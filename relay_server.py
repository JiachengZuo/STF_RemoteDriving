#!/usr/bin/env python3
# Remote Driving Framework - Relay Server
# Central hub: forwards sensor data Vehicle->Operator, forwards controls Operator->Vehicle
# QCNet delay compensation: predicts future trajectories asynchronously.

import socket
import threading
import time
import sys
import os
import subprocess
import heapq
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import (RELAY_HOST, RELAY_PORT, QCNet_IPC_PORT, QCNet_PYTHON,
                    QCNet_CHECKPOINT, QCNet_SERVER_PATH, QCNet_ROOT)
from trajectory_buffer import TrajectoryBuffer


class RelayServer:
    def __init__(self, host=RELAY_HOST, port=RELAY_PORT, delay_ms=0,
                 enable_qcnet=True):
        self.host = host
        self.port = port
        self.delay_ms = delay_ms
        self.enable_qcnet = enable_qcnet
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

        self.vehicle_conn = None
        self.operator_conn = None
        self.vehicle_lock = threading.Lock()
        self.operator_lock = threading.Lock()
        self.running = True

        # Delay queues: (deadline, msg) pushed by receiver, popped by forwarder
        # Using heapq for O(log n) insert/pop by deadline
        self._v2o_queue = []      # vehicle -> operator: sensor_data
        self._o2v_queue = []      # operator -> vehicle: controls
        self._v2o_lock = threading.Lock()
        self._o2v_lock = threading.Lock()
        self._v2o_event = threading.Event()   # signaled when queue non-empty
        self._o2v_event = threading.Event()

        self.stats = {
            'frames_forwarded': 0,
            'controls_forwarded': 0,
            'qcnet_inferences': 0,
            'start_time': time.time(),
        }

        # QCNet state
        self.trajectory_buffer = TrajectoryBuffer()
        self.latest_prediction = None  # {trajs: [6,60,2] tensor, probs: [6] tensor}
        self.prediction_lock = threading.Lock()
        self.qcnet_process = None
        self.qcnet_client = None

        # Lane data (sent once by vehicle, forwarded to operator on connect)
        self.lane_data = None
        self.lane_data_lock = threading.Lock()

        # Start QCNet subprocess and inference thread
        if self.enable_qcnet:
            self._start_qcnet()
            self._inference_thread = threading.Thread(
                target=self._inference_worker, daemon=True)
            self._inference_thread.start()

        # Start delay-forwarding threads
        self._v2o_thread = threading.Thread(target=self._forward_v2o, daemon=True)
        self._v2o_thread.start()
        self._o2v_thread = threading.Thread(target=self._forward_o2v, daemon=True)
        self._o2v_thread.start()

    def log(self, msg):
        ts = time.time() - self.stats['start_time']
        print(f"[{ts:8.2f}s] [Relay] {msg}")

    # ------------------------------------------------------------------
    # QCNet inference
    # ------------------------------------------------------------------

    def _start_qcnet(self):
        """Launch QCNet inference server as a subprocess."""
        try:
            self.qcnet_process = subprocess.Popen(
                [QCNet_PYTHON, '-u', QCNet_SERVER_PATH,
                 '--port', str(QCNet_IPC_PORT),
                 '--checkpoint', QCNet_CHECKPOINT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Read lines until we see "READY"
            for line in self.qcnet_process.stdout:
                line = line.strip()
                self.log(f"QCNet process: {line}")
                if line.startswith('[QCNet Server] READY'):
                    break

            # Import TCP client (only if QCNet enabled)
            from qcnet_client import QCNetClient
            self.qcnet_client = QCNetClient(port=QCNet_IPC_PORT)
            self.log(f"QCNet TCP client ready on port {QCNet_IPC_PORT}")

            # Drain remaining stdout so pipe buffer doesn't fill over long runs
            import threading as _th
            def _drain_qcnet_stdout():
                for line in self.qcnet_process.stdout:
                    line = line.strip()
                    if line:
                        if 'error' in line.lower() or 'exception' in line.lower():
                            self.log(f"QCNet process: {line}")
                self.log("QCNet process stdout closed")
            _th.Thread(target=_drain_qcnet_stdout, daemon=True).start()
        except Exception as e:
            self.log(f"QCNet startup failed: {e}. Running without predictions.")
            self.enable_qcnet = False

    def _inference_worker(self):
        """Thread: periodically run QCNet inference on latest trajectory buffer.

        Runs asynchronously from forwarding — never blocks frame delivery.
        """
        self.log("QCNet inference worker started")
        last_infer_time = 0
        min_interval = 0.12  # max ~8 inferences/sec (tight loop for smooth BEV)

        while self.running and self.enable_qcnet:
            time.sleep(0.03)  # poll every 30ms

            now = time.time()
            if now - last_infer_time < min_interval:
                continue

            if not self.trajectory_buffer.is_ready():
                continue

            # Build HeteroData and send to QCNet.
            # CRITICAL: capture agent_origins at the SAME TIME as the input data,
            # so trajectories and origins are always aligned.
            try:
                data_dict = self.trajectory_buffer.to_hetero_data()
                agent_origins = self.trajectory_buffer.agent_origins()
                result = self.qcnet_client.predict(data_dict)

                if result is not None:
                    result['agent_origins'] = agent_origins
                    result['prediction_time'] = now  # for continuous BEV interpolation
                    with self.prediction_lock:
                        self.latest_prediction = result
                    self.stats['qcnet_inferences'] += 1
                    last_infer_time = now

                    if self.stats['qcnet_inferences'] % 30 == 1:
                        self.log(f"QCNet inference #{self.stats['qcnet_inferences']}: "
                                 f"buffer steps={self.trajectory_buffer.valid_count()}, "
                                 f"agents={len(agent_origins)}")
            except Exception as e:
                self.log(f"QCNet inference error: {e}")

    def _stop_qcnet(self):
        """Clean up QCNet process and client."""
        if self.qcnet_client:
            try:
                self.qcnet_client.close()
            except Exception:
                pass
            self.qcnet_client = None
        if self.qcnet_process:
            try:
                self.qcnet_process.terminate()
                self.qcnet_process.wait(timeout=5)
            except Exception:
                try:
                    self.qcnet_process.kill()
                except Exception:
                    pass
            self.qcnet_process = None

    # ------------------------------------------------------------------
    # Delay-forwarding threads (pop from queue when deadline expires)
    # ------------------------------------------------------------------

    def _forward_v2o(self):
        """Thread: pop expired frames from delay queue, forward only the latest.

        Drops intermediate frames to avoid TCP backpressure when operator
        falls behind. This ensures control messages can always get through.
        """
        last_forward_time = 0
        min_forward_interval = 0.04  # ~25 FPS max to operator (reduces delay)

        while self.running:
            self._v2o_event.wait(timeout=0.025)  # wake every 25ms for ~25 FPS
            self._v2o_event.clear()

            latest_msg = None
            skipped = 0

            with self._v2o_lock:
                while self._v2o_queue:
                    deadline, msg = self._v2o_queue[0]
                    if deadline > time.time():
                        break
                    heapq.heappop(self._v2o_queue)
                    latest_msg = msg
                    skipped += 1

            if latest_msg is None:
                continue

            # Rate-limit forwarding to operator (max ~25 FPS)
            now = time.time()
            if now - last_forward_time < min_forward_interval:
                continue
            last_forward_time = now

            # Only forward the latest expired frame; skip intermediate ones
            self.stats['frames_forwarded'] += skipped
            fid = latest_msg.get('frame_id', 0)
            if self.stats['frames_forwarded'] % 30 == 1:
                fps = self.stats['frames_forwarded'] / max(1, time.time() - self.stats['start_time'])
                qcnet_info = f", qcnet={self.stats['qcnet_inferences']}" if self.enable_qcnet else ""
                self.log(f"Forwarding frame #{fid} (total: {self.stats['frames_forwarded']}, "
                         f"{fps:.1f} FPS, delay={self.delay_ms}ms, "
                         f"queue={len(self._v2o_queue)}, skipped={skipped-1}{qcnet_info})")

            # Attach QCNet predictions if available
            if self.enable_qcnet:
                with self.prediction_lock:
                    pred = self.latest_prediction
                if pred is not None:
                    # Use agent_origins captured AT INFERENCE TIME (guaranteed aligned)
                    origins_list = pred.get('agent_origins', None)
                    if origins_list is None:
                        origins_list = self.trajectory_buffer.agent_origins()
                    origin_x, origin_y, origin_yaw = (
                        origins_list[0] if origins_list else (0.0, 0.0, 0.0))

                    trajs_list = pred['trajs'].tolist()
                    probs_list = pred['probs'].tolist()
                    num_agents = len(trajs_list)
                    trajs_rounded_all = [
                        [[[round(v, 2) for v in pt] for pt in mode]
                         for mode in agent_trajs]
                        for agent_trajs in trajs_list
                    ]
                    probs_rounded = [
                        [round(p, 4) for p in agent_probs]
                        for agent_probs in probs_list
                    ]
                    origins_rounded = [[round(v, 4) for v in o] for o in origins_list]

                    latest_msg['qcnet_trajs'] = trajs_rounded_all
                    latest_msg['qcnet_probs'] = probs_rounded
                    latest_msg['qcnet_num_agents'] = num_agents
                    latest_msg['qcnet_origin'] = [round(origin_x, 2),
                                                   round(origin_y, 2),
                                                   round(origin_yaw, 4)]
                    latest_msg['qcnet_agent_origins'] = origins_rounded
                    latest_msg['qcnet_prediction_time'] = pred.get('prediction_time', 0)

            with self.operator_lock:
                if self.operator_conn:
                    try:
                        send_message(self.operator_conn, latest_msg)
                    except Exception as e:
                        self.log(f"Operator write failed: {e}")
                        self.operator_conn = None

    def _forward_o2v(self):
        """Thread: pop control messages from delay queue when deadline expires,
        forward to vehicle."""
        while self.running:
            self._o2v_event.wait(timeout=0.05)
            self._o2v_event.clear()

            while True:
                with self._o2v_lock:
                    if not self._o2v_queue:
                        break
                    deadline, msg = self._o2v_queue[0]
                    if deadline > time.time():
                        break
                    heapq.heappop(self._o2v_queue)

                # Forward outside the lock
                self.stats['controls_forwarded'] += 1
                if self.stats['controls_forwarded'] % 30 == 1:
                    self.log(f"Forwarding control (total: {self.stats['controls_forwarded']}): "
                             f"throttle={msg.get('throttle', 0):.2f}, "
                             f"steer={msg.get('steer', 0):.2f}, "
                             f"brake={msg.get('brake', 0):.2f}")

                with self.vehicle_lock:
                    if self.vehicle_conn:
                        try:
                            send_message(self.vehicle_conn, msg)
                        except Exception as e:
                            self.log(f"Vehicle write failed: {e}")
                            self.vehicle_conn = None

    # ------------------------------------------------------------------
    # Receiver threads (non-blocking: push into delay queues)
    # ------------------------------------------------------------------

    def _handle_vehicle(self, conn, addr):
        """Thread: receive sensor_data from vehicle, push into delay queue."""
        self.log(f"Vehicle connected from {addr}")
        try:
            while self.running:
                msg = recv_message(conn)
                msg_type = msg.get('type', 'unknown')

                if msg_type == 'sensor_data':
                    # Record time when relay received this frame
                    server_recv_time = time.time()
                    msg['server_recv_time'] = server_recv_time
                    deadline = server_recv_time + self.delay_ms / 1000.0

                    # Feed trajectory buffer for QCNet
                    if self.enable_qcnet:
                        loc = msg.get('location', [0, 0, 0])
                        rot = msg.get('rotation', [0, 0, 0])
                        vel = msg.get('velocity', [0, 0, 0])
                        yaw_rad = math.radians(rot[0])
                        surrounding = msg.get('surrounding', [])
                        self.trajectory_buffer.append(
                            loc[0], loc[1], yaw_rad, vel[0], vel[1],
                            surrounding=surrounding)

                    with self._v2o_lock:
                        heapq.heappush(self._v2o_queue, (deadline, msg))
                        # Cap queue at 100 frames: drop oldest if overflow
                        while len(self._v2o_queue) > 100:
                            heapq.heappop(self._v2o_queue)
                    self._v2o_event.set()

                elif msg_type == 'ping':
                    send_message(conn, {'type': 'pong'})
                    continue

        except (ConnectionError, OSError) as e:
            self.log(f"Vehicle disconnected: {e}")
        finally:
            self.log("Vehicle handler exiting")
            try:
                conn.close()
            except Exception:
                pass

    def _handle_operator(self, conn, addr):
        """Thread: receive control messages from operator, push into delay queue."""
        self.log(f"Operator connected from {addr}")

        # Send lane data immediately on operator connect
        with self.lane_data_lock:
            ld = self.lane_data
        if ld:
            try:
                send_message(conn, {'type': 'lane_data', 'lane_data': ld})
                self.log(f"Sent lane data to operator: {ld.get('num_lanes_summary', '?')}")
            except Exception as e:
                self.log(f"Failed to send lane data: {e}")

        try:
            while self.running:
                msg = recv_message(conn)
                msg_type = msg.get('type', 'unknown')

                if msg_type == 'control':
                    recv_time = time.time()
                    deadline = recv_time + self.delay_ms / 1000.0

                    with self._o2v_lock:
                        heapq.heappush(self._o2v_queue, (deadline, msg))
                    self._o2v_event.set()

                elif msg_type == 'ping':
                    send_message(conn, {'type': 'pong'})
                    continue

        except (ConnectionError, OSError) as e:
            self.log(f"Operator disconnected: {e}")
        finally:
            self.log("Operator handler exiting")
            try:
                conn.close()
            except Exception:
                pass

    def run(self):
        """Main loop: accept connections, dispatch to handler threads."""
        self.log(f"Relay server listening on {self.host}:{self.port}")
        self.log("Waiting for vehicle and operator connections...")

        try:
            while self.running:
                conn, addr = self.sock.accept()

                # Read registration message to identify role
                try:
                    msg = recv_message(conn)
                except Exception as e:
                    self.log(f"Failed to read registration from {addr}: {e}")
                    conn.close()
                    continue

                role = msg.get('role', '')
                self.log(f"New connection from {addr}, role={role}")

                if role == 'vehicle':
                    # Store lane data if provided in registration
                    ld = msg.get('lane_data', None)
                    if ld:
                        with self.lane_data_lock:
                            self.lane_data = ld
                        self.log(f"Received lane data: {ld.get('num_lanes_summary', 'unknown')}")

                    # Close old vehicle connection if exists
                    with self.vehicle_lock:
                        old = self.vehicle_conn
                        self.vehicle_conn = conn
                    if old:
                        self.log("Replacing old vehicle connection")
                        try:
                            old.close()
                        except Exception:
                            pass
                    t = threading.Thread(target=self._handle_vehicle,
                                         args=(conn, addr), daemon=True)
                    t.start()

                elif role == 'operator':
                    with self.operator_lock:
                        old = self.operator_conn
                        self.operator_conn = conn
                    if old:
                        self.log("Replacing old operator connection")
                        try:
                            old.close()
                        except Exception:
                            pass
                    t = threading.Thread(target=self._handle_operator,
                                         args=(conn, addr), daemon=True)
                    t.start()

                else:
                    self.log(f"Unknown role '{role}', closing connection")
                    conn.close()

        except KeyboardInterrupt:
            self.log("Shutting down...")
        finally:
            self.running = False
            self._stop_qcnet()
            self.sock.close()
            # Print stats
            elapsed = max(1, time.time() - self.stats['start_time'])
            self.log(f"Stats: {self.stats['frames_forwarded']} frames, "
                     f"{self.stats['controls_forwarded']} controls, "
                     f"{self.stats['qcnet_inferences']} QCNet inferences "
                     f"in {elapsed:.1f}s")
            self.log("Relay server stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remote Driving Relay Server')
    parser.add_argument('--delay', type=int, default=0,
                        help='Artificial delay in milliseconds (simulates network latency)')
    parser.add_argument('--host', default=RELAY_HOST, help='Relay server host')
    parser.add_argument('--port', type=int, default=RELAY_PORT, help='Relay server port')
    parser.add_argument('--no-qcnet', action='store_true',
                        help='Disable QCNet trajectory prediction')
    args = parser.parse_args()

    server = RelayServer(host=args.host, port=args.port, delay_ms=args.delay,
                         enable_qcnet=not args.no_qcnet)
    if args.delay > 0:
        server.log(f"Delay simulation enabled: {args.delay}ms")
    if args.no_qcnet:
        server.log("QCNet disabled (--no-qcnet)")
    server.run()


if __name__ == '__main__':
    main()