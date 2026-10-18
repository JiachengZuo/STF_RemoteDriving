#!/usr/bin/env python3
# Remote Driving Framework - Relay Server
# Central hub: forwards sensor data Vehicle->Operator, forwards controls Operator->Vehicle
# Future: insert QCNet delay compensation here

import socket
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import RELAY_HOST, RELAY_PORT


class RelayServer:
    def __init__(self, host=RELAY_HOST, port=RELAY_PORT):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

        self.vehicle_conn = None
        self.operator_conn = None
        self.vehicle_lock = threading.Lock()
        self.operator_lock = threading.Lock()
        self.running = True

        self.stats = {
            'frames_forwarded': 0,
            'controls_forwarded': 0,
            'start_time': time.time(),
        }

    def log(self, msg):
        ts = time.time() - self.stats['start_time']
        print(f"[{ts:8.2f}s] [Relay] {msg}")

    def _handle_vehicle(self, conn, addr):
        """Thread: receive sensor_data from vehicle, forward to operator."""
        self.log(f"Vehicle connected from {addr}")
        try:
            while self.running:
                msg = recv_message(conn)
                msg_type = msg.get('type', 'unknown')

                if msg_type == 'sensor_data':
                    self.stats['frames_forwarded'] += 1
                    fid = msg.get('frame_id', 0)
                    if self.stats['frames_forwarded'] % 30 == 1:
                        fps = self.stats['frames_forwarded'] / max(1, time.time() - self.stats['start_time'])
                        self.log(f"Forwarding frame #{fid} (total: {self.stats['frames_forwarded']}, "
                                 f"{fps:.1f} FPS, {len(msg.get('image', ''))} bytes b64)")

                elif msg_type == 'ping':
                    send_message(conn, {'type': 'pong'})
                    continue

                # Forward to operator if connected
                with self.operator_lock:
                    if self.operator_conn:
                        try:
                            send_message(self.operator_conn, msg)
                        except Exception as e:
                            self.log(f"Operator write failed: {e}")
                            self.operator_conn = None
                    else:
                        # Drop frames if no operator connected
                        if self.stats['frames_forwarded'] % 60 == 1:
                            self.log("No operator connected, dropping frame")

        except (ConnectionError, OSError) as e:
            self.log(f"Vehicle disconnected: {e}")
        finally:
            self.log("Vehicle handler exiting")
            try:
                conn.close()
            except Exception:
                pass

    def _handle_operator(self, conn, addr):
        """Thread: receive control messages from operator, forward to vehicle."""
        self.log(f"Operator connected from {addr}")
        try:
            while self.running:
                msg = recv_message(conn)
                msg_type = msg.get('type', 'unknown')

                if msg_type == 'control':
                    self.stats['controls_forwarded'] += 1
                    if self.stats['controls_forwarded'] % 30 == 1:
                        self.log(f"Forwarding control (total: {self.stats['controls_forwarded']}): "
                                 f"throttle={msg.get('throttle', 0):.2f}, "
                                 f"steer={msg.get('steer', 0):.2f}, "
                                 f"brake={msg.get('brake', 0):.2f}")

                elif msg_type == 'ping':
                    send_message(conn, {'type': 'pong'})
                    continue

                # Forward to vehicle if connected
                with self.vehicle_lock:
                    if self.vehicle_conn:
                        try:
                            send_message(self.vehicle_conn, msg)
                        except Exception as e:
                            self.log(f"Vehicle write failed: {e}")
                            self.vehicle_conn = None
                    else:
                        self.log("No vehicle connected, dropping control")

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
            self.sock.close()
            # Print stats
            elapsed = max(1, time.time() - self.stats['start_time'])
            self.log(f"Stats: {self.stats['frames_forwarded']} frames, "
                     f"{self.stats['controls_forwarded']} controls in {elapsed:.1f}s")
            self.log("Relay server stopped.")


def main():
    server = RelayServer()
    server.run()


if __name__ == '__main__':
    main()