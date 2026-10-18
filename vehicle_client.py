#!/usr/bin/env python3
# Remote Driving Framework - Vehicle Client
# Connects to CARLA server, spawns ego vehicle + sensors,
# streams camera/telemetry to Relay Server, receives controls from Relay Server.

import carla
import socket
import threading
import time
import sys
import os
import weakref
import math
import collections
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message, encode_image
from config import (CARLA_HOST, CARLA_PORT, RELAY_HOST, RELAY_PORT,
                    CAMERA_WIDTH, CAMERA_HEIGHT, JPEG_QUALITY, SYNC_MODE,
                    FIXED_DELTA_SECONDS, CAMERA_POS_X, CAMERA_POS_Z, CAMERA_FOV)
from lane_extractor import extract_lanes


# ==============================================================================
# -- Sensors (adapted from manual_control.py) ----------------------------------
# ==============================================================================

class CollisionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self._recent_collision = False
        self._last_collision_frame = -100
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self._recent_collision = True
        self._last_collision_frame = event.frame

    def has_recent_collision(self, current_frame, window=20):
        return (current_frame - self._last_collision_frame) < window

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


# ==============================================================================
# -- Vehicle Client ------------------------------------------------------------
# ==============================================================================

class VehicleClient:
    def __init__(self):
        self.relay_sock = None
        self.carla_client = None
        self.world = None
        self.player = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.running = False

        # Thread-safe camera data
        self.camera_lock = threading.Lock()
        self.latest_frame = None  # dict with image_bgr, timestamp, frame_id

        # Latest control
        self.control_lock = threading.Lock()
        self.latest_control = None  # dict

        self.frame_count = 0
        self.last_sent_frame_id = -1
        self.start_time = 0

    def log(self, msg):
        ts = time.time() - self.start_time if self.start_time else 0
        print(f"[{ts:8.2f}s] [Vehicle] {msg}")

    # -- CARLA Setup --

    def connect_carla(self):
        self.log(f"Connecting to CARLA server at {CARLA_HOST}:{CARLA_PORT}...")
        self.carla_client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.carla_client.set_timeout(10.0)
        self.world = self.carla_client.get_world()

        if SYNC_MODE:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
            settings.no_rendering_mode = True  # no rendering on vehicle side
            self.world.apply_settings(settings)
            self.log(f"Synchronous mode enabled, delta={FIXED_DELTA_SECONDS}s, no rendering")

        self.log(f"Connected to CARLA map: {self.world.get_map().name}")

    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        bp = random.choice(bp_lib.filter('vehicle.*'))
        bp.set_attribute('role_name', 'remote_ego')
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in this map")

        for sp in spawn_points[:20] + random.sample(spawn_points, min(20, len(spawn_points))):
            self.player = self.world.try_spawn_actor(bp, sp)
            if self.player is not None:
                break

        if self.player is None:
            raise RuntimeError("Failed to spawn vehicle after trying multiple spawn points")

        self.log(f"Spawned {self.player.type_id} at {self.player.get_location()}")

        # Spawn multiple NPC vehicles so we have a populated scene
        tm = self.carla_client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm_port = tm.get_port()
        self._npc_vehicles = []

        # Collect available spawn points that aren't occupied by ego
        ego_loc = self.player.get_location()
        npc_count = 0
        npc_target = 15  # spawn ~15 NPC vehicles

        vehicle_bps = bp_lib.filter('vehicle.*')
        available_sps = []
        for sp in spawn_points:
            d = math.sqrt((sp.location.x - ego_loc.x)**2 + (sp.location.y - ego_loc.y)**2)
            if d > 5.0:  # don't spawn on top of ego
                available_sps.append(sp)
        random.shuffle(available_sps)

        for sp in available_sps[:npc_target]:
            bp_npc = random.choice(vehicle_bps)
            bp_npc.set_attribute('role_name', 'remote_npc')
            if bp_npc.has_attribute('color'):
                bp_npc.set_attribute('color',
                    random.choice(bp_npc.get_attribute('color').recommended_values))
            v = self.world.try_spawn_actor(bp_npc, sp)
            if v is not None:
                v.set_autopilot(True, self.tm_port)
                self._npc_vehicles.append(v)
                npc_count += 1

        if npc_count > 0:
            self.log(f"Spawned {npc_count} NPC vehicles with autopilot via TrafficManager")
        else:
            self.log("Could not spawn any NPC vehicles")

        # Spawn collision sensor
        self.collision_sensor = CollisionSensor(self.player)

        # Spawn RGB camera (first-person view)
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(FIXED_DELTA_SECONDS))

        camera_transform = carla.Transform(
            carla.Location(x=CAMERA_POS_X, z=CAMERA_POS_Z),
            carla.Rotation(pitch=0)
        )
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player,
            attachment_type=carla.AttachmentType.Rigid
        )
        weak_self = weakref.ref(self)
        self.camera_sensor.listen(lambda image: VehicleClient._camera_callback(weak_self, image))
        self.log(f"Camera sensor spawned: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, FOV={CAMERA_FOV}")

    @staticmethod
    def _camera_callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Convert BGRA -> BGR numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # strip alpha

        with self.camera_lock:
            self.latest_frame = {
                'image_bgr': array,
                'timestamp': image.timestamp,
                'frame_id': image.frame,
            }

    # -- Relay Connection --

    def connect_relay(self):
        self.log(f"Connecting to relay server at {RELAY_HOST}:{RELAY_PORT}...")
        self.relay_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.relay_sock.connect((RELAY_HOST, RELAY_PORT))

        # Extract lane data for BEV display
        self.log("Extracting CARLA lane waypoints...")
        t0 = time.time()
        lane_data = extract_lanes(self.world, sample_distance=8.0, max_distance=200.0)
        self.log(f"Lane extraction: {lane_data['num_lanes_summary']} "
                 f"in {time.time() - t0:.1f}s")

        register_msg = {
            'type': 'register',
            'role': 'vehicle',
            'lane_data': lane_data,
        }
        send_message(self.relay_sock, register_msg)
        self.log("Registered with relay server as vehicle (with lane data)")

    # -- Data Streaming --

    def _get_surrounding_agents(self, max_range=100.0, max_agents=20):
        """Query CARLA for surrounding vehicles within range.

        Returns list of dicts with id, x, y, yaw_rad, vx, vy for each agent.
        """
        ego_loc = self.player.get_location()
        ego_id = self.player.id
        vehicles = self.world.get_actors().filter('vehicle.*')

        agents = []
        for v in vehicles:
            if v.id == ego_id:
                continue
            vloc = v.get_location()
            dx = vloc.x - ego_loc.x
            dy = vloc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_range:
                continue

            vvel = v.get_velocity()
            vrot = v.get_transform().rotation
            agents.append({
                'id': v.id,
                'x': round(vloc.x, 2),
                'y': round(vloc.y, 2),
                'yaw_rad': round(math.radians(vrot.yaw), 4),
                'vx': round(vvel.x, 2),
                'vy': round(vvel.y, 2),
                'dist': round(dist, 1),
            })
            if len(agents) >= max_agents:
                break

        return agents

    def _send_frame(self, frame_data):
        """Encode and send one camera frame + telemetry to relay."""
        b64_image = encode_image(frame_data['image_bgr'])
        if b64_image is None:
            return

        # Gather telemetry
        vel = self.player.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        loc = self.player.get_location()
        rot = self.player.get_transform().rotation

        collision = self.collision_sensor.has_recent_collision(frame_data['frame_id'])

        # Detect surrounding vehicles for multi-agent QCNet prediction
        surrounding = self._get_surrounding_agents()

        msg = {
            'type': 'sensor_data',
            'timestamp': frame_data['timestamp'],
            'frame_id': frame_data['frame_id'],
            'image': b64_image,
            'image_width': CAMERA_WIDTH,
            'image_height': CAMERA_HEIGHT,
            'speed_kmh': round(speed_kmh, 1),
            'velocity': [round(vel.x, 2), round(vel.y, 2), round(vel.z, 2)],
            'location': [round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)],
            'rotation': [round(rot.yaw, 2), round(rot.pitch, 2), round(rot.roll, 2)],
            'collision': collision,
            'surrounding': surrounding,
        }

        try:
            send_message(self.relay_sock, msg)
            return True
        except Exception as e:
            self.log(f"Failed to send frame: {e}")
            return False

    def _control_recv_thread(self):
        """Background thread: receive control messages from relay."""
        self.log("Control receiver thread started")
        try:
            while self.running:
                msg = recv_message(self.relay_sock)
                if msg.get('type') == 'control':
                    with self.control_lock:
                        self.latest_control = {
                            'throttle': float(msg.get('throttle', 0)),
                            'steer': float(msg.get('steer', 0)),
                            'brake': float(msg.get('brake', 0)),
                            'reverse': bool(msg.get('reverse', False)),
                            'hand_brake': bool(msg.get('hand_brake', False)),
                            'gear': int(msg.get('gear', 0)),
                            'manual_gear_shift': bool(msg.get('manual_gear_shift', False)),
                        }
                elif msg.get('type') == 'pong':
                    pass  # ignore heartbeat responses
        except (ConnectionError, OSError) as e:
            self.log(f"Control receiver thread exiting: {e}")
            self.running = False

    # -- Main Loop --

    def apply_latest_control(self):
        """Apply the most recently received control to the vehicle."""
        with self.control_lock:
            c = self.latest_control
            self.latest_control = None
        if c is None:
            return

        control = carla.VehicleControl()
        control.throttle = max(0.0, min(1.0, c['throttle']))
        control.steer = max(-1.0, min(1.0, c['steer']))
        control.brake = max(0.0, min(1.0, c['brake']))
        control.reverse = c['reverse']
        control.hand_brake = c['hand_brake']
        control.gear = c['gear']
        control.manual_gear_shift = c['manual_gear_shift']
        self.player.apply_control(control)

    def _enforce_speed_limit(self):
        """Enforce 10 km/h speed cap smoothly on EVERY tick."""
        MAX_SPEED_KMH = 10.0
        limit_mps = MAX_SPEED_KMH / 3.6

        vel = self.player.get_velocity()
        speed_mps = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        speed_kmh = speed_mps * 3.6

        if speed_mps >= limit_mps:
            # Smooth proportional throttle reduction: no hard brake.
            # At 10.1 km/h → throttle *= 0.3, at 12+ km/h → throttle = 0.
            overshoot = speed_kmh - MAX_SPEED_KMH
            factor = max(0.0, 1.0 - overshoot * 0.4)
            ctrl = self.player.get_control()
            ctrl.throttle *= factor
            if overshoot > 2.0:
                ctrl.brake = max(ctrl.brake, 0.1)
            self.player.apply_control(ctrl)

    def run(self):
        self.start_time = time.time()
        self.log("=== Vehicle Client Starting ===")

        try:
            self.connect_carla()
            self.spawn_vehicle()
            self.connect_relay()

            self.running = True

            # Start control receiver thread
            ctrl_thread = threading.Thread(target=self._control_recv_thread, daemon=True)
            ctrl_thread.start()

            # Main loop: tick world, send latest frame, apply controls
            # In synchronous mode, TM auto-advances with world.tick()
            self.log("Entering main loop...")
            while self.running:
                if SYNC_MODE:
                    self.world.tick()

                # Apply received control before sending frame (so the frame
                # reflects the most recent command)
                self.apply_latest_control()

                # ALSO enforce speed limit EVERY tick — not only when
                # a new control arrives. This guarantees the vehicle never
                # exceeds 10 km/h regardless of operator behavior.
                self._enforce_speed_limit()

                # Send new camera frame (only when frame_id changes)
                with self.camera_lock:
                    frame = self.latest_frame

                if frame is not None and frame['frame_id'] != self.last_sent_frame_id:
                    self.last_sent_frame_id = frame['frame_id']
                    self.frame_count += 1
                    self._send_frame(frame)

                    if self.frame_count % 60 == 1:
                        fps = self.frame_count / max(1, time.time() - self.start_time)
                        self.log(f"Sent {self.frame_count} frames, {fps:.1f} FPS")

        except KeyboardInterrupt:
            self.log("Interrupted by user")
        except Exception as e:
            self.log(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self.cleanup()

    def cleanup(self):
        self.log("Cleaning up...")
        self.running = False

        # Destroy NPC vehicles (disable autopilot first)
        for v in getattr(self, '_npc_vehicles', []):
            try:
                v.set_autopilot(False)
                v.destroy()
            except Exception:
                pass

        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        if self.player:
            self.player.destroy()
        if self.relay_sock:
            try:
                self.relay_sock.close()
            except Exception:
                pass

        # Restore CARLA settings
        if self.world and SYNC_MODE:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
                self.log("Restored CARLA settings")
            except Exception:
                pass

        self.log("Vehicle client stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remote Driving Vehicle Client')
    parser.add_argument('--host', default=CARLA_HOST, help='CARLA server host')
    parser.add_argument('--port', type=int, default=CARLA_PORT, help='CARLA server port')
    parser.add_argument('--relay-host', default=RELAY_HOST, help='Relay server host')
    parser.add_argument('--relay-port', type=int, default=RELAY_PORT, help='Relay server port')
    args = parser.parse_args()

    # Override config with CLI args
    import config
    config.CARLA_HOST = args.host
    config.CARLA_PORT = args.port
    config.RELAY_HOST = args.relay_host
    config.RELAY_PORT = args.relay_port

    client = VehicleClient()
    client.run()


if __name__ == '__main__':
    main()