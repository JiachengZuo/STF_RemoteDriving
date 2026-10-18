#!/usr/bin/env python3
# Remote Driving Framework - Operator Client
# Connects to Relay Server, receives camera/telemetry via TCP,
# displays with pygame, captures keyboard input, sends controls.

import socket
import threading
import time
import sys
import os
import math
import numpy as np

try:
    import pygame
    from pygame.locals import (
        K_ESCAPE, K_SPACE, K_w, K_a, K_s, K_d, K_q,
        K_UP, K_DOWN, K_LEFT, K_RIGHT,
        K_F1, K_h, K_SLASH, K_BACKSPACE,
        KMOD_CTRL, KMOD_SHIFT
    )
except ImportError:
    raise RuntimeError('pygame is required. Install with: pip install pygame')

try:
    import cv2
except ImportError:
    raise RuntimeError('opencv-python is required. Install with: pip install opencv-python')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message, decode_image
from config import RELAY_HOST, RELAY_PORT, CAMERA_WIDTH, CAMERA_HEIGHT, BEV_WIDTH, BEV_RANGE
from bev_renderer import BEVRenderer


# ==============================================================================
# -- HUD (adapted from manual_control.py) --------------------------------------
# ==============================================================================

class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        font_big = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        self._font = font_small
        self._font_big = font_big
        self._info_text = []
        self._show_info = True

        # Received data
        self.speed_kmh = 0.0
        self.location = (0, 0, 0)
        self.collision = False
        self.data_fps = 0.0
        self.display_fps = 0.0
        self.control = {'throttle': 0, 'steer': 0, 'brake': 0, 'reverse': False, 'hand_brake': False}
        self.total_frames = 0
        self.delay_ms = 0

    def toggle_info(self):
        self._show_info = not self._show_info

    def tick(self, clock):
        pass

    def render(self, display):
        if not self._show_info:
            return

        # Semi-transparent background panel (left side)
        info_surface = pygame.Surface((260, self.dim[1]))
        info_surface.set_alpha(180)
        info_surface.fill((0, 0, 0))
        display.blit(info_surface, (0, 0))

        # Header
        title = self._font_big.render('REMOTE OPERATOR', True, (0, 255, 128))
        display.blit(title, (10, 8))

        texts = [
            '',
            f'Data FPS:    {self.data_fps:5.1f}',
            f'Display FPS:  {self.display_fps:5.1f}',
            f'Frames recv:  {self.total_frames}',
            '',
            f'Speed:    {self.speed_kmh:6.1f} km/h  (max 10)',
            f'Location: ({self.location[0]:.1f}, {self.location[1]:.1f})',
            '',
            f'Throttle: {self.control["throttle"]:.2f}',
            f'Steer:    {self.control["steer"]:.2f}',
            f'Brake:    {self.control["brake"]:.2f}',
            f'Reverse:  {"ON" if self.control["reverse"] else "off"}',
            f'Handbrake: {"ON" if self.control["hand_brake"] else "off"}',
        ]

        if self.collision:
            texts.insert(5, '*** COLLISION! ***')

        y = 32
        for line in texts:
            if y + 20 > self.dim[1]:
                break
            if line == '*** COLLISION! ***':
                surf = self._font_big.render(line, True, (255, 50, 50))
            else:
                surf = self._font.render(line, True, (200, 200, 200))
            display.blit(surf, (10, y))
            y += 18

        # Throttle/Steer/Brake bars (right side)
        bar_x = self.dim[0] - 140
        for label, val, bar_y in [
            ('Throttle', self.control['throttle'], self.dim[1] - 80),
            ('Steer', (self.control['steer'] + 1) / 2, self.dim[1] - 55),
            ('Brake', self.control['brake'], self.dim[1] - 30),
        ]:
            lbl = self._font.render(label, True, (180, 180, 180))
            display.blit(lbl, (bar_x - 65, bar_y))
            # Background
            pygame.draw.rect(display, (60, 60, 60), (bar_x, bar_y + 4, 100, 12))
            # Value
            bar_w = max(2, int(val * 100))
            color = (0, 255, 128) if label == 'Throttle' else (255, 200, 50) if label == 'Steer' else (255, 80, 80)
            pygame.draw.rect(display, color, (bar_x, bar_y + 4, bar_w, 12))

        # Delay indicator
        delay_color = (255, 255, 50) if self.delay_ms > 200 else (100, 255, 100)
        delay_text = self._font.render(f'Delay: ~{self.delay_ms:.0f}ms', True, delay_color)
        display.blit(delay_text, (10, self.dim[1] - 25))


# ==============================================================================
# -- Operator Client -----------------------------------------------------------
# ==============================================================================

class OperatorClient:
    def __init__(self):
        self.relay_sock = None
        self.running = False

        # Thread-safe shared state
        self.data_lock = threading.Lock()
        self.latest_image_bgr = None  # numpy array (H, W, 3)
        self.latest_surface = None    # pygame surface for display
        self.latest_telemetry = {}
        self.latest_delay_ms = 0.0
        self._smooth_delay_ms = 0.0  # EWMA-smoothed for stable BEV compensation
        self.latest_qcnet_data = {}  # qcnet_trajs, qcnet_probs, qcnet_origin
        self.lane_data = None        # lane centerlines + boundaries from relay
        self.total_frames = 0
        self.data_fps_tracker = []  # timestamps of recent frames

        # Control state (read/written in main thread, no lock needed)
        self.control = {
            'throttle': 0.0,
            'steer': 0.0,
            'brake': 0.0,
            'reverse': False,
            'hand_brake': False,
            'gear': 0,
            'manual_gear_shift': False,
        }
        self.steer_cache = 0.0
        self._last_sent_control = None
        self._control_send_interval = 0  # counter: send every N frames

        # HUD
        self.hud = None
        self.clock = None

    def log(self, msg):
        print(f"[Operator] {msg}")

    # -- Network --

    def connect_relay(self):
        self.log(f"Connecting to relay server at {RELAY_HOST}:{RELAY_PORT}...")
        self.relay_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.relay_sock.connect((RELAY_HOST, RELAY_PORT))
        send_message(self.relay_sock, {'type': 'register', 'role': 'operator'})
        self.log("Registered with relay server as operator")

    def send_control(self):
        """Send current control state to relay if changed from last sent."""
        msg = {
            'type': 'control',
            'throttle': round(self.control['throttle'], 2),
            'steer': round(self.control['steer'], 2),
            'brake': round(self.control['brake'], 2),
            'reverse': self.control['reverse'],
            'hand_brake': self.control['hand_brake'],
            'gear': self.control['gear'],
            'manual_gear_shift': self.control['manual_gear_shift'],
        }
        # Only send if control changed from last sent
        if msg == self._last_sent_control:
            return
        self._last_sent_control = msg
        try:
            send_message(self.relay_sock, msg)
        except Exception as e:
            self.log(f"Failed to send control: {e}")

    # -- Data Receiver Thread --

    def _data_recv_thread(self):
        """Background thread: continuously receive sensor_data from relay."""
        self.log("Data receiver thread started")
        try:
            while self.running:
                msg = recv_message(self.relay_sock)
                if msg.get('type') == 'lane_data':
                    with self.data_lock:
                        self.lane_data = msg.get('lane_data')
                    self.log(f"Received lane data: {msg.get('lane_data', {}).get('num_lanes_summary', '?')}")
                    continue
                if msg.get('type') != 'sensor_data':
                    continue

                # Decode image
                b64_image = msg.get('image', '')
                if not b64_image:
                    continue

                img_bgr = decode_image(b64_image)
                if img_bgr is None:
                    continue

                # Convert BGR → RGB for pygame
                img_rgb = img_bgr[:, :, ::-1]
                surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

                                # Compute round-trip delay from server timestamp
                server_recv_time = msg.get('server_recv_time', 0)
                if server_recv_time > 0:
                    computed_delay_ms = (time.time() - server_recv_time) * 1000.0
                else:
                    computed_delay_ms = 0.0

                                # Extract QCNet predictions
                qcnet_data = {}
                if 'qcnet_trajs' in msg:
                    qcnet_data['trajs'] = msg['qcnet_trajs']
                    qcnet_data['probs'] = msg['qcnet_probs']
                    qcnet_data['origin'] = msg.get('qcnet_origin', [0, 0, 0])
                    qcnet_data['agent_origins'] = msg.get('qcnet_agent_origins', None)
                    qcnet_data['prediction_time'] = msg.get('qcnet_prediction_time', 0)

                with self.data_lock:
                    self.latest_image_bgr = img_bgr
                    self.latest_surface = surface
                    self.latest_telemetry = msg
                    # EWMA-smooth delay for stable BEV compensation.
                    # Alpha=0.3: responsive enough to track real changes,
                    # damped enough to avoid jitter from per-frame variance.
                    raw_delay = computed_delay_ms
                    if self._smooth_delay_ms == 0.0:
                        smoothed = raw_delay  # snap on first reading
                    else:
                        alpha = 0.3
                        smoothed = (1 - alpha) * self._smooth_delay_ms + alpha * raw_delay
                    self._smooth_delay_ms = smoothed
                    self.latest_delay_ms = smoothed
                    if qcnet_data:
                        self.latest_qcnet_data = qcnet_data
                    self.total_frames += 1
                    if self.total_frames == 1 and qcnet_data:
                        trajs_shape = f"shape {len(qcnet_data['trajs'])} agents"
                        self.log(f"QCNet data received: {trajs_shape}, "
                                 f"ego probs={[round(p,3) for p in qcnet_data['probs'][0]]}, "
                                 f"origin={qcnet_data.get('origin')}")
                    self.data_fps_tracker.append(time.time())
                    # Keep last 60 entries
                    if len(self.data_fps_tracker) > 60:
                        self.data_fps_tracker = self.data_fps_tracker[-60:]

        except (ConnectionError, OSError) as e:
            self.log(f"Data receiver thread exiting: {e}")
            self.running = False

    # -- Keyboard Input --

    def parse_keys(self, keys, milliseconds):
        """Parse keyboard state, update self.control. Adapted from manual_control.py."""
        # Throttle (server-side hard limit at 10 km/h for QCNet accuracy)
        if keys[K_w] or keys[K_UP]:
            self.control['throttle'] = min(self.control['throttle'] + 0.1, 1.0)
        else:
            self.control['throttle'] = 0.0

        # Brake
        if keys[K_s] or keys[K_DOWN]:
            self.control['brake'] = min(self.control['brake'] + 0.2, 1.0)
        else:
            self.control['brake'] = 0.0

        # Steer
        steer_increment = 5e-4 * milliseconds
        if keys[K_a] or keys[K_LEFT]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_d] or keys[K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control['steer'] = round(self.steer_cache, 1)

        # Hand brake
        self.control['hand_brake'] = bool(keys[K_SPACE])

    def parse_events(self):
        """Parse pygame events. Returns True to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return True
                elif event.key == K_q and pygame.key.get_mods() & KMOD_CTRL:
                    return True
                elif event.key == K_BACKSPACE:
                    # Quick reverse gear toggle via Backspace
                    self.control['reverse'] = not self.control['reverse']
                    self.log(f"Reverse: {'ON' if self.control['reverse'] else 'OFF'}")
                elif event.key == K_F1:
                    self.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    self._print_help()
        return False

    def _print_help(self):
        help_text = """
Remote Driving Operator Controls:
    W/UP      : throttle
    S/DOWN    : brake
    A/LEFT    : steer left
    D/RIGHT   : steer right
    SPACE     : hand-brake
    Backspace : toggle reverse gear
    F1        : toggle HUD
    ESC       : quit
        """
        self.log(help_text)

    # -- Main Loop --

    def run(self):
        self.log("=== Operator Client Starting ===")
        pygame.init()
        pygame.font.init()

        display_w, display_h = CAMERA_WIDTH + BEV_WIDTH, CAMERA_HEIGHT
        display = pygame.display.set_mode((display_w, display_h),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Remote Driving Operator')
        display.fill((0, 0, 0))
        pygame.display.flip()

        self.hud = HUD(CAMERA_WIDTH, display_h)  # HUD only on camera side
        self.bev_renderer = BEVRenderer(width=BEV_WIDTH, height=display_h,
                                        world_range=BEV_RANGE)
        self.clock = pygame.time.Clock()

        try:
            self.connect_relay()

            self.running = True
            recv_thread = threading.Thread(target=self._data_recv_thread, daemon=True)
            recv_thread.start()

            self.log("Entering main loop...")
            while self.running:
                self.clock.tick_busy_loop(60)

                # Handle events
                if self.parse_events():
                    break

                # Parse keyboard input → update control → send
                keys = pygame.key.get_pressed()
                self.parse_keys(keys, self.clock.get_time())
                self.send_control()

                # Render
                with self.data_lock:
                    surface = self.latest_surface
                    telemetry = self.latest_telemetry
                    delay_ms = self.latest_delay_ms
                    qcnet_data = dict(self.latest_qcnet_data)

                if surface is not None:
                    display.blit(surface, (0, 0))
                else:
                    display.fill((20, 20, 20))
                    # Waiting text
                    font = pygame.font.Font(pygame.font.get_default_font(), 24)
                    wait_text = font.render('Waiting for video feed...', True, (255, 255, 255))
                    display.blit(wait_text, (CAMERA_WIDTH // 2 - 100, display_h // 2))

                # Update HUD with telemetry
                if telemetry:
                    self.hud.speed_kmh = telemetry.get('speed_kmh', 0.0)
                    loc = telemetry.get('location', [0, 0, 0])
                    self.hud.location = tuple(loc)
                    self.hud.collision = telemetry.get('collision', False)
                    self.hud.control = dict(self.control)
                    self.hud.total_frames = self.total_frames
                    self.hud.delay_ms = delay_ms

                    # Feed trail to BEV renderer
                    self.bev_renderer.add_trail_point(loc[0], loc[1])

                # Compute data FPS
                with self.data_lock:
                    tracker = list(self.data_fps_tracker)
                if len(tracker) > 1:
                    self.hud.data_fps = (len(tracker) - 1) / max(0.001, tracker[-1] - tracker[0])
                self.hud.display_fps = self.clock.get_fps()

                self.hud.render(display)

                # --- BEV Panel (right side) ---
                # Update lane data in BEV renderer when available
                with self.data_lock:
                    ld = self.lane_data
                if ld is not None and self.bev_renderer.lane_data is None:
                    self.bev_renderer.lane_data = ld
                    self.log("BEV lane data loaded")

                if telemetry and qcnet_data:
                    loc = telemetry.get('location', [0, 0, 0])
                    rot = telemetry.get('rotation', [0, 0, 0])
                    origin = qcnet_data.get('origin', None)
                    agent_origins = qcnet_data.get('agent_origins', None)
                    pred_time = qcnet_data.get('prediction_time', 0)

                    # Build surrounding_positions for BEV markers
                    surrounding = telemetry.get('surrounding', [])
                    surr_pos = [(a['x'], a['y'], a['yaw_rad']) for a in surrounding]

                    # Continuous BEV delay compensation:
                    #   elapsed_since_pred = time since QCNet ran (seconds)
                    #   effective_delay = elapsed_since_pred + measured RTT delay
                    #   step_idx = effective_delay / 50ms_per_step
                    # This makes the BEV move smoothly along the predicted
                    # trajectory every frame, rather than jumping only when
                    # a new QCNet prediction arrives.
                    if pred_time > 0:
                        elapsed_since_pred = time.time() - pred_time
                    else:
                        elapsed_since_pred = 0

                    # USE smoothed delay for BEV compensation (delay_ms
                    # is already EWMA-smoothed from the data thread).
                    # Add 50ms safety margin so BEV leads the camera slightly.
                    # Total includes elapsed time since prediction (continuous
                    # interpolation) + measured RTT delay.
                    effective_delay_ms = (elapsed_since_pred * 1000.0) + delay_ms + 50.0

                    bev_surface = self.bev_renderer.render(
                        ego_x=loc[0],
                        ego_y=loc[1],
                        ego_yaw=math.radians(rot[0]),
                        trajectories=qcnet_data.get('trajs'),
                        probs=qcnet_data.get('probs'),
                        origin=origin,
                        agent_origins=agent_origins,
                        surrounding_positions=surr_pos,
                        delay_ms=effective_delay_ms,
                    )
                    display.blit(bev_surface, (CAMERA_WIDTH, 0))

                    # Periodic BEV smoothness log (every ~5s at 60 FPS)
                    if self.total_frames % 300 == 1:
                        self.log(f"BEV: "
                                 f"{len(qcnet_data.get('trajs', []))} agents, "
                                 f"delay~{delay_ms:.0f}ms, "
                                 f"pred_age~{max(0, elapsed_since_pred*1000):.0f}ms, "
                                 f"eff_step~{effective_delay_ms/50:.1f}, "
                                 f"disp_fps={self.hud.display_fps:.1f}")
                else:
                    # Dark placeholder
                    bev_placeholder = pygame.Surface((BEV_WIDTH, display_h))
                    bev_placeholder.fill((20, 22, 26))
                    # QCNet status
                    if qcnet_data:
                        n_agents = len(qcnet_data.get('trajs', []))
                        status = f'QCNet: {n_agents} agents'
                    else:
                        status = 'QCNet: warming up...'
                    status_text = pygame.font.Font(
                        pygame.font.get_default_font(), 13).render(
                            status, True, (120, 120, 120))
                    bev_placeholder.blit(status_text, (10, 10))
                    display.blit(bev_placeholder, (CAMERA_WIDTH, 0))

                # Separator line
                pygame.draw.line(display, (80, 80, 80),
                                 (CAMERA_WIDTH, 0), (CAMERA_WIDTH, display_h), 2)

                pygame.display.flip()

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
        if self.relay_sock:
            try:
                self.relay_sock.close()
            except Exception:
                pass
        pygame.quit()
        self.log("Operator client stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remote Driving Operator Client')
    parser.add_argument('--relay-host', default=RELAY_HOST, help='Relay server host')
    parser.add_argument('--relay-port', type=int, default=RELAY_PORT, help='Relay server port')
    args = parser.parse_args()

    import config
    config.RELAY_HOST = args.relay_host
    config.RELAY_PORT = args.relay_port

    client = OperatorClient()
    client.run()


if __name__ == '__main__':
    main()