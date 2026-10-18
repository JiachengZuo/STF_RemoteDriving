#!/usr/bin/env python3
# Remote Driving Framework - BEV Renderer
# Renders a bird's-eye view with ego vehicle and predicted trajectories
# using pygame. Designed for side-by-side display in operator client.

import math
import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError('pygame is required. Install with: pip install pygame')


class BEVRenderer:
    """Renders a top-down BEV view of ego vehicle + QCNet predictions.

    Coordinate mapping (CARLA → screen):
        x (forward) → y_up on screen (inverted: -x → up)
        y (right)  → x_right on screen
    """

    def __init__(self, width=500, height=640, world_range=80.0):
        self.width = width
        self.height = height
        self.range = world_range  # meters from center to edge
        self.scale = self.height / (2 * self.range)  # px per meter
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # Font for labels
        self._font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self._font_small = pygame.font.Font(pygame.font.get_default_font(), 10)

        # Rolling history trail
        self._trail = []  # list of (x, y) world coords, max 50

        # Cached lane data (updated externally)
        self.lane_data = None

    def add_trail_point(self, wx, wy):
        """Add a point to the historical trail."""
        self._trail.append((wx, wy))
        if len(self._trail) > 50:
            self._trail = self._trail[-50:]

    def _world_to_pixel(self, wx, wy, ego_x, ego_y, ego_yaw):
        """Convert world coordinate to BEV pixel.

        First translate to ego-centric, then rotate so ego forward = up.
        """
        # Translate
        dx = wx - ego_x
        dy = wy - ego_y

        # Rotate: CARLA yaw=0 = +X. Forward = (cos(yaw), sin(yaw)).
        # Screen: up=forward, right=right.
        # Inverse rotation R(-yaw) to get ego-relative forward/right components.
        cos_yaw = math.cos(-ego_yaw)   # = cos(yaw)
        sin_yaw = math.sin(-ego_yaw)   # = -sin(yaw)
        rx = dx * cos_yaw - dy * sin_yaw  # rotated forward
        ry = dx * sin_yaw + dy * cos_yaw  # rotated right

        # Scale and center
        px = self.center_x + int(ry * self.scale)   # lateral → x
        py = self.center_y - int(rx * self.scale)   # forward → y (up = negative)
        return px, py

    def render(self, ego_x, ego_y, ego_yaw, trajectories=None, probs=None,
               origin=None, agent_origins=None, surrounding_positions=None,
               delay_ms=0):
        """Render the full BEV scene with multi-agent predictions.

        Uses QCNet max-prob trajectory to compensate for network delay by
        shifting the BEV center forward to the predicted current position.

        Args:
            ego_x, ego_y: ego delayed position (meters) from sensor_data
            ego_yaw: ego heading (radians)
            trajectories: list of [6, 60, 2] — QCNet predictions per agent
            probs: list of [6] — mode probabilities per agent
            origin: ego-only origin (backward compat)
            agent_origins: list of (ox, oy, oyaw) per agent
            surrounding_positions: list of (wx, wy, yaw) for markers
            delay_ms: measured round-trip delay. Used to look up the
                      corresponding QCNet prediction step and shift the
                      BEV center forward (delay compensation).

        Returns:
            pygame.Surface
        """
        surf = pygame.Surface((self.width, self.height))
        surf.fill((25, 28, 32))  # dark charcoal background

        # --- Grid ---
        for i in range(-int(self.range), int(self.range) + 1, 10):
            alpha = 80 if i == 0 else 40
            color = (50, 60, 70, alpha)
            # Horizontal grid line (constant forward distance)
            y_px = self.center_y - int(i * self.scale)
            if 0 <= y_px < self.height:
                pygame.draw.line(surf, color[:3], (0, y_px), (self.width, y_px), 1)
            # Vertical grid line (constant lateral distance)
            x_px = self.center_x + int(i * self.scale)
            if 0 <= x_px < self.width:
                pygame.draw.line(surf, color[:3], (x_px, 0), (x_px, self.height), 1)

        # --- Delay compensation: predict ego position at now+delay ---
        # This shifts the entire BEV forward so the operator sees where
        # the vehicle SHOULD be now, not the delayed position.
        if trajectories is not None and len(trajectories) > 0 and delay_ms > 0:
            ego_trajs = np.asarray(trajectories[0])
            ego_probs = np.asarray(probs[0])
            best_mode = int(np.argmax(ego_probs))
            ego_traj = ego_trajs[best_mode]

            # Continuous interpolation: effective delay includes both
            # measured RTT delay AND elapsed time since prediction was made.
            # This makes the BEV move smoothly along the trajectory every
            # frame, rather than jumping only when a new prediction arrives.
            raw_step = delay_ms / 50.0
            if raw_step < 59.0:
                step_idx = int(raw_step)
                ego_pred_x = float(ego_traj[step_idx, 0])
                ego_pred_y = float(ego_traj[step_idx, 1])
            else:
                # Extrapolate beyond trajectory using average velocity
                # from last 5 trajectory points (linear extrapolation).
                last_pts = ego_traj[54:60]  # indices 54..59
                d = last_pts[1:] - last_pts[:-1]  # [5, 2] per-step deltas
                avg_vel = d.mean(axis=0)  # [2] average per-step velocity
                overshoot = raw_step - 59.0
                ego_pred_x = float(ego_traj[59, 0] + avg_vel[0] * overshoot)
                ego_pred_y = float(ego_traj[59, 1] + avg_vel[1] * overshoot)

            if agent_origins is not None and len(agent_origins) > 0:
                ref_x, ref_y, ref_yaw = agent_origins[0]
            elif origin is not None:
                ref_x, ref_y, ref_yaw = origin
            else:
                ref_x, ref_y, ref_yaw = ego_x, ego_y, ego_yaw
            # CARLA yaw=0 = +X direction. Standard rotation R(yaw) = [[cos,-sin],[sin,cos]].
            # QCNet agent-centric: x=forward, y=right.
            c = math.cos(ref_yaw)
            s = math.sin(ref_yaw)
            display_ego_x = ref_x + ego_pred_x * c - ego_pred_y * s
            display_ego_y = ref_y + ego_pred_x * s + ego_pred_y * c
        else:
            display_ego_x = ego_x
            display_ego_y = ego_y

        # --- Range rings ---
        for r in [10, 20, 30, 40]:
            color = (40, 45, 50)
            pygame.draw.circle(surf, color, (self.center_x, self.center_y),
                               int(r * self.scale), 1)

        # --- Historical trail ---
        if len(self._trail) >= 2:
            pts = [self._world_to_pixel(wx, wy, display_ego_x, display_ego_y, ego_yaw)
                   for wx, wy in self._trail[-30:]]
            for i in range(len(pts) - 1):
                alpha = int(40 + 100 * i / len(pts))
                color = (0, alpha, alpha)  # teal fading
                pygame.draw.line(surf, color, pts[i], pts[i + 1], 2)

        # --- Lane lines ---
        lane_data = self.lane_data
        if lane_data is not None:
            for i, pts in enumerate(lane_data.get('centerlines', [])):
                lane_type = lane_data.get('lane_types', ['Driving'] * len(
                    lane_data.get('centerlines', [])))
                ltype = lane_type[i] if i < len(lane_type) else 'Driving'
                if ltype == 'Driving':
                    color = (80, 80, 60)
                elif ltype == 'Sidewalk':
                    color = (50, 50, 55)
                elif ltype == 'Shoulder':
                    color = (40, 45, 40)
                else:
                    color = (45, 45, 45)

                px_pts = []
                for wx, wy in pts:
                    px, py = self._world_to_pixel(wx, wy, display_ego_x,
                                                  display_ego_y, ego_yaw)
                    if 0 <= px < self.width and 0 <= py < self.height:
                        px_pts.append((px, py))
                if len(px_pts) >= 2:
                    pygame.draw.lines(surf, color, False, px_pts, 1)

            for pts in lane_data.get('boundaries', []):
                px_pts = []
                for wx, wy in pts:
                    px, py = self._world_to_pixel(wx, wy, display_ego_x,
                                                  display_ego_y, ego_yaw)
                    if 0 <= px < self.width and 0 <= py < self.height:
                        px_pts.append((px, py))
                if len(px_pts) >= 2:
                    pygame.draw.lines(surf, (60, 60, 60), False, px_pts, 1)

        # --- Predicted trajectories (QCNet, max-prob mode only per agent) ---
        if trajectories is not None and probs is not None:
            num_agents = len(trajectories)

            for agent_idx in range(num_agents):
                trajs = np.asarray(trajectories[agent_idx])  # [6, 60, 2]
                probs_arr = np.asarray(probs[agent_idx])     # [6]

                # Pick max-prob mode only
                best_mode = int(np.argmax(probs_arr))
                prob = probs_arr[best_mode]
                traj = trajs[best_mode]  # [60, 2]

                # Each agent's prediction is in agent-centric coords relative
                # to ITS OWN origin (position + heading at last historical step)
                if agent_origins is not None and agent_idx < len(agent_origins):
                    ref_x, ref_y, ref_yaw = agent_origins[agent_idx]
                elif origin is not None and agent_idx == 0:
                    ref_x, ref_y, ref_yaw = origin
                elif agent_idx == 0:
                    ref_x, ref_y, ref_yaw = ego_x, ego_y, ego_yaw
                else:
                    continue  # skip: no origin data for this surrounding agent

                cos_yaw = math.cos(ref_yaw)
                sin_yaw = math.sin(ref_yaw)

                # Convert agent-centric to world: standard rotation R(yaw).
                world_pts = []
                for step in range(0, 60, 2):  # subsample: every 2nd step
                    ax, ay = traj[step]  # agent-centric displacement
                    wx = ref_x + ax * cos_yaw - ay * sin_yaw
                    wy = ref_y + ax * sin_yaw + ay * cos_yaw
                    world_pts.append((wx, wy))

                # Convert to screen pixels (centered on display_ego)
                px_pts = [self._world_to_pixel(wx, wy, display_ego_x,
                                               display_ego_y, ego_yaw)
                          for wx, wy in world_pts]

                if len(px_pts) >= 2:
                    if agent_idx == 0:
                        # Ego: bright cyan
                        color = (0, 220, 255)
                        width = 3
                    else:
                        # Surrounding agents: orange
                        color = (255, 140, 30)
                        width = 1
                    pygame.draw.lines(surf, color, False, px_pts, width=width)

                    # Draw probability label at end (ego only)
                    if agent_idx == 0 and px_pts[-1][0] is not None:
                        lbl = self._font_small.render(f'{prob:.2f}', True, color)
                        surf.blit(lbl, (px_pts[-1][0] + 3, px_pts[-1][1] - 6))

                    # --- Ego waypoint guides (1-second intervals) ---
                    # Show where ego should be 1s, 2s, 3s from now.
                    # Helps operator navigate even if camera feed freezes.
                    if agent_idx == 0:
                        # Compute waypoints at 1s intervals without subsampling
                        full_world = []
                        for step in range(60):
                            ax, ay = traj[step]
                            wx = ref_x + ax * cos_yaw - ay * sin_yaw
                            wy = ref_y + ax * sin_yaw + ay * cos_yaw
                            full_world.append((wx, wy))

                        for sec in [1, 2, 3]:
                            step = sec * 20  # 20 steps/s at 0.05s per step
                            if step >= 60:
                                break
                            wpx, wpy = full_world[step]
                            spx, spy = self._world_to_pixel(
                                wpx, wpy, display_ego_x, display_ego_y, ego_yaw)
                            if 0 <= spx < self.width and 0 <= spy < self.height:
                                # Fading circles: closer=brighter
                                alpha_val = max(40, 200 - sec * 60)
                                r = int(4 + sec * 1.5)  # radius grows with distance
                                pygame.draw.circle(
                                    surf, (0, 255, 140, alpha_val),
                                    (spx, spy), r, 2)
                                # Time label
                                t_lbl = self._font_small.render(
                                    f'{sec}s', True, (0, 255, 140, alpha_val))
                                surf.blit(t_lbl, (spx + r + 2, spy - 6))

        # --- Surrounding agent position markers (boxes) ---
        if surrounding_positions:
            for wx, wy, wyaw in surrounding_positions:
                px, py = self._world_to_pixel(wx, wy, display_ego_x,
                                              display_ego_y, ego_yaw)
                if 0 <= px < self.width and 0 <= py < self.height:
                    # Box: w=lateral(side-to-side), h=longitudinal(front-to-rear)
                    bw, bh = 5, 8
                    box_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
                    box_surf.fill((255, 160, 50))
                    pygame.draw.rect(box_surf, (0, 0, 0, 128), box_surf.get_rect(), 1)
                    # Convert CARLA yaw to screen rotation:
                    # CARLA: yaw=0→east(+x→screenUP), clockwise positive
                    # pygame: rotate ccw. Screen-relative heading = wyaw - ego_yaw
                    # py_angle = -(screen_heading_deg) = degrees(ego_yaw - wyaw)
                    angle = math.degrees(ego_yaw - wyaw)
                    rotated = pygame.transform.rotate(box_surf, angle)
                    rot_rect = rotated.get_rect(center=(px, py))
                    surf.blit(rotated, rot_rect)

        
        # --- Ego vehicle marker (box, centered on screen) ---
        ego_w, ego_h = 8, 12
        ego_box = pygame.Surface((ego_w, ego_h), pygame.SRCALPHA)
        ego_box.fill((0, 255, 100))
        pygame.draw.rect(ego_box, (0, 0, 0, 128), ego_box.get_rect(), 1)
        surf.blit(ego_box, ego_box.get_rect(center=(self.center_x, self.center_y)))

        # --- Scale legend ---
        legend_y = self.height - 30
        bar_start_x = 10
        bar_len = int(10 * self.scale)  # 10m bar
        pygame.draw.line(surf, (150, 150, 150),
                         (bar_start_x, legend_y),
                         (bar_start_x + bar_len, legend_y), 2)
        lbl = self._font_small.render('10m', True, (150, 150, 150))
        surf.blit(lbl, (bar_start_x + bar_len + 4, legend_y - 6))

        # --- Title ---
        title = self._font.render('QCNet BEV', True, (140, 140, 140))
        surf.blit(title, (4, 4))

        return surf
