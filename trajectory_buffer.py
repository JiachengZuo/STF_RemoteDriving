#!/usr/bin/env python3
# Remote Driving Framework - Trajectory Buffer
# Maintains a rolling window of vehicle states for QCNet inference.
# Supports multiple agents: ego + surrounding vehicles.
# 50 historical steps + 60 future placeholder = 110 total timesteps.

import math
import torch
import numpy as np

NUM_HISTORICAL_STEPS = 50
NUM_FUTURE_STEPS = 60
NUM_TOTAL_STEPS = NUM_HISTORICAL_STEPS + NUM_FUTURE_STEPS  # 110


class TrajectoryBuffer:
    """Rolling buffer of vehicle states: position, heading, velocity.

    QCNet needs at least 30 valid historical steps (num_t2m_steps=30)
    for temporal attention in the decoder. Buffer starts filling from
    the moment the vehicle starts streaming.

    Multi-agent: ego (index 0) + N surrounding vehicles (indices 1..N).
    """

    def __init__(self):
        self._count = 0  # how many steps have been appended
        self._ready = False

        # Ego agent data (always present)
        self.ego_positions = torch.zeros(NUM_TOTAL_STEPS, 2)
        self.ego_headings = torch.zeros(NUM_TOTAL_STEPS)
        self.ego_velocities = torch.zeros(NUM_TOTAL_STEPS, 2)
        self.ego_valid_mask = torch.zeros(NUM_TOTAL_STEPS, dtype=torch.bool)
        self.ego_predict_mask = torch.zeros(NUM_TOTAL_STEPS, dtype=torch.bool)
        self.ego_predict_mask[NUM_HISTORICAL_STEPS:] = True

        # Surrounding agents: dict of agent_id -> {positions, headings, velocities, valid_mask}
        # agent_id is CARLA actor id (int) for stable tracking across ticks
        self.surrounding_agents = {}  # agent_id -> dict
        self._agent_order = []        # list of agent_ids in insertion order

    def _new_agent_buffers(self):
        """Create fresh buffers for a new surrounding agent."""
        buf = {
            'positions': torch.zeros(NUM_TOTAL_STEPS, 2),
            'headings': torch.zeros(NUM_TOTAL_STEPS),
            'velocities': torch.zeros(NUM_TOTAL_STEPS, 2),
            'valid_mask': torch.zeros(NUM_TOTAL_STEPS, dtype=torch.bool),
            'predict_mask': torch.zeros(NUM_TOTAL_STEPS, dtype=torch.bool),
        }
        # Mark future steps (50..109) as predicted — same as ego.
        # Without this, QCNet outputs garbage/uniform filler for NPCs.
        buf['predict_mask'][NUM_HISTORICAL_STEPS:] = True
        return buf

    def _shift_buffers(self, buffers_dict):
        """Shift all buffers in a dict left by one timestep."""
        for key in ['positions', 'headings', 'velocities', 'valid_mask']:
            buf = buffers_dict[key]
            buf[:-1] = buf[1:].clone()

    def _append_step(self, buffers_dict, x, y, yaw_rad, vx, vy):
        """Insert new data at the last historical slot (index 49)."""
        new_idx = NUM_HISTORICAL_STEPS - 1
        buffers_dict['positions'][new_idx] = torch.tensor([x, y])
        buffers_dict['headings'][new_idx] = yaw_rad
        buffers_dict['velocities'][new_idx] = torch.tensor([vx, vy])
        buffers_dict['valid_mask'][new_idx] = True
        # vector_repr convention: step 0 is always invalid
        buffers_dict['valid_mask'][0] = False

    def append(self, ego_x, ego_y, ego_yaw_rad, ego_vx, ego_vy,
               surrounding=None):
        """Push a new observation into the buffer (FIFO shift).

        Args:
            ego_x, ego_y: ego world coordinates (meters)
            ego_yaw_rad: ego heading angle (radians)
            ego_vx, ego_vy: ego velocity components in world frame (m/s)
            surrounding: list of dicts for surrounding agents, each with:
                id: CARLA actor id (int, used as stable key)
                x, y, yaw_rad, vx, vy: same as ego
                If None or empty, no surrounding agents this tick.
        """
        # Shift ego buffers
        self._shift_buffers({'positions': self.ego_positions,
                             'headings': self.ego_headings,
                             'velocities': self.ego_velocities,
                             'valid_mask': self.ego_valid_mask})
        self._append_step({'positions': self.ego_positions,
                           'headings': self.ego_headings,
                           'velocities': self.ego_velocities,
                           'valid_mask': self.ego_valid_mask},
                          ego_x, ego_y, ego_yaw_rad, ego_vx, ego_vy)

        # Track which agents we see this tick for cleanup
        seen_ids = set()
        if surrounding:
            for agent in surrounding:
                agent_id = agent['id']
                seen_ids.add(agent_id)

                if agent_id not in self.surrounding_agents:
                    self.surrounding_agents[agent_id] = self._new_agent_buffers()
                    self._agent_order.append(agent_id)

                buffers = self.surrounding_agents[agent_id]
                self._shift_buffers(buffers)
                self._append_step(buffers,
                                  agent['x'], agent['y'], agent['yaw_rad'],
                                  agent['vx'], agent['vy'])

        # Shift and invalidate agents not seen this tick
        gone_ids = set(self.surrounding_agents.keys()) - seen_ids
        for agent_id in gone_ids:
            buffers = self.surrounding_agents[agent_id]
            self._shift_buffers(buffers)
            # No append — the new step is invalid (zero-filled from shift)
            # Track how long since last seen; drop if gone too long
            valid_count = buffers['valid_mask'][:NUM_HISTORICAL_STEPS].sum().item()
            if valid_count < 5:
                del self.surrounding_agents[agent_id]
                if agent_id in self._agent_order:
                    self._agent_order.remove(agent_id)

        # Cap at 20 surrounding agents to keep inference fast
        while len(self._agent_order) > 20:
            oldest_id = self._agent_order.pop(0)
            del self.surrounding_agents[oldest_id]

        self._count += 1

        # Ready when we have enough valid historical steps for ego
        if not self._ready and self.valid_count() >= 30:
            self._ready = True

    def valid_count(self):
        """Number of valid historical steps for ego."""
        return self.ego_valid_mask[:NUM_HISTORICAL_STEPS].sum().item()

    def is_ready(self):
        """Buffer has enough data for QCNet inference."""
        return self._ready

    def num_agents(self):
        """Total number of agents: 1 (ego) + N (surrounding)."""
        return 1 + len(self._agent_order)

    def origin(self):
        """Get origin (position, yaw) at the last historical step for ego."""
        idx = NUM_HISTORICAL_STEPS - 1
        return (
            self.ego_positions[idx, 0].item(),
            self.ego_positions[idx, 1].item(),
            self.ego_headings[idx].item(),
        )

    def agent_origins(self):
        """Get origins for ALL agents (ego + surrounding).

        Returns list of (x, y, yaw) tuples, one per agent.
        Index 0 = ego, indices 1..N = surrounding agents.
        """
        idx = NUM_HISTORICAL_STEPS - 1
        origins = [(
            self.ego_positions[idx, 0].item(),
            self.ego_positions[idx, 1].item(),
            self.ego_headings[idx].item(),
        )]
        for agent_id in self._agent_order:
            buf = self.surrounding_agents[agent_id]
            origins.append((
                buf['positions'][idx, 0].item(),
                buf['positions'][idx, 1].item(),
                buf['headings'][idx].item(),
            ))
        return origins

    def to_hetero_data(self):
        """Build a minimal HeteroData for QCNet inference.

        Supports N agents: ego at index 0, surrounding at 1..N.
        Returns a dict that can be pickled and sent to the QCNet process.
        The QCNet process reconstructs it as torch_geometric.data.HeteroData.
        """
        # Build agent data: ego first, then surrounding in insertion order
        all_positions = []
        all_headings = []
        all_velocities = []
        all_valid_mask = []
        all_predict_mask = []
        all_ids = []
        all_types = []
        all_categories = []

        # Ego (index 0)
        all_positions.append(self.ego_positions)
        all_headings.append(self.ego_headings)
        all_velocities.append(self.ego_velocities)
        all_valid_mask.append(self.ego_valid_mask)
        all_predict_mask.append(self.ego_predict_mask.clone())
        all_ids.append('carla_ego')
        all_types.append(0)   # vehicle
        all_categories.append(3)  # FOCAL_TRACK

        # Surrounding agents
        for agent_id in self._agent_order:
            buf = self.surrounding_agents[agent_id]
            all_positions.append(buf['positions'])
            all_headings.append(buf['headings'])
            all_velocities.append(buf['velocities'])
            all_valid_mask.append(buf['valid_mask'])
            all_predict_mask.append(buf['predict_mask'].clone())
            all_ids.append(f'carla_{agent_id}')
            all_types.append(0)
            all_categories.append(3)

        num_agents = len(all_positions)

        agent_data = {
            'num_nodes': num_agents,
            'av_index': 0,  # ego is always index 0
            'valid_mask': torch.stack(all_valid_mask),      # [A, 110]
            'predict_mask': torch.stack(all_predict_mask),   # [A, 110]
            'id': all_ids,
            'type': torch.tensor(all_types, dtype=torch.uint8),
            'category': torch.tensor(all_categories, dtype=torch.uint8),
            'position': torch.stack(all_positions),          # [A, 110, 2]
            'heading': torch.stack(all_headings),             # [A, 110]
            'velocity': torch.stack(all_velocities),          # [A, 110, 2]
        }

        # Empty map data (QCNet handles this gracefully)
        map_data = {
            'map_polygon': {
                'num_nodes': 0,
                'position': torch.zeros(0, 2),
                'orientation': torch.zeros(0),
                'type': torch.zeros(0, dtype=torch.uint8),
                'is_intersection': torch.zeros(0, dtype=torch.uint8),
            },
            'map_point': {
                'num_nodes': 0,
                'position': torch.zeros(0, 2),
                'orientation': torch.zeros(0),
                'magnitude': torch.zeros(0),
                'type': torch.zeros(0, dtype=torch.uint8),
                'side': torch.zeros(0, dtype=torch.uint8),
            },
            'edges': {
                'pt2pl': torch.zeros(2, 0, dtype=torch.long),
                'pl2pl': torch.zeros(2, 0, dtype=torch.long),
                'pl2pl_type': torch.zeros(0, dtype=torch.uint8),
            },
        }

        return {'agent': agent_data, 'map': map_data}
