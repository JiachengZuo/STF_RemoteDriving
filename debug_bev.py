#!/usr/bin/env python3
"""Headless BEV debug: connect, render one frame, save to PNG."""
import socket, sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import RELAY_HOST, RELAY_PORT, BEV_WIDTH, BEV_RANGE, CAMERA_HEIGHT
from bev_renderer import BEVRenderer

os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.init()
pygame.font.init()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RELAY_HOST, RELAY_PORT))
send_message(sock, {'type': 'register', 'role': 'operator'})
print("Registered, waiting for QCNet data...")

renderer = BEVRenderer(width=BEV_WIDTH, height=CAMERA_HEIGHT, world_range=BEV_RANGE)

while True:
    msg = recv_message(sock)
    if msg.get('type') == 'lane_data':
        renderer.lane_data = msg.get('lane_data')
        print("Lane data loaded")
        continue
    if msg.get('type') != 'sensor_data' or 'qcnet_trajs' not in msg:
        continue

    trajs = msg['qcnet_trajs']
    probs = msg['qcnet_probs']
    origins = msg.get('qcnet_agent_origins', None)
    origin = msg.get('qcnet_origin', [0, 0, 0])
    loc = msg.get('location', [0, 0, 0])
    rot = msg.get('rotation', [0, 0, 0])
    surrounding = msg.get('surrounding', [])
    surr_pos = [(a['x'], a['y'], a['yaw_rad']) for a in surrounding]
    server_recv_time = msg.get('server_recv_time', 0)
    delay_ms = (time.time() - server_recv_time) * 1000 if server_recv_time else 0

    print(f"\nFrame #{msg.get('frame_id')}")
    print(f"  agents: {len(trajs)} trajs, {len(origins) if origins else 'None'} origins")
    print(f"  ego_at: ({loc[0]:.1f}, {loc[1]:.1f}), yaw={rot[0]:.1f}")
    print(f"  delay_ms: {delay_ms:.0f}")
    print(f"  surrounding: {len(surrounding)} vehicles")

    for a in range(len(trajs)):
        best = max(range(len(probs[a])), key=lambda i: probs[a][i])
        has_origin = "YES" if origins and a < len(origins) else "MISSING"
        print(f"  Agent[{a}]: mode={best} prob={probs[a][best]:.3f} origin={has_origin}")
        if origins and a < len(origins):
            o = origins[a]
            print(f"           ref=({o[0]:.1f},{o[1]:.1f}) yaw={o[2]:.3f}")
            traj_best = trajs[a][best]
            print(f"           traj[0]=({traj_best[0][0]:.2f},{traj_best[0][1]:.2f}) traj[59]=({traj_best[59][0]:.2f},{traj_best[59][1]:.2f})")

    # Render
    bev = renderer.render(
        ego_x=loc[0], ego_y=loc[1], ego_yaw=math.radians(rot[0]),
        trajectories=trajs, probs=probs,
        origin=origin, agent_origins=origins,
        surrounding_positions=surr_pos, delay_ms=delay_ms + 50,
    )

    pygame.image.save(bev, '/tmp/bev_debug.png')
    print(f"\n  Saved BEV to /tmp/bev_debug.png")
    break

sock.close()
print("Done.")
