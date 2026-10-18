#!/usr/bin/env python3
"""Debug: inspect trajectory points for all agents."""
import socket, sys, os, time, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import RELAY_HOST, RELAY_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RELAY_HOST, RELAY_PORT))
send_message(sock, {'type': 'register', 'role': 'operator'})
print("Waiting for QCNet frame...")

while True:
    msg = recv_message(sock)
    if msg.get('type') != 'sensor_data' or 'qcnet_trajs' not in msg:
        continue
    
    trajs = msg['qcnet_trajs']
    probs = msg['qcnet_probs']
    origins = msg.get('qcnet_agent_origins', [])
    loc = msg.get('location', [0,0,0])
    surrounding = msg.get('surrounding', [])
    
    print(f"\nFrame #{msg['frame_id']} ego=({loc[0]:.1f},{loc[1]:.1f})")
    print(f"  QCNet: {len(trajs)} agents, {len(surrounding)} sensor surr, origins={len(origins)}")
    
    for a in range(min(len(trajs), 15)):
        t = np.asarray(trajs[a])  # [6, 60, 2]
        p = np.asarray(probs[a])  # [6]
        best = int(np.argmax(p))
        traj = t[best]  # [60, 2]
        
        # Sample key points
        pts = {}
        for step in [0, 20, 40, 59]:
            pts[step] = (round(traj[step][0], 3), round(traj[step][1], 3))
        
        # Compute range of trajectory
        x_vals = traj[:, 0]
        y_vals = traj[:, 1]
        x_range = round(float(x_vals.max() - x_vals.min()), 3)
        y_range = round(float(y_vals.max() - y_vals.min()), 3)
        
        role = "EGO" if a == 0 else f"NPC{a}"
        ori = f"({origins[a][0]:.1f},{origins[a][1]:.1f})" if a < len(origins) else "NONE"
        print(f"  [{role}] mode={best} prob={p[best]:.3f} origin={ori}")
        print(f"         pts: t0={pts[0]} t20={pts[20]} t40={pts[40]} t59={pts[59]}")
        print(f"         range: dx={x_range} dy={y_range}")
    
    break

sock.close()
print("Done.")
