#!/usr/bin/env python3
"""Debug: dump surrounding + QCNet details for multiple frames."""
import socket, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import RELAY_HOST, RELAY_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RELAY_HOST, RELAY_PORT))
send_message(sock, {'type': 'register', 'role': 'operator'})
print("Registered, collecting 5 QCNet frames...")

n = 0
while n < 5:
    msg = recv_message(sock)
    if msg.get('type') != 'sensor_data':
        continue
    if 'qcnet_trajs' not in msg:
        continue

    trajs = msg['qcnet_trajs']
    probs = msg['qcnet_probs']
    origins = msg.get('qcnet_agent_origins', [])
    surrounding = msg.get('surrounding', [])
    loc = msg.get('location', [0, 0, 0])
    server_recv_time = msg.get('server_recv_time', 0)
    delay = (time.time() - server_recv_time) * 1000 if server_recv_time else 0

    print(f"\nFrame #{n+1} #{msg.get('frame_id')} | delay={delay:.0f}ms | ego=({loc[0]:.1f},{loc[1]:.1f})")
    print(f"  QCNet: {len(trajs)} agents, origins={len(origins) if origins else 0}")
    print(f"  sensor surrounding: {len(surrounding)} vehicles")
    for a in range(len(trajs)):
        best = max(range(len(probs[a])), key=lambda i: probs[a][i])
        has_origin = origins and a < len(origins)
        print(f"    Agent[{a}]: mode={best} prob={probs[a][best]:.3f} origin={'YES' if has_origin else 'NO'}")
    n += 1
    time.sleep(0.3)

sock.close()
print("\nDone.")
