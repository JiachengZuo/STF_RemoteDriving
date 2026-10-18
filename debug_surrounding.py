#!/usr/bin/env python3
"""Debug: check how many surrounding vehicles the vehicle side sees."""
import socket, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message, send_message
from config import RELAY_HOST, RELAY_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RELAY_HOST, RELAY_PORT))
send_message(sock, {'type': 'register', 'role': 'operator'})
print("Listening for surrounding data...")

n = 0
while n < 10:
    msg = recv_message(sock)
    if msg.get('type') != 'sensor_data':
        continue
    surrounding = msg.get('surrounding', [])
    loc = msg.get('location', [0, 0, 0])
    print(f"Frame #{msg.get('frame_id')} ego=({loc[0]:.1f},{loc[1]:.1f}) surrounding={len(surrounding)}", end="")
    if surrounding:
        print(f" IDs={[s['id'] for s in surrounding[:3]]}")
    else:
        print(" (empty)")
    n += 1

sock.close()
