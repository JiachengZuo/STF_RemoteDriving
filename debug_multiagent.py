#!/usr/bin/env python3
"""Debug script: check multi-agent data flow end to end."""
import socket, json, time, sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import recv_message

RELAY_HOST = '127.0.0.1'
RELAY_PORT = 8890

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((RELAY_HOST, RELAY_PORT))
    # Register as operator
    send_msg = json.dumps({'type': 'register', 'role': 'operator'})
    sock.sendall((send_msg + '\n').encode())

    frames_checked = 0
    for _ in range(200):
        msg = recv_message(sock)
        if msg.get('type') != 'sensor_data':
            continue

        frames_checked += 1
        surrounding = msg.get('surrounding', [])

        if frames_checked <= 3 or frames_checked % 50 == 0:
            print(f"\n=== Frame {frames_checked} (ID={msg.get('frame_id')}) ===")
            print(f"  Ego: loc={msg.get('location')}, vel={msg.get('velocity')}")
            print(f"  Surrounding agents: {len(surrounding)}")
            for a in surrounding[:5]:
                print(f"    ID={a['id']:5d}  pos=({a['x']:8.2f}, {a['y']:8.2f})  "
                      f"vel=({a['vx']:6.2f}, {a['vy']:6.2f})  dist={a['dist']:5.1f}m")
            if len(surrounding) > 5:
                print(f"    ... and {len(surrounding)-5} more")

            # Check QCNet data
            if 'qcnet_trajs' in msg:
                trajs = msg['qcnet_trajs']
                probs = msg['qcnet_probs']
                n_agents = len(trajs)
                print(f"  QCNet: {n_agents} agents' predictions")
                # Agent 0 (ego): mode1 trajectory first 5 points
                if len(trajs) > 0:
                    m1 = trajs[0][1]  # mode 1 = 2nd mode
                    print(f"    Ego mode1 first 3 pts: {m1[:3]}")
                    print(f"    Ego mode1 last 3 pts: {m1[-3:]}")
                # Agent 1 (surrounding): mode1 trajectory
                if len(trajs) > 1:
                    m1 = trajs[1][1]
                    print(f"    Agent1 mode1 first 3 pts: {m1[:3]}")
                    print(f"    Agent1 mode1 last 3 pts: {m1[-3:]}")
            else:
                print(f"  QCNet: not yet available")

    sock.close()
    print(f"\nChecked {frames_checked} frames")

if __name__ == '__main__':
    main()
