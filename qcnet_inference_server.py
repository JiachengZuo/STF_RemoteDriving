#!/usr/bin/env python3
# Remote Driving Framework - QCNet Inference Server
# Standalone process running in QCNet conda env.
# Binds TCP socket, accepts one connection at a time,
# receives HeteroData (pickle), runs inference, sends results back.

import sys
import os
import pickle
import struct
import socket
import time
import argparse
import threading

import torch
from torch_geometric.data import HeteroData


def load_model(checkpoint_path):
    """Load QCNet model from checkpoint."""
    qcnet_root = os.environ.get('QCNet_ROOT',
                                os.path.join(os.path.dirname(__file__),
                                             '../../../QCNet'))
    sys.path.insert(0, qcnet_root)

    from predictors import QCNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QCNet.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model, device


def build_hetero_data(data_dict):
    """Reconstruct HeteroData from the dict sent by relay."""
    agent = data_dict['agent']
    mp = data_dict['map']

    data = HeteroData()

    # Agent node
    data['agent'].num_nodes = agent['num_nodes']
    data['agent'].av_index = agent['av_index']
    data['agent'].valid_mask = agent['valid_mask']
    data['agent'].predict_mask = agent['predict_mask']
    data['agent'].id = agent['id']
    data['agent'].type = agent['type']
    data['agent'].category = agent['category']
    data['agent'].position = agent['position']
    data['agent'].heading = agent['heading']
    data['agent'].velocity = agent['velocity']

    # Map polygon
    mp_poly = mp['map_polygon']
    data['map_polygon'].num_nodes = mp_poly['num_nodes']
    data['map_polygon'].position = mp_poly['position']
    data['map_polygon'].orientation = mp_poly['orientation']
    data['map_polygon'].type = mp_poly['type']
    data['map_polygon'].is_intersection = mp_poly['is_intersection']

    # Map point
    mp_pt = mp['map_point']
    data['map_point'].num_nodes = mp_pt['num_nodes']
    data['map_point'].position = mp_pt['position']
    data['map_point'].orientation = mp_pt['orientation']
    data['map_point'].magnitude = mp_pt['magnitude']
    data['map_point'].type = mp_pt['type']
    data['map_point'].side = mp_pt['side']

    # Map edges
    edges = mp['edges']
    data['map_point', 'to', 'map_polygon'].edge_index = edges['pt2pl']
    data['map_polygon', 'to', 'map_polygon'].edge_index = edges['pl2pl']
    data['map_polygon', 'to', 'map_polygon'].type = edges['pl2pl_type']

    return data


def recv_exact(sock, n):
    """Receive exactly n bytes."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_pickle(sock):
    """Receive a pickle object with 4-byte length prefix."""
    raw_len = recv_exact(sock, 4)
    if raw_len is None:
        return None
    data_len = struct.unpack('>I', raw_len)[0]
    data = recv_exact(sock, data_len)
    if data is None:
        return None
    return pickle.loads(data)


def send_pickle(sock, obj):
    """Send a pickle object with 4-byte length prefix."""
    data = pickle.dumps(obj)
    sock.sendall(struct.pack('>I', len(data)) + data)


def main():
    parser = argparse.ArgumentParser(description='QCNet Inference Server')
    parser.add_argument('--port', type=int, default=8891,
                        help='TCP server port')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/djhuai/zuo/stf_二期/QCNet/checkpoints/QCNet_AV2.ckpt',
                        help='Path to QCNet checkpoint')
    args = parser.parse_args()

    print(f"[QCNet Server] Loading model from {args.checkpoint}...", flush=True)
    t0 = time.time()
    model, device = load_model(args.checkpoint)
    print(f"[QCNet Server] Model loaded in {time.time() - t0:.1f}s on {device}", flush=True)

    # Bind TCP server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('127.0.0.1', args.port))
    server_sock.listen(5)
    server_sock.settimeout(1.0)
    print(f"[QCNet Server] READY on 127.0.0.1:{args.port}", flush=True)

    request_count = 0
    running = True
    try:
        while running:
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue

            t_start = time.time()
            try:
                # Receive HeteroData dict
                data_dict = recv_pickle(conn)
                if data_dict is None:
                    conn.close()
                    continue

                request_count += 1

                # Build HeteroData and run inference
                data = build_hetero_data(data_dict)
                data = data.to(device)

                with torch.no_grad():
                    pred = model(data)

                # Extract predictions for ALL agents
                # pred['loc_refine_pos']: [A, 6, 60, 2], pred['pi']: [A, 6]
                loc_refine_pos = pred['loc_refine_pos']   # [A, 6, 60, 2]
                pi_logits = pred['pi']                    # [A, 6]
                pi_probs = torch.softmax(pi_logits, dim=-1)  # [A, 6]

                result = {
                    'trajs': loc_refine_pos.cpu(),    # [A, 6, 60, 2]
                    'probs': pi_probs.cpu(),           # [A, 6]
                    'num_agents': loc_refine_pos.shape[0],
                }

                # Send back
                send_pickle(conn, result)
                conn.close()

                infer_ms = (time.time() - t_start) * 1000
                if request_count % 50 == 1:
                    print(f"[QCNet Server] Requests: {request_count}, "
                          f"last: {infer_ms:.1f}ms")

            except Exception as e:
                print(f"[QCNet Server] Request error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        running = False
        print(f"\n[QCNet Server] Shutting down ({request_count} requests)")
    finally:
        server_sock.close()


if __name__ == '__main__':
    main()
