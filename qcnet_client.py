#!/usr/bin/env python3
# Remote Driving Framework - QCNet Client
# Simple TCP client for QCNet inference server. Uses pickle over TCP
# with 4-byte big-endian length prefix.

import pickle
import struct
import socket
import threading


def send_pickle(sock, obj):
    """Send a pickle-able object with 4-byte length prefix."""
    data = pickle.dumps(obj)
    sock.sendall(struct.pack('>I', len(data)) + data)


def recv_pickle(sock):
    """Receive a pickle-able object with 4-byte length prefix."""
    raw_len = _recv_exact(sock, 4)
    if raw_len is None:
        return None
    data_len = struct.unpack('>I', raw_len)[0]
    data = _recv_exact(sock, data_len)
    if data is None:
        return None
    return pickle.loads(data)


def _recv_exact(sock, n):
    """Receive exactly n bytes from socket."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class QCNetClient:
    """Thread-safe client for the QCNet inference server.

    Opens a fresh TCP connection per inference request (stateless,
    avoids ZMQ's REQ/REP state-machine issues).
    """

    def __init__(self, host='127.0.0.1', port=8891, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = threading.Lock()

    def predict(self, hetero_data_dict):
        """Send HeteroData dict, get predictions back.

        Returns dict with 'trajs' and 'probs', or None on error.
        """
        with self._lock:
            return self._predict_locked(hetero_data_dict)

    def _predict_locked(self, hetero_data_dict):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            send_pickle(sock, hetero_data_dict)
            result = recv_pickle(sock)
            return result
        except Exception as e:
            print(f"[QCNet Client] Error: {e}")
            return None
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def close(self):
        pass  # stateless, nothing to close
