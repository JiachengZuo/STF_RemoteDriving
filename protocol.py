# Remote Driving Framework - Message Protocol Helpers
# JSON + newline delimiter over TCP

import json
import base64
import numpy as np
import cv2

# Per-connection buffers (keyed by socket id) to handle
# messages that arrive split across recv() calls
_streams = {}


def encode_message(msg_dict):
    """Encode a message dict to bytes with newline terminator."""
    return (json.dumps(msg_dict) + '\n').encode('utf-8')


def send_message(conn, msg_dict):
    """Send a message dict over a socket."""
    data = encode_message(msg_dict)
    conn.sendall(data)


class MessageStream:
    """Per-connection message stream with internal buffer."""

    def __init__(self):
        self._buffer = b''

    def recv_message(self, conn):
        """Receive one newline-delimited JSON message from a socket.
        Blocks until a complete message is received."""
        while True:
            if b'\n' in self._buffer:
                msg_bytes, self._buffer = self._buffer.split(b'\n', 1)
                return json.loads(msg_bytes.decode('utf-8'))
            chunk = conn.recv(65536)
            if not chunk:
                raise ConnectionError('Connection closed by peer')
            self._buffer += chunk


def recv_message(conn):
    """Receive one newline-delimited JSON message from a socket.
    Uses a global per-connection buffer so bytes from multiple recv() calls
    are correctly handled."""
    key = id(conn)
    if key not in _streams:
        _streams[key] = MessageStream()
    return _streams[key].recv_message(conn)


def cleanup_stream(conn):
    """Remove the stream buffer for a closed connection."""
    key = id(conn)
    _streams.pop(key, None)


def encode_image(np_array_bgr):
    """Encode a BGR numpy array (H, W, 3) to a base64 JPEG string."""
    success, jpeg_bytes = cv2.imencode('.jpg', np_array_bgr,
                                        [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not success:
        return None
    return base64.b64encode(jpeg_bytes.tobytes()).decode('utf-8')


def decode_image(b64_string):
    """Decode a base64 JPEG string to a BGR numpy array (H, W, 3)."""
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
