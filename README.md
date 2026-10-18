# Remote Driving Framework

Three-tier remote driving system: Vehicle Client ↔ Relay Server ↔ Operator Client.

## Architecture

```
Operator Client          Relay Server (8890)         Vehicle Client         CARLA Server (2000)
(pygame + keyboard)      (data routing)              (sensors + controls)   (simulation)
     │                        │                           │                      │
     │── control ────────────►│── control ───────────────►│── apply_control() ──►│
     │                        │                           │                      │
     │◄── sensor_data ────────│◄── sensor_data ───────────│◄── camera callback ───│
```

## Requirements

- Python 3.10
- Conda environment: `TCP_new` (has carla, pygame, opencv, numpy)

## Startup (4 Terminals)

All terminals use the same conda environment:

```bash
conda activate TCP_new
cd /home/djhuai/zuo/stf_二期/PythonAPI/examples
```

### Terminal 1 — CARLA Server

```bash
bash /home/djhuai/carla.sh
```

### Terminal 2 — Relay Server

```bash
conda activate TCP_new
cd /home/djhuai/zuo/stf_二期/PythonAPI/examples
python remote_driving/relay_server.py
```

### Terminal 3 — Vehicle Client

```bash
conda activate TCP_new
cd /home/djhuai/zuo/stf_二期/PythonAPI/examples
python remote_driving/vehicle_client.py
```

### Terminal 4 — Operator Client

```bash
conda activate TCP_new
cd /home/djhuai/zuo/stf_二期/PythonAPI/examples
python remote_driving/operator_client.py
```

## Operator Controls

| Key | Action |
|-----|--------|
| W / ↑ | Throttle |
| S / ↓ | Brake |
| A / ← | Steer left |
| D / → | Steer right |
| Space | Hand brake |
| Backspace | Toggle reverse |
| F1 | Toggle HUD |
| ESC | Quit |

## Simulating Network Delay

Edit `relay_server.py`, add `time.sleep(N)` in `_handle_vehicle()` before `send_message()`:

```python
# Before forwarding to operator, insert artificial delay:
time.sleep(0.3)  # 300ms delay
send_message(self.operator_conn, msg)
```

## CLI Arguments

```bash
# Vehicle client
python remote_driving/vehicle_client.py --host 127.0.0.1 --port 2000 --relay-host 127.0.0.1 --relay-port 8890

# Operator client
python remote_driving/operator_client.py --relay-host 127.0.0.1 --relay-port 8890

# Relay server (edit config.py to change port)
python remote_driving/relay_server.py
```

## File Structure

```
remote_driving/
├── README.md              # This file
├── __init__.py            # Package marker
├── config.py              # Shared config (ports, camera settings)
├── protocol.py            # Message serialization (JSON + base64 JPEG)
├── relay_server.py        # Central relay: forwards data ↔ controls
├── vehicle_client.py      # Vehicle side: CARLA sensors + relay connection
└── operator_client.py     # Operator side: pygame display + keyboard input
```
