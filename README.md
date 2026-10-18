# Remote Driving Framework with QCNet Motion Prediction

Three-tier remote driving system with multi-agent motion prediction for network delay compensation.

```
Operator Client          Relay Server (8890)         Vehicle Client         CARLA Server (2000)
  pygame                 data routing + delay       sensors + controls    simulation
  keyboard control       QCNet inference (8891)      TrafficManager (NPCs)  Town10HD_Opt
     │                        │                           │                      │
     │── control ────────────►│── control ───────────────►│── apply_control() ──►│
     │                        │                           │                      │
     │◄── sensor_data+traj ───│◄── sensor_data ───────────│◄── camera callback ───│
                               QCNet subprocess (8891)
```

## Features

- **Remote driving** over TCP with real-time camera feed (960×640, ~25 FPS)
- **Delay compensation** via QCNet multi-agent motion prediction
  - Predicts ego + surrounding NPC trajectories (50 hist steps → 60 future steps)
  - BEV (Bird's Eye View) shifts forward to compensate for network round-trip delay
  - Continuous interpolation: BEV advances smoothly along predicted trajectory every frame (62.5 FPS)
- **Multi-agent BEV display** (640×640, ±80m range)
  - Ego trajectory: bright cyan, width 3
  - NPC trajectories: orange, width 1 (max-prob mode only, up to 20 agents)
  - Lane centerlines and boundaries rendered on BEV
  - Ego waypoint guides: 1s/2s/3s green circles along predicted path
  - Box markers (ego green 8×12, NPCs orange 5×8)
  - Historical trail with fading teal line
- **HUD overlay** displaying speed, location, delay, FPS, control state
- **Speed limit** (10 km/h) for accurate prediction at low speeds
- **Configurable artificial delay** for network latency simulation (relay `--delay`)
- **TrafficManager** spawning 15 NPC vehicles with autopilot

## Requirements

### Environments

| Component | Conda Env | Key Packages |
|-----------|-----------|-------------|
| Relay Server | `TCP_new` | carla, pygame, numpy, torch |
| Vehicle Client | `TCP_new` | carla, pygame, numpy, torch |
| Operator Client | `TCP_new` | carla, pygame, opencv, numpy |
| QCNet Server | `QCNet` | torch (2.0.1+CUDA), torch_geometric, pytorch_lightning |

### QCNet Checkpoint

Download the QCNet checkpoint and place it at:
```
/home/djhuai/zuo/stf_二期/QCNet/checkpoints/QCNet_AV2.ckpt
```

## Quick Start (5 Terminals)

All from the examples directory:

```bash
cd /home/djhuai/zuo/stf_二期/PythonAPI/examples
```

### Terminal 1 — CARLA Server

```bash
bash /home/djhuai/carla.sh
```

### Terminal 2 — Relay Server (with 200ms simulated delay)

```bash
conda activate TCP_new
python remote_driving/relay_server.py --delay 200
```

### Terminal 3 — Vehicle Client

```bash
conda activate TCP_new
python remote_driving/vehicle_client.py
```

### Terminal 4 — QCNet Inference Server

```bash
conda activate QCNet
python remote_driving/qcnet_inference_server.py \
    --port 8891 \
    --checkpoint /home/djhuai/zuo/stf_二期/QCNet/checkpoints/QCNet_AV2.ckpt
```

> **Note:** The relay server automatically launches a QCNet subprocess in `QCNet` env.
> The standalone QCNet server (Terminal 4) is only needed for debugging or manual control.

### Terminal 5 — Operator Client

```bash
conda activate TCP_new
python remote_driving/operator_client.py
```

## Controls

| Key | Action |
|-----|--------|
| W / ↑ | Throttle |
| S / ↓ | Brake |
| A / ← | Steer left |
| D / → | Steer right |
| Space | Hand brake |
| Backspace | Toggle reverse |
| F1 | Toggle HUD |
| ESC / Ctrl+Q | Quit |

## CLI Arguments

### relay_server.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--delay` | 0 | Artificial delay in milliseconds |
| `--host` | 127.0.0.1 | Relay server host |
| `--port` | 8890 | Relay server port |
| `--no-qcnet` | false | Disable QCNet trajectory prediction |

### vehicle_client.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 127.0.0.1 | CARLA server host |
| `--port` | 2000 | CARLA server port |
| `--relay-host` | 127.0.0.1 | Relay server host |
| `--relay-port` | 8890 | Relay server port |

### operator_client.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--relay-host` | 127.0.0.1 | Relay server host |
| `--relay-port` | 8890 | Relay server port |

## Configuration

All shared settings in `config.py`:

```python
# Network
RELAY_HOST = '127.0.0.1'
RELAY_PORT = 8890
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000

# Camera (first-person windshield view)
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 640
JPEG_QUALITY = 60
CAMERA_POS_X = 1.5      # mounted at windshield
CAMERA_POS_Z = 1.4
CAMERA_FOV = 90

# Simulation
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05   # 20 FPS simulation steps

# BEV Display
BEV_WIDTH = 640
BEV_RANGE = 80.0              # ±80 meters

# QCNet
QCNet_IPC_PORT = 8891
QCNet_PYTHON = '/home/djhuai/anaconda3/envs/QCNet/bin/python'
QCNet_CHECKPOINT = '/home/djhuai/zuo/stf_二期/QCNet/checkpoints/QCNet_AV2.ckpt'
QCNet_ROOT = '/home/djhuai/zuo/stf_二期/QCNet'
```

## How Delay Compensation Works

1. **Vehicle** captures sensor data (camera + telemetry + surrounding agents) at 60 FPS
2. **Relay** records `server_recv_time`, delays by `--delay` ms, feeds trajectory buffer to QCNet
3. **QCNet** predicts 60 future steps (3 seconds at 50ms/step) for ego + NPCs
4. **Operator** computes `effective_delay = prediction_age + smoothed_RTT + 50ms`, translates to QCNet step index
5. **BEV** shifts the entire display forward by the predicted displacement at that step
6. **Continuous interpolation**: between QCNet inferences, `eff_step` increases every frame as `pred_age` grows

This means the operator sees where the vehicle IS NOW (not where it was at the delayed camera frame), with NPC trajectories also compensated.

## Architecture

### Data Flow

```
CARLA World
  │
  ├─ Camera sensor → JPEG base64
  ├─ Telemetry (loc, rot, velocity, speed)
  └─ Surrounding agents (id, pos, yaw, velocity)
  │
  ▼
Vehicle Client ── TCP (JSON+newline) ──► Relay Server
  │                                         │
  │                                    ┌────┴────┐
  │                                    │ Delay Queue │── deadline ──► Operator Client
  │                                    │ (heapq)      │                  │
  │                                    ├───────────┤    ◄── controls ────┘
  │                                    │ Trajectory │                  │
  │                                    │ Buffer     │                  │
  │                                    │ (110 steps)│                  │
  │                                    └────┬───────┘                  │
  │                                         │                          │
  │                                    QCNet Subprocess                │
  │                                    (conda QCNet)                   │
  │                                    TCP+pickle (8891)               │
  │                                         │                          │
  │                                    ┌────▼──────┐                   │
  │                                    │ Prediction │                  │
  │                                    │ [A,6,60,2] │                  │
  │                                    └────┬───────┘                  │
  │                                         │                          │
  │                                    Attach to frame ───────────────►│
  │                                                                   │
  │◄── controls ──────────────────────────────────────────────────────┘
  │
  ▼
apply_control() → CARLA Vehicle
```

### Files

```
remote_driving/
├── README.md                 # This file
├── __init__.py               # Package marker
├── config.py                 # Shared configuration
├── protocol.py               # JSON+newline TCP, base64 JPEG encode/decode
├── relay_server.py           # Central hub: delay simulation, QCNet dispatch
├── vehicle_client.py         # CARLA sensors, TrafficManager, control application
├── operator_client.py        # pygame display, keyboard input, BEV rendering
├── bev_renderer.py           # BEV surface: lanes, trails, trajectories, markers
├── trajectory_buffer.py      # Rolling 110-step buffer for QCNet input
├── qcnet_client.py           # TCP+pickle client to QCNet inference server
├── qcnet_inference_server.py # Standalone QCNet inference process
├── lane_extractor.py         # CARLA lane waypoint extraction
│
├── debug_qcnet.py            # Debug: dump QCNet predictions from relay
├── debug_surrounding.py      # Debug: check surrounding agent count
├── debug_traj.py             # Debug: inspect trajectory point values
├── debug_bev.py              # Debug: headless BEV render to PNG
├── debug_multiagent.py       # Debug: multi-agent trajectory dump
└── check_actors.py           # Debug: query CARLA world vehicle count
```

### Key Design Decisions

- **TCP + JSON+newline** for data channel (simple, reliable, human-readable)
- **TCP + pickle (4-byte length prefix)** for QCNet IPC (efficient tensor serialization)
- **Separate QCNet process** in conda `QCNet` env (PyTorch 2.0.1+CUDA) vs `TCP_new` (PyTorch 2.13)
- **Delay queues (heapq)** for symmetric delay simulation (vehicle→operator and operator→vehicle)
- **Rate limiting** at relay forwarding: max ~25 FPS to operator
- **EWMA delay smoothing** (alpha=0.3) for stable BEV compensation
- **Continuous trajectory interpolation**: uses elapsed time since prediction to smoothly advance along trajectory
- **Hard speed limit** (10 km/h) at vehicle side: progressive throttle reduction with minimal braking
- **Agent origins captured at inference time** to guarantee trajectory-origin alignment
