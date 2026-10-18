# Remote Driving Framework - Shared Configuration

# Network
RELAY_HOST = '127.0.0.1'
RELAY_PORT = 8890
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000

# Camera
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 640
JPEG_QUALITY = 60

# Simulation
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05  # 20 FPS simulation

# Camera transform: first-person view (windshield position)
CAMERA_POS_X = 1.5
CAMERA_POS_Z = 1.4
CAMERA_FOV = 90

# BEV (Bird's Eye View) display
BEV_WIDTH = 640
BEV_RANGE = 80.0  # meters from center to edge

# QCNet inference
QCNet_IPC_PORT = 8891
QCNet_PYTHON = '/home/djhuai/anaconda3/envs/QCNet/bin/python'
QCNet_CHECKPOINT = '/home/djhuai/zuo/stf_二期/QCNet/checkpoints/QCNet_AV2.ckpt'
QCNet_ROOT = '/home/djhuai/zuo/stf_二期/QCNet'
QCNet_SERVER_PATH = '/home/djhuai/zuo/stf_二期/PythonAPI/examples/remote_driving/qcnet_inference_server.py'
