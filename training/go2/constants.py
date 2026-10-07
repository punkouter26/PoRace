"""Go2 constants shared by every training env. Single source of truth for ordering.

Order everywhere (qpos[7:], qvel[6:], ctrl, obs slices, action index): FL, FR, RL, RR x hip, thigh, calf.
docs/joint_order.md is the human copy of the same table and must stay in sync.
"""
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
ASSET_PATH = ROOT_PATH / "assets" / "go2"
SCENE_XML = ASSET_PATH / "scene_porace.xml"
ROBOT_XML = ASSET_PATH / "go2_porace.xml"

LEGS = ["FL", "FR", "RL", "RR"]
JOINT_NAMES = [f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")]
ACTUATOR_NAMES = [f"{leg}_{j}" for leg in LEGS for j in ("hip", "thigh", "calf")]
FEET_SITES = [f"{leg}_foot" for leg in LEGS]
FEET_GEOMS = LEGS
FEET_POS_SENSOR = [f"{leg}_pos" for leg in LEGS]
FEET_LINVEL_SENSOR = [f"{leg}_global_linvel" for leg in LEGS]
FEET_CONTACT_SENSOR = [f"{leg}_floor_found" for leg in LEGS]

ROOT_BODY = "base"
IMU_SITE = "imu"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

NUM_CUBES = 4
CUBE_BODIES = [f"cube{i}" for i in range(NUM_CUBES)]
CUBE_PARK_QPOS = [[20.0 + 2 * i, 20.0, 0.05, 1.0, 0.0, 0.0, 0.0] for i in range(NUM_CUBES)]

# Contract values. check_model.py asserts the XML matches these.
SIM_DT = 0.004
CTRL_DT = 0.02
DECIMATION = 5
KP = 35.0
KD = 0.5
FORCERANGE = 24.0
ACTION_SCALE = 0.5
NQ_ROBOT = 19
NV_ROBOT = 18
NU = 12
NQ = NQ_ROBOT + 7 * NUM_CUBES
NV = NV_ROBOT + 6 * NUM_CUBES
DEFAULT_POSE = [0.0, 0.9, -1.8] * 4
TOTAL_ROBOT_MASS = 6.921 + 4 * (0.678 + 1.152 + 0.241352)

OBS_JOYSTICK = 48  # local_linvel 3, gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12, command 3
OBS_GETUP = 42     # gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12
