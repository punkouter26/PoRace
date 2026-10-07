"""B.5 zero-brain gate, Python side. Same scenario as Unity: reset to home, ctrl = default pose, step 2 s on
CPU MuJoCo, record every control step. Then compare with docs/parity/unity_zero_brain.json if it exists.
Run: uv run --no-project --with mujoco==3.15.0 --with numpy python zero_brain.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from go2 import constants as C  # noqa: E402

PARITY_DIR = C.ROOT_PATH.parent / "docs" / "parity"
SECONDS = 2.0
QPOS_TOL = 1e-3   # rad / m per element
HEIGHT_TOL = 2e-3

m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
d.ctrl[:] = np.array(C.DEFAULT_POSE, dtype=np.float32)  # ctrl is float32 in Unity (MjActuator.Control) and in Warp
frames = []
steps = int(round(SECONDS / m.opt.timestep))
# Unity records in MjScene.postUpdateEvent, i.e. after mj_step of physics step i where i % 5 == 0.
for i in range(steps + 1):
    mujoco.mj_step(m, d)
    if i % C.DECIMATION == 0:
        frames.append({"t": float(d.time), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "ctrl": d.ctrl.tolist(),
                       "sensordata": d.sensordata.tolist()})
PARITY_DIR.mkdir(parents=True, exist_ok=True)
(PARITY_DIR / "python_zero_brain.json").write_text(json.dumps({"frames": frames}))
print(f"python: {len(frames)} frames, final z={frames[-1]['qpos'][2]:.4f}")

unity_path = PARITY_DIR / "unity_zero_brain.json"
if not unity_path.exists():
    print("no unity_zero_brain.json yet; run the Testbed scene with ParityRecorder")
    sys.exit(0)
u = json.loads(unity_path.read_text())["frames"]
n = min(len(u), len(frames))
assert n > 50, f"too few unity frames: {len(u)}"
worst = 0.0
for k in range(n):
    p, q = np.array(frames[k]["qpos"][:19]), np.array(u[k]["qpos"][:19])
    assert abs(frames[k]["t"] - u[k]["t"]) < 1e-6, f"frame {k}: time {frames[k]['t']} vs {u[k]['t']}"
    diff = np.abs(p - q)
    worst = max(worst, float(diff.max()))
    assert diff[7:].max() < QPOS_TOL, f"frame {k} t={u[k]['t']:.3f}: joint diff {diff[7:].max():.2e} idx {diff[7:].argmax()}"
    assert diff[2] < HEIGHT_TOL, f"frame {k}: height diff {diff[2]:.2e}"
print(f"ZERO-BRAIN PARITY OK over {n} frames ({n * C.CTRL_DT:.2f} s), worst |dqpos| = {worst:.2e}")
