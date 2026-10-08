"""G1 zero-brain gate: PD hold at the knees_bent pose (ctrl = default pose, float32) for 1 s on CPU MuJoCo, recorded
after the 10th physics step of every control interval, then compared with Unity's docs/parity/unity_g1_zero_brain.json.
  host:  uv run --no-project --with mujoco==3.15.0 --with numpy python training/g1/zero_brain.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "training" / "g1" / "assets" / "g1_porace.xml"
PARITY = ROOT / "docs" / "parity"
SECONDS, DECIMATION, TOL = 1.0, 10, 1e-3

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)
d.ctrl[:] = m.key_qpos[kid][7:].astype(np.float32)
frames = []
for i in range(int(round(SECONDS / m.opt.timestep)) + DECIMATION):
  mujoco.mj_step(m, d)
  if i % DECIMATION == DECIMATION - 1:
    frames.append({"t": float(d.time), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "ctrl": d.ctrl.tolist()})
PARITY.mkdir(parents=True, exist_ok=True)
(PARITY / "python_g1_zero_brain.json").write_text(json.dumps({"frames": frames}))
print(f"python: {len(frames)} frames, final pelvis z={frames[-1]['qpos'][2]:.4f}")

up = PARITY / "unity_g1_zero_brain.json"
if not up.exists():
  print("no unity_g1_zero_brain.json yet"); sys.exit(0)
u = json.loads(up.read_text())["frames"]
n = min(len(u), len(frames)); assert n >= 40, f"too few unity frames: {len(u)}"
worst = 0.0
for k in range(n):
  assert abs(frames[k]["t"] - u[k]["t"]) < 1e-6, f"frame {k}: time {frames[k]['t']} vs {u[k]['t']}"
  diff = np.abs(np.array(frames[k]["qpos"]) - np.array(u[k]["qpos"]))
  worst = max(worst, float(diff.max()))
  assert diff.max() < TOL, f"frame {k} t={u[k]['t']:.3f}: qpos diff {diff.max():.2e} at index {int(diff.argmax())}"
print(f"G1 ZERO-BRAIN PARITY OK over {n} frames ({n * DECIMATION * m.opt.timestep:.2f} s), worst |dqpos| = {worst:.2e}")
