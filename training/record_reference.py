"""C.5: roll a trained policy for N seconds (noise off, fixed command) on CPU MuJoCo and write
docs/parity/reference_trajectory.json: per 50 Hz frame {t, qpos, qvel, ctrl, obs, action} in canonical order,
plus the initial state and command. Unity replays obs through the ONNX (ReplayHarness) and runs closed-loop from the
same initial state (compare_trajectory.py).

  python record_reference.py --env joystick --params runs/r1/params.pkl --command 0.5 0 0 --seconds 5
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.io import model as brax_model

from export_onnx import policy_from_params
from go2 import constants as C

Q, V = slice(7, C.NQ_ROBOT), slice(6, C.NV_ROBOT)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--env", choices=["joystick", "getup"], default="joystick")
  ap.add_argument("--params", required=True)
  ap.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0])
  ap.add_argument("--seconds", type=float, default=5.0)
  ap.add_argument("--out", default=str(C.ROOT_PATH.parent / "docs" / "parity" / "reference_trajectory.json"))
  a = ap.parse_args()

  policy = policy_from_params(brax_model.load_params(a.params))  # numpy fn: obs -> action (deterministic)
  m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  mujoco.mj_forward(m, d)
  imu = m.site(C.IMU_SITE).id
  adr = {s: m.sensor_adr[m.sensor(s).id] for s in [C.GYRO_SENSOR, C.LOCAL_LINVEL_SENSOR]}
  default = np.array(C.DEFAULT_POSE, np.float32)
  last_act = np.zeros(C.NU, np.float32)
  cmd_target = np.array(a.command, np.float32); cmd = np.zeros(3, np.float32)
  d.ctrl[:] = default
  frames = []
  n_ctrl = int(round(a.seconds / C.CTRL_DT))
  for k in range(n_ctrl):
    gravity = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    cmd = cmd + np.clip(cmd_target - cmd, -C.CMD_SLEW_STEP, C.CMD_SLEW_STEP)  # same slew limit as Unity
    if a.env == "joystick":
      obs = np.concatenate([d.sensordata[adr[C.LOCAL_LINVEL_SENSOR]:adr[C.LOCAL_LINVEL_SENSOR] + 3], d.sensordata[adr[C.GYRO_SENSOR]:adr[C.GYRO_SENSOR] + 3],
                            gravity, d.qpos[Q] - default, d.qvel[V], last_act, cmd]).astype(np.float32)
    else:
      obs = np.concatenate([d.sensordata[adr[C.GYRO_SENSOR]:adr[C.GYRO_SENSOR] + 3], gravity, d.qpos[Q] - default, d.qvel[V], last_act]).astype(np.float32)
    action = np.clip(policy(obs), -1, 1).astype(np.float32)
    target = (default if a.env == "joystick" else d.qpos[Q].astype(np.float32)) + C.ACTION_SCALE * action
    d.ctrl[:] = target.astype(np.float32)
    for _ in range(C.DECIMATION):
      mujoco.mj_step(m, d)
    frames.append({"t": float(d.time), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "ctrl": d.ctrl.tolist(),
                   "obs": obs.tolist(), "action": action.tolist()})
    last_act = action
  out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({"env": a.env, "command": a.command, "ctrl_dt": C.CTRL_DT, "decimation": C.DECIMATION,
                             "init_qpos": m.key_qpos[0].tolist(), "frames": frames}))
  z = [f["qpos"][2] for f in frames]
  print(f"wrote {len(frames)} frames to {out}; base z min/max {min(z):.3f}/{max(z):.3f}, final x {frames[-1]['qpos'][0]:.2f} m")


if __name__ == "__main__":
  main()
