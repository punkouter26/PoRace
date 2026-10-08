"""G1 on CPU MuJoCo with the exported numpy policy: pass-bar eval and the Unity reference trajectory.
Observation, action, command slew and gait clock are exactly what Assets/Scripts/G1Controller.cs does.

  python g1/sim.py eval   --params runs/g1/params.pkl [--seeds 10]
  python g1/sim.py record --params runs/g1/params.pkl --command 0.6 0 0 --seconds 5 --out /docs/parity/g1_reference.json
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_onnx import policy_from_params  # noqa: E402

XML = Path(__file__).resolve().parent / "assets" / "g1_porace.xml"
SIM_DT, DECIMATION, ACTION_SCALE, GAIT_HZ, NU = 0.002, 10, 0.5, 1.375, 29
CTRL_DT = SIM_DT * DECIMATION
CMD_SLEW_STEP = 6.0 * CTRL_DT


def wrap(p): return (p + np.pi) % (2 * np.pi) - np.pi


class G1Sim:
  def __init__(self, params):
    self.policy = policy_from_params(params)
    self.m = mujoco.MjModel.from_xml_path(str(XML)); self.d = mujoco.MjData(self.m)
    O = mujoco.mjtObj
    self.kid = mujoco.mj_name2id(self.m, O.mjOBJ_KEY, "knees_bent")
    self.default = self.m.key_qpos[self.kid][7:].astype(np.float32)
    self.imu = self.m.site("imu_in_pelvis").id
    adr = lambda n: self.m.sensor_adr[self.m.sensor(n).id]
    self.a_lin, self.a_gyro, self.a_up = adr("local_linvel_pelvis"), adr("gyro_pelvis"), adr("upvector_torso")
    self.pelvis = self.m.body("pelvis").id

  def reset(self, rng=None):
    mujoco.mj_resetDataKeyframe(self.m, self.d, self.kid)
    if rng is not None: self.d.qpos[7:] += rng.uniform(-0.05, 0.05, NU)
    self.d.ctrl[:] = self.default
    mujoco.mj_forward(self.m, self.d)
    self.last = np.zeros(NU, np.float32); self.cmd = np.zeros(3, np.float32); self.k = 0

  def obs(self):
    d = self.d
    gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    n = max(0, self.k - 1); dphi = 2 * np.pi * CTRL_DT * GAIT_HZ
    ph = np.array([wrap(n * dphi), wrap(np.pi + n * dphi)])
    return np.concatenate([d.sensordata[self.a_lin:self.a_lin + 3], d.sensordata[self.a_gyro:self.a_gyro + 3], gravity, self.cmd,
                           d.qpos[7:] - self.default, d.qvel[6:], self.last, np.cos(ph), np.sin(ph)]).astype(np.float32)

  def control_step(self, command):
    self.cmd = self.cmd + np.clip(np.asarray(command, np.float32) - self.cmd, -CMD_SLEW_STEP, CMD_SLEW_STEP)
    o = self.obs()
    act = self.policy(o).astype(np.float32)   # the trainer does not clip actions
    self.d.ctrl[:] = (self.default + ACTION_SCALE * act).astype(np.float32)
    for _ in range(DECIMATION): mujoco.mj_step(self.m, self.d)
    self.last = act; self.k += 1
    return o, act

  def upright(self): return self.d.sensordata[self.a_up + 2] > 0.0
  def vel(self): return self.d.sensordata[self.a_lin:self.a_lin + 3], self.d.sensordata[self.a_gyro:self.a_gyro + 3]


def evaluate(params, seeds):
  """G1-R0: stand 10 s on a zero command. G1-R1: 1 standstill start at 1.0 m/s + 4 random commands, 3 s each,
  mean velocity error < 0.25 and never fallen. G1-R2: 15 N.s push at 2 s while walking at 0.5 m/s, upright to 6 s."""
  sim = G1Sim(params); res = {"r0": 0, "r1": 0, "r2": 0}; errs_all = []
  for seed in range(seeds):
    rng = np.random.default_rng(seed)
    sim.reset(rng); ok = True
    for _ in range(int(10 / CTRL_DT)): sim.control_step([0, 0, 0]); ok &= bool(sim.upright())
    res["r0"] += ok
    sim.reset(rng); ok = True; errs = []
    for cmd in [[1.0, 0, 0]] + [[rng.uniform(-0.8, 1.0), rng.uniform(-0.4, 0.4), rng.uniform(-0.8, 0.8)] for _ in range(4)]:
      for k in range(int(3 / CTRL_DT)):
        sim.control_step(cmd); ok &= bool(sim.upright())
        if k >= int(1 / CTRL_DT):
          v, w = sim.vel(); errs.append(np.linalg.norm([v[0] - cmd[0], v[1] - cmd[1], 0.5 * (w[2] - cmd[2])]))
    e = float(np.mean(errs)); errs_all.append(e); res["r1"] += ok and e < 0.25
    sim.reset(rng); ok = True; steps = 0
    for k in range(int(6 / CTRL_DT)):
      if k == int(2 / CTRL_DT):
        ang = rng.uniform(0, 2 * np.pi); sim.d.xfrc_applied[sim.pelvis, :3] = np.array([np.cos(ang), np.sin(ang), 0]) * 15.0 / 0.1; steps = 50
      sim.control_step([0.5, 0, 0])
      if steps > 0:
        steps -= DECIMATION
        if steps <= 0: sim.d.xfrc_applied[:] = 0
      if k > int(2.5 / CTRL_DT): ok &= bool(sim.upright())
    res["r2"] += ok
  need = {"r0": seeds, "r1": seeds, "r2": int(0.9 * seeds)}
  for r in ("r0", "r1", "r2"):
    extra = f" (mean vel err {np.mean(errs_all):.3f}, worst {np.max(errs_all):.3f})" if r == "r1" else ""
    print(f"G1-{r.upper()}: {res[r]}/{seeds} passed (need {need[r]}) -> {'PASS' if res[r] >= need[r] else 'FAIL'}{extra}")
  return all(res[r] >= need[r] for r in res)


def record(params, command, seconds, out):
  sim = G1Sim(params); sim.reset(); frames = []
  for _ in range(int(round(seconds / CTRL_DT))):
    o, act = sim.control_step(command)
    frames.append({"t": float(sim.d.time), "qpos": sim.d.qpos.tolist(), "qvel": sim.d.qvel.tolist(), "ctrl": sim.d.ctrl.tolist(),
                   "obs": o.tolist(), "action": act.tolist()})
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  Path(out).write_text(json.dumps({"creature": "g1", "command": list(command), "ctrl_dt": CTRL_DT, "frames": frames}))
  x = [f["qpos"][0] for f in frames]; z = [f["qpos"][2] for f in frames]
  print(f"wrote {len(frames)} frames to {out}; pelvis z min/max {min(z):.3f}/{max(z):.3f}, final x {x[-1]:.2f} m, upright {bool(sim.upright())}")


if __name__ == "__main__":
  from brax.io import model as brax_model
  ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["eval", "record"]); ap.add_argument("--params", required=True)
  ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--command", type=float, nargs=3, default=[0.6, 0, 0])
  ap.add_argument("--seconds", type=float, default=5.0); ap.add_argument("--out", default="/docs/parity/g1_reference.json")
  a = ap.parse_args(); P = brax_model.load_params(a.params)
  if a.what == "eval": raise SystemExit(0 if evaluate(P, a.seeds) else 1)
  record(P, a.command, a.seconds, a.out)
