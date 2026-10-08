"""Pass-bar evaluation on CPU MuJoCo with the exported numpy policy (same math as the ONNX).

  python eval.py --rung r0 --params runs/r1/params.pkl        # stand 10 s, zero command, 10 seeds
  python eval.py --rung r1 --params runs/r1/params.pkl        # velocity tracking error < 0.2 m/s, 10 seeds
  python eval.py --rung r2 --params runs/r2/params.pkl        # 30 N.s push + 1 kg cube drop, 9/10 seeds
  python eval.py --rung r3 --params runs/r3/params.pkl        # getup: standing within 3 s from fallen, 9/10 seeds
"""
import argparse

import mujoco
import numpy as np
from brax.io import model as brax_model

from export_onnx import policy_from_params
from go2 import constants as C

Q, V = slice(7, C.NQ_ROBOT), slice(6, C.NV_ROBOT)


class Sim:
  def __init__(self, params, env):
    self.policy = policy_from_params(params)
    self.env = env
    self.m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    self.d = mujoco.MjData(self.m)
    self.imu = self.m.site(C.IMU_SITE).id
    self.adr = {s: self.m.sensor_adr[self.m.sensor(s).id] for s in [C.GYRO_SENSOR, C.LOCAL_LINVEL_SENSOR, C.UPVECTOR_SENSOR]}
    self.base = self.m.body(C.ROOT_BODY).id
    self.default = np.array(C.DEFAULT_POSE, np.float32)

  def sensor(self, s, n=3):
    return self.d.sensordata[self.adr[s]:self.adr[s] + n]

  def reset(self, rng, fallen=False):
    mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
    if fallen:
      self.d.qpos[2] = 0.5
      q = rng.normal(size=4); self.d.qpos[3:7] = q / np.linalg.norm(q)
      self.d.qpos[Q] = rng.uniform(self.m.jnt_range[1:13, 0], self.m.jnt_range[1:13, 1])
    self.d.ctrl[:] = self.d.qpos[Q]
    mujoco.mj_forward(self.m, self.d)
    if fallen:
      for _ in range(int(0.5 / C.SIM_DT)):
        mujoco.mj_step(self.m, self.d)
    self.last_act = np.zeros(C.NU, np.float32)

  def control_step(self, cmd):
    gravity = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    if self.env == "joystick":
      obs = np.concatenate([self.sensor(C.LOCAL_LINVEL_SENSOR), self.sensor(C.GYRO_SENSOR), gravity, self.d.qpos[Q] - self.default,
                            self.d.qvel[V], self.last_act, cmd])
      base = self.default
    else:
      obs = np.concatenate([self.sensor(C.GYRO_SENSOR), gravity, self.d.qpos[Q] - self.default, self.d.qvel[V], self.last_act])
      base = self.d.qpos[Q]
    act = np.clip(self.policy(obs.astype(np.float32)), -1, 1).astype(np.float32)
    self.d.ctrl[:] = (base + C.ACTION_SCALE * act).astype(np.float32)
    for _ in range(C.DECIMATION):
      mujoco.mj_step(self.m, self.d)
    self.last_act = act

  def upright(self):
    return self.sensor(C.UPVECTOR_SENSOR)[2] > 0

  def push(self, impulse, rng):
    ang = rng.uniform(0, 2 * np.pi); n = 25  # 0.1 s
    self.d.xfrc_applied[self.base, :3] = np.array([np.cos(ang), np.sin(ang), 0]) * impulse / (n * C.SIM_DT)
    self._push_steps = n

  def drop_cube(self, k):
    q = C.NQ_ROBOT + 7 * k
    self.d.qpos[q:q + 3] = self.d.qpos[0:3] + [0, 0, 1.0]
    self.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    self.d.qvel[C.NV_ROBOT + 6 * k:C.NV_ROBOT + 6 * k + 6] = 0


def run(rung, params, seeds):
  env = "getup" if rung == "r3" else "joystick"
  sim = Sim(params, env)
  passes = 0; details = []
  for seed in range(seeds):
    rng = np.random.default_rng(seed)
    if rung == "r0":
      sim.reset(rng); ok = True
      for k in range(int(10 / C.CTRL_DT)):
        sim.control_step(np.zeros(3, np.float32)); ok &= sim.upright()
      details.append(f"seed {seed}: {'upright' if ok else 'FELL'} z={sim.d.qpos[2]:.3f}")
    elif rung == "r1":
      # Randomized per seed: joint offsets at start, a 1.5 m/s command from standstill, then 4 random commands.
      sim.reset(rng); errs = []; ok = True
      sim.d.qpos[Q] += rng.uniform(-0.1, 0.1, 12); mujoco.mj_forward(sim.m, sim.d)
      cmds = [[1.5, 0, 0]] + [[rng.uniform(-1.0, 1.5), rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 1.0)] for _ in range(4)]
      for cmd in cmds:
        cmd = np.array(cmd, np.float32)
        for k in range(int(3 / C.CTRL_DT)):
          sim.control_step(cmd); ok &= sim.upright()
          if k >= int(1 / C.CTRL_DT):  # after 1 s transient
            v = sim.sensor(C.LOCAL_LINVEL_SENSOR); w = sim.sensor(C.GYRO_SENSOR)
            errs.append(np.linalg.norm([v[0] - cmd[0], v[1] - cmd[1], 0.5 * (w[2] - cmd[2])]))
      e = float(np.mean(errs)); e0 = float(np.mean(errs[:100])); ok &= e < 0.2
      details.append(f"seed {seed}: mean vel err {e:.3f} (1.5 m/s from standstill: {e0:.3f}) {'OK' if ok else 'FAIL'}")
    elif rung == "r2":
      sim.reset(rng); ok = True; cmd = np.array([0.5, 0, 0], np.float32)
      for k in range(int(8 / C.CTRL_DT)):
        if k == int(2 / C.CTRL_DT): sim.push(30.0, rng)
        if k == int(5 / C.CTRL_DT): sim.drop_cube(0)
        if getattr(sim, "_push_steps", 0) > 0:
          sim._push_steps -= C.DECIMATION
          if sim._push_steps <= 0: sim.d.xfrc_applied[:] = 0
        sim.control_step(cmd)
        if k > int(2.5 / C.CTRL_DT): ok &= sim.upright()
      details.append(f"seed {seed}: {'survived' if ok else 'FELL'}")
    else:  # r3
      sim.reset(rng, fallen=True); ok = False; t_up = None
      for k in range(int(5 / C.CTRL_DT)):
        sim.control_step(None)
        if sim.upright() and sim.d.qpos[2] > 0.22 and t_up is None: t_up = k * C.CTRL_DT
      ok = t_up is not None and t_up <= 3.0 and sim.upright()
      details.append(f"seed {seed}: {'up at %.2fs' % t_up if t_up is not None else 'never up'} {'OK' if ok else 'FAIL'}")
    passes += ok
  need = {"r0": seeds, "r1": seeds, "r2": int(0.9 * seeds), "r3": int(0.9 * seeds)}[rung]
  print("\n".join(details))
  print(f"{rung.upper()}: {passes}/{seeds} passed (need {need}) -> {'PASS' if passes >= need else 'FAIL'}")
  return passes >= need


def run_combo(loco_params, getup_params, seeds):
  """Walk at 0.5 m/s, get flipped onto a random fallen pose at 2 s, stand up with the getup policy, switch back
  (same rule as Unity: Getup when upvector z < 0, Locomotion after 0.5 s with z > 0.9) and walk again.
  Pass: back in Locomotion within 5 s of the flip and mean forward speed > 0.3 m/s over the last 2 s of 10 s."""
  sim = Sim(loco_params, "joystick"); getup = policy_from_params(getup_params); loco = sim.policy
  passes = 0; cmd = np.array([0.5, 0, 0], np.float32)
  for seed in range(seeds):
    rng = np.random.default_rng(seed); sim.reset(rng); sim.env, sim.policy = "joystick", loco
    active = "loco"; upright_for = 0.0; t_back = None; vx = []
    for k in range(int(10 / C.CTRL_DT)):
      t = k * C.CTRL_DT
      if k == int(2 / C.CTRL_DT):
        while True:  # resample until the body z axis points clearly downward-ish, so it really falls
          q = rng.normal(size=4); q /= np.linalg.norm(q)
          if 1 - 2 * (q[1] ** 2 + q[2] ** 2) < -0.2: break
        sim.d.qpos[3:7] = q; sim.d.qpos[2] = 0.5
        sim.d.qpos[Q] = rng.uniform(sim.m.jnt_range[1:13, 0], sim.m.jnt_range[1:13, 1]); sim.d.qvel[:C.NV_ROBOT] = 0
        mujoco.mj_forward(sim.m, sim.d)
      up = sim.sensor(C.UPVECTOR_SENSOR)[2]
      upright_for = upright_for + C.CTRL_DT if up > 0.9 else 0.0
      if active == "loco" and up < 0: active = "getup"; sim.last_act[:] = 0
      elif active == "getup" and upright_for > 0.5:
        active = "loco"; sim.last_act[:] = 0
        if t > 2 and t_back is None: t_back = t - 2
      sim.env, sim.policy = ("joystick", loco) if active == "loco" else ("getup", getup)
      sim.control_step(cmd)
      if t >= 8: vx.append(sim.sensor(C.LOCAL_LINVEL_SENSOR)[0])
    v = float(np.mean(vx)); ok = t_back is not None and t_back <= 5.0 and v > 0.3 and sim.upright() and active == "loco"
    print(f"seed {seed}: back to walking after {t_back if t_back is None else round(t_back, 2)} s, final speed {v:.2f} m/s {'OK' if ok else 'FAIL'}")
    passes += ok
  need = int(0.9 * seeds)
  print(f"COMBO: {passes}/{seeds} passed (need {need}) -> {'PASS' if passes >= need else 'FAIL'}")
  return passes >= need


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--rung", choices=["r0", "r1", "r2", "r3", "combo"], required=True)
  ap.add_argument("--getup_params")
  ap.add_argument("--params", required=True)
  ap.add_argument("--seeds", type=int, default=10)
  a = ap.parse_args()
  if a.rung == "combo":
    raise SystemExit(0 if run_combo(brax_model.load_params(a.params), brax_model.load_params(a.getup_params), a.seeds) else 1)
  raise SystemExit(0 if run(a.rung, brax_model.load_params(a.params), a.seeds) else 1)
