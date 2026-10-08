"""Per-command tracking diagnostic for a joystick checkpoint (CPU MuJoCo, exported numpy policy).
  python diag_r1.py /work/runs/r1b/checkpoints/000045875200
"""
import sys

import numpy as np
from brax.io import model
from brax.training.agents.ppo import checkpoint as ck

import eval as ev
from go2 import constants as C

path = sys.argv[1]
params = ck.load(path) if not path.endswith(".pkl") else model.load_params(path)
sim = ev.Sim(params, "joystick")
rng = np.random.default_rng(0)
print(f"{'command (vx, vy, wz)':26s} {'vx':>6s} {'vy':>6s} {'wz':>6s} {'err':>6s} upright")
errs = []
for cmd in [[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0], [0.5, 0, -1.0]]:
  sim.reset(rng); c = np.array(cmd, np.float32); v = []; up = True
  for k in range(int(4 / C.CTRL_DT)):
    sim.control_step(c); up &= bool(sim.upright())
    if k >= int(1 / C.CTRL_DT):
      l, g = sim.sensor(C.LOCAL_LINVEL_SENSOR), sim.sensor(C.GYRO_SENSOR)
      v.append([l[0], l[1], g[2]])
  m = np.mean(v, 0); e = float(np.linalg.norm([m[0] - c[0], m[1] - c[1], 0.5 * (m[2] - c[2])])); errs.append(e)
  print(f"{str(cmd):26s} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {e:6.2f} {up}")
print("mean err", round(float(np.mean(errs)), 3))
