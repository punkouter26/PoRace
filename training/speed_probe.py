"""Top-speed probe for a locomotion checkpoint (CPU MuJoCo, exported numpy policy, command slew as deployed).
  python speed_probe.py runs/r4            # newest checkpoint in runs/r4/checkpoints
  python speed_probe.py runs/r4/params.pkl
"""
import sys
from pathlib import Path

import numpy as np
from brax.io import model

import eval as ev
from go2 import constants as C

p = Path(sys.argv[1])
if p.is_dir():
  from brax.training.agents.ppo import checkpoint as ck
  steps = sorted(d for d in (p / "checkpoints").iterdir() if d.is_dir() and d.name.isdigit())
  params = ck.load(str(steps[-1].resolve())); label = steps[-1].name
  model.save_params(str(p / f"params_{label}.pkl"), params)
else:
  params = model.load_params(str(p)); label = p.name
sim = ev.Sim(params, "joystick"); rng = np.random.default_rng(0)
print(f"{label}: forward command from standstill -> speed reached (mean over seconds 3-6)")
for vx in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
  sim.reset(rng); v = []; up = True
  for k in range(300):
    sim.control_step(np.array([vx, 0, 0], np.float32)); up &= bool(sim.upright())
    if k >= 150: v.append(sim.sensor(C.LOCAL_LINVEL_SENSOR)[0])
  print(f"  cmd {vx:.1f} -> {np.mean(v):5.2f} m/s  {'upright' if up else 'FELL'}")
