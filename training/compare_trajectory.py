"""C.9 closed-loop gate: Unity run vs trainer reference over the same seconds from the same initial state and command.

  python compare_trajectory.py docs/parity/reference_trajectory.json docs/parity/unity_closed_loop.json

Metrics (contract): upright the whole time; mean forward speed within 0.1 m/s; foot-contact cadence within 10 %;
mean |actuator force| within 15 %. Foot contacts and actuator forces are recomputed from qpos/qvel/ctrl with CPU
MuJoCo so both sides are measured identically.
"""
import json
import sys

import mujoco
import numpy as np

from go2 import constants as C

m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
d = mujoco.MjData(m)
FEET = [m.geom(g).id for g in C.FEET_GEOMS]
FOOT_R = float(m.geom_size[FEET[0], 0])
Q, V = slice(7, C.NQ_ROBOT), slice(6, C.NV_ROBOT)


def analyse(frames):
  z, vx, contacts, forces, upright = [], [], [], [], []
  for f in frames:
    d.qpos[:] = f["qpos"]; d.qvel[:] = f["qvel"]; d.ctrl[:] = f["ctrl"]
    mujoco.mj_forward(m, d)
    z.append(d.qpos[2]); vx.append(d.qvel[0])
    upright.append(d.site_xmat[m.site(C.IMU_SITE).id].reshape(3, 3)[2, 2] > 0)
    contacts.append([d.geom_xpos[g][2] < FOOT_R + 0.005 for g in FEET])
    forces.append(np.clip(C.KP * (d.ctrl - d.qpos[Q]) - C.KD * d.qvel[V], -C.FORCERANGE, C.FORCERANGE))
  contacts = np.array(contacts)
  touchdowns = int(np.sum((contacts[1:] & ~contacts[:-1])))  # total foot touchdown events
  secs = frames[-1]["t"] - frames[0]["t"]
  return dict(upright=bool(np.all(upright)), mean_vx=float(np.mean(vx)), cadence_hz=touchdowns / max(secs, 1e-6) / 4,
              mean_abs_force=float(np.mean(np.abs(forces))), min_z=float(np.min(z)))


def main(ref_path, unity_path):
  ref = json.load(open(ref_path))["frames"]; uni = json.load(open(unity_path))["frames"]
  n = min(len(ref), len(uni))
  r, u = analyse(ref[:n]), analyse(uni[:n])
  checks = {
      "upright": u["upright"] and r["upright"],
      "speed": abs(u["mean_vx"] - r["mean_vx"]) <= 0.1,
      "cadence": abs(u["cadence_hz"] - r["cadence_hz"]) <= 0.1 * max(r["cadence_hz"], 1e-6) + 0.05,
      "force": abs(u["mean_abs_force"] - r["mean_abs_force"]) <= 0.15 * max(r["mean_abs_force"], 1e-6),
  }
  q_err = max(float(np.abs(np.array(a["qpos"][:19]) - np.array(b["qpos"][:19])).max()) for a, b in zip(ref[:n], uni[:n]))
  print(f"frames={n} ({n * C.CTRL_DT:.2f} s)   max |dqpos| robot = {q_err:.2e} (informational; chaos grows it)")
  for k in ["upright", "mean_vx", "cadence_hz", "mean_abs_force", "min_z"]:
    print(f"  {k:15s} ref={r[k]!s:>8}  unity={u[k]!s:>8}")
  ok = all(checks.values())
  print("CLOSED-LOOP PARITY", "PASS" if ok else "FAIL", {k: ("ok" if v else "FAIL") for k, v in checks.items()})
  return ok


if __name__ == "__main__":
  raise SystemExit(0 if main(sys.argv[1], sys.argv[2]) else 1)
