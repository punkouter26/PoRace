"""Derive the PoRace G1 model from the mujoco_playground stock G1 (menagerie unitree_g1 + playground feet-only scene).

The MuJoCo Unity plugin has no <contact><pair> and no joint `actuatorfrcrange`, so both are rewritten into
equivalent supported forms, and the SAME physics is then used by the trainer and by Unity:
  - contact pairs foot-floor (condim 3, friction 0.6) and foot-foot  -> contype/conaffinity 1 on floor + feet geoms
  - hand-thigh pairs                                                 -> dropped (arms are not used for balance here)
  - joint actuatorfrcrange                                           -> forcerange on that joint's (single) actuator
Outputs:
  training/g1/assets/g1_porace.xml      physics only (visual meshes stripped: every body has an explicit inertial)
  Assets/MuJoCo/g1/g1_unity.xml + STLs   same physics plus the visual meshes
  docs/g1/model_summary.json
Run in the trainer container:  python g1/make_model.py
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from etils import epath
from mujoco_playground._src.locomotion.g1 import base as g1base, g1_constants as c

HERE = Path(__file__).resolve().parent
OUT_TRAIN = HERE / "assets" / "g1_porace.xml"
OUT_UNITY = Path("/assets/MuJoCo/g1")
DOCS = Path("/docs/g1")
O = mujoco.mjtObj

assets = g1base.get_assets()
stock = mujoco.MjModel.from_xml_string(epath.Path(c.FEET_ONLY_FLAT_TERRAIN_XML).read_text(), assets=assets)
tmp = Path("/tmp/g1_flat.xml"); mujoco.mj_saveLastXML(str(tmp), stock)
root = ET.fromstring(tmp.read_text())

# --- contact pairs -> geom collision bits
contact = root.find("contact")
if contact is not None: root.remove(contact)
for g in root.iter("geom"):
  if g.get("name") in ("floor", "left_foot", "right_foot"):
    g.set("contype", "1"); g.set("conaffinity", "1"); g.set("condim", "3"); g.set("friction", "0.6 0.005 0.0001")
    g.attrib.pop("priority", None)

# --- joint actuatorfrcrange -> actuator forcerange. The saved XML keeps defaults classes, so the per-joint limits are
# taken from the compiled model and the attribute is removed wherever it appears (joints and default classes).
for j in root.iter("joint"):
  j.attrib.pop("actuatorfrcrange", None); j.attrib.pop("actuatorfrclimited", None)
jname = lambda i: mujoco.mj_id2name(stock, O.mjOBJ_JOINT, i)
frc = {jname(i): stock.jnt_actfrcrange[i] for i in range(1, stock.njnt) if stock.jnt_actfrclimited[i]}
for a in root.find("actuator"):
  lo, hi = frc[a.get("joint")]; a.set("forcerange", f"{lo:g} {hi:g}"); a.set("forcelimited", "true")
assert len(frc) == 29, len(frc)

# contact sensors that referenced the dropped hand-thigh pairs are not used by the env reward at scale 0... keep geoms, drop nothing.
opt = root.find("option")
opt.set("integrator", "Euler"); opt.set("cone", "pyramidal")


def write(path: Path, r: ET.Element, meshdir: str):
  comp = r.find("compiler")
  if comp is None: comp = ET.SubElement(r, "compiler")
  comp.set("meshdir", meshdir); comp.set("angle", comp.get("angle", "radian"))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(ET.tostring(r, encoding="unicode"))


# --- Unity version: keep meshes, copy the STL files next to the XML.
unity = ET.fromstring(ET.tostring(root))
(OUT_UNITY / "assets").mkdir(parents=True, exist_ok=True)
n = 0
for mesh in unity.iter("mesh"):
  f = Path(mesh.get("file")).name; mesh.set("file", f)
  if f in assets:
    b = assets[f]
    # Unity's STL importer treats any file starting with "solid" as ASCII; these are binary STLs with that header text.
    if b[:5].lower() == b"solid" and len(b) == 84 + 50 * int.from_bytes(b[80:84], "little"): b = b"PoRace binary STL".ljust(80, b" ") + b[80:]
    (OUT_UNITY / "assets" / f).write_bytes(b); n += 1
write(OUT_UNITY / "g1_unity.xml", unity, "assets")

# --- Trainer version: strip visual meshes (physics does not depend on them).
train = ET.fromstring(ET.tostring(root))
asset = train.find("asset")
for mesh in list(asset.findall("mesh")): asset.remove(mesh)
for parent in list(train.iter()):
  for g in list(parent.findall("geom")):
    if g.get("mesh") is not None or g.get("type") == "mesh": parent.remove(g)
write(OUT_TRAIN, train, ".")

# --- Verify: both derived models equal the stock one in every dynamic quantity; hold test against stock.
mt = mujoco.MjModel.from_xml_path(str(OUT_TRAIN)); mu = mujoco.MjModel.from_xml_path(str(OUT_UNITY / "g1_unity.xml"))
for m in (mt, mu):
  assert (m.nq, m.nv, m.nu, m.nbody) == (stock.nq, stock.nv, stock.nu, stock.nbody)
  for f in ["body_mass", "body_inertia", "body_ipos", "body_pos", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss",
            "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "key_qpos"]:
    assert np.allclose(getattr(m, f), getattr(stock, f), atol=1e-9), f
  assert m.opt.timestep == stock.opt.timestep and m.opt.iterations == stock.opt.iterations and m.opt.ls_iterations == stock.opt.ls_iterations
  assert m.opt.disableflags == stock.opt.disableflags and m.opt.solver == stock.opt.solver and m.npair == 0
  assert np.allclose(m.actuator_forcerange, stock.jnt_actfrcrange[1:], atol=1e-9) and m.actuator_forcelimited.all()
assert mt.nmesh == 0 and mu.nmesh == stock.nmesh


def hold(m, seconds=2.0):
  d = mujoco.MjData(m); kid = mujoco.mj_name2id(m, O.mjOBJ_KEY, "knees_bent")
  mujoco.mj_resetDataKeyframe(m, d, kid); d.ctrl[:] = m.key_qpos[kid][7:]
  for _ in range(int(seconds / m.opt.timestep)): mujoco.mj_step(m, d)
  return d.qpos.copy(), d


qs, _ = hold(stock); qt, dt_ = hold(mt); qu, _ = hold(mu)
print(f"2 s PD hold at knees_bent: stock z={qs[2]:.4f}  porace z={qt[2]:.4f}  max|dq| stock-vs-porace={np.abs(qs - qt).max():.2e}  trainer-vs-unity={np.abs(qt - qu).max():.2e}")
print("contacts after hold:", [(mujoco.mj_id2name(mt, O.mjOBJ_GEOM, c_.geom1), mujoco.mj_id2name(mt, O.mjOBJ_GEOM, c_.geom2)) for c_ in dt_.contact[:dt_.ncon]][:6])

N = lambda t, k: [mujoco.mj_id2name(mt, t, i) for i in range(k)]
kid = mujoco.mj_name2id(mt, O.mjOBJ_KEY, "knees_bent")
DOCS.mkdir(parents=True, exist_ok=True)
(DOCS / "model_summary.json").write_text(json.dumps({
    "timestep": mt.opt.timestep, "decimation": 10, "nq": mt.nq, "nv": mt.nv, "nu": mt.nu, "mass": float(mt.body_mass.sum()),
    "joints": N(O.mjOBJ_JOINT, mt.njnt)[1:], "actuators": N(O.mjOBJ_ACTUATOR, mt.nu), "default_pose": mt.key_qpos[kid][7:].tolist(),
    "init_qpos": mt.key_qpos[kid].tolist(), "kp": mt.actuator_gainprm[:, 0].tolist(), "damping": mt.dof_damping[6:].tolist(),
    "armature": mt.dof_armature[6:].tolist(), "forcerange": mt.actuator_forcerange.tolist(), "ctrlrange": mt.actuator_ctrlrange.tolist(),
    "hold_qpos_2s": qt.tolist()}, indent=1))
print(f"OK  nq={mt.nq} nu={mt.nu} mass={mt.body_mass.sum():.3f} kg  trainer xml {OUT_TRAIN.stat().st_size} bytes, unity xml + {n} meshes "
      f"({sum(f.stat().st_size for f in (OUT_UNITY / 'assets').iterdir()) / 1e6:.1f} MB)")
