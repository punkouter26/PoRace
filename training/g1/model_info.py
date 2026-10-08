"""G1 (playground stock env model): print the contract values and write the flattened Unity model + summary.
  python g1/model_info.py     (in the trainer container; needs the baked menagerie)
"""
import json, re, shutil, sys
from pathlib import Path
import mujoco, numpy as np
from etils import epath
from mujoco_playground._src.locomotion.g1 import base as g1base, g1_constants as c

assets = g1base.get_assets()
xml = epath.Path(c.FEET_ONLY_FLAT_TERRAIN_XML).read_text()
m = mujoco.MjModel.from_xml_string(xml, assets=assets)
N = lambda t, n: [mujoco.mj_id2name(m, t, i) for i in range(n)]
O = mujoco.mjtObj
print("nq nv nu nbody ngeom nsensor nkey nmesh", m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.nsensor, m.nkey, m.nmesh)
print("opt dt", m.opt.timestep, "integrator", m.opt.integrator, "cone", m.opt.cone, "impratio", m.opt.impratio, "iter", m.opt.iterations,
      "ls", m.opt.ls_iterations, "disable", m.opt.disableflags, "solver", m.opt.solver, "noslip", m.opt.noslip_iterations)
print("keys", N(O.mjOBJ_KEY, m.nkey), "mass", round(float(m.body_mass.sum()), 4))
print("gaintype", sorted(set(m.actuator_gaintype.tolist())), "biastype", sorted(set(m.actuator_biastype.tolist())), "dyntype", sorted(set(m.actuator_dyntype.tolist())))
print("kp", sorted(set(np.round(m.actuator_gainprm[:, 0], 3).tolist())), "bias1", sorted(set(np.round(m.actuator_biasprm[:, 1], 3).tolist())), "bias2", sorted(set(np.round(m.actuator_biasprm[:, 2], 3).tolist())))
print("damping", sorted(set(np.round(m.dof_damping[6:], 3).tolist())), "armature", sorted(set(np.round(m.dof_armature[6:], 4).tolist())), "frictionloss", sorted(set(np.round(m.dof_frictionloss[6:], 3).tolist())))
print("forcelimited", int(m.actuator_forcelimited.sum()), "ctrllimited", int(m.actuator_ctrllimited.sum()), "jnt actfrclimited", int(m.jnt_actfrclimited.sum()))
print("collision geoms", [(mujoco.mj_id2name(m, O.mjOBJ_GEOM, g), int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_condim[g])) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
print("sensors", [(n, int(m.sensor_type[i]), int(m.sensor_dim[i])) for i, n in enumerate(N(O.mjOBJ_SENSOR, m.nsensor))])
print("sites", N(O.mjOBJ_SITE, m.nsite))
kid = mujoco.mj_name2id(m, O.mjOBJ_KEY, "knees_bent")
joints = N(O.mjOBJ_JOINT, m.njnt)
summary = {"timestep": m.opt.timestep, "nq": m.nq, "nv": m.nv, "nu": m.nu, "mass": float(m.body_mass.sum()), "joints": joints[1:],
           "actuators": N(O.mjOBJ_ACTUATOR, m.nu), "default_pose": m.key_qpos[kid][7:].tolist(), "init_qpos": m.key_qpos[kid].tolist(),
           "kp": m.actuator_gainprm[:, 0].tolist(), "kd_bias": m.actuator_biasprm[:, 2].tolist(), "ctrlrange": m.actuator_ctrlrange.tolist(),
           "forcerange": m.actuator_forcerange.tolist(), "jnt_range": m.jnt_range[1:].tolist()}
out = Path("/docs/g1"); out.mkdir(parents=True, exist_ok=True)
(out / "model_summary.json").write_text(json.dumps(summary, indent=1))
print("actuator->joint identity:", all(m.actuator_trnid[i, 0] == i + 1 for i in range(m.nu)), "default pose", np.round(m.key_qpos[kid][7:], 3).tolist(), "z", m.key_qpos[kid][2])

# Flatten for Unity: one XML, meshes copied next to it.
if len(sys.argv) > 1:
  dst = Path(sys.argv[1]); (dst / "assets").mkdir(parents=True, exist_ok=True)
  spec = mujoco.MjSpec.from_string(xml, assets=assets); spec.compile()
  flat = spec.to_xml()
  flat = flat.replace("<option ", '<option integrator="Euler" cone="pyramidal" ', 1) if "integrator=" not in flat else flat
  files = set(re.findall(r'file="([^"]+)"', flat)); n = 0
  for f in files:
    base = Path(f).name
    if base in assets: (dst / "assets" / base).write_bytes(assets[base]); n += 1
    flat = flat.replace(f'file="{f}"', f'file="{base}"')
  flat = re.sub(r'meshdir="[^"]*"', 'meshdir="assets"', flat)
  if 'meshdir=' not in flat: flat = flat.replace("<compiler ", '<compiler meshdir="assets" ', 1)
  (dst / "g1_unity.xml").write_text(flat)
  chk = mujoco.MjModel.from_xml_path(str(dst / "g1_unity.xml"))
  assert (chk.nq, chk.nv, chk.nu, chk.ngeom) == (m.nq, m.nv, m.nu, m.ngeom) and abs(chk.body_mass.sum() - m.body_mass.sum()) < 1e-9
  print(f"flattened -> {dst / 'g1_unity.xml'} ({len(flat)} chars, {n} mesh files)")
