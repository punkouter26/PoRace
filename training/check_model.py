"""Phase A.6: assert scene_porace.xml matches the contract. Run: uv run --with mujoco python check_model.py"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from go2 import constants as C  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
d = mujoco.MjData(m)


def names(objtype, n):
    return [mujoco.mj_id2name(m, objtype, i) for i in range(n)]


# Sizes and order.
assert m.nu == C.NU, m.nu
assert m.nq == C.NQ and m.nv == C.NV, (m.nq, m.nv)
assert names(mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu) == C.ACTUATOR_NAMES
joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
assert joint_names[1:13] == C.JOINT_NAMES, joint_names
for i in range(m.nu):
    assert m.actuator_trnid[i, 0] == i + 1, "actuator i must drive joint i+1"
assert m.jnt_qposadr[1] == 7 and m.jnt_dofadr[1] == 6
assert m.nkey == 1 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0) == "home"
assert np.allclose(m.key_qpos[0, 7:19], C.DEFAULT_POSE)
assert np.allclose(m.key_ctrl[0], C.DEFAULT_POSE)

# Options.
assert abs(m.opt.timestep - C.SIM_DT) < 1e-12
assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
assert m.opt.impratio == 100 and m.opt.iterations == 1 and m.opt.ls_iterations == 5
assert m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
assert np.allclose(m.opt.gravity, [0, 0, -9.81])

# Actuators and joints.
assert np.allclose(m.actuator_gainprm[:, 0], C.KP)
assert np.allclose(m.actuator_biasprm[:, 1], -C.KP)
assert np.allclose(m.actuator_biasprm[:, 2], -C.KD)
assert np.allclose(m.actuator_forcerange, [[-C.FORCERANGE, C.FORCERANGE]] * 12)
assert np.allclose(m.dof_damping[6:18], C.KD) and np.allclose(m.dof_armature[6:18], 0.01)
assert (m.actuator_ctrllimited == 1).all() and (m.actuator_forcelimited == 1).all()

# Mass.
robot_bodies = [b for b in range(m.nbody) if m.body_rootid[b] == 1]
assert len(robot_bodies) == 13
robot_mass = float(m.body_mass[robot_bodies].sum())
assert abs(robot_mass - C.TOTAL_ROBOT_MASS) < 1e-6, robot_mass
for k, body in enumerate(C.CUBE_BODIES):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    assert abs(m.body_mass[bid] - 1.0) < 1e-9
    assert m.jnt_qposadr[m.body_jntadr[bid]] == C.NQ_ROBOT + 7 * k

# Sensors exist by name.
for s in [C.GYRO_SENSOR, C.LOCAL_LINVEL_SENSOR, C.UPVECTOR_SENSOR, C.GLOBAL_LINVEL_SENSOR, C.GLOBAL_ANGVEL_SENSOR,
          C.ACCELEROMETER_SENSOR, *C.FEET_POS_SENSOR, *C.FEET_LINVEL_SENSOR, *C.FEET_CONTACT_SENSOR]:
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0, s

# No robot self-penetration at home; feet are the only robot geoms touching the floor.
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in C.FEET_GEOMS}
for c in d.contact[: d.ncon]:
    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    pair = {c.geom1, c.geom2}
    assert not (b1 in robot_bodies and b2 in robot_bodies), f"self-contact {c.geom1} {c.geom2}"
    if floor in pair and (b1 in robot_bodies or b2 in robot_bodies):
        assert pair & feet, f"non-foot robot geom touches floor: {pair}"

# Passive hold at home ctrl for 2 s: must stay standing, cubes must stay parked.
d.ctrl[:] = m.key_ctrl[0]
for _ in range(int(2.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
assert d.qpos[2] > 0.2, f"robot collapsed under passive PD hold, z={d.qpos[2]:.3f}"
assert abs(d.qpos[2] - 0.27) < 0.05, d.qpos[2]  # PD sag under gravity is ~3 cm; recorded in summary
assert np.abs(d.qpos[7:19] - C.DEFAULT_POSE).max() < 0.3  # gravity load vs Kp=35 gives ~0.2 rad sag
for k in range(C.NUM_CUBES):
    z = d.qpos[C.NQ_ROBOT + 7 * k + 2]
    assert 0.03 < z < 0.07, f"cube{k} not resting on floor, z={z}"
assert np.abs(d.qvel).max() < 0.05, "scene not at rest after 2 s"

summary = {
    "timestep": m.opt.timestep, "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
    "robot_mass": robot_mass,
    "bodies": names(mujoco.mjtObj.mjOBJ_BODY, m.nbody),
    "joints": joint_names,
    "actuators": C.ACTUATOR_NAMES,
    "sensors": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i), "adr": int(m.sensor_adr[i]),
                 "dim": int(m.sensor_dim[i])} for i in range(m.nsensor)],
    "home_qpos": m.key_qpos[0].tolist(),
    "settled_qpos_after_2s_hold": d.qpos.tolist(),
}
out = C.ROOT_PATH.parent / "docs" / "model_summary.json"
out.write_text(json.dumps(summary, indent=1))
print(f"OK  nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} mass={robot_mass:.3f} settled z={d.qpos[2]:.4f} -> {out}")
