"""Domain randomization for Go2 (contract: mass, friction, PD gains ±10-20 %). Ported from go1/randomize.py.
Ids are resolved by name from the CPU model so cube bodies/geoms are never touched."""
import jax
import mujoco
from mujoco import mjx

from go2 import constants as C

_MJ = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
FLOOR_GEOM_ID = _MJ.geom("floor").id
TORSO_BODY_ID = _MJ.body(C.ROOT_BODY).id
ROBOT_BODY_IDS = [b for b in range(_MJ.nbody) if _MJ.body_rootid[b] == TORSO_BODY_ID]
assert len(ROBOT_BODY_IDS) == 13
D = slice(6, C.NV_ROBOT)   # robot joint dofs
Q = slice(7, C.NQ_ROBOT)


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    rng, k = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(jax.random.uniform(k, minval=0.4, maxval=1.0))
    rng, k = jax.random.split(rng)
    dof_frictionloss = model.dof_frictionloss.at[D].set(model.dof_frictionloss[D] * jax.random.uniform(k, (12,), minval=0.9, maxval=1.1))
    rng, k = jax.random.split(rng)
    dof_armature = model.dof_armature.at[D].set(model.dof_armature[D] * jax.random.uniform(k, (12,), minval=1.0, maxval=1.05))
    rng, k = jax.random.split(rng)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + jax.random.uniform(k, (3,), minval=-0.05, maxval=0.05))
    rng, k = jax.random.split(rng)
    scale = jax.random.uniform(k, (len(ROBOT_BODY_IDS),), minval=0.9, maxval=1.1)
    body_mass = model.body_mass
    for i, b in enumerate(ROBOT_BODY_IDS):
      body_mass = body_mass.at[b].set(body_mass[b] * scale[i])
    rng, k = jax.random.split(rng)
    body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + jax.random.uniform(k, minval=-1.0, maxval=1.0))
    rng, k = jax.random.split(rng)
    qpos0 = model.qpos0.at[Q].set(model.qpos0[Q] + jax.random.uniform(k, (12,), minval=-0.05, maxval=0.05))
    # PD gains ±20 %: Kp lives in gainprm[0] and biasprm[1] (= -Kp); Kd in biasprm[2] and joint damping.
    rng, kp, kd = jax.random.split(rng, 3)
    sp = jax.random.uniform(kp, (12,), minval=0.8, maxval=1.2)
    sd = jax.random.uniform(kd, (12,), minval=0.8, maxval=1.2)
    gainprm = model.actuator_gainprm.at[:, 0].set(model.actuator_gainprm[:, 0] * sp)
    biasprm = model.actuator_biasprm.at[:, 1].set(model.actuator_biasprm[:, 1] * sp)
    biasprm = biasprm.at[:, 2].set(biasprm[:, 2] * sd)
    dof_damping = model.dof_damping.at[D].set(model.dof_damping[D] * sd)
    return geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature, gainprm, biasprm, dof_damping

  keys = ["geom_friction", "body_ipos", "body_mass", "qpos0", "dof_frictionloss", "dof_armature",
          "actuator_gainprm", "actuator_biasprm", "dof_damping"]
  values = rand_dynamics(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in keys})
  model = model.tree_replace(dict(zip(keys, values)))
  return model, in_axes
