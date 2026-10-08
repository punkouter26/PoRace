"""Go2 fall recovery (R3). Ported from mujoco_playground go1/getup.py onto the Go2 scene (robot slices Q/V).
Obs 'state' (42): gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12. Action: qpos + 0.5 * action.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from go2 import constants as C
from go2.base import Go2Env, Q, V


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=C.CTRL_DT, sim_dt=C.SIM_DT, episode_length=300, drop_from_height_prob=0.6, settle_time=0.5,
      action_repeat=1, action_scale=C.ACTION_SCALE, soft_joint_pos_limit_factor=0.95,
      energy_termination_threshold=np.inf,
      noise_config=config_dict.create(level=1.0, scales=config_dict.create(joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05)),
      reward_config=config_dict.create(scales=config_dict.create(
          orientation=1.0, torso_height=1.0, posture=1.0, stand_still=1.0, action_rate=-0.001,
          dof_pos_limits=-0.1, torques=-1e-5, dof_acc=-2.5e-7, dof_vel=-0.1)),
      impl="warp", naconmax=30 * 8192, njmax=250,
  )


class Getup(Go2Env):
  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config=config, config_overrides=config_overrides)
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[Q])
    self._lowers, self._uppers = self.mj_model.jnt_range[1:13].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._z_des = 0.275
    self._up_vec = jp.array([0.0, 0.0, -1.0])

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    """Root at 0.5 m with random orientation and joint angles; cubes stay parked."""
    rng, orientation_rng, qpos_rng = jax.random.split(rng, 3)
    qpos = self._init_q
    qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.5]))
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)
    return qpos.at[Q].set(jax.random.uniform(qpos_rng, (12,), minval=self._lowers, maxval=self._uppers))

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    qpos = jp.where(jax.random.bernoulli(k1, self._config.drop_from_height_prob), self._get_random_qpos(k2), self._init_q)
    qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.5, maxval=0.5))
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[Q], impl=self.mjx_model.impl.value,
                             naconmax=self._config.naconmax, njmax=self._config.njmax)
    data = mjx.forward(self.mjx_model, data)
    data = mjx_env.step(self.mjx_model, data, qpos[Q], self._settle_steps)
    data = data.replace(time=0.0)
    info = {"rng": rng, "last_act": jp.zeros(self.mjx_model.nu), "last_last_act": jp.zeros(self.mjx_model.nu)}
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics["nan_resets"] = jp.zeros(())
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = state.data.qpos[Q] + action * self._config.action_scale
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
    obs = self._get_obs(data, state.info)
    done = jp.sum(jp.abs(data.actuator_force * data.qvel[V])) > self._config.energy_termination_threshold
    rewards = self._get_reward(data, action, state.info)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()  # see joystick.py
    done = done | bad
    obs = jax.tree_util.tree_map(jp.nan_to_num, obs)
    reward = jp.where(bad, 0.0, reward)
    rewards = {k: jp.nan_to_num(v) for k, v in rewards.items()}
    state.metrics["nan_resets"] = bad.astype(jp.float32)
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    return state.replace(data=data, obs=obs, reward=reward, done=jp.float32(done))

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
    ns = self._config.noise_config.scales
    gyro, gravity = self.get_gyro(data), self.get_gravity(data)
    joint_angles, joint_vel = data.qpos[Q], data.qvel[V]
    state = jp.concatenate([
        self.noisy(info, gyro, ns.gyro), self.noisy(info, gravity, ns.gravity),
        self.noisy(info, joint_angles, ns.joint_pos) - self._default_pose, self.noisy(info, joint_vel, ns.joint_vel),
        info["last_act"],
    ])
    privileged_state = jp.hstack([
        state, gyro, self.get_accelerometer(data), self.get_local_linvel(data), self.get_global_angvel(data),
        joint_angles, joint_vel, data.actuator_force, data.site_xpos[self._imu_site_id][2],
    ])
    return {"state": state, "privileged_state": privileged_state}

  def _get_reward(self, data, action, info):
    torso_height = data.site_xpos[self._imu_site_id][2]
    qpos, qvel, torques = data.qpos[Q], data.qvel[V], data.actuator_force
    gravity = self.get_gravity(data)
    is_upright = jp.sum(jp.square(self._up_vec - gravity)) < 0.01
    height = jp.minimum(torso_height, self._z_des)
    is_at_height = (self._z_des - height) < 0.005
    gate = is_upright * is_at_height
    return {
        "orientation": jp.exp(-2.0 * jp.sum(jp.square(self._up_vec - gravity))),
        "torso_height": jp.exp(height) - 1.0,
        "posture": is_upright * jp.exp(-0.5 * jp.sum(jp.square(qpos - self._default_pose))),
        "stand_still": gate * jp.exp(-0.5 * jp.sum(jp.square(action))),
        "action_rate": jp.sum(jp.square(action - info["last_act"])) + jp.sum(jp.square(action - 2 * info["last_act"] + info["last_last_act"])),
        "torques": jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques)),
        "dof_pos_limits": jp.sum(-jp.clip(qpos - self._soft_lowers, None, 0.0) + jp.clip(qpos - self._soft_uppers, 0.0, None)),
        "dof_acc": jp.sum(jp.square(data.qacc[V])),
        "dof_vel": jp.sum(jp.square(jp.maximum(jp.abs(qvel) - 2.0 * jp.pi, 0.0))),
    }
