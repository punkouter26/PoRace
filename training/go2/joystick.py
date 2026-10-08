"""Go2 joystick locomotion (R0/R1/R2). Ported from mujoco_playground go1/joystick.py.
Changes: Go2 scene with 4 pooled cubes (robot slices Q/V), optional falling-cube disturbance, DR hooks in randomize.py.
Obs 'state' (48): local_linvel 3, gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12, command 3.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from go2 import constants as C
from go2.base import Go2Env, Q, V


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=C.CTRL_DT, sim_dt=C.SIM_DT, episode_length=1000, action_repeat=1, action_scale=C.ACTION_SCALE,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(level=1.0, scales=config_dict.create(
          joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1)),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-5.0,
              dof_pos_limits=-1.0, pose=0.5, termination=-1.0, stand_still=-1.0, torques=-0.0002,
              action_rate=-0.01, energy=-0.001, feet_clearance=-2.0, feet_height=-0.2, feet_slip=-0.1,
              feet_air_time=0.1),
          tracking_sigma=0.25, max_foot_height=0.1),
      pert_config=config_dict.create(enable=False, velocity_kick=[0.0, 3.0], kick_durations=[0.05, 0.2],
                                     kick_wait_times=[1.0, 3.0]),
      cube_config=config_dict.create(enable=False, wait_times=[1.0, 3.0], drop_height=1.0, xy_jitter=0.15),
      command_config=config_dict.create(a=[1.5, 0.8, 1.2], b=[0.9, 0.25, 0.5]),
      impl="warp", naconmax=20 * 8192, njmax=128,  # full-collision model: a fallen dog has ~25 contacts; 4-8/env (go1 feet-only) drops contacts and bodies sink through the floor
  )


class Joystick(Go2Env):
  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config=config, config_overrides=config_overrides)
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[Q])
    self._lowers, self._uppers = self.mj_model.jnt_range[1:13].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
    self._torso_body_id = self._mj_model.body(C.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._feet_site_id = np.array([self._mj_model.site(n).id for n in C.FEET_SITES])
    adr = []
    for s in C.FEET_LINVEL_SENSOR:
      sid = self._mj_model.sensor(s).id
      a = self._mj_model.sensor_adr[sid]
      adr.append(list(range(a, a + self._mj_model.sensor_dim[sid])))
    self._foot_linvel_sensor_adr = jp.array(adr)
    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
    self._cube_park = jp.array(C.CUBE_PARK_QPOS)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[0:2].set(qpos[0:2] + jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5))
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[Q], impl=self.mjx_model.impl.value,
                             naconmax=self._config.naconmax, njmax=self._config.njmax)
    data = mjx.forward(self.mjx_model, data)

    rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
    pc = self._config.pert_config
    pert_duration_seconds = jax.random.uniform(k2, minval=pc.kick_durations[0], maxval=pc.kick_durations[1])
    info = {
        "rng": rng,
        "command": jax.random.uniform(k5, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a),
        "steps_until_next_cmd": jp.round(jax.random.exponential(k4) * 5.0 / self.dt).astype(jp.int32),
        "last_act": jp.zeros(self.mjx_model.nu), "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4), "last_contact": jp.zeros(4, dtype=bool), "swing_peak": jp.zeros(4),
        "steps_until_next_pert": jp.round(jax.random.uniform(k1, minval=pc.kick_wait_times[0], maxval=pc.kick_wait_times[1]) / self.dt).astype(jp.int32),
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": jp.round(pert_duration_seconds / self.dt).astype(jp.int32),
        "steps_since_last_pert": 0, "pert_steps": 0, "pert_dir": jp.zeros(3),
        "pert_mag": jax.random.uniform(k3, minval=pc.velocity_kick[0], maxval=pc.velocity_kick[1]),
        "steps_until_next_cube": self._sample_cube_wait(k6), "next_cube": 0,
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics["swing_peak"] = jp.zeros(())
    metrics["nan_resets"] = jp.zeros(())
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _sample_cube_wait(self, key):
    cc = self._config.cube_config
    return jp.round(jax.random.uniform(key, minval=cc.wait_times[0], maxval=cc.wait_times[1]) / self.dt).astype(jp.int32)

  def _maybe_drop_cube(self, state: mjx_env.State) -> mjx_env.State:
    """Every 1-3 s, park the next pooled cube `drop_height` above the base (zero velocity) so it falls on the robot."""
    cc = self._config.cube_config
    fire = state.info["steps_until_next_cube"] <= 0
    state.info["rng"], k1, k2 = jax.random.split(state.info["rng"], 3)
    base = state.data.qpos[0:3]
    xy = base[:2] + jax.random.uniform(k1, (2,), minval=-cc.xy_jitter, maxval=cc.xy_jitter)
    cube_q = jp.concatenate([xy, base[2:3] + cc.drop_height, jp.array([1.0, 0.0, 0.0, 0.0])])
    k = state.info["next_cube"]
    qadr = C.NQ_ROBOT + 7 * k
    vadr = C.NV_ROBOT + 6 * k
    qpos = jax.lax.dynamic_update_slice(state.data.qpos, cube_q, (qadr,))
    qvel = jax.lax.dynamic_update_slice(state.data.qvel, jp.zeros(6), (vadr,))
    data = state.data.replace(qpos=jp.where(fire, qpos, state.data.qpos), qvel=jp.where(fire, qvel, state.data.qvel))
    state.info["next_cube"] = jp.where(fire, (k + 1) % C.NUM_CUBES, k)
    state.info["steps_until_next_cube"] = jp.where(fire, self._sample_cube_wait(k2), state.info["steps_until_next_cube"] - 1)
    return state.replace(data=data)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    if self._config.cube_config.enable:
      state = self._maybe_drop_cube(state)
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    contact = jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_fz = data.site_xpos[self._feet_site_id][..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)
    rewards = self._get_reward(data, action, state.info, done, first_contact, contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    # Solver blow-ups (iterations=1) leave NaN in a few envs per 1e6 steps; a NaN fall check never terminates and
    # poisons the obs normalizer. Terminate + sanitize so the autoreset wrapper swaps the env back to its reset state.
    bad = self.blown_up(data)
    done = done | bad
    obs = self.clean_obs(obs)
    reward = jp.where(bad, 0.0, reward)
    rewards = {k: jp.nan_to_num(v) for k, v in rewards.items()}

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], k1, k2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(state.info["steps_until_next_cmd"] <= 0,
                                     self.sample_command(k1, state.info["command"]), state.info["command"])
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32), state.info["steps_until_next_cmd"])
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.nan_to_num(jp.mean(state.info["swing_peak"]))
    state.metrics["nan_resets"] = bad.astype(jp.float32)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return self.get_upvector(data)[-1] < 0.0

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
    ns = self._config.noise_config.scales
    gyro, gravity = self.get_gyro(data), self.get_gravity(data)
    joint_angles, joint_vel, linvel = data.qpos[Q], data.qvel[V], self.get_local_linvel(data)
    state = jp.hstack([
        self.noisy(info, linvel, ns.linvel), self.noisy(info, gyro, ns.gyro), self.noisy(info, gravity, ns.gravity),
        self.noisy(info, joint_angles, ns.joint_pos) - self._default_pose, self.noisy(info, joint_vel, ns.joint_vel),
        info["last_act"], info["command"],
    ])
    privileged_state = jp.hstack([
        state, gyro, self.get_accelerometer(data), gravity, linvel, self.get_global_angvel(data),
        joint_angles - self._default_pose, joint_vel, data.actuator_force, info["last_contact"],
        data.sensordata[self._foot_linvel_sensor_adr].ravel(), info["feet_air_time"],
        data.xfrc_applied[self._torso_body_id, :3],
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],
    ])
    return {"state": state, "privileged_state": privileged_state}

  def _get_reward(self, data, action, info, done, first_contact, contact):
    sig = self._config.reward_config.tracking_sigma
    cmd = info["command"]
    cmd_norm = jp.linalg.norm(cmd)
    local_vel, gyro = self.get_local_linvel(data), self.get_gyro(data)
    up = self.get_upvector(data)
    qpos, qvel, torques = data.qpos[Q], data.qvel[V], data.actuator_force
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    max_h = self._config.reward_config.max_foot_height
    return {
        "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - local_vel[:2])) / sig),
        "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / sig),
        "lin_vel_z": jp.square(self.get_global_linvel(data)[2]),
        "ang_vel_xy": jp.sum(jp.square(self.get_global_angvel(data)[:2])),
        "orientation": jp.sum(jp.square(up[:2])),
        "stand_still": jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01),
        "termination": done,
        "pose": jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * jp.array([1.0, 1.0, 0.1] * 4))),
        "torques": jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques)),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "energy": jp.sum(jp.abs(qvel) * jp.abs(torques)),
        "feet_slip": jp.sum(jp.sum(jp.square(vel_xy), axis=-1) * contact) * (cmd_norm > 0.01),
        "feet_clearance": jp.sum(jp.abs(foot_z - max_h) * jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))),
        "feet_height": jp.sum(jp.square(info["swing_peak"] / max_h - 1.0) * first_contact) * (cmd_norm > 0.01),
        "feet_air_time": jp.sum((info["feet_air_time"] - 0.1) * first_contact) * (cmd_norm > 0.01),
        "dof_pos_limits": jp.sum(-jp.clip(qpos - self._soft_lowers, None, 0.0) + jp.clip(qpos - self._soft_uppers, 0.0, None)),
    }

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng):
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state):
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      force = u_t * self._torso_mass * state.info["pert_mag"] / state.info["pert_duration_seconds"]
      xfrc = jp.zeros((self.mjx_model.nbody, 6)).at[self._torso_body_id, :3].set(force * state.info["pert_dir"])
      state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
      state.info["steps_since_last_pert"] = jp.where(state.info["pert_steps"] >= state.info["pert_duration"], 0,
                                                    state.info["steps_since_last_pert"])
      state.info["pert_steps"] += 1
      return state

    def wait(state):
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      due = state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"]
      state.info["pert_steps"] = jp.where(due, 0, state.info["pert_steps"])
      state.info["pert_dir"] = jp.where(due, gen_dir(rng), state.info["pert_dir"])
      return state.replace(data=state.data.replace(xfrc_applied=jp.zeros((self.mjx_model.nbody, 6))))

    return jax.lax.cond(state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"], apply_pert, wait, state)

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    return x_k - w_k * (x_k - y_k * z_k)
