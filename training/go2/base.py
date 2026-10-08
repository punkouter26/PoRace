"""Go2 base env: loads scene_porace.xml as-is (gains, timestep, sensors baked into the XML)."""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from go2 import constants as C

Q = slice(7, C.NQ_ROBOT)   # robot joint qpos
V = slice(6, C.NV_ROBOT)   # robot joint qvel / dof


class Go2Env(mjx_env.MjxEnv):
  def __init__(self, config: config_dict.ConfigDict,
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None) -> None:
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    assert abs(self._mj_model.opt.timestep - self._config.sim_dt) < 1e-12, "sim_dt must equal the XML timestep"
    assert self._mj_model.nq == C.NQ and self._mj_model.nu == C.NU
    self._mj_model.vis.global_.offwidth = 1920
    self._mj_model.vis.global_.offheight = 1080
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = str(C.SCENE_XML)
    self._imu_site_id = self._mj_model.site(C.IMU_SITE).id
    self._feet_floor_found_sensor = [self._mj_model.sensor(s).id for s in C.FEET_CONTACT_SENSOR]
    self._cube_qpos_adr = [self._mj_model.jnt_qposadr[self._mj_model.body(b).jntadr[0]] for b in C.CUBE_BODIES]
    assert self._cube_qpos_adr == [C.NQ_ROBOT + 7 * k for k in range(C.NUM_CUBES)]

  def get_upvector(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.UPVECTOR_SENSOR)
  def get_gravity(self, data): return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])
  def get_global_linvel(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.GLOBAL_LINVEL_SENSOR)
  def get_global_angvel(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.GLOBAL_ANGVEL_SENSOR)
  def get_local_linvel(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.LOCAL_LINVEL_SENSOR)
  def get_accelerometer(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.ACCELEROMETER_SENSOR)
  def get_gyro(self, data): return mjx_env.get_sensor_data(self.mj_model, data, C.GYRO_SENSOR)
  def get_feet_pos(self, data):
    return jp.vstack([mjx_env.get_sensor_data(self.mj_model, data, s) for s in C.FEET_POS_SENSOR])

  def blown_up(self, data) -> jax.Array:
    """NaN, runaway velocity, or a body that escaped the arena (e.g. fell through a dropped floor contact)."""
    return (jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
            | (jp.abs(data.qvel[:C.NV_ROBOT]).max() > 500.0) | (jp.abs(data.qpos[:3]).max() > 100.0))

  @staticmethod
  def clean_obs(obs):
    return jax.tree_util.tree_map(lambda o: jp.clip(jp.nan_to_num(o), -100.0, 100.0), obs)

  def noisy(self, info, x, scale):
    info["rng"], k = jax.random.split(info["rng"])
    return x + (2 * jax.random.uniform(k, shape=x.shape) - 1) * self._config.noise_config.level * scale

  @property
  def xml_path(self) -> str: return self._xml_path
  @property
  def action_size(self) -> int: return self._mjx_model.nu
  @property
  def mj_model(self) -> mujoco.MjModel: return self._mj_model
  @property
  def mjx_model(self) -> mjx.Model: return self._mjx_model
