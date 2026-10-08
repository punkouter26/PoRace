"""Brax PPO on the Go2 envs (MuJoCo Warp). Mirrors mujoco_playground/learning/train_jax_ppo.py.

  python train.py --env joystick --timesteps 200000000 --logdir runs/r1
  python train.py --env joystick --pert --cubes --dr --restore runs/r1/checkpoints --logdir runs/r2
  python train.py --env getup --logdir runs/r3
Outputs: <logdir>/tb (TensorBoard), <logdir>/checkpoints/<step>, <logdir>/params.pkl (brax.io.model).
"""
import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# brax 0.14.2 still calls jax.device_put_replicated, removed in jax 0.11. Drop-in from the pmap migration guide.
try:
  jax.device_put_replicated
except AttributeError:
  def _device_put_replicated(x, devices):
    sh = jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(devices), ("d",)), jax.sharding.PartitionSpec("d"))
    return jax.tree_util.tree_map(
        lambda a: jax.device_put(jnp.broadcast_to(jnp.asarray(a), (len(devices),) + jnp.shape(a)), sh), x)
  jax.device_put_replicated = _device_put_replicated

from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from tensorboardX import SummaryWriter

from go2 import getup, joystick, randomize

ENVS = {"joystick": (joystick.Joystick, joystick.default_config, "Go1JoystickFlatTerrain"),
        "getup": (getup.Getup, getup.default_config, "Go1Getup")}


def make_env(name, pert=False, cubes=False):
  cls, cfg_fn, _ = ENVS[name]
  cfg = cfg_fn()
  if name == "joystick":
    cfg.pert_config.enable = pert
    cfg.cube_config.enable = cubes
  return cls(config=cfg), cfg


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--env", choices=ENVS, default="joystick")
  ap.add_argument("--timesteps", type=int, default=None)
  ap.add_argument("--num_envs", type=int, default=None)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--logdir", default=None)
  ap.add_argument("--restore", default=None, help="checkpoint dir (latest step) or a specific step dir")
  ap.add_argument("--pert", action="store_true"); ap.add_argument("--cubes", action="store_true"); ap.add_argument("--dr", action="store_true")
  a = ap.parse_args()

  env, cfg = make_env(a.env, a.pert, a.cubes)
  eval_env, _ = make_env(a.env, a.pert, a.cubes)
  ppo_params = locomotion_params.brax_ppo_config(ENVS[a.env][2])
  if a.timesteps is not None: ppo_params.num_timesteps = a.timesteps
  if a.num_envs is not None: ppo_params.num_envs = a.num_envs
  logdir = Path(a.logdir or f"runs/{a.env}-{time.strftime('%Y%m%d-%H%M%S')}").resolve()
  ckpt = logdir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
  (ckpt / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str))
  (logdir / "ppo_params.json").write_text(json.dumps(ppo_params.to_dict(), indent=1, default=str))
  writer = SummaryWriter(str(logdir / "tb"))

  restore = None
  if a.restore:
    p = Path(a.restore)
    steps = sorted([d for d in p.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name)) if p.is_dir() else []
    restore = str((steps[-1] if steps else p).resolve())  # orbax requires an absolute path
    print("restoring from", restore)

  training_params = dict(ppo_params)
  network_factory = functools.partial(ppo_networks.make_ppo_networks, **training_params.pop("network_factory"))
  num_eval_envs = training_params.pop("num_eval_envs", 128)
  if a.dr:
    training_params["randomization_fn"] = randomize.domain_randomize
  t0 = time.monotonic()

  def progress(num_steps, metrics):
    for k, v in metrics.items():
      writer.add_scalar(k, float(v), num_steps)
    writer.flush()
    r = metrics.get("eval/episode_reward", float("nan"))
    print(f"{num_steps:>12,d}  reward={r:8.3f}  {time.monotonic() - t0:7.0f}s", flush=True)

  make_inference_fn, params, _ = ppo.train(
      environment=env, eval_env=eval_env, progress_fn=progress, seed=a.seed, num_eval_envs=num_eval_envs,
      network_factory=network_factory, restore_checkpoint_path=restore, save_checkpoint_path=str(ckpt),
      wrap_env_fn=wrapper.wrap_for_brax_training, **training_params)
  brax_model.save_params(str(logdir / "params.pkl"), params)
  print("saved", logdir / "params.pkl", "in", f"{time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
  main()
