"""Find where NaN enters: batched random-action rollouts on Warp, report first NaN step and which fields.
  python nan_probe.py --envs 4096 --steps 300 [--cubes] [--pert] [--feet_only_collision]
"""
import argparse
import time

import jax
import jax.numpy as jp
import numpy as np

from go2 import joystick


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--envs", type=int, default=4096); ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--cubes", action="store_true"); ap.add_argument("--pert", action="store_true")
  ap.add_argument("--seed", type=int, default=0); ap.add_argument("--wrapped", action="store_true")
  a = ap.parse_args()
  cfg = joystick.default_config(); cfg.cube_config.enable = a.cubes; cfg.pert_config.enable = a.pert
  env = joystick.Joystick(config=cfg)
  if a.wrapped:  # same wrappers as brax training: episode limit + autoreset on done
    from mujoco_playground import wrapper
    env = wrapper.wrap_for_brax_training(env, episode_length=1000)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(a.seed)
    s = reset(jax.random.split(rng, a.envs))
  else:
    reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
    rng = jax.random.PRNGKey(a.seed)
    s = reset(jax.random.split(rng, a.envs))
  nan_resets = 0
  t0 = time.time(); first = None
  for k in range(a.steps):
    rng, kk = jax.random.split(rng)
    act = jax.random.uniform(kk, (a.envs, 12), minval=-1, maxval=1)
    s = step(s, act)
    bad_obs = jp.isnan(s.obs["state"]).any(-1)
    bad_q = jp.isnan(s.data.qpos).any(-1)
    bad_r = jp.isnan(s.reward)
    n = int(bad_obs.sum()), int(bad_q.sum()), int(bad_r.sum())
    nan_resets += int(s.metrics["nan_resets"].sum()) if "nan_resets" in s.metrics else 0
    if any(n) and first is None:
      first = k
      i = int(jp.argmax(bad_obs | bad_q | bad_r))
      print(f"first NaN at step {k}: envs obs/qpos/reward = {n}; env {i}")
      print("  qpos nan idx", np.where(np.isnan(np.asarray(s.data.qpos[i])))[0].tolist())
      print("  qvel nan idx", np.where(np.isnan(np.asarray(s.data.qvel[i])))[0].tolist())
      print("  obs  nan idx", np.where(np.isnan(np.asarray(s.obs['state'][i])))[0].tolist())
      print("  reward terms nan:", [t for t, v in s.metrics.items() if bool(jp.isnan(v[i]))])
      print("  max |qvel| over batch:", float(jp.nanmax(jp.abs(s.data.qvel))), " done frac:", float(s.done.mean()))
    if k % 50 == 0:
      print(f"step {k}: nan envs obs/qpos/reward {n}, max|qvel| {float(jp.nanmax(jp.abs(s.data.qvel))):.1f}, done {float(s.done.mean()):.3f}", flush=True)
  print(f"done {a.steps} steps x {a.envs} envs in {time.time()-t0:.0f}s; first NaN step = {first}; nan_resets total = {nan_resets}")


if __name__ == "__main__":
  main()
