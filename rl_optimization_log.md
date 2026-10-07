# rl_optimization_log.md

Runs, decisions and gate results. Newest at the bottom.

## 2026-10-07 Phase 0 / A: toolchain and model derivation

**Machine:** Windows 11, RTX 5070 Ti Laptop 12 GB, driver 610.78, CUDA 13.3 toolkit, Docker Desktop 29.7 with GPU passthrough verified (`nvidia-smi` inside `nvidia/cuda:13.0.1`).

**Decision: trainer runs in Docker, not native Windows.** JAX has no CUDA wheels for native Windows and MJX/Warp needs JAX on the GPU to share device buffers. Image: `training/Dockerfile` (Ubuntu 24.04, Python 3.12, jax[cuda13] 0.11.2, mujoco 3.15.0, mujoco-warp 3.15.0, playground 0.2.0, brax 0.14.2).

**Decision: port playground Go1 envs to Go2.** mujoco_playground has no Go2 env. Same 12-joint topology; the port is paths, constants and the joint order (Go2 menagerie order is FL, FR, RL, RR; Go1 playground is FR, FL, RR, RL).

**Decision: PD gains baked into the XML.** Playground applied Kp=35 / Kd=0.5 at runtime on top of the menagerie file (gainprm 50). `go2_porace.xml` carries gainprm 35, biasprm 0 -35 -0.5, joint damping 0.5, so the trainer and Unity load literally the same numbers.

**Decision: cubes park on the floor at (20+2k, 20, 0.05).** First attempt parked them under the floor at z=-10; MuJoCo's infinite plane treats anything below it as a deep penetration. Found by `check_model.py`.

**Unity plugin facts (org.mujoco 3.15.0) that shape Phase B:**
- Ignores MJCF `timestep` and `gravity`; takes them from Unity Time and Physics settings. Must set fixedDeltaTime = 0.004.
- Importer has no `<include>`, no `<keyframe>`, no `<contact>` sensor. `flatten_for_unity.py` writes `Assets/MuJoCo/go2_unity.xml`; C# sets the home pose and finds sensors by name.
- `MjGlobalSettings` does not parse `ls_iterations` or the `eulerdamp` flag. Set them on `mjModel.opt` from C# after scene creation and assert in the B.3 readback.
- Unity regenerates MJCF from its component tree at play time, so model fields (not just the file) must be compared against the trainer. Added to B.3.

**check_model.py result (CPU MuJoCo 3.15.0):** nq=47 nv=42 nu=12 ngeom=61, robot mass 15.206 kg, no self-contacts at home, only feet touch the floor. Passive 2 s hold at home ctrl settles to base z = 0.2395 m (3 cm sag) and joint sag up to 0.26 rad on the calves. Those settled values are in `docs/model_summary.json` and are the zero-brain reference for Unity.
