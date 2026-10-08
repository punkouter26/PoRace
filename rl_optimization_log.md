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

## 2026-10-07 Phase B: Unity ingestion and zero-brain parity

**Result: ZERO-BRAIN PARITY OK.** 2 s passive PD hold, 101 control frames, worst |dqpos| = 4.75e-8 (tolerance 1e-3). Unity model regenerated from the component tree matches the trainer model in every physical field (`training/compare_models.py`); the only differences are equivalent quaternion sign flips, visual-mesh bounds and the floor plane's render spacing.

Four things that broke parity before it passed, all now handled in code:
1. **Element order is not stable.** The plugin orders bodies/joints/actuators by EntityId, which changes between runs (cubes came first on the second run). `Go2Model.cs` resolves every qpos/dof/ctrl/sensor address by name; `ParityRecorder` dumps canonical order.
2. **Generated names get numeric suffixes** unless `MjGlobalSettings.UseRawGameObjectNames` is on. Set in the Testbed scene.
3. **`MjActuator.OnSyncState` rewrites `mjData.ctrl` from its `Control` field after every step**, silently discarding ctrl written in `ctrlCallback`. `Go2Controller.SetCtrl` writes both. Control is float32, so the Python reference uses float32 ctrl too (Warp is float32 anyway).
4. **`ls_iterations`, `eulerdamp` and `ccd_iterations` are not parsed by the plugin.** Set on `mjModel.opt` in `postInitEvent`; `ModelCheck.cs` asserts them.

Also: the plugin at tag 3.15.0 does not compile on Unity 6000.6 (`GetInstanceID` is a hard error); patched to `GetEntityId` in the embedded copy. The editor's player loop does not advance while the editor window is unfocused, so parity runs use the headless standalone build.

Passive-hold settled state in both engines: base z 0.2395 m, calf sag up to 0.26 rad (Kp 35 vs ~15 kg).

## 2026-10-07 Phase C plumbing (trainer)

- Ported playground `go1` joystick/getup to `training/go2/` on `scene_porace.xml`. Obs 48 / 42 confirmed on Warp.
- **njmax 40 (go1 default) silently drops the pooled cubes through the floor**: the four condim-6 pyramidal feet already use 40 constraint rows. Joystick now uses njmax 128, naconmax 8*8192.
- playground 0.2.0 (PyPI) differs from GitHub main: model upload is `mjx.put_model(m, impl=)`.
- brax 0.14.2 calls `jax.device_put_replicated`, removed in jax 0.11.2. `train.py` installs a drop-in shim before importing brax.
- Exporter matches brax exactly: normalize is `(x - mean) / std` with std already containing `std_eps`; MLP layers `hidden_i`, swish, final `tanh(mean)` of the NormalTanh head.
- `training/gate.sh <run>` = C.8/C.9 in one go: export -> eval R0/R1 -> reference -> assign ONNX + rebuild headless player -> replay -> closed loop -> compare.

## 2026-10-07 Sanity run + gate harness dry run (throwaway policy `runs/sanity`)

- `train.py --timesteps 2000000 --num_envs 2048`: brax rounded to 14.7M steps, 809 s on the RTX 5070 Ti (about 5 min of that is JIT). Final eval reward 16.2. The policy already stands (R0 3/3) but does not walk (0.01 m/s on a 0.5 m/s command).
- Exporter: ONNX vs numpy reference 2.97e-6 worst over 1000 random obs.
- Unity replay gate on the trainer's own 250-frame reference: worst |da| = 3.58e-7 (tol 1e-4). Inference Engine CPU backend reproduces the JAX policy.
- Closed-loop 5 s from the same state and command: PASS on all four metrics (upright, speed, cadence, force). max |dqpos| drifts to 2.8e-2 by 5 s, expected float32 chaos in a closed loop; the metrics, not raw qpos, are the gate.
- Git Bash mangles `/docs` style container paths: `MSYS_NO_PATHCONV=1` in gate.sh. CPU-only tools run with `JAX_PLATFORMS=cpu` to skip the CUPTI probe.
- Rung 1 full run launched: `train.py --env joystick --logdir runs/r1` (200M steps, 8192 envs, no pert/DR).

## 2026-10-07 Rung 1 attempt 1 FAILED: NaN poisoning (run discarded, relaunched)

- Symptom: TensorBoard `eval/episode_reward`, `training/policy_loss`, KL all NaN from the 22M-step eval on, while `tracking_lin_vel` and episode length looked healthy. The 92M checkpoint had NaN in every policy tensor and in the obs normalizer. CPU eval still showed "standing" only because MuJoCo resets data when it detects NaN, which disguised the failure.
- Root cause (`training/nan_probe.py`): the Warp solver (iterations=1, ls_iterations=5, full-collision body geoms) occasionally blows a fallen env up to NaN, about 1 env per 4096 x 170 random-action steps. The termination test `upvector_z < 0` is False on NaN, so the env never reset and fed NaN into the running observation statistics, which then made every obs NaN.
- Fix: joystick/getup `step` now terminates on any NaN in qpos/qvel, zeroes the reward, `nan_to_num`s obs and metrics, and logs `nan_resets`; the brax autoreset wrapper then swaps the env back to its reset state (`where_done` on data/obs/info). Verified by injecting NaN: done=1, obs finite, nan_resets=1. Unity `AutoReset` has the same guard (NaN base state -> home keyframe).
- Lesson: watch `training/policy_loss` for NaN in the first 5 evals of every run; a healthy-looking tracking curve is not proof.

## 2026-10-07 Rung 1 attempt 2 FAILED: contacts dropped, dogs sink through the floor (relaunched as attempt 3)

- Same NaN signature at the 22M eval, but `nan_resets` = 0: the physics never went NaN. The obs normalizer std had grown to 1e6 and base/joint positions to 1e5: bodies were falling without end. NaN ctrl from the dead policy is clamped by ctrlrange, which is why physics looked finite and tracking "improved".
- Root cause: `naconmax = 8 * 8192` (go1 feet-only sizing, 4-8 contacts per env). The full-collision Go2 lying on the floor has ~25 contacts; early in training most envs are fallen, the budget overflows, Warp drops contacts, the dog falls through the floor, and `upvector_z < 0` never fires.
- Fix: joystick `naconmax = 30 * 8192` (the go1 getup value), `blown_up()` guard in base.py (NaN, |qvel| > 500, |base pos| > 100 m) terminates such envs, obs are `nan_to_num` + clipped to ±100. Probe with the new budget: max|obs| 41, min base z 0.16 over 300 wrapped random-action steps (before: velocities to 285 and positions to 1e5).
- Side effect: the larger budget needs more GPU memory; the probe OOM'd at 4096 envs with the viewer open. Attempt 3 runs with 8192 envs and memory is being watched; fallback is 20 * 8192.
- `eval/episode_nan_resets` is now also the dropped-contact detector: it must stay 0.
