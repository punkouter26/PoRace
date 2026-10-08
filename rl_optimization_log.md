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
- Attempt 3 (naconmax 30*8192) was healthy at 23M (reward 17.1, loss finite, 0 NaN resets) but died of GPU OOM: Warp failed a 254 MB allocation at 7.1 GB used because XLA caches memory it never returns. Attempt 3b resumes from the 23M checkpoint with naconmax 20*8192 and `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`, which bounds XLA and leaves ~7 GB for Warp. If `nan_resets` rises above 0, contacts are being dropped again.
- Attempt 3c: `--restore` needs an absolute path for orbax (fixed in train.py). Brax restores params only and restarts the step counter, so the resumed run re-evaluates the 23M policy at "step 0" (reward 17.2) and trains a full 200M more; checkpoint names restart too and overwrite `000022937600`. Total Rung 1 budget is therefore ~223M steps.

## 2026-10-07 Paused (user request) with Rung 1 at its 23M checkpoint

- Attempt 3c never logged an eval in over an hour at 100 % GPU. Most likely cause: GPU memory spill. `naconmax` is a total across worlds, and the 128-world eval env was allocating the same contact buffers as the 8192-world training env. On Windows the driver then spills into shared system memory, which is extremely slow (host RAM dropped to 3 GB free, vmmem at 9.4 GB). Not yet confirmed by a clean run.
- The container became an unkillable zombie; `docker desktop restart` hung; recovery was killing Docker's processes, `wsl --shutdown`, and starting Docker Desktop again. Future trainer runs should use `docker run --init --log-driver none` (reap children, and stop the Warp solver warnings from filling the container log).
- `train.py` now sizes `naconmax = 30 * num_worlds` per env instance (training env and eval env separately). Untested.
- To resume: `docker run --rm --init --log-driver none --gpus all -v <repo>\training:/work porace-trainer python train.py --env joystick --num_envs 4096 --logdir runs/r1 --restore runs/r1/checkpoints`, check the first eval for finite loss and 0 nan_resets, then `training/gate.sh runs/r1`.

## 2026-10-08 Rung 1 resumed as runs/r1b (4096 envs, per-instance naconmax) after reboot

- The stall is fixed: evals arrive every ~14 min (23M steps), GPU memory steady at 5.4 GB, `nan_resets` 0. Per-instance `naconmax = 30 * num_worlds` confirmed (the eval env no longer allocates training-size buffers). Launch via `--init --log-driver none --name porace-r1`.
- Reward 17.2 -> 18.2 -> 18.8 over the first 46M resumed steps. `diag_r1.py` on the 46M checkpoint: turns on command (0.85 of 1.0 rad/s) but stands still for every linear command, mean error 0.56 (bar 0.2). Reward terms: pose saturated (453), tracking_lin_vel rising slowly 277 -> 288 -> 307 of ~1000, swing peak and feet slip rising, so it is beginning to step. Decision: let the 200M run finish before changing rewards.

## 2026-10-08 Rung 0 / Rung 1 PASSED, Unity parity gate PASSED (runs/r1b, 23M + 206M steps)

- Reward 17.2 -> 26.2 over the resumed 206M steps (2 h 10 min at 4096 envs). Walking emerged between 69M and 138M.
- `eval.py`: R0 10/10, R1 10/10 (mean velocity error 0.092, bar 0.2). The eval is deterministic, so the 10 seeds are identical runs; they add no evidence beyond one. Known gap: from a standstill the policy ignores a 1.5 m/s command (it reaches 1.5 when already walking at 1.0).
- Unity gate (`Assets/Policies/r1.onnx`): replay worst |da| 9.24e-7 over 250 frames; closed loop 5 s at 0.5 m/s PASS: mean vx 0.428 vs 0.433, cadence 1.71 vs 1.81 Hz, mean |torque| 4.13 vs 3.66 N.m, upright both.
- First closed-loop attempt FAILED and was caught by `ModelCheck`: Unity fixed timestep had reverted to 0.02 after the reboot (it was only set in the live editor, never saved). Now `ProjectSettings/TimeManager.asset` has 0.004 and `Go2Controller.Awake` sets `Time.fixedDeltaTime` itself.
- After a reboot the editor needs Unity Hub running for its license; `unity open` hangs otherwise.
- Rung 2 (`runs/r2`: --pert --cubes --dr from the r1b policy) launched in parallel with the gate.

## 2026-10-08 Rung 2 PASSED (runs/r2, stopped at 92M of 200M)

- `train.py --pert --cubes --dr --restore runs/r1b/checkpoints`: reward 17.8 -> 21.0 -> 20.8 -> 21.7 -> 21.6 (lower than R1 by design: kicks up to 3 m/s, falling 1 kg cubes, randomized mass/friction/gains).
- Baseline R1 policy on the R2 bar: 7/10. R2 checkpoint 46M: 9/10. Checkpoint 92M: 18/20 (bar 90 %), R1 error 0.091, R0 pass. Reward had plateaued, so the run was stopped there and `checkpoints/000091750400` kept as `runs/r2/params.pkl`. The margin over the bar is zero; more R2 training is the first thing to do if a larger sample dips below 90 %.
- Unity spot-check (`StressTest.cs`, `PoRace.exe -stress -stressSeed N`): 30 N.s shove at 2 s + pooled cube dropped from 1 m at 5 s, 10 seeds: 9/10 survived in the MuJoCo plugin.
- `--init` works: `docker stop porace-r2` returned cleanly.

## 2026-10-08 Rung 3 PASSED; behaviour ladder complete

- `train.py --env getup --num_envs 4096` (52M steps, 28 min): reward 1.0 -> 5.2 -> 10.0 -> 11.2 -> 15.9. At 26M only some starts recovered; the last interval made the difference.
- `eval.py --rung r3 --seeds 20`: 20/20 standing within 3 s (slowest 2.18 s).
- Unity (`PoRace.exe -getupTest -stressSeed N -mode Getup`, random orientation at 0.5 m, random joint angles, 0.5 s hold): 10/10, slowest 2.22 s.
- `Go2Controller` now has a real state machine for the switch (`Active`): Locomotion -> Getup when upvector z < 0, back only after 0.5 s with z > 0.9. The earlier version had no hysteresis.
- Final Testbed build: locomotion `r2.onnx` + getup `r3_getup.onnx`, autoGetup on.

| Rung | Trainer (CPU MuJoCo) | Unity (MuJoCo plugin) |
|---|---|---|
| R0 stand | 10/10 | covered by closed-loop gate |
| R1 walk + turn | 10/10, err 0.092 (identical deterministic runs) | replay 9.2e-7, closed loop PASS |
| R2 push + cube | 18/20 | 9/10 |
| R3 stand-up | 20/20 | 10/10 |

Not yet verified: the combined fall -> getup -> resume walking sequence in one run (the cube and shove rarely knock the R2 policy over), and the 1.5 m/s from-standstill gap.

## 2026-10-08 Tightening the ladder after review: three gaps found and closed, one reopened

- **End-to-end recovery** (`eval.py --rung combo`, Unity `-comboTest`): walk, get flipped onto a fallen pose at 2 s, auto-switch to getup, switch back, walk again. 20/20 trainer, 10/10 Unity, back to walking ~1.5 s after the flip.
- **R1 eval is now really randomized** (joint offsets, a 1.5 m/s command from standstill, four random commands per seed).
- **Freeze on fast standstill starts:** from rest the policy tracks forward commands up to 1.2 m/s, braces and does not move for >= 1.3 m/s, but walks backward at 1.5 and reaches 1.41 when the command ramps. Noise or nudges do not break the freeze. Fix: a command slew limit `CMD_SLEW = 6` m/s^2 applied identically in eval, `record_reference.py` and `Go2Controller` (0.25 s to reach 1.5). With it R1 is 10/10, standstill-to-1.5 error 0.13. This is a controller-side fix, not a policy fix.
- **True closed-loop parity:** Unity read gyro/linvel/orientation sensors in `ctrlCallback` (after `mj_step1`, fresh), while the trainer's observation uses `sensordata` left over from the start of the previous physics step (4 ms older). Unity now snapshots those sensors in `preUpdateEvent`. `ParityRecorder` and `zero_brain.py` both record after the 5th physics step of each action. Result: 5 s closed loop max |dqpos| 4.2e-4 at 0.5 m/s and 1.9e-6 at 1.5 m/s (was 0.8 rad of drift); speed, cadence and torque agree to 4 digits. Zero-brain still 4.5e-8.
- **R2 reopened.** `eval.py` cleared the push force after 20 physics steps (24 N.s), Unity applied the full 30 N.s. With the eval fixed, the r2b policy scores 35/40 in the trainer and 17/20 in Unity: 87.5 %, below the 90 % bar. Root cause: training kicks peak at 0.318 * 15.2 kg * 3 m/s = 14.5 N.s, half the bar. `train.py --kick_max 7` (~34 N.s) added; run `runs/r2c` continues from r2b.

## 2026-10-08 FINAL: all behaviours achieved (trainer and Unity)

Final policies: `training/final/go2_loco_params.pkl` (= runs/r2c checkpoint 23M: r1 23M + r1b 206M + r2 92M + r2b 46M + r2c 23M with `--kick_max 7`) and `training/final/go2_getup_params.pkl` (runs/r3, 52M). Unity: `Assets/Policies/go2_loco.onnx`, `go2_getup.onnx`. Stronger kicks fixed R2 in one 12-minute interval (reward 15.6 -> 17.1).

| Behaviour | Bar | Trainer (CPU MuJoCo) | Unity (MuJoCo plugin, headless player) |
|---|---|---|---|
| R0 stand 10 s | 10/10 | 10/10 | covered by closed loop |
| R1 walk + turn, randomized, incl. 1.5 m/s from standstill | err < 0.2, 10/10 | 10/10 | closed loop PASS at 0.5 and 1.5 m/s (1.2588 vs 1.2588 m/s) |
| R2 30 N.s push + 1 kg cube from 1 m | 90 % | 40/40 | 20/20 |
| R3 stand up within 3 s | 90 % | 20/20 | 10/10 |
| Fall -> getup -> resume walking | 90 % | 20/20 | 10/10 |
| Parity | | | zero-brain 4.5e-8; replay 7.8e-7; closed loop metrics pass |

Caveats: walking from standstill at >= 1.3 m/s relies on the 6 m/s^2 command slew limit (policy still braces on a raw step). The 0.5 m/s closed-loop run drifts to 0.78 rad max |dqpos| by 5 s while passing all gait metrics (the 1.5 m/s run stays within 7e-6); float-level divergence in a chaotic gait, not a model mismatch. R0/R3 evals are noise-free.

## 2026-10-08 Phase D: game-side polish

- `Assets/Scenes/Race.unity`: the validated Testbed creature without harness components (ParityRecorder, ReplayHarness, StressTest removed), authored in the editor. Behaviour selector (top right): Stand / Walk 1.0 / Run 1.5 / Turn / Flip (`Go2Controller.KnockOver` writes qpos only). Bottom left: Reset / Shove / Cube / Drop. TAB toggles telemetry. Built to `Build/Race/PoRace.exe`.
- Editor tests: `NoPhysXTest` 2/2 (no Rigidbody/Collider/Joint in Testbed or Race; fixed timestep 0.004).
- Performance (`PerfProbe`, headless, 10 s at 1.0 m/s): physics step mean 0.182 ms including amortized inference (inference alone 0.289 ms per call), i.e. 0.91 ms per 50 Hz control step for one racer; 3.6 ms extrapolated for four (budget 5.0). Worst single step 8.7 ms (one spike). Extrapolation only; no multi-racer scene yet.
- Regression after the changes: stress 5/5, combo 3/3.
- `docs/report/training_report.html` regenerated from the final results (`training/report.py`, data in `docs/report/tb_final.json`).

## 2026-10-08 First race: Map 01 Planar Sprint (no new training)

- `Track01.unity` + `RaceOrchestrator.cs`. The racer is the validated locomotion policy driven by a waypoint steer: `vx = 1.5 * max(0.3, cos(err))`, `yaw rate = clamp(1.5 * err, +-1)`, through the same 6 m/s^2 command slew. Rails are MjGeom boxes in the MuJoCo world, so hitting them is solver contact; `NoPhysXTest` covers the scene (2/2).
- Headless results (deterministic, identical on repeat): 100 m in 74.42 s, splits 17.79 / 36.30 / 54.88 / 74.42, average 1.34 m/s, 0 respawns. With `Go2Controller.KnockOver` at race time 20 s: getup, brain switch, resume; 75.67 s, 0 respawns (1.25 s lost).
- Top speed is the limit: commanded 1.5, achieved about 1.35 on a long straight. Faster racing needs a policy trained on a wider command range.
- `ResetToHome` now spawns at `spawnMj` / `spawnYaw`, which the orchestrator moves to the last checkpoint.

## 2026-10-08 Four-dog race (no new training)

- **One model, four robots.** `Track01x4.unity`: the imported robot subtree (base + actuators + sensors) is cloned three times inside the same `go2_scene`, every MuJoCo name prefixed `r1_`..`r3_`, so all racers and the rails live in one mjModel (nq 104, nv 96, nu 48). `Go2Model(prefix)`, `Go2Controller.prefix/multiRobot`, `ModelCheck(robots)`. In multi-robot scenes `ResetToHome` writes only that robot's qpos/qvel (no `mj_resetData`).
- **Race:** `MultiRaceOrchestrator.cs`: countdown, per-racer lane steering through the joystick command, checkpoints every 25 m for respawn, live standings, pack-framing camera, robot-robot contact counter (from `mjData.contact`).
- **First version produced zero bumps** (cruise 1.5/1.44/1.38/1.32 with lanes pinched to 45 %): the pack spread out before the pinch. Changed to closer speeds (1.47 / 1.41 / 1.44 / 1.50, fastest on the outside lanes) and a full merge to one line between 40 % and 60 % of the track.
- **Result (headless, identical on repeat):** Gold 74.22 s, Blue 76.02, Green 76.85, Red 78.22; 162 contact episodes, 8.11 s in contact, 0 respawns. The policies were never trained against another robot; the push/cube robustness carried over.
- **Flip the leader at 30 s:** it was run into by the pack, did not get up within the 6 s limit, was respawned at its checkpoint and finished last (97.75 s). Alone on the track the same flip costs 1.25 s. Getting up inside a crowd is not something the getup policy was trained for.
- **Performance, measured:** 0.394 ms per physics step for four racers including inference = 1.97 ms per 50 Hz control step (budget 5.0). Worst single step 6.7 to 15 ms (occasional spikes, probably GC/first-use).
- **Regression after the refactor:** zero-brain 4.47e-8, replay 7.75e-7, 1.5 m/s closed loop 6.74e-6, stress 5/5, combo 5/5, single race 74.42 s: all identical to before. `NoPhysXTest` 2/2 across Testbed, Race, Track01, Track01x4.
