# PoRace / PoMotion - Build Plan (tasks.md)

Creature: Unitree Go2 (menagerie go2_mjx.xml, full-collision). Trainer: mujoco_playground on MuJoCo Warp, Brax PPO.
Engine: Unity 6000.6.0f1 URP, org.mujoco plugin, Unity Inference Engine (com.unity.ai.inference). PhysX never simulates.
Contract: 4 ms sim step, 50 Hz control (decimation 5), PD gainprm 35 / biasprm 0 -35 -0.5, forcerange +-24, joint damping 0.5.
Mark `[x]` when done. Each task is one commit.

## Phase 0: Repo & Toolchain
- [ ] 0.1 `git init`, Unity .gitignore, commit the template project as-is.
- [ ] 0.2 Create `training/` with a uv/venv: mujoco, mujoco-warp, mujoco_playground, brax, onnx, onnxruntime. Pin versions in `training/pyproject.toml`.
- [ ] 0.3 Smoke test: import mujoco_playground and run the stock Go1 joystick env for 10 steps on Warp. Record GPU, driver, versions in `rl_optimization_log.md`.
- [ ] 0.4 Install org.mujoco into Unity from the mujoco release (`unity/` package + mujoco.dll). Install com.unity.ai.inference. Commit `Packages/manifest.json`.
- [ ] 0.5 Restart / verify Unity MCP connection (failed this session). Needed for Phase B scene authoring.

## Phase A: Rig & Physics Body Derivation
- [ ] A.1 Copy menagerie `unitree_go2/` (go2_mjx.xml + assets/*.obj) into `training/assets/go2/`. Record menagerie commit hash.
- [ ] A.2 Author `go2_porace.xml`: full-collision body geoms (port from go1_mjx_fullcollisions.xml classes), explicit option block (timestep 0.004, Euler, pyramidal cone, impratio 100, iterations 1, ls_iterations 5, eulerdamp disable), gainprm/biasprm/damping baked in, no inheritrange.
- [ ] A.3 Add sensors on the imu site: gyro, velocimeter (local_linvel), framezaxis (upvector), framelinvel/frameangvel (global), accelerometer, foot positions, foot floor contact sensors. Match go1 `sensor_fullcollision.xml` names.
- [ ] A.4 Add 4 pooled cubes to `scene_porace.xml`: body at z=-10 with freejoint, box geom size 0.05, mass 1, friction 0.6, condim 3. Floor plane friction 0.6, condim 3, contype 1, conaffinity 0.
- [ ] A.5 Lock the joint/actuator order table (FL,FR,RL,RR x hip,thigh,calf) into `training/go2/constants.py` and `docs/joint_order.md`. One table, referenced by both Python and C#.
- [ ] A.6 `training/check_model.py`: load XML, assert nu=12, nq=19+4*7, actuator order, forcerange, timestep, zero robot self-penetration at keyframe home, total mass ~15.2 kg. Dump `docs/model_summary.json`.

## Phase B: Early Unity Ingestion & Zero-Brain Parity (CRITICAL, before any training)
- [ ] B.1 Import `scene_porace.xml` via MjcfImporter into `Assets/Scenes/Testbed.unity`. Verify hierarchy: MjScene, MjBody x(13 + 4 cubes), MjGeom, MjHingeJoint x12, MjFreeJoint x5, MjActuator x12, sensors.
- [ ] B.2 No-PhysX assertion: editor test `NoPhysXTest.cs` fails if any Rigidbody / Collider / Joint / CharacterController exists in the scene. Added to the test runner.
- [ ] B.3 Set `Time.fixedDeltaTime = 0.004`, gravity (0,-9.81,0) in Physics Manager, MjGlobalSettings solver options to contract values. Runtime log at scene start reads back `mjModel.opt.*` and asserts.
- [ ] B.4 `Go2Controller.cs`: subscribe to `MjScene.ctrlCallback`, step counter mod 5, build obs (48 / 42) from `mjData.sensordata`, `qpos`, `qvel`; write 12 ctrl values. Starts in "hold home pose" mode (no network).
- [ ] B.5 Zero-brain gate: 2 s passive hold at home ctrl in Unity and in Python (`training/zero_brain.py`). Dump qpos every ctrl step from both; assert max diff < 1e-3 rad per joint, base height within 2 mm. Log result in `rl_optimization_log.md`.
- [ ] B.6 `CubePool.cs`: holds the 4 cube MjBody/MjFreeJoint refs; `Fire(pos, vel)` writes qpos/qvel of the next free cube via C# bindings; `Park()` returns it to z=-10 with zero velocity. No Instantiate/Destroy.
- [ ] B.7 `Shove.cs`: impulse via `mjData.xfrc_applied` on the base body for N physics steps.
- [ ] B.8 Testbed scene: 9:16 portrait Game view, follow camera, ground plane, uGUI HUD (TL title / TC fps+tick / TR behaviour dropdown / BL reset+shove / BR version).
- [ ] B.9 Dummy ONNX: `training/export_onnx.py` emits `zero_policy.onnx` (48 -> 12 zeros). Load with Inference Engine, run every 5th step, verify ctrl == default pose and the dog stands as in B.5.
- [ ] B.10 `PolicyRunner.cs`: loads an ONNX, float[] obs -> float[12]. `ReplayTest.cs` reads `reference_trajectory.json` and asserts max abs action diff < 1e-4. Run against the zero policy to prove the harness itself.
- [ ] B.11 Commit tag `phase-b-parity-scaffold`.

## Phase C: Phased Training & Verification Loop
### Trainer plumbing
- [ ] C.1 `training/go2/joystick.py`, `getup.py`, `base.py`, `randomize.py`: port from playground go1, pointing at `scene_porace.xml`, joint order from constants. Add DR for Kp/Kd x U(0.8,1.2).
- [ ] C.2 Pooled-cube projectile disturbance in joystick env: every 1-3 s activate one cube above the torso with random offset, drop from 1 m, recycle after 2 s.
- [ ] C.3 `training/train.py`: Brax PPO with playground go1 hyperparams, TensorBoard logging, checkpoints every 5M steps, seedable. `training/eval.py`: N-seed rollout reporting pass-bar metrics per rung.
- [ ] C.4 `training/export_onnx.py` for real checkpoints: bake obs normalizer, deterministic mean output, opset 17, batch 1. Verify onnxruntime vs JAX < 1e-5 on 1000 random obs.
- [ ] C.5 `training/record_reference.py`: 5 s rollout -> `reference_trajectory.json` (qpos, qvel, sensordata, obs, action per 50 Hz frame) plus initial state and command.

### Rung 0 / Rung 1
- [ ] C.6 Train locomotion, no perturbation, no DR (R0/R1 baseline). Log run in `rl_optimization_log.md`.
- [ ] C.7 Eval R0 (zero command, 10 s, 10 seeds) and R1 (velocity error < 0.2 m/s, 10 seeds). Viewer inspection.
- [ ] C.8 Export R1 ONNX + reference_trajectory.json.
- [ ] C.9 EARLY VERIFICATION GATE in Unity: replay test (< 1e-4), closed-loop 5 s comparison (upright, speed +-0.1 m/s, cadence +-10 %, actuator force +-15 %). If divergent: STOP, fix timestep / solver / ctrl timing / gains, re-run B.5, repeat. Record outcome.
- [ ] C.10 Commit tag `rung1-parity-pass`.

### Rung 2
- [ ] C.11 Continue from R1 checkpoint with kicks, cubes, and DR on.
- [ ] C.12 Eval R2: 30 N.s push and 1 kg cube hit, 9/10 seeds. Export ONNX.
- [ ] C.13 Unity spot-check: fire pool cubes and shove in the testbed, dog survives. Log.

### Rung 3
- [ ] C.14 Train getup policy (drops + fallen poses, settle 0.5 s). Eval: standing within 3 s, 9/10 seeds. Export ONNX.
- [ ] C.15 Mode switch in `Go2Controller.cs`: locomotion <-> getup on projected-gravity z sign with 0.5 s upright hysteresis. Spot-check: knock it over with cubes, it gets up and resumes walking.
- [ ] C.16 Commit tag `ladder-complete`.

## Phase D: Engine Polish & Game Loop (Step 5)
- [ ] D.1 Author `Race.unity` via Unity MCP in-editor (no procedural scene code): 9:16 camera rig, ground, HUD anchors, behaviour selector wired to command presets (stand / walk / turn / getup demo).
- [ ] D.2 Auto-reset: fallen > 5 s without recovery, stalled > 10 s, or out of bounds -> write keyframe home into qpos/qvel, zero ctrl, park cubes.
- [ ] D.3 Telemetry HUD (TAB): state (RACING / STUMBLE / FALLEN_RECOVERING), base velocity, sim tick rate, inference ms.
- [ ] D.4 Performance: profile one dog at 60 FPS portrait; inference + mj_step < 5 ms per control step. Record numbers.
- [ ] D.5 Re-run full in-engine ladder spot-check. Commit tag `v0.1-playable`.

## Phase E: Later (tracked, not scheduled)
- [ ] E.1 Android: build mujoco C library for arm64, swap plugin native binary, portrait build.
- [ ] E.2 CreatureDefinition / MapDefinition registries and pre-race menu (PRD FR-01..03).
- [ ] E.3 Rough terrain rung (heightfield scan obs) and multi-agent racing rung.
- [ ] E.4 G1 / H1 / DeepMimic biped onboarding through the same gates.
