# PoRace / PoMotion - Build Plan (tasks.md)

Creature: Unitree Go2 (menagerie go2_mjx.xml, full-collision). Trainer: mujoco_playground on MuJoCo Warp, Brax PPO.
Engine: Unity 6000.6.0f1 URP, org.mujoco plugin, Unity Inference Engine (com.unity.ai.inference). PhysX never simulates.
Contract: 4 ms sim step, 50 Hz control (decimation 5), PD gainprm 35 / biasprm 0 -35 -0.5, forcerange +-24, joint damping 0.5.
Mark `[x]` when done. Each task is one commit.

## Phase 0: Repo & Toolchain
- [x] 0.1 `git init`, Unity .gitignore, commit the template project as-is.
- [x] 0.2 Trainer runs in Docker (no JAX CUDA wheels on native Windows): `training/Dockerfile` + pinned `training/pyproject.toml`. Build `porace-trainer` image.
- [x] 0.3 Smoke test: import mujoco_playground and run the stock Go1 joystick env for 10 steps on Warp. Record GPU, driver, versions in `rl_optimization_log.md`.
- [x] 0.4 Install org.mujoco into Unity from the mujoco release (`unity/` package + mujoco.dll). Install com.unity.ai.inference. Commit `Packages/manifest.json`.
- [x] 0.5 (superseded: the editor is driven through the Unity CLI pipeline instead of Unity MCP) Restart / verify Unity MCP connection. Needed for Phase B scene authoring.

## Phase A: Rig & Physics Body Derivation
- [x] A.1 Copy menagerie `unitree_go2/` (go2_mjx.xml + assets/*.obj) into `training/assets/go2/`. Record menagerie commit hash.
- [x] A.2 Author `go2_porace.xml`: full-collision body geoms (port from go1_mjx_fullcollisions.xml classes), explicit option block (timestep 0.004, Euler, pyramidal cone, impratio 100, iterations 1, ls_iterations 5, eulerdamp disable), gainprm/biasprm/damping baked in, no inheritrange.
- [x] A.3 Add sensors on the imu site: gyro, velocimeter (local_linvel), framezaxis (upvector), framelinvel/frameangvel (global), accelerometer, foot positions, foot floor contact sensors. Match go1 `sensor_fullcollision.xml` names.
- [x] A.4 Add 4 pooled cubes to `scene_porace.xml`: parked on the floor at (20+2k, 20, 0.05) with freejoint, box geom size 0.05, mass 1, friction 0.6, condim 3. Floor plane friction 0.6, condim 3, contype 1, conaffinity 0. (Parking below the plane = deep penetration; rejected.)
- [x] A.5 Lock the joint/actuator order table (FL,FR,RL,RR x hip,thigh,calf) into `training/go2/constants.py` and `docs/joint_order.md`. One table, referenced by both Python and C#.
- [x] A.6 `training/check_model.py`: load XML, assert nu=12, nq=19+4*7, actuator order, forcerange, timestep, zero robot self-penetration at keyframe home, total mass ~15.2 kg. Dump `docs/model_summary.json`.

- [x] A.7 `training/flatten_for_unity.py` -> `Assets/MuJoCo/go2_unity.xml` + meshes (importer has no include/keyframe/contact support).

## Phase B: Early Unity Ingestion & Zero-Brain Parity (CRITICAL, before any training)
- [x] B.1 Import `Assets/MuJoCo/go2_unity.xml` via MjcfImporter into `Assets/Scenes/Testbed.unity`. Verify hierarchy: MjScene, MjBody x(13 + 4 cubes), MjGeom, MjHingeJoint x12, MjFreeJoint x5, MjActuator x12, sensors.
- [x] B.2 No-PhysX assertion: editor test `NoPhysXTest.cs` fails if any Rigidbody / Collider / Joint / CharacterController exists in the scene. Added to the test runner.
- [x] B.3 Set `Time.fixedDeltaTime = 0.004`, gravity (0,-9.81,0) in Physics Manager, MjGlobalSettings solver options to contract values. The plugin does not parse `ls_iterations` / `eulerdamp`: set them on `mjModel.opt` from C# after scene creation. Runtime readback asserts opt.*, nq/nv/nu, body masses, actuator gainprm/biasprm, geom friction against `docs/model_summary.json` (Unity regenerates MJCF from components).
- [x] B.4 `Go2Controller.cs`: subscribe to `MjScene.ctrlCallback`, step counter mod 5, build obs (48 / 42) from `mjData.sensordata`, `qpos`, `qvel`; write 12 ctrl values. Starts in "hold home pose" mode (no network).
- [x] B.5 Zero-brain gate: 2 s passive hold at home ctrl in Unity and in Python (`training/zero_brain.py`). PASSED: worst |dqpos| 4.75e-8 over 101 frames. Runs via headless player `Build/Testbed/PoRace.exe -batchmode -nographics -parityDir <dir>` (editor play loop does not tick while the editor is unfocused). `training/compare_models.py` diffs the Unity-regenerated model field by field.
- [x] B.6 (code in place, exercised in C.13) `CubePool.cs`: holds the 4 cube MjBody/MjFreeJoint refs; `Fire(pos, vel)` writes qpos/qvel of the next free cube via C# bindings; `Park()` returns it to z=-10 with zero velocity. No Instantiate/Destroy.
- [x] B.7 (code in place, exercised in C.13) `Shove.cs`: impulse via `mjData.xfrc_applied` on the base body for N physics steps.
- [x] B.8 Testbed scene: 9:16 portrait Game view, follow camera, ground plane, uGUI HUD (TL title / TC fps+tick / TR behaviour dropdown / BL reset+shove / BR version).
- [x] B.9 Dummy ONNX: `training/export_onnx.py` emits `zero_policy.onnx` (48 -> 12 zeros). Load with Inference Engine, run every 5th step, verify ctrl == default pose and the dog stands as in B.5.
- [x] B.10 `PolicyRunner.cs`: loads an ONNX, float[] obs -> float[12]. `ReplayHarness.cs` (`PoRace.exe -batchmode -replay <json>`) reads `reference_trajectory.json` and asserts max abs action diff < 1e-4. Run against the zero policy to prove the harness itself.
- [x] B.11 Commit tag `phase-b-parity-scaffold`.

## Phase C: Phased Training & Verification Loop
### Trainer plumbing
- [x] C.1 `training/go2/joystick.py`, `getup.py`, `base.py`, `randomize.py`: port from playground go1, pointing at `scene_porace.xml`, joint order from constants. Add DR for Kp/Kd x U(0.8,1.2).
- [x] C.2 Pooled-cube projectile disturbance in joystick env: every 1-3 s activate one cube above the torso with random offset, drop from 1 m, recycle after 2 s.
- [x] C.3 `training/train.py`: Brax PPO with playground go1 hyperparams, TensorBoard logging, checkpoints every 5M steps, seedable. `training/eval.py`: N-seed rollout reporting pass-bar metrics per rung.
- [x] C.4 `training/export_onnx.py --params <params.pkl>`: bakes brax obs normalizer ((x-mean)/std, std already has eps) + MLP (swish) + tanh(mean), opset 17, batch 1. Verifies onnxruntime vs numpy reference on 1000 random obs. (Real-checkpoint verification happens at C.8.)
- [x] C.5 `training/record_reference.py`: 5 s rollout -> `reference_trajectory.json` (qpos, qvel, sensordata, obs, action per 50 Hz frame) plus initial state and command.

### Rung 0 / Rung 1
- [x] C.6 Train locomotion, no perturbation, no DR (R0/R1 baseline). Log run in `rl_optimization_log.md`. (runs/r1 in progress; pipeline + gate harness proven on runs/sanity)
- [x] C.7 Eval R0 (zero command, 10 s, 10 seeds) and R1 (velocity error < 0.2 m/s, 10 seeds). Viewer inspection.
- [x] C.8 Export R1 ONNX + reference_trajectory.json.
- [x] C.9 EARLY VERIFICATION GATE in Unity: replay test (< 1e-4), closed-loop 5 s comparison (upright, speed +-0.1 m/s, cadence +-10 %, actuator force +-15 %). If divergent: STOP, fix timestep / solver / ctrl timing / gains, re-run B.5, repeat. Record outcome.
- [x] C.10 Commit tag `rung1-parity-pass`.

### Rung 2
- [x] C.11 Continue from R1 checkpoint with kicks, cubes, and DR on.
- [x] C.12 Eval R2: 30 N.s push and 1 kg cube hit, 9/10 seeds. Export ONNX.
- [x] C.13 Unity spot-check: fire pool cubes and shove in the testbed, dog survives. Log.

### Rung 3
- [x] C.14 Train getup policy (drops + fallen poses, settle 0.5 s). Eval: standing within 3 s, 9/10 seeds. Export ONNX.
- [x] C.15 Mode switch in `Go2Controller.cs`: locomotion <-> getup on projected-gravity z sign with 0.5 s upright hysteresis. Spot-check: knock it over with cubes, it gets up and resumes walking.
- [x] C.16 Commit tag `ladder-complete`.

## Phase D: Engine Polish & Game Loop (Step 5)
- [x] D.1 Author `Race.unity` via Unity MCP in-editor (no procedural scene code): 9:16 camera rig, ground, HUD anchors, behaviour selector wired to command presets (stand / walk / turn / getup demo).
- [x] D.2 Auto-reset: fallen > 5 s without recovery, stalled > 10 s, or out of bounds -> write keyframe home into qpos/qvel, zero ctrl, park cubes.
- [x] D.3 Telemetry HUD (TAB): state (RACING / STUMBLE / FALLEN_RECOVERING), base velocity, sim tick rate, inference ms.
- [x] D.4 Performance: profile one dog at 60 FPS portrait; inference + mj_step < 5 ms per control step. Record numbers.
- [x] D.5 Re-run full in-engine ladder spot-check. Commit tag `v0.1-playable`.

## Phase F: First race (Map 01, Planar Sprint)
- [x] F.1 `Assets/Scenes/Track01.unity`: 100 m straight corridor; side rails are native MuJoCo world geoms (MjGeom boxes); track surface, start/finish lines, distance markers and finish arch are visual-only meshes (no colliders).
- [x] F.2 `RaceOrchestrator.cs`: 3 s countdown, waypoint steering through the joystick command (vx = cruise, yaw rate from heading error), split times at 25/50/75 m, finish line, checkpoint respawn, Restart button.
- [x] F.3 Headless race tests: `PoRace.exe -batchmode -nographics -raceTest -timeScale 4 [-raceFlipAt 20]`. Plain 74.42 s (1.34 m/s, 0 respawns); flipped at 20 s: 75.67 s.
- [x] F.4 Four racers: `Assets/Scenes/Track01x4.unity` + `MultiRaceOrchestrator.cs`. Four Go2s in ONE MuJoCo model (robot subtree cloned in-editor with `r1_`..`r3_` name prefixes), each with its own policies; lanes merge to a single line mid-track. Headless `-raceTest`: all finish (74.2 / 76.0 / 76.9 / 78.2 s), 162 robot-robot contact episodes (8.1 s), 0 respawns; 1.97 ms per control step measured (budget 5.0).
- [x] F.5 `MapDefinition` assets in `Assets/Resources/Maps` discovered by `MapRegistry`; `TrackBuilder` (editor) builds MuJoCo rails + visuals from the centerline. Map 02 Flat Oval (97.6 m lap, two 180 degree turns of radius 6 m), laps, track-frame chase camera. All four finish 2 laps with no respawns.
- [x] F.6 Randomized races: seeded lane draw and 6 % cruise-speed spread per race (`-seed N` reproduces a race).
- [x] F.7 Pre-race menu (`Menu.unity`, `MenuController`): map selector from the registry, racers 1-4, laps 1-5, Start; Menu button on the race HUD. One combined build `Build/PoRace/PoRace.exe`.
- [ ] F.8 Banked turns (PRD Map 02 calls for 15 degree banking): needs a policy trained on slopes.

## Phase E: Later (tracked, not scheduled)
- [ ] E.1 Android: build mujoco C library for arm64, swap plugin native binary, portrait build.
- [ ] E.2 CreatureDefinition / MapDefinition registries and pre-race menu (PRD FR-01..03).
- [ ] E.3 Rough terrain rung (heightfield scan obs) and multi-agent racing rung.
- [ ] E.4 G1 / H1 / DeepMimic biped onboarding through the same gates.
