# Go2 joint / actuator / state order (locked)

Same index everywhere: `qpos[7+i]`, `qvel[6+i]`, `ctrl[i]`, `action[i]`, obs joint slices.
Python copy: `training/go2/constants.py`. C# copy: `Go2Controller.cs` (must match this table).

| i | joint | actuator | class | joint range (rad) | ctrlrange (rad) | forcerange (N·m) | default pose |
|---|---|---|---|---|---|---|---|
| 0 | FL_hip_joint | FL_hip | abduction | -1.0472 .. 1.0472 | -0.9472 .. 0.9472 | ±24 | 0.0 |
| 1 | FL_thigh_joint | FL_thigh | hip | -1.5708 .. 3.4907 | -1.4 .. 2.5 | ±24 | 0.9 |
| 2 | FL_calf_joint | FL_calf | knee | -2.7227 .. -0.83776 | -2.6227 .. -0.84776 | ±24 | -1.8 |
| 3 | FR_hip_joint | FR_hip | abduction | same | same | ±24 | 0.0 |
| 4 | FR_thigh_joint | FR_thigh | hip | same | same | ±24 | 0.9 |
| 5 | FR_calf_joint | FR_calf | knee | same | same | ±24 | -1.8 |
| 6 | RL_hip_joint | RL_hip | abduction | same | same | ±24 | 0.0 |
| 7 | RL_thigh_joint | RL_thigh | hip | same | same | ±24 | 0.9 |
| 8 | RL_calf_joint | RL_calf | knee | same | same | ±24 | -1.8 |
| 9 | RR_hip_joint | RR_hip | abduction | same | same | ±24 | 0.0 |
| 10 | RR_thigh_joint | RR_thigh | hip | same | same | ±24 | 0.9 |
| 11 | RR_calf_joint | RR_calf | knee | same | same | ±24 | -1.8 |

Note: menagerie Go2 uses the same thigh range for front and rear legs in go2_mjx.xml (the non-MJX go2.xml differs). We ship the MJX variant.

## Free joints
- `qpos[0:7]` base position xyz + quaternion wxyz. `qvel[0:6]` base linear + angular velocity (world frame linear, body frame angular, MuJoCo convention).
- Cubes: `qpos[19+7k : 26+7k]`, `qvel[18+6k : 24+6k]`, k = 0..3. Parked resting on the floor at (20+2k, 20, 0.05).

## Actuator
General affine PD: force = 35 · (ctrl − q) − 0.5 · qdot, clamped to ±24 N·m. Joint damping 0.5, armature 0.01.

## Action → ctrl
- Locomotion: `ctrl = default_pose + 0.5 · clip(action, −1, 1)`
- Getup: `ctrl = qpos[7:] + 0.5 · clip(action, −1, 1)`
Held for 5 physics steps of 4 ms (50 Hz control).

## Observation (joystick, 48)
| slice | content | source |
|---|---|---|
| 0:3 | base linear velocity, body frame | sensor `local_linvel` |
| 3:6 | base angular velocity, body frame | sensor `gyro` |
| 6:9 | projected gravity (−upvector)... see note | sensor `upvector` rotated: gravity = R_imuᵀ · (0,0,−1) |
| 9:21 | joint pos − default pose | `qpos[7:19]` |
| 21:33 | joint vel | `qvel[6:18]` |
| 33:45 | last action | policy memory |
| 45:48 | command (vx, vy, yaw rate) | user |

Getup (42) drops slices 0:3 and 45:48.

## Sensors by name (C# looks these up with mj_name2id; never by address)
gyro, local_linvel, accelerometer, position, upvector, forwardvector, global_linvel, global_angvel, orientation,
FL/FR/RL/RR_global_linvel, FL/FR/RL/RR_pos. Contact sensors (`*_floor_found`) exist only in the training scene.
