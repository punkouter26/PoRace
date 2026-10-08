using System;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRace {

public enum Go2Mode { HoldHome, Locomotion, Getup }

/// <summary>
/// Drives the Go2 inside MjScene. Runs the policy every 5th physics step (50 Hz on a 4 ms step)
/// from MjScene.ctrlCallback, which fires between mj_step1 and mj_step2 exactly like the trainer.
/// Joint order and obs layout: docs/joint_order.md. All model indices come from Go2Model (by name).
/// </summary>
public unsafe class Go2Controller : MonoBehaviour, IParitySource {
  // ---- contract (training/go2/constants.py) ----
  public const float SimDt = 0.004f;
  public const int Decimation = 5;
  public const float ActionScale = 0.5f;
  public const int Nu = 12;
  public const int ObsJoystick = 48;
  public const int ObsGetup = 42;
  public static readonly float[] DefaultPose =
      { 0f, 0.9f, -1.8f, 0f, 0.9f, -1.8f, 0f, 0.9f, -1.8f, 0f, 0.9f, -1.8f };
  public const float HomeHeight = 0.27f;
  public const float CmdSlewStep = 6.0f * SimDt * Decimation;  // 6 m/s^2 (rad/s^2) per control step = constants.CMD_SLEW_STEP

  public Go2Mode mode = Go2Mode.HoldHome;
  public ModelAsset locomotionModel;
  public ModelAsset getupModel;
  [Tooltip("vx (m/s), vy (m/s), yaw rate (rad/s)")]
  public Vector3 command = Vector3.zero;
  public bool autoGetup = true;
  [Tooltip("MuJoCo name prefix of this robot's bodies/joints/actuators/sensors (\"\" for the first, \"r1_\" ... for clones).")]
  public string prefix = "";
  [Tooltip("True when several robots share the MuJoCo model: resets then touch only this robot's state.")]
  public bool multiRobot = false;
  [Tooltip("Where ResetToHome puts the robot, MuJoCo frame (x, y, z up) and heading about z. A race moves this to the last checkpoint.")]
  public Vector3 spawnMj = new Vector3(0f, 0f, HomeHeight);
  public float spawnYaw = 0f;

  public Go2Model Model { get; private set; }
  // IParitySource: canonical order = base, 12 joints, then the 4 pooled cubes (training/go2/constants.py layout).
  public bool Ready => Model != null;
  int IParitySource.Decimation => Decimation;
  public int Nq => 47; public int Nv => 42; int IParitySource.Nu => Nu;
  public void Canonical(MujocoLib.mjData_* d, double[] qpos, double[] qvel, double[] ctrl) { Model.CanonicalQpos(d, qpos); Model.CanonicalQvel(d, qvel); Model.CanonicalCtrl(d, ctrl); }
  public bool Upright { get; private set; } = true;
  public int ControlStep { get; private set; }
  public float LastInferenceMs { get; private set; }
  public float[] LastAction => _lastAction;
  public float[] CurrentObs => _obs;

  /// <summary>Policy actually driving the joints this step (differs from `mode` while auto-recovering).</summary>
  public Go2Mode Active { get; private set; } = Go2Mode.HoldHome;
  Go2Mode _requested = (Go2Mode)(-1);
  double _holdUntil = -1; readonly float[] _holdTargets = new float[Nu];

  /// <summary>Hold the current joint angles as PD targets until sim time `until` (used after teleporting the robot).</summary>
  public void HoldCurrentPose(MujocoLib.mjData_* d, double until) {
    for (int i = 0; i < Nu; i++) { _holdTargets[i] = (float)d->qpos[Model.JointQpos[i]]; SetCtrl(d, i, _holdTargets[i]); }
    _holdUntil = until;
  }

  Vector3 _cmd;  // slew-limited command actually fed to the policy
  public Vector3 FilteredCommand => _cmd;
  PolicyRunner _loco, _getup;
  readonly MjActuator[] _act = new MjActuator[Nu];  // MjActuator.OnSyncState rewrites ctrl from .Control after every step
  readonly float[] _obs = new float[ObsJoystick];
  readonly float[] _lastAction = new float[Nu];
  int _physicsStep;
  float _uprightSince;

  void Awake() {
    Time.fixedDeltaTime = SimDt;  // MuJoCo plugin steps at Unity's fixed timestep; never trust the project setting alone
    MjScene.Instance.postInitEvent += OnSceneInit;
    MjScene.Instance.preUpdateEvent += OnPreStep;
    MjScene.Instance.ctrlCallback += OnCtrl;
  }

  // Trainer convention: after mj_step, sensordata still holds the values computed at the START of that step, while
  // qpos/qvel are post-integration. ctrlCallback runs after mj_step1, where sensordata is already refreshed, so the
  // sensor part of the observation is snapshotted here (before mj_step1) to keep the same 4 ms sensor age.
  readonly double[] _sens = new double[11];  // linvel 3, gyro 3, quat 4, upZ 1
  void OnPreStep(object sender, MjStepArgs a) {
    if (Model == null) return;
    var d = a.data; var M = Model;
    for (int i = 0; i < 3; i++) { _sens[i] = d->sensordata[M.LinvelAdr + i]; _sens[3 + i] = d->sensordata[M.GyroAdr + i]; }
    for (int i = 0; i < 4; i++) _sens[6 + i] = d->sensordata[M.QuatAdr + i];
    _sens[10] = d->sensordata[M.UpAdr + 2];
  }

  void OnDestroy() {
    if (!MjScene.InstanceExists) return;
    MjScene.Instance.postInitEvent -= OnSceneInit;
    MjScene.Instance.preUpdateEvent -= OnPreStep;
    MjScene.Instance.ctrlCallback -= OnCtrl;
  }

  void Start() {
    var args = System.Environment.GetCommandLineArgs();  // headless harness: -command vx vy yaw  -mode HoldHome|Locomotion|Getup
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-command" && i + 3 < args.Length)
        command = new Vector3(float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture),
                              float.Parse(args[i + 2], System.Globalization.CultureInfo.InvariantCulture),
                              float.Parse(args[i + 3], System.Globalization.CultureInfo.InvariantCulture));
      if (args[i] == "-mode" && i + 1 < args.Length) mode = (Go2Mode)Enum.Parse(typeof(Go2Mode), args[i + 1]);
    }
    if (locomotionModel != null) _loco = new PolicyRunner(locomotionModel, ObsJoystick, Nu);
    if (getupModel != null) _getup = new PolicyRunner(getupModel, ObsGetup, Nu);
  }

  void OnApplicationQuit() {
    _loco?.Dispose();
    _getup?.Dispose();
  }

  void OnSceneInit(object sender, MjStepArgs a) {
    var m = a.model;
    // Plugin does not parse these two; the trainer runs with them. See rl_optimization_log.md.
    m->opt.ls_iterations = 5;
    m->opt.disableflags |= (int)MujocoLib.mjtDisableBit.mjDSBL_EULERDAMP;
    Model = new Go2Model(m, prefix);
    ModelCheck.Assert(m, Model, multiRobot ? FindObjectsByType<Go2Controller>(FindObjectsSortMode.None).Length : 1);
    m->opt.ccd_iterations = 35;
    foreach (var act in FindObjectsByType<MjActuator>(FindObjectsSortMode.None))
      for (int i = 0; i < Nu; i++) if (act.MujocoId == Model.ActId[i]) _act[i] = act;
    for (int i = 0; i < Nu; i++) if (_act[i] == null) throw new InvalidOperationException($"MjActuator for {Go2Model.ActuatorNames[i]} not found");
    ResetToHome(a.model, a.data);
  }

  public void ResetToHome() {
    var s = MjScene.Instance;
    ResetToHome(s.Model, s.Data);
    s.SyncUnityToMjState();
  }

  void ResetToHome(MujocoLib.mjModel_* m, MujocoLib.mjData_* d) {
    var M = Model;
    if (!multiRobot) MujocoLib.mj_resetData(m, d);  // single robot: full reset (also zeroes sim time, used by the parity harness)
    else { for (int i = 0; i < 6; i++) d->qvel[M.BaseDof + i] = 0; for (int i = 0; i < Nu; i++) d->qvel[M.JointDof[i]] = 0; }
    d->qpos[M.BaseQpos] = spawnMj.x; d->qpos[M.BaseQpos + 1] = spawnMj.y; d->qpos[M.BaseQpos + 2] = spawnMj.z;
    d->qpos[M.BaseQpos + 3] = Math.Cos(spawnYaw * 0.5); d->qpos[M.BaseQpos + 4] = 0; d->qpos[M.BaseQpos + 5] = 0; d->qpos[M.BaseQpos + 6] = Math.Sin(spawnYaw * 0.5);
    for (int i = 0; i < Nu; i++) { d->qpos[M.JointQpos[i]] = DefaultPose[i]; SetCtrl(d, i, DefaultPose[i]); }
    Array.Clear(_lastAction, 0, Nu);
    _cmd = Vector3.zero;
    _physicsStep = 0; ControlStep = 0;
    if (!multiRobot) CubePool.Instance?.ParkAll(M, d);
    _uprightSince = 0f; _holdUntil = -1; Active = mode; _requested = mode;
    MujocoLib.mj_forward(m, d);
  }

  void OnCtrl(object sender, MjStepArgs a) {
    if (_physicsStep++ % Decimation != 0) return;  // hold ctrl for 5 steps
    ControlStep++;
    var d = a.data; var M = Model;
    double upZ = _sens[10];
    Upright = upZ > 0.0;  // upvector z, same as trainer termination
    if (upZ > 0.9) _uprightSince += SimDt * Decimation; else _uprightSince = 0f;

    if (d->time < _holdUntil) {  // settle phase (trainer getup reset holds ctrl = qpos for 0.5 s)
      for (int i = 0; i < Nu; i++) SetCtrl(d, i, _holdTargets[i]);
      Array.Clear(_lastAction, 0, Nu);
      return;
    }

    // Behaviour switch with hysteresis: fall -> Getup; back to Locomotion only after 0.5 s solidly upright.
    if (mode != _requested) { _requested = mode; Active = mode; }
    if (autoGetup && _getup != null && _loco != null && _requested == Go2Mode.Locomotion) {
      if (Active == Go2Mode.Locomotion && !Upright) { Active = Go2Mode.Getup; Array.Clear(_lastAction, 0, Nu); _cmd = Vector3.zero; }
      else if (Active == Go2Mode.Getup && _uprightSince > 0.5f) { Active = Go2Mode.Locomotion; Array.Clear(_lastAction, 0, Nu); }
    }
    var active = Active;

    float[] action;
    var t0 = Time.realtimeSinceStartupAsDouble;
    switch (active) {
      case Go2Mode.Locomotion when _loco != null:
        _cmd = new Vector3(Mathf.MoveTowards(_cmd.x, command.x, CmdSlewStep), Mathf.MoveTowards(_cmd.y, command.y, CmdSlewStep),
                           Mathf.MoveTowards(_cmd.z, command.z, CmdSlewStep));
        BuildObs(d, withLinvelAndCommand: true);
        action = _loco.Run(_obs, ObsJoystick);
        for (int i = 0; i < Nu; i++) SetCtrl(d, i, DefaultPose[i] + ActionScale * Mathf.Clamp(action[i], -1f, 1f));
        break;
      case Go2Mode.Getup when _getup != null:
        BuildObs(d, withLinvelAndCommand: false);
        action = _getup.Run(_obs, ObsGetup);
        for (int i = 0; i < Nu; i++) SetCtrl(d, i, (float)d->qpos[M.JointQpos[i]] + ActionScale * Mathf.Clamp(action[i], -1f, 1f));
        break;
      default:
        action = null;
        BuildObs(d, withLinvelAndCommand: true);  // keep obs recorded even without a brain
        for (int i = 0; i < Nu; i++) SetCtrl(d, i, DefaultPose[i]);
        break;
    }
    LastInferenceMs = (float)((Time.realtimeSinceStartupAsDouble - t0) * 1000.0);
    if (action != null) Array.Copy(action, _lastAction, Nu); else Array.Clear(_lastAction, 0, Nu);
  }

  /// <summary>Drop the slew-limited command to zero so it ramps up again. The policy braces instead of walking when
  /// it is at rest with a forward command of 1.3 m/s or more (see rl_optimization_log.md); a race uses this to un-stick
  /// a racer that was bumped to a halt.</summary>
  public void RestartCommandRamp() { _cmd = Vector3.zero; }

  Transform _baseTr;
  /// <summary>Unity transform of this robot's base body (kept in sync by the MuJoCo plugin); for cameras and UI.</summary>
  public Transform BaseTransform { get { if (_baseTr == null) { var go = GameObject.Find(prefix + "base"); if (go != null) _baseTr = go.transform; } return _baseTr; } }

  /// <summary>Gameplay: flip the robot onto its back in place (state write only, no scene change) to demo recovery.</summary>
  public void KnockOver() {
    var d = MjScene.Instance.Data; var M = Model; if (M == null) return;
    d->qpos[M.BaseQpos + 2] = 0.45;
    d->qpos[M.BaseQpos + 3] = 0; d->qpos[M.BaseQpos + 4] = 1; d->qpos[M.BaseQpos + 5] = 0; d->qpos[M.BaseQpos + 6] = 0;  // 180 deg about x
    for (int i = 0; i < 6; i++) d->qvel[M.BaseDof + i] = 0;
    MujocoLib.mj_forward(MjScene.Instance.Model, d);
  }

  /// <summary>ctrl is float32 on both sides: the trainer runs Warp in float32 and MjActuator.Control is a float.</summary>
  void SetCtrl(MujocoLib.mjData_* d, int i, float v) {
    d->ctrl[Model.ActId[i]] = v;
    _act[i].Control = v;
  }

  /// <summary>Joystick: linvel 3, gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12, cmd 3.
  /// Getup: gyro 3, gravity 3, qpos-default 12, qvel 12, last_act 12.</summary>
  public void BuildObs(MujocoLib.mjData_* d, bool withLinvelAndCommand) {
    var M = Model;
    int k = 0;
    if (withLinvelAndCommand) for (int i = 0; i < 3; i++) _obs[k++] = (float)_sens[i];
    for (int i = 0; i < 3; i++) _obs[k++] = (float)_sens[3 + i];
    // Projected gravity: R_imu^T * (0,0,-1), from the imu site quaternion (w,x,y,z).
    double w = _sens[6], x = _sens[7], y = _sens[8], z = _sens[9];
    _obs[k++] = (float)(-2 * (x * z - w * y));
    _obs[k++] = (float)(-2 * (y * z + w * x));
    _obs[k++] = (float)(-(1 - 2 * (x * x + y * y)));
    for (int i = 0; i < Nu; i++) _obs[k++] = (float)(d->qpos[M.JointQpos[i]] - DefaultPose[i]);
    for (int i = 0; i < Nu; i++) _obs[k++] = (float)d->qvel[M.JointDof[i]];
    for (int i = 0; i < Nu; i++) _obs[k++] = _lastAction[i];
    if (withLinvelAndCommand) { _obs[k++] = _cmd.x; _obs[k++] = _cmd.y; _obs[k++] = _cmd.z; }
  }
}

}
