using System;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRace {

/// <summary>What the parity recorder needs from any creature controller.</summary>
public unsafe interface IParitySource {
  bool Ready { get; }
  int Decimation { get; }
  int Nq { get; } int Nv { get; } int Nu { get; }
  void Canonical(MujocoLib.mjData_* d, double[] qpos, double[] qvel, double[] ctrl);
  float[] CurrentObs { get; }
  float[] LastAction { get; }
}

/// <summary>
/// Unitree G1 inside MjScene: 500 Hz physics, policy at 50 Hz (every 10th step) from MjScene.ctrlCallback.
/// Observation and action follow mujoco_playground's G1 joystick env (training/g1): local linear velocity, gyro,
/// projected gravity (pelvis IMU), command, joint pos/vel, last action and a two-foot gait phase clock.
/// Model addresses are resolved by name; sensors are snapshotted before mj_step1 (trainer sensor age), as for the Go2.
/// </summary>
public unsafe class G1Controller : MonoBehaviour, IParitySource {
  public enum Mode { HoldPose, Locomotion }
  public Mode mode = Mode.HoldPose;
  public ModelAsset locomotionModel;
  [Tooltip("vx (m/s), vy (m/s), yaw rate (rad/s)")] public Vector3 command = Vector3.zero;

  public bool Ready => _ready;
  public int Decimation => G1Spec.Decimation;
  public int Nq => 7 + G1Spec.Nu; public int Nv => 6 + G1Spec.Nu; public int Nu => G1Spec.Nu;
  public float[] CurrentObs => _obs;
  public float[] LastAction => _lastAction;
  public bool Upright { get; private set; } = true;
  public float LastInferenceMs { get; private set; }
  public int BaseQpos { get; private set; }
  public int BaseDof { get; private set; }

  const int N = G1Spec.Nu;
  readonly int[] _jq = new int[N], _jd = new int[N], _act = new int[N];
  readonly MjActuator[] _actComp = new MjActuator[N];
  readonly float[] _obs = new float[G1Spec.ObsDim], _lastAction = new float[N];
  readonly double[] _sens = new double[11];   // linvel 3, gyro 3, quat 4, torso up z
  int _linvel, _gyro, _quat, _up, _step, _ctrlStep; bool _ready;
  PolicyRunner _policy; Vector3 _cmd;

  void Awake() {
    Time.fixedDeltaTime = G1Spec.SimDt;
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-command" && i + 3 < args.Length)
        command = new Vector3(float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture),
                              float.Parse(args[i + 2], System.Globalization.CultureInfo.InvariantCulture),
                              float.Parse(args[i + 3], System.Globalization.CultureInfo.InvariantCulture));
      if (args[i] == "-mode" && i + 1 < args.Length && Enum.TryParse(args[i + 1], out Mode m)) mode = m;
    }
    MjScene.Instance.postInitEvent += OnInit;
    MjScene.Instance.preUpdateEvent += OnPre;
    MjScene.Instance.ctrlCallback += OnCtrl;
  }

  void Start() { if (locomotionModel != null) _policy = new PolicyRunner(locomotionModel, G1Spec.ObsDim, N); }
  void OnApplicationQuit() { _policy?.Dispose(); }

  static int Id(MujocoLib.mjModel_* m, MujocoLib.mjtObj t, string name) {
    int id = MujocoLib.mj_name2id(m, (int)t, name);
    if (id < 0) throw new InvalidOperationException($"G1: {t} '{name}' missing from model");
    return id;
  }

  void OnInit(object sender, MjStepArgs a) {
    var m = a.model;
    m->opt.ls_iterations = 5; m->opt.iterations = 3;
    m->opt.disableflags |= (int)MujocoLib.mjtDisableBit.mjDSBL_EULERDAMP;
    int pelvis = Id(m, MujocoLib.mjtObj.mjOBJ_BODY, "pelvis");
    for (int j = 0; j < (int)m->njnt; j++) if (m->jnt_bodyid[j] == pelvis && m->jnt_type[j] == (int)MujocoLib.mjtJoint.mjJNT_FREE) { BaseQpos = m->jnt_qposadr[j]; BaseDof = m->jnt_dofadr[j]; }
    for (int i = 0; i < N; i++) {
      int j = Id(m, MujocoLib.mjtObj.mjOBJ_JOINT, G1Spec.Joints[i]);
      _jq[i] = m->jnt_qposadr[j]; _jd[i] = m->jnt_dofadr[j];
      _act[i] = Id(m, MujocoLib.mjtObj.mjOBJ_ACTUATOR, G1Spec.Joints[i]);   // actuators are named after their joints
    }
    foreach (var c in FindObjectsByType<MjActuator>(FindObjectsSortMode.None)) for (int i = 0; i < N; i++) if (c.MujocoId == _act[i]) _actComp[i] = c;
    _linvel = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, "local_linvel_pelvis")];
    _gyro = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")];
    _quat = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, "orientation_pelvis")];
    _up = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, "upvector_torso")];
    Check(m, pelvis);
    _ready = true;
    ResetToHome(m, a.data);
  }

  /// <summary>The model Unity regenerated must equal the trainer's (docs/g1/model_summary.json).</summary>
  void Check(MujocoLib.mjModel_* m, int pelvis) {
    void Must(bool ok, string msg) { if (!ok) throw new InvalidOperationException("[G1Check] " + msg); }
    Must((int)m->nq == 36 && (int)m->nv == 35 && (int)m->nu == 29, $"sizes nq={m->nq} nv={m->nv} nu={m->nu}");
    Must(Math.Abs(m->opt.timestep - G1Spec.SimDt) < 1e-9, $"timestep {m->opt.timestep}");
    Must(m->opt.integrator == (int)MujocoLib.mjtIntegrator.mjINT_EULER && m->opt.cone == (int)MujocoLib.mjtCone.mjCONE_PYRAMIDAL, "integrator/cone");
    Must(m->opt.iterations == 3 && m->opt.ls_iterations == 5 && m->opt.solver == 2, $"solver {m->opt.solver} iterations {m->opt.iterations}/{m->opt.ls_iterations}");
    Must(Math.Abs(m->opt.impratio - 1) < 1e-9, $"impratio {m->opt.impratio}");
    double mass = 0; for (int b = 0; b < (int)m->nbody; b++) if (m->body_rootid[b] == pelvis) mass += m->body_mass[b];
    Must(Math.Abs(mass - G1Spec.Mass) < 1e-3, $"mass {mass}");
    for (int i = 0; i < N; i++) {
      int a = _act[i];
      Must(Math.Abs(m->actuator_gainprm[a * 10] - G1Spec.Kp[i]) < 1e-6 && Math.Abs(m->actuator_biasprm[a * 10 + 1] + G1Spec.Kp[i]) < 1e-6, $"kp[{i}] {m->actuator_gainprm[a * 10]}");
      Must(Math.Abs(m->actuator_biasprm[a * 10 + 2]) < 1e-9, $"kv[{i}] {m->actuator_biasprm[a * 10 + 2]}");
      Must(Math.Abs(m->actuator_forcerange[a * 2 + 1] - G1Spec.ForceMax[i]) < 1e-6 && m->actuator_forcelimited[a] != 0, $"forcerange[{i}] {m->actuator_forcerange[a * 2 + 1]}");
      Must(Math.Abs(m->dof_damping[_jd[i]] - G1Spec.Damping[i]) < 1e-6, $"damping[{i}] {m->dof_damping[_jd[i]]}");
      Must(Math.Abs(m->dof_frictionloss[_jd[i]] - 0.1) < 1e-6, $"frictionloss[{i}] {m->dof_frictionloss[_jd[i]]}");
      Must(_actComp[i] != null, $"MjActuator component for {G1Spec.Joints[i]}");
    }
    Debug.Log($"[G1Check] OK nq={m->nq} nv={m->nv} nu={m->nu} mass={mass:F4} dt={m->opt.timestep}");
  }

  void ResetToHome(MujocoLib.mjModel_* m, MujocoLib.mjData_* d) {
    MujocoLib.mj_resetData(m, d);
    for (int i = 0; i < 7; i++) d->qpos[BaseQpos + i] = G1Spec.InitQpos[i];
    for (int i = 0; i < N; i++) { d->qpos[_jq[i]] = G1Spec.DefaultPose[i]; SetCtrl(d, i, G1Spec.DefaultPose[i]); }
    Array.Clear(_lastAction, 0, N); _cmd = Vector3.zero; _step = 0; _ctrlStep = 0;
    MujocoLib.mj_forward(m, d);
  }

  void SetCtrl(MujocoLib.mjData_* d, int i, float v) { d->ctrl[_act[i]] = v; _actComp[i].Control = v; }

  void OnPre(object sender, MjStepArgs a) {
    if (!_ready) return; var d = a.data;
    for (int i = 0; i < 3; i++) { _sens[i] = d->sensordata[_linvel + i]; _sens[3 + i] = d->sensordata[_gyro + i]; }
    for (int i = 0; i < 4; i++) _sens[6 + i] = d->sensordata[_quat + i];
    _sens[10] = d->sensordata[_up + 2];
  }

  void OnCtrl(object sender, MjStepArgs a) {
    if (!_ready || _step++ % G1Spec.Decimation != 0) return;
    var d = a.data;
    Upright = _sens[10] > 0.0;
    _cmd = new Vector3(Mathf.MoveTowards(_cmd.x, command.x, Go2Controller.CmdSlewStep), Mathf.MoveTowards(_cmd.y, command.y, Go2Controller.CmdSlewStep),
                       Mathf.MoveTowards(_cmd.z, command.z, Go2Controller.CmdSlewStep));
    BuildObs(d);
    var t0 = Time.realtimeSinceStartupAsDouble;
    if (mode == Mode.Locomotion && _policy != null) {
      var act = _policy.Run(_obs, G1Spec.ObsDim);
      for (int i = 0; i < N; i++) { _lastAction[i] = act[i]; SetCtrl(d, i, G1Spec.DefaultPose[i] + G1Spec.ActionScale * act[i]); }   // trainer does not clip actions
    } else {
      for (int i = 0; i < N; i++) { _lastAction[i] = 0f; SetCtrl(d, i, G1Spec.DefaultPose[i]); }
    }
    LastInferenceMs = (float)((Time.realtimeSinceStartupAsDouble - t0) * 1000.0);
    _ctrlStep++;
  }

  /// <summary>linvel 3, gyro 3, gravity 3, command 3, q - default 29, qvel 29, last action 29, cos(phase) 2, sin(phase) 2.
  /// The trainer builds the observation before advancing the phase, so the reset obs and the first step obs share phase 0.</summary>
  void BuildObs(MujocoLib.mjData_* d) {
    int k = 0;
    for (int i = 0; i < 3; i++) _obs[k++] = (float)_sens[i];
    for (int i = 0; i < 3; i++) _obs[k++] = (float)_sens[3 + i];
    double w = _sens[6], x = _sens[7], y = _sens[8], z = _sens[9];
    _obs[k++] = (float)(-2 * (x * z - w * y)); _obs[k++] = (float)(-2 * (y * z + w * x)); _obs[k++] = (float)(-(1 - 2 * (x * x + y * y)));
    _obs[k++] = _cmd.x; _obs[k++] = _cmd.y; _obs[k++] = _cmd.z;
    for (int i = 0; i < N; i++) _obs[k++] = (float)(d->qpos[_jq[i]] - G1Spec.DefaultPose[i]);
    for (int i = 0; i < N; i++) _obs[k++] = (float)d->qvel[_jd[i]];
    for (int i = 0; i < N; i++) _obs[k++] = _lastAction[i];
    double dphi = 2 * Math.PI * G1Spec.SimDt * G1Spec.Decimation * G1Spec.GaitHz;
    int n = Math.Max(0, _ctrlStep - 1);
    double pl = Wrap(n * dphi), pr = Wrap(Math.PI + n * dphi);
    _obs[k++] = (float)Math.Cos(pl); _obs[k++] = (float)Math.Cos(pr); _obs[k++] = (float)Math.Sin(pl); _obs[k++] = (float)Math.Sin(pr);
  }
  static double Wrap(double p) { p = (p + Math.PI) % (2 * Math.PI); if (p < 0) p += 2 * Math.PI; return p - Math.PI; }

  public void Canonical(MujocoLib.mjData_* d, double[] qpos, double[] qvel, double[] ctrl) {
    for (int i = 0; i < 7; i++) qpos[i] = d->qpos[BaseQpos + i];
    for (int i = 0; i < 6; i++) qvel[i] = d->qvel[BaseDof + i];
    for (int i = 0; i < N; i++) { qpos[7 + i] = d->qpos[_jq[i]]; qvel[6 + i] = d->qvel[_jd[i]]; ctrl[i] = d->ctrl[_act[i]]; }
  }
}

}
