using System;
using Mujoco;

namespace PoRace {

/// <summary>
/// Name-resolved addresses into the model Unity regenerated from its component tree. Unity orders bodies,
/// joints and actuators by EntityId, which is NOT stable between runs, so nothing may use a hard-coded index.
/// Canonical order (docs/joint_order.md): base free joint, 12 joints FL/FR/RL/RR x hip/thigh/calf, cubes 0..3.
/// </summary>
public unsafe sealed class Go2Model {
  public static readonly string[] Legs = { "FL", "FR", "RL", "RR" };
  public static readonly string[] JointNames = Build("{0}_{1}_joint");
  public static readonly string[] ActuatorNames = Build("{0}_{1}");
  public const int Nu = 12, NumCubes = 4;

  public int BaseBody, BaseQpos, BaseDof;
  public readonly int[] JointQpos = new int[Nu], JointDof = new int[Nu], ActId = new int[Nu];
  public readonly int[] CubeBody = new int[NumCubes], CubeQpos = new int[NumCubes], CubeDof = new int[NumCubes];
  public int GyroAdr, LinvelAdr, QuatAdr, UpAdr;

  static string[] Build(string fmt) {
    var r = new string[Nu]; int k = 0;
    foreach (var leg in Legs) foreach (var j in new[] { "hip", "thigh", "calf" }) r[k++] = string.Format(fmt, leg, j);
    return r;
  }

  public static int Id(MujocoLib.mjModel_* m, MujocoLib.mjtObj type, string name) {
    int id = MujocoLib.mj_name2id(m, (int)type, name);
    if (id < 0) throw new InvalidOperationException($"{type} '{name}' missing from model");
    return id;
  }

  static int FreeJointOf(MujocoLib.mjModel_* m, int body) {
    for (int j = 0; j < (int)m->njnt; j++)
      if (m->jnt_bodyid[j] == body && m->jnt_type[j] == (int)MujocoLib.mjtJoint.mjJNT_FREE) return j;
    throw new InvalidOperationException($"body {body} has no free joint");
  }

  public readonly string Prefix;
  public Go2Model(MujocoLib.mjModel_* m, string prefix = "") {
    Prefix = prefix ?? "";
    BaseBody = Id(m, MujocoLib.mjtObj.mjOBJ_BODY, Prefix + "base");
    int bj = FreeJointOf(m, BaseBody);
    BaseQpos = m->jnt_qposadr[bj]; BaseDof = m->jnt_dofadr[bj];
    for (int i = 0; i < Nu; i++) {
      int j = Id(m, MujocoLib.mjtObj.mjOBJ_JOINT, Prefix + JointNames[i]);
      JointQpos[i] = m->jnt_qposadr[j]; JointDof[i] = m->jnt_dofadr[j];
      ActId[i] = Id(m, MujocoLib.mjtObj.mjOBJ_ACTUATOR, Prefix + ActuatorNames[i]);
      if (m->actuator_trnid[ActId[i] * 2] != j) throw new InvalidOperationException($"actuator {ActuatorNames[i]} does not drive {JointNames[i]}");
    }
    for (int k = 0; k < NumCubes; k++) {
      CubeBody[k] = Id(m, MujocoLib.mjtObj.mjOBJ_BODY, "cube" + k);
      int j = FreeJointOf(m, CubeBody[k]);
      CubeQpos[k] = m->jnt_qposadr[j]; CubeDof[k] = m->jnt_dofadr[j];
    }
    GyroAdr = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, Prefix + "gyro")];
    LinvelAdr = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, Prefix + "local_linvel")];
    QuatAdr = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, Prefix + "orientation")];
    UpAdr = m->sensor_adr[Id(m, MujocoLib.mjtObj.mjOBJ_SENSOR, Prefix + "upvector")];
  }

  /// <summary>qpos in canonical order: base 7, joints 12, cubes 4x7 (= training/go2/constants.py layout).</summary>
  public void CanonicalQpos(MujocoLib.mjData_* d, double[] dst) {
    int k = 0;
    for (int i = 0; i < 7; i++) dst[k++] = d->qpos[BaseQpos + i];
    for (int i = 0; i < Nu; i++) dst[k++] = d->qpos[JointQpos[i]];
    for (int c = 0; c < NumCubes; c++) for (int i = 0; i < 7; i++) dst[k++] = d->qpos[CubeQpos[c] + i];
  }

  public void CanonicalQvel(MujocoLib.mjData_* d, double[] dst) {
    int k = 0;
    for (int i = 0; i < 6; i++) dst[k++] = d->qvel[BaseDof + i];
    for (int i = 0; i < Nu; i++) dst[k++] = d->qvel[JointDof[i]];
    for (int c = 0; c < NumCubes; c++) for (int i = 0; i < 6; i++) dst[k++] = d->qvel[CubeDof[c] + i];
  }

  public void CanonicalCtrl(MujocoLib.mjData_* d, double[] dst) {
    for (int i = 0; i < Nu; i++) dst[i] = d->ctrl[ActId[i]];
  }
}

}
