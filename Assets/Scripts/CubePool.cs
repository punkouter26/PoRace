using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>
/// Pre-allocated projectile cubes (cube0..cube3 in go2_unity.xml). Nothing is instantiated at runtime;
/// firing writes qpos/qvel of the next cube and parking writes it back to its rest slot.
/// Parking slots match training/go2/constants.py CUBE_PARK_QPOS. Addresses come from Go2Model (by name).
/// </summary>
public unsafe class CubePool : MonoBehaviour {
  public static CubePool Instance { get; private set; }
  public const int Count = Go2Model.NumCubes;
  public Go2Controller controller;
  int _next;

  void Awake() { Instance = this; if (controller == null) controller = GetComponent<Go2Controller>(); }
  void OnDestroy() { if (Instance == this) Instance = null; }

  public static Vector3 ParkPosition(int k) => new Vector3(20f + 2f * k, 20f, 0.05f);  // MuJoCo frame (x, y, z-up)

  public void ParkAll(Go2Model M, MujocoLib.mjData_* d) {
    for (int k = 0; k < Count; k++) Place(M, d, k, ParkPosition(k), Vector3.zero);
    _next = 0;
  }

  static void Place(Go2Model M, MujocoLib.mjData_* d, int k, Vector3 mjPos, Vector3 mjVel) {
    int q = M.CubeQpos[k], v = M.CubeDof[k];
    d->qpos[q] = mjPos.x; d->qpos[q + 1] = mjPos.y; d->qpos[q + 2] = mjPos.z;
    d->qpos[q + 3] = 1; d->qpos[q + 4] = 0; d->qpos[q + 5] = 0; d->qpos[q + 6] = 0;
    d->qvel[v] = mjVel.x; d->qvel[v + 1] = mjVel.y; d->qvel[v + 2] = mjVel.z;
    d->qvel[v + 3] = 0; d->qvel[v + 4] = 0; d->qvel[v + 5] = 0;
  }

  /// <summary>Place the next pooled cube at a MuJoCo-frame position with a MuJoCo-frame velocity.</summary>
  public int Fire(Vector3 mjPos, Vector3 mjVel) {
    int k = _next; _next = (_next + 1) % Count;
    Place(controller.Model, MjScene.Instance.Data, k, mjPos, mjVel);
    return k;
  }

  Vector3 BasePos() {
    var d = MjScene.Instance.Data; int q = controller.Model.BaseQpos;
    return new Vector3((float)d->qpos[q], (float)d->qpos[q + 1], (float)d->qpos[q + 2]);
  }

  /// <summary>Drop a cube from 1 m above the base with a small random horizontal offset (training C.2).</summary>
  public int DropOnRobot() {
    var o = Random.insideUnitCircle * 0.15f;
    return Fire(BasePos() + new Vector3(o.x, o.y, 1.0f), Vector3.zero);
  }

  /// <summary>Throw a cube at the base from a random horizontal direction, 2 m away.</summary>
  public int ThrowAtRobot(float speed = 3f) {
    float ang = Random.Range(0f, Mathf.PI * 2f);
    var dir = new Vector3(Mathf.Cos(ang), Mathf.Sin(ang), 0f);
    var pos = BasePos() - dir * 2f; pos.z += 0.2f;
    return Fire(pos, dir * speed);
  }
}

}
