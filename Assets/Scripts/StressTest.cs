using System;
using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>C.13 in-engine robustness spot-check, same scenario as training/eval.py r2:
/// 30 N·s push at 2 s, pooled cube dropped from 1 m at 5 s, must be upright from 2.5 s to 8 s.
/// Headless: PoRace.exe -batchmode -nographics -stress [-stressSeed N] -mode Locomotion -command 0.5 0 0 -seconds 0</summary>
public unsafe class StressTest : MonoBehaviour {
  public Go2Controller controller;
  public Shove shove;
  public CubePool cubes;
  bool _on, _pushed, _dropped, _done; bool _ok = true; double _minUp = 1;
  bool _getupTest, _placed; double _upAt = -1, _fellAt = -1;
  static readonly float[] Lo = { -1.0472f, -1.5708f, -2.7227f }, Hi = { 1.0472f, 3.4907f, -0.83776f };

  void Awake() {
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-stress") _on = true;
      if (args[i] == "-getupTest") { _on = true; _getupTest = true; }
      if (args[i] == "-comboTest") { _on = true; _combo = true; }
      if (args[i] == "-stressSeed" && i + 1 < args.Length) UnityEngine.Random.InitState(int.Parse(args[i + 1]));
    }
    if (!_on) return;
    var ar = controller.GetComponent<AutoReset>(); if (ar != null) ar.enabled = false;  // a reset would fake a recovery
    MjScene.Instance.postUpdateEvent += OnPostStep;
  }

  void OnPostStep(object sender, MjStepArgs a) {
    if (_done || controller.Model == null) return;
    double t = a.data->time; double up = a.data->sensordata[controller.Model.UpAdr + 2];
    if (_getupTest) { GetupStep(a, t, up); return; }
    if (_combo) { ComboStep(a, t, up); return; }
    if (!_pushed && t >= 2.0) { shove.PushRandom(); _pushed = true; }
    if (!_dropped && t >= 5.0) { cubes.DropOnRobot(); _dropped = true; }
    if (t > 2.5) { _minUp = Math.Min(_minUp, up); if (up <= 0 && _ok) { _ok = false; _fellAt = t; } }
    if (t >= 8.0) {
      _done = true;
      Debug.Log($"[Stress] {(_ok ? "SURVIVED" : "FELL")} minUpZ={_minUp:F3} baseZ={a.data->qpos[controller.Model.BaseQpos + 2]:F3} fellAt={_fellAt:F2}s pushDir=({shove.LastDir.x:F2},{shove.LastDir.y:F2})");
      Application.Quit(_ok ? 0 : 1);
    }
  }

  /// <summary>R3 in engine, same as training/eval.py r3: random orientation at 0.5 m with random joint angles,
  /// hold for 0.5 s, then the policy must be upright with base above 0.22 m within 3 s and still up at 5 s.</summary>
  void GetupStep(MjStepArgs a, double t, double up) {
    var d = a.data; var M = controller.Model;
    if (!_placed) {
      _placed = true;
      var q = UnityEngine.Random.rotationUniform;
      d->qpos[M.BaseQpos] = 0; d->qpos[M.BaseQpos + 1] = 0; d->qpos[M.BaseQpos + 2] = 0.5;
      d->qpos[M.BaseQpos + 3] = q.w; d->qpos[M.BaseQpos + 4] = q.x; d->qpos[M.BaseQpos + 5] = q.y; d->qpos[M.BaseQpos + 6] = q.z;
      for (int i = 0; i < 12; i++) d->qpos[M.JointQpos[i]] = UnityEngine.Random.Range(Lo[i % 3], Hi[i % 3]);
      for (int i = 0; i < 6; i++) d->qvel[M.BaseDof + i] = 0;
      for (int i = 0; i < 12; i++) d->qvel[M.JointDof[i]] = 0;
      controller.HoldCurrentPose(d, t + 0.5);
      MujocoLib.mj_forward(a.model, d);
      _t0 = t + 0.5; return;
    }
    double z = d->qpos[M.BaseQpos + 2];
    if (t > _t0 && _upAt < 0 && up > 0 && z > 0.22) _upAt = t - _t0;
    if (t >= _t0 + 5.0) {
      _done = true;
      bool ok = _upAt >= 0 && _upAt <= 3.0 && up > 0;
      Debug.Log($"[Getup] {(ok ? "PASS" : "FAIL")} upAt={_upAt:F2}s finalUpZ={up:F3} baseZ={z:F3} active={controller.Active}");
      Application.Quit(ok ? 0 : 1);
    }
  }
  double _t0;

  /// <summary>End-to-end recovery, same as training/eval.py combo: walk at the commanded speed, get flipped onto a
  /// fallen pose at 2 s, auto-switch to the getup policy, switch back, and be walking again (> 0.3 m/s over 8-10 s).</summary>
  bool _combo, _flipped, _sawGetup; double _backAt = -1, _vSum; int _vN;
  void ComboStep(MjStepArgs a, double t, double up) {
    var d = a.data; var M = controller.Model;
    if (!_flipped && t >= 2.0) {
      _flipped = true;
      Quaternion q; do { q = UnityEngine.Random.rotationUniform; } while (1 - 2 * (q.x * q.x + q.y * q.y) > -0.2f);
      d->qpos[M.BaseQpos + 2] = 0.5;
      d->qpos[M.BaseQpos + 3] = q.w; d->qpos[M.BaseQpos + 4] = q.x; d->qpos[M.BaseQpos + 5] = q.y; d->qpos[M.BaseQpos + 6] = q.z;
      for (int i = 0; i < 12; i++) d->qpos[M.JointQpos[i]] = UnityEngine.Random.Range(Lo[i % 3], Hi[i % 3]);
      for (int i = 0; i < 6; i++) d->qvel[M.BaseDof + i] = 0;
      for (int i = 0; i < 12; i++) d->qvel[M.JointDof[i]] = 0;
      MujocoLib.mj_forward(a.model, d);
      return;
    }
    if (_flipped && controller.Active == Go2Mode.Getup) _sawGetup = true;
    if (_sawGetup && _backAt < 0 && controller.Active == Go2Mode.Locomotion) _backAt = t - 2.0;
    if (t >= 8.0) { _vSum += d->sensordata[M.LinvelAdr]; _vN++; }
    if (t >= 10.0) {
      _done = true;
      double v = _vSum / Math.Max(1, _vN);
      bool ok = _sawGetup && _backAt >= 0 && _backAt <= 5.0 && v > 0.3 && up > 0;
      Debug.Log($"[Combo] {(ok ? "PASS" : "FAIL")} sawGetup={_sawGetup} backAfter={_backAt:F2}s finalSpeed={v:F2} upZ={up:F2}");
      Application.Quit(ok ? 0 : 1);
    }
  }
}

}
