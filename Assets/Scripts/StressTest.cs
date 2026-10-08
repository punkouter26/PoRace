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

  void Awake() {
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-stress") _on = true;
      if (args[i] == "-stressSeed" && i + 1 < args.Length) UnityEngine.Random.InitState(int.Parse(args[i + 1]));
    }
    if (_on) MjScene.Instance.postUpdateEvent += OnPostStep;
  }

  void OnPostStep(object sender, MjStepArgs a) {
    if (_done || controller.Model == null) return;
    double t = a.data->time; double up = a.data->sensordata[controller.Model.UpAdr + 2];
    if (!_pushed && t >= 2.0) { shove.PushRandom(); _pushed = true; }
    if (!_dropped && t >= 5.0) { cubes.DropOnRobot(); _dropped = true; }
    if (t > 2.5) { _minUp = Math.Min(_minUp, up); if (up <= 0) _ok = false; }
    if (t >= 8.0) {
      _done = true;
      Debug.Log($"[Stress] {(_ok ? "SURVIVED" : "FELL")} minUpZ={_minUp:F3} baseZ={a.data->qpos[controller.Model.BaseQpos + 2]:F3}");
      Application.Quit(_ok ? 0 : 1);
    }
  }
}

}
