using System;
using System.Diagnostics;
using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>D.4: measures wall time of one physics step (mj_step + policy inference when it runs) between MjScene
/// preUpdate and postUpdate. Headless: PoRace.exe -batchmode -nographics -perf 10 logs the average after N seconds.</summary>
public unsafe class PerfProbe : MonoBehaviour {
  public Go2Controller controller;
  readonly Stopwatch _sw = new Stopwatch();
  double _stepMs, _maxMs, _infMs; long _n, _nInf; float _seconds = -1; bool _done;
  public double MeanStepMs => _n > 0 ? _stepMs / _n : 0;

  void Awake() {
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i + 1 < args.Length; i++) if (args[i] == "-perf") _seconds = float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture);
    MjScene.Instance.preUpdateEvent += (s, a) => _sw.Restart();
    MjScene.Instance.postUpdateEvent += OnPost;
  }

  void OnPost(object sender, MjStepArgs a) {
    double ms = _sw.Elapsed.TotalMilliseconds;
    if (a.data->time < 1.0) return;  // skip warm-up (first inference allocates)
    _stepMs += ms; _n++; if (ms > _maxMs) _maxMs = ms;
    if (_n % Go2Controller.Decimation == 0) { _infMs += controller.LastInferenceMs; _nInf++; }
    if (!_done && _seconds > 0 && a.data->time >= 1.0 + _seconds) {
      _done = true;
      double perCtrl = MeanStepMs * Go2Controller.Decimation;
      UnityEngine.Debug.Log($"[Perf] physics step mean {MeanStepMs:F4} ms max {_maxMs:F3} ms | inference mean {_infMs / Math.Max(1, _nInf):F4} ms | per 50 Hz control step (5 mj_step + 1 inference) {perCtrl:F3} ms | x4 racers {perCtrl * 4:F3} ms (budget 5.0)");
      Application.Quit(0);
    }
  }
}

}
