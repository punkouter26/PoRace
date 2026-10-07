using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>Records canonical-order qpos/qvel/ctrl plus obs/action every control step for `seconds`, then writes
/// docs/parity/<fileName>. training/zero_brain.py and training/compare_trajectory.py read it.</summary>
public unsafe class ParityRecorder : MonoBehaviour {
  public Go2Controller controller;
  public float seconds = 2f;
  public string fileName = "unity_zero_brain.json";
  public bool quitWhenDone = false;

  readonly List<string> _frames = new();
  readonly double[] _qpos = new double[47], _qvel = new double[42], _ctrl = new double[12];
  int _physicsStep;
  bool _done;

  void Awake() { MjScene.Instance.postUpdateEvent += OnPostStep; }
  void OnDestroy() { if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= OnPostStep; }

  void OnPostStep(object sender, MjStepArgs a) {
    if (_done || controller == null || controller.Model == null) return;
    // ctrlCallback ran inside this same step when _physicsStep % Decimation == 0; record after mj_step2.
    if (_physicsStep++ % Go2Controller.Decimation != 0) return;
    var d = a.data; var M = controller.Model;
    M.CanonicalQpos(d, _qpos); M.CanonicalQvel(d, _qvel); M.CanonicalCtrl(d, _ctrl);
    var sb = new StringBuilder();
    sb.Append("{\"t\":").Append(d->time.ToString("R", CultureInfo.InvariantCulture));
    Append(sb, "qpos", _qpos); Append(sb, "qvel", _qvel); Append(sb, "ctrl", _ctrl);
    Append(sb, "obs", controller.CurrentObs); Append(sb, "action", controller.LastAction);
    sb.Append('}');
    _frames.Add(sb.ToString());
    if (d->time >= seconds) Finish();
  }

  static void Append(StringBuilder sb, string key, double[] v) {
    sb.Append(",\"").Append(key).Append("\":[");
    for (int i = 0; i < v.Length; i++) { if (i > 0) sb.Append(','); sb.Append(v[i].ToString("R", CultureInfo.InvariantCulture)); }
    sb.Append(']');
  }

  static void Append(StringBuilder sb, string key, float[] v) {
    sb.Append(",\"").Append(key).Append("\":[");
    for (int i = 0; i < v.Length; i++) { if (i > 0) sb.Append(','); sb.Append(v[i].ToString("R", CultureInfo.InvariantCulture)); }
    sb.Append(']');
  }

  void Finish() {
    _done = true;
    var dir = Path.Combine(Application.dataPath, "..", "docs", "parity");
    var args = System.Environment.GetCommandLineArgs();  // headless player: -parityDir <abs path>
    for (int i = 0; i + 1 < args.Length; i++) if (args[i] == "-parityDir") dir = args[i + 1];
    Directory.CreateDirectory(dir);
    var path = Path.GetFullPath(Path.Combine(dir, fileName));
    File.WriteAllText(path, "{\"frames\":[\n" + string.Join(",\n", _frames) + "\n]}");
    Debug.Log($"[ParityRecorder] wrote {_frames.Count} frames to {path}");
    if (quitWhenDone) {
#if UNITY_EDITOR
      UnityEditor.EditorApplication.isPlaying = false;
#else
      Application.Quit();
#endif
    }
  }
}

}
