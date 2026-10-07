using System;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace PoRace {

/// <summary>
/// B.10 replay gate: feeds recorded observations from reference_trajectory.json into the in-engine policy and
/// asserts max |action_unity - action_trainer| < 1e-4. Headless: `PoRace.exe -batchmode -nographics -replay <json>`.
/// Frames are {"obs":[...],"action":[...]} objects; parsing is deliberately minimal (no JSON dependency).
/// </summary>
public class ReplayHarness : MonoBehaviour {
  public Go2Controller controller;
  public float tolerance = 1e-4f;

  void Start() {
    var args = Environment.GetCommandLineArgs();
    string path = null;
    for (int i = 0; i + 1 < args.Length; i++) if (args[i] == "-replay") path = args[i + 1];
    if (path == null) return;
    int code = Run(path, controller.locomotionModel, Go2Controller.ObsJoystick) ? 0 : 1;
    Application.Quit(code);
  }

  public bool Run(string path, Unity.InferenceEngine.ModelAsset asset, int obsDim) {
    using var policy = new PolicyRunner(asset, obsDim, Go2Controller.Nu);
    var text = File.ReadAllText(path);
    int n = 0; float worst = 0f; int worstFrame = -1;
    int pos = 0;
    while (true) {
      var obs = NextArray(text, "\"obs\"", ref pos); if (obs == null) break;
      var act = NextArray(text, "\"action\"", ref pos); if (act == null) break;
      if (obs.Length != obsDim) { Debug.LogError($"[Replay] frame {n}: obs {obs.Length} != {obsDim}"); return false; }
      var y = policy.Run(obs, obsDim);
      for (int i = 0; i < Go2Controller.Nu; i++) {
        float d = Mathf.Abs(y[i] - act[i]);
        if (d > worst) { worst = d; worstFrame = n; }
      }
      n++;
    }
    bool ok = n > 0 && worst < tolerance;
    Debug.Log($"[Replay] {(ok ? "PASS" : "FAIL")} frames={n} worst|da|={worst:E2} at frame {worstFrame} tol={tolerance:E0}");
    return ok;
  }

  static float[] NextArray(string s, string key, ref int pos) {
    int k = s.IndexOf(key, pos, StringComparison.Ordinal); if (k < 0) return null;
    int a = s.IndexOf('[', k), b = s.IndexOf(']', a);
    pos = b + 1;
    var parts = s.Substring(a + 1, b - a - 1).Split(',');
    var r = new float[parts.Length];
    for (int i = 0; i < parts.Length; i++) r[i] = float.Parse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture);
    return r;
  }
}

}
