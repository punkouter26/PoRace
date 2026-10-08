using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Mujoco;
using UnityEngine;
using UnityEngine.UI;

namespace PoRace {

[Serializable]
public class Racer {
  public string name = "Racer";
  public Go2Controller controller;
  public AutoReset autoReset;
  public float laneY;          // MuJoCo y of this racer's lane (m)
  public float cruiseSpeed = 1.5f;
  public Color color = Color.white;
  [NonSerialized] public float distance;
  [NonSerialized] public double finishTime = -1;
  [NonSerialized] public int place, checkpoint;
  [NonSerialized] public RaceState state;
}

/// <summary>
/// Multi-racer sprint (PRD 4.2 / FR-13): N robots in ONE MuJoCo model, each driven by its own copy of the validated
/// locomotion + getup policies. Racers bump through solver contact only. Steering is the joystick command:
/// vx = cruise, yaw rate toward a look-ahead point on the racer's lane. Lanes pinch toward the centre in the middle of
/// the track so racers have to jostle. MuJoCo frame: x forward along the track, y left.
/// Headless: PoRace.exe -batchmode -nographics -raceTest [-timeScale 4] [-raceFlipAt 20]
/// </summary>
public unsafe class MultiRaceOrchestrator : MonoBehaviour {
  public List<Racer> racers = new List<Racer>();
  public Text raceText, standingsText;
  public FollowCam followCam;
  public float trackLength = 100f;
  public float countdownSeconds = 3f;
  public float lookAhead = 4f;
  public float steerGain = 1.5f;
  public float checkpointSpacing = 25f;
  [Tooltip("Lanes shrink to this fraction of their spacing around the middle of the track (1 = straight lanes).")]
  public float pinch = 0.45f;
  public float halfWidth = 1.65f;

  public RacePhase Phase { get; private set; } = RacePhase.Countdown;
  public double RaceTime { get; private set; }
  double _t0 = -1, _raceStart, _sim, _flipAt = -1; int _finished, _flips; bool _test, _testDone;
  readonly System.Diagnostics.Stopwatch _sw = new System.Diagnostics.Stopwatch();
  double _stepMs, _maxMs; long _steps;

  void Awake() {
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-raceTest") _test = true;
      if (args[i] == "-timeScale" && i + 1 < args.Length) Time.timeScale = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
      if (args[i] == "-raceFlipAt" && i + 1 < args.Length) _flipAt = double.Parse(args[i + 1], CultureInfo.InvariantCulture);
    }
    MjScene.Instance.preUpdateEvent += (s, a) => _sw.Restart();
    MjScene.Instance.postUpdateEvent += OnPostStep;
    foreach (var r in racers) Spawn(r, 0f);
  }

  /// <summary>Lane centre at track position x: full spacing at the ends, pinched in the middle third.</summary>
  public float LaneAt(Racer r, float x) {
    float u = Mathf.Clamp01(x / trackLength);
    float k = Mathf.SmoothStep(1f, pinch, Mathf.InverseLerp(0.15f, 0.4f, u)) * (u < 0.5f ? 1f : 0f)
            + Mathf.SmoothStep(pinch, 1f, Mathf.InverseLerp(0.6f, 0.85f, u)) * (u >= 0.5f ? 1f : 0f);
    return r.laneY * k;
  }

  void Spawn(Racer r, float x) {
    r.controller.spawnMj = new Vector3(x, LaneAt(r, x), Go2Controller.HomeHeight);
    r.controller.spawnYaw = 0f;
  }

  public void Restart() {
    Phase = RacePhase.Countdown; _t0 = -1; RaceTime = 0; _finished = 0; Bumps = 0; BumpSteps = 0;
    foreach (var r in racers) {
      r.distance = 0; r.finishTime = -1; r.place = 0; r.checkpoint = 0;
      r.controller.mode = Go2Mode.Locomotion; r.controller.command = Vector3.zero;
      Spawn(r, 0f); r.controller.ResetToHome();
    }
  }

  void OnPostStep(object sender, MjStepArgs a) {
    double ms = _sw.Elapsed.TotalMilliseconds;
    var d = a.data; double t = d->time; _sim = t;
    if (racers.Count == 0 || racers[0].controller.Model == null) return;
    if (t > 1.0) { _stepMs += ms; _steps++; if (ms > _maxMs) _maxMs = ms; }
    if (_t0 < 0) _t0 = t;

    if (Phase == RacePhase.Countdown && t - _t0 >= countdownSeconds) { Phase = RacePhase.Racing; _raceStart = t; }
    if (Phase == RacePhase.Racing) RaceTime = t - _raceStart;
    if (Phase == RacePhase.Racing && _flipAt >= 0 && RaceTime >= _flipAt) { _flipAt = -1; Leader().controller.KnockOver(); _flips++; }

    foreach (var r in racers) {
      var M = r.controller.Model; var c = r.controller;
      float x = (float)d->qpos[M.BaseQpos], y = (float)d->qpos[M.BaseQpos + 1];
      double qw = d->qpos[M.BaseQpos + 3], qx = d->qpos[M.BaseQpos + 4], qy = d->qpos[M.BaseQpos + 5], qz = d->qpos[M.BaseQpos + 6];
      float yaw = (float)Math.Atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
      r.state = r.autoReset != null ? r.autoReset.State : RaceState.RACING;
      if (Phase == RacePhase.Countdown || r.finishTime >= 0) { c.command = Vector3.zero; if (r.finishTime < 0) r.distance = 0; continue; }

      r.distance = Mathf.Clamp(x, 0f, trackLength);
      if (x >= trackLength) { r.finishTime = RaceTime; r.place = ++_finished; r.distance = trackLength; c.command = Vector3.zero; continue; }
      int cp = Mathf.FloorToInt(x / checkpointSpacing);
      if (cp > r.checkpoint) { r.checkpoint = cp; Spawn(r, cp * checkpointSpacing); }

      float tx = x + lookAhead, ty = LaneAt(r, tx);
      float err = Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, Mathf.Atan2(ty - y, tx - x) * Mathf.Rad2Deg) * Mathf.Deg2Rad;
      c.command = new Vector3(r.cruiseSpeed * Mathf.Max(0.3f, Mathf.Cos(err)), 0f, Mathf.Clamp(steerGain * err, -1f, 1f));
      if (Mathf.Abs(y) > halfWidth + 1f) c.ResetToHome();  // left the corridor: back to the last checkpoint
    }
    if (Phase == RacePhase.Racing && _finished == racers.Count) Phase = RacePhase.Finished;

    // Robot-vs-robot contacts this step (bump mechanics are pure solver contact; this only counts them).
    if (Phase == RacePhase.Racing) {
      var m = a.model; bool any = false;
      for (int i = 0; i < d->ncon; i++) {
        int b1 = m->body_rootid[m->geom_bodyid[d->contact[i].geom1]], b2 = m->body_rootid[m->geom_bodyid[d->contact[i].geom2]];
        if (b1 != b2 && IsRobotRoot(b1) && IsRobotRoot(b2)) { any = true; break; }
      }
      if (any) { BumpSteps++; if (!_inBump) { Bumps++; _inBump = true; } } else _inBump = false;
    }

    if (_test && !_testDone && (Phase == RacePhase.Finished || RaceTime > 300)) {
      _testDone = true;
      bool ok = Phase == RacePhase.Finished;
      var order = racers.OrderBy(r => r.finishTime < 0 ? double.MaxValue : r.finishTime).ToList();
      Debug.Log($"[Race4] {(ok ? "FINISHED" : "DNF")} " + string.Join(" | ", order.Select(r =>
          $"{r.name} {(r.finishTime >= 0 ? r.finishTime.ToString("F2", CultureInfo.InvariantCulture) + "s" : "DNF@" + r.distance.ToString("F0") + "m")} resets={(r.autoReset != null ? r.autoReset.Resets : 0)}")) + $" flips={_flips} bumps={Bumps} bumpTime={BumpSteps * Go2Controller.SimDt:F2}s");
      double mean = _steps > 0 ? _stepMs / _steps : 0;
      Debug.Log($"[Perf4] {racers.Count} racers: physics step mean {mean:F4} ms max {_maxMs:F3} ms | per 50 Hz control step {mean * Go2Controller.Decimation:F3} ms (budget 5.0)");
      Application.Quit(ok ? 0 : 1);
    }
  }

  /// <summary>Distinct robot-robot contact episodes, and physics steps spent in contact.</summary>
  public int Bumps { get; private set; }
  public int BumpSteps { get; private set; }
  bool _inBump;
  bool IsRobotRoot(int body) { foreach (var r in racers) if (r.controller.Model.BaseBody == body) return true; return false; }

  Transform _anchor;
  public Racer Leader() => racers.OrderByDescending(r => r.finishTime >= 0 ? trackLength + 1000 - (float)r.finishTime : r.distance).First();

  void Update() {
    if (racers.Count == 0) return;
    var order = racers.OrderByDescending(r => r.finishTime >= 0 ? trackLength + 1000 - (float)r.finishTime : r.distance).ToList();
    if (followCam != null) {
      // Frame the pack: an anchor at the mean position of the racers still running (all of them once finished).
      if (_anchor == null) { _anchor = new GameObject("PackAnchor").transform; followCam.target = _anchor; }
      var live = order.Where(r => r.finishTime < 0 && r.controller.BaseTransform != null).ToList();
      if (live.Count == 0) live = order.Where(r => r.controller.BaseTransform != null).ToList();
      if (live.Count > 0) {
        var sum = Vector3.zero; foreach (var r in live) sum += r.controller.BaseTransform.position;
        _anchor.position = sum / live.Count;
      }
    }
    if (raceText != null) {
      string head = Phase == RacePhase.Countdown ? Mathf.CeilToInt(Mathf.Max(0f, countdownSeconds - (float)(_sim - Math.Max(0, _t0)))).ToString()
                  : Phase == RacePhase.Racing ? "GO" : "FINISH";
      raceText.text = $"{head}\n{RaceTime:F2} s";
    }
    if (standingsText != null) {
      var sb = new System.Text.StringBuilder();
      for (int i = 0; i < order.Count; i++) {
        var r = order[i];
        string col = ColorUtility.ToHtmlStringRGB(r.color);
        string st = r.finishTime >= 0 ? r.finishTime.ToString("F2", CultureInfo.InvariantCulture) + " s" : $"{r.distance:F0} m" + (r.state == RaceState.RACING ? "" : "  " + (r.state == RaceState.STUMBLE ? "stumble" : "DOWN"));
        sb.Append($"{i + 1}. <color=#{col}>{r.name}</color>  {st}\n");
      }
      sb.Append($"<size=18>bumps {Bumps}</size>");
      standingsText.text = sb.ToString();
    }
  }
}

}
