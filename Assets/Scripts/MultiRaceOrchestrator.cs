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
  [Tooltip("Root of this racer's MuJoCo subtree. Null for the first racer, which is always present.")]
  public GameObject mjRoot;
  public float cruiseSpeed = 1.5f;
  public Color color = Color.white;
  [NonSerialized] public float lane;           // lateral offset from the centerline, +left (m)
  [NonSerialized] public float speed;          // this race's cruise speed after the random draw
  [NonSerialized] public float distance;       // metres raced, laps included
  [NonSerialized] public int segment, lap, checkpoint, place;
  [NonSerialized] public double finishTime = -1;
  [NonSerialized] public RaceState state;
  [NonSerialized] public bool active = true;
  [NonSerialized] public float stuckFor; [NonSerialized] public int unsticks;
}

/// <summary>
/// Multi-racer race on any MapDefinition (PRD 4.2 / FR-13): N robots in ONE MuJoCo model, each driven by its own copy
/// of the validated locomotion + getup policies. Racers bump through solver contact only. Steering is the joystick
/// command: vx = cruise, yaw rate toward a look-ahead point on the racer's lane of the map centerline.
/// Lane draw and speed variation are random per race (seeded). MuJoCo frame: x, y on the ground, z up.
/// Headless: PoRace.exe -batchmode -nographics -raceTest [-timeScale 4] [-seed 7] [-racers 4] [-laps 2] [-raceFlipAt 20]
/// </summary>
public unsafe class MultiRaceOrchestrator : MonoBehaviour {
  public MapDefinition map;
  public List<Racer> racers = new List<Racer>();
  public Text raceText, standingsText;
  public FollowCam followCam;
  public float countdownSeconds = 3f;
  public float steerGain = 1.5f;
  [Tooltip("Each race draws a cruise speed in [1 - speedSpread, 1] x the racer's cruiseSpeed.")]
  public float speedSpread = 0.06f;
  public bool randomizeGrid = true;

  public RacePhase Phase { get; private set; } = RacePhase.Countdown;
  public double RaceTime { get; private set; }
  public int Laps { get; private set; } = 1;
  public int Seed { get; private set; }
  public int Bumps { get; private set; }
  public int BumpSteps { get; private set; }
  float _total; double _t0 = -1, _raceStart, _sim, _flipAt = -1; int _finished, _flips; bool _test, _testDone, _inBump;
  readonly System.Diagnostics.Stopwatch _sw = new System.Diagnostics.Stopwatch();
  double _stepMs, _maxMs; long _steps;
  List<Racer> _live = new List<Racer>();
  Transform _anchor; Vector3 _camForward = Vector3.right;

  void Awake() {
    int n = Mathf.Clamp(RaceConfig.fromMenu ? RaceConfig.racers : racers.Count, 1, racers.Count);
    int laps = RaceConfig.fromMenu ? RaceConfig.laps : 0; int seed = RaceConfig.fromMenu ? RaceConfig.seed : -1;
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-raceTest") _test = true;
      if (i + 1 >= args.Length) continue;
      if (args[i] == "-timeScale") Time.timeScale = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
      if (args[i] == "-raceFlipAt") _flipAt = double.Parse(args[i + 1], CultureInfo.InvariantCulture);
      if (args[i] == "-seed") seed = int.Parse(args[i + 1]);
      if (args[i] == "-racers") n = Mathf.Clamp(int.Parse(args[i + 1]), 1, racers.Count);
      if (args[i] == "-laps") laps = int.Parse(args[i + 1]);
    }
    Laps = map.closed ? Mathf.Clamp(laps > 0 ? laps : map.defaultLaps, 1, 5) : 1;
    _total = map.Length * Laps;
    Seed = seed >= 0 ? seed : Environment.TickCount & 0xFFFF;

    // Racers beyond the chosen count are removed from the MuJoCo model before MjScene compiles it (in Start).
    for (int i = 0; i < racers.Count; i++) {
      racers[i].active = i < n;
      if (i >= n) { if (racers[i].mjRoot != null) racers[i].mjRoot.SetActive(false); racers[i].controller.gameObject.SetActive(false); }
    }
    _live = racers.Where(r => r.active).ToList();
    Draw();
    MjScene.Instance.preUpdateEvent += (s, a) => _sw.Restart();
    MjScene.Instance.postUpdateEvent += OnPostStep;
  }

  /// <summary>Random lane draw and speed variation for this race, reproducible from Seed.</summary>
  void Draw() {
    var rng = new System.Random(Seed);
    var lanes = map.LaneOffsets(_live.Count).ToList();
    if (randomizeGrid) lanes = lanes.OrderBy(_ => rng.Next()).ToList();
    for (int i = 0; i < _live.Count; i++) {
      var r = _live[i];
      r.lane = lanes[i];
      r.speed = r.cruiseSpeed * (1f - speedSpread * (float)rng.NextDouble());
      r.distance = 0; r.segment = 0; r.lap = 0; r.checkpoint = 0; r.finishTime = -1; r.place = 0; r.stuckFor = 0; r.unsticks = 0;
      Spawn(r, 0f);
    }
  }

  void Spawn(Racer r, float s) {
    var p = map.LanePoint(s, r.lane, out var t);
    r.controller.spawnMj = new Vector3(p.x, p.y, Go2Controller.HomeHeight);
    r.controller.spawnYaw = Mathf.Atan2(t.y, t.x);
  }

  public void Restart() {
    Phase = RacePhase.Countdown; _t0 = -1; RaceTime = 0; _finished = 0; Bumps = 0; BumpSteps = 0;
    Seed = (Seed * 1103515245 + 12345) & 0xFFFF;   // a new draw each restart
    Draw();
    foreach (var r in _live) { r.controller.mode = Go2Mode.Locomotion; r.controller.command = Vector3.zero; r.controller.ResetToHome(); }
  }

  /// <summary>Advance the racer's segment so its projection stays inside it; returns metres raced including laps.</summary>
  float Progress(Racer r, Vector2 pos, out float lateral) {
    int n = map.SegmentCount; float t = 0, len = 1; Vector2 dir = Vector2.right, a = Vector2.zero;
    for (int guard = 0; guard < n + 1; guard++) {
      a = map.P(r.segment); dir = (map.P(r.segment + 1) - a); len = dir.magnitude; dir /= len;
      t = Vector2.Dot(pos - a, dir);
      if (t <= len) break;
      if (r.segment + 1 < n) r.segment++;
      else if (map.closed) { r.segment = 0; r.lap++; }
      else break;   // past the end of an open track
    }
    lateral = Vector2.Dot(pos - a, MapDefinition.Left(dir));
    float along = map.closed ? Mathf.Clamp(t, 0f, len) : Mathf.Max(0f, t);
    return r.lap * map.Length + map.SegmentStart(r.segment) + along;
  }

  void OnPostStep(object sender, MjStepArgs a) {
    double ms = _sw.Elapsed.TotalMilliseconds;
    var d = a.data; double t = d->time; _sim = t;
    if (_live.Count == 0 || _live[0].controller.Model == null) return;
    if (t > 1.0) { _stepMs += ms; _steps++; if (ms > _maxMs) _maxMs = ms; }
    if (_t0 < 0) _t0 = t;

    if (Phase == RacePhase.Countdown && t - _t0 >= countdownSeconds) { Phase = RacePhase.Racing; _raceStart = t; }
    if (Phase == RacePhase.Racing) RaceTime = t - _raceStart;
    if (Phase == RacePhase.Racing && _flipAt >= 0 && RaceTime >= _flipAt) { _flipAt = -1; Leader().controller.KnockOver(); _flips++; }

    foreach (var r in _live) {
      var M = r.controller.Model; var c = r.controller;
      var pos = new Vector2((float)d->qpos[M.BaseQpos], (float)d->qpos[M.BaseQpos + 1]);
      double qw = d->qpos[M.BaseQpos + 3], qx = d->qpos[M.BaseQpos + 4], qy = d->qpos[M.BaseQpos + 5], qz = d->qpos[M.BaseQpos + 6];
      float yaw = (float)Math.Atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
      r.state = r.autoReset != null ? r.autoReset.State : RaceState.RACING;
      if (Phase == RacePhase.Countdown || r.finishTime >= 0) { c.command = Vector3.zero; continue; }

      float dist = Progress(r, pos, out float lateral);
      r.distance = Mathf.Min(dist, _total);
      if (dist >= _total) { r.finishTime = RaceTime; r.place = ++_finished; c.command = Vector3.zero; continue; }
      int cp = Mathf.FloorToInt(dist / map.checkpointSpacing);
      if (cp > r.checkpoint) { r.checkpoint = cp; Spawn(r, cp * map.checkpointSpacing); }

      var target = map.LanePoint(dist + map.lookAhead, r.lane, out _);
      float err = Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, Mathf.Atan2(target.y - pos.y, target.x - pos.x) * Mathf.Rad2Deg) * Mathf.Deg2Rad;
      c.command = new Vector3(r.speed * Mathf.Max(0.3f, Mathf.Cos(err)), 0f, Mathf.Clamp(steerGain * err, -1.2f, 1.2f));
      // Un-stick: upright, asked to go fast, but (nearly) stopped for half a second -> restart the command ramp.
      float vx = (float)d->qvel[M.BaseDof], vy = (float)d->qvel[M.BaseDof + 1];
      bool stuck = c.Active == Go2Mode.Locomotion && c.FilteredCommand.x > 1.0f && vx * vx + vy * vy < 0.3f * 0.3f;
      r.stuckFor = stuck ? r.stuckFor + Go2Controller.SimDt : 0f;
      if (r.stuckFor > 0.5f) { c.RestartCommandRamp(); r.stuckFor = 0f; r.unsticks++; }
      if (Mathf.Abs(lateral) > map.halfWidth + 1f) c.ResetToHome();   // left the track: back to the last checkpoint
    }
    if (Phase == RacePhase.Racing && _finished == _live.Count) Phase = RacePhase.Finished;

    // Robot-vs-robot contacts this step (bump mechanics are pure solver contact; this only counts them).
    if (Phase == RacePhase.Racing) {
      var m = a.model; bool any = false;
      for (int i = 0; i < d->ncon; i++) {
        int b1 = m->body_rootid[m->geom_bodyid[d->contact[i].geom1]], b2 = m->body_rootid[m->geom_bodyid[d->contact[i].geom2]];
        if (b1 != b2 && IsRobotRoot(b1) && IsRobotRoot(b2)) { any = true; break; }
      }
      if (any) { BumpSteps++; if (!_inBump) { Bumps++; _inBump = true; } } else _inBump = false;
    }

    if (_test && !_testDone && (Phase == RacePhase.Finished || RaceTime > 600)) {
      _testDone = true;
      bool ok = Phase == RacePhase.Finished;
      Debug.Log($"[Race4] {(ok ? "FINISHED" : "DNF")} map={map.displayName} laps={Laps} length={_total:F1}m seed={Seed} | " + string.Join(" | ", Standings().Select(r =>
          $"{r.name} {(r.finishTime >= 0 ? r.finishTime.ToString("F2", CultureInfo.InvariantCulture) + "s" : "DNF@" + r.distance.ToString("F0") + "m")} lane={r.lane:F1} v={r.speed:F2} resets={(r.autoReset != null ? r.autoReset.Resets : 0)} unstuck={r.unsticks}"))
          + $" | flips={_flips} bumps={Bumps} bumpTime={BumpSteps * Go2Controller.SimDt:F2}s");
      double mean = _steps > 0 ? _stepMs / _steps : 0;
      Debug.Log($"[Perf4] {_live.Count} racers: physics step mean {mean:F4} ms max {_maxMs:F3} ms | per 50 Hz control step {mean * Go2Controller.Decimation:F3} ms (budget 5.0)");
      Application.Quit(ok ? 0 : 1);
    }
  }

  bool IsRobotRoot(int body) { foreach (var r in _live) if (r.controller.Model.BaseBody == body) return true; return false; }
  List<Racer> Standings() => _live.OrderByDescending(r => r.finishTime >= 0 ? 1e6f - (float)r.finishTime : r.distance).ToList();
  public Racer Leader() { var s = Standings(); return s.FirstOrDefault(r => r.finishTime < 0) ?? s[0]; }

  void Update() {
    if (_live.Count == 0) return;
    var order = Standings();
    if (followCam != null) {
      // Frame the pack: an anchor at the mean position of the racers still running, looking along the track there.
      if (_anchor == null) { _anchor = new GameObject("PackAnchor").transform; followCam.target = _anchor; }
      var run = order.Where(r => r.finishTime < 0 && r.controller.BaseTransform != null).ToList();
      if (run.Count == 0) run = order.Where(r => r.controller.BaseTransform != null).ToList();
      if (run.Count > 0) {
        var sum = Vector3.zero; float dsum = 0; foreach (var r in run) { sum += r.controller.BaseTransform.position; dsum += r.distance; }
        _anchor.position = sum / run.Count;
        map.PointAt(dsum / run.Count, out var tan);
        _camForward = Vector3.Slerp(_camForward, new Vector3(tan.x, 0f, tan.y), 1f - Mathf.Exp(-2.5f * Time.deltaTime)).normalized;
        followCam.forward = _camForward;
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
        string st = r.finishTime >= 0 ? r.finishTime.ToString("F2", CultureInfo.InvariantCulture) + " s"
                  : (map.closed ? $"L{Mathf.Min(r.lap + 1, Laps)}/{Laps}  " : "") + $"{r.distance:F0} m" + (r.state == RaceState.RACING ? "" : r.state == RaceState.STUMBLE ? "  stumble" : "  DOWN");
        sb.Append($"{i + 1}. <color=#{ColorUtility.ToHtmlStringRGB(r.color)}>{r.name}</color>  {st}\n");
      }
      sb.Append($"<size=18>{map.displayName}   bumps {Bumps}</size>");
      standingsText.text = sb.ToString();
    }
  }
}

}
