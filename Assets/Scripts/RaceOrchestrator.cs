using System;
using System.Collections.Generic;
using System.Globalization;
using Mujoco;
using UnityEngine;
using UnityEngine.UI;

namespace PoRace {

public enum RacePhase { Countdown, Racing, Finished }

/// <summary>
/// One-racer race loop on a waypoint track (PRD 4.2, FR-13). Everything is measured in MuJoCo sim time and the
/// MuJoCo frame (x forward, y left, z up). The racer is steered only through the joystick command of the validated
/// locomotion policy: vx from a cruise speed, yaw rate from the heading error to the next waypoint.
/// Headless check: PoRace.exe -batchmode -nographics -raceTest [-timeScale 4]
/// </summary>
public unsafe class RaceOrchestrator : MonoBehaviour {
  public Go2Controller controller;
  public AutoReset autoReset;
  public Text raceText;
  [Tooltip("Waypoints in the MuJoCo frame (x, y), metres. The last one is the finish line.")]
  public List<Vector2> waypoints = new List<Vector2>();
  public float cruiseSpeed = 1.5f;
  public float halfWidth = 1.5f;
  public float countdownSeconds = 3f;
  public float waypointRadius = 1.5f;
  public float steerGain = 1.5f;

  public RacePhase Phase { get; private set; } = RacePhase.Countdown;
  public double RaceTime { get; private set; }
  public float Distance { get; private set; }
  public int Respawns { get; private set; }
  readonly List<double> _splits = new List<double>();
  int _next; double _t0 = -1, _raceStart; float _trackLength, _doneLength;
  bool _test, _testDone; double _lastSim;

  void Awake() {
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-raceTest") _test = true;
      if (args[i] == "-raceFlipAt" && i + 1 < args.Length) _flipAt = double.Parse(args[i + 1], CultureInfo.InvariantCulture);  // race time, s
      if (args[i] == "-timeScale" && i + 1 < args.Length) Time.timeScale = float.Parse(args[i + 1], CultureInfo.InvariantCulture);
    }
    MjScene.Instance.postUpdateEvent += OnPostStep;
  }

  void Start() {
    _trackLength = 0f; var prev = Vector2.zero;
    foreach (var w in waypoints) { _trackLength += Vector2.Distance(prev, w); prev = w; }
    Restart(resetRobot: false);
  }

  public void Restart() => Restart(true);

  void Restart(bool resetRobot) {
    Phase = RacePhase.Countdown; _next = 0; _t0 = -1; _splits.Clear(); RaceTime = 0; Distance = 0; _doneLength = 0; Respawns = 0;
    SetSpawn(Vector2.zero, waypoints.Count > 0 ? waypoints[0] : Vector2.right);
    controller.mode = Go2Mode.Locomotion; controller.command = Vector3.zero;
    if (resetRobot) controller.ResetToHome();
  }

  void SetSpawn(Vector2 at, Vector2 towards) {
    controller.spawnMj = new Vector3(at.x, at.y, Go2Controller.HomeHeight);
    controller.spawnYaw = Mathf.Atan2(towards.y - at.y, towards.x - at.x);
  }

  void OnPostStep(object sender, MjStepArgs a) {
    var M = controller.Model; if (M == null || waypoints.Count == 0) return;
    var d = a.data; double t = d->time; _lastSim = t;
    if (t < _lastSimSeen) { _t0 = -1; }  // sim time went backwards: the robot was reset (mj_resetData zeroes time)
    _lastSimSeen = t;
    if (_t0 < 0) { _t0 = t; if (Phase == RacePhase.Racing) _raceStart = t - RaceTime; }

    var pos = new Vector2((float)d->qpos[M.BaseQpos], (float)d->qpos[M.BaseQpos + 1]);
    double qw = d->qpos[M.BaseQpos + 3], qx = d->qpos[M.BaseQpos + 4], qy = d->qpos[M.BaseQpos + 5], qz = d->qpos[M.BaseQpos + 6];
    float yaw = (float)Math.Atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));

    switch (Phase) {
      case RacePhase.Countdown:
        controller.command = Vector3.zero;
        if (t - _t0 >= countdownSeconds) { Phase = RacePhase.Racing; _raceStart = t; }
        break;
      case RacePhase.Racing:
        RaceTime = t - _raceStart;
        if (_flipAt >= 0 && RaceTime >= _flipAt) { _flipAt = -1; controller.KnockOver(); _flips++; }
        var target = waypoints[_next];
        var from = _next == 0 ? Vector2.zero : waypoints[_next - 1];
        float segLen = Vector2.Distance(from, target);
        Distance = _doneLength + Mathf.Clamp(Vector2.Dot(pos - from, (target - from).normalized), 0f, segLen);
        bool last = _next == waypoints.Count - 1;
        // Intermediate waypoints count when within radius; the finish counts when the line is crossed.
        bool reached = last ? Vector2.Dot(pos - from, (target - from).normalized) >= segLen : Vector2.Distance(pos, target) < waypointRadius;
        if (reached) {
          _splits.Add(RaceTime); _doneLength += segLen;
          if (last) { Phase = RacePhase.Finished; Distance = _trackLength; controller.command = Vector3.zero; break; }
          SetSpawn(target, waypoints[_next + 1]); _next++; target = waypoints[_next];
        }
        float err = Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, Mathf.Atan2(target.y - pos.y, target.x - pos.x) * Mathf.Rad2Deg) * Mathf.Deg2Rad;
        controller.command = new Vector3(cruiseSpeed * Mathf.Max(0.3f, Mathf.Cos(err)), 0f, Mathf.Clamp(steerGain * err, -1f, 1f));
        if (Mathf.Abs(DistanceToCenterline(pos, from, target)) > halfWidth + 1f) { controller.ResetToHome(); }  // jumped the rails
        break;
      case RacePhase.Finished:
        controller.command = Vector3.zero;
        break;
    }
    if (autoReset != null) Respawns = autoReset.Resets;

    if (_test && !_testDone && (Phase == RacePhase.Finished || t > 400)) {
      _testDone = true;
      bool ok = Phase == RacePhase.Finished;
      Debug.Log($"[Race] {(ok ? "FINISHED" : "DNF")} time={RaceTime:F2}s distance={Distance:F1}m splits=[{string.Join(", ", _splits.ConvertAll(s => s.ToString("F2", CultureInfo.InvariantCulture)))}] respawns={Respawns} flips={_flips} avgSpeed={(RaceTime > 0 ? Distance / RaceTime : 0):F2}m/s");
      Application.Quit(ok ? 0 : 1);
    }
  }
  double _lastSimSeen, _flipAt = -1; int _flips;

  static float DistanceToCenterline(Vector2 p, Vector2 a, Vector2 b) {
    var dir = (b - a).normalized; var n = new Vector2(-dir.y, dir.x);
    return Vector2.Dot(p - a, n);
  }

  void Update() {
    if (raceText == null) return;
    string head = Phase == RacePhase.Countdown ? (_t0 < 0 ? "READY" : Mathf.CeilToInt(Mathf.Max(0f, countdownSeconds - (float)(_lastSim - _t0))).ToString())
                : Phase == RacePhase.Racing ? "GO" : "FINISH";
    string splits = _splits.Count == 0 ? "" : "\n" + string.Join("  ", _splits.ConvertAll(s => s.ToString("F1", CultureInfo.InvariantCulture)));
    raceText.text = $"{head}\n{RaceTime:F2} s   {Distance:F0} / {_trackLength:F0} m{splits}";
  }
}

}
