using System.Collections.Generic;
using UnityEngine;

namespace PoRace {

/// <summary>
/// A track as data (PRD FR-02). The centerline is a polyline in the MuJoCo frame (x forward, y left, metres);
/// everything else (rails, visuals, start grid, checkpoints, steering targets) is derived from it.
/// Assets live in Assets/Resources/Maps so MapRegistry can discover them at runtime (FR-03).
/// </summary>
[CreateAssetMenu(menuName = "PoRace/Map Definition", fileName = "Map")]
public class MapDefinition : ScriptableObject {
  public string displayName = "Track";
  [Tooltip("Scene that contains this map's geometry (must be in the build).")]
  public string sceneName = "";
  public string terrain = "flat";
  [Range(1, 5)] public int difficulty = 1;
  public bool closed = false;
  [Range(1, 5)] public int defaultLaps = 1;
  public float halfWidth = 1.65f;
  [Range(1, 4)] public int gridSlots = 4;
  public float laneSpacing = 0.8f;
  public float checkpointSpacing = 25f;
  public float lookAhead = 4f;
  public List<Vector2> centerline = new List<Vector2>();
  [Tooltip("Lane spacing multiplier over one lap (x = 0..1 of lap length). Below 1 squeezes the racers together.")]
  public AnimationCurve laneScale = AnimationCurve.Constant(0f, 1f, 1f);

  float[] _cum; float _length; int _n;

  void EnsureBuilt() {
    int segs = closed ? centerline.Count : centerline.Count - 1;
    if (_cum != null && _n == segs && _cum.Length == segs + 1) return;
    _n = segs; _cum = new float[segs + 1];
    for (int i = 0; i < segs; i++) _cum[i + 1] = _cum[i] + Vector2.Distance(P(i), P(i + 1));
    _length = _cum[segs];
  }

  public int SegmentCount { get { EnsureBuilt(); return _n; } }
  /// <summary>Length of one lap (closed) or of the whole track (open).</summary>
  public float Length { get { EnsureBuilt(); return _length; } }
  public Vector2 P(int i) => centerline[closed ? ((i % centerline.Count) + centerline.Count) % centerline.Count : Mathf.Clamp(i, 0, centerline.Count - 1)];
  public float SegmentStart(int i) { EnsureBuilt(); return _cum[i]; }
  public float SegmentLength(int i) { EnsureBuilt(); return _cum[i + 1] - _cum[i]; }

  /// <summary>Point and unit tangent at arc length s. Closed tracks wrap; open tracks extrapolate past the ends.</summary>
  public Vector2 PointAt(float s, out Vector2 tangent) {
    EnsureBuilt();
    if (closed) s = Mathf.Repeat(s, _length);
    int i = 0;
    if (s >= _length) i = _n - 1;
    else if (s > 0) { while (i < _n - 1 && _cum[i + 1] <= s) i++; }
    Vector2 a = P(i), b = P(i + 1);
    tangent = (b - a).normalized;
    return a + tangent * (s - _cum[i]);
  }

  public static Vector2 Left(Vector2 tangent) => new Vector2(-tangent.y, tangent.x);

  /// <summary>Lane centre at arc length s for a lane offset (positive = left of travel), with the lap's squeeze applied.</summary>
  public Vector2 LanePoint(float s, float laneOffset, out Vector2 tangent) {
    var p = PointAt(s, out tangent);
    float u = closed ? Mathf.Repeat(s, Length) / Length : Mathf.Clamp01(s / Length);
    return p + Left(tangent) * laneOffset * laneScale.Evaluate(u);
  }

  /// <summary>Lane offsets for n racers, centred on the track, leftmost first.</summary>
  public float[] LaneOffsets(int n) {
    var r = new float[n];
    for (int i = 0; i < n; i++) r[i] = ((n - 1) * 0.5f - i) * laneSpacing;
    return r;
  }
}

/// <summary>Runtime discovery of every map asset under Resources/Maps (PRD FR-03).</summary>
public static class MapRegistry {
  public static MapDefinition[] All() {
    var maps = Resources.LoadAll<MapDefinition>("Maps");
    System.Array.Sort(maps, (a, b) => string.CompareOrdinal(a.name, b.name));
    return maps;
  }
}

/// <summary>Choices made on the pre-race screen, carried across the scene load. Defaults give a full grid.</summary>
public static class RaceConfig {
  public static int racers = 4;
  public static int laps = 0;        // 0 = the map's default
  public static int seed = -1;       // -1 = different every race
  public static bool fromMenu = false;
}

}
