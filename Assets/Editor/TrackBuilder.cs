using System.Collections.Generic;
using Mujoco;
using PoRace;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Editor-time track authoring from a MapDefinition. Physics = side rails as native MuJoCo world geoms (MjGeom boxes
/// under the MuJoCo scene root). Visuals = plain meshes without colliders. Nothing here runs in the player.
/// MuJoCo (x, y, z up) maps to Unity (x, z, y); a MuJoCo yaw of theta is a Unity rotation of -theta about Y.
/// </summary>
public static class TrackBuilder {
  const string MapDir = "Assets/Resources/Maps";

  static Material Mat(string name, Color col) {
    if (!AssetDatabase.IsValidFolder("Assets/Materials")) AssetDatabase.CreateFolder("Assets", "Materials");
    var path = "Assets/Materials/" + name + ".mat";
    var m = AssetDatabase.LoadAssetAtPath<Material>(path);
    if (m == null) { m = new Material(MjcfImporter.GetLitShader()); AssetDatabase.CreateAsset(m, path); }
    m.color = col; if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", col);
    EditorUtility.SetDirty(m); return m;
  }

  static MapDefinition MapAsset(string file) {
    if (!AssetDatabase.IsValidFolder("Assets/Resources")) AssetDatabase.CreateFolder("Assets", "Resources");
    if (!AssetDatabase.IsValidFolder(MapDir)) AssetDatabase.CreateFolder("Assets/Resources", "Maps");
    var path = MapDir + "/" + file + ".asset";
    var m = AssetDatabase.LoadAssetAtPath<MapDefinition>(path);
    if (m == null) { m = ScriptableObject.CreateInstance<MapDefinition>(); AssetDatabase.CreateAsset(m, path); }
    return m;
  }

  /// <summary>Creates / refreshes the launch maps: 01 Planar Sprint and 02 Flat Oval.</summary>
  [MenuItem("PoRace/Create Launch Maps")]
  public static void CreateMaps() {
    var s = MapAsset("Map01_PlanarSprint");
    s.displayName = "Planar Sprint"; s.sceneName = "Track01x4"; s.terrain = "flat"; s.difficulty = 1; s.closed = false; s.defaultLaps = 1;
    s.halfWidth = 1.65f; s.gridSlots = 4; s.laneSpacing = 0.8f; s.checkpointSpacing = 25f; s.lookAhead = 4f;
    s.centerline = new List<Vector2> { new Vector2(0, 0), new Vector2(100, 0) };
    // All four lanes merge to one line between 40 % and 60 % of the track, then spread again.
    s.laneScale = new AnimationCurve(new Keyframe(0f, 1f, 0, 0), new Keyframe(0.15f, 1f, 0, 0), new Keyframe(0.4f, 0f, 0, 0), new Keyframe(0.6f, 0f, 0, 0), new Keyframe(0.85f, 1f, 0, 0), new Keyframe(1f, 1f, 0, 0));
    EditorUtility.SetDirty(s);

    var o = MapAsset("Map02_FlatOval");
    o.displayName = "Flat Oval"; o.sceneName = "Track02"; o.terrain = "flat"; o.difficulty = 2; o.closed = true; o.defaultLaps = 2;
    o.halfWidth = 2.0f; o.gridSlots = 4; o.laneSpacing = 0.8f; o.checkpointSpacing = 12f; o.lookAhead = 2.5f;
    const float L = 30f, R = 6f; const int N = 14;
    var pts = new List<Vector2> { new Vector2(5, 0), new Vector2(L, 0) };
    for (int i = 1; i <= N; i++) { float a = Mathf.PI * i / N; pts.Add(new Vector2(L + R * Mathf.Sin(a), R - R * Mathf.Cos(a))); }       // left turn, east end
    pts.Add(new Vector2(0, 2 * R));
    for (int i = 1; i <= N; i++) { float a = Mathf.PI * i / N; pts.Add(new Vector2(-R * Mathf.Sin(a), R + R * Mathf.Cos(a))); }          // left turn, west end
    o.centerline = pts;   // closes back to (5, 0)
    // Full grid spacing on the start straight, then the lanes squeeze into a racing line (35 %) for the rest of the lap.
    o.laneScale = new AnimationCurve(new Keyframe(0f, 1f, 0, 0), new Keyframe(0.1f, 1f, 0, 0), new Keyframe(0.24f, 0.35f, 0, 0), new Keyframe(0.9f, 0.35f, 0, 0), new Keyframe(1f, 1f, 0, 0));
    EditorUtility.SetDirty(o);
    AssetDatabase.SaveAssets();
    Debug.Log($"[TrackBuilder] maps: {s.displayName} {s.Length:F1} m, {o.displayName} {o.Length:F1} m per lap ({o.centerline.Count} points)");
  }

  static Quaternion Yaw(Vector2 dir) => Quaternion.Euler(0f, -Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg, 0f);
  static Vector3 U(Vector2 mj, float height) => new Vector3(mj.x, height, mj.y);

  /// <summary>Copies `sourceScene` (a scene that already holds the racers) to `targetScene` and rebuilds its track
  /// geometry from the map: MuJoCo rails, visual surface, start/finish line, checkpoint marks.</summary>
  public static void BuildScene(string mapAssetPath, string sourceScene, string targetScene, bool extendOpenEnds = true) {
    var map = AssetDatabase.LoadAssetAtPath<MapDefinition>(mapAssetPath);
    var src = EditorSceneManager.OpenScene(sourceScene, OpenSceneMode.Single);
    if (sourceScene != targetScene) { EditorSceneManager.SaveScene(src, targetScene, true); }
    var scene = EditorSceneManager.OpenScene(targetScene, OpenSceneMode.Single);
    var mjRoot = GameObject.Find("go2_scene");
    foreach (var oldName in new[] { "Track01", "TrackVisuals" }) { var g = GameObject.Find(oldName); if (g != null) Object.DestroyImmediate(g); }
    var oldRails = mjRoot.transform.Find("rails"); if (oldRails != null) Object.DestroyImmediate(oldRails.gameObject);

    Material mA = Mat("TrackA", new Color(0.30f, 0.32f, 0.36f)), mB = Mat("TrackB", new Color(0.36f, 0.38f, 0.42f)),
             mRail = Mat("Rail", new Color(0.85f, 0.25f, 0.15f)), mLine = Mat("Line", Color.white), mMark = Mat("Marker", new Color(0.95f, 0.8f, 0.1f));
    var cube = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
    var rails = new GameObject("rails"); rails.transform.SetParent(mjRoot.transform, false);
    var vis = new GameObject("TrackVisuals");
    System.Action<string, Vector3, Quaternion, Vector3, Material> V = (name, pos, rot, scale, m) => {
      var go = new GameObject(name); go.transform.SetParent(vis.transform, false); go.transform.SetPositionAndRotation(pos, rot); go.transform.localScale = scale;
      go.AddComponent<MeshFilter>().sharedMesh = cube; go.AddComponent<MeshRenderer>().sharedMaterial = m;
    };
    System.Action<string, Vector2, Vector2> Rail = (name, a, b) => {
      var dir = (b - a); float len = dir.magnitude; dir /= len;
      var go = new GameObject(name); go.transform.SetParent(rails.transform, false);
      go.transform.SetPositionAndRotation(U((a + b) * 0.5f, 0.15f), Yaw(dir));
      var g = go.AddComponent<MjGeom>(); g.ShapeType = MjShapeComponent.ShapeTypes.Box; g.Box.Extents = new Vector3(len * 0.5f + 0.06f, 0.15f, 0.05f);
      go.AddComponent<MeshFilter>(); go.AddComponent<MjMeshFilter>(); go.AddComponent<MeshRenderer>().sharedMaterial = mRail;
    };

    int n = map.SegmentCount; float w = map.halfWidth;
    // Offset polylines with vertex normals (average of the adjoining segment normals) so corner rails join up.
    var left = new List<Vector2>(); var right = new List<Vector2>(); var centre = new List<Vector2>();
    int verts = map.closed ? n : n + 1;
    for (int i = 0; i < verts; i++) {
      Vector2 p = map.P(i), dPrev = (map.closed || i > 0) ? (p - map.P(i - 1)).normalized : (map.P(i + 1) - p).normalized, dNext = (map.closed || i < n) ? (map.P(i + 1) - p).normalized : dPrev;
      var nrm = (MapDefinition.Left(dPrev) + MapDefinition.Left(dNext)).normalized;
      float k = 1f / Mathf.Max(0.5f, Vector2.Dot(nrm, MapDefinition.Left(dNext)));   // miter length
      if (!map.closed && extendOpenEnds) p += (i == 0 ? -dNext * 6f : i == n ? dPrev * 6f : Vector2.zero);
      centre.Add(p); left.Add(p + nrm * w * k); right.Add(p - nrm * w * k);
    }
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % verts;
      Rail("rail_L" + i, left[i], left[j]); Rail("rail_R" + i, right[i], right[j]);
      Vector2 a = centre[i], b = centre[j], dir = (b - a); float len = dir.magnitude; dir /= len;
      // Surface: long segments are cut into ~5 m tiles so alternating shades show speed.
      int tiles = Mathf.Max(1, Mathf.RoundToInt(len / 5f));
      for (int t = 0; t < tiles; t++) {
        Vector2 c = a + dir * (len * (t + 0.5f) / tiles);
        V($"tile_{i}_{t}", U(c, -0.012f), Yaw(dir), new Vector3(len / tiles + 0.12f, 0.02f, 2f * w + 0.3f), ((i + t) % 2 == 0) ? mA : mB);
      }
    }
    // Start / finish line and checkpoint marks, across the track at their arc position.
    var p0 = map.PointAt(0f, out var t0);
    V("start_line", U(p0 + t0 * 0.6f, -0.0015f), Yaw(t0), new Vector3(0.15f, 0.003f, 2f * w), mLine);
    for (float s = map.checkpointSpacing; s < map.Length - 0.5f; s += map.checkpointSpacing) { var p = map.PointAt(s, out var tt); V("mark_" + Mathf.RoundToInt(s), U(p, -0.0015f), Yaw(tt), new Vector3(0.1f, 0.003f, 2f * w), mMark); }
    var pf = map.closed ? p0 : map.PointAt(map.Length, out t0); var nf = MapDefinition.Left(t0);
    V("arch_l", U(pf + nf * (w + 0.2f), 0.9f), Yaw(t0), new Vector3(0.1f, 1.8f, 0.1f), mLine);
    V("arch_r", U(pf - nf * (w + 0.2f), 0.9f), Yaw(t0), new Vector3(0.1f, 1.8f, 0.1f), mLine);
    V("arch_top", U(pf, 1.8f), Yaw(t0), new Vector3(0.12f, 0.15f, 2f * w + 0.5f), mMark);

    // Race wiring: the orchestrator reads the map; racer start transforms follow the grid at arc length 0.
    var race = Object.FindFirstObjectByType<MultiRaceOrchestrator>(FindObjectsInactive.Include);
    race.map = map; EditorUtility.SetDirty(race);
    var lanes = map.LaneOffsets(race.racers.Count);
    for (int k = 0; k < race.racers.Count; k++) {
      var r = race.racers[k]; var gp = map.LanePoint(0f, lanes[k], out var gt);
      var baseGo = GameObject.Find(r.controller.prefix + "base");
      if (baseGo != null) { baseGo.transform.SetPositionAndRotation(new Vector3(gp.x, baseGo.transform.position.y, gp.y), Yaw(gt)); }
      if (k > 0 && r.mjRoot == null) r.mjRoot = GameObject.Find("racer" + k);
    }
    int phys = Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length + Object.FindObjectsByType<Collider>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length;
    EditorSceneManager.SaveScene(scene); AssetDatabase.SaveAssets();
    Debug.Log($"[TrackBuilder] {targetScene}: map '{map.displayName}' {map.Length:F1} m, {n} segments, {rails.transform.childCount} MuJoCo rail geoms, {vis.transform.childCount} visuals, PhysX components={phys}");
  }

  [MenuItem("PoRace/Rebuild Launch Tracks")]
  public static void RebuildAll() {
    CreateMaps();
    BuildScene(MapDir + "/Map01_PlanarSprint.asset", "Assets/Scenes/Track01x4.unity", "Assets/Scenes/Track01x4.unity");
    BuildScene(MapDir + "/Map02_FlatOval.asset", "Assets/Scenes/Track01x4.unity", "Assets/Scenes/Track02.unity");
    EditorSceneManager.OpenScene("Assets/Scenes/Testbed.unity", OpenSceneMode.Single);
  }
}
