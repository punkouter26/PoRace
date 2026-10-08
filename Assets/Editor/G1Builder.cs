using Mujoco;
using PoRace;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>Editor-time authoring of the G1 parity testbed: imports Assets/MuJoCo/g1/g1_unity.xml through the MuJoCo
/// importer (no hand-built hierarchy), adds the controller, the parity recorder, a camera and a light.</summary>
public static class G1Builder {
  [MenuItem("PoRace/Build G1 Testbed")]
  public static void Build() {
    var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    var root = new MjImporterWithAssets().ImportFile(System.IO.Path.GetFullPath("Assets/MuJoCo/g1/g1_unity.xml"));
    root.name = "g1_scene";
    var gs = Object.FindFirstObjectByType<MjGlobalSettings>();
    gs.UseRawGameObjectNames = true; EditorUtility.SetDirty(gs);

    var rig = new GameObject("G1Rig");
    var ctrl = rig.AddComponent<G1Controller>(); ctrl.mode = G1Controller.Mode.HoldPose;
    var rec = rig.AddComponent<ParityRecorder>(); rec.controller = ctrl; rec.seconds = 1f; rec.fileName = "unity_g1_zero_brain.json"; rec.quitWhenDone = true;

    var cam = new GameObject("Main Camera"); cam.tag = "MainCamera"; cam.AddComponent<Camera>();
    cam.transform.position = new Vector3(2.4f, 1.3f, -2.4f); cam.transform.LookAt(new Vector3(0f, 0.7f, 0f));
    var light = new GameObject("Directional Light"); var l = light.AddComponent<Light>(); l.type = LightType.Directional;
    light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

    EditorSceneManager.SaveScene(scene, "Assets/Scenes/G1Testbed.unity");
    int bodies = Object.FindObjectsByType<MjBody>(FindObjectsSortMode.None).Length, acts = Object.FindObjectsByType<MjActuator>(FindObjectsSortMode.None).Length;
    int phys = Object.FindObjectsByType<Rigidbody>(FindObjectsSortMode.None).Length + Object.FindObjectsByType<Collider>(FindObjectsSortMode.None).Length;
    EditorSceneManager.OpenScene("Assets/Scenes/Testbed.unity", OpenSceneMode.Single);
    Debug.Log($"[G1Builder] G1Testbed.unity saved: MjBody={bodies} (want 30) MjActuator={acts} (want 29) PhysX components={phys}");
  }
}
