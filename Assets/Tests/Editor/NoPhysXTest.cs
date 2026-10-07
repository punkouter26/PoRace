using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoRace.Tests {

/// <summary>Physics engine isolation mandate: no PhysX component may exist in any PoRace scene.</summary>
public class NoPhysXTest {
  static readonly string[] Scenes = { "Assets/Scenes/Testbed.unity", "Assets/Scenes/Race.unity" };

  [Test]
  public void ScenesContainNoPhysXComponents() {
    foreach (var path in Scenes.Where(p => System.IO.File.Exists(p))) {
      var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);
      try {
        foreach (var root in scene.GetRootGameObjects()) {
          Assert.IsEmpty(root.GetComponentsInChildren<Rigidbody>(true), $"{path}: Rigidbody found");
          Assert.IsEmpty(root.GetComponentsInChildren<Collider>(true), $"{path}: Collider found");
          Assert.IsEmpty(root.GetComponentsInChildren<Joint>(true), $"{path}: Joint found");
          Assert.IsEmpty(root.GetComponentsInChildren<CharacterController>(true), $"{path}: CharacterController found");
          Assert.IsEmpty(root.GetComponentsInChildren<ArticulationBody>(true), $"{path}: ArticulationBody found");
        }
      } finally {
        EditorSceneManager.CloseScene(scene, true);
      }
    }
  }

  [Test]
  public void FixedTimestepMatchesTrainer() {
    Assert.AreEqual(0.004f, Time.fixedDeltaTime, 1e-7f, "Time.fixedDeltaTime must be 0.004 (MuJoCo plugin ignores MJCF timestep)");
    Assert.AreEqual(new Vector3(0f, -9.81f, 0f), Physics.gravity, "Physics.gravity is the value MjScene feeds to MuJoCo");
  }
}

}
