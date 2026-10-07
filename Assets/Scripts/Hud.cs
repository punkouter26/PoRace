using Mujoco;
using UnityEngine;
using UnityEngine.UI;

namespace PoRace {

/// <summary>Portrait HUD. Anchors: TL title | TC fps + sim tick | TR behaviour | BL reset / shove / cube | BR version.
/// Buttons drive the creature only through MuJoCo state (CubePool, Shove, Go2Controller.ResetToHome).</summary>
public class Hud : MonoBehaviour {
  public Go2Controller controller;
  public CubePool cubes;
  public Shove shove;
  public Text title, telemetry, behaviour, version;
  int _ticks; float _fpsTimer; int _frames; float _fps;

  void Awake() {
    MjScene.Instance.postUpdateEvent += (s, a) => _ticks++;
  }

  void Update() {
    _frames++; _fpsTimer += Time.unscaledDeltaTime;
    if (_fpsTimer >= 0.5f) { _fps = _frames / _fpsTimer; _frames = 0; _fpsTimer = 0f; }
    if (telemetry) telemetry.text = $"{_fps:F0} fps  tick {_ticks}  ctrl {controller.ControlStep}  inf {controller.LastInferenceMs:F2} ms";
    if (behaviour) behaviour.text = $"{controller.mode}{(controller.Upright ? "" : " FALLEN")}";
    if (version) version.text = Application.version;
    if (title) title.text = "PoRace Go2";
  }

  public void OnReset() => controller.ResetToHome();
  public void OnShove() => shove.PushRandom();
  public void OnCube() => cubes.ThrowAtRobot();
  public void OnDrop() => cubes.DropOnRobot();
  public void OnMode(int m) => controller.mode = (Go2Mode)m;
}

}
