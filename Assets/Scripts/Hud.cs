using Mujoco;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace PoRace {

/// <summary>Portrait HUD. Anchors: TL title | TC fps + sim tick | TR behaviour / race state | BL reset / shove / cube | BR version.
/// TAB toggles the telemetry block (D.3). Buttons drive the creature only through MuJoCo state.</summary>
public class Hud : MonoBehaviour {
  public Go2Controller controller;
  public CubePool cubes;
  public Shove shove;
  public AutoReset autoReset;
  public Text title, telemetry, behaviour, version;
  public bool telemetryVisible = true;
  int _ticks, _lastTicks; float _fpsTimer; int _frames; float _fps, _tickRate;

  void Awake() {
    MjScene.Instance.postUpdateEvent += (s, a) => _ticks++;
    if (autoReset == null) autoReset = controller.GetComponent<AutoReset>();
  }

  void Update() {
    if (Keyboard.current != null && Keyboard.current.tabKey.wasPressedThisFrame) telemetryVisible = !telemetryVisible;
    _frames++; _fpsTimer += Time.unscaledDeltaTime;
    if (_fpsTimer >= 0.5f) {
      _fps = _frames / _fpsTimer; _tickRate = (_ticks - _lastTicks) / _fpsTimer;
      _frames = 0; _fpsTimer = 0f; _lastTicks = _ticks;
    }
    if (title) title.text = "PoRace Go2";
    if (version) version.text = "v" + Application.version;
    var state = autoReset != null ? autoReset.State.ToString() : (controller.Upright ? "RACING" : "FALLEN_RECOVERING");
    if (behaviour) behaviour.text = $"{controller.mode}\n{state}";
    if (telemetry) {
      telemetry.enabled = telemetryVisible;
      var d = MjScene.InstanceExists ? MjScene.Instance : null;
      var vel = SpeedText();
      telemetry.text = $"{_fps:F0} fps | sim {_tickRate:F0} Hz | ctrl {controller.ControlStep} | inf {controller.LastInferenceMs:F2} ms\n{vel} | cmd {controller.command}";
    }
  }

  unsafe string SpeedText() {
    if (!MjScene.InstanceExists || controller.Model == null || MjScene.Instance.Data == null) return "v -";
    var d = MjScene.Instance.Data; var M = controller.Model;
    return $"v ({d->qvel[M.BaseDof]:F2}, {d->qvel[M.BaseDof + 1]:F2}) m/s";
  }

  public void OnReset() => controller.ResetToHome();
  public void OnShove() => shove.PushRandom();
  public void OnCube() => cubes.ThrowAtRobot();
  public void OnDrop() => cubes.DropOnRobot();
  public void OnMode(int m) => controller.mode = (Go2Mode)m;
}

}
