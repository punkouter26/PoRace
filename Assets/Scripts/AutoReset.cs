using Mujoco;
using UnityEngine;

namespace PoRace {

public enum RaceState { RACING, STUMBLE, FALLEN_RECOVERING }

/// <summary>D.2: race state from MuJoCo state only, and automatic reset to the home keyframe when the dog stays down,
/// stalls, or leaves the arena. Reset = Go2Controller.ResetToHome (qpos/qvel/ctrl buffers + parked cubes), no scene reload.</summary>
public unsafe class AutoReset : MonoBehaviour {
  public Go2Controller controller;
  public float fallenSeconds = 5f;
  public float stallSeconds = 10f;
  public float stallSpeed = 0.05f;   // m/s; only counts while a non-zero command is active
  public float arenaRadius = 15f;
  public RaceState State { get; private set; } = RaceState.RACING;
  public int Resets { get; private set; }

  float _fallenFor, _stalledFor;

  void Awake() { MjScene.Instance.postUpdateEvent += OnPostStep; }
  void OnDestroy() { if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= OnPostStep; }

  void OnPostStep(object sender, MjStepArgs a) {
    var M = controller.Model; if (M == null) return;
    var d = a.data;
    float dt = Go2Controller.SimDt;
    double upZ = d->sensordata[M.UpAdr + 2];
    float x = (float)d->qpos[M.BaseQpos], y = (float)d->qpos[M.BaseQpos + 1];
    float speed = Mathf.Sqrt((float)(d->qvel[M.BaseDof] * d->qvel[M.BaseDof] + d->qvel[M.BaseDof + 1] * d->qvel[M.BaseDof + 1]));

    State = upZ < 0.0 ? RaceState.FALLEN_RECOVERING : upZ < 0.7 ? RaceState.STUMBLE : RaceState.RACING;
    _fallenFor = State == RaceState.FALLEN_RECOVERING ? _fallenFor + dt : 0f;
    bool wantsToMove = controller.mode == Go2Mode.Locomotion && controller.command.sqrMagnitude > 0.01f;
    _stalledFor = wantsToMove && speed < stallSpeed ? _stalledFor + dt : 0f;

    if (_fallenFor > fallenSeconds || _stalledFor > stallSeconds || x * x + y * y > arenaRadius * arenaRadius) {
      Debug.Log($"[AutoReset] {State} fallen={_fallenFor:F1}s stalled={_stalledFor:F1}s pos=({x:F1},{y:F1}) -> home");
      controller.ResetToHome();
      _fallenFor = _stalledFor = 0f; Resets++;
    }
  }
}

}
