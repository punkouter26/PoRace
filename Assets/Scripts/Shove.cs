using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>External push on the base body via mjData.xfrc_applied, mirroring the trainer's velocity kicks.
/// Impulse (N·s) is spread over durationSteps physics steps as a constant force.</summary>
public unsafe class Shove : MonoBehaviour {
  public float impulse = 30f;        // N·s, contract R2 bar
  public int durationSteps = 25;     // 0.1 s at 4 ms
  int _baseBody = -1, _remaining;
  Vector3 _force;

  void Awake() {
    MjScene.Instance.postInitEvent += (s, a) => _baseBody = MujocoLib.mj_name2id(a.model, (int)MujocoLib.mjtObj.mjOBJ_BODY, "base");
    MjScene.Instance.preUpdateEvent += OnPreStep;
  }

  /// <summary>Push in a MuJoCo-frame horizontal direction (unit vector).</summary>
  public Vector3 LastDir { get; private set; }
  public void Push(Vector3 mjDir) {
    LastDir = mjDir.normalized;
    _force = mjDir.normalized * (impulse / (durationSteps * Go2Controller.SimDt));
    _remaining = durationSteps;
  }

  public void PushRandom() {
    float ang = Random.Range(0f, Mathf.PI * 2f);
    Push(new Vector3(Mathf.Cos(ang), Mathf.Sin(ang), 0f));
  }

  void OnPreStep(object sender, MjStepArgs a) {
    if (_baseBody < 0) return;
    var f = a.data->xfrc_applied + 6 * _baseBody;
    if (_remaining > 0) { f[0] = _force.x; f[1] = _force.y; f[2] = _force.z; _remaining--; }
    else { f[0] = 0; f[1] = 0; f[2] = 0; }
  }
}

}
