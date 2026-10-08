using UnityEngine;

namespace PoRace {

/// <summary>D.1 portrait follow camera: trails the base body (a Unity transform the MuJoCo plugin keeps in sync).</summary>
public class FollowCam : MonoBehaviour {
  public Transform target;            // the "base" MjBody transform
  public Vector3 offset = new Vector3(1.2f, 0.9f, -1.6f);
  public float smooth = 4f;

  void LateUpdate() {
    if (target == null) return;
    var want = target.position + offset;
    transform.position = Vector3.Lerp(transform.position, want, 1f - Mathf.Exp(-smooth * Time.deltaTime));
    transform.LookAt(target.position + Vector3.up * 0.1f);
  }
}

}
