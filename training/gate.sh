#!/usr/bin/env bash
# Early Verification Gate (C.8/C.9): export ONNX, eval pass bars, record reference, replay in Unity, closed-loop compare.
#   training/gate.sh runs/r1 [vx vy yaw]      (run from the repo root in Git Bash; Unity player already built with the ONNX)
set -euo pipefail
export MSYS_NO_PATHCONV=1   # keep /work /docs /assets container paths intact under Git Bash
RUN=${1:?run dir, e.g. runs/r1}; VX=${2:-0.5}; VY=${3:-0}; WZ=${4:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WROOT="$(cygpath -w "$ROOT")"
T="docker run --rm -e JAX_PLATFORMS=cpu -v ${WROOT}\\training:/work -v ${WROOT}\\docs:/docs -v ${WROOT}\\Assets:/assets porace-trainer"
NAME=$(basename "$RUN")

echo "== export ONNX"
$T python export_onnx.py --params "$RUN/params.pkl" --out "/assets/Policies/${NAME}.onnx"
echo "== eval R0 / R1"
$T python eval.py --rung r0 --params "$RUN/params.pkl"
$T python eval.py --rung r1 --params "$RUN/params.pkl"
echo "== reference trajectory (5 s, command $VX $VY $WZ)"
$T python record_reference.py --env joystick --params "$RUN/params.pkl" --command "$VX" "$VY" "$WZ" --seconds 5 --out /docs/parity/reference_trajectory.json

echo "== assign ${NAME}.onnx to the Testbed controller and rebuild the headless player"
unity command eval --caller plugin --skill unity-cli --no-banner -- "UnityEditor.AssetDatabase.Refresh(); UnityEditor.SceneManagement.EditorSceneManager.OpenScene(\"Assets/Scenes/Testbed.unity\"); var c = UnityEngine.Object.FindFirstObjectByType<PoRace.Go2Controller>(); c.locomotionModel = UnityEditor.AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(\"Assets/Policies/${NAME}.onnx\"); c.mode = PoRace.Go2Mode.Locomotion; UnityEditor.EditorUtility.SetDirty(c); UnityEditor.SceneManagement.EditorSceneManager.SaveOpenScenes(); UnityEngine.Debug.Log(\"[gate] policy=\" + (c.locomotionModel != null ? c.locomotionModel.name : \"NULL\"));" 60000 >/dev/null
unity command build --caller plugin --skill unity-cli --no-banner -- StandaloneWindows64 Build/Testbed/PoRace.exe "" "" "" true false >/dev/null
for i in $(seq 1 60); do sleep 10; st=$(unity command build_status --caller plugin --skill unity-cli --format json --no-banner | python -c "import sys,json; print(json.load(sys.stdin)['data']['result'].get('status'))"); [ "$st" = "completed" ] && break; done
echo "build: $st"

echo "== Unity replay (policy output parity)"
"$ROOT/Build/Testbed/PoRace.exe" -batchmode -nographics -logFile "$ROOT/Temp/replay.log" -replay "${WROOT}\\docs\\parity\\reference_trajectory.json"
grep -E "\[Replay\]" "$ROOT/Temp/replay.log"
echo "== Unity closed loop (5 s)"
"$ROOT/Build/Testbed/PoRace.exe" -batchmode -nographics -logFile "$ROOT/Temp/closed.log" -mode Locomotion -command "$VX" "$VY" "$WZ" -seconds 5 -parityFile unity_closed_loop.json -parityDir "${WROOT}\\docs\\parity"
grep -E "ModelCheck|ParityRecorder|Exception" "$ROOT/Temp/closed.log"
$T python compare_trajectory.py /docs/parity/reference_trajectory.json /docs/parity/unity_closed_loop.json
