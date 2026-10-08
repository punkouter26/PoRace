"""Export a policy to ONNX (opset 17, batch 1, float32) for Unity Inference Engine.

  --zero                 : B.9 dummy policy, obs[1,48] -> zeros[1,12]
  --params <params.pkl>  : Brax PPO params saved by train.py (C.4). Bakes the obs normalizer and the MLP, outputs tanh(mean).
Run (zero): uv run --no-project --with onnx==1.23.2 --with onnxruntime==1.30.0 --with numpy python export_onnx.py --zero
Run (real, in the trainer container): python export_onnx.py --params runs/r1/params.pkl --out /work/../Assets/Policies/go2_loco.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
from go2 import constants as C  # noqa: E402

OUT_DIR = C.ROOT_PATH.parent / "Assets" / "Policies"
NORM_EPS = 0.0  # brax running_statistics: std already = sqrt(var + std_eps), normalize divides by std directly


def unpack_params(params):
  """Brax PPO params -> (mean, std, [(W, b), ...]) with the policy head cut to the action mean."""
  norm, policy = params[0], params[1]
  mean, std = norm.mean, norm.std
  if isinstance(mean, dict):  # playground envs return {"state", "privileged_state"}; the policy sees "state"
    mean, std = mean["state"], std["state"]
  mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
  p = policy["params"]
  keys = sorted(p.keys(), key=lambda k: int(k.split("_")[1]))
  layers = [(np.asarray(p[k]["kernel"], np.float32), np.asarray(p[k]["bias"], np.float32)) for k in keys]
  w, b = layers[-1]
  assert w.shape[1] % 2 == 0, w.shape  # NormalTanhDistribution: [mean, logstd]; action size = half (12 for Go2, 29 for G1)
  nu = w.shape[1] // 2
  layers[-1] = (w[:, :nu], b[:nu])
  return mean, std, layers


def policy_from_params(params):
  """numpy reference of the exported graph: obs -> tanh(mlp((obs-mean)/(std+eps))). Deterministic brax policy."""
  mean, std, layers = unpack_params(params)

  def run(obs):
    x = (np.asarray(obs, np.float32) - mean) / (std + NORM_EPS)
    for i, (w, b) in enumerate(layers):
      x = x @ w + b
      if i < len(layers) - 1:
        x = x * (1.0 / (1.0 + np.exp(-x)))  # swish
    return np.tanh(x)
  return run


def build_graph(layers, obs_dim, act_dim, name, norm_mean=None, norm_std=None, tanh=True):
  nodes, inits = [], []
  x = "obs"
  if norm_mean is not None:
    inits += [numpy_helper.from_array(norm_mean.astype(np.float32), "norm_mean"),
              numpy_helper.from_array((norm_std + NORM_EPS).astype(np.float32), "norm_std_eps")]
    nodes += [helper.make_node("Sub", [x, "norm_mean"], ["x_c"]), helper.make_node("Div", ["x_c", "norm_std_eps"], ["x_n"])]
    x = "x_n"
  for i, (w, b) in enumerate(layers):
    inits += [numpy_helper.from_array(w.astype(np.float32), f"W{i}"), numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
    nodes += [helper.make_node("MatMul", [x, f"W{i}"], [f"h{i}"]), helper.make_node("Add", [f"h{i}", f"b{i}"], [f"z{i}"])]
    x = f"z{i}"
    if i < len(layers) - 1:
      nodes += [helper.make_node("Sigmoid", [x], [f"s{i}"]), helper.make_node("Mul", [x, f"s{i}"], [f"a{i}"])]
      x = f"a{i}"
  nodes.append(helper.make_node("Tanh" if tanh else "Identity", [x], ["action"]))
  graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
                            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, act_dim])], inits)
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="porace")
  model.ir_version = 9
  onnx.checker.check_model(model)
  return model


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--zero", action="store_true")
  ap.add_argument("--params")
  ap.add_argument("--out")
  a = ap.parse_args()
  if a.zero:
    model = build_graph([(np.zeros((C.OBS_JOYSTICK, C.NU)), np.zeros(C.NU))], C.OBS_JOYSTICK, C.NU, "zero_policy", tanh=False)
    name, ref = "zero_policy.onnx", None
  else:
    from brax.io import model as brax_model
    params = brax_model.load_params(a.params)
    mean, std, layers = unpack_params(params)
    model = build_graph(layers, len(mean), layers[-1][0].shape[1], "policy", mean, std)
    name, ref = Path(a.params).parent.name + ".onnx", policy_from_params(params)
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  out = Path(a.out) if a.out else OUT_DIR / name
  onnx.save(model, str(out))
  import onnxruntime as ort
  sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  obs_dim = sess.get_inputs()[0].shape[1]
  worst = 0.0
  for _ in range(1000):
    x = np.random.randn(1, obs_dim).astype(np.float32)
    y = sess.run(None, {"obs": x})[0]
    assert y.shape[0] == 1
    if ref is not None:
      worst = max(worst, float(np.abs(y[0] - ref(x[0])).max()))
    else:
      assert np.all(y == 0)
  print(f"OK {out} obs={obs_dim} act={y.shape[1]} opset=17 onnx-vs-numpy worst {worst:.2e}")
