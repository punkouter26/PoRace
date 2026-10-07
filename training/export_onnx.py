"""Export a policy to ONNX (opset 17, batch 1, float32) for Unity Inference Engine.

  --zero                 : B.9 dummy policy, obs[1,48] -> zeros[1,12]
  --checkpoint <dir>     : Brax PPO checkpoint (C.4): bakes the obs normalizer and the MLP, outputs the mean.
Run: uv run --no-project --with onnx==1.23.2 --with onnxruntime==1.30.0 --with numpy python export_onnx.py --zero
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


def build_mlp(layers, obs_dim, act_dim, name, norm_mean=None, norm_std=None):
    """layers: list of (W[in,out], b[out]); activation swish between layers (Brax default), tanh on the mean."""
    nodes, inits = [], []
    x = "obs"
    if norm_mean is not None:
        inits += [numpy_helper.from_array(norm_mean.astype(np.float32), "norm_mean"),
                  numpy_helper.from_array(norm_std.astype(np.float32), "norm_std")]
        nodes += [helper.make_node("Sub", [x, "norm_mean"], ["x_c"]), helper.make_node("Div", ["x_c", "norm_std"], ["x_n"])]
        x = "x_n"
    for i, (w, b) in enumerate(layers):
        inits += [numpy_helper.from_array(w.astype(np.float32), f"W{i}"), numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes += [helper.make_node("MatMul", [x, f"W{i}"], [f"h{i}"]), helper.make_node("Add", [f"h{i}", f"b{i}"], [f"z{i}"])]
        x = f"z{i}"
        if i < len(layers) - 1:
            nodes += [helper.make_node("Sigmoid", [x], [f"s{i}"]), helper.make_node("Mul", [x, f"s{i}"], [f"a{i}"])]
            x = f"a{i}"
    nodes.append(helper.make_node("Identity", [x], ["action"]))
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
                              [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, act_dim])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="porace")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def export_zero():
    w = np.zeros((C.OBS_JOYSTICK, C.NU)); b = np.zeros(C.NU)
    model = build_mlp([(w, b)], C.OBS_JOYSTICK, C.NU, "zero_policy")
    return model, "zero_policy.onnx"


def export_checkpoint(path):
    # Brax PPO params: (normalizer_params, policy_params). Layer naming hidden_0.. from brax.training.networks.
    from brax.training.acme import running_statistics  # noqa: F401  (import check)
    from orbax import checkpoint as ocp
    params = ocp.PyTreeCheckpointer().restore(path)
    norm, policy = params[0], params[1]
    mean, std = np.asarray(norm.mean), np.asarray(norm.std)
    p = policy["params"]
    keys = sorted(p.keys(), key=lambda k: int(k.split("_")[1]))
    layers = [(np.asarray(p[k]["kernel"]), np.asarray(p[k]["bias"])) for k in keys]
    # Brax policy head outputs [mean, logstd]; keep the mean half, then tanh (NormalTanhDistribution mode).
    w, b = layers[-1]
    layers[-1] = (w[:, : C.NU], b[: C.NU])
    model = build_mlp(layers, len(mean), C.NU, "go2_policy", mean, std)
    # tanh on the mean: append node
    g = model.graph
    g.node[-1].output[0] = "pre_tanh"
    g.node.append(helper.make_node("Tanh", ["pre_tanh"], ["action"]))
    onnx.checker.check_model(model)
    return model, Path(path).name + ".onnx"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    a = ap.parse_args()
    model, name = export_zero() if a.zero else export_checkpoint(a.checkpoint)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else OUT_DIR / name
    onnx.save(model, str(out))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    obs_dim = sess.get_inputs()[0].shape[1]
    y = sess.run(None, {"obs": np.random.randn(1, obs_dim).astype(np.float32)})[0]
    assert y.shape == (1, C.NU), y.shape
    if a.zero:
        assert np.all(y == 0)
    print(f"OK {out} obs={obs_dim} act={y.shape[1]} opset=17")
