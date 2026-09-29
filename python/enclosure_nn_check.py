#!/usr/bin/python3
"""Enclosure fork: the C++ network evaluation against a PyTorch forward pass (plan step 5).

    python enclosure_nn_check.py make CHECKPOINT [-config b6c96] [-seed 0]
    python export_model_pytorch.py -checkpoint CHECKPOINT -export-dir DIR -model-name NAME -filename-prefix NAME
    katago runenclosurenntests DIR/NAME.bin ../cpp/tests/data/enclosure_inputs.txt CPP_OUT
    python enclosure_nn_check.py compare CHECKPOINT ../cpp/tests/data/enclosure_inputs.txt CPP_OUT

`make` writes a randomly initialised model whose every parameter and normalisation statistic is then
perturbed, so no layer is an identity (fixup initialises some to zero). `compare` builds each position's
input from the fixture (the Enclosure repository's reference inputs, which NNInputs::fillRowV8 equals),
runs the model and compares with the C++ output: the policy over the legal moves, the win/loss/no-result
probabilities, the score mean and lead, and the ownership. Exit code 0 when all agree within float32
tolerance.
"""

import argparse
import sys

import numpy as np
import torch

from katago.train import modelconfigs
from katago.train.load_model import load_model
from katago.train.model_pytorch import Model

SIZE = 19
PY_PLANES = 219       # the Enclosure repository's input_planes: 192 edge, 5 scalar, 22 feature planes
PY_EDGE_PLANES = 192
PY_FEATURES_START = 197
V8_SPATIAL = 215


def read_inputs(path):
    """label -> (spatial (215, 19, 19), global (5,)) from the inputs fixture."""
    positions = {}
    lines = open(path).read().splitlines()
    for i in range(0, len(lines), 3):
        label = lines[i].split()[1]
        glob = np.array([float(v) for v in lines[i + 1].split()[1:]], np.float32)
        idx = np.array([int(v) for v in lines[i + 2].split()[2:]], np.int64)
        spatial = np.zeros((V8_SPATIAL, SIZE, SIZE), np.float32)
        spatial[0] = 1.0
        pos, plane = idx // PY_PLANES, idx % PY_PLANES
        chans = np.where(plane < PY_EDGE_PLANES, 1 + plane, plane - 4)
        spatial[chans, pos // SIZE, pos % SIZE] = 1.0
        positions[label] = (spatial, glob)
    return positions


def read_cpp(path):
    rows = {}
    for line in open(path):
        w = line.split()
        label, pla = w[0], int(w[1])
        win, loss, noresult, scoremean, lead = map(float, w[2:7])
        assert w[7] == "own"
        own = np.array([float(v) for v in w[8:8 + SIZE * SIZE]])
        assert w[8 + SIZE * SIZE] == "pol"
        n = int(w[9 + SIZE * SIZE])
        pol = w[10 + SIZE * SIZE:]
        assert len(pol) == 2 * n
        actions = np.array([int(v) for v in pol[0::2]])
        probs = np.array([float(v) for v in pol[1::2]])
        rows[label] = dict(value=np.array([win, loss, noresult]), scoremean=scoremean, lead=lead, own=own,
                           actions=actions, probs=probs)
    return rows


def make(args):
    torch.manual_seed(args.seed)
    config = modelconfigs.config_of_name[args.config]
    model = Model(config, SIZE)
    model.initialize()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.add_(0.05 * torch.randn_like(p))
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.add_(0.05 * torch.randn_like(b))
            elif name.endswith("running_var") or name.endswith("running_std"):
                b.mul_(torch.exp(0.1 * torch.randn_like(b)))
    torch.save({"config": config, "model": model.state_dict()}, args.checkpoint)
    print(f"wrote {args.checkpoint} ({args.config}, seed {args.seed})")


def compare(args):
    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu")
    model.eval()
    inputs = read_inputs(args.inputs)
    cpp = read_cpp(args.cpp_out)
    labels = list(cpp)
    worst = dict(policy=0.0, value=0.0, score=0.0, own=0.0)
    failures = 0
    with torch.no_grad():
        for start in range(0, len(labels), 32):
            batch = labels[start:start + 32]
            spatial = torch.from_numpy(np.stack([inputs[l][0] for l in batch]))
            glob = torch.from_numpy(np.stack([inputs[l][1] for l in batch]))
            outputs = model.postprocess_output(model(spatial, glob))[0]
            policy_logits, value_logits = outputs[0], outputs[1]
            ownership_pretanh, scoremean, lead = outputs[4], outputs[8], outputs[10]
            for i, label in enumerate(batch):
                c = cpp[label]
                logits = policy_logits[i, 0].numpy()[c["actions"]].astype(np.float64)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                # As NNEvaluator under the fork's rules (area scoring, positional ko): no result is impossible,
                # so its probability is 0 and win and loss are renormalised.
                value = np.zeros(3)
                value[:2] = torch.softmax(value_logits[i, :2].double(), dim=0).numpy()
                own = np.tanh(ownership_pretanh[i, 0].double().numpy().reshape(-1))
                diffs = dict(
                    policy=np.abs(probs - c["probs"]).max() / max(probs.max(), 1e-9),
                    value=np.abs(value - c["value"]).max(),
                    score=max(abs(float(scoremean[i]) - c["scoremean"]), abs(float(lead[i]) - c["lead"])) / model.scoremean_multiplier,
                    own=np.abs(own - c["own"]).max(),
                )
                for k, d in diffs.items():
                    worst[k] = max(worst[k], d)
                if max(diffs.values()) > args.tolerance:
                    failures += 1
                    if failures <= 10:
                        print(f"FAIL {label}: " + ", ".join(f"{k} {d:.2e}" for k, d in diffs.items()))
    print(f"{len(labels)} positions; largest differences: " + ", ".join(f"{k} {d:.2e}" for k, d in worst.items())
          + f"; {failures} over {args.tolerance}")
    return 0 if failures == 0 and len(labels) > 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("checkpoint")
    m.add_argument("-config", default="b6c96")
    m.add_argument("-seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("checkpoint")
    c.add_argument("inputs")
    c.add_argument("cpp_out")
    c.add_argument("-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if args.cmd == "make":
        make(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
