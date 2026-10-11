#!/usr/bin/python3
"""Enclosure fork: the C++ network evaluation against a PyTorch forward pass (plan step 5).

    python enclosure_nn_check.py make CHECKPOINT [-config b6c96] [-seed 0]
    python export_model_pytorch.py -checkpoint CHECKPOINT -export-dir DIR -model-name NAME -filename-prefix NAME
    testnnoutputs DIR/NAME.bin ../cpp/tests/data/enclosure_inputs.txt DUMP FP16 NHWC search,ownership
    python enclosure_nn_check.py compare CHECKPOINT DUMP [-tolerance 1e-3]

`make` writes a randomly initialised model whose every parameter and normalisation statistic is then
perturbed, so no layer is an identity (fixup initialises some to zero). `testnnoutputs` (the fork's test,
katago/cpp/tests/enclosure/build.sh) evaluates the exported model and writes each position's input rows
with the outputs. `compare` runs the checkpoint on those same rows and compares the outputs: the policy
over the legal moves, the win/loss/no-result probabilities, the score mean and lead, and the ownership.
So it checks the network alone. testnninputs checks the rows against the inputs fixture. Exit code 0 when
all agree within the tolerance.
"""

import argparse
import sys

import numpy as np
import torch

from enclosure_nn_outputs_check import WHITE, read_dump
from katago.train import modelconfigs
from katago.train.load_model import load_model
from katago.train.model_pytorch import Model

SIZE = 19


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
    model, swa_model, _ = load_model(args.checkpoint, use_swa=args.use_swa, device="cpu")
    if args.use_swa:  # the weights that the run's exports carry (export_model_for_selfplay.sh -use-swa)
        model = swa_model
    model.eval()
    c_in = getattr(model, "num_bin_input_features", 215)
    positions = read_dump(args.dump)
    worst = dict(policy=0.0, value=0.0, score=0.0, own=0.0)
    failures = 0
    with torch.no_grad():
        for start in range(0, len(positions), 32):
            batch = positions[start:start + 32]
            spatial = torch.zeros(len(batch), c_in, SIZE * SIZE)
            for n, p in enumerate(batch):
                for i, v in p["spatial"]:
                    if p["nhwc"]:
                        spatial[n, i % c_in, i // c_in] = v
                    else:
                        spatial[n, i // (SIZE * SIZE), i % (SIZE * SIZE)] = v
            glob = torch.tensor(np.stack([p["global"] for p in batch]), dtype=torch.float32)
            outputs = model.postprocess_output(model(spatial.view(-1, c_in, SIZE, SIZE), glob))[0]
            policy_logits, value_logits = outputs[0], outputs[1]
            ownership_pretanh, scoremean, lead = outputs[4], outputs[8], outputs[10]
            for i, p in enumerate(batch):
                # The dump's policy is at the legal actions and pass, -1 at a pass that is not legal.
                actions = np.array(sorted(a for a, v in p["policy"].items() if v >= 0))
                cpp_probs = np.array([p["policy"][a] for a in actions])
                logits = policy_logits[i, 0].numpy()[actions].astype(np.float64)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                # As NNEvaluator under the fork's rules (area scoring, positional ko): no result is impossible,
                # so its probability is 0 and win and loss are renormalised.
                value = np.zeros(3)
                value[:2] = torch.softmax(value_logits[i, :2].double(), dim=0).numpy()
                own = np.tanh(ownership_pretanh[i, 0].double().numpy().reshape(-1))
                # The dump's value, score and ownership are in white's view, PyTorch's in the mover's.
                white = p["pla"] == WHITE
                sign = 1.0 if white else -1.0
                cpp_value = p["value"] if white else p["value"][[1, 0, 2]]
                diffs = dict(
                    policy=np.abs(probs - cpp_probs).max() / max(probs.max(), 1e-9),
                    value=np.abs(value - cpp_value).max(),
                    score=max(abs(sign * float(scoremean[i]) - p["score"][0]),
                              abs(sign * float(lead[i]) - p["score"][2])) / model.scoremean_multiplier,
                    own=np.abs(own - sign * p["owner"]).max(),
                )
                for k, d in diffs.items():
                    worst[k] = max(worst[k], d)
                if max(diffs.values()) > args.tolerance:
                    failures += 1
                    if failures <= 10:
                        print(f"FAIL {p['label']}: " + ", ".join(f"{k} {d:.2e}" for k, d in diffs.items()))
    print(f"{len(positions)} positions; largest differences: "
          + ", ".join(f"{k} {d:.2e}" for k, d in worst.items()) + f"; {failures} over {args.tolerance}")
    return 0 if failures == 0 and len(positions) > 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("checkpoint")
    m.add_argument("-config", default="b6c96")
    m.add_argument("-seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("checkpoint")
    c.add_argument("dump", help="testnnoutputs' output, with OUTPUTS search,ownership")
    c.add_argument("-use-swa", dest="use_swa", action="store_true",
                   help="the checkpoint's SWA weights, for a model exported with -use-swa (a run's models)")
    c.add_argument("-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if args.cmd == "make":
        make(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
