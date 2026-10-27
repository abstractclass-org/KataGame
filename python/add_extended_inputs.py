#!/usr/bin/python3
"""KataGame fork: give a training checkpoint the inputs version 10, 11 or 12, and scale its outputs in points.

    python add_extended_inputs.py CHECKPOINT OUT -inputs-version VERSION [-old-score-scale S]

The inputs version 10 (2026-10-26; C++ NNInputs V10, model version 19) is version 9's with the game's extended inputs:
its extended planes after its planes, and its extended global inputs after version 9's (katago/train/games:
NUM_EXTENDED_SPATIAL_INPUTS and NUM_EXTENDED_GLOBAL_INPUTS). Blueprints' run bpj3 starts from a network of bpj2 this
way. The inputs version 11 (2026-10-27; C++ NNInputs V11, model version 20) is version 10's with the game's turn inputs
after its global inputs (games.turn_global_inputs). Its planes keep their count, but its history planes show the last
turns in place of the last actions: training absorbs that change of meaning. Blueprints' run bpj4 starts from bpj3's
last checkpoint this way (the engine repository's games/blueprints/docs/decisions/port.md). The trunk's first
convolution (conv_spatial) reads the new planes through more input channels, and its first global layer
(linear_global) reads the new global inputs through more weight columns.

This sets the config's inputs_version to VERSION and adds those channels and columns as zeros, after the old ones, in
the model, the SWA model and the optimizer's state, so that the network computes exactly what it did until training
moves them, whatever the new inputs hold. The optimizer keeps every state by position (no parameter is added): the new
channels and columns start at 0 in each state tensor of the widened weights (SGD's momentum, Adam's moments). A
checkpoint without an optimizer or an SWA model (a network alone) gets the rest. The checkpoint's inputs must be the
version before VERSION: 9 for 10, 10 for 11, 11 for 12.

-inputs-version 12 gives a checkpoint of the inputs version 11 the version 12 (2026-10-27; C++ NNInputs V12, model
version 21): version 11's inputs with the game's planes of closed areas after its planes (katago/train/games:
NUM_AREA_SPATIAL_INPUTS), as zero input channels of conv_spatial in the same way. Version 12 adds no global input.
A run of version 11 of Blueprints takes the areas closed by each player's walls this way, with no new run: the
trainer reads the rows of version 11 in its window with those planes at 0
(data_processing_pytorch.fit_binary_input_nchw).

-old-score-scale S names the game's SCORE_SCALE that trained the checkpoint, and this code's SCORE_SCALE s replaces it.
It multiplies by F = S / s the outputs that become points through a plain factor, KataGo's score multiplier of 20
times the game's SCORE_SCALE (Model.postprocess_output in the trainer, the model file's multipliers in the C++
NNEvaluator): the score mean, the lead and the 3 TD scores. These outputs then keep their values in points under the
new multipliers. They are rows of each value head's last linear layers, so F multiplies those rows of the weights and
the biases, in the model and the SWA model. It also multiplies SGD's momentum of those rows, a step of the weights, so
that the next steps move the outputs in points as before (the losses of these outputs divide by SCORE_SCALE squared).
Blueprints' 12 -> 20 (bpj3 -> bpj4) gives F = 0.6. Run it with the code of the new score scale: when s is S, it
refuses. The score's standard deviation and short-term error (a softplus times a multiplier) and the score belief
(bins SCORE_SCALE points wide) have no such factor: training moves them. An optimizer state other than SGD's momentum,
or a policy that predicts each move's final margin (policy output 7, in points too), is refused.

Writes OUT, which must not exist: the checkpoint stays as it was. Run it on a checkpoint that no trainer writes at the
time (a copy).
"""

import argparse
import copy
import math
import os
import sys

import torch

from enclosure_add_regret_heads import group_names
from katago.train import modelconfigs
from katago.train.model_pytorch import Model

# Each inputs version that this gives, and the version that it needs.
PREVIOUS = {10: 9, 11: 10, 12: 11}
# The outputs that become points through a plain factor (Model.postprocess_output): rows of each value head's last
# linear layers, the score mean and the lead, then the 3 TD scores.
SCORE_ROWS = {"linear_miscvaluehead": [0, 2], "linear_moremiscvaluehead": [5, 6, 7]}


def widen(t, old_n, new_n):
    """t with new_n - old_n zeros after its old_n entries on dimension 1."""
    assert t.dim() >= 2 and t.shape[1] == old_n, (tuple(t.shape), old_n)
    shape = list(t.shape)
    shape[1] = new_n - old_n
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=1)


def scale_rows(t, rows, factor):
    """t with its entries rows on dimension 0 multiplied by factor."""
    t = t.clone()
    t[rows] *= factor
    return t


def key_of(state_dict, name):
    """The key of the parameter name in state_dict, after a wrapper's prefix ("_orig_mod." of a compiled model,
    "module." of the SWA model)."""
    keys = [k for k in state_dict if k == name or k.endswith("." + name)]
    assert len(keys) == 1, (name, keys)
    return keys[0]


def network(config):
    """A model of config, on its game's board."""
    return Model(config, modelconfigs.get_game(config).MAX_BOARD_LEN)


def optimizer_states(ckpt, names):
    """{name: (index, state)} of the parameters names in ckpt's optimizer, by their positions in
    train.get_param_groups' order. A parameter that has no state yet gets an empty one, and a checkpoint without an
    optimizer (a network alone) none."""
    if "optimizer" not in ckpt:
        return {}
    indices = {}
    for saved, (_, group) in zip(ckpt["optimizer"]["param_groups"], group_names(network(ckpt["config"]))):
        assert len(saved["params"]) == len(group), "the saved optimizer does not match this code's parameter groups"
        for p, n in zip(saved["params"], group):
            if n in names:
                indices[n] = p
    assert set(indices) == set(names), f"the optimizer's groups lack {set(names) - set(indices)}"
    return {n: (i, ckpt["optimizer"]["state"].get(i, {})) for n, i in indices.items()}


def convert(ckpt, inputs_version):
    """ckpt (a loaded checkpoint) with the inputs version inputs_version (10, 11 or 12), in place. Returns what it
    widened, one line each."""
    if inputs_version not in PREVIOUS:
        raise ValueError(f"the inputs version {inputs_version}: this gives {' or '.join(map(str, PREVIOUS))}")
    config = ckpt["config"]
    version = modelconfigs.get_inputs_version(config)
    if version != PREVIOUS[inputs_version]:
        raise ValueError(f"the checkpoint's inputs are version {version}, not {PREVIOUS[inputs_version]}")
    new_config = copy.deepcopy(config)
    new_config["inputs_version"] = inputs_version
    # The weights that read the inputs, on dimension 1: conv_spatial (out, in, y, x) and linear_global (out, in), with
    # their counts of inputs before and after, if these change.
    sizes = {}
    for name, count in (("conv_spatial.weight", modelconfigs.get_num_bin_input_features),
                        ("linear_global.weight", modelconfigs.get_num_global_input_features)):
        old_n, new_n = count(config), count(new_config)
        if old_n != new_n:
            sizes[name] = (old_n, new_n)
    if not sizes:
        raise ValueError(f"the game {modelconfigs.get_game(config).NAME} has no inputs new in version {inputs_version}")

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        for name, (old_n, new_n) in sizes.items():
            key = key_of(ckpt[part], name)
            ckpt[part][key] = widen(ckpt[part][key], old_n, new_n)
            touched.append(f"{part} {key}: {old_n} -> {new_n}")

    ckpt["config"] = new_config
    for name, (index, state) in optimizer_states(ckpt, sizes).items():
        old_n, new_n = sizes[name]
        widened = []
        for k, v in state.items():
            if torch.is_tensor(v) and v.dim() >= 2:
                state[k] = widen(v, old_n, new_n)
                widened.append(k)
        touched.append(f"optimizer {name} (state {index}: {', '.join(widened) or 'none'})")
    return touched


def scale_scores(ckpt, factor):
    """ckpt (a loaded checkpoint) with its outputs in points (SCORE_ROWS) multiplied by factor, in place. Returns what
    it scaled, one line each."""
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError(f"the score factor {factor} is not a positive number")
    model = network(ckpt["config"])
    if model.policy_head.num_policy_outputs > 7:
        raise ValueError("the policy predicts each move's final margin (policy output 7), which this does not scale")
    # The weight and the bias of each value head's layers (the intermediate head's too), with their rows.
    rows = {n: SCORE_ROWS[n.split(".")[-2]] for n, _ in model.named_parameters() if n.split(".")[-2] in SCORE_ROWS}
    states = optimizer_states(ckpt, rows)
    for name, (_, state) in states.items():
        for k, v in state.items():
            if torch.is_tensor(v) and k != "momentum_buffer":
                raise ValueError(f"the optimizer's state {k} of {name}: this scales SGD's momentum_buffer only")

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        for name, r in rows.items():
            key = key_of(ckpt[part], name)
            ckpt[part][key] = scale_rows(ckpt[part][key], r, factor)
            touched.append(f"{part} {key}: rows {r} x {factor}")
    for name, (index, state) in states.items():
        if torch.is_tensor(state.get("momentum_buffer")):
            state["momentum_buffer"] = scale_rows(state["momentum_buffer"], rows[name], factor)
            touched.append(f"optimizer {name} (state {index}: momentum_buffer rows {rows[name]} x {factor})")
    return touched



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="the checkpoint to read")
    parser.add_argument("out", help="the new checkpoint, which must not exist")
    parser.add_argument("-inputs-version", type=int, choices=sorted(PREVIOUS), required=True,
                        help="the inputs version to give: 10 to a checkpoint of 9, 11 to a checkpoint of 10, "
                             "12 to a checkpoint of 11")
    parser.add_argument("-old-score-scale", type=float,
                        help="the game's SCORE_SCALE that trained the checkpoint: give its outputs in points this "
                             "code's SCORE_SCALE")
    args = parser.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists: give a new path")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    touched = convert(ckpt, args.inputs_version)
    if args.old_score_scale is not None:
        scale = modelconfigs.get_game(ckpt["config"]).SCORE_SCALE
        if args.old_score_scale == scale:
            sys.exit(f"this code's SCORE_SCALE is {scale} too: run this with the code of the new score scale")
        touched += scale_scores(ckpt, args.old_score_scale / scale)
    torch.save(ckpt, args.out)
    print(f"{args.out}: {args.checkpoint} with the inputs version {args.inputs_version}\n  " + "\n  ".join(touched))


if __name__ == "__main__":
    main()
