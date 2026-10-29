#!/usr/bin/python3
"""KataGame fork: give a training checkpoint of the inputs version 13 AZdb's members: the inputs version 14 and the
intrinsic value head.

    python add_diversity_members.py CHECKPOINT OUT [-seed S] [-latent-scale F]

AZdb (Zahavy et al., "Diversifying AI: Towards Creative Chess with AlphaZero", arXiv 2308.09175; the engine
repository's games/blueprints/docs/research/diversity-players.md, "The cost on this machine", "How it starts") trains
a team of members in one network. The inputs version 14 (2026-10-29; C++ NNInputs V14, model version 23) is version
13's with the latent after its global inputs: modelconfigs.DIVERSITY_MEMBERS inputs, 1 at the member's. The
intrinsic value head (model_pytorch.ValueHead, config intrinsic_head) gives two outputs on the shared trunk: A, the
intrinsic return of the player to move, and B, that of the other player, both as the member of the latent.

This sets the config's inputs_version to 14 and intrinsic_head, in the model, the SWA model and the optimizer's state:

- The trunk's first global layer (linear_global) reads the latent through DIVERSITY_MEMBERS new weight columns after
  the old ones. Member 0's column is 0, so that member 0 computes what the checkpoint's network computed, whatever
  member 0's input holds. Each other member's column is random and small: KataGo's initialization of that layer
  (Model.initialize) times F, LATENT_SCALE by default. With zero weights every member would start as the checkpoint's
  network, every member's estimate of psi would be equal, and every reward would be 0 ("How it starts"). With these
  weights the members differ a little at the first export, and training sets how much.
- The intrinsic value head is new, with KataGo's initialization, as enclosure_add_regret_heads.py gives RGSC's heads.

S seeds both draws (default 0), so that a conversion repeats. The model and the SWA model get the same new weights.
The optimizer's state is rebuilt by parameter name, as enclosure_add_regret_heads.py rebuilds it: every old parameter
keeps its state, the latent's columns start at 0 in each state tensor of linear_global (SGD's momentum), and the
head's parameters start without state, as in a fresh optimizer. A checkpoint without an optimizer or an SWA model (a
network alone) gets the rest.

Writes OUT, which must not exist: the checkpoint stays as it was. Run it on a checkpoint that no trainer writes at the
time (a copy).
"""

import argparse
import copy
import os
import sys

import torch

from add_extended_inputs import key_of, network
from enclosure_add_regret_heads import group_names
from katago.train import modelconfigs

# The inputs version that this gives, and the version that it needs.
VERSION, PREVIOUS = 14, 13
# The scale of the other members' latent columns against KataGo's initialization of linear_global.
LATENT_SCALE = 0.1
GLOBAL = "linear_global.weight"
HEAD = "value_head.intrinsic_value."


def convert(ckpt, seed=0, latent_scale=LATENT_SCALE):
    """ckpt (a loaded checkpoint of the inputs version 13) with the inputs version 14 and the intrinsic value head, in
    place. Returns what it changed, one line each."""
    config = ckpt["config"]
    version = modelconfigs.get_inputs_version(config)
    if version != PREVIOUS:
        raise ValueError(f"the checkpoint's inputs are version {version}, not {PREVIOUS}")
    if config.get("intrinsic_head"):
        raise ValueError("the checkpoint has the intrinsic value head already")
    if not latent_scale >= 0:
        raise ValueError(f"the latent's scale {latent_scale} is not 0 or more")
    new_config = copy.deepcopy(config)
    new_config["inputs_version"] = VERSION
    new_config["intrinsic_head"] = True
    old_n = modelconfigs.get_num_global_input_features(config)
    new_n = modelconfigs.get_num_global_input_features(new_config)
    assert new_n == old_n + modelconfigs.DIVERSITY_MEMBERS, (old_n, new_n)

    # The new weights, drawn once for both the model and the SWA model: a fresh network's initialization.
    old_model = network(config)
    new_model = network(new_config)
    torch.manual_seed(seed)
    new_model.initialize()
    fresh = new_model.state_dict()
    latent = fresh[GLOBAL][:, old_n:].clone() * latent_scale
    latent[:, 0] = 0.0
    added = [k for k in fresh if k not in old_model.state_dict()]
    assert added and all(k.startswith(HEAD) for k in added), added

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        key = key_of(ckpt[part], GLOBAL)
        old = ckpt[part][key]
        assert old.shape[1] == old_n, (tuple(old.shape), old_n)
        ckpt[part][key] = torch.cat([old, latent.to(old.dtype)], dim=1)
        prefix = key[:-len(GLOBAL)]
        for k in added:
            ckpt[part][prefix + k] = fresh[k].clone()
        touched.append(f"{part} {key}: {old_n} -> {new_n}; {len(added)} tensors of the intrinsic value head")

    ckpt["config"] = new_config
    if "optimizer" in ckpt:
        touched.append(rebuild_optimizer(ckpt, old_model, new_model, old_n, new_n))
    return touched


def rebuild_optimizer(ckpt, old_model, new_model, old_n, new_n):
    """ckpt's optimizer state by parameter name for new_model's parameter groups (train.get_param_groups' order), with
    linear_global's state tensors widened by zero columns. Returns what it did, one line."""
    old_groups, new_groups = group_names(old_model), group_names(new_model)
    opt = ckpt["optimizer"]
    assert [len(g["params"]) for g in opt["param_groups"]] == [len(ns) for _, ns in old_groups], \
        "the saved optimizer does not match this code's parameter groups"
    old_index = {}
    for saved, (_, ns) in zip(opt["param_groups"], old_groups):
        for i, n in zip(saved["params"], ns):
            old_index[n] = i
    settings = {g: saved for saved, (g, _) in zip(opt["param_groups"], old_groups)}
    new_state, new_param_groups, i, widened = {}, [], 0, []
    for g, ns in new_groups:
        group = {k: v for k, v in settings[g].items() if k != "params"}
        group["params"] = []
        for n in ns:
            if n in old_index and old_index[n] in opt["state"]:
                state = opt["state"][old_index[n]]
                if n == GLOBAL:
                    for k, v in state.items():
                        if torch.is_tensor(v) and v.dim() >= 2:
                            assert v.shape[1] == old_n, (k, tuple(v.shape), old_n)
                            state[k] = torch.cat([v, v.new_zeros((v.shape[0], new_n - old_n))], dim=1)
                            widened.append(k)
                new_state[i] = state
            group["params"].append(i)
            i += 1
        new_param_groups.append(group)
    assert len(new_state) == len(opt["state"]), "some optimizer state was not carried over"
    ckpt["optimizer"] = {"state": new_state, "param_groups": new_param_groups}
    return (f"optimizer ({len(new_state)} states kept, {GLOBAL} widened: {', '.join(widened) or 'none'}; "
            f"{i - len(old_index)} new parameters without state)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="the checkpoint to read, of the inputs version 13")
    parser.add_argument("out", help="the new checkpoint, which must not exist")
    parser.add_argument("-seed", type=int, default=0, help="the seed of the new weights (default 0)")
    parser.add_argument("-latent-scale", type=float, default=LATENT_SCALE,
                        help="the other members' latent columns against KataGo's initialization of linear_global "
                             f"(default {LATENT_SCALE})")
    args = parser.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists: give a new path")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    touched = convert(ckpt, args.seed, args.latent_scale)
    torch.save(ckpt, args.out)
    print(f"{args.out}: {args.checkpoint} with AZdb's members (the inputs version {VERSION}, the intrinsic value "
          "head)\n  " + "\n  ".join(touched))


if __name__ == "__main__":
    main()
