#!/usr/bin/python3
"""Enclosure fork: add RGSC's regret value and regret ranking heads to a training checkpoint.

    python enclosure_add_regret_heads.py CHECKPOINT [--swing]

With --swing (2026-10-11), it adds the second source's pair instead: the swing value and swing ranking heads (config
swing_heads; the checkpoint must have the regret heads already), and keeps the original as CHECKPOINT.before-swing.

RGSC (owner, 2026-10-04; v2/docs/decisions/rgsc.md) adds two heads to the value head (model_pytorch.RegretHead), present when
the config has regret_heads. For a checkpoint trained before, this sets the flag and adds both heads with KataGo's
initialization (Model.initialize), in the model and the SWA model. The heads read the detached trunk, so the rest
of the network computes exactly what it did.

The optimizer state is rebuilt by parameter name: train.py builds its parameter groups from add_reg_dict, which
now lists the new parameters too, so the saved state no longer matches by position (it would be dropped whole,
momentum included). Every existing parameter keeps its state; the new ones start without any, as in a fresh
optimizer. Saves in place, keeping the original as CHECKPOINT.before-regret. Run only while no trainer uses the
checkpoint.
"""

import copy
import shutil
import sys

import torch

from katago.train.model_pytorch import Model

ANCHOR = "value_head.conv1.weight"  # a key every checkpoint has, to find the state dicts' prefixes


def group_names(model):
    """[(group name, [parameter names])] in train.get_param_groups' order (non-empty reg_dict groups)."""
    names = {id(p): n for n, p in model.named_parameters()}
    reg_dict = {}
    model.add_reg_dict(reg_dict)
    return [(g, [names[id(p)] for p in ps]) for g, ps in reg_dict.items() if len(ps) > 0]


def prefix_of(state_dict):
    keys = [k for k in state_dict if k.endswith(ANCHOR)]
    assert len(keys) == 1, keys
    return keys[0][:-len(ANCHOR)]


# Each pair of heads: its config key, the module names of its heads, and the backup's suffix.
PAIRS = {
    "regret": ("regret_heads", (".regret_value.", ".regret_ranking."), ".before-regret"),
    "swing": ("swing_heads", (".swing_value.", ".swing_ranking."), ".before-swing"),
}


def main():
    path = sys.argv[1]
    pair = "swing" if "--swing" in sys.argv[2:] else "regret"
    key, modules, backup = PAIRS[pair]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if config.get(key):
        print(f"{path}: already has the {pair} heads")
        return
    if pair == "swing" and not config.get("regret_heads"):
        sys.exit(f"{path}: add the regret heads first")
    new_config = copy.deepcopy(config)
    new_config[key] = True
    old_model = Model(config, 19)
    new_model = Model(new_config, 19)
    new_model.initialize()
    fresh = new_model.state_dict()
    added = [k for k in fresh if k not in old_model.state_dict()]
    assert added and all(any(m in k for m in modules) for k in added), added

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        prefix = prefix_of(ckpt[part])
        for k in added:
            ckpt[part][prefix + k] = fresh[k].clone()
        touched.append(f"{part} ({len(added)} tensors)")

    # Optimizer: old index -> name, new name -> index; group settings carried over by group name.
    old_groups, new_groups = group_names(old_model), group_names(new_model)
    opt = ckpt["optimizer"]
    assert [len(g["params"]) for g in opt["param_groups"]] == [len(ns) for _, ns in old_groups], \
        "the saved optimizer does not match this code's parameter groups"
    old_index = {}
    for saved, (_, ns) in zip(opt["param_groups"], old_groups):
        for i, n in zip(saved["params"], ns):
            old_index[n] = i
    settings = {g: saved for saved, (g, _) in zip(opt["param_groups"], old_groups)}
    new_state, new_param_groups, i = {}, [], 0
    for g, ns in new_groups:
        group = {k: v for k, v in settings[g].items() if k != "params"}
        group["params"] = []
        for n in ns:
            if n in old_index and old_index[n] in opt["state"]:
                new_state[i] = opt["state"][old_index[n]]
            group["params"].append(i)
            i += 1
        new_param_groups.append(group)
    assert len(new_state) == len(opt["state"]), "some optimizer state was not carried over"
    ckpt["optimizer"] = {"state": new_state, "param_groups": new_param_groups}
    touched.append(f"optimizer ({len(new_state)} states kept, {i - len(old_index)} new parameters without state)")

    ckpt["config"] = new_config
    shutil.copy2(path, path + backup)
    torch.save(ckpt, path)
    print(f"{path}: {pair} heads added to " + "; ".join(touched))


if __name__ == "__main__":
    main()
