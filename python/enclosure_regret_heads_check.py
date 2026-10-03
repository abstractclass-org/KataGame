#!/usr/bin/python3
"""Enclosure fork: check enclosure_add_regret_heads.py on a copy of a checkpoint.

    python enclosure_regret_heads_check.py ORIGINAL CONVERTED [ROWS.npz]

Loads both as train.py does (strict model load, the optimizer from its saved state), then checks that:
every existing output is unchanged (the heads read the detached trunk); the optimizer's momentum carried over by
parameter name; the regret losses run on rows with made-up regret targets and send gradients only into the two
new heads. With ROWS (a shuffled .npz of training rows) the forward uses real inputs, else random ones.
"""

import sys

import numpy as np
import torch

from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(ckpt["config"], 19)
    sd = ckpt["model"]
    prefix = [k for k in sd if k.endswith("value_head.conv1.weight")][0][:-len("value_head.conv1.weight")]
    model.load_state_dict({k[len(prefix):]: v for k, v in sd.items()}, strict=True)
    model.eval()
    return ckpt, model


def inputs(rows_path, n, config):
    if rows_path:
        d = np.load(rows_path)
        packed = d["binaryInputNCHWPacked"][:n]
        c = packed.shape[1]
        bits = np.unpackbits(packed, axis=2)[:, :, :19 * 19].reshape(-1, c, 19, 19)
        return torch.tensor(bits, dtype=torch.float32), torch.tensor(d["globalInputNC"][:n], dtype=torch.float32), d
    torch.manual_seed(0)
    spatial = (torch.rand(n, modelconfigs.get_num_bin_input_features(config), 19, 19) < 0.2).float()
    spatial[:, 0] = 1.0  # on-board mask
    return spatial, torch.randn(n, modelconfigs.get_num_global_input_features(config)), None


def main():
    orig_path, conv_path = sys.argv[1], sys.argv[2]
    rows = sys.argv[3] if len(sys.argv) > 3 else None
    orig_ckpt, orig = load(orig_path)
    conv_ckpt, conv = load(conv_path)
    assert conv_ckpt["config"].get("regret_heads") and not orig_ckpt["config"].get("regret_heads")

    spatial, glob, d = inputs(rows, 64, orig_ckpt["config"])
    with torch.no_grad():
        a = orig(spatial, glob)[0]
        b = conv(spatial, glob)[0]
    for i, (x, y) in enumerate(zip(a, b)):
        same = torch.equal(x, y)
        print(f"output {i}: {'identical' if same else 'DIFFERS'} {tuple(x.shape)}")
        assert same or i == len(a) - 1
    assert torch.count_nonzero(a[-1]) == 0 and torch.count_nonzero(b[-1]) > 0
    print("regret outputs (raw):", b[-1][:4].tolist())

    # Optimizer: momentum by parameter name.
    names = {id(p): n for n, p in conv.named_parameters()}

    def by_name(model, ckpt):
        reg = {}
        model.add_reg_dict(reg)
        order = [names_of[id(p)] for g, ps in reg.items() if ps for p in ps]
        return {order[i]: s for i, s in ckpt["optimizer"]["state"].items()}

    names_of = {id(p): n for n, p in orig.named_parameters()}
    old = by_name(orig, orig_ckpt)
    names_of = names
    new = by_name(conv, conv_ckpt)
    assert old.keys() <= new.keys() and len(new) == len(old)
    for n in old:
        assert torch.equal(old[n]["momentum_buffer"], new[n]["momentum_buffer"]), n
    print(f"optimizer: {len(new)} momentum buffers carried over by name, all equal")
    sgd = torch.optim.SGD([{"params": ps} for g, ps in (lambda r: (conv.add_reg_dict(r), r)[1])({}).items() if ps],
                          lr=1.0, momentum=0.9)
    sgd.load_state_dict(conv_ckpt["optimizer"])
    print("optimizer: loads into this code's parameter groups")

    # Losses with made-up regret targets: gradients only into the new heads.
    conv.train()
    metrics = Metrics(1, conv)
    batch_size = spatial.shape[0]
    global_targets = torch.zeros(batch_size, 80)
    global_targets[:, 70] = torch.rand(batch_size) * 2.0
    global_targets[:, 71] = 1.0
    out = conv.postprocess_output(conv(spatial, glob))[0]
    pred_regret, pred_ranking = out[-2], out[-1]
    w = global_targets[:, 71]
    gw = torch.ones(batch_size)
    lv = metrics.loss_regret_value_samplewise(pred_regret, global_targets[:, 70], w, gw).sum()
    lr = metrics.loss_regret_ranking(pred_ranking, global_targets[:, 70], w, gw)
    (lv + lr).backward()
    with_grad = sorted({n.split(".")[1] for n, p in conv.named_parameters()
                        if p.grad is not None and torch.count_nonzero(p.grad) > 0})
    print(f"losses: regret value {lv.item():.3f} (sum of {batch_size}), ranking {lr.item():.3f}; gradients in {with_grad}")
    assert with_grad == ["regret_ranking", "regret_value"], with_grad
    print("all checks passed")


if __name__ == "__main__":
    main()
