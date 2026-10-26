#!/usr/bin/python3
"""Enclosure fork: check enclosure_add_regret_heads.py on a copy of a checkpoint, of any game.

    python enclosure_regret_heads_check.py ORIGINAL CONVERTED [ROWS.npz ...]

Loads both as train.py does (strict model load, the optimizer from its saved state), then checks that:
every existing output is unchanged (the heads read the detached trunk), in the model and, when the checkpoints have
one, the SWA model (2026-10-27); the optimizer's momentum carried over by parameter name; the regret losses run on
rows with made-up regret targets and send gradients only into the two new heads. With ROWS (.npz files of training
rows, up to 128 rows from them) the forward uses real inputs, else random ones. The models are built on the board of
the config's game (2026-10-27: Blueprints' 67 x 67 view). The pair added is the one CONVERTED has and ORIGINAL lacks:
the regret heads, or (--swing, 2026-10-11) the swing heads.
"""

import sys

import numpy as np
import torch

from enclosure_add_regret_heads import prefix_of
from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


def model_of(config, sd):
    """A Model of the config's game, with the state dict sd (after a wrapper's prefix), in eval mode."""
    model = Model(config, modelconfigs.get_game(config).MAX_BOARD_LEN)
    prefix = prefix_of(sd)
    model.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=True)
    model.eval()
    return model


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    swa = model_of(ckpt["config"], ckpt["swa_model"]) if "swa_model" in ckpt else None
    return ckpt, model_of(ckpt["config"], ckpt["model"]), swa


def inputs(rows_paths, n, config):
    pos_len = modelconfigs.get_game(config).MAX_BOARD_LEN
    if rows_paths:
        spatial, glob, got = [], [], 0
        for path in rows_paths:
            d = np.load(path)
            packed = d["binaryInputNCHWPacked"][:n - got]
            c = packed.shape[1]
            spatial.append(np.unpackbits(packed, axis=2)[:, :, :pos_len * pos_len].reshape(-1, c, pos_len, pos_len))
            glob.append(d["globalInputNC"][:len(packed)])
            got += len(packed)
            if got >= n:
                break
        return (torch.tensor(np.concatenate(spatial), dtype=torch.float32),
                torch.tensor(np.concatenate(glob), dtype=torch.float32))
    torch.manual_seed(0)
    spatial = (torch.rand(n, modelconfigs.get_num_bin_input_features(config), pos_len, pos_len) < 0.2).float()
    spatial[:, 0] = 1.0  # on-board mask
    return spatial, torch.randn(n, modelconfigs.get_num_global_input_features(config))


def main():
    orig_path, conv_path = sys.argv[1], sys.argv[2]
    rows = sys.argv[3:]
    orig_ckpt, orig, orig_swa = load(orig_path)
    conv_ckpt, conv, conv_swa = load(conv_path)
    assert (orig_swa is None) == (conv_swa is None), "one checkpoint has an SWA model, the other not"
    swing = bool(conv_ckpt["config"].get("swing_heads")) and not orig_ckpt["config"].get("swing_heads")
    if not swing:
        assert conv_ckpt["config"].get("regret_heads") and not orig_ckpt["config"].get("regret_heads")
    # The pair's raw output columns (out_regret), its postprocessed outputs, its target columns and its heads.
    cols, keep = (slice(2, 4), slice(0, 2)) if swing else (slice(0, 2), slice(2, 4))
    out_value, out_ranking = (17, 18) if swing else (15, 16)
    target, flag = (72, 73) if swing else (70, 71)
    heads = ["swing_ranking", "swing_value"] if swing else ["regret_ranking", "regret_value"]

    spatial, glob = inputs(rows, 128 if rows else 64, orig_ckpt["config"])
    print(f"inputs: {len(spatial)} rows ({'self-play rows' if rows else 'random'})")
    for name, before, after in (("model", orig, conv), ("SWA model", orig_swa, conv_swa)):
        if before is None:
            continue
        with torch.no_grad():
            a = before(spatial, glob)[0]
            b = after(spatial, glob)[0]
        for i, (x, y) in enumerate(zip(a, b)):
            same = torch.equal(x, y)
            print(f"{name} output {i}: {'identical' if same else 'DIFFERS'} {tuple(x.shape)}")
            assert same or i == len(a) - 1
        assert torch.equal(a[-1][:, keep], b[-1][:, keep]), "the other pair's outputs changed"
        assert torch.count_nonzero(a[-1][:, cols]) == 0 and torch.count_nonzero(b[-1][:, cols]) > 0
        print(f"{name}: {heads[1].split('_')[0]} outputs (raw):", b[-1][:4, cols].tolist())

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

    # Losses with made-up regret targets: gradients only into the new heads. One group of the ranking loss, 32 rows,
    # so that the backward pass on a large board stays small.
    conv.train()
    metrics = Metrics(1, conv)
    spatial, glob = spatial[:32], glob[:32]
    batch_size = spatial.shape[0]
    global_targets = torch.zeros(batch_size, 80)
    global_targets[:, target] = torch.rand(batch_size) * 2.0
    global_targets[:, flag] = 1.0
    out = conv.postprocess_output(conv(spatial, glob))[0]
    pred_regret, pred_ranking = out[out_value], out[out_ranking]
    w = global_targets[:, flag]
    gw = torch.ones(batch_size)
    lv = metrics.loss_regret_value_samplewise(pred_regret, global_targets[:, target], w, gw).sum()
    lr = metrics.loss_regret_ranking(pred_ranking, global_targets[:, target], w, gw)
    (lv + lr).backward()
    with_grad = sorted({n.split(".")[1] for n, p in conv.named_parameters()
                        if p.grad is not None and torch.count_nonzero(p.grad) > 0})
    print(f"losses: regret value {lv.item():.3f} (sum of {batch_size}), ranking {lr.item():.3f}; gradients in {with_grad}")
    assert with_grad == heads, with_grad
    print("all checks passed")


if __name__ == "__main__":
    main()
