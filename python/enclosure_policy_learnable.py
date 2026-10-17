#!/usr/bin/python3
"""Enclosure fork: can the policy head learn from our rows at all? (2026-10-02, v2a's p0loss stuck near ln 17,329)
    python enclosure_policy_learnable.py CHECKPOINT SHUFFLED_TRAIN_DIR [STEPS]
A copy of the checkpoint's model trains on the policy loss alone (Adam, lr 1e-3, batch 64, no warm-up) over all
but the last two shuffled files, then is scored on those two, before and after. If the targets' move numbering
agrees with the policy outputs', the held-out cross-entropy falls well below its start (and toward the targets'
own entropy); if they disagree, a convolutional policy cannot generalise and it stays near ln(policy size).
Also reported: the share of the model's policy on moves the target uses (the legal, searched moves).
"""
import glob
import math
import sys

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.games.enclosure import POLICY_CHANNELS
from katago.train.model_pytorch import policy_len

SIZE = 19


def batches(files, model, batch_size, randomize):
    return read_npz_training_data(files, batch_size=batch_size, world_size=1, rank=0, pos_len=SIZE,
                                  device=DEVICE, randomize_symmetries=randomize, include_meta=False,
                                  model_config=model.config)


def policy_terms(model, batch):
    out = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
    logp = torch.log_softmax(out[0][:, 0, :].float(), dim=1)
    t = batch["policyTargetsNCMove"][:, 0, :].float()
    t = t / t.sum(dim=1, keepdim=True)
    ce = -(t * logp).sum(dim=1)
    entropy = -(t * torch.log(t.clamp_min(1e-30))).sum(dim=1)
    mass = (logp.exp() * (t > 0)).sum(dim=1)
    return ce, entropy, mass


def score(model, files, limit=40):
    model.eval()
    ce, h, mass, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(batches(files, model, 64, False)):
            c, e, m = policy_terms(model, b)
            ce += c.sum().item(); h += e.sum().item(); mass += m.sum().item(); n += len(c)
            if i + 1 >= limit:
                break
    return ce / n, h / n, mass / n


def main():
    checkpoint, data_dir = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    train_files, held_out = files[:-2], files[-2:]
    torch.manual_seed(0)
    model, _, _ = load_model(checkpoint, use_swa=False, device=DEVICE)
    print(f"policy size {policy_len(SIZE, POLICY_CHANNELS)}, ln = {math.log(policy_len(SIZE, POLICY_CHANNELS)):.3f}; "
          f"{len(train_files)} training files, {len(held_out)} held out")
    ce, h, mass = score(model, held_out)
    print(f"before: held-out policy CE {ce:.3f}, target entropy {h:.3f}, model mass on target moves {mass:.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    done = 0
    while done < steps:
        for b in batches(train_files, model, 64, True):
            model.train()
            c, _, _ = policy_terms(model, b)
            opt.zero_grad()
            c.mean().backward()
            opt.step()
            done += 1
            if done % 50 == 0:
                print(f"  step {done}: training batch CE {c.mean().item():.3f}", flush=True)
            if done >= steps:
                break
    ce, h, mass = score(model, held_out)
    print(f"after {steps} steps: held-out policy CE {ce:.3f}, target entropy {h:.3f}, model mass on target moves {mass:.4f}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    main()
