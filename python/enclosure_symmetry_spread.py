#!/usr/bin/python3
"""Enclosure fork: how far a network's outputs move between the 8 board symmetries, checkpoint by checkpoint.

    python enclosure_symmetry_spread.py positions OUT.npz --tdata DIR [--files 64] [--per-file 2] [--seed 0]
    python enclosure_symmetry_spread.py measure RUN_DIR [--every-rows 250000] [--watch SECONDS] [--threads 2]
    python enclosure_symmetry_spread.py measure RUN_DIR --only NAME [NAME ...]

The network is not built symmetric. Training shows each row in a random symmetry (katago/train/enclosure_symmetry.py),
and the network learns the symmetry from that alone, so its outputs move with the board's orientation by less as it
trains (v2/docs/decisions/build.md, "The network's spread over the 8 symmetries"). This measures that on fixed
positions: each position goes through the network as it is and in each of the 7 other symmetries, with the trainer's
own mapping and the SWA weights that the run exports. For each symmetry, the identity's policy is mapped into the
symmetry's frame and compared with the network's own policy there.

- policyTV: the total variation between the two (the share of the probability that moves), mean over the positions
  and symmetries. The policy is the softmax over every move and pass, as the trainer computes it: the engine also
  leaves out the illegal moves, which a trained network gives almost nothing.
- topAgree: the share of (position, symmetry) pairs whose top move is the identity's top move. lightvector's study of
  KataGo's released networks (July 2026) finds 90 to 95%. topAgreeClear: the same, only on the positions where the
  identity's top move leads the second by more than CLEAR_GAP (counted in clearPositions): near ties decide topAgree
  (v2/docs/decisions/build.md, "The training's symmetry handling").
- winSD: the standard deviation of the mover's win chance over the 8 symmetries, mean over the undecided positions
  (mean win chance 0.1 to 0.9, counted in winPositions). leadSD: the same for the lead in points, over all positions.

positions: the newest --files .npz files under DIR (self-play's tdata folders, by time), --per-file rows from each,
spread evenly through the file. It keeps their inputs, the same for every checkpoint after.

measure: each checkpoint in RUN_DIR/models and models_extra (NAME-sS-dD with model.ckpt) exported after the run's last
input change (the RUN_DIR/*_since markers: older networks read other inputs), not yet in RUN_DIR/symmetry.jsonl, at
least --every-rows data rows after the last one measured of its size. It appends one line for each. First it measures
again each line that lacks a number this version computes, and rewrites the file. With --watch it looks again every
SECONDS, forever. With --only it measures just the checkpoints named and prints them, writing nothing.
The positions are RUN_DIR/symmetry_positions.npz.
"""

import argparse
import datetime
import json
import pathlib
import re
import time

import numpy as np
import torch

from katago.train import modelconfigs
from katago.train.data_processing_pytorch import fit_global_input_nc
from katago.train.enclosure_symmetry import NUM_SYMMETRIES, apply_symmetry_inputs, apply_symmetry_policy
from katago.train.load_model import load_checkpoint, load_model_state_dict, load_swa_model_state_dict
from katago.train.model_pytorch import Model

SIZE = 19
NAME = re.compile(r"(.+)-s(\d+)-d(\d+)")
FUTUREPOS = "value_head.conv_futurepos.weight"
CLEAR_GAP = 0.15


def load(path):
    """The checkpoint's SWA weights (what the run exports), or its raw weights when it has none; (network, which).
    A checkpoint from before the territory income (2026-10-03) has 2 future-position outputs, padded to 4 with zeros
    as enclosure_checkpoint_losses.py does (this reads neither)."""
    from torch.optim.swa_utils import AveragedModel
    state = load_checkpoint(path)
    model = Model(state["config"], SIZE)
    model.initialize()

    def pad(sd, key):
        if key in sd and sd[key].shape[0] == 2:
            w = torch.zeros((4, *sd[key].shape[1:]), dtype=sd[key].dtype)
            w[:2] = sd[key]
            sd[key] = w
        return sd

    model.load_state_dict(pad(load_model_state_dict(state), FUTUREPOS))
    swa = load_swa_model_state_dict(state)
    if swa is None:
        return model.eval(), "raw"
    averaged = AveragedModel(model, device="cpu")
    averaged.load_state_dict(pad(swa, "module." + FUTUREPOS))
    return averaged.module.eval(), "swa"


def make_positions(args):
    files = sorted(pathlib.Path(args.tdata).glob("**/*.npz"), key=lambda p: p.stat().st_mtime)[-args.files:]
    rng = np.random.default_rng(args.seed)
    packed, glob, sources = [], [], []
    for f in files:
        with np.load(f) as npz:
            n = npz["globalInputNC"].shape[0]
            rows = sorted({int(r) for r in (np.arange(args.per_file) + rng.random()) * n / args.per_file})
            packed.append(npz["binaryInputNCHWPacked"][rows])
            glob.append(npz["globalInputNC"][rows])
            sources += [f"{f.parent.parent.name}/{f.name}:{r}" for r in rows]
    np.savez_compressed(args.out, binaryInputNCHWPacked=np.concatenate(packed), globalInputNC=np.concatenate(glob),
                        sources=np.array(sources))
    print(f"{len(sources)} positions from {len(files)} files into {args.out}")


def load_positions(path):
    with np.load(path) as npz:
        binary = np.unpackbits(npz["binaryInputNCHWPacked"], axis=2)[:, :, :SIZE * SIZE]
        binary = binary.reshape(binary.shape[0], binary.shape[1], SIZE, SIZE).astype(np.float32)
        return torch.from_numpy(binary), npz["globalInputNC"]


@torch.no_grad()
def spread(model, binary, global_raw):
    """The four numbers of the docstring, for one network on the positions."""
    g = torch.from_numpy(fit_global_input_nc(global_raw, modelconfigs.get_num_global_input_features(model.config)))
    g = g.float()
    policies, wins, leads = [], [], []
    for s in range(NUM_SYMMETRIES):
        out = model(apply_symmetry_inputs(binary, s), g)[0]  # the main heads
        policies.append(torch.softmax(out[0][:, 0].float(), dim=1))
        wins.append(torch.softmax(out[1].float(), dim=1)[:, 0])
        leads.append(out[2][:, 2].float() * model.lead_multiplier)
    tv, agree = [], []
    top2 = policies[0].topk(2, dim=1).values
    clear = (top2[:, 0] - top2[:, 1]) > CLEAR_GAP
    for s in range(1, NUM_SYMMETRIES):
        expected = apply_symmetry_policy(policies[0].unsqueeze(1), s, binary, g)[:, 0]  # the identity's, in frame s
        tv.append(0.5 * (expected - policies[s]).abs().sum(dim=1))
        agree.append((expected.argmax(dim=1) == policies[s].argmax(dim=1)).float())
    agree = torch.stack(agree)  # symmetries, positions
    wins, leads = torch.stack(wins, dim=1), torch.stack(leads, dim=1)
    undecided = (wins.mean(dim=1) >= 0.1) & (wins.mean(dim=1) <= 0.9)
    return {"policyTV": round(float(torch.stack(tv).mean()), 4), "topAgree": round(float(agree.mean()), 4),
            "topAgreeClear": round(float(agree[:, clear].mean()), 4) if bool(clear.any()) else None,
            "clearPositions": int(clear.sum()),
            "winSD": round(float(wins[undecided].std(dim=1).mean()), 4) if bool(undecided.any()) else None,
            "winPositions": int(undecided.sum()), "leadSD": round(float(leads.std(dim=1).mean()), 2)}


def checkpoint_dirs(run):
    """Every exported checkpoint's folder by name: models/ holds the exports of the network that self-play plays (at
    the time), models_extra/ the other trainers'."""
    return {d.name: d for kind in ("models", "models_extra") if (run / kind).is_dir()
            for d in (run / kind).iterdir() if NAME.fullmatch(d.name)}


def candidates(run, done, every_rows):
    """The checkpoints to measure now, oldest first: after the last input change, one per every_rows of each size."""
    since = max((int(p.read_text().split()[0]) for p in run.glob("*_since")), default=0)
    last = {}
    for name in done:
        m = NAME.fullmatch(name)
        if m:
            last[m.group(1)] = max(last.get(m.group(1), -1), int(m.group(3)))
    found = []
    for d in checkpoint_dirs(run).values():
        m, ckpt = NAME.fullmatch(d.name), d / "model.ckpt"
        if d.name not in done and ckpt.exists() and ckpt.stat().st_mtime >= since:
            found.append((m.group(1), int(m.group(3)), int(m.group(2)), d))
    chosen = []
    for size, rows, samples, d in sorted(found, key=lambda f: (f[0], f[1], f[2])):
        if rows >= last.get(size, -every_rows) + every_rows:
            chosen.append((size, rows, samples, d))
            last[size] = rows
    return sorted(chosen, key=lambda c: c[1])


def fill_missing(run, out, binary, global_raw):
    """Measure again each line of out that lacks a number spread() now computes, and rewrite out after each."""
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    dirs = checkpoint_dirs(run)
    for i, row in enumerate(rows):
        if "topAgreeClear" in row or row["model"] not in dirs or not (dirs[row["model"]] / "model.ckpt").exists():
            continue
        start = time.time()
        net, weights = load(str(dirs[row["model"]] / "model.ckpt"))
        rows[i] = {**row, "weights": weights, **spread(net, binary, global_raw), "seconds": round(time.time() - start)}
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(out)
        print(f"filled {row['model']}", flush=True)
        del net


def measure(args):
    run = pathlib.Path(args.run)
    out = run / "symmetry.jsonl"
    torch.set_num_threads(args.threads)
    binary, global_raw = load_positions(run / "symmetry_positions.npz")
    if not args.only:
        fill_missing(run, out, binary, global_raw)
    while True:
        done = {json.loads(line)["model"] for line in out.read_text().splitlines()} if out.exists() else set()
        todo = candidates(run, done, args.every_rows)
        if args.only:
            dirs = checkpoint_dirs(run)
            todo = [(m.group(1), int(m.group(3)), int(m.group(2)), dirs[n])
                    for n in args.only for m in [NAME.fullmatch(n)]]
        for size, rows, samples, d in todo:
            start = time.time()
            net, weights = load(str(d / "model.ckpt"))
            exported = datetime.datetime.fromtimestamp((d / "model.ckpt").stat().st_mtime)
            row = {"model": d.name, "size": size, "samples": samples, "rows": rows,
                   "exported": exported.isoformat(" ", "seconds"),
                   "weights": weights, "positions": binary.shape[0], **spread(net, binary, global_raw),
                   "seconds": round(time.time() - start)}
            if not args.only:
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            del net
        if not args.watch or args.only:
            return
        time.sleep(args.watch)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("positions")
    p.add_argument("out")
    p.add_argument("--tdata", required=True)
    p.add_argument("--files", type=int, default=64)
    p.add_argument("--per-file", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("measure")
    m.add_argument("run")
    m.add_argument("--every-rows", type=int, default=250000)
    m.add_argument("--watch", type=int, default=0)
    m.add_argument("--threads", type=int, default=2)
    m.add_argument("--only", nargs="+")
    args = ap.parse_args()
    make_positions(args) if args.command == "positions" else measure(args)


if __name__ == "__main__":
    main()
