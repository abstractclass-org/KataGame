#!/usr/bin/python3
"""KataGame fork (2026-11-03, the engine repository's abscls-ra93.3): the C++ network evaluation against a PyTorch
forward pass on a training run's rows (a holdout), for any game, on every output that the search reads:

    python nn_rows_check.py write ROWS_FILE NPZ [NPZ ...] -size LEN [-rows-per-file 32]
    testnnrows MODEL ROWS_FILE OUT_FILE FP16 NHWC [EXACT [BATCH]]     (katagame/cpp/tests/enclosure/testnnrows.cpp)
    python nn_rows_check.py compare CHECKPOINT ROWS_FILE OUT_FILE [-use-swa] [-tolerance 1e-3] [-torch-cache NPZ]

`write` takes the first rows of each NPZ (a shuffled file, so rows of many games and of every part of a game), unpacks
their spatial inputs, and writes them with their global inputs in testnnrows' format. testnnrows evaluates MODEL, the
checkpoint's export (export_model_pytorch.py; with -use-swa, the SWA weights, which a run's exports carry), on those
rows and writes its outputs. `compare` runs the checkpoint's network on the same rows in fp32 on the CPU. It passes the
engine's outputs through the model's own post-processing (Model.postprocess_single_heads_output: the multipliers, the
softplus of the error heads, RGSC's sigmoid), so both sides read as the search reads them, and compares per row:

- policy, policyopt: the first and the optimistic policy output (the engine's two policy channels, which the search
  blends by its policy optimism), as probabilities over the legal moves (games.legal_moves; every entry for a game
  without it): the largest difference over the largest probability;
- value: the win and loss probabilities (the fork's NNEvaluator renormalises them: no result is impossible);
- score: the score mean and the lead, each over its multiplier (in the network's units);
- uncertainty: the score's stdev, the time left's variance and the short-term winloss and score errors, which the
  search's uncertainty weighting reads, each over its multiplier;
- intrinsic: AZdb's returns A and B (the members' search), relative to their size (at least 1);
- regret: RGSC's regret value (0 to 4) over 4 and its ranking relative to its size (at least 1), and the swing
  regret's, where the engine computes them (self-play's CUDA backend; the Eigen backend does not);
- own: the ownership after tanh on the board's points (not used to choose a move).

It also prints the largest differences of the raw policy and value logits. Exit code 0 when every row agrees within
the tolerance on every measure.
"""

import argparse
import os
import sys

import numpy as np
import torch

from katago.train import games, modelconfigs
from katago.train.load_model import load_model

MEASURES = ("policy", "policyopt", "value", "score", "uncertainty", "intrinsic", "regret", "own")
# The post-processed outputs (Model.postprocess_single_heads_output) that the comparison reads, by index.
SCALARS = {"scoremean": 8, "scorestdev": 9, "lead": 10, "vtime": 11, "stverror": 12, "stserror": 13,
           "regret": 15, "ranking": 16, "swing": 17, "swingranking": 18}
OPTIMISTIC = 5  # the policy output that the export writes as the engine's optimistic channel (write_policy_head)


def write(args):
    area = args.size * args.size
    spatial, global_ = [], []
    for path in args.npz:
        with np.load(path) as z:
            packed = z["binaryInputNCHWPacked"][:args.rows_per_file]
            global_.append(z["globalInputNC"][:args.rows_per_file].astype("<f4"))
        if packed.shape[2] * 8 < area:
            raise ValueError(f"{path}: {packed.shape[2] * 8} bits a plane, fewer than a board of {args.size}")
        spatial.append(np.unpackbits(packed, axis=2)[:, :, :area].astype("<f4"))
    spatial = np.concatenate(spatial)
    global_ = np.concatenate(global_)
    n, c = spatial.shape[0], spatial.shape[1]
    with open(args.rows_file, "wb") as f:
        f.write(f"nnrows {n} {c} {args.size} {global_.shape[1]}\n".encode("ascii"))
        f.write(spatial.tobytes())
        f.write(global_.tobytes())
    print(f"wrote {n} rows of {c} planes and {global_.shape[1]} global inputs from {len(args.npz)} files")
    return 0


def read_rows(path):
    with open(path, "rb") as f:
        word, n, c, size, g = f.readline().split()
        assert word == b"nnrows", word
        n, c, size, g = int(n), int(c), int(size), int(g)
        spatial = np.frombuffer(f.read(4 * n * c * size * size), dtype="<f4").reshape(n, c, size, size)
        global_ = np.frombuffer(f.read(4 * n * g), dtype="<f4").reshape(n, g)
    return spatial, global_


def read_outputs(path):
    """testnnrows' sections: {name: (rows, count) float64}."""
    with open(path, "rb") as f:
        word, n, *sections = f.readline().decode("ascii").split()
        assert word == "nnrowsout", word
        n = int(n)
        names = [s.split(":")[0] for s in sections]
        counts = [int(s.split(":")[1]) for s in sections]
        data = np.frombuffer(f.read(4 * n * sum(counts)), dtype="<f4").reshape(n, sum(counts)).astype(np.float64)
    return dict(zip(names, np.split(data, np.cumsum(counts)[:-1], axis=1)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def post(model, value, score, extra):
    """The post-processed scalars of raw outputs in the engine's layout (score: misc value 0-3 and more-misc 0-1, as
    write_value_head exports them; extra: RGSC's 4 then AZdb's 2), through the model's own post-processing."""
    n = value.shape[0]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))  # noqa: E731
    misc, moremisc = torch.zeros(n, 10, dtype=torch.float64), torch.zeros(n, 8, dtype=torch.float64)
    misc[:, 0:4], moremisc[:, 0:2] = t(score[:, 0:4]), t(score[:, 4:6])
    empty = torch.zeros(n, 1, dtype=torch.float64)
    out = model.postprocess_single_heads_output(
        (empty, t(value), misc, moremisc, empty, empty, empty, empty, empty, t(extra)))
    result = {k: out[i].numpy() for k, i in SCALARS.items()}
    result["intrinsic"] = out[19].numpy()
    return result


def torch_outputs(model, game, spatial, global_, cache=None):
    """The network's outputs on the rows: the policy logits of the first and the optimistic output, the value logits,
    the post-processed scalars (SCALARS, intrinsic), the ownership before tanh, and the legal moves; from ``cache`` (an
    .npz that an earlier call wrote) when it exists."""
    if cache is not None and os.path.exists(cache):
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    parts = {}
    with torch.no_grad():
        for start in range(0, spatial.shape[0], 32):
            binary = torch.from_numpy(spatial[start:start + 32].copy())
            glob = torch.from_numpy(global_[start:start + 32].copy())
            out = model.postprocess_output(model(binary, glob))[0]
            legal = games.legal_moves(game, binary, glob)
            policy = out[0].double().numpy()
            batch = {"policy": policy[:, 0], "policyopt": policy[:, OPTIMISTIC], "value": out[1].double().numpy(),
                     "own": out[4].double().numpy().reshape(policy.shape[0], -1), "intrinsic": out[19].double().numpy(),
                     "legal": legal.numpy() if legal is not None else np.ones(policy[:, 0].shape, dtype=bool)}
            batch.update({k: out[i].double().numpy() for k, i in SCALARS.items()})
            for k, v in batch.items():
                parts.setdefault(k, []).append(v)
    result = {k: np.concatenate(v) for k, v in parts.items()}
    if cache is not None:
        np.savez(cache, **result)
    return result


def compare(args):
    model, swa_model, _ = load_model(args.checkpoint, use_swa=args.use_swa, device="cpu")
    if args.use_swa:
        model = swa_model
    model.eval()
    game = modelconfigs.get_game(model.config)
    spatial, global_ = read_rows(args.rows_file)
    cpp = read_outputs(args.out_file)
    n = spatial.shape[0]
    assert cpp["policy"].shape[0] == n, (cpp["policy"].shape, n)
    ref = torch_outputs(model, game, spatial, global_, args.torch_cache)
    assert ref["policy"].shape == cpp["policy"].shape, (ref["policy"].shape, cpp["policy"].shape)
    # A section that the output lacks, or holds as NaN (a head that the model or the backend does not compute), is
    # left out of the comparison.
    present = {k for k in ("policyopt", "intrinsic", "regret") if k in cpp and not np.isnan(cpp[k]).any()}
    has_regret = "regret" in present
    extra = np.zeros((n, 6))
    for k, cut in (("regret", slice(0, 4)), ("intrinsic", slice(4, 6))):
        if k in present:
            extra[:, cut] = cpp[k]
    with torch.no_grad():
        mine = post(model, cpp["value"], cpp["score"], extra)
    mult = {"scoremean": model.scoremean_multiplier, "lead": model.lead_multiplier,
            "scorestdev": model.scorestdev_multiplier, "vtime": model.variance_time_multiplier,
            "stverror": model.shortterm_value_error_multiplier, "stserror": model.shortterm_score_error_multiplier}
    worst = {k: 0.0 for k in MEASURES}
    raw = dict(policy_logit=0.0, policyopt_logit=0.0, value_logit=0.0)
    failures = 0
    for r in range(n):
        moves = ref["legal"][r]
        diffs = {}
        for k in ("policy", "policyopt"):
            if k not in cpp or (k == "policyopt" and k not in present):
                continue
            t_logits, c_logits = ref[k][r][moves], cpp[k][r][moves]
            t_probs, c_probs = softmax(t_logits), softmax(c_logits)
            diffs[k] = np.abs(t_probs - c_probs).max() / max(t_probs.max(), 1e-9)
            raw[k + "_logit"] = max(raw[k + "_logit"], np.abs(t_logits - c_logits).max())
        diffs["value"] = np.abs(softmax(ref["value"][r][:2]) - softmax(cpp["value"][r][:2])).max()
        raw["value_logit"] = max(raw["value_logit"], np.abs(ref["value"][r] - cpp["value"][r]).max())
        rel = lambda k: abs(ref[k][r] - mine[k][r]) / mult[k]  # noqa: E731
        diffs["score"] = max(rel("scoremean"), rel("lead"))
        diffs["uncertainty"] = max(rel("scorestdev"), rel("vtime"), rel("stverror"), rel("stserror"))
        if "intrinsic" in present:
            diffs["intrinsic"] = (np.abs(ref["intrinsic"][r] - mine["intrinsic"][r]).max()
                                  / max(1.0, np.abs(ref["intrinsic"][r]).max()))
        if has_regret:
            diffs["regret"] = max(abs(ref["regret"][r] - mine["regret"][r]) / 4.0,
                                  abs(ref["swing"][r] - mine["swing"][r]) / 4.0,
                                  abs(ref["ranking"][r] - mine["ranking"][r]) / max(1.0, abs(ref["ranking"][r])),
                                  abs(ref["swingranking"][r] - mine["swingranking"][r])
                                  / max(1.0, abs(ref["swingranking"][r])))
        on_board = spatial[r, 0].reshape(-1) > 0
        own = np.abs(np.tanh(ref["own"][r]) - np.tanh(cpp["own"][r]))[on_board]
        diffs["own"] = own.max() if own.size else 0.0
        for k, d in diffs.items():
            worst[k] = max(worst[k], d)
        if max(diffs.values()) > args.tolerance:
            failures += 1
            if failures <= 10:
                print(f"FAIL row {r}: " + ", ".join(f"{k} {d:.2e}" for k, d in diffs.items()))
    shown = [k for k in MEASURES if k not in ("policyopt", "intrinsic", "regret") or k in present]
    missing = [k for k in ("policyopt", "intrinsic", "regret") if k not in present]
    print(f"{n} rows; largest differences: " + ", ".join(f"{k} {worst[k]:.2e}" for k in shown)
          + (f" (not in the output: {', '.join(missing)})" if missing else "")
          + "; raw: " + ", ".join(f"{k} {d:.2e}" for k, d in raw.items()
                                  if k != "policyopt_logit" or "policyopt" in present)
          + f"; {failures} over {args.tolerance}")
    return 0 if failures == 0 and n > 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("rows_file")
    w.add_argument("npz", nargs="+")
    w.add_argument("-rows-per-file", dest="rows_per_file", type=int, default=32)
    w.add_argument("-size", type=int, required=True, help="the network's board length (the game's MAX_BOARD_LEN)")
    c = sub.add_parser("compare")
    c.add_argument("checkpoint")
    c.add_argument("rows_file")
    c.add_argument("out_file", help="testnnrows' output on rows_file")
    c.add_argument("-use-swa", dest="use_swa", action="store_true",
                   help="the checkpoint's SWA weights, for a model exported with -use-swa (a run's models)")
    c.add_argument("-tolerance", type=float, default=1e-3)
    c.add_argument("-torch-cache", dest="torch_cache", default=None,
                   help="an .npz of PyTorch's outputs on rows_file: read when it exists, else written (one network's)")
    args = parser.parse_args()
    return write(args) if args.cmd == "write" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
