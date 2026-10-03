#!/usr/bin/python3
"""Enclosure fork: the C++ model's outputs against the PyTorch checkpoint's, on the same input rows.

    python enclosure_nn_outputs_check.py CHECKPOINT DUMP [DUMP ...]

Each DUMP comes from the fork's testnnoutputs (katago/cpp/tests/enclosure/testnnoutputs.cpp), which evaluated a
model exported from CHECKPOINT (export_model_pytorch.py, without -use-swa) on positions and wrote the input rows
the network got with its outputs. This runs CHECKPOINT's model on the same rows and reports, per output, the
largest difference: the value (win, loss, no result), the ownership, RGSC's regret value and ranking, and the
exported training-only outputs (the aux values and the 9 aux maps). The C++ value and ownership are in white's
view; the rest, like PyTorch's outputs, in the view of the player to move.
"""

import sys

import numpy as np
import torch

from katago.train.model_pytorch import Model

S = 19
BLACK, WHITE = 1, 2


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(ckpt["config"], S)
    sd = ckpt["model"]
    prefix = [k for k in sd if k.endswith("value_head.conv1.weight")][0][:-len("value_head.conv1.weight")]
    model.load_state_dict({k[len(prefix):]: v for k, v in sd.items()}, strict=True)
    model.eval()
    return model, ckpt["config"]


def read_dump(path):
    positions, cur = [], None
    with open(path) as f:
        for line in f:
            word, *rest = line.split()
            if word == "position":
                cur = {"label": rest[0], "pla": int(rest[1]), "nhwc": rest[2] == "nhwc"}
                positions.append(cur)
            elif word == "spatial":
                cur["spatial"] = [(int(i), float(v)) for i, v in (e.split(":") for e in rest[1:])]
            else:
                cur[word] = np.array([float(x) for x in rest], dtype=np.float64)
    return positions


def main():
    model, config = load_model(sys.argv[1])
    c_in = model.num_bin_input_features if hasattr(model, "num_bin_input_features") else 215
    worst_all = {}
    for dump in sys.argv[2:]:
        positions = read_dump(dump)
        spatial = torch.zeros(len(positions), c_in, S * S)
        glob = torch.zeros(len(positions), len(positions[0]["global"]))
        for n, p in enumerate(positions):
            for i, v in p["spatial"]:
                if p["nhwc"]:
                    spatial[n, i % c_in, i // c_in] = v
                else:
                    spatial[n, i // (S * S), i % (S * S)] = v
            glob[n] = torch.tensor(p["global"])
        with torch.no_grad():
            (out_policy, out_value, out_misc, out_moremisc, out_ownership, out_scoring, out_futurepos, out_seki,
             out_scorebelief, out_regret) = model(spatial.view(-1, c_in, S, S), glob)[0]
        worst = {}

        def note(name, diff):
            worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(diff))))

        for n, p in enumerate(positions):
            value = torch.softmax(out_value[n], dim=0).numpy()  # mover's view
            cpp_value = p["value"] if p["pla"] == WHITE else p["value"][[1, 0, 2]]
            note("value", value - cpp_value)
            owner = torch.tanh(out_ownership[n, 0]).numpy().reshape(-1)
            note("ownership", owner - (p["owner"] if p["pla"] == WHITE else -p["owner"]))
            note("regret value", torch.sigmoid(out_regret[n, 0]).item() * 4.0 - p["regret"][0])
            note("regret ranking", out_regret[n, 1].item() - p["regret"][1])
            aux_values = torch.cat((out_misc[n, 4:10], out_moremisc[n, 2:8])).numpy()
            note("aux values", aux_values - p["auxvalues"])
            aux_map = torch.cat((out_futurepos[n], out_seki[n], out_scoring[n])).numpy().reshape(9, -1)
            cpp_map = p["auxmap"].reshape(9, -1)
            for c, name in enumerate(["future pos 0", "future pos 1", "income mover", "income opponent",
                                      "cut edge 0", "cut edge 1", "cut edge 2", "cut edge 3", "occupancy"]):
                note("aux map: " + name, aux_map[c] - cpp_map[c])
        print(f"{dump}: {len(positions)} positions; largest differences:")
        for name, d in worst.items():
            print(f"  {name:22s} {d:.2e}")
            worst_all[name] = max(worst_all.get(name, 0.0), d)
        scale = {"regret ranking": float(out_regret[:, 1].abs().max()), "aux values": float(out_misc.abs().max())}
        print(f"  (output magnitudes: ranking up to {scale['regret ranking']:.2f}, misc values up to {scale['aux values']:.2f})")


if __name__ == "__main__":
    main()
