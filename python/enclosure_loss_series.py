#!/usr/bin/python3
"""Enclosure fork: print a V2 network's training losses over time (one line per sampled training window).

    python enclosure_loss_series.py METRICS_TRAIN_JSON [KEY ...]

Reported losses are sums over a batch's rows divided by its row count (metrics_logging.log_metrics), including
rows that carry no target for that loss.
"""

import json
import sys

KEYS = ("sbpdfloss", "sbcdfloss", "smloss", "fploss", "skloss", "sloss", "iloss", "oloss")


def main():
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    keys = sys.argv[2:] or KEYS
    step = max(1, len(rows) // 40)
    print("nsamp " + " ".join(keys))
    for r in rows[::step] + [rows[-1]]:
        print(int(r["nsamp"]), " ".join(f"{r.get(k, float('nan')):.4f}" for k in keys))


if __name__ == "__main__":
    main()
