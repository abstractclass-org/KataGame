#!/usr/bin/python3
"""Enclosure fork: RGSC's analysis figures, ours after the RGSC paper (Tsai et al., 2026) and the Gomoku thesis
(Lin, 2026), on one page.

    python enclosure_rgsc_report.py --variants V.json --selectors S.json --analysis A.json --snapshots DIR --out DIR

Inputs: enclosure_regret_variants.py --json (regret's definitions, the value's accuracy, example games),
enclosure_rgsc_selectors.py --json (the heads' picks: the paper's Figure 7), enclosure_rgsc_analysis.py --json (restart
depths by model: the paper's Figures 16-18), and the buffer snapshots of v2/scripts/rgsc_snapshots.sh (regret at entry
against the last update: the paper's Figure 8, the thesis's Figures 4.5-4.6). Needs matplotlib; the training
environment does not carry it, so run it in an environment of its own.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2f6fd0", "#e08a1e", "#2e9a5a", "#d0442f", "#888888", "#8a5ac8"


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=110)
    plt.close(fig)
    return name


def fig_selectors(s, out):
    names = ["random", "ranking", "regret value", "shallowest", "best"]
    colors = [GREY, ORANGE, BLUE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vals = [s["picks"][n] for n in names]
    errs = [1.96 * s["pick_sd"][n] for n in names]
    ax.bar(names, vals, yerr=errs, color=colors, capsize=4)
    for i, n in enumerate(names):
        ax.text(i, vals[i] + 0.02, f"{vals[i]:.3f}\nmove {int(s['depths'][n])}", ha="center", fontsize=8)
    ax.set_ylabel("realized regret of the pick")
    ax.set_title(f"One pick per game ({s['games']} games): who finds high regret?")
    ax.set_ylim(0, max(vals) * 1.3)
    return save(fig, out, "selectors.png")


def fig_heads_by_depth(s, out):
    rows = s["corr_by_depth"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.2, [r["ranking"] for r in rows], 0.4, color=ORANGE, label="ranking head")
    ax.bar(x + 0.2, [r["regret value"] for r in rows], 0.4, color=BLUE, label="regret value head")
    ax.set_xticks(x, [r["turns"] for r in rows])
    ax.set_xlabel("turns")
    ax.set_ylabel("rank correlation with realized regret")
    ax.set_title("The heads within each depth (0 = no skill beyond depth)")
    ax.legend(fontsize=8)
    return save(fig, out, "heads_by_depth.png")


def fig_curriculum(a, out):
    models, buckets = a["curriculum"]["models"], a["curriculum"]["buckets"]
    # Few restarts are sampled per model, so models are pooled in bins of 500,000 data rows, weighted by their restarts.
    bins = {}
    for m in models:
        b = int(m["row"] // 500_000)
        n, s = bins.get(b, (0, np.zeros(len(buckets))))
        bins[b] = (n + m["starts"], s + m["starts"] * np.array(m["shares"]))
    keys = sorted(bins)
    rows = np.array([(k + 0.5) * 0.5 for k in keys])
    shares = np.array([bins[k][1] / bins[k][0] for k in keys]).T
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(buckets)))
    ax.stackplot(rows, shares, labels=[f"moves {b}" for b in buckets], colors=colors)
    ax.set_xlabel("data row of the self-play model (millions)")
    ax.set_ylabel("share of restart states")
    ax.set_title("Restart states by opening length, by model (the paper's Figures 16-18)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return save(fig, out, "curriculum.png")


def snapshot_history(snap_dir, since=None):
    """Each buffer state over the snapshots: its regret when first seen and when last seen, and its restarts. With
    since (YYYYMMDD-HHMM), only the snapshots taken from then on (their names are the time they were taken)."""
    files = sorted(f for f in glob.glob(os.path.join(snap_dir, "*.txt")) if since is None or os.path.basename(f) >= since)
    seen = {}
    for k, f in enumerate(files):
        for line in open(f):
            w = line.split()
            if not w or w[0] != "entry":
                continue
            key = tuple(w[4:4 + 2 * int(w[3])])
            regret, restarts = float(w[1]), int(w[2])
            if key not in seen:
                seen[key] = {"first": k, "r0": regret, "n0": restarts}
            seen[key].update(last=k, r1=regret, n1=restarts)
    return files, seen


def fig_entry_vs_last(snap_dir, out, since=None):
    files, seen = snapshot_history(snap_dir, since)
    if len(files) < 2:
        return None, "fewer than two snapshots"
    # States seen to enter after the first snapshot (their entry regret is known), and updated by a restart since.
    entered = [s for s in seen.values() if s["first"] > 0]
    updated = [s for s in entered if s["n1"] > s["n0"]]
    pool = updated if len(updated) >= 20 else [s for s in seen.values() if s["n1"] > s["n0"]]
    note = (f"{len(files)} snapshots; {len(seen)} states seen, {len(entered)} entered after the first snapshot, "
            f"{len(updated)} of those restarted since"
            + ("" if pool is updated else f" (too few: all {len(pool)} states restarted between snapshots shown)"))
    if not pool:
        return None, note
    r0 = np.array([s["r0"] for s in pool])
    r1 = np.array([s["r1"] for s in pool])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    bins = np.linspace(0, max(r0.max(), r1.max()) * 1.05, 30)
    ax1.hist(r0, bins=bins, alpha=0.6, color=BLUE, label=f"first seen (mean {r0.mean():.3f})")
    ax1.hist(r1, bins=bins, alpha=0.6, color=ORANGE, label=f"last seen (mean {r1.mean():.3f})")
    ax1.set_xlabel("regret")
    ax1.set_ylabel("states")
    ax1.legend(fontsize=8)
    ax1.set_title("Buffer regret, first against last (Figure 8)" + (f", from {since}" if since else ""))
    d = r1 - r0
    ax2.hist(d, bins=30, color=BLUE)
    ax2.axvline(0, color=GREY, ls="--")
    ax2.set_xlabel("ΔR = last − first")
    ax2.set_title(f"ΔR (mean {d.mean():+.3f}, median {np.median(d):+.3f})")
    return save(fig, out, f"entry_vs_last_since_{since}.png" if since else "entry_vs_last.png"), note


def fig_definitions(v, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    styles = {"vanilla": BLUE, "discounted": ORANGE, "swing": GREEN}
    for name, color in styles.items():
        y = np.array(v["by_move"][name])
        ax.plot(y / y.max(), color=color, label=f"{name} (max {y.max():.3f})")
    ax.set_xlabel("move")
    ax.set_ylabel("mean regret, scaled to its maximum")
    ax.set_title(f"Regret's definitions by move ({v['games']:,} games, γ {v['gamma']})")
    ax.legend(fontsize=8)
    return save(fig, out, "definitions.png")


def fig_accuracy(v, out):
    acc = v["accuracy"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(acc["sign_right"], color=BLUE, label="value's sign right")
    ax.plot(acc["mean_abs_v"], color=GREEN, label="mean |v|")
    ax.plot(acc["mean_err"], color=RED, label="mean (v − z)²")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xlabel("move")
    ax.set_title("The search value along a game: uncertain early, not wrong")
    ax.legend(fontsize=8)
    return save(fig, out, "accuracy.png")


def fig_pick_depths(v, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.arange(0, 125, 5)
    for name, color in (("vanilla", BLUE), ("discounted", ORANGE), ("swing", GREEN), ("vanilla by depth", PURPLE)):
        ax.hist(v["pick_depths"][name], bins=bins, histtype="step", lw=2, color=color, label=name, density=True)
    ax.set_xlabel("move of the per-game pick")
    ax.set_ylabel("density")
    ax.set_title("Where each definition picks a game's state")
    ax.legend(fontsize=8)
    return save(fig, out, "pick_depths.png")


def fig_examples(v, out):
    ex = v["examples"]
    if not ex:
        return None
    n = len(ex)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 2.6 * ((n + 1) // 2)), squeeze=False)
    for ax, e in zip(axes.flat, ex):
        val = np.array(e["value"])
        ax.plot(val, color="#333333", lw=1.4, label="search value v (White)")
        ax.axhline(e["z"], color=RED, ls="--", lw=1, label=f"outcome z = {e['z']:+.0f}")
        van = np.array(e["vanilla"])
        sw = np.array(e["swing"])
        ax.plot(van / van.max(), color=BLUE, lw=1, label="vanilla (scaled)")
        ax.plot(sw / sw.max(), color=GREEN, lw=1, label="swing (scaled)")
        ax.axvline(e["vanilla_pick"], color=BLUE, ls=":", lw=1.5)
        ax.axvline(e["swing_pick"], color=GREEN, ls=":", lw=1.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("move")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Example games: vanilla picks (blue dotted) against swing picks (green dotted)")
    return save(fig, out, "examples.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--selectors", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", action="append", default=[], metavar="YYYYMMDD-HHMM[=LABEL]",
                    help="also draw Figure 8 from the snapshots taken from this time on (repeatable), e.g. after a change")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    v, s, a = (json.load(open(p)) for p in (args.variants, args.selectors, args.analysis))
    entry_png, entry_note = fig_entry_vs_last(args.snapshots, args.out)
    since_figs = []
    for spec in args.since:
        when, _, label = spec.partition("=")
        png, note = fig_entry_vs_last(args.snapshots, args.out, when)
        since_figs.append((f"Figure 8 again, from {when}" + (f": {label}" if label else ""), png,
                           f"Only the snapshots taken from {when} on. {note}."))
    figs = [
        ("The paper's Figure 7 / the thesis's Figure 4.1: one state picked per game",
         fig_selectors(s, args.out),
         "Mean realized regret (Eq. 2) of the state each selector picks in a game, with the median move of the picks. "
         "Both heads beat a random pick, as in both papers. Picking the game's first state beats both heads."),
        ("The heads within each depth", fig_heads_by_depth(s, args.out),
         "Rank correlation of each head with the realized regret among states of the same depth. Early, where restart "
         "states come from, the heads see little beyond depth."),
        ("The paper's Figures 16-18: restart states by opening length", fig_curriculum(a, args.out),
         "The share of restart states by their move number, for each self-play model (from the training rows)."),
        ("The paper's Figure 8 / the thesis's Figures 4.5-4.6: buffer regret, first against last", entry_png,
         f"{entry_note}. A shift to the left means restarts teach the network those states."),
        *since_figs,
        ("Why vanilla regret picks the first moves: the value along a game", fig_accuracy(v, args.out),
         "Early, the search value is near 0 (uncertain), so (v − z)² is near 1 at every early state. Eq. 2 averages "
         "the error from a state to the end, and the error falls as the value converges on the outcome, so the "
         "average is largest at the start. Not a bug: the definition rewards uncertainty, not mistakes."),
        ("Regret's definitions by move", fig_definitions(v, args.out),
         "Vanilla (Eq. 2), step-discounted (the thesis), and value swing: Σ γ^k (v_{t+k+1} − v_{t+k})² / Σ γ^k, the "
         "TD error of prioritized experience replay. The discount moves the picks only a few moves later."),
        ("Where each definition picks a game's state", fig_pick_depths(v, args.out),
         "Vanilla and discounted pick the first moves; vanilla's percentile within its depth picks late upsets; swing "
         "spreads its picks over the whole game."),
        ("Example games", fig_examples(v, args.out),
         "The search value (black) against the outcome (red). Swing picks the state just before the value turns."),
    ]
    parts = []
    for title, png, text in figs:
        img = f'<img src="{png}" alt="{title}">' if png else "<p><em>No data yet.</em></p>"
        parts.append(f"<section><h2>{title}</h2>{img}<p>{text}</p></section>")
    page = f"""<!doctype html><meta charset="utf-8"><title>RGSC analysis</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:16px auto;max-width:1000px;padding:0 16px;color:#222;background:#fff}}
img{{max-width:100%;border:1px solid #eee}} h2{{font-size:17px;margin-top:28px}} p{{color:#444}}</style>
<h1>RGSC analysis</h1>
<p>Our versions of the analysis figures of the RGSC paper (Tsai et al., 2026) and of the Gomoku thesis (Lin, 2026).
The buffer's states on the board: <a href="buffer_top_regret.html">by regret</a>,
<a href="buffer_top_restarts.html">by restarts</a>.</p>
{"".join(parts)}"""
    open(os.path.join(args.out, "report.html"), "w", encoding="utf-8").write(page)
    print(f"wrote {args.out}/report.html")


if __name__ == "__main__":
    main()
