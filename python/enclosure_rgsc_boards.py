#!/usr/bin/python3
"""Enclosure fork: RGSC's buffer states drawn on the board, as the RGSC paper's Figure 13 shows its examples.

    python enclosure_rgsc_boards.py BUFFER_FILE OUT.html [--top 24] [--by regret|restarts]

Each buffer entry (program/rgsc.cpp's file: "entry REGRET RESTARTS LENGTH LOC PLAYER ...") is replayed in the Python
engine and drawn as an SVG board: each player's edges, the last turn's edges (protected) thicker, each player's
enclosed regions filled. Under each board: its regret, restarts drawn, depth in moves, the side to move and the
scores. A move's Loc is MOVE_LOC_BASE + the engine's action index (katago/cpp/game/board.h), MOVE_LOC_BASE being
(19 + 1) * (19 + 2) + 1 = 421 for the 19-point board.
"""

import argparse
import html
import os
import sys

import numpy as np
import shapely

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from enclosure.engine import SIZE, Position, decode_action  # noqa: E402

MOVE_LOC_BASE = (19 + 1) * (19 + 2) + 1
PASS_LOC = 1
COLORS = ("#2f6fd0", "#d0442f")  # blue, red
CELL, MARGIN = 20, 14


def entries(path):
    out = []
    for line in open(path):
        w = line.split()
        if not w or w[0] != "entry":
            continue
        regret, restarts, length = float(w[1]), int(w[2]), int(w[3])
        moves = [(int(w[4 + 2 * i]), int(w[5 + 2 * i])) for i in range(length)]
        out.append({"regret": regret, "restarts": restarts, "moves": moves})
    return out


def replay(moves):
    p = Position()
    for loc, _ in moves:
        p.play(17328 if loc == PASS_LOC else loc - MOVE_LOC_BASE)
    return p


def regions(edges):
    if len(edges) < 3:
        return []
    noded = shapely.union_all(shapely.linestrings(np.array(list(edges), dtype=float)))
    return list(shapely.get_parts(shapely.polygonize(shapely.get_parts(noded))))


def svg(p):
    w = 2 * MARGIN + (SIZE - 1) * CELL
    xy = lambda pt: (MARGIN + pt[0] * CELL, MARGIN + (SIZE - 1 - pt[1]) * CELL)  # y up, as the site draws it
    out = [f'<svg viewBox="0 0 {w} {w}" width="{w}" height="{w}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{w}" height="{w}" fill="#fbfaf7"/>']
    for k in range(2):
        for poly in regions(p.edges[k]):
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(c) for c in poly.exterior.coords))
            out.append(f'<polygon points="{pts}" fill="{COLORS[k]}" fill-opacity="0.18" stroke="none"/>')
    for x in range(SIZE):
        for y in range(SIZE):
            cx, cy = xy((x, y))
            out.append(f'<circle cx="{cx}" cy="{cy}" r="1.1" fill="#b9b4a8"/>')
    for k in range(2):
        for a, b in p.edges[k]:
            (x1, y1), (x2, y2) = xy(a), xy(b)
            thick = 3.4 if (a, b) in p.protected else 1.8
            out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS[k]}" stroke-width="{thick}" '
                       f'stroke-linecap="round"/>')
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--by", choices=("regret", "restarts"), default="regret")
    args = ap.parse_args()
    es = entries(args.buffer)
    bad = [e for e in es if e["moves"] and (decode_action(e["moves"][0][0] - MOVE_LOC_BASE)[0] not in ((0, 9), (3, 9)))]
    if bad:
        raise SystemExit(f"{len(bad)} entries whose first move does not start at a blue start node: wrong MOVE_LOC_BASE?")
    es.sort(key=lambda e: (-e[args.by], -e["regret"]))
    cards = []
    for e in es[:args.top]:
        p = replay(e["moves"])
        side = ("Blue", "Red")[p.player]
        cards.append(f'<figure>{svg(p)}<figcaption>regret {e["regret"]:.3f} · restarts {e["restarts"]} · '
                     f'move {len(e["moves"])} · {side} to move · score {p.scores[0]:.0f}–{p.scores[1]:.0f}'
                     f'</figcaption></figure>')
    regrets = [e["regret"] for e in es]
    depths = [len(e["moves"]) for e in es]
    page = f"""<!doctype html><meta charset="utf-8"><title>RGSC buffer states</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#222}}
figure{{display:inline-block;margin:6px;text-align:center}} figcaption{{font-size:12px;color:#555}}</style>
<h1>RGSC buffer states</h1>
<p>{html.escape(os.path.basename(args.buffer))}: {len(es)} states, regret {min(regrets):.3f}–{max(regrets):.3f},
depth {min(depths)}–{max(depths)} moves. Top {args.top} by {args.by}. Thick edges: the last turn's (protected).
Shaded: each player's enclosed regions.</p>
{"".join(cards)}"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(page)
    print(f"wrote {args.out}: {min(args.top, len(es))} boards of {len(es)} states")


if __name__ == "__main__":
    main()
