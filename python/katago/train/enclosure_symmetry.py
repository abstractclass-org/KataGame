"""Enclosure fork: board symmetries of training rows (KataGo's random symmetry augmentation).

KataGo mirrors and rotates every plane as pixels. In Enclosure an edge is stored at its lower end (the
canonical endpoint, lexicographic in (x, y)) in the plane of its offset, so under a flip an edge can move
to its other end and change offset: the edge planes and the policy are permuted as a whole, not pixel by
pixel. Point planes (on board, feature planes, ownership and the other value targets) transform as pixels.

Rows written with ENCLOSURE_EDGE_BOTH_ENDS (owner, 2026-10-03) also mark each edge at its upper end, in the
opposite offset's plane. Their edge planes are a plain permutation (each entry to its source's image, as a
move from the source), told apart row by row by any entry in a downward offset, which one-end rows never set:
old and new rows share the training window.

Symmetries are numbered as the C++ SymmetryHelpers: bit 0 flips y, bit 1 flips x, bit 2 transposes, applied
in that order. The images agree with SymmetryHelpers::getSymPosition and getSymAction (checked by
enclosure_symmetry_check.py against testsymmetry -dump).
"""

import numpy as np
import torch

SIZE = 19
NUM_SYMMETRIES = 8
NUM_OFFSETS = 48
OFFSETS = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
OFFSET_INDEX = {o: k for k, o in enumerate(OFFSETS)}
NUM_ACTIONS = SIZE * SIZE * NUM_OFFSETS   # Pass is the next index.

# V8 spatial inputs (NNInputs::NUM_FEATURES_SPATIAL_V8): on board, then 4 groups of edge planes, then the
# feature planes, whose plane 9 * player + 0 marks that player's nodes.
EDGE_PLANES_START = 1
NUM_EDGE_GROUPS = 4
FEATURE_PLANES_START = EDGE_PLANES_START + NUM_EDGE_GROUPS * NUM_OFFSETS
FEATURES_PER_PLAYER = 9


def sym_xy(x, y, symmetry):
    if symmetry & 2:
        x = SIZE - 1 - x
    if symmetry & 1:
        y = SIZE - 1 - y
    if symmetry & 4:
        x, y = y, x
    return x, y


def action_of(x, y, dx, dy):
    return (y * SIZE + x) * NUM_OFFSETS + OFFSET_INDEX[(dx, dy)]


def _tables(symmetry):
    """Per action (source, offset) with both ends on the board: its image from the image of the source
    (forward) and from the image's lower end (canonical), and its destination point; -1 off the board.
    Stored edges (lower end first) use edge: the canonical image, only for actions from their lower end."""
    forward = np.full(NUM_ACTIONS, -1, np.int64)
    canonical = np.full(NUM_ACTIONS, -1, np.int64)
    edge = np.full(NUM_ACTIONS, -1, np.int64)
    dest = np.full(NUM_ACTIONS, -1, np.int64)
    for y in range(SIZE):
        for x in range(SIZE):
            for k, (dx, dy) in enumerate(OFFSETS):
                bx, by = x + dx, y + dy
                if not (0 <= bx < SIZE and 0 <= by < SIZE):
                    continue
                a = (y * SIZE + x) * NUM_OFFSETS + k
                lower = (x, y) < (bx, by)
                sax, say = sym_xy(x, y, symmetry)
                sbx, sby = sym_xy(bx, by, symmetry)
                forward[a] = action_of(sax, say, sbx - sax, sby - say)
                if (sbx, sby) < (sax, say):
                    sax, say, sbx, sby = sbx, sby, sax, say
                canonical[a] = action_of(sax, say, sbx - sax, sby - say)
                if lower:
                    edge[a] = canonical[a]
                dest[a] = by * SIZE + bx
    return forward, canonical, edge, dest


_TABLES = {}


def tables(symmetry, device):
    key = (symmetry, str(device))
    if key not in _TABLES:
        _TABLES[key] = tuple(torch.from_numpy(t).to(device) for t in _tables(symmetry))
    return _TABLES[key]


def apply_symmetry_points(tensor, symmetry):
    """Point planes (..., y, x): the value at a point moves to the point's image."""
    if symmetry & 2:
        tensor = tensor.flip(-1)
    if symmetry & 1:
        tensor = tensor.flip(-2)
    if symmetry & 4:
        tensor = tensor.transpose(-2, -1)
    return tensor


# Offsets toward a lexicographically lower point: only an edge's upper end is stored in them.
DOWNWARD = torch.tensor([(dx, dy) < (0, 0) for dx, dy in OFFSETS])


def _permute_edges(values, symmetry, both_ends=None):
    """values (N, G, NUM_ACTIONS) indexed by (point, offset): the images' entries. One-end rows (an edge at its
    lower end) move by canonical edge; rows with both_ends (N,) set move entry by entry (forward)."""
    forward, _, edge, _ = tables(symmetry, values.device)
    flat = values.reshape(-1, NUM_ACTIONS)
    out = torch.zeros_like(flat)
    valid = edge >= 0
    out[:, edge[valid]] = flat[:, valid]
    if both_ends is not None and bool(both_ends.any()):
        rows = both_ends.repeat_interleave(values.shape[1])  # (N * G,)
        moved = torch.zeros_like(flat[rows])
        valid = forward >= 0
        moved[:, forward[valid]] = flat[rows][:, valid]
        out[rows] = moved
    return out.view(values.shape)


def apply_symmetry_inputs(binary_inputs, symmetry):
    """V8 spatial inputs (N, C, 19, 19)."""
    if symmetry == 0:
        return binary_inputs
    n = binary_inputs.shape[0]
    out = apply_symmetry_points(binary_inputs, symmetry).clone()
    edges = binary_inputs[:, EDGE_PLANES_START:FEATURE_PLANES_START]  # N, groups * offsets, y, x
    # N, group, offset, y, x -> N, group, (y, x, offset): the action index layout.
    edges = edges.reshape(n, NUM_EDGE_GROUPS, NUM_OFFSETS, SIZE * SIZE)
    both_ends = (edges[:, :, DOWNWARD.to(edges.device)] != 0).flatten(1).any(dim=1)  # (N,)
    edges = edges.permute(0, 1, 3, 2).reshape(n, NUM_EDGE_GROUPS, NUM_ACTIONS)
    edges = _permute_edges(edges, symmetry, both_ends)
    edges = edges.reshape(n, NUM_EDGE_GROUPS, SIZE * SIZE, NUM_OFFSETS).permute(0, 1, 3, 2).reshape(n, NUM_EDGE_GROUPS * NUM_OFFSETS, SIZE, SIZE)
    out[:, EDGE_PLANES_START:FEATURE_PLANES_START] = edges
    return out


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """Policy-shaped targets (N, C, NUM_ACTIONS + 1), pass last, for the positions of binary_inputs (before
    the symmetry). A move between two of the mover's nodes is listed from its lower end, so its image is
    listed from the image's lower end; any other move from the image of its source (the mover's node)."""
    if symmetry == 0:
        return policy
    n, c, _ = policy.shape
    forward, canonical, _, dest = tables(symmetry, policy.device)
    player = global_inputs[:, 0].round().long()  # 0 blue, 1 red
    node_planes = binary_inputs[:, FEATURE_PLANES_START:FEATURE_PLANES_START + 2 * FEATURES_PER_PLAYER:FEATURES_PER_PLAYER]
    mover_nodes = node_planes[torch.arange(n, device=policy.device), player].reshape(n, SIZE * SIZE) > 0.5  # N, points
    valid = forward >= 0
    dest_is_node = torch.zeros(n, NUM_ACTIONS, dtype=torch.bool, device=policy.device)
    dest_is_node[:, valid] = mover_nodes[:, dest[valid]]
    target = torch.where(dest_is_node, canonical, forward)  # N, actions; -1 where off the board
    moves = policy[:, :, :NUM_ACTIONS]
    index = torch.where(target >= 0, target, torch.full_like(target, NUM_ACTIONS))  # dump slot for off-board
    padded = torch.zeros(n, c, NUM_ACTIONS + 1, dtype=policy.dtype, device=policy.device)
    padded.scatter_add_(2, index.unsqueeze(1).expand(n, c, NUM_ACTIONS), moves)
    out = padded[:, :, :NUM_ACTIONS]
    return torch.cat((out, policy[:, :, NUM_ACTIONS:]), dim=2)
