"""Blueprints' module for the trainer (katago/train/games/__init__.py lists the members): its facts, as the C++
encoding part gives them (game/blueprints/gameencoding.h), then the augmentation of its training rows.

The game allows 4 of KataGo's symmetries: the identity, the turn by 180 degrees (3) and two reflections (4, 7), the
ones that keep the hex neighbours on the axial square. A point of the board moves as a pixel. A blueprint moves cell
by cell (the symmetry's linear part on axial offsets) and then goes back to the site's frame, where its lowest cell
(the least r, then the least q) is at (0, 0). So:

- the hand's planes and the picked cell's planes move by the linear part and then by their blueprint's shift;
- a placement's anchor (its (0, 0) cell) moves as a pixel and then by its blueprint's shift: the image's anchor is the
  lowest image of the stamped cells;
- the policy's cells of the hand (picks, and drops after a pick) move as the hand's points, and its placements as the
  anchors.

Each row's shifts come from its own hand planes: the mover's blueprints (with the picked cell, whose blueprint keeps
its whole frame until the drop) and the opponent's. The second policy target (the next move) needs the next mover's
blueprints after the row's move, and the row's value targets say who the next mover is (slot C1). The blueprints are
the row's own, except after a drop with a move of the same player to come: the drop changes two blueprints of the
mover, and the row does not say which drop was played. Those rows map with the row's frames, an error in that target
only.
"""

import torch

from . import TARGET_OWNERSHIP, apply_symmetry_points

# The members of the game's module (C++ game/blueprints/gameencoding.h).
NAME = "blueprints"
MAX_BOARD_LEN = 61       # the axial square |q|, |r| <= 30 from its middle, which follows the game
POLICY_CHANNELS = 6      # place blueprint 1-3, then a cell of blueprint 1-3 (a pick, or a drop after a pick)
NUM_SPATIAL_INPUTS = 59  # 60 with the on-board plane
NUM_GLOBAL_INPUTS = 17
HISTORY_PLANE = 51       # the points that each of the last 4 actions stamped and took, newest first
NUM_HISTORY_PLANES = 8
SYMMETRIES = (1 << 0) | (1 << 3) | (1 << 4) | (1 << 7)
TARGETS = TARGET_OWNERSHIP   # C0: the final board (+1 the mover's cells, -1 the opponent's)
SCORE_SCALE = 12.0
GAME_LENGTH = 120.0      # the site's moves of a game

# The encoding's facts that the augmentation reads (gamerules.h, gameencoding.h). A plane's index in a batch is
# 1 + its index here: KataGo's on-board plane comes first.
RADIUS, HAND, HAND_X, HAND_Y = 30, 3, 30, 1
MY_PLACES, THEIR_PLACES = 10, 19   # + k: fits, + 3 + k: takes 1+ stars, + 6 + k: takes 3+ stars
MY_HAND, THEIR_HAND = 30, 39       # + k: cells, + 3 + k: stars, + 6 + k: walls
PICKED = 48                        # + k: the picked cell of blueprint k
NEXT_MOVER_SLOT = 1                # the value targets' slot that says who moves next (+1 the row's mover, -1 not)
_KEY = 1000                        # a cell's (r, q) as one integer, for the lowest cell: (r + 100) * _KEY + q + 100


def _linear(q, r, symmetry):
    """The symmetry's linear part on axial offsets (C++ sym_offset): flip x, flip y, then transpose."""
    if symmetry & 2:
        q = -q
    if symmetry & 1:
        r = -r
    if symmetry & 4:
        q, r = r, q
    return q, r


def _pixel(x, y, symmetry):
    """A board point's image (C++ GameInterface::symPoint, apply_symmetry_points)."""
    last = MAX_BOARD_LEN - 1
    if symmetry & 2:
        x = last - x
    if symmetry & 1:
        y = last - y
    if symmetry & 4:
        x, y = y, x
    return x, y


def _shifts(inputs, symmetry, hand, picked=None):
    """Each row's shift of each blueprint, (q, r) tensors (N, HAND): the lowest cell of the image of the blueprint's
    cells on the hand planes ``hand`` (with the picked cell's planes ``picked``), (0, 0) for an empty blueprint."""
    n = inputs.shape[0]
    planes = [inputs[:, 1 + hand:1 + hand + HAND]]
    if picked is not None:
        planes.append(inputs[:, 1 + picked:1 + picked + HAND])
    keys = torch.full((n * HAND,), 10**9, dtype=torch.long, device=inputs.device)
    for block in planes:
        row, k, y, x = (block > 0.5).nonzero(as_tuple=True)
        q, r = _linear(x - HAND_X, y - HAND_Y, symmetry)
        keys.scatter_reduce_(0, row * HAND + k, (r + 100) * _KEY + q + 100, reduce="amin")
    empty = keys == 10**9
    mq = torch.where(empty, 0, keys % _KEY - 100).view(n, HAND)
    mr = torch.where(empty, 0, keys // _KEY - 100).view(n, HAND)
    return mq, mr


def _hand_image(x, y, k, row, symmetry, mq, mr):
    """The image of hand points (x, y) of blueprint k in rows ``row``: the linear part, then the blueprint's shift."""
    q, r = _linear(x - HAND_X, y - HAND_Y, symmetry)
    return q - mq[row, k] + HAND_X, r - mr[row, k] + HAND_Y


def _anchor_image(x, y, k, row, symmetry, mq, mr):
    """The image of placement anchors (x, y) of blueprint k in rows ``row``: the pixel, then the blueprint's shift."""
    x, y = _pixel(x, y, symmetry)
    return x + mq[row, k], y + mr[row, k]


def _remap_planes(out, inputs, first, count, image):
    """Writes the image of the planes ``first`` to ``first + count - 1`` of ``inputs`` into ``out``, where they were
    pixels: image(x, y, k, row) gives each point's image, k the plane's blueprint (its offset modulo HAND)."""
    block = inputs[:, 1 + first:1 + first + count]
    out[:, 1 + first:1 + first + count] = 0
    row, j, y, x = (block > 0.5).nonzero(as_tuple=True)
    nx, ny = image(x, y, j % HAND, row)
    inside = (nx >= 0) & (nx < MAX_BOARD_LEN) & (ny >= 0) & (ny < MAX_BOARD_LEN)
    out[row[inside], 1 + first + j[inside], ny[inside], nx[inside]] = block[row, j, y, x][inside]


def apply_symmetry_inputs(binary_inputs, symmetry):
    """The input planes (N, C, y, x)."""
    if symmetry == 0:
        return binary_inputs
    out = apply_symmetry_points(binary_inputs, symmetry).clone()
    mine = _shifts(binary_inputs, symmetry, MY_HAND, PICKED)
    theirs = _shifts(binary_inputs, symmetry, THEIR_HAND)
    hand = lambda m: (lambda x, y, k, row: _hand_image(x, y, k, row, symmetry, *m))
    anchor = lambda m: (lambda x, y, k, row: _anchor_image(x, y, k, row, symmetry, *m))
    _remap_planes(out, binary_inputs, MY_HAND, 3 * HAND, hand(mine))
    _remap_planes(out, binary_inputs, THEIR_HAND, 3 * HAND, hand(theirs))
    _remap_planes(out, binary_inputs, PICKED, HAND, hand(mine))
    _remap_planes(out, binary_inputs, MY_PLACES, 3 * HAND, anchor(mine))
    _remap_planes(out, binary_inputs, THEIR_PLACES, 3 * HAND, anchor(theirs))
    return out


def _map_policy(policy, symmetry, mine, theirs, same):
    """Policy-shaped tensors (N, C, MAX_BOARD_LEN^2 * POLICY_CHANNELS + 1) at the symmetry: channel c of row n moves
    with the mover's shifts ``mine`` where same[n, c], else with the opponent's ``theirs``. Pass stays last."""
    n, c, size = policy.shape
    points = policy[:, :, :-1].reshape(n, c, MAX_BOARD_LEN, MAX_BOARD_LEN, POLICY_CHANNELS)
    out = torch.zeros_like(points)
    row, chan, y, x, action = points.nonzero(as_tuple=True)
    values = points[row, chan, y, x, action]
    k = action % HAND
    use_mine = same[row, chan]
    mq = torch.where(use_mine, mine[0][row, k], theirs[0][row, k])
    mr = torch.where(use_mine, mine[1][row, k], theirs[1][row, k])
    ax, ay = _pixel(x, y, symmetry)
    hq, hr = _linear(x - HAND_X, y - HAND_Y, symmetry)
    place = action < HAND
    nx = torch.where(place, ax + mq, hq - mq + HAND_X)
    ny = torch.where(place, ay + mr, hr - mr + HAND_Y)
    inside = (nx >= 0) & (nx < MAX_BOARD_LEN) & (ny >= 0) & (ny < MAX_BOARD_LEN)
    out.index_put_((row[inside], chan[inside], ny[inside], nx[inside], action[inside]), values[inside],
                   accumulate=True)
    return torch.cat((out.reshape(n, c, -1), policy[:, :, -1:]), dim=2)


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """A policy of the mover's moves (N, C, policy): the q-value targets."""
    if symmetry == 0:
        return policy
    mine = _shifts(binary_inputs, symmetry, MY_HAND, PICKED)
    theirs = _shifts(binary_inputs, symmetry, THEIR_HAND)
    same = torch.ones(policy.shape[:2], dtype=torch.bool, device=policy.device)
    return _map_policy(policy, symmetry, mine, theirs, same)


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    """The policy targets (N, 2, policy): channel 0 the mover's move, channel 1 the next move.

    The row says who makes the next move: the value targets' slot C1 holds +1 when it is the row's mover and -1 when
    it is the opponent (C++ training_targets, NEXT_MOVER_SLOT; no loss reads the slot)."""
    if symmetry == 0:
        return policy
    mine = _shifts(binary_inputs, symmetry, MY_HAND, PICKED)
    theirs = _shifts(binary_inputs, symmetry, THEIR_HAND)
    same_next = value_targets[:, NEXT_MOVER_SLOT, 0, 0] > 0
    same = torch.stack((torch.ones_like(same_next), same_next), dim=1)
    return _map_policy(policy, symmetry, mine, theirs, same)
