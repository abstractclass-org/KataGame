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
blueprints after the row's move, and the row says who the next mover is (next_movers). The blueprints are the row's
own, except after a drop with a move of the same player to come: the drop changes two blueprints of the mover, and the
row does not say which drop was played. So in those rows the second policy target has the weight 0 (mask_targets), at
every symmetry.

The view has an orientation (the C++ gamerules.h): self-play can turn each position's or each game's view by 60 or
120 degrees, and the row is in its view's frame, the hand's planes too. So the augmentation does not read the
orientation, and its 4 symmetries on top of the views' 3 orientations give the hex grid's 12. The second policy target
is a policy of the next position's view. When that view has another orientation (the global targets G78 and G79), the
policy lies in a turned frame that the row does not give, so it has the weight 0 too. With one orientation a game, no
row has a next view of another orientation.

The extended inputs of the inputs version 10 (the C++ gameencoding.h, for bpj3): the mover's cells that the opponent's
placements in its own view take, and the minimap with each player's view on it, move as pixels (the minimap's bins of
the rules board map to bins). The legal hand actions move as the hand's points of the mover's blueprints. The window
boxes are global inputs: each box's two corners move as offsets by the symmetry's linear part, and the turn by 180
degrees and the second reflection (3 and 7) swap the corners (apply_symmetry_extended_global_inputs). The rows of the
older versions have none of these. The planes of closed areas of the inputs version 12 (the areas closed by each
player's walls) are points of the board, so they move as pixels too.

The value targets are points of the row's view, so they move as pixels: the final board (C0), the cells captured
before the game's end (C1), the stars 4 and 16 moves later (C2-3), where the points are earned (C4), and the cells
that go in the rest of the mover's turn (C5, the opponent's cells) and in the opponent's next turn (C6, the mover's
cells). The loss of C5-6 reads only the cells that can go: the row's cells of that player that are not walls
(income_mask). The loss of C1 reads only the row's cells that are not walls: the others hold the class 2, which it
leaves out. Each row trains the targets that its version has (G77, gameencoding.h): the rows of versions 0 and 1 have
the flag C68 and the weight C34 at 0, so no loss reads their C1 or C4.
"""

import torch

from . import (TARGET_FUTURE_PIECES, TARGET_INCOME, TARGET_OWNERSHIP, TARGET_SCORING, TARGET_SEKI,
               apply_symmetry_points)

# The members of the game's module (C++ game/blueprints/gameencoding.h).
NAME = "blueprints"
MAX_BOARD_LEN = 67       # the view: |q|, |r| <= 33 from the mover's window, which follows its play (gamerules.h)
POLICY_CHANNELS = 6      # place blueprint 1-3, then a cell of blueprint 1-3 (a pick, or a drop after a pick)
NUM_SPATIAL_INPUTS = 89  # 90 with the on-board plane
NUM_GLOBAL_INPUTS = 22   # the last, 21: the mover's effective margin (with the komi) / 1000
HISTORY_PLANE = 81       # the points that each of the last 4 actions (turns in V11) stamped and took, newest first
NUM_HISTORY_PLANES = 8
SYMMETRIES = (1 << 0) | (1 << 3) | (1 << 4) | (1 << 7)
# C0: the final board (+1 the mover's cells, -1 the opponent's). C1: the cells captured before the game's end. C2-3: the
# stars 4 and 16 moves later. C4: where the points are earned. C5-6: the cells that go in the rest of the mover's turn
# and in the opponent's next turn.
TARGETS = TARGET_OWNERSHIP | TARGET_SEKI | TARGET_FUTURE_PIECES | TARGET_SCORING | TARGET_INCOME
SCORE_SCALE = 20.0       # people's median |margin|, 409 points, / KataGo's score multiplier of 20 (gameencoding.h)
GAME_LENGTH = 132.0      # the actions of a game: 120 of the site's moves, and one more for each edit (C++)
# KataGo's spatial losses read the board's area as the area of play: nearly every one of Go's 361 points ends owned. The
# view's 4,489 hexes hold the window, and about 360 of them hold a cell at the game's end (bpj3's rows: a mean of 363,
# a median of 361). So the ownership (C0), the future positions (C2-3) and the scoring slot (C4) read 361 hexes, and
# every coefficient is KataGo's or Enclosure's (LOSS_COEFFICIENTS' defaults). C1 and C5-6 average over their own cells.
LOSS_AREA = 361.0

# The encoding's facts that the augmentation reads (gamerules.h, gameencoding.h). A plane's index in a batch is
# 1 + its index here: KataGo's on-board plane comes first.
VIEW_RADIUS, HAND, HAND_X, HAND_Y = 33, 3, 33, 1
MINE, THEIRS = 0, 3                # each player's cells, + 0 plain, + 1 star, + 2 wall
# A player's placements, + level * HAND + k for blueprint k: fits (level 0), takes at least 1 to 4 enemy stars (levels
# 1 to 4), scores at least 25, 100 and 400 points (levels 5 to 7). All of them are anchors.
MY_PLACES, THEIR_PLACES = 10, 34
STAR_LEVELS, NUM_POINT_LEVELS = 4, 3
PLACE_PLANES = (1 + STAR_LEVELS + NUM_POINT_LEVELS) * HAND
MY_HAND, THEIR_HAND = 60, 69       # + k: cells, + 3 + k: stars, + 6 + k: walls
PICKED = 78                        # + k: the picked cell of blueprint k
PICKED_INPUT = 9                   # the global input: 1 while a pick waits for its drop
NEXT_POLICY_WEIGHT = 28            # the global target: the weight of the second policy target (KataGo's C28)
# Who makes the next move (+1 the row's mover, -1 the opponent, 0 none): the global target G76 in the rows of the
# targets' version 1 or later (G77), and the value targets' slot C1 in older rows, on every point. From version 2, C1
# holds the cells captured before the game's end.
NEXT_MOVER_TARGET, TARGETS_VERSION_TARGET, NEXT_MOVER_SLOT = 76, 77, 1
# The orientations of the row's view and of the next position's view (0 to 5, both 0 in the rows from before the
# orientations): the global targets G78 and G79.
ORIENTATION_TARGET, NEXT_ORIENTATION_TARGET = 78, 79
# The extended inputs (the inputs version 10): the planes after the game's 89, and the global inputs after V9's.
NUM_EXTENDED_SPATIAL_INPUTS = 12
NUM_EXTENDED_GLOBAL_INPUTS = 20
MINE_THREATENED = 89               # the mover's cells that the opponent's placements in its own view take
LEGAL_HAND = 90                    # + k: the hand's points of blueprint k where a pick or a drop is legal now
MINIMAP = 93                       # + 3 * side + type (side 0 the mover's): the minimap's bins that hold such a cell
MY_WINDOW, THEIR_WINDOW = 99, 100  # the minimap's bins that each player's view meets
BOX_FIELDS = 5                     # a window box: 1 when there, then its corners' x0, x1, y0, y1 / 67
# The turn inputs (the inputs version 11): after the extended global inputs, for each of the last 4 turns (the newest
# first, the turn in progress) and each of its 2 moves in their order, 1 when the move was an edit. The history planes
# then show the last turns in place of the last actions: turn h's in the planes of action h.
NUM_TURN_GLOBAL_INPUTS = 8
NUM_HISTORY_TURNS = 4
# The planes of closed areas (the inputs version 12), after the extended planes: the hexes of the areas closed by the
# mover's walls and by the opponent's walls. They move as pixels.
NUM_AREA_SPATIAL_INPUTS = 2
MY_CLOSED, THEIR_CLOSED = 101, 102
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


def _hand_image(x, y, symmetry, mq, mr):
    """The image of hand points (x, y), each of a blueprint whose shift is (mq, mr): the linear part, then the shift."""
    q, r = _linear(x - HAND_X, y - HAND_Y, symmetry)
    return q - mq + HAND_X, r - mr + HAND_Y


def _anchor_image(x, y, symmetry, mq, mr):
    """The image of placement anchors (x, y), each of a blueprint whose shift is (mq, mr): the pixel, then the shift."""
    x, y = _pixel(x, y, symmetry)
    return x + mq, y + mr


def _inside(x, y):
    """Whether the points (x, y) are on the board."""
    return (x >= 0) & (x < MAX_BOARD_LEN) & (y >= 0) & (y < MAX_BOARD_LEN)


def _remap_planes(out, inputs, first, count, image, symmetry, shifts):
    """Writes the image of the planes ``first`` to ``first + count - 1`` of ``inputs`` into ``out``, where they were
    pixels: ``image`` (_hand_image or _anchor_image) moves each point by the ``shifts`` of its row and of its plane's
    blueprint (the plane's offset modulo HAND)."""
    block = inputs[:, 1 + first:1 + first + count]
    out[:, 1 + first:1 + first + count] = 0
    row, j, y, x = (block > 0.5).nonzero(as_tuple=True)
    k = j % HAND
    nx, ny = image(x, y, symmetry, shifts[0][row, k], shifts[1][row, k])
    inside = _inside(nx, ny)
    out[row[inside], 1 + first + j[inside], ny[inside], nx[inside]] = block[row, j, y, x][inside]


def apply_symmetry_inputs(binary_inputs, symmetry):
    """The input planes (N, C, y, x), with the extended planes in the rows of the inputs versions 10 to 12, and the
    planes of closed areas, pixels, in the rows of version 12."""
    if symmetry == 0:
        return binary_inputs
    out = apply_symmetry_points(binary_inputs, symmetry).clone()
    mine = _shifts(binary_inputs, symmetry, MY_HAND, PICKED)
    theirs = _shifts(binary_inputs, symmetry, THEIR_HAND)
    _remap_planes(out, binary_inputs, MY_HAND, 3 * HAND, _hand_image, symmetry, mine)
    _remap_planes(out, binary_inputs, THEIR_HAND, 3 * HAND, _hand_image, symmetry, theirs)
    _remap_planes(out, binary_inputs, PICKED, HAND, _hand_image, symmetry, mine)
    _remap_planes(out, binary_inputs, MY_PLACES, PLACE_PLANES, _anchor_image, symmetry, mine)
    _remap_planes(out, binary_inputs, THEIR_PLACES, PLACE_PLANES, _anchor_image, symmetry, theirs)
    if binary_inputs.shape[1] > 1 + NUM_SPATIAL_INPUTS:
        _remap_planes(out, binary_inputs, LEGAL_HAND, HAND, _hand_image, symmetry, mine)
    return out


def apply_symmetry_extended_global_inputs(extended, symmetry):
    """The window boxes (N, NUM_EXTENDED_GLOBAL_INPUTS) at the symmetry: each box's corners (x0, y0) and (x1, y1) move
    by the linear part (C++ sym_offset), and the symmetries that flip both axes (3 and 7) swap the corners, as the
    image's box has the images of the other corners first (C++ box_inputs)."""
    if symmetry == 0:
        return extended
    boxes = extended.reshape(extended.shape[0], -1, BOX_FIELDS)
    x0, y0 = _linear(boxes[:, :, 1], boxes[:, :, 3], symmetry)
    x1, y1 = _linear(boxes[:, :, 2], boxes[:, :, 4], symmetry)
    if symmetry & 3 == 3:
        x0, y0, x1, y1 = x1, y1, x0, y0
    out = torch.stack((boxes[:, :, 0], x0, x1, y0, y1), dim=2)
    return out.reshape(extended.shape)


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
    ax, ay = _anchor_image(x, y, symmetry, mq, mr)
    hx, hy = _hand_image(x, y, symmetry, mq, mr)
    place = action < HAND
    nx = torch.where(place, ax, hx)
    ny = torch.where(place, ay, hy)
    inside = _inside(nx, ny)
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


def next_movers(value_targets, global_targets):
    """Who makes each row's next move (N,): +1 the row's mover, -1 the opponent, 0 none (C++ training_targets; no loss
    reads it). G76 in the rows of the targets' version 1 or later (G77), the slot C1 in older rows. Torch tensors or
    numpy arrays."""
    new = global_targets[:, TARGETS_VERSION_TARGET] >= 1
    return new * global_targets[:, NEXT_MOVER_TARGET] + ~new * value_targets[:, NEXT_MOVER_SLOT, 0, 0]


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    """The policy targets (N, 2, policy): channel 0 the mover's move, channel 1 the next move, with the frames of the
    player that next_movers names."""
    if symmetry == 0:
        return policy
    mine = _shifts(binary_inputs, symmetry, MY_HAND, PICKED)
    theirs = _shifts(binary_inputs, symmetry, THEIR_HAND)
    same_next = next_movers(value_targets, global_targets) > 0
    same = torch.stack((torch.ones_like(same_next), same_next), dim=1)
    return _map_policy(policy, symmetry, mine, theirs, same)


def mask_targets(binary_inputs, global_inputs, value_targets, global_targets):
    """The rows' global targets (N, 80): the weight of the second policy target is 0 in the rows of a drop (a pick
    waits for its drop) with a move of the same player to come, and in the rows whose next position's view has another
    orientation: the row does not give that target's frames (the module's docstring). That weight also weighs the
    target's soft loss."""
    unmapped = (global_inputs[:, PICKED_INPUT] > 0.5) & (next_movers(value_targets, global_targets) > 0)
    unmapped |= global_targets[:, ORIENTATION_TARGET] != global_targets[:, NEXT_ORIENTATION_TARGET]
    out = global_targets.clone()
    out[:, NEXT_POLICY_WEIGHT].masked_fill_(unmapped, 0.0)
    return out


def income_mask(binary_inputs):
    """The points where the loss of C5-6 reads the target (N, 2, y, x): the row's cells that can go, not walls. C5 the
    opponent's, C6 the mover's (the input planes of the cells, which move as pixels)."""
    planes = binary_inputs[:, 1:]
    return torch.stack((planes[:, THEIRS] + planes[:, THEIRS + 1], planes[:, MINE] + planes[:, MINE + 1]), dim=1)
