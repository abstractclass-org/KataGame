"""The trainer's games: one module for each game, katago/train/games/<name>.py, with the game's facts that the model
and data code read. A model config names its game ("game"). A config without one is Enclosure's, as every model before
the game interface (design 07 of the engine repository).

Each module mirrors the members of the C++ encoding part that the trainer needs (KataGame's GAMES.md, "The network
encoding part"), with the same names and meanings:

  NAME                  the game's name, as the C++ build option GAME gives it.
  MAX_BOARD_LEN         the network's board: the trainer's pos_len.
  POLICY_CHANNELS       policy channels per point. A policy has MAX_BOARD_LEN^2 * POLICY_CHANNELS entries, then pass.
  NUM_SPATIAL_INPUTS    the input planes that the game writes. KataGo's on-board plane comes first, so the network
                        reads 1 + NUM_SPATIAL_INPUTS planes.
  NUM_GLOBAL_INPUTS     the global inputs that the game writes. The inputs version 9 adds KataGo's 2 playout inputs
                        after them (modelconfigs.get_num_global_input_features).
  HISTORY_PLANE,        the history planes, the game's planes HISTORY_PLANE to HISTORY_PLANE + NUM_HISTORY_PLANES - 1
  NUM_HISTORY_PLANES    (0 is the plane after the on-board plane), newest first: the history dropout clears them
                        (data_processing_pytorch.apply_history_dropout). A game without history has 0 planes.
  SYMMETRIES            the allowed subset of KataGo's 8 symmetries: bit s for symmetry s, the identity included.
  TARGETS               the optional training targets that the game writes (the TARGET_ bits below).
  SCORE_SCALE           the game's score spread against Go's. KataGo's constants in points scale by it, and the
                        score distribution's bins are SCORE_SCALE points wide.
  GAME_LENGTH           a game's mean length in actions (the front's moves), against Go's GO_GAME_LENGTH
                        (model_pytorch).

and the augmentation, for each allowed symmetry (KataGo's numbers: bit 0 flips y, bit 1 flips x, bit 2 transposes):

  apply_symmetry_inputs(binary_inputs, symmetry)                              the input planes (N, C, y, x).
  apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets)
                        the policy targets (N, 2, policy): channel 0 the mover's move, channel 1 the next move. The
                        other arguments are the row's, before the symmetry.
  apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs)       a policy of the mover's moves (the
                        q-value targets).

The value targets are planes of points, so every game maps them with apply_symmetry_points.

A module may also give these optional members (mask_targets and loss_coefficients below give the defaults):

  mask_targets(binary_inputs, global_inputs, value_targets, global_targets)
                        the rows' global targets (N, 80), with weight 0 for each target that the game's augmentation
                        cannot map in a row. The arguments are the rows' as read, before any symmetry. The trainer
                        applies it to every batch, at every symmetry and without the symmetries, so that the weights do
                        not depend on the draw. Default: the global targets as they are.
  LOSS_COEFFICIENTS     a dict: the coefficient of each of the trainer's losses that it names, of LOSS_COEFFICIENTS
                        below. Default: those of LOSS_COEFFICIENTS below.
  LOSS_AREA             the area, in points, that KataGo's spatial losses read where KataGo reads the board's area
                        (metrics_pytorch): the ownership loss and the scoring slot's loss sum over the row's points and
                        divide by it, the future positions' loss by its square root, and the income loss without an
                        income_mask by twice it. For a game whose board is much larger than the points where its play
                        is. Default: the row's on-board points, as KataGo's.
  income_mask(binary_inputs)
                        the points where the loss of C5-6 reads the target (N, 2, y, x), from the batch's input planes
                        at its symmetry: the loss averages each channel over its points, then over the channels that
                        have points. Default: none, and the loss averages over the board's points and both channels.
  NUM_EXTENDED_SPATIAL_INPUTS, NUM_EXTENDED_GLOBAL_INPUTS
                        the game's extended inputs (C++ GameInterface), which the inputs version 10 adds: its extended
                        planes after its planes, and its extended global inputs after the inputs version 9's (KataGo's 2
                        playout inputs). apply_symmetry_inputs maps the extended planes too, in the rows that have them.
                        Default: 0 and 0 (extended_spatial_inputs, extended_global_inputs).
  apply_symmetry_extended_global_inputs(extended, symmetry)
                        the extended global inputs (N, NUM_EXTENDED_GLOBAL_INPUTS) at the symmetry, for a game whose
                        extended global inputs name points (apply_symmetry_global_inputs). Default: as they are.
  NUM_TURN_GLOBAL_INPUTS, NUM_HISTORY_TURNS
                        the game's turn inputs (C++ GameInterface), which the inputs version 11 adds after version 10's
                        global inputs, and the turns that they and the history planes then show in place of the last
                        actions, the newest first. Each turn has NUM_HISTORY_PLANES // NUM_HISTORY_TURNS planes and
                        NUM_TURN_GLOBAL_INPUTS // NUM_HISTORY_TURNS turn inputs, in its order. The turn inputs name no
                        point. In the rows of the inputs version 11 the history dropout draws once for each turn, and
                        clears the turn's planes and turn inputs together
                        (data_processing_pytorch.apply_history_dropout). A game with turn inputs gives
                        NUM_HISTORY_TURNS, which divides both counts. Default: 0 and none (turn_global_inputs).
"""

import importlib

DEFAULT = "enclosure"

# The optional training targets: the slots of valueTargetsNCHW that a game may write (C++ GameTarget, gameinterface.h).
# The rows hold every slot, C0, C4 and C5-6 stored x 120 (C++ TargetValues). A slot that the game does not write has
# the weight 0 in the rows, by its weight or its flag, but for C0: KataGo's C27 weighs it with the score's targets, so
# the trainer leaves out C0 itself (metrics_pytorch).
TARGET_OWNERSHIP = 1 << 0      # C0, its weight C27
TARGET_SEKI = 1 << 1           # C1: a class per point (0, +1 the mover's, -1 the opponent's, 2 none), flagged in C68
TARGET_FUTURE_PIECES = 1 << 2  # C2-3, its weight C33
TARGET_SCORING = 1 << 3        # C4, its weight C34
TARGET_INCOME = 1 << 4         # C5-6, flagged in C69
TARGET_NEXT_PIECES = 1 << 5    # C7, flagged in C75: the next mover's pieces after the row's move (not a loss)

# The trainer's losses whose coefficients a game's LOSS_COEFFICIENTS can set, and their defaults (metrics_pytorch).
LOSS_COEFFICIENTS = {
    "ownership": 1.5,  # C0, KataGo's
    "seki": 1.0,       # C1, with the classes of Enclosure's edges that get cut: Enclosure's
    "income": 1.5,     # C5-6, Enclosure's
}

# A training row as the C++ row writer lays it out (dataio/trainingwrite.cpp, TrainingWriteBuffers): the channels of
# its arrays that do not depend on the game. The shuffle holds rows uncompressed, so a row's bytes bound its memory.
ROW_POLICY_TARGETS = 2           # policyTargetsNCMove, int16 a move (POLICY_TARGET_NUM_CHANNELS)
ROW_GLOBAL_TARGETS = 80          # globalTargetsNC, float32 (GLOBAL_TARGET_NUM_CHANNELS)
ROW_EXTRA_SCORE_RADIUS = 60      # scoreDistrN, int8: 2 * (points + 60) bins (NNPos::EXTRA_SCORE_DISTR_RADIUS)
ROW_VALUE_TARGETS = 8            # valueTargetsNCHW, int8 a point (GameInterface::TargetValues::NUM_CHANNELS)
ROW_QVALUE_TARGETS = 3           # qValueTargetsNCMove, int16 a move (QVALUE_SPATIAL_TARGET_NUM_CHANNELS)
ROW_PLAYOUT_INPUTS = 2           # the global inputs that the inputs version 9 adds (modelconfigs.PLAYOUT_GLOBAL_INPUTS)
# KataGo's row of Go on 19 x 19, as its shuffle.py measures it (UNCOMPRESSED_BYTES_PER_ROW_REQUIRED_19 and
# UNCOMPRESSED_BYTES_PER_ROW_QVALUES_19): the size that upstream's row counts are for.
GO_ROW_BYTES = 7675


def extended_spatial_inputs(game):
    """The game's extended planes, which the inputs version 10 adds (0 for a game without them)."""
    return getattr(game, "NUM_EXTENDED_SPATIAL_INPUTS", 0)


def extended_global_inputs(game):
    """The game's extended global inputs, which the inputs version 10 adds after version 9's (0 for a game without)."""
    return getattr(game, "NUM_EXTENDED_GLOBAL_INPUTS", 0)


def turn_global_inputs(game):
    """The game's turn inputs, which the inputs version 11 adds after version 10's (0 for a game without them)."""
    return getattr(game, "NUM_TURN_GLOBAL_INPUTS", 0)


def row_bytes(game, inputs_version=9):
    """The bytes of one training row of the game, uncompressed: the row writer's arrays (ROW_ above) on the game's
    largest board, without the metadata inputs, which the shuffle leaves out. Self-play writes rows of the inputs
    version 9 by default (C++ NNModelVersion::defaultModelVersion), whatever its network reads, and of version 10 or 11
    with its config's inputsVersion."""
    points = game.MAX_BOARD_LEN ** 2
    moves = points * game.POLICY_CHANNELS + 1  # the policy: every point's channels, then pass
    global_inputs = game.NUM_GLOBAL_INPUTS + (ROW_PLAYOUT_INPUTS if inputs_version >= 9 else 0)
    planes = 1 + game.NUM_SPATIAL_INPUTS
    if inputs_version >= 10:
        global_inputs += extended_global_inputs(game)
        planes += extended_spatial_inputs(game)
    if inputs_version >= 11:
        global_inputs += turn_global_inputs(game)
    return (
        # binaryInputNCHWPacked: KataGo's on-board plane, then the game's planes, a bit a point
        planes * ((points + 7) // 8)
        + 4 * global_inputs                                # globalInputNC
        + 2 * ROW_POLICY_TARGETS * moves
        + 4 * ROW_GLOBAL_TARGETS
        + 2 * (points + ROW_EXTRA_SCORE_RADIUS)
        + ROW_VALUE_TARGETS * points
        + 2 * ROW_QVALUE_TARGETS * moves
    )


MEMBERS = (
    "NAME", "MAX_BOARD_LEN", "POLICY_CHANNELS", "NUM_SPATIAL_INPUTS", "NUM_GLOBAL_INPUTS", "HISTORY_PLANE",
    "NUM_HISTORY_PLANES", "SYMMETRIES", "TARGETS", "SCORE_SCALE", "GAME_LENGTH",
    "apply_symmetry_inputs", "apply_symmetry_policy_targets", "apply_symmetry_policy",
)


def get(name: str):
    """The trainer's module of the game NAME. Raises ValueError for a game without one, or a module that lacks a
    member."""
    try:
        game = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{name}":
            raise
        raise ValueError(f"the trainer has no module for the game {name!r} (katago/train/games/{name}.py)") from None
    missing = [m for m in MEMBERS if not hasattr(game, m)]
    if missing:
        raise ValueError(f"the trainer's module of the game {name!r} lacks {', '.join(missing)}")
    if game.NAME != name:
        raise ValueError(f"the trainer's module katago/train/games/{name}.py names the game {game.NAME!r}")
    unknown = sorted(set(getattr(game, "LOSS_COEFFICIENTS", {})) - set(LOSS_COEFFICIENTS))
    if unknown:
        raise ValueError(f"the trainer's module of the game {name!r} sets the coefficient of no loss named "
                         f"{', '.join(unknown)} (LOSS_COEFFICIENTS: {', '.join(LOSS_COEFFICIENTS)})")
    turns = getattr(game, "NUM_HISTORY_TURNS", 0)
    if turn_global_inputs(game) and not (turns > 0 and game.NUM_HISTORY_PLANES % turns == 0 and
                                         turn_global_inputs(game) % turns == 0):
        raise ValueError(f"the trainer's module of the game {name!r} has turn inputs, so it gives NUM_HISTORY_TURNS, "
                         "which divides NUM_HISTORY_PLANES and NUM_TURN_GLOBAL_INPUTS")
    return game


def symmetries(game):
    """The game's allowed symmetries, ascending."""
    return [s for s in range(8) if game.SYMMETRIES & (1 << s)]


def mask_targets(game, binary_inputs, global_inputs, value_targets, global_targets):
    """The rows' global targets after the game's mask_targets, or as they are for a game without it."""
    mask = getattr(game, "mask_targets", None)
    if mask is None:
        return global_targets
    return mask(binary_inputs, global_inputs, value_targets, global_targets)


def apply_symmetry_global_inputs(game, global_inputs, symmetry):
    """The rows' global inputs (N, C) at the symmetry: in the rows of the inputs versions 10 and 11, the game's extended
    global inputs (after version 9's) through its apply_symmetry_extended_global_inputs. The other global inputs name no
    point, so they stay, as do the rows of the older versions and of a game without the member."""
    mapping = getattr(game, "apply_symmetry_extended_global_inputs", None)
    extended = extended_global_inputs(game)
    if mapping is None or symmetry == 0 or extended == 0:
        return global_inputs
    start = game.NUM_GLOBAL_INPUTS + ROW_PLAYOUT_INPUTS
    if global_inputs.shape[1] not in (start + extended, start + extended + turn_global_inputs(game)):
        return global_inputs
    out = global_inputs.clone()
    out[:, start:start + extended] = mapping(global_inputs[:, start:start + extended], symmetry)
    return out


def loss_coefficients(game):
    """The game's coefficient of each loss of LOSS_COEFFICIENTS: its LOSS_COEFFICIENTS, and the default for each loss
    that it does not name."""
    return {**LOSS_COEFFICIENTS, **getattr(game, "LOSS_COEFFICIENTS", {})}


def loss_area(game):
    """The game's LOSS_AREA, or None for a game whose spatial losses read the row's on-board points."""
    return getattr(game, "LOSS_AREA", None)


def apply_symmetry_points(tensor, symmetry):
    """Planes of points (..., y, x): the value at a point moves to the point's image."""
    if symmetry & 2:
        tensor = tensor.flip(-1)
    if symmetry & 1:
        tensor = tensor.flip(-2)
    if symmetry & 4:
        tensor = tensor.transpose(-2, -1)
    return tensor
