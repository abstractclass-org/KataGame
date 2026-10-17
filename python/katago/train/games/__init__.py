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
  GAME_LENGTH           a game's mean length in moves, against Go's GO_GAME_LENGTH (model_pytorch).

and the augmentation, for each allowed symmetry (KataGo's numbers: bit 0 flips y, bit 1 flips x, bit 2 transposes):

  apply_symmetry_inputs(binary_inputs, symmetry)                              the input planes (N, C, y, x).
  apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets)
                        the policy targets (N, 2, policy): channel 0 the mover's move, channel 1 the next move. The
                        other arguments are the row's, before the symmetry.
  apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs)       a policy of the mover's moves (the
                        q-value targets).

The value targets are planes of points, so every game maps them with apply_symmetry_points.
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
    return game


def symmetries(game):
    """The game's allowed symmetries, ascending."""
    return [s for s in range(8) if game.SYMMETRIES & (1 << s)]


def apply_symmetry_points(tensor, symmetry):
    """Planes of points (..., y, x): the value at a point moves to the point's image."""
    if symmetry & 2:
        tensor = tensor.flip(-1)
    if symmetry & 1:
        tensor = tensor.flip(-2)
    if symmetry & 4:
        tensor = tensor.transpose(-2, -1)
    return tensor
