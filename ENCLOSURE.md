# KataGo for Enclosure: the fork's design

`katago/` (upstream `d91ea85` plus our changes) turns KataGo into an
Enclosure bot. The plan, its build order and the approved departures are in
`docs/katago-fork-plan.md` of this repository. This file records the
design decisions inside the code.

## Principle

KataGo's methods, parameters and structure stay as they are. The game layer
(`cpp/game/`) is replaced. Every other file changes only where it relies on
Go: moves as points, alternating turns, Go's input features, and the policy
size. Go-only concepts (ko, superko, encore phases, komi, handicap, the
button, pass-alive areas) keep their names and signatures, fixed at their
neutral values. Callers therefore compile unchanged, and nothing in them
branches on Enclosure.

## Moves and points (step 2)

`Loc` stays a 16-bit `short`, but it now names two different things:

- **A point**, where KataGo indexes the board: ownership, areas, network
  input and output planes, symmetries. It keeps KataGo's layout,
  `Location::getLoc(x, y, 19)`, within `Board::MAX_ARR_SIZE` (421). Only
  the game layer and the network input/output code handle points.
- **A move**: `NULL_LOC` (0), `PASS_LOC` (1), or `MOVE_LOC_BASE` (421,
  = `Board::MAX_ARR_SIZE`) + our action index `(y * 19 + x) * 48 + offset`.
  Offsets follow the Enclosure engine's order: dx, then dy, each from -3
  to 3, without (0, 0). So a move `Loc` runs from 421 to 17,748.
  - Moves start after the points, so no value is both (owner, 2026-09-29;
    see "Moves where points are expected").
  - Everything that stores, compares, hashes or passes moves (the search
    tree, histories, training rows) is unchanged.
  - `NNPos::locToPos` maps a move to its policy index, the action index,
    with pass last at 17,328.

A `Move` is still `(Loc, Player)`.

## Board and Rules (step 2)

- **`Rules` is unchanged.** It is plain data (ko, scoring and tax rules,
  komi), and Enclosure reads none of it. Configurations set komi to 0.
- **`Board` holds an `enclosure::Position`** (`game/enclosure/engine.h`,
  generated from the Enclosure repository's verified native engine by its
  `tools/extract_katago_engine.py`).
  - Callers outside `game/` use a few members (survey 2026-09-29): the size
    (`x_size`, `y_size` = 19), `colors` (74 uses), `pos_hash`, `ko_loc`,
    and Go stone logic (`getNumLiberties`, `isSimpleEye`, captures).
  - `colors[point]` = `C_BLACK` / `C_WHITE` where the point is one of Blue's
    / Red's nodes, else `C_EMPTY`: display and "whose point is this" keep
    working.
  - `pos_hash`: Zobrist over edges, protected and placed edges, scores and
    the player to move (what the engine's `same_position` compares).
  - `ko_loc` = `NULL_LOC`.
  - The Go stone logic stays declared with neutral answers. Each caller is
    reviewed at steps 3-4 (most are the Go input features, replaced
    anyway).

## Players and turns

Blue moves first and is `P_BLACK`; Red is `P_WHITE`. The turn order is
1-2-2 (blue's single opening edge, then turns of two edges). The next player
therefore comes from the history (`BoardHistory::presumedNextMovePla`),
never from `getOpp` of the last mover. Every `getOpp` on the move sequence in
`search/` (29 places) is changed to ask the position (step 3).

## Turn order in the search (step 3)

KataGo already fits two-edge turns. Each `SearchNode` stores its own
`nextPla`, and statistics are kept from White's side and read with the
chooser's sign, so nothing negates per ply. What breaks is only where the
next player is *derived* from the last mover. The survey (2026-09-29) of
the 29 `getOpp` calls in `search/`:

- **Next player after a move: change to `hist.presumedNextMovePla`** (set
  from the engine after every move).
  - `Search::makeMove` (the root);
  - `Search::playoutDescend` (both descent branches);
  - the four PV replays in `analysisdata.cpp`.
- **The player who made the move into a node: change to
  `moveHistory.back().pla`.**
  - Subtree value bias in `Search::allocateOrFindNode` (on in self-play,
    `subtreeValueBiasFactor = 0.30`). Its key's previous move stays the
    literal previous move, which may be the same player's first edge. Its
    local pattern is adapted to edges (below).
- **"The other player" as an identity: unchanged.** Visit-cap contempt,
  pondering's assumed side, and playout doubling advantage all name a player,
  not the next mover.
- **Go-only, kept neutral:**
  - the pattern bonus table (`avoidRepeatedPatternUtility`, default 0) and
    anti-mirror (`antiMirror`, default off);
  - pass-alive pruning in `rootPruneUselessMoves` (passes never occur in a
    search);
  - `getEndingWhiteScoreBonus`, which only moves the score, and score
    utility is 0.

Outside `search/`, the self-play game loop (`program/play.cpp`,
`playutils.cpp`) and the analysis engine (`command/analysis.cpp`) get the
same change. GTP, SGF, book and start-position tools are not on our path and
stay as they are.

Training rows need no change for turn order: KataGo's opponent-policy target
is the next move's policy, whoever plays it (`policyTargetsByTurn[i + 1]`),
and value targets are written from the row's `nextPla`.

**Graph search:** `Board::simpleRepetitionBoundGt` returns true, not Go's
safe false. Enclosure positions never recur (the move count only rises and
is hashed), so the state hash alone identifies a node. False would chain
the move path into every graph hash, and transposed turns (A then B, B then
A) would never share a node.

Tests: `katago runenclosuretests` (`tests/testenclosuresearch.cpp`), run in
a Debug build (asserts on, `-D_GLIBCXX_ASSERTIONS`) with KataGo's
no-network evaluator (`debugSkipNeuralNet`: random logits and values, exact
results at the end of the game). Graph search and subtree value bias are on.

- **Tree invariants.** Random games are searched at edges 1-4, 31-32 and
  117-120: blue's opening edge, each edge of a turn, and the last turns.
  - Every node's `nextPla` is the history's.
  - Every child's move is legal for that player.
  - The PV replays with the history's players.
- **Transposed turns share a graph hash.**
- **Two edges, one mover.**
  - Red plays edges 118-119 as one turn; Blue's single edge 120 ends the
    game.
  - Red holds two sides of a 3×3 square; only the other two sides together
    win. That was checked exhaustively on the Python engine: 2 of Red's
    45,601 two-edge turns win, and all others lose.
  - The search runs 100,000 visits with c_score = 0 and flattened random
    priors (`nnPolicyTemperature` 100). The root considers 20 first edges
    (KataGo's allowMoves): both winners and 18 losers. Red's second edge,
    the node under test, keeps every legal move. With raw random priors,
    first-play urgency keeps a low-prior second edge untried for thousands
    of visits, because every tried line loses.
  - The search must pick a winning first edge and value it as a Red win.
  - After `makeMove`, the root must still be Red, with that child's visits
    kept (tree reuse between our two edges).
  - It must then complete the square.
  - Negating values across Red's own turn would avoid the win.
- **Checked against the bug it targets:** with `getOpp` put back into
  `playoutDescend`, the tests fail at once.
- **Result (2026-09-29):**
  - All 10 invariant positions pass (55-1,387 nodes each).
  - Red opens with I9-F9 (win value 0.997); the root keeps 93,891 visits
    after `makeMove`; Red completes the square with F6-F9 (0.999).

**Terminal scores:** `ScoreValue::whiteScoreMeanSqOfScoreGridded` asserted
Go's half-point grid. Enclosure scores are exact areas, so a final score is
a point mass (mean square = score²), in line with D1's Enclosure score
bins.

## Subtree value bias (owner, 2026-09-29)

KataGo buckets search nodes by the mover, the last move, the move before it,
and a 5×5 window of point colours and atari around the last move. Each
bucket's average error of the network against deeper search then corrects
new nodes in the same bucket (`docs/KataGoMethods.md`).

- **Enclosure keeps the intention:** the same local tactic in the same local
  shape shares a bucket.
- **`LocalPatternHasher::getHash`** centres a **7×7** window on the edge's
  source point, which covers the whole edge. It hashes every edge of either
  player with an endpoint in the window (owner and exact edge), plus
  whether each is protected.
  - Point colours alone would miss which nodes are joined: a closed wall and
    four loose nodes would share a bucket.
  - Protection plays the part of Go's atari flag.
- **Unchanged:** λ = 0.30 (the self-play config), α, and the key's last two
  moves. The mover comes from the history.
- **`getHashWithSym`** (the pattern bonus table, off by default) refuses
  until edge symmetries exist (step 4).

## Moves where points are expected (found 2026-09-29; blocks step 3)

At first, move `Loc`s were 2 + action and shared the values 2-420 with
points. Code that treats a move as a point then gives silently wrong answers
for those moves and reads past the end of the array for the rest. KataGo's
`assert`s are compiled out in Release.

- **Fix (owner): disjoint ranges.** Moves start at `MOVE_LOC_BASE = 421`,
  so a move used as a point is always out of range.
- **Tests run in a Debug build** (asserts on, `-D_GLIBCXX_ASSERTIONS`), so
  a miss fails loudly.

Found so far:

- **Arrays of `Board::MAX_ARR_SIZE` indexed by a move:**
  - `rootSymDupLoc` and `rootSymRepresentativeLoc` (`search.h`);
  - the subtree value bias Zobrist table (`subtreevaluebiastable.cpp`, on
    in self-play);
  - the pattern bonus table.
- **Arrays of `MAX_ARR_SIZE` holding one entry per legal move,** while
  Enclosure positions have up to about 600 legal edges:
  - `chooseIndexWithTemperature` (`searchhelpers.cpp`), which picks the
    played move.
- **Loops over points to list legal moves:**
  - `playutils.cpp`'s random-legal-move choosers loop `loc` over
    `0..MAX_ARR_SIZE`. With base 2 they saw only moves 2-420, the first 419
    edges; with base 421, none.
  - Self-play's random opening moves would therefore all fall in one corner,
    with no error.
- **Point functions given a move:** `isOnBoard(moveLoc)` in the root
  checks and the avoid-move filter (`searchhelpers.cpp`), and
  `isAdjacentToPla(moveLoc, ...)`.

Every such place is resized to `Board::NUM_MOVE_LOCS`, or to the policy
size, or given the point a move means. Also found and fixed:

- **`getEndingWhiteScoreBonus`** (Go's dame-filling bonus, on in self-play)
  read ownership at `whiteOwnerMap[action index]`, past its 361 entries. It
  returns 0 now, which is exactly neutral: it only shifts the score, and
  score utility is 0.
- **`NNPos::locToPos` meant both** a move's policy index and a point's
  spatial index. It now takes moves only. Points use
  `NNPos::pointToPos`.
- **Root symmetry pruning** refuses to run until edge symmetries exist
  (step 4). It is off by default and in self-play.

## Policy indexing (moved from step 4 into step 3)

The search cannot expand Enclosure moves without it, so it comes first.

- **`NNPos::MAX_NN_POLICY_SIZE`** = 19 × 19 × 48 + 1 = 17,329.
- **`locToPos`** maps a move to its action index, with pass last at
  17,328.
- **The no-network evaluator (`debugSkipNeuralNet`)** fills every policy
  entry.
- **`Board` keeps its legal moves as a bitset**, updated with the position,
  so `isLegal` is a lookup. The evaluator asks it once for each of the
  17,329 entries.
- **Left for step 4:** the network's inputs, symmetries, and how the
  backends lay out the 48 channels per point.

## Network inputs (step 4.1)

Input version **V8** (`NNInputs::fillRowV8`) is the Enclosure network's
input (the Enclosure repository's `net.input_planes`), laid out the KataGo
way:

- **Spatial, 215 planes** (`NUM_FEATURES_SPATIAL_V8`):
  - 0: on board, all ones. KataGo's backends and PyTorch model take the
    board mask from channel 0, so it stays first.
  - 1-192: the observation's edge planes, `4 × 48`: each player's edges,
    then each player's protected or just-placed edges, set at the edge's
    lower end (the canonical `Edge`'s a) in the offset's plane; with
    `ENCLOSURE_EDGE_BOTH_ENDS=1` (v2a from 2026-10-03, cycle 31) also at
    its upper end in the opposite offset's plane.
  - 193-214: the 22 feature planes (`features.h`, `planes.feature_planes`):
    9 per player, then 4 history planes. In v2a, plane 7 of each player
    (held ground) is empty under `ENCLOSURE_NO_HELD_GROUND=1` (from
    2026-10-03 10:06), and holds the turn-safe plane under
    `ENCLOSURE_TURN_SAFE=1` (`planes.turn_safe_edges`; plane 8, openings,
    stays empty).
- **Global, 5** (`NUM_FEATURES_GLOBAL_V8`): the observation's scalars (player
  to move, edges left in the turn / 2, move count / 120, and both scores /
  (18 · 18 · 61)). The plan puts the scalars here, not in constant planes.
- **Absolute colours,** as the Enclosure network: plane k is Blue's for
  k = 0. The player to move is a global input. (KataGo's Go inputs are
  relative to the mover; the owner chose "our feature planes, as KataGo uses
  its Go features".)
- **Model versions 8-17** keep their architecture meanings but all read V8;
  Go networks (and versions 3-7) are refused. Go's `fillRowV3`-`V7` stay in
  the file, unreachable, to keep the diff from upstream small.
- **The NN cache hash** already includes the history (step 2's
  `historyHash`), so positions differing only in history planes are
  cached apart.
- `maxHistory <= 0` (KataGo's no-history evaluations) drops the 4 history
  planes.
- **Test:** `tests/enclosure/testnninputs.cpp` against
  `tests/data/enclosure_inputs.txt` (the Enclosure repository's
  `tools/export_katago_inputs.py`: 336 positions from 40 archived and 40
  random games, with Python's reference inputs). Every spatial entry
  (NHWC and NCHW) and every global matches; a changed fixture entry fails.

## Symmetries (step 4.2, D5)

**Measured** (`tests/enclosure/testsymmetry.cpp`, 2026-09-29): for each of
KataGo's 8 symmetries, 100 random games were played on the board and on
its image, each move mapped. Over 12,000 moves per symmetry, the image game
was always the image of the game: legal moves (one to one), edges,
protected and placed edges, history, and exact scores. The feature planes
also moved point by point.

| Symmetry | Rules respect it | Fixes the start position |
|---|---|---|
| 0 identity | yes | yes |
| 1 flipY (y → 18 − y) | yes | yes |
| 2-7 (flipX, transposes) | yes | no (Blue's start moves right or vertical) |

So the rules are symmetric under all 8, but only the identity and the
mirror map real games to real games: under the other 6, Blue starts on the
right or on a vertical line, which never happens. How many to use is the
owner's decision (D5 assumed only the mirror; see `docs/decisions/katago-fork.md`).
**Decided (owner): all 8** (exact labels, KataGo's method, and reversible
without new data; an arena A/B of 2 against 8 can revisit it).
The code handles all 8. `SymmetryHelpers::NUM_SYMMETRIES` sets the count,
and KataGo's numbering puts flipY at 1, so a count of 2 means identity and
mirror.

**How symmetries work in the fork:**

- **They act on positions, not on input planes.** An edge's image is the
  canonical edge between its endpoints' images, so under a flip a
  vertical edge re-anchors at its other end: the edge planes are not a pixel
  and channel permutation. `SymmetryHelpers::getSymEdge`,
  `getSymPosition` and `getSymBoard` map edges; `getSymAction` and
  `getSymMoveLoc` map moves. Between two of the mover's own nodes, only one
  direction is listed as legal, so the image move is looked up on the image
  board.
- **Network evaluation (`NNEvaluator::evaluate`).** The symmetry is chosen
  where the inputs are filled: at random (a per-thread generator seeded
  from the evaluator's seed), or the default, as the server thread did. The
  inputs are the image position's, and the backend gets the identity. On
  return, the policy is mapped back move by move and the ownership point by
  point, before the softmax and the cache. The backends' pixel symmetry is
  never used.
- **Root symmetry pruning** (`markDuplicateMoveLocs`): a symmetry is valid
  when the position (edges, protected and placed edges) is its own image.
  Of each set of equivalent legal moves, the lowest action is searched.
  `computeRootSymRepresentativeLocs` and the analysis output map moves with
  `getSymMoveLoc`, and principal variations move by move on the image
  boards (`getSymMoveLocs`).
- **`LocalPatternHasher::getHashWithSym`** hashes the image board's window
  around the image move.
- **Tests:**
  - `testsymmetry`: the measurement above. Also 153,600 round trips and
    compositions (inverse maps back; s1 then s2 is `compose(s1, s2)`), and
    root pruning at the start, where the valid symmetries are {0, 1} and
    69 legal moves become 36 (3 lie on the axis).
  - `katago runenclosuretests`: the two-edges-one-mover square, searched
    on its image under every symmetry, must find the image of the winning
    turn; and root pruning at the start searches one move of each mirror
    pair, while the analysis lists both, each with a legal principal
    variation.
- **Training-data augmentation** (KataGo's Python, step 6) mirrors pixels
  too. It must re-anchor edge planes and remap policy channels the same
  way, and is tested against these helpers.

## Score utility (step 4.3; revised 2026-10-03)

**Since 2026-10-03 (owner): KataGo's score utility, with an Enclosure score
unit.** Before, c_score = 0 (owner, option (b), win/loss only).

- KataGo's utility: factor x (2/pi) atan((margin - centre) / (scale x unit)),
  the margin being the signed final score difference. KataGo's unit is the
  board's side (sqrtBoardArea, 19: the square root of Go's largest score).
  Enclosure's largest total score is the largest area, 18 x 18, paid at each
  of 61 turn ends, so the unit is sqrt(18 x 18 x 61), about 141 points
  (`Board::scoreUnit`; iviazovetskyi's bot uses 150). Only the search's score
  utility and the score values it reports read it (`searchhelpers.cpp`,
  `reportedsearchvalues.cpp`, `searchresults.cpp`); komi, handicap and old
  model versions keep sqrtBoardArea.
- KataGo's values on top: self-play (`selfplay_enclosure.cfg`, which v2a uses, and `selfplay1.cfg`) static 0, dynamic 0.40,
  centre zero weight 0.25, centre scale 0.50 (about 70 points); play and
  analysis (`analysis_example.cfg`, `match_example.cfg`) static 0.10, dynamic
  0.30 (centre zero weight 0.20, scale 0.75: about 105 points), as upstream's
  defaults.
- The gauntlet's `analysis_enclosure.cfg` keeps 0 for now, so its ratings
  stay comparable across the run.
- The score value tables clamp scores beyond Go's range; with the unit, the
  scaled scores stay within them.

## Score distribution bins (D1, owner 2026-09-29)

842 bins as KataGo, each 12 points wide, covering margins of ±5,052 (every
margin in `net1`'s self-play, 99% of full-length human games; measurements
in `docs/decisions/katago-fork.md`). Larger margins go to the end bins. Only the
training rows and the Python head use the distribution (step 6).

## The network's policy head, export and CPU backend (step 5, 2026-09-29)

- **Policy layout:** every policy output has 48 channels per point,
  channel = output × 48 + offset. So policy entry (y · 19 + x) · 48 +
  offset (the action index) reads the NHWC output in place. Pass stays one
  logit per output.
  - PyTorch (`model_pytorch.py`): `conv2p` has `num_policy_outputs × 48`
    channels. The head returns (N, outputs, 17,329) with pass last.
    `NUM_OFFSETS` and `policy_len` hold the numbers.
  - Export (`export_model_pytorch.py`): the chosen outputs (regular and
    optimistic, as upstream) are written as blocks of 48 channels.
  - C++ (`desc.cpp`): `p2Conv.outChannels` must be `policyOutChannels ×
    48`. The Eigen backend sizes its buffers and decodes the blocks (the
    optimism blend per entry).
  - CUDA (`cudaandrocmbackend.inc`, 2026-09-30):
    - the host and device policy buffers hold 48 channels per output;
    - the pass buffer holds one logit per output;
    - the decode reads NHWC or NCHW;
    - the backend refuses any symmetry but the identity.

    Tested with `katago runenclosurenntests MODEL INPUTS OUT [fp16] [nhwc]`
    on the RTX 5080 against PyTorch:
    - FP32 with TF32 off (`NVIDIA_TF32_OVERRIDE=0`), NCHW and NHWC: within
      7e-6, as the CPU;
    - FP32 with TF32 on (cuDNN's default on this GPU): within 3e-3;
    - FP16, NHWC and NCHW: within 6e-3.

    TensorRT comes later.
- **Inputs in PyTorch:** `modelconfigs` gives 215 spatial and 5 global
  inputs. The model's first layers read them from there instead of Go's
  hard-coded 22 and 19.
- **Go's score parity** (the komi feature times a parity vector in the
  score-belief head) is kept as a no-op: the vector is zero. The last global
  input is a score here, not komi.
- **Score-belief bin centres** are in points: 12 × (bin − 421 + 0.5)
  (D1).
- **No result:** the fork never sets `isNoResult`. Its rules (area scoring,
  positional ko) take KataGo's branch that zeroes the no-result output, so
  win and loss share the probability, as under Go's area rules.
- **Test:** `katago/python/enclosure_nn_check.py` with `katago
  runenclosurenntests`. A b6c96 is initialised at random, then every
  parameter and statistic is perturbed so that no layer is an identity. It
  is exported, evaluated in C++ (Eigen, `NNEvaluator`) and in PyTorch on the
  332 unfinished positions of `tests/data/enclosure_inputs.txt`.
  - Largest differences: policy 6e-6 (relative), value 6e-7, score 4e-6
    (in units of the model's score multiplier, 240 points after the score
    retune), ownership 7e-6. The tolerance is 1e-3.
  - A model with other weights fails at every position.
- **Training augmentation** (`katago/train/enclosure_symmetry.py`, used by
  `data_processing_pytorch.py`):
  - point planes and the value targets move as pixels;
  - the edge planes and the policy and q-value targets go through index
    maps. A stored edge goes to its image's lower end. A move between two
    of the mover's nodes (read from the node feature planes before the
    symmetry) goes to its image's lower end; any other move goes to the
    image of its source.
  - Symmetries are numbered as in C++.
  - Test: `katago/python/enclosure_symmetry_check.py` against `testsymmetry
    -dump`. For 40 positions × 8 symmetries, the augmented inputs equal the
    C++ inputs of the image position, and every legal move lands on
    `getSymAction`'s image. Scrambled symmetry labels fail.

## Score scale (step 6, owner 2026-09-29)

**Decided:** scores stay in raw points, and every constant KataGo tuned to
Go's point scale is scaled by `SCORE_SCALE` = 12 (C++ `NNPos::SCORE_SCALE`,
Python `model_pytorch.SCORE_SCALE`). That is D1's bin width against
KataGo's 1-point bins, from the measured margins. Training then weighs every
score term exactly as KataGo does on Go-sized scores (owner: "scale-invariant
retune"). The rule:
- quantities in points × s;
- squared points × s²;
- each loss weight ÷ its loss's growth: a Huber loss whose argument and
  delta both scale by s grows by s², and the squared-error one by s⁴.

| Constant | KataGo | Enclosure | Why KataGo picked it |
|---|---|---|---|
| Output multipliers: score mean, stdev, lead, short-term scores | 20 | 240 | Paper A.5: "an arbitrary reasonable scaling factor so that on typical data the neural net would only need to output values around unit scale" |
| Short-term score error multiplier | 150 (squared points) | 21,600 | Same aim, for a squared error (model v13 rescaled it) |
| Score distribution range | ±(361 + 60) points | ±5,052 points | Paper A.5: "an upper bound for the plausible final score difference", the board area plus 60 for komi |
| Score target cap | 421 | 5,052 | The same bound, so targets stay in range |
| Q-value score storage in int16 | ×60 | ×5 | Keeps ±421 within 32,000 at 1/60-point precision; ×5 keeps ±5,052 within it |
| Huber delta: score mean | 12 | 144 | Code comment: Huber is "meanlike locally and ... medianlike for very large possible losses". The specific deltas: source not yet found (to trace in KataGo's commits and issues). The paper's first version used 10. |
| Huber delta: lead / stdev / q-value score / short-term score | 8 / 10 / 12 / 12 | 96 / 120 / 144 / 144 | As above |
| Huber delta: short-term score error | 100 (squared points) | 14,400 | As above, in squared points |
| Irreducible error floor: short-term score | 1e-4 ("one hundredth of a point", squared) | 1.44e-2 | A regulariser: a hundredth of the unit |
| Loss weights: score mean 0.0015, lead 0.006, stdev 0.001, short-term score 0.0004, q-value score 0.0008 | as listed | ÷144 | Source not yet found (to trace in KataGo's commits and issues). The paper's first version used 0.004 for the self-predictions. Dividing keeps each term's share of the total loss whatever the original reason. |
| Loss weight: short-term score error | 0.00002 | ÷20,736 | A fourth-degree loss |
| Score distribution pdf and cdf losses | 0.02 each | unchanged | Over bins, not points (paper: pdf and cdf 0.02) |

The paper's base weights, for reference: value 1.5, policy 1.0, opponent
policy 0.15, ownership 1.5/b², score pdf and cdf 0.02 each, L2 3e-5.

## Training rows and the trainer (step 6, 2026-09-29)

- **Rows** (`trainingwrite.cpp`):
  - policy targets of 17,329 moves;
  - D2 ownership in channel 0, × 120 with KataGo's stochastic rounding. It
    is computed from the game's positions after the row, at every turn end;
  - score targets capped and binned as above;
  - Go's seki, future positions and area targets zero, with weight 0 (the
    seki loss is zeroed in Python, since its weight is the ownership
    weight);
  - no-result never occurs.
- **History dropout** (owner: KataGo-style), in
  `data_processing_pytorch.apply_history_dropout`:
  - KataGo's draws (each item stops with probability 0.02, older ones
    first) on the 4 history planes;
  - the current- and previous-turn edge planes carry protection and are
    never dropped;
  - measured on 200,000 rows: planes kept 98.0 / 96.0 / 94.1 / 92.3 %,
    against 0.98^k.
- **Soft-policy mask** (metrics): a move is on the board when its source
  point is (48 moves per point).
- **Tests:**
  - `katago runenclosurerowtests OUT [SEED]` plays one self-play game with
    the no-network evaluator and writes its rows as text.
  - The Enclosure repository's `tools/check_katago_rows.py` replays the
    game in its engine and checks every row: inputs equal `input_planes`,
    policy targets on legal moves, D2 ownership within the storage
    rounding, result, score and the score distribution's mean, q-value
    range, and Go targets zero. Three games, 360 rows: 0 failures.
  - `--npz` writes them for `katago/python/enclosure_rows_check.py`. It
    reads them with KataGo's `read_npz_training_data` (symmetries and
    history dropout on), then runs the forward pass, the metrics and one
    backward pass (no optimizer step). 21 batches: shapes as expected, and
    every loss and gradient finite. The Go-only losses are 0.
  - Writing real `.npz` files still needs libzip (`sudo apt install
    libzip-dev`, the owner's, before step 8). The test uses the writer's
    text output instead.

## Spatial targets (step 6; design for the owner, 2026-09-29)

**Built (owner, 2026-10-03: "add all of our ideas and just run"), in KataGo's existing slots, so rows keep their
format and old and new rows mix in the shuffle:**
- **1. Future nodes** in C2-3: KataGo's future-position code as it is (`board.colors` marks each player's nodes;
  boards are one edge apart), 8 and 32 edges later, its loss (tanh, squared error, [1, 0.25], ÷√area, 0.25),
  weight C33 = 1.
- **2. Edges that get cut** in C1 (the seki slot): +1 / -1 / 0 on points on edges, 2 on points on no edge; the
  seki head's three sign logits, cross-entropy averaged over the points on edges (not the whole board: the
  question exists only there), weight 1 times the ownership weight, on rows flagged in C68 (rows written
  before have zeros, which would teach that nothing is ever cut). KataGo's adaptive seki scale and neutral
  logit are not used.
- **3. Region occupancy** in C4 (the scoring slot): one horizon, the 8th turn end later (4 rounds; the last if the
  game ends sooner), x 120; KataGo's area loss (squared error with a square-root transform, 0.25), weight C34
  = the value weight. The 1-round horizon is left out on its merits too: the current regions are an input,
  the future nodes cover 8 edges ahead, and D2 the whole game.
- Checked by `tools/check_katago_rows.py` (recomputes all three from the engine) and
  `enclosure_rows_check.py` (KataGo's trainer, every loss finite).

The design as written on 2026-09-29:

**Decided (owner):**
- KataGo's Go-only spatial targets (seki, future stone positions, final
  area/territory) get weight 0, with their heads kept.
- The ownership target is D2: per point, the mean over the remaining turn
  ends of +1 inside the mover's regions, −1 inside the opponent's, and 0
  otherwise.
- The owner asked for Enclosure-specific targets instead. They are worked
  out below, to be added one at a time (weight 0 until each is tested in
  the arena).

KataGo's reasons, from its code comments: auxiliary spatial targets give
the trunk dense, per-point gradients about the future beyond the one
policy and value number per position. Future positions use a fixed horizon
(so a fixed number of points are "wrong" whatever the board size), and the
far horizon has a quarter of the weight, since it is harder to predict.

### 1. Future nodes (KataGo's future positions, for nodes)

- **Target,** from the row's mover's side: per point, +1 if it is one of
  the mover's nodes, −1 if one of the opponent's, 0 otherwise. It is taken
  at two horizons, near and far, and past the end of the game the final
  position is used.
  - KataGo takes the board 8 and 32 moves later (`trainingwrite.cpp`), 4
    and 16 rounds of Go. An Enclosure round is 2 turns, 4 edges.
  - Mapping rounds gives 16 and 64 edges, over half of a 120-edge game.
    Scaling by game length (about 250 moves in Go) gives about 4 and 16
    edges.
  - The choice belongs with the other time horizons below.
- **Head and loss:** KataGo's `futurepos` head (2 channels) and its loss,
  unchanged: squared error of tanh, horizon weights [1, 0.25], summed and
  divided by √area, weight 0.25.
- **What it teaches:** where each side will build next (and which nodes
  vanish when their edges are cut). It is a spatial summary of the next
  moves, like Go's.

### 2. Edges that get cut (the counterpart of Go's dead stones and seki)

- **Target:** per lattice point on a current edge (its endpoints and the
  lattice points it passes through), 3 classes: no current edge there is
  cut before the game ends; one of the mover's edges there is cut; one of
  the opponent's is cut. (If both, the mover's.) An edge counts as cut when
  that edge is removed, even if an identical edge is placed again later.
- **Head and loss:** the `seki` head's 3 sign logits, with cross-entropy
  per point divided by the board area. KataGo's adaptive seki scaling is
  replaced by a plain weight, tuned when tested: it tracks how rare seki
  is, which does not carry over.
- **What it teaches:** which walls are weak. It is the fragility a static
  input plane cannot show: protection lasts one turn, and a cut depends on
  the opponent's reach and tempo.

### 3. Region occupancy at several horizons (both players together)

- **The owner's point:** a next-turn region gain leans greedy. So the
  target takes both players together, over horizons out to the end of the
  game.
- **Target,** from the mover's side: per point, +1 inside the mover's
  regions, −1 inside the opponent's, 0 in neither or both. It is taken at
  the turn end 1 round and 4 rounds later.
  - The change from now needs no separate target: the current regions are
    an input (the enclosed feature planes), so predicting the future
    occupancy is predicting the gain and loss of both players at once.
  - D2 ownership (the mean over all remaining turn ends) is the
    end-of-game member of the same ladder. So the network learns near,
    middle and whole-game occupancy, and no single horizon dominates.
- **Head and loss:** a new 2-channel 1×1 head, like `futurepos`, with its
  loss form (weights [1, 0.25], ÷√area). The `scoring` head has one channel
  only, so it stays at weight 0.
- **Why not "next turn's gain" alone:** a short horizon rewards whatever
  closes a region now. The 4-round horizon and D2 carry what the opponent
  does in return.

**Time horizons: split by what they measure (owner, 2026-09-30).**

KataGo sets its time constants from the board area. In Go, the area sets
both the game's length and the reach of its tactics; Enclosure keeps the
19×19 board but has a shorter game.

The only source is KataGo's methods doc: the short-term targets average
"roughly the next 6 turns, 16 turns, and 50 turns ... on a 19x19 board, and
correspondingly shorter for smaller boards in proportion to the board area".
It gives two purposes:
- lower-variance value feedback (bias-variance);
- predicting what a search concludes "a few turns later", which the
  uncertainty weighting and the optimistic policy use.

Both count decisions, not fractions of a game.

- **Look-ahead horizons are KataGo's 19×19 formulas, unscaled, counted in
  edges** (the decisions in our search tree). This covers the short-term
  value and score targets (about 6, 20 and 64 edges, that is 3, 10 and 32
  turns), the short-term error targets (about 6 edges), value-surprise
  weighting, and the future-node design's horizons (8 and 32 edges).
- **Game-phase constants scale by game length** (below): the earliest
  resignation, early forks and opening moves.
- **The variance-time multiplier** keeps outputs near unit scale. It is
  scaled provisionally and will be set from the measured size of its target
  once self-play data exists.

An intermediate version scaled everything by game length (2, 7 and 23
edges). The owner found 2 edges, about one turn, too short, which led to
this split.

Game-phase constants:
- `Board::TIME_SCALE` (Python `model_pytorch.TIME_SCALE`) = 120 / 328 ≈
  0.366: Enclosure's 120 edges against the measured mean length of KataGo's
  own 19×19 self-play games without handicap or setup stones.
  - Source: katagoarchive.org kata1 training games. On 2020-12-08, 732
    games: mean 328.7, median 329, 10th-90th percentile 294-363. On
    2026-09-26, 51,094 games: mean 328.1, median 327, 292-366.
  - KataGo plays games out rather than resigning, so they are longer than
    human games.
  - An earlier guess of 250 moves was replaced by this measurement (owner:
    find the source of every number).
- `Board::GAME_LENGTH_AREA` = 361 × 0.366 ≈ 132 stands in for the board
  area in the game-phase constants:
  - the earliest self-play resignation (A / 5);
  - the early-fork move (`earlyForkGameExpectedMoveProp` × A);
  - the number of opening policy moves (`policyInitAreaProp` × A).
- Provisionally, until its target is measured: the variance-time output
  multiplier (40 → 14.6) and its Huber delta (50 → 18.3), with the loss
  weight ÷ 0.366².
- (The look-ahead formulas keep the board area. KataGo's code comment on
  the short-term constants: they "used to be 'nicer' numbers 0.18, 0.06,
  0.02" until a fix to the functional form, with values nudged to keep
  9×9-19×19 unchanged.)
- Owner: the specific numbers are what made sense in Go, not only their
  scale. They may deserve tuning for Enclosure later.

**Data:** all three come from the finished game. Self-play already keeps
every position of the game, and at the end each row's targets are read
from the positions h rounds later (nodes, region masks from the engine's
`enclosed_area`, and which edges survive).

## Sizes at 17,329 actions (steps 4-6)

- **`NNOutput` stores a dense policy** of `MAX_NN_POLICY_SIZE` floats,
  69 KB per result.
  - Search trees hold one per expanded node: self-play's 128 games at 600
    visits is about 5 GB.
  - The self-play config's `nnCacheSizePowerOfTwo = 21` would be 145 GB. It
    must come down (2^15 is about 2.3 GB). This is a setting, not a code
    change.
  - A sparse policy (legal moves only, about 30 times smaller) waits until
    memory is measured.
- **Training rows:**
  - The two policy targets are 69 KB a row (KataGo's 19×19 rows total 5.5
    KB), and the q-value targets 104 KB. The file format stays KataGo's;
    the mostly-zero policy compresses on disk.
  - Memory is set by the shuffle's `-approx-rows-per-out-file` (upstream
    70,000 rows, about 10 GB of float32 policy per file in the trainer) and
    its process count.
  - Both are measured in step 6 before choosing values.

## BoardHistory surface (from the survey of callers outside `game/`)

- **Kept with Enclosure meaning:**
  - `moveHistory`, `isLegal`, `makeBoardMoveAssumeLegal`,
    `makeBoardMoveTolerant`, `isLegalTolerant`, `getRecentBoard`, `clear`;
  - `isGameFinished` (after line 120), `winner`,
    `finalWhiteMinusBlackScore` (red minus blue), `isNoResult` (the rules'
    draw when the player to move has no legal edge);
  - `presumedNextMovePla`, `initialBoard`, `initialTurnNumber`,
    `getCurrentTurnNumber`, `endAndScoreGameNow`, `setWinnerByResignation`,
    `isResignation`.
- **Neutral, for Go only:**
  - `encorePhase` = 0, `superKoBanned` and `koRecapBlocked` all false;
  - `setKomi` and `currentSelfKomi` (komi 0), `hasButton` = false;
  - handicap functions: none;
  - `passWouldEndPhase` and `passWouldEndGame` false (a pass is only the
    site's timeout);
  - `endGameIfAllPassAlive`: nothing to do.

## Verification (step 2)

The game layer must equal the Enclosure repository's engine, which is
verified against both sites: legal moves, results and scores on the archived
games (the 3,801-game native diff) and on random games, move by move.
`tests/` gains an Enclosure test that replays them from a fixture file
exported by the Enclosure repository.

## Build and resources

Built in `~/katago-build` with the Eigen CPU backend. While the Enclosure
bot YshanV1 is playing, builds and tests run only on CPUs 14-15 at nice 19,
and nothing uses the GPU.
