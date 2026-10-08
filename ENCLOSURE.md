# KataGo for Enclosure: the fork's design

`katago/` (upstream `d91ea85` plus our changes) turns KataGo into an Enclosure bot. The plan, its build order and the
approved departures are in `v2/docs/katago-fork-plan.md` of this repository. This file records the design decisions
inside the code. The "steps" below are the steps of that plan. The network itself, drawn and compared with KataGo's,
is in [ENCLOSURE_NETWORK.md](ENCLOSURE_NETWORK.md).

## Principle

KataGo's methods, parameters and structure stay as they are. The fork replaces the game layer (`cpp/game/`). Every
other file changes only where it relies on Go: moves as points, alternating turns, Go's input features, and the
policy size. Concepts that only Go has (ko, superko, encore phases, komi, handicap, the button, pass-alive areas) keep
their names and signatures, fixed at their neutral values. So their callers compile unchanged, and nothing in them
branches on Enclosure.

## Moves and points (step 2)

`Loc` stays a 16-bit `short`, but it now names two different things:

- **A point**, where KataGo indexes the board: ownership, areas, the planes of the network's inputs and outputs, and
  symmetries. A point keeps KataGo's layout, `Location::getLoc(x, y, 19)`, within `Board::MAX_ARR_SIZE` (421). Only
  the game layer and the network's input and output code handle points.
- **A move**: `NULL_LOC` (0), `PASS_LOC` (1), or `MOVE_LOC_BASE` (421, which is `Board::MAX_ARR_SIZE`) + our action
  index `(y * 19 + x) * 48 + offset`. The offsets follow the Enclosure engine's order: dx, then dy, each from -3 to 3,
  without (0, 0). So a move `Loc` goes from 421 to 17,748.
  - Moves start after the points, so no value is both (see "Moves where points are expected").
  - Everything that stores, compares, hashes or passes moves (the search tree, histories, training rows) is
    unchanged.
  - `NNPos::locToPos` maps a move to its policy index, which is the action index. Pass is last, at 17,328.

A `Move` is still `(Loc, Player)`.

## Board and Rules (step 2)

- **`Rules` is unchanged.** It is plain data (ko, scoring and tax rules, komi), and Enclosure reads none of it. The
  configurations set komi to 0.
- **`Board` holds an `enclosure::Position`** (`game/enclosure/engine.h`). It was generated from the Enclosure
  repository's verified native engine by `v2/tools/extract_katago_engine.py`.
  - The fork's copies are now edited here directly, and they are ahead of the native source. They hold V2's input
    options (no_maxflow, edges at both ends, the turn-safe plane), the income areas (`clipped_area`), and faster
    feature planes ([speed.md](../v2/docs/decisions/speed.md) "Faster features").
  - A new extraction would drop these changes. So port a native change by hand.
- Callers outside `game/` use a few members (from a survey of the callers): the size (`x_size`, `y_size` = 19),
  `colors` (74 uses), `pos_hash`, `ko_loc`, and Go's stone logic (`getNumLiberties`, `isSimpleEye`, captures).
  - `colors[point]` is `C_BLACK` where the point is one of Blue's nodes, `C_WHITE` where it is one of Red's, and
    `C_EMPTY` otherwise. So the display and "whose point is this" continue to work.
  - `pos_hash`: Zobrist hashing over the edges, the protected and placed edges, the scores and the player to move.
    These are the fields that the engine's `same_position` compares.
  - `ko_loc` = `NULL_LOC`.
  - Go's stone logic stays declared, with neutral answers. Steps 3-4 reviewed each caller. Most callers are the Go
    input features, which the fork replaces anyway.

## Players and turns

Blue moves first and is `P_BLACK`. Red is `P_WHITE`. The turn order is 1-2-2: Blue's single opening edge, then turns
of two edges. So the next player comes from the history (`BoardHistory::presumedNextMovePla`), never from `getOpp` of
the last mover. In `search/`, every `getOpp` on the sequence of moves (29 places) now asks the position (step 3).

## Turn order in the search (step 3)

KataGo already fits two-edge turns. Each `SearchNode` stores its own `nextPla`. The search keeps statistics from White's side
and read with the sign of the player who chooses, so nothing negates per ply. Only the places that *derive* the next
player from the last mover break. A survey of the 29 `getOpp` calls in `search/` found three kinds:

- **The next player after a move: changed to `hist.presumedNextMovePla`**, which the engine sets after every move.
  - `Search::makeMove` (the root).
  - `Search::playoutDescend` (both descent branches).
  - the four replays of the principal variation in `analysisdata.cpp`.
- **The player who made the move into a node: changed to `moveHistory.back().pla`.**
  - Subtree value bias in `Search::allocateOrFindNode` (on in self-play, `subtreeValueBiasFactor = 0.30`). The
    previous move of its key stays the literal previous move, which can be the same player's first edge. Its local
    pattern is adapted to edges (below).
- **"The other player" as an identity: unchanged.** Visit-cap contempt, the side that pondering assumes, and playout
  doubling advantage all name a player, not the next mover.
- **Go only, kept neutral:**
  - the pattern bonus table (`avoidRepeatedPatternUtility`, default 0) and anti-mirror (`antiMirror`, default off).
  - pass-alive pruning in `rootPruneUselessMoves` (passes never occur in a search).
  - `getEndingWhiteScoreBonus`: it returns 0. Go's bonus reads the ownership at the move's point, and an edge has no
    point. With anti-mirror off and pass suppression for territory scoring only, the search computes the ownership
    at its root but uses it to choose no move.

Outside `search/`, the self-play game loop (`program/play.cpp`, `playutils.cpp`) and the analysis engine
(`command/analysis.cpp`) get the same change. GTP, SGF, the book and the start-position tools are not on our path,
and they stay as they are.

Training rows need no change for turn order. KataGo's opponent-policy target is the policy of the next move, whoever
plays it (`policyTargetsByTurn[i + 1]`). The value targets are written from the row's `nextPla`.

**Graph search:** `Board::simpleRepetitionBoundGt` returns true, not Go's safe false. Enclosure positions never
recur, because the move count only increases and is hashed. So the state hash alone identifies a node. With false,
the move path would go into every graph hash, and transposed turns (A then B, B then A) would never share a node.

**Tests:** `katago runenclosuretests` (`tests/testenclosuresearch.cpp`). They run in a Debug build (asserts on,
`-D_GLIBCXX_ASSERTIONS`) with KataGo's evaluator that uses no network (`debugSkipNeuralNet`: random logits and values,
exact results at the end of the game). Graph search and subtree value bias are on.

- **Tree invariants.** The tests search random games at edges 1-4, 31-32 and 117-120: Blue's opening edge, each edge
  of a turn, and the last turns.
  - Every node's `nextPla` is the history's.
  - Every child's move is legal for that player.
  - The principal variation replays with the history's players.
- **Transposed turns share a graph hash.**
- **Two edges, one mover.**
  - Red plays edges 118-119 as one turn. Blue's single edge 120 ends the game.
  - Red holds two sides of a 3×3 square. Only the other two sides together win. An exhaustive check on the Python
    engine showed this: 2 of Red's 45,601 two-edge turns win, and all others lose.
  - The search runs 100,000 visits with c_score = 0 and flattened random priors (`nnPolicyTemperature` 100). The root
    considers 20 first edges (KataGo's allowMoves): both winners and 18 losers. Red's second edge, the node under
    test, keeps every legal move. With raw random priors, first-play urgency leaves a second edge with a low prior
    untried for thousands of visits, because every tried line loses.
  - The search must select a winning first edge and value it as a Red win.
  - After `makeMove`, the root must still be Red, and it must keep that child's visits (tree reuse between our two
    edges).
  - The search must then complete the square.
  - A search that negates values across Red's own turn would avoid the win.
- **Checked against the bug that it targets:** with `getOpp` put back into `playoutDescend`, the tests fail
  immediately.
- **Result:**
  - All 10 invariant positions pass (55-1,387 nodes each).
  - Red opens with I9-F9 (win value 0.997). The root keeps 93,891 visits after `makeMove`. Red completes the square
    with F6-F9 (0.999).

**Terminal scores:** `ScoreValue::whiteScoreMeanSqOfScoreGridded` asserted Go's half-point grid. Enclosure scores are
exact areas, so a final score is a point mass (mean square = score²). This agrees with the Enclosure score bins
("Score distribution bins").

## Subtree value bias

KataGo puts search nodes into buckets by the mover, the last move, the move before it, and a 5×5 window of point
colours and atari around the last move. The average error of the network against deeper search in each bucket then
corrects new nodes in the same bucket (`docs/KataGoMethods.md`).

- **Enclosure keeps the intention:** the same local tactic in the same local shape shares a bucket.
- **`LocalPatternHasher::getHash`** centres a **7×7** window on the edge's source point, which covers the whole edge.
  It hashes every edge of either player with an endpoint in the window (the owner and the exact edge), and whether
  each edge is protected.
  - Point colours alone would miss which nodes are joined: a closed wall and four loose nodes would share a bucket.
  - Protection has the role of Go's atari flag.
- **Unchanged:** λ = 0.30 (the self-play config), α, and the last two moves of the key. The mover comes from the
  history.
- **`getHashWithSym`** (the pattern bonus table, off by default) refused to run until edge symmetries existed
  (step 4).

## Moves where points are expected (this blocked step 3)

At first, move `Loc`s were 2 + action, and they shared the values 2-420 with points. Code that treats a move as a
point then gives wrong answers with no error for those moves, and reads past the end of the array for the other
moves. KataGo's `assert`s are compiled out in Release.

- **The fix: ranges that do not overlap.** Moves start at `MOVE_LOC_BASE = 421`, so a move used
  as a point is always out of range.
- **Tests run in a Debug build** (asserts on, `-D_GLIBCXX_ASSERTIONS`), so a miss fails loudly.

What the fix found:

- **Arrays of `Board::MAX_ARR_SIZE` indexed by a move:**
  - `rootSymDupLoc` and `rootSymRepresentativeLoc` (`search.h`).
  - the Zobrist table of the subtree value bias (`subtreevaluebiastable.cpp`, on in self-play).
  - the pattern bonus table.
- **Arrays of `MAX_ARR_SIZE` that hold one entry per legal move,** while Enclosure positions have up to about 600
  legal edges:
  - `chooseIndexWithTemperature` (`searchhelpers.cpp`), which selects the played move.
- **Loops over points to list legal moves:**
  - The random-legal-move choosers of `playutils.cpp` loop `loc` over `0..MAX_ARR_SIZE`. With base 2 they saw only
    moves 2-420, the first 419 edges. With base 421, they saw none.
  - So all of self-play's random opening moves would fall in one corner, with no error.
- **Point functions given a move:** `isOnBoard(moveLoc)` in the root checks and in the avoid-move filter
  (`searchhelpers.cpp`), and `isAdjacentToPla(moveLoc, ...)`.

Each such place now has the size `Board::NUM_MOVE_LOCS` or the policy size, or gets the point that a move means. The
same search also found and fixed these:

- **`getEndingWhiteScoreBonus`** (Go's dame-filling bonus, on in self-play) read ownership at
  `whiteOwnerMap[action index]`, past its 361 entries. It now returns 0, which is exactly neutral: it only shifts the
  score.
- **`NNPos::locToPos` meant both** a move's policy index and a point's spatial index. It now takes only moves. Points
  use `NNPos::pointToPos`.
- **Root symmetry pruning** refused to run until edge symmetries existed (step 4). It is off by default and in
  self-play.

## Policy indexing (moved from step 4 into step 3)

The search cannot expand Enclosure moves without it, so it comes first.

- **`NNPos::MAX_NN_POLICY_SIZE`** = 19 × 19 × 48 + 1 = 17,329.
- **`locToPos`** maps a move to its action index. Pass is last, at 17,328.
- **The evaluator without a network (`debugSkipNeuralNet`)** fills every policy entry.
- **`Board` keeps its legal moves as a bitset**, updated with the position, so `isLegal` is a lookup. The evaluator
  asks it once for each of the 17,329 entries.
- **Left for step 4:** the network's inputs, the symmetries, and how the backends lay out the 48 channels per point.

## Network inputs (step 4.1)

Input version **V8** (`NNInputs::fillRowV8`) is the input of the Enclosure network (the Enclosure repository's
`net.input_planes`), laid out in KataGo's way:

- **Spatial, 215 planes** (`NUM_FEATURES_SPATIAL_V8`):
  - 0: on board, all ones. KataGo's backends and PyTorch model take the board mask from channel 0, so it stays first.
  - 1-192: the observation's edge planes, `4 × 48`: each player's edges, then each player's protected or just-placed
    edges. An edge is set at its lower end (the canonical `Edge`'s a), in the plane of its offset, and at its upper
    end, in the plane of the opposite offset.
  - 193-214: the 22 feature planes (`features.h`, `planes.feature_planes`): 9 per player, then 4 history planes.
    Plane 7 of each player holds the turn-safe plane (`planes.turn_safe_edges`), where V1 had held ground. Plane 8,
    V1's openings, stays empty.
  - Until 2026-10-08 these three were environment options (`ENCLOSURE_EDGE_BOTH_ENDS`, `ENCLOSURE_TURN_SAFE`,
    `ENCLOSURE_NO_HELD_GROUND`), on in v2a since 2026-10-03 (v2/docs/design/03-network-inputs.md).
- **Global, 5** (`NUM_FEATURES_GLOBAL_V8`): the observation's scalars. These are the player to move, the edges left in
  the turn / 2, the move count / 120, and both scores / (18 · 18 · 61). The plan puts the scalars here, not in
  constant planes.
- **Absolute colours,** as in the Enclosure network: plane k is Blue's for k = 0. The player to move is a global
  input. (KataGo's Go inputs are relative to the mover. The choice was "our feature planes, as KataGo uses its Go
  features".)
- **Model versions 8-17** keep their meanings for the architecture, but all of them read V8. The fork refuses Go
  networks (and versions 3-7). Go's `fillRowV3`-`V7` stay in the file, unreachable, to keep the diff from upstream
  small.
- **The NN cache hash** already includes the history (step 2's `historyHash`). So the cache keeps apart positions
  that differ only in their history planes.
- `maxHistory <= 0` (KataGo's evaluations without history) drops the 4 history planes.
- **Test:** `tests/enclosure/testnninputs.cpp` against `tests/data/enclosure_inputs.txt` (from the Enclosure
  repository's `v2/tools/export_katago_inputs.py`: 336 positions from 40 archived and 40 random games, with Python's
  reference inputs). Every spatial entry (NHWC and NCHW) and every global entry matches. A changed fixture entry
  fails.

## Symmetries (step 4.2)

**Measured** (`tests/enclosure/testsymmetry.cpp`): for each of KataGo's 8 symmetries, the test played 100 random games
on the board and on its image, with each move mapped. Over 12,000 moves per symmetry, the image game was always the
image of the game: the legal moves (one to one), the edges, the protected and placed edges, the history, and the
exact scores. The feature planes also moved point by point.

| Symmetry | Rules respect it | Fixes the start position |
|---|---|---|
| 0 identity | yes | yes |
| 1 flipY (y → 18 − y) | yes | yes |
| 2-7 (flipX, transposes) | yes | no (Blue's start moves to the right or becomes vertical) |

So the rules are symmetric under all 8. But only the identity and the mirror map real games to real games. Under the
other 6, Blue starts on the right or on a vertical line, which never happens. The plan expected only the mirror, so
the number to use became a decision (see `v2/docs/decisions/build.md`). **Decided: all 8.** The
labels are exact, it is KataGo's method, and it can be reversed without new data. An arena A/B test of 2 against 8
can look at it again. The code handles all 8. `SymmetryHelpers::NUM_SYMMETRIES` sets the count, and KataGo's
numbering puts flipY at 1. So a count of 2 means the identity and the mirror.

**How symmetries work in the fork:**

- **They act on positions, not on input planes.** An edge's image is the canonical edge between the images of its
  endpoints. So under a flip, a vertical edge moves its anchor to its other end, and the edge planes are not a
  permutation of pixels and channels.
  - `SymmetryHelpers::getSymEdge`, `getSymPosition` and `getSymBoard` map edges. `getSymAction` and
    `getSymMoveLoc` map moves.
  - Between two of the mover's own nodes, only one direction is listed as legal. So the image move is looked up on
    the image board.
- **Network evaluation (`NNEvaluator::evaluate`).** The symmetry is chosen where the inputs are filled: at random (a
  generator per thread, seeded from the evaluator's seed), or the default, as the server thread did. The inputs are
  those of the image position, and the backend gets the identity. On return, the policy is mapped back move by move
  and the ownership point by point, before the softmax and the cache. The backends' pixel symmetry is never used.
- **Root symmetry pruning** (`markDuplicateMoveLocs`): a symmetry is valid when the position (edges, protected and
  placed edges) is its own image. Of each set of equivalent legal moves, the search uses the lowest action.
  `computeRootSymRepresentativeLocs` and the analysis output map moves with `getSymMoveLoc`. They map principal
  variations move by move on the image boards (`getSymMoveLocs`).
- **`LocalPatternHasher::getHashWithSym`** hashes the window of the image board around the image move.
- **Tests:**
  - `testsymmetry`: the measurement above. It also does 153,600 round trips and compositions (the inverse maps back,
    s1 then s2 is `compose(s1, s2)`). It checks root pruning at the start, where the valid symmetries are {0, 1} and
    69 legal moves become 36 (3 are on the axis).
  - `katago runenclosuretests`: the test of two edges by one mover, searched on its image under every symmetry, must
    find the image of the winning turn. At the start, root pruning searches one move of each mirror pair, while the
    analysis lists both, each with a legal principal variation.
- **Training-data augmentation** (KataGo's Python, step 6) also mirrors pixels. It must move the anchors of edge
  planes and remap policy channels in the same way. Tests check it against these helpers.

## Score utility (step 4.3, revised later)

**Now: KataGo's score utility, with an Enclosure score unit.** At first, c_score was 0 (the choice
then: win/loss only).

- **KataGo's utility:** factor × (2/π) atan((margin − centre) / (scale × unit)). The margin is the signed final score
  difference.
  - KataGo's unit is the board's side (sqrtBoardArea, 19: the square root of Go's largest score).
  - Enclosure's largest total score is the largest area, 18 × 18, paid at each of 61 turn ends. So the unit is
    sqrt(18 × 18 × 61), about 141 points (`Board::scoreUnit`). (iviazovetskyi's bot uses 150.)
  - Only the search's score utility and the score values that it reports read the unit (`searchhelpers.cpp`,
    `reportedsearchvalues.cpp`, `searchresults.cpp`, and the dynamic centre's cap in `search.cpp`). The first change
    missed that cap, which was then 9.5 points in self-play instead of 70.
  - Komi, handicap and old model versions keep sqrtBoardArea.
- **KataGo's values on top:**
  - Self-play (`selfplay_enclosure.cfg`, which v2a uses, and `selfplay1.cfg`): static 0, dynamic 0.40, centre zero
    weight 0.25, centre scale 0.50 (about 70 points).
  - Play and analysis (`analysis_example.cfg`, `match_example.cfg`): static 0.10, dynamic 0.30 (centre zero weight
    0.20, scale 0.75: about 105 points), as upstream's defaults.
- **The gauntlet's `analysis_enclosure.cfg`:** first 0, then KataGo's 0.10 and 0.30, in a new rating pool.
- The score value tables clamp scores beyond Go's range. With the unit, the scaled scores stay within them.

## Score distribution bins

842 bins, as in KataGo, each 12 points wide. They cover margins of ±5,052: every margin in `net1`'s self-play, and 99%
of the full-length human games (the measurements are in `v2/docs/decisions/build.md`). Larger margins go to the end
bins. Only the training rows and the Python head use the distribution (step 6).

## The network's policy head, export and CPU backend (step 5)

- **Policy layout:** every policy output has 48 channels per point, channel = output × 48 + offset. So the policy
  entry (y · 19 + x) · 48 + offset (the action index) reads the NHWC output in place. Pass stays one logit per output.
  - PyTorch (`model_pytorch.py`): `conv2p` has `num_policy_outputs × 48` channels. The head returns (N, outputs,
    17,329) with pass last. `NUM_OFFSETS` and `policy_len` hold the numbers.
  - Export (`export_model_pytorch.py`): the chosen outputs (regular and optimistic, as upstream) are written as blocks
    of 48 channels.
  - C++ (`desc.cpp`): `p2Conv.outChannels` must be `policyOutChannels × 48`. The Eigen backend sizes its buffers and
    decodes the blocks (with the optimism blend per entry).
  - CUDA (`cudaandrocmbackend.inc`):
    - the host and device policy buffers hold 48 channels per output.
    - the pass buffer holds one logit per output.
    - the decode reads NHWC or NCHW.
    - the backend refuses every symmetry except the identity.

    Tested with `katago runenclosurenntests MODEL INPUTS OUT [fp16] [nhwc]` on the RTX 5080, against PyTorch:
    - FP32 with TF32 off (`NVIDIA_TF32_OVERRIDE=0`), NCHW and NHWC: within 7e-6, as the CPU.
    - FP32 with TF32 on (cuDNN's default on this GPU): within 3e-3.
    - FP16, NHWC and NCHW: within 6e-3.

    TensorRT came later.
- **Inputs in PyTorch:** `modelconfigs` gives 215 spatial and 5 global inputs. The model's first layers read them from
  there, not from Go's hard-coded 22 and 19.
- **Go's score parity** (the komi feature times a parity vector in the score-belief head) stays as a no-op: the vector
  is zero. Here the last global input is a score, not komi.
- **Score-belief bin centres** are in points: 12 × (bin − 421 + 0.5) (the Enclosure score bins).
- **No result:** the fork never sets `isNoResult`. Its rules (area scoring, positional ko) take KataGo's branch that
  sets the no-result output to zero. So win and loss share the probability, as under Go's area rules.
- **Test:** `katago/python/enclosure_nn_check.py` with `katago runenclosurenntests`. The test initialises a b6c96 at random.
  Then it perturbs every parameter and statistic, so that no layer is an identity. It exports the model and
  evaluates it in C++ (Eigen, `NNEvaluator`) and in PyTorch on the 332 unfinished positions of
  `tests/data/enclosure_inputs.txt`.
  - The largest differences: policy 6e-6 (relative), value 6e-7, score 4e-6 (in units of the model's score
    multiplier, 240 points after the score scale below), ownership 7e-6. The tolerance is 1e-3.
  - A model with other weights fails at every position.
- **Training augmentation** (`katago/train/enclosure_symmetry.py`, used by `data_processing_pytorch.py`):
  - point planes and the value targets move as pixels.
  - the edge planes and the policy and q-value targets go through index maps. A stored edge goes to the lower end of
    its image. A move between two of the mover's nodes (read from the node feature planes before the symmetry) goes
    to the lower end of its image. Any other move goes to the image of its source.
  - Symmetries have the same numbers as in C++.
  - Test: `katago/python/enclosure_symmetry_check.py` against `testsymmetry -dump`. For 40 positions × 8 symmetries,
    the augmented inputs equal the C++ inputs of the image position, and every legal move lands on the image from
    `getSymAction`. Scrambled symmetry labels fail.

## Score scale (step 6)

**Decided:** scores stay in raw points, and every constant that KataGo tuned to Go's point scale is scaled by
`SCORE_SCALE` = 12 (C++ `NNPos::SCORE_SCALE`, Python `model_pytorch.SCORE_SCALE`). This is the width of the
Enclosure score bins against KataGo's bins of 1 point, from the measured margins. Training then weighs every score
term exactly as KataGo does on scores of Go's size ("scale-invariant retune"). The rule:

- quantities in points: × s.
- squared points: × s².
- each loss weight: ÷ the growth of its loss. A Huber loss whose argument and delta both scale by s grows by s². The
  squared-error loss grows by s⁴.

| Constant | KataGo | Enclosure | Why KataGo selected it |
|---|---|---|---|
| Output multipliers: score mean, stdev, lead, short-term scores | 20 | 240 | Paper A.5: "an arbitrary reasonable scaling factor so that on typical data the neural net would only need to output values around unit scale" |
| Short-term score error multiplier | 150 (squared points) | 21,600 | The same aim, for a squared error (model v13 rescaled it) |
| Score distribution range | ±(361 + 60) points | ±5,052 points | Paper A.5: "an upper bound for the plausible final score difference", the board area plus 60 for komi |
| Score target cap | 421 | 5,052 | The same bound, so that targets stay in range |
| Q-value score storage in int16 | ×60 | ×5 | ×60 keeps ±421 within 32,000 at a precision of 1/60 point. ×5 keeps ±5,052 within it. |
| Huber delta: score mean | 12 | 144 | Code comment: Huber is "meanlike locally and ... medianlike for very large possible losses". The specific deltas: source not found yet (to trace in KataGo's commits and issues). The paper's first version used 10. |
| Huber delta: lead / stdev / q-value score / short-term score | 8 / 10 / 12 / 12 | 96 / 120 / 144 / 144 | As above |
| Huber delta: short-term score error | 100 (squared points) | 14,400 | As above, in squared points |
| Floor of the irreducible error: short-term score | 1e-4 ("one hundredth of a point", squared) | 1.44e-2 | A regulariser: a hundredth of the unit |
| Loss weights: score mean 0.0015, lead 0.006, stdev 0.001, short-term score 0.0004, q-value score 0.0008 | as listed | ÷144 | Source not found yet (to trace in KataGo's commits and issues). The paper's first version used 0.004 for the self-predictions. The division keeps each term's share of the total loss, whatever the original reason. |
| Loss weight: short-term score error | 0.00002 | ÷20,736 | A loss of the fourth degree |
| Score distribution pdf and cdf losses | 0.02 each | unchanged | Over bins, not points (paper: pdf and cdf 0.02) |

The paper's base weights, for reference: value 1.5, policy 1.0, opponent policy 0.15, ownership 1.5/b², score pdf
and cdf 0.02 each, L2 3e-5.

## Training rows and the trainer (step 6)

- **Rows** (`trainingwrite.cpp`):
  - policy targets of 17,329 moves.
  - the Enclosure ownership target (see "Spatial targets") in channel 0, × 120 with KataGo's stochastic rounding.
    It is computed from the game's positions after the row, at every turn end.
  - score targets, capped and binned as above.
  - Go's seki, future-position and area targets are zero, with weight 0. (The seki loss is set to zero in Python,
    because its weight is the ownership weight.)
  - no-result never occurs.
- **History dropout** (in KataGo's style), in `data_processing_pytorch.apply_history_dropout`:
  - KataGo's draws (each item stops with probability 0.02, older ones first) on the 4 history planes.
  - the edge planes of the current and previous turns carry protection, and they are never dropped.
  - measured on 200,000 rows: planes kept 98.0 / 96.0 / 94.1 / 92.3 %, against 0.98^k.
- **Soft-policy mask** (metrics): a move is on the board when its source point is (48 moves per point).
- **Tests:**
  - `katago runenclosurerowtests OUT [SEED]` plays one self-play game with the evaluator that uses no network, and
    writes its rows as text.
  - The Enclosure repository's `v2/tools/check_katago_rows.py` replays the game in its engine and checks every row: the inputs equal
    `input_planes`, the policy targets are on legal moves, the ownership is right within the storage rounding, the
    result, the score and the mean of the score distribution are right, the q-values are in range, and the Go targets
    are zero. Three games, 360 rows: 0 failures.
  - `--npz` writes the rows for `katago/python/enclosure_rows_check.py`. It reads them with KataGo's
    `read_npz_training_data` (symmetries and history dropout on). Then it runs the forward pass, the metrics and one
    backward pass (no optimizer step). 21 batches: the shapes are as expected, and every loss and gradient is finite.
    The losses that only Go has are 0.
  - Real `.npz` files need libzip (`libzip-dev`, installed before step 8). Before that, the test used
    the writer's text output.

## KataGo's training and self-play knobs: what V2 uses

The fork runs KataGo's trainer and self-play as published. So most of these are KataGo's defaults, not decisions.
This section lists them so that a reader sees them as choices. "On" and "off" describe v2a (`v2/runs/v2a.env`,
`selfplay_enclosure.cfg`, `train.log`). To change one, use KataGo's own flag or key.

- **Training** (`python/train.py`, `python/shuffle.py`):
  - **Learning rate:** KataGo's base rate (`pslr_batch` 3e-5 a sample), times a third for the self-play trainer
    (`-lr-scale 0.333`). No schedule. Available: `-lr-scale` (a multiplier), `-lr-schedule` (explicit steps by
    samples trained), `-lr-scale-auto`/`-auto2` (KataGo's step schedules, with their first cuts at 20M-550M samples),
    `-lr-cycle-*` (cyclic).
  - **Rates by parameter group: on, the defaults.** The output layers of all heads together run at half the trunk's
    rate (`-head-lr-factor 0.5`). Biases and norms have their own factors. There is no rate per head.
  - **Head loss weights: on**, KataGo's (`metrics_pytorch.py`, with the score weights scaled by `SCORE_SCALE`, see
    "Score scale"). Under SGD, a head's loss weight is in effect the learning rate of that head's own layers. It also
    sets how hard the head pulls on the shared trunk. So it is the lever for one head. (Use KataGo's
    levers, this one or `-lr-schedule`, not new code.)
  - **Optimizer:** SGD with momentum 0.9 and lookahead (0.5, 6). `-use-muon` and `-use-adamw` exist, unused.
  - **Averaged exported weights (SWA): on**, a sample every 80,000 positions (`-swa-period-samples`).
  - **Weight-decay normalisation and gradient-norm clipping: on** (`wdnormal_batch`, `gnorm_cap_batch` 2,500).
  - **Training window: on.** It grows with the data (`-expand-window-per-row 0.4`, `-taper-window-exponent 0.65`).
  - **Training passes per row: 5** (`MAX_TRAIN_PER_DATA`). At 8, the outcome heads overfit
    ([strength.md](../v2/docs/decisions/strength.md) "The outcome heads overfit").
  - **Validation: on.** `shuffle.sh` keeps 1% of the files out of training (`VALIDATE=1`), and `metrics_val.json`
    holds their losses. (Upstream's `synchronous_loop.sh` turns validation off. Then the loss curves are training
    losses only, and they rise when the data gets harder, even while the network improves.)
  - **Batch** 256, FP16 training.
- **Self-play** (`selfplay_enclosure.cfg`):
  - **Playout cap randomization: on.** 25% of searches are full searches at 200 visits. The rest are at 40
    (`cheapSearchProb 0.75`, `cheapSearchTargetWeight 0`: only full searches train the policy). 200/40 matches
    iviazovetskyi's self-play. It is a named departure from KataGo's 600/100 and 1000/200.
  - **Surprise weighting: on**, `policySurpriseDataWeight 0.5`, `valueSurpriseDataWeight 0.1`.
  - **Variety: on**, with Go's values: early forks (`earlyForkGameProb 0.04`), root Dirichlet noise (total 10.83),
    root policy temperature 1.25 early and then 1.1, and move temperature 0.75 early, decaying to 0.15 with a
    half-life of 19 (`chosenMoveTemperature*`).
  - **Search:** `cpuctExploration 1.1`, `fpuReductionMax 0.2`, `valueWeightExponent 0.5`, `subtreeValueBiasFactor
    0.30` ("Subtree value bias" above).
  - **Shortcuts: on.** `reduceVisits` at a winrate of 0.9 held for 3 turns, and `switchNetsMidGame`.
  - **Komi randomisation: on** ([komi.md](../v2/docs/decisions/komi.md)). **Lead estimation: on**, as KataGo's
    (`estimateLeadProb 0.05`).
  - **Gatekeeper: off.** v2a runs no gatekeeper, so no new network is tested before self-play takes it. The
    gauntlet rates the networks afterwards.

## Spatial targets (step 6)

**Built ("add all of our ideas and just run").** The targets use KataGo's existing slots, so the rows keep
their format, and old and new rows mix in the shuffle:

- **1. Future nodes** in C2-3: KataGo's future-position code as it is (`board.colors` marks each player's nodes, and
  the boards are one edge apart). The horizons are 8 and 32 edges later. The loss is KataGo's (tanh, squared error,
  [1, 0.25], ÷√area, 0.25), with weight C33 = 1.
- **2. Edges that get cut** in C1 (the seki slot): +1 / -1 / 0 on points on edges, and 2 on points on no edge.
  - The seki head's three sign logits, with cross-entropy averaged over the points on edges. (Not the whole board:
    the question exists only there.)
  - Weight 1 times the ownership weight, on rows flagged in C68. Rows written earlier have zeros, which would teach
    that nothing is ever cut.
  - KataGo's adaptive seki scale and neutral logit are not used.
- **3. Region occupancy** in C4 (the scoring slot): one horizon, the 8th turn end later (4 rounds, or the last turn
  end if the game ends sooner), × 120. KataGo's area loss (squared error with a square-root transform, 0.25), with
  weight C34 = the value weight.
  - The 1-round horizon is left out on its merits too. The current regions are an input, the future nodes cover 8
    edges ahead, and the ownership target covers the whole game.
- `v2/tools/check_katago_rows.py` checks the rows (it computes all three again from the engine).
  `enclosure_rows_check.py` checks KataGo's trainer (every loss finite).

The design as first written:

**Decided:**

- KataGo's spatial targets that only Go has (seki, future stone positions, final area/territory) get weight 0. Their
  heads stay.
- **The ownership target:** per point, the mean over the remaining turn ends of +1 inside the mover's regions, −1
  inside the opponent's, and 0 otherwise.
- We wanted targets specific to Enclosure in place of Go's. The sections below work them out. They were to
  be added one at a time (weight 0 until the arena tested each one).

KataGo's reasons, from its code comments: auxiliary spatial targets give the trunk dense gradients per point about
the future, beyond the one policy and value number per position. Future positions use a fixed horizon, so that a
fixed number of points are "wrong" whatever the board size. The far horizon has a quarter of the weight, because it is
harder to predict.

### 1. Future nodes (KataGo's future positions, for nodes)

- **Target,** from the side of the row's mover, per point: +1 for one of the mover's nodes, −1 for one of the
  opponent's, 0 otherwise. The target uses two horizons, near and far. Past the end of the game, the final position
  is used.
  - KataGo takes the board 8 and 32 moves later (`trainingwrite.cpp`), which is 4 and 16 rounds of Go. An Enclosure
    round is 2 turns, 4 edges.
  - A mapping by rounds gives 16 and 64 edges, more than half of a game of 120 edges. A scaling by game length
    (about 250 moves in Go) gives about 4 and 16 edges.
  - The choice belongs with the other time horizons below.
- **Head and loss:** KataGo's `futurepos` head (2 channels) and its loss, unchanged: the squared error of tanh, horizon
  weights [1, 0.25], summed and divided by √area, weight 0.25.
- **What it teaches:** where each side will build next, and which nodes disappear when their edges are cut. It is a
  spatial summary of the next moves, as in Go.

### 2. Edges that get cut (the counterpart of Go's dead stones and seki)

- **Target:** per lattice point on a current edge (its endpoints and the lattice points that it goes through), 3
  classes:
  - no current edge there is cut before the game ends.
  - one of the mover's edges there is cut.
  - one of the opponent's edges there is cut. (If both, the mover's.)

  An edge counts as cut when that edge is removed, even if an identical edge is placed again later.
- **Head and loss:** the 3 sign logits of the `seki` head, with cross-entropy per point divided by the board area.
  A plain weight, tuned when tested, replaces KataGo's adaptive seki scaling. That scaling follows how rare seki is,
  which does not carry over.
- **What it teaches:** which walls are weak. A static input plane cannot show this fragility: protection lasts one
  turn, and a cut depends on the opponent's reach and tempo.

### 3. Region occupancy at several horizons (both players together)

- **Point:** a target of next-turn region gain leans greedy. So the target takes both players together,
  over horizons out to the end of the game.
- **Target,** from the mover's side: per point, +1 inside the mover's regions, −1 inside the opponent's, and 0 in
  neither or both. The target uses the turn end 1 round and 4 rounds later.
  - The change from now needs no separate target. The current regions are an input (the enclosed feature planes), so
    to predict the future occupancy is to predict the gain and loss of both players at once.
  - The ownership target (the mean over all remaining turn ends) is the end-of-game member of the same ladder. So the
    network learns near, middle and whole-game occupancy, and no single horizon dominates.
- **Head and loss:** a new 2-channel 1×1 head, like `futurepos`, with the form of its loss (weights [1, 0.25],
  ÷√area). The `scoring` head has only one channel, so it stays at weight 0.
- **Why not "next turn's gain" alone:** a short horizon rewards whatever closes a region now. The 4-round horizon and
  the ownership target carry what the opponent does in return.

**Time horizons: divided by what they measure.**

KataGo sets its time constants from the board area. In Go, the area sets both the length of the game and the reach of
its tactics. Enclosure keeps the 19×19 board but has a shorter game.

The only source is KataGo's methods document. The short-term targets average "roughly the next 6 turns, 16 turns, and
50 turns ... on a 19x19 board, and correspondingly shorter for smaller boards in proportion to the board area". It
gives two purposes:

- value feedback with lower variance (bias-variance).
- a prediction of what a search concludes "a few turns later", which the uncertainty weighting and the optimistic
  policy use.

Both purposes count decisions, not fractions of a game.

- **Look-ahead horizons are KataGo's 19×19 formulas, not scaled, counted in edges** (the decisions in our search
  tree). This covers the short-term value and score targets (about 6, 20 and 64 edges, which are 3, 10 and 32 turns),
  the short-term error targets (about 6 edges), value-surprise weighting, and the horizons of the future-node design
  (8 and 32 edges).
- **Game-phase constants scale by the length of the game** (below): the earliest resignation, early forks and opening
  moves.
- **The variance-time multiplier** keeps outputs near unit scale. It is scaled provisionally. It will be set from the
  measured size of its target when self-play data exists.

An intermediate version scaled everything by the length of the game (2, 7 and 23 edges). But 2 edges,
about one turn, was too short, and this led to the division.

Game-phase constants:

- `Board::TIME_SCALE` (Python `model_pytorch.TIME_SCALE`) = 120 / 328 ≈ 0.366: Enclosure's 120 edges against the
  measured mean length of KataGo's own 19×19 self-play games without handicap or setup stones.
  - Source: the kata1 training games on katagoarchive.org. A sample of 732 games: mean 328.7, median 329, 10th-90th
    percentile 294-363. A later sample of 51,094 games: mean 328.1, median 327, 292-366.
  - KataGo plays games to the end and does not resign, so its games are longer than human games.
  - This measurement replaced an earlier guess of 250 moves (find the source of every number).
- `Board::GAME_LENGTH_AREA` = 361 × 0.366 ≈ 132 replaces the board area in the game-phase constants:
  - the earliest self-play resignation (A / 5).
  - the early-fork move (`earlyForkGameExpectedMoveProp` × A).
  - the number of opening policy moves (`policyInitAreaProp` × A).
- Provisionally, until its target is measured: the variance-time output multiplier (40 → 14.6) and its Huber delta
  (50 → 18.3), with the loss weight ÷ 0.366².
- The look-ahead formulas keep the board area. KataGo's code comment on the short-term constants says that they
  "used to be 'nicer' numbers 0.18, 0.06, 0.02" until a fix to the functional form, with values nudged to keep
  9×9-19×19 unchanged.
- The specific numbers are what made sense in Go, not only their scale. They can need tuning for
  Enclosure later.

**Data:** all three targets come from the finished game. Self-play already keeps every position of the game. At the
end, the writer reads the targets of each row from the positions h rounds later: the nodes, the region masks from the
engine's `enclosed_area`, and which edges survive.

## Sizes at 17,329 actions (steps 4-6)

- **`NNOutput` stores a dense policy** of `MAX_NN_POLICY_SIZE` floats, 69 KB per result.
  - Search trees hold one per expanded node. Self-play's 128 games at 600 visits use about 5 GB.
  - The self-play config's `nnCacheSizePowerOfTwo = 21` would use 145 GB. It must become smaller (2^15 is about
    2.3 GB). This is a setting, not a code change.
  - A sparse policy (legal moves only, about 30 times smaller) waits until memory is measured.
- **Training rows:**
  - The two policy targets are 69 KB a row (KataGo's 19×19 rows are 5.5 KB in total), and the q-value targets are
    104 KB. The file format stays KataGo's. The policy is mostly zeros, so it compresses on disk.
  - The shuffle's `-approx-rows-per-out-file` (upstream 70,000 rows, about 10 GB of float32 policy per file in the
    trainer) and its process count set the memory.
  - Step 6 measures both before it chooses values.

## BoardHistory surface (from the survey of callers outside `game/`)

- **Kept, with an Enclosure meaning:**
  - `moveHistory`, `isLegal`, `makeBoardMoveAssumeLegal`, `makeBoardMoveTolerant`, `isLegalTolerant`,
    `getRecentBoard`, `clear`.
  - `isGameFinished` (after line 120), `winner`, `finalWhiteMinusBlackScore` (red minus blue), `isNoResult` (the
    rules' draw when the player to move has no legal edge).
  - `presumedNextMovePla`, `initialBoard`, `initialTurnNumber`, `getCurrentTurnNumber`, `endAndScoreGameNow`,
    `setWinnerByResignation`, `isResignation`.
- **Neutral, for Go only:**
  - `encorePhase` = 0, `superKoBanned` and `koRecapBlocked` all false.
  - `setKomi` and `currentSelfKomi` (komi 0), `hasButton` = false.
  - handicap functions: none.
  - `passWouldEndPhase` and `passWouldEndGame` false (a pass is only the site's timeout).
  - `endGameIfAllPassAlive`: nothing to do.

## Verification (step 2)

The game layer must equal the engine of the Enclosure repository, which is verified against both sites. The legal
moves, the results and the scores must agree, move by move, on the archived games (the 3,801-game native diff) and on
random games. `tests/` has an Enclosure test that replays them from a fixture file that the Enclosure repository
exports.

## Build and resources

The CPU build is in `~/ws/game-ai/enclosure/katago-build`, with the Eigen backend. The CUDA build and its use of the
machine are in [v2/README.md](../v2/README.md) and [docs/toolchain.md](../docs/toolchain.md).
