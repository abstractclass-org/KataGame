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

Tests (behaviour, run with asserts on):

- **Two edges, one mover:** near line 120, blue has a pair of edges that
  wins and pairs that lose. A search with a uniform network
  (`debugSkipNeuralNet`) and terminal results must choose the winning first
  edge and value it as a win for blue. Negating across the turn would pick
  the worst edge.
- **Tree reuse between our two edges:** after `makeMove` of the first edge,
  the root keeps that child's visits and its `nextPla` is still the same
  player.
- **Transposed turns share a node** in the graph search (A then B equals B
  then A).
- **PV players** follow the 1-2-2 order.

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
