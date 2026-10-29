# The game interface: the C++ reference

**Status: decided, built.** KataGame plays a game that supplies the game interface. The build chooses one game, as it
chooses one backend. This file is the C++ reference: the members, the conventions, the game layer, the front's changes
to KataGo for every game, the checks, and how to add a game. Its last part is the map of the code that reached
Enclosure directly, with the member that replaced each reference. The decision and its reasons are design 07 of the
engine repository (`docs/design/07-game-interface.md`), and the record of what the second game showed is its design 06,
step 6. Enclosure's own design stays in [ENCLOSURE.md](ENCLOSURE.md).

The interface exists. Enclosure and Connect-4 (`cpp/game/connect4/`) implement both parts, and each game has its
`katago` binary. The front (`Board`, `BoardHistory`), the search, self-play, GTP and the analysis engine reach the
rules through the rules part. The network code reaches the encoding part: the policy layout and the backends
(`NNPos`), the inputs (`NNInputs`), the symmetries (`SymmetryHelpers`) and the training rows (`TrainingWriteBuffers`).
No C++ code outside the game layer names a game, and a check keeps it so ("The game layer"). The trainer reads a
game's facts from the game's module (`python/katago/train/games/`), and a model file names its game ("The model file's
game"). The training scripts (`python/selfplay/`) run a run of any game, and `katago version` names the binary's game
("The training scripts"). The map below gives each step's work.

* [The files](#the-files)
* [Choosing the game](#choosing-the-game)
* [The game layer](#the-game-layer)
* [Conventions](#conventions)
* [The front: KataGo's changes for every game](#the-front-katagos-changes-for-every-game)
* [The rules part](#the-rules-part)
* [The network encoding part](#the-network-encoding-part)
* [The bot part](#the-bot-part)
* [The checks](#the-checks)
* [Adding a game](#adding-a-game)
* [The map: Enclosure outside the game layer](#the-map-enclosure-outside-the-game-layer)

## The files

| File | What it holds |
|---|---|
| `cpp/game/gameinterface.h` | The contract: every member with its meaning, the two static checks, `GameInterface::checkRules<R>()` and `checkEncoding<E>()`, and `GameInterface::symPoint`, KataGo's symmetry of a point. |
| `cpp/game/chosenrules.h` | The choice of game, rules part only: the namespace alias `chosengame` and `GameRules`. The Python binding builds against it. |
| `cpp/game/chosengame.h` | Both parts: `chosenrules.h` and `GameEncoding`. |
| `cpp/game/<game>/gamerules.h` | The game's rules part: `struct GameRules` in the game's namespace, then its check. Enclosure's own position field of the analysis engine is in `game/enclosure/queryposition.h`. |
| `cpp/game/<game>/gameencoding.h` | The game's network encoding part: `struct GameEncoding`, then its check. |
| `cpp/game/botinterface.h`, `cpp/game/chosenbot.h` | The bot part's contract, its check `GameInterface::checkBot<B>()`, and the chosen game's `GameBot`. |
| `cpp/game/<game>/gamebot.h` | The game's bot part (optional): `struct GameBot`, then its check. |
| `cpp/search/alphabeta.h` | The alpha-beta search for every game with a bot part. Only the Python binding compiles it. |
| `cpp/game/board.*`, `cpp/game/boardhistory.*` | The front: KataGo's `Board` and `BoardHistory`, which the search calls. They reach the game's rules through the members only. |
| `cpp/game/<game>/ownfiles.txt` | The game's own code outside the game layer (optional): the files that only the game's binary compiles, and that may name the game. |
| `cpp/tests/checkgamelayer.py` | The check that no code outside the game layer names a game. Each build of `katago` and `tests/enclosure/build.sh` run it. |
| `cpp/tests/checkgamelayer_test.py` | The test of the check, on temporary trees with planted uses (Enclosure, Connect-4 and a made-up game with a short name). |
| `cpp/tests/enclosure/testgameinterface.cpp` | Enclosure's members, against the front's code where the front still has a copy, and the defaults of the optional members with a stub game. |
| `cpp/tests/enclosure/testgameareas.cpp` | The look-ahead area and the komi area (optional members of the encoding part) through the front, for the build's game: the short-term value targets' lambdas and the komi draw's standard deviation, against each game's pinned areas, and the defaults with stub games. |
| `cpp/tests/enclosure/testcheapsearch.cpp` | Self-play's full and cheap searches (`play.cpp`'s `GameRunner`), for the build's game: the share of full searches among the actions that draw one, and with `cheapSearchPerMove`, the search of an action that continues a move. |
| `cpp/tests/enclosure/testdiversitysearch.cpp` | AZdb's diversity term in the search, for a build's game that gives AZdb's statistics (Blueprints): member 0's search against KataGo's, D on chains, a transposition, the member's choice by its reward, each member's evaluations, the refusals and the tree's clear. |
| `cpp/tests/enclosure/testkomi.cpp` | Komi through the front, for the build's game: the result and the value targets at komis that decide a game, the global inputs at the history's komi, the even-komi search's bracket (an optional member of the encoding part) against KataGo's code and each game's pinned bracket, the two rules where the win-loss value keeps its sign up to the limit (no even komi, or KataGo's bracket end), each game's pinned lead step, and a self-play config's komi draws. |

All members are static, and every file is a header. So the compiler sees the same code as in a build for one game, and
the WebAssembly build, the binding and the standalone tests need no new source file.

## Choosing the game

- **CMake:** `-DGAME=enclosure` (the default). The games are the folders of `cpp/game/` that hold a `gamerules.h`. An
  unknown name stops `cmake` with the list of games. A game without `gameencoding.h` also stops it: `katago` needs
  both parts, and the Python binding needs only the rules part. The option defines `KATAGAME_GAME=<name>`.
- **The headers:** `chosenrules.h` includes `<name>/gamerules.h` and makes `chosengame` the alias of the namespace
  `<name>`. `chosengame.h` includes `<name>/gameencoding.h`. A game's folder and its namespace have the same name.
- **Other builds** (the WebAssembly module, the fork's standalone tests) define nothing and get Enclosure. So every
  build command of today still works. The Python binding defines `KATAGAME_GAME` for each game that it builds
  (`core/binding/build.sh` in the engine repository).
- **Every build of `katago` checks both parts:** `board.h` and `rules.h` include `chosengame.h`, so each file that
  uses the front checks both parts. `rules.cpp`, which includes no `Board`, includes `chosenrules.h` first, before
  `rules.h`: it shows that the rules part compiles without the encoding part.
- **The game's own sources:** CMake compiles the `.cpp` files that `game/<name>/ownfiles.txt` lists into the binary of
  that game only ("The game layer").

## The game layer

The game layer is what a game supplies, and the only code that names a game:

- **The game's folder, `cpp/game/<game>/`:** its rules part, its encoding part, its bot part, and the code of its own
  that they call (Enclosure's `engine.h`, `features.h`, `notation.h`). This code reaches no code of the front: the
  front calls it.
- **The choice of game,** `cpp/game/chosen*.h`. They name the default game, Enclosure, for a build that names none.

Every other file of `cpp/` (except `external/`) reaches the game through the interface only: `GameRules`,
`GameEncoding` and `GameBot`, and their members. The front (`Board`, `BoardHistory`, `rules.*`), the interface's
headers (`gameinterface.h`, `botinterface.h`), the search, the network code, the commands and the tests of every game
are in this part.

**A game's own code outside the game layer.** Some code is a game's own, and it uses the front, so it cannot be in the
game's folder: a game's own commands and its tests of the search. The game lists these files in
`game/<game>/ownfiles.txt`, one path a line, with the reason for each. Only these files may name the game, and only
that game's binary compiles them. A command of the game's own registers itself with a static `MainCmds::GameCommand`
(`main.h`): `main.cpp` then dispatches and lists it, and names no game. Enclosure's own files:

| File | What it is |
|---|---|
| `command/enclosureprotocol.cpp`, `.h` | Enclosure's engine protocol (design 04 of the engine repository): the describe action with its edges, nodes and regions, the cut edge, the outcome's text, the position check, the history planes' edges. The WebAssembly module builds it too. |
| `command/enclosureengine.cpp` | The protocol's native command, `katago enclosure-engine` (a `GameCommand`). |
| `tests/testenclosuresearch.cpp` | The search under two-edge turns, `katago runenclosuretests` (a `GameCommand`). |
| `tests/enclosure/` | The standalone tests of Enclosure's members through the front (`tests/enclosure/build.sh`). |

Blueprints' own files:

| File | What it is |
|---|---|
| `command/blueprintsprotocol.cpp`, `.h` | Blueprints' engine protocol (the engine repository's `games/blueprints/README.md`, "The browser build"): Enclosure's requests in strategyjam.org's moves, with a placement outside the view played through `place_at_site`, the cells and the hands in the site's form, each move's footprint, captured cells and points, an edit's values, and the ownership with its view's place. An analysis searches the root's views (`rootViews`, `program/rootviews.cpp`) in slices, and `describe` gives the cover. The WebAssembly module builds it too. |
| `command/blueprintsengine.cpp` | The protocol's native command, `katago blueprints-engine` (a `GameCommand`). |

**The check** (`cpp/tests/checkgamelayer.py`, Python's standard library only) fails when a file outside the game layer:

- names a game's namespace (`enclosure::`, `namespace enclosure`, `using namespace enclosure`, an alias such as
  `namespace e = enclosure`).
- includes a header of a game's folder (an `#include` path with `enclosure/` in it), or a header of a game's own code
  outside the game layer (`command/enclosureprotocol.h`), whose code only that game's binary has.
- names the chosen game's namespace (`chosengame::`) beyond the aliases of `chosen*.h`.
- names a member of `GameRules`, `GameEncoding` or `GameBot` that the interface does not have (Enclosure's
  `GameEncoding::sym_edge`). The interface's members are the names that its checks detect: `G::<member>` in
  `gameinterface.h` and `botinterface.h`.
- names an optional member directly, beside its helper: a game may lack it. The optional members are those that a
  helper of `gameinterface.h` or `botinterface.h` calls after its `if constexpr` on the member's detector.
- reads a field of the game's position (`board.pos.edges` names Enclosure's `Position`).

A name counts only as a namespace or as a folder of an include path. The fork's code keeps Enclosure's name in
comments, strings, macros and the names of its generic commands, so a word of a line does not count. A namespace must
have the game's exact name: `hexagon::` is not `hex::`, and `cargo::` is not `go::`. `std::hex` lacks the trailing `::`.
A namespace use still counts after another scope (`app::hex::`) or at the global scope (`::hex::`). The engine
repository's Python checks (`name_pattern` in `pipeline/tests/checkgamenames.py`) share this rule for namespace uses.
They also check identifiers and strings, so they flag a method such as `bytes.hex()`. This C++ check does not flag it.
KataGame cannot import the Python rule, so it keeps its own copy. Each copy names the other.

A game's own file may name its own game, and no other: `tests/enclosure/` cannot include Connect-4's headers. The
check also fails when a game's list names a file that does not exist. Each build of `katago` runs it before it
compiles (the CMake target `checkgamelayer`, when CMake finds Python 3), and so does `tests/enclosure/build.sh`. On
the drive of the training machine (`/mnt/f` in WSL), it reads the 266 files in about 2 s. Connect-4's binary
(`-DGAME=connect4`) builds from the same files as Enclosure's, its own folder aside.

```
python3 tests/checkgamelayer.py            # from katagame/cpp
python3 tests/checkgamelayer_test.py       # its test
```

## Conventions

- **Players:** 0 is the first player (KataGo's Black), 1 the second (White). Enclosure's Blue is 0.
- **Actions:** an `int` from 0 to `NUM_ACTIONS - 1`. Pass is the last, `PASS_ACTION`. Every game has pass. A game whose
  rules have no pass never lists it as legal (Connect-4).
- **Actions on the largest board:** a game numbers its actions on a board of `MAX_BOARD_LEN`, point by point:
  `(y * MAX_BOARD_LEN + x) * POLICY_CHANNELS + channel`, pass last. So an action keeps its number on every board size,
  and `NUM_ACTIONS` is `MAX_BOARD_LEN² × POLICY_CHANNELS + 1`.
- **One copy of the formula in each game:** the front never writes it. It uses an action as its policy index, and it
  asks `action_point` for a move's point. A game writes its numbering once, and its members and its own code call
  that copy: Enclosure's `action_at` and `action_source` (`engine.h`), Connect-4's `action_at`, `action_x` and
  `action_y`. `testgameinterface` checks Enclosure's `point_action` and `action_point` against the policy index.
- **An action is its policy index:** the network's board is the game's largest board (`nnXLen = nnYLen =
  MAX_BOARD_LEN`). So the network's policy index of an action's point and channel is the action itself. The search,
  the evaluator and the backends use an action as its policy index and need no map. The encoding part's check refuses
  a `NUM_ACTIONS` that breaks this rule.
- **Points:** `(x, y)`, `0 <= x < x_size`, `0 <= y < y_size`, with `y = 0` at the bottom (row 1 in the notation).
  Arrays of points are `[y * x_size + x]`.
- **Symmetries:** KataGo's 8 and its numbers: bit `0x2` flips x, bit `0x1` flips y, then bit `0x4` transposes
  (`GameInterface::symPoint`). A game allows a subset. The identity is always in it. A game that allows a transpose
  plays on square boards only: the front's on-board plane does not change under a symmetry.
- **No exceptions:** the WebAssembly build has none. A parser returns `false` or `-1` and gives the reason.

## The front: KataGo's changes for every game

KataGo's methods, parameters and structure stay as they are. The fork replaces Go's rules with the game interface. Every
other file changes only where it relied on Go: moves as points, turns that alternate, Go's input features, the policy's
size, and Go's score scale. Concepts that only Go has (ko, superko, encore phases, handicap, the button, pass-alive
areas) keep their names and signatures, fixed at neutral values. So their callers compile unchanged, and nothing in them
branches on a game.

The fork made these changes for Enclosure, its first game, and every game now uses them. [ENCLOSURE.md](ENCLOSURE.md)
keeps Enclosure's values and the tests of each change on Enclosure.

### Moves and points

A `Loc` stays a 16-bit `short`, but it names two different things:

- **A point,** where KataGo indexes the board: ownership, areas, the planes of the network's inputs and outputs, and the
  symmetries. A point keeps KataGo's layout (`Location::getLoc`), within `Board::MAX_ARR_SIZE`.
- **A move:** `NULL_LOC` (0), `PASS_LOC` (1), or `Board::MOVE_LOC_BASE` plus the game's action. `MOVE_LOC_BASE` is
  `MAX_ARR_SIZE`, so the moves start after the points, and every move `Loc` is below `Board::NUM_MOVE_LOCS`.
  Everything that stores, compares, hashes or passes moves (the search tree, the histories, the training rows) is
  KataGo's. A `Move` is still `(Loc, Player)`.

**Why the two ranges do not overlap:** a non-pass move used as a point is outside every point array. Overlapping ranges
would let some such errors produce wrong answers without a failure. Release builds remove KataGo's `assert`s. The search
tests use a Debug build with assertions and `-D_GLIBCXX_ASSERTIONS`, so these errors fail. The affected arrays now use
`NUM_MOVE_LOCS` or the policy size. Callers that need a point use `action_point`. These callers changed:

- Arrays of `MAX_ARR_SIZE` indexed by a move: `rootSymDupLoc` and `rootSymRepresentativeLoc` (`search.h`), the
  Zobrist table of the subtree value bias (`subtreevaluebiastable.cpp`), and the pattern bonus table.
- Arrays of `MAX_ARR_SIZE` that hold one entry for each legal move, which a game can have more of than points:
  `chooseIndexWithTemperature` (`searchhelpers.cpp`), which chooses the played move.
- Loops over points that list the legal moves: the choosers of a random legal move in `playutils.cpp`. With moves
  among the points, all of self-play's random opening moves would fall in one corner, with no error.
- Point functions given a move: `isOnBoard(moveLoc)` in the root checks and in the filter of avoided moves
  (`searchhelpers.cpp`), and `isAdjacentToPla(moveLoc, ...)`.
- A point array read at a move: `rootSafeArea` (`search.h`), in Go's prune of useless root moves, which now skips a
  move ("Go's features, kept neutral"). The array is now a `std::vector`, not a raw array, so that
  `-D_GLIBCXX_ASSERTIONS` catches such a read.
- `NNPos::locToPos` took both a move and a point. It now takes moves only, and gives the action, which is the policy
  index. A point's spatial index comes from `NNPos::pointToPos`.

**SGF records** (`dataio/sgf.cpp`, `writeSgfLoc`): a move writes the game's protocol text (`Location::toString`),
since its `Loc` is not a point. A point (the start's pieces, `AB` and `AW`) keeps SGF's letters, which reach 52 points:
a board larger than 52 (Blueprints' view of 67) writes its points as `x,y` numbers. No reader reads these back: the
records are self-play's logs. The dashboard counts their moves (`;B[` and `;W[`).

**The legal moves:** `Board` keeps the legal actions as a bitset (`bitset<NUM_ACTIONS>`), updated with the position,
so `isLegal` is a lookup. The evaluator reads it by policy index, and the evaluator that uses no network
(`debugSkipNeuralNet`) fills every policy entry.

### The player to move in the search

A turn can have several moves, as in Enclosure's 1-2-2 order. The next player comes from `next_player` through
`BoardHistory::presumedNextMovePla`. `getOpp` of the last mover cannot give it. KataGo already fits such turns: each
`SearchNode` stores its own `nextPla`. The search keeps statistics from White's side and reads them with the choosing
player's sign. It does not negate values at each move.

Only the places that derive the next player from the last mover had to change. A survey of the 29 `getOpp` calls in
`search/` found three kinds:

- **The next player after a move:** now `hist.presumedNextMovePla`, which the front sets after every move. These are
  `Search::makeMove` (the root), `Search::playoutDescend` (both branches of the descent), and the four replays of the
  principal variation in `analysisdata.cpp`.
- **The player who made the move into a node:** now `moveHistory.back().pla`. This is the subtree value bias in
  `Search::allocateOrFindNode`. The previous move of its key stays the literal previous move, which can be the same
  player's.
- **"The other player" as an identity:** unchanged. The visit-cap contempt, the side that pondering assumes, and the
  playout doubling advantage each name a player, not the next mover.

The self-play game loop (`program/play.cpp`, `playutils.cpp`) and the analysis engine (`command/analysis.cpp`) take the
next player in the same way. GTP checks a move's player against the position (`turnError`). The training rows need no
change. KataGo's opponent-policy target is the policy of the next move, whoever plays it. Value targets use the row's
player to move. [Enclosure's search tests](ENCLOSURE.md#turn-order-in-the-search-step-3) check this.

- **The graph search:** `Board::simpleRepetitionBoundGt` returns true, not Go's safe false. Neither game's positions
  recur, so the position's hash alone identifies a node. Transposed turns (A then B, B then A) share that node. With
  false, the moves would go into every graph hash. A game whose positions can recur needs KataGo's repetition handling
  ("Assumptions that stay in the front and the search", below).
- **A final score is a point mass:** `ScoreValue::whiteScoreMeanSqOfScoreGridded` returns the square of the score.
  KataGo's Go version spreads a score over Go's half-point grid, which the games' exact scores do not have.

### Go's features, kept neutral

These stay declared, with neutral answers, so that their callers compile unchanged:

- **`Board`:** `ko_loc` is `NULL_LOC`. Go's stone logic (liberties, eyes, captures, ladders) gives neutral answers.
- **`BoardHistory`:** `encorePhase` is 0, `superKoBanned` and `koRecapBlocked` are all false, `hasButton` is false,
  and there are no handicap stones. `passWouldEndPhase` and `passWouldEndGame` are false, and
  `endGameIfAllPassAlive` does nothing. The rules' `Rules` struct is plain data (ko, scoring and tax rules, komi),
  and the games read only its komi.
- **The search:** the pattern bonus table (`avoidRepeatedPatternUtility`, 0 by default) and anti-mirror (`antiMirror`,
  off by default) remain optional. `getEndingWhiteScoreBonus` returns 0. This disables Go's bonus for filling neutral
  points, which read ownership at the move's point.
- **The prune of useless root moves** (`rootPruneUselessMoves`) prunes no move of a game. When the opponent's last
  four moves are passes, Go's prune removes each root move on a point of its pass-alive area (`rootSafeArea`).
  `Search::isAllowedRootMove` skips that test for a move `Loc`, which is not a point. A test at the move's point
  (`action_point`) would remove real moves. A game's `rootSafeArea` is its area now (`GameRules::area`), not a safe
  area, and a later move can change it. For example, an Enclosure edge that cuts an enemy edge can open the enemy's
  region. A game that has useless moves needs a prune of its own. The three games' self-play configs keep the setting
  on, with no effect. `runenclosuretests` checks this after seven passes.
- **No result:** the fork never sets `isNoResult`. KataGo's branch for rules without a no-result outcome then sets
  that output to zero, so a win and a loss share the probability, as under Go's area rules.

### The network at a symmetry

- **A symmetry acts on the position, not on the input planes.** The evaluator (`NNEvaluator::evaluate`) chooses the
  symmetry when it fills the inputs. It uses the cache key (`nnSymmetryFromHash`), a random draw, or the server's
  symmetry. Each thread's random generator starts from the evaluator's seed. The inputs describe the image position
  (`sym_position`). The backend gets the identity, and CUDA refuses any other symmetry.

  The evaluator maps the returned policy back action by action (`SymActionMap`) and ownership point by point, before the
  softmax and cache. It never uses the backends' pixel symmetries. Thus Enclosure's edges get exact images even though
  they do not move as pixels.
- **Root symmetry pruning** (`markDuplicateMoveLocs`): a symmetry is valid when the image has the position's hash. Of
  each set of equal legal moves, the search uses the lowest action. `computeRootSymRepresentativeLocs` and the
  analysis output map moves with `getSymMoveLoc`, and map principal variations move by move on the image boards
  (`getSymMoveLocs`).
- **`LocalPatternHasher::getHashWithSym`** hashes the window of the image board around the image move.

### The policy head

- **Each policy output has `POLICY_CHANNELS` channels for each point.** Its convolution channel is `output *
  POLICY_CHANNELS + channel`. The policy entry of an action, `(y * MAX_BOARD_LEN + x) * POLICY_CHANNELS + channel`,
  reads the NHWC output in place. Pass stays one logit for each output.
  - PyTorch (`model_pytorch.py`): `conv2p` has `num_policy_outputs × POLICY_CHANNELS` channels. The head returns (N,
    outputs, `NUM_ACTIONS`), with pass last.
  - The export (`export_model_pytorch.py`) writes the chosen outputs (the regular and the optimistic, as upstream) as
    blocks of `POLICY_CHANNELS` channels.
  - C++ (`desc.cpp`): `p2Conv.outChannels` must be `policyOutChannels × POLICY_CHANNELS`. The Eigen and CUDA backends
    size their buffers for it and decode the blocks. The pass buffer holds one logit for each output.
- **Go's score parity** (the komi input times a parity vector in the score-belief head) stays a no-op: the vector is
  zero. A game whose last global input is komi would need KataGo's.

### The score scale

Scores stay in the game's points. Every constant that KataGo tuned to Go's points scales by the game's `SCORE_SCALE`
(s), its score spread against Go's. Training then weighs every score term as KataGo does on scores of Go's size. The
rule:

- a quantity in points: × s.
- squared points: × s².
- a loss weight: ÷ the growth of its loss. A Huber loss whose argument and delta both scale by s grows by s². A
  squared-error loss on squared points grows by s⁴.

| Constant | KataGo (s = 1) | Scales by | Why KataGo selected it |
|---|---|---|---|
| Output multipliers: score mean, stdev, lead, short-term scores | 20 | s | The paper, A.5: "an arbitrary reasonable scaling factor so that on typical data the neural net would only need to output values around unit scale" |
| Short-term score error multiplier | 150 (squared points) | s² | The same aim, for a squared error (model version 13 rescaled it) |
| Score distribution range, score target cap | ±(361 + 60) points | s, on the game's largest board: ±(`MAX_BOARD_LEN²` + 60) × s | The paper, A.5: "an upper bound for the plausible final score difference", the board's area plus 60 for komi |
| Score distribution bins | 1 point wide | s: 2 × (`MAX_BOARD_LEN²` + 60) bins, s points wide | The same bound, in bins of one point |
| Q-value score storage in int16 | × 60 | ÷ s | × 60 keeps ±421 within 32,000 at a precision of 1/60 point |
| Huber deltas: score mean, lead, stdev, q-value score, short-term score | 12, 8, 10, 12, 12 | s | A code comment: Huber is "meanlike locally and ... medianlike for very large possible losses". The specific deltas: source not found yet. The paper's first version used 10. |
| Huber delta: short-term score error | 100 (squared points) | s² | As above, in squared points |
| Floor of the irreducible error: short-term score | 1e-4 (squared points) | s² | A regulariser: a hundredth of a point, squared |
| Loss weights: score mean, lead, stdev, short-term score, q-value score | 0.0015, 0.006, 0.001, 0.0004, 0.0008 | ÷ s² | Source not found yet. The paper's first version used 0.004 for the self-predictions. The division keeps each term's share of the total loss, whatever the first reason. |
| Loss weight: short-term score error | 0.00002 | ÷ s⁴ | A loss of the fourth degree |
| Score distribution pdf and cdf losses | 0.02 each | none | Over bins, not points (the paper: pdf and cdf 0.02) |
| GTP resignation (`command/gtp.cpp`, `shouldResign`): the bound on the score, the handicap test, the buffer below schedule | 361, 0.9, 5 points | s | The board's area as the largest score, and "Always require at least a 5 point buffer" before a handicap game resigns. Only GTP play uses them. |

The paper's base weights, for reference: value 1.5, policy 1.0, opponent policy 0.15, ownership 1.5/b², score pdf and
cdf 0.02 each, L2 3e-5. The search's score utility unit is a separate constant (`SCORE_UNIT`), since one scale for both
would change Enclosure's search (design 07 of the engine repository, "Why"). Enclosure's s is 12
([ENCLOSURE.md](ENCLOSURE.md#score-scale-step-6), "Score scale"). Connect-4's s is 1, so its point multipliers and loss
weights keep KataGo's values. Its score target cap uses its largest board: `7² + 60 = 109` points. Blueprints' s is 20:
the median margin of people's games, 409 points, over KataGo's score multiplier of 20
(`game/blueprints/gameencoding.h`).

### The game's length

KataGo sets its time constants from the board's area. In Go, the area sets both the game's length and the reach of
its tactics. A game can have Go's board and a shorter game, so the fork divides the constants by what they measure
(`board.h`):

- **The look-ahead horizons count decisions:** KataGo's 19 × 19 formulas, unchanged, counted in moves of the search
  tree. These are the short-term value and score targets, the short-term error targets, and value-surprise weighting.
  They read the game's look-ahead area (`Board::lookaheadArea`, the optional member `LOOKAHEAD_AREA`): the board's
  area, unless the game's board is larger than the reach of its decisions.
- **The game-phase constants are fractions of a game.** `Board::TIME_SCALE` is `GAME_LENGTH / GO_GAME_LENGTH`.
  `Board::GAME_LENGTH_AREA` is `19 * 19 * TIME_SCALE`. It replaces the board's area (A) in these constants:
  - earliest self-play resignation: `1 + int(A / 5)`.
  - early-fork move: `earlyForkGameExpectedMoveProp * A`.
  - opening policy moves: `policyInitAreaProp * A`.
  - GTP time allocation: `TimeControls::getTime` uses A for game-length estimates and counts moves played in place of
    stones.
  - GTP resignation: `shouldResign` uses A for the minimum game length and handicap catch-up.
  - the move temperature's half-life (`Search::interpolateEarly`, `chosenMoveTemperatureHalflife` and the root's
    policy temperature): KataGo's half-life × √A / 19.

  The trainer's variance-time constants scale by `TIME_SCALE` too (`model_pytorch.py`).
- **The front counts actions.** The constants above count the history's actions (KataGo's moves), one search each,
  and the variance-time target counts rows. So `GAME_LENGTH` counts actions. In Enclosure and Connect-4 an action is
  a move of the game. In Blueprints an edit is two actions, a pick and its drop, so its `GAME_LENGTH` is the mean
  count of actions of its self-play games, 132 (120 moves of the site and the edits). The count of a game changes with
  its edits: 120 actions in the shortest game, 146 in the longest. Go's length changes at least as much: 292 to 366
  moves from the 10th to the 90th percentile (below). Two places count the game's moves instead
  (`moves_played`): GTP's time allocation, which Blueprints' play does not use, and the turn number of a history that
  starts from a position (the engine protocol, GTP), which then leaves out the edits before that position.
- **`GO_GAME_LENGTH` = 328** is the measured mean length of KataGo's own 19 × 19 self-play games, without handicap or
  setup stones. The source is the kata1 training games on katagoarchive.org. A sample of 732 games: mean 328.7, median
  329, 10th to 90th percentile 294 to 363. A later sample of 51,094 games: mean 328.1, median 327, 292 to 366. KataGo
  plays its games to the end and does not resign, so its games are longer than human games. The specific numbers are
  Go's, and a game can need other values later.

### Reproducible analysis and search in slices

For an engine protocol (design 04 of the engine repository), two additions that change nothing by default:

- **`nnSymmetryFromHash`** (a setting, off by default): each network evaluation's symmetry comes from its cache key,
  not from a random draw (`NNEvaluator::setSymmetryFromHash`, `nneval.cpp`). Different positions still get different
  symmetries, so a search keeps the mean over orientations. The same position always gets the same one, so a result
  does not depend on earlier requests. Self-play keeps its random draws.
- **`Search::endManualSearch`** ends a search that KataGo's manual interface runs (`beginSearch`,
  `runSinglePlayout`), as `runWholeSearch` ends its own. An engine module runs a search in slices with it.

### The search of the root's views

For a game whose network sees a part of a larger board (the views of "Optional members"), the analysis engine, GTP's
genmove and Blueprints' engine protocol add `rootViews` (`program/rootviews.cpp`, the fields and the commands in
`docs/Analysis_Engine.md` and `docs/GTP_Extensions.md`, the protocol's in the engine repository's
`games/blueprints/README.md`). It is off when a config leaves it out, and Blueprints' config turns it on. It searches
the root in each view of a cover (a pass), then the child positions of the best candidates (a check). In GTP, each
search takes its share of the clock's time for the move, and a move that the position's own view cannot name is
`MOVE/VIEW`.
`Search::fixedScoreCenter` gives these searches one centre of the score utility in place of each root's centre
(`computeRootValues`). Outside them it is `NAN`, and each search takes its root's centre, as in KataGo.

`RootViews::Searches` holds the passes and the checks one at a time, for a caller that runs each search. `next()`
gives each search's root and its share of the limits. `done()` takes the search at its end. `result()` gives the
response at any time, also with a search that has not ended. `RootViews::search` runs them on an `AsyncBot` (the
analysis engine and GTP). Blueprints' engine protocol (`command/blueprintsprotocol.cpp`) runs them in its slices,
with the manual interface of `Search`, so a reply during the search gives the candidates so far.

## The rules part

`struct GameRules`. The Python binding needs this part, and the encoding part's symmetries when the game has that
part. It needs KataGo's `core/hash.h` (`Hash128`). Enclosure's `hash` also needs `core/rand.h`, and its protocol's
parser `core/global.h`: a program that calls them links `core/rand.cpp`, `core/hash.cpp` and their dependencies,
`core/global.cpp` among them.

| Member | Meaning | Enclosure |
|---|---|---|
| `NAME` | The game's name, as in `-DGAME` and the folder. | `"enclosure"` |
| `Position` | A copyable value: the whole state that the rules read. | `enclosure::Position` (`engine.h`) |
| `MAX_BOARD_LEN` | The longest side of the game's boards, and the network's board (`NNPos::MAX_BOARD_LEN`). The front takes the build's largest board (`COMPILE_MAX_BOARD_LEN`) from it, unless the build asks for a larger one (`USE_BIGGER_BOARDS_EXPENSIVE`). | 19 |
| `NUM_ACTIONS` | The number of actions, pass included. | `PASS + 1` = 17,329 |
| `PASS_ACTION` | Pass: `NUM_ACTIONS - 1`. | `PASS` = 17,328 |
| `new_game(xSize, ySize, p)` | Sets the start position on a board of that size. False when the game is not played on it. | 19 × 19 only |
| `x_size(p)`, `y_size(p)` | The board's size. | 19 |
| `next_player(p)` | The player to move, from the position. A turn can have several moves. | `p.player` |
| `turn_moves_left(p)` | The moves left in the mover's turn, this one included. 1 in a game that alternates. | `p.moves_left` (1-2-2) |
| `moves_played(p)` | The moves since the start: the game's progress. | `p.move_count` |
| `terminal(p)` | Whether the game is over. | after 120 edges |
| `legal_actions(p, out)` | The legal actions, ascending. None when terminal. | `Position::legal_actions` (pass only when no edge is legal) |
| `is_legal(p, action)` | One action's legality, false out of range. | `Position::is_legal` |
| `play(p, action)` | Plays a legal action, unchecked. | `Position::play` (pass forfeits the rest of the turn) |
| `result(p)` | At the end, the first player's return: 1, 0 or -1. | `Position::result` |
| `score(p, player)` | The player's points in the game's units, final at the end. KataGo's margin is `score(p, 1) - score(p, 0)`, with the rules' komi added. | `p.scores[player]` |
| `hash(p)` | A hash of the whole position, the player to move included, so that equal hashes mean equal positions. The front also mixes in KataGo's hash of the player to move. | Board's Zobrist hash, from the same seed ([Enclosure's design](ENCLOSURE.md#board-and-rules-step-2)) |
| `pieces(p, owner)` | `owner[y * x_size + x]`: the player whose piece is on the point, or -1. KataGo's `colors`. | each player's nodes |
| `area(p, owner)` | `owner[y * x_size + x]`: the player who owns the point now, or -1. KataGo's area: the final ownership at the end. | the points that only one player's regions enclose |
| `action_text(action)` | The action in the game's notation (design 04 of the engine repository), for a legal action or pass. | `notation::move_name` (an edge that leaves the board has no name) |
| `parse_action(p, text, why)` | The legal action of `p` that the text names, or -1 and the reason. | `notation::parse_move` |
| `position_text(p)` | The one-line position. | `notation::position_string`: the five fields of design 04 |
| `parse_position(text, p, why)` | Sets `p` from a one-line position, or returns false and the reason. | `notation::parse_position` (the history planes come back empty) |

### Optional members

A game may leave out an optional member. For each one, `gameinterface.h` has a detector (as the checks have) and a
helper, which calls the game's member when the game has it, and gives the default when it does not. The front calls
the helper. The rules part's check requires only that the members of a pair come together.

| Members | Meaning | Default (helper) | Enclosure |
|---|---|---|---|
| `protocol_text(action)`, `parse_protocol_text(text, action)` | An action other than pass as the engine protocol (GTP, the analysis engine) writes and reads it. The parser reads trimmed text in any case, does not check legality, and returns false for text that names no action. The front reads and writes pass (`pass`). | `action_text`, and the action whose `action_text` is the text, ignoring case (`protocolText`, `parseProtocolText`). The parser tries each action, so it suits games with few actions. | The form of the move server and the sites: column letters A to S (I included, unlike the notation), such as `D10-D13` |
| `NUM_LOCAL_KEYS`, `LOCAL_WINDOW`, `local_hash(p, cx, cy, xRadius, yRadius, flipPlayers, keys)` | The position near the point (cx, cy), in the window of xRadius and yRadius: the search's local pattern hash (the subtree value bias, the pattern bonus). The front draws `NUM_LOCAL_KEYS` random keys for it, after KataGo's keys. `flipPlayers` swaps the players. `LOCAL_WINDOW` is the side of the subtree value bias's window (odd), large enough for a move's reach. | Each point of the window on the board hashed with its piece (`pieces`), as KataGo hashes stones: KataGo's keys `zobristLocalPattern` by its colors (empty, the first player, the second), without its atari flag. KataGo's window of 5 (`localHash`, `localWindow`). | The edges of either player with an end in the window, and whether each is protected: the hash of before the interface, byte for byte (3 × 17,328 keys, in the order that the hasher drew them). The window of 7, which covers a whole edge from its source. |
| `QUERY_POSITION_FIELD`, `parse_query_position(value, p, why)` | A position field of the game's own in an analysis query, beside the generic field `position` (the one-line position): its key, and the reader of its value. The reader is a template on the JSON type, so that the rules part does not need the JSON library. It may throw the JSON library's exceptions, which the analysis engine catches. | None: the analysis engine takes `position` only (`queryPositionField`, `parseQueryPosition`) | `enclosurePosition` (`queryposition.h`): the fields of `Position` as JSON, with the last turns' edges for the history planes, which the one-line position leaves out. The move server sends it. |
| `STANDARD_X_SIZE`, `STANDARD_Y_SIZE` | The game's standard board, on which `new_game` succeeds: the front's default board (`Board()`, `Board::DEFAULT_X_SIZE` and `DEFAULT_Y_SIZE`, GTP's board when its config names none) and the Python binding's `Position()`. | The largest board, `MAX_BOARD_LEN` × `MAX_BOARD_LEN` (`standardXSize`, `standardYSize`) | None: the default gives 19 × 19. Connect-4 gives 7 × 6. |
| `TURN_MOVES` | The moves of a full turn, after which the clock's increment comes. The time controls give each move its share of the increment (`increment / TURN_MOVES`), and a move with more of its turn to come takes at most the time left divided by the moves left (`turn_moves_left`). | 1, a game that alternates (`turnMoves`) | 2: the site's increment comes once a turn of two edges (games/enclosure/docs/design/18-play-through-gtp.md) |
| `replay_tolerates(p, action)` | Whether a replayed game (GTP, the analysis engine, SGF files, start positions: `BoardHistory::isLegalTolerant`) also plays this action of `p` where it is not legal: what a site plays beside the legal actions. `play` must accept it. Any `int` (false out of range). The search still plays the legal actions only. | None: a replay plays the legal actions only (`replayTolerates`) | A pass at any time: the site's pass on a clock's timeout, which forfeits the rest of the turn |
| `continues_move(p)` | Whether the next action continues the move of the action before it, by the same player: one move of the game played as two actions. Self-play's `cheapSearchPerMove` reads it (`play.cpp`): such an action takes the search limits of the action before it, full or cheap, in place of its own draw. | False: each action is a move (`continuesMove`) | None: each edge is a move. Blueprints gives it: a drop, after its pick, continues the edit. |
| `SETTINGS`, `set_setting(name, value, why)` | The game's own settings: each one's key in a config (a `std::array` of `const char*`), and the setter, which reads a value's text and returns false with the reason for a name or a value that the game does not have. Every command reads them from its config before its first game (`Setup::initializeSession`). A setting holds for the whole process, and the rules can read it, so a game replays only under the settings that played it. A config without a setting's key keeps the game's default. The Python binding gives `SETTINGS` and `set_setting(name, value)`. | None (`settingNames`, `setSetting`) | None. Blueprints gives `viewOrientation`, the rule of its views' orientations (the engine repository's `games/blueprints/README.md`). |
| `view_text(p)`, `set_view(p, text, why)`, `view_cover(p, views)` | The views of a game whose network sees a part of a larger board, each a text of the game's own (`""` for the view that the rules give the position). `view_text` gives the view that a caller set on `p`, or `""`. `set_view` sets `p`'s view until `p`'s next action, which ends it: the legal actions, what each action means, the inputs, the symmetries' images and the hash are then that view's, and every later position has the rules' view. `""` gives the rules' view back, and a text that names no view returns false with the reason. `view_cover` gives views that hold every legal move of `p` between them, each move an action of `p` with one of them set: `{""}` when the rules' view holds them all. The analysis engine's and GTP's `rootViews` search each view of the cover (`docs/Analysis_Engine.md`, `docs/GTP_Extensions.md`). The Python binding gives `Position.view()`, `set_view(text)` and `view_cover()`. | One view, `""` (`viewText`, `setView`, `viewCover`) | None. Blueprints' views are `q,r,o`: the middle's site coordinates and the orientation (the engine repository's `games/blueprints/README.md`). |
| `state_text(p)`, `parse_state(text, p, why)` | The whole state that the network reads, as one line: the one-line position and what it leaves out that the inputs read (`input_hash`: the earlier turns of the history planes). Equal states give equal texts. `parse_state` gives the state back, with the same `hash`, `input_hash`, legal actions and inputs, or returns false and the reason. A view that a caller set is not part of the text. RGSC's buffer (`program/rgsc.h`) keeps a game's states in it: its entries are then exact states, each with the context of its evaluation (the komi and the asymmetric playouts), and a restart starts from the state itself with that context. `testrgscstates` checks a game's round trip at every position of random games. | None: RGSC keeps the moves from the empty board, with their unordered key (`hasStateText`, `stateText`, `parseState`) | None: Enclosure's buffer keeps moves. Blueprints gives it: its one-line position with each blueprint's cells sorted, then the hexes that the last 4 actions filled and captured (the engine repository's `games/blueprints/README.md`, "RGSC"). |
| `draw_start(p, random)` | The game's own draws in the start position `p` of a new game. Self-play makes them for each game from the standard start (`GameInitializer`, `play.cpp`), `random()` gives a uniform `uint64_t` from self-play's random numbers, and the game calls it only for the draws that it makes: a game that draws nothing leaves self-play's other draws as they are. The position then holds the draws, its one-line position too, so a replay or an analysis from that text gives them back. Self-play's SGF records do not hold them. The commands that start a game in another way (GTP, the analysis engine) draw nothing. The Python binding gives `Position.draw_start(random)`. | None: no draw (`drawStart`) | None. Blueprints draws the view's orientation of a game under `viewOrientation = randomGame`. |

### A game's notation, and a site's own terms

A game's rules part supplies four forms of notation:

- `action_text` and `parse_action`: moves in the game's notation.
- `position_text` and `parse_position`: a one-line position.
- `protocol_text` and `parse_protocol_text`: one action in the engine protocol.
- `QUERY_POSITION_FIELD` and `parse_query_position`: an optional start-position field in an analysis query.

A port of a strategyjam.org game uses the site's move strings in its protocol (abscls-r93). The planned backend
(abscls-ezid) accepts the site's record, `{initial, moves}`, and returns a site move string. It maps that record to
KataGame's analysis query:

1. Put `initial` in the game's query field when the record needs a nonstandard start. The game's `parse_query_position`
   reads that state.
2. Give each action its colour from the game's turn order: `[["B", MOVE], ...]`. A turn with several actions
   (`TURN_MOVES` above 1) needs one protocol move for each action. The backend must split a site's combined turn string
   into actions before building the query. `parse_protocol_text` reads one action and cannot split a turn into several
   moves.
3. Map the site's pass string to `pass`, which the front reads and writes itself. Map an output pass back to the site's
   form.

A game with no site version (Connect-4) uses the notation of design 04 of the engine repository.

## The network encoding part

`struct GameEncoding`. The search, self-play and the trainer need it. The Python binding uses only its symmetries
(`SYMMETRIES`, `sym_position`, `SymActionMap`), when the game has this part, for the conformance suite. Input versions
8 and 9 (model versions 17 and 18) are the game's inputs: the front's on-board plane, then the game's planes, and the
game's global inputs (V9 adds KataGo's 2 playout inputs after them). Input version 10 (model version 19) adds the
game's extended inputs to V9's (`NUM_EXTENDED_SPATIAL_INPUTS`, below): its extended planes after its planes, and its
extended global inputs after V9's. So a new run can read more inputs, while the networks and the rows of V9 stay as
they are. Input version 11 (model version 20) is V10's with the game's history of turns (`NUM_TURN_GLOBAL_INPUTS`,
below): the history planes show the last turns in place of the last actions, and the game's turn inputs follow V10's
global inputs. Input version 12 (model version 21) is V11's with the game's planes of closed areas
(`NUM_AREA_SPATIAL_INPUTS`, below) after its planes. Input version 13 (model version 22) has V12's inputs with the
game's labels in their canonical order (`CanonicalActionMap`, below), and the policy in that order too. Input version
14 (model version 23) is V13's with AZdb's latent after its global inputs, for every game: 5 inputs, 1 at the member's
(`MiscNNInputParams::diversityMember`, 0 by default) and 0 elsewhere. The network's cache folds the member into its
key (`NNInputs::getHash`), as KataGo folds the playout doubling advantage: member 0 keeps the key of an evaluation
without a member ("AZdb's members", below). Self-play writes its rows in V9 unless its config sets `inputsVersion` (10,
11, 12, 13 or 14).

| Member | Meaning | Enclosure |
|---|---|---|
| `Rules` | The game's rules part. | `GameRules` |
| `NUM_SPATIAL_INPUTS` | The input planes that the game writes. The front puts KataGo's on-board plane first, so the network sees `1 + NUM_SPATIAL_INPUTS` planes. | 214: 4 × 48 edge planes, 22 feature planes (V8 has 215 with the on-board plane) |
| `NUM_GLOBAL_INPUTS` | The global inputs that the game writes. | 5 (V9 adds KataGo's 2 playout inputs after them) |
| `spatial_inputs(p, options, set)` | Calls `set(x, y, plane)` for each input that is 1, at the image under `options.symmetry`. A template on `set`, so that each writer of the front is inlined. | `fillRowV8`'s edge planes (both ends of each edge) and feature planes |
| `global_inputs(p, options, out)` | Writes the global inputs. | the observation's scalars, with komi as a starting score |
| `HISTORY_PLANE`, `NUM_HISTORY_PLANES` | The history planes: the game's planes that show what the position keeps of earlier turns. `spatial_inputs` sets none of them without `options.history`, and the trainer's history dropout clears them. A game without history has 0 planes. | the last 4 feature planes: 210 to 213 (the network's 211 to 214) |
| `input_hash(p)` | What the inputs read beyond `hash(p)`. The front mixes it into the network cache's key. | the history planes' edges (`BoardHistory`'s history hash) |
| `POLICY_CHANNELS` | Policy channels per point. | 48: one per offset |
| `action_point(a, x, y, channel)` | The point and the channel of an action other than pass. | the edge's source point and offset |
| `point_action(x, y, channel)` | The action at that point and channel: `(y * MAX_BOARD_LEN + x) * POLICY_CHANNELS + channel`, which is also its policy index. | `(y * 19 + x) * 48 + channel` |
| `SYMMETRIES` | The allowed symmetries: bit s for KataGo's symmetry s. The evaluator, the search and the images of positions use these only. | `0xFF`: all 8 |
| `sym_position(p, s)` | The image of the position. | every edge mapped (`sym_edge`), the history's too |
| `SymActionMap` | `SymActionMap(p, s)(a)`: the image of a legal action of `p` or pass, an action of `sym_position(p, s)`. Built once for each position and symmetry. | the edge's image, listed from its other end when that end comes first and is one of the mover's nodes |
| `TARGETS` | The optional training targets that the game writes. | all six (below) |
| `training_targets(positions, n, row, mover, values)` | The game's own targets of a training row: the slots of its own targets (`OWN_TARGETS`, by default all of `TARGETS` but `FRONT_TARGETS`), from the game's positions `positions(j)`, `0 <= j < n`, with the row at `positions(row)`, from the view of `mover`. It writes the values before the row's storage (`TargetValues`), then its global targets, and returns the slots that hold a target in this row. A template on `positions`. A game without own targets can leave it out. | C0, C1, C4 and C5-6 (below), from the positions after the row |
| `SCORE_SCALE` | The game's score spread against Go's. KataGo's constants in points scale by it ("The score scale"). | 12 (ENCLOSURE.md, "Score scale") |
| `SCORE_UNIT` | The search's score utility unit, in points, where KataGo uses the square root of the board's area. | 140.5845 = √(18 · 18 · 61) |
| `GAME_LENGTH` | A game's mean length in actions (the front's moves), for KataGo's game-phase constants ("The game's length"). | 120 |

The encoding part's optional members work as the rules part's ("Optional members"): `gameinterface.h` has a detector
and a helper for each. The front calls the helpers of the two areas through `Board`, and the row writer calls
`ownTargets`. The encoding part's check requires each area to be positive, the bracket's two members to come
together, the lead's step to be 0 or on the half-point grid, and `OWN_TARGETS` to hold the slots that the table gives.
Enclosure and Connect-4 give none. So their areas, their bracket with its rule, and their lead's step are those of
KataGo's code (`testgameareas` and `testkomi` pin them), and the front computes each slot of `FRONT_TARGETS` in their
`TARGETS`.

| Member | Meaning | Default (helper) | Blueprints |
|---|---|---|---|
| `LOOKAHEAD_AREA` | The area A that KataGo's look-ahead horizons read where KataGo reads the board's area ("The game's length"). The short-term value targets C4-7, C8-11 and C12-15 have lambda = 1 - 1 / (1 + A × c), for c = 0.176, 0.056 and 0.016: a mean look-ahead of A × c moves. Value-surprise weighting uses c = 0.016. | The board's area, `x_size * y_size` (`lookaheadArea`, `Board::lookaheadArea`) | 361, Go's: its 67 × 67 view holds the window, and its moves do not reach that far. The look-ahead is about 64, 20 and 6 actions, against 790, 251 and 72 with the view's 4,489, in a game of about 132 actions. |
| `KOMI_AREA` | The area A that KataGo's komi sampler reads where KataGo reads the board's area (`PlayUtils::chooseExtraBlackAndKomi`): each configured standard deviation times √A / 19, and the number of handicap stones. The two spreads of `fancyKomiVarying` (`play.cpp`) still read the board's area. | The board's area (`komiArea`, `Board::komiArea`) | 361: the sampler draws a komi in the points that the config gives. The view's 4,489 multiplied each standard deviation by 67 / 19. |
| `KOMI_BRACKET_STEP`, `KOMI_BRACKET_LIMIT` | The bracket of KataGo's even-komi search (`PlayUtils::evenKomiOffset`, for `adjustKomiToEven` and `computeLead`), in points. Around its guess, the search tries the offsets step, 2 × step and so on up to the limit, until White's win-loss value changes sign, then bisects on the half-point grid. Whole or half points. When the value keeps its sign up to the limit, a game that gives its bracket has no even komi: the search stops after the bracket's offsets, the lead's search gives the position no lead target, and the compensation of an uneven game keeps the game's komi. | KataGo's 1 and 32: the offsets 1, 2, 4, 8, 16 and 32, and KataGo's rule, which bisects to the bracket's end and takes it as the even komi (`komiBracketStep`, `komiBracketLimit`, `komiBracketEndIsEven`) | 20 (`SCORE_SCALE`) and 3,000: the offsets 20, 40, 80 and so on to 2,560, then 3,000, the reach of self-play's widest komi. KataGo's 32 points are less than 2 of Go's points at this score scale. Beyond 3,000 points from the guess, no even komi. |
| `KOMI_LEAD_STEP` | The step of `computeLead`'s smoothing over a coarse score, in points. When the even komi is not a multiple of 2 × step, the lead reads the win-loss value at the 4 komis step apart around it, averages each pair 2 × step apart, and interpolates. KataGo's half point is for Go's area scoring, where the margins of a full board have one parity, so the result changes every 2 points of komi. 0, or whole or half points. 0: no smoothing, the lead is the search's even komi (KataGo's rule for territory scoring). | KataGo's half point times `SCORE_SCALE` (`komiLeadStep`): Enclosure's 6 points, Connect-4's half point | 0: Blueprints' margins are whole points of both parities (426 even and 311 odd in the site's 737 finished games), so each half point of komi can change the result. |
| `OWN_TARGETS` | The slots of `TARGETS` that the game computes itself (`training_targets`). It holds every slot of `TARGETS` that is not one of the `FRONT_TARGETS`, and it can hold those too: the front then leaves that slot to the game. The check requires both. | `TARGETS` without the `FRONT_TARGETS` (`ownTargets`) | All of its `TARGETS`, C2-3 too: the front's future positions read each later position's own view at the row's points, and Blueprints' view moves with the mover's window. |
| `NUM_EXTENDED_SPATIAL_INPUTS`, `NUM_EXTENDED_GLOBAL_INPUTS` | The game's extended inputs, which input version 10 adds to version 9's. With `options.extended`, `spatial_inputs` also sets the planes `NUM_SPATIAL_INPUTS` to `NUM_SPATIAL_INPUTS + NUM_EXTENDED_SPATIAL_INPUTS - 1`, and `global_inputs` writes its extended global inputs after its own. The front puts the extended planes after the game's planes, and the extended global inputs after KataGo's 2 playout inputs. A game gives both or neither. | 0 and 0: V10 is V9 (`extendedSpatialInputs`, `extendedGlobalInputs`) | 12 and 20: the opponent's threats in its own view, the legal hand actions, the minimap of the whole board, and the window boxes (the engine repository's `games/blueprints/README.md`, "The network"). |
| `NUM_TURN_GLOBAL_INPUTS` | The game's turn inputs, which input version 11 adds to version 10's, for a game whose history planes count actions, not turns. With `options.turnHistory`, the history planes show the game's last turns in place of its last actions, and `global_inputs` writes the turn inputs after its extended global inputs: what the planes do not show of those turns. The front puts them after V10's global inputs. Without `options.history` they are 0, as the history planes. | 0: V11 is V10 (`turnGlobalInputs`), as its history planes already count turns | 8: for each of the 4 turns of the history planes (the turn in progress, then the 3 whole turns before it) and each of its 2 moves, 1 when the move was an edit, a pick and its drop (the engine repository's `games/blueprints/README.md`, "The network"). |
| `NUM_AREA_SPATIAL_INPUTS` | The game's planes of closed areas, which input version 12 adds to version 11's: areas that the rules close to a player for the rest of the game. Whether a point is closed depends on cells beyond the convolutions' reach, so the game computes it, as KataGo gives Go's pass-alive areas as input planes. With `options.areas`, `spatial_inputs` also sets the planes after the extended ones, and the front puts them after V11's planes. They are points of the board: the symmetries move them as pixels. | 0: V12 is V11 (`areaSpatialInputs`) | 2: the hexes of the areas closed by the mover's walls and by the opponent's walls, which the other player can never fill (the engine repository's `games/blueprints/README.md`, "The network"). |
| `CanonicalActionMap` | A type, for a game whose position holds labels that no rule reads: input version 13 reads the labels in an order that depends only on what they label, so that two positions that differ only in their labels give the same inputs. With `options.canonical`, `spatial_inputs` and `global_inputs` write each label's inputs at its place in that order. `CanonicalActionMap(p)` gives the network's frame of p's actions: `operator()` maps each legal action and pass to its policy index in that order, and `listed(action)` says whether the search lists the action. An action that gives the same position as a listed action, up to the labels, is not listed, and it maps to that action's index: the two are one edge of the search, with one prior. The order must not depend on the symmetries, so the front maps an action through `SymActionMap`, then through this (`NNInputs`' `NetActionMap`), and the network's evaluation lists the listed actions only. The training rows of V13 write the policy targets at these indices, the next move's in the next position's frame. The trainer reads that target with the row's inputs, so wherever it weighs that target (the game's `mask_targets`), the next position's order of the next mover's labels must be the row's. The rules keep every action legal. | None: every action listed at its own index, and V13 is V12 (`CanonicalActions`) | Each hand in the order of its blueprints' cells: more cells first, then more stars, then more walls, then the lesser shape up to the hex grid's 12 symmetries, then the lowest number of the blueprint's equal blueprints, then the blueprint with the pick, then the lower number. Of equal blueprints, the lowest-numbered one's actions are listed (the engine repository's `games/blueprints/README.md`, "The network"). |
| `Diversity` | A type, for AZdb's reward ("AZdb's members", below): the statistics of a position, φ before its normalization. `NUM_STATISTICS` counts them, `statistics(p, player, out)` writes them from the player's side for the position after a move of that player, `MOVES` is each player's moves in a game (the reward's divisor), and `Normalizer`, built from each statistic's mean and standard deviation, gives the normalized statistics (`z(x, out)`), moves its running estimates by one finished side (`update(x, positions)`) and gives them (`mean(i)`, `sd(i)`) for self-play's league. | None: 0 statistics, 1 move, and every reward is 0 (`diversityStatistics`, `diversityMoves`, `diversityStatisticsOf`, `DiversityNormalizer`) | Its 57 statistics, standardized with a floor of 0.1 on the standard deviation, clipped to [-3, 3] and weighted by group (`game/blueprints/diversity.h`), and 60 moves |

`InputOptions` holds what the front gives the input members:

| Field | Meaning |
|---|---|
| `symmetry` | An allowed symmetry. The inputs are those of the image. |
| `history` | False: leave out what the position keeps of earlier turns (KataGo's `maxHistory <= 0`). |
| `komi` | The rules' komi, for a game whose inputs encode it. |
| `extended` | True: the game's extended inputs too (input version 10). |
| `turnHistory` | True: the history of the game's last turns in place of its last actions, and the game's turn inputs (input version 11). |
| `areas` | True: the game's planes of closed areas too (input version 12). |
| `canonical` | True: the game's labels in their canonical order (`CanonicalActionMap`, input version 13). |

**AZdb's members** (Zahavy et al., arXiv 2308.09175, and the engine repository's
`games/blueprints/docs/research/diversity-players.md`): a team of 5 policies in one network, which reads the member as
V14's latent. The front holds what does not depend on the game (`program/azdb.h`): a game's copy of the members'
state (each member's estimate of ψ, the statistics' mean and standard deviation, and ℓ0), the reward of a move of
member i ≥ 1, (1 − (ℓᵢ / ℓ0)³) φ · (ψⁱ − ψʲ*) / `MOVES` with j* its nearest other member, and the targets of the
intrinsic value head. Member 0's reward is 0. The game gives only φ before its normalization, through the optional
member `Diversity` (above). A game without it gives every member the reward 0.

- **The intrinsic value head** (`ValueHeadDesc::intrinsicHead`, the model file's second value-head option) has RGSC's
  head shape with 2 outputs on the trunk: A, the intrinsic return of the player to move, and B, that of the other
  player, both as the member of the latent. They are returns, not values from White's side: `NNOutput::intrinsicA`
  and `intrinsicB` keep their sign for either player. The CUDA and Eigen backends compute the head. The other backends
  leave it out (`hasIntrinsic` false).
- **Self-play** plays the members with `diversityMembers = 5` (`command/selfplay.cpp`, `program/azdbleague.h`).
  The matchmaker is PSRO-NASH (the paper's Appendix B): each game draws its chooser i and the chooser's player
  uniformly, and its opponent from the other player's maximum-entropy Nash mixture over the members 0 to i of a
  table of results, kept as AlphaStar's league keeps its own (decayed counts, 0.99 a game of a pair). Two
  members play with two searches of the same network (`Play::setDiversityMember`, the seam with the search), and
  only the chooser's rows train (`MatchPairer::BotSpec::trains`, `FinishedGameData::diversityTrains`): the other
  player's turns have the weight 0 before KataGo spreads the game's weight by surprise, and the row writer leaves out
  their rows and side positions. The league holds the table, each member's
  estimate of ψ and the statistics' normalization: the game threads share it, each game takes a copy at its start
  (`diversityState`), and `diversityStateFile` keeps it across restarts. Without `diversityMembers`, member 0
  plays alone, as before.
- **The rows of V14** keep each row's latent (the member of the row's player, `FinishedGameData::diversityMembers`)
  and 4 more global targets after the 80 (C80-83): A's target and its weight, then B's. The rows of earlier versions
  keep their 80. A move's reward is from the side of the player who made it, at the position after the action that
  ends the move. A turn is a run of actions of one player. A's target at a row of player P is the rewards of P's moves
  from the row up to the start of P's 5th turn after the row's turn (`AZdb::RETURN_TURNS`), then output A of the
  search's root evaluation there (`FinishedGameData::intrinsicAByTurn`), or 0 when the game ends before. B's is the
  rewards of the other player's moves in the same span, as P's member, then output B there. A trains on the rows of
  members 1 to 4, B only where both players are the same member, and neither where the root has no output or the
  record stops before the span's end. Without a state (`FinishedGameData::diversityState`), every weight is 0.
  Self-play gives each row the member of its player's search.
- **The search** of member i ≥ 1 (the settings `diversityMember` and `diversityLambda`, λ, 0.9 by default) adds
  w s_P D to each node's utility, with w = (1 − λ) / λ (`Search::diversityWeight`) and P the player that the search is
  for. D is member i's intrinsic return: the rewards of P's moves from the root to the node, then output A where P
  moves at the node and output B where the opponent moves, 0 at the game's end. The opponent minimizes the same
  utility. A move's reward is a node constant (`SearchNode::diversityReward`), from the rows' code
  (`AZdb::State::phiOfMove`) when the search creates the node. Each node adds it once to its average, as KataGo adds a
  pattern bonus, so a node that two paths share has one value. The root has no reward. The win-loss and score values
  stay KataGo's. Every evaluation of the search reads member i's latent. Member 0, or λ = 1, gives w = 0: the search
  is KataGo's, with no reward, no D and no read of the head. With w > 0 the search refuses to start in four cases: a
  model without the head, a backend that does not compute it (`NeuralNet::computesIntrinsicHead`), no copy of the
  game's state (`Search::setDiversityState`), and the eval cache. A member of 1 to 4 needs a model of V14. The search
  clears its tree when P changes, as for the pattern bonus, and when the state, the member or w changes.
  `testdiversitysearch` checks the term ("The checks").

**The training targets** are the slots of KataGo's spatial targets (`valueTargetsNCHW`) and the two that the fork
added. `TARGETS` declares the slots that the game writes. The front computes C2-3 and C7 from `pieces`
(`FRONT_TARGETS`), unless the game computes them itself (`OWN_TARGETS`). The game gives each other slot its own
meaning and computes it (`training_targets`). The front writes 0 in each slot that the game does not write, and the
slots with a weight or a flag of their own (C1, C2-3, C4, C5-6, C7) then have weight 0. C0 has none: C27 weighs it, so
the trainer reads `TARGETS` to leave it out.

The front stores the game's values (`GameInterface::TargetValues`). C0, C4 and C5-6 hold a value in [-1, 1], times
120 and rounded at random so that the expected value is exact (KataGo's storage of its scoring target). C1, C2-3 and
C7 hold an integer as it is (KataGo's seki and future positions: -1, 0 or 1). The random rounding
goes slot by slot (C0 to C7, of which only C0, C4, C5 and C6 draw) and point by point. After the slots, the game can
write the global targets G76 to G79, which KataGo leaves unused. The slots that the member returns give the row's
flags and weights: C68 for C1, C69 for C5-6, and C33 and C34, the weights of the game's own C2-3 and of C4. C27 stays
KataGo's weight of C0, C1 and the score's targets: it is set when the row has the game's end.

| Flag | Slot | Enclosure's meaning (ENCLOSURE.md, "Spatial targets") |
|---|---|---|
| `TARGET_OWNERSHIP` | C0 | the future occupancy: the mean over the turn ends after the row of +1 inside the mover's regions, -1 inside the opponent's |
| `TARGET_SEKI` | C1 | the edges that get cut |
| `TARGET_FUTURE_PIECES` | C2-3 | the nodes 8 and 32 edges later (KataGo's future positions, from `pieces`) |
| `TARGET_SCORING` | C4 | the region occupancy 4 rounds later |
| `TARGET_INCOME` | C5-6 | the territory income of each player |
| `TARGET_NEXT_PIECES` | C7 | the next mover's nodes after the row's move, for the symmetries of policy channel 1 (not a loss) |

### Connect-4's encoding part

`game/connect4/gameencoding.h` is a short example of the part. The engine repository's `games/connect4/README.md`,
"The network", gives the reasons.

| Member | Connect-4 |
|---|---|
| `NUM_SPATIAL_INPUTS`, `spatial_inputs` | 3: the mover's discs, the opponent's discs and the landing points (the point where a column's next disc lands, for each column that is not full). The network sees 4 planes. |
| `NUM_GLOBAL_INPUTS`, `global_inputs` | 1: 1 when the mover is the first player. Komi is not an input: Connect-4 plays at komi 0. |
| `HISTORY_PLANE`, `NUM_HISTORY_PLANES`, `input_hash` | 0, 0 and an empty hash: a position holds the whole state of the game |
| `POLICY_CHANNELS`, `action_point`, `point_action` | 1: a move is its landing point (`action_at`). Pass is never legal. |
| `SYMMETRIES`, `sym_position`, `SymActionMap` | `0x5`: the identity and the mirror (symmetry 2, flip x), which keep gravity. The mirror moves column x to column `x_size - 1 - x`, the discs and the moves alike. |
| `TARGETS`, `training_targets` | C0 only: the final board as ownership (`area` of the game's last position), +1 on the mover's discs and -1 on the opponent's |
| `SCORE_SCALE` | 1: the score is the solver's, in discs (`score`, Pascal Pons's convention, gamesolver.org) |
| `SCORE_UNIT` | 6.4807 = √(7 · 6): KataGo's rule, the square root of the board's area, on the standard board |
| `GAME_LENGTH` | 41: a game of perfect play on 7 × 6, where the first player wins with its last disc |

The value and score targets are KataGo's, from the rules' result and score. A draw counts as half a win: self-play's
`drawEquivalentWinsForWhite` is 0.5, with no random radius (`configs/training/selfplay_connect4.cfg`), since the
inputs do not show a draw's value. A row's score target is the solver's score of the game's end, from the view of the
row's player to move.

### The trainer's module

The trainer reads a game's facts from the game's module, `python/katago/train/games/<game>.py`, which mirrors the
encoding part. `python/katago/train/games/__init__.py` lists the members, and `games.get(name)` refuses a module that
lacks one. A model config names its game (`"game"`, Enclosure's without it), and `modelconfigs.get_game(config)` gives
its module. The config's `"inputs_version"` (8 without it, 9, 10, 11, 12, 13 or 14) gives the game's global inputs,
V9 adds KataGo's two playout inputs, V10 the game's extended inputs, V11 its turn inputs, V12 its planes of closed
areas, V13 the canonical order of its labels, and V14 AZdb's latent after V13's global inputs. The config's
`"intrinsic_head"` gives the model AZdb's intrinsic value head, whose loss is the squared error on the rows' C80-83,
weighted as the value's loss.
`train.py -inputs-version 9` gives a new model V9's inputs (Connect-4's run). A checkpoint keeps its version.
`python/add_extended_inputs.py CHECKPOINT OUT -inputs-version 10` gives a checkpoint of V9 the extended inputs as zero
input channels of the first convolution and zero columns of `linear_global`, in the model, the SWA model and the
optimizer's state, so that its network computes what it did until training moves them (Blueprints' bpj3 from bpj2).
`-inputs-version 11` gives a checkpoint of V10 the turn inputs as zero columns of `linear_global` in the same way (bpj4
from bpj3). The planes keep their count, and training absorbs the new meaning of the history planes (turns, not
actions). `-inputs-version 12` gives a checkpoint of V11 the planes of closed areas in the same way, as zero input
channels of the first convolution. The shuffle merges rows of V11 and V12 at V12's planes, and the reader fits a row's
planes to its model: a row of V11 gets V12's planes at 0, and a row of V12 loses them for a model of V11. So a run
switches from V11 to V12 with a converted checkpoint and no new run, while its window holds rows of both. Rows of other
versions stay apart: the shuffle refuses files whose planes and global inputs both differ (V9's and V10's, or V10's and
V12's), and the reader refuses any other count of planes. `-inputs-version 13` gives a checkpoint of V12 the version
13 and changes no weight: V13 adds no input. The network then reads each label at its place in the canonical order,
and it computes what it did wherever that order is the labels' own. Rows of V12 and V13 have the same shapes, so the
shuffle and the reader cannot tell them apart: a run that switches to V13 trains on its rows of V13 only.
`python/add_diversity_members.py CHECKPOINT OUT` gives a checkpoint of V13 V14's latent and the intrinsic value head:
member 0's column of `linear_global` is 0, so that member 0 computes what the network did, and the other members'
columns are KataGo's initialization of that layer times 0.1, so that they differ a little. The head is new, with
KataGo's initialization, except its last layer, which starts at 0 so that the trunk gets no gradient through the head
until it has learned (`-random-last-layer` keeps KataGo's initialization there). The reader pads rows of 80 global
targets with zeros to 84. A run of V14 trains on its own rows only: the shuffle refuses to mix files of 84 global
targets with files of 80, and the reader refuses rows of an earlier version for a model of V14, because those rows
hold no member's latent. For a new
`SCORE_SCALE`, `-old-score-scale S`, run with the new code, multiplies by S over the new `SCORE_SCALE` the outputs that
a plain multiplier puts in points: the score mean, the lead and the TD scores. These outputs keep their values in
points. Training moves the score's standard deviation and the score belief.

| Member | Meaning | Enclosure (`games/enclosure.py`) | Connect-4 (`games/connect4.py`) |
|---|---|---|---|
| `NAME`, `MAX_BOARD_LEN`, `POLICY_CHANNELS`, `NUM_SPATIAL_INPUTS`, `NUM_GLOBAL_INPUTS`, `SYMMETRIES`, `TARGETS`, `SCORE_SCALE`, `GAME_LENGTH` | As in C++. `MAX_BOARD_LEN` is the trainer's `pos_len` (the model refuses another), and the score distribution's bins are `SCORE_SCALE` points wide. | As in C++ | As in C++ |
| `HISTORY_PLANE`, `NUM_HISTORY_PLANES` | As in C++ (the game's planes, newest first), for the history dropout. | The last 4 feature planes: 210, 4 | None: 0, 0 |
| `apply_symmetry_inputs`, `apply_symmetry_policy_targets`, `apply_symmetry_policy` | The augmentation of the input planes, the policy targets (both channels) and a policy of the mover's moves (the q-values), at an allowed symmetry. The value targets move as pixels in every game (`games.apply_symmetry_points`). | An edge moves as an edge, with each channel's player's nodes | Every plane and the policy move as pixels, as KataGo's do. The mirror flips x on the network's 7 × 7 board, which a row of the standard board fills in width. |
| `mask_targets` (optional) | The rows' global targets, with weight 0 for each target that the augmentation cannot map in a row. The trainer applies it to every batch, at every symmetry and without the symmetries (`games.mask_targets`: none without the member). Blueprints gives it: the second policy target (C28) after a drop with a move of the same player to come. | None | None |
| `LOSS_COEFFICIENTS` (optional) | The coefficient of each loss that it names, of `games.LOSS_COEFFICIENTS`: `ownership` (C0), `seki` (C1) and `income` (C5-6). `games.get` refuses another name. | None: `games.LOSS_COEFFICIENTS`, 1.5, 1 and 1.5 (KataGo's ownership, Enclosure's edges that get cut and its income) | None |
| `LOSS_AREA` (optional) | The area, in points, that KataGo's spatial losses read where KataGo reads the board's area (`games.loss_area`, `Metrics.spatial_area`): the ownership loss and the scoring slot's loss sum over the row's points and divide by it, the future positions' loss by its square root, and the income loss without `income_mask` by twice it. KataGo's area is the row's on-board points, nearly all of which end owned in Go. Blueprints gives 361: its view's 4,489 hexes hold the window, and about 360 of them hold a cell at the game's end. | None: the row's on-board points | None |
| `income_mask(binary_inputs)` (optional) | The points where the loss of C5-6 reads its target, from the batch's input planes at its symmetry: the loss averages each channel over its points, then over the channels that have points. Blueprints gives the cells of each player that can go (not walls). | None: the board's points, both channels together | None |
| `NUM_EXTENDED_SPATIAL_INPUTS`, `NUM_EXTENDED_GLOBAL_INPUTS` (optional) | As in C++. `apply_symmetry_inputs` maps the extended planes too, in the rows that have them (V10). | None: 0, 0 (`games.extended_spatial_inputs`, `games.extended_global_inputs`) | None |
| `apply_symmetry_extended_global_inputs(extended, symmetry)` (optional) | The extended global inputs of V10's rows at the symmetry, for a game whose extended global inputs name points. The trainer applies it to every batch at its symmetry (`games.apply_symmetry_global_inputs`). Blueprints gives it: its window boxes' corners. | None: as they are | None |
| `NUM_TURN_GLOBAL_INPUTS`, `NUM_HISTORY_TURNS` (optional) | As in C++, and the turns that V11's history planes and turn inputs show, newest first: each turn has `NUM_HISTORY_PLANES` // `NUM_HISTORY_TURNS` planes and `NUM_TURN_GLOBAL_INPUTS` // `NUM_HISTORY_TURNS` turn inputs. The turn inputs name no point. In V11's rows the history dropout draws once for each turn, as KataGo draws once for each history move, and clears the turn's planes and turn inputs together, as KataGo's pass inputs go with their moves. A game with turn inputs gives `NUM_HISTORY_TURNS`, which divides both counts (`games.get` checks it). Blueprints: 4 turns, each 2 planes and 2 turn inputs. | None: 0 (`games.turn_global_inputs`) | None |
| `NUM_AREA_SPATIAL_INPUTS` (optional) | As in C++. The planes are points of the board, so `apply_symmetry_inputs` moves them as pixels. Blueprints: 2. | None: 0 (`games.area_spatial_inputs`) | None |
| `CANONICAL_LABELS` (optional) | True for a game with a canonical order of its labels (C++ `CanonicalActionMap`), which V13's rows hold. The order does not depend on the symmetries, so the augmentation maps V13's rows as V12's. Blueprints: True. | None: False, V13 is V12 (`games.canonical_labels`) | None |

The model's heads are the same in every game. The rows give each slot that the game does not write the weight 0, but
not C0 (above). So the trainer leaves out C0 for a game without `TARGET_OWNERSHIP`.

### The model file's game

The export (`python/export_model_pytorch.py`) writes the config's game into the model file, in the header's model
option H, the last spare option: 0 for Enclosure, as every file from before holds, and the game's name for another
game. The loader (`neuralnet/desc.cpp`, `checkModelGame`) reads 0 as Enclosure (`GameInterface::UNNAMED_MODEL_GAME`).
It also reads a file of a version below 15, which has no option H, as Enclosure's. It refuses a network of another
game than the build's: `NAME: the network is for the game connect4, and this binary plays enclosure`. So every binary,
from before too, loads every Enclosure network, and a binary from before refuses another game's network ("model failed
to parse unused params"). A file that writes `enclosure` there (no export writes it) loads in an Enclosure build. The
ONNX metadata (`neuralnet/onnxmodelbuilder.cpp`) does not carry the game yet, so a `.onnx` file is not checked: its
change waits for the TensorRT and ONNX backends (abscls-bd7). Design 07 of the engine repository, "The model file's
game", gives the reasons.

## The bot part

`struct GameBot`, optional. The alpha-beta search (`cpp/search/alphabeta.h`) needs it with the rules part. The Python
binding includes both when the game has `gamebot.h`, and then gives the search (`AlphaBeta`). `katago` compiles
neither: KataGo's search does not use them, so a heuristic changes no output of `katago`. Design 07 of the engine
repository, "The alpha-beta search", gives the search's values and its rules.

| Member | Meaning | Enclosure | Connect-4 |
|---|---|---|---|
| `Rules` | The game's rules part. | `GameRules` | `GameRules` |
| `evaluate(p)` | A heuristic value of a position that is not terminal, from the view of the player to move. The search keeps it within ±`WIN_VALUE / 2`, below every won or lost end. | The mover's final margin if every region stayed as it is (the greedy bot's measure) | Centre columns and open threes |
| `order_actions(p, actions)` (optional) | Reorders the legal actions, the most promising first, and keeps the order of equal ones (a stable sort). Without it, they keep the search's order: ascending, or random with a seed. The search puts its table's move first. | None | The centre columns first |

## The checks

**A game that lacks a member does not compile.** Each game's header ends with its part's check. A missing member gives
one error that names it. This is the `katago` build with `next_player` removed from Enclosure's `GameRules`:

```
game/../game/enclosure/../gameinterface.h:177:35: error: static assertion failed: game interface, rules part: the
game lacks next_player(const Position&) -> int
make[3]: *** [CMakeFiles/katago.dir/build.make:429: CMakeFiles/katago.dir/game/board.cpp.o] Error 1
```

**No code outside the game layer names a game** (`tests/checkgamelayer.py`, "The game layer"). Each build of `katago`
runs it first. Here the check fails on a test that every game's binary compiles. Enclosure's board is put back in
the test:

```
checkgamelayer: 1 references to a game outside the game layer (game/<game>/, game/chosen*.h). Reach the game through
GameRules, GameEncoding or GameBot, or list a file of the game's own in game/<game>/ownfiles.txt:
  tests/testenclosurenn.cpp:35: names the namespace enclosure: const int nnLen = enclosure::SIZE;
```

**The rules part compiles alone.** A program that includes `game/chosenrules.h` builds and runs in a copy of the game
layer without `gameencoding.h`, `features.h` and `chosengame.h`.

**An action is its policy index.** The encoding part's check refuses a game whose `NUM_ACTIONS` is not
`MAX_BOARD_LEN² × POLICY_CHANNELS + 1`. This is Enclosure with `POLICY_CHANNELS = NOFF - 1`:

```
game/enclosure/../gameinterface.h:295:36: error: static assertion failed: game interface, encoding part: an action is
its policy index, so NUM_ACTIONS is MAX_BOARD_LEN * MAX_BOARD_LEN * POLICY_CHANNELS + 1
```

**Enclosure's members, and the defaults:** `testgameinterface` replays the 500 games of
`tests/data/enclosure_games.txt` and 60 random games (with timeout passes and komi). The front calls both parts, so
each member has one copy. At each position the test checks:

- the rules part against itself: the player to move and the edges left (against the 1-2-2 turn order), `is_legal`
  against the legal actions (every action at each 200th position), the moves and the position in the notation and the
  engine protocol's moves (each read back), and the result at the end.
- the encoding part through the front: the inputs (`fillRowV8` with and without history and at the game's komi, and
  `fillRowV8Packed` at each symmetry) against the on-board plane and the members, the history planes (the only planes
  that the history changes, and empty without it), and the policy index (`NNPos::locToPos`).
- the symmetries: the images of the legal actions are the image's legal actions, the inputs at a symmetry are the
  image's inputs, and the inverse symmetry maps the image back.
- at compile time: the allowed symmetries (all 8, and the helpers on a subset).

It also checks the optional members: Enclosure's (the protocol's moves `A10-D10` and `S19-P16`, through `Location`
too, the window's keys and the query field), and the defaults with a stub game that has none of them (the protocol's
text read back in any case, and the window's hash at a corner, key by key, with and without `flipPlayers`). The front
has no copy of the members any more, so the test pins the values of KataGame before the interface in the first game:
`pos_hash`, the situation hash and the local pattern hash at three symmetries, the images of the position at each
symmetry (three plies), and the training targets of four rows (the slots and the values). The training targets'
values are checked to lie in each slot's range at every row of that game.

`testnninputs` checks the inputs against the Python reference of the Enclosure repository, and the row written at a
symmetry against the image board's row. `testsymmetry` plays random games on a board and on its image at once, each
move mapped by the action map: the image game stays the image of the game.

It also checks the members of the turn, the pass and the standard board: Enclosure's values (two edges a turn, a pass
at any time and nothing else in a replay, 19 × 19) through the front too (`Board()`, `BoardHistory::isLegalTolerant`),
and the defaults with the stub game, also with a standard board of its own. Neither game has settings, and the
default's setter refuses a name with a reason. Neither draws at the start: the default leaves the start as it is and
takes no random number. Enclosure's training targets keep the digests that the test pins, and Enclosure leaves C2-3, C7 and the global targets to the front. In the engine
repository, `games/blueprints/tests/test_targets.py` checks Blueprints' own targets against the game record.

Build and run it as the other standalone tests (`tests/enclosure/build.sh`, which runs the check of the game layer
first), from `katagame/cpp`:

```
TESTS=testgameinterface bash tests/enclosure/build.sh BUILD_DIR OUT_DIR
OUT_DIR/testgameinterface tests/data/enclosure_games.txt
```

**The areas of KataGo's constants:** `testgameareas` runs on a build of any game. It checks the defaults of
`LOOKAHEAD_AREA` and `KOMI_AREA` with stub games, and the build's game on its standard board: both areas against the
values that the test pins for each game, and the komi sampler's standard deviation for one configured as 100 points.
It prints the lambdas of the short-term value targets beside their values with the board's area. A new game fails it
until the test pins the game's areas. It builds and runs as `testgameinterface` does, with no argument. In the engine
repository, `games/blueprints/tools/td_lambdas.py` checks the lambdas in Blueprints' rows.

**Komi:** `testkomi` runs on a build of any game, with a self-play config as an optional argument. It plays a random
game to its end and checks the history's result at komis of both signs, at 0, and at the komis that decide the game
(the half point nearest the opposite of the earned margin, a draw when the margin is on the half-point grid, and half
a point to either side): the final margin with the komi, the winner,
White's wins in the value targets (half a win at a margin of 0), the final score target, each side's own komi, and the
network cache's hash, which changes with a nonzero komi only. It checks that the front's global inputs are the game's
at the history's komi, and prints the inputs that change. It checks the even-komi search: the defaults with stub
games, and the build's game against the bracket, the rule and the lead's step that the test pins for each game. With
KataGo's bracket and rule, `evenKomiOffset` tries the same offsets and gives the same result as a copy of KataGo's
code on 1,929 win-loss curves, each an even komi. With the game's bracket, it finds the sign change of a monotone curve
within half a point out to the limit, and the test prints how many values it reads. Where the value keeps its sign up
to the limit (flat curves, and curves that cross beyond the limit, on KataGo's bracket and the game's), KataGo's rule
gives KataGo's result, the far end, as an even komi. The other rule gives the far end at once, after the bracket's
offsets, and no even komi. With a config, it makes 200,000 new games with `play.cpp`'s
`GameInitializer`, as self-play does, and
checks the shares of draws at 0 and beyond 3 standard deviations of `komiBigStdev` against the config's probabilities
(the bigger draw overrides the big draw, which overrides `komiStdev`), and that each draw is on the half-point grid.

**Self-play's cheap searches:** `testcheapsearch SELFPLAY_CONFIG MODEL GAMES [OVERRIDES]` runs on a build of any game,
with a network of the game. A small random one is enough: `python/enclosure_nn_check.py make` with `-config b2c16`,
then its export (the model loader refuses `b1c6nbt`: "trunk: all numbers of channels must be positive"). It plays the
games one after the other as self-play does (`play.cpp`'s `GameRunner`), with fixed seeds and one search thread, and
the config with the overrides (`key=value,...`, as `-override-config`). For each game it prints a digest of the
actions, their full or cheap searches and their weights, so that two builds or two settings that draw the same
searches in the same order print the same digests. It checks that the share of full searches among the actions that
draw their own is `1 - cheapSearchProb` within 4 standard deviations. With `cheapSearchPerMove`, it checks that each
action that continues a move (`continues_move`: Blueprints' drop) has the decision and the weight of the action before
it. Without it, those actions draw their own. It prints the share of full searches among the picks and among their
drops: the choice of a pick depends on its search, so with the setting the drops' share is the picks', which can
differ from `1 - cheapSearchProb`. Two runs play the same games only with the same config and overrides: the size of
the network's cache (`nnCacheSizePowerOfTwo`) changes the games once the cache is full. KataGo's asymmetric playouts
(`normalAsymmetricPlayoutProb`) need at least 5 visits in each search. In a game with them, a cheap search of fewer
visits stops the test with an error ("asymmetric playout doubling resulted in fewer than 5 visits"). So with very few
visits, set that probability to 0, or use a set of games that has no such game.

**AZdb's search:** `testdiversitysearch MODEL MODEL_WITHOUT_HEAD` runs on a build of a game that gives AZdb's statistics
(Blueprints), with two networks of the game of the inputs version 14: one with the intrinsic value head, one without
it. Small random ones are enough (the test's header gives the commands). It checks that member 0 searches as KataGo's
search, and D at each node of chains where each depth allows one move. It checks a node that two orders of P's moves
share, a member's choice of the move with the larger reward, and each node's reward in a member's search with
self-play's weightings. It also checks the members' evaluations in one cache, the refusals, and when the search keeps
and clears its tree.

**Connect-4's encoding part.** The conformance suite checks that its mirror maps the legal moves onto the image's. With
a Connect-4 binary and network, the engine repository's `games/connect4/tests/test_network.py` checks two more things.
Value, score and ownership agree exactly after mapping the mirrored outputs back. The policy agrees within a relative
tolerance of 1e-6, because its softmax sums moves in a different order. The rows of one self-play game hold each
position's inputs and its game's targets. The network check against PyTorch (the random test network of
`python/enclosure_nn_check.py` with `-game connect4`, and `testnnoutputs` on `tests/data/connect4_positions.txt`) names
the game.

## Adding a game

Connect-4 is the template. The engine repository's `games/connect4/README.md`, "A new game from this template", gives
the same steps. Its `docs/design/06-organization-and-game-interface.md`, "What a third game touches", lists the shared
code and the work that remains for a third game.

1. **The rules part.** Make the folder `cpp/game/<game>/` with `gamerules.h` (`struct GameRules` in the namespace
   `<game>`). End it with its check: `static_assert(GameInterface::checkRules<GameRules>(), "...")`. Connect-4's
   `gamerules.h` is a short example. Give the optional members where the defaults do not fit ("Optional members": the
   standard board, the turn, the pass). The notation is in this part: a game that a site already plays speaks the
   site's terms ("A game's notation, and a site's own terms").
2. **The Python module and the tests.** In the engine repository, make `games/<game>/` a Python package and build the
   game's rules module with `sh core/binding/build.sh <game>`. The module's common API needs no code of the game's own.
   The game's Python extras, if any, go in `games/<game>/binding/extras.h`. The conformance suite
   (`core/tests/test_conformance.py`) checks each game's module (design 07, "The Python binding"). Put the game's own
   tests in `games/<game>/tests/`. Compare the rules with published results where they exist. Connect-4's
   `test_rules.py` checks position counts and solves small boards to the end. The random bot (`core/bots.py`) and the
   arena play the game at once: `python -m core.arena --game <game> --run runs/<game>-league --league random r2=random`.
3. **The bot part** (optional). For the alpha-beta search, add `cpp/game/<game>/gamebot.h` (`struct GameBot`, a
   heuristic), ending with `static_assert(GameInterface::checkBot<GameBot>(), "...")`. The binding then gives the search
   after a rebuild with `sh core/binding/build.sh <game>`. The arena's player `ab` then plays the game. A game's own
   bots and the arena's fixed opponents for it go in the optional module `games/<game>/bots.py` (design 07, "The
   pipeline").
4. **The encoding part.** Add `cpp/game/<game>/gameencoding.h` (`struct GameEncoding`), ending with
   `static_assert(GameInterface::checkEncoding<GameEncoding>(), "...")`. Connect-4's is a short example. A board
   larger than the reach of the game's moves gives `LOOKAHEAD_AREA` and `KOMI_AREA` ("The network encoding part"), and
   `tests/enclosure/testgameareas.cpp` pins the game's areas in any case. A game whose margins are much larger than
   Go's gives `KOMI_BRACKET_STEP` and `KOMI_BRACKET_LIMIT`, a game whose margins have no parity gives
   `KOMI_LEAD_STEP` 0, and `tests/enclosure/testkomi.cpp` pins the game's bracket, its rule and its lead's step in any
   case. Build with
   `-DGAME=<game>`, in a build folder for that game and backend: `pipeline/scripts/build_game.sh <game>` builds both
   backends and rebuilds the binding with the encoding's symmetries. Repeat the conformance suite to check them. Code of
   the game's own that uses the front (a command or a test of the search) goes outside the folder, in a file named after
   the game, and the game lists it in `cpp/game/<game>/ownfiles.txt` ("The game layer").
5. **The trainer's module.** Add `python/katago/train/games/<game>.py` with the members of `games/__init__.py` ("The
   trainer's module"), the values of the game's encoding part. `train.py -game <game>` then makes a model of the game,
   and its exported files name the game. Check the network against PyTorch: a random test network
   (`python/enclosure_nn_check.py make CHECKPOINT -game <game>`, then its export), `testnnoutputs` on positions of the
   game in `cpp/tests/data/`, then `enclosure_nn_check.py compare CHECKPOINT DUMP`. The engine repository's
   `build_verify.sh` and `verify_build.sh` are Enclosure's own, so a new game runs these by hand (Connect-4's README,
   "The network").
6. **The run.** Add the game's self-play config `cpp/configs/training/selfplay_<game>.cfg` and its analysis config
   `cpp/configs/analysis_<game>.cfg` (the load check, and the engine repository's network bot, `core/networkbot.py`,
   which plays the game's networks in its arena). Each starts from KataGo's (`selfplay1.cfg`, `analysis_example.cfg`)
   and says why on each line that differs: Connect-4's sets its board, komi 0, a draw's value, its symmetries at the
   root, and a smaller cache of evaluations in the analysis config (`nnCacheSizePowerOfTwo = 16`), since each engine
   of the gauntlet and the ratings allocates the whole cache. Then the first run, in the engine repository (its
   design 07, "The pipeline"), in WSL from its root after `. pipeline/env.sh`. Connect-4's README ("The run") gives
   each command for its run `c4a`:
   1. **The binaries and the rules module:** `bash pipeline/scripts/build_game.sh <game>` builds `katago` for the game
      with the settings of the machine's builds `builds/cuda` and `builds/eigen`, in `$ENGINES_BUILDS/<game>-cuda`
      and `<game>-eigen`, and the game's rules module, which the gauntlet imports. It is a heavy job: start it
      through the org folder's `scripts/machine run`.
   2. **The settings:** `games/<game>/runs/RUN.env`. `KATAGO_BIN` is the CUDA build, for self-play.
      `GAUNTLET_KATAGO_BIN` is the Eigen build, for the gauntlet's matches on the CPU. Start from Connect-4's
      `c4a.env`. The asynchronous setup's defaults fit any game (`python/selfplay/async_parts.sh`), so the settings
      need no shuffle sizes and no thread counts: the shuffle's row counts come from the game's row size (the
      trainer's module, `row_bytes`), at most KataGo's, and each part's thread pools stay within the run's CPUs
      (`CPUS`). A first run of a few minutes to an hour is a pilot (`PILOT=1`, as `c4a.env`): until the first export,
      self-play plays a game in flight a CPU, and the files of a network's rows, the shuffle and the trainer's wait for
      data come at shorter intervals, so that the run trains on its networks' own rows.
   3. **The overrides:** a check, or a run from a worktree with builds of its own, changes settings for one start with
      `RUN_OVERRIDES` (`pipeline/scripts/lib.sh`), and puts its run in another folder with `ENGINES_RUNS`. It needs no
      copy of the settings. `DRY_RUN=1 bash pipeline/scripts/start.sh RUN` shows the settings in force.
   4. **The start:** `bash pipeline/scripts/start.sh RUN` starts self-play, the shuffle, the export and the trainers.
      Then start self-play's watchdog and the gauntlet (`pipeline/scripts/gauntlet.sh RUN`), which rates each export.
      A short run needs a gauntlet round as short as its exports' interval (`GAUNTLET_ARGS`: c4a's round is every 10
      minutes).
   5. **The dashboard:** the live dashboard shows the runs of `$ENGINES_RUNS` with the main tree's settings and
      gauntlets. For a run from a worktree or in another folder, `bash pipeline/scripts/dashboard.sh RUN` serves the
      run's own dashboard on a free port. The Arena tab replays a league's games when the game has the module
      `games/<game>/replay.py` (`replay(game)` and `start()`, from the game's rules binding: Enclosure's and
      Blueprints') and a board in `core/dashboard.html`. A league of another game lists its games with no replay.
   6. **The stop:** stop the watchdog first. Stop the gauntlet's process group, then run `bash pipeline/scripts/stop.sh
      RUN`. Connect-4's README shows how to find the PIDs. Stop a run's own dashboard with `systemctl --user stop
      RUN-dashboard`. The last export can come after the gauntlet's last round. If that occurs, start the gauntlet again
      for one round. Find its new PID before stopping its process group. The engine repository's shared Python (`core/`)
      names no game: it finds a run's folder and its models with the run helpers of `core/runs.py`, for every game, and
      it reaches the game through lookups by the game's name. A game can add the optional module `games/<game>/bots.py`
      (its own bots, and its own player of a network). Without it, the arena plays the game's networks through
      `core/networkbot.py`. The dashboard's replay board draws only Enclosure's board, from Enclosure's
      `games/enclosure/replay.py`, so it lists another game's leagues without replays (design 07 of the engine
      repository, "The pipeline").
7. **What a new game still meets.** Read the engine repository's `docs/design/06-organization-and-game-interface.md`,
   "What a third game touches". Check the front's assumptions, the worktree setup, the machine's limits and the bot's
   clock before the first run.

The front reaches the game through the members alone (the map below), and the check of the game layer keeps it so.
So a second game's binary builds from its folder, with the front as it is.

## The map: Enclosure outside the game layer

Each reference to Enclosure outside `cpp/game/`, and each Enclosure constant in the trainer
(`python/katago/train/`), with the member that replaces it. Most callers keep the front's names (`Board`,
`BoardHistory`, `NNPos`, `NNInputs`, `SymmetryHelpers`). Only those names' definitions change. So KataGo's search keeps
its calls, and upstream releases still merge with few conflicts. The fork's tests (`cpp/tests/enclosure/`,
`cpp/tests/testenclosure*.cpp`) stay Enclosure's tests and are not in the map.

Every row is done, with these exceptions, each with its reason in its row: the OpenCL and ONNX Runtime backends and the
ONNX metadata, which wait for a machine with their toolchains (abscls-bd7), and two parts of the trainer that are not
the game's (Go's score parity vector, an option for experiments). The last sections list the assumptions of the
fork's games that stay in the front, and the names that stay.

### The front (`cpp/game/board.*`, `cpp/game/boardhistory.*`)

These files are in `cpp/game/`, outside the game layer, but they held Enclosure's code that the members replace. The
step of the rules (abscls-mkd.2) moved them onto the members and deleted their copies of the members' code: every row
is done.

| Front | Member |
|---|---|
| `Board(x, y)`, `Board::init`: 19 × 19 only, `enclosure::Position()` | Done: `new_game` (the front's error names `NAME`) |
| `x_size`, `y_size` | Done: `x_size`, `y_size`, also in `setPosition` |
| `colors` (`syncFromPosition`) | Done: `pieces` |
| `pos_hash` (`syncFromPosition`), the Enclosure tables of `initHash` (`ZOBRIST_EDGE_HASH`, `_PROTECTED_`, `_PLACED_`, `_MOVE_COUNT_`, `_MOVES_LEFT_`), `edgeIndex` | Done: `hash`, then `^ ZOBRIST_PLAYER_HASH[nextPla()]` as before. The tables moved to the game, which draws them at its first call. The situation hashes add the player's hash again, so the two cancel: that is the behaviour of before, and a change of it changes the keys. |
| `legalActions`, `getLegalMoveLocs`, `computeLegalActions`, `bitset<enclosure::PASS + 1>` | Done: `legal_actions`, `bitset<NUM_ACTIONS>`. The cache of the legal set stays in the front. |
| `isLegal` | Done: `is_legal` before the legal set exists, as before |
| `nextPla` | Done: `next_player` |
| `isGameOver` | Done: `terminal` |
| `isEmpty` | Done: `moves_played(p) == 0` |
| `playMoveAssumeLegal`, `playMoveRecorded`, `undo`, `MoveRecord::before` | Done: `play`, and a copy of `Position` |
| `getScore` | Done: `score` |
| `calculateArea`, `calculateIndependentLifeArea` | Done: `area` |
| `setPosition(enclosure::Position)` | Done: `setPosition(GameRules::Position)` |
| `MOVE_LOC_BASE`, `NUM_MOVE_LOCS`, `moveLocOfAction`, `actionOfMoveLoc` | Done: they stay in the front. A move `Loc` is `MOVE_LOC_BASE` + action, with `NUM_MOVE_LOCS = MOVE_LOC_BASE + PASS_ACTION` and pass from `PASS_ACTION`. |
| `scoreUnit()` | Done: `SCORE_UNIT` |
| `TIME_SCALE`, `GAME_LENGTH_AREA` | Done: `GAME_LENGTH / GO_GAME_LENGTH` (`GO_GAME_LENGTH` stays: it is Go's) |
| `printBoard` (the position's fields) | Done: `moves_played`, `turn_moves_left` and `score`, so that its lines (GTP's `showboard`, the logs) stay as they were. `position_text` would change them. |
| `Location::toString`, `tryOfString` (moves in letters A-S, with I) | Done: the protocol's move text and its parser (optional, `protocolText`, `parseProtocolText`), which Enclosure fills with this form. The front keeps `pass` and `null`. |
| `BoardHistory::endAndScoreGameNow` (margin from `getScore`) | Done: `score` |
| `BoardHistory` `historyHash` (in `getSituationRulesAndKoHash`) | Done: `input_hash` |
| `COMPILE_MAX_BOARD_LEN` (default 19) | Done: `MAX_BOARD_LEN`, unless the build asks for more (`USE_BIGGER_BOARDS_EXPENSIVE`) |

### The search, self-play and the engines (the rules)

The step of the rules (abscls-mkd.2) did this section. The search, self-play, GTP and the analysis engine name no
game's namespace. Where a row keeps a front name (`Board::GAME_LENGTH_AREA`, `NNPos::SCORE_SCALE`), the name's
definition reads the member.

| File | Reference | Member |
|---|---|---|
| `command/analysis.cpp` | The query field `enclosurePosition`: `enclosure::Position`'s fields, read with `enclosure::SIZE`, `MAX_MOVES`, `HISTORY_TURNS` and `make_edge`, then `Board::setPosition` | Done: the generic field `position`, read by `parse_position`. The history stays outside the one-line position (design 04). `enclosurePosition` stays through the game's own position field (optional, `QUERY_POSITION_FIELD`): the move server sends it. Its parser is an Enclosure extra (`game/enclosure/queryposition.h`). |
| `command/analysis.cpp` | `Board::NUM_MOVE_LOCS` (`avoidMoveUntilByLoc`) | Done: front (stays) |
| `command/enclosureprotocol.cpp`, `.h` | Enclosure's engine protocol (design 04): `notation::move_name`, `parse_move`, `position_string` and `parse_position`, `position_check::check`, the `enclosed_area` polygons (the describe action), `snap(scores)` (the outcome), `board.pos.legal_actions` and `terminal`, `enclosure::SIZE` (the board's and the network's size) | Done: `action_text`, `parse_action`, `position_text`, `parse_position`, `legal_actions`, `terminal`, `play`, `next_player`, `turn_moves_left`, `moves_played`, `score`, `x_size`, `MAX_BOARD_LEN`. Enclosure's own parts stay: the history planes' edges, the describe action (edges, nodes, regions), the cut edge, the outcome's text and the position check. |
| `command/enclosureengine.cpp` | The command `enclosure-engine`, `configs/analysis_enclosure.cfg` | Names (see "Names that stay") |
| `command/gtp.cpp` | `setEnclosureStart(enclosure::Position)`, `board.setPosition` | Done: `setStart(GameRules::Position)`, a start that `parse_position` reads |
| `command/gtp.cpp` | `board.pos.terminal`, `enclosureTurnError` (the player to move comes from the position) | Done: `terminal` (`Board::isGameOver`), `next_player`, in `turnError` |
| `command/gtp.cpp` | `Board::GAME_LENGTH_AREA` (resignation), `NNPos::SCORE_SCALE` (the resignation bounds) | Done: `GAME_LENGTH` (through `Board`), `SCORE_SCALE` (through `NNPos`, abscls-mkd.4) |
| `command/gtpsession.h` | `setStart(enclosure::Position)` | Done: `GameRules::Position` |
| `search/timecontrols.cpp` | `GAME_LENGTH_AREA`, `board.pos.move_count`, `board.pos.moves_left == 2` | Done: `GAME_LENGTH`, `moves_played`, `turn_moves_left(p) > 1` |
| `search/localpattern.cpp`, `.h` | The pattern hash: Zobrist keys for each edge (`2 × enclosure::PASS`) and protected edge, `Board::edgeIndex`, the move's source point (`actionOfMoveLoc / NOFF`, `enclosure::SIZE`), `board.pos.edges`, `protected_`, `contains` | Done: the move's point from `action_point`. The window's content: the position near a point (optional, `localHash`), which Enclosure fills with this edge hash, with its keys drawn as before. Without it: a hash of `pieces` in the window. |
| `search/search.cpp`, `search.h`, `searchhelpers.cpp`, `searchexplorehelpers.cpp`, `searchresults.cpp`, `patternbonustable.cpp`, `subtreevaluebiastable.cpp` | `Board::NUM_MOVE_LOCS`, `MOVE_LOC_BASE` | Done: front (stays) |
| `search/subtreevaluebiastable.cpp` | The window of 7 × 7, so that it covers a whole edge (KataGo's is 5 × 5) | Done: `LOCAL_WINDOW` (optional, given with the position near a point, through `localWindow`) |
| `search/search.cpp`, `searchhelpers.cpp`, `searchresults.cpp`, `reportedsearchvalues.cpp` | `rootBoard.scoreUnit()` | Done: `SCORE_UNIT` (through `Board::scoreUnit`) |
| `search/searchhelpers.cpp` | `Board::GAME_LENGTH_AREA` (the half-lives of `interpolateEarly`) | Done: `GAME_LENGTH` |
| `program/play.cpp` | `Board::GAME_LENGTH_AREA` (the earliest resignation, early forks) | Done: `GAME_LENGTH` |
| `program/play.cpp` | `NNPos::SCORE_SCALE` (the bounds of the komi spreads) | Done: `SCORE_SCALE` (through `NNPos`, abscls-mkd.4) |
| `program/play.cpp` | `endAndScoreGameNow(board, finalOwnership)`, `calculateArea` (final full area) | Done: `area` |
| `program/playutils.cpp` | `NNPos::SCORE_SCALE`, `SCORE_TARGET_CAP` (score steps and caps), `Board::GAME_LENGTH_AREA` (the opening moves), `Board::NUM_MOVE_LOCS` | Done: `SCORE_SCALE` and the cap (through `NNPos`, abscls-mkd.4), `GAME_LENGTH`, front |
| `command/evalsgf.cpp`, `command/writetrainingdata.cpp`, `program/playutils.cpp` | `endAndScoreGameNow(board, area)`, `calculateArea` | Done: `area` |
| `search/search.cpp` | `rootBoard.calculateArea` (the root's safe area) | Done: `area` |

### The network inputs, the symmetries and the training rows

The step of the inputs and the rows (abscls-mkd.3) did this section. The network inputs, the symmetries and the
training-row writer name no game's namespace.

| File | Reference | Member |
|---|---|---|
| `neuralnet/nninputs.h` | `#include "../game/enclosure/features.h"` | Done (abscls-mkd.4, for `NNPos`): `game/chosengame.h` |
| `neuralnet/nninputs.h` | `V8_EDGE_PLANES_START`, `V8_FEATURE_PLANES_START`, `NUM_FEATURES_SPATIAL_V8` (`4 * enclosure::NOFF + enclosure::FEATURE_PLANES`), `NUM_FEATURES_GLOBAL_V8`/`V9` (`enclosure::SCALARS`) | Done: `1 + NUM_SPATIAL_INPUTS`, `NUM_GLOBAL_INPUTS` (+ 2 in V9). The two plane starts are Enclosure's (its tests keep them). |
| `neuralnet/nninputs.h` | `NUM_SPATIAL_ELTS_V8` (`enclosure::SIZE`) | Done: `NNPos::MAX_BOARD_AREA` (`MAX_BOARD_LEN`) |
| `neuralnet/nninputs.h`, `.cpp` | `SymmetryHelpers::getSymEdge`, `getSymPosition`, `getSymBoard` | Done: `sym_position`, and `SymActionMap` over the legal set (`getSymBoard` maps it). `getSymEdge` and `getSymPosition` are deleted. |
| `neuralnet/nninputs.h`, `.cpp` | `SymmetryHelpers::getSymAction`, `SymActionMap` (its `node[enclosure::SIZE][...]`), `getSymMoveLoc`, `getSymMoveLocs` | Done: `SymActionMap` (the front's derives from the game's, built from a `Board`). `getSymAction` is deleted. `getSymMoveLoc` takes the board that the move is played on, as `getSymLoc` does. A move that is not legal there has no image (`getSymMoveLocs` gives the null move). |
| `neuralnet/nninputs.h` | `SymmetryHelpers::NUM_SYMMETRIES` (the first N symmetries) | Done: `SYMMETRIES` (a set, so that a game may allow symmetry 2 without 1), through `SymmetryHelpers::ALLOWED` (their list), `NUM_ALLOWED_SYMMETRIES` and `isAllowed`. `NUM_SYMMETRIES` stays KataGo's 8. |
| `neuralnet/nninputs.cpp` | `fillRowV8Entries`: `board.pos.observation`, `enclosure::feature_planes`, `PLANES`, `NOFF`, `OFFS`, `OFF_INDEX`, `PER_PLAYER`, `MAX_OBS_IDX`, `SCALARS`, the komi's score input (`18 * 18 * 61`), `maxHistory <= 0` | Done: `spatial_inputs`, `global_inputs` (`options.symmetry`, `options.komi`, `options.history`). The on-board plane stays the front's. |
| `neuralnet/nninputs.cpp` | `getSymXY` (`enclosure::SIZE`) | Done: deleted. The game maps its points (`GameInterface::symPoint`). |
| `neuralnet/nninputs.cpp` | `markDuplicateMoveLocs`: the image's edges, protected and placed edges against the position's | Done: `hash(sym_position(p, s)) == hash(p)`, for the allowed symmetries |
| `neuralnet/nninputs.cpp`, `nneval.cpp` | `game/enclosure/bitscan.h` (`bitsFirst`, `bitsNext` over the legal set) | Done (abscls-mkd.4, for the evaluator's gather): `core/bitscan.h`, a helper that reads a bitset, not the game |
| `neuralnet/nninputs.cpp` | `Board::playerOfIndex(board.pos.player)`, `assert(nnXLen == enclosure::SIZE)` | Done: `next_player` (`Board::nextPla`), `NNPos::MAX_BOARD_LEN` |
| `neuralnet/nneval.cpp` | `enclosureInputs` (input versions 8 and 9), `board.pos.terminal`, `SymmetryHelpers::SymActionMap` | Done: `gameInputs` (versions 8 and 9 are the game's), `terminal` (`Board::isGameOver`), `SymActionMap`. The random symmetry (from the cache key, at random, the server's) and the root's sample of symmetries (`averageMultipleSymmetries`) draw an allowed one. |
| `program/setup.cpp`, `playutils.cpp`, `command/gtp.cpp` | `nnForcedSymmetry` and `rootNumSymmetriesToSample` up to 8, `getFullSymmetryNNOutput` and GTP's raw network outputs (`kata-raw-nn`) at all 8 | Done: the allowed symmetries (`isAllowed`, `NUM_ALLOWED_SYMMETRIES`) |
| `dataio/trainingwrite.cpp` | The future occupancy (C0) and the income (C5-6): `enclosure::enclosed_area` at each turn end | Done: `TARGETS`, and `training_targets`, which Enclosure fills with this code |
| `dataio/trainingwrite.cpp` | The edges that get cut (C1: `Edge`, `igcd`, `contains`), the region occupancy 4 rounds later (C4) | Done: the same. KataGo's Go scoring of C4 (`finalWhiteScoring`) is no game's target and is deleted. |
| `dataio/trainingwrite.cpp` | The future nodes (C2-3) and the next mover's nodes (C7): `colors`, `Board::playerOfIndex(nextBoard.pos.player)` | Done: `pieces` (`Board::colors`), `next_player` (`Board::nextPla`), for a game whose `TARGETS` have them |
| `dataio/trainingwrite.cpp` | `enclosure::SIZE`, `N_POINTS`, `Board::indexOfPlayer` | Done: `x_size`, `y_size` (the row's board), and the row's player to move as the game's `mover` |
| `dataio/trainingwrite.cpp` | `NNPos::SCORE_SCALE` (score bins, the q-value score's storage), `SCORE_TARGET_CAP` | Done: `SCORE_SCALE` (through `NNPos`, abscls-mkd.4) |
| `dataio/trainingwrite.h` | The row format's comments (C5-7, C68, C69, C75) | Done: `TARGETS`, with Enclosure's meaning beside each slot |
| `game/rules.h`, `command/analysis.cpp` | The komi range, `±5052` (`(361 + 60) * 12`), and the analysis engine's message | Done: `(MAX_BOARD_LEN² + 60) * SCORE_SCALE`, the score target cap (`playutils.cpp` checks it against `NNPos::SCORE_TARGET_CAP`). `rules.h` includes both parts, after `rules.cpp`'s check of the rules part. |
| `cpp/game/boardhistory.cpp` | `historyHash` | Done (the step of the rules): `input_hash` |

### The policy layout and the backends

Done (abscls-mkd.4). `NNPos` takes the network's board, the policy's channels and size, and the score scale from the
game. The evaluator, the model description and the backends name no game. An action is its policy index (see
"Conventions"), so none of them needs `action_point` or `point_action`.

| File | Reference | Member |
|---|---|---|
| `neuralnet/nninputs.h` | `NNPos::NUM_OFFSETS = enclosure::NOFF`, `MAX_NN_POLICY_SIZE` (`static_assert == enclosure::PASS + 1`), `locToPos` and `posToLoc` (`assert(... == enclosure::SIZE)`, the action as the policy index) | Done: `NNPos::POLICY_CHANNELS`, `MAX_NN_POLICY_SIZE = NUM_ACTIONS`, and `NNPos::MAX_BOARD_LEN = MAX_BOARD_LEN` (the network's board, which can be smaller than the build's `Board::MAX_LEN`). `locToPos` and `posToLoc` assert that the network is the largest board. |
| `neuralnet/nninputs.h` | `NNPos::SCORE_SCALE = 12.0`, `SCORE_TARGET_CAP` | Done: `SCORE_SCALE`. The cap comes from the game's largest board. |
| `neuralnet/desc.cpp`, `.h` | `NNPos::NUM_OFFSETS` (the policy convolution's channels per output) | Done: `NNPos::POLICY_CHANNELS` |
| `neuralnet/cudaandrocmbackend.inc` | `NNPos::NUM_OFFSETS` (15 places), and the policy gather's `static_assert(enclosure::PASS < (1 << 15))` and `testAssert(a <= enclosure::PASS)` | Done: `NNPos::POLICY_CHANNELS`, and the gather's checks on `MAX_NN_POLICY_SIZE` |
| `neuralnet/eigenbackend.cpp`, `trtbackend.cpp`, `onnxmodelbuilder.cpp` | `NNPos::NUM_OFFSETS` | Done: `NNPos::POLICY_CHANNELS` (and in the WebAssembly module's `evaluator.cpp` and `jsbackend.cpp`) |
| `neuralnet/nneval.cpp` | `enclosure::PASS` (the policy gather, the legal mask, `policySize == PASS + 1`), `board.legalActions()` | Done: `NNPos::PASS_POS` (`PASS_ACTION`), the front's legal set read by policy index. The evaluator refuses a network board other than the largest board when it starts. |
| `neuralnet/openclbackend.cpp`, `onnxbackend.cpp`, `cudaandrocmhelpers.*`, `modelversion.*`, `onnxmodelbuilder.h` | Comments: the policy channels, the input versions | Done for `cudaandrocmhelpers.h`. Not done for `openclbackend.cpp` and `onnxbackend.cpp`: they never took the fork's policy. They copy one policy channel per point, size their buffers for it, and apply the symmetry themselves. Neither builds on the training machine, so their change waits for a machine with their toolchains (abscls-bd7). The input versions: done (`modelversion.*` say that versions 8 and 9 are the game's inputs, abscls-mkd.3). The model's game: in the model file's header since the trainer's step ("The model file's game"). The ONNX metadata does not carry it yet (abscls-bd7). |

### The trainer (`python/katago/train/`)

Done (abscls-mkd.5). The trainer reads a game's facts from the game's module ("The trainer's module"), and a model
file names its game ("The model file's game"). The model and data code hold no Enclosure constant.

| Constant | Where | Member |
|---|---|---|
| `NUM_OFFSETS = 48`, `policy_len(pos_len)` | `model_pytorch.py` (the policy convolution and its reshape), `metrics_pytorch.py` (`policy_len`, the soft-policy mask), `enclosure_symmetry.py` | Done: `POLICY_CHANNELS`, and `policy_len(pos_len, policy_channels)`. `enclosure_symmetry.py` is now Enclosure's module, `games/enclosure.py`. |
| `SCORE_BIN_WIDTH`, `SCORE_SCALE = 12.0` | `model_pytorch.py` (the score bins, the output multipliers), `metrics_pytorch.py` (the Huber deltas, the loss weights, the q-value score's storage `60 / SCORE_SCALE`) | Done: `SCORE_SCALE`, also the bins' width |
| `TIME_SCALE = 120 / 328` | `model_pytorch.py`, `metrics_pytorch.py` (the variance-time Huber delta) | Done: `GAME_LENGTH / GO_GAME_LENGTH` (`GO_GAME_LENGTH` = 328 stays: it is Go's) |
| 215 spatial inputs, 5 or 7 global inputs | `modelconfigs.get_num_bin_input_features`, `get_num_global_input_features` | Done: `1 + NUM_SPATIAL_INPUTS`, `NUM_GLOBAL_INPUTS` (+ 2 in V9, `PLAYOUT_GLOBAL_INPUTS`) |
| `ENCLOSURE_HISTORY_PLANES_START = 211`, `ENCLOSURE_NUM_HISTORY_PLANES = 4` | `data_processing_pytorch.apply_history_dropout` | Done: `1 + HISTORY_PLANE`, `NUM_HISTORY_PLANES` (the network's planes count the on-board plane) |
| `enclosure_symmetry` (`SIZE`, `NUM_OFFSETS`, `OFFSETS`, the edge planes' anchors, `NUM_SYMMETRIES = 8`) | `data_processing_pytorch.py` (the augmentation) | Done: `SYMMETRIES`, and the module's `apply_symmetry_inputs`, `apply_symmetry_policy_targets` and `apply_symmetry_policy` |
| `VALUE_TARGETS_NCHW_CHANNELS = 8`, the targets in C1, C4, C5-6, C7 and their flags C68, C69, C75 | `data_processing_pytorch.py`, `metrics_pytorch.py` | Done: `TARGETS` leaves out C0 for a game without `TARGET_OWNERSHIP`, and the rows' weights and flags leave out the other slots. The 8 slots and the flags are the rows' format, which every game shares. |
| The storage × 120 of C0, C4 and C5-6 | `metrics_pytorch.py` | Done: the rows' storage for every game (`TargetValues::STORAGE_SCALE`) |
| The future-position head's 4 outputs: KataGo's 2, then the income logits | `model_pytorch.py` (`conv_futurepos`), `metrics_pytorch.py` | Done: every game has the 4 outputs. The last 2 train only with `TARGET_INCOME`. |
| The score parity vector, kept at zero (Go's komi parity) | `model_pytorch.py` | None: Go's. A game whose last global input is komi would need KataGo's. |
| `load_model(..., pos_len=19)` | `load_model.py` | Done: `MAX_BOARD_LEN` |
| `ENCLOSURE_ZERO_INPUT_CHANNELS` | `data_processing_pytorch.py` | None: an option for experiments (a name) |

The RGSC heads and their targets (`model_pytorch.py`, `metrics_pytorch.py`, `cpp/program/rgsc.h`) come from the
search, not from the game. They are not in the map. What the game chooses is how RGSC's buffer keeps its states: moves
from the empty board, or the game's state texts (`state_text`, an optional member of the rules part).

Two files beside the trainer also held Enclosure's facts:

- `python/export_model_pytorch.py`: the export of the policy convolution (`NUM_OFFSETS` channels for each output), and
  the V9 inputs (7 global inputs) in the model version. Done: the policy head's channels, the game's inputs, and the
  game in the model file. `-game` names the game of a random-initialized model.
- `python/shuffle.py`: the rows' C5-7 channels, V9's 7 global inputs and V12's planes (`pad_inputs_to_widest`,
  `pad_planes_to_widest`). No change: these are the rows' format, which every game shares, and the padding takes the
  widest file's width.

### The training scripts (`python/selfplay/`)

Done (abscls-mkd.7). The asynchronous setup runs a run of any game. The engine repository's pipeline gives it the
run's game (`GAME`) from the run's settings, `games/GAME/runs/RUN.env` (design 07 of the engine repository, "The
pipeline").

| File | Reference | Now |
|---|---|---|
| `enclosure_async.sh` | Enclosure's asynchronous setup: `configs/training/selfplay_enclosure.cfg`, the load check's `configs/analysis_enclosure.cfg`, the binary's default `builds/cuda/katago` of the engine repository's work space | Done: `async_start.sh`, for the game `GAME`: `selfplay_GAME.cfg` (`SELFPLAY_CONFIG`), `analysis_GAME.cfg` (`ANALYSIS_CONFIG`), and `KATAGO_BIN`, which the caller sets. It refuses a binary of another game, a game without a self-play config and a game without a trainer's module, before it makes a folder. Done (abscls-mkd.14): `analysis_connect4.cfg` is Connect-4's load check. |
| `async_start.sh` (the levers' defaults) | The shuffle's row counts were Enclosure's (2,900 and 3,300 rows, for its rows of 186 KB), and no thread pool had a default, so each Python part and Eigen's self-play took the machine's cores. A run of another game copied KataGo's counts and the thread settings into its settings (abscls-mkd.14). | Done (abscls-mkd.19): `async_parts.sh` (`run_defaults`, `game_defaults`) gives every lever its default, for any game. The shuffle's counts are KataGo's for the same bytes of the game's rows, at most KataGo's: the trainer's module gives the row's bytes (`row_bytes`). When the run has fewer CPUs than the machine, Eigen's self-play takes a network thread a CPU of the run, and each Python part one torch thread. A pilot (`PILOT=1`, `pilot_selfplay.sh`) is a short run. The engine repository's dry run of a start prints the levers in force (its design 07, "The pipeline"). |
| `enclosure_async_stop.sh`, `enclosure_parts.sh` | The parts' stop, the trainers and the parts' limits | Done: `async_stop.sh`, `async_parts.sh`. They held no game's code, and the limits' defaults are the same. |
| `train.sh` | `-pos-len 19` | Done: `-pos-len` from `POS_LEN`, which `async_start.sh` sets to the game's `MAX_BOARD_LEN` (the trainer's module). Upstream's 19 without it. `async_start.sh` gives every trainer `-game GAME`. |
| `main.cpp` (`katago version`) | The binary's game was not visible from outside | Done: the line `Game: NAME` (`GameRules::NAME`), which `async_start.sh` checks against the run's game |
| `enclosure_loop.sh` | v2a's synchronous loop, retired on 2026-10-05 | Stays Enclosure's (the engine repository's `games/enclosure/runs/ownscripts.txt`) |

### The last references (the contract step)

The last step of design 07's order of the work (expand, migrate, contract) is the contract step (abscls-mkd.8). It did
this section and added the check of the game layer. It also decided the three
assumptions of Enclosure in the search that the step of the rules left, and the action's formula, written out in 12
places of Enclosure's code.

| File | Reference | Member |
|---|---|---|
| `command/enclosureprotocol.cpp`, `.h`, `command/enclosureengine.cpp` | Enclosure's engine protocol and its command `enclosure-engine`, built into every `katago` binary | Done: Enclosure's own files (`game/enclosure/ownfiles.txt`), in Enclosure's binary only. `enclosure-engine` is a `MainCmds::GameCommand`. |
| `tests/testenclosuresearch.cpp`, `command/runtests.cpp`, `tests/tests.h` | Enclosure's search tests (its edges, `enclosure::`) and their command `runenclosuretests` | Done: Enclosure's own file, which now holds the command (a `GameCommand`). `runtests.cpp` and `tests.h` no longer name them. |
| `main.cpp`, `main.h` | `MainCmds::enclosureEngine`, `runenclosuretests` | Done: the game's own commands, which register themselves (`MainCmds::GameCommand`). `main.cpp` dispatches them and lists them in its help. |
| `tests/testenclosurenn.cpp` | Enclosure's board, 19: the network's board, the ownership's 361 points, the rows' board | Done: `NNPos::MAX_BOARD_LEN`, so `runenclosurenntests` and `runenclosurerowtests` work for any game |
| `search/timecontrols.cpp` | `increment / 2`: an edge counts half the increment, which comes once a turn of two edges. The first edge of a turn takes at most half the time left. | Done: `increment / TURN_MOVES` (optional, `turnMoves`: 1 by default, Enclosure's 2), and at most the time left divided by `turn_moves_left` |
| `game/boardhistory.cpp` | `isLegalTolerant`: a pass at any time, as Enclosure's timeout pass | Done: what the game's `replay_tolerates` accepts beside the legal moves (optional, `replayTolerates`: nothing by default, Enclosure's pass). Without it a replay plays the legal moves only, so Connect-4's GTP and analysis engine refuse a pass. |
| `game/board.*`, `command/gtp.cpp`, `tests/tests.h` | `Board::DEFAULT_LEN`: the default board is square, 19 or the largest board | Done: `Board::DEFAULT_X_SIZE`, `DEFAULT_Y_SIZE`, the game's standard board (optional `STANDARD_X_SIZE`, `STANDARD_Y_SIZE`: the largest board by default, Connect-4's 7 × 6). The Python binding's `Position()` starts on it too, so Connect-4 needs no extras. |
| `game/enclosure/engine.h`, `features.h`, `notation.h`, `gamerules.h`, `gameencoding.h`, Enclosure's tests and binding extras | The action's formula, `(y * SIZE + x) * NOFF + k` and its inverse | Done: one copy, `action_at` and `action_source` (`engine.h`), which `point_action` and `action_point` call, and `action_between` for the edge between two points. The turn's two edges are one constant, `TURN_EDGES`. |

### Assumptions that stay in the front and the search

These do not name a game, but they restrict the games that the front supports. A game that needs these features must
first adapt their callers:

- `Board::simpleRepetitionBoundGt` and `KoHashTable`: a position never recurs (Enclosure's edge count only rises), so
  the graph search merges transposed turns. A game whose positions can recur needs KataGo's repetition handling.
  Connect-4's positions do not recur either (a disc stays).
- A move as a point: the opening books (`book/book.cpp`, `command/genbook.cpp`) and the moves of a start position
  (`command/startposes.cpp`) map a move with `getSymLoc`, as Go's moves are points. A game's move is a move `Loc`
  (`MOVE_LOC_BASE` plus the action), so a game that uses them needs the game's images of moves there
  (`getSymMoveLoc`). They use the allowed symmetries only ("The second game's build", below). `dataio/sgf.cpp`
  transposes a Go handicap on a 19 × 19 board, which no game of the fork reaches.

### The second game's build (abscls-mkd.13)

Connect-4's binary showed four assumptions of the front that Enclosure's boards and policy hide. Each is done, and
Enclosure's outputs do not change:

| File | Assumption | Change |
|---|---|---|
| `search/searchnode.h` | A node's children are in three arrays of 8, 56 and the policy's size less 64 (`SearchChildrenSizes`). A policy smaller than 64 (Connect-4: 50) gives a negative size. | Done: the third size is at least the second, so a game with fewer moves never grows to it. |
| `program/setup.cpp`, `command/gtp.cpp` | A command sizes the network by the boards that it plays (self-play's, a match's, GTP's), and asks for that size exactly (`requireExactNNLen`). Connect-4's board, 7 × 6, is smaller than its network's, and the evaluator refuses a network board other than the game's largest. | Done: a board up to the game's largest gets the largest board, and only a board that fills it asks for the exact size. GTP keeps the game's standard board as its default board, and makes its network again only when the board changes that. |
| `tests/testenclosurenn.cpp`, `tests/enclosure/testnnoutputs.cpp` | The positions' board fills the network's. | Done: the exact size only when the standard board fills the network's board. |
| `search/patternbonustable.cpp`, `search/searchresults.cpp`, `book/book.cpp`, `command/genbook.cpp`, `command/startposes.cpp` | All 8 symmetries, where a feature that Enclosure does not use loops over them. `getSymBoard` asserts that its symmetry is allowed. | Done: the allowed symmetries (`SymmetryHelpers::ALLOWED`): the pattern bonus and the avoided patterns (with the move's image from `getSymMoveLoc`, as the table's keys are moves), the analysis engine's `symHash`, the book's hash, its choice of a node's symmetry and its sampled positions (`genbook`'s random symmetry and its mean over the symmetries), and the random symmetry of a start position. With all 8 allowed, the loops are the same. |

### Names that stay

These name Enclosure but hold no game code. Each stays, for its reason:

- The query field `enclosurePosition`: the protocol, which the live move server sends. The analysis engine reads it
  through the game's own query field (`QUERY_POSITION_FIELD`), so the engine names no game.
- The commands `enclosure-engine` and `runenclosuretests`, and `configs/*_enclosure.cfg`: Enclosure's own commands and
  settings. Only Enclosure's binary has the commands.
- The commands `runenclosurerowtests` and `runenclosurenntests` (`main.cpp`, `main.h`, `command/runtests.cpp`, and
  their functions `Tests::runEnclosureRowTests`, `runEnclosureNNTests`): generic code with Enclosure's name.
  `build_verify.sh` runs them on the production binary too, which knows these names only, and the engine
  repository's tools call them, Connect-4's checks too (`games/connect4/tests/test_network.py`). A generic name,
  with the old one as an alias, is later work.
- The network check against PyTorch, `python/enclosure_nn_check.py` and `tests/enclosure/testnnoutputs.cpp`: generic
  code with Enclosure's name. `verify_build.sh` calls them by these names, and the WebAssembly build includes
  `testnnoutputs.cpp` by its path.
- The options of the fork's speed work: `ENCLOSURE_FUSED_CONV` (`cudaandrocmbackend.inc`), `ENCLOSURE_EIGEN_SIMD`,
  `ENCLOSURE_WINOGRAD_NV`, `ENCLOSURE_WINOGRAD_NT`, `ENCLOSURE_DIRECT_CONV`, `ENCLOSURE_EVAL_RAW`
  (`eigenbackend.cpp`), `ENCLOSURE_COOP_THREADS` (`nneval.*`), `ENCLOSURE_NO_POLICY_GATHER` (`nneval.cpp`). The build
  scripts and the runs' settings set them. A rename needs the old names as fallbacks, as the pipeline's renames have
  (`ENGINES_*`, with `ENCLOSURE_*` and `V2*` as fallbacks).

### Changes of the fork that are already generic

These files mention Enclosure in comments only. Their code asks the position (the player to move, the end) through the
front, or scales a constant through a front name in the tables above: `command/dumponnx.cpp`, `evalsgf.cpp`,
`genbook.cpp`, `selfplay.cpp`, `writetrainingdata.cpp`, `core/os.h`, `core/threadaffinity.h`, `dataio/sgf.h`,
`neuralnet/cudaandrocmhelpers.*`, `program/play.h`, `rgsc.h`, `selfplaymanager.cpp`, `setup.cpp`,
`search/analysisdata.cpp`, `asyncbot.*`, `searchnnhelpers.cpp`, `searchupdatehelpers.cpp`.
