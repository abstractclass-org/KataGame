# The game interface: the C++ reference

**Status: decided, being built.** KataGame plays a game that supplies the game interface. The build chooses one game, as
it chooses one backend. This file is the C++ reference: the members, the conventions, the checks, and how to add a
game. Its last part is the map of the code that still reaches Enclosure directly, with the member that replaces each
reference. The decision and its reasons are design 07 of the engine repository
(`docs/design/07-game-interface.md`). Enclosure's own design stays in [ENCLOSURE.md](ENCLOSURE.md).

The interface exists. Enclosure implements both parts. Connect-4 (`cpp/game/connect4/`) implements its rules part. The
front (`Board`, `BoardHistory`), the search, self-play, GTP and the analysis engine reach the rules through the rules
part. The policy layout and the backends read the interface (`NNPos`). The rest of the network code (the inputs, the
symmetries, the training rows) still uses Enclosure directly: it moves onto the encoding part step by step, and the map
below gives the work of each step.

* [The files](#the-files)
* [Choosing the game](#choosing-the-game)
* [Conventions](#conventions)
* [The rules part](#the-rules-part)
* [The network encoding part](#the-network-encoding-part)
* [The members still to build](#the-members-still-to-build)
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
| `cpp/game/board.*`, `cpp/game/boardhistory.*` | The front: KataGo's `Board` and `BoardHistory`, which the search calls. They reach the game's rules through the members only. |
| `cpp/tests/enclosure/testgameinterface.cpp` | Enclosure's members, against the front's code where the front still has a copy, and the defaults of the optional members with a stub game. |

All members are static, and every file is a header. So the compiler sees the same code as in a build for one game, and
the WebAssembly build, the binding and the standalone tests need no new source file.

## Choosing the game

- **CMake:** `-DGAME=enclosure` (the default). The games are the folders of `cpp/game/` that hold a `gamerules.h`. An
  unknown name stops `cmake` with the list of games. A game without `gameencoding.h` (Connect-4 for now) also stops
  it: `katago` needs both parts, and the Python binding needs only the rules part. The option defines
  `KATAGAME_GAME=<name>`.
- **The headers:** `chosenrules.h` includes `<name>/gamerules.h` and makes `chosengame` the alias of the namespace
  `<name>`. `chosengame.h` includes `<name>/gameencoding.h`. A game's folder and its namespace have the same name.
- **Other builds** (the WebAssembly module, the fork's standalone tests) define nothing and get Enclosure. So every
  build command of today still works. The Python binding defines `KATAGAME_GAME` for each game that it builds
  (`core/binding/build.sh` in the engine repository).
- **Every build of `katago` checks both parts:** `board.h` includes `chosengame.h`, so each file that uses the front
  checks both parts. `rules.cpp`, which includes no `Board`, includes `chosenrules.h` alone: it shows that the rules part
  compiles without the encoding part.

## Conventions

- **Players:** 0 is the first player (KataGo's Black), 1 the second (White). Enclosure's Blue is 0.
- **Actions:** an `int` from 0 to `NUM_ACTIONS - 1`. Pass is the last, `PASS_ACTION`. Every game has pass. A game whose
  rules have no pass never lists it as legal (Connect-4).
- **Actions on the largest board:** a game numbers its actions on a board of `MAX_BOARD_LEN`, point by point:
  `(y * MAX_BOARD_LEN + x) * POLICY_CHANNELS + channel`, pass last. So an action keeps its number on every board size,
  and `NUM_ACTIONS` is `MAX_BOARD_LEN² × POLICY_CHANNELS + 1`.
- **An action is its policy index:** the network's board is the game's largest board (`nnXLen = nnYLen =
  MAX_BOARD_LEN`). So the network's policy index of an action's point and channel is the action itself. The search,
  the evaluator and the backends use an action as its policy index and need no map. The encoding part's check refuses
  a `NUM_ACTIONS` that breaks this rule.
- **Points:** `(x, y)`, `0 <= x < x_size`, `0 <= y < y_size`, with `y = 0` at the bottom (row 1 in the notation).
  Arrays of points are `[y * x_size + x]`.
- **Symmetries:** KataGo's 8 and its numbers: bit `0x2` flips x, bit `0x1` flips y, then bit `0x4` transposes
  (`GameInterface::symPoint`). A game allows a subset. The identity is always in it.
- **No exceptions:** the WebAssembly build has none. A parser returns `false` or `-1` and gives the reason.

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
| `hash(p)` | A hash of the whole position, the player to move included, so that equal hashes mean equal positions. The front also mixes in KataGo's hash of the player to move. | Board's Zobrist hash, from the same seed (see below) |
| `pieces(p, owner)` | `owner[y * x_size + x]`: the player whose piece is on the point, or -1. KataGo's `colors`. | each player's nodes |
| `area(p, owner)` | `owner[y * x_size + x]`: the player who owns the point now, or -1. KataGo's area: the final ownership at the end. | the points that only one player's regions enclose |
| `action_text(action)` | The action in the game's notation (design 04 of the engine repository), for a legal action or pass. | `notation::move_name` (an edge that leaves the board has no name) |
| `parse_action(p, text, why)` | The legal action of `p` that the text names, or -1 and the reason. | `notation::parse_move` |
| `position_text(p)` | The one-line position. | `notation::position_string`: the five fields of design 04 |
| `parse_position(text, p, why)` | Sets `p` from a one-line position, or returns false and the reason. | `notation::parse_position` (the history planes come back empty) |

**Enclosure's hash** covers the edges by owner, the protected and the placed edges, the edge count, the edges left in
the turn and the scores. Its tables are the ones that `Board::initHash` drew before the interface: `GameRules` draws
them from the same seed in the same order, at its first call (a function-local static, which is thread safe), so
`hash(p)` keeps its values. The player to move is in the hash through the edge count and the edges
left in the turn, which fix it in every position that a game reaches.

### Optional members

A game may leave out an optional member. For each one, `gameinterface.h` has a detector (as the checks have) and a
helper, which calls the game's member when the game has it, and gives the default when it does not. The front calls
the helper. The rules part's check requires only that the members of a pair come together.

| Members | Meaning | Default (helper) | Enclosure |
|---|---|---|---|
| `protocol_text(action)`, `parse_protocol_text(text, action)` | An action other than pass as the engine protocol (GTP, the analysis engine) writes and reads it. The parser reads trimmed text in any case, does not check legality, and returns false for text that names no action. The front reads and writes pass (`pass`). | `action_text`, and the action whose `action_text` is the text, ignoring case (`protocolText`, `parseProtocolText`). The parser tries each action, so it suits games with few actions. | The form of the move server and the sites: column letters A to S (I included, unlike the notation), such as `D10-D13` |
| `NUM_LOCAL_KEYS`, `LOCAL_WINDOW`, `local_hash(p, cx, cy, xRadius, yRadius, flipPlayers, keys)` | The position near the point (cx, cy), in the window of xRadius and yRadius: the search's local pattern hash (the subtree value bias, the pattern bonus). The front draws `NUM_LOCAL_KEYS` random keys for it, after KataGo's keys. `flipPlayers` swaps the players. `LOCAL_WINDOW` is the side of the subtree value bias's window (odd), large enough for a move's reach. | Each point of the window on the board hashed with its piece (`pieces`), as KataGo hashes stones: KataGo's keys `zobristLocalPattern` by its colors (empty, the first player, the second), without its atari flag. KataGo's window of 5 (`localHash`, `localWindow`). | The edges of either player with an end in the window, and whether each is protected: the hash of before the interface, byte for byte (3 × 17,328 keys, in the order that the hasher drew them). The window of 7, which covers a whole edge from its source. |
| `QUERY_POSITION_FIELD`, `parse_query_position(value, p, why)` | A position field of the game's own in an analysis query, beside the generic field `position` (the one-line position): its key, and the reader of its value. The reader is a template on the JSON type, so that the rules part does not need the JSON library. It may throw the JSON library's exceptions, which the analysis engine catches. | None: the analysis engine takes `position` only (`queryPositionField`, `parseQueryPosition`) | `enclosurePosition` (`queryposition.h`): the fields of `Position` as JSON, with the last turns' edges for the history planes, which the one-line position leaves out. The move server sends it. |

## The network encoding part

`struct GameEncoding`. The search, self-play and the trainer need it. The Python binding uses only its symmetries
(`SYMMETRIES`, `sym_position`, `SymActionMap`), when the game has this part, for the conformance suite.

| Member | Meaning | Enclosure |
|---|---|---|
| `Rules` | The game's rules part. | `GameRules` |
| `NUM_SPATIAL_INPUTS` | The input planes that the game writes. The front puts KataGo's on-board plane first, so the network sees `1 + NUM_SPATIAL_INPUTS` planes. | 214: 4 × 48 edge planes, 22 feature planes (V8 has 215 with the on-board plane) |
| `NUM_GLOBAL_INPUTS` | The global inputs that the game writes. | 5 (V9 adds KataGo's 2 playout inputs after them) |
| `spatial_inputs(p, options, set)` | Calls `set(x, y, plane)` for each input that is 1, at the image under `options.symmetry`. A template on `set`, so that each writer of the front is inlined. | `fillRowV8`'s edge planes (both ends of each edge) and feature planes |
| `global_inputs(p, options, out)` | Writes the global inputs. | the observation's scalars, with komi as a starting score |
| `input_hash(p)` | What the inputs read beyond `hash(p)`. The front mixes it into the network cache's key. | the history planes' edges (`BoardHistory`'s history hash) |
| `POLICY_CHANNELS` | Policy channels per point. | 48: one per offset |
| `action_point(a, x, y, channel)` | The point and the channel of an action other than pass. | the edge's source point and offset |
| `point_action(x, y, channel)` | The action at that point and channel: `(y * MAX_BOARD_LEN + x) * POLICY_CHANNELS + channel`, which is also its policy index. | `(y * 19 + x) * 48 + channel` |
| `SYMMETRIES` | The allowed symmetries: bit s for KataGo's symmetry s. | `0xFF`: all 8 |
| `sym_position(p, s)` | The image of the position. | every edge mapped: `SymmetryHelpers::getSymPosition` |
| `SymActionMap` | `SymActionMap(p, s)(a)`: the image of a legal action of `p`, an action of `sym_position(p, s)`. Built once for each position and symmetry. | `SymmetryHelpers::SymActionMap` (the edge's direction can change) |
| `TARGETS` | The optional training targets that the game writes. | all six (below) |
| `SCORE_SCALE` | The game's score spread against Go's. KataGo's constants in points scale by it. | 12 (ENCLOSURE.md, "Score scale") |
| `SCORE_UNIT` | The search's score utility unit, in points, where KataGo uses the square root of the board's area. | 140.5845 = √(18 · 18 · 61) |
| `GAME_LENGTH` | A game's mean length in moves, for KataGo's game-phase constants. | 120 |

`InputOptions` holds what the front gives the input members:

| Field | Meaning |
|---|---|
| `symmetry` | An allowed symmetry. The inputs are those of the image. |
| `history` | False: leave out what the position keeps of earlier turns (KataGo's `maxHistory <= 0`). |
| `komi` | The rules' komi, for a game whose inputs encode it. |

**The training targets** are the slots of KataGo's spatial targets (`valueTargetsNCHW`) and the two that the fork
added. The game gives each slot that it writes its own meaning. The front writes 0, with weight 0, in each other slot.
`TARGETS` declares the slots. The member that computes a game's own targets comes with the step of the training rows
(see "The members still to build").

| Flag | Slot | Enclosure's meaning (ENCLOSURE.md, "Spatial targets") |
|---|---|---|
| `TARGET_OWNERSHIP` | C0 | the future occupancy: the mean over the turn ends after the row of +1 inside the mover's regions, -1 inside the opponent's |
| `TARGET_SEKI` | C1 | the edges that get cut |
| `TARGET_FUTURE_PIECES` | C2-3 | the nodes 8 and 32 edges later (KataGo's future positions, from `pieces`) |
| `TARGET_SCORING` | C4 | the region occupancy 4 rounds later |
| `TARGET_INCOME` | C5-6 | the territory income of each player |
| `TARGET_NEXT_PIECES` | C7 | the next mover's nodes after the row's move, for the symmetries of policy channel 1 (not a loss) |

## The members still to build

These are decided (design 07). Each comes with the step that first needs it.

| Member | Part | What it gives | Enclosure | Without it | Step |
|---|---|---|---|---|---|
| The history planes | encoding | The first plane and the count of the planes that `InputOptions::history` leaves out | The last 4 feature planes | Every game has it | The inputs and the rows (abscls-mkd.3) |
| A game's own training targets | encoding | The values per point of the slots that the game gives its own meaning, from the game's positions after the row, before KataGo's storage | Today's targets in `trainingwrite.cpp` | Every slot in `TARGETS` that the front does not compute from `pieces` needs it | The inputs and the rows (abscls-mkd.3) |

## The checks

**A game that lacks a member does not compile.** Each game's header ends with its part's check. A missing member gives
one error that names it. This is the `katago` build with `next_player` removed from Enclosure's `GameRules`:

```
game/../game/enclosure/../gameinterface.h:177:35: error: static assertion failed: game interface, rules part: the
game lacks next_player(const Position&) -> int
make[3]: *** [CMakeFiles/katago.dir/build.make:429: CMakeFiles/katago.dir/game/board.cpp.o] Error 1
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
`tests/data/enclosure_games.txt` and 60 random games (with timeout passes and komi). The front calls the rules part,
so the rules part has one copy. Where the front still has a copy of a member, the test compares the two. At each
position it checks:

- the rules part against itself: the player to move and the edges left (against the 1-2-2 turn order), `is_legal`
  against the legal actions (every action at each 200th position), the moves and the position in the notation and the
  engine protocol's moves (each read back), and the result at the end.
- the encoding part against `NNInputs`, `NNPos` and `SymmetryHelpers`: the inputs (`fillRowV8` with and without history
  and at the game's komi, and `fillRowV8Packed` at each symmetry), the policy index (`NNPos::locToPos`), the image
  position and the action map (the images of the legal actions are the image's legal actions).
- the constants that `NNPos` and `SymmetryHelpers` still hold.

It also checks the optional members: Enclosure's (the protocol's moves `A10-D10` and `S19-P16`, through `Location`
too, the window's keys and the query field), and the defaults with a stub game that has none of them (the protocol's
text read back in any case, and the window's hash at a corner, key by key, with and without `flipPlayers`). The front
has no copy of the hash any more, so the test pins the values of KataGame before the interface at three positions of
the first game: `pos_hash`, the situation hash and the local pattern hash at three symmetries.

Build and run it as the other standalone tests (`tests/enclosure/build.sh`), from `katagame/cpp`:

```
TESTS=testgameinterface bash tests/enclosure/build.sh BUILD_DIR OUT_DIR
OUT_DIR/testgameinterface tests/data/enclosure_games.txt
```

## Adding a game

1. Make the folder `cpp/game/<game>/` with `gamerules.h` (`struct GameRules` in the namespace `<game>`) and
   `gameencoding.h` (`struct GameEncoding`). End each with its check:
   `static_assert(GameInterface::checkRules<GameRules>(), "...")`. Connect-4's `gamerules.h` is a short example of
   the rules part.
2. Build with `-DGAME=<game>`, in a build folder for that game and backend.
3. In the engine repository, make `games/<game>/` a Python package and build the game's rules module with
   `sh core/binding/build.sh <game>`. The module's common API needs no code of the game's own. The game's Python
   extras, if any, go in `games/<game>/binding/extras.h`. The conformance suite (`core/tests/test_conformance.py`)
   checks the module (design 07, "The Python binding").

A second game builds only after the front reaches Enclosure through the members alone (the map below).

## The map: Enclosure outside the game layer

Each reference to Enclosure outside `cpp/game/`, and each Enclosure constant in the trainer
(`python/katago/train/`), with the member that replaces it. Most callers keep the front's names (`Board`,
`BoardHistory`, `NNPos`, `NNInputs`, `SymmetryHelpers`). Only those names' definitions change. So KataGo's search keeps
its calls, and upstream releases still merge with few conflicts. The fork's tests (`cpp/tests/enclosure/`,
`cpp/tests/testenclosure*.cpp`) stay Enclosure's tests and are not in the map.

### The front (`cpp/game/board.*`, `cpp/game/boardhistory.*`)

These files are inside the game layer, but they held Enclosure's code that the members replace. The step of the rules
(abscls-mkd.2) moved them onto the members and deleted their copies of the members' code: every row is done.

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

| File | Reference | Member |
|---|---|---|
| `neuralnet/nninputs.h` | `#include "../game/enclosure/features.h"` | Done (abscls-mkd.4, for `NNPos`): `game/chosengame.h` |
| `neuralnet/nninputs.h` | `V8_EDGE_PLANES_START`, `V8_FEATURE_PLANES_START`, `NUM_FEATURES_SPATIAL_V8` (`4 * enclosure::NOFF + enclosure::FEATURE_PLANES`), `NUM_FEATURES_GLOBAL_V8`/`V9` (`enclosure::SCALARS`) | `1 + NUM_SPATIAL_INPUTS`, `NUM_GLOBAL_INPUTS` (+ 2 in V9) |
| `neuralnet/nninputs.h` | `NUM_SPATIAL_ELTS_V8` (`enclosure::SIZE`) | `MAX_BOARD_LEN` |
| `neuralnet/nninputs.h`, `.cpp` | `SymmetryHelpers::getSymEdge`, `getSymPosition`, `getSymBoard` | `sym_position`, and `SymActionMap` over the legal set (`getSymBoard` maps it) |
| `neuralnet/nninputs.h`, `.cpp` | `SymmetryHelpers::getSymAction`, `SymActionMap` (its `node[enclosure::SIZE][...]`), `getSymMoveLoc`, `getSymMoveLocs` | `SymActionMap` |
| `neuralnet/nninputs.h` | `SymmetryHelpers::NUM_SYMMETRIES` (the first N symmetries) | `SYMMETRIES` (a set, so that a game may allow symmetry 2 without 1) |
| `neuralnet/nninputs.cpp` | `fillRowV8Entries`: `board.pos.observation`, `enclosure::feature_planes`, `PLANES`, `NOFF`, `OFFS`, `OFF_INDEX`, `PER_PLAYER`, `MAX_OBS_IDX`, `SCALARS`, the komi's score input (`18 * 18 * 61`), `maxHistory <= 0` | `spatial_inputs`, `global_inputs` (`options.komi`, `options.history`). The on-board plane stays the front's. |
| `neuralnet/nninputs.cpp` | `getSymXY` (`enclosure::SIZE`) | The front's point map (KataGo's, on the board's size) |
| `neuralnet/nninputs.cpp` | `markDuplicateMoveLocs`: the image's edges, protected and placed edges against the position's | `hash(sym_position(p, s)) == hash(p)` |
| `neuralnet/nninputs.cpp`, `nneval.cpp` | `game/enclosure/bitscan.h` (`bitsFirst`, `bitsNext` over the legal set) | Done (abscls-mkd.4, for the evaluator's gather): `core/bitscan.h`, a helper that reads a bitset, not the game |
| `neuralnet/nninputs.cpp` | `Board::playerOfIndex(board.pos.player)`, `assert(nnXLen == enclosure::SIZE)` | `next_player`, `x_size`/`MAX_BOARD_LEN` |
| `neuralnet/nneval.cpp` | `enclosureInputs` (input versions 8 and 9), `board.pos.terminal`, `SymmetryHelpers::SymActionMap` | the game's input versions, `terminal`, `SymActionMap` |
| `dataio/trainingwrite.cpp` | The future occupancy (C0) and the income (C5-6): `enclosure::enclosed_area` at each turn end | `TARGETS`, and the member that computes a game's own targets |
| `dataio/trainingwrite.cpp` | The edges that get cut (C1: `Edge`, `igcd`, `contains`), the region occupancy 4 rounds later (C4) | The same |
| `dataio/trainingwrite.cpp` | The future nodes (C2-3) and the next mover's nodes (C7): `colors`, `Board::playerOfIndex(nextBoard.pos.player)` | `pieces`, `next_player` |
| `dataio/trainingwrite.cpp` | `enclosure::SIZE`, `N_POINTS`, `Board::indexOfPlayer` | `x_size`, `y_size`, `next_player` |
| `dataio/trainingwrite.cpp` | `NNPos::SCORE_SCALE` (score bins, the q-value score's storage), `SCORE_TARGET_CAP` | `SCORE_SCALE` |
| `dataio/trainingwrite.h` | The row format's comments (C5-7, C68, C69, C75) | `TARGETS` |
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
| `neuralnet/openclbackend.cpp`, `onnxbackend.cpp`, `cudaandrocmhelpers.*`, `modelversion.*`, `onnxmodelbuilder.h` | Comments: the policy channels, the input versions | Done for `cudaandrocmhelpers.h`. Not done for `openclbackend.cpp` and `onnxbackend.cpp`: they never took the fork's policy. They copy one policy channel per point, size their buffers for it, and apply the symmetry themselves. Neither builds on the training machine, so their change waits for a machine with their toolchains (abscls-bd7). The input versions: the inputs' step. The model's game: the trainer's step. |

### The trainer (`python/katago/train/`)

The trainer reads a game's facts from one module for that game. Each constant below goes into Enclosure's module.

| Constant | Where | Member it mirrors |
|---|---|---|
| `NUM_OFFSETS = 48`, `policy_len(pos_len)` | `model_pytorch.py` (the policy convolution and its reshape), `metrics_pytorch.py` (`policy_len`, the soft-policy mask), `enclosure_symmetry.py` | `POLICY_CHANNELS`, `NUM_ACTIONS` |
| `SCORE_BIN_WIDTH`, `SCORE_SCALE = 12.0` | `model_pytorch.py` (the score bins, the output multipliers), `metrics_pytorch.py` (the Huber deltas, the loss weights, the q-value score's storage `60 / SCORE_SCALE`) | `SCORE_SCALE` |
| `TIME_SCALE = 120 / 328` | `model_pytorch.py`, `metrics_pytorch.py` (the variance-time Huber delta) | `GAME_LENGTH / 328` |
| 215 spatial inputs, 5 or 7 global inputs | `modelconfigs.get_num_bin_input_features`, `get_num_global_input_features` | `1 + NUM_SPATIAL_INPUTS`, `NUM_GLOBAL_INPUTS` (+ 2 in V9) |
| `ENCLOSURE_HISTORY_PLANES_START = 211`, `ENCLOSURE_NUM_HISTORY_PLANES = 4` | `data_processing_pytorch.apply_history_dropout` | The member that names the history planes (the first plane and the count), the planes that `InputOptions::history = false` leaves out |
| `enclosure_symmetry` (`SIZE`, `NUM_OFFSETS`, `OFFSETS`, the edge planes' anchors, `NUM_SYMMETRIES = 8`) | `data_processing_pytorch.py` (the augmentation) | `SYMMETRIES`, `sym_position`, `SymActionMap`, in Python |
| `VALUE_TARGETS_NCHW_CHANNELS = 8`, the targets in C1, C4, C5-6, C7 and their flags C68, C69, C75 | `data_processing_pytorch.py`, `metrics_pytorch.py` | `TARGETS` |
| The storage × 120 of C0, C4 and C5-6 | `metrics_pytorch.py` | `TARGETS` (the slot's storage: KataGo's for C0 and C4) |
| The future-position head's 4 outputs: KataGo's 2, then the income logits | `model_pytorch.py` (`conv_futurepos`), `metrics_pytorch.py` | `TARGET_INCOME` |
| The score parity vector, kept at zero (Go's komi parity) | `model_pytorch.py` | None: Go's. A game whose last global input is komi would need KataGo's. |
| `load_model(..., pos_len=19)` | `load_model.py` | `MAX_BOARD_LEN` |
| `ENCLOSURE_ZERO_INPUT_CHANNELS` | `data_processing_pytorch.py` | None: an option for experiments (a name) |

The RGSC heads and their targets (`model_pytorch.py`, `metrics_pytorch.py`, `cpp/program/rgsc.h`) come from the
search, not from the game. They are not in the map.

Two files beside the trainer also hold Enclosure's facts, for the trainer's step:

- `python/export_model_pytorch.py`: the export of the policy convolution (`NUM_OFFSETS` channels for each output),
  and the V9 inputs (7 global inputs) in the model version.
- `python/shuffle.py`: the rows' C5-7 channels and V9's 7 global inputs (`pad_global_inputs_to_widest`).

### Enclosure's assumptions that stay in the front and the search

These do not name Enclosure, but they hold for Enclosure's rules only. Each waits for the step of a second game that
needs it:

- `search/timecontrols.cpp`: an edge counts half the time increment (`increment / 2`), since Enclosure's increment
  comes once a turn of two edges. A game whose turns have one move needs its own count of moves for a turn.
- `BoardHistory::isLegalTolerant`: a pass is allowed at any time, as Enclosure's timeout pass. A game whose rules have
  no pass needs the front to refuse it in GTP and the analysis engine.
- `Board::simpleRepetitionBoundGt` and `KoHashTable`: a position never recurs (Enclosure's edge count only rises), so
  the graph search merges transposed turns. A game whose positions can recur needs KataGo's repetition handling.
- `Board::DEFAULT_LEN`: the default board (`Board()`) is square, 19 or the largest board. A game whose standard board
  is not square (Connect-4's 7 × 6) needs a default size of its own.
- `command/enclosureprotocol.cpp`, `enclosureengine.cpp`: Enclosure's engine protocol (design 04), built into every
  `katago` binary. It reaches the rules through the members, and its own parts are Enclosure's, so a build of another
  game must leave it out.

### Names that stay

These name Enclosure but hold no game code. The step that removes the last game names decides which to rename:

- the query field `enclosurePosition` (an alias, for the move server), the commands `enclosure-engine`,
  `runenclosuretests`, `runenclosurerowtests` and `runenclosurenntests` (`main.cpp`, `main.h`,
  `command/runtests.cpp`), and `configs/*_enclosure.cfg`.
- the options of the fork's speed work: `ENCLOSURE_FUSED_CONV` (`cudaandrocmbackend.inc`), `ENCLOSURE_EIGEN_SIMD`,
  `ENCLOSURE_WINOGRAD_NV`, `ENCLOSURE_WINOGRAD_NT`, `ENCLOSURE_DIRECT_CONV`, `ENCLOSURE_EVAL_RAW`
  (`eigenbackend.cpp`), `ENCLOSURE_COOP_THREADS` (`nneval.*`), `ENCLOSURE_NO_POLICY_GATHER` (`nneval.cpp`).

### Changes of the fork that are already generic

These files mention Enclosure in comments only. Their code asks the position (the player to move, the end) through the
front, or scales a constant through a front name in the tables above: `command/dumponnx.cpp`, `evalsgf.cpp`,
`genbook.cpp`, `selfplay.cpp`, `writetrainingdata.cpp`, `core/os.h`, `core/threadaffinity.h`, `dataio/sgf.h`,
`neuralnet/cudaandrocmhelpers.*`, `program/play.h`, `rgsc.h`, `selfplaymanager.cpp`, `setup.cpp`,
`search/analysisdata.cpp`, `asyncbot.*`, `searchnnhelpers.cpp`, `searchupdatehelpers.cpp`.
