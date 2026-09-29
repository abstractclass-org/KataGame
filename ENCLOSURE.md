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
- **A move**: `NULL_LOC` (0), `PASS_LOC` (1), or 2 + our action index
  `(y * 19 + x) * 48 + offset`. Offsets follow the Enclosure engine's
  order: dx, then dy, each from -3 to 3, without (0, 0). So a move `Loc`
  runs up to 17,329.
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
