#ifndef GAME_GAMEINTERFACE_H_
#define GAME_GAMEINTERFACE_H_

/*
 * The game interface: what each game in KataGame supplies (GAMES.md).
 *
 * A game is a folder game/<name>/ with two structs in its namespace:
 * - GameRules, the rules part: the position, the actions, play, the end, the result, the hash and the notation. The
 *   Python binding builds against this part alone.
 * - GameEncoding, the network encoding part: the network's inputs, the policy layout, the symmetries, the training
 *   targets that the game writes, and the scales of the score and of a game's length for KataGo's constants.
 * The build chooses one game (game/chosenrules.h, game/chosengame.h). KataGo's Board and BoardHistory stay the front
 * that the search calls, and they reach the game through these members only. All members are static: the compiler
 * sees the same code as in a build for that one game.
 *
 * checkRules<R>() and checkEncoding<E>() are the static assertions. Each game's header calls the one for its part,
 * so a game that lacks a member fails to compile, and the message names the member. The rules part's check does not
 * need the encoding part.
 *
 * Conventions:
 * - Players are 0 (the first player, KataGo's Black) and 1 (the second, KataGo's White).
 * - An action is an int from 0 to NUM_ACTIONS - 1. Pass is the last, PASS_ACTION. Every game has it, and a game whose
 *   rules have no pass never lists it as legal. Actions are numbered on the largest board (MAX_BOARD_LEN), so an
 *   action keeps its number on every board size. An action is its policy index (point_action), so NUM_ACTIONS is
 *   MAX_BOARD_LEN * MAX_BOARD_LEN * POLICY_CHANNELS + 1. The network's board is the largest board.
 * - A point is (x, y), with 0 <= x < x_size and 0 <= y < y_size, and y = 0 at the bottom (row 1 in the notation).
 * - Symmetries are KataGo's 8 (bit 0x2 flips x, bit 0x1 flips y, bit 0x4 then transposes): symPoint below.
 * - No member throws: the WebAssembly build has no exceptions. A parser returns false or -1 and says why.
 *
 * The rules part (struct R):
 *   NAME                    const char*: the game's name, as the build option and the folder give it.
 *   Position                a copyable value: the whole state that the rules read.
 *   MAX_BOARD_LEN           int: the longest side of the game's boards. The front will take the build's largest board
 *                           (COMPILE_MAX_BOARD_LEN) from it.
 *   NUM_ACTIONS             int: the number of actions, pass included.
 *   PASS_ACTION             int: pass, NUM_ACTIONS - 1.
 *   new_game(xSize, ySize, position)
 *                           bool: sets the start position on a board of that size, or returns false when the game
 *                           is not played on it.
 *   x_size(p), y_size(p)    int: the board's size.
 *   next_player(p)          int: the player to move, from the position (a turn can have several moves).
 *   turn_moves_left(p)      int: the moves left in the mover's turn, this one included (1 in a game that alternates).
 *   moves_played(p)         int: the moves played since the start, a measure of the game's progress.
 *   terminal(p)             bool: whether the game is over.
 *   legal_actions(p, out)   void: the legal actions in ascending order (std::vector<int>), none when terminal.
 *   is_legal(p, action)     bool: whether one action is legal (false for a number out of range).
 *   play(p, action)         void: plays a legal action (legality is not checked).
 *   result(p)               float: at the end, the first player's return: 1 a win, 0 a draw, -1 a loss.
 *   score(p, player)        double: the player's points, in the game's units, final at the end. KataGo's margin is
 *                           score(p, 1) - score(p, 0) (White's minus Black's), plus the rules' komi.
 *   hash(p)                 Hash128: a hash of the whole position, the player to move included, so that equal hashes
 *                           mean equal positions. The front also mixes in KataGo's hash of the player to move.
 *   pieces(p, owner)        void: owner[y * x_size + x] is the player whose piece is on the point, or -1 (int8_t).
 *                           KataGo's colors, ownership display and future-position targets read it.
 *   area(p, owner)          void: owner[y * x_size + x] is the player who owns the point now, or -1 (int8_t): KataGo's
 *                           area, which gives the final ownership at the end of a game.
 *   action_text(action)     std::string: the action in the game's notation (docs/design/04-notation-and-protocol.md).
 *   parse_action(p, text, why)
 *                           int: the legal action of p that text names, or -1 with the reason in *why (if not
 *                           null).
 *   position_text(p)        std::string: the one-line position.
 *   parse_position(text, p, why)
 *                           bool: sets p from a one-line position, or returns false with the reason in why.
 *
 * Optional members of the rules part. A game may leave each out: the front calls the helper below, which calls the
 * game's member when the game has it and gives the default when it does not.
 *   protocol_text(action), parse_protocol_text(text, action)
 *                           std::string, bool: an action other than pass as the engine protocol (GTP, the analysis
 *                           engine) writes and reads it. The parser reads text as it comes (trimmed, in any case),
 *                           does not check legality, and returns false for text that names no action. The front reads
 *                           and writes pass ("pass"). A game gives both or neither. Default: action_text, and the
 *                           action whose action_text is text, ignoring case (protocolText, parseProtocolText).
 *   NUM_LOCAL_KEYS, LOCAL_WINDOW, local_hash(p, cx, cy, xRadius, yRadius, flipPlayers, keys)
 *                           int, int, Hash128: the position near the point (cx, cy), in the window of the points (x, y)
 *                           with |x - cx| <= xRadius and |y - cy| <= yRadius: the local pattern hash of the search
 *                           (the subtree value bias, the pattern bonus). keys holds the NUM_LOCAL_KEYS random keys
 *                           that the front draws for the game. flipPlayers swaps the two players. LOCAL_WINDOW is the
 *                           side of the subtree value bias's window (odd), large enough for a move's reach. A game
 *                           gives all three or none. Default: each point of the window hashed with its piece, as
 *                           KataGo hashes stones, in KataGo's window of 5 (localHash, localWindow).
 *   QUERY_POSITION_FIELD, parse_query_position(value, p, why)
 *                           const char*, bool: a position field of the game's own in an analysis query, beside the
 *                           generic field "position" (the one-line position): its key, and the reader of its value, a
 *                           JSON value (a template on the JSON type, so that the rules part does not need the JSON
 *                           library). The reader may throw the JSON library's exceptions, which the analysis engine
 *                           catches (the WebAssembly build has no analysis engine). A game gives both or neither.
 *                           Default: none.
 *   STANDARD_X_SIZE, STANDARD_Y_SIZE
 *                           int, int: the game's standard board, on which new_game succeeds: the front's default board
 *                           (Board(), GTP's board when its config names none) and the Python binding's Position(). A
 *                           game gives both or neither. Default: the largest board, MAX_BOARD_LEN x MAX_BOARD_LEN
 *                           (standardXSize, standardYSize).
 *   TURN_MOVES              int: the moves of a full turn, after which the clock's increment comes: the time controls
 *                           give each move its share of the increment. Default: 1, a game that alternates (turnMoves).
 *   replay_tolerates(p, action)
 *                           bool: whether a replayed game (GTP, the analysis engine, SGF files, start positions) also
 *                           plays this action of p where it is not legal: what a site plays beside the legal actions,
 *                           such as a pass on a clock's timeout. play must accept it. Any int (false out of range).
 *                           The search still plays the legal actions only. Default: none (replayTolerates).
 *   continues_move(p)       bool: whether the next action continues the move of the action before it, by the same
 *                           player: Blueprints' drop, the second action of an edit, after its pick. Self-play's
 *                           cheapSearchPerMove reads it. Default: false, each action is a move (continuesMove).
 *   SETTINGS, set_setting(name, value, why)
 *                           a list of const char* (std::array), bool: the game's own settings, which every command
 *                           reads from its config before its first game (Setup::initializeSession): each one's key in
 *                           the config, and the setter, which reads a value's text and returns false with the reason
 *                           for a name or a value that the game does not have. A setting holds for the whole process,
 *                           and the rules read it, so a game replays only under the settings that played it. A config
 *                           without a setting's key keeps the game's default. Blueprints' viewOrientation. A game gives
 *                           both or neither. Default: none (settingNames, setSetting).
 *   draw_start(p, random)   void: the game's own draws in the start position p of a new game, which self-play's games
 *                           from the standard start make (GameInitializer). random() gives a uniform uint64_t from
 *                           self-play's random numbers: the game calls it only for the draws that it makes, so a game
 *                           that draws nothing leaves self-play's other draws as they are. The position then holds the
 *                           draws, its one-line position too: a replay or an analysis from that text gives them back.
 *                           Self-play's SGF records do not hold them. Blueprints' view orientation under
 *                           viewOrientation = randomGame. Default: none (drawStart).
 *   view_text(p), set_view(p, text, why), view_cover(p, views)
 *                           std::string, bool, void: the views of a game whose network sees a part of a larger board,
 *                           each a text of the game's own ("" for the view that the rules give the position).
 *                           view_text gives the view that a caller set on p, or "". set_view sets p's view until p's
 *                           next action (play ends it, and every later position has the rules' view): the legal
 *                           actions, what each action means, the inputs, the symmetries' images and the hash are then
 *                           that view's. "" gives the rules' view back. It returns false with the reason for a text
 *                           that names no view. view_cover gives views (std::vector<std::string>) that hold every
 *                           legal move of p between them, each move as an action of p with one of the views set: {""}
 *                           when the rules' view holds them all. The analysis engine's rootViews searches each view
 *                           of the cover. A game gives all three or none. Blueprints' views. Default: one view, ""
 *                           (viewText, setView, viewCover).
 *
 * The network encoding part (struct E):
 *   Rules                   the game's rules part (its GameRules).
 *   NUM_SPATIAL_INPUTS      int: the input planes that the game writes. The front puts KataGo's on-board plane
 *                           first, so the network sees 1 + NUM_SPATIAL_INPUTS planes.
 *   NUM_GLOBAL_INPUTS       int: the global inputs that the game writes.
 *   spatial_inputs(p, options, set)
 *                           void: calls set(x, y, plane) once for each input that is 1 (every other input is 0),
 *                           with 0 <= plane < NUM_SPATIAL_INPUTS, at the point of the image under options.symmetry.
 *                           A template on set, so that each caller's writer is inlined.
 *   global_inputs(p, options, out)
 *                           void: writes NUM_GLOBAL_INPUTS floats.
 *   HISTORY_PLANE, NUM_HISTORY_PLANES
 *                           int, int: the history planes, the game's planes HISTORY_PLANE to HISTORY_PLANE +
 *                           NUM_HISTORY_PLANES - 1. They show what the position keeps of earlier turns, and
 *                           spatial_inputs sets none of them without options.history. The trainer's history dropout
 *                           clears them. A game without history has 0 planes.
 *   input_hash(p)           Hash128: what the inputs read beyond hash(p), such as earlier turns. The front mixes it
 *                           into the key of the network's cache.
 *   POLICY_CHANNELS         int: policy channels per point.
 *   action_point(action, x, y, channel)
 *                           void: the point and the channel of an action other than pass.
 *   point_action(x, y, channel)
 *                           int: the action at that point and channel, (y * MAX_BOARD_LEN + x) * POLICY_CHANNELS +
 *                           channel, which is also its policy index. Pass is last.
 *   SYMMETRIES              unsigned: the allowed subset of KataGo's 8 symmetries, bit s for symmetry s. Bit 0 (the
 *                           identity) is always set. A game that allows a transpose (bit 4 to 7) plays on square
 *                           boards only.
 *   sym_position(p, symmetry)
 *                           Position: the image of the position.
 *   SymActionMap            a type: SymActionMap(p, symmetry) maps each legal action of p and pass (operator(),
 *                           int to int) to its image, an action of sym_position(p, symmetry).
 *   TARGETS                 unsigned: the optional training targets that the game writes (GameTarget bits).
 *   training_targets(positions, numPositions, row, mover, values)
 *                           unsigned: the game's own training targets of a training row, the slots of its own targets
 *                           (ownTargets: all of TARGETS but the FRONT_TARGETS, by default). positions(j) is the game's
 *                           position after j moves (const Position&), for 0 <= j < numPositions, and the row is at
 *                           positions(row). mover is the row's player to move. The game writes the values before the
 *                           row's storage (TargetValues), in values[channel * area + y * x_size + x], area = x_size *
 *                           y_size, and its global targets after them, where the front has written 0. It returns the
 *                           slots (GameTarget bits) that hold a target in this row, which the front's weights and
 *                           flags of the slots read (TrainingWriteBuffers). A template on positions. A game without
 *                           own targets can leave it out.
 *   SCORE_SCALE             double: the game's score spread against Go's. KataGo's constants in points scale by it
 *                           (the score distribution's bin width, the output multipliers, the Huber deltas).
 *   SCORE_UNIT              double: the search's score utility unit, in points, where KataGo uses the square root
 *                           of the board's area.
 *   GAME_LENGTH             double: a game's mean length in moves, for KataGo's game-phase constants (the earliest
 *                           resignation, early forks, the opening moves, the time controls).
 *
 * Optional members of the encoding part, as those of the rules part (the helper below gives the default):
 *   LOOKAHEAD_AREA          double: the area that KataGo's look-ahead horizons read where KataGo reads the board's area
 *                           (the short-term value and score targets, value-surprise weighting). The horizons count
 *                           decisions, so a game whose board is larger than the reach of its decisions gives Go's 361,
 *                           and the horizons count its moves as KataGo's count Go's. Default: the board's area,
 *                           x_size * y_size (lookaheadArea).
 *   KOMI_AREA               double: the area that KataGo's komi sampler reads where KataGo reads the board's area
 *                           (chooseExtraBlackAndKomi: the configured standard deviations times sqrt(area) / 19, and
 *                           the handicap's stones). With 361, the sampler draws a komi in the points that the config
 *                           gives. The spreads of fancyKomiVarying still read the board's area. Default: the board's
 *                           area, x_size * y_size (komiArea).
 *   KOMI_BRACKET_STEP, KOMI_BRACKET_LIMIT
 *                           double, double: the bracket of KataGo's even-komi search (PlayUtils::evenKomiOffset), in
 *                           points. Around its guess, the search tries the offsets KOMI_BRACKET_STEP, twice that, and
 *                           so on, up to KOMI_BRACKET_LIMIT, until the win-loss value changes sign. Then it bisects on
 *                           the half-point grid. When the value keeps its sign up to the limit, a game that gives its
 *                           bracket has no even komi: the lead's search gives the position no lead target, and the
 *                           compensation of an uneven game keeps the game's komi. Default: KataGo's 1 and 32, the
 *                           offsets 1, 2, 4, 8, 16 and 32, and KataGo's rule, which takes the bracket's end as the
 *                           even komi (komiBracketStep, komiBracketLimit, komiBracketEndIsEven).
 *   KOMI_LEAD_STEP          double: the step of the lead's smoothing over a coarse score (PlayUtils::computeLead), in
 *                           points. When the even komi is not a multiple of 2 * KOMI_LEAD_STEP, computeLead reads the
 *                           win-loss value at the komis KOMI_LEAD_STEP apart around it, averages each pair 2 *
 *                           KOMI_LEAD_STEP apart, and interpolates. KataGo's is half a point, for Go's area scoring:
 *                           there the margins of a full board have one parity, so the result changes every 2 points
 *                           of komi. 0 for a game whose margins have no parity: the lead is the search's even komi, as
 *                           in KataGo for territory scoring. Default: KataGo's half point times SCORE_SCALE
 *                           (komiLeadStep).
 *   OWN_TARGETS             unsigned: the slots of TARGETS that the game computes itself (training_targets). It holds
 *                           every slot of TARGETS that is not one of the FRONT_TARGETS, and can hold those too: a game
 *                           whose pieces() do not give such a target computes it, and the front then leaves that slot
 *                           to the game. Default: TARGETS without the FRONT_TARGETS (ownTargets).
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/hash.h"

namespace GameInterface {

//What the front gives spatial_inputs and global_inputs.
struct InputOptions {
  int symmetry = 0;     //An allowed symmetry: the inputs are those of the position's image.
  bool history = true;  //False: leave out what the position keeps of earlier turns (KataGo's maxHistory <= 0).
  double komi = 0.0;    //The rules' komi, for a game whose inputs encode it.
};

//The optional training targets: the slots of KataGo's spatial targets (its training rows' valueTargetsNCHW) and the
//two that the fork added. A game gives each slot that it writes its own meaning (its GameEncoding says which), and the
//front writes 0 in each slot that the game does not write. The slots with a weight or a flag of their own (C1, C2-3,
//C4, C5-6, C7) then have weight 0. C0 has none: KataGo's C27 weighs it with C1 and the score's targets, so the
//trainer reads TARGETS to leave out C0 for a game without TARGET_OWNERSHIP.
enum GameTarget : unsigned {
  TARGET_OWNERSHIP = 1u << 0,      //C0: KataGo's ownership.
  TARGET_SEKI = 1u << 1,           //C1: KataGo's seki areas.
  TARGET_FUTURE_PIECES = 1u << 2,  //C2-3: the front's, the pieces 8 and 32 moves later (pieces()), KataGo's future
                                   //positions: -1, 0 or 1 a point.
  TARGET_SCORING = 1u << 3,        //C4: KataGo's scoring (final area).
  TARGET_INCOME = 1u << 4,         //C5-6: a value per point for each player.
  TARGET_NEXT_PIECES = 1u << 5,    //C7: the front's, the next mover's pieces after the row's move.
};
//The slots that the front can compute from pieces(): it computes each of them that the game's TARGETS hold and its own
//targets do not (OWN_TARGETS). The game's training_targets gives every other slot of its TARGETS.
constexpr unsigned FRONT_TARGETS = TARGET_FUTURE_PIECES | TARGET_NEXT_PIECES;

//The values of a row's targets before the front stores them (TrainingWriteBuffers::addRow): channel c of
//valueTargetsNCHW is values[c * area + y * x_size + x], then the global targets that a game can write. The front
//stores the slots that the game writes as follows, channel by channel (C0 to C7) and point by point:
//- C0 (ownership), C4 (scoring), C5-6 (income): a value in [-1, 1], times 120, rounded at random to an integer so
//  that the expected value is exact (KataGo's storage of its scoring target).
//- C1 (seki), C2-3 (future pieces), C7 (next pieces): an integer, as it is (KataGo's seki and future positions: -1, 0
//  or 1).
//- The global targets G76 to G79 of the row (globalTargetsNC), which KataGo leaves unused: values[NUM_CHANNELS * area +
//  i] is G(FIRST_GLOBAL + i), as it is.
struct TargetValues {
  static constexpr int NUM_CHANNELS = 8;  //C0 to C7 (the rows' valueTargetsNCHW)
  static constexpr float STORAGE_SCALE = 120.0f;
  static constexpr bool storedAsIs(int channel) { return channel == 1 || channel == 2 || channel == 3 || channel == 7; }
  static constexpr int FIRST_GLOBAL = 76, NUM_GLOBALS = 4;
  //The floats of a row's values on a board of that area.
  static constexpr int size(int area) { return NUM_CHANNELS * area + NUM_GLOBALS; }
};

//KataGo's symmetry of a point on a board of xSize x ySize (SymmetryHelpers::getSymLoc): flip x, flip y, then
//transpose.
inline void symPoint(int& x, int& y, int symmetry, int xSize, int ySize) {
  if(symmetry & 0x2) x = xSize - x - 1;
  if(symmetry & 0x1) y = ySize - y - 1;
  if(symmetry & 0x4) std::swap(x, y);
}

//A set of KataGo's symmetries (bit s for symmetry s, as SYMMETRIES): whether it holds a symmetry, how many it holds,
//and the list of them in ascending order.
constexpr bool hasSymmetry(unsigned set, int symmetry) {
  return symmetry >= 0 && symmetry < 8 && ((set >> symmetry) & 1u) != 0;
}
constexpr int numSymmetries(unsigned set) {
  int n = 0;
  for(int s = 0; s < 8; s++)
    n += hasSymmetry(set, s);
  return n;
}
template<unsigned SET> constexpr std::array<int, numSymmetries(SET)> symmetryList() {
  std::array<int, numSymmetries(SET)> list{};
  int n = 0;
  for(int s = 0; s < 8; s++)
    if(hasSymmetry(SET, s))
      list[n++] = s;
  return list;
}

//The game of a network whose model file holds 0 in option H (neuralnet/desc.cpp): Enclosure's files hold 0 there, as
//every file from before the game interface, so that every binary loads them.
constexpr const char* UNNAMED_MODEL_GAME = "enclosure";

//A writer for spatial_inputs, for the detection below.
struct SetEntry {
  void operator()(int x, int y, int plane) const;
};
//A JSON value for the detection of parse_query_position, a template on the JSON type.
struct QueryValue {};
//A game's positions for the detection of training_targets: positions(j).
template<class P> struct PositionAt {
  const P& operator()(int j) const;
};

//Each detector is true when the expression (the rest of the arguments) is valid and converts to the type.
#define GAMEINTERFACE_DETECT(NAME, TYPE, ...) \
  template<class G, class = void> struct NAME : std::false_type {}; \
  template<class G> struct NAME<G, std::enable_if_t<std::is_convertible<decltype(__VA_ARGS__), TYPE>::value>> \
    : std::true_type {};

template<class G, class = void> struct HasPosition : std::false_type {};
template<class G> struct HasPosition<G, std::void_t<typename G::Position>> : std::true_type {};

namespace rules {
  template<class G> using P = typename G::Position;
  GAMEINTERFACE_DETECT(HasName, const char*, G::NAME)
  GAMEINTERFACE_DETECT(HasMaxBoardLen, int, std::integral_constant<int, G::MAX_BOARD_LEN>::value)
  GAMEINTERFACE_DETECT(HasNumActions, int, std::integral_constant<int, G::NUM_ACTIONS>::value)
  GAMEINTERFACE_DETECT(HasPassAction, int, std::integral_constant<int, G::PASS_ACTION>::value)
  GAMEINTERFACE_DETECT(HasNewGame, bool, G::new_game(0, 0, std::declval<P<G>&>()))
  GAMEINTERFACE_DETECT(HasXSize, int, G::x_size(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasYSize, int, G::y_size(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasNextPlayer, int, G::next_player(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasTurnMovesLeft, int, G::turn_moves_left(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasMovesPlayed, int, G::moves_played(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasTerminal, bool, G::terminal(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasLegalActions, void,
                       G::legal_actions(std::declval<const P<G>&>(), std::declval<std::vector<int>&>()))
  GAMEINTERFACE_DETECT(HasIsLegal, bool, G::is_legal(std::declval<const P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasPlay, void, G::play(std::declval<P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasResult, float, G::result(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasScore, double, G::score(std::declval<const P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasHash, Hash128, G::hash(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasPieces, void, G::pieces(std::declval<const P<G>&>(), (int8_t*)nullptr))
  GAMEINTERFACE_DETECT(HasArea, void, G::area(std::declval<const P<G>&>(), (int8_t*)nullptr))
  GAMEINTERFACE_DETECT(HasActionText, std::string, G::action_text(0))
  GAMEINTERFACE_DETECT(HasParseAction, int,
                       G::parse_action(std::declval<const P<G>&>(), std::declval<const std::string&>(),
                                       (std::string*)nullptr))
  GAMEINTERFACE_DETECT(HasPositionText, std::string, G::position_text(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasParsePosition, bool,
                       G::parse_position(std::declval<const std::string&>(), std::declval<P<G>&>(),
                                         std::declval<std::string&>()))
  //The optional members.
  GAMEINTERFACE_DETECT(HasProtocolText, std::string, G::protocol_text(0))
  GAMEINTERFACE_DETECT(HasParseProtocolText, bool,
                       G::parse_protocol_text(std::declval<const std::string&>(), std::declval<int&>()))
  GAMEINTERFACE_DETECT(HasNumLocalKeys, int, std::integral_constant<int, G::NUM_LOCAL_KEYS>::value)
  GAMEINTERFACE_DETECT(HasLocalWindow, int, std::integral_constant<int, G::LOCAL_WINDOW>::value)
  GAMEINTERFACE_DETECT(HasLocalHash, Hash128,
                       G::local_hash(std::declval<const P<G>&>(), 0, 0, 0, 0, false, (const Hash128*)nullptr))
  GAMEINTERFACE_DETECT(HasQueryPositionField, const char*, G::QUERY_POSITION_FIELD)
  GAMEINTERFACE_DETECT(HasParseQueryPosition, bool,
                       G::template parse_query_position<QueryValue>(std::declval<const QueryValue&>(),
                                                                    std::declval<P<G>&>(), std::declval<std::string&>()))
  GAMEINTERFACE_DETECT(HasStandardXSize, int, std::integral_constant<int, G::STANDARD_X_SIZE>::value)
  GAMEINTERFACE_DETECT(HasStandardYSize, int, std::integral_constant<int, G::STANDARD_Y_SIZE>::value)
  GAMEINTERFACE_DETECT(HasTurnMoves, int, std::integral_constant<int, G::TURN_MOVES>::value)
  GAMEINTERFACE_DETECT(HasReplayTolerates, bool, G::replay_tolerates(std::declval<const P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasContinuesMove, bool, G::continues_move(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasSettings, const char*, *std::begin(G::SETTINGS))
  GAMEINTERFACE_DETECT(HasSetSetting, bool,
                       G::set_setting(std::declval<const std::string&>(), std::declval<const std::string&>(),
                                      std::declval<std::string&>()))
  GAMEINTERFACE_DETECT(HasDrawStart, void, G::draw_start(std::declval<P<G>&>(), std::declval<uint64_t (&)()>()))
  GAMEINTERFACE_DETECT(HasViewText, std::string, G::view_text(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasSetView, bool,
                       G::set_view(std::declval<P<G>&>(), std::declval<const std::string&>(),
                                   std::declval<std::string&>()))
  GAMEINTERFACE_DETECT(HasViewCover, void,
                       G::view_cover(std::declval<const P<G>&>(), std::declval<std::vector<std::string>&>()))
}

template<class R> constexpr bool checkRules() {
  using namespace rules;
  static_assert(HasName<R>::value, "game interface, rules part: the game lacks NAME (const char*, the game's name)");
  static_assert(HasPosition<R>::value, "game interface, rules part: the game lacks Position (the type of a position)");
  static_assert(HasMaxBoardLen<R>::value, "game interface, rules part: the game lacks MAX_BOARD_LEN (constexpr int)");
  static_assert(HasNumActions<R>::value, "game interface, rules part: the game lacks NUM_ACTIONS (constexpr int)");
  static_assert(HasPassAction<R>::value, "game interface, rules part: the game lacks PASS_ACTION (constexpr int)");
  static_assert(HasNewGame<R>::value,
                "game interface, rules part: the game lacks new_game(int xSize, int ySize, Position&) -> bool");
  static_assert(HasXSize<R>::value, "game interface, rules part: the game lacks x_size(const Position&) -> int");
  static_assert(HasYSize<R>::value, "game interface, rules part: the game lacks y_size(const Position&) -> int");
  static_assert(HasNextPlayer<R>::value,
                "game interface, rules part: the game lacks next_player(const Position&) -> int");
  static_assert(HasTurnMovesLeft<R>::value,
                "game interface, rules part: the game lacks turn_moves_left(const Position&) -> int");
  static_assert(HasMovesPlayed<R>::value,
                "game interface, rules part: the game lacks moves_played(const Position&) -> int");
  static_assert(HasTerminal<R>::value, "game interface, rules part: the game lacks terminal(const Position&) -> bool");
  static_assert(HasLegalActions<R>::value,
                "game interface, rules part: the game lacks legal_actions(const Position&, std::vector<int>&)");
  static_assert(HasIsLegal<R>::value,
                "game interface, rules part: the game lacks is_legal(const Position&, int action) -> bool");
  static_assert(HasPlay<R>::value, "game interface, rules part: the game lacks play(Position&, int action)");
  static_assert(HasResult<R>::value, "game interface, rules part: the game lacks result(const Position&) -> float");
  static_assert(HasScore<R>::value,
                "game interface, rules part: the game lacks score(const Position&, int player) -> double");
  static_assert(HasHash<R>::value, "game interface, rules part: the game lacks hash(const Position&) -> Hash128");
  static_assert(HasPieces<R>::value,
                "game interface, rules part: the game lacks pieces(const Position&, int8_t* owner)");
  static_assert(HasArea<R>::value, "game interface, rules part: the game lacks area(const Position&, int8_t* owner)");
  static_assert(HasActionText<R>::value,
                "game interface, rules part: the game lacks action_text(int action) -> std::string");
  static_assert(HasParseAction<R>::value,
                "game interface, rules part: the game lacks parse_action(const Position&, const std::string&, "
                "std::string* why) -> int");
  static_assert(HasPositionText<R>::value,
                "game interface, rules part: the game lacks position_text(const Position&) -> std::string");
  static_assert(HasParsePosition<R>::value,
                "game interface, rules part: the game lacks parse_position(const std::string&, Position&, "
                "std::string& why) -> bool");
  if constexpr(HasNumActions<R>::value && HasPassAction<R>::value)
    static_assert(R::PASS_ACTION == R::NUM_ACTIONS - 1, "game interface, rules part: PASS_ACTION is the last action");
  static_assert(HasProtocolText<R>::value == HasParseProtocolText<R>::value,
                "game interface, rules part: the game gives protocol_text(int action) -> std::string and "
                "parse_protocol_text(const std::string&, int& action) -> bool together, or neither");
  static_assert(HasNumLocalKeys<R>::value == HasLocalHash<R>::value && HasLocalWindow<R>::value == HasLocalHash<R>::value,
                "game interface, rules part: the game gives NUM_LOCAL_KEYS (constexpr int), LOCAL_WINDOW (constexpr "
                "int) and local_hash(const Position&, int cx, int cy, int xRadius, int yRadius, bool flipPlayers, "
                "const Hash128* keys) -> Hash128 together, or none");
  if constexpr(HasLocalWindow<R>::value)
    static_assert(R::LOCAL_WINDOW > 0 && R::LOCAL_WINDOW % 2 == 1, "game interface, rules part: LOCAL_WINDOW is odd");
  static_assert(HasQueryPositionField<R>::value == HasParseQueryPosition<R>::value,
                "game interface, rules part: the game gives QUERY_POSITION_FIELD (const char*) and the template "
                "parse_query_position(const Json&, Position&, std::string& why) -> bool together, or neither");
  static_assert(HasStandardXSize<R>::value == HasStandardYSize<R>::value,
                "game interface, rules part: the game gives STANDARD_X_SIZE and STANDARD_Y_SIZE (constexpr int) "
                "together, or neither");
  if constexpr(HasStandardXSize<R>::value && HasStandardYSize<R>::value && HasMaxBoardLen<R>::value)
    static_assert(R::STANDARD_X_SIZE >= 1 && R::STANDARD_X_SIZE <= R::MAX_BOARD_LEN && R::STANDARD_Y_SIZE >= 1 &&
                    R::STANDARD_Y_SIZE <= R::MAX_BOARD_LEN,
                  "game interface, rules part: the standard board fits the largest board");
  if constexpr(HasTurnMoves<R>::value)
    static_assert(R::TURN_MOVES >= 1, "game interface, rules part: a turn has at least one move (TURN_MOVES)");
  static_assert(HasSettings<R>::value == HasSetSetting<R>::value,
                "game interface, rules part: the game gives SETTINGS (the settings' names, const char*) and "
                "set_setting(const std::string& name, const std::string& value, std::string& why) -> bool together, "
                "or neither");
  static_assert(HasViewText<R>::value == HasSetView<R>::value && HasViewCover<R>::value == HasSetView<R>::value,
                "game interface, rules part: the game gives view_text(const Position&) -> std::string, "
                "set_view(Position&, const std::string&, std::string& why) -> bool and view_cover(const Position&, "
                "std::vector<std::string>&) together, or none");
  return true;
}

//The helpers of the optional members: each calls the game's member when the game has it, else gives the default.

//An action other than pass as the engine protocol writes it.
template<class R> std::string protocolText(int action) {
  if constexpr(rules::HasProtocolText<R>::value)
    return R::protocol_text(action);
  else
    return R::action_text(action);
}

//The action other than pass that text names in the engine protocol, legal or not. False when it names none. The
//default compares text with action_text of each action, so it suits games with few actions.
template<class R> bool parseProtocolText(const std::string& text, int& action) {
  if constexpr(rules::HasParseProtocolText<R>::value)
    return R::parse_protocol_text(text, action);
  else {
    auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; };
    auto same = [&](char a, char b) { return lower(a) == lower(b); };
    for(int a = 0; a < R::PASS_ACTION; a++) {
      const std::string name = R::action_text(a);
      if(name.size() == text.size() && std::equal(name.begin(), name.end(), text.begin(), same)) {
        action = a;
        return true;
      }
    }
    return false;
  }
}

//The random keys that the front draws for the game's local_hash: 0 without it.
template<class R> constexpr int numLocalKeys() {
  if constexpr(rules::HasLocalHash<R>::value)
    return R::NUM_LOCAL_KEYS;
  else
    return 0;
}

//The side of the subtree value bias's window: the game's, else KataGo's 5.
template<class R> constexpr int localWindow() {
  if constexpr(rules::HasLocalHash<R>::value)
    return R::LOCAL_WINDOW;
  else
    return 5;
}

//The position near (cx, cy) in the window of xRadius and yRadius. gameKeys: the numLocalKeys<R>() keys of the game's
//local_hash. pointKeys: the default's keys, KataGo's zobristLocalPattern. The default is KataGo's hash of the stones
//without its atari flag: each point of the window on the board, with the key
//pointKeys[state * windowArea + (dy + yRadius) * windowWidth + (dx + xRadius)], where the state is KataGo's color: 0
//empty, 1 the first player's piece, 2 the second player's (swapped when flipPlayers). Points off the board are left
//out.
template<class R>
Hash128 localHash(const typename R::Position& p, int cx, int cy, int xRadius, int yRadius, bool flipPlayers,
                  const Hash128* gameKeys, const Hash128* pointKeys) {
  if constexpr(rules::HasLocalHash<R>::value) {
    (void)pointKeys;
    return R::local_hash(p, cx, cy, xRadius, yRadius, flipPlayers, gameKeys);
  }
  else {
    (void)gameKeys;
    const int xSize = R::x_size(p), ySize = R::y_size(p);
    int8_t owner[R::MAX_BOARD_LEN * R::MAX_BOARD_LEN];
    R::pieces(p, owner);
    const int width = 2 * xRadius + 1, windowArea = width * (2 * yRadius + 1);
    Hash128 h;
    for(int y = std::max(0, cy - yRadius); y <= std::min(ySize - 1, cy + yRadius); y++)
      for(int x = std::max(0, cx - xRadius); x <= std::min(xSize - 1, cx + xRadius); x++) {
        const int o = owner[y * xSize + x];
        const int state = o < 0 ? 0 : 1 + (flipPlayers ? 1 - o : o);
        h ^= pointKeys[state * windowArea + (y - cy + yRadius) * width + (x - cx + xRadius)];
      }
    return h;
  }
}

//The key of the game's own position field in an analysis query, or null without it.
template<class R> constexpr const char* queryPositionField() {
  if constexpr(rules::HasQueryPositionField<R>::value)
    return R::QUERY_POSITION_FIELD;
  else
    return nullptr;
}

//Reads the game's own position field (queryPositionField). A game without it reads none.
template<class R, class Json> bool parseQueryPosition(const Json& value, typename R::Position& p, std::string& why) {
  if constexpr(rules::HasQueryPositionField<R>::value)
    return R::parse_query_position(value, p, why);
  else {
    (void)value;
    (void)p;
    why = "the game has no position field of its own";
    return false;
  }
}

//The game's standard board: the game's, else its largest board.
template<class R> constexpr int standardXSize() {
  if constexpr(rules::HasStandardXSize<R>::value)
    return R::STANDARD_X_SIZE;
  else
    return R::MAX_BOARD_LEN;
}
template<class R> constexpr int standardYSize() {
  if constexpr(rules::HasStandardYSize<R>::value)
    return R::STANDARD_Y_SIZE;
  else
    return R::MAX_BOARD_LEN;
}

//The moves of a full turn, which share the clock's increment: the game's, else 1.
template<class R> constexpr int turnMoves() {
  if constexpr(rules::HasTurnMoves<R>::value)
    return R::TURN_MOVES;
  else
    return 1;
}

//Whether a replayed game also plays this action where it is not legal: the game's, else none.
template<class R> bool replayTolerates(const typename R::Position& p, int action) {
  if constexpr(rules::HasReplayTolerates<R>::value)
    return R::replay_tolerates(p, action);
  else {
    (void)p;
    (void)action;
    return false;
  }
}

//Whether the next action continues the move of the action before it: the game's, else never.
template<class R> bool continuesMove(const typename R::Position& p) {
  if constexpr(rules::HasContinuesMove<R>::value)
    return R::continues_move(p);
  else {
    (void)p;
    return false;
  }
}

//The game's own settings, by their keys in a config: the game's, else none.
template<class R> std::vector<std::string> settingNames() {
  if constexpr(rules::HasSettings<R>::value)
    return std::vector<std::string>(std::begin(R::SETTINGS), std::end(R::SETTINGS));
  else
    return {};
}

//Sets one of the game's own settings from a value's text, or returns false with the reason. A game without settings
//has none to set.
template<class R> bool setSetting(const std::string& name, const std::string& value, std::string& why) {
  if constexpr(rules::HasSetSetting<R>::value)
    return R::set_setting(name, value, why);
  else {
    (void)value;
    why = std::string("the game ") + R::NAME + " has no setting " + name;
    return false;
  }
}

//The game's own draws in a new game's start position p, from random(), a uniform uint64_t a call: the game's, else
//none (random() is not called).
template<class R, class Random> void drawStart(typename R::Position& p, Random&& random) {
  if constexpr(rules::HasDrawStart<R>::value)
    R::draw_start(p, random);
  else {
    (void)p;
    (void)random;
  }
}

//Whether the game has views of its own (view_text, set_view, view_cover).
template<class R> constexpr bool hasViews() { return rules::HasSetView<R>::value; }

//The view that a caller set on p: the game's, else "" (the rules' view, the only one).
template<class R> std::string viewText(const typename R::Position& p) {
  if constexpr(rules::HasViewText<R>::value)
    return R::view_text(p);
  else {
    (void)p;
    return "";
  }
}

//Sets p's view until its next action, or returns false with the reason: the game's, else "" alone (no change).
template<class R> bool setView(typename R::Position& p, const std::string& text, std::string& why) {
  if constexpr(rules::HasSetView<R>::value)
    return R::set_view(p, text, why);
  else {
    (void)p;
    if(text.empty())
      return true;
    why = std::string("the game ") + R::NAME + " has one view, \"\", not \"" + text + "\"";
    return false;
  }
}

//Views that hold every legal move of p between them: the game's, else the rules' view, {""}.
template<class R> void viewCover(const typename R::Position& p, std::vector<std::string>& views) {
  if constexpr(rules::HasViewCover<R>::value)
    R::view_cover(p, views);
  else {
    (void)p;
    views.assign(1, "");
  }
}

namespace encoding {
  template<class G> using P = typename G::Rules::Position;
  template<class G, class = void> struct HasRules : std::false_type {};
  template<class G> struct HasRules<G, std::void_t<P<G>>> : std::true_type {};
  template<class G, class = void> struct HasSymActionMap : std::false_type {};
  template<class G> struct HasSymActionMap<G, std::enable_if_t<std::is_convertible<
    decltype(std::declval<const typename G::SymActionMap&>()(0)), int>::value &&
    std::is_constructible<typename G::SymActionMap, const P<G>&, int>::value>> : std::true_type {};
  GAMEINTERFACE_DETECT(HasNumSpatialInputs, int, std::integral_constant<int, G::NUM_SPATIAL_INPUTS>::value)
  GAMEINTERFACE_DETECT(HasNumGlobalInputs, int, std::integral_constant<int, G::NUM_GLOBAL_INPUTS>::value)
  GAMEINTERFACE_DETECT(HasSpatialInputs, void,
                       G::spatial_inputs(std::declval<const P<G>&>(), std::declval<const InputOptions&>(),
                                         std::declval<SetEntry&>()))
  GAMEINTERFACE_DETECT(HasGlobalInputs, void,
                       G::global_inputs(std::declval<const P<G>&>(), std::declval<const InputOptions&>(),
                                        (float*)nullptr))
  GAMEINTERFACE_DETECT(HasHistoryPlane, int, std::integral_constant<int, G::HISTORY_PLANE>::value)
  GAMEINTERFACE_DETECT(HasNumHistoryPlanes, int, std::integral_constant<int, G::NUM_HISTORY_PLANES>::value)
  GAMEINTERFACE_DETECT(HasInputHash, Hash128, G::input_hash(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasPolicyChannels, int, std::integral_constant<int, G::POLICY_CHANNELS>::value)
  GAMEINTERFACE_DETECT(HasActionPoint, void,
                       G::action_point(0, std::declval<int&>(), std::declval<int&>(), std::declval<int&>()))
  GAMEINTERFACE_DETECT(HasPointAction, int, G::point_action(0, 0, 0))
  GAMEINTERFACE_DETECT(HasSymmetries, unsigned, std::integral_constant<unsigned, G::SYMMETRIES>::value)
  GAMEINTERFACE_DETECT(HasSymPosition, P<G>, G::sym_position(std::declval<const P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasTargets, unsigned, std::integral_constant<unsigned, G::TARGETS>::value)
  GAMEINTERFACE_DETECT(HasTrainingTargets, unsigned,
                       G::training_targets(std::declval<const PositionAt<P<G>>&>(), 0, 0, 0, (float*)nullptr))
  GAMEINTERFACE_DETECT(HasScoreScale, double, G::SCORE_SCALE)
  GAMEINTERFACE_DETECT(HasScoreUnit, double, G::SCORE_UNIT)
  GAMEINTERFACE_DETECT(HasGameLength, double, G::GAME_LENGTH)
  //The optional members.
  GAMEINTERFACE_DETECT(HasLookaheadArea, double, G::LOOKAHEAD_AREA)
  GAMEINTERFACE_DETECT(HasKomiArea, double, G::KOMI_AREA)
  GAMEINTERFACE_DETECT(HasKomiBracketStep, double, G::KOMI_BRACKET_STEP)
  GAMEINTERFACE_DETECT(HasKomiBracketLimit, double, G::KOMI_BRACKET_LIMIT)
  GAMEINTERFACE_DETECT(HasKomiLeadStep, double, G::KOMI_LEAD_STEP)
  GAMEINTERFACE_DETECT(HasOwnTargets, unsigned, std::integral_constant<unsigned, G::OWN_TARGETS>::value)
}

//The slots of TARGETS that the game computes itself (training_targets): the game's, else all but the FRONT_TARGETS.
template<class E> constexpr unsigned ownTargets() {
  if constexpr(encoding::HasOwnTargets<E>::value)
    return E::OWN_TARGETS;
  else
    return E::TARGETS & ~FRONT_TARGETS;
}

template<class E> constexpr bool checkEncoding() {
  using namespace encoding;
  static_assert(HasRules<E>::value,
                "game interface, encoding part: the game lacks Rules (its rules part, with a Position)");
  static_assert(HasNumSpatialInputs<E>::value,
                "game interface, encoding part: the game lacks NUM_SPATIAL_INPUTS (constexpr int)");
  static_assert(HasNumGlobalInputs<E>::value,
                "game interface, encoding part: the game lacks NUM_GLOBAL_INPUTS (constexpr int)");
  static_assert(HasSpatialInputs<E>::value,
                "game interface, encoding part: the game lacks spatial_inputs(const Position&, const InputOptions&, "
                "set(x, y, plane))");
  static_assert(HasGlobalInputs<E>::value,
                "game interface, encoding part: the game lacks global_inputs(const Position&, const InputOptions&, "
                "float*)");
  static_assert(HasHistoryPlane<E>::value && HasNumHistoryPlanes<E>::value,
                "game interface, encoding part: the game lacks HISTORY_PLANE and NUM_HISTORY_PLANES (constexpr int, "
                "the history planes; 0 planes without history)");
  static_assert(HasInputHash<E>::value,
                "game interface, encoding part: the game lacks input_hash(const Position&) -> Hash128");
  static_assert(HasPolicyChannels<E>::value,
                "game interface, encoding part: the game lacks POLICY_CHANNELS (constexpr int)");
  static_assert(HasActionPoint<E>::value,
                "game interface, encoding part: the game lacks action_point(int action, int& x, int& y, int& channel)");
  static_assert(HasPointAction<E>::value,
                "game interface, encoding part: the game lacks point_action(int x, int y, int channel) -> int");
  static_assert(HasSymmetries<E>::value,
                "game interface, encoding part: the game lacks SYMMETRIES (constexpr unsigned)");
  static_assert(HasSymPosition<E>::value,
                "game interface, encoding part: the game lacks sym_position(const Position&, int symmetry) -> "
                "Position");
  static_assert(HasSymActionMap<E>::value,
                "game interface, encoding part: the game lacks SymActionMap (constructed from (const Position&, int "
                "symmetry), int operator()(int action) const)");
  static_assert(HasTargets<E>::value, "game interface, encoding part: the game lacks TARGETS (constexpr unsigned)");
  static_assert(HasScoreScale<E>::value, "game interface, encoding part: the game lacks SCORE_SCALE (double)");
  static_assert(HasScoreUnit<E>::value, "game interface, encoding part: the game lacks SCORE_UNIT (double)");
  static_assert(HasGameLength<E>::value, "game interface, encoding part: the game lacks GAME_LENGTH (double)");
  if constexpr(HasSymmetries<E>::value)
    static_assert((E::SYMMETRIES & 1u) != 0 && E::SYMMETRIES <= 0xFFu,
                  "game interface, encoding part: SYMMETRIES is a subset of KataGo's 8 with the identity");
  if constexpr(HasNumSpatialInputs<E>::value && HasHistoryPlane<E>::value && HasNumHistoryPlanes<E>::value)
    static_assert(E::HISTORY_PLANE >= 0 && E::NUM_HISTORY_PLANES >= 0 &&
                    E::HISTORY_PLANE + E::NUM_HISTORY_PLANES <= E::NUM_SPATIAL_INPUTS,
                  "game interface, encoding part: the history planes are planes of the game's");
  if constexpr(HasTargets<E>::value && HasOwnTargets<E>::value)
    static_assert((E::OWN_TARGETS & ~E::TARGETS) == 0 && (E::TARGETS & ~FRONT_TARGETS & ~E::OWN_TARGETS) == 0,
                  "game interface, encoding part: OWN_TARGETS holds slots of TARGETS only, and every slot of TARGETS "
                  "that the front does not compute (FRONT_TARGETS)");
  if constexpr(HasTargets<E>::value)
    static_assert(ownTargets<E>() == 0 || HasTrainingTargets<E>::value,
                  "game interface, encoding part: the game computes slots of its TARGETS itself (OWN_TARGETS, or a "
                  "slot that the front does not compute), so the game gives training_targets(positions, int "
                  "numPositions, int row, int mover, float* values) -> unsigned");
  if constexpr(HasRules<E>::value && HasPolicyChannels<E>::value) {
    using R = typename E::Rules;
    if constexpr(rules::HasMaxBoardLen<R>::value && rules::HasNumActions<R>::value)
      static_assert(R::NUM_ACTIONS == R::MAX_BOARD_LEN * R::MAX_BOARD_LEN * E::POLICY_CHANNELS + 1,
                    "game interface, encoding part: an action is its policy index, so NUM_ACTIONS is MAX_BOARD_LEN * "
                    "MAX_BOARD_LEN * POLICY_CHANNELS + 1");
  }
  if constexpr(HasLookaheadArea<E>::value)
    static_assert(E::LOOKAHEAD_AREA > 0, "game interface, encoding part: LOOKAHEAD_AREA is positive");
  if constexpr(HasKomiArea<E>::value)
    static_assert(E::KOMI_AREA > 0, "game interface, encoding part: KOMI_AREA is positive");
  static_assert(HasKomiBracketStep<E>::value == HasKomiBracketLimit<E>::value,
                "game interface, encoding part: the game gives both KOMI_BRACKET_STEP and KOMI_BRACKET_LIMIT, or "
                "neither");
  if constexpr(HasKomiBracketStep<E>::value && HasKomiBracketLimit<E>::value)
    static_assert(E::KOMI_BRACKET_STEP >= 0.5 && E::KOMI_BRACKET_LIMIT >= E::KOMI_BRACKET_STEP &&
                    E::KOMI_BRACKET_STEP * 2 == (double)(long long)(E::KOMI_BRACKET_STEP * 2) &&
                    E::KOMI_BRACKET_LIMIT * 2 == (double)(long long)(E::KOMI_BRACKET_LIMIT * 2),
                  "game interface, encoding part: KOMI_BRACKET_STEP is at least half a point, KOMI_BRACKET_LIMIT is "
                  "at least KOMI_BRACKET_STEP, and both are whole or half points");
  if constexpr(HasKomiLeadStep<E>::value)
    static_assert(E::KOMI_LEAD_STEP >= 0.0 && E::KOMI_LEAD_STEP * 2 == (double)(long long)(E::KOMI_LEAD_STEP * 2),
                  "game interface, encoding part: KOMI_LEAD_STEP is 0, or whole or half points");
  return true;
}

//The area of KataGo's look-ahead horizons on a board of xSize x ySize: the game's, else the board's.
template<class E> constexpr double lookaheadArea(int xSize, int ySize) {
  if constexpr(encoding::HasLookaheadArea<E>::value) {
    (void)xSize;
    (void)ySize;
    return E::LOOKAHEAD_AREA;
  }
  else
    return (double)xSize * ySize;
}

//The area of KataGo's komi sampler on a board of xSize x ySize: the game's, else the board's.
template<class E> constexpr double komiArea(int xSize, int ySize) {
  if constexpr(encoding::HasKomiArea<E>::value) {
    (void)xSize;
    (void)ySize;
    return E::KOMI_AREA;
  }
  else
    return (double)xSize * ySize;
}

//The bracket of KataGo's even-komi search, in points: its first offset and its largest. The game's, else KataGo's.
template<class E> constexpr double komiBracketStep() {
  if constexpr(encoding::HasKomiBracketStep<E>::value)
    return E::KOMI_BRACKET_STEP;
  else
    return 1.0;
}
template<class E> constexpr double komiBracketLimit() {
  if constexpr(encoding::HasKomiBracketLimit<E>::value)
    return E::KOMI_BRACKET_LIMIT;
  else
    return 32.0;
}
//Whether the even-komi search takes its bracket's end as the even komi when the win-loss value keeps its sign up to
//the limit: KataGo's rule, for a game that keeps KataGo's bracket. A game that gives its bracket has no even komi there.
template<class E> constexpr bool komiBracketEndIsEven() { return !encoding::HasKomiBracketLimit<E>::value; }

//The step of the lead's smoothing over a coarse score, in points: the game's, else KataGo's half point times the
//game's SCORE_SCALE.
template<class E> constexpr double komiLeadStep() {
  if constexpr(encoding::HasKomiLeadStep<E>::value)
    return E::KOMI_LEAD_STEP;
  else
    return 0.5 * E::SCORE_SCALE;
}

//The game's own training targets of a row (training_targets): none for a game without own targets. values holds
//TargetValues::size(area) floats.
template<class E, class Positions>
unsigned trainingTargets(const Positions& positions, int numPositions, int row, int mover, float* values) {
  if constexpr(encoding::HasTrainingTargets<E>::value)
    return E::training_targets(positions, numPositions, row, mover, values);
  else {
    (void)positions; (void)numPositions; (void)row; (void)mover; (void)values;
    return 0;
  }
}

#undef GAMEINTERFACE_DETECT

}  // namespace GameInterface

#endif  // GAME_GAMEINTERFACE_H_
