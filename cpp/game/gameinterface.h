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
 *   action keeps its number on every board size.
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
 *   input_hash(p)           Hash128: what the inputs read beyond hash(p), such as earlier turns. The front mixes it
 *                           into the key of the network's cache.
 *   POLICY_CHANNELS         int: policy channels per point.
 *   action_point(action, x, y, channel)
 *                           void: the point and the channel of an action other than pass.
 *   point_action(x, y, channel)
 *                           int: the action at that point and channel. The policy's index of an action is
 *                           (y * nnXLen + x) * POLICY_CHANNELS + channel, and pass is last.
 *   SYMMETRIES              unsigned: the allowed subset of KataGo's 8 symmetries, bit s for symmetry s. Bit 0 (the
 *                           identity) is always set.
 *   sym_position(p, symmetry)
 *                           Position: the image of the position.
 *   SymActionMap            a type: SymActionMap(p, symmetry) maps each legal action of p and pass (operator(),
 *                           int to int) to its image, an action of sym_position(p, symmetry).
 *   TARGETS                 unsigned: the optional training targets that the game writes (GameTarget bits).
 *   SCORE_SCALE             double: the game's score spread against Go's. KataGo's constants in points scale by it
 *                           (the score distribution's bin width, the output multipliers, the Huber deltas).
 *   SCORE_UNIT              double: the search's score utility unit, in points, where KataGo uses the square root
 *                           of the board's area.
 *   GAME_LENGTH             double: a game's mean length in moves, for KataGo's game-phase constants (the earliest
 *                           resignation, early forks, the opening moves, the time controls).
 */

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
//front writes 0, with weight 0, in each slot that the game does not write.
enum GameTarget : unsigned {
  TARGET_OWNERSHIP = 1u << 0,      //C0: KataGo's ownership.
  TARGET_SEKI = 1u << 1,           //C1: KataGo's seki areas.
  TARGET_FUTURE_PIECES = 1u << 2,  //C2-3: the pieces 8 and 32 moves later (pieces()), KataGo's future positions.
  TARGET_SCORING = 1u << 3,        //C4: KataGo's scoring (final area).
  TARGET_INCOME = 1u << 4,         //C5-6: a value per point for each player.
  TARGET_NEXT_PIECES = 1u << 5,    //C7: the next mover's pieces after the row's move.
};

//KataGo's symmetry of a point on a board of xSize x ySize (SymmetryHelpers::getSymLoc): flip x, flip y, then
//transpose.
inline void symPoint(int& x, int& y, int symmetry, int xSize, int ySize) {
  if(symmetry & 0x2) x = xSize - x - 1;
  if(symmetry & 0x1) y = ySize - y - 1;
  if(symmetry & 0x4) std::swap(x, y);
}

//A writer for spatial_inputs, for the detection below.
struct SetEntry {
  void operator()(int x, int y, int plane) const;
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
  return true;
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
  GAMEINTERFACE_DETECT(HasInputHash, Hash128, G::input_hash(std::declval<const P<G>&>()))
  GAMEINTERFACE_DETECT(HasPolicyChannels, int, std::integral_constant<int, G::POLICY_CHANNELS>::value)
  GAMEINTERFACE_DETECT(HasActionPoint, void,
                       G::action_point(0, std::declval<int&>(), std::declval<int&>(), std::declval<int&>()))
  GAMEINTERFACE_DETECT(HasPointAction, int, G::point_action(0, 0, 0))
  GAMEINTERFACE_DETECT(HasSymmetries, unsigned, std::integral_constant<unsigned, G::SYMMETRIES>::value)
  GAMEINTERFACE_DETECT(HasSymPosition, P<G>, G::sym_position(std::declval<const P<G>&>(), 0))
  GAMEINTERFACE_DETECT(HasTargets, unsigned, std::integral_constant<unsigned, G::TARGETS>::value)
  GAMEINTERFACE_DETECT(HasScoreScale, double, G::SCORE_SCALE)
  GAMEINTERFACE_DETECT(HasScoreUnit, double, G::SCORE_UNIT)
  GAMEINTERFACE_DETECT(HasGameLength, double, G::GAME_LENGTH)
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
  return true;
}

#undef GAMEINTERFACE_DETECT

}  // namespace GameInterface

#endif  // GAME_GAMEINTERFACE_H_
