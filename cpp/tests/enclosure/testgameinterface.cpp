// The game interface (game/gameinterface.h): Enclosure's implementation (game/enclosure/gamerules.h,
// gameencoding.h), and the defaults of the optional members.
//
//   testgameinterface tests/data/enclosure_games.txt
//
// The front (Board, BoardHistory, NNInputs, NNPos, SymmetryHelpers, the training rows) calls both parts, so each member
// has one copy. The test replays the fixture's games (as testgamelayer does) and random games with timeout passes and
// komi. At each position:
// - the rules part, against itself: the player to move and the edges left (against the 1-2-2 turn order), is_legal
//   against legal_actions, the move and the position in the notation (each read back), the engine protocol's moves
//   (each read back), and at the end the result;
// - the encoding part: the front's rows against the members (NNInputs::fillRowV8, and fillRowV8Packed at each
//   symmetry), with and without the history planes and at several komis; the history planes (the only planes that
//   the history changes); the policy layout (NNPos); and the symmetries: the action map onto the image's legal
//   actions, the inputs at a symmetry against the image's inputs, and the inverse of each image.
// In the fixture's first game, the hashes (the position's, the situation's and the local pattern's), the images of
// the position and the training targets are checked against pinned values. A stub game without the optional members
// checks their defaults.

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

#include "../../core/rand.h"
#include "../../game/boardhistory.h"
#include "../../game/chosengame.h"
#include "../../game/enclosure/position_check.h"
#include "../../neuralnet/nninputs.h"
#include "../../search/localpattern.h"

using namespace std;

using R = GameRules;
using E = GameEncoding;

static_assert(E::NUM_SPATIAL_INPUTS + 1 == NNInputs::NUM_FEATURES_SPATIAL_V8, "the on-board plane, then the game's");
static_assert(E::NUM_GLOBAL_INPUTS == NNInputs::NUM_FEATURES_GLOBAL_V8, "the game's global inputs are V8's");

static int failures = 0;
static void fail(const string& what) {
  if(failures < 20)
    fprintf(stderr, "FAIL %s\n", what.c_str());
  failures++;
}

static int checkedPositions = 0, inputChecks = 0, symmetryChecks = 0, allActionChecks = 0;

//The entries of the front's row (V8, NHWC) that are 1, as the game's members give them: the on-board plane first,
//then the game's planes.
template<class SetIndex>
static void forEachInput(const R::Position& p, const GameInterface::InputOptions& options, SetIndex set) {
  const int planes = NNInputs::NUM_FEATURES_SPATIAL_V8;
  for(int pos = 0; pos < 19 * 19; pos++)
    set(pos * planes);
  E::spatial_inputs(p, options, [&](int x, int y, int plane) {
    set((y * 19 + x) * planes + NNInputs::V8_GAME_PLANES_START + plane);
  });
}

static void interfaceRow(const R::Position& p, const GameInterface::InputOptions& options, float* bin, float* glob) {
  fill(bin, bin + NNInputs::NUM_FEATURES_SPATIAL_V8 * 19 * 19, 0.0f);
  forEachInput(p, options, [&](int i) { bin[i] = 1.0f; });
  E::global_inputs(p, options, glob);
}

static void interfaceBits(const R::Position& p, const GameInterface::InputOptions& options, uint32_t* bits) {
  fill(bits, bits + NNInputs::NUM_SPATIAL_WORDS_V8, 0u);
  forEachInput(p, options, [&](int i) { bits[i >> 5] |= 1u << (i & 31); });
}

//The whole state that the members see: the rules' hash and the inputs' hash.
static Hash128 fullHash(const R::Position& p) { return R::hash(p) ^ E::input_hash(p); }

//FNV-1a over bytes, for the pinned digests.
static uint64_t digest(uint64_t h, const void* data, size_t n) {
  const unsigned char* b = (const unsigned char*)data;
  for(size_t i = 0; i < n; i++)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}
static const uint64_t DIGEST_START = 14695981039346656037ULL;

//expectedMoves: the moves played, or -1 if unknown.
static void checkPosition(const Board& board, const BoardHistory& hist, const string& where, int expectedMoves) {
  const int index = checkedPositions++;
  const R::Position& p = board.pos;
  const Player pla = board.nextPla();

  //The rules part.
  if(expectedMoves >= 0) {
    int wantPlayer, wantLeft;
    enclosure::position_check::expected_turn(expectedMoves, wantPlayer, wantLeft);
    if(R::moves_played(p) != expectedMoves || R::next_player(p) != wantPlayer || R::turn_moves_left(p) != wantLeft)
      fail(where + ": moves_played, next_player or turn_moves_left is not the 1-2-2 turn order's");
  }
  vector<int> legal;
  R::legal_actions(p, legal);
  for(int a : legal)
    if(!R::is_legal(p, a))
      fail(where + ": is_legal rejects a listed action");
  if(index % 200 == 0) {
    allActionChecks++;
    set<int> listed(legal.begin(), legal.end());
    for(int a = -1; a <= R::NUM_ACTIONS; a++)
      if(R::is_legal(p, a) != (listed.count(a) > 0))
        fail(where + ": is_legal differs from legal_actions at action " + Global::intToString(a));
  }
  for(int a : legal) {
    string why;
    if(R::parse_action(p, R::action_text(a), &why) != a)
      fail(where + ": action_text does not read back (" + R::action_text(a) + ": " + why + ")");
    if(a == R::PASS_ACTION)
      continue;
    int read = -1;
    if(!GameInterface::parseProtocolText<R>(Global::toLower(GameInterface::protocolText<R>(a)), read) || read != a)
      fail(where + ": the protocol's text does not read back (" + GameInterface::protocolText<R>(a) + ")");
  }
  if(index % 10 == 0) {
    R::Position read;
    string why;
    if(!R::parse_position(R::position_text(p), read, why) || R::hash(read) != R::hash(p))
      fail(where + ": position_text does not read back (" + why + ")");
  }

  //The encoding part: the inputs, with and without history and at the history's komi.
  const int numBin = NNInputs::NUM_FEATURES_SPATIAL_V8 * 19 * 19;
  static vector<float> bin(numBin), bin2(numBin);
  float glob[NNInputs::NUM_FEATURES_GLOBAL_V8], glob2[NNInputs::NUM_FEATURES_GLOBAL_V8];
  if(index % 3 == 0) {
    inputChecks++;
    static vector<float> withHistory(numBin);
    for(int history = 1; history >= 0; history--) {
      MiscNNInputParams params;
      if(!history)
        params.maxHistory = 0;
      GameInterface::InputOptions options;
      options.history = history != 0;
      options.komi = hist.rules.komi;
      NNInputs::fillRowV8(board, hist, pla, params, 19, 19, true, bin.data(), glob);
      interfaceRow(p, options, bin2.data(), glob2);
      if(bin != bin2)
        fail(where + ": NNInputs::fillRowV8 differs from the on-board plane and spatial_inputs");
      for(int i = 0; i < NNInputs::NUM_FEATURES_GLOBAL_V8; i++)
        if(glob[i] != glob2[i])
          fail(where + ": NNInputs::fillRowV8 differs from global_inputs");
      if(history)
        withHistory = bin2;
    }
    //The history changes the history planes only, and without it they are empty (bin2 is the row without history).
    const int planes = NNInputs::NUM_FEATURES_SPATIAL_V8;
    for(int i = 0; i < numBin; i++) {
      const int plane = i % planes - NNInputs::V8_GAME_PLANES_START;
      const bool historyPlane = plane >= E::HISTORY_PLANE && plane < E::HISTORY_PLANE + E::NUM_HISTORY_PLANES;
      if(historyPlane ? bin2[i] != 0.0f : bin2[i] != withHistory[i])
        fail(where + ": the inputs without history differ outside the history planes, or set one");
    }
  }

  //The policy layout: an action's point and channel give its policy index.
  for(int a : legal) {
    if(a == R::PASS_ACTION) {
      if(NNPos::locToPos(Board::PASS_LOC, 19, 19, 19) != R::PASS_ACTION)
        fail(where + ": pass is not the policy's last entry");
      continue;
    }
    int x, y, channel;
    E::action_point(a, x, y, channel);
    if(NNPos::locToPos(Board::moveLocOfAction(a), 19, 19, 19) != (y * 19 + x) * E::POLICY_CHANNELS + channel ||
       E::point_action(x, y, channel) != a)
      fail(where + ": action_point or point_action differs from NNPos::locToPos");
  }

  //The symmetries: the image position and its inverse, the action map, and the image's inputs.
  if(index % 10 == 0) {
    static vector<uint32_t> bits(NNInputs::NUM_SPATIAL_WORDS_V8), bits2(NNInputs::NUM_SPATIAL_WORDS_V8),
      bits3(NNInputs::NUM_SPATIAL_WORDS_V8);
    float glob3[NNInputs::NUM_FEATURES_GLOBAL_V8];
    for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++) {
      symmetryChecks++;
      const R::Position image = E::sym_position(p, s);
      if(fullHash(E::sym_position(image, SymmetryHelpers::invert(s))) != fullHash(p))
        fail(where + ": the inverse symmetry does not map the image back");
      //The images of the legal actions are the image position's legal actions, listed by the rules.
      const E::SymActionMap actionMap(p, s);
      vector<int> mapped, imageLegal;
      for(int a : legal)
        mapped.push_back(actionMap(a));
      sort(mapped.begin(), mapped.end());
      R::legal_actions(image, imageLegal);
      if(mapped != imageLegal)
        fail(where + ": the action map's images are not the image position's legal actions");
      //The front writes the members' inputs at the symmetry, and they are the image position's.
      MiscNNInputParams params;
      GameInterface::InputOptions options;
      options.komi = hist.rules.komi;
      interfaceBits(image, options, bits3.data());
      E::global_inputs(image, options, glob3);
      options.symmetry = s;
      NNInputs::fillRowV8Packed(board, hist, pla, params, 19, 19, true, bits.data(), glob, s);
      interfaceBits(p, options, bits2.data());
      if(bits != bits2)
        fail(where + ": NNInputs::fillRowV8Packed differs from spatial_inputs at symmetry " + Global::intToString(s));
      if(bits2 != bits3 || !equal(glob, glob + NNInputs::NUM_FEATURES_GLOBAL_V8, glob3))
        fail(where + ": the inputs at symmetry " + Global::intToString(s) + " are not the image position's");
    }
  }
}

//A stub game without the optional members: a 3 x 3 board, where a move puts the mover's piece on an empty point and
//the players alternate. The game ends on a full board, and pass is never legal.
namespace stub {
struct Position {
  int8_t cell[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  int moves = 0;
};
struct GameRules {
  static constexpr const char* NAME = "stub";
  using Position = stub::Position;
  static constexpr int MAX_BOARD_LEN = 3;
  static constexpr int NUM_ACTIONS = 10;
  static constexpr int PASS_ACTION = 9;
  static bool new_game(int xSize, int ySize, Position& p) {
    p = Position();
    return xSize == 3 && ySize == 3;
  }
  static int x_size(const Position&) { return 3; }
  static int y_size(const Position&) { return 3; }
  static int next_player(const Position& p) { return p.moves % 2; }
  static int turn_moves_left(const Position&) { return 1; }
  static int moves_played(const Position& p) { return p.moves; }
  static bool terminal(const Position& p) { return p.moves == 9; }
  static void legal_actions(const Position& p, vector<int>& out) {
    out.clear();
    for(int a = 0; a < 9; a++)
      if(p.cell[a] < 0)
        out.push_back(a);
  }
  static bool is_legal(const Position& p, int a) { return a >= 0 && a < 9 && p.cell[a] < 0; }
  static void play(Position& p, int a) { p.cell[a] = (int8_t)next_player(p); p.moves++; }
  static float result(const Position&) { return 0.f; }
  static double score(const Position&, int) { return 0.0; }
  static Hash128 hash(const Position& p) {
    Hash128 h;
    for(int i = 0; i < 9; i++)
      h.hash0 = h.hash0 * 3 + (uint64_t)(p.cell[i] + 1);
    return h;
  }
  static void pieces(const Position& p, int8_t* owner) { copy(p.cell, p.cell + 9, owner); }
  static void area(const Position& p, int8_t* owner) { pieces(p, owner); }
  //GTP points: A1 to C3, row 1 at the bottom.
  static string action_text(int a) {
    return a == PASS_ACTION ? string("pass") : string(1, char('A' + a % 3)) + char('1' + a / 3);
  }
  static int parse_action(const Position& p, const string& text, string* why) {
    for(int a = 0; a < 9; a++)
      if(text == action_text(a) && is_legal(p, a))
        return a;
    if(why != nullptr)
      *why = "not a legal point";
    return -1;
  }
  static string position_text(const Position& p) {
    string s;
    for(int8_t c : p.cell)
      s += c < 0 ? '.' : c == 0 ? 'x' : 'o';
    return s;
  }
  static bool parse_position(const string&, Position&, string& why) {
    why = "not read";
    return false;
  }
};
static_assert(GameInterface::checkRules<GameRules>(), "the stub game's rules part");
//The stub game with a standard board of its own, 3 x 2 (the optional pair STANDARD_X_SIZE, STANDARD_Y_SIZE).
struct WithStandardBoard : GameRules {
  static constexpr int STANDARD_X_SIZE = 3, STANDARD_Y_SIZE = 2;
};
static_assert(GameInterface::checkRules<WithStandardBoard>(), "the stub game with a standard board");
}  // namespace stub

//The hashes of before the interface, pinned: the values that KataGame main (192acb41) gives in the fixture's first
//game, at the ply given. The checks above run the same code on both sides, so only these catch a change of the
//hash's tables (their seed or their order), of the history's hash or of the local pattern keys. The local hashes
//are those of the move A10-A7, from a hasher with a window of 7 and the keys of the seed below.
struct PinnedHashes {
  int ply;
  const char* pos;          //Board::pos_hash
  const char* situation;    //BoardHistory::getSituationRulesAndKoHash, komi 0.5
  const char* local[3];     //getHash, getHashWithSym at symmetry 3, and at symmetry 6 with flipColors
};
static const PinnedHashes PINNED[] = {
  {0, "287C6CE6850489F7A8E988B7A6CA371C", "A25646C97749329A6DDC71C87606496A",
   {"400F8B95A2DA2E06620DAEFEB1CF30EA", "E39B3ADFA44F4145EDC2100273C52E1B", "BC71B3CC7650008A77B1AF372E57E74A"}},
  {7, "0994BAAD5524EA138DABC7ECA714F1DA", "AD8747636BE134CB0338D768B06576C7",
   {"0BA5F626F542F39C966AA57B43DAE712", "B97527EAD3B027788D84081A5289138A", "CF235627EE8EA9373219A76A1537BC51"}},
  {120, "EC5A1BAB5CF860F5435C016BCE0FA104", "FB5129F8DB826F3E0EE65F9F61E337A6", {nullptr, nullptr, nullptr}},
};
static int pinnedChecks = 0;
static void checkPinnedHashes(const Board& board, const BoardHistory& hist, int ply) {
  static LocalPatternHasher hasher;
  static bool hasherReady = false;
  if(!hasherReady) {
    Rand rand("probe local pattern keys");
    hasher.init(7, 7, rand);
    hasherReady = true;
  }
  for(const PinnedHashes& pin : PINNED) {
    if(pin.ply != ply)
      continue;
    pinnedChecks++;
    const Player pla = board.nextPla();
    const string where = "the pinned hashes at ply " + Global::intToString(ply);
    if(board.pos_hash.toString() != pin.pos)
      fail(where + ": the position's hash");
    if(BoardHistory::getSituationRulesAndKoHash(board, hist, pla, 0.5).toString() != pin.situation)
      fail(where + ": the situation's hash");
    if(pin.local[0] == nullptr)
      continue;
    const Loc loc = Location::ofString("A10-A7", board);
    if(hasher.getHash(board, loc, pla).toString() != pin.local[0] ||
       hasher.getHashWithSym(board, loc, pla, 3, false).toString() != pin.local[1] ||
       hasher.getHashWithSym(board, loc, pla, 6, true).toString() != pin.local[2])
      fail(where + ": the local pattern hashes");
  }
}

//The fixture's first game, position by position, pinned where the front had a copy of the members' code. Both pins
//are the values of KataGame main (192acb41).
//- The images of the position (E::sym_position) at each symmetry: a digest of their full hashes, at the plies given.
//  Before the front called the encoding part, this test checked sym_position against SymmetryHelpers::getSymPosition.
//- The training targets (E::training_targets) of the rows given: the slots that hold a target and a digest of the
//  values. The rows of a self-play game, written by trainingwrite.cpp before the targets moved into the game and by
//  the game's member after, are byte-identical.
struct PinnedImages {
  int ply;
  uint64_t images;
};
static const PinnedImages PINNED_IMAGES[] = {
  {0, 0x200D6413A3AD1DE1ULL}, {7, 0x4CE6774F84207634ULL}, {60, 0x9E0538511F520929ULL}};
struct PinnedTargets {
  int row;
  unsigned slots;
  uint64_t values;
};
static constexpr unsigned GAME_SLOTS = GameInterface::TARGET_OWNERSHIP | GameInterface::TARGET_SEKI |
                                       GameInterface::TARGET_SCORING | GameInterface::TARGET_INCOME;
static const PinnedTargets PINNED_TARGETS[] = {
  {0, GAME_SLOTS, 0x924A4FEA6C761C27ULL}, {7, GAME_SLOTS, 0xCE7144B166577825ULL},
  {59, GAME_SLOTS, 0xF554CD95F68E5B36ULL}, {118, GAME_SLOTS, 0xB30C52306B6E56A2ULL}};
static int pinnedGameChecks = 0;
static void checkPinnedGame(const vector<R::Position>& game) {
  for(const PinnedImages& pin : PINNED_IMAGES) {
    if(pin.ply >= (int)game.size())
      continue;
    pinnedGameChecks++;
    uint64_t h = DIGEST_START;
    for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++) {
      const Hash128 image = fullHash(E::sym_position(game[pin.ply], s));
      h = digest(h, &image.hash0, sizeof(image.hash0));
      h = digest(h, &image.hash1, sizeof(image.hash1));
    }
    if(h != pin.images)
      fail("the images at ply " + Global::intToString(pin.ply) + ": digest " + Global::uint64ToHexString(h));
  }
  //Every row's targets, each value in its slot's range (GameInterface::TargetValues; the front's slots stay 0), and
  //the pinned rows' digests.
  const int area = 19 * 19;
  vector<float> values(GameInterface::TargetValues::NUM_CHANNELS * area);
  for(int row = 0; row + 1 < (int)game.size(); row++) {
    fill(values.begin(), values.end(), 0.0f);
    const unsigned slots = GameInterface::trainingTargets<E>([&](int j) -> const R::Position& { return game[j]; },
                                                              (int)game.size(), row, R::next_player(game[row]),
                                                              values.data());
    for(int c = 0; c < GameInterface::TargetValues::NUM_CHANNELS; c++)
      for(int q = 0; q < area; q++) {
        const float v = values[c * area + q];
        const bool ok = c == GameInterface::TargetValues::CLASS_CHANNEL
                          ? (v == -1.0f || v == 0.0f || v == 1.0f || v == 2.0f)
                      : (c == 2 || c == 3 || c == 7) ? v == 0.0f
                      : (v >= -1.0f && v <= 1.0f);
        if(!ok) {
          fail("the training targets of row " + Global::intToString(row) + ": a value out of its slot's range");
          return;
        }
      }
    for(const PinnedTargets& pin : PINNED_TARGETS) {
      if(pin.row != row)
        continue;
      pinnedGameChecks++;
      const uint64_t h = digest(DIGEST_START, values.data(), values.size() * sizeof(float));
      if(slots != pin.slots || h != pin.values)
        fail("the training targets of row " + Global::intToString(row) + ": slots " + Global::intToString((int)slots) +
             ", digest " + Global::uint64ToHexString(h));
    }
  }
}

//The optional members: Enclosure has them, and the stub game gets the defaults.
static void checkOptionalMembers() {
  using S = stub::GameRules;
  static_assert(GameInterface::rules::HasProtocolText<R>::value && GameInterface::rules::HasLocalHash<R>::value &&
                  GameInterface::rules::HasQueryPositionField<R>::value &&
                  GameInterface::rules::HasParseQueryPosition<R>::value,
                "Enclosure has the optional members");
  static_assert(!GameInterface::rules::HasProtocolText<S>::value && !GameInterface::rules::HasLocalHash<S>::value &&
                  !GameInterface::rules::HasQueryPositionField<S>::value &&
                  !GameInterface::rules::HasParseQueryPosition<S>::value,
                "the stub game has none");

  //The turn, the replay and the standard board. Enclosure: two edges a turn, a replay that passes at any time (its
  //timeout pass), and the default standard board, its largest. The stub game: the defaults (one move a turn, a replay
  //of the legal moves only, the largest board), and its standard board when it gives one.
  static_assert(GameInterface::turnMoves<R>() == 2 && GameInterface::standardXSize<R>() == 19 &&
                  GameInterface::standardYSize<R>() == 19,
                "Enclosure's turn and standard board");
  static_assert(GameInterface::turnMoves<S>() == 1 && GameInterface::standardXSize<S>() == 3 &&
                  GameInterface::standardYSize<S>() == 3,
                "the defaults: one move a turn, the largest board");
  static_assert(GameInterface::standardXSize<stub::WithStandardBoard>() == 3 &&
                  GameInterface::standardYSize<stub::WithStandardBoard>() == 2,
                "the game's standard board");
  {
    const R::Position start = R::Position();
    vector<int> legal;
    R::legal_actions(start, legal);
    const int edge = legal.front();
    if(!GameInterface::replayTolerates<R>(start, R::PASS_ACTION) || GameInterface::replayTolerates<R>(start, edge) ||
       GameInterface::replayTolerates<R>(start, -1) ||
       GameInterface::replayTolerates<S>(stub::Position(), S::PASS_ACTION))
      fail("the replay's tolerance: Enclosure's pass at any time, nothing else, and none for the stub game");
    //The front: Board() is the standard board, and a replayed game passes at any time, while an edge is legal.
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    if(board.x_size != 19 || board.y_size != 19)
      fail("Board() is not Enclosure's standard board, 19 x 19");
    if(hist.isLegal(board, Board::PASS_LOC, P_BLACK) || !hist.isLegalTolerant(board, Board::PASS_LOC, P_BLACK) ||
       !hist.isLegalTolerant(board, Board::moveLocOfAction(edge), P_BLACK))
      fail("a replayed game of Enclosure does not pass at any time, or does not play a legal edge");
  }

  //Enclosure: the protocol's moves as before the interface (letters A to S, with I), and its own window keys.
  const int a10d10 = enclosure::action_between(0, 9, 3, 9);
  const int s19 = enclosure::action_between(18, 18, 15, 15);
  int read = -1;
  Loc loc;
  if(GameInterface::protocolText<R>(a10d10) != "A10-D10" || GameInterface::protocolText<R>(s19) != "S19-P16" ||
     R::action_text(s19) == "S19-P16")
    fail("Enclosure's protocol text is not the form with I (A10-D10, S19-P16)");
  if(!GameInterface::parseProtocolText<R>("a10-d10", read) || read != a10d10 ||
     GameInterface::parseProtocolText<R>("A10-E10", read) || GameInterface::parseProtocolText<R>("T1-T2", read) ||
     GameInterface::parseProtocolText<R>("A10", read) || GameInterface::parseProtocolText<R>("pass", read))
    fail("Enclosure's protocol parser");
  if(!Location::tryOfString(" a10-d10 ", 19, 19, loc) || loc != Board::moveLocOfAction(a10d10) ||
     !Location::tryOfString("Pass", 19, 19, loc) || loc != Board::PASS_LOC ||
     Location::toString(Board::moveLocOfAction(a10d10), 19, 19) != "A10-D10" ||
     Location::toString(Board::PASS_LOC, 19, 19) != "pass")
    fail("Location's moves through the protocol's text");
  if(GameInterface::numLocalKeys<R>() != 3 * enclosure::PASS)
    fail("Enclosure's local hash keys");
  if(string(GameInterface::queryPositionField<R>()) != "enclosurePosition")
    fail("Enclosure's query position field");

  //The stub game: the protocol's text is action_text, read back in any case.
  for(int a = 0; a < 9; a++)
    if(GameInterface::protocolText<S>(a) != S::action_text(a) ||
       !GameInterface::parseProtocolText<S>(Global::toLower(S::action_text(a)), read) || read != a)
      fail("the default protocol text of action " + Global::intToString(a));
  if(GameInterface::parseProtocolText<S>("D1", read) || GameInterface::parseProtocolText<S>("", read) ||
     GameInterface::parseProtocolText<S>("pass", read) || GameInterface::parseProtocolText<S>("B22", read))
    fail("the default protocol parser reads text that names no action");

  //The stub game: no window keys, KataGo's window, and the default hash of the window at a corner, with radius 1. The
  //first player's piece is at A1 (0, 0), the second's at B2 (1, 1). A point's key is
  //keys[state * 9 + (dy + 1) * 3 + (dx + 1)], KataGo's colors: 0 empty, 1 the first player, 2 the second. The five
  //points of the window off the board are left out, as KataGo leaves them out.
  if(GameInterface::numLocalKeys<S>() != 0 || GameInterface::queryPositionField<S>() != nullptr)
    fail("the stub game has window keys or a query field");
  if(GameInterface::localWindow<S>() != 5 || GameInterface::localWindow<R>() != 7)
    fail("the subtree value bias's window is not KataGo's 5 for the stub game and 7 for Enclosure");
  Rand rand("testgameinterface window keys");
  Hash128 keys[4 * 9];
  for(Hash128& k : keys)
    k = Hash128(rand.nextUInt64(), rand.nextUInt64());
  stub::Position p;
  S::play(p, 0);
  S::play(p, 4);
  auto expected = [&](bool flip) {
    Hash128 h;
    h ^= keys[(flip ? 2 : 1) * 9 + 4];  //A1: dx = 0, dy = 0
    h ^= keys[0 * 9 + 5];               //B1: empty
    h ^= keys[0 * 9 + 7];               //A2: empty
    h ^= keys[(flip ? 1 : 2) * 9 + 8];  //B2
    return h;
  };
  for(bool flip : {false, true})
    if(GameInterface::localHash<S>(p, 0, 0, 1, 1, flip, nullptr, keys) != expected(flip))
      fail(string("the default window hash, flipPlayers ") + (flip ? "true" : "false"));
  string why;
  if(GameInterface::parseQueryPosition<S>(0, p, why) || why.empty())
    fail("the stub game reads a query position field");
}

static constexpr bool allSymmetriesInOrder() {
  for(int i = 0; i < SymmetryHelpers::NUM_ALLOWED_SYMMETRIES; i++)
    if(SymmetryHelpers::ALLOWED[i] != i)
      return false;
  return true;
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testgameinterface GAMES_FILE\n");
    return 2;
  }
  Board::initHash();

  //The allowed symmetries: all 8 for Enclosure, and a subset as Connect-4's (the identity and the left-right mirror,
  //symmetry 2). Enclosure's own targets.
  static_assert(SymmetryHelpers::NUM_ALLOWED_SYMMETRIES == 8 && allSymmetriesInOrder(), "Enclosure allows all 8");
  constexpr std::array<int, 2> mirror = GameInterface::symmetryList<0x5u>();
  static_assert(mirror[0] == 0 && mirror[1] == 2 && !GameInterface::hasSymmetry(0x5u, 1) &&
                  !GameInterface::hasSymmetry(0xFFu, 8) && !GameInterface::hasSymmetry(0xFFu, -1),
                "the identity and the left-right mirror (Connect-4's), and symmetries out of range");
  static_assert(GameInterface::encoding::HasTrainingTargets<E>::value &&
                  (E::TARGETS & ~GameInterface::FRONT_TARGETS) != 0,
                "Enclosure computes its own targets");
  {
    R::Position other;
    if(R::new_game(9, 9, other) || R::new_game(19, 18, other))
      fail("new_game accepts a board other than 19 x 19");
  }
  checkOptionalMembers();

  //The fixture's games.
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  int games = 0;
  string line;
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "game")
      continue;
    games++;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    //A pass (the player to move has no legal edge) forfeits the rest of the turn: the turn order then shifts.
    bool passed = false;
    vector<R::Position> positions;
    for(int i = 0; i < n; i++) {
      getline(in, line);
      istringstream row(line);
      int action;
      row >> action;
      checkPosition(board, hist, label + " move " + Global::intToString(i + 1), passed ? -1 : i);
      if(games == 1)
        checkPinnedHashes(board, hist, i);
      positions.push_back(board.pos);
      hist.makeBoardMoveAssumeLegal(board, Board::moveLocOfAction(action), board.nextPla(), NULL);
      passed = passed || action == R::PASS_ACTION;
    }
    checkPosition(board, hist, label + " end", passed ? -1 : n);
    positions.push_back(board.pos);
    if(games == 1) {
      checkPinnedHashes(board, hist, n);
      checkPinnedGame(positions);
    }
    getline(in, line);
    istringstream tail(line);
    int result;
    tail >> word >> result;
    if(!R::terminal(board.pos) || R::result(board.pos) != (result > 0 ? 1.f : result < 0 ? -1.f : 0.f))
      fail(label + ": result differs from the fixture's");
  }

  //Random games, with timeout passes (a pass while edges are legal forfeits the turn) and komi.
  const int randomGames = 60;
  {
    Rand rand("testgameinterface random games");
    const float komis[] = {0.0f, 0.5f, -0.5f, 12.0f, -37.5f, 144.0f};
    for(int g = 0; g < randomGames; g++) {
      Board board;
      BoardHistory hist(board, P_BLACK, Rules(), 0);
      hist.setKomi(komis[g % (sizeof(komis) / sizeof(komis[0]))]);
      int ply = 0;
      while(!board.isGameOver()) {
        checkPosition(board, hist, "random game " + Global::intToString(g) + " ply " + Global::intToString(ply), -1);
        vector<int> legal;
        R::legal_actions(board.pos, legal);
        const Player pla = board.nextPla();
        if(rand.nextBool(0.05))
          hist.makeBoardMoveTolerant(board, Board::PASS_LOC, pla);
        else
          hist.makeBoardMoveAssumeLegal(board, Board::moveLocOfAction(legal[rand.nextUInt((uint32_t)legal.size())]),
                                        pla, NULL);
        ply++;
      }
      checkPosition(board, hist, "random game " + Global::intToString(g) + " end", -1);
      const double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK);
      if(R::result(board.pos) != (margin < 0 ? 1.f : margin > 0 ? -1.f : 0.f))
        fail("random game " + Global::intToString(g) + ": result is not the sign of the scores");
    }
  }

  if(pinnedChecks != (int)(sizeof(PINNED) / sizeof(PINNED[0])) ||
     pinnedGameChecks != (int)(sizeof(PINNED_IMAGES) / sizeof(PINNED_IMAGES[0]) +
                               sizeof(PINNED_TARGETS) / sizeof(PINNED_TARGETS[0])))
    fail("the fixture's first game did not reach every pinned ply");
  printf("%d fixture games and %d random games: %d positions, %d with the inputs checked, %d symmetry checks, "
         "%d with every action checked, %d with pinned hashes, %d failures\n",
         games, randomGames, checkedPositions, inputChecks, symmetryChecks, allActionChecks, pinnedChecks, failures);
  return failures == 0 ? 0 : 1;
}
