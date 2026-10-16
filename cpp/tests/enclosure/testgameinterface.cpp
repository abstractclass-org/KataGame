// The game interface (game/gameinterface.h) against the front: each member of Enclosure's implementation
// (game/enclosure/gamerules.h, gameencoding.h) gives what Board, BoardHistory, NNInputs and SymmetryHelpers give
// today, the code that the front will call it from.
//
//   testgameinterface tests/data/enclosure_games.txt
//
// It replays the fixture's games (as testgamelayer does) and random games with timeout passes and komi. A second
// position, played through GameRules::play only, follows each game. At each position:
// - the rules part: the player to move and the edges left (against the 1-2-2 turn order), terminal, the legal
//   actions, is_legal, the scores, the hash (with the player's hash, Board::pos_hash), the pieces (Board::colors),
//   the area (Board::calculateArea), the move and the position in the notation (each read back), and at the end the
//   result;
// - the encoding part: the inputs (NNInputs::fillRowV8, and fillRowV8Packed at each symmetry), with and without the
//   history planes and at several komis, the input hash (the network cache's key), the policy layout (NNPos), the
//   image position and the action map (SymmetryHelpers), and the constants.

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

#include "../../core/rand.h"
#include "../../game/boardhistory.h"
#include "../../game/chosengame.h"
#include "../../game/enclosure/position_check.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

using R = GameRules;
using E = GameEncoding;

static_assert(R::NUM_ACTIONS == NNPos::MAX_NN_POLICY_SIZE, "one policy entry per action");
static_assert(E::NUM_SPATIAL_INPUTS + 1 == NNInputs::NUM_FEATURES_SPATIAL_V8, "the on-board plane, then the game's");
static_assert(E::NUM_GLOBAL_INPUTS == NNInputs::NUM_FEATURES_GLOBAL_V8, "the game's global inputs are V8's");
static_assert(E::POLICY_CHANNELS == NNPos::NUM_OFFSETS, "the policy's channels per point");
static_assert(R::MAX_BOARD_LEN == Board::MAX_LEN, "the build's largest board");

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
  E::spatial_inputs(p, options, [&](int x, int y, int plane) { set((y * 19 + x) * planes + 1 + plane); });
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

//An owner from pieces() or area() as Board's color.
static Color colorOf(int8_t owner) { return owner < 0 ? C_EMPTY : Board::playerOfIndex(owner); }

//Whether the owners of every point equal Board's colors there.
static bool sameColors(const int8_t* owner, const Color* colors) {
  for(int y = 0; y < 19; y++)
    for(int x = 0; x < 19; x++)
      if(colorOf(owner[y * 19 + x]) != colors[Location::getLoc(x, y, 19)])
        return false;
  return true;
}

//The whole state that the members see: the rules' hash and the inputs' hash.
static Hash128 fullHash(const R::Position& p) { return R::hash(p) ^ E::input_hash(p); }

//replayed: the same game, played through GameRules::play only. expectedMoves: the moves played, or -1 if unknown.
static void checkPosition(const Board& board, const BoardHistory& hist, const R::Position& replayed,
                          const string& where, int expectedMoves) {
  const int index = checkedPositions++;
  const R::Position& p = board.pos;
  const Player pla = board.nextPla();

  //The rules part.
  if(R::next_player(p) != Board::indexOfPlayer(pla))
    fail(where + ": next_player differs from Board::nextPla");
  if(expectedMoves >= 0) {
    int wantPlayer, wantLeft;
    enclosure::position_check::expected_turn(expectedMoves, wantPlayer, wantLeft);
    if(R::moves_played(p) != expectedMoves || R::next_player(p) != wantPlayer || R::turn_moves_left(p) != wantLeft)
      fail(where + ": moves_played, next_player or turn_moves_left is not the 1-2-2 turn order's");
  }
  if(R::terminal(p) != board.isGameOver())
    fail(where + ": terminal differs from Board::isGameOver");
  if((R::moves_played(p) == 0) != board.isEmpty())
    fail(where + ": moves_played is not 0 exactly at the start");
  vector<Loc> legalLocs;
  board.getLegalMoveLocs(legalLocs);
  vector<int> legal, frontLegal;
  R::legal_actions(p, legal);
  for(Loc loc : legalLocs)
    frontLegal.push_back(Board::actionOfMoveLoc(loc));
  if(legal != frontLegal)
    fail(where + ": legal_actions differs from Board::getLegalMoveLocs");
  for(int a : legal)
    if(!R::is_legal(p, a) || !board.isLegal(Board::moveLocOfAction(a), pla, false))
      fail(where + ": is_legal rejects a listed action");
  if(index % 200 == 0) {
    allActionChecks++;
    set<int> listed(legal.begin(), legal.end());
    for(int a = -1; a <= R::NUM_ACTIONS; a++)
      if(R::is_legal(p, a) != (listed.count(a) > 0))
        fail(where + ": is_legal differs from legal_actions at action " + Global::intToString(a));
  }
  if((R::hash(p) ^ Board::ZOBRIST_PLAYER_HASH[pla]) != board.pos_hash)
    fail(where + ": hash with the player's hash differs from Board::pos_hash");
  if(R::hash(replayed) != R::hash(p))
    fail(where + ": the position played through GameRules::play differs from Board's");
  for(int k = 0; k < 2; k++)
    if(R::score(p, k) != board.getScore(Board::playerOfIndex(k)))
      fail(where + ": score differs from Board::getScore");
  int8_t owner[19 * 19];
  R::pieces(p, owner);
  if(!sameColors(owner, board.colors))
    fail(where + ": pieces differ from Board::colors");
  if(index % 3 == 0) {
    Color area[Board::MAX_ARR_SIZE];
    board.calculateArea(area, false, false, false, false);
    R::area(p, owner);
    if(!sameColors(owner, area))
      fail(where + ": area differs from Board::calculateArea");
  }
  for(int a : legal) {
    string why;
    if(R::parse_action(p, R::action_text(a), &why) != a)
      fail(where + ": action_text does not read back (" + R::action_text(a) + ": " + why + ")");
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
    for(int history = 0; history < 2; history++) {
      MiscNNInputParams params;
      if(!history)
        params.maxHistory = 0;
      GameInterface::InputOptions options;
      options.history = history != 0;
      options.komi = hist.rules.komi;
      NNInputs::fillRowV8(board, hist, pla, params, 19, 19, true, bin.data(), glob);
      interfaceRow(p, options, bin2.data(), glob2);
      if(bin != bin2)
        fail(where + ": spatial_inputs differ from NNInputs::fillRowV8");
      for(int i = 0; i < NNInputs::NUM_FEATURES_GLOBAL_V8; i++)
        if(glob[i] != glob2[i])
          fail(where + ": global_inputs differ from NNInputs::fillRowV8");
    }
    if(hist.rules.komi == 0.0f &&
       BoardHistory::getSituationRulesAndKoHash(board, hist, pla, 0.5) !=
         (BoardHistory::getSituationAndSimpleKoHash(board, pla) ^ E::input_hash(p)))
      fail(where + ": input_hash is not what the network cache's key adds");
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

  //The symmetries: the image position, the action map, and the image's inputs.
  if(index % 10 == 0) {
    static vector<uint32_t> bits(NNInputs::NUM_SPATIAL_WORDS_V8), bits2(NNInputs::NUM_SPATIAL_WORDS_V8);
    for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++) {
      symmetryChecks++;
      const R::Position image = E::sym_position(p, s);
      if(fullHash(image) != fullHash(SymmetryHelpers::getSymPosition(p, s)))
        fail(where + ": sym_position differs from SymmetryHelpers::getSymPosition");
      //The images of the legal actions are the image position's legal actions, listed by the rules.
      const E::SymActionMap actionMap(p, s);
      const SymmetryHelpers::SymActionMap frontMap(board, s);
      vector<int> mapped, imageLegal;
      for(int a : legal) {
        mapped.push_back(actionMap(a));
        if(actionMap(a) != frontMap(a))
          fail(where + ": SymActionMap differs from SymmetryHelpers::SymActionMap");
      }
      sort(mapped.begin(), mapped.end());
      R::legal_actions(image, imageLegal);
      if(mapped != imageLegal)
        fail(where + ": the action map's images are not the image position's legal actions");
      MiscNNInputParams params;
      GameInterface::InputOptions options;
      options.symmetry = s;
      options.komi = hist.rules.komi;
      NNInputs::fillRowV8Packed(board, hist, pla, params, 19, 19, true, bits.data(), glob, s);
      interfaceBits(p, options, bits2.data());
      if(bits != bits2)
        fail(where + ": spatial_inputs differ from NNInputs::fillRowV8Packed at symmetry " + Global::intToString(s));
    }
  }
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testgameinterface GAMES_FILE\n");
    return 2;
  }
  Board::initHash();

  //The constants that the front holds today.
  if(E::SCORE_SCALE != NNPos::SCORE_SCALE)
    fail("SCORE_SCALE differs from NNPos::SCORE_SCALE");
  if(E::SCORE_UNIT != Board::scoreUnit())
    fail("SCORE_UNIT differs from Board::scoreUnit");
  if(E::GAME_LENGTH / Board::GO_GAME_LENGTH != Board::TIME_SCALE)
    fail("GAME_LENGTH differs from the game length in Board::TIME_SCALE");
  if(E::SYMMETRIES != (1u << SymmetryHelpers::NUM_SYMMETRIES) - 1)
    fail("SYMMETRIES differs from SymmetryHelpers::NUM_SYMMETRIES");
  {
    R::Position start, other;
    Board board;
    if(!R::new_game(19, 19, start) || (R::hash(start) ^ Board::ZOBRIST_PLAYER_HASH[P_BLACK]) != board.pos_hash)
      fail("new_game(19, 19) is not Board's start");
    if(R::new_game(9, 9, other) || R::new_game(19, 18, other))
      fail("new_game accepts a board other than 19 x 19");
    if(R::x_size(start) != board.x_size || R::y_size(start) != board.y_size)
      fail("x_size or y_size differs from Board's");
  }

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
    R::Position replayed;
    R::new_game(19, 19, replayed);
    //A pass (the player to move has no legal edge) forfeits the rest of the turn: the turn order then shifts.
    bool passed = false;
    for(int i = 0; i < n; i++) {
      getline(in, line);
      istringstream row(line);
      int action;
      row >> action;
      checkPosition(board, hist, replayed, label + " move " + Global::intToString(i + 1), passed ? -1 : i);
      hist.makeBoardMoveAssumeLegal(board, Board::moveLocOfAction(action), board.nextPla(), NULL);
      R::play(replayed, action);
      passed = passed || action == R::PASS_ACTION;
    }
    checkPosition(board, hist, replayed, label + " end", passed ? -1 : n);
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
      R::Position replayed;
      R::new_game(19, 19, replayed);
      int ply = 0;
      while(!board.isGameOver()) {
        checkPosition(board, hist, replayed,
                      "random game " + Global::intToString(g) + " ply " + Global::intToString(ply), -1);
        vector<int> legal;
        R::legal_actions(board.pos, legal);
        const Player pla = board.nextPla();
        if(rand.nextBool(0.05)) {
          hist.makeBoardMoveTolerant(board, Board::PASS_LOC, pla);
          R::play(replayed, R::PASS_ACTION);
        }
        else {
          const int a = legal[rand.nextUInt((uint32_t)legal.size())];
          hist.makeBoardMoveAssumeLegal(board, Board::moveLocOfAction(a), pla, NULL);
          R::play(replayed, a);
        }
        ply++;
      }
      checkPosition(board, hist, replayed, "random game " + Global::intToString(g) + " end", -1);
      const double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK);
      if(R::result(board.pos) != (margin < 0 ? 1.f : margin > 0 ? -1.f : 0.f))
        fail("random game " + Global::intToString(g) + ": result is not the sign of the scores");
    }
  }

  printf("%d fixture games and %d random games: %d positions, %d with the inputs checked, %d symmetry checks, "
         "%d with every action checked, %d failures\n",
         games, randomGames, checkedPositions, inputChecks, symmetryChecks, allActionChecks, failures);
  return failures == 0 ? 0 : 1;
}
