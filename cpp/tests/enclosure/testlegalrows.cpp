// KataGame fork (abscls-shfz.9): the legal moves in the training rows (LEGAL_MOVES_IN_ROWS), for the build's game. It
// names no game of the game layer, so it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with
// TESTS=testlegalrows):
//
//   testlegalrows [DIR]
//
// A game without the member: the rows have no legal moves, and the file of rows no array of them. A game with it: each
// row of self-play's games holds the moves that the rules make legal in its position, and no other, a bit for each
// policy index. The rows are those of the game, of its side positions, and of a reanalyzed game. The test computes the
// legal moves from the board alone (expectedLegal), for the test games endings (4 x 4, a point a move) and slides
// (5 x 5 and 6 x 6 boards in the policy of 6 x 6, with an opening that places the opponent's pieces, 13 channels a
// point and several hundred slots) and for Tak (the same boards and kind of opening, 255 channels), from its position's
// text (tpsboard.h).
// The writer stops at a row with visits on a move that is not legal in its position, and at a row whose position has
// no legal move, with the row's name.
//
// With DIR, the rows of the games go there as a training file (.npz), for the trainer's loader and its test of the same
// rows (the rows of two sizes, with the array).
//
// Prints each failure and the counts. Exits 1 on a failure.

#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/fileutils.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../dataio/trainingwrite.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/modelversion.h"
#include "../../neuralnet/nneval.h"
#include "../../neuralnet/nninputs.h"
#include "../../program/play.h"
#include "../../search/search.h"
#include "tpsboard.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

static constexpr bool HOLDS = TrainingWriteBuffers::HAS_LEGAL_MOVES;
static const string GAME = GameRules::NAME;
//A game of two sizes of board in the largest one's policy, with an opening of 2 plies that places pieces alone.
static const bool TWO_SIZES = GAME == "slides" || GAME == "tak";
static constexpr int SLOTS = NNPos::POLICY_CHANNELS * NNPos::MAX_BOARD_AREA + 1;
static constexpr int LEN = NNPos::MAX_BOARD_LEN;
static const int DX[4] = {0, 0, 1, -1}, DY[4] = {1, -1, 0, 0};

//The legal moves of a position as policy indices, for a test game, from the board's colors and the turn, which the
//side positions and the game's positions give, and for Tak from its position's text. An independent copy of the rules.
static set<int> expectedLegal(const Board& board, int turn) {
  set<int> legal;
  if(GAME == "tak") {
    const Tps::Board tps(GameRules::position_text(board.pos));
    check(tps.ply == turn, "the turn of a position is its ply");
    return tps.legalMoves(LEN);
  }
  const int n = board.x_size;
  auto at = [&](int x, int y) { return board.colors[Location::getLoc(x, y, n)]; };
  const Player mover = board.nextPla();
  for(int y = 0; y < n; y++) {
    for(int x = 0; x < n; x++) {
      const int point = y * LEN + x;
      if(GAME == "endings") {
        if(at(x, y) == C_EMPTY)
          legal.insert(point);
      }
      else if(GAME == "slides") {
        if(at(x, y) == C_EMPTY)
          legal.insert(point * NNPos::POLICY_CHANNELS);
        else if(turn >= 2 && at(x, y) == mover) {
          for(int d = 0; d < 4; d++) {
            for(int k = 1; k <= 3; k++) {
              const int px = x + DX[d] * k, py = y + DY[d] * k;
              if(px < 0 || py < 0 || px >= n || py >= n || at(px, py) != C_EMPTY)
                break;
              legal.insert(point * NNPos::POLICY_CHANNELS + 1 + 4 * (k - 1) + d);
            }
          }
        }
      }
    }
  }
  return legal;
}

//A row's mask, as the policy indices that it has.
static set<int> maskOf(const TrainingWriteBuffers& b, int64_t r) {
  set<int> mask;
  const uint8_t* bits = b.legalMovesNMovePacked.data + r * b.packedPolicyLen;
  check(b.packedPolicyLen == (SLOTS + 7) / 8, "the bytes of a row's legal moves: " + to_string(b.packedPolicyLen));
  for(int i = 0; i < b.packedPolicyLen * 8; i++)
    if(bits[i >> 3] & (0x80 >> (i & 7))) {
      check(i < SLOTS, "a bit after the policy's slots: " + to_string(i));
      mask.insert(i);
    }
  return mask;
}

struct Counts {
  int rows = 0, sideRows = 0, reanalyzedRows = 0, openingRows = 0, smallRows = 0, games = 0;
  int64_t moves = 0;
};

//A row's mask and its visited moves against the position's legal moves.
static void checkRow(const TrainingWriteBuffers& b, int64_t r, const Board& board, int turn, Counts& counts,
                     const string& where) {
  const set<int> mask = maskOf(b, r);
  const set<int> legal = expectedLegal(board, turn);
  check(!mask.empty() && mask == legal, where + ": the mask has " + to_string(mask.size()) + " moves, the rules " +
        to_string(legal.size()));
  const int64_t n = b.policyTargetsNCMove.shape[2];
  for(int64_t a = 0; a < n; a++)
    check(b.policyTargetsNCMove.data[r * 2 * n + a] == 0 || mask.count((int)a) > 0,
          where + ": visits on the move " + to_string(a) + ", which is not in the mask");
  if(board.x_size < LEN) {
    counts.smallRows++;
    for(const int a : mask)
      check((a / NNPos::POLICY_CHANNELS) % LEN < board.x_size && (a / NNPos::POLICY_CHANNELS) / LEN < board.y_size,
            where + ": a move off the smaller board");
  }
  if(TWO_SIZES && turn < 2) {
    counts.openingRows++;
    for(const int a : mask)
      check(a % NNPos::POLICY_CHANNELS == 0, where + ": a move in the opening that is not the first channel's");
  }
  counts.moves += (int64_t)mask.size();
}

static NNEvaluator* newEvaluator(Logger& logger) {
  ConfigParser cfg;
  cfg.overrideKey("onnxProvider", "cpu");
  const int len = NNPos::MAX_BOARD_LEN;
  NNEvaluator* nnEval = new NNEvaluator(
    "/dev/null", "/dev/null", "", &logger, 16, len, len, false, true, 12, 8, true, "", enabled_t::False, 1,
    vector<int>{0}, "testlegalrows evaluator", false, 0, cfg);
  nnEval->spawnServerThreads();
  return nnEval;
}

static MatchPairer::BotSpec botSpecOf(NNEvaluator* nnEval) {
  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "test";
  botSpec.nnEval = nnEval;
  botSpec.baseParams.maxVisits = 8;
  botSpec.baseParams.staticScoreUtilityFactor = 0.0;
  botSpec.baseParams.dynamicScoreUtilityFactor = 0.0;
  botSpec.baseParams.chosenMoveTemperature = 1.0;
  botSpec.baseParams.chosenMoveTemperatureEarly = 1.0;
  botSpec.baseParams.rootNoiseEnabled = true;
  return botSpec;
}

//A game from self-play, with side positions, on a board of the size (0: the game's own).
static FinishedGameData* playGame(NNEvaluator* nnEval, Logger& logger, int size, const string& seed) {
  MatchPairer::BotSpec botSpec = botSpecOf(nnEval);
  PlaySettings playSettings;
  playSettings.forSelfPlay = true;
  playSettings.sidePositionProb = 0.3;
  Rules rules;
  Board board;
  if(size > 0)
    board = Board(size, size);
  BoardHistory hist(board, P_BLACK, rules, 0);
  Rand rand("testlegalrows " + seed);
  OtherGameProperties otherGameProps;
  return Play::runGame(
    board, board.nextPla(), hist, ExtraBlackAndKomi(), botSpec, botSpec, "testlegalrows search " + seed, false, true,
    logger, false, false, 1000, []() noexcept { return false; }, nullptr, playSettings, otherGameProps, rand, nullptr,
    nullptr);
}

//Writes the game's rows with the data's weights, and checks them.
template<class Check>
static void writeRows(const FinishedGameData& data, const Check& checkWritten) {
  ostream nowhere(nullptr);  //the rows are read in memory (beforeWrite)
  TrainingDataWriter writer(&nowhere, NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion), 4096, 1.0,
                            LEN, LEN, 1, "testlegalrows writer");
  writer.beforeWrite = checkWritten;
  writer.writeGame(data);
  writer.flushIfNonempty();
}

//The error of a write that must stop: its text, or "" for no error.
template<class Write>
static string errorOf(const Write& write) {
  try {
    write();
  }
  catch(const StringError& e) {
    return e.what();
  }
  return "";
}

//A move of the position that the rules do not make legal: a piece on a point with a piece, and for endings a move of a
//point with a piece. It is a valid policy index.
static Loc illegalMove(const Board& board) {
  for(int y = 0; y < board.y_size; y++)
    for(int x = 0; x < board.x_size; x++)
      if(board.colors[Location::getLoc(x, y, board.x_size)] != C_EMPTY)
        return Board::moveLocOfAction((y * LEN + x) * NNPos::POLICY_CHANNELS);
  return Board::NULL_LOC;
}

static void checkStops(NNEvaluator* nnEval, Logger& logger, int size) {
  FinishedGameData* data = playGame(nnEval, logger, size, "stops " + to_string(size));
  data->targetWeightByTurn.assign(data->targetWeightByTurn.size(), 1.0f);
  auto write = [&]() { writeRows(*data, [](const TrainingWriteBuffers&) {}); };
  check(errorOf(write).empty(), "the good rows stop the writer");
  //A visited move that is not legal, in a game's row: the 5th, where there are pieces on the board.
  const int turn = 5;
  vector<Board> boards = {data->startBoard};
  for(const Move& move : data->endHist.moveHistory) {
    boards.push_back(boards.back());
    boards.back().playMoveAssumeLegal(move.loc, move.pla);
  }
  vector<PolicyTargetMove>& moves = *data->policyTargetsByTurn[turn].policyTargets;
  const PolicyTargetMove good = moves[0];
  moves[0] = PolicyTargetMove(illegalMove(boards[turn]), 10);
  string error = errorOf(write);
  const string gameName = data->gameHash.toString();
  check(error.find("The row of turn " + to_string(turn) + " of the game " + gameName) != string::npos &&
        error.find("not legal in its position") != string::npos, "the error of a row's illegal visits: " + error);
  moves[0] = good;
  check(errorOf(write).empty(), "the good rows stop the writer again");
  //A visited move that is not legal, in a side position's row.
  check(!data->sidePositions.empty(), "a side position to check");
  if(!data->sidePositions.empty()) {
    SidePosition* sp = data->sidePositions[0];
    const float weight = sp->targetWeight;
    sp->targetWeight = 1.0f;
    const vector<PolicyTargetMove> target = sp->policyTarget;
    sp->policyTarget = {PolicyTargetMove(illegalMove(sp->board), 10)};
    error = errorOf(write);
    check(error.find("The side position of turn ") != string::npos && error.find(gameName) != string::npos &&
          error.find("not legal in its position") != string::npos, "the error of a side position's illegal visits: " +
          error);
    sp->policyTarget = target;
    //A position without a legal move, which no searched position is.
    const Board goodBoard = sp->board;
    const BoardHistory goodHist = sp->hist;
    const Player goodPla = sp->pla;
    sp->board = data->endHist.getRecentBoard(0);
    sp->hist = data->endHist;
    sp->pla = sp->board.nextPla();
    sp->policyTarget.clear();
    error = errorOf(write);
    check(error.find("without legal moves") != string::npos, "the error of a position without legal moves: " + error);
    sp->board = goodBoard;
    sp->hist = goodHist;
    sp->pla = goodPla;
    sp->policyTarget = target;
    sp->targetWeight = weight;
  }
  delete data;
}

static void checkHeld(NNEvaluator* nnEval, Logger& logger, const string& outDir, Counts& counts) {
  TrainingDataWriter* files = nullptr;
  if(!outDir.empty())
    files = new TrainingDataWriter(outDir, NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion), 4096,
                                   1.0, LEN, LEN, "testlegalrows files");
  const vector<int> sizes = TWO_SIZES ? vector<int>{LEN, LEN - 1} : vector<int>{LEN};
  for(int g = 0; g < 40; g++) {
    const int size = sizes[g % sizes.size()];
    FinishedGameData* data = playGame(nnEval, logger, size, to_string(g));
    const string where = "game " + to_string(g) + " of " + to_string(size);
    counts.games++;
    if(files != nullptr)
      files->writeGame(*data);
    //The positions of the game's turns, with the colors of their boards.
    const int startTurn = (int)data->startHist.moveHistory.size();
    vector<Board> boards = {data->startBoard};
    for(size_t t = startTurn; t < data->endHist.moveHistory.size(); t++) {
      boards.push_back(boards.back());
      boards.back().playMoveAssumeLegal(data->endHist.moveHistory[t].loc, data->endHist.moveHistory[t].pla);
    }
    //The side positions' rows alone, each at weight 1, so that row r is the r-th side position's.
    const vector<float> weights = data->targetWeightByTurn;
    data->targetWeightByTurn.assign(weights.size(), 0.0f);
    for(SidePosition* sp : data->sidePositions)
      sp->targetWeight = 1.0f;
    writeRows(*data, [&](const TrainingWriteBuffers& b) {
      check(b.curRows == (int)data->sidePositions.size(), where + ": the side positions' rows");
      for(int64_t r = 0; r < b.curRows; r++) {
        const SidePosition* sp = data->sidePositions[r];
        const int turn = (int)sp->hist.getCurrentTurnNumber();
        checkRow(b, r, sp->board, turn, counts, where + ", side position " + to_string(r));
        counts.sideRows++;
      }
    });
    //The game's own rows, at weight 1 each, so that row r is turn r's, and again as reanalyzed without the outcome.
    data->targetWeightByTurn.assign(weights.size(), 1.0f);
    vector<SidePosition*> sidePositions;
    sidePositions.swap(data->sidePositions);
    auto checkGame = [&](const TrainingWriteBuffers& b, int& rows) {
      check(b.curRows == (int)weights.size(), where + ": the rows of its turns");
      for(int64_t r = 0; r < b.curRows; r++) {
        check((int)b.globalTargetsNC.data[r * b.globalTargetsNC.shape[1] + 51] == startTurn + (int)r, where +
              ": the turn of the row " + to_string(r));
        checkRow(b, r, boards[r], startTurn + (int)r, counts, where + ", the row of turn " + to_string(startTurn + r));
        rows++;
      }
    };
    writeRows(*data, [&](const TrainingWriteBuffers& b) { checkGame(b, counts.rows); });
    ReanalysisData reanalyzed;
    reanalyzed.wasReanalyzed = true;
    reanalyzed.usedOutcomeTargets = false;
    data->reanalysisByTurn.assign(weights.size(), reanalyzed);
    writeRows(*data, [&](const TrainingWriteBuffers& b) { checkGame(b, counts.reanalyzedRows); });
    data->reanalysisByTurn.clear();
    data->targetWeightByTurn = weights;
    data->sidePositions.swap(sidePositions);  //data owns them
    delete data;
  }
  check(counts.rows > 0 && counts.sideRows > 0 && counts.reanalyzedRows == counts.rows,
        "the rows: none of a game, of a side position, or of a reanalyzed game");
  if(TWO_SIZES)
    check(counts.openingRows > 0 && counts.smallRows > 0, "the rows: none of the opening, or of the smaller board");
  if(files != nullptr) {
    files->flushIfNonempty();
    printf("wrote the rows of %d games to %s\n", counts.games, outDir.c_str());
    delete files;
  }
}

//The text of a game's rows, as the writer's debug output gives it.
static string textOf(const FinishedGameData& data) {
  ostringstream out;
  TrainingDataWriter writer(&out, NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion), 4096, 1.0, LEN,
                            LEN, 1, "testlegalrows text");
  writer.writeGame(data);
  writer.flushIfNonempty();
  return out.str();
}

int main(int argc, char** argv) {
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  printf("%s: the rows hold their legal moves: %s, %d slots\n", GAME.c_str(), HOLDS ? "yes" : "no", SLOTS);
  check((GAME != "endings" && !TWO_SIZES) || HOLDS, "the test games and Tak have the member");

  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = newEvaluator(logger);
  Counts counts;
  FinishedGameData* sample = playGame(nnEval, logger, HOLDS ? LEN : 0, "sample");
  sample->targetWeightByTurn.assign(sample->targetWeightByTurn.size(), 1.0f);
  const string text = textOf(*sample);
  check((text.find("legalMovesNMovePacked") != string::npos) == HOLDS, "the array's name in the rows' text");
  if(HOLDS) {
    checkHeld(nnEval, logger, argc > 1 ? argv[1] : "", counts);
    checkStops(nnEval, logger, LEN);
    if(TWO_SIZES)
      checkStops(nnEval, logger, LEN - 1);
    printf("%d games: %d rows, %d side positions, %d reanalyzed; of all three kinds, %d of the opening and %d on the "
           "smaller board; %.1f legal moves a row\n", counts.games, counts.rows, counts.sideRows, counts.reanalyzedRows,
           counts.openingRows, counts.smallRows, (double)counts.moves / std::max(1, counts.rows + counts.sideRows +
           counts.reanalyzedRows));
  }
  else {
    writeRows(*sample, [&](const TrainingWriteBuffers& b) {
      check(b.legalMovesNMovePacked.shape[0] == 1, "the array of a game without the member has one row");
    });
  }
  delete sample;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  printf("%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
