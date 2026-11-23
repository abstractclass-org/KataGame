// KataGame fork (abscls-shfz.7): the end of a game, through the front, for the build's game: the rules' end
// (GameInterface::gameEnd) and draws as a class of their own (GameInterface::drawsAsNoResult). It names no game of the
// game layer, so it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with TESTS=testgameends):
//
//   testgameends
//
// A game without the members keeps KataGo's rule: the winner is the sign of the margin with komi, and a draw is a scored
// game without a winner, half a win. The test game endings (tests/games/endings, -DGAME=endings) has both members and
// every kind of ending. There the test computes each end from the board's pieces alone (expectedEnd).
//
// 1. The history (BoardHistory) at the end of random games, at several komis: the winner, the final score, isScored,
//    isNoResult, the kind of ending and the SGF record's result, also after a second call of endAndScoreGameNow and in a
//    history that starts at the end. And the same game stopped at each position before its end, as at the move cap:
//    KataGo's rule, no kind, and never a no result.
// 2. The evaluator: the no-result output is kept for a game with draws as a class, and 0 in every other game, under the
//    default rules (positional ko: KataGo keeps the output under a simple ko). The three outputs sum to 1.
// 3. The search: the value of each terminal child of the positions before the end (win-loss and no result), and that a
//    draw is worth half a win.
// 4. Self-play: a game's value targets (the last one is the result) and its rows (C0-2, C20, C27, C52 and C62), for games
//    that end by their rules and for games at the move cap, with RGSC's outcome of each (Rgsc::gameOutcome).
//
// Prints each failure and the counts of the endings. Exits 1 on a failure.

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../dataio/sgf.h"
#include "../../dataio/trainingwrite.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/modelversion.h"
#include "../../neuralnet/nneval.h"
#include "../../neuralnet/nninputs.h"
#include "../../program/play.h"
#include "../../program/rgsc.h"
#include "../../search/search.h"
#include "../../search/searchnode.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

static constexpr bool OWN_END = GameInterface::hasGameEnd<GameRules>();
static constexpr bool DRAW_CLASS = GameInterface::drawsAsNoResult<GameEncoding>();
static const bool TEST_GAME = string(GameRules::NAME) == "endings";

//An end as the test expects it.
struct Expected {
  Player winner = C_EMPTY;
  double margin = 0.0;  //White's minus Black's, with komi
  int kind = -1;
  bool scored = true;
};

//The kinds of the test game's endings (tests/games/endings/gamerules.h).
enum { LINE = 0, COUNT = 1, COUNT_TIE = 2, EMPTY_CORNER = 3 };

//The end of the test game from the board's pieces: a full row or column, then the count of the upper rows on a full
//board, then the empty corner. In another game KataGo's rule.
static Expected expectedEnd(const Board& board, double komi) {
  Expected e;
  e.margin = board.getScore(P_WHITE) - board.getScore(P_BLACK) + komi;
  e.winner = e.margin > 0 ? P_WHITE : e.margin < 0 ? P_BLACK : C_EMPTY;
  if(!TEST_GAME)
    return e;
  const int n = board.x_size;
  auto at = [&](int x, int y) { return board.colors[Location::getLoc(x, y, n)]; };
  int pieces = 0;
  double upper[2] = {0, 0};
  Player line = C_EMPTY;
  for(int i = 0; i < n; i++) {
    int row[2] = {0, 0}, column[2] = {0, 0};
    for(int j = 0; j < n; j++) {
      if(at(j, i) != C_EMPTY)
        row[at(j, i) == P_WHITE]++;
      if(at(i, j) != C_EMPTY)
        column[at(i, j) == P_WHITE]++;
    }
    pieces += row[0] + row[1];
    if(i >= n / 2)
      for(int c = 0; c < 2; c++)
        upper[c] += row[c];
    for(int c = 0; c < 2; c++)
      if(row[c] == n || column[c] == n)
        line = c == 1 ? P_WHITE : P_BLACK;
  }
  e.margin = upper[1] - upper[0] + komi;
  if(line != C_EMPTY) {
    e.winner = line;
    e.kind = LINE;
    e.scored = false;
  }
  else if(pieces == n * n) {
    e.winner = e.margin > 0 ? P_WHITE : e.margin < 0 ? P_BLACK : C_EMPTY;
    e.kind = e.winner == C_EMPTY ? COUNT_TIE : COUNT;
  }
  else {
    e.winner = C_EMPTY;
    e.kind = EMPTY_CORNER;
    e.scored = false;
  }
  return e;
}

static void checkEnd(const BoardHistory& hist, const Expected& e, const string& where) {
  const bool noResult = DRAW_CLASS && e.winner == C_EMPTY;
  check(hist.isGameFinished && !hist.isResignation, where + ": the game is not over");
  check(hist.winner == e.winner, where + ": the winner is " + PlayerIO::playerToString(hist.winner) + ", not " +
        PlayerIO::playerToString(e.winner));
  check(hist.isNoResult == noResult, where + ": isNoResult");
  check(hist.isScored == (e.scored && !noResult), where + ": isScored");
  const double score = noResult ? 0.0 : e.margin;
  check(std::fabs(hist.finalWhiteMinusBlackScore - score) < 1e-4, where + ": the final score is " +
        to_string(hist.finalWhiteMinusBlackScore) + ", not " + to_string(score));
  check(hist.rulesEnd.kind == e.kind, where + ": the kind of ending is " + to_string(hist.rulesEnd.kind) + ", not " +
        to_string(e.kind));
  if(OWN_END || DRAW_CLASS)
    check(hist.rulesEnd.scored == e.scored, where + ": the rules' scored");
  //The record: a draw "0", a win that the score did not decide without a score.
  const string text = WriteSgf::gameResultNoSgfTag(hist);
  const string letter = e.winner == P_WHITE ? "W+" : "B+";
  const string expectedText = e.winner == C_EMPTY ? string("0")
    : letter + (e.scored ? Global::doubleToString(e.winner == P_WHITE ? hist.finalWhiteMinusBlackScore
                                                                     : -hist.finalWhiteMinusBlackScore) : string());
  check(text == expectedText, where + ": the record's result is " + text + ", not " + expectedText);
  //An override of the score (GTP's printsgf) changes the score of a scored win only.
  if(e.winner != C_EMPTY)
    check(WriteSgf::gameResultNoSgfTag(hist, e.winner == P_WHITE ? 7.0 : -7.0) == letter + (e.scored ? "7" : ""),
          where + ": the record's result with another score");
  else if(noResult)
    check(WriteSgf::gameResultNoSgfTag(hist, 7.0) == "0", where + ": the record of a draw with another score");
}

static bool sameEnd(const BoardHistory& a, const BoardHistory& b) {
  return a.isGameFinished == b.isGameFinished && a.winner == b.winner && a.isNoResult == b.isNoResult &&
    a.isScored == b.isScored && a.finalWhiteMinusBlackScore == b.finalWhiteMinusBlackScore &&
    a.rulesEnd.kind == b.rulesEnd.kind && a.rulesEnd.scored == b.rulesEnd.scored &&
    a.rulesEnd.result == b.rulesEnd.result;
}

//A random legal move of the position.
static Loc randomMove(const Board& board, Rand& rand) {
  static vector<int> legal;
  GameRules::legal_actions(board.pos, legal);
  return Board::moveLocOfAction(legal[rand.nextUInt((uint32_t)legal.size())]);
}

//A name of an end for the counts: the kind, and for a line whether the score says another result.
static string endName(const Expected& e) {
  string name = "kind " + to_string(e.kind) + (e.winner == P_WHITE ? " white" : e.winner == P_BLACK ? " black" : " draw");
  if(!e.scored) {
    const Player byScore = e.margin > 0 ? P_WHITE : e.margin < 0 ? P_BLACK : C_EMPTY;
    name += byScore == e.winner ? " as the score" : " against the score";
  }
  return name;
}

static void checkHistory(int games, map<string, int>& ends) {
  Rand rand("testgameends history");
  const double komis[] = {0.0, 0.5, -0.5, 2.0, -2.0, 7.5, -7.5};
  int stopped = 0;
  for(int g = 0; g < games; g++) {
    Rules rules;
    rules.komi = (float)komis[g % 7];
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0);
    const string game = "game " + to_string(g) + " at komi " + to_string(rules.komi);
    while(!hist.isGameFinished) {
      check(hist.rulesEnd.kind == -1, game + ": a kind before the end");
      //Stopped here, as at the move cap: the sign of the margin, scored, no kind.
      {
        BoardHistory copy(hist);
        copy.endAndScoreGameNow(board);
        const double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK) + rules.komi;
        const Player bySign = margin > 0 ? P_WHITE : margin < 0 ? P_BLACK : C_EMPTY;
        check(copy.isGameFinished && copy.isScored && !copy.isNoResult && copy.winner == bySign &&
              copy.rulesEnd.kind == -1 && std::fabs(copy.finalWhiteMinusBlackScore - margin) < 1e-4,
              game + ": stopped after " + to_string(hist.moveHistory.size()) + " moves, not KataGo's rule");
        stopped++;
      }
      const Loc loc = randomMove(board, rand);
      hist.makeBoardMoveAssumeLegal(board, loc, board.nextPla(), NULL);
    }
    const Expected e = expectedEnd(board, rules.komi);
    ends[endName(e)]++;
    checkEnd(hist, e, game);
    BoardHistory again(hist);
    again.endAndScoreGameNow(board);
    check(sameEnd(again, hist), game + ": a second call of endAndScoreGameNow changes the end");
    BoardHistory atEnd(board, P_BLACK, rules, 0);
    check(sameEnd(atEnd, hist), game + ": a history that starts at the end has another end");
    again.clear(Board(), P_BLACK, rules, 0);
    check(!again.isGameFinished && !again.isNoResult && again.rulesEnd.kind == -1, game + ": clear keeps the end");
  }
  printf("the history: %d games, stopped at %d positions before their ends\n", games, stopped);
}

static NNEvaluator* newEvaluator(Logger& logger) {
  ConfigParser cfg;
  cfg.overrideKey("onnxProvider", "cpu");
  const int len = NNPos::MAX_BOARD_LEN;
  NNEvaluator* nnEval = new NNEvaluator(
    "/dev/null", "/dev/null", "", &logger, 16, len, len, false, true, 12, 8, true, "", enabled_t::False, 1,
    vector<int>{0}, "testgameends evaluator", false, 0, cfg);
  nnEval->spawnServerThreads();
  return nnEval;
}

static void checkEvaluator(NNEvaluator* nnEval) {
  Rand rand("testgameends evaluator positions");
  Board board;
  BoardHistory hist(board, P_BLACK, Rules(), 0);
  int evaluated = 0;
  double largest = 0.0;
  for(int i = 0; i < 10 && !hist.isGameFinished; i++) {
    NNResultBuf buf;
    MiscNNInputParams params;
    nnEval->evaluate(board, hist, board.nextPla(), params, buf, false, false);
    const NNOutput& out = *buf.result;
    const string where = "the evaluation after " + to_string(i) + " moves";
    check(std::fabs(out.whiteWinProb + out.whiteLossProb + out.whiteNoResultProb - 1.0) < 1e-5,
          where + ": the value outputs do not sum to 1");
    check(DRAW_CLASS ? out.whiteNoResultProb > 0.0f : out.whiteNoResultProb == 0.0f,
          where + ": the no-result output is " + to_string(out.whiteNoResultProb));
    largest = std::max(largest, (double)out.whiteNoResultProb);
    evaluated++;
    const Loc loc = randomMove(board, rand);
    hist.makeBoardMoveAssumeLegal(board, loc, board.nextPla(), NULL);
  }
  printf("the evaluator: %d positions, no-result output up to %.3f\n", evaluated, largest);
}

static SearchParams testParams(int64_t visits) {
  SearchParams params;
  params.maxVisits = visits;
  params.staticScoreUtilityFactor = 0.0;
  params.dynamicScoreUtilityFactor = 0.0;
  return params;
}

//The children that end the game, of the positions one move before the end of random games.
static void checkSearch(NNEvaluator* nnEval, Logger& logger, int games, map<string, int>& ends) {
  Rand rand("testgameends search");
  const SearchParams params = testParams(60);
  check(params.noResultUtilityForWhite == 0.0, "noResultUtilityForWhite is not 0 by default");
  for(int g = 0; g < games; g++) {
    Rules rules;
    rules.komi = g % 2 == 0 ? 0.0f : 0.5f;
    Board board, before;
    BoardHistory hist(board, P_BLACK, rules, 0), histBefore(hist);
    while(!hist.isGameFinished) {
      before = board;
      histBefore = hist;
      const Loc loc = randomMove(board, rand);
      hist.makeBoardMoveAssumeLegal(board, loc, board.nextPla(), NULL);
    }
    Search bot(params, nnEval, &logger, "testgameends search " + to_string(g));
    bot.setPosition(before.nextPla(), before, histBefore);
    bot.runWholeSearch(before.nextPla());
    ConstSearchNodeChildrenReference children = bot.rootNode->getChildren();
    const int numChildren = children.iterateAndCountChildren();
    for(int i = 0; i < numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        continue;
      const Loc loc = children[i].getMoveLocRelaxed();
      Board after(before);
      BoardHistory histAfter(histBefore);
      histAfter.makeBoardMoveAssumeLegal(after, loc, before.nextPla(), NULL);
      ReportedSearchValues values;
      if(!histAfter.isGameFinished || !bot.getNodeValues(child, values))
        continue;
      const Expected e = expectedEnd(after, rules.komi);
      const string where = "the search of game " + to_string(g) + ", a child with " + endName(e);
      const double winLoss = e.winner == P_WHITE ? 1.0 : e.winner == P_BLACK ? -1.0 : 0.0;
      const double noResult = DRAW_CLASS && e.winner == C_EMPTY ? 1.0 : 0.0;
      check(std::fabs(values.winLossValue - winLoss) < 1e-6 && std::fabs(values.noResultValue - noResult) < 1e-6,
            where + ": the value " + to_string(values.winLossValue) + ", no result " + to_string(values.noResultValue));
      //A draw is worth half a win, as KataGo's split gives: the utility is the win-loss value's.
      check(std::fabs(values.utility - winLoss * params.winLossUtilityFactor) < 1e-6,
            where + ": the utility " + to_string(values.utility));
      ends[endName(e)]++;
    }
  }
}

struct RowCounts {
  int rows = 0, capRows = 0, drawRows = 0;
};

//A game's rows: the final result (C0-2, in the mover's view), the score target and its weight (C20, C27), and the flags
//of the move cap and of a finished game (C52, C62).
static void checkRows(const TrainingWriteBuffers& b, const FinishedGameData& data, const ValueTargets& last,
                      RowCounts& counts, const string& where) {
  const int64_t n = b.globalTargetsNC.shape[1];
  const bool capped = data.hitTurnLimit;
  //The mover's side of each turn of the game.
  vector<Player> movers;
  for(const Move& m : data.endHist.moveHistory)
    movers.push_back(m.pla);
  for(int64_t r = 0; r < b.curRows; r++) {
    const float* c = b.globalTargetsNC.data + r * n;
    const int turn = (int)c[51];
    const bool white = movers.at(turn) == P_WHITE;
    const string row = where + ", the row of turn " + to_string(turn);
    check(c[0] == (white ? last.win : last.loss) && c[1] == (white ? last.loss : last.win) && c[2] == last.noResult,
          row + ": the result " + to_string(c[0]) + ", " + to_string(c[1]) + ", " + to_string(c[2]));
    check(std::fabs(c[0] + c[1] + c[2] - 1.0f) < 1e-6f, row + ": the result does not sum to 1");
    const bool noResult = data.endHist.isNoResult;
    check(c[27] == (noResult ? 0.0f : 1.0f), row + ": C27 " + to_string(c[27]));
    check(c[20] == (noResult ? 0.0f : white ? last.score : -last.score), row + ": C20 " + to_string(c[20]));
    check(c[52] == (capped ? 1.0f : 0.0f) && c[62] == (capped ? 0.0f : 1.0f), row + ": C52 and C62");
    check(c[70] == 0.0f && c[71] == 0.0f, row + ": a regret target without RGSC");
    counts.rows++;
    counts.capRows += capped;
    counts.drawRows += noResult;
  }
}

static void checkSelfplay(NNEvaluator* nnEval, Logger& logger, int games, map<string, int>& ends) {
  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "test";
  botSpec.nnEval = nnEval;
  botSpec.baseParams = testParams(6);
  botSpec.baseParams.chosenMoveTemperature = 1.0;
  botSpec.baseParams.chosenMoveTemperatureEarly = 1.0;
  botSpec.baseParams.rootNoiseEnabled = true;
  PlaySettings playSettings;
  playSettings.forSelfPlay = true;
  const int len = NNPos::MAX_BOARD_LEN;
  const int inputsVersion = NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  RowCounts counts;
  int capped = 0;
  for(int g = 0; g < games; g++) {
    Rules rules;
    rules.komi = g % 4 == 0 ? 0.5f : 0.0f;
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0);
    Rand rand("testgameends selfplay " + to_string(g));
    OtherGameProperties otherGameProps;
    const bool stops = g % 5 == 4;  //these games stop at a move cap of 5
    //The test game starts from random openings, so that its games reach each kind of ending.
    for(int i = TEST_GAME ? (stops ? g % 4 : g % 11) : 0; i > 0; i--) {
      Board nextBoard(board);
      BoardHistory nextHist(hist);
      nextHist.makeBoardMoveAssumeLegal(nextBoard, randomMove(board, rand), board.nextPla(), NULL);
      if(nextHist.isGameFinished)
        break;
      board = nextBoard;
      hist = nextHist;
    }
    FinishedGameData* data = Play::runGame(
      board, board.nextPla(), hist, ExtraBlackAndKomi(), botSpec, botSpec, "testgameends search " + to_string(g), false, true,
      logger, false, false, stops ? 5 : 1000, []() noexcept { return false; }, nullptr, playSettings, otherGameProps,
      rand, nullptr, nullptr);
    const string where = "self-play's game " + to_string(g) + (stops ? ", stopped at 5 moves" : "");
    const ValueTargets& last = data->whiteValueTargetsByTurn.back();
    Player winner;
    double margin;
    if(stops) {
      //The move cap: KataGo's rule for the value target, and no outcome for RGSC.
      capped++;
      check(data->hitTurnLimit && !data->endHist.isGameFinished && !data->endHist.isNoResult &&
            data->endHist.rulesEnd.kind == -1, where + ": the game did not stop at the cap");
      check(!Rgsc::gameOutcome(data->endHist, data->hitTurnLimit).has_value(), where + ": RGSC has an outcome");
      const Board& end = data->endHist.getRecentBoard(0);
      margin = end.getScore(P_WHITE) - end.getScore(P_BLACK) + rules.komi;
      winner = margin > 0 ? P_WHITE : margin < 0 ? P_BLACK : C_EMPTY;
    }
    else {
      const Board& end = data->endHist.getRecentBoard(0);
      const Expected e = expectedEnd(end, rules.komi);
      check(!data->hitTurnLimit, where + ": the game hit the cap");
      checkEnd(data->endHist, e, where);
      ends[endName(e)]++;
      winner = e.winner;
      margin = e.margin;
      const optional<double> outcome = Rgsc::gameOutcome(data->endHist, data->hitTurnLimit);
      check(outcome.has_value() && *outcome == (winner == P_WHITE ? 1.0 : winner == P_BLACK ? -1.0 : 0.0),
            where + ": RGSC's outcome");
    }
    const bool draw = DRAW_CLASS && !stops && winner == C_EMPTY;
    check(last.noResult == (draw ? 1.0f : 0.0f), where + ": the no-result target " + to_string(last.noResult));
    check(last.win == (draw ? 0.0f : winner == P_WHITE ? 1.0f : winner == P_BLACK ? 0.0f : 0.5f),
          where + ": the win target " + to_string(last.win));
    check(last.loss == (draw ? 0.0f : 1.0f - last.win), where + ": the loss target " + to_string(last.loss));
    if(draw)
      check(last.score == 0.0f, where + ": the score target of a draw");
    else if(winner != C_EMPTY)
      check(std::fabs(last.score - margin) < 1e-4, where + ": the score target " + to_string(last.score));
    ostream nowhere(nullptr);  //the rows are read in memory (beforeWrite)
    TrainingDataWriter writer(&nowhere, inputsVersion, 4096, 1.0, len, len, 1, "testgameends writer");
    writer.beforeWrite = [&](const TrainingWriteBuffers& b) { checkRows(b, *data, last, counts, where); };
    writer.writeGame(*data);
    writer.flushIfNonempty();
    delete data;
  }
  check(counts.rows > 0 && counts.capRows > 0, "the rows: none, or none of a game at the cap");
  if(TEST_GAME)
    check(counts.drawRows > 0, "the rows: none of a draw");
  printf("self-play: %d games, %d at the cap; %d rows, %d of them at the cap, %d of a draw as no result\n", games, capped,
         counts.rows, counts.capRows, counts.drawRows);
}

static void printEnds(const char* where, const map<string, int>& ends) {
  printf("  %s:", where);
  for(const auto& [name, count] : ends)
    printf(" %s %d;", name.c_str(), count);
  printf("\n");
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  printf("%s: the rules' end %s, draws as a class %s\n", GameRules::NAME, OWN_END ? "yes" : "no", DRAW_CLASS ? "yes" : "no");
  check(!TEST_GAME || (OWN_END && DRAW_CLASS), "the test game has both members");

  map<string, int> historyEnds, searchEnds, selfplayEnds;
  checkHistory(TEST_GAME ? 4000 : 20, historyEnds);
  printEnds("the games' ends", historyEnds);
  if(TEST_GAME)
    for(const char* name : {"kind 0 white as the score", "kind 0 white against the score", "kind 0 black as the score",
                            "kind 0 black against the score", "kind 1 white", "kind 1 black", "kind 2 draw",
                            "kind 3 draw as the score", "kind 3 draw against the score"})
      check(historyEnds.count(name) > 0, string("no game of the history's ended by ") + name);

  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = newEvaluator(logger);
  checkEvaluator(nnEval);
  checkSearch(nnEval, logger, TEST_GAME ? 200 : 4, searchEnds);
  printEnds("the search's terminal children", searchEnds);
  checkSelfplay(nnEval, logger, TEST_GAME ? 200 : 5, selfplayEnds);
  printEnds("self-play's ends", selfplayEnds);
  if(TEST_GAME)
    for(const auto* ends : {&searchEnds, &selfplayEnds}) {
      set<char> kinds;
      for(const auto& [name, count] : *ends)
        kinds.insert(name[5]);
      check(kinds.size() == 4, "the search or self-play saw " + to_string(kinds.size()) + " of the 4 kinds of ending");
    }
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  printf("%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
