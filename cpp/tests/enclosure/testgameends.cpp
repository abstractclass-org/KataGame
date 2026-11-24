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
// 2. The evaluator: the no-result output is kept for a game with draws as a class. In every other game it is 0 under
//    the default rules (positional ko) and kept under a simple ko, as KataGo does. The three outputs sum to 1.
// 3. The search: the value of each terminal child of the positions before the end (win-loss and no result), and that a
//    draw is worth half a win.
// 4. Self-play: a game's value targets (the last one is the result) and its rows (C0-2, C20, C27, C52 and C62), for games
//    that end by their rules and for games at the move cap, with RGSC's outcome of each (Rgsc::gameOutcome).
// 5. The targets of one ending (GameInterface::endingClasses), in the same rows: which rows hold the score's targets
//    (C27), the game's own (C33, C34, C68, C69 and the spatial slots) and the ending (G76, G77), for each kind of
//    ending, at the cap, in side positions and in reanalyzed rows without the game's outcome. A game without ending
//    classes keeps KataGo's C27. The test game has them, and the test computes each of its targets from the game's
//    boards.
//
// Prints each failure and the counts of the endings. Exits 1 on a failure.

#include <cmath>
#include <cstdio>
#include <functional>
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
static constexpr bool ENDINGS = GameInterface::endingClasses<GameEncoding>() > 0;
static const bool TEST_GAME = string(GameRules::NAME) == "endings";

//KataGo's rule: White's minus Black's score with komi, and the winner by its sign.
static double marginOf(const Board& board, double komi) {
  return board.getScore(P_WHITE) - board.getScore(P_BLACK) + komi;
}
static Player winnerOfMargin(double margin) { return margin > 0 ? P_WHITE : margin < 0 ? P_BLACK : C_EMPTY; }
//White's return of a winner.
static double returnOf(Player winner) { return winner == P_WHITE ? 1.0 : winner == P_BLACK ? -1.0 : 0.0; }

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
  e.margin = marginOf(board, komi);
  e.winner = winnerOfMargin(e.margin);
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
    e.winner = winnerOfMargin(e.margin);
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
  if(!e.scored)
    name += winnerOfMargin(e.margin) == e.winner ? " as the score" : " against the score";
  return name;
}

//The ends that a check saw: how many of each name, and the kinds.
struct Ends {
  map<string, int> counts;
  set<int> kinds;
  void add(const Expected& e) {
    counts[endName(e)]++;
    kinds.insert(e.kind);
  }
};

static void checkHistory(int games, Ends& ends) {
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
        const double margin = marginOf(board, rules.komi);
        check(copy.isGameFinished && copy.isScored && !copy.isNoResult && copy.winner == winnerOfMargin(margin) &&
              copy.rulesEnd.kind == -1 && std::fabs(copy.finalWhiteMinusBlackScore - margin) < 1e-4,
              game + ": stopped after " + to_string(hist.moveHistory.size()) + " moves, not KataGo's rule");
        stopped++;
      }
      const Loc loc = randomMove(board, rand);
      hist.makeBoardMoveAssumeLegal(board, loc, board.nextPla(), NULL);
    }
    const Expected e = expectedEnd(board, rules.komi);
    ends.add(e);
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

//The records of ends that the score did not decide, each with a margin: no score, whatever score the caller gives
//(GTP's printsgf), also for a draw of a game whose draws are no class of their own, which no game here has.
static void checkUnscoredRecords() {
  BoardHistory hist;
  hist.isGameFinished = true;
  hist.finalWhiteMinusBlackScore = 3.0f;
  for(const Player winner : {C_EMPTY, P_WHITE, P_BLACK}) {
    hist.winner = winner;
    const string expected = winner == P_WHITE ? "W+" : winner == P_BLACK ? "B+" : "0";
    for(const double score : {std::nan(""), 3.0, -3.0, 0.0})
      check(WriteSgf::gameResultNoSgfTag(hist, score) == expected,
            "the record of an unscored end, " + expected + ", with the score " + to_string(score));
  }
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
    for(const int koRule : {Rules::KO_POSITIONAL, Rules::KO_SIMPLE}) {
      BoardHistory h(hist);
      h.rules.koRule = koRule;
      const bool kept = DRAW_CLASS || koRule == Rules::KO_SIMPLE;
      NNResultBuf buf;
      MiscNNInputParams params;
      nnEval->evaluate(board, h, board.nextPla(), params, buf, true, false);
      const NNOutput& out = *buf.result;
      const string where = "the evaluation after " + to_string(i) + " moves, ko rule " + to_string(koRule);
      check(std::fabs(out.whiteWinProb + out.whiteLossProb + out.whiteNoResultProb - 1.0) < 1e-5,
            where + ": the value outputs do not sum to 1");
      check(kept ? out.whiteNoResultProb > 0.0f : out.whiteNoResultProb == 0.0f,
            where + ": the no-result output is " + to_string(out.whiteNoResultProb));
      if(koRule == Rules::KO_POSITIONAL)
        largest = std::max(largest, (double)out.whiteNoResultProb);
    }
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
static void checkSearch(NNEvaluator* nnEval, Logger& logger, int games, Ends& ends) {
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
      const double winLoss = returnOf(e.winner);
      const double noResult = DRAW_CLASS && e.winner == C_EMPTY ? 1.0 : 0.0;
      check(std::fabs(values.winLossValue - winLoss) < 1e-6 && std::fabs(values.noResultValue - noResult) < 1e-6,
            where + ": the value " + to_string(values.winLossValue) + ", no result " + to_string(values.noResultValue));
      //A draw is worth half a win, as KataGo's split gives: the utility is the win-loss value's.
      check(std::fabs(values.utility - winLoss * params.winLossUtilityFactor) < 1e-6,
            where + ": the utility " + to_string(values.utility));
      ends.add(e);
    }
  }
}

struct RowCounts {
  int rows = 0, capRows = 0, drawRows = 0;
  int sideRows = 0, reanalyzedRows = 0;  //the rows that do not hold the game's outcome
  map<int, int> endings;                 //the rows of each ending's class
};

//The class of the test game's ending for the row's mover: its line, the opponent's line, its count, the opponent's
//count, a tie of the count, a draw by the corner.
static int expectedClass(const Expected& e, Player mover) {
  switch(e.kind) {
  case LINE: return e.winner == mover ? 0 : 1;
  case COUNT: return e.winner == mover ? 2 : 3;
  case COUNT_TIE: return 4;
  default: return 5;
  }
}

//A row's score distribution: its bins sum to 100, and their mean is the score, 0 without a score's target.
static void checkScoreDistribution(const TrainingWriteBuffers& b, int64_t r, double score, const string& row) {
  const int64_t len = b.scoreDistrN.shape[1];
  const int8_t* s = b.scoreDistrN.data + r * len;
  int sum = 0;
  double mean = 0.0;
  for(int64_t i = 0; i < len; i++) {
    sum += s[i];
    mean += s[i] * ((double)(i - len / 2) + 0.5) * NNPos::SCORE_SCALE / 100.0;
  }
  check(sum == 100, row + ": the score distribution sums to " + to_string(sum));
  check(std::fabs(mean - score) < 1e-6, row + ": the score distribution's mean " + to_string(mean));
}

//The test game's targets of a row of a game, from the game's boards: boards[turn] is the row's, and the last one is the
//game's end, or its position at the cap.
static void checkEndingTargets(const TrainingWriteBuffers& b, int64_t r, const FinishedGameData& data,
                               const vector<Board>& boards, int turn, Player mover, RowCounts& counts,
                               const string& where) {
  const float* c = b.globalTargetsNC.data + r * b.globalTargetsNC.shape[1];
  const bool ended = !data.hitTurnLimit;
  const Expected e = expectedEnd(boards.back(), data.endHist.rules.komi);
  const bool line = ended && e.kind == LINE, count = ended && (e.kind == COUNT || e.kind == COUNT_TIE);
  const bool reply = turn + 2 < (int)boards.size();
  const double score = count ? (mover == P_WHITE ? e.margin : -e.margin) : 0.0;
  const string row = where + (ended ? ", of a game that ended by " + endName(e) : ", of a game at the cap");
  check(c[27] == (count ? 1.0f : 0.0f), row + ": C27 " + to_string(c[27]));
  check(std::fabs(c[20] - score) < 1e-6, row + ": C20 " + to_string(c[20]));
  checkScoreDistribution(b, r, score, row);
  check(c[34] == (line ? 1.0f : 0.0f), row + ": C34 " + to_string(c[34]));
  check(c[33] == 1.0f, row + ": C33 " + to_string(c[33]));
  check(c[68] == (ended ? 1.0f : 0.0f), row + ": C68 " + to_string(c[68]));
  check(c[69] == (reply ? 1.0f : 0.0f), row + ": C69 " + to_string(c[69]));
  const int endingClass = expectedClass(e, mover);
  check(c[76] == (ended ? (float)endingClass : 0.0f), row + ": G76 " + to_string(c[76]));
  check(c[77] == (ended ? 1.0f : 0.0f), row + ": G77 " + to_string(c[77]));
  if(ended)
    counts.endings[endingClass]++;

  const int n = boards.back().x_size, area = n * n;
  const int8_t* v = b.valueTargetsNCHW.data + r * b.valueTargetsNCHW.shape[1] * area;
  auto side = [&](const Board& board, int x, int y) {
    const Color color = board.colors[Location::getLoc(x, y, n)];
    return color == C_EMPTY ? 0 : color == mover ? 1 : -1;
  };
  const int last = (int)boards.size() - 1, winnerSide = e.winner == mover ? 1 : -1;
  vector<string> bad(8);
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++) {
      int inRow = 0, inColumn = 0;  //the line winner's pieces in the point's row and column
      for(int j = 0; line && j < n; j++) {
        inRow += side(boards.back(), j, y) == winnerSide;
        inColumn += side(boards.back(), x, j) == winnerSide;
      }
      const int now = side(boards[turn], x, y), end = side(boards.back(), x, y);
      const int expected[8] = {
        count ? 120 * end : 0,
        !ended ? 0 : now != 0 ? 2 : end,
        side(boards[std::min(turn + 2, last)], x, y),
        side(boards[std::min(turn + 4, last)], x, y),
        inRow == n || inColumn == n ? 120 * end : 0,
        reply && side(boards[turn + 1], x, y) != now ? 120 : 0,
        reply && side(boards[turn + 2], x, y) != side(boards[turn + 1], x, y) ? 120 : 0,
        0};
      for(int i = 0; i < 8; i++)
        if(v[i * area + y * n + x] != expected[i])
          bad[i] = to_string(v[i * area + y * n + x]) + " in place of " + to_string(expected[i]) + " at " +
            to_string(x) + "," + to_string(y);
    }
  for(int i = 0; i < 8; i++)
    check(bad[i].empty(), row + ": the slot C" + to_string(i) + " holds " + bad[i]);
}

//The rows that do not hold the game's outcome: a side position's, and a reanalyzed one's that leaves it out. They hold
//no target but the policy's and the values'.
static void checkRowsWithoutOutcome(const TrainingWriteBuffers& b, int& rows, const string& where) {
  for(int64_t r = 0; r < b.curRows; r++) {
    const float* c = b.globalTargetsNC.data + r * b.globalTargetsNC.shape[1];
    const string row = where + ", row " + to_string(r);
    for(const int i : {20, 27, 33, 34, 69, 76, 77})
      check(c[i] == 0.0f, row + ": C" + to_string(i) + " " + to_string(c[i]));
    //KataGo's flag of the pass-alive mode, which is on in these games (checkSelfplay). C27 cancels it, but not in a
    //game with ending classes, where C68 is C1's flag alone.
    check(c[68] == (ENDINGS ? 0.0f : 1.0f), row + ": C68 " + to_string(c[68]));
    checkScoreDistribution(b, r, 0.0, row);
    const int64_t size = b.valueTargetsNCHW.shape[1] * b.valueTargetsNCHW.shape[2] * b.valueTargetsNCHW.shape[3];
    bool empty = true;
    for(int64_t i = 0; i < size; i++)
      empty = empty && b.valueTargetsNCHW.data[r * size + i] == 0;
    check(empty, row + ": a spatial target");
    rows++;
  }
}

//A game's rows: the final result (C0-2, in the mover's view), the score target and its weight (C20, C27), and the flags
//of the move cap and of a finished game (C52, C62).
static void checkRows(const TrainingWriteBuffers& b, const FinishedGameData& data, const ValueTargets& last,
                      RowCounts& counts, const string& where) {
  const int64_t n = b.globalTargetsNC.shape[1];
  const bool capped = data.hitTurnLimit;
  //The mover's side of each turn of the game, and its boards from its first row on.
  vector<Player> movers;
  for(const Move& m : data.endHist.moveHistory)
    movers.push_back(m.pla);
  const int startTurn = (int)data.startHist.moveHistory.size();
  vector<Board> boards = {data.startBoard};
  for(size_t t = startTurn; t < movers.size(); t++) {
    boards.push_back(boards.back());
    boards.back().playMoveAssumeLegal(data.endHist.moveHistory[t].loc, movers[t]);
  }
  for(int64_t r = 0; r < b.curRows; r++) {
    const float* c = b.globalTargetsNC.data + r * n;
    const int turn = (int)c[51];
    const bool white = movers.at(turn) == P_WHITE;
    const string row = where + ", the row of turn " + to_string(turn);
    check(c[0] == (white ? last.win : last.loss) && c[1] == (white ? last.loss : last.win) && c[2] == last.noResult,
          row + ": the result " + to_string(c[0]) + ", " + to_string(c[1]) + ", " + to_string(c[2]));
    check(std::fabs(c[0] + c[1] + c[2] - 1.0f) < 1e-6f, row + ": the result does not sum to 1");
    const bool noResult = data.endHist.isNoResult;
    if(TEST_GAME)
      checkEndingTargets(b, r, data, boards, turn - startTurn, movers.at(turn), counts, row);
    else {
      check(c[27] == (noResult ? 0.0f : 1.0f), row + ": C27 " + to_string(c[27]));
      check(c[20] == (noResult ? 0.0f : white ? last.score : -last.score), row + ": C20 " + to_string(c[20]));
    }
    check(c[52] == (capped ? 1.0f : 0.0f) && c[62] == (capped ? 0.0f : 1.0f), row + ": C52 and C62");
    check(c[70] == 0.0f && c[71] == 0.0f, row + ": a regret target without RGSC");
    counts.rows++;
    counts.capRows += capped;
    counts.drawRows += noResult;
  }
}

static void checkSelfplay(NNEvaluator* nnEval, Logger& logger, int games, Ends& ends) {
  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "test";
  botSpec.nnEval = nnEval;
  botSpec.baseParams = testParams(6);
  botSpec.baseParams.chosenMoveTemperature = 1.0;
  botSpec.baseParams.chosenMoveTemperatureEarly = 1.0;
  botSpec.baseParams.rootNoiseEnabled = true;
  botSpec.baseParams.alwaysComputePassAliveUnderSuicideRules = enabled_t::True;  //KataGo's C68
  PlaySettings playSettings;
  playSettings.forSelfPlay = true;
  playSettings.sidePositionProb = 0.2;
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
      margin = marginOf(end, rules.komi);
      winner = winnerOfMargin(margin);
    }
    else {
      const Board& end = data->endHist.getRecentBoard(0);
      const Expected e = expectedEnd(end, rules.komi);
      check(!data->hitTurnLimit, where + ": the game hit the cap");
      checkEnd(data->endHist, e, where);
      ends.add(e);
      winner = e.winner;
      margin = e.margin;
      const optional<double> outcome = Rgsc::gameOutcome(data->endHist, data->hitTurnLimit);
      check(outcome.has_value() && *outcome == returnOf(winner), where + ": RGSC's outcome");
    }
    const bool draw = DRAW_CLASS && !stops && winner == C_EMPTY;
    check(last.noResult == (draw ? 1.0f : 0.0f), where + ": the no-result target " + to_string(last.noResult));
    check(last.win == (draw ? 0.0f : (float)ScoreValue::whiteWinsOfWinner(winner, 0.5)),
          where + ": the win target " + to_string(last.win));
    check(last.loss == (draw ? 0.0f : 1.0f - last.win), where + ": the loss target " + to_string(last.loss));
    if(draw)
      check(last.score == 0.0f, where + ": the score target of a draw");
    else if(winner != C_EMPTY)
      check(std::fabs(last.score - margin) < 1e-4, where + ": the score target " + to_string(last.score));
    auto write = [&](const function<void(const TrainingWriteBuffers&)>& checkWritten) {
      ostream nowhere(nullptr);  //the rows are read in memory (beforeWrite)
      TrainingDataWriter writer(&nowhere, inputsVersion, 4096, 1.0, len, len, 1, "testgameends writer");
      writer.beforeWrite = checkWritten;
      writer.writeGame(*data);
      writer.flushIfNonempty();
    };
    //The game's own rows, its side positions' alone, and its own rows again as reanalyzed without the outcome.
    vector<SidePosition*> sidePositions;
    sidePositions.swap(data->sidePositions);
    write([&](const TrainingWriteBuffers& b) { checkRows(b, *data, last, counts, where); });
    const vector<float> weights = data->targetWeightByTurn;
    data->targetWeightByTurn.assign(weights.size(), 0.0f);
    data->sidePositions.swap(sidePositions);
    write([&](const TrainingWriteBuffers& b) {
      checkRowsWithoutOutcome(b, counts.sideRows, where + ", a side position");
    });
    data->sidePositions.swap(sidePositions);
    data->targetWeightByTurn = weights;
    ReanalysisData reanalyzed;
    reanalyzed.wasReanalyzed = true;
    reanalyzed.usedOutcomeTargets = false;
    data->reanalysisByTurn.assign(weights.size(), reanalyzed);
    write([&](const TrainingWriteBuffers& b) {
      checkRowsWithoutOutcome(b, counts.reanalyzedRows, where + ", reanalyzed");
    });
    data->sidePositions.swap(sidePositions);
    delete data;
  }
  check(counts.rows > 0 && counts.capRows > 0, "the rows: none, or none of a game at the cap");
  check(counts.reanalyzedRows > 0, "the rows: none reanalyzed");
  if(TEST_GAME)
    check(counts.drawRows > 0 && counts.sideRows > 0 && counts.endings.size() == 6,
          "the rows: none of a draw, none of a side position, or not every ending's class");
  printf("self-play: %d games, %d at the cap; %d rows, %d of them at the cap, %d of a draw as no result\n", games, capped,
         counts.rows, counts.capRows, counts.drawRows);
  printf("  the rows without the game's outcome: %d of side positions, %d reanalyzed\n", counts.sideRows,
         counts.reanalyzedRows);
  if(!counts.endings.empty()) {
    printf("  the rows' endings:");
    for(const auto& [endingClass, n] : counts.endings)
      printf(" class %d %d;", endingClass, n);
    printf("\n");
  }
}

static void printEnds(const char* where, const Ends& ends) {
  printf("  %s:", where);
  for(const auto& [name, count] : ends.counts)
    printf(" %s %d;", name.c_str(), count);
  printf("\n");
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  printf("%s: the rules' end %s, draws as a class %s, ending classes %d\n", GameRules::NAME, OWN_END ? "yes" : "no",
         DRAW_CLASS ? "yes" : "no", GameInterface::endingClasses<GameEncoding>());
  check(!TEST_GAME || (OWN_END && DRAW_CLASS && ENDINGS), "the test game has the members");

  Ends historyEnds, searchEnds, selfplayEnds;
  checkHistory(TEST_GAME ? 4000 : 20, historyEnds);
  printEnds("the games' ends", historyEnds);
  if(TEST_GAME)
    for(const char* name : {"kind 0 white as the score", "kind 0 white against the score", "kind 0 black as the score",
                            "kind 0 black against the score", "kind 1 white", "kind 1 black", "kind 2 draw",
                            "kind 3 draw as the score", "kind 3 draw against the score"})
      check(historyEnds.counts.count(name) > 0, string("no game of the history's ended by ") + name);

  checkUnscoredRecords();

  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = newEvaluator(logger);
  checkEvaluator(nnEval);
  checkSearch(nnEval, logger, TEST_GAME ? 200 : 4, searchEnds);
  printEnds("the search's terminal children", searchEnds);
  checkSelfplay(nnEval, logger, TEST_GAME ? 200 : 5, selfplayEnds);
  printEnds("self-play's ends", selfplayEnds);
  if(TEST_GAME)
    for(const Ends* ends : {&searchEnds, &selfplayEnds})
      check(ends->kinds.size() == 4,
            "the search or self-play saw " + to_string(ends->kinds.size()) + " of the 4 kinds of ending");
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  printf("%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
