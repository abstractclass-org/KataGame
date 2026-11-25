// KataGame fork: komi through the front, for the build's game. The komi K is points added to White's final margin
// (BoardHistory::endAndScoreGameNow), and KataGo's even-komi search looks for the K that makes a position even.
//
//   testkomi [SELFPLAY_CONFIG]
//
// - The result: a random game to its end, then the history's result at komis of both signs, at 0, and at the komis
//   that decide the game (the half point nearest the opposite of the earned margin, and half a point to either side):
//   the final margin with K, its winner, White's wins in the value targets (a margin of 0 is half a win), the final
//   score target, each side's own komi, and the network cache's hash, which changes with a nonzero K only. In a game
//   whose rules give the end (GameInterface::gameEnd), the winner is the rules'. In a game whose draws are no results,
//   a draw has the final score 0 (testgameends checks both).
// - The inputs: at a position of that game, the front's global inputs (NNInputs::fillRowV8, which gives the game the
//   history's komi) are the game's global_inputs at that komi. The test prints the global inputs that change with K.
// - The even-komi search (PlayUtils::evenKomiOffset): the defaults with stub games, and the build's game's bracket,
//   rule and lead step (GameInterface::komiLeadStep) against the values pinned below. With KataGo's bracket (1 to 32
//   points) and KataGo's rule, the search tries the same offsets and gives the same result as KataGo's own code
//   (kataGoOffset, a copy) on many win-loss curves, and it always finds an even komi. With the game's bracket, it finds
//   the sign change of a monotone curve within half a point out to the bracket's limit, ends at the limit beyond it,
//   and the test prints the values that it reads. Beyond the limit, and on curves that keep their sign, the rule
//   decides: KataGo's takes the far end as the even komi after the bisection. The other finds no even komi, and stops
//   after the bracket's offsets.
// - The draws of the komi (play.cpp's GameInitializer), against the shares that the config gives them: each mean, each
//   spread, the whole komis, the evened games, each komi and the komis up to each, within the binomial interval that
//   leaves out 1e-7 on each side. Every komi is on the half-point grid.
//   - The keys of a size of board (komiMeans6 and the others), in configs made here of the two largest sizes of square
//     board that the build's game plays, or of its only one: the new games, the asymmetric games, the starts from a
//     position, the forks, the komi around an even komi, and the frozen starts. Each key in place of the general one,
//     with the other size at the general ones. A new game's random numbers are in KataGo's order, and a size's keys of
//     the general values give the general games, draw for draw. The boards that are not square, and the positions of a
//     size that the config does not have. The configs that cannot be meant stop at the load, with their key's name.
//   - With a self-play config: its new games, on each of its boards.

#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"
#include "../../program/play.h"
#include "../../program/playutils.h"
#include "../../search/searchparams.h"

using namespace std;

static int failures = 0;

static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

//A game that gives none of the even-komi search's members, and one that gives them all.
struct NoBracket {
  static constexpr double SCORE_SCALE = 12.0;
};
struct OwnBracket {
  static constexpr double SCORE_SCALE = 12.0;
  static constexpr double KOMI_BRACKET_STEP = 6.0;
  static constexpr double KOMI_BRACKET_LIMIT = 500.0;
  static constexpr double KOMI_LEAD_STEP = 0.0;
};
static_assert(GameInterface::komiBracketStep<NoBracket>() == 1.0 && GameInterface::komiBracketLimit<NoBracket>() == 32.0 &&
                GameInterface::komiBracketEndIsEven<NoBracket>() && GameInterface::komiLeadStep<NoBracket>() == 6.0,
              "KataGo's bracket and rule by default, and KataGo's lead step times SCORE_SCALE");
static_assert(GameInterface::komiBracketStep<OwnBracket>() == 6.0 &&
                GameInterface::komiBracketLimit<OwnBracket>() == 500.0 &&
                !GameInterface::komiBracketEndIsEven<OwnBracket>() && GameInterface::komiLeadStep<OwnBracket>() == 0.0,
              "a game's own bracket, with no even komi beyond it, and its own lead step");

//The even-komi search of each game of KataGame: its bracket's first offset and limit, whether the bracket's end counts
//as the even komi, and the lead's step. Blueprints' scores reach thousands of points, and its margins have no parity.
//The other games, and the games of the tests, give none of these, so they keep KataGo's bracket and rule, and KataGo's
//lead step times SCORE_SCALE.
static bool pinnedSearch(double& step, double& limit, bool& endIsEven, double& leadStep) {
  if(strcmp(GameRules::NAME, "blueprints") == 0) {
    step = 20.0;
    limit = 3000.0;
    endIsEven = false;
    leadStep = 0.0;
    return true;
  }
  if(strcmp(GameRules::NAME, "enclosure") == 0 || strcmp(GameRules::NAME, "connect4") == 0 ||
     strcmp(GameRules::NAME, "endings") == 0 || strcmp(GameRules::NAME, "slides") == 0) {
    step = 1.0;
    limit = 32.0;
    endIsEven = true;
    leadStep = strcmp(GameRules::NAME, "enclosure") == 0 ? 6.0 : 0.5;
    return true;
  }
  return false;
}

//KataGo's last stage of the even-komi search, as playutils.cpp had it before the game's bracket (getNaiveEvenKomiHelper,
//from "Grow window outward"), with winLossAt in place of evalWinLoss.
static double kataGoOffset(const function<double(double)>& winLossAt) {
  double lowerDelta;
  double upperDelta;
  double lowerWinLoss;
  double upperWinLoss;
  {
    double winLossZero = winLossAt(0);
    if(winLossZero < 0) {
      lowerDelta = 0.0;
      lowerWinLoss = winLossZero;
      for(int i = 0; i<=5; i++) {
        upperDelta = round(pow(2.0,i));
        upperWinLoss = winLossAt(upperDelta);
        if(upperWinLoss >= 0)
          break;
      }
    }
    else {
      upperDelta = 0.0;
      upperWinLoss = winLossZero;
      for(int i = 0; i<=5; i++) {
        lowerDelta = -round(pow(2.0,i));
        lowerWinLoss = winLossAt(lowerDelta);
        if(lowerWinLoss <= 0)
          break;
      }
    }
  }
  while(upperDelta - lowerDelta > 0.50001) {
    double midDelta = 0.5 * (lowerDelta + upperDelta);
    double midWinLoss = winLossAt(midDelta);
    if(midWinLoss < 0) {
      lowerDelta = midDelta;
      lowerWinLoss = midWinLoss;
    }
    else {
      upperDelta = midDelta;
      upperWinLoss = midWinLoss;
    }
  }
  double finalDelta;
  if(lowerWinLoss >= upperWinLoss - 1e-30)
    finalDelta = 0.5 * (lowerDelta + upperDelta);
  else if(upperWinLoss <= 0)
    finalDelta = upperDelta;
  else if(lowerWinLoss >= 0)
    finalDelta = lowerDelta;
  else
    finalDelta = lowerDelta + (upperDelta - lowerDelta) * (0-lowerWinLoss) / (upperWinLoss-lowerWinLoss);
  return finalDelta;
}

//A move of a random game that often scores: of 20 random legal moves, the one after which the mover's margin is the
//largest. NULL_LOC when no move is legal.
static Loc scoringMove(const Board& board, Player pla, Rand& rand) {
  vector<Loc> legal;
  board.getLegalMoveLocs(legal);
  Loc best = Board::NULL_LOC;
  double bestMargin = 0.0;
  for(int i = 0; i < 20 && !legal.empty(); i++) {
    const Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
    Board after(board);
    after.playMoveAssumeLegal(loc, pla);
    const double margin = after.getScore(pla) - after.getScore(getOpp(pla));
    if(best == Board::NULL_LOC || margin > bestMargin) {
      best = loc;
      bestMargin = margin;
    }
  }
  return best;
}

//The result, the score target and the cache's hash at each komi, on the end of a random game. Then the front's global
//inputs at a position of that game.
static void checkResults() {
  Board board;
  Player pla = P_BLACK;
  BoardHistory hist(board, pla, Rules(), 0);
  Rand rand("testkomi game");
  Board middle;
  BoardHistory middleHist(hist);
  while(!hist.isGameFinished) {
    const Loc loc = scoringMove(board, pla, rand);
    if(loc == Board::NULL_LOC) {
      check(false, "the random game has a legal move until its end");
      return;
    }
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = hist.presumedNextMovePla;
    if(hist.moveHistory.size() == 20 && !hist.isGameFinished) {
      middle = board;
      middleHist = hist;
    }
  }
  const double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK);
  printf("a random game of %d moves: White's earned margin %g\n", (int)hist.moveHistory.size(), margin);
  const Hash128 hashAtZero = BoardHistory::getSituationRulesAndKoHash(board, hist, pla, 0.5);
  //The komis that decide the game: the half point nearest the opposite of the earned margin (a draw when the margin is
  //on the half-point grid, as Blueprints' whole points are), and half a point to either side. A komi is a half point.
  const double even = 0.5 * round(-2.0 * margin);
  const double komis[] = {0.0, 12.5, -12.5, 3000.0, -3000.0, even, even + 0.5, even - 0.5};
  for(double komi : komis) {
    BoardHistory h(hist);
    h.setKomi((float)komi);
    h.endAndScoreGameNow(board);
    const GameInterface::GameEnd end = GameInterface::gameEnd<GameRules>(board.pos, komi);
    const Player winner = end.result < 0 ? P_WHITE : end.result > 0 ? P_BLACK : C_EMPTY;
    const bool noResult = GameInterface::drawsAsNoResult<GameEncoding>() && winner == C_EMPTY;
    const double d = noResult ? 0.0 : margin + komi;
    const string at = "komi " + to_string(komi) + ": ";
    if(!GameInterface::hasGameEnd<GameRules>())
      check(winner == (d > 0 ? P_WHITE : d < 0 ? P_BLACK : C_EMPTY),
            at + "the winner is the sign of the margin with komi");
    check(h.winner == winner, at + "the winner is the rules'");
    check(h.isNoResult == noResult, at + "a no result is a draw of a game whose draws are no results");
    check(h.finalWhiteMinusBlackScore == (float)d, at + "the final margin counts komi");
    const double wins = ScoreValue::whiteWinsOfWinner(h.winner, 0.5);
    check(wins == (winner == P_WHITE ? 1.0 : winner == P_BLACK ? 0.0 : 0.5),
          at + "White's wins: 1, 0, or half a win in a draw");
    check(ScoreValue::whiteScoreDrawAdjust(h.finalWhiteMinusBlackScore, 0.5, h) == (float)d,
          at + "the final score target is the margin with komi");
    check(h.currentSelfKomi(P_WHITE, 0.5) == (float)komi && h.currentSelfKomi(P_BLACK, 0.5) == -(float)komi,
          at + "each side's own komi");
    check((BoardHistory::getSituationRulesAndKoHash(board, h, pla, 0.5) == hashAtZero) == (komi == 0.0),
          at + "the cache's hash changes with a nonzero komi only");
    printf("komi %9.1f: White's margin %9.1f, winner %s, White's wins %.1f\n", komi, (double)h.finalWhiteMinusBlackScore,
           winner == P_WHITE ? "White" : winner == P_BLACK ? "Black" : "none (a draw)", wins);
  }

  //The front's global inputs at a position of the game: the game's own, at the history's komi.
  const Player mover = middle.nextPla();
  vector<float> bin(NNInputs::NUM_FEATURES_SPATIAL_V8 * NNPos::MAX_BOARD_AREA);
  vector<float> glob(NNInputs::NUM_FEATURES_GLOBAL_V8), own(NNInputs::NUM_FEATURES_GLOBAL_V8);
  vector<float> atZero(NNInputs::NUM_FEATURES_GLOBAL_V8);
  for(double komi : {0.0, 12.5, -12.5, 3000.0, -3000.0}) {
    BoardHistory h(middleHist);
    h.setKomi((float)komi);
    NNInputs::fillRowV8(middle, h, mover, MiscNNInputParams(), NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, false,
                        bin.data(), glob.data());
    GameInterface::InputOptions options;
    options.komi = komi;
    GameEncoding::global_inputs(middle.pos, options, own.data());
    check(glob == own, "komi " + to_string(komi) + ": the front's global inputs are the game's at the history's komi");
    if(komi == 0.0)
      atZero = glob;
    string changed;
    for(int i = 0; i < NNInputs::NUM_FEATURES_GLOBAL_V8; i++)
      if(glob[i] != atZero[i])
        changed += " " + to_string(i) + " (" + to_string(atZero[i]) + " -> " + to_string(glob[i]) + ")";
    if(komi != 0.0)
      printf("after move 20, %s to move, komi %.1f: the global inputs that change:%s\n",
             mover == P_WHITE ? "White" : "Black", komi, changed.empty() ? " none" : changed.c_str());
  }
}

//The bracket's offsets among the values that the search read (tried): after the value at 0, step, then each the double
//of the last, or the limit.
static int bracketOffsets(const vector<double>& tried, double step, double limit) {
  int offsets = 0;
  while(offsets + 1 < (int)tried.size() && fabs(tried[offsets + 1]) == min(step * pow(2.0, offsets), limit))
    offsets++;
  return offsets;
}

//The even-komi search: KataGo's bracket against KataGo's code, the game's bracket on monotone curves, and the two
//rules where the win-loss value keeps its sign up to the limit.
static void checkBracket() {
  double step = 0.0, limit = 0.0, leadStep = -1.0;
  bool endIsEven = false;
  check(pinnedSearch(step, limit, endIsEven, leadStep), "the build's game has a pinned even-komi search");
  const double gameStep = GameInterface::komiBracketStep<GameEncoding>();
  const double gameLimit = GameInterface::komiBracketLimit<GameEncoding>();
  const bool gameEndIsEven = GameInterface::komiBracketEndIsEven<GameEncoding>();
  const double gameLeadStep = GameInterface::komiLeadStep<GameEncoding>();
  check(gameStep == step && gameLimit == limit && gameEndIsEven == endIsEven && gameLeadStep == leadStep,
        "the game's bracket, its rule and its lead step");
  printf("game %s: the even-komi search's bracket, offsets from %g up to %g points; beyond it, %s; the lead's step %g "
         "points\n", GameRules::NAME, gameStep, gameLimit,
         gameEndIsEven ? "the bracket's end is the even komi (KataGo's rule)" : "no even komi", gameLeadStep);

  //KataGo's bracket and rule: on curves that cross 0 at c with a width s, some with a wobble that crosses them more
  //than once, and on flat ones, the same offsets in the same order and the same result as KataGo's code.
  int curves = 0;
  vector<function<double(double)>> shapes;
  for(double c = -40.0; c <= 40.0; c += 0.25)
    for(double s : {0.3, 3.0, 30.0})
      for(double wobble : {0.0, 0.3})
        shapes.push_back([=](double d) noexcept { return tanh((d - c) / s) + wobble * sin(7.0 * d); });
  for(double flat : {-0.5, 0.0, 0.5})
    shapes.push_back([=](double) noexcept { return flat; });
  for(const auto& f : shapes) {
    vector<double> tried, kataGoTried;
    double offset;
    const bool found =
      PlayUtils::evenKomiOffset([&](double d) { tried.push_back(d); return f(d); }, 1.0, 32.0, true, offset);
    const double kataGo = kataGoOffset([&](double d) { kataGoTried.push_back(d); return f(d); });
    check(found && tried == kataGoTried && offset == kataGo,
          "KataGo's bracket and rule: an even komi, with the offsets and the result of KataGo's code");
    curves++;
  }
  printf("KataGo's bracket (1 to 32 points) and rule: the same offsets and results as KataGo's code on %d win-loss "
         "curves, each an even komi\n", curves);

  //The game's bracket: a monotone curve that crosses 0 at c, as wide as the bracket's limit, so that it does not go
  //flat between the bracket's offsets.
  const double width = gameLimit / 6.0;
  vector<double> crossings = {0.25, -5.5, 37.25, 299.5, -1234.5, 2999.5, -2999.75, 3500.0, -3500.0};
  for(double& c : crossings)
    c *= gameLimit / 3000.0;
  for(double c : crossings) {
    vector<double> tried;
    const auto f = [=](double d) { return tanh((d - c) / width); };
    double offset;
    const bool found = PlayUtils::evenKomiOffset(
      [&](double d) { tried.push_back(d); return f(d); }, gameStep, gameLimit, gameEndIsEven, offset
    );
    bool grid = true;
    for(double d : tried)
      grid = grid && Rules::komiIsIntOrHalfInt((float)d) && fabs(d) <= gameLimit;
    check(grid, "the game's bracket: each offset is on the half-point grid, within the limit");
    const int offsets = bracketOffsets(tried, gameStep, gameLimit);
    if(fabs(c) <= gameLimit)
      check(found && fabs(offset - c) <= 0.5,
            "the game's bracket: an even komi at the sign change, within half a point, at " + to_string(c));
    else
      check(offset == (c > 0 ? gameLimit : -gameLimit), "the game's bracket: the limit, beyond it, at " + to_string(c));
    printf("a crossing at %8.2f: %2d values, %2d of them the bracket's offsets (the last %g), the result %8.2f (%s)\n",
           c, (int)tried.size(), offsets, tried[offsets], offset, found ? "an even komi" : "no even komi");
  }

  //The rules where the value keeps its sign up to the limit, on KataGo's bracket and on the game's: flat curves, and
  //a monotone one that crosses 0 beyond the limit on either side. KataGo's rule: the far end after the bisection (a
  //flat curve's last window of half a point, which KataGo averages), an even komi. The check of KataGo's bracket above
  //compares such curves with KataGo's code. The other rule: the far end at once, after the value at 0 and the
  //bracket's offsets, and no even komi. A curve flat at 0 has an even komi within the first offset, and a curve that
  //crosses 0 at the limit has one there.
  for(const pair<double, double>& bracket : {make_pair(1.0, 32.0), make_pair(gameStep, gameLimit)}) {
    const double s = bracket.first, l = bracket.second;
    vector<pair<string, function<double(double)>>> keeping = {
      {"flat at -0.5", [](double) noexcept { return -0.5; }},
      {"flat at 0.5", [](double) noexcept { return 0.5; }},
      {"crossing beyond +limit", [=](double d) noexcept { return tanh((d - 1.25 * l) / l); }},
      {"crossing beyond -limit", [=](double d) noexcept { return tanh((d + 1.25 * l) / l); }},
    };
    for(const auto& named : keeping) {
      const auto& f = named.second;
      const double end = f(0.0) < 0 ? l : -l;
      for(bool rule : {true, false}) {
        vector<double> tried;
        double offset;
        const bool found =
          PlayUtils::evenKomiOffset([&](double d) { tried.push_back(d); return f(d); }, s, l, rule, offset);
        const string what = "bracket " + to_string(s) + " to " + to_string(l) + ", " + named.first + ", " +
                            (rule ? "KataGo's rule" : "no even komi beyond the limit");
        if(rule)
          check(found && fabs(offset - end) <= 0.5, what + ": an even komi, the far end");
        else {
          check(!found && offset == end, what + ": no even komi, the far end");
          check((int)tried.size() == 1 + bracketOffsets(tried, s, l) && fabs(tried.back()) == l,
                what + ": the value at 0 and the bracket's offsets only");
        }
      }
    }
    double offset;
    check(PlayUtils::evenKomiOffset([](double) noexcept { return 0.0; }, s, l, false, offset) && fabs(offset) <= s,
          "a curve flat at 0: an even komi within the first offset");
    check(PlayUtils::evenKomiOffset([=](double d) noexcept { return tanh((d - l) / l); }, s, l, false, offset) &&
            fabs(offset - l) <= 0.5,
          "a curve that crosses 0 at the limit: an even komi there");
  }
  printf("curves that keep their sign up to the limit, on KataGo's bracket and the game's: the far end under both "
         "rules, an even komi under KataGo's, and no even komi under the other, after the bracket's offsets\n");
}

//How the games on a board draw their komi, as a config gives it: play.cpp's KomiDraws, read and written here apart from
//the code under test. Without a mean, the games have the fair komi (komiAuto), and their first draw is around 0.
struct Draws {
  vector<double> means, shares;
  double stdev = 0.0, bigProb = 0.0, big = 10.0, biggerProb = 0.0, bigger = 30.0, integerProb = 1.0, forkProb = 0.0;

  bool fair() const { return means.empty(); }
  vector<double> bases() const { return fair() ? vector<double>(1, 0.0) : means; }
  vector<double> baseShares() const {
    vector<double> result = fair() ? vector<double>(1, 1.0) : shares;
    double sum = 0.0;
    for(double s : result)
      sum += s;
    for(double& s : result)
      s /= sum;
    return result;
  }
  //The sampler's spreads (chooseExtraBlackAndKomi) and their shares: the bigger draw overrides the big one, which
  //overrides komiStdev. Each spread is in the config's points times sqrt(komi area) / 19.
  vector<pair<double, double>> spreads(const Board& board) const {
    const double scale = sqrt(board.komiArea()) / 19.0;
    const double pBigger = bigger > 0 ? biggerProb : 0.0;
    const double pBig = big > 0 ? bigProb * (1.0 - pBigger) : 0.0;
    return {{stdev * scale, 1.0 - pBig - pBigger}, {big * scale, pBig}, {bigger * scale, pBigger}};
  }
};

//The shares of the komis that draws gives on a board around the bases, by twice the komi. The draw around a base is a
//Gaussian truncated at 3 standard deviations, rounded to one of its two nearest half points with the shares that keep
//its mean, and clipped at the rules' range. A whole komi moves half a point up or down in the games that do not allow
//it (setKomiWithNoise).
static map<int, double> expectedKomis(
  const Draws& draws, const Board& board, const vector<double>& bases, const vector<double>& baseShares
) {
  const int steps = 120000;
  static vector<double> z, weight;  //the truncated Gaussian, in steps
  if(z.empty()) {
    double sum = 0.0;
    for(int j = 0; j < steps; j++) {
      z.push_back(6.0 * (j + 0.5) / steps - 3.0);
      weight.push_back(exp(-0.5 * z[j] * z[j]));
      sum += weight[j];
    }
    for(double& w : weight)
      w /= sum;
  }
  const int range = 2 * (int)Rules::MAX_USER_KOMI;
  vector<double> rounded(2 * range + 1, 0.0);
  auto add = [&](double komi, double share) {
    const double lower = floor(2.0 * komi), up = 2.0 * komi - lower;
    rounded[range + max(-range, min(range, (int)lower))] += share * (1.0 - up);
    rounded[range + max(-range, min(range, (int)lower + 1))] += share * up;
  };
  for(size_t i = 0; i < bases.size(); i++) {
    for(const pair<double, double>& spread : draws.spreads(board)) {
      const double share = baseShares[i] * spread.second;
      if(share <= 0.0)
        continue;
      if(spread.first <= 0.0)
        add(bases[i], share);
      else
        for(int j = 0; j < steps; j++)
          add(bases[i] + spread.first * z[j], share * weight[j]);
    }
  }
  map<int, double> result;
  for(int k = -range; k <= range; k++) {
    const double share = rounded[range + k];
    if(share <= 0.0)
      continue;
    const double moved = k % 2 == 0 ? 1.0 - draws.integerProb : 0.0;
    if(moved < 1.0)
      result[k] += share * (1.0 - moved);
    if(moved > 0.0) {
      result[k - 1] += share * 0.5 * moved;
      result[k + 1] += share * 0.5 * moved;
    }
  }
  return result;
}

//The counts of n draws of probability p that at most 1e-7 of the draws are below, and at most 1e-7 above. Past the mode
//the shares of the counts fall faster than the geometric series of two of them, which bounds the shares of the rest.
static pair<int, int> binomialInterval(int n, double p) {
  if(p <= 0.0 || p >= 1.0)
    return p <= 0.0 ? make_pair(0, 0) : make_pair(n, n);
  const double tail = 1e-7, odds = p / (1.0 - p);
  const int mode = (int)min((double)n, floor((n + 1.0) * p));
  const double atMode = exp(lgamma(n + 1.0) - lgamma(mode + 1.0) - lgamma(n - mode + 1.0) + mode * log(p) +
                            (n - mode) * log1p(-p));
  int lo = mode, hi = mode;
  for(double share = atMode; hi < n; hi++) {
    share *= (double)(n - hi) / (hi + 1) * odds;  //of hi + 1
    const double next = (double)(n - hi - 1) / (hi + 2) * odds;
    if(next < 1.0 && share / (1.0 - next) <= tail)
      break;
  }
  for(double share = atMode; lo > 0; lo--) {
    share *= (double)lo / (n - lo + 1) / odds;  //of lo - 1
    const double next = (double)(lo - 1) / (n - lo + 2) / odds;
    if(next < 1.0 && share / (1.0 - next) <= tail)
      break;
  }
  return make_pair(lo, hi);
}

//Checks a count of n draws against its share. Returns its text, with the interval.
static string checkCount(int count, int n, double share, const string& what) {
  const pair<int, int> interval = binomialInterval(n, min(1.0, share));
  check(count >= interval.first && count <= interval.second,
        what + ": " + to_string(count) + " of " + to_string(n) + ", expected " + to_string(interval.first) + " to " +
          to_string(interval.second));
  return Global::strprintf("%d [%d, %d]", count, interval.first, interval.second);
}

struct Drawn {
  Board board;
  double komi;
  ExtraBlackAndKomi extra;
};
static Drawn drawGame(
  GameInitializer& gameInit, const InitialPosition* fork, const Sgf::PositionSample* start,
  const PlaySettings& settings = PlaySettings()
) {
  Drawn drawn;
  Player pla;
  BoardHistory hist;
  SearchParams params;  //Self-play's form, which also draws noResultStdev and drawRandRadius into the params.
  OtherGameProperties otherGameProps;
  gameInit.createGame(drawn.board, pla, hist, drawn.extra, params, fork, settings, otherGameProps, start,
                      BoardHistoryModes());
  drawn.komi = hist.rules.komi;
  return drawn;
}

static string boardText(int x, int y) { return to_string(x) + " x " + to_string(y); }

//The draws of the games on one board, counted, against what the config gives.
struct Counted {
  int n = 0, offGrid = 0, fairGames = 0, evenedGames = 0;
  double sum = 0.0, largest = 0.0;
  map<int, int> komis;  //by twice the komi
  map<float, int> means, stdevs;

  void add(const Drawn& d) {
    n++;
    offGrid += !Rules::komiIsIntOrHalfInt((float)d.komi);
    fairGames += d.extra.makeGameFairForEmptyBoard;
    evenedGames += d.extra.makeGameFair;
    sum += d.komi;
    largest = max(largest, fabs(d.komi));
    komis[(int)lround(2.0 * d.komi)]++;
    means[d.extra.komiMean]++;
    stdevs[d.extra.komiStdev]++;
  }

  //The means, the spreads, the whole komis and the evened games (makeGameFair, in a share evenedShare of the games).
  //Then each komi and the komis up to each: the single counts of a wide distribution are small, and the sums hold its
  //form. fair: the games that are not evened have the fair komi.
  void checkAgainst(
    const Draws& draws, const Board& board, const vector<double>& bases, const vector<double>& baseShares,
    double evenedShare, bool fair, const string& what
  ) const {
    check(n > 0 && offGrid == 0, what + ": games, each with a komi on the half-point grid");
    check(fairGames == (fair ? n - evenedGames : 0), what + ": the games with the fair komi");
    string text;

    map<float, double> meanShares;
    for(size_t i = 0; i < bases.size(); i++)
      meanShares[(float)bases[i]] += baseShares[i];
    for(const pair<const float, int>& m : means)
      check(meanShares.count(m.first) > 0, what + ": a mean of " + to_string(m.first));
    for(const pair<const float, double>& m : meanShares)
      text += Global::strprintf("; mean %g: ", m.first) + checkCount(
        means.count(m.first) ? means.at(m.first) : 0, n, m.second, what + ", mean " + to_string(m.first));

    //The sampler multiplies in single precision, so a count is of the spread nearest to it.
    map<float, double> stdevShares;
    for(const pair<double, double>& s : draws.spreads(board))
      stdevShares[(float)s.first] += s.second;
    map<float, int> stdevCounts;
    for(const pair<const float, int>& c : stdevs) {
      float nearest = stdevShares.begin()->first;
      for(const pair<const float, double>& s : stdevShares)
        nearest = fabs(s.first - c.first) < fabs(nearest - c.first) ? s.first : nearest;
      check(fabs(nearest - c.first) <= 1e-5f * (1.0f + nearest), what + ": a spread of " + to_string(c.first));
      stdevCounts[nearest] += c.second;
    }
    for(const pair<const float, double>& s : stdevShares)
      text += Global::strprintf("; spread %.4g: ", s.first) +
              checkCount(stdevCounts[s.first], n, s.second, what + ", spread " + to_string(s.first));

    const map<int, double> shares = expectedKomis(draws, board, bases, baseShares);
    int whole = 0, upTo = 0;
    double wholeShare = 0.0, shareUpTo = 0.0;
    for(const pair<const int, int>& k : komis) {
      whole += k.first % 2 == 0 ? k.second : 0;
      check(shares.count(k.first) > 0, what + ": a komi of " + to_string(0.5 * k.first));
    }
    for(const pair<const int, double>& s : shares) {
      const int count = komis.count(s.first) ? komis.at(s.first) : 0;
      upTo += count;
      shareUpTo += s.second;
      wholeShare += s.first % 2 == 0 ? s.second : 0.0;
      checkCount(count, n, s.second, what + ", komi " + to_string(0.5 * s.first));
      checkCount(upTo, n, shareUpTo, what + ", komis up to " + to_string(0.5 * s.first));
    }
    text += "; whole komis: " + checkCount(whole, n, wholeShare, what + ", whole komis");
    text += "; evened: " + checkCount(evenedGames, n, evenedShare, what + ", evened games");
    printf("%s: %d games%s; %d komis of %d, the mean %.3f, the largest %g%s\n", what.c_str(), n, text.c_str(),
           (int)komis.size(), (int)shares.size(), sum / n, largest, fair ? "; the others get the fair komi" : "");
  }
};

//The draws of a board in a config: each key with the board's size after it, or the general key.
static Draws drawsOf(ConfigParser& cfg, int x, int y) {
  const string size = x == y ? to_string(x) : "none";
  auto value = [&](const string& key, double absent) {
    return cfg.contains(key + size) ? cfg.getDouble(key + size) : cfg.contains(key) ? cfg.getDouble(key) : absent;
  };
  Draws d;
  if(cfg.contains("komiMeans" + size)) {
    d.means = cfg.getDoubles("komiMeans" + size);
    d.shares = {1.0};
    if(cfg.contains("komiMeanRelProbs" + size))
      d.shares = cfg.getDoubles("komiMeanRelProbs" + size);
  }
  else if(cfg.contains("komiMean") && !(cfg.contains("komiAuto" + size) && cfg.getBool("komiAuto" + size))) {
    d.means = cfg.getDoubles("komiMean");
    d.shares = {1.0};
  }
  d.stdev = value("komiStdev", d.stdev);
  d.bigProb = value("komiBigStdevProb", d.bigProb);
  d.big = value("komiBigStdev", d.big);
  d.biggerProb = value("komiBiggerStdevProb", d.biggerProb);
  d.bigger = value("komiBiggerStdev", d.bigger);
  d.integerProb = value("komiAllowIntegerProb", d.integerProb);
  return d;
}

//The komi draws of a self-play config's new games, on each board that it has.
static void checkDraws(const char* path) {
  ConfigParser cfg(path);
  Logger logger(nullptr, false, true, false, false);
  GameInitializer gameInit(cfg, logger, "testkomi draws");
  map<pair<int, int>, Counted> counted;
  const int n = 200000;
  for(int i = 0; i < n; i++) {
    const Drawn d = drawGame(gameInit, NULL, NULL);
    counted[make_pair(d.board.x_size, d.board.y_size)].add(d);
  }
  printf("%s: %d new games\n", path, n);
  for(const pair<const pair<int, int>, Counted>& c : counted) {
    const int x = c.first.first, y = c.first.second;
    const Draws draws = drawsOf(cfg, x, y);
    c.second.checkAgainst(draws, Board(x, y), draws.bases(), draws.baseShares(), 0.0, draws.fair(),
                          boardText(x, y) + ", new games");
  }
}

static string numbers(const vector<double>& values) {
  string text;
  for(double v : values)
    text += (text.empty() ? "" : ",") + Global::strprintf("%g", v);
  return text;
}

//The config's lines of a size: all its keys.
static string linesOf(const Draws& d, int size) {
  auto line = [&](const string& key, const string& value) { return key + to_string(size) + " = " + value + "\n"; };
  return (d.fair() ? line("komiAuto", "true") :
                     line("komiMeans", numbers(d.means)) + line("komiMeanRelProbs", numbers(d.shares))) +
    line("komiStdev", numbers({d.stdev})) + line("komiBigStdevProb", numbers({d.bigProb})) +
    line("komiBigStdev", numbers({d.big})) + line("komiBiggerStdevProb", numbers({d.biggerProb})) +
    line("komiBiggerStdev", numbers({d.bigger})) + line("komiAllowIntegerProb", numbers({d.integerProb})) +
    line("forkCompensateKomiProb", numbers({d.forkProb}));
}

//A config of these sizes of square board, each in the same share of the games, with lines added.
static unique_ptr<ConfigParser> configOf(const vector<int>& sizes, const string& lines) {
  string edges, probs;
  for(int size : sizes) {
    edges += (edges.empty() ? "" : ",") + to_string(size);
    probs += probs.empty() ? "1" : ",1";
  }
  istringstream in("koRules = SIMPLE\nscoringRules = AREA\ntaxRules = NONE\nmultiStoneSuicideLegals = false\n"
                   "hasButtons = false\nbSizes = " + edges + "\nbSizeRelProbs = " + probs + "\n" + lines);
  return make_unique<ConfigParser>(in);
}

static Draws drawsWith(
  const vector<double>& means, const vector<double>& shares, double stdev, double bigProb, double big,
  double biggerProb, double bigger, double integerProb, double forkProb
) {
  Draws d;
  d.means = means;
  d.shares = shares;
  d.stdev = stdev;
  d.bigProb = bigProb;
  d.big = big;
  d.biggerProb = biggerProb;
  d.bigger = bigger;
  d.integerProb = integerProb;
  d.forkProb = forkProb;
  return d;
}

static Sgf::PositionSample startOn(int size) {
  Sgf::PositionSample start;
  start.board = Board(size, size);
  start.nextPla = P_BLACK;
  start.initialTurnNumber = 0;
  start.hintLoc = Board::NULL_LOC;
  start.weight = 1.0;
  return start;
}

//The sizes (two, or the game's only one) with draws of their own, at every start of GameInitializer. The config gives
//no general mean, and general spreads and shares that no size here has.
static void checkSizes(const vector<int>& sizes, const vector<Draws>& draws, const string& name) {
  string lines = "komiStdev = 40\nkomiBigStdevProb = 0.5\nkomiBigStdev = 45\nkomiBiggerStdevProb = 0.5\n"
                 "komiBiggerStdev = 50\nkomiAllowIntegerProb = 0.6\nforkCompensateKomiProb = 0.6\n";
  for(size_t i = 0; i < sizes.size(); i++)
    lines += linesOf(draws[i], sizes[i]);
  Logger logger(nullptr, false, false, false, false);
  const unique_ptr<ConfigParser> cfg = configOf(sizes, lines);
  GameInitializer gameInit(*cfg, logger, "testkomi " + name);
  check(cfg->unusedKeys().empty(), name + ": GameInitializer reads every key");
  //Asymmetric games: the komi of 4 in 10 is evened.
  PlaySettings asymmetric;
  asymmetric.normalAsymmetricPlayoutProb = 1.0;
  asymmetric.maxAsymmetricRatio = 4.0;
  asymmetric.minAsymmetricCompensateKomiProb = 0.4;

  const int n = 40000;
  map<int, Counted> news, uneven;
  for(int i = 0; i < n; i++) {
    const Drawn d = drawGame(gameInit, NULL, NULL);
    news[d.board.x_size].add(d);
    const Drawn u = drawGame(gameInit, NULL, NULL, asymmetric);
    uneven[u.board.x_size].add(u);
  }
  for(size_t i = 0; i < sizes.size(); i++) {
    const int size = sizes[i];
    const Draws& d = draws[i];
    const Board board(size, size);
    const string what = name + ", " + boardText(size, size);
    checkCount(news[size].n, n, 1.0 / sizes.size(), what + ", the size's games");
    news[size].checkAgainst(d, board, d.bases(), d.baseShares(), 0.0, d.fair(), what + ", new games");
    uneven[size].checkAgainst(d, board, d.bases(), d.baseShares(), 0.4, d.fair(), what + ", asymmetric games");

    //The starts from a position of this size, with the forks' share of evened games (the config has no
    //sgfCompensateKomiProb). The forks of a game of this size, around its komi. And the komi around the even komi that
    //Play::runGame finds at any start (the fair komi and every compensation), here 6.5: the start's spread and whole
    //komis.
    Sgf::PositionSample start = startOn(size);
    const double parent = -1.5;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    hist.setKomi((float)parent);
    const InitialPosition fork(board, hist, P_BLACK, true, false, false, 1.0);
    Rand rand("testkomi evened " + what);
    Counted starts, forks, evened;
    for(int j = 0; j < n / 2; j++) {
      Drawn s = drawGame(gameInit, NULL, &start);
      starts.add(s);
      s.extra.komiMean = 6.5f;
      s.extra.makeGameFair = s.extra.makeGameFairForEmptyBoard = false;
      PlayUtils::setKomiWithNoise(s.extra, hist, rand);
      s.komi = hist.rules.komi;
      evened.add(s);
      forks.add(drawGame(gameInit, &fork, NULL));
    }
    starts.checkAgainst(d, board, d.bases(), d.baseShares(), d.forkProb, d.fair(), what + ", starts from a position");
    forks.checkAgainst(d, board, {parent}, {1.0}, d.forkProb, false, what + ", forks");
    evened.checkAgainst(d, board, {6.5}, {1.0}, 0.0, false, what + ", around an even komi");

    //A start with its context frozen keeps its size and its komi.
    start.frozenContext = true;
    start.komi = -12.5f;
    for(int j = 0; j < 1000; j++) {
      const Drawn f = drawGame(gameInit, NULL, &start, asymmetric);
      check(f.komi == -12.5 && f.board.x_size == size && f.extra.komiMean == -12.5f && f.extra.komiStdev == 0.0f &&
              !f.extra.makeGameFair && !f.extra.makeGameFairForEmptyBoard,
            what + ": a frozen start keeps its size and its komi");
    }
  }
}

//The random numbers of a new game, taken here in KataGo's order: the size, the rules, the game's own draws at its
//start, the spreads and the whole komi, then the komi. Several means take one before the spreads, and a single mean
//takes none, so the games of a config without a size's key are those of the fork before these keys.
static void checkStream(const vector<int>& sizes) {
  const Draws single = drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 20, 0.5, 0);
  const Draws means = drawsWith({-4, 1, 6}, {2, 1, 3}, 3, 0.2, 8, 0.05, 20, 0.5, 0);
  Logger logger(nullptr, false, false, false, false);
  for(const Draws& d : {single, means}) {
    string lines = "komiMean = 2\nkomiStdev = 3\nkomiBigStdevProb = 0.2\nkomiBigStdev = 8\nkomiBiggerStdevProb = 0.05\n"
                   "komiBiggerStdev = 20\nkomiAllowIntegerProb = 0.5\n";
    if(d.means.size() > 1)
      for(int size : sizes)
        lines += linesOf(d, size);
    const unique_ptr<ConfigParser> cfg = configOf(sizes, lines);
    GameInitializer gameInit(*cfg, logger, "testkomi stream");
    Rand rand("testkomi stream");
    const vector<double> sizeProbs(sizes.size(), 1.0 / sizes.size());
    for(int i = 0; i < 2000; i++) {
      const int size = sizes[rand.nextUInt(sizeProbs.data(), sizeProbs.size())];
      for(int rule = 0; rule < 5; rule++)
        rand.nextUInt(1);
      Board board(size, size);
      if constexpr(GameInterface::rules::HasDrawStart<GameRules>::value) {
        GameRules::Position start = board.pos;
        GameInterface::drawStart<GameRules>(start, [&rand]() { return rand.nextUInt64(); });
        board.setPosition(start);
      }
      const double mean = d.means.size() > 1 ? d.means[rand.nextUInt(d.shares.data(), d.shares.size())] : d.means[0];
      BoardHistory hist(board, P_BLACK, Rules(), 0);
      const ExtraBlackAndKomi extra = PlayUtils::chooseExtraBlackAndKomi(
        (float)mean, 3.0f, 0.5, 0.0, 0, 0.2, 8.0f, 0.05, 20.0f, board, rand
      );
      PlayUtils::setKomiWithNoise(extra, hist, rand);
      const Drawn game = drawGame(gameInit, NULL, NULL);
      check(game.board.x_size == size && game.board.pos_hash == board.pos_hash && game.extra.komiMean == (float)mean &&
              game.extra.komiStdev == extra.komiStdev && game.komi == hist.rules.komi,
            "the random numbers of a new game, in KataGo's order, game " + to_string(i));
    }
  }
}

//One of a size's keys in place of the general one, and the other size's games with the general ones.
static void checkGeneralKeys(const vector<int>& sizes) {
  const int size = sizes.back(), other = sizes.front();
  const string S = to_string(size);
  const string spreads = "komiStdev = 3\nkomiBigStdevProb = 0.2\nkomiBigStdev = 8\nkomiBiggerStdevProb = 0.05\n"
                         "komiBiggerStdev = 20\nkomiAllowIntegerProb = 0.5\nforkCompensateKomiProb = 0.75\n";
  const string mean = "komiMean = 2\n" + spreads, fair = "komiAuto = true\n" + spreads;
  Logger logger(nullptr, false, false, false, false);

  //Draw for draw: a size's keys of the general values give the general games, as a single mean takes no random number.
  const Draws general = drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 20, 0.5, 0.75);
  const unique_ptr<ConfigParser> plainCfg = configOf(sizes, mean);
  const unique_ptr<ConfigParser> ownCfg = configOf(sizes, mean + linesOf(general, size));
  GameInitializer plainInit(*plainCfg, logger, "testkomi same"), ownInit(*ownCfg, logger, "testkomi same");
  for(int i = 0; i < 20000; i++) {
    const Drawn a = drawGame(plainInit, NULL, NULL), b = drawGame(ownInit, NULL, NULL);
    check(a.komi == b.komi && a.board.x_size == b.board.x_size && a.extra.komiStdev == b.extra.komiStdev &&
            a.extra.allowInteger == b.extra.allowInteger,
          "a size's keys of the general values draw the general games");
  }

  const Draws generalFair = drawsWith({}, {}, 3, 0.2, 8, 0.05, 20, 0.5, 0.75);
  const struct { string general, own; Draws draws; } merged[] = {
    {mean, "komiStdev" + S + " = 5\n", drawsWith({2}, {1}, 5, 0.2, 8, 0.05, 20, 0.5, 0.75)},
    {mean, "komiMeans" + S + " = -1\n", drawsWith({-1}, {1}, 3, 0.2, 8, 0.05, 20, 0.5, 0.75)},
    {mean, "komiBigStdevProb" + S + " = 0\n", drawsWith({2}, {1}, 3, 0, 8, 0.05, 20, 0.5, 0.75)},
    {mean, "komiBigStdev" + S + " = 4\n", drawsWith({2}, {1}, 3, 0.2, 4, 0.05, 20, 0.5, 0.75)},
    {mean, "komiBiggerStdevProb" + S + " = 1\n", drawsWith({2}, {1}, 3, 0.2, 8, 1, 20, 0.5, 0.75)},
    {mean, "komiBiggerStdev" + S + " = 25\n", drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 25, 0.5, 0.75)},
    {mean, "komiAllowIntegerProb" + S + " = 0\n", drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 20, 0, 0.75)},
    {mean, "komiAllowIntegerProb" + S + " = 1\n", drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 20, 1, 0.75)},
    {mean, "forkCompensateKomiProb" + S + " = 0.1\n", drawsWith({2}, {1}, 3, 0.2, 8, 0.05, 20, 0.5, 0.1)},
    {mean, "komiAuto" + S + " = true\n", generalFair},
    {mean, "komiAuto" + S + " = false\n", general},
    {fair, "komiMeans" + S + " = 0, 4\nkomiMeanRelProbs" + S + " = 4, 1\n",
     drawsWith({0, 4}, {4, 1}, 3, 0.2, 8, 0.05, 20, 0.5, 0.75)},
    {fair, "komiAuto" + S + " = false\nkomiMeans" + S + " = 1\n",
     drawsWith({1}, {1}, 3, 0.2, 8, 0.05, 20, 0.5, 0.75)},
  };
  for(const auto& m : merged) {
    const unique_ptr<ConfigParser> cfg = configOf(sizes, m.general + m.own);
    GameInitializer gameInit(*cfg, logger, "testkomi merged");
    const Board board(size, size);
    const InitialPosition fork(board, BoardHistory(board, P_BLACK, Rules(), 0), P_BLACK, true, false, false, 1.0);
    Counted own, others, forks;
    for(int i = 0; i < 20000; i++) {
      const Drawn d = drawGame(gameInit, NULL, NULL);
      (d.board.x_size == size ? own : others).add(d);
      forks.add(drawGame(gameInit, &fork, NULL));
    }
    const string what = "only " + m.own.substr(0, m.own.find('\n')) + (m.general == fair ? " (komiAuto)" : "");
    own.checkAgainst(m.draws, board, m.draws.bases(), m.draws.baseShares(), 0.0, m.draws.fair(), what);
    forks.checkAgainst(m.draws, board, {0.0}, {1.0}, m.draws.forkProb, false, what + ", forks");
    const Draws& g = m.general == fair ? generalFair : general;
    if(other != size)
      others.checkAgainst(g, Board(other, other), g.bases(), g.baseShares(), 0.0, g.fair(), what + ", the other size");
  }
}

//The boards that are not square, of a game that plays them, have the general keys.
static void checkRectangles(const vector<int>& sizes) {
  const int size = sizes.back(), other = sizes.front();
  const Draws small = drawsWith({-3}, {1}, 1, 0, 10, 0, 30, 1, 0);
  const Draws large = drawsWith({4, 6}, {1, 2}, 2, 0, 10, 0, 30, 0, 0);
  const Draws general = drawsWith({1}, {1}, 5, 0, 10, 0, 30, 0.5, 0);
  Logger logger(nullptr, false, false, false, false);
  const unique_ptr<ConfigParser> cfg = configOf(
    sizes, "allowRectangleProb = 0.8\nkomiMean = 1\nkomiStdev = 5\nkomiAllowIntegerProb = 0.5\n" +
             linesOf(small, other) + linesOf(large, size));
  GameInitializer gameInit(*cfg, logger, "testkomi rectangles");
  map<pair<int, int>, Counted> counted;
  for(int i = 0; i < 80000; i++) {
    const Drawn d = drawGame(gameInit, NULL, NULL);
    counted[make_pair(d.board.x_size, d.board.y_size)].add(d);
  }
  check(counted.size() == 4, "two square boards, and two that are not");
  for(const pair<const pair<int, int>, Counted>& c : counted) {
    const int x = c.first.first, y = c.first.second;
    const Draws& d = x != y ? general : x == size ? large : small;
    c.second.checkAgainst(d, Board(x, y), d.bases(), d.baseShares(), 0.0, false, "rectangles, " + boardText(x, y));
  }
}

//The error of a config's load, or nothing.
static string loadError(const vector<int>& sizes, const string& lines) {
  Logger logger(nullptr, false, false, false, false);
  const unique_ptr<ConfigParser> cfg = configOf(sizes, lines);
  try {
    GameInitializer gameInit(*cfg, logger, "testkomi load");
  }
  catch(const StringError& e) {
    return e.what();
  }
  return "";
}

//A config that cannot be meant stops at the load, with the key's name.
static void checkRefusals(const vector<int>& sizes) {
  const int size = sizes.back(), other = sizes.front();
  const string S = to_string(size), O = to_string(other), X = to_string(size + 1);  //X: of no size here
  const string mean = "komiMean = 0\n";
  const string top = Global::strprintf("%.0f", Rules::MAX_USER_KOMI);
  const string two = "komiMeans" + S + " = 0,2\n", probs = "komiMeanRelProbs" + S;
  vector<pair<string, string>> refused;  //the lines, and what the error says
  for(const string key : {"komiMeans", "komiMeanRelProbs", "komiStdev", "komiBigStdevProb", "komiBigStdev",
                          "komiBiggerStdevProb", "komiBiggerStdev", "komiAllowIntegerProb", "forkCompensateKomiProb"})
    refused.push_back({mean + key + X + " = 1\n", "'" + key + X + "'"});
  refused.push_back({mean + "komiAuto" + X + " = true\n", "'komiAuto" + X + "'"});
  refused.push_back({mean + "komiMeans0" + S + " = 1\n", "'komiMeans0" + S + "'"});
  refused.push_back({mean + "komiMeans99999999999 = 1\n", "'komiMeans99999999999'"});
  refused.push_back({mean + "komiMean" + S + " = 1\n", "'komiMean" + S + "'"});
  refused.push_back({mean + "komiMeansRelProbs" + S + " = 1\n", "'komiMeansRelProbs" + S + "'"});
  refused.push_back({mean + "komiMeans = 1\n", "'komiMeans'"});
  refused.push_back({mean + "komiMeanRelProbs = 1\n", "'komiMeanRelProbs'"});
  for(const string values : {"1", "1,1,1", "1,-1", "3,-1", "0,0", "1,nan", "1,x", "1,inf"})
    refused.push_back({mean + two + probs + " = " + values + "\n", probs});
  refused.push_back({mean + two, probs});
  refused.push_back({mean + probs + " = 1\n", probs});
  refused.push_back({mean + "komiMeans" + S + " = 0\n" + probs + " = 1,1\n", probs});
  refused.push_back({mean + "komiMeans" + S + " = 0\n" + probs + " = 0\n", probs});
  refused.push_back({mean + "komiMeans" + S + " = 0\nkomiAuto" + S + " = true\n", "komiMeans" + S});
  refused.push_back({mean + "komiMeans" + S + " = 0\nkomiAuto" + S + " = true\n", "komiAuto" + S});
  for(const string& value : {top + ".5", "-" + top + ".5", string("nan"), string("inf"), string("x")})
    refused.push_back({mean + "komiMeans" + S + " = 0," + value + "\n" + probs + " = 1,1\n", "'komiMeans" + S + "'"});
  refused.push_back({mean + "komiStdev" + S + " = -1\n", "'komiStdev" + S + "'"});
  refused.push_back({mean + "komiBigStdevProb" + S + " = -0.5\n", "'komiBigStdevProb" + S + "'"});
  refused.push_back({mean + "komiAllowIntegerProb" + S + " = 1.5\n", "'komiAllowIntegerProb" + S + "'"});
  refused.push_back({mean + "forkCompensateKomiProb" + S + " = 2\n", "'forkCompensateKomiProb" + S + "'"});
  refused.push_back({mean + "komiAuto = true\n", "komiMean"});
  refused.push_back({"", "komiMeans" + O});
  refused.push_back({"komiStdev" + S + " = 1\n", "komiMeans" + O});
  refused.push_back({"komiAuto = true\nkomiAuto" + S + " = false\n", "komiMeans" + S});
  if(other != size) {
    refused.push_back({"komiMeans" + S + " = 0\n", "komiMeans" + O});
    refused.push_back({"komiMeans" + O + " = 0\n", "komiMeans" + S});
    refused.push_back({"komiMeans" + O + " = 0\nkomiMeans" + S + " = 0\nallowRectangleProb = 0.5\n", O + "x" + S});
  }
  for(const pair<string, string>& r : refused) {
    const string error = loadError(sizes, r.first);
    check(error.find(r.second) != string::npos, "refused, with " + r.second + ": " + r.first + "-> " + error);
  }
  string own;
  for(int s : sizes)
    own += "komiMeans" + to_string(s) + " = 0\n";
  vector<string> accepted = {
    mean, "komiAuto = true\n", own, own + "komiAuto" + S + " = false\n", "komiAuto = true\n" + own,
    mean + "komiMeans" + S + " = " + top + ", -" + top + "\n" + probs + " = 0, 1\n",
    mean + "komiMeans" + S + " = 0.3\n" + probs + " = 1e-5\n"
  };
  if(other != size)
    accepted.push_back(mean + own + "allowRectangleProb = 0.5\n");
  for(const string& lines : accepted)
    check(loadError(sizes, lines).empty(), "accepted: " + lines + "-> " + loadError(sizes, lines));
  printf("%d configs that cannot be meant stop at the load with their key's name, and %d others load\n",
         (int)refused.size(), (int)accepted.size());
}

//A start from a position of another size than the config's (bSizes) has the general draws. Without a general komi, the
//load stops at such a position of hintPosesDir, and a game's start at another. A frozen start needs no general komi.
static void checkOtherBoards(int size, int other) {
  const string general = "komiMean = -3\nkomiStdev = 8\nkomiAllowIntegerProb = 0\n";
  const string S = to_string(size), O = to_string(other);
  const string own = "komiMeans" + S + " = 5\nkomiStdev" + S + " = 1\nkomiAllowIntegerProb" + S + " = 1\n";
  const Draws draws = drawsWith({-3}, {1}, 8, 0, 10, 0, 30, 0, 0);
  Logger logger(nullptr, false, false, false, false);
  Sgf::PositionSample start = startOn(other);
  {
    const unique_ptr<ConfigParser> cfg = configOf({size}, general + own);
    GameInitializer gameInit(*cfg, logger, "testkomi another board");
    Counted starts;
    for(int i = 0; i < 20000; i++)
      starts.add(drawGame(gameInit, NULL, &start));
    starts.checkAgainst(draws, start.board, draws.bases(), draws.baseShares(), 0.0, false,
                        "starts from a position of another board, " + boardText(other, other));
  }
  {
    const unique_ptr<ConfigParser> cfg = configOf({size}, own);
    GameInitializer gameInit(*cfg, logger, "testkomi no general komi");
    string error;
    try {
      drawGame(gameInit, NULL, &start);
    }
    catch(const StringError& e) {
      error = e.what();
    }
    check(error.find("komiMean") != string::npos && error.find(O + "x" + O) != string::npos,
          "a start on a board without a komi stops: " + error);
    Sgf::PositionSample frozen = start;
    frozen.frozenContext = true;
    frozen.komi = 3.5f;
    check(drawGame(gameInit, NULL, &frozen).komi == 3.5, "a frozen start on a board without a komi keeps its own");
  }
  char dir[] = "/tmp/testkomiXXXXXX";
  if(mkdtemp(dir) == NULL) {
    check(false, "a temporary folder");
    return;
  }
  const string file = string(dir) + "/a.hintposes.txt";
  ofstream out(file);
  out << Sgf::PositionSample::toJsonLine(start) << endl;
  out.close();
  const string hints = "hintPosesProb = 0.5\nhintPosesDir = " + string(dir) + "\n";
  const string error = loadError({size}, own + hints);
  check(error.find("komiMean") != string::npos && error.find(O + "x" + O) != string::npos,
        "a hint position on a board without a komi stops the load: " + error);
  const unique_ptr<ConfigParser> cfg = configOf({size}, general + own + hints);
  GameInitializer gameInit(*cfg, logger, "testkomi hint positions");
  Counted hinted;
  for(int i = 0; i < 40000; i++) {
    const Drawn d = drawGame(gameInit, NULL, NULL);
    if(d.board.x_size == other)
      hinted.add(d);
  }
  checkCount(hinted.n, 40000, 0.5, "the hint positions' games");
  hinted.checkAgainst(draws, start.board, draws.bases(), draws.baseShares(), 0.0, false,
                      "hint positions of another board, " + boardText(other, other));
  remove(file.c_str());
  rmdir(dir);
}

//The komi's keys of a size of board, in configs of the two largest sizes of square board that the build's game plays,
//or of its only one.
static void checkKeysOfSizes() {
  vector<int> sizes;
  bool rectangles = false;
  GameRules::Position p;
  for(int n = 2; n <= Board::MAX_LEN; n++) {
    if(GameRules::new_game(n, n, p))
      sizes.push_back(n);
    rectangles = rectangles || (n < Board::MAX_LEN && GameRules::new_game(n, n + 1, p));
  }
  if(sizes.empty()) {
    printf("the game plays no square board: no keys of a size\n");
    return;
  }
  sizes.erase(sizes.begin(), sizes.end() - min((size_t)2, sizes.size()));

  //One mean, with whole komis in every game. Several means, in shares that no other order of theirs has, and every
  //spread. Means off the grid and below 0, with no spread. No whole komis, with a mean of share 0 and shares that do
  //not sum to 1. A mean below 0 with bigger draws, and no whole komis. The fair komi.
  const Draws one = drawsWith({0}, {1}, 3, 0, 10, 0, 30, 1, 0.25);
  const Draws several = drawsWith({0, 2}, {1, 3}, 1.5, 0.2, 8, 0.05, 20, 0.5, 0.75);
  const Draws offGrid = drawsWith({-3.1, 0.3, 5.25}, {5, 1, 2}, 0, 0, 10, 0, 30, 1, 0);
  const Draws noWhole = drawsWith({-2.5, 1.5, 4}, {0, 0.5, 0.25}, 0, 0.1, 4, 0, 30, 0, 1);
  const Draws below = drawsWith({-4.5}, {7}, 6, 0, 10, 0.3, 12, 0, 0.5);
  const Draws fair = drawsWith({}, {}, 2, 0.5, 5, 0, 30, 0.75, 0.1);
  const vector<vector<Draws>> configs = {
    {one, several}, {several, one}, {offGrid, noWhole}, {below, fair}, {fair, offGrid}
  };
  for(size_t i = 0; i < configs.size(); i++)
    checkSizes(sizes, vector<Draws>(configs[i].end() - sizes.size(), configs[i].end()), "config " + to_string(i + 1));
  checkStream(sizes);
  checkGeneralKeys(sizes);
  checkRefusals(sizes);
  if(sizes.size() > 1) {
    if(rectangles)
      checkRectangles(sizes);
    checkOtherBoards(sizes.back(), sizes.front());
  }
}

int main(int argc, char** argv) {
  if(argc > 2) {
    fprintf(stderr, "usage: testkomi [SELFPLAY_CONFIG]\n");
    return 2;
  }
  Board::initHash();
  checkResults();
  checkBracket();
  checkKeysOfSizes();
  if(argc == 2)
    checkDraws(argv[1]);
  if(failures > 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("testkomi: all checks pass\n");
  return 0;
}
