// KataGame fork: komi through the front, for the build's game. The komi K is points added to White's final margin
// (BoardHistory::endAndScoreGameNow), and KataGo's even-komi search looks for the K that makes a position even.
//
//   testkomi [SELFPLAY_CONFIG]
//
// - The result: a random game to its end, then the history's result at komis of both signs, at 0, and at the komis
//   that decide the game (the half point nearest the opposite of the earned margin, and half a point to either side):
//   the final margin with K, its winner, White's wins in the value targets (a margin of 0 is half a win), the final
//   score target, each side's own komi, and the network cache's hash, which changes with a nonzero K only.
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
// - With a self-play config: the komi draws of new games, which play.cpp's GameInitializer makes as in self-play,
//   against the shares that the config's probabilities give: the draws of exactly 0, and the draws beyond 3 standard
//   deviations of the middle spread (komiBigStdev). Every draw is on the half-point grid, and none is beyond 3
//   standard deviations of the widest spread.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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
//The other games give none of these, so they keep KataGo's bracket and rule, and KataGo's lead step times SCORE_SCALE.
static bool pinnedSearch(double& step, double& limit, bool& endIsEven, double& leadStep) {
  if(strcmp(GameRules::NAME, "blueprints") == 0) {
    step = 12.0;
    limit = 3000.0;
    endIsEven = false;
    leadStep = 0.0;
    return true;
  }
  if(strcmp(GameRules::NAME, "enclosure") == 0 || strcmp(GameRules::NAME, "connect4") == 0) {
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
    const double d = margin + komi;
    const Player winner = d > 0 ? P_WHITE : d < 0 ? P_BLACK : C_EMPTY;
    const string at = "komi " + to_string(komi) + ": ";
    check(h.winner == winner, at + "the winner is the sign of the margin with komi");
    check(h.finalWhiteMinusBlackScore == (float)d, at + "the final margin counts komi");
    const double wins = ScoreValue::whiteWinsOfWinner(h.winner, 0.5);
    check(wins == (d > 0 ? 1.0 : d < 0 ? 0.0 : 0.5), at + "White's wins: 1, 0, or half a win at a margin of 0");
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

//The standard normal distribution function.
static double normalCdf(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }

//The komi draws of a self-play config's new games.
static void checkDraws(const char* path) {
  ConfigParser cfg(path);
  //The spreads, for the expected shares, with play.cpp's defaults. GameInitializer checks their ranges and draws.
  auto value = [&](const char* key, double absent) { return cfg.contains(key) ? cfg.getDouble(key) : absent; };
  const double komiMean = value("komiMean", 0.0);
  const double komiStdev = value("komiStdev", 0.0);
  const double bigProb = value("komiBigStdevProb", 0.0);
  const double bigStdev = value("komiBigStdev", 10.0);
  const double biggerProb = value("komiBiggerStdevProb", 0.0);
  const double biggerStdev = value("komiBiggerStdev", 30.0);
  printf("%s: komiMean %g, komiStdev %g, komiBigStdev %g at %g, komiBiggerStdev %g at %g\n", path, komiMean, komiStdev,
         bigStdev, bigProb, biggerStdev, biggerProb);

  //The sampler's spreads (chooseExtraBlackAndKomi): the bigger draw overrides the big one, which overrides komiStdev.
  //Each spread is in the config's points times sqrt(komi area) / 19.
  const double scale = sqrt(Board().komiArea()) / 19.0;
  const double pBigger = biggerStdev > 0 && biggerProb > 0 ? biggerProb : 0.0;
  const double pBig = bigStdev > 0 ? bigProb * (1.0 - pBigger) : 0.0;
  const double pBase = 1.0 - pBig - pBigger;
  struct Spread { double prob, stdev; };
  const Spread spreads[3] = {{pBase, komiStdev * scale}, {pBig, bigStdev * scale}, {pBigger, biggerStdev * scale}};
  //The expected shares, at a komiMean of 0. A draw of exactly 0: the spreads of 0, and the draws of the others that the
  //rounding to half points takes to 0 (half a point times the density at 0, of a Gaussian truncated at 3 sigma). A
  //draw beyond 3 sigma of the big spread: each wider spread's draws beyond it.
  double zeroShare = 0.0, beyondShare = 0.0;
  const double edge = 3.0 * bigStdev * scale;
  const double truncated = 2.0 * normalCdf(3.0) - 1.0;
  for(const Spread& s : spreads) {
    if(s.stdev == 0.0)
      zeroShare += s.prob;
    else {
      zeroShare += s.prob * 0.5 / (sqrt(2.0 * acos(-1.0)) * s.stdev * truncated);
      if(s.stdev * 3.0 > edge)
        beyondShare += s.prob * 2.0 * (normalCdf(3.0) - normalCdf(edge / s.stdev)) / truncated;
    }
  }

  Logger logger(nullptr, false, true, false, false);
  GameInitializer gameInit(cfg, logger, "testkomi draws");
  const int n = 200000;
  int zeros = 0, beyond = 0, offGrid = 0;
  double largest = 0.0, sum = 0.0;
  for(int i = 0; i < n; i++) {
    Board board;
    Player pla;
    BoardHistory hist;
    ExtraBlackAndKomi extraBlackAndKomi;
    SearchParams params;  //Self-play's form, which also draws noResultStdev and drawRandRadius into the params.
    OtherGameProperties otherGameProps;
    gameInit.createGame(board, pla, hist, extraBlackAndKomi, params, NULL, PlaySettings(), otherGameProps, NULL,
                        BoardHistoryModes());
    const double komi = hist.rules.komi;
    zeros += komi == 0.0;
    beyond += fabs(komi) > edge;
    offGrid += !Rules::komiIsIntOrHalfInt((float)komi);
    largest = max(largest, fabs(komi));
    sum += komi;
  }
  const double widest = 3.0 * max({spreads[0].stdev, spreads[1].stdev, spreads[2].stdev});
  printf("%d draws: %.4f%% at 0 (expected %.4f%%), %.4f%% beyond %g points (expected %.4f%%), the largest %g points "
         "(the widest spread's 3 sigma: %g), the mean %.2f, %d off the half-point grid\n",
         n, 100.0 * zeros / n, 100.0 * zeroShare, 100.0 * beyond / n, edge, 100.0 * beyondShare, largest, widest, sum / n,
         offGrid);
  const double zeroSigma = sqrt(zeroShare * (1.0 - zeroShare) / n), beyondSigma = sqrt(beyondShare * (1.0 - beyondShare) / n);
  check(komiMean != 0.0 || fabs((double)zeros / n - zeroShare) <= 5.0 * zeroSigma + 1e-9, "the share of draws at 0");
  check(fabs((double)beyond / n - beyondShare) <= 5.0 * beyondSigma + 1e-9, "the share of draws beyond the big spread");
  check(offGrid == 0, "every draw on the half-point grid");
  check(largest <= widest + 0.5, "no draw beyond 3 sigma of the widest spread");
}

int main(int argc, char** argv) {
  if(argc > 2) {
    fprintf(stderr, "usage: testkomi [SELFPLAY_CONFIG]\n");
    return 2;
  }
  Board::initHash();
  checkResults();
  checkBracket();
  if(argc == 2)
    checkDraws(argv[1]);
  if(failures > 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("testkomi: all checks pass\n");
  return 0;
}
