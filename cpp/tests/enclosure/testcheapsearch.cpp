// KataGame fork: self-play's full or cheap search of each action (play.cpp), with PlaySettings::cheapSearchPerMove on
// or off, for the build's game.
//
//   testcheapsearch SELFPLAY_CONFIG MODEL GAMES [OVERRIDES]
//
// The test plays GAMES self-play games one after the other, as self-play plays them (play.cpp's GameRunner, with a
// fork of a game feeding a later game), with the config and then OVERRIDES (key=value,..., as -override-config). The
// seeds are fixed (the games, the searches, the network's symmetries), and the test plays one game at a time with one
// search thread, so a build plays the same games each time. The test sets policySurpriseDataWeight and
// valueSurpriseDataWeight to 0, so that each action's recorded weight is the weight of its search's limits.
// For each action of the game's loop (after the moves of policy init or of a fork), the test reads the decision that
// the game recorded: cheap or full (wasCheapSearchByTurn), and the weight (targetWeightByTurnUnrounded). It replays
// the game to know which actions continue the move of the action before them (GameInterface::continuesMove:
// Blueprints' drops, after their picks).
// - A digest of each game: its actions, decisions and weights, and its final margin. Two builds, or two settings, that
//   draw the same searches in the same order give the same digests.
// - The draws: the actions that draw their decision (with the setting off, all of them; with it on, all but the
//   actions that continue a move, and the loop's first action draws too). A draw comes before its search, so the share
//   of full searches among the draws is 1 - cheapSearchProb, within 4 standard deviations.
// - With the setting on: each action that continues the move of the action before it has that action's decision and
//   weight. With it off, those actions draw.
// - The test prints the share of full searches among all the actions, the picks and the drops. Whether an action is a
//   pick depends on its search (a cheap search can choose picks more or less often than a full one), so with the
//   setting on, the drops' share is the picks' share, which can differ from 1 - cheapSearchProb, and so can the share
//   among all the actions.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../dataio/trainingwrite.h"
#include "../../neuralnet/nneval.h"
#include "../../program/play.h"
#include "../../program/playsettings.h"
#include "../../program/setup.h"

using namespace std;

static int failures = 0;

static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

//64-bit FNV-1a, over the bytes of a value.
struct Digest {
  uint64_t h = 1469598103934665603ULL;
  template<class T> void add(const T& v) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    for(size_t i = 0; i < sizeof(T); i++)
      h = (h ^ b[i]) * 1099511628211ULL;
  }
};

//The counts over the games: the actions of the loops, and how many of them had a full search.
struct Counts {
  int actions = 0, actionsFull = 0;
  int draws = 0, drawsFull = 0;          //the actions that drew their decision
  int drops = 0, dropsFull = 0;          //the actions that continue the move of the action before them, in the loop
  int picks = 0, picksFull = 0;          //the actions before those
  int bothFull = 0;                      //the drops with their picks, both full
  int differ = 0;                        //the drops whose decision or weight is not their pick's
};

static void readGame(int gameIdx, const FinishedGameData& data, bool perMove, Counts& counts) {
  const size_t numMoves = data.targetWeightByTurnUnrounded.size();
  if(data.wasCheapSearchByTurn.size() != numMoves) {
    check(false, "a decision for each action");
    return;
  }
  Board board = data.startBoard;
  BoardHistory hist = data.startHist;
  Player pla = data.startPla;
  const size_t startIdx = data.startHist.moveHistory.size();
  Digest digest;
  int full = 0, drops = 0;
  for(size_t t = 0; t < numMoves; t++) {
    const Move move = data.endHist.moveHistory[startIdx + t];
    check(move.pla == pla && hist.isLegal(board, move.loc, move.pla), "the game replays");
    const bool cheap = data.wasCheapSearchByTurn[t];
    const float weight = data.targetWeightByTurnUnrounded[t];
    digest.add(move.loc);
    digest.add(cheap);
    digest.add(weight);
    full += !cheap;
    //The loop's first action has no action before it in the loop: it draws.
    const bool drop = t > 0 && GameInterface::continuesMove<GameRules>(board.pos);
    if(drop) {
      const bool pickCheap = data.wasCheapSearchByTurn[t - 1];
      drops++;
      counts.drops++;
      counts.dropsFull += !cheap;
      counts.picks++;
      counts.picksFull += !pickCheap;
      counts.bothFull += !cheap && !pickCheap;
      counts.differ += cheap != pickCheap || weight != data.targetWeightByTurnUnrounded[t - 1];
    }
    if(!(perMove && drop)) {
      counts.draws++;
      counts.drawsFull += !cheap;
    }
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    pla = hist.presumedNextMovePla;
  }
  counts.actions += (int)numMoves;
  counts.actionsFull += full;
  digest.add(data.endHist.finalWhiteMinusBlackScore);
  printf("game %d: digest %016llx, %d actions after %d before the loop, %d full, %d drops\n", gameIdx,
         (unsigned long long)digest.h, (int)numMoves, (int)startIdx, full, drops);
}

static double share(int hits, int n) { return n > 0 ? (double)hits / n : 0.0; }

int main(int argc, char** argv) {
  if(argc < 4 || argc > 5) {
    fprintf(stderr, "usage: testcheapsearch SELFPLAY_CONFIG MODEL GAMES [OVERRIDES]\n");
    return 2;
  }
  const int numGames = atoi(argv[3]);
  if(numGames <= 0) {
    fprintf(stderr, "GAMES must be positive\n");
    return 2;
  }
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg(argv[1]);
  if(argc == 5)
    cfg.overrideKeys(ConfigParser::parseCommaSeparated(argv[4]));
  cfg.overrideKeys(map<string, string>{{"policySurpriseDataWeight", "0"}, {"valueSurpriseDataWeight", "0"},
                                       {"numSearchThreads", "1"}});
  Logger logger(nullptr, false, true, false, false);
  Rand seedRand("testcheapsearch");
  const SearchParams baseParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
  const PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, false);
  const bool perMove = playSettings.cheapSearchPerMove;
  GameRunner* gameRunner = new GameRunner(cfg, "testcheapsearch games", playSettings, logger);
  Setup::initializeSession(cfg);
  NNEvaluator* nnEval;
  {
    const GameInitializer* init = gameRunner->getGameInitializer();
    const bool exactNNLen = init->getMinBoardXSize() == init->getMaxBoardXSize() &&
                            init->getMinBoardYSize() == init->getMaxBoardYSize();
    nnEval = Setup::initializeNNEvaluator(argv[2], argv[2], "", cfg, logger, seedRand, 1, init->getMaxBoardXSize(),
                                          init->getMaxBoardYSize(), Setup::MaxBatchSizeRequest::requireFromConfig(),
                                          exactNNLen, false, Setup::SETUP_FOR_OTHER);
  }
  printf("game %s, cheapSearchPerMove %s, cheapSearchProb %g, %d visits (%d in a cheap search)\n", GameRules::NAME,
         perMove ? "on" : "off", playSettings.cheapSearchProb, (int)baseParams.maxVisits,
         playSettings.cheapSearchVisits);

  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = nnEval->getModelName();
  botSpec.nnEval = nnEval;
  botSpec.baseParams = baseParams;
  ForkData* forkData = new ForkData();
  Counts counts;
  for(int g = 0; g < numGames; g++) {
    FinishedGameData* data = gameRunner->runGame("testcheapsearch game " + to_string(g), botSpec, botSpec, forkData,
                                                 NULL, logger, nullptr, nullptr, nullptr, nullptr, nullptr);
    check(data != NULL, "the game ends");
    if(data == NULL)
      break;
    readGame(g, *data, perMove, counts);
    delete data;
  }

  const double q = 1.0 - playSettings.cheapSearchProb;
  const Counts& c = counts;
  printf("draws: %d of %d actions, %d full (%.4f, expected %.4f)\n", c.draws, c.actions, c.drawsFull,
         share(c.drawsFull, c.draws), q);
  printf("full searches: %.4f of the actions, %.4f of the %d picks, %.4f of their drops; %d with both full (%.4f); "
         "%d drops differ from their pick\n", share(c.actionsFull, c.actions), share(c.picksFull, c.picks), c.picks,
         share(c.dropsFull, c.drops), c.bothFull, share(c.bothFull, c.drops), c.differ);
  const double sigma = c.draws > 0 ? sqrt(q * (1.0 - q) / c.draws) : 0.0;
  check(fabs(share(c.drawsFull, c.draws) - q) <= 4.0 * sigma + 1e-12, "the share of full searches among the draws");
  if(perMove)
    check(c.differ == 0, "each drop takes its pick's decision and weight");
  else if(c.drops > 0) {
    const double dropSigma = sqrt(q * (1.0 - q) / c.drops);
    check(fabs(share(c.dropsFull, c.drops) - q) <= 4.0 * dropSigma, "the drops draw their own");
  }

  delete forkData;
  delete gameRunner;
  delete nnEval;
  ScoreValue::freeTables();
  if(failures > 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("testcheapsearch: all checks pass\n");
  return 0;
}
