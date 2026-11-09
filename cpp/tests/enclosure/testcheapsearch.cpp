// KataGame fork: self-play's full or cheap search of each action (play.cpp), with PlaySettings::cheapSearchPerMove and
// cheapSearchPerTurn on or off, for the build's game.
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
// the game to know the actions that follow the action before them, of two kinds: the actions that continue its move
// (GameInterface::continuesMove: Blueprints' drops, after their picks), and the actions in its turn (the same player
// moves again: Enclosure's second edges, and Blueprints' drops and second moves).
// - A digest of each game: its actions, decisions and weights, and its final margin. Two builds, or two settings, that
//   draw the same searches in the same order give the same digests.
// - The draws: the actions that draw their decision (with both settings off, all of them; with cheapSearchPerMove on,
//   all but the actions that continue a move; with cheapSearchPerTurn on, all but the actions in the turn of the
//   action before them; the loop's first action draws too). A draw comes before its search, so the share of full
//   searches among the draws is 1 - cheapSearchProb, within 4 standard deviations.
// - With a setting on: each action that it binds to the action before it has that action's decision and weight. With
//   both off, the actions of each kind draw their own.
// - The test prints, for each kind, the share of full searches among the actions before (the leaders) and among the
//   actions that follow them. Whether an action is a pick depends on its search (a cheap search can choose picks more
//   or less often than a full one), so with a setting on, the followers' share is their leaders', which can differ
//   from 1 - cheapSearchProb, and so can the share among all the actions. Enclosure's turns do not depend on the
//   search: a turn's first edge is a leader whatever its search.

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
  uint64_t h = 14695981039346656037ULL;
  template<class T> void add(const T& v) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    for(size_t i = 0; i < sizeof(T); i++)
      h = (h ^ b[i]) * 1099511628211ULL;
  }
};

//The actions of one kind that follow the action before them in the loop (their leader), and their full searches.
struct Followers {
  int n = 0, full = 0;
  int leadersFull = 0;                   //the followers whose leader had a full search
  int bothFull = 0;                      //the followers with their leaders, both full
  int differ = 0;                        //the followers whose decision or weight is not their leader's
  void add(bool cheap, float weight, bool leaderCheap, float leaderWeight) {
    n++;
    full += !cheap;
    leadersFull += !leaderCheap;
    bothFull += !cheap && !leaderCheap;
    differ += cheap != leaderCheap || weight != leaderWeight;
  }
};

//The counts over the games: the actions of the loops, and how many of them had a full search.
struct Counts {
  int actions = 0, actionsFull = 0;
  int draws = 0, drawsFull = 0;          //the actions that drew their decision
  Followers moves;                       //the actions that continue the move of the action before them (drops)
  Followers turns;                       //the actions in the turn of the action before them (the same player again)
};

static void readGame(int gameIdx, const FinishedGameData& data, bool perMove, bool perTurn, Counts& counts) {
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
  int full = 0, drops = 0, inTurn = 0;
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
    const bool sameTurn = t > 0 && data.endHist.moveHistory[startIdx + t - 1].pla == pla;
    check(!drop || sameTurn, "an action that continues a move is in its turn");
    const bool leaderCheap = t > 0 && data.wasCheapSearchByTurn[t - 1];
    const float leaderWeight = t > 0 ? data.targetWeightByTurnUnrounded[t - 1] : 0.0f;
    if(drop) {
      drops++;
      counts.moves.add(cheap, weight, leaderCheap, leaderWeight);
    }
    if(sameTurn) {
      inTurn++;
      counts.turns.add(cheap, weight, leaderCheap, leaderWeight);
    }
    if(!(perMove && drop) && !(perTurn && sameTurn)) {
      counts.draws++;
      counts.drawsFull += !cheap;
    }
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    pla = hist.presumedNextMovePla;
  }
  counts.actions += (int)numMoves;
  counts.actionsFull += full;
  digest.add(data.endHist.finalWhiteMinusBlackScore);
  printf("game %d: digest %016llx, %d actions after %d before the loop, %d full, %d drops, %d in a turn's\n", gameIdx,
         (unsigned long long)digest.h, (int)numMoves, (int)startIdx, full, drops, inTurn);
}

static double share(int hits, int n) { return n > 0 ? (double)hits / n : 0.0; }

//Whether hits of n trials are the share q, within 4 standard deviations.
static bool nearShare(int hits, int n, double q) {
  return n > 0 && fabs(share(hits, n) - q) <= 4.0 * sqrt(q * (1.0 - q) / n);
}

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
  const bool perTurn = playSettings.cheapSearchPerTurn;
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
  printf("game %s, cheapSearchPerMove %s, cheapSearchPerTurn %s, cheapSearchProb %g, %d visits (%d in a cheap "
         "search)\n", GameRules::NAME, perMove ? "on" : "off", perTurn ? "on" : "off", playSettings.cheapSearchProb,
         (int)baseParams.maxVisits, playSettings.cheapSearchVisits);

  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = nnEval->getModelName();
  botSpec.nnEval = nnEval;
  botSpec.baseParams = baseParams;
  ForkData* forkData = new ForkData();
  Counts c;
  for(int g = 0; g < numGames; g++) {
    FinishedGameData* data = gameRunner->runGame("testcheapsearch game " + to_string(g), botSpec, botSpec, forkData,
                                                 NULL, logger, nullptr, nullptr, nullptr, nullptr, nullptr);
    check(data != NULL, "the game ends");
    if(data == NULL)
      break;
    readGame(g, *data, perMove, perTurn, c);
    delete data;
  }

  const double q = 1.0 - playSettings.cheapSearchProb;
  printf("draws: %d of %d actions, %d full (%.4f, expected %.4f)\n", c.draws, c.actions, c.drawsFull,
         share(c.drawsFull, c.draws), q);
  printf("full searches: %.4f of the actions\n", share(c.actionsFull, c.actions));
  //A follower's leader is the action before it: a drop's pick, a second edge's first edge.
  const auto print = [&](const char* what, const Followers& f) {
    printf("%s: %d; full searches: %.4f of their leaders, %.4f of them; %d with both full (%.4f); %d differ from "
           "their leader\n", what, f.n, share(f.leadersFull, f.n), share(f.full, f.n), f.bothFull,
           share(f.bothFull, f.n), f.differ);
  };
  print("actions that continue a move (drops)", c.moves);
  print("actions in the turn of the action before them", c.turns);
  check(nearShare(c.drawsFull, c.draws, q), "the share of full searches among the draws");
  //With a setting on, the actions that it binds take their leader's search. Each action that continues a move is in
  //its turn, so cheapSearchPerTurn binds the drops too.
  if(perMove || perTurn)
    check(c.moves.differ == 0, "each drop takes its pick's decision and weight");
  else if(c.moves.n > 0)
    check(nearShare(c.moves.full, c.moves.n, q), "the drops draw their own");
  if(perTurn)
    check(c.turns.differ == 0, "each action in the turn of the action before it takes that action's search");
  else if(c.turns.n > 0 && !(perMove && c.moves.n > 0))
    check(nearShare(c.turns.full, c.turns.n, q), "the actions in a turn draw their own");

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
