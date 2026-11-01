// KataGame fork (abscls-il4e.11): self-play's search of the root's views (PlaySettings::rootViews, program/play.cpp),
// for the build's game, which must have views of its own (GameInterface::hasViews: Blueprints).
//
//   testselfplayviews SELFPLAY_CONFIG MODEL GAMES [OVERRIDES [OUT_DIR]]
//
// The test plays GAMES self-play games one after the other, as self-play plays them (play.cpp's GameRunner, with a
// fork of a game feeding a later game), with the config, then OVERRIDES (key=value,..., as -override-config), then
// rootViews=true, one search thread, and no reduced or asymmetric visits. MODEL "/dev/null" is KataGo's evaluator
// without a network (random policy and values). With diversityMember 1 to 4 in OVERRIDES, both players are that
// member, with a fixed state of the members (each statistic's mean 0 and standard deviation 1, as
// runenclosurerowtests'), and MODEL must have the intrinsic value head. The seeds are fixed, so a build plays the same
// games each time. The checks:
// - The game replays from its first position, each move in its view (FinishedGameData::playMove), to its end.
// - The moves before the loop (policy init, a fork) are in the rules' view.
// - Each move of the loop is in a view of its position's cover (RootViews::cover), the rules' view ("") when the
//   cover is the rules' view alone. Each move of the turn's policy target is a legal action of the position in the
//   move's view: the move's view is its search's.
// - A full search has the whole of its limits in each of its searches (reduceVisits off: the turn's visits are the
//   config's maxVisits), with one view or more.
// - The action of one view in another (GameInterface::viewAction) on the first positions of each game whose cover has
//   more views: for each view of the cover and each of its legal actions, in each other view of the cover and in the
//   rules' view, the action with the same child position, or -1 when that view has none.
// - RGSC's played positions (Rgsc::playedEntry, with the moves' views) after the game's first move in a view: the
//   entry's state restores to the replayed position.
// - RGSC's tree candidate (Rgsc::TreeTracker) at the root of the game's first search in another view than the rules',
//   after a move: the candidate's state restores, with the candidate's key, to the position in the rules' view.
// - The games hold a move in another view than the rules' and a full search whose cover has more views: a run whose
//   games lack either fails, as it did not test them.
// With OUT_DIR, the games' rows (TrainingDataWriter, the config's inputsVersion) go to OUT_DIR as .npz files, and their
// records (WriteSgf) to OUT_DIR/games.sgfs, for games/blueprints/tools/check_rows.py --records. The test prints, over
// the games: the actions of the loops, those in another view than the rules', the full and the cheap searches whose
// cover has more views, and the rows whose second policy target the writer moved into the row's view or lost.
// With OUT_DIR, also (abscls-3tbn):
// - No row has a policy target of sum 0 with its weight above 0 (C26 for the first, C28 for the second): the trainer
//   divides each target by its sum.
// - Next policies of target 0 in the row's view. A search's policy lists each child of its root, and a child's target
//   can be 0 (KataGo's policy target pruning, or the rounding to whole targets), so a row whose view keeps only such
//   moves of the next policy has a target of sum 0. Such rows are few in the games, so the test also makes them. After
//   the writer takes the game's rows, the test takes each next policy that the row's view moves, where the view keeps
//   a move and leaves out a move of target above 0, and sets the kept moves' targets to 0. A second writer writes the
//   game again without its side positions, its text to no file. Each row whose next policy keeps only moves of target
//   0 must have no second policy target (C28 0), and the writer must count it in numRowsNextPolicyLost. The games
//   must give such a row.
// Exit code 0 when every check holds.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../dataio/sgf.h"
#include "../../dataio/trainingwrite.h"
#include "../../neuralnet/modelversion.h"
#include "../../neuralnet/nneval.h"
#include "../../program/play.h"
#include "../../program/playsettings.h"
#include "../../program/rgsc.h"
#include "../../program/rootviews.h"
#include "../../program/setup.h"

using namespace std;

static int failures = 0;

static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 30)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

//The child position of the action on p, by its hash.
static Hash128 childHash(const GameRules::Position& p, int action) {
  GameRules::Position c = p;
  GameRules::play(c, action);
  return GameRules::hash(c);
}

//p in the view (GameInterface::setView): "" leaves it as it is.
static GameRules::Position inView(const GameRules::Position& p, const string& view) {
  GameRules::Position out = p;
  string why;
  const bool suc = GameInterface::setView<GameRules>(out, view, why);
  testAssert(suc);
  return out;
}

//GameInterface::viewAction from each view of the cover of p (in the rules' view) to each other view and to the rules'
//view, against the actions with the same child. Returns the actions compared.
static int64_t checkViewActions(const GameRules::Position& p, const vector<string>& cover) {
  vector<string> views = cover;
  bool hasRules = false;
  for(const string& v : views)
    hasRules = hasRules || v.empty();
  if(!hasRules)
    views.push_back("");
  //Each view's legal actions, and its actions by their child positions.
  vector<GameRules::Position> positions;
  vector<vector<int>> actions(views.size());
  vector<map<Hash128, int>> byChild(views.size());
  for(size_t v = 0; v < views.size(); v++) {
    positions.push_back(inView(p, views[v]));
    GameRules::legal_actions(positions[v], actions[v]);
    for(int a : actions[v]) {
      const Hash128 h = childHash(positions[v], a);
      check(byChild[v].count(h) == 0, "two actions of one view with one child");
      byChild[v][h] = a;
    }
  }
  int64_t compared = 0;
  for(size_t from = 0; from < views.size(); from++) {
    if(from >= cover.size())
      break;  //the rules' view added above: no move of the cover comes from it
    for(int a : actions[from]) {
      const Hash128 h = childHash(positions[from], a);
      for(size_t to = 0; to < views.size(); to++) {
        auto found = byChild[to].find(h);
        const int expected = found == byChild[to].end() ? -1 : found->second;
        const int got = GameInterface::viewAction<GameRules>(positions[from], a, positions[to]);
        check(got == expected, "viewAction from view '" + views[from] + "' to '" + views[to] + "' of action " +
              to_string(a) + ": " + to_string(got) + ", the same child's " + to_string(expected));
        compared++;
      }
    }
  }
  return compared;
}

struct Counts {
  int64_t actions = 0, inView = 0, before = 0;
  int64_t fullMore = 0, cheapMore = 0, fullOne = 0, cheapOne = 0;
  int64_t viewActions = 0, viewPositions = 0, rgscChecked = 0, trackerChecked = 0;
};

//RGSC's tree tracker at a search's root in a set view: viewed is board, a position of the game after the moves of hist,
//in that view. The tracker's candidate restores, with its key, to board, the state in the rules' view.
static void checkTreeCandidate(const Board& board, const Board& viewed, const BoardHistory& hist,
                               const Rgsc::Context& context, const string& where) {
  Rgsc::Settings settings;
  Rgsc::Buffer buffer(settings);
  Rgsc::TreeTracker tracker(&buffer);
  NNOutput output;
  output.hasRegret = true;
  output.regretValue = 1.0f;
  output.regretRanking = 1.0f;
  //The reason: the hash of a set view keys the view, and the state's text leaves it out.
  check(Rgsc::stateKey(viewed, context) != Rgsc::stateKey(board, context), where + ": a set view changes the key");
  tracker.offer(output, viewed, hist, context, 0);
  check(tracker.found(), where + ": the tracker takes the root in a view");
  if(!tracker.found())
    return;
  const Rgsc::Entry e = tracker.candidate();
  Board restored;
  string why;
  const bool suc = Rgsc::restoreState(e, restored, why);
  check(suc, where + ": RGSC's tree candidate at a root in a view restores with its key: " + why);
  check(suc && GameRules::hash(restored.pos) == GameRules::hash(board.pos),
        where + ": RGSC's tree candidate at a root in a view is the position in the rules' view");
}

struct RowCounts {
  int64_t rows = 0;
  int64_t zeroRows = 0;  //the rows at a turn of zeroTurns
};

//The checks of a file's rows (TrainingDataWriter::beforeWrite): no policy target of sum 0 with its weight above 0, and
//no second policy target in the rows at a turn of zeroTurns (C51, the row's turn).
static void checkRows(const TrainingWriteBuffers& b, const set<int>& zeroTurns, RowCounts& counts,
                      const string& where) {
  const int64_t numTargets = b.globalTargetsNC.shape[1];
  const int64_t channels = b.policyTargetsNCMove.shape[1];
  const int64_t policySize = b.policyTargetsNCMove.shape[2];
  for(int64_t r = 0; r < b.curRows; r++) {
    const float* global = b.globalTargetsNC.data + r * numTargets;
    const int turn = (int)global[51];
    for(int c = 0; c < 2; c++) {
      const int16_t* target = b.policyTargetsNCMove.data + (r * channels + c) * policySize;
      int64_t sum = 0;
      for(int64_t p = 0; p < policySize; p++)
        sum += target[p];
      const float weight = global[c == 0 ? 26 : 28];
      check(sum > 0 || weight == 0, where + ", the row of turn " + to_string(turn) + ": policy target " + to_string(c) +
            " of sum " + to_string(sum) + " with its weight " + to_string(weight));
    }
    counts.rows++;
    if(zeroTurns.count(turn) > 0) {
      counts.zeroRows++;
      check(global[28] == 0, where + ", the row of turn " + to_string(turn) +
            ": a next policy with only moves of target 0 in the row's view has a second policy target");
    }
  }
}

struct ZeroCounts {
  int64_t found = 0;  //rows whose next policy keeps only moves of target 0 in the game as it was played
  int64_t made = 0;   //rows whose next policy the test changed so
};

//The next policies of target 0 in the row's view (the case of abscls-3tbn). For each row of the game, its next policy
//in its view as TrainingDataWriter::writeGame gives it: the next position in the view of GameInterface::viewInFrame,
//and each move of the next search's policy at its action there (GameInterface::viewAction), or left out. The test
//counts a policy that keeps moves of target 0 only, and does not change it. A policy that keeps a move and leaves out a
//move of target above 0 gets its kept moves' targets set to 0. Returns the turns (moveHistory's indices) of the rows of
//both kinds.
static set<int> zeroNextPolicies(FinishedGameData& data, ZeroCounts& counts) {
  set<int> turns;
  const size_t start = data.startHist.moveHistory.size();
  const size_t numMoves = data.endHist.moveHistory.size();
  Board board = data.startHist.initialBoard;
  BoardHistory hist(board, data.startHist.initialPla, data.startHist.rules, data.startHist.initialEncorePhase);
  for(size_t i = 0; i < numMoves; i++) {
    Board row = board;
    data.setMoveView(i, row);
    data.playMove(i, board, hist);  //board: the next position, as writeGame's posHistForFutureBoards
    if(i < start || i + 1 >= numMoves)
      continue;
    vector<PolicyTargetMove>* next = data.policyTargetsByTurn[i + 1 - start].policyTargets;
    const string frameView = GameInterface::viewInFrame<GameRules>(board.pos, row.pos);
    const string& searchView = data.moveView(i + 1);
    if(next == NULL || frameView == searchView)
      continue;
    Board searched = board;
    FinishedGameData::setBoardView(searched, searchView);
    Board frame = board;
    FinishedGameData::setBoardView(frame, frameView);
    vector<size_t> kept;
    int64_t keptSum = 0;
    bool droppedTarget = false;
    for(size_t m = 0; m < next->size(); m++) {
      const PolicyTargetMove& move = (*next)[m];
      if(GameInterface::viewAction<GameRules>(searched.pos, Board::actionOfMoveLoc(move.loc), frame.pos) >= 0) {
        kept.push_back(m);
        keptSum += move.policyTarget;
      }
      else
        droppedTarget = droppedTarget || move.policyTarget > 0;
    }
    if(kept.empty() || (keptSum > 0 && !droppedTarget))
      continue;
    if(keptSum == 0)
      counts.found++;
    else {
      for(size_t m : kept)
        (*next)[m].policyTarget = 0;
      counts.made++;
    }
    turns.insert((int)i);
  }
  return turns;
}

static void readGame(int gameIdx, const FinishedGameData& data, int64_t maxVisits, Counts& counts) {
  const size_t start = data.startHist.moveHistory.size();
  const size_t numMoves = data.endHist.moveHistory.size();
  check(data.moveViews.size() == numMoves, "a view for each move");
  for(size_t i = 0; i < start; i++)
    check(data.moveView(i).empty(), "the moves before the loop in the rules' view");
  counts.before += start;
  //The replay from the game's first position, each move in its view.
  Board board = data.startHist.initialBoard;
  BoardHistory hist(board, data.startHist.initialPla, data.startHist.rules, data.startHist.initialEncorePhase);
  for(size_t i = 0; i < start; i++)
    data.playMove(i, board, hist);
  check(GameRules::hash(board.pos) == GameRules::hash(data.startBoard.pos), "the moves before the loop replay");
  int checkedPositions = 0;
  size_t firstInView = 0;
  for(size_t t = 0; t + start < numMoves; t++) {
    const size_t i = start + t;
    const string& view = data.moveView(i);
    vector<string> cover = RootViews::cover(board);
    bool inCover = false;
    for(const string& v : cover)
      inCover = inCover || v == view;
    check(inCover, "game " + to_string(gameIdx) + " move " + to_string(i) + ": its view '" + view + "' in its cover");
    const bool more = cover.size() > 1;
    const bool cheap = data.wasCheapSearchByTurn[t];
    counts.actions++;
    counts.inView += !view.empty();
    (more ? (cheap ? counts.cheapMore : counts.fullMore) : (cheap ? counts.cheapOne : counts.fullOne))++;
    if(!cheap)
      check(data.policyTargetsByTurn[t].unreducedNumVisits == maxVisits,
            "game " + to_string(gameIdx) + " move " + to_string(i) + ": a full search of " +
            to_string(data.policyTargetsByTurn[t].unreducedNumVisits) + " visits, not the config's " +
            to_string(maxVisits));
    //The turn's policy target: legal actions of the position in the move's view.
    Board viewed = board;
    FinishedGameData::setBoardView(viewed, view);
    if(data.policyTargetsByTurn[t].policyTargets != NULL)
      for(const PolicyTargetMove& m : *data.policyTargetsByTurn[t].policyTargets)
        check(GameRules::is_legal(viewed.pos, Board::actionOfMoveLoc(m.loc)),
              "game " + to_string(gameIdx) + " move " + to_string(i) + ": a policy target outside its view");
    check(GameRules::is_legal(viewed.pos, Board::actionOfMoveLoc(data.endHist.moveHistory[i].loc)),
          "game " + to_string(gameIdx) + " move " + to_string(i) + ": the move in its view");
    if(more && checkedPositions < 2) {
      counts.viewActions += checkViewActions(board.pos, cover);
      counts.viewPositions++;
      checkedPositions++;
    }
    if(firstInView == 0 && !view.empty()) {
      firstInView = i + 1;
      if(i > 0) {  //the tracker leaves out a state without moves before it, the game's first
        checkTreeCandidate(board, viewed, hist,
                           Rgsc::Context::of(data.endHist.rules.komi, data.playoutDoublingAdvantagePla,
                                             data.playoutDoublingAdvantage),
                           "game " + to_string(gameIdx) + " move " + to_string(i));
        counts.trackerChecked++;
      }
    }
    data.playMove(i, board, hist);
  }
  check(GameRules::hash(board.pos) == GameRules::hash(data.endHist.getRecentBoard(0).pos),
        "game " + to_string(gameIdx) + " replays to its end");
  //RGSC's played position after the first move in a view.
  if(firstInView > 0 && firstInView < numMoves) {
    const Rgsc::Context context = Rgsc::Context::of(
      data.endHist.rules.komi, data.playoutDoublingAdvantagePla, data.playoutDoublingAdvantage);
    const Rgsc::Entry entry = Rgsc::playedEntry(data.endHist, data.moveViews, firstInView, context);
    Board restored;
    string why;
    const bool suc = Rgsc::restoreState(entry, restored, why);
    check(suc, "RGSC's played entry restores: " + why);
    Board replayed = data.startHist.initialBoard;
    BoardHistory replayHist(replayed, data.startHist.initialPla, data.startHist.rules, 0);
    for(size_t i = 0; i < firstInView; i++)
      data.playMove(i, replayed, replayHist);
    check(suc && GameRules::hash(restored.pos) == GameRules::hash(replayed.pos),
          "RGSC's played entry after a move in a view is the replayed position");
    counts.rgscChecked++;
  }
  printf("game %d: %d actions after %d before the loop, %d in another view\n", gameIdx,
         (int)(numMoves - start), (int)start, (int)std::count_if(data.moveViews.begin(), data.moveViews.end(),
                                                                [](const string& v) { return !v.empty(); }));
}

int main(int argc, char** argv) {
  if(argc < 4 || argc > 6) {
    fprintf(stderr, "usage: testselfplayviews SELFPLAY_CONFIG MODEL GAMES [OVERRIDES [OUT_DIR]]\n");
    return 2;
  }
  const int numGames = atoi(argv[3]);
  if(numGames <= 0) {
    fprintf(stderr, "GAMES must be positive\n");
    return 2;
  }
  if(!GameInterface::hasViews<GameRules>()) {
    fprintf(stderr, "the game %s has no views of its own\n", GameRules::NAME);
    return 2;
  }
  const string outDir = argc == 6 ? argv[5] : "";
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg(argv[1]);
  if(argc >= 5 && string(argv[4]) != "")
    cfg.overrideKeys(ConfigParser::parseCommaSeparated(argv[4]));
  cfg.overrideKeys(map<string, string>{{"rootViews", "true"}, {"reduceVisits", "false"}, {"numSearchThreads", "1"},
                                       {"normalAsymmetricPlayoutProb", "0"}, {"handicapAsymmetricPlayoutProb", "0"}});
  Logger logger(nullptr, false, true, false, false);
  Rand seedRand("testselfplayviews");
  const SearchParams baseParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
  const PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, false);
  const int inputsVersion = cfg.contains("inputsVersion") ? cfg.getInt("inputsVersion", 0, 10000)
    : NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  GameRunner* gameRunner = new GameRunner(cfg, "testselfplayviews games", playSettings, logger);
  Setup::initializeSession(cfg);
  NNEvaluator* nnEval;
  int nnLen;
  {
    const GameInitializer* init = gameRunner->getGameInitializer();
    nnLen = init->getMaxBoardXSize();
    nnEval = Setup::initializeNNEvaluator(argv[2], argv[2], "", cfg, logger, seedRand, 1, init->getMaxBoardXSize(),
                                          init->getMaxBoardYSize(), Setup::MaxBatchSizeRequest::requireFromConfig(),
                                          true, false, Setup::SETUP_FOR_OTHER);
  }
  printf("game %s, %d visits (%d in a cheap search), member %d\n", GameRules::NAME, (int)baseParams.maxVisits,
         playSettings.cheapSearchVisits, baseParams.diversityMember);

  //A member 1 to 4 searches with its term, from a fixed state: member i's psi at statistic s is 0.1 i ((s mod 7) - 3),
  //each statistic's mean 0 and standard deviation 1, l0 1 (runenclosurerowtests' AZDB_TEST_PSI).
  std::shared_ptr<const AZdb::State<GameEncoding>> state;
  std::function<void(const MatchPairer::BotSpec&, Search*)> attach = nullptr;
  if(baseParams.diversityMember != 0) {
    const int n = GameInterface::diversityStatistics<GameEncoding>();
    vector<double> psi((size_t)AZdb::NUM_MEMBERS * n), mean(n, 0.0), sd(n, 1.0);
    for(int i = 0; i < AZdb::NUM_MEMBERS; i++)
      for(int s = 0; s < n; s++)
        psi[(size_t)i * n + s] = 0.1 * i * ((s % 7) - 3);
    state = std::make_shared<AZdb::State<GameEncoding>>(psi.data(), mean.data(), sd.data(), 1.0);
    attach = [state](const MatchPairer::BotSpec&, Search* search) { search->setDiversityState(state); };
  }

  //Declared before the writers, whose callbacks refer to them, so that they outlive the writers.
  std::ofstream noFile;  //never opened: the second writer's text goes nowhere
  RowCounts rowCounts, zeroRowCounts;
  ZeroCounts zeroCounts;
  set<int> zeroTurns;
  std::unique_ptr<TrainingDataWriter> writer, zeroWriter;
  std::ofstream sgfOut;
  if(outDir != "") {
    writer.reset(new TrainingDataWriter(outDir, inputsVersion, 10000, 1.0, nnLen, nnLen, "testselfplayviews rows"));
    writer->beforeWrite = [&](const TrainingWriteBuffers& b) { checkRows(b, set<int>(), rowCounts, "the rows"); };
    sgfOut.open(outDir + "/games.sgfs");
    zeroWriter.reset(new TrainingDataWriter(&noFile, inputsVersion, 10000, 1.0, nnLen, nnLen, 1,
                                            "testselfplayviews rows with targets of 0"));
    zeroWriter->beforeWrite = [&](const TrainingWriteBuffers& b) {
      checkRows(b, zeroTurns, zeroRowCounts, "the rows with next policies of target 0");
    };
  }

  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = nnEval->getModelName();
  botSpec.nnEval = nnEval;
  botSpec.baseParams = baseParams;
  ForkData* forkData = new ForkData();
  Counts c;
  for(int g = 0; g < numGames; g++) {
    FinishedGameData* data = gameRunner->runGame("testselfplayviews game " + to_string(g), botSpec, botSpec, forkData,
                                                 NULL, logger, nullptr, nullptr, nullptr, attach, nullptr);
    check(data != NULL, "the game ends");
    if(data == NULL)
      break;
    readGame(g, *data, baseParams.maxVisits, c);
    if(writer != nullptr) {
      data->diversityMembers[0] = data->diversityMembers[1] = baseParams.diversityMember;
      data->diversityState = state;
      writer->writeGame(*data);
      WriteSgf::writeSgf(sgfOut, data->bName, data->wName, data->endHist, data, false, true);
      sgfOut << endl;
      //After the game's rows and record: the game again, with next policies of target 0 in the row's view. It has no
      //side positions here, so each row at a turn of zeroTurns is a row of the game's own turns.
      zeroTurns = zeroNextPolicies(*data, zeroCounts);
      vector<SidePosition*> sidePositions;
      sidePositions.swap(data->sidePositions);
      zeroWriter->writeGame(*data);
      zeroWriter->flushIfNonempty();
      sidePositions.swap(data->sidePositions);
    }
    delete data;
  }
  if(writer != nullptr) {
    writer->flushIfNonempty();
    printf("rows: %lld, with the second policy target in the row's view %lld, without it %lld\n",
           (long long)writer->numRowsWritten(), (long long)writer->numRowsNextPolicyTranslated(),
           (long long)writer->numRowsNextPolicyLost());
    printf("rows checked for policy targets of sum 0 with a weight: %lld. Next policies with only moves of target 0 "
           "in the row's view: %lld in the games, %lld made; the second writer's rows at their turns %lld of %lld, "
           "without the second policy target %lld\n",
           (long long)rowCounts.rows, (long long)zeroCounts.found, (long long)zeroCounts.made,
           (long long)zeroRowCounts.zeroRows, (long long)zeroRowCounts.rows,
           (long long)zeroWriter->numRowsNextPolicyLost());
    check(rowCounts.rows == writer->numRowsWritten(), "the check sees each row");
    check(zeroRowCounts.zeroRows > 0, "a row whose next policy keeps only moves of target 0");
    check(zeroWriter->numRowsNextPolicyLost() >= zeroRowCounts.zeroRows,
          "the writer counts each row whose next policy keeps only moves of target 0 as a row without it");
  }
  printf("actions: %lld in the loops (%lld before them), %lld in another view than the rules'; covers of more views: "
         "%lld full and %lld cheap searches (of one view: %lld full, %lld cheap)\n",
         (long long)c.actions, (long long)c.before, (long long)c.inView, (long long)c.fullMore, (long long)c.cheapMore,
         (long long)c.fullOne, (long long)c.cheapOne);
  printf("viewAction: %lld actions compared on %lld positions; RGSC's entries checked: %lld played, %lld of the tree\n",
         (long long)c.viewActions, (long long)c.viewPositions, (long long)c.rgscChecked, (long long)c.trackerChecked);
  check(c.inView > 0, "a move in another view than the rules'");
  check(c.fullMore > 0, "a full search with more views");

  delete forkData;
  delete gameRunner;
  delete nnEval;
  ScoreValue::freeTables();
  if(failures > 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("testselfplayviews: all checks pass\n");
  return 0;
}
