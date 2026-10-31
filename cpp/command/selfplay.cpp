#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/threadaffinity.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../program/rgsc.h"
#include "../program/azdbleague.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <limits>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

//-----------------------------------------------------------------------------------------


//Enclosure fork: one buffer's candidate from a game from the normal start: the tracker's tree state (its regret from
//the source's value head, at least the buffer's lowest) or a played position that ranks higher (its regret from the
//game), later positions first, as the authors' code; states already in the buffer are skipped.
//KataGame fork (2026-10-27): in a game that keeps states, the played positions are the game's states, replayed from
//its first position, with the game's context (its komi and asymmetric playouts).
static void rgscOfferCandidate(
  Rgsc::Buffer& buffer, const Rgsc::TreeTracker* tracker, const vector<float>& rankingByTurn,
  const vector<float>& regretByTurn, const FinishedGameData& gameData
) {
  const BoardHistory& hist = gameData.endHist;
  const size_t start = gameData.startHist.moveHistory.size();
  testAssert(hist.moveHistory.size() >= start + regretByTurn.size());
  const Rgsc::Context context = Rgsc::Context::of(
    hist.rules.komi, gameData.playoutDoublingAdvantagePla, gameData.playoutDoublingAdvantage);
  const vector<uint64_t> keys = Rgsc::playedKeys(hist, start, regretByTurn.size(), context);
  bool found = tracker->found();
  float best = tracker->ranking();
  double regret = found ? std::max(buffer.floorRegret(), (double)tracker->regretValue()) : 0.0;
  size_t played = 0;  //the length of the played position chosen, 0 for the tracker's state
  for(size_t k = regretByTurn.size(); k-- > 0; ) {
    const size_t length = start + k;  //the position the move of turn k was searched in
    const float ranking = rankingByTurn[k];
    if(length == 0 || !(ranking > best))
      continue;
    if(buffer.contains(keys[k]))
      continue;
    found = true;
    best = ranking;
    regret = regretByTurn[k];
    played = length;
  }
  if(!found)
    return;
  //KataGame fork (2026-10-29): the side of the search that produced the state (the tracker's, or a played position's
  //mover), and that search's AZdb member, the game's member of that side.
  Rgsc::Entry entry;
  if(played == 0) {
    entry = tracker->candidate();
    entry.origin = Rgsc::ORIGIN_TREE;
  }
  else {
    entry = Rgsc::playedEntry(hist, played, context);
    entry.origin = Rgsc::ORIGIN_PLAYED;
    entry.side = Board::indexOfPlayer(hist.moveHistory[played].pla);
  }
  entry.regret = regret;
  entry.member = gameData.diversityMembers[entry.side];
  buffer.offer(std::move(entry));
}

//Enclosure fork (owner, 2026-10-04): RGSC after a game (program/rgsc.h): each turn's regret target (Eq. 2); the
//start state's EMA update after a restart, or, after a game from the normal start, its candidate offered to the
//buffer (the highest regret ranking among the tracker's tree states and the played positions, later positions
//first, as the authors' code). Returns whether to write the game: not during the warm-up, nor a restart that
//searched no move. A restart whose start state left the buffer meanwhile is written (the authors' code drops it;
//see below).
//Enclosure fork (2026-10-11): the swing source. Each turn's swing regret target too, whenever RGSC is on; after a
//restart from the swing buffer, the EMA update from the start state's swing regret; and, when the swing buffer
//exists, a second candidate from a game from the normal start, by the swing ranking. The warm-up follows the
//vanilla buffer only.
//KataGame fork (2026-10-27): no swing targets with rgscSwingScale 0; each game's regret targets by kind of action
//(Rgsc::Buffer::addRegrets); and the buffer's summary in the log at each save.
static bool rgscAfterGame(
  Rgsc::Buffer& buffer, Rgsc::Buffer* swingBuffer, const Rgsc::Settings& settings, FinishedGameData* gameData,
  Rgsc::Source source, uint64_t restartKey, const Rgsc::TreeTracker* tracker, const Rgsc::TreeTracker* swingTracker,
  Logger& logger
) {
  const vector<float>& selected = gameData->selectedWhiteWinLossByTurn;
  const bool hasRegret = !selected.empty();
  const bool restarted = source != Rgsc::SOURCE_NONE;
  gameData->rgscSource = (int)source;
  if(hasRegret) {
    const Player winner = gameData->endHist.winner;
    const double whiteOutcome = winner == P_WHITE ? 1.0 : winner == P_BLACK ? -1.0 : 0.0;
    gameData->regretByTurn = Rgsc::regretsOfGame(selected, whiteOutcome);
    if(settings.swingScale > 0)
      gameData->swingByTurn =
        Rgsc::swingsOfGame(selected, whiteOutcome, settings.swingGamma, settings.swingScale, settings.swingCap);
    buffer.addRegrets(gameData->regretByTurn, gameData->continuesMoveByTurn);
  }
  //A model exported before the heads were added gives no rankings: the game still gets its regret targets (so the
  //heads train on them), but offers nothing and does not count toward the warm-up, which starts with the first
  //model that has the heads.
  const bool modelHasHeads = !gameData->regretRankingByTurn.empty() && !std::isnan(gameData->regretRankingByTurn[0]);
  if(!modelHasHeads && !restarted)
    return true;
  bool keep = true;
  if(restarted) {
    //Enclosure fork (2026-10-10), a departure from the authors' code, which drops a restarted game whose start
    //state left the buffer meanwhile. With 384 games in flight against one buffer (the authors run 32), about two
    //thirds of the restarts were dropped (games/enclosure/docs/decisions/rgsc.md, "Why the share of restarted games fell").
    //Such a game is ordinary self-play from a state that was in the buffer when it was drawn, so it is written.
    //Only its EMA update is skipped: its state is gone.
    const bool fromSwing = source == Rgsc::SOURCE_SWING;
    testAssert(!fromSwing || swingBuffer != NULL);
    Rgsc::Buffer& from = fromSwing ? *swingBuffer : buffer;
    if(!hasRegret) {
      keep = false;
      from.countDroppedRestart();
    }
    else if(!from.updateAfterRestart(restartKey, fromSwing ? gameData->swingByTurn[0] : gameData->regretByTurn[0]))
      from.countRestartStateGone();
  }
  else if(
    tracker != NULL && hasRegret && gameData->usedInitialPosition == 0 && gameData->mode == FinishedGameData::MODE_NORMAL &&
    gameData->regretRankingByTurn.size() == selected.size()
  ) {
    rgscOfferCandidate(buffer, tracker, gameData->regretRankingByTurn, gameData->regretByTurn, *gameData);
    //A model without the swing heads ranks every state NaN, so it offers nothing to the swing buffer.
    if(swingBuffer != NULL && swingTracker != NULL && gameData->swingRankingByTurn.size() == selected.size())
      rgscOfferCandidate(*swingBuffer, swingTracker, gameData->swingRankingByTurn, gameData->swingByTurn, *gameData);
  }
  const int64_t finishedBefore = buffer.countFinishedGame();
  if(swingBuffer != NULL)
    swingBuffer->countFinishedGame();
  if(finishedBefore < settings.warmupGames)
    keep = false;
  if((finishedBefore + 1) % 50 == 0) {
    buffer.save();
    logger.write(buffer.summary());
    if(swingBuffer != NULL) {
      swingBuffer->save();
      logger.write(swingBuffer->summary());
    }
  }
  return keep;
}

//KataGame fork (2026-10-29): AZdb in self-play (program/azdbleague.h; the engine repository's
//games/blueprints/docs/research/diversity-players.md, "The matchmaker and one trainer"). With diversityMembers =
//AZdb::NUM_MEMBERS, each game's members come from PSRO-NASH, each player's search plays its member
//(SearchParams::diversityMember), only the chooser's rows train, and the league (the table of results, ψ, μ and σ)
//lives in diversityStateFile, which a restart reads again. A game that starts from a state of RGSC's buffer, either
//buffer, is member 0 against itself, and the league takes nothing from it (AZdb::restartMatch). Without it
//(diversityMembers = 1, the default), member 0 plays alone, as before.
//- diversityStateFile: the league's file. diversitySeedFile: the seed of μ and σ (games/blueprints/runs/
//  diversity_seed.json), needed while the league's file does not exist.
//- diversityL0: ℓ0, the distance at which a member's reward is 0.
//- diversityRewardScale: the reward's scale k, a multiplier of every member's reward (AZdb::State), in the rows'
//  targets and in the search's node rewards alike. 1 by default, the design's reward. Like ℓ0, the setting holds also
//  after a restart, and the league's file records it for the searches outside self-play (AZdb::readState).
//- diversityGamesFile: a line for each written game, for the analysis (AZdb::gameRecord, and the game's start:
//  rgsc_source as the rows' C74, and after a restart the member and side that produced its state, Rgsc::Entry).
//  Default: OUTPUT_DIR/azdb-games.jsonl.
//- The search's λ of members 1 to 4 is the search parameter diversityLambda.
struct AZdbSettings {
  bool enabled = false;
  string stateFile;
  string seedFile;
  string gamesFile;
  double l0 = 0.0;
  double rewardScale = 1.0;

  static AZdbSettings load(ConfigParser& cfg, const string& outputDir, int inputsVersion) {
    AZdbSettings s;
    const int members = cfg.contains("diversityMembers") ? cfg.getInt("diversityMembers", 1, AZdb::NUM_MEMBERS) : 1;
    if(members == 1)
      return s;
    //The members' state (AZdb::State) holds every member of the latent, and each member's nearest member is among
    //them all.
    if(members != AZdb::NUM_MEMBERS)
      throw StringError("AZdb: diversityMembers is 1 (member 0 alone) or " + Global::intToString(AZdb::NUM_MEMBERS) +
                        " (every member of the latent)");
    s.enabled = true;
    if(inputsVersion < 14)
      throw StringError("AZdb: the rows of the members hold their latent from the inputs version 14, not " +
                        Global::intToString(inputsVersion) + " (inputsVersion)");
    s.stateFile = cfg.getString("diversityStateFile");
    s.seedFile = cfg.contains("diversitySeedFile") ? cfg.getString("diversitySeedFile") : "";
    s.gamesFile = cfg.contains("diversityGamesFile") ? cfg.getString("diversityGamesFile") : outputDir + "/azdb-games.jsonl";
    s.l0 = cfg.getDouble("diversityL0", 0.0, 1e9);
    if(!(s.l0 > 0.0))
      throw StringError("AZdb: diversityL0 is positive");
    if(cfg.contains("diversityRewardScale")) {
      s.rewardScale = cfg.getDouble("diversityRewardScale", 0.0, 1e9);
      if(!(s.rewardScale > 0.0))
        throw StringError("AZdb: diversityRewardScale is positive");
    }
    return s;
  }
};

//The league of a run's AZdb members, from its file, or from the seed when the file does not exist yet.
static AZdb::League<GameEncoding>* loadAZdbLeague(const AZdbSettings& settings, Logger& logger) {
  constexpr int n = AZdb::League<GameEncoding>::N;
  vector<double> mean(n, 0.0), sd(n, 1.0);
  const bool resumes = FileUtils::exists(settings.stateFile);
  if(!resumes) {
    if(settings.seedFile == "")
      throw StringError("AZdb: " + settings.stateFile + " does not exist, and diversitySeedFile gives no seed");
    AZdb::readSeed(settings.seedFile, n, mean, sd);
  }
  AZdb::League<GameEncoding>* league =
    new AZdb::League<GameEncoding>(settings.l0, mean.data(), sd.data(), settings.rewardScale);
  league->load(settings.stateFile);
  logger.write(string("AZdb: ") + (resumes ? "the league read again from " + settings.stateFile :
                                   "a new league from the seed " + settings.seedFile) + "; " + league->summary());
  return league;
}

//Refuses a model that cannot play AZdb's members: one without their latent (the inputs version 14) or without the
//intrinsic value head, or a backend that does not compute the head (it writes NaN there).
static void checkAZdbModel(const NNEvaluator* nnEval) {
  const string name = nnEval->getModelName();
  const int netInputsVersion = NNModelVersion::getInputsVersion(nnEval->getModelVersion());
  if(netInputsVersion < 14)
    throw StringError("AZdb: the model " + name + " reads the inputs version " + Global::intToString(netInputsVersion) +
                      ", without the members' latent (version 14: add_diversity_members.py)");
  if(!nnEval->hasIntrinsicHead())
    throw StringError("AZdb: the model " + name + " has no intrinsic value head (add_diversity_members.py)");
  if(!nnEval->computesIntrinsicHead())
    throw StringError("AZdb: this backend does not compute the intrinsic value head of the model " + name +
                      " (TensorRT, ONNX, OpenCL and Metal write NaN there): play AZdb's members with the CUDA or "
                      "Eigen backend");
}

//The lines of AZdb's games for the analysis, shared by the game threads.
struct AZdbGamesLog {
  std::mutex mutex;
  ofstream out;
};

//AZdb after a written game: the league takes the game's sides and result (nothing from a restart), the game's line
//goes to the log of games with its start (the rows' rgscSource, and a restart's restartEntry), and every 50 games the
//league is saved.
static void azdbAfterGame(
  AZdb::League<GameEncoding>& league, const AZdbSettings& settings, AZdbGamesLog& gamesLog, const AZdb::Match& match,
  const FinishedGameData& gameData, const Rgsc::Entry& restartEntry, Logger& logger
) {
  //The game's positions, one action apart from its start.
  vector<Board> boards;
  {
    Board board(gameData.startBoard);
    BoardHistory hist(gameData.startHist);
    boards.push_back(board);
    for(size_t t = gameData.startHist.moveHistory.size(); t < gameData.endHist.moveHistory.size(); t++) {
      const Move& move = gameData.endHist.moveHistory[t];
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
      boards.push_back(board);
    }
  }
  vector<double> side[2];
  AZdb::sideStatistics<GameEncoding>(
    [&](int j) -> const GameRules::Position& { return boards[j].pos; }, (int)boards.size() - 1, side);
  const BoardHistory& end = gameData.endHist;
  const double firstScore = !end.isGameFinished ? std::numeric_limits<double>::quiet_NaN()
    : end.winner == P_BLACK ? 1.0 : end.winner == P_WHITE ? 0.0 : 0.5;
  const int64_t games = league.finishGame(match, side, *gameData.diversityState, firstScore);
  {
    nlohmann::json record = AZdb::gameRecord<GameEncoding>(match, end.rules.komi, firstScore, side);
    record["rgsc_source"] = gameData.rgscSource;
    if(match.restart) {
      record["start_member"] = restartEntry.member;
      record["start_side"] = restartEntry.side;
    }
    const string line = record.dump();
    std::lock_guard<std::mutex> lock(gamesLog.mutex);
    gamesLog.out << line << "\n";
    gamesLog.out.flush();
  }
  if(games % 50 == 0) {
    league.save(settings.stateFile);
    logger.write(league.summary());
  }
}

int MainCmds::selfplay(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelsDir;
  string outputDir;
  int64_t maxGamesTotal = ((int64_t)1) << 62;
  try {
    KataGoCommandLine cmd("Generate training data via self play.");
    cmd.addConfigFileArg("","");
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> modelsDirArg("","models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");
    TCLAP::ValueArg<string> maxGamesTotalArg("","max-games-total","Terminate after this many games",false,string(),"NGAMES");
    cmd.add(modelsDirArg);
    cmd.add(outputDirArg);
    cmd.add(maxGamesTotalArg);
    cmd.parseArgs(args);

    modelsDir = modelsDirArg.getValue();
    outputDir = outputDirArg.getValue();
    string maxGamesTotalStr = maxGamesTotalArg.getValue();
    if(maxGamesTotalStr != "") {
      bool suc = Global::tryStringToInt64(maxGamesTotalStr,maxGamesTotal);
      if(!suc || maxGamesTotal <= 0)
        throw StringError("-max-games-total must be a positive integer");
    }

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("models-dir",modelsDir);
    checkDirNonEmpty("output-dir",outputDir);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outputDir);
  MakeDir::make(modelsDir);

  Logger logger(&cfg);
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(outputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");

  logger.write("Self Play Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  //Enclosure fork (owner, 2026-10-05): optional CPU lists for the NN server threads and the game threads, e.g.
  //"12-15" and "0-11" (core/threadaffinity.h); set before any NNEvaluator spawns its server threads.
  if(cfg.contains("nnServerThreadCpus"))
    ThreadAffinity::setServerCpus(cfg.getString("nnServerThreadCpus"));
  if(cfg.contains("gameThreadCpus"))
    ThreadAffinity::setGameCpus(cfg.getString("gameThreadCpus"));
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Width and height of the board to use when writing data, typically 19
  const int dataBoardLen = cfg.getInt("dataBoardLen",3,Board::MAX_LEN);
  const int inputsVersion =
    cfg.contains("inputsVersion") ?
    cfg.getInt("inputsVersion",0,10000) :
    NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  //Max number of games that we will allow to be queued up and not written out
  const int maxDataQueueSize = cfg.getInt("maxDataQueueSize",1,1000000);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);
  const double firstFileRandMinProp = cfg.getDouble("firstFileRandMinProp",0.0,1.0);

  const int64_t logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

  const bool switchNetsMidGame = cfg.getBool("switchNetsMidGame");
  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);

  //Initialize object for randomizing game settings and running games
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  //Enclosure fork (owner, 2026-10-04): RGSC's buffer, shared by the game threads, kept across runs in its file.
  const Rgsc::Settings rgscSettings = Rgsc::Settings::load(cfg);
  //KataGame fork (2026-10-29): AZdb's members (AZdbSettings).
  const AZdbSettings azdbSettings = AZdbSettings::load(cfg, outputDir, inputsVersion);
  bool autoCleanupAllButLatestIfUnused = true;
  SelfplayManager* manager = new SelfplayManager(maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);

  //KataGame fork (2026-10-27): the buffers load after the game's settings (initializeSession), as a game that keeps
  //states restores them under its settings.
  Rgsc::Buffer* rgscBuffer = NULL;
  if(rgscSettings.enabled) {
    rgscBuffer = new Rgsc::Buffer(rgscSettings);
    rgscBuffer->load();
    logger.write("RGSC: buffer of " + Global::uint64ToString(rgscBuffer->size()) + " states loaded from " +
                 rgscSettings.bufferFile + (Rgsc::keepsStates() ? " (the game's states)" : " (moves)") + "; " +
                 rgscBuffer->summary());
  }
  //Enclosure fork (2026-10-11): the swing source's buffer, when its restarts are on. Its targets are written either
  //way (rgscAfterGame).
  Rgsc::Buffer* rgscSwingBuffer = NULL;
  if(rgscSettings.enabled && rgscSettings.swingStartProb > 0) {
    rgscSwingBuffer = new Rgsc::Buffer(rgscSettings.forSwingBuffer(), "RGSC swing");
    rgscSwingBuffer->load();
    logger.write("RGSC swing: buffer of " + Global::uint64ToString(rgscSwingBuffer->size()) + " states loaded from " +
                 rgscSettings.swingBufferFile + "; restarts with probability " +
                 Global::doubleToString(rgscSettings.swingStartProb) + " once it holds " +
                 Global::intToString(rgscSettings.bufferSize) + " states");
  }
  if(rgscSettings.enabled)
    logger.write("RGSC swing targets: gamma " + Global::doubleToString(rgscSettings.swingGamma) + ", scale " +
                 Global::doubleToString(rgscSettings.swingScale) + ", cap " +
                 Global::doubleToString(rgscSettings.swingCap) +
                 (rgscSettings.swingScale > 0 ? "" : " (scale 0: no swing targets)"));
  //KataGame fork (2026-10-29): AZdb's league, shared by the game threads, kept across runs in its file, and the log of
  //its games.
  AZdb::League<GameEncoding>* azdbLeague = NULL;
  AZdbGamesLog azdbGamesLog;
  if(azdbSettings.enabled) {
    azdbLeague = loadAZdbLeague(azdbSettings, logger);
    FileUtils::open(azdbGamesLog.out, azdbSettings.gamesFile, std::ios::app);
    logger.write("AZdb: " + Global::intToString(AZdb::NUM_MEMBERS) + " members, l0 " +
                 Global::doubleToString(azdbSettings.l0) + ", the reward's scale " +
                 Global::doubleToString(azdbSettings.rewardScale) + "; the league in " + azdbSettings.stateFile +
                 ", the games' lines in " + azdbSettings.gamesFile);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting self play");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting self play" << endl;

  //Time the whole self-play run for reporting overall computational throughput.
  ClockTimer selfplayTimer;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  //Returns true if a new net was loaded.
  auto loadLatestNeuralNetIntoManager =
    [inputsVersion,&manager,maxRowsPerTrainFile,firstFileRandMinProp,dataBoardLen,
     &modelsDir,&outputDir,&logger,&cfg,numGameThreads,&azdbSettings,
     minBoardXSizeUsed,maxBoardXSizeUsed,minBoardYSizeUsed,maxBoardYSizeUsed](const string* lastNetName) -> bool {

    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool foundModel = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);

    //No new neural nets yet
    if(!foundModel || (lastNetName != NULL && *lastNetName == modelName))
      return false;
    if(modelName == "random" && lastNetName != NULL && *lastNetName != "random") {
      logger.write("WARNING: " + *lastNetName + " was the previous model, but now no model was found. Continuing with prev model instead of using random");
      return false;
    }

    logger.write("Found new neural net " + modelName);

    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const bool disableFP16 = false;
    const string expectedSha256 = "";

    Rand rand;
     NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);
    //KataGame fork (2026-10-29): AZdb's members need a model and a backend that play them.
    if(azdbSettings.enabled)
      checkAZdbModel(nnEval);

    string modelOutputDir = outputDir + "/" + modelName;
    string sgfOutputDir = modelOutputDir + "/sgfs";
    string tdataOutputDir = modelOutputDir + "/tdata";

    //Try repeatedly to make directories, in case the filesystem is unhappy with us as we try to make the same dirs as another process.
    //Wait a random amount of time in between each failure.
    int maxTries = 5;
    for(int i = 0; i<maxTries; i++) {
      bool success = false;
      try {
        MakeDir::make(modelOutputDir);
        MakeDir::make(sgfOutputDir);
        MakeDir::make(tdataOutputDir);
        success = true;
      }
      catch(const StringError& e) {
        logger.write(string("WARNING, error making directories, trying again shortly: ") + e.what());
        success = false;
      }

      if(success)
        break;
      else {
        if(i == maxTries-1) {
          logger.write("ERROR: Could not make selfplay model directories, is something wrong with the filesystem?");
          //Just give up and wait for the next model.
          return false;
        }
        double sleepTime = 10.0 + rand.nextDouble() * 30.0;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
        continue;
      }
    }

    {
      ofstream out;
      FileUtils::open(out,modelOutputDir + "/" + "selfplay-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
      out << cfg.getContents();
      out.close();
    }

    //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
    //simply controls the input feature version for the written data
    TrainingDataWriter* tdataWriter = new TrainingDataWriter(
      tdataOutputDir, inputsVersion, maxRowsPerTrainFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs");
    }

    logger.write("Model loading loop thread loaded new neural net " + nnEval->getModelName());
    manager->loadModelAndStartDataWriting(nnEval, tdataWriter, sgfOut);
    return true;
  };

  //Initialize the initial neural net
  {
    bool success = loadLatestNeuralNetIntoManager(NULL);
    if(!success)
      throw StringError("Either could not load latest neural net or access/write appopriate directories");
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  //Shared across all game loop threads
  std::atomic<int64_t> numGamesStarted(0);
  ForkData* forkData = new ForkData();
  auto gameLoop = [
    &gameRunner,
    &manager,
    &logger,
    switchNetsMidGame,
    &numGamesStarted,
    &forkData,
    maxGamesTotal,
    &baseParams,
    &gameSeedBase,
    rgscBuffer,
    rgscSwingBuffer,
    &rgscSettings,
    azdbLeague,
    &azdbSettings,
    &azdbGamesLog,
    maxBoardXSizeUsed,
    maxBoardYSizeUsed
  ](int threadIdx) {
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    string prevModelName;
    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;
      NNEvaluator* nnEval = manager->acquireLatest();
      testAssert(nnEval != NULL);

      if(prevModelName != nnEval->getModelName()) {
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new neural net: " + prevModelName);
      }

      //Callback that runGame will call periodically to ask us if we have a new neural net
      std::function<NNEvaluator*()> checkForNewNNEval = [&manager,&nnEval,&prevModelName,&logger,&threadIdx]() -> NNEvaluator* {
        NNEvaluator* newNNEval = manager->acquireLatest();
        testAssert(newNNEval != NULL);
        if(newNNEval == nnEval) {
          manager->release(newNNEval);
          return NULL;
        }
        manager->release(nnEval);

        nnEval = newNNEval;
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " changing midgame to new neural net: " + prevModelName);
        return nnEval;
      };

      FinishedGameData* gameData = NULL;

      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx < maxGamesTotal) {
        manager->countOneGameStarted(nnEval);
        MatchPairer::BotSpec botSpecB;
        botSpecB.botIdx = 0;
        botSpecB.botName = nnEval->getModelName();
        botSpecB.nnEval = nnEval;
        botSpecB.baseParams = baseParams;
        MatchPairer::BotSpec botSpecW = botSpecB;

        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        //Enclosure fork (owner, 2026-10-04): RGSC. With probability rgscStartProb the game starts exactly at a state
        //drawn from the buffer (and takes no fork position); otherwise a tracker collects its searches' candidate.
        //Enclosure fork (2026-10-11): else, with probability rgscSwingStartProb / (1 - rgscStartProb), from a state of
        //the swing buffer (when it is full), so that rgscSwingStartProb of the games start there.
        Rgsc::Entry rgscEntry;
        bool rgscRestarted = false;
        Rgsc::Source rgscSource = Rgsc::SOURCE_NONE;
        Sgf::PositionSample rgscStart;
        std::unique_ptr<Rgsc::TreeTracker> rgscTracker;
        std::unique_ptr<Rgsc::TreeTracker> rgscSwingTracker;
        std::function<void(const MatchPairer::BotSpec&, Search*)> rgscAttach = nullptr;
        if(rgscBuffer != NULL) {
          if(rgscBuffer->maybeSample(thisLoopSeedRand, rgscEntry))
            rgscSource = Rgsc::SOURCE_VANILLA;
          else if(
            rgscSwingBuffer != NULL && rgscSettings.startProb < 1.0 &&
            thisLoopSeedRand.nextBool(rgscSettings.swingStartProb / (1.0 - rgscSettings.startProb)) &&
            rgscSwingBuffer->sample(thisLoopSeedRand, rgscEntry)
          )
            rgscSource = Rgsc::SOURCE_SWING;
          //KataGame fork (2026-10-27): the entry's start (Rgsc::startOf): its moves, or in a game that keeps states the
          //state itself, with its turn number and its context frozen. A state that does not come back with its key
          //leaves the buffer, and the game starts from the normal start.
          string why;
          if(
            rgscSource != Rgsc::SOURCE_NONE &&
            !Rgsc::startOf(rgscEntry, maxBoardXSizeUsed, maxBoardYSizeUsed, rgscStart, why)
          ) {
            (rgscSource == Rgsc::SOURCE_SWING ? rgscSwingBuffer : rgscBuffer)->reject(rgscEntry.key);
            logger.write("RGSC: an entry rejected at its restart: " + why);
            rgscSource = Rgsc::SOURCE_NONE;
          }
          if(rgscSource != Rgsc::SOURCE_NONE)
            rgscRestarted = true;
          else {
            rgscTracker = std::make_unique<Rgsc::TreeTracker>(rgscBuffer);
            if(rgscSwingBuffer != NULL)
              rgscSwingTracker = std::make_unique<Rgsc::TreeTracker>(rgscSwingBuffer, Rgsc::SOURCE_SWING);
            Rgsc::TreeTracker* tracker = rgscTracker.get();
            Rgsc::TreeTracker* swingTracker = rgscSwingTracker.get();
            rgscAttach = [tracker, swingTracker](const MatchPairer::BotSpec&, Search* search) {
              search->rgscTracker = tracker;
              search->rgscSwingTracker = swingTracker;
            };
          }
        }
        //KataGame fork (2026-10-29): AZdb. The matchmaker draws the game's members, and the game takes its copy of the
        //league's state. Two members play with two bots, one search each, and each search gets its member and the
        //copy. A game that starts from a state of RGSC's buffer is member 0 against itself (AZdb::restartMatch).
        //Without AZdb nothing here draws or changes.
        AZdb::Match azdbMatch;
        std::shared_ptr<const AZdb::State<GameEncoding>> azdbState;
        std::function<void(const MatchPairer::BotSpec&, Search*)> attach = rgscAttach;
        if(azdbLeague != NULL) {
          azdbMatch = rgscRestarted ? AZdb::restartMatch() : azdbLeague->draw(thisLoopSeedRand);
          azdbState = azdbLeague->snapshot();
          botSpecB.botIdx = azdbMatch.members[Board::indexOfPlayer(P_BLACK)];
          botSpecW.botIdx = azdbMatch.members[Board::indexOfPlayer(P_WHITE)];
          //Only the chooser's rows train (both players' in a game of the chooser against itself).
          botSpecB.trains = botSpecB.botIdx == azdbMatch.chooser;
          botSpecW.trains = botSpecW.botIdx == azdbMatch.chooser;
          botSpecB.baseParams.diversityMember = botSpecB.botIdx;
          botSpecW.baseParams.diversityMember = botSpecW.botIdx;
          attach = [&rgscAttach, azdbState](const MatchPairer::BotSpec& spec, Search* search) {
            if(rgscAttach != nullptr)
              rgscAttach(spec, search);
            search->setDiversityState(azdbState);
          };
        }
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, rgscRestarted ? NULL : forkData, rgscRestarted ? &rgscStart : NULL, logger,
          shouldStopFunc,
          shouldPause,
          (switchNetsMidGame ? checkForNewNNEval : nullptr),
          attach,
          nullptr
        );
        //The rows' rewards and targets come from the game's copy of the state. Their members (each player's latent)
        //and which of them train came with the bots (Play::runGame).
        if(gameData != NULL && azdbLeague != NULL)
          gameData->diversityState = azdbState;
        if(gameData != NULL && rgscBuffer != NULL) {
          bool keep = rgscAfterGame(
            *rgscBuffer, rgscSwingBuffer, rgscSettings, gameData, rgscSource, rgscEntry.key, rgscTracker.get(),
            rgscSwingTracker.get(), logger);
          if(!keep) {
            delete gameData;
            gameData = NULL;
            //Not written, but the loop goes on (a NULL game otherwise means stop).
            manager->release(nnEval);
            continue;
          }
        }
        //KataGame fork (2026-10-29): AZdb's league takes the written games only: their positions are those of rows.
        if(gameData != NULL && azdbLeague != NULL)
          azdbAfterGame(*azdbLeague, azdbSettings, azdbGamesLog, azdbMatch, *gameData, rgscEntry, logger);
      }

      //NULL gamedata will happen when the game is interrupted by shouldStop, which means we should also stop.
      //Or when we run out of total games.
      bool shouldContinue = gameData != NULL;
      //Note that if we've gotten a newNNEval, we're actually pushing the game as data for the new one, rather than the old one!
      if(gameData != NULL)
        manager->enqueueDataToWrite(nnEval,gameData);

      manager->release(nnEval);

      if(!shouldContinue)
        break;
    }

    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };
  auto gameLoopProtected = [&logger,&gameLoop](int threadIdx) {
    ThreadAffinity::applyGame();  //Enclosure fork: gameThreadCpus, if set
    Logger::logThreadUncaught("game loop", &logger, [&](){ gameLoop(threadIdx); });
  };

  //Looping thread for polling for new neural nets and loading them in
  std::mutex modelLoadMutex;
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&modelLoadMutex,&modelLoadSleepVar,&logger,&manager,&loadLatestNeuralNetIntoManager]() {
    logger.write("Model loading loop thread starting");
    auto nextProgressLog = std::chrono::steady_clock::now();

    while(true) {
      if(shouldStop.load())
        break;
      string lastNetName = manager->getLatestModelName();
      bool success = loadLatestNeuralNetIntoManager(&lastNetName);
      (void)success;

      //Enclosure fork (2026-10-10): once a minute, the network rows evaluated by every model so far. The watchdog
      //(pipeline/scripts/watchdog_selfplay.sh) tests that this count rises. Games log only when they start or end, and at
      //high visits none ends for many minutes after a start, while a hang in the CUDA driver stops the evaluations.
      if(std::chrono::steady_clock::now() >= nextProgressLog) {
        logger.write("Progress: NN rows processed " + Global::uint64ToString(manager->getTotalNumRowsProcessed()));
        nextProgressLog = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      }

      if(shouldStop.load())
        break;

      //Sleep for a while and then re-poll
      std::unique_lock<std::mutex> lock(modelLoadMutex);
      modelLoadSleepVar.wait_for(lock, std::chrono::seconds(20), [](){return shouldStop.load();});
    }

    logger.write("Model loading loop thread terminating");
  };
  auto modelLoadLoopProtected = [&logger,&modelLoadLoop]() {
    Logger::logThreadUncaught("model load loop", &logger, modelLoadLoop);
  };

  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(gameLoopProtected,i);
  }
  std::thread modelLoadLoopThread(modelLoadLoopProtected);

  //Wait for all game threads to stop
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  //If by now somehow shouldStop is not true, set it to be true since all game threads are toast
  shouldStop.store(true);

  //Wake up the model loading thread rather than waiting for it to wake up on its own, and
  //wait for it to die.
  {
    //Lock so that we don't race where we notify the loading thread to wake when it's still in
    //its own critical section but not yet slept, and to ensure the two agree on shouldStop.
    std::lock_guard<std::mutex> lock(modelLoadMutex);
    modelLoadSleepVar.notify_all();
  }
  modelLoadLoopThread.join();

  //At this point, nothing else except possibly data write loops are running, within the selfplay manager.
  delete manager;

  //Overall self-play totals (per-model NN/data/moves breakdowns are logged above by the manager).
  logger.write("Total games: " + Global::int64ToString(numGamesStarted.load(std::memory_order_relaxed)));
  logger.write("Total selfplay runtime (seconds): " + Global::doubleToString(selfplayTimer.getSeconds()));

  //Enclosure fork (owner, 2026-10-04): RGSC's buffer, kept for the next run.
  if(rgscBuffer != NULL) {
    rgscBuffer->save();
    logger.write(rgscBuffer->summary() + "; saved to " + rgscSettings.bufferFile);
    delete rgscBuffer;
  }
  if(rgscSwingBuffer != NULL) {
    rgscSwingBuffer->save();
    logger.write(rgscSwingBuffer->summary() + "; saved to " + rgscSettings.swingBufferFile);
    delete rgscSwingBuffer;
  }
  //KataGame fork (2026-10-29): AZdb's league, kept for the next run.
  if(azdbLeague != NULL) {
    azdbLeague->save(azdbSettings.stateFile);
    logger.write(azdbLeague->summary() + "; saved to " + azdbSettings.stateFile);
    delete azdbLeague;
  }

  //Delete and clean up everything else
  NeuralNet::globalCleanup();
  delete forkData;
  delete gameRunner;
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
