#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../program/rgsc.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <chrono>
#include <cmath>
#include <csignal>

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


//Enclosure fork (owner, 2026-10-04): RGSC after a game (program/rgsc.h): each turn's regret target (Eq. 2); the
//start state's EMA update after a restart, or, after a game from the normal start, its candidate offered to the
//buffer (the highest regret ranking among the tracker's tree states and the played positions, later positions
//first, as the authors' code). Returns whether to write the game: not when its start state left the buffer
//meanwhile (the authors' code drops such games) nor during the warm-up.
static bool rgscAfterGame(
  Rgsc::Buffer& buffer, const Rgsc::Settings& settings, FinishedGameData* gameData,
  bool restarted, uint64_t restartKey, const Rgsc::TreeTracker* tracker
) {
  const vector<float>& selected = gameData->selectedWhiteWinLossByTurn;
  const bool hasRegret = !selected.empty();
  if(hasRegret) {
    const Player winner = gameData->endHist.winner;
    const double whiteOutcome = winner == P_WHITE ? 1.0 : winner == P_BLACK ? -1.0 : 0.0;
    gameData->regretByTurn = Rgsc::regretsOfGame(selected, whiteOutcome);
  }
  //A model exported before the heads were added gives no rankings: the game still gets its regret targets (so the
  //heads train on them), but offers nothing and does not count toward the warm-up, which starts with the first
  //model that has the heads.
  const bool modelHasHeads = !gameData->regretRankingByTurn.empty() && !std::isnan(gameData->regretRankingByTurn[0]);
  if(!modelHasHeads && !restarted)
    return true;
  bool keep = true;
  if(restarted) {
    if(!hasRegret || !buffer.updateAfterRestart(restartKey, gameData->regretByTurn[0])) {
      keep = false;
      buffer.countDroppedRestart();
    }
  }
  else if(
    tracker != NULL && hasRegret && gameData->usedInitialPosition == 0 && gameData->mode == FinishedGameData::MODE_NORMAL &&
    gameData->regretRankingByTurn.size() == selected.size()
  ) {
    const vector<Move>& allMoves = gameData->endHist.moveHistory;
    const size_t start = gameData->startHist.moveHistory.size();
    testAssert(allMoves.size() >= start + selected.size());
    bool found = tracker->found();
    float best = tracker->ranking();
    double regret = found ? std::max(buffer.floorRegret(), (double)tracker->regretValue()) : 0.0;
    uint64_t key = tracker->key();
    vector<Move> moves = found ? tracker->moves() : vector<Move>();
    for(size_t k = selected.size(); k-- > 0; ) {
      const size_t length = start + k;  //the position the move of turn k was searched in
      const float ranking = gameData->regretRankingByTurn[k];
      if(length == 0 || !(ranking > best))
        continue;
      const uint64_t positionKey = Rgsc::keyOf(allMoves, length);
      if(buffer.contains(positionKey))
        continue;
      found = true;
      best = ranking;
      regret = gameData->regretByTurn[k];
      key = positionKey;
      moves.assign(allMoves.begin(), allMoves.begin() + length);
    }
    if(found)
      buffer.offer(key, regret, moves);
  }
  const int64_t finishedBefore = buffer.countFinishedGame();
  if(finishedBefore < settings.warmupGames)
    keep = false;
  if((finishedBefore + 1) % 50 == 0)
    buffer.save();
  return keep;
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
  Rgsc::Buffer* rgscBuffer = NULL;
  if(rgscSettings.enabled) {
    rgscBuffer = new Rgsc::Buffer(rgscSettings);
    rgscBuffer->load();
    logger.write("RGSC: buffer of " + Global::uint64ToString(rgscBuffer->size()) + " states loaded from " + rgscSettings.bufferFile);
  }
  bool autoCleanupAllButLatestIfUnused = true;
  SelfplayManager* manager = new SelfplayManager(maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);

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
     &modelsDir,&outputDir,&logger,&cfg,numGameThreads,
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
    &rgscSettings,
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
        Rgsc::Entry rgscEntry;
        bool rgscRestarted = false;
        Sgf::PositionSample rgscStart;
        std::unique_ptr<Rgsc::TreeTracker> rgscTracker;
        std::function<void(const MatchPairer::BotSpec&, Search*)> rgscAttach = nullptr;
        if(rgscBuffer != NULL) {
          if(rgscBuffer->maybeSample(thisLoopSeedRand, rgscEntry)) {
            rgscRestarted = true;
            rgscStart.board = Board(maxBoardXSizeUsed, maxBoardYSizeUsed);
            rgscStart.nextPla = P_BLACK;
            rgscStart.moves = rgscEntry.moves;
            rgscStart.initialTurnNumber = 0;
            rgscStart.hintLoc = Board::NULL_LOC;
            rgscStart.weight = 1.0;
            rgscStart.trainingWeight = 1.0;
            rgscStart.exactStart = true;
          }
          else {
            rgscTracker = std::make_unique<Rgsc::TreeTracker>(rgscBuffer);
            Rgsc::TreeTracker* tracker = rgscTracker.get();
            rgscAttach = [tracker](const MatchPairer::BotSpec&, Search* search) { search->rgscTracker = tracker; };
          }
        }
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, rgscRestarted ? NULL : forkData, rgscRestarted ? &rgscStart : NULL, logger,
          shouldStopFunc,
          shouldPause,
          (switchNetsMidGame ? checkForNewNNEval : nullptr),
          rgscAttach,
          nullptr
        );
        if(gameData != NULL && rgscBuffer != NULL) {
          bool keep = rgscAfterGame(*rgscBuffer, rgscSettings, gameData, rgscRestarted, rgscEntry.key, rgscTracker.get());
          if(!keep) {
            delete gameData;
            gameData = NULL;
            //Not written, but the loop goes on (a NULL game otherwise means stop).
            manager->release(nnEval);
            continue;
          }
        }
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
    Logger::logThreadUncaught("game loop", &logger, [&](){ gameLoop(threadIdx); });
  };

  //Looping thread for polling for new neural nets and loading them in
  std::mutex modelLoadMutex;
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&modelLoadMutex,&modelLoadSleepVar,&logger,&manager,&loadLatestNeuralNetIntoManager]() {
    logger.write("Model loading loop thread starting");

    while(true) {
      if(shouldStop.load())
        break;
      string lastNetName = manager->getLatestModelName();
      bool success = loadLatestNeuralNetIntoManager(&lastNetName);
      (void)success;

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
