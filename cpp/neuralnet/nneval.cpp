#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/test.h"
#include "../core/threadaffinity.h"
#include "../core/bitscan.h"

#include <algorithm>
#include <chrono>

using namespace std;

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    includeOwnerMap(false),
    includeAuxOutputs(false),
    symmetryFromHash(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowSpatialBits(),
    rowSpatialPacked(false),
    policyGather(),
    policyScratch(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0)
{}

NNResultBuf::~NNResultBuf() {
}

const float* NNResultBuf::spatialRowFloats(int n) {
  if(rowSpatialPacked) {
    testAssert(n <= (int)rowSpatialBits.size() * 32);
    if((int)rowSpatialBuf.size() < n)
      rowSpatialBuf.resize(n);
    NNInputs::unpackSpatialBits(rowSpatialBits.data(), rowSpatialBuf.data(), n);
  }
  return rowSpatialBuf.data();
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  ConfigParser& cfg
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   numEffectiveDevices(NeuralNet::getNumEffectiveDevices(cfg, gpuIdxByServerThr)),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_numCacheHits(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   maxRowsToSendPerBatch(maxBatchSz),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  //Enclosure fork: a policy index is an action, so the network's board is the game's largest board (NNPos).
  if(nnXLen != NNPos::MAX_BOARD_LEN || nnYLen != NNPos::MAX_BOARD_LEN)
    throw StringError("The nnEval board size must be the game's largest board, " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    modelHasAux = desc.valueHead.auxVersion >= 1;
    modelHasRegret = !desc.valueHead.regretHeads.empty();
    modelHasIntrinsic = desc.valueHead.intrinsicVersion >= 1;
    //Export stage 2: copied before createComputeContext, which may scale the model's weights for its backend.
    modelHasPolicyAux = modelHasAux && desc.policyHead.auxVersion >= 1;
    if(modelHasAux && desc.valueHead.auxVersion >= 2) {
      const ValueHeadDesc& v = desc.valueHead;
      ScoreBeliefParams& sb = scoreBeliefParams;
      sb.len = v.scoreBeliefLen;
      sb.binWidth = v.scoreBinWidth;
      sb.offsetScale = v.scoreOffsetScale;
      sb.c2 = v.sbS2Mul.outChannels;
      sb.k = v.sbS3Mul.outChannels;
      sb.activation = v.v2Activation.activation;
      sb.offW = v.sbS2OffMul.weights;
      sb.s3W = v.sbS3Mul.weights;
      sb.s3B = v.sbS3Bias.weights;
      testAssert((int)sb.offW.size() == sb.c2 && (int)sb.s3W.size() == sb.c2 * sb.k && (int)sb.s3B.size() == sb.k);
    }
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      homeDataDirOverride,
      usingFP16Mode,loadedModel,cfg
    );
    //Enclosure fork (2026-10-21): the training-only outputs only from a backend that computes them.
    modelHasAux = modelHasAux && NeuralNet::computesAuxOutputs(computeContext);
    modelHasPolicyAux = modelHasPolicyAux && modelHasAux;
    backendComputesIntrinsic = NeuralNet::computesIntrinsicHead(computeContext);
    // Snapshot postProcessParams only after createComputeContext: backends may apply
    // config-dependent transforms to the model desc there (e.g. the ONNX backend's
    // scale8 workaround multiplies outputScaleMultiplier by 8).
    postProcessParams = desc.postProcessParams;
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    modelHasAux = false;
    modelHasPolicyAux = false;
    modelHasRegret = false;
    modelHasIntrinsic = false;
    backendComputesIntrinsic = false;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  // Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  // Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();

  if(computeContext != NULL)
    NeuralNet::freeComputeContext(computeContext);
  computeContext = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  delete nnCacheTable;
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getMaxRowsToSendPerBatch() const {
  return maxRowsToSendPerBatch.load(std::memory_order_acquire);
}
void NNEvaluator::setMaxRowsToSendPerBatch(int maxRows) {
  if(maxRows <= 0 || maxRows > maxBatchSize)
    throw StringError("Invalid setting for max rows to send per batch");
  maxRowsToSendPerBatch.store(maxRows,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
  return numEffectiveDevices;
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
const std::vector<int>& NNEvaluator::getGpuIdxByServerThread() const {
  return gpuIdxByServerThread;
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
bool NNEvaluator::getRequireExactNNLen() const {
  return requireExactNNLen;
}
int NNEvaluator::getModelVersion() const {
  return modelVersion;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

int64_t NNEvaluator::getNumModelParameters() const {
  return NeuralNet::getModelDesc(loadedModel).getNumParameters();
}

bool NNEvaluator::modelHasAnyTransformerBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks();
}

bool NNEvaluator::modelHasAnyNestedBottleneckBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyNestedBottleneckBlocks();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}

bool NNEvaluator::supportsShorttermError() const {
  return modelVersion >= 9;
}

bool NNEvaluator::modelPreferPassAliveUnderSuicideRules() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferPassAliveUnderSuicideRules;
}

bool NNEvaluator::modelPreferExcludeTerritoryAdjacentToAtari() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferExcludeTerritoryAdjacentToAtari;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  currentDefaultSymmetry.store(s, std::memory_order_release);
}
bool NNEvaluator::getSymmetryFromHash() const {
  return currentSymmetryFromHash.load(std::memory_order_acquire);
}
void NNEvaluator::setSymmetryFromHash(bool b) {
  currentSymmetryFromHash.store(b, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}
uint64_t NNEvaluator::numCacheHits() const {
  return m_numCacheHits.load(std::memory_order_relaxed);
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
  m_numCacheHits.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  ThreadAffinity::applyServer();  //Enclosure fork (owner, 2026-10-05): nnServerThreadCpus, if self-play set it
  Rand rand(randSeedThisThread);

  // We're in big trouble if this raises an exception (e.g. out of GPU memory) and causes possibly
  // the only nnEval thread to die, so let the exception escape and terminate the process. But log
  // and print the error first, since on some platforms an exception escaping a thread aborts
  // without printing anything. cerr is used unconditionally rather than only when there is no
  // logger, since the logger may be temporarily disabled (e.g. the benchmark suppresses logging
  // while respawning server threads).
  try {
    nnEval->serve(rand,gpuIdxForThisThread,serverThreadIdx);
  }
  catch(const std::exception& e) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write(string("ERROR: NN server thread failed: ") + e.what());
    cerr << (string("ERROR: NN server thread failed: ") + e.what()) << endl;
    throw;
  }
  catch(...) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write("ERROR: NN server thread failed with an exception that is not a std::exception");
    cerr << "ERROR: NN server thread failed with an exception that is not a std::exception" << endl;
    throw;
  }
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

#ifdef NNEVAL_SYNCHRONOUS
  //Enclosure fork (games/enclosure/wasm): no threads; one compute handle for evaluate() to use in the calling thread.
  if(syncServerBuf != NULL)
    throw StringError("NNEvaluator::spawnServerThreads called twice in synchronous mode");
  serverThreadsIsUsingFP16.assign(1,0);
  syncRand = new Rand(randSeed + ":NNEvalSynchronous");
  if(loadedModel != NULL) {
    syncHandle = NeuralNet::createComputeHandle(
      computeContext, loadedModel, logger, maxBatchSize, requireExactNNLen, inputsUseNHWC,
      gpuIdxByServerThread.empty() ? 0 : gpuIdxByServerThread[0], 0
    );
    serverThreadsIsUsingFP16[0] = NeuralNet::isUsingFP16(syncHandle) ? 1 : 0;
  }
  syncServerBuf = new NNServerBuf(*this, loadedModel);
#ifdef ENCLOSURE_COOP_THREADS
  syncFreeLanes.push_back({syncHandle, syncServerBuf});
#endif
  return;
#endif

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  numServerThreadsStartingUp = numThreads;
  for(int i = 0; i<numThreads; i++) {
    int gpuIdxForThisThread = gpuIdxByServerThread[i];
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(
      &serveEvals,randSeedThisThread,this,gpuIdxForThisThread,i
    );
    serverThreads.push_back(thread);
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
}

void NNEvaluator::killServerThreads() {
#ifdef NNEVAL_SYNCHRONOUS
#ifdef ENCLOSURE_COOP_THREADS
  //Every lane is free here (no evaluation is in flight). The first is syncHandle and syncServerBuf, freed below.
  for(const auto& lane : syncFreeLanes)
    if(lane.first != syncHandle) {
      NeuralNet::freeComputeHandle(lane.first);
      delete lane.second;
    }
  syncFreeLanes.clear();
#endif
  if(syncHandle != NULL)
    NeuralNet::freeComputeHandle(syncHandle);
  delete syncServerBuf;
  delete syncRand;
  syncHandle = NULL;
  syncServerBuf = NULL;
  syncRand = NULL;
  serverThreadsIsUsingFP16.clear();
  return;
#endif
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();

  // Can unset now that threads are dead
  isKilled = false;

  testAssert(numOngoingEvals == 0);
  testAssert(numWaitingEvals == 0);
  testAssert(numEvalsToAwaken == 0);
}

void NNEvaluator::fillRowBufs(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  int symmetry
) const {
  const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
  //Enclosure fork (owner, 2026-10-06): symmetry, the image's row written from board, for V8 to V14 only. KataGame
  //fork: V8 to V14 are the game's inputs (NNInputs::NUM_FEATURES_SPATIAL_V8, V10 with its extended inputs, V11 with
  //its turns too, V12 with its planes of closed areas too, V13 with its labels in their canonical order, V14 with
  //AZdb's latent too).
  const bool gameInputs = inputsVersion >= 8 && inputsVersion <= 14;
  testAssert(symmetry == 0 || gameInputs);
  //V8 to V14 rows are packed (NNResultBuf::rowSpatialBits); the float row is sized only for the other versions.
  if(!gameInputs && buf.rowSpatialBuf.size() < rowSpatialLen)
    buf.rowSpatialBuf.resize(rowSpatialLen);
  const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
  if(buf.rowGlobalBuf.size() < rowGlobalLen)
    buf.rowGlobalBuf.resize(rowGlobalLen);
  const int rowMetaLen = numInputMetaChannels;
  if(buf.rowMetaBuf.size() < rowMetaLen)
    buf.rowMetaBuf.resize(rowMetaLen);

  static_assert(NNModelVersion::latestInputsVersionImplemented == 14, "");
  //Enclosure fork (owner, 2026-10-06): V8 rows packed, one bit per entry (NNResultBuf::rowSpatialBits). This replaces
  //the float row cleared by its last entries (2026-10-05): the packed row is 9.7 KB, cleared whole. V9 rows are V8's
  //with two more global inputs (2026-10-13), V10 rows V9's with the game's extended inputs (2026-10-26), V11 rows
  //V10's planes with the game's turns (2026-10-27), V12 rows V11's with the game's planes of closed areas
  //(2026-10-27), V13 rows V12's with the game's labels in their canonical order (2026-10-29), and V14 rows V13's with
  //AZdb's latent after their global inputs (2026-10-29).
  buf.rowSpatialPacked = gameInputs;
  if(gameInputs) {
    testAssert(rowSpatialLen == NNInputs::gameSpatialFeatures(inputsVersion) * NNPos::MAX_BOARD_AREA);
    const size_t words = NNInputs::gameSpatialWords(inputsVersion);
    if(buf.rowSpatialBits.size() < words)
      buf.rowSpatialBits.resize(words);
    if(inputsVersion == 14)
      NNInputs::fillRowV14Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else if(inputsVersion == 13)
      NNInputs::fillRowV13Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else if(inputsVersion == 12)
      NNInputs::fillRowV12Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else if(inputsVersion == 11)
      NNInputs::fillRowV11Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else if(inputsVersion == 10)
      NNInputs::fillRowV10Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else if(inputsVersion == 9)
      NNInputs::fillRowV9Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
    else
      NNInputs::fillRowV8Packed(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBits.data(), buf.rowGlobalBuf.data(), symmetry);
  }
  else if(inputsVersion == 3)
    NNInputs::fillRowV3(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 4)
    NNInputs::fillRowV4(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 5)
    NNInputs::fillRowV5(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 6)
    NNInputs::fillRowV6(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 7)
    NNInputs::fillRowV7(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else
    ASSERT_UNREACHABLE;

  if(rowMetaLen > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    SGFMetadata::fillMetadataRow(
      sgfMeta,
      buf.rowMetaBuf.data(),
      nextPlayer,
      board.x_size*board.y_size
    );
    buf.hasRowMeta = true;
  }
  else {
    buf.hasRowMeta = false;
  }
}

void NNEvaluator::maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx) {
  if(gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
    return;
  // Warmup currently only matters on CUDA, where cuDNN lazily compiles an SDPA execution plan per
  // batch size on first use. Other backends: nothing to warm up for now.
#if !defined(USE_CUDA_BACKEND)
  (void)serverThreadIdx;
  return;
#else
  // Only transformer models build the lazy SDPA graphs; skip the (otherwise harmless but wasteful)
  // warmup passes for plain convnets.
  if(!NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks())
    return;

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) +
      ": warming up transformer graphs for batch sizes 1.." + Global::intToString(maxBatchSize)
    );
  }

  // Empty board of the configured size, default rules/params. Outputs are discarded; we only want
  // the forward passes to trigger graph compilation for every batch size that will be seen.
  Board board(nnXLen, nnYLen);
  //Featurize the way this model expects (a no-op under Tromp-Taylorish rules, but robust if the
  //warmup rules ever change).
  BoardHistory history(
    board, P_BLACK, Rules::getTrompTaylorish(), 0,
    BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
  );
  MiscNNInputParams nnInputParams;
  SGFMetadata sgfMeta;
  const SGFMetadata* sgfMetaPtr = NULL;
  if(numInputMetaChannels > 0) {
    sgfMeta = SGFMetadata::makeDummyWarmupProfile();
    sgfMetaPtr = &sgfMeta;
  }

  // Mark the handle as warming up so the backend treats lazy-graph-compilation failures (e.g. cudnn
  // SDPA) leniently, falling back to a custom kernel instead of failing hard. Restored when done.
  bool prevIsWarmup = NeuralNet::setIsWarmup(gpuHandle, true);

  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);

  // Reusable per-row input; identical for every row since it's an empty board.
  std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
  std::vector<NNResultBuf*> resultBufs;
  ownedBufs.reserve(maxBatchSize);
  resultBufs.reserve(maxBatchSize);
  for(int i = 0; i < maxBatchSize; i++) {
    ownedBufs.push_back(std::make_unique<NNResultBuf>());
    NNResultBuf* buf = ownedBufs.back().get();
    fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
    buf->symmetry = 0;
    buf->policyOptimism = nnInputParams.policyOptimism;
    resultBufs.push_back(buf);
  }

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    std::vector<NNOutput*> outputs;
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      NNOutput* out = new NNOutput();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      out->whiteOwnerMap = NULL;
      resultBufs[row]->policyScratch.resize(NNPos::MAX_NN_POLICY_SIZE);
      out->policyDense = resultBufs[row]->policyScratch.data();  //Enclosure fork: compact policy
      outputs.push_back(out);
    }
    NeuralNet::getOutput(gpuHandle, inputBuffers, batchSize, resultBufs.data(), outputs);
    for(NNOutput* out : outputs)
      delete out;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::setIsWarmup(gpuHandle, prevIsWarmup);
#endif
}

NNEvalBenchmarkResult NNEvaluator::benchmarkPureForward(
  int numWarmups,
  int numIterations,
  const std::vector<int>& boardSizes
) {
  if(numIterations <= 0)
    throw StringError("benchmarkPureForward: numIterations must be positive");
  if(numWarmups < 0)
    throw StringError("benchmarkPureForward: numWarmups must be nonnegative");
  if(boardSizes.size() <= 0)
    throw StringError("benchmarkPureForward: no board sizes specified");
  if(debugSkipNeuralNet || loadedModel == NULL || computeContext == NULL)
    throw StringError("benchmarkPureForward requires a real neural net model");
  if(serverThreads.size() > 0)
    throw StringError("benchmarkPureForward: server threads must not be spawned");
  for(int bSize: boardSizes) {
    if(bSize < 2 || bSize > nnXLen || bSize > nnYLen)
      throw StringError("benchmarkPureForward: board size " + Global::intToString(bSize) + " out of range for NN buffer size");
    if(requireExactNNLen && (bSize != nnXLen || bSize != nnYLen))
      throw StringError("benchmarkPureForward: requireExactNNLen is set but board size " + Global::intToString(bSize) + " does not fill the NN buffer");
  }

  const int numThreadsToUse = (int)gpuIdxByServerThread.size();
  const int batchSize = maxBatchSize;
  testAssert(numThreadsToUse > 0);

  // Left populated after the benchmark returns, deliberately: callers query
  // isAnyThreadUsingFP16() afterward to report what precision was measured.
  {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if(serverThreadsIsUsingFP16.size() < (size_t)numThreadsToUse)
      serverThreadsIsUsingFP16.resize(numThreadsToUse,0);
  }

  NNEvalBenchmarkResult result;
  result.batchSize = batchSize;
  result.numThreads = numThreadsToUse;
  result.numIterations = numIterations;
  result.perThreadIterationSeconds.resize(numThreadsToUse);
  result.perThreadMedianSeconds.assign(numThreadsToUse, 0.0);
  result.perThreadNNEvalsPerSec.assign(numThreadsToUse, 0.0);
  result.sumMedianNNEvalsPerSec = 0.0;
  result.actualWallSeconds = 0.0;
  result.actualWallNNEvalsPerSec = 0.0;

  std::atomic<int> readyCount(0);
  std::atomic<bool> startFlag(false);
  std::atomic<bool> anyError(false);
  std::mutex errorMutex;
  std::exception_ptr firstError;

  // Per-thread end-of-timed-loop timestamps, so that wall time excludes compute handle
  // teardown (freeing the model is slow and an early finisher's frees can also perturb
  // other threads' tail iterations, but that perturbation is at least visible in samples).
  std::vector<std::chrono::steady_clock::time_point> threadEndTimes(numThreadsToUse);

  std::vector<std::thread> threads;
  threads.reserve(numThreadsToUse);
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    threads.emplace_back([&,threadIdx]() {
      try {
        ComputeHandle* gpuHandle = NeuralNet::createComputeHandle(
          computeContext,
          loadedModel,
          logger,
          maxBatchSize,
          requireExactNNLen,
          inputsUseNHWC,
          gpuIdxByServerThread[threadIdx],
          threadIdx
        );
        // Declared before the try so that on an exception it is destroyed after the catch frees
        // the compute handle, matching the normal teardown order below.
        std::unique_ptr<NNServerBuf> serverBuf;
        try {
          // Mirror serve(): one compute handle and one set of input buffers per server thread,
          // with the buffers allocated only after createComputeHandle has bound this thread to
          // its GPU (see serve() for why the order matters).
          serverBuf = std::make_unique<NNServerBuf>(*this, loadedModel);
          maybeWarmupComputeHandle(gpuHandle, threadIdx);
          {
            std::lock_guard<std::mutex> lock(bufferMutex);
            if((size_t)threadIdx < serverThreadsIsUsingFP16.size())
              serverThreadsIsUsingFP16[threadIdx] = NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
          }

          MiscNNInputParams nnInputParams;
          SGFMetadata sgfMeta;
          const SGFMetadata* sgfMetaPtr = NULL;
          if(numInputMetaChannels > 0) {
            sgfMeta = SGFMetadata::makeDummyWarmupProfile();
            sgfMetaPtr = &sgfMeta;
          }

          // Empty boards, cycling through the requested sizes across the rows of the batch.
          // Position content doesn't affect speed, but board size does when masking is active.
          std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
          std::vector<NNResultBuf*> resultBufs;
          ownedBufs.reserve(batchSize);
          resultBufs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            int bSize = boardSizes[row % boardSizes.size()];
            Board board(bSize, bSize);
            BoardHistory history(
              board, P_BLACK, Rules::getTrompTaylorish(), 0,
              BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
            );
            ownedBufs.push_back(std::make_unique<NNResultBuf>());
            NNResultBuf* buf = ownedBufs.back().get();
            fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
            buf->symmetry = 0;
            buf->policyOptimism = nnInputParams.policyOptimism;
            resultBufs.push_back(buf);
          }

          std::vector<std::unique_ptr<NNOutput>> ownedOutputs;
          std::vector<NNOutput*> outputs;
          ownedOutputs.reserve(batchSize);
          outputs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            ownedOutputs.push_back(std::make_unique<NNOutput>());
            NNOutput* out = ownedOutputs.back().get();
            out->nnXLen = nnXLen;
            out->nnYLen = nnYLen;
            out->whiteOwnerMap = NULL;
            resultBufs[row]->policyScratch.resize(NNPos::MAX_NN_POLICY_SIZE);
            out->policyDense = resultBufs[row]->policyScratch.data();  //Enclosure fork: compact policy
            outputs.push_back(out);
          }

          for(int i = 0; i < numWarmups; i++)
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);

          readyCount.fetch_add(1);
          while(!startFlag.load(std::memory_order_acquire))
            std::this_thread::yield();

          std::vector<double>& times = result.perThreadIterationSeconds[threadIdx];
          times.reserve(numIterations);
          for(int i = 0; i < numIterations; i++) {
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(t1-t0).count());
          }
          threadEndTimes[threadIdx] = std::chrono::steady_clock::now();
        }
        catch(...) {
          NeuralNet::freeComputeHandle(gpuHandle);
          throw;
        }
        NeuralNet::freeComputeHandle(gpuHandle);
      }
      catch(...) {
        anyError.store(true);
        std::lock_guard<std::mutex> lock(errorMutex);
        if(firstError == nullptr)
          firstError = std::current_exception();
      }
    });
  }

  while(readyCount.load() < numThreadsToUse && !anyError.load())
    std::this_thread::yield();
  std::chrono::steady_clock::time_point wallStart = std::chrono::steady_clock::now();
  startFlag.store(true, std::memory_order_release);
  for(std::thread& t: threads)
    t.join();

  if(firstError != nullptr)
    std::rethrow_exception(firstError);

  std::chrono::steady_clock::time_point wallEnd = wallStart;
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++)
    wallEnd = std::max(wallEnd, threadEndTimes[threadIdx]);

  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    std::vector<double> sorted = result.perThreadIterationSeconds[threadIdx];
    std::sort(sorted.begin(), sorted.end());
    double median =
      sorted.size() % 2 == 1 ? sorted[sorted.size()/2] :
      0.5 * (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]);
    result.perThreadMedianSeconds[threadIdx] = median;
    result.perThreadNNEvalsPerSec[threadIdx] = median > 0.0 ? (double)batchSize / median : 0.0;
    result.sumMedianNNEvalsPerSec += result.perThreadNNEvalsPerSec[threadIdx];
  }
  result.actualWallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
  if(result.actualWallSeconds > 0.0)
    result.actualWallNNEvalsPerSec =
      (double)numThreadsToUse * (double)batchSize * (double)numIterations / result.actualWallSeconds;
  return result;
}

void NNEvaluator::serve(
  Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL) {
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx
    );

    // Warm up lazily-compiled backend graphs before reporting this thread as started.
    maybeWarmupComputeHandle(gpuHandle, serverThreadIdx);
  }

  // Allocate input buffers only after createComputeHandle, which binds this thread to its GPU.
  // On the CUDA backend the pinned host allocation in the buffers initializes a context on the
  // thread's current device, so doing it before the GPU binding would leave a stray context
  // (and some wasted VRAM) on GPU 0 even when only other GPUs are configured.
  NNServerBuf buf(*this, loadedModel);

  {
    lock_guard<std::mutex> lock(bufferMutex);
    testAssert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, maxRowsToSendPerBatch.load(std::memory_order_acquire));
    bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
    // Queue being closed is a signal that we're done.
    if(!gotAnything)
      break;

    int numRows = (int)resultBufs.size();
    testAssert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();
        resultBuf->policyScratch.resize(NNPos::MAX_NN_POLICY_SIZE);  //Enclosure fork: compact policy
        resultBuf->result->policyDense = resultBuf->policyScratch.data();

        float* policyProbs = resultBuf->result->policyDense;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        // At this point, these aren't probabilities, since this is before the postprocessing
        // that happens for each result. These just need to be unnormalized log probabilities.
        // Illegal move filtering happens later.
        //Enclosure fork: one entry per edge, not per point (see NNPos).
        for(int pos = 0; pos<NNPos::getPassPos(nnXLen,nnYLen); pos++)
          policyProbs[pos] = (float)rand.nextGaussian();
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[nnXLen*nnYLen];
          for(int i = 0; i<nnXLen*nnYLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              whiteOwnerMap[pos] = (float)rand.nextGaussian() * 0.20f;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        // These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->whiteScoreMean = (float)whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = (float)whiteScoreMeanSq;
        resultBuf->result->whiteLead = (float)whiteScoreMean;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->shorttermScoreError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        testAssert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        //Enclosure fork (compact policy): the backend writes the dense policy into the requester's scratch.
        resultBufs[row]->policyScratch.resize(NNPos::MAX_NN_POLICY_SIZE);
        emptyOutput->policyDense = resultBufs[row]->policyScratch.data();
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        if(resultBufs[row]->includeAuxOutputs && modelHasAux) {
          emptyOutput->auxMap = new float[NNOutput::NUM_AUX_MAP_CHANNELS*nnXLen*nnYLen];
          emptyOutput->auxValues = new float[NNOutput::NUM_AUX_VALUES];
          if(modelHasPolicyAux)
            emptyOutput->auxPolicy = new float[(size_t)NNOutput::NUM_AUX_POLICIES*NNPos::MAX_NN_POLICY_SIZE];
          if(scoreBeliefParams.len > 0)
            emptyOutput->scoreBeliefRaw = new float[scoreBeliefParams.c2 + scoreBeliefParams.k];
        }
        outputBuf.push_back(emptyOutput);
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry =
              SymmetryHelpers::ALLOWED[rand.nextUInt(SymmetryHelpers::NUM_ALLOWED_SYMMETRIES)];
          else {
            testAssert(SymmetryHelpers::isAllowed(defaultSymmetry));
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf);
      testAssert(outputBuf.size() == numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    // Lock and update stats before looping again
    lock.lock();
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

#ifdef NNEVAL_SYNCHRONOUS
//Enclosure fork (games/enclosure/wasm): serve()'s code for one row (no batching), in the calling thread.
void NNEvaluator::evaluateSynchronously(NNResultBuf* resultBuf) {
  if(syncServerBuf == NULL || syncHandle == NULL)
    throw StringError("NNEvaluator: synchronous evaluation needs a model and spawnServerThreads first");
  NNOutput* emptyOutput = new NNOutput();
  emptyOutput->nnXLen = nnXLen;
  emptyOutput->nnYLen = nnYLen;
  resultBuf->policyScratch.resize(NNPos::MAX_NN_POLICY_SIZE);
  emptyOutput->policyDense = resultBuf->policyScratch.data();
  if(resultBuf->includeOwnerMap)
    emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
  else
    emptyOutput->whiteOwnerMap = NULL;
  if(resultBuf->includeAuxOutputs && modelHasAux) {
    emptyOutput->auxMap = new float[NNOutput::NUM_AUX_MAP_CHANNELS*nnXLen*nnYLen];
    emptyOutput->auxValues = new float[NNOutput::NUM_AUX_VALUES];
    if(modelHasPolicyAux)
      emptyOutput->auxPolicy = new float[(size_t)NNOutput::NUM_AUX_POLICIES*NNPos::MAX_NN_POLICY_SIZE];
    if(scoreBeliefParams.len > 0)
      emptyOutput->scoreBeliefRaw = new float[scoreBeliefParams.c2 + scoreBeliefParams.k];
  }
  if(resultBuf->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
    if(currentDoRandomize.load(std::memory_order_acquire))
      resultBuf->symmetry = SymmetryHelpers::ALLOWED[syncRand->nextUInt(SymmetryHelpers::NUM_ALLOWED_SYMMETRIES)];
    else
      resultBuf->symmetry = currentDefaultSymmetry.load(std::memory_order_acquire);
  }
  vector<NNOutput*> outputBuf(1, emptyOutput);
#ifdef ENCLOSURE_COOP_THREADS
  //Enclosure fork (the practice bot): a lane of its own, as the call can wait while another evaluation starts.
  std::pair<ComputeHandle*, NNServerBuf*> lane;
  if(!syncFreeLanes.empty()) {
    lane = syncFreeLanes.back();
    syncFreeLanes.pop_back();
  }
  else {
    lane = {NeuralNet::createComputeHandle(computeContext, loadedModel, logger, maxBatchSize, requireExactNNLen,
                                           inputsUseNHWC, gpuIdxByServerThread.empty() ? 0 : gpuIdxByServerThread[0], 0),
            new NNServerBuf(*this, loadedModel)};
  }
  NeuralNet::getOutput(lane.first, lane.second->inputBuffers, 1, &resultBuf, outputBuf);
  syncFreeLanes.push_back(lane);
#else
  NeuralNet::getOutput(syncHandle, syncServerBuf->inputBuffers, 1, &resultBuf, outputBuf);
#endif
  testAssert(outputBuf.size() == 1);
  m_numRowsProcessed.fetch_add(1, std::memory_order_relaxed);
  m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
  resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[0]);
  resultBuf->hasResult = true;
  //The evaluation's end as serve() ends a batch: wake the threads that wait for it in waitForNextNNEvalIfAny (a search
  //thread at a finished game). With one thread none waits; the practice bot's cooperative threads can.
  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals -= 1;
  if(numWaitingEvals > 0) {
    numEvalsToAwaken += numWaitingEvals;
    numWaitingEvals = 0;
    waitingForFinish.notify_all();
  }
}
#endif

void NNEvaluator::waitForNextNNEvalIfAny() {
  unique_lock<std::mutex> lock(bufferMutex);
  if(numOngoingEvals <= 0)
    return;

  numWaitingEvals++;
  while(numEvalsToAwaken <= 0 && !isKilled)
    waitingForFinish.wait(lock);
  numEvalsToAwaken--;
}


static double softPlus(double x) {
  // Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

static const int daggerPattern[9][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,2,1,0,0,0,0,0},
  {0,3,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};
static bool daggerMatch(const Board& board, Player nextPla, Loc& banned, int symmetry) {
  for(int yi = 0; yi < 9; yi++) {
    for(int xi = 0; xi < 8; xi++) {
      int y = yi;
      int x = xi;
      if((symmetry & 0x1) != 0)
        std::swap(x,y);
      if((symmetry & 0x2) != 0)
        x = board.x_size-1-x;
      if((symmetry & 0x4) != 0)
        y = board.y_size-1-y;
      Loc loc = Location::getLoc(x,y,board.x_size);
      int m = daggerPattern[yi][xi];
      if(m == 0 && board.colors[loc] != C_EMPTY)
        return false;
      if(m == 1 && board.colors[loc] != nextPla)
        return false;
      if(m == 2 && board.colors[loc] != getOpp(nextPla))
        return false;
      if(m == 3)
        banned = loc;
    }
  }
  return true;
}

//Enclosure fork: the symmetry of an evaluation is chosen by the thread asking for it (see evaluate).
//One generator per thread, seeded from the first evaluator it serves.
static Rand& clientSymmetryRand(const string& randSeed) {
  static std::atomic<uint64_t> numClientThreads(0);
  thread_local std::unique_ptr<Rand> rand;
  if(rand == nullptr)
    rand = std::make_unique<Rand>(randSeed + ":NNEvalClientThread:" + Global::uint64ToString(numClientThreads.fetch_add(1)));
  return *rand;
}

//KataGame fork (2026-10-29): the moves that the network's search lists: board's legal moves, less, from the inputs
//version 13, those that another listed move stands for (NetActionMap::listed: Blueprints' actions of a blueprint equal
//to a lower-numbered one). A reference to board's own list, or to this thread's copy until its next call.
static const std::bitset<GameRules::NUM_ACTIONS>& listedMoves(const Board& board,
                                                             const SymmetryHelpers::NetActionMap& netAction) {
  const std::bitset<GameRules::NUM_ACTIONS>& legal = board.legalActions();
  if(!netAction.labels)
    return legal;
  static thread_local std::bitset<GameRules::NUM_ACTIONS> listed;
  listed = legal;
  for(size_t i = bitsFirst(legal); i < (size_t)NNPos::PASS_POS; i = bitsNext(legal, i))
    if(!netAction.listed((int)i))
      listed.reset(i);
  return listed;
}

//Enclosure fork: map an output computed on the image of board under symmetry back to board. Policy
//entries follow the moves (only board's legal moves matter; the rest are masked afterwards), and
//ownership follows the points.
//With mapPolicy false (owner, 2026-10-06: the request listed its policy entries already in the network's frame,
//NNResultBuf::policyGather) the policy is read through that list instead and left as it is here.
//KataGame fork (2026-10-29): the moves map through netAction (SymmetryHelpers::NetActionMap), which from the inputs
//version 13 also gives the network's frame of the game's labels, even at symmetry 0, and legal holds the moves that
//the search lists (listedMoves). The points move at a symmetry only: the labels' frame leaves them as they are.
static void mapOutputBack(NNOutput& output, const Board& board, const std::bitset<GameRules::NUM_ACTIONS>& legal,
                          const SymmetryHelpers::NetActionMap& netAction, int symmetry, int policySize,
                          bool mapPolicy) {
  static thread_local std::vector<float> buf;
  if(mapPolicy) {
    buf.assign(policySize, 0.0f);
    for(int i = 0; i < policySize; i++)
      if(i == NNPos::PASS_POS || legal[i])
        buf[i] = output.policyDense[netAction(i)];
    std::copy(buf.begin(), buf.end(), output.policyDense);
  }
  if(symmetry != 0 && output.whiteOwnerMap != NULL) {
    const int nnXLen = output.nnXLen, nnYLen = output.nnYLen;
    buf.assign(nnXLen * nnYLen, 0.0f);
    for(int y = 0; y < board.y_size; y++)
      for(int x = 0; x < board.x_size; x++) {
        Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
        buf[NNPos::xyToPos(x, y, nnXLen)] = output.whiteOwnerMap[NNPos::pointToPos(symLoc, board.x_size, nnXLen)];
      }
    std::copy(buf.begin(), buf.end(), output.whiteOwnerMap);
  }
  //Enclosure fork (owner, 2026-10-04): the aux maps follow the points too, channel by channel.
  if(symmetry != 0 && output.auxMap != NULL) {
    const int nnXLen = output.nnXLen, nnYLen = output.nnYLen;
    for(int c = 0; c < NNOutput::NUM_AUX_MAP_CHANNELS; c++) {
      float* channel = output.auxMap + c * nnXLen * nnYLen;
      buf.assign(nnXLen * nnYLen, 0.0f);
      for(int y = 0; y < board.y_size; y++)
        for(int x = 0; x < board.x_size; x++) {
          Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
          buf[NNPos::xyToPos(x, y, nnXLen)] = channel[NNPos::pointToPos(symLoc, board.x_size, nnXLen)];
        }
      std::copy(buf.begin(), buf.end(), channel);
    }
  }
  //Export stage 2: policy outputs 1-4 follow the actions, as policyProbs (the legal actions and pass).
  if(output.auxPolicy != NULL) {
    for(int a = 0; a < NNOutput::NUM_AUX_POLICIES; a++) {
      float* policy = output.auxPolicy + (size_t)a * NNPos::MAX_NN_POLICY_SIZE;
      buf.assign(policySize, 0.0f);
      for(int i = 0; i < policySize; i++)
        if(i == NNPos::PASS_POS || legal[i])
          buf[i] = policy[netAction(i)];
      std::copy(buf.begin(), buf.end(), policy);
    }
  }
}

//Enclosure fork (owner, 2026-10-04), export stage 2: the score belief from the backend's s2 and mixture logits, as
//model_pytorch.ValueHead.forward: per bin i, h = act(s2 + offW * offsetScale * (i - len/2 + 0.5)), logits = s3 h + b;
//each component a softmax over the bins; the mixture weighted by the softmax of the mixture logits. The parity term
//is zero in the Enclosure fork. The backend's values carry the output scale (outputScaleMultiplier undoes it).
void NNEvaluator::finishScoreBelief(NNOutput& output) const {
  const ScoreBeliefParams& sb = scoreBeliefParams;
  const float scale = (float)postProcessParams.outputScaleMultiplier;
  std::vector<float> s2(sb.c2), mix(sb.k), h(sb.c2);
  for(int j = 0; j < sb.c2; j++)
    s2[j] = output.scoreBeliefRaw[j] * scale;
  for(int a = 0; a < sb.k; a++)
    mix[a] = output.scoreBeliefRaw[sb.c2 + a] * scale;
  auto activate = [&](float x) -> float {
    switch(sb.activation) {
      case ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
      case ACTIVATION_MISH: {
        double softplus = x > 20.0f ? (double)x : log1p(exp((double)x));
        return (float)(x * tanh(softplus));
      }
      case ACTIVATION_SILU: return (float)(x / (1.0 + exp(-(double)x)));
      default: return x;
    }
  };
  std::vector<double> logits((size_t)sb.len * sb.k);
  const int mid = sb.len / 2;
  for(int i = 0; i < sb.len; i++) {
    const float offset = sb.offsetScale * ((float)(i - mid) + 0.5f);
    for(int j = 0; j < sb.c2; j++)
      h[j] = activate(s2[j] + sb.offW[j] * offset);
    for(int a = 0; a < sb.k; a++) {
      double v = sb.s3B[a];
      for(int j = 0; j < sb.c2; j++)
        v += (double)h[j] * sb.s3W[(size_t)j * sb.k + a];
      logits[(size_t)i * sb.k + a] = v;
    }
  }
  //log-softmax of each component over the bins, plus the log mixture weights, then the mixture's probabilities.
  std::vector<double> logWeight(sb.k);
  {
    double m = *std::max_element(mix.begin(), mix.end()), sum = 0.0;
    for(int a = 0; a < sb.k; a++)
      sum += exp(mix[a] - m);
    for(int a = 0; a < sb.k; a++)
      logWeight[a] = mix[a] - m - log(sum);
  }
  for(int a = 0; a < sb.k; a++) {
    double m = -1e300, sum = 0.0;
    for(int i = 0; i < sb.len; i++)
      m = std::max(m, logits[(size_t)i * sb.k + a]);
    for(int i = 0; i < sb.len; i++)
      sum += exp(logits[(size_t)i * sb.k + a] - m);
    const double logNorm = m + log(sum);
    for(int i = 0; i < sb.len; i++)
      logits[(size_t)i * sb.k + a] += logWeight[a] - logNorm;
  }
  delete[] output.scoreBelief;
  output.scoreBeliefLen = sb.len;
  output.scoreBelief = new float[sb.len];
  for(int i = 0; i < sb.len; i++) {
    double p = 0.0;
    for(int a = 0; a < sb.k; a++)
      p += exp(logits[(size_t)i * sb.k + a]);
    output.scoreBelief[i] = (float)p;
  }
  delete[] output.scoreBeliefRaw;
  output.scoreBeliefRaw = NULL;
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams,
  NNResultBuf& buf,
  bool includeOwnerMap,
  Rand& rand,
  int numSymmetriesToSample
) {
  MiscNNInputParams nnInputParams = baseNNInputParams;
  vector<std::shared_ptr<NNOutput>> ptrs;
  //KataGame fork: a sample of the game's allowed symmetries.
  std::array<int, SymmetryHelpers::NUM_ALLOWED_SYMMETRIES> symmetryIndexes = SymmetryHelpers::ALLOWED;
  testAssert(numSymmetriesToSample <= SymmetryHelpers::NUM_ALLOWED_SYMMETRIES);
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_ALLOWED_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; // Skip cache since there's no guarantee which symmetry is in the cache
    evaluate(
      board, history, nextPlayer, sgfMeta,
      nnInputParams,
      buf, skipCacheThisIteration, includeOwnerMap
    );
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache,
    includeOwnerMap
  );
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  testAssert(!isKilled);
  buf.hasResult = false;

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(numInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(numInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    //Enclosure fork: a cached result also lacks what was asked when it has no aux outputs.
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL) &&
       !(buf.includeAuxOutputs && modelHasAux && buf.result->auxMap == NULL))
    {
      m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  //Enclosure fork: a symmetry acts on the position, not on the input planes (an edge's planes re-anchor
  //under a flip), so the backends' pixel symmetry cannot be used. The symmetry is chosen here as the
  //server thread would (random, or the default), the inputs are the image position's, the backend
  //gets the identity, and the outputs are mapped back when they arrive. KataGame fork: one of the game's allowed
  //symmetries (SymmetryHelpers::ALLOWED).
  int symmetry = nnInputParams.symmetry;
  if(symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
    //Enclosure fork: from the cache key (setSymmetryFromHash, NNResultBuf::symmetryFromHash)
    if(currentSymmetryFromHash.load(std::memory_order_acquire) || buf.symmetryFromHash)
      symmetry =
        SymmetryHelpers::ALLOWED[(nnHash.hash0 ^ (nnHash.hash1 >> 7)) % SymmetryHelpers::NUM_ALLOWED_SYMMETRIES];
    else if(currentDoRandomize.load(std::memory_order_acquire))
      symmetry =
        SymmetryHelpers::ALLOWED[clientSymmetryRand(randSeed).nextUInt(SymmetryHelpers::NUM_ALLOWED_SYMMETRIES)];
    else
      symmetry = currentDefaultSymmetry.load(std::memory_order_acquire);
  }
  testAssert(SymmetryHelpers::isAllowed(symmetry));
  //Enclosure fork (owner, 2026-10-06): no image board: the inputs are the image's, written from board
  //(fillRowBufs' symmetry), and moves map through symAction (SymmetryHelpers::SymActionMap). KataGame fork
  //(2026-10-29): from the inputs version 13, through the network's frame of the game's labels too (NetActionMap), and
  //the network gives a policy over the moves that its search lists only (listedMoves): legalBits stands for
  //board.legalActions() below.
  const SymmetryHelpers::NetActionMap netAction(board, symmetry, inputsVersion);
  const std::bitset<GameRules::NUM_ACTIONS>& legalBits = listedMoves(board, netAction);

  if(!debugSkipNeuralNet) {
    fillRowBufs(board, history, nextPlayer, sgfMeta, nnInputParams, buf, symmetry);
  }

  //Enclosure fork (owner, 2026-10-06): the policy entries read below (mapOutputBack, then the masking): the legal
  //moves, ascending, and pass, in the network's frame (NNResultBuf::policyGather).
  buf.policyGather.clear();
  //ENCLOSURE_NO_POLICY_GATHER=1 (tests only): no list, so the backends copy the whole policy as before the gather.
  static const bool noPolicyGather = getenv("ENCLOSURE_NO_POLICY_GATHER") != NULL;
  if(!debugSkipNeuralNet && !noPolicyGather) {
    for(size_t i = bitsFirst(legalBits); i < (size_t)NNPos::PASS_POS; i = bitsNext(legalBits, i))
      buf.policyGather.push_back((uint16_t)netAction((int)i));
    buf.policyGather.push_back((uint16_t)NNPos::PASS_POS);
  }

  buf.symmetry = 0;
  buf.policyOptimism = nnInputParams.policyOptimism;

  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

#ifdef NNEVAL_SYNCHRONOUS
  evaluateSynchronously(&buf);
#else
  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();
#endif

  //Enclosure fork (owner, 2026-10-06): with the policy entries listed (NNResultBuf::policyGather), the post-processing
  //below reads those only (listedPolicy), already through the symmetry. Not with the dagger hack (Go's), which masks
  //moves beyond the board's legality.
  const bool listedPolicy = !buf.policyGather.empty() && !nnInputParams.avoidMYTDaggerHack;
  if(symmetry != 0 || netAction.labels)
    mapOutputBack(*buf.result, board, legalBits, netAction, symmetry, policySize, !listedPolicy);

  // Perform postprocessing on the result - turn the nn output into probabilities
  // As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  // and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  // and causing policy weights to be different, which would reduce performance of successive searches in a game
  // by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    buf.result->whiteLead = resultWithoutOwnerMap->whiteLead;
    buf.result->varTimeLeft = resultWithoutOwnerMap->varTimeLeft;
    buf.result->shorttermWinlossError = resultWithoutOwnerMap->shorttermWinlossError;
    buf.result->shorttermScoreError = resultWithoutOwnerMap->shorttermScoreError;
    //Enclosure fork: the compact policy (the new output's dense scratch is left unread).
    buf.result->legalPos = resultWithoutOwnerMap->legalPos;
    buf.result->legalProbs = resultWithoutOwnerMap->legalProbs;
    std::copy(resultWithoutOwnerMap->policyMask, resultWithoutOwnerMap->policyMask + NNPos::MAX_BOARD_AREA, buf.result->policyMask);
    std::copy(resultWithoutOwnerMap->policyStart, resultWithoutOwnerMap->policyStart + NNPos::MAX_BOARD_AREA, buf.result->policyStart);
    buf.result->policyPassSlot = resultWithoutOwnerMap->policyPassSlot;
    buf.result->policyDense = NULL;
    buf.result->policyOptimismUsed = (float)resultWithoutOwnerMap->policyOptimismUsed;
    buf.result->nnXLen = resultWithoutOwnerMap->nnXLen;
    buf.result->nnYLen = resultWithoutOwnerMap->nnYLen;
    buf.result->hasRegret = resultWithoutOwnerMap->hasRegret;
    buf.result->regretValue = resultWithoutOwnerMap->regretValue;
    buf.result->regretRanking = resultWithoutOwnerMap->regretRanking;
    buf.result->hasSwing = resultWithoutOwnerMap->hasSwing;
    buf.result->swingValue = resultWithoutOwnerMap->swingValue;
    buf.result->swingRanking = resultWithoutOwnerMap->swingRanking;
    buf.result->hasIntrinsic = resultWithoutOwnerMap->hasIntrinsic;
    buf.result->intrinsicA = resultWithoutOwnerMap->intrinsicA;
    buf.result->intrinsicB = resultWithoutOwnerMap->intrinsicB;
    testAssert(!includeOwnerMap || buf.result->whiteOwnerMap != NULL);
  }
  else {
    float policyOutputScaling = postProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;
    if(listedPolicy) {
    //Enclosure fork (owner, 2026-10-06; trick 2's follow-up): the listed entries only (the legal moves ascending, then
    //pass, in the network's frame), through the same steps and float operations as the dense loops below and in the
    //same order, so the results are identical: an entry not listed is illegal, and an illegal entry adds exactly 0
    //to a sum and never wins the max (pass is always listed, as the passing hack reads it even when illegal).
    const std::vector<uint16_t>& gather = buf.policyGather;
    const float* dense = buf.result->policyDense;
    const int n = (int)gather.size();
    testAssert(n >= 1 && gather[n-1] == NNPos::PASS_POS);
    static thread_local std::vector<float> vals;
    static thread_local std::vector<uint16_t> listedPos;
    static thread_local std::vector<char> legal;
    vals.resize(n);
    listedPos.resize(n);
    legal.resize(n);
    testAssert(nextPlayer == history.presumedNextMovePla);
    {
      const bool canMove = !history.isGameFinished && nextPlayer == board.nextPla() && !board.isGameOver();
      int j = 0;
      for(size_t i = bitsFirst(legalBits); i < (size_t)NNPos::PASS_POS; i = bitsNext(legalBits, i))
        listedPos[j++] = (uint16_t)i;
      listedPos[j++] = (uint16_t)NNPos::PASS_POS;
      testAssert(j == n);
      for(j = 0; j < n; j++)
        legal[j] = canMove && legalBits.test(listedPos[j]);
    }
    float maxPolicy = -1e25f;
    int legalCount = 0;
    for(int j = 0; j<n; j++) {
      float policyValue;
      if(legal[j]) {
        legalCount += 1;
        policyValue = dense[gather[j]] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;
      vals[j] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }
    testAssert(legalCount > 0);
    float policySum = 0.0f;
    if(nnInputParams.enablePassingHacks) {
      float maxPassPolicySumFactor = 19.0f;
      for(int j = 0; j<n-1; j++) {
        vals[j] = legal[j] ? exp(vals[j] - maxPolicy) : 0.0f;
        policySum += vals[j];
      }
      vals[n-1] = std::max(1e-20f, std::min(exp(vals[n-1] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += vals[n-1];
    }
    else {
      for(int j = 0; j<n; j++) {
        vals[j] = legal[j] ? exp(vals[j] - maxPolicy) : 0.0f;
        policySum += vals[j];
      }
    }
    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int j = 0; j<n; j++)
        if(legal[j])
          vals[j] = uniform;
    }
    else {
      for(int j = 0; j<n; j++)
        if(legal[j])
          vals[j] = vals[j] / policySum;
    }
    buf.result->setPolicyFromLegal(listedPos.data(), vals.data(), legal.data(), n);
    buf.result->policyDense = NULL;
    testAssert((int)buf.result->legalPos.size() == legalCount);
    }
    else {
    //The dense path (no listed entries: the no-network test mode, the dagger hack), as upstream on 17,330 entries.
    float* policy = buf.result->policyDense;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    testAssert(nextPlayer == history.presumedNextMovePla);
    //Enclosure fork: the legal moves listed once, then read directly. Policy index i is move MOVE_LOC_BASE + i,
    //the last one pass (NNPos::posToLoc), so history.isLegal(board, posToLoc(i), nextPlayer) is the game-state
    //check (the same for every i) and bit i of the list; per entry it was 17,329 full calls per evaluation.
    {
      const bool canMove = !history.isGameFinished && nextPlayer == board.nextPla() && !board.isGameOver();
      for(int i = 0; i<policySize; i++)
        isLegal[i] = canMove && legalBits.test(i);
    }

    if(nnInputParams.avoidMYTDaggerHack && xSize >= 13 && ySize >= 13) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        Loc banned = Board::NULL_LOC;
        if(daggerMatch(board, nextPlayer, banned, symmetry)) {
          if(banned != Board::NULL_LOC) {
            isLegal[NNPos::locToPos(banned,xSize,nnXLen,nnYLen)] = false;
          }
        }
      }
    }

    for(int i = 0; i<policySize; i++) {
      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    testAssert(legalCount > 0);

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = isLegal[i] ? exp(policy[i] - maxPolicy) : 0.0f;  //Enclosure fork: as below, exactly 0 if illegal
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      testAssert(passPos == policySize-1);
      int i = passPos;
      policy[i] = std::max(1e-20f, std::min(exp(policy[i] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += policy[i];
    }
    else {
      //Enclosure fork: exp only for legal entries; an illegal one holds -1e30, whose exp here is exactly 0 (about
      //440 legal moves against 17,329 entries).
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? exp(policy[i] - maxPolicy) : 0.0f;
        policySum += policy[i];
      }
    }

    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }

    // Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    // Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    // Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    //Enclosure fork (owner, 2026-10-05/06: compact policy storage): the legal entries (exactly those left >= 0 above)
    //kept, the dense scratch released.
    buf.result->compactPolicy();
    testAssert((int)buf.result->legalPos.size() == legalCount);
    }

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    // Fix up the value as well. Note that the neural net gives us back the value from the perspective
    // of the player so we need to negate that to make it the white value.
    if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      // Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(!isfinite(probSum) || !isfinite(scoreValue)) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }

    }
    else if(modelVersion >= 4) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      double lead;
      double varTimeLeft;
      double shorttermWinlossError;
      double shorttermScoreError;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;
        double scoreMeanPreScaled = buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq * postProcessParams.outputScaleMultiplier;
        double leadPreScaled = buf.result->whiteLead * postProcessParams.outputScaleMultiplier;
        double varTimeLeftPreSoftplus = buf.result->varTimeLeft * postProcessParams.outputScaleMultiplier;
        double shorttermWinlossErrorPreSoftplus = buf.result->shorttermWinlossError * postProcessParams.outputScaleMultiplier;
        double shorttermScoreErrorPreSoftplus = buf.result->shorttermScoreError * postProcessParams.outputScaleMultiplier;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * postProcessParams.scoreMeanMultiplier;
        double scoreStdev = softPlus(scoreStdevPreSoftplus) * postProcessParams.scoreStdevMultiplier;
        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
        lead = leadPreScaled * postProcessParams.leadMultiplier;
        varTimeLeft = softPlus(varTimeLeftPreSoftplus) * postProcessParams.varianceTimeMultiplier;

        // scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        // noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);
        lead = lead * (1.0-noResultProb);

        if(modelVersion >= 14) {
          {
            double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
            shorttermWinlossError = sqrt(s * s * postProcessParams.shorttermValueErrorMultiplier);
          }
          {
            double s = softPlus(shorttermScoreErrorPreSoftplus * 0.5);
            shorttermScoreError = sqrt(s * s * postProcessParams.shorttermScoreErrorMultiplier);
          }
        }
        else if(modelVersion >= 10) {
          shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * postProcessParams.shorttermValueErrorMultiplier);
          shorttermScoreError = sqrt(softPlus(shorttermScoreErrorPreSoftplus) * postProcessParams.shorttermScoreErrorMultiplier);
        }
        else {
          shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
          shorttermScoreError = softPlus(shorttermScoreErrorPreSoftplus) * 10.0;
        }

        if(
          !isfinite(probSum) ||
          !isfinite(scoreMean) ||
          !isfinite(scoreMeanSq) ||
          !isfinite(lead) ||
          !isfinite(varTimeLeft) ||
          !isfinite(shorttermWinlossError) ||
          !isfinite(shorttermScoreError)
        ) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits
               << " " << scoreMean << " " << scoreMeanSq
               << " " << lead << " " << varTimeLeft
               << " " << shorttermWinlossError << " " << shorttermScoreError
               << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = (float)lead;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = -(float)lead;
      }

      if(modelVersion >= 9) {
        buf.result->varTimeLeft = (float)varTimeLeft;
        buf.result->shorttermWinlossError = (float)shorttermWinlossError;
        buf.result->shorttermScoreError = (float)shorttermScoreError;
      }
      else {
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  //Enclosure fork (owner, 2026-10-04): RGSC's heads (when not kept from the cached result above) and the aux outputs,
  //with the output scale (the backends' scale8 leaves every output divided by 8). The regret value through the
  //authors' sigmoid x 4; the ranking score raw. The aux outputs stay raw (model_pytorch's head outputs).
  if(buf.result->hasRegret && !hadResultWithoutOwnerMap) {
    double rawRegret = buf.result->regretValue * postProcessParams.outputScaleMultiplier;
    buf.result->regretValue = (float)(4.0 / (1.0 + exp(-rawRegret)));
    buf.result->regretRanking = (float)(buf.result->regretRanking * postProcessParams.outputScaleMultiplier);
  }
  //The swing regret's heads, the same way (Enclosure fork, 2026-10-11).
  if(buf.result->hasSwing && !hadResultWithoutOwnerMap) {
    double rawSwing = buf.result->swingValue * postProcessParams.outputScaleMultiplier;
    buf.result->swingValue = (float)(4.0 / (1.0 + exp(-rawSwing)));
    buf.result->swingRanking = (float)(buf.result->swingRanking * postProcessParams.outputScaleMultiplier);
  }
  //KataGame fork (2026-10-29): AZdb's intrinsic returns, raw outputs (model_pytorch's intrinsic_value) with the output
  //scale. They are returns of the member, so they keep their sign for either player.
  if(buf.result->hasIntrinsic && !hadResultWithoutOwnerMap) {
    buf.result->intrinsicA = (float)(buf.result->intrinsicA * postProcessParams.outputScaleMultiplier);
    buf.result->intrinsicB = (float)(buf.result->intrinsicB * postProcessParams.outputScaleMultiplier);
  }
  if(buf.result->auxMap != NULL) {
    for(int c = 0; c<NNOutput::NUM_AUX_MAP_CHANNELS; c++)
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        float& v = buf.result->auxMap[c*nnXLen*nnYLen + pos];
        v = (y >= board.y_size || x >= board.x_size) ? 0.0f : v * (float)postProcessParams.outputScaleMultiplier;
      }
  }
  if(buf.result->auxValues != NULL) {
    for(int i = 0; i<NNOutput::NUM_AUX_VALUES; i++)
      buf.result->auxValues[i] *= (float)postProcessParams.outputScaleMultiplier;
  }
  //Export stage 2: policy outputs 1-4 raw with the output scale, 0 at illegal actions (as mapOutputBack leaves them
  //under a symmetry); the score belief finished here.
  if(buf.result->auxPolicy != NULL) {
    for(int a = 0; a<NNOutput::NUM_AUX_POLICIES; a++) {
      float* policy = buf.result->auxPolicy + (size_t)a * NNPos::MAX_NN_POLICY_SIZE;
      for(int i = 0; i<policySize; i++)
        policy[i] = (i == NNPos::PASS_POS || legalBits.test(i)) ? policy[i] * (float)postProcessParams.outputScaleMultiplier : 0.0f;
    }
  }
  if(buf.result->scoreBeliefRaw != NULL)
    finishScoreBelief(*buf.result);

  // Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          // Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          // but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }


  // And record the nnHash in the result and put it into the table
  //Enclosure fork (2026-10-21): not a result with the training-only outputs. Only checks and the engine protocol's maps
  //ask for them, one position at a time, and they make an entry several times larger (policy outputs 1-4 alone are
  //277 KB). An entry of the same position without them stays, so the search's evaluations do not change.
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL && buf.result->auxMap == NULL)
    nnCacheTable->set(buf.result);

}

// Uncomment this to lower the effective hash size down to one where we get true collisions
// #define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  // Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}
