// KataGame fork (2026-11-03, the engine repository's abscls-ra93.3): a model's outputs on given input rows, for the
// check of the network against PyTorch on rows that no position replays (a training run's holdout rows):
//
//   testnnrows MODEL ROWS_FILE OUT_FILE FP16 NHWC [EXACT [BATCH]]
//
// It names no game, so it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with TESTS=testnnrows).
// katagame/python/nn_rows_check.py writes ROWS_FILE and compares OUT_FILE with PyTorch's outputs on the same rows.
//
// ROWS_FILE: a line "nnrows <rows> <channels> <len> <globals>", then the rows' spatial inputs (float32, little-endian,
// NCHW: channel, then y, then x, on the network's len x len board), then their global inputs (float32). Each batch of
// BATCH rows (default 16) goes through the backend as NNEvaluator sends it (NeuralNet::getOutput: symmetry 0, no packed
// rows), with FP16 and NHWC as testnnoutputs (the CUDA backend's modes; the Eigen backend needs NHWC 1) and
// requireExactNNLen EXACT (0 or 1; default: whether the game's standard board fills the network's, as testnnoutputs and
// self-play). Each batch is evaluated twice: with no policy optimism, and with the policy optimism 1, which gives the
// optimistic policy output alone.
//
// OUT_FILE: a line "nnrowsout <rows>" and then NAME:COUNT for each section of a row, then per row its sections in
// float32. Each is what the backend gives NNEvaluator, times the model's output scale as NNEvaluator multiplies it
// (outputScaleMultiplier: the CUDA backend runs a network without transformer blocks with its activations scaled by
// 1/8, ModelDesc::applyScale8ToReduceActivations), and otherwise as it is, before NNEvaluator's softmax, sigmoid and
// softplus, in the view of the player to move:
//   policy     the logits of the first policy output (entry (y * len + x) * POLICY_CHANNELS + channel, then pass)
//   policyopt  the logits of the optimistic policy output, in the same order
//   value      the value logits (win, loss, no result)
//   score      the 6 score outputs (score mean, its stdev before softplus, lead, time left's variance, the short-term
//              winloss and score errors)
//   intrinsic  AZdb's outputs A and B (NaN for a model without the head)
//   regret     RGSC's regret value and ranking, then the swing regret's (NaN where the model or the backend has none:
//              the Eigen backend does not compute them)
//   own        the ownership before tanh (y * len + x)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/timer.h"
#include "../../neuralnet/nneval.h"
#include "../../neuralnet/nninterface.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc < 6 || argc > 8) {
    fprintf(stderr, "usage: testnnrows MODEL ROWS_FILE OUT_FILE FP16 NHWC [EXACT [BATCH]]\n");
    return 2;
  }
  const string modelFile = argv[1];
  const bool fp16 = string(argv[4]) == "1";
  const bool nhwc = string(argv[5]) == "1";
  const bool exact = argc >= 7 ? string(argv[6]) == "1" : NNPos::fillsNNBoard(Board::DEFAULT_X_SIZE, Board::DEFAULT_Y_SIZE);
  const int maxBatch = argc >= 8 ? atoi(argv[7]) : 16;
  if(maxBatch < 1) {
    fprintf(stderr, "BATCH is 1 or more\n");
    return 2;
  }

  ifstream in(argv[2], ios::binary);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 2;
  }
  string header;
  getline(in, header);
  istringstream head(header);
  string word;
  int rows = 0, channels = 0, len = 0, globals = 0;
  if(!(head >> word >> rows >> channels >> len >> globals) || word != "nnrows" || rows < 1) {
    fprintf(stderr, "%s: not a rows file (its first line: %s)\n", argv[2], header.c_str());
    return 2;
  }
  const int S = NNPos::MAX_BOARD_LEN;
  const int area = S * S;
  if(len != S) {
    fprintf(stderr, "the rows are on a board of %d, the network's is %d\n", len, S);
    return 2;
  }
  vector<float> spatial((size_t)rows * channels * area);
  vector<float> global((size_t)rows * globals);
  in.read((char*)spatial.data(), (streamsize)(spatial.size() * sizeof(float)));
  in.read((char*)global.data(), (streamsize)(global.size() * sizeof(float)));
  if(!in) {
    fprintf(stderr, "%s: shorter than its header says\n", argv[2]);
    return 2;
  }

  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  map<string, string> kvs = {{"cudaUseNHWC", nhwc ? "true" : "false"}};
  ConfigParser cfg(kvs);
  Logger logger(nullptr, false, true, false, false);
  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
  const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
  if(desc.numInputChannels != channels || desc.numInputGlobalChannels != globals || desc.numInputMetaChannels != 0) {
    fprintf(stderr, "the model reads %d spatial and %d global inputs (and %d meta), the rows hold %d and %d\n",
            desc.numInputChannels, desc.numInputGlobalChannels, desc.numInputMetaChannels, channels, globals);
    return 2;
  }
  ComputeContext* context = NeuralNet::createComputeContext(
    {0}, &logger, S, S, "", fp16 ? enabled_t::True : enabled_t::False, loadedModel, cfg);
  ComputeHandle* handle = NeuralNet::createComputeHandle(context, loadedModel, &logger, maxBatch, exact, nhwc, 0, 0);
  //Read after createComputeContext, as NNEvaluator reads it: a backend may change it there.
  const float scale = desc.postProcessParams.outputScaleMultiplier;
  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatch, S, S);

  vector<unique_ptr<NNResultBuf>> bufs;
  vector<NNResultBuf*> bufPtrs;
  for(int i = 0; i < maxBatch; i++) {
    bufs.push_back(make_unique<NNResultBuf>());
    bufPtrs.push_back(bufs.back().get());
  }

  ofstream out(argv[3], ios::binary);
  if(!out) {
    fprintf(stderr, "cannot open %s\n", argv[3]);
    return 2;
  }
  const int policyLen = NNPos::MAX_NN_POLICY_SIZE;
  out << "nnrowsout " << rows << " policy:" << policyLen << " policyopt:" << policyLen << " value:3 score:6 intrinsic:2"
      << " regret:4 own:" << area << "\n";
  const float none = std::numeric_limits<float>::quiet_NaN();
  vector<float> row;
  //The evaluations' seconds (the first getOutput of each batch), after the first batch (which builds the backend's plans).
  double seconds = 0.0;
  int timedRows = 0;
  bool hasRegret = false;
  for(int start = 0; start < rows; start += maxBatch) {
    const int n = std::min(maxBatch, rows - start);
    //The batch twice: no policy optimism, then the policy optimism 1 (the optimistic output alone).
    vector<unique_ptr<NNOutput>> owned[2];
    vector<NNOutput*> outputs[2];
    vector<vector<float>> policies[2];
    for(int pass = 0; pass < 2; pass++) {
      policies[pass].reserve(n);  //each output points into its vector
      for(int i = 0; i < n; i++) {
        const int r = start + i;
        NNResultBuf& buf = *bufs[i];
        buf.rowSpatialPacked = false;
        buf.rowSpatialBuf.resize((size_t)channels * area);  //every entry is written below
        const float* src = spatial.data() + (size_t)r * channels * area;
        for(int c = 0; c < channels; c++)
          for(int p = 0; p < area; p++)
            buf.rowSpatialBuf[nhwc ? (size_t)p * channels + c : (size_t)c * area + p] = src[(size_t)c * area + p];
        buf.rowGlobalBuf.assign(global.begin() + (size_t)r * globals, global.begin() + (size_t)(r + 1) * globals);
        buf.hasRowMeta = false;
        buf.symmetry = 0;
        buf.policyOptimism = pass;
        buf.policyGather.clear();
        policies[pass].emplace_back(policyLen, 0.0f);
        owned[pass].push_back(make_unique<NNOutput>());
        NNOutput* o = owned[pass].back().get();
        o->nnXLen = S;
        o->nnYLen = S;
        o->policyDense = policies[pass].back().data();
        o->whiteOwnerMap = new float[area];
        outputs[pass].push_back(o);
      }
      ClockTimer timer;
      NeuralNet::getOutput(handle, inputBuffers, n, bufPtrs.data(), outputs[pass]);
      if(pass == 0 && start > 0) {
        seconds += timer.getSeconds();
        timedRows += n;
      }
    }
    for(int i = 0; i < n; i++) {
      const NNOutput& o = *outputs[0][i];
      row.assign(o.policyDense, o.policyDense + policyLen);
      row.insert(row.end(), outputs[1][i]->policyDense, outputs[1][i]->policyDense + policyLen);
      for(float v : {o.whiteWinProb, o.whiteLossProb, o.whiteNoResultProb, o.whiteScoreMean, o.whiteScoreMeanSq,
                     o.whiteLead, o.varTimeLeft, o.shorttermWinlossError, o.shorttermScoreError})
        row.push_back(v);
      row.push_back(o.hasIntrinsic ? o.intrinsicA : none);
      row.push_back(o.hasIntrinsic ? o.intrinsicB : none);
      row.push_back(o.hasRegret ? o.regretValue : none);
      row.push_back(o.hasRegret ? o.regretRanking : none);
      row.push_back(o.hasSwing ? o.swingValue : none);
      row.push_back(o.hasSwing ? o.swingRanking : none);
      hasRegret = hasRegret || o.hasRegret;
      row.insert(row.end(), o.whiteOwnerMap, o.whiteOwnerMap + area);
      for(float& v : row)
        v *= scale;
      out.write((const char*)row.data(), (streamsize)(row.size() * sizeof(float)));
    }
  }
  fprintf(stderr, "wrote %d rows (fp16 %d, nhwc %d, exact %d, batch %d, fp16 used %d, model version %d, output scale %g, "
          "regret %d); after the first batch, %d rows in %.3f s, %.1f rows a second\n", rows, (int)fp16, (int)nhwc,
          (int)exact, maxBatch, (int)NeuralNet::isUsingFP16(handle), desc.modelVersion, (double)scale, (int)hasRegret,
          timedRows, seconds, seconds > 0.0 ? timedRows / seconds : 0.0);
  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::freeComputeHandle(handle);
  NeuralNet::freeComputeContext(context);
  NeuralNet::freeLoadedModel(loadedModel);
  NeuralNet::globalCleanup();
  return 0;
}
