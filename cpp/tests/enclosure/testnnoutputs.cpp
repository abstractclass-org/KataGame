// Enclosure fork (owner, 2026-10-04): the exported training-only outputs and RGSC's heads, against PyTorch.
//
//   testnnoutputs MODEL INPUTS_FILE OUT_FILE FP16 NHWC
//
// Replays each position of INPUTS_FILE (tests/data/enclosure_inputs.txt, the positions of testnninputs), evaluates
// it at symmetry 0 with the ownership and the aux outputs (FP16, NHWC: 0 or 1, the CUDA backend's modes), and
// writes per position the input rows the network got and its outputs:
//   position <label> <player to move: 1 black, 2 white> <input layout: nchw or nhwc>
//   spatial <count> <index in that layout>:<value> ...
//   global <values>
//   value <win> <loss> <noresult>          (white's view, as NNEvaluator reports them)
//   owner <x * y values>                   (white's view, after tanh)
//   regret <value> <ranking>
//   auxvalues <12 values>
//   auxmap <9 * x * y values>
//   auxpolicy <count> <action>:<4 comma-separated logits> ...   (export stage 2: policy outputs 1-4 at the legal
//                                                               actions and pass)
//   scorebelief <len> <bin width> <len probabilities>            (export stage 2)
// katago/python/enclosure_nn_outputs_check.py runs the PyTorch checkpoint on the same inputs and compares.

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../neuralnet/nneval.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc != 6) {
    fprintf(stderr, "usage: testnnoutputs MODEL INPUTS_FILE OUT_FILE FP16 NHWC\n");
    return 2;
  }
  const string modelFile = argv[1];
  const bool fp16 = string(argv[4]) == "1";
  const bool nhwc = string(argv[5]) == "1";
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();

  //trtIncludeAuxOutputs: the TensorRT backend builds the aux outputs into its engine only when asked (2026-10-04);
  //the CUDA backend ignores it. TensorRT is NCHW whatever cudaUseNHWC says: run it with NHWC 0.
  map<string, string> kvs = {{"cudaUseNHWC", nhwc ? "true" : "false"}, {"trtIncludeAuxOutputs", "true"}};
  ConfigParser cfg(kvs);
  Logger logger(nullptr, false, true, false, false);
  const int S = enclosure::SIZE;
  NNEvaluator* nnEval = new NNEvaluator(
    "test", modelFile, "", &logger, 8, S, S, true, nhwc, 10, 4, false, "",
    fp16 ? enabled_t::True : enabled_t::False, 1, {0}, "testnnoutputs", false, 0, cfg);
  nnEval->spawnServerThreads();
  if(!nnEval->hasAuxOutputs() || !nnEval->hasRegretHeads()) {
    fprintf(stderr, "the model has no aux outputs or no regret heads\n");
    return 1;
  }
  const bool stage2 = nnEval->hasAuxPolicy() && nnEval->hasScoreBelief();

  ifstream in(argv[2]);
  ofstream out(argv[3]);
  if(!in || !out) {
    fprintf(stderr, "cannot open the input or the output file\n");
    return 2;
  }
  out.precision(9);
  string line;
  int positions = 0;
  //One buffer for every position, as a search thread keeps (its input row refilled packed, NNEvaluator::fillRowBufs).
  NNResultBuf buf;
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    bool legal = true;
    for(int i = 0; i < n && legal; i++) {
      int action;
      head >> action;
      Loc loc = Board::moveLocOfAction(action);
      Player pla = hist.presumedNextMovePla;
      legal = hist.isLegal(board, loc, pla);
      if(legal)
        hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    }
    if(!legal || hist.isGameFinished)
      continue;
    const Player pla = hist.presumedNextMovePla;
    MiscNNInputParams params;
    params.symmetry = 0;
    buf.includeAuxOutputs = true;
    nnEval->evaluate(board, hist, pla, params, buf, true, true);
    const NNOutput& o = *buf.result;
    if(o.auxMap == NULL || o.auxValues == NULL || o.whiteOwnerMap == NULL || !o.hasRegret ||
       (stage2 && (o.auxPolicy == NULL || o.scoreBelief == NULL))) {
      fprintf(stderr, "%s: missing outputs\n", label.c_str());
      return 1;
    }

    out << "position " << label << " " << (int)pla << " " << (nhwc ? "nhwc" : "nchw") << "\n";
    vector<pair<int, float>> nonzero;
    const float* rowSpatial = buf.spatialRowFloats(NNInputs::NUM_SPATIAL_ELTS_V8);
    for(int i = 0; i < NNInputs::NUM_SPATIAL_ELTS_V8; i++)
      if(rowSpatial[i] != 0.0f)
        nonzero.push_back({i, rowSpatial[i]});
    out << "spatial " << nonzero.size();
    for(auto& e : nonzero)
      out << " " << e.first << ":" << e.second;
    out << "\nglobal";
    for(float g : buf.rowGlobalBuf)
      out << " " << g;
    out << "\nvalue " << o.whiteWinProb << " " << o.whiteLossProb << " " << o.whiteNoResultProb;
    out << "\nowner";
    for(int i = 0; i < S * S; i++)
      out << " " << o.whiteOwnerMap[i];
    out << "\nregret " << o.regretValue << " " << o.regretRanking;
    out << "\nauxvalues";
    for(int i = 0; i < NNOutput::NUM_AUX_VALUES; i++)
      out << " " << o.auxValues[i];
    out << "\nauxmap";
    for(int i = 0; i < NNOutput::NUM_AUX_MAP_CHANNELS * S * S; i++)
      out << " " << o.auxMap[i];
    out << "\n";
    vector<int> actions;
    for(int i = 0; i <= enclosure::PASS; i++)
      if(i == enclosure::PASS || board.legalActions()[i])
        actions.push_back(i);
    //The move policy (output 0) at the legal actions and pass: a build compared with another (2026-10-08, the
    //WebAssembly bot against the CUDA build, v2/wasm).
    const PolicyView policy = o.policy();
    out << "policy " << actions.size();
    for(int i : actions)
      out << " " << i << ":" << policy[i];
    out << "\n";
    if(stage2) {
      out << "auxpolicy " << actions.size();
      for(int i : actions) {
        out << " " << i << ":";
        for(int a = 0; a < NNOutput::NUM_AUX_POLICIES; a++)
          out << (a ? "," : "") << o.auxPolicy[(size_t)a * NNPos::MAX_NN_POLICY_SIZE + i];
      }
      out << "\nscorebelief " << o.scoreBeliefLen << " " << nnEval->getScoreBinWidth();
      for(int i = 0; i < o.scoreBeliefLen; i++)
        out << " " << o.scoreBelief[i];
      out << "\n";
    }
    positions++;
  }
  fprintf(stderr, "wrote %d positions (fp16 %d, nhwc %d, fp16 used %d, stage 2 %d)\n", positions, (int)fp16,
          (int)nhwc, (int)nnEval->isAnyThreadUsingFP16(), (int)stage2);
  nnEval->killServerThreads();
  delete nnEval;
  NeuralNet::globalCleanup();
  return 0;
}
