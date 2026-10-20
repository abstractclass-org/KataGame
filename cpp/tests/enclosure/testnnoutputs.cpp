// Enclosure fork (owner, 2026-10-04): a model's outputs on fixed positions, for two checks:
//   - the export against PyTorch: katagame/python/enclosure_nn_check.py compare runs the checkpoint on the same inputs
//     and compares (OUTPUTS search,ownership), for any game; katagame/python/enclosure_nn_outputs_check.py compares
//     every output of an Enclosure model (OUTPUTS search,ownership,training);
//   - one build against another (backends, precisions, the WebAssembly bot against the CUDA build), on the outputs
//     that both compute: games/enclosure/wasm/compare_dumps.py compares two dumps (OUTPUTS search,ownership).
//
//   testnnoutputs MODEL INPUTS_FILE OUT_FILE FP16 NHWC OUTPUTS
//
// KataGame fork: it names no game, so it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with
// TESTS=testnnoutputs). A position is the game's actions from the start of its standard board.
//
// Replays each position of INPUTS_FILE (tests/data/enclosure_inputs.txt, the positions of testnninputs; Connect-4's:
// tests/data/connect4_positions.txt), evaluates it at symmetry 0 (FP16, NHWC: 0 or 1, the CUDA backend's modes; the
// Eigen backend needs NHWC 1), and writes per position the input rows the network got and the outputs of the groups
// that OUTPUTS names (comma-separated):
//   position <label> <player to move: 1 black, 2 white> <input layout: nchw or nhwc>
//   spatial <count> <index in that layout>:<value> ...
//   global <values>
// search: what the search reads (the value, the score utility, the uncertainty weighting, the policy with its
// optimism blend):
//   value <win> <loss> <noresult>          (white's view, as NNEvaluator reports them)
//   score <mean> <mean sq> <lead> <var time left> <short-term winloss error> <short-term score error>
//                                          (white's view where it has one, as NNEvaluator reports them; the lead
//                                          and the time-left variance are only reported)
//   policy <count> <action>:<probability> ...   (the move policy at the legal actions and pass)
// ownership: computed at the search's root, reported, and not used to choose a move:
//   owner <x * y values>                   (white's view, after tanh)
// training: the training-only outputs (since 2026-10-21 the Eigen backend computes them too, except RGSC's heads,
// which only self-play reads, so its dumps have no regret and no swing line):
//   regret <value> <ranking>
//   swing <value> <ranking>                (models with the swing heads, 2026-10-11)
//   auxvalues <12 values>
//   auxmap <9 * x * y values>
//   auxpolicy <count> <action>:<4 comma-separated logits> ...   (export stage 2: policy outputs 1-4 at the legal
//                                                               actions and pass)
//   scorebelief <len> <bin width> <len probabilities>            (export stage 2)

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../neuralnet/nneval.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc != 7) {
    fprintf(stderr, "usage: testnnoutputs MODEL INPUTS_FILE OUT_FILE FP16 NHWC OUTPUTS (search,ownership,training)\n");
    return 2;
  }
  const string modelFile = argv[1];
  const bool fp16 = string(argv[4]) == "1";
  const bool nhwc = string(argv[5]) == "1";
  bool search = false, ownership = false, training = false;
  {
    istringstream groups(argv[6]);
    string group;
    while(getline(groups, group, ',')) {
      if(group == "search")
        search = true;
      else if(group == "ownership")
        ownership = true;
      else if(group == "training")
        training = true;
      else {
        fprintf(stderr, "unknown output group %s (search, ownership, training)\n", group.c_str());
        return 2;
      }
    }
  }
#ifdef USE_EIGEN_BACKEND
  const bool regret = false;  //the Eigen backend does not compute RGSC's heads
#else
  const bool regret = training;
#endif
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();

  //trtIncludeAuxOutputs: the TensorRT backend builds the aux outputs into its engine only when asked (2026-10-04);
  //the CUDA backend ignores it. TensorRT is NCHW whatever cudaUseNHWC says: run it with NHWC 0.
  map<string, string> kvs = {{"cudaUseNHWC", nhwc ? "true" : "false"}, {"trtIncludeAuxOutputs", "true"}};
  ConfigParser cfg(kvs);
  Logger logger(nullptr, false, true, false, false);
  const int S = NNPos::MAX_BOARD_LEN;  //the network's board, the game's largest
  //The positions start on the game's standard board, which need not fill it (Connect-4: 7 x 6).
  NNEvaluator* nnEval = new NNEvaluator(
    "test", modelFile, "", &logger, 8, S, S, NNPos::fillsNNBoard(Board::DEFAULT_X_SIZE, Board::DEFAULT_Y_SIZE), nhwc, 10, 4, false, "",
    fp16 ? enabled_t::True : enabled_t::False, 1, {0}, "testnnoutputs", false, 0, cfg);
  nnEval->spawnServerThreads();
  if(training && (!nnEval->hasAuxOutputs() || (regret && !nnEval->hasRegretHeads()))) {
    fprintf(stderr, "the model has no aux outputs or no regret heads\n");
    return 1;
  }
  const bool stage2 = training && nnEval->hasAuxPolicy() && nnEval->hasScoreBelief();

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
    buf.includeAuxOutputs = training;
    nnEval->evaluate(board, hist, pla, params, buf, true, ownership);
    const NNOutput& o = *buf.result;
    if((ownership && o.whiteOwnerMap == NULL) ||
       (training && (o.auxMap == NULL || o.auxValues == NULL || (regret && !o.hasRegret))) ||
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
    out << "\n";
    vector<int> actions;
    for(int i = 0; i <= NNPos::PASS_POS; i++)
      if(i == NNPos::PASS_POS || board.legalActions()[i])
        actions.push_back(i);
    if(search) {
      out << "value " << o.whiteWinProb << " " << o.whiteLossProb << " " << o.whiteNoResultProb;
      out << "\nscore " << o.whiteScoreMean << " " << o.whiteScoreMeanSq << " " << o.whiteLead << " " << o.varTimeLeft
          << " " << o.shorttermWinlossError << " " << o.shorttermScoreError;
      const PolicyView policy = o.policy();
      out << "\npolicy " << actions.size();
      for(int i : actions)
        out << " " << i << ":" << policy[i];
      out << "\n";
    }
    if(ownership) {
      out << "owner";
      for(int i = 0; i < S * S; i++)
        out << " " << o.whiteOwnerMap[i];
      out << "\n";
    }
    if(training) {
      if(regret) {
        out << "regret " << o.regretValue << " " << o.regretRanking << "\n";
        if(o.hasSwing)
          out << "swing " << o.swingValue << " " << o.swingRanking << "\n";
      }
      out << "auxvalues";
      for(int i = 0; i < NNOutput::NUM_AUX_VALUES; i++)
        out << " " << o.auxValues[i];
      out << "\nauxmap";
      for(int i = 0; i < NNOutput::NUM_AUX_MAP_CHANNELS * S * S; i++)
        out << " " << o.auxMap[i];
      out << "\n";
    }
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
  fprintf(stderr, "wrote %d positions (fp16 %d, nhwc %d, fp16 used %d, outputs %s, stage 2 %d)\n", positions,
          (int)fp16, (int)nhwc, (int)nnEval->isAnyThreadUsingFP16(), argv[6], (int)stage2);
  nnEval->killServerThreads();
  delete nnEval;
  NeuralNet::globalCleanup();
  return 0;
}
