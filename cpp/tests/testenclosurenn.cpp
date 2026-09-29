// Enclosure fork: a network's evaluation in C++ (plan step 5), for comparison with PyTorch.
//
//   katago runenclosurenntests MODEL_FILE INPUTS_FILE OUT_FILE [fp16] [nhwc]
//
// fp16 and nhwc select those backend modes (the CUDA backend's usual FP16 path uses NHWC).
// Replays each position of INPUTS_FILE (tests/data/enclosure_inputs.txt), evaluates it with NNEvaluator
// (symmetry 0) and writes one line per position to OUT_FILE:
//   <label> <player to move 0 blue, 1 red> <win> <loss> <noresult> <scoremean> <lead>
//     own <361 ownership values> pol <legal count> <action> <probability> ...
// Values, scores and ownership are from the player to move's side, as the network outputs them.
// The Enclosure repository's katago/python/enclosure_nn_check.py compares them with a PyTorch forward pass.

#include "../tests/tests.h"

#include <fstream>
#include <iomanip>
#include <sstream>

#include "../dataio/trainingwrite.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../program/play.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestSearchCommon;

void Tests::runEnclosureNNTests(const string& modelFile, const string& inputsFile, const string& outFile, bool useFP16, bool useNHWC) {
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cerr);
  NNEvaluator* nnEval = startNNEval(modelFile, logger, "", 19, 19, 0, true, useNHWC, useFP16, false, true);

  ifstream in(inputsFile);
  if(!in)
    throw StringError("cannot open " + inputsFile);
  ofstream out(outFile);
  out << setprecision(9);
  string line;
  int positions = 0;
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    for(int i = 0; i < n; i++) {
      int action;
      head >> action;
      Loc loc = Board::moveLocOfAction(action);
      testAssert(hist.isLegal(board, loc, hist.presumedNextMovePla));
      hist.makeBoardMoveAssumeLegal(board, loc, hist.presumedNextMovePla, NULL);
    }
    if(hist.isGameFinished)
      continue;  //The network is not asked about finished games
    Player pla = hist.presumedNextMovePla;
    MiscNNInputParams params;
    params.symmetry = 0;
    NNResultBuf buf;
    nnEval->evaluate(board, hist, pla, params, buf, true, true);
    const NNOutput& o = *buf.result;
    //NNEvaluator reports White's side; the network, and this file, the mover's.
    const bool white = pla == P_WHITE;
    const double sign = white ? 1.0 : -1.0;
    out << label << " " << Board::indexOfPlayer(pla) << " "
        << (white ? o.whiteWinProb : o.whiteLossProb) << " " << (white ? o.whiteLossProb : o.whiteWinProb) << " "
        << o.whiteNoResultProb << " " << sign * o.whiteScoreMean << " " << sign * o.whiteLead << " own";
    for(int i = 0; i < 19 * 19; i++)
      out << " " << sign * o.whiteOwnerMap[i];
    vector<Loc> legal;
    board.getLegalMoveLocs(legal);
    out << " pol " << legal.size();
    for(Loc loc : legal) {
      int pos = NNPos::locToPos(loc, 19, 19, 19);
      out << " " << Board::actionOfMoveLoc(loc) << " " << o.policyProbs[pos];
    }
    out << "\n";
    positions++;
  }
  cout << positions << " positions evaluated into " << outFile << endl;
  delete nnEval;
  NeuralNet::globalCleanup();
}

// Enclosure fork: training rows from a self-play game (plan step 6).
//
//   katago runenclosurerowtests OUT_FILE [SEED [NPZ_DIR]]
//
// With NPZ_DIR, the same rows are also written as KataGo's .npz training files there (needs libzip).
// Plays one self-play game with the no-network evaluator (random policy and values, exact results at the end),
// writes every row through TrainingDataWriter's text output, and appends the game's moves ("moves <n> <action>
// ..."). The Enclosure repository's tools/check_katago_rows.py replays the game in its engine and checks every row.
void Tests::runEnclosureRowTests(const string& outFile, const string& seed, const string& npzDir) {
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cerr);
  NNEvaluator* nnEval = startNNEval("/dev/null", logger, seed, 19, 19, 0, true, false, false, true, false);

  ofstream out(outFile);
  const int inputsVersion = NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  const int maxRows = 256;
  const double firstFileMinRandProp = 1.0;
  const int debugOnlyWriteEvery = 1;
  TrainingDataWriter dataWriter(&out, inputsVersion, maxRows, firstFileMinRandProp, 19, 19, debugOnlyWriteEvery, seed + "dwriter");

  SearchParams params;
  params.maxVisits = 20;
  params.useGraphSearch = true;
  params.staticScoreUtilityFactor = 0.0;
  params.dynamicScoreUtilityFactor = 0.0;
  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "test";
  botSpec.nnEval = nnEval;
  botSpec.baseParams = params;

  Board initialBoard;
  BoardHistory initialHist(initialBoard, P_BLACK, Rules(), 0);
  ExtraBlackAndKomi extraBlackAndKomi;
  PlaySettings playSettings;
  playSettings.forSelfPlay = true;
  Rand rand(seed + "play");
  OtherGameProperties otherGameProps;
  auto shouldStop = []() noexcept { return false; };
  FinishedGameData* gameData = Play::runGame(
    initialBoard, P_BLACK, initialHist, extraBlackAndKomi,
    botSpec, botSpec,
    seed + "search",
    false, true,
    logger, false, false,
    1000, shouldStop,
    nullptr,
    playSettings, otherGameProps,
    rand,
    nullptr,
    nullptr
  );
  testAssert(gameData->endHist.isGameFinished);
  dataWriter.writeGame(*gameData);
  dataWriter.flushIfNonempty();
  if(npzDir != "") {
    TrainingDataWriter npzWriter(npzDir, inputsVersion, maxRows, firstFileMinRandProp, 19, 19, seed + "dwriter");
    npzWriter.writeGame(*gameData);
    npzWriter.flushIfNonempty();
  }
  const vector<Move>& moves = gameData->endHist.moveHistory;
  out << "moves " << moves.size();
  for(const Move& m : moves)
    out << " " << Board::actionOfMoveLoc(m.loc);
  out << "\n";
  cout << "game of " << moves.size() << " moves, red minus blue " << gameData->endHist.finalWhiteMinusBlackScore
       << ", rows written to " << outFile << endl;
  delete gameData;
  delete nnEval;
  NeuralNet::globalCleanup();
}
