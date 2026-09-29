// Enclosure fork: a network's evaluation in C++ (plan step 5), for comparison with PyTorch.
//
//   katago runenclosurenntests MODEL_FILE INPUTS_FILE OUT_FILE
//
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

#include "../neuralnet/nneval.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestSearchCommon;

void Tests::runEnclosureNNTests(const string& modelFile, const string& inputsFile, const string& outFile) {
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cerr);
  NNEvaluator* nnEval = startNNEval(modelFile, logger, "", 19, 19, 0, true, false, false, false, true);

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
