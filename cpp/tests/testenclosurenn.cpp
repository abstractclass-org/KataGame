// Enclosure fork: a network's evaluation in C++ (plan step 5), for comparison with PyTorch.
//
//   katago runenclosurenntests MODEL_FILE INPUTS_FILE OUT_FILE [fp16] [nhwc] [sym=N]
//
// sym=N evaluates at symmetry N instead of 0 (for comparing two builds byte for byte; PyTorch is compared at 0).
// fp16 and nhwc select those backend modes (the CUDA backend's usual FP16 path uses NHWC).
// Replays each position of INPUTS_FILE (tests/data/enclosure_inputs.txt), evaluates it with NNEvaluator
// (symmetry 0) and writes one line per position to OUT_FILE:
//   <label> <player to move 0 blue, 1 red> <win> <loss> <noresult> <scoremean> <lead>
//     own <ownership values, one per point of the network's board> pol <legal count> <action> <probability> ...
// Values, scores and ownership are from the player to move's side, as the network outputs them.
// The Enclosure repository's katagame/python/enclosure_nn_check.py compares them with a PyTorch forward pass.

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

void Tests::runEnclosureNNTests(const string& modelFile, const string& inputsFile, const string& outFile, bool useFP16, bool useNHWC, int symmetry) {
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cerr);
  //Enclosure fork: the inputs in the backend's layout (2026-10-04: TensorRT takes NCHW only; it was NHWC always).
  //The network's board is the game's largest board (Enclosure: 19 x 19). The positions start on the game's standard
  //board, which need not fill it (Connect-4: 7 x 6).
  const int nnLen = NNPos::MAX_BOARD_LEN;
  NNEvaluator* nnEval = startNNEval(modelFile, logger, "", nnLen, nnLen, 0, useNHWC, useNHWC, useFP16, false,
                                    NNPos::fillsNNBoard(Board::DEFAULT_X_SIZE, Board::DEFAULT_Y_SIZE));

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
    //sym=N (2026-10-06): another symmetry, for comparing builds (the outputs mapped back; PyTorch is checked at 0).
    params.symmetry = symmetry;
    NNResultBuf buf;
    nnEval->evaluate(board, hist, pla, params, buf, true, true);
    const NNOutput& o = *buf.result;
    //NNEvaluator reports White's side; the network, and this file, the mover's.
    const bool white = pla == P_WHITE;
    const double sign = white ? 1.0 : -1.0;
    out << label << " " << Board::indexOfPlayer(pla) << " "
        << (white ? o.whiteWinProb : o.whiteLossProb) << " " << (white ? o.whiteLossProb : o.whiteWinProb) << " "
        << o.whiteNoResultProb << " " << sign * o.whiteScoreMean << " " << sign * o.whiteLead << " own";
    for(int i = 0; i < nnLen * nnLen; i++)
      out << " " << sign * o.whiteOwnerMap[i];
    vector<Loc> legal;
    board.getLegalMoveLocs(legal);
    out << " pol " << legal.size();
    for(Loc loc : legal) {
      int pos = NNPos::locToPos(loc, board.x_size, nnLen, nnLen);
      out << " " << Board::actionOfMoveLoc(loc) << " " << o.policy()[pos];
    }
    out << "\n";
    positions++;
  }
  cout << positions << " positions evaluated into " << outFile << endl;
  delete nnEval;
  NeuralNet::globalCleanup();
}

// The members' latent of each row of a writer's text output (TrainingWriteBuffers::writeToTextOstream): the member
// whose input is 1, one for each row.
static vector<int> latentsOfRows(const string& text, int64_t rows) {
  const size_t at = text.find("globalInputNC\n");
  testAssert(at != string::npos);
  istringstream in(text.substr(at));
  string line;
  getline(in, line);  // the section's name
  getline(in, line);  // the array's header
  vector<int> out;
  for(int64_t r = 0; r < rows; r++) {
    testAssert((bool)getline(in, line));
    istringstream values(line);
    vector<float> row;
    float v;
    while(values >> v)
      row.push_back(v);
    testAssert((int)row.size() == NNInputs::NUM_FEATURES_GLOBAL_V14);
    int member = -1;
    for(int i = 0; i < NNInputs::NUM_DIVERSITY_MEMBERS; i++)
      if(row[NNInputs::NUM_FEATURES_GLOBAL_V13 + i] == 1.0f) {
        testAssert(member == -1);
        member = i;
      }
    out.push_back(member);
  }
  return out;
}

// KataGame fork (2026-10-29): AZdb's matchmaker trains only the chooser's rows (FinishedGameData::diversityTrains).
// The game written again as a game of member 2 (the first player) against member 4: with the first player's rows
// only, the second's only, and both. The first two hold the rows of their player's moves, each with its member's
// latent, and together the rows of both. Every row of the game has a whole weight, so that no draw decides a row.
static void checkChoosersRows(FinishedGameData& data, int inputsVersion, int nnLen, int maxRows, const string& seed) {
  const int savedMembers[2] = {data.diversityMembers[0], data.diversityMembers[1]};
  data.diversityMembers[0] = 2;
  data.diversityMembers[1] = 4;
  testAssert(data.sidePositions.empty());
  int64_t movesOf[2] = {0, 0};
  {
    BoardHistory hist(data.startHist);
    Board board(data.startBoard);
    Player pla = data.startPla;
    for(size_t t = data.startHist.moveHistory.size(); t < data.endHist.moveHistory.size(); t++) {
      const double weight = data.targetWeightByTurn[t - data.startHist.moveHistory.size()];
      testAssert(weight == std::floor(weight));
      movesOf[Board::indexOfPlayer(pla)] += (int64_t)weight;
      const Move& move = data.endHist.moveHistory[t];
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
      pla = hist.presumedNextMovePla;
    }
  }
  int64_t rows[3];
  const bool trains[3][2] = {{true, false}, {false, true}, {true, true}};
  for(int k = 0; k < 3; k++) {
    ostringstream sink;
    TrainingDataWriter writer(&sink, inputsVersion, maxRows, 1.0, nnLen, nnLen, 1, seed + "chooser");
    data.diversityTrains[0] = trains[k][0];
    data.diversityTrains[1] = trains[k][1];
    writer.writeGame(data);
    // The game's rows fit in one file, so that none has left the buffer.
    rows[k] = writer.numRowsInBuffer();
    testAssert(rows[k] < maxRows);
    writer.flushIfNonempty();
    if(k < 2)
      for(int member : latentsOfRows(sink.str(), rows[k]))
        testAssert(member == (k == 0 ? 2 : 4));
  }
  data.diversityTrains[0] = data.diversityTrains[1] = true;
  data.diversityMembers[0] = savedMembers[0];
  data.diversityMembers[1] = savedMembers[1];
  testAssert(movesOf[0] > 0 && movesOf[1] > 0);
  testAssert(rows[0] == movesOf[0] && rows[1] == movesOf[1] && rows[2] == rows[0] + rows[1]);
  cout << "the chooser's rows: " << rows[0] << " of the first player, " << rows[1] << " of the second, " << rows[2]
       << " of both" << endl;
}

// KataGame fork (2026-10-29): a player whose rows do not train (MatchPairer::BotSpec::trains, AZdb's chooser) has the
// weight 0 at each of its turns before KataGo spreads the game's weight by surprise, so that its turns take none of it.
// A game with surprise weights where the second player does not train: its turns have the weight 0, and the first
// player's weights before their random rounding (targetWeightByTurnUnrounded) sum to its count of turns, each turn's
// weight 1 before the spread.
static void checkWeightsOfTurnsThatTrain(const MatchPairer::BotSpec& botSpec, Logger& logger, const string& seed) {
  MatchPairer::BotSpec botSpecB = botSpec;
  MatchPairer::BotSpec botSpecW = botSpec;
  botSpecW.trains = false;
  Board initialBoard;
  BoardHistory initialHist(initialBoard, P_BLACK, Rules(), 0);
  ExtraBlackAndKomi extraBlackAndKomi;
  PlaySettings playSettings;
  playSettings.forSelfPlay = true;
  playSettings.policySurpriseDataWeight = 0.5;
  playSettings.valueSurpriseDataWeight = 0.1;
  Rand rand(seed + "weights");
  OtherGameProperties otherGameProps;
  auto shouldStop = []() noexcept { return false; };
  FinishedGameData* gameData = Play::runGame(
    initialBoard, P_BLACK, initialHist, extraBlackAndKomi,
    botSpecB, botSpecW,
    seed + "weights search",
    false, true,
    logger, false, false,
    1000, shouldStop,
    nullptr,
    playSettings, otherGameProps,
    rand,
    nullptr,
    nullptr
  );
  testAssert(gameData->diversityTrains[0] && !gameData->diversityTrains[1]);
  double sums[2] = {0.0, 0.0};
  int turns[2] = {0, 0};
  const size_t start = gameData->startHist.moveHistory.size();
  testAssert(gameData->targetWeightByTurnUnrounded.size() == gameData->targetWeightByTurn.size());
  for(size_t t = 0; t < gameData->targetWeightByTurn.size(); t++) {
    const int p = Board::indexOfPlayer(gameData->endHist.moveHistory[start + t].pla);
    sums[p] += gameData->targetWeightByTurnUnrounded[t];
    turns[p]++;
    if(p == 1)
      testAssert(gameData->targetWeightByTurn[t] == 0.0f);
  }
  testAssert(turns[0] > 0 && turns[1] > 0);
  testAssert(sums[1] == 0.0);
  testAssert(std::fabs(sums[0] - turns[0]) < 1e-3 * turns[0]);
  cout << "the weights of the turns that train: " << sums[0] << " over the first player's " << turns[0]
       << " turns, 0 over the second player's " << turns[1] << endl;
  delete gameData;
}

// Enclosure fork: training rows from a self-play game (plan step 6).
//
//   katago runenclosurerowtests OUT_FILE [SEED [NPZ_DIR [INPUTS_VERSION]]]
//
// With NPZ_DIR, the same rows are also written as KataGo's .npz training files there (needs libzip). The rows hold the
// inputs of INPUTS_VERSION, by default those of self-play's default (NNModelVersion::defaultModelVersion).
// Plays one self-play game with the no-network evaluator (random policy and values, exact results at the end),
// writes every row through TrainingDataWriter's text output, and appends the game's moves ("moves <n> <action>
// ..."). The Enclosure repository's games/enclosure/tools/check_katago_rows.py replays the game in its engine and checks every row.
// KataGame fork (2026-10-29): from the inputs version 14, both players are AZdb's member 2, with a fixed state
// (AZDB_TEST_PSI below, each statistic's mean 0 and standard deviation 1, l0 1). The evaluator has no intrinsic head,
// so the rows hold targets of the intrinsic value head only where their span reaches the game's end (program/azdb.h),
// and their latent is member 2's. Then the game, as one of member 2 against member 4, checks that only the chooser's
// rows are written (checkChoosersRows), and a second game, whose second player does not train, checks the turns'
// weights (checkWeightsOfTurnsThatTrain), apart from the output.
void Tests::runEnclosureRowTests(const string& outFile, const string& seed, const string& npzDir, int inputsVersion) {
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cerr);
  const int nnLen = NNPos::MAX_BOARD_LEN;
  NNEvaluator* nnEval = startNNEval("/dev/null", logger, seed, nnLen, nnLen, 0, true, false, false, true, false);

  ofstream out(outFile);
  const int maxRows = 256;
  const double firstFileMinRandProp = 1.0;
  const int debugOnlyWriteEvery = 1;
  TrainingDataWriter dataWriter(&out, inputsVersion, maxRows, firstFileMinRandProp, nnLen, nnLen, debugOnlyWriteEvery, seed + "dwriter");

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
  if(inputsVersion >= 14) {
    //AZDB_TEST_PSI: member i's psi at statistic s is 0.1 i ((s mod 7) - 3).
    const int n = GameInterface::diversityStatistics<GameEncoding>();
    vector<double> psi((size_t)AZdb::NUM_MEMBERS * n), mean(n, 0.0), sd(n, 1.0);
    for(int i = 0; i < AZdb::NUM_MEMBERS; i++)
      for(int s = 0; s < n; s++)
        psi[(size_t)i * n + s] = 0.1 * i * ((s % 7) - 3);
    gameData->diversityMembers[0] = gameData->diversityMembers[1] = 2;
    gameData->diversityState = std::make_shared<AZdb::State<GameEncoding>>(psi.data(), mean.data(), sd.data(), 1.0);
  }
  dataWriter.writeGame(*gameData);
  dataWriter.flushIfNonempty();
  if(inputsVersion >= 14) {
    checkChoosersRows(*gameData, inputsVersion, nnLen, maxRows, seed);
    checkWeightsOfTurnsThatTrain(botSpec, logger, seed);
  }
  if(npzDir != "") {
    TrainingDataWriter npzWriter(npzDir, inputsVersion, maxRows, firstFileMinRandProp, nnLen, nnLen, seed + "dwriter");
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
