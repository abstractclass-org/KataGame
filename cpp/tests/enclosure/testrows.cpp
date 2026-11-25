// KataGame fork (abscls-shfz.11): the training rows of self-play's games with a plain record of each game, for a check
// of a game's rows outside the C++ code (the engine repository's games/tak/tools/check_rows.py). It names no game, so
// it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with TESTS=testrows).
//
//   testrows SELFPLAY_CONFIG MODEL GAMES OUT_DIR [OVERRIDES [STARTS]]
//
// It plays GAMES games one after the other as self-play plays them (play.cpp's GameRunner), with the config and then
// OVERRIDES (key=value,..., as -override-config), fixed seeds and one search thread. MODEL can be /dev/null, KataGo's
// evaluator without a network. With STARTS, a file of states (testpositions.h: state LABEL KOMI TEXT), every second
// game starts from its next state, at that state's komi (an exact start with a frozen context, as RGSC's restarts). The
// rows go to OUT_DIR as a training file (.npz, of the config's inputsVersion), and the records to
// OUT_DIR/records.jsonl. A line is a game: its hash (as the rows' C41 to C46 hold it), its komi, the state of its first
// row's position (the text, and the turn of it), its moves from there as the protocol writes them, whether it hit the
// move cap, and the states of its side positions with their turns. Exit code 0 when every game ends.

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/makedir.h"
#include "../../core/rand.h"
#include "../../dataio/trainingwrite.h"
#include "../../neuralnet/modelversion.h"
#include "../../neuralnet/nneval.h"
#include "../../program/play.h"
#include "../../program/playsettings.h"
#include "../../program/rgsc.h"
#include "../../program/setup.h"
#include "../../external/nlohmann_json/json.hpp"
#include "testpositions.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc < 5 || argc > 7) {
    fprintf(stderr, "usage: testrows SELFPLAY_CONFIG MODEL GAMES OUT_DIR [OVERRIDES [STARTS]]\n");
    return 2;
  }
  const int numGames = atoi(argv[3]);
  const string outDir = argv[4];
  Board::initHash();
  ScoreValue::initTables();
  ConfigParser cfg(argv[1]);
  if(argc >= 6 && argv[5][0] != '\0')
    cfg.overrideKeys(ConfigParser::parseCommaSeparated(argv[5]));
  cfg.overrideKeys(map<string, string>{{"numSearchThreads", "1"}});
  Logger logger(nullptr, false, true, false, false);
  Rand seedRand("testrows");
  const SearchParams baseParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
  const PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, false);
  unique_ptr<GameRunner> gameRunner = make_unique<GameRunner>(cfg, "testrows games", playSettings, logger);
  Setup::initializeSession(cfg);
  const GameInitializer* init = gameRunner->getGameInitializer();
  const bool exactNNLen = init->getMinBoardXSize() == init->getMaxBoardXSize() &&
                          init->getMinBoardYSize() == init->getMaxBoardYSize();
  unique_ptr<NNEvaluator> nnEval(Setup::initializeNNEvaluator(
    argv[2], argv[2], "", cfg, logger, seedRand, 1, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    Setup::MaxBatchSizeRequest::requireFromConfig(), exactNNLen, false, Setup::SETUP_FOR_OTHER));

  vector<Rgsc::Entry> starts;
  if(argc == 7) {
    ifstream in(argv[6]);
    string label;
    Board board;
    BoardHistory hist;
    while(nextTestPosition(in, label, board, hist))
      starts.push_back(Rgsc::stateEntry(board, Rgsc::Context::of(hist.rules.komi, C_EMPTY, 0.0),
                                        GameRules::moves_played(board.pos)));
    if(starts.empty()) {
      fprintf(stderr, "no state in %s\n", argv[6]);
      return 2;
    }
  }

  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = nnEval->getModelName();
  botSpec.nnEval = nnEval.get();
  botSpec.baseParams = baseParams;
  const int inputsVersion = cfg.contains("inputsVersion")
    ? cfg.getInt("inputsVersion", 0, 10000)
    : NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  MakeDir::make(outDir);
  ofstream records(outDir + "/records.jsonl");
  unique_ptr<TrainingDataWriter> writer = make_unique<TrainingDataWriter>(
    outDir, inputsVersion, 1000000, 1.0, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, "testrows writer");
  int64_t turns = 0, sidePositions = 0;
  int capped = 0;
  for(int g = 0; g < numGames; g++) {
    Sgf::PositionSample start;
    const bool fromState = !starts.empty() && g % 2 == 1;
    if(fromState) {
      const Rgsc::Entry& entry = starts[(g / 2) % starts.size()];
      string why;
      Board board;
      if(!Rgsc::restoreState(entry, board, why) || !Rgsc::startOf(entry, board.x_size, board.y_size, start, why)) {
        fprintf(stderr, "game %d: no start: %s\n", g, why.c_str());
        return 1;
      }
    }
    unique_ptr<FinishedGameData> data(gameRunner->runGame(
      "testrows game " + to_string(g), botSpec, botSpec, NULL, fromState ? &start : NULL, logger, nullptr, nullptr,
      nullptr, nullptr, nullptr));
    if(data == NULL) {
      fprintf(stderr, "game %d did not end\n", g);
      return 1;
    }
    nlohmann::json record;
    record["hash"] = data->gameHash.toString();
    record["komi"] = data->endHist.rules.komi;
    record["state"] = GameInterface::stateText<GameRules>(data->startBoard.pos);
    const size_t first = data->startHist.moveHistory.size();
    record["turn"] = first;
    Board board = data->startBoard;
    vector<string> moves;
    for(size_t t = first; t < data->endHist.moveHistory.size(); t++) {
      const Move& m = data->endHist.moveHistory[t];
      moves.push_back(Location::toString(m.loc, board));
      board.playMoveAssumeLegal(m.loc, m.pla);
    }
    record["moves"] = moves;
    record["capped"] = data->hitTurnLimit;
    record["from_state"] = fromState;
    nlohmann::json side = nlohmann::json::array();
    for(const SidePosition* sp: data->sidePositions)  //the turn as the row writer gives it (C51)
      side.push_back({sp->hist.getCurrentTurnNumber() - data->endHist.initialTurnNumber,
                      GameInterface::stateText<GameRules>(sp->board.pos)});
    record["side"] = side;
    records << record.dump() << "\n";
    writer->writeGame(*data);
    turns += (int64_t)moves.size();
    sidePositions += (int64_t)data->sidePositions.size();
    capped += data->hitTurnLimit;
  }
  writer->flushIfNonempty();
  printf("%s: %d games (%d at the move cap), %lld turns and %lld side positions, the inputs version %d, in %s\n",
         GameRules::NAME, numGames, capped, (long long)turns, (long long)sidePositions, inputsVersion, outDir.c_str());
  writer.reset();
  gameRunner.reset();
  nnEval.reset();
  ScoreValue::freeTables();
  return 0;
}
