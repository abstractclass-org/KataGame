// Blueprints' engine protocol (command/blueprintsprotocol.cpp) as a native command, the same code as the WebAssembly
// module (games/enclosure/wasm/engine.cpp, built for Blueprints): one JSON request a line on standard input, one JSON
// reply a line on standard output, in the order of the requests, until standard input closes.
//
//   katago blueprints-engine -config configs/analysis_blueprints.cfg [-model MODEL] [-override-config "k=v,k=v"]
//                            [-build NAME]
//
// -model loads a network at start (as {"action": "load_model", "path": MODEL}); its reply is the first line.
#include "../command/blueprintsprotocol.h"
#include "../command/commandline.h"
#include "../core/global.h"
#include "../game/board.h"
#include "../main.h"
#include "../neuralnet/nninputs.h"
#include "../external/nlohmann_json/json.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

namespace {

int blueprintsEngine(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string configFile, modelFile, overrides, build;
  KataGoCommandLine cmd("Run Blueprints' engine protocol: JSON lines on standard input and output.");
  try {
    TCLAP::ValueArg<string> configArg("", "config", "The analysis settings (configs/analysis_blueprints.cfg)", true, "", "FILE");
    TCLAP::ValueArg<string> modelArg("", "model", "A network to load at start", false, "", "FILE");
    TCLAP::ValueArg<string> overrideArg("", "override-config", "Settings over the config: k=v,k=v", false, "", "KEYVALUEPAIRS");
    TCLAP::ValueArg<string> buildArg("", "build", "The build's name in replies (default: native)", false, "native", "NAME");
    cmd.add(configArg);
    cmd.add(modelArg);
    cmd.add(overrideArg);
    cmd.add(buildArg);
    cmd.parseArgs(args);
    configFile = configArg.getValue();
    modelFile = modelArg.getValue();
    overrides = overrideArg.getValue();
    build = buildArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ifstream in(configFile);
  if(!in) {
    cerr << "Error: cannot read " << configFile << endl;
    return 1;
  }
  stringstream text;
  text << in.rdbuf() << "\n";
  stringstream pairs(overrides);
  for(string kv; getline(pairs, kv, ',');) {
    const size_t eq = kv.find('=');
    if(eq != string::npos)
      text << Global::trim(kv.substr(0, eq)) << " = " << Global::trim(kv.substr(eq + 1)) << "\n";
  }
  BlueprintsProtocol::setup(text.str(), build);

  if(!modelFile.empty()) {
    const nlohmann::json load = {{"id", "start"}, {"action", "load_model"}, {"path", modelFile}};
    cout << BlueprintsProtocol::handle(load.dump()) << endl;
  }
  for(string line; getline(cin, line);) {
    if(Global::trim(line).empty())
      continue;
    cout << BlueprintsProtocol::handle(line) << endl;
  }
  return 0;
}

//Blueprints' own command (game/blueprints/ownfiles.txt): only Blueprints' binary has it.
const MainCmds::GameCommand BLUEPRINTS_ENGINE(
  "blueprints-engine", "Blueprints' engine protocol (design 04): JSON lines on standard input and output",
  blueprintsEngine);

}  // namespace
