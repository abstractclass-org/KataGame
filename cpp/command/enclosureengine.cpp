// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md) as a native command, for the analysis
// site's offline jobs and our bots. The protocol is command/enclosureprotocol.cpp, the same code as the WebAssembly
// module (v2/wasm/engine.cpp); this is its transport: one JSON request a line on standard input, one JSON reply a line
// on standard output, in order, until standard input closes.
//
//   katago enclosure-engine -config configs/analysis_enclosure.cfg [-model MODEL] [-override-config "k=v,k=v"]
//                           [-build NAME]
//
// -model loads a network at start (as {"action": "load_model", "path": MODEL}); its reply is the first line. The
// settings' search threads (numSearchThreadsPerAnalysisThread) search one query faster; with more than one, results
// are not reproducible, and a search in slices still runs on one thread.
#include "../command/enclosureprotocol.h"
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

int MainCmds::enclosureEngine(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string configFile, modelFile, overrides, build;
  KataGoCommandLine cmd("Run the Enclosure engine protocol: JSON lines on standard input and output.");
  try {
    TCLAP::ValueArg<string> configArg("", "config", "The analysis settings (configs/analysis_enclosure.cfg)", true, "", "FILE");
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
  EnclosureProtocol::setup(text.str(), build);

  if(!modelFile.empty()) {
    const nlohmann::json load = {{"id", "start"}, {"action", "load_model"}, {"path", modelFile}};
    cout << EnclosureProtocol::handle(load.dump()) << endl;
  }
  for(string line; getline(cin, line);) {
    if(Global::trim(line).empty())
      continue;
    cout << EnclosureProtocol::handle(line) << endl;
  }
  return 0;
}
