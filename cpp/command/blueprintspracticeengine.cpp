// Blueprints' practice bot as a native GTP engine (command/blueprintspractice.h; the engine repository's
// games/blueprints/README.md, "The practice bot"): the session of the browser module, blueprints-bot.wasm, with the
// CUDA or Eigen backend, search threads, the network from a file and GTP's clock. One command a line on standard input,
// its GTP response on standard output, until quit or the end of standard input.
//
//   katago blueprints-practice -config configs/gtp_blueprints.cfg -model MODEL [-override-config "k=v,k=v"]
//                              [-build NAME]
//
// The engine loads the network at blueprints-start SEED [THREADS], with the config's settings and the session's over
// them (GTP's rules, the search threads and the seed). THREADS goes up to KataGo's 4096. The config's logs are KataGo's
// GTP logs (logDir, logSearchInfo, logAllGTPCommunication), the commands that the session sends GTP among them.
#include "../command/blueprintspractice.h"
#include "../command/commandline.h"
#include "../command/gtpsession.h"
#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/sha2.h"
#include "../game/board.h"
#include "../main.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../external/nlohmann_json/json.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

namespace {

// The most search threads, as KataGo's config reader takes numSearchThreads (Setup::loadParams).
const int MAX_THREADS = 4096;

class NativeHost : public BlueprintsPractice::Host {
 public:
  NativeHost(ConfigParser& cfg_, const string& modelFile_, const string& build_)
    : cfg(cfg_), modelFile(modelFile_), buildName(build_) {}

  string build() const override { return buildName; }
  int maxThreads() const override { return MAX_THREADS; }
  bool clock() const override { return true; }

  GTPSession* startEngine(const map<string, string>& settings) override {
    cfg.overrideKeys(settings);
    return GTPSessions::start(cfg, modelFile);
  }

  // The network's name as the engine read it (kata-get-models), and the file's SHA-256 and bytes.
  BlueprintsPractice::Network network(GTPSession* session) const override {
    BlueprintsPractice::Network n;
    bool quit = false;
    const string models = GTPSessions::handle(session, "kata-get-models", quit);
    const nlohmann::json list =
      nlohmann::json::parse(models.size() > 1 && models[0] == '=' ? models.substr(1) : "null", nullptr, false);
    if(list.is_array() && !list.empty() && list[0].contains("internalName") && list[0]["internalName"].is_string())
      n.name = list[0]["internalName"].get<string>();
    ifstream in(modelFile, ios::binary);
    const vector<char> bytes((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    char hash[65];
    SHA2::get256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), hash);
    n.sha256 = string(hash, 64);
    n.bytes = bytes.size();
    return n;
  }

 private:
  ConfigParser& cfg;
  string modelFile;
  string buildName;
};

int blueprintsPractice(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile, build;
  KataGoCommandLine cmd("Run Blueprints' practice bot as a GTP engine, in strategyjam.org's moves.");
  try {
    cmd.addConfigFileArg("", "gtp_blueprints.cfg");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();
    TCLAP::ValueArg<string> buildArg("", "build", "The build's name in blueprints-bot-info (default: native)", false,
                                     "native", "NAME");
    cmd.add(buildArg);
    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    build = buildArg.getValue();
    cmd.getConfig(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  NativeHost host(cfg, modelFile, build);
  BlueprintsPractice::setHost(&host);
  for(string line; getline(cin, line);) {
    if(Global::trim(line).empty())
      continue;
    bool quit = false;
    cout << BlueprintsPractice::handle(line, quit) << flush;
    if(quit)
      break;
  }
  BlueprintsPractice::finish();
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}

//Blueprints' own command (game/blueprints/ownfiles.txt): only Blueprints' binary has it.
const MainCmds::GameCommand BLUEPRINTS_PRACTICE(
  "blueprints-practice", "Blueprints' practice bot: GTP in strategyjam.org's moves, with GTP's clock", blueprintsPractice);

}  // namespace
