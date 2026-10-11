// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md) as a native command, for the analysis
// site's offline jobs and our bots. The protocol is command/enclosureprotocol.cpp, the same code as the WebAssembly
// module (v2/wasm/engine.cpp); this is its transport: one JSON request a line on standard input, one JSON reply a line
// on standard output, until standard input closes.
//
//   katago enclosure-engine -config configs/analysis_enclosure.cfg [-model MODEL] [-override-config "k=v,k=v"]
//                           [-build NAME] [-parallel N]
//
// -model loads a network at start (as {"action": "load_model", "path": MODEL}); its reply is the first line. The
// settings' search threads (numSearchThreadsPerAnalysisThread) search one query faster; with more than one, results
// are not reproducible, and a search in slices still runs on one thread.
//
// With one engine (the default) the replies come in the order of the requests. -parallel N runs N engines, each in
// its own thread with its own search, sharing one network evaluator, so that the GPU gets one batch from all their
// searches (many positions at once, as KataGo's analysis engine with numAnalysisThreads). Each reply then comes when
// it is ready, matched by its "id":
// - a new search (an analysis) goes to an engine that holds no search in slices; when all do, it waits in a backlog
//   (the reading of requests goes on), and the next engine to free itself takes it; a search whose id an engine
//   already runs goes to that engine, which ends it and starts the new one (as one engine);
// - "continue" and "terminate" go to the engine that runs that id;
// - "load_model" goes to every engine, and one reply comes back (the others only when they fail);
// - other actions go to the engine with the least work.
// So a client must continue or terminate each search in slices that it begins: until then, its engine takes no new
// search.
#include "../command/enclosureprotocol.h"
#include "../command/commandline.h"
#include "../core/global.h"
#include "../game/board.h"
#include "../main.h"
#include "../neuralnet/nninputs.h"
#include "../external/nlohmann_json/json.hpp"

#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

using namespace std;

namespace {

// -parallel: one slot for each engine, run by its own thread.
struct Task {
  string line;
  string owns;  // the search (its id, as JSON text) that this request starts or continues, or ""
  string ends;  // the search that this request ends (terminate's target), or ""
  bool quiet = false;  // load_model's copies for engines after the first: written only when they fail
  bool resets = false;  // load_model: it ends any search in slices that the engine holds
};

struct Slot {
  deque<Task> tasks;
  bool busy = false;
  string holds;  // the search in slices that this engine holds between its slices, or ""
};

class Parallel {
 public:
  explicit Parallel(int n) : slots(n) {}

  int run(const string& modelFile) {
    vector<thread> threads;
    for(size_t i = 0; i < slots.size(); i++)
      threads.emplace_back([this, i]() { work(i); });
    if(!modelFile.empty()) {
      const nlohmann::json load = {{"id", "start"}, {"action", "load_model"}, {"path", modelFile}};
      toEvery(load.dump());
      unique_lock<mutex> lock(m);
      freed.wait(lock, [this]() { return allIdle(); });  // its reply is the first line, as with one engine
    }
    for(string line; getline(cin, line);) {
      if(Global::trim(line).empty())
        continue;
      dispatch(line);
    }
    {
      lock_guard<mutex> lock(m);
      closing = true;
    }
    queued.notify_all();
    for(thread& t : threads)
      t.join();
    return 0;
  }

 private:
  vector<Slot> slots;
  map<string, size_t> owner;  // a search in slices (its id) -> the engine that holds it
  deque<Task> backlog;  // new searches waiting for an engine that holds no search in slices
  mutex m, out;
  condition_variable queued, freed;
  bool closing = false;

  // Whether a search (its id) waits in the backlog (its continue and terminate then wait there too, behind it).
  bool inBacklog(const string& sid) const {
    for(const Task& t : backlog)
      if(t.owns == sid || t.ends == sid)
        return true;
    return false;
  }

  // Slot i, free now, takes the oldest search of the backlog with the requests that wait behind it for the same search
  // (under the lock); whether it took one.
  bool takeBacklog(size_t i) {
    Slot& slot = slots[i];
    if(backlog.empty() || slot.busy || !slot.tasks.empty() || !slot.holds.empty())
      return false;
    const string sid = backlog.front().owns;
    owner[sid] = i;
    slot.holds = sid;
    for(auto it = backlog.begin(); it != backlog.end();) {
      if(it->owns == sid || it->ends == sid) {
        slot.tasks.push_back(*it);
        it = backlog.erase(it);
      }
      else
        ++it;
    }
    return true;
  }

  bool allIdle() const {
    for(const Slot& s : slots)
      if(s.busy || !s.tasks.empty())
        return false;
    return true;
  }

  // An engine with nothing to do and no search in slices, or -1.
  int freeSlot() const {
    for(size_t i = 0; i < slots.size(); i++)
      if(!slots[i].busy && slots[i].tasks.empty() && slots[i].holds.empty())
        return (int)i;
    return -1;
  }

  size_t leastLoaded() const {
    size_t best = 0;
    for(size_t i = 1; i < slots.size(); i++)
      if(slots[i].tasks.size() + slots[i].busy < slots[best].tasks.size() + slots[best].busy)
        best = i;
    return best;
  }

  void toEvery(const string& line) {
    {
      lock_guard<mutex> lock(m);
      for(size_t i = 0; i < slots.size(); i++)
        slots[i].tasks.push_back(Task{line, "", "", i > 0, true});
    }
    queued.notify_all();
  }

  void dispatch(const string& line) {
    const nlohmann::json q = nlohmann::json::parse(line, nullptr, false);
    const bool object = q.is_object();
    const string id = object && q.contains("id") ? q["id"].dump() : "";
    const string action = object && q.contains("action") && q["action"].is_string() ? q["action"].get<string>() : "";
    if(action == "load_model") {
      toEvery(line);
      return;
    }
    {
      unique_lock<mutex> lock(m);
      Task t{line, "", "", false, false};
      size_t target;
      const bool analysis = object && !q.contains("action") && !id.empty();
      const string ends = action == "terminate" ? (object && q.contains("terminateId") ? q["terminateId"].dump() : id) : "";
      if((action == "continue" && inBacklog(id)) || (action == "terminate" && inBacklog(ends))) {
        t.owns = action == "continue" ? id : "";
        t.ends = ends;
        backlog.push_back(t);  // its search waits for an engine: so does this request, behind it
        return;
      }
      if(analysis || action == "continue") {
        auto it = owner.find(id);
        if(it != owner.end())
          target = it->second;
        else if(analysis && inBacklog(id)) {
          t.owns = id;
          backlog.push_back(t);  // the same id again, while the first waits: it waits behind it (and then ends it)
          return;
        }
        else if(analysis) {
          t.owns = id;
          if(freeSlot() < 0) {
            backlog.push_back(t);  // no engine free: the next one to free itself takes it
            return;
          }
          target = (size_t)freeSlot();
        }
        else
          target = leastLoaded();  // a continue for no running search: its engine replies with the error
        if(analysis || it != owner.end()) {
          t.owns = id;
          owner[id] = target;
          slots[target].holds = id;  // reserved until its reply shows whether the search goes on
        }
      }
      else if(action == "terminate") {
        t.ends = ends;
        auto it = owner.find(t.ends);
        target = it != owner.end() ? it->second : leastLoaded();
      }
      else
        target = leastLoaded();
      slots[target].tasks.push_back(t);
    }
    queued.notify_all();
  }

  void work(size_t i) {
    Slot& slot = slots[i];
    for(;;) {
      Task t;
      {
        unique_lock<mutex> lock(m);
        queued.wait(lock, [&]() { return !slot.tasks.empty() || closing; });
        if(slot.tasks.empty() && !takeBacklog(i))
          return;  // input closed, and nothing left for this engine
        t = slot.tasks.front();
        slot.tasks.pop_front();
        slot.busy = true;
      }
      const string reply = EnclosureProtocol::handle(t.line);
      bool during = false;
      if(!t.owns.empty()) {
        const nlohmann::json r = nlohmann::json::parse(reply, nullptr, false);
        during = r.is_object() && r.value("isDuringSearch", false);
      }
      if(!t.quiet || reply.find("\"error\"") != string::npos) {
        lock_guard<mutex> lock(out);
        cout << reply << endl;
      }
      {
        lock_guard<mutex> lock(m);
        auto drop = [&](const string& sid) {
          auto it = owner.find(sid);
          if(it != owner.end() && it->second == i)
            owner.erase(it);
          if(slot.holds == sid)
            slot.holds.clear();
        };
        if(!t.owns.empty() && !during)
          drop(t.owns);
        if(!t.ends.empty())
          drop(t.ends);
        if(t.resets && !slot.holds.empty())
          drop(slot.holds);
        slot.busy = false;
        takeBacklog(i);
      }
      freed.notify_all();
    }
  }
};

}  // namespace

int MainCmds::enclosureEngine(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string configFile, modelFile, overrides, build;
  int parallel = 1;
  KataGoCommandLine cmd("Run the Enclosure engine protocol: JSON lines on standard input and output.");
  try {
    TCLAP::ValueArg<string> configArg("", "config", "The analysis settings (configs/analysis_enclosure.cfg)", true, "", "FILE");
    TCLAP::ValueArg<string> modelArg("", "model", "A network to load at start", false, "", "FILE");
    TCLAP::ValueArg<string> overrideArg("", "override-config", "Settings over the config: k=v,k=v", false, "", "KEYVALUEPAIRS");
    TCLAP::ValueArg<string> buildArg("", "build", "The build's name in replies (default: native)", false, "native", "NAME");
    TCLAP::ValueArg<int> parallelArg("", "parallel", "Engines, one for each thread, sharing one network evaluator (default 1)", false, 1, "N");
    cmd.add(configArg);
    cmd.add(modelArg);
    cmd.add(overrideArg);
    cmd.add(buildArg);
    cmd.add(parallelArg);
    cmd.parseArgs(args);
    configFile = configArg.getValue();
    modelFile = modelArg.getValue();
    overrides = overrideArg.getValue();
    build = buildArg.getValue();
    parallel = parallelArg.getValue();
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
  if(parallel < 1) {
    cerr << "Error: -parallel is at least 1" << endl;
    return 1;
  }
  if(parallel > 1) {
    EnclosureProtocol::setParallel(parallel);
    return Parallel(parallel).run(modelFile);
  }

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
