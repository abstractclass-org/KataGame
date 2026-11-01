#ifndef COMMAND_BLUEPRINTSPRACTICE_H_
#define COMMAND_BLUEPRINTSPRACTICE_H_

// Blueprints' practice bot (the engine repository's games/blueprints/README.md, "The practice bot"): KataGo's GTP
// engine (command/gtp.cpp, one command at a time through command/gtpsession.h) with the practice interface, in
// strategyjam.org's strings (command/blueprintssite.h). This is the one code of the session, for its two hosts:
// - the browser module, blueprints-bot.wasm (the engine repository's games/blueprints/wasm/practice.cpp): one search
//   thread, the network from the page's bytes, moves to a number of visits, no clock;
// - the native engine, katago blueprints-practice (command/blueprintspracticeengine.cpp): the CUDA or Eigen backend,
//   search threads, the network from a file, and GTP's clock (kata-time_settings and time_left), for play under a
//   clock such as strategyjam.org's.
// The commands, the moves, the settings and the search are the same in both, so the two give the same moves at the
// same settings, seed and visits. The host gives only what differs: its build's name, its search threads, whether it
// takes the clock, its settings and its network.

#include <cstdint>
#include <map>
#include <string>

class GTPSession;

namespace BlueprintsPractice {

// The network of a session, for blueprints-bot-info.
struct Network {
  std::string name;
  std::string sha256;
  uint64_t bytes = 0;
};

// What differs between the hosts of the session.
class Host {
 public:
  virtual ~Host() {}
  // The build's name, for blueprints-bot-info.
  virtual std::string build() const = 0;
  // The most search threads that blueprints-start takes: 1 in the browser module.
  virtual int maxThreads() const = 0;
  // Whether the host takes GTP's clock: kata-time_settings and time_left (the native engine). The browser module
  // refuses them: KataGo's handlers of the time commands throw on some arguments, and a throw stops a module with no
  // WebAssembly exceptions.
  virtual bool clock() const = 0;
  // Why blueprints-start cannot start the session now (the browser module: no network yet), or "".
  virtual std::string notReady() const { return ""; }
  // KataGo's GTP session: the host's settings with `settings` over them (each key of `settings` wins).
  virtual GTPSession* startEngine(const std::map<std::string, std::string>& settings) = 0;
  // The network of the session that startEngine gave.
  virtual Network network(GTPSession* session) const = 0;
};

// The host of the commands. It must outlive them, and it is set once, before the first command.
void setHost(Host* host);

// One command line: its GTP response, "= TEXT\n\n" or "? ERROR\n\n". Sets quit after the command quit.
std::string handle(const std::string& line, bool& quit);

// Ends the session (its search and pondering stop), if it started. The native engine calls it before it exits.
void finish();

}  // namespace BlueprintsPractice

#endif  // COMMAND_BLUEPRINTSPRACTICE_H_
