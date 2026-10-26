// Blueprints' engine protocol (the engine repository's docs/design/04-notation-and-protocol.md, with Blueprints'
// fields: its games/blueprints/README.md, "The browser build"). The same code runs in the WebAssembly module
// (games/enclosure/wasm/engine.cpp, built for Blueprints by games/enclosure/wasm/build.sh) and in the native command
// (command/blueprintsengine.cpp, `katago blueprints-engine`); each gives it a transport. No exceptions leave it: a bad
// request gets an error reply. It has the interface of Enclosure's protocol (command/enclosureprotocol.h), without the
// GPU module's network reader (modelReader), so the CPU module's transport serves either game.
//
// One engine (one search, its tree, its search in slices) for the process: the WebAssembly module and the native
// command each have one thread that calls handle().
#ifndef COMMAND_BLUEPRINTSPROTOCOL_H_
#define COMMAND_BLUEPRINTSPROTOCOL_H_

#include <cstddef>
#include <string>

namespace BlueprintsProtocol {
  // The base settings (an analysis config's text, e.g. configs/analysis_blueprints.cfg with any overrides appended)
  // and the build's name, given before the first request.
  void setup(const std::string& configText, const std::string& build);
  // A buffer for a network's bytes (a .bin model, as model-float16.bin); the load_model action without a path then
  // loads it, and frees it once loaded. The buffer stays valid until the next call or that load.
  char* modelBuffer(size_t len);
  // One request (a JSON object, as text) -> its reply (a JSON object on one line, no newline).
  std::string handle(const std::string& request);
}

#endif  // COMMAND_BLUEPRINTSPROTOCOL_H_
