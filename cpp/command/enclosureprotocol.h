// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md), one engine per process. The same code
// runs in the WebAssembly module (v2/wasm/engine.cpp) and in the native command (command/enclosureengine.cpp,
// `katago enclosure-engine`); each gives it a transport. No exceptions leave it: a bad request gets an error reply.
#ifndef COMMAND_ENCLOSUREPROTOCOL_H_
#define COMMAND_ENCLOSUREPROTOCOL_H_

#include <cstddef>
#include <string>

namespace EnclosureProtocol {
  // The base settings (an analysis config's text, e.g. configs/analysis_enclosure.cfg with any overrides appended)
  // and the build's name, given before the first request.
  void setup(const std::string& configText, const std::string& build);
  // A buffer for a network's bytes (a .bin model, as model-float16.bin); the load_model action without a path then
  // loads it. The buffer stays valid until the next call.
  char* modelBuffer(size_t len);
  // One request (a JSON object, as text) -> its reply (a JSON object on one line, no newline).
  std::string handle(const std::string& request);
}

#endif  // COMMAND_ENCLOSUREPROTOCOL_H_
