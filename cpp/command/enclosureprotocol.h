// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md). The same code runs in the WebAssembly
// module (v2/wasm/engine.cpp) and in the native command (command/enclosureengine.cpp, `katago enclosure-engine`); each
// gives it a transport. No exceptions leave it: a bad request gets an error reply.
//
// One engine (one search, its tree, its search in slices) for each thread that calls handle(). The WebAssembly module
// has one thread, so one engine. The native command's -parallel runs one engine in each of several threads, and
// setParallel makes their engines share one network evaluator.
#ifndef COMMAND_ENCLOSUREPROTOCOL_H_
#define COMMAND_ENCLOSUREPROTOCOL_H_

#include <cstddef>
#include <functional>
#include <string>

namespace EnclosureProtocol {
  // The base settings (an analysis config's text, e.g. configs/analysis_enclosure.cfg with any overrides appended)
  // and the build's name, given before the first request.
  void setup(const std::string& configText, const std::string& build);
  // Before the first request, when `engines` threads (more than one) will each run an engine: the engines that load
  // the same network with the same settings then share one network evaluator, created for all their evaluations at
  // once, so that its batches gather every engine's evaluations. Native builds only.
  void setParallel(int engines);
  // A buffer for a network's bytes (a .bin model, as model-float16.bin); the load_model action without a path then
  // loads it, and frees it once loaded. The buffer stays valid until the next call or that load.
  char* modelBuffer(size_t len);
  // In place of the buffer: a network of `len` bytes that the caller holds, with its SHA-256 (hex), read in chunks
  // through read(offset, dst, n) -> bytes copied, so that the engine never holds a copy of it (the GPU module's page,
  // which runs many engines of one network). The load_model action without a path then loads it.
  void modelReader(size_t len, const std::string& sha256, const std::function<size_t(size_t, char*, size_t)>& read);
  // One request (a JSON object, as text) -> its reply (a JSON object on one line, no newline).
  std::string handle(const std::string& request);
}

#endif  // COMMAND_ENCLOSUREPROTOCOL_H_
