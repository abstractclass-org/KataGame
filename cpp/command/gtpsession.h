#ifndef COMMAND_GTPSESSION_H_
#define COMMAND_GTPSESSION_H_

//Enclosure fork (v2/docs/design/18-play-through-gtp.md): KataGo's GTP engine one command at a time, for a host without
//standard input or threads (the WebAssembly bot, v2/wasm/bot.cpp). The session is the one MainCmds::gtp drives from
//standard input (command/gtp.cpp, GTPSession).

#include <string>

class ConfigParser;
class GTPSession;
namespace enclosure { struct Position; }

namespace GTPSessions {
  //A session with the settings cfg, which must outlive it, and the network in nnModelFile.
  GTPSession* start(ConfigParser& cfg, const std::string& nnModelFile);
  //One command line: what the engine prints for it, which is its GTP response. Sets quit after the quit command.
  std::string handle(GTPSession* session, const std::string& line, bool& quit);
  //A new game from the set-up start (the practice bot, v2/docs/design/22-practice-bot.md), as clear_board starts one
  //from the standard start: undo stops at it, and its turn number is its edge count. The caller checks the start.
  void setStart(GTPSession* session, const enclosure::Position& start);
}

#endif  // COMMAND_GTPSESSION_H_
