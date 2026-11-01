#ifndef COMMAND_GTPSESSION_H_
#define COMMAND_GTPSESSION_H_

//Enclosure fork (games/enclosure/docs/design/18-play-through-gtp.md): KataGo's GTP engine one command at a time, for a host without
//standard input or threads (the WebAssembly bot, games/enclosure/wasm/bot.cpp). The session is the one MainCmds::gtp drives from
//standard input (command/gtp.cpp, GTPSession).

#include <string>

#include "../game/chosenrules.h"

class ConfigParser;
class GTPSession;

namespace GTPSessions {
  //A session with the settings cfg, which must outlive it, and the network in nnModelFile.
  GTPSession* start(ConfigParser& cfg, const std::string& nnModelFile);
  //One command line: what the engine prints for it, which is its GTP response. Sets quit after the quit command.
  std::string handle(GTPSession* session, const std::string& line, bool& quit);
  //A new game from the set-up start (the practice bot, games/enclosure/docs/design/22-practice-bot.md), as clear_board starts one
  //from the standard start: undo stops at it, and its turn number is its moves played. The caller checks the start.
  void setStart(GTPSession* session, const GameRules::Position& start);
  //KataGame fork (Blueprints' native practice engine, command/blueprintspracticeengine.cpp): ends the session as
  //MainCmds::gtp ends its own (its search and pondering stop), and deletes it.
  void finish(GTPSession* session);
}

#endif  // COMMAND_GTPSESSION_H_
