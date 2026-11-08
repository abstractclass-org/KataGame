// Blueprints' practice bot, the session of both hosts (command/blueprintspractice.h; the engine repository's
// games/blueprints/README.md, "The practice bot").
//
// The response is GTP's: "= TEXT" or "? ERROR", then a blank line. The moves are strategyjam.org's strings
// (command/blueprintssite.h): a placement "K@q,r" (blueprint K's anchor cell on the board's hex q,r), an edit
// "I:q,r>J:q,r" as one move, and "pass". The commands:
// - blueprints-start SEED [THREADS]: starts the session with the host's network and SEED (decimal, up to 2^64 - 1) for
//   the search's and the symmetries' random numbers. THREADS (default 1) is the search threads, up to the host's: 1 in
//   the browser module.
// - blueprints-bot-info: one JSON line with the interface's version, the build, the rules, the network (name, SHA-256,
//   bytes) and the search settings in force (kata-get-params). It works before blueprints-start too (no network, no
//   settings).
// - play COLOR MOVE, genmove COLOR, undo, clear_board: KataGo's, with the site's moves. COLOR is b (Blue, KataGo's
//   Black) or w (Red). An edit is two actions of the engine, the pick and the drop: play sends both, genmove searches
//   both (a second genmove, from the pick), and undo takes both back. A turn is two moves after Blue's first, so a
//   takeback of a turn is two undos.
// - blueprints-set-game [MOVE ...]: a new game from the start with the moves played, as clear_board and then play for
//   each move, for a page that reloads its game or a client that joins a game. undo takes them back as any other
//   moves. A refused move leaves the game as it was.
// - blueprints-set-position POSITION: a new game from a set-up start, the engine's one-line position (as the engine
//   protocol's queries take it), as clear_board starts one from the standard start. undo stops at the start, and the
//   early phase of the move temperature counts the start's moves played. A position that no game reaches is refused
//   with the position check's first reason (game/blueprints/positioncheck.h), a pick that waits for its drop too. A
//   refused position leaves the game as it was.
// - kata-set-param SETTING VALUE: maxVisits and the chosen move's settings (chosenMoveTemperature and the other
//   chosenMove* settings), each in the range of KataGo's config reader.
// - kata-time_settings ... and time_left COLOR TIME STONES, with a host that takes the clock (the native engine):
//   KataGo's. GTP's clock then sets each search's time. An edit is one move of the clock: between the pick and the
//   drop, the session gives GTP the mover's time left (the last time_left, less the time since it came), and the
//   fork's clock gives the drop no share of the increment, which the pick's search had (search/timecontrols.cpp).
// - komi K, kata-get-param, kata-get-params, protocol_version, name, version, quit: KataGo's. list_commands and
//   known_command answer for the host's list.
// Every other command is refused ("? unknown command"), in both hosts: the browser module has no WebAssembly
// exceptions, so a throw stops it, and KataGo's handlers of other commands throw on their arguments or on files.
// genmove searches until maxVisits or the clock's time, and as the native genmove does with the root's views: the usual
// search, with the tree kept from move to move (and from pondering, with ponderingEnabled), when the rule's view holds
// every legal placement; else the passes and the checks (program/rootviews.h), which keep no tree. Resignation is off
// (GTP's default).
//
// A placement whose cells do not all lie in the mover's view is the move in the view whose middle is its anchor hex, as
// GTP takes a move of another view ("MOVE/VIEW", command/gtp.cpp), and genmove's answer in another view becomes the
// site's string. The session keeps the game's position beside GTP's, to read and write the site's strings.
#include "../command/blueprintspractice.h"

#include "../command/blueprintssite.h"
#include "../command/gtpsession.h"
#include "../core/global.h"
#include "../game/blueprints/positioncheck.h"
#include "../game/board.h"
#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using json = nlohmann::json;
using std::string;
using std::vector;
namespace bp = blueprints;
namespace site = BlueprintsSite;
static_assert(std::is_same<GameRules, bp::GameRules>::value, "Blueprints' build (KATAGAME_GAME=blueprints)");

namespace {

// The interface's version, by the engine protocol's rules (design 04). Its versions: the README's "The practice bot".
const char* INTERFACE_VERSION = "0.1.3";

// The commands of every host (the head of this file), and those of a host with a clock. Every other command is refused
// before the engine.
const vector<string> COMMANDS = {
  "protocol_version", "name", "version", "known_command", "list_commands", "quit",
  "blueprints-start", "blueprints-bot-info", "blueprints-set-game", "blueprints-set-position",
  "clear_board", "komi", "play", "genmove", "undo", "kata-get-param", "kata-get-params", "kata-set-param",
};
const vector<string> CLOCK_COMMANDS = {"kata-time_settings", "time_left"};

// The settings that kata-set-param takes, with the ranges of KataGo's config reader (Setup::loadParams), as Enclosure's
// practice bot takes them.
struct Setting {
  const char* key;
  bool whole;  // a whole number (ConfigParser::getInt64), else a number (getDouble)
  double lo, hi;
};
const Setting SETTINGS[] = {
  {"maxVisits", true, 1.0, double(int64_t(1) << 50)},
  {"chosenMoveTemperature", false, 0.0, 5.0},
  {"chosenMoveTemperatureEarly", false, 0.0, 5.0},
  {"chosenMoveTemperatureHalflife", false, 0.1, 100000.0},
  {"chosenMoveTemperatureOnlyBelowProb", false, 0.0, 1.0},
  {"chosenMoveSubtract", false, 0.0, 1.0e10},
  {"chosenMovePrune", false, 0.0, 1.0e10},
};

// One action of the engine as GTP played it, and the view that named it: "" for the position's own view, else a view's
// text "q,r,o" (GTP's MOVE/VIEW).
struct Action {
  int action;
  string view;
};
// One of the site's moves: its actions (two for an edit, the pick and the drop).
using Move = vector<Action>;

// The last time_left of a colour, for an edit's drop: its time, its stones, and when it came.
struct ClockReading {
  bool set = false;
  double seconds = 0;
  string stones;
  std::chrono::steady_clock::time_point at;
};

BlueprintsPractice::Host* HOST = nullptr;
GTPSession* SESSION = nullptr;
BlueprintsPractice::Network NETWORK;
uint64_t SEED = 0;
int THREADS = 1;
vector<Move> MOVES;  // the session's game since START
bp::Position* START = nullptr;  // the session's start: the standard start, or a set-up start (blueprints-set-position)
bp::Position* POS = nullptr;  // the position after MOVES, as GTP has it
ClockReading CLOCK[2];  // by GTP's colour: 0 Black (Blue), 1 White (Red)

// A position for the work of one command beside the game's own (on the heap: a new position clears its whole rules
// board, and a copy into one reads only the copy's cells). 0: a game replayed (blueprints-set-game); 1: a copy for one
// move.
bp::Position& scratch(int i) {
  static bp::Position* positions[2] = {new bp::Position(), new bp::Position()};
  return *positions[i];
}

string ok(const string& text) { return "= " + text + "\n\n"; }
string fail(const string& text) { return "? " + text + "\n\n"; }

string engine(const string& line) {
  bool quit = false;
  return GTPSessions::handle(SESSION, line, quit);
}

// Whether a GTP response is an error ("? TEXT\n\n").
bool failed(const string& response) { return response.empty() || response[0] != '='; }

// The text of a GTP response ("= TEXT\n\n").
string text(const string& response) { return Global::trim(response.size() > 1 ? response.substr(1) : ""); }

// The commands of the host.
vector<string> commands() {
  vector<string> list = COMMANDS;
  if(HOST->clock())
    list.insert(list.end(), CLOCK_COMMANDS.begin(), CLOCK_COMMANDS.end());
  return list;
}

// GTP's colour of the player to move at p.
string colorOf(const bp::Position& p) { return bp::turn_of(p.ply) == 0 ? "b" : "w"; }

// GTP's text of an action: the move, then "/" and its view when it has one (command/gtp.cpp, moveInViewText).
string gtpText(const Action& a) {
  return GameRules::action_text(a.action) + (a.view.empty() ? "" : "/" + a.view);
}

// Plays an action at p as GTP plays it: in its view, which the move then ends.
void playAction(bp::Position& p, const Action& a) {
  string why;
  if(!a.view.empty())
    GameRules::set_view(p, a.view, why);
  GameRules::play(p, a.action);
}

// GTP's text of a move (genmove's answer) at p: its action and its view. False for any other text ("resign").
bool readGtpMove(const string& gtp, Action& a) {
  const size_t slash = gtp.rfind('/');
  a.view = slash == string::npos ? "" : gtp.substr(slash + 1);
  const string move = gtp.substr(0, slash);
  if(move == "pass") {
    a.action = GameRules::PASS_ACTION;
    return a.view.empty();
  }
  return GameRules::parse_protocol_text(move, a.action);
}

// The site's string of the action at p, read in its view.
string siteOf(const bp::Position& p, const Action& a) {
  if(a.action == GameRules::PASS_ACTION)
    return "pass";
  if(a.view.empty())
    return site::siteText(p, a.action);
  bp::Position& v = scratch(1);
  v = p;
  string why;
  GameRules::set_view(v, a.view, why);
  return site::siteText(v, a.action);
}

// The engine's actions of the site's move `text` at p, for the player to move: one for a placement or a pass, the pick
// and the drop for an edit. False with the reason when the move is not legal, or when no view of the engine names it.
// The engine protocol reads the move with the same code (site::readSiteMove).
bool actionsOf(const bp::Position& p, const string& text, Move& out, string& why) {
  out.clear();
  const site::SiteMove m = site::readSiteMove(p, text, scratch(1));
  if(m.kind == site::SiteMove::ACTIONS) {
    for(int action : m.actions)
      out.push_back({action, ""});
    return true;
  }
  if(m.kind != site::SiteMove::OUTSIDE) {
    why = m.why;
    return false;
  }
  // Outside the mover's view: the action in the view whose middle is the anchor hex, in orientation 0, which holds
  // every cell of the placement (a blueprint spans less than a view's radius from its anchor cell).
  bp::Position& v = scratch(1);
  v = p;
  string unused;
  if(GameRules::set_view(v, bp::cell_text(m.q, m.r) + ",0", unused)) {
    const int action = site::placementAction(v, m.k, m.q, m.r, text);
    if(action >= 0) {
      out.push_back({action, GameRules::view_text(v)});
      return true;
    }
  }
  // The site's rule allows it, but a cell lies off the rules board, which the engine's actions do not reach (the
  // engine protocol moves the cells for it: bp::place_at_site).
  why = text + ": a cell of it lies past this build's rules board of " + std::to_string(bp::RULES_LEN) + " x " +
        std::to_string(bp::RULES_LEN) + " hexes, where the bot cannot play";
  return false;
}

// POS after MOVES, from START.
void replay() {
  *POS = *START;
  for(const Move& m : MOVES)
    for(const Action& a : m)
      playAction(*POS, a);
}

// The session's game from the standard start, with no moves.
void newGame() {
  GameRules::new_game(bp::VIEW, bp::VIEW, *START);
  MOVES.clear();
  replay();
}

// The seat of GTP's colour COLOR (0 for b, Blue; 1 for w, Red), as bp::turn_of and CLOCK number them, or -1 when the
// colour does not parse.
int seatOf(const string& color) {
  Player pla;
  if(!PlayerIO::tryParsePlayer(color, pla))
    return -1;
  return pla == P_BLACK ? 0 : 1;
}

// GTP's refusal of a move by COLOR (command/gtp.cpp, turnError), or "".
string turnError(const string& color) {
  const int seat = seatOf(color);
  if(seat < 0)
    return "Could not parse color: '" + color + "'";
  if(GameRules::terminal(*POS))
    return "the game is over";
  if(seat != bp::turn_of(POS->ply))
    return "it is " + colorOf(*POS) + "'s turn";
  return "";
}

string start(const vector<string>& args) {
  if(args.size() != 2 && args.size() != 3)
    return fail("usage: blueprints-start SEED [THREADS]");
  const string notReady = HOST->notReady();
  if(!notReady.empty())
    return fail(notReady);
  char* end = nullptr;
  const unsigned long long seed = std::strtoull(args[1].c_str(), &end, 10);
  if(end == args[1].c_str() || *end != '\0')
    return fail("the seed is not a whole number: " + args[1]);
  int64_t threads = 1;
  if(args.size() == 3 && (!Global::tryStringToInt64(args[2], threads) || threads < 1 || threads > HOST->maxThreads()))
    return fail(HOST->maxThreads() == 1
                  ? "this build searches with one thread: THREADS is 1, not " + args[2]
                  : "THREADS is a whole number from 1 to " + std::to_string(HOST->maxThreads()) + ", not " + args[2]);
  if(SESSION != nullptr)
    return fail("the session has started; another seed needs a new module or engine process");
  SEED = seed;
  THREADS = int(threads);
  // The session's settings over the host's: GTP's rules (the one key of GTP that the analysis settings lack: the
  // native GTP engine of games/blueprints/tools/gtp_root_views.py adds it too), the search threads and the seed.
  SESSION = HOST->startEngine({
    {"rules", "tromp-taylor"},
    {"numSearchThreads", std::to_string(THREADS)},
    {"searchRandSeed", "practice:" + std::to_string(seed)},
    {"nnRandSeed", "practice:" + std::to_string(seed)},
  });
  NETWORK = HOST->network(SESSION);
  START = new bp::Position();
  POS = new bp::Position();
  newGame();
  return ok("");
}

// kata-set-param SETTING VALUE, checked as KataGo's config reader checks it (Global::tryStringToInt64 and
// tryStringToDouble are its parsers), so that KataGo's handler cannot throw.
string setParam(const vector<string>& args, const string& line) {
  if(args.size() != 3)
    return fail("usage: kata-set-param SETTING VALUE");
  for(const Setting& s : SETTINGS) {
    if(args[1] != s.key)
      continue;
    bool good;
    if(s.whole) {
      int64_t x = 0;
      good = Global::tryStringToInt64(args[2], x) && x >= s.lo && x <= s.hi;
    }
    else {
      double x = 0;
      good = Global::tryStringToDouble(args[2], x) && x >= s.lo && x <= s.hi;  // NaN fails both comparisons
    }
    if(!good)
      return fail(args[1] + " is " + (s.whole ? "a whole number from " + Global::int64ToString(int64_t(s.lo)) + " to " +
                                                  Global::int64ToString(int64_t(s.hi))
                                              : "a number from " + Global::doubleToString(s.lo) + " to " +
                                                  Global::doubleToString(s.hi)));
    return engine(line);
  }
  return fail("the settings are maxVisits and the chosenMove* settings, not " + args[1]);
}

string info() {
  json reply = {{"interface", INTERFACE_VERSION}, {"build", HOST->build()}, {"rules", site::RULES}};
  if(SESSION == nullptr) {
    reply["network"] = nullptr;
    reply["settings"] = nullptr;
  }
  else {
    reply["network"] = {{"name", NETWORK.name}, {"sha256", NETWORK.sha256}, {"bytes", NETWORK.bytes}};
    const string params = engine("kata-get-params");
    json settings = failed(params) ? json() : json::parse(text(params), nullptr, false);
    if(settings.is_discarded())
      settings = json();
    settings["seed"] = std::to_string(SEED);
    settings["threads"] = THREADS;
    reply["settings"] = settings;
  }
  return ok(reply.dump());
}

string play(const vector<string>& args) {
  if(args.size() != 3)
    return fail("usage: play COLOR MOVE");
  const string turn = turnError(args[1]);
  if(!turn.empty())
    return fail(turn);
  Move m;
  string why;
  if(!actionsOf(*POS, args[2], m, why))
    return fail(why);
  for(size_t i = 0; i < m.size(); i++) {
    const string response = engine("play " + args[1] + " " + gtpText(m[i]));
    if(failed(response)) {
      for(size_t j = 0; j < i; j++)
        engine("undo");
      return response;
    }
  }
  for(const Action& a : m)
    playAction(*POS, a);
  MOVES.push_back(m);
  return ok("");
}

// time_left COLOR TIME STONES: GTP's, and the reading for an edit's drop when GTP takes it.
string timeLeft(const vector<string>& args, const string& line) {
  const string response = engine(line);
  const int seat = args.size() == 4 ? seatOf(args[1]) : -1;
  double seconds = 0;
  if(!failed(response) && seat >= 0 && Global::tryStringToDouble(args[2], seconds)) {
    ClockReading& c = CLOCK[seat];
    c.set = true;
    c.seconds = seconds;
    c.stones = args[3];
    c.at = std::chrono::steady_clock::now();
  }
  return response;
}

// Before an edit's drop: GTP's clock of the mover gets the time left after the pick, the last time_left less the time
// since it came. Nothing without a time_left since the last genmove (always so in a host with no clock, which refuses
// time_left).
void clockBeforeDrop(const string& color) {
  const int seat = seatOf(color);
  if(seat < 0 || !CLOCK[seat].set)
    return;
  const ClockReading& c = CLOCK[seat];
  const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - c.at).count();
  engine("time_left " + color + " " + Global::doubleToString(std::max(0.0, c.seconds - spent)) + " " + c.stones);
}

// genmove COLOR: GTP's move, and after a pick GTP's drop from it (a second genmove), as the site's string of the move.
// The colour's clock reading counts for this move only: the next move needs its own time_left.
string genmove(const vector<string>& args, const string& line) {
  const string response = engine(line);
  if(failed(response))
    return response;
  string color = args.size() > 1 ? args[1] : "";
  Action first;
  if(!readGtpMove(text(response), first))
    return ok(text(response));  // "resign", which the play settings turn off
  string move = siteOf(*POS, first);
  Move m{first};
  const bool pick = site::isPick(*POS, first.action);
  playAction(*POS, first);
  if(pick) {
    clockBeforeDrop(color);
    const string dropResponse = engine(line);
    Action drop;
    if(failed(dropResponse) || !readGtpMove(text(dropResponse), drop)) {
      // No drop: the pick goes back, and the game is as it was.
      engine("undo");
      replay();
      return failed(dropResponse) ? dropResponse : fail("no drop after the pick " + move + ": " + text(dropResponse));
    }
    move += ">" + siteOf(*POS, drop);
    m.push_back(drop);
    playAction(*POS, drop);
  }
  const int seat = seatOf(color);
  if(seat >= 0)
    CLOCK[seat].set = false;
  MOVES.push_back(m);
  return ok(move);
}

string undo() {
  if(MOVES.empty())
    return fail("cannot undo");
  for(size_t i = 0; i < MOVES.back().size(); i++) {
    const string response = engine("undo");
    if(failed(response))
      return response;
  }
  MOVES.pop_back();
  replay();
  return ok("");
}

// blueprints-set-game [MOVE ...]: every move is checked first, from the start, so that a refused move changes nothing.
string setGame(const vector<string>& args) {
  vector<Move> moves;
  bp::Position& p = scratch(0);
  GameRules::new_game(bp::VIEW, bp::VIEW, p);
  for(size_t i = 1; i < args.size(); i++) {
    Move m;
    string why;
    if(!actionsOf(p, args[i], m, why))
      return fail("move " + std::to_string(i - 1) + ": " + why);
    for(const Action& a : m)
      playAction(p, a);
    moves.push_back(m);
  }
  const string cleared = engine("clear_board");
  if(failed(cleared))
    return cleared;
  newGame();
  for(size_t i = 0; i < moves.size(); i++) {
    for(const Action& a : moves[i]) {
      const string response = engine("play " + colorOf(*POS) + " " + gtpText(a));
      if(failed(response)) {
        engine("clear_board");
        newGame();
        return fail("move " + std::to_string(i) + ": " + text(response));
      }
      playAction(*POS, a);
    }
    MOVES.push_back(moves[i]);
  }
  return ok("");
}

// blueprints-set-position POSITION: the position's fields follow the command's name. The start is read and checked
// first, so that a refused position changes nothing. GTP's game then starts from it (GTPSessions::setStart): its turn
// number is the start's moves played, and undo stops there.
string setPosition(const vector<string>& args) {
  if(args.size() < 2)
    return fail("usage: blueprints-set-position POSITION");
  bp::Position& start = scratch(0);
  string why;
  if(!bp::position_check::read_start(Global::concat(args, " ", 1, args.size()), start, why))
    return fail(why);
  const vector<bp::position_check::Reason> reasons = bp::position_check::check(start);
  if(!reasons.empty())
    return fail(reasons[0].text);
  GTPSessions::setStart(SESSION, start);
  *START = start;
  MOVES.clear();
  replay();
  return ok("");
}

}  // namespace

void BlueprintsPractice::setHost(Host* host) { HOST = host; }

std::string BlueprintsPractice::handle(const std::string& line, bool& quit) {
  quit = false;
  std::istringstream in(line);
  vector<string> args;
  for(string w; in >> w;)
    args.push_back(w);
  if(args.empty())
    return fail("empty command");
  const string& command = args[0];
  const vector<string> known = commands();
  const auto isKnown = [&known](const string& c) { return std::find(known.begin(), known.end(), c) != known.end(); };
  if(!isKnown(command))
    return fail("unknown command");
  if(command == "list_commands") {
    string list;
    for(const string& c : known)
      list += (list.empty() ? "" : "\n") + c;
    return ok(list);
  }
  if(command == "known_command")
    return ok(args.size() == 2 && isKnown(args[1]) ? "true" : "false");
  if(command == "blueprints-bot-info")
    return info();
  if(command == "blueprints-start")
    return start(args);
  if(SESSION == nullptr)
    return fail("no session: send blueprints-start SEED first");
  if(command == "blueprints-set-game")
    return setGame(args);
  if(command == "blueprints-set-position")
    return setPosition(args);
  if(command == "play")
    return play(args);
  if(command == "genmove")
    return genmove(args, line);
  if(command == "undo")
    return undo();
  if(command == "kata-set-param")
    return setParam(args, line);
  if(command == "time_left")
    return timeLeft(args, line);
  const string response = GTPSessions::handle(SESSION, line, quit);
  if(!failed(response) && command == "clear_board")
    newGame();
  return response;
}

void BlueprintsPractice::finish() {
  if(SESSION != nullptr)
    GTPSessions::finish(SESSION);
  SESSION = nullptr;
}
