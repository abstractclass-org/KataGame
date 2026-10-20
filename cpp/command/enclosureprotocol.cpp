// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md). See enclosureprotocol.h.
//
// Requests (each has an "id", echoed in its reply; an error reply is {"id", "error", "field"}):
//   {"action": "query_version"}                       the protocol, the build, the search (the first build that
//                                                     searched as this one), the model, the rules, trained outputs
//   {"action": "load_model", "path"?, "name", "settings"}
//                                                     a network: the file at "path" (the native engine), else the
//                                                     model buffer; settings override the base settings
//   {"action": "describe", "position", "moves"}      the position for a board: edges, nodes, regions, legal moves, and
//                                                     "check": whether "position" could occur in a game, with reasons
//                                                     (position_check.h); an analysis of one that cannot is refused
//   {"action": "describe_game", "position", "moves", "from"?}
//                                                     every position of the game from "from" on, with the moves played
//                                                     once: describe's fields without nodes, legal and check, and the
//                                                     edge that each move cuts
//   {"action": "play", "position", "moves", "move"}  whether the move is legal, what it cuts, the new position
//   {"position", "moves", "maxVisits", ...}          an analysis (no action), as KataGo's analysis engine; with
//                                                     "slice": {"visits": N} or {"ms": T}, one slice of it at a time
//   {"action": "continue"}                           the next slice of the search with this id
//   {"action": "terminate", "terminateId"}           ends that search; its last results, "terminated": true
// A position is "position" (the notation; default the standard start) followed by "moves" (edges, "pass").
// "history" (newest turn first, at most 4 turns of at most 2 edges) gives the last turns' edges of a position whose
// moves are not known, for the network's history planes. "overrideSettings" (KataGo's field) takes "maxTime"
// (seconds), which limits a search by time, and the chosen move's settings (chosenMoveTemperature and the other
// chosenMove* keys); other keys are refused. "chooseMove": true adds "chosenMove" to a search's final reply: the move
// KataGo's GTP engine would play, drawn with the chosen-move temperature, seeded by "seed" (a string or an integer,
// default 0) and the position. A position's turn number is its edge count, so the temperature's early phase follows
// the game. "includeWeakWalls" and "includeFutureNodes" add the root's maps from the network's training-only heads
// (rootMaps), from one evaluation of the root, not from the search.
// The rules come through the game interface (GameRules: the moves and the position in the notation, the legal moves,
// the end, the scores). Enclosure's own parts stay Enclosure's: the history planes' edges, the describe action's edges,
// nodes and regions, the cut edge, the outcome's text and the position check.
// Results are reproducible: each evaluation's symmetry comes from its cache key (nnSymmetryFromHash), and the search's
// random numbers (root noise) from the query's "seed" and the position, so the same request gives the same reply
// whatever came before, and a search in slices equals one whole search (with one search
// thread; more threads search faster and are not reproducible).
//
// One engine for each thread (PER_ENGINE): the native command's -parallel runs one in each of its threads. Their
// engines share one network evaluator (setParallel, acquireEvaluator), so that one batch gathers the evaluations of
// every engine. A batch's size can change the network's last digits on a GPU, so a reply then equals the reply of one
// engine alone to rounding; with batches of one, exactly.
#include "../command/enclosureprotocol.h"

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/sha2.h"
#include "../core/test.h"
#include "../core/timer.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/enclosure/notation.h"
#include "../game/enclosure/position_check.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../search/search.h"
#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

// An engine's state is per thread in native builds (the native command's -parallel), and a plain global in the
// WebAssembly module, which has one thread.
#ifdef __wasm__
#define PER_ENGINE
#else
#define PER_ENGINE thread_local
#endif

using json = nlohmann::json;
using std::string;
using std::vector;
namespace nt = enclosure::notation;

namespace {

// The interface's version (design 04, "Versioning"). Before 1.0, a breaking change takes the next minor version and a
// compatible change the next patch ("0.6" is 0.6.0).
const char* PROTOCOL = "0.6.2";
// The first build that searched as this one does (design 04, "The same search"), given as each reply's "search". A
// change that can alter an analysis reply sets it to "", and the build then names itself. After that build's release,
// it takes the released build's name. A change that leaves every analysis reply as it was keeps it.
const char* SEARCH_SINCE = "fdfe5ad";
const char* RULES = "online";  // the online variant (enclosure/README.md "Rules"), 19 x 19
const int DEFAULT_VISITS = 500;

// The process's settings (setup, setParallel), the same for every engine.
struct Process {
  string baseConfig, build = "unknown";
  int engines = 1;
};
Process P;

// A network that the caller holds and lends through a reader (modelReader): its size, SHA-256 and reader.
struct PendingReader {
  size_t len = 0;
  string sha256;
  std::function<size_t(size_t, char*, size_t)> read;
};

struct Engine {
  vector<char> pending;  // a network's bytes, written by the caller (modelBuffer), until load_model reads them
  PendingReader pendingReader;  // or the caller's reader of them (modelReader)
  string modelName, modelSha;
  size_t modelBytes = 0;
  Logger* logger = nullptr;
  ConfigParser* cfg = nullptr;
  NNEvaluator* nnEval = nullptr;
  string evalKey;  // the shared evaluator's key (sharedEvals), empty when the engine has its own
  Search* search = nullptr;
  SearchParams params;
};
PER_ENGINE Engine E;

// With several engines (setParallel): one evaluator for each network (its SHA-256) and settings text, shared by the
// engines that load the same, and deleted when the last of them loads another.
struct SharedEval {
  NNEvaluator* nnEval = nullptr;
  int users = 0;
};
std::mutex sharedMutex;
std::map<string, SharedEval> sharedEvals;

struct Fail {
  string field, why;
};

json errorReply(const json& id, const Fail& f) {
  json r = {{"id", id}, {"error", f.why}};
  if(!f.field.empty())
    r["field"] = f.field;
  return r;
}


const json* field(const json& q, const char* key) {
  auto it = q.find(key);
  return it == q.end() ? nullptr : &*it;
}

// A query's option such as "includeOwnership": true only when given as true.
bool flag(const json& q, const char* key) {
  const json* b = field(q, key);
  return b != nullptr && b->is_boolean() && b->get<bool>();
}

// The rules and size fields are optional; anything but the online variant at 19 x 19 is refused.
bool checkVariant(const json& q, Fail& f) {
  if(const json* r = field(q, "rules"))
    if(!r->is_string() || r->get<string>() != RULES) {
      f = {"rules", string("the only rules are \"") + RULES + "\""};
      return false;
    }
  for(const char* key : {"boardXSize", "boardYSize"})
    if(const json* s = field(q, key))
      if(!s->is_number_integer() || s->get<int>() != enclosure::SIZE) {
        f = {key, "the board is 19 x 19"};
        return false;
      }
  return true;
}

// "history": the last turns' edges, newest first, into the position's history planes.
bool readHistory(const json& q, enclosure::Position& p, Fail& f) {
  const json* h = field(q, "history");
  if(h == nullptr)
    return true;
  if(!h->is_array() || h->size() > (size_t)enclosure::HISTORY_TURNS) {
    f = {"history", "history is a list of at most " + std::to_string(enclosure::HISTORY_TURNS) + " turns, newest first"};
    return false;
  }
  for(size_t t = 0; t < h->size(); t++) {
    const json& turn = (*h)[t];
    if(!turn.is_array() || turn.size() > 2) {
      f = {"history", "a history turn is a list of at most 2 edges"};
      return false;
    }
    for(size_t i = 0; i < turn.size(); i++) {
      string why = "edges are strings";
      enclosure::Edge e;
      if(!turn[i].is_string() || !nt::parse_edge(turn[i].get<string>(), e, &why)) {
        f = {"history", why};
        return false;
      }
      p.history[t][i] = e;
    }
    p.history_n[t] = (uint8_t)turn.size();
  }
  return true;
}

// The query's position: "position" (default the start) with its "history", then "moves". Fills board, hist and the
// player to move. "each", if given, gets the position before the moves and the position after each move.
bool buildPosition(
  const json& q, Board& board, BoardHistory& hist, Player& pla, Fail& f,
  const std::function<void(const enclosure::Position&)>& each = nullptr
) {
  if(!checkVariant(q, f))
    return false;
  GameRules::Position start;
  if(const json* p = field(q, "position")) {
    string why;
    if(!p->is_string() || !GameRules::parse_position(p->get<string>(), start, why)) {
      f = {"position", p->is_string() ? why : "a position is a string"};
      return false;
    }
  }
  if(!readHistory(q, start, f))
    return false;
  board = Board(GameRules::x_size(start), GameRules::y_size(start));
  board.setPosition(start);
  pla = board.nextPla();
  Rules rules = Rules::getTrompTaylorish();
  if(const json* k = field(q, "komi")) {
    if(!k->is_number()) {
      f = {"komi", "komi is a number: points added to Red's final score"};
      return false;
    }
    rules.komi = k->get<float>();
  }
  hist = BoardHistory(board, pla, rules, 0);
  // The game's turn number counts edges from the standard start, so a position in the middle of a game is at its
  // edge count: the move temperature's early phase (Search::interpolateEarly) is then the game's, not the query's.
  hist.setInitialTurnNumber(GameRules::moves_played(start));
  if(each)
    each(board.pos);
  if(const json* moves = field(q, "moves")) {
    if(!moves->is_array()) {
      f = {"moves", "moves is a list of edges"};
      return false;
    }
    for(size_t i = 0; i < moves->size(); i++) {
      const json& m = (*moves)[i];
      string why = "moves are strings";
      const int action = m.is_string() ? GameRules::parse_action(board.pos, m.get<string>(), &why) : -1;
      if(action < 0 || !hist.makeBoardMoveTolerant(board, Board::moveLocOfAction(action), pla, false)) {
        f = {"moves", "move " + std::to_string(i) + ": " + why};
        return false;
      }
      pla = hist.presumedNextMovePla;
      if(each)
        each(board.pos);
    }
  }
  return true;
}

vector<string> sortedEdgeNames(const vector<enclosure::Edge>& edges) {
  vector<string> out;
  vector<enclosure::Edge> es = edges;
  std::sort(es.begin(), es.end(), enclosure::position_check::edge_less);
  for(const auto& e : es)
    out.push_back(nt::edge_name(e));
  return out;
}

vector<string> nodeNames(const vector<enclosure::Edge>& edges) {
  std::set<std::pair<int, int>> nodes;
  for(const auto& e : edges) {
    nodes.insert({e.ax, e.ay});
    nodes.insert({e.bx, e.by});
  }
  vector<string> out;
  for(const auto& [x, y] : nodes)
    out.push_back(nt::point_name(x, y));
  return out;
}

vector<string> legalNames(const GameRules::Position& p) {
  vector<int> legal;
  GameRules::legal_actions(p, legal);
  vector<string> out;
  for(int a : legal)
    out.push_back(GameRules::action_text(a));
  std::sort(out.begin(), out.end());
  return out;
}

json outcome(const GameRules::Position& p) {
  if(!GameRules::terminal(p))
    return nullptr;
  const double d = enclosure::snap(GameRules::score(p, 0) - GameRules::score(p, 1));
  const string result = d > 0 ? "b+" + nt::score_text(d) : d < 0 ? "r+" + nt::score_text(-d) : "0";
  return {{"result", result}};
}

// The position check (design 04, "The position check"; learn-enclosure's request 16) of a query's start, the position
// before its moves (hist.initialBoard; the moves themselves are checked as legal moves): {"possible", "reasons"}, each
// reason {"code", "text", "edges"}. A query with no position starts from the standard start, which passes.
json positionCheck(const enclosure::Position& start) {
  json reasons = json::array();
  for(const auto& r : enclosure::position_check::check(start))
    reasons.push_back({{"code", r.code}, {"text", r.text}, {"edges", r.edges}});
  return {{"possible", reasons.empty()}, {"reasons", reasons}};
}

// The fields of a position that a board shows: describe's, without the nodes, the legal moves and the check.
json boardFields(const enclosure::Position& p) {
  json regions = json::object(), areas = json::array(), edges = json::object();
  for(int side = 0; side < 2; side++) {
    const string name(1, nt::PLAYERS[side]);
    vector<vector<std::pair<double, double>>> polygons;
    const double area = enclosure::enclosed_area(p.edges[side], nullptr, nullptr, &polygons);
    json list = json::array();
    for(const auto& poly : polygons) {
      json pts = json::array();
      for(const auto& [x, y] : poly)
        pts.push_back({Global::roundDynamic(x, 9), Global::roundDynamic(y, 9)});
      list.push_back(pts);
    }
    regions[name] = std::move(list);
    areas.push_back(area);
    edges[name] = sortedEdgeNames(p.edges[side]);
  }
  return {
    {"position", GameRules::position_text(p)},
    {"mover", string(1, nt::PLAYERS[GameRules::next_player(p)])},
    {"movesLeft", GameRules::turn_moves_left(p)},
    {"moveCount", GameRules::moves_played(p)},
    {"scores", {GameRules::score(p, 0), GameRules::score(p, 1)}},
    {"edges", std::move(edges)},
    {"protected", sortedEdgeNames(p.protected_)},
    {"placed", sortedEdgeNames(p.placed)},
    {"regions", std::move(regions)},
    {"areas", std::move(areas)},
    {"outcome", outcome(p)},
  };
}

json describe(const json& q) {
  Board board;
  BoardHistory hist;
  Player pla;
  Fail f;
  if(!buildPosition(q, board, hist, pla, f))
    return errorReply(q["id"], f);
  const enclosure::Position& p = board.pos;
  json reply = boardFields(p), nodes = json::object();
  for(int side = 0; side < 2; side++)
    nodes[string(1, nt::PLAYERS[side])] = nodeNames(p.edges[side]);
  reply["id"] = q["id"];
  reply["nodes"] = std::move(nodes);
  reply["legal"] = legalNames(p);
  reply["check"] = positionCheck(hist.initialBoard.pos);
  return reply;
}

// The opponent's edge that a move from "before" to "after" cut, or null. A move removes at most one opponent edge and
// adds none.
json cutEdge(const enclosure::Position& before, const enclosure::Position& after) {
  const int enemy = 1 - before.player;
  if(after.edges[enemy].size() == before.edges[enemy].size())
    return nullptr;
  for(const auto& e : before.edges[enemy])
    if(!enclosure::contains(after.edges[enemy], e))
      return nt::edge_name(e);
  return nullptr;
}

// Every position of a game in one request (learn-enclosure's request 18): "position" and "moves" as describe's, played
// once. "from" (default 0) is the first position in the reply: 0 is the position before the moves, k the position after
// k moves. Each position has the board's fields (boardFields) and "cuts": the opponent's edge that the move to it cut,
// as play gives it (null for the position before the moves).
json describeGame(const json& q) {
  int from = 0;
  if(const json* fr = field(q, "from")) {
    const json* moves = field(q, "moves");
    const long long n = moves != nullptr && moves->is_array() ? (long long)moves->size() : 0;
    if(!fr->is_number_integer() || fr->get<long long>() < 0 || fr->get<long long>() > n)
      return errorReply(q["id"], {"from", "from is the first position's index: 0 to the number of moves"});
    from = fr->get<int>();
  }
  json positions = json::array();
  enclosure::Position before;
  int k = 0;
  const auto each = [&](const enclosure::Position& p) {
    if(k >= from) {
      json r = boardFields(p);
      r["cuts"] = k == 0 ? json(nullptr) : cutEdge(before, p);
      positions.push_back(std::move(r));
    }
    before = p;
    k++;
  };
  Board board;
  BoardHistory hist;
  Player pla;
  Fail f;
  if(!buildPosition(q, board, hist, pla, f, each))
    return errorReply(q["id"], f);
  return {{"id", q["id"]}, {"positions", std::move(positions)}};
}

json play(const json& q) {
  Board board;
  BoardHistory hist;
  Player pla;
  Fail f;
  if(!buildPosition(q, board, hist, pla, f))
    return errorReply(q["id"], f);
  const json* m = field(q, "move");
  if(m == nullptr || !m->is_string())
    return errorReply(q["id"], {"move", "play needs a move"});
  const enclosure::Position& before = board.pos;
  string why;
  const int action = GameRules::parse_action(before, m->get<string>(), &why);
  if(action < 0)
    return {{"id", q["id"]}, {"legal", false}, {"why", why}};
  enclosure::Position after = before;
  GameRules::play(after, action);  // PASS forfeits the rest of the turn (a timeout)
  const int enemy = 1 - before.player;
  const json cuts = cutEdge(before, after);
  const vector<string> nodesBefore = nodeNames(before.edges[enemy]), nodesAfter = nodeNames(after.edges[enemy]);
  json removed = json::array();
  for(const string& n : nodesBefore)
    if(std::find(nodesAfter.begin(), nodesAfter.end(), n) == nodesAfter.end())
      removed.push_back(n);
  return {
    {"id", q["id"]}, {"legal", true}, {"move", GameRules::action_text(action)}, {"cuts", cuts},
    {"removedNodes", removed}, {"position", GameRules::position_text(after)}, {"outcome", outcome(after)},
  };
}

string searchName() {
  return SEARCH_SINCE[0] != '\0' ? string(SEARCH_SINCE) : P.build;
}

json engineTag() {
  return {{"protocol", PROTOCOL}, {"build", P.build}, {"search", searchName()},
          {"model", E.modelSha.empty() ? json(nullptr) : json(E.modelSha)}};
}

json modelInfo() {
  return {{"name", E.modelName}, {"sha256", E.modelSha}, {"bytes", E.modelBytes}};
}

// Whether the loaded network gives the root's maps (rootMaps): a network that exports its training-only heads, on a
// backend that computes them (the GPU module's graph has none).
bool hasMaps() {
  return E.nnEval != nullptr && E.nnEval->hasAuxOutputs();
}

json queryVersion(const json& q) {
  json trained = json::array({"winrate", "scoreSelfplay", "scoreStdev", "scoreLead", "policy", "ownership"});
  if(hasMaps()) {
    trained.push_back("weakWalls");
    trained.push_back("futureNodes");
  }
  return {
    {"id", q["id"]}, {"protocol", PROTOCOL}, {"build", P.build}, {"search", searchName()},
    {"model", E.nnEval != nullptr ? modelInfo() : json(nullptr)}, {"rules", RULES}, {"trained", std::move(trained)},
  };
}

string sha256(const vector<char>& bytes) {
  char hash[65];
  SHA2::get256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), hash);  // of every byte
  return hash;
}

// The engine's evaluator given back: its own deleted, or the shared one's use counted down (deleted with its last).
void releaseEvaluator() {
  if(E.nnEval == nullptr)
    return;
  if(E.evalKey.empty())
    delete E.nnEval;
  else {
    std::lock_guard<std::mutex> lock(sharedMutex);
    auto it = sharedEvals.find(E.evalKey);
    if(it != sharedEvals.end() && --it->second.users == 0) {
      delete it->second.nnEval;
      sharedEvals.erase(it);
    }
  }
  E.nnEval = nullptr;
  E.evalKey.clear();
}

// The engine's evaluator for the network `file` with the settings `text`: its own with one engine, else the one that
// every engine loading the same network and settings shares, made for all their evaluations at once (so that, by
// default, a batch can hold one evaluation from each engine's search threads).
void acquireEvaluator(const string& file, const string& text, Setup::MaxBatchSizeRequest batch) {
  const int threads = std::max(1, E.params.numThreads);
  auto make = [&](int concurrent) {
    Rand seedRand("engine");
    return Setup::initializeNNEvaluator(
      "engine", file, "", *E.cfg, *E.logger, seedRand, concurrent, GameRules::MAX_BOARD_LEN, GameRules::MAX_BOARD_LEN,
      batch, true,
      false, Setup::SETUP_FOR_ANALYSIS
    );
  };
  if(P.engines <= 1) {
    E.nnEval = make(threads);
    return;
  }
  const string key = E.modelSha + "\n" + text;
  std::lock_guard<std::mutex> lock(sharedMutex);
  SharedEval& s = sharedEvals[key];
  if(s.nnEval == nullptr)
    s.nnEval = make(threads * P.engines);
  s.users++;
  E.nnEval = s.nnEval;
  E.evalKey = key;
}

json loadModel(const json& q) {
  const json* path = field(q, "path");
  if(path != nullptr && !path->is_string())
    return errorReply(q["id"], {"path", "path is a model file's path"});
  vector<char> fileBytes;
  if(path != nullptr) {
    std::ifstream in(path->get<string>(), std::ios::binary);
    if(!in)
      return errorReply(q["id"], {"path", "cannot read " + path->get<string>()});
    fileBytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  else {
#ifdef MODEL_FROM_MEMORY
    if(E.pending.empty() && !E.pendingReader.read)
      return errorReply(q["id"], {"", "no network: give a path, or fill the model buffer first"});
#else
    return errorReply(q["id"], {"path", "this build loads a network from a file: give its path"});
#endif
  }
  // By default each evaluation's symmetry from its cache key, so that the same request gives the same result (design
  // 04, "Reproducible analysis"); then the base settings and the request's settings, which can change it (a later line
  // wins).
  string text = "nnSymmetryFromHash = true\n" + P.baseConfig + "\n";
  if(const json* s = field(q, "settings")) {
    if(!s->is_object())
      return errorReply(q["id"], {"settings", "settings is an object of analysis settings"});
    for(auto it = s->begin(); it != s->end(); ++it) {
      const string value = it.value().is_string() ? it.value().get<string>() : it.value().dump();
      text += it.key() + " = " + value + "\n";
    }
  }
  delete E.search;
  E.search = nullptr;
  releaseEvaluator();
  delete E.cfg;
  string file;
  if(path != nullptr) {
    E.modelSha = sha256(fileBytes);
    E.modelBytes = fileBytes.size();
    file = path->get<string>();
  }
  else if(E.pendingReader.read) {
    // The caller's bytes, read where it holds them: the module never copies the whole network.
    E.modelSha = E.pendingReader.sha256;
    E.modelBytes = E.pendingReader.len;
#ifdef MODEL_FROM_MEMORY
    ModelDesc::registerInMemoryModelReader("engine", E.pendingReader.len, E.pendingReader.read);
#endif
    E.pendingReader = PendingReader();
    file = "memory:engine";
  }
  else {
    E.modelSha = sha256(E.pending);
    E.modelBytes = E.pending.size();
#ifdef MODEL_FROM_MEMORY
    ModelDesc::registerInMemoryModel("engine", E.pending.data(), E.pending.size());
#endif
    file = "memory:engine";
  }
  std::istringstream in(text);
  E.cfg = new ConfigParser(in, true);
  E.cfg->applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");
  if(E.logger == nullptr)
    E.logger = new Logger(nullptr, false, false, false);
  const json* name = field(q, "name");
  E.modelName = name != nullptr && name->is_string() ? name->get<string>() : (path != nullptr ? file : "");
  E.params = Setup::loadSingleParams(*E.cfg, Setup::SETUP_FOR_ANALYSIS);
  // As many evaluations at once as search threads (of every engine, with several); a batch size named in the settings
  // is taken as it is.
  const Setup::MaxBatchSizeRequest batch = E.cfg->contains("nnMaxBatchSize")
    ? Setup::MaxBatchSizeRequest::explicitSize(E.cfg->getInt("nnMaxBatchSize", 1, 65536))
    : Setup::MaxBatchSizeRequest::fromConcurrency();
  acquireEvaluator(file, text, batch);
  if(path == nullptr) {
    // The evaluator holds what it took from the network, and each load_model brings its own: the bytes go now.
    vector<char>().swap(E.pending);
#ifdef MODEL_FROM_MEMORY
    ModelDesc::registerInMemoryModel("engine", nullptr, 0);
#endif
  }
  E.search = new Search(E.params, E.nnEval, E.logger, "engine");
  return {{"id", q["id"]}, {"model", modelInfo()}};
}

// A move string of KataGo's protocol (Enclosure's: "A10-D10", columns A-S) -> the game's notation.
json ourMove(const json& m) {
  Loc loc;
  if(!m.is_string() || !Location::tryOfString(m.get<string>(), GameRules::MAX_BOARD_LEN, GameRules::MAX_BOARD_LEN, loc))
    return m;
  return GameRules::action_text(Board::actionOfMoveLoc(loc));
}

void dropGoFields(json& j) {
  for(const char* key : {"scoreMean", "noResultValue", "rawNoResultProb"})
    j.erase(key);
}

// The root's maps from the network's training-only heads (the engine repository's
// games/enclosure/docs/design/06-ownership-and-spatial-targets.md), for "includeWeakWalls" and "includeFutureNodes":
// one evaluation of the root, with the root's inputs and symmetry, not the search (design 04, "The root's maps"). Called
// after setPosition. NNOutput::auxMap holds them raw from the mover's side, point (x, y) at y * nnXLen + x of each
// channel: 0-1 the future nodes 8 and 32 edges later (tanh: +1 the mover's node, -1 the opponent's), 4-6 the edges
// that get cut (softmax: no cut, a cut of the mover's edge, a cut of the opponent's). Each map is a list of the 361
// points, y * 19 + x (row 1 first), as the ownership:
//   "weakWalls": {"b", "r"}: at each point of the player's edges (their ends and the points they pass through), the
//                chance that an edge of the player's there is cut before the game ends; null at the other points.
//   "futureNodes": {"8", "32"}: at each point, the expected +1 for a Blue node 8 (32) edges later and -1 for a Red
//                  one: Blue's side, as the ownership.
json rootMaps(const json& q, const Board& board, Player pla) {
  json maps = json::object();
  const bool walls = flag(q, "includeWeakWalls"), nodes = flag(q, "includeFutureNodes");
  if(!walls && !nodes)
    return maps;
  NNResultBuf buf;
  buf.includeAuxOutputs = true;
  E.search->computeRootNNEvaluation(buf, false);
  const NNOutput& out = *buf.result;
  testAssert(out.auxMap != NULL);
  const int S = enclosure::SIZE, area = out.nnXLen * out.nnYLen;
  const int mover = pla == P_BLACK ? 0 : 1;
  auto channel = [&](int c, int x, int y) { return (double)out.auxMap[c * area + NNPos::xyToPos(x, y, out.nnXLen)]; };
  // Rounded as KataGo rounds the ownership (Search::getAverageTreeOwnership), and never -0.
  auto rounded = [](double v) { return Global::roundStatic(v, 1000000.0) + 0.0; };
  if(walls) {
    json map = json::object();
    for(int side = 0; side < 2; side++) {
      uint8_t on[enclosure::SIZE * enclosure::SIZE];
      enclosure::edge_points(board.pos.edges[side], on);
      const int cut = side == mover ? 5 : 6;
      json list = json::array();
      for(int y = 0; y < S; y++)
        for(int x = 0; x < S; x++) {
          if(!on[x * S + y]) {
            list.push_back(nullptr);
            continue;
          }
          const double l[3] = {channel(4, x, y), channel(5, x, y), channel(6, x, y)};
          const double top = std::max({l[0], l[1], l[2]});
          const double sum = std::exp(l[0] - top) + std::exp(l[1] - top) + std::exp(l[2] - top);
          list.push_back(rounded(std::exp(l[cut - 4] - top) / sum));
        }
      map[string(1, nt::PLAYERS[side])] = std::move(list);
    }
    maps["weakWalls"] = std::move(map);
  }
  if(nodes) {
    json map = json::object();
    const double sign = mover == 0 ? 1.0 : -1.0;
    const char* horizons[2] = {"8", "32"};
    for(int c = 0; c < 2; c++) {
      json list = json::array();
      for(int y = 0; y < S; y++)
        for(int x = 0; x < S; x++)
          list.push_back(rounded(sign * std::tanh(channel(c, x, y))));
      map[horizons[c]] = std::move(list);
    }
    maps["futureNodes"] = std::move(map);
  }
  return maps;
}

// An analysis query's reply from the search's tree as it is now (during a search in slices, or after it), with the
// root's maps (rootMaps) that its query asked for.
json analysisReply(
  const json& q, const Board& board, const BoardHistory& hist, Player pla, bool during, const json& maps
) {
  int pvLen = 15;
  if(const json* l = field(q, "analysisPVLen"))
    if(l->is_number_integer())
      pvLen = std::max(0, l->get<int>());
  json out;
  if(!E.search->getAnalysisJson(P_BLACK, pvLen, false, false, flag(q, "includeOwnership"), false,
                                flag(q, "includeMovesOwnership"), false, flag(q, "includePVVisits"), false, out))
    return errorReply(q["id"], {"", "the search gave no analysis"});
  json moveInfos = json::array();
  for(json& mi : out["moveInfos"]) {
    // In a position that is its own image, the search keeps one edge of each set of equivalent edges (root symmetry
    // pruning), and KataGo lists each other edge of the set with that edge's values: "isSymmetryOf" names it (design
    // 04, "Symmetric twins").
    if(mi.contains("isSymmetryOf"))
      mi["isSymmetryOf"] = ourMove(mi["isSymmetryOf"]);
    mi["move"] = ourMove(mi["move"]);
    if(mi.contains("pv"))
      for(json& m : mi["pv"])
        m = ourMove(m);
    dropGoFields(mi);
    moveInfos.push_back(std::move(mi));
  }
  json rootInfo = std::move(out["rootInfo"]);
  dropGoFields(rootInfo);
  rootInfo["currentPlayer"] = string(1, nt::PLAYERS[pla == P_BLACK ? 0 : 1]);
  json reply = {
    {"id", q["id"]}, {"isDuringSearch", during}, {"turnNumber", hist.moveHistory.size()}, {"engine", engineTag()},
    {"rootInfo", std::move(rootInfo)}, {"moveInfos", std::move(moveInfos)},
  };
  if(flag(q, "chooseMove") && !during) {
    // The move KataGo's GTP engine would play (Search::getChosenMoveLoc): a draw over the play selection values with
    // the chosen-move temperature. Its seed comes from the query's "seed" (default 0) and the position, never from
    // earlier requests (design 04, "Reproducible analysis"), so the same request draws the same move. A bot that
    // must not repeat itself sends a new seed each time.
    const json* s = field(q, "seed");
    const string seed = s == nullptr ? "0" : s->is_string() ? s->get<string>() : std::to_string(s->get<uint64_t>());
    Rand rand("chosenMove:" + seed + ":" + board.pos_hash.toString());
    const Loc loc = E.search->getChosenMoveLoc(rand);
    reply["chosenMove"] = loc == Board::NULL_LOC ? json(nullptr) : ourMove(Location::toString(loc, board));
  }
  if(flag(q, "includePolicy")) {
    // The root's network policy over every legal edge, by edge (the fork's policy index is the action index).
    static PER_ENGINE float probs[NNPos::MAX_NN_POLICY_SIZE];
    if(E.search->getPolicy(probs)) {
      vector<int> legal;
      GameRules::legal_actions(board.pos, legal);
      vector<std::pair<double, string>> rows;
      for(int a : legal)
        rows.push_back({probs[a], GameRules::action_text(a)});
      std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
      json policy = json::array();
      for(const auto& [prob, name] : rows)
        policy.push_back({name, Global::roundDynamic(prob, 8)});
      reply["policy"] = std::move(policy);
    }
  }
  if(out.contains("ownership"))
    reply["ownership"] = std::move(out["ownership"]);  // per point, index y * 19 + x (row 1 first), Blue's side
  for(auto it = maps.begin(); it != maps.end(); ++it)
    reply[it.key()] = it.value();
  return reply;
}

// A search in slices (design 04, "Search in slices"): the query that started it, its position, and where it is. The
// playouts are run as runWholeSearch runs them (search.cpp), only paused between slices, so the slices of a search
// add up to one whole search of the same visits.
struct Running {
  json query;
  Board board;
  BoardHistory hist;
  Player pla = P_BLACK;
  SearchThread* thread = nullptr;
  int64_t maxVisits = 0, nonPlayoutVisits = 0, playouts = 0;
  int64_t sliceVisits = 0;  // 0: by time
  double sliceSeconds = 0;
  json maps;  // the root's maps (rootMaps), for every reply of the search
};
PER_ENGINE Running* RUN = nullptr;

void endRunning() {
  if(RUN == nullptr)
    return;
  E.search->endManualSearch(*RUN->thread);
  delete RUN->thread;
  delete RUN;
  RUN = nullptr;
}

// One slice of the running search; its reply. When the search reaches its visits, it ends.
json runSlice() {
  Running& r = *RUN;
  ClockTimer timer;
  int64_t done = 0;
  while(r.playouts + r.nonPlayoutVisits < r.maxVisits) {
    if(r.sliceVisits > 0 ? done >= r.sliceVisits : (done >= 1 && timer.getSeconds() >= r.sliceSeconds))
      break;
    const double left = double(r.maxVisits - r.playouts - r.nonPlayoutVisits);
    if(E.search->runSinglePlayout(*r.thread, left)) {
      r.playouts++;
      done++;
    }
  }
  const bool during = r.playouts + r.nonPlayoutVisits < r.maxVisits;
  const json reply = analysisReply(r.query, r.board, r.hist, r.pla, during, r.maps);
  if(!during)
    endRunning();
  return reply;
}

json analyze(const json& q) {
  if(E.search == nullptr)
    return errorReply(q["id"], {"", "no network loaded (load_model)"});
  Board board;
  BoardHistory hist;
  Player pla;
  Fail f;
  if(!buildPosition(q, board, hist, pla, f))
    return errorReply(q["id"], f);
  if(board.isGameOver())
    return errorReply(q["id"], {"position", "the game is over"});
  // A position that no game reaches gives numbers that mean nothing: refused, with its reasons (design 04).
  const json check = positionCheck(hist.initialBoard.pos);
  if(!check["possible"].get<bool>()) {
    json reply = errorReply(q["id"], {"position", check["reasons"][0]["text"].get<string>()});
    reply["reasons"] = check["reasons"];
    return reply;
  }
  int visits = DEFAULT_VISITS;
  if(const json* v = field(q, "maxVisits")) {
    if(!v->is_number_integer() || v->get<int>() < 1)
      return errorReply(q["id"], {"maxVisits", "maxVisits is a positive integer"});
    visits = v->get<int>();
  }
  SearchParams p = E.params;
  double maxTime = 1e20;
  if(const json* o = field(q, "overrideSettings")) {
    if(!o->is_object())
      return errorReply(q["id"], {"overrideSettings", "overrideSettings is an object"});
    for(auto it = o->begin(); it != o->end(); ++it) {
      if(it.key() == "maxTime") {
        if(!it.value().is_number() || it.value().get<double>() <= 0)
          return errorReply(q["id"], {"overrideSettings", "maxTime is in seconds, positive"});
        maxTime = it.value().get<double>();
        continue;
      }
      // The chosen move's settings, with the ranges KataGo's config reader allows (Setup::loadParams).
      const struct { const char* key; double* value; double lo, hi; } choice[] = {
        {"chosenMoveTemperature", &p.chosenMoveTemperature, 0.0, 5.0},
        {"chosenMoveTemperatureEarly", &p.chosenMoveTemperatureEarly, 0.0, 5.0},
        {"chosenMoveTemperatureHalflife", &p.chosenMoveTemperatureHalflife, 0.1, 100000.0},
        {"chosenMoveTemperatureOnlyBelowProb", &p.chosenMoveTemperatureOnlyBelowProb, 0.0, 1.0},
        {"chosenMoveSubtract", &p.chosenMoveSubtract, 0.0, 1.0e10},
        {"chosenMovePrune", &p.chosenMovePrune, 0.0, 1.0e10},
      };
      bool known = false;
      for(const auto& c : choice) {
        if(it.key() != c.key)
          continue;
        known = true;
        if(!it.value().is_number() || it.value().get<double>() < c.lo || it.value().get<double>() > c.hi)
          return errorReply(q["id"], {"overrideSettings", it.key() + " is a number from " + Global::doubleToString(c.lo) +
                                                            " to " + Global::doubleToString(c.hi)});
        *c.value = it.value().get<double>();
      }
      if(!known)
        return errorReply(q["id"], {"overrideSettings", "the settings are maxTime and the chosenMove* settings, not " +
                                                          it.key()});
    }
  }
  const json* seed = field(q, "seed");
  if(seed != nullptr && !seed->is_string() && !seed->is_number_unsigned())
    return errorReply(q["id"], {"seed", "seed is a string or a non-negative integer"});
  int64_t sliceVisits = 0;
  double sliceSeconds = 0;
  const json* slice = field(q, "slice");
  if(slice != nullptr) {
    const json* sv = slice->is_object() ? field(*slice, "visits") : nullptr;
    const json* sm = slice->is_object() ? field(*slice, "ms") : nullptr;
    if(sv != nullptr && sv->is_number_integer() && sv->get<int64_t>() > 0)
      sliceVisits = sv->get<int64_t>();
    else if(sm != nullptr && sm->is_number() && sm->get<double>() > 0)
      sliceSeconds = sm->get<double>() / 1000.0;
    else
      return errorReply(q["id"], {"slice", "slice is {\"visits\": N} or {\"ms\": T}, positive"});
    if(maxTime < 1e20)
      return errorReply(q["id"], {"overrideSettings", "a search in slices has no maxTime: stop it by terminate"});
  }
  for(const char* key : {"includeWeakWalls", "includeFutureNodes"})
    if(flag(q, key) && !hasMaps())
      return errorReply(q["id"], {key, string(key) + " needs the network's training-only heads, which this network or "
                                                     "this module does not give (the GPU module gives none)"});
  endRunning();  // one search at a time: a new query ends the one before
  p.maxVisits = visits;
  p.maxPlayouts = (int64_t)1 << 50;
  p.maxTime = maxTime;
  E.search->setParams(p);
  // As KataGo's analysis engine: the search keeps ownership only when asked (analysis.cpp).
  E.search->setAlwaysIncludeOwnerMap(flag(q, "includeOwnership") || flag(q, "includeMovesOwnership"));
  E.search->setPosition(pla, board, hist);
  // Before the search, which it leaves as it is: the evaluation with the maps stays out of the evaluator's cache.
  json maps = rootMaps(q, board, pla);
  // Root noise (wideRootNoise, KataGo's analysis default) draws from the search thread's random numbers. KataGo seeds
  // them with the engine's seed, the position and the count of searches begun (search.cpp, makeSeed). Seeded with the
  // query's "seed" in place of the engine's, and the count reset, the same request draws the same noise whatever came
  // before (design 04, "Reproducible analysis"). A search in slices keeps one thread, so its draws are those of one
  // whole search.
  E.search->randSeed = "analysis:" + (seed == nullptr ? string("0") : seed->is_string() ? seed->get<string>()
                                                                    : std::to_string(seed->get<uint64_t>()));
  E.search->numSearchesBegun = 0;
  if(slice == nullptr) {
    E.search->runWholeSearch(pla);
    return analysisReply(q, board, hist, pla, false, maps);
  }
  RUN = new Running{q, board, hist, pla, nullptr, visits, 0, 0, sliceVisits, sliceSeconds, std::move(maps)};
  E.search->beginSearch(false);
  RUN->nonPlayoutVisits = E.search->getRootVisits();
  RUN->thread = new SearchThread(0, *E.search);
  return runSlice();
}

// "continue" and "terminate" name the running search by its query's id ("terminate" also takes KataGo's terminateId).
bool isRunning(const json& id) { return RUN != nullptr && RUN->query["id"] == id; }

json continueQuery(const json& q) {
  if(!isRunning(q["id"]))
    return errorReply(q["id"], {"id", "no search in slices runs with this id"});
  return runSlice();
}

json terminateQuery(const json& q) {
  const json id = q.contains("terminateId") ? q["terminateId"] : q["id"];
  if(!isRunning(id))
    return errorReply(q["id"], {"terminateId", "no search in slices runs with this id"});
  json reply = analysisReply(RUN->query, RUN->board, RUN->hist, RUN->pla, false, RUN->maps);
  reply["terminated"] = true;
  endRunning();
  return reply;
}

json handleJson(const string& text) {
  json q = json::parse(text, nullptr, false);
  if(q.is_discarded() || !q.is_object())
    return {{"error", "the request is not a JSON object"}};
  if(!q.contains("id"))
    return {{"error", "the request has no id"}, {"field", "id"}};
  const json* action = field(q, "action");
  if(action == nullptr)
    return analyze(q);
  const string a = action->is_string() ? action->get<string>() : "";
  if(a == "query_version") return queryVersion(q);
  if(a == "load_model") { endRunning(); return loadModel(q); }
  if(a == "describe") return describe(q);
  if(a == "describe_game") return describeGame(q);
  if(a == "play") return play(q);
  if(a == "continue") return continueQuery(q);
  if(a == "terminate") return terminateQuery(q);
  return errorReply(q["id"], {"action", "unknown action: " + action->dump()});
}

}  // namespace

void EnclosureProtocol::setup(const string& configText, const string& build) {
  P.baseConfig = configText;
  P.build = build;
}

void EnclosureProtocol::setParallel(int engines) {
  P.engines = std::max(1, engines);
}

char* EnclosureProtocol::modelBuffer(size_t len) {
  E.pendingReader = PendingReader();
  E.pending.assign(std::max<size_t>(len, 1), 0);
  return E.pending.data();
}

void EnclosureProtocol::modelReader(
  size_t len, const string& sha256, const std::function<size_t(size_t, char*, size_t)>& read
) {
  vector<char>().swap(E.pending);
  E.pendingReader.len = len;
  E.pendingReader.sha256 = sha256;
  E.pendingReader.read = read;
}

string EnclosureProtocol::handle(const string& request) {
  return handleJson(request).dump();
}
