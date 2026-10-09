// Enclosure fork: the engine protocol (docs/design/04-notation-and-protocol.md). See enclosureprotocol.h.
//
// Requests (each has an "id", echoed in its reply; an error reply is {"id", "error", "field"}):
//   {"action": "query_version"}                       the protocol, the build, the model, the rules, trained outputs
//   {"action": "load_model", "path"?, "name", "settings"}
//                                                     a network: the file at "path" (the native engine), else the
//                                                     model buffer; settings override the base settings
//   {"action": "describe", "position", "moves"}      the position for a board: edges, nodes, regions, legal moves
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
// the game.
// Results are reproducible: each evaluation's symmetry comes from its cache key (nnSymmetryFromHash), so the same
// request gives the same reply whatever came before, and a search in slices equals one whole search (with one search
// thread; more threads search faster and are not reproducible).
#include "../command/enclosureprotocol.h"

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/sha2.h"
#include "../core/timer.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/enclosure/notation.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../search/search.h"
#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <vector>

using json = nlohmann::json;
using std::string;
using std::vector;
namespace nt = enclosure::notation;

namespace {

const char* PROTOCOL = "1.1";
const char* RULES = "online";  // the online variant (enclosure/README.md "Rules"), 19 x 19
const int DEFAULT_VISITS = 500;

struct Engine {
  string baseConfig, build = "unknown";
  vector<char> pending;  // a network's bytes, written by the caller (modelBuffer)
  vector<char> model;    // a network loaded from memory: its bytes, which the evaluator reads ("memory:")
  string modelName, modelSha;
  size_t modelBytes = 0;
  Logger* logger = nullptr;
  ConfigParser* cfg = nullptr;
  NNEvaluator* nnEval = nullptr;
  Search* search = nullptr;
  SearchParams params;
};
Engine E;

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
// player to move.
bool buildPosition(const json& q, Board& board, BoardHistory& hist, Player& pla, Fail& f) {
  if(!checkVariant(q, f))
    return false;
  enclosure::Position start;
  if(const json* p = field(q, "position")) {
    string why;
    if(!p->is_string() || !nt::parse_position(p->get<string>(), start, why)) {
      f = {"position", p->is_string() ? why : "a position is a string"};
      return false;
    }
  }
  if(!readHistory(q, start, f))
    return false;
  board = Board(enclosure::SIZE, enclosure::SIZE);
  board.setPosition(start);
  pla = start.player == 0 ? P_BLACK : P_WHITE;  // Blue is KataGo's Black
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
  hist.setInitialTurnNumber(start.move_count);
  if(const json* moves = field(q, "moves")) {
    if(!moves->is_array()) {
      f = {"moves", "moves is a list of edges"};
      return false;
    }
    for(size_t i = 0; i < moves->size(); i++) {
      const json& m = (*moves)[i];
      string why = "moves are strings";
      const int action = m.is_string() ? nt::parse_move(m.get<string>(), board.pos, &why) : -1;
      if(action < 0 || !hist.makeBoardMoveTolerant(board, Board::moveLocOfAction(action), pla, false)) {
        f = {"moves", "move " + std::to_string(i) + ": " + why};
        return false;
      }
      pla = hist.presumedNextMovePla;
    }
  }
  return true;
}

vector<string> sortedEdgeNames(const vector<enclosure::Edge>& edges) {
  vector<string> out;
  vector<enclosure::Edge> es = edges;
  std::sort(es.begin(), es.end(), [](const enclosure::Edge& a, const enclosure::Edge& b) {
    return std::tie(a.ax, a.ay, a.bx, a.by) < std::tie(b.ax, b.ay, b.bx, b.by);
  });
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

vector<string> legalNames(const enclosure::Position& p) {
  vector<int> legal;
  p.legal_actions(legal);
  vector<string> out;
  for(int a : legal)
    out.push_back(nt::move_name(a));
  std::sort(out.begin(), out.end());
  return out;
}

json outcome(const enclosure::Position& p) {
  if(!p.terminal())
    return nullptr;
  const double d = enclosure::snap(p.scores[0] - p.scores[1]);
  const string result = d > 0 ? "b+" + nt::score_text(d) : d < 0 ? "r+" + nt::score_text(-d) : "0";
  return {{"result", result}};
}

json describe(const json& q) {
  Board board;
  BoardHistory hist;
  Player pla;
  Fail f;
  if(!buildPosition(q, board, hist, pla, f))
    return errorReply(q["id"], f);
  const enclosure::Position& p = board.pos;
  json regions = json::object(), areas = json::array(), edges = json::object(), nodes = json::object();
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
    regions[name] = list;
    areas.push_back(area);
    edges[name] = sortedEdgeNames(p.edges[side]);
    nodes[name] = nodeNames(p.edges[side]);
  }
  return {
    {"id", q["id"]},
    {"position", nt::position_string(p)},
    {"mover", string(1, nt::PLAYERS[p.player])},
    {"movesLeft", p.moves_left},
    {"moveCount", p.move_count},
    {"scores", {p.scores[0], p.scores[1]}},
    {"edges", edges},
    {"protected", sortedEdgeNames(p.protected_)},
    {"placed", sortedEdgeNames(p.placed)},
    {"nodes", nodes},
    {"regions", regions},
    {"areas", areas},
    {"legal", legalNames(p)},
    {"outcome", outcome(p)},
  };
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
  const int action = nt::parse_move(m->get<string>(), before, &why);
  if(action < 0)
    return {{"id", q["id"]}, {"legal", false}, {"why", why}};
  enclosure::Position after = before;
  after.play(action);  // PASS forfeits the rest of the turn (a timeout)
  const int enemy = 1 - before.player;
  json cuts = nullptr;
  for(const auto& e : before.edges[enemy])
    if(!enclosure::contains(after.edges[enemy], e))
      cuts = nt::edge_name(e);
  const vector<string> nodesBefore = nodeNames(before.edges[enemy]), nodesAfter = nodeNames(after.edges[enemy]);
  json removed = json::array();
  for(const string& n : nodesBefore)
    if(std::find(nodesAfter.begin(), nodesAfter.end(), n) == nodesAfter.end())
      removed.push_back(n);
  return {
    {"id", q["id"]}, {"legal", true}, {"move", nt::move_name(action)}, {"cuts", cuts},
    {"removedNodes", removed}, {"position", nt::position_string(after)}, {"outcome", outcome(after)},
  };
}

json engineTag() {
  return {{"protocol", PROTOCOL}, {"build", E.build}, {"model", E.modelSha.empty() ? json(nullptr) : json(E.modelSha)}};
}

json modelInfo() {
  return {{"name", E.modelName}, {"sha256", E.modelSha}, {"bytes", E.modelBytes}};
}

json queryVersion(const json& q) {
  return {
    {"id", q["id"]}, {"protocol", PROTOCOL}, {"build", E.build},
    {"model", E.nnEval != nullptr ? modelInfo() : json(nullptr)}, {"rules", RULES},
    {"trained", {"winrate", "scoreSelfplay", "scoreStdev", "scoreLead", "policy", "ownership"}},
  };
}

string sha256(const vector<char>& bytes) {
  char hash[65];
  SHA2::get256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), hash);  // of every byte
  return hash;
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
    if(E.pending.empty())
      return errorReply(q["id"], {"", "no network: give a path, or fill the model buffer first"});
#else
    return errorReply(q["id"], {"path", "this build loads a network from a file: give its path"});
#endif
  }
  // By default each evaluation's symmetry from its cache key, so that the same request gives the same result (design
  // 04, "Reproducible analysis"); then the base settings and the request's settings, which can change it (a later line
  // wins).
  string text = "nnSymmetryFromHash = true\n" + E.baseConfig + "\n";
  if(const json* s = field(q, "settings")) {
    if(!s->is_object())
      return errorReply(q["id"], {"settings", "settings is an object of analysis settings"});
    for(auto it = s->begin(); it != s->end(); ++it) {
      const string value = it.value().is_string() ? it.value().get<string>() : it.value().dump();
      text += it.key() + " = " + value + "\n";
    }
  }
  delete E.search;
  delete E.nnEval;
  delete E.cfg;
  E.search = nullptr;
  E.nnEval = nullptr;
  string file;
  if(path != nullptr) {
    E.model.clear();
    E.modelSha = sha256(fileBytes);
    E.modelBytes = fileBytes.size();
    file = path->get<string>();
  }
  else {
    E.model.swap(E.pending);
    E.pending.clear();
    E.modelSha = sha256(E.model);
    E.modelBytes = E.model.size();
#ifdef MODEL_FROM_MEMORY
    ModelDesc::registerInMemoryModel("engine", E.model.data(), E.model.size());
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
  // As many evaluations at once as search threads; a batch size named in the settings is taken as it is.
  const Setup::MaxBatchSizeRequest batch = E.cfg->contains("nnMaxBatchSize")
    ? Setup::MaxBatchSizeRequest::explicitSize(E.cfg->getInt("nnMaxBatchSize", 1, 65536))
    : Setup::MaxBatchSizeRequest::fromConcurrency();
  Rand seedRand("engine");
  E.nnEval = Setup::initializeNNEvaluator(
    "engine", file, "", *E.cfg, *E.logger, seedRand, std::max(1, E.params.numThreads), enclosure::SIZE,
    enclosure::SIZE, batch, true, false, Setup::SETUP_FOR_ANALYSIS
  );
  E.search = new Search(E.params, E.nnEval, E.logger, "engine");
  return {{"id", q["id"]}, {"model", modelInfo()}};
}

// A KataGo move string of the fork ("A10-D10", columns A-S) -> our notation.
json ourMove(const json& m) {
  Loc loc;
  if(!m.is_string() || !Location::tryOfString(m.get<string>(), enclosure::SIZE, enclosure::SIZE, loc))
    return m;
  return nt::move_name(Board::actionOfMoveLoc(loc));
}

void dropGoFields(json& j) {
  for(const char* key : {"scoreMean", "noResultValue", "rawNoResultProb", "isSymmetryOf"})
    j.erase(key);
}

// An analysis query's reply from the search's tree as it is now (during a search in slices, or after it).
json analysisReply(const json& q, const Board& board, const BoardHistory& hist, Player pla, bool during) {
  auto flag = [&](const char* key) { const json* b = field(q, key); return b != nullptr && b->is_boolean() && b->get<bool>(); };
  int pvLen = 15;
  if(const json* l = field(q, "analysisPVLen"))
    if(l->is_number_integer())
      pvLen = std::max(0, l->get<int>());
  json out;
  if(!E.search->getAnalysisJson(P_BLACK, pvLen, false, false, flag("includeOwnership"), false,
                                flag("includeMovesOwnership"), false, flag("includePVVisits"), false, out))
    return errorReply(q["id"], {"", "the search gave no analysis"});
  json moveInfos = json::array();
  for(json mi : out["moveInfos"]) {
    if(mi.contains("isSymmetryOf"))
      continue;  // KataGo lists symmetric twins of a move too; each legal edge once here
    mi["move"] = ourMove(mi["move"]);
    if(mi.contains("pv"))
      for(json& m : mi["pv"])
        m = ourMove(m);
    dropGoFields(mi);
    moveInfos.push_back(mi);
  }
  json rootInfo = out["rootInfo"];
  dropGoFields(rootInfo);
  rootInfo["currentPlayer"] = string(1, nt::PLAYERS[pla == P_BLACK ? 0 : 1]);
  json reply = {
    {"id", q["id"]}, {"isDuringSearch", during}, {"turnNumber", hist.moveHistory.size()}, {"engine", engineTag()},
    {"rootInfo", rootInfo}, {"moveInfos", moveInfos},
  };
  if(flag("chooseMove") && !during) {
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
  if(flag("includePolicy")) {
    // The root's network policy over every legal edge, by edge (the fork's policy index is the action index).
    static float probs[NNPos::MAX_NN_POLICY_SIZE];
    if(E.search->getPolicy(probs)) {
      vector<int> legal;
      board.pos.legal_actions(legal);
      vector<std::pair<double, string>> rows;
      for(int a : legal)
        rows.push_back({probs[a], nt::move_name(a)});
      std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
      json policy = json::array();
      for(const auto& [prob, name] : rows)
        policy.push_back({name, Global::roundDynamic(prob, 8)});
      reply["policy"] = policy;
    }
  }
  if(out.contains("ownership"))
    reply["ownership"] = out["ownership"];  // per point, index y * 19 + x (row 1 first), Blue's side
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
};
Running* RUN = nullptr;

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
  const json reply = analysisReply(r.query, r.board, r.hist, r.pla, during);
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
  if(board.pos.terminal())
    return errorReply(q["id"], {"position", "the game is over"});
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
  auto flag = [&](const char* key) { const json* b = field(q, key); return b != nullptr && b->is_boolean() && b->get<bool>(); };
  endRunning();  // one search at a time: a new query ends the one before
  p.maxVisits = visits;
  p.maxPlayouts = (int64_t)1 << 50;
  p.maxTime = maxTime;
  E.search->setParams(p);
  // As KataGo's analysis engine: the search keeps ownership only when asked (analysis.cpp).
  E.search->setAlwaysIncludeOwnerMap(flag("includeOwnership") || flag("includeMovesOwnership"));
  E.search->setPosition(pla, board, hist);
  if(slice == nullptr) {
    E.search->runWholeSearch(pla);
    return analysisReply(q, board, hist, pla, false);
  }
  RUN = new Running{q, board, hist, pla, nullptr, visits, 0, 0, sliceVisits, sliceSeconds};
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
  json reply = analysisReply(RUN->query, RUN->board, RUN->hist, RUN->pla, false);
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
  if(a == "play") return play(q);
  if(a == "continue") return continueQuery(q);
  if(a == "terminate") return terminateQuery(q);
  return errorReply(q["id"], {"action", "unknown action: " + action->dump()});
}

}  // namespace

void EnclosureProtocol::setup(const string& configText, const string& build) {
  E.baseConfig = configText;
  E.build = build;
}

char* EnclosureProtocol::modelBuffer(size_t len) {
  E.pending.assign(std::max<size_t>(len, 1), 0);
  return E.pending.data();
}

string EnclosureProtocol::handle(const string& request) {
  return handleJson(request).dump();
}
