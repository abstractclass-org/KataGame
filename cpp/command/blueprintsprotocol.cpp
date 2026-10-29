// Blueprints' engine protocol: Enclosure's (command/enclosureprotocol.cpp, the engine repository's
// docs/design/04-notation-and-protocol.md), version 0.7.1 (Enclosure's engine stays 0.6.x), with Blueprints' fields and
// strategyjam.org's moves. See blueprintsprotocol.h. The engine repository's games/blueprints/README.md, "The browser
// build", gives the fields.
//
// Requests (each has an "id", echoed in its reply; an error reply is {"id", "error", "field"}):
//   {"action": "query_version"}                       the protocol, the build, the search (the first build that
//                                                     searched as this one), the model, the rules, trained outputs
//   {"action": "load_model", "path"?, "name", "settings"}
//                                                     a network: the file at "path" (the native engine), else the
//                                                     model buffer; settings override the base settings
//   {"action": "describe", "position", "moves"}      the position for a board (its cells, hands, scores, mover, moves
//                                                     left, move count, outcome, view, the views of its cover) and
//                                                     every legal move
//   {"action": "describe_game", "position", "moves", "from"?}
//                                                     every position of the game from "from" on, with the moves played
//                                                     once: the board's fields, and what the move to each position did
//                                                     (its footprint, the cells that it captured, its points)
//   {"action": "play", "position", "moves", "move"}  whether the move is legal, what it did, the new position
//   {"position", "moves", "maxVisits", "rootViews"?, ...}
//                                                     an analysis (no action), as KataGo's analysis engine; with
//                                                     "slice": {"visits": N} or {"ms": T}, one slice of it at a time
//   {"action": "continue"}                           the next slice of the search with this id
//   {"action": "terminate", "terminateId"}           ends that search; its last results, "terminated": true
// A position is "position" (the one-line position, default the start) followed by "moves", in strategyjam.org's
// strings: a placement "K@q,r" (blueprint K's anchor cell on the board's hex q,r), an edit "I:q,r>J:q,r" and "pass".
// Every move that the engine gives is such a string: the legal moves, the candidates and the moves of their lines.
//
// The site's moves meet the engine's actions here only (games/blueprints/README.md, "The port"):
// - A placement whose cells all lie in the mover's view is one action of the view. Any other placement that the site's
//   rule allows plays through place_at_site on the rules board, and the replies name the moves sent that played so
//   ("movesOutsideView", describe_game's "outsideView"). The search of the root's views gives such a placement as a
//   candidate: an action of another view, named by the site's string as every move is.
// - An edit is two actions of the mover, the pick "I:q,r" and the drop "J:q,r". A candidate edit has the values of its
//   pick's node, and its move joins the pick with its best drop: the next move of its line, or the drop of the highest
//   prior at the pick's node when the line stops at the pick. A line joins each pick with its drop.
// - The rules board holds 511 x 511 hexes: a placement that would make the cells span more is refused, with an error
//   that names the limit.
// The ownership is the network's at the root, over the view's hexes, with the view's place ("view"): KataGo's average
// over the search's tree would add the maps of nodes whose views lie elsewhere (each mover's view follows its play).
// Results are reproducible, as Enclosure's: each evaluation's symmetry comes from its cache key, and the search's
// random numbers from the query's "seed" and the position.
//
// The root's views (program/rootviews.h; games/blueprints/README.md, "The browser build"): an analysis's "rootViews"
// (by default the settings' rootViews, true in analysis_blueprints.cfg) searches the cover of the root's views. When
// the rule's view holds every legal placement, the search and the reply are those without it. One other view that
// holds them: the search of that view. More views: their passes and checks (RootViews::Searches), run one after
// another in the slices, each with its share of the visits. The reply then has the candidates of every view by the
// site's strings, and the root's values are the best candidate's, with no raw values. rootInfo gives the cover
// ("views") and the searches. The policy and the ownership of such a search are the network's at the root in the
// rule's view.
#include "../command/blueprintsprotocol.h"
#include "../command/blueprintssite.h"

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/sha2.h"
#include "../core/timer.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/blueprints/gamerules.h"
#include "../neuralnet/nneval.h"
#include "../program/rootviews.h"
#include "../program/setup.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

using json = nlohmann::json;
using std::string;
using std::vector;
namespace bp = blueprints;
using Pos = bp::Position;

static_assert(std::is_same<GameRules, bp::GameRules>::value, "the protocol of Blueprints' build");

namespace {

// The interface's version (design 04, "Versioning"). Blueprints' build took 0.7.0 for the search of the root's views
// (rootViews on by default, no raw values in a reply of more views), which breaks a 0.6 client. 0.7.1 accepts the query
// whose passes find no candidate, which 0.7.0 refused: its move is the first pass's policy move. Enclosure's engine
// stays 0.6.x (0.6.2 adds its own maps).
const char* PROTOCOL = "0.7.1";
// The first build that searched as this one does (design 04, "The same search"), given as each reply's "search". A
// change that can alter an analysis reply sets it to "", and the build then names itself. After that build's release,
// it takes the released build's name.
const char* SEARCH_SINCE = "";
const int DEFAULT_VISITS = 500;
const char* KINDS[3] = {"plain", "star", "wall"};  // the site's names of a cell's kinds, bp::PLAIN to bp::WALL

struct Process {
  string baseConfig, build = "unknown";
};
Process P;

struct Engine {
  vector<char> pending;  // a network's bytes, written by the caller (modelBuffer), until load_model reads them
  string modelName, modelSha;
  size_t modelBytes = 0;
  Logger* logger = nullptr;
  ConfigParser* cfg = nullptr;
  NNEvaluator* nnEval = nullptr;
  Search* search = nullptr;
  SearchParams params;
  // The settings' rootViews (false when absent), each analysis's default, and the passes' and checks' settings.
  bool rootViews = false;
  RootViews::Settings rootViewSettings;
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

// A query's option such as "includeOwnership": true only when given as true.
bool flag(const json& q, const char* key) {
  const json* b = field(q, key);
  return b != nullptr && b->is_boolean() && b->get<bool>();
}

// A position for the work of one request beside the game's own: a copy into it reads only the copy's box of cells
// (bp::Cells), where a new Position clears all 261,121 points of its rules board.
Pos& scratch(int i) {
  static Pos* positions[3] = {new Pos(), new Pos(), new Pos()};
  return *positions[i];
}

// A game of a request: the front's board and history, the site's moves played, and those that played outside the
// view. On the heap: a BoardHistory holds 7 boards of Blueprints' rules board.
struct Game {
  Board board;
  BoardHistory hist;
  int moves = 0;
  vector<int> outside;
  Player pla() const { return board.nextPla(); }  // the player to move, as the history takes it
};

// The site's rules, its moves and the engine's actions (command/blueprintssite.h, which the practice bot shares).
using namespace BlueprintsSite;

string playerName(int player) { return player == 0 ? "b" : "r"; }

// Calls f(k, x, y) once for each placement of the mover's blueprint k, its anchor cell on the rules point (x, y), on
// the board or off it, that the site's rule allows (siteFits), with span its cells' span with the placement
// (spanWith). A legal placement touches a cell of the mover, so one of its cells lies on an empty hex that touches
// one: the anchors come from those hexes.
template<class F>
void forEachSitePlacement(const Pos& p, F&& f) {
  const int mover = bp::turn_of(p.ply);
  const bp::Box cells = bp::cells_box(p);
  if(!cells.any)
    return;
  // Each point once, in order.
  auto sorted = [](vector<std::pair<int, int>>& points) {
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
  };
  vector<std::pair<int, int>> frontier, anchors;
  for(int y = cells.y0; y <= cells.y1; y++)
    for(int x = cells.x0; x <= cells.x1; x++)
      if(bp::cell_owner(p.cell[y * bp::RULES_LEN + x]) == mover)
        for(const auto& d : bp::DIRS)
          if(!bp::at(p, x + d[0], y + d[1]))
            frontier.push_back({x + d[0], y + d[1]});
  sorted(frontier);
  for(int k = 0; k < bp::HAND; k++) {
    const bp::Blueprint& b = p.hand[mover][k];
    anchors.clear();
    for(const auto& [fx, fy] : frontier)
      for(int i = 0; i < b.n; i++)
        anchors.push_back({fx - b.q[i], fy - b.r[i]});
    sorted(anchors);
    for(const auto& [x, y] : anchors)
      if(siteFits(p, k, x, y))
        f(k, x, y, spanWith(cells, b, x, y));
  }
}

enum class Played { OK, ILLEGAL, LIMIT };

// Plays one action of the engine in g, through the front.
void playAction(Game& g, int action) {
  g.hist.makeBoardMoveTolerant(g.board, Board::moveLocOfAction(action), g.pla(), false);
}

// Plays the site's move `text` in g (readSiteMove): a placement as one action of the view, or through place_at_site
// outside it (then *outsideView); an edit as its pick and its drop; a pass. ILLEGAL or LIMIT (the rules board), with
// the reason in why, and g as it was.
Played playSiteMove(Game& g, const string& text, bool& outsideView, string& why) {
  outsideView = false;
  const SiteMove m = readSiteMove(g.board.pos, text, scratch(0));
  if(m.kind == SiteMove::ILLEGAL || m.kind == SiteMove::LIMIT) {
    why = m.why;
    return m.kind == SiteMove::ILLEGAL ? Played::ILLEGAL : Played::LIMIT;
  }
  if(m.kind == SiteMove::ACTIONS) {
    for(int action : m.actions)
      playAction(g, action);
    return Played::OK;
  }
  // Outside the view: on the rules board, and the front's history starts again from the position after it. The
  // position keeps its own record of the last actions (the network's history planes), and the turn number goes on.
  Pos& next = scratch(0);
  next = g.board.pos;
  bp::place_at_site(next, m.k, m.q, m.r);
  const int64_t turn = g.hist.getCurrentTurnNumber();
  const Rules rules = g.hist.rules;
  g.board.setPosition(next);
  g.hist.clear(g.board, g.pla(), rules, 0);
  g.hist.setInitialTurnNumber(turn + 1);
  outsideView = true;
  return Played::OK;
}

// The rules and size fields are optional; anything but the site's rules is refused, and the board has no size.
bool checkVariant(const json& q, Fail& f) {
  if(const json* r = field(q, "rules"))
    if(!r->is_string() || r->get<string>() != RULES) {
      f = {"rules", string("the only rules are \"") + RULES + "\""};
      return false;
    }
  for(const char* key : {"boardXSize", "boardYSize"})
    if(field(q, key) != nullptr) {
      f = {key, "Blueprints' board has no edge, so no size"};
      return false;
    }
  return true;
}

// The query's position: "position" (default the start), then "moves", into g. "each", if given, gets the position
// before the moves and the position after each move, with whether that move played outside the view.
bool buildPosition(const json& q, Game& g, Fail& f, const std::function<void(const Pos&, bool)>& each = nullptr) {
  if(!checkVariant(q, f))
    return false;
  Pos& start = scratch(1);
  if(const json* p = field(q, "position")) {
    string why;
    if(!p->is_string() || !GameRules::parse_position(p->get<string>(), start, why)) {
      f = {"position", p->is_string() ? why : "a position is a string"};
      return false;
    }
    if(start.picked >= 0) {
      f = {"position", "a pick waits for its drop: the site's positions are between moves, and an edit is one move"};
      return false;
    }
  }
  else
    GameRules::new_game(bp::VIEW, bp::VIEW, start);
  g.board.setPosition(start);
  Rules rules = Rules::getTrompTaylorish();
  if(const json* k = field(q, "komi")) {
    if(!k->is_number()) {
      f = {"komi", "komi is a number: points added to Red's final score"};
      return false;
    }
    rules.komi = k->get<float>();
  }
  g.hist.clear(g.board, g.pla(), rules, 0);
  // The game's turn number counts from the start, so the move temperature's early phase is the game's.
  g.hist.setInitialTurnNumber(GameRules::moves_played(start));
  if(each)
    each(g.board.pos, false);
  if(const json* moves = field(q, "moves")) {
    if(!moves->is_array()) {
      f = {"moves", "moves is a list of strategyjam.org's moves"};
      return false;
    }
    for(size_t i = 0; i < moves->size(); i++) {
      const json& m = (*moves)[i];
      string why = "moves are strings";
      bool outside = false;
      if(!m.is_string() || playSiteMove(g, m.get<string>(), outside, why) != Played::OK) {
        f = {"moves", "move " + std::to_string(i) + ": " + why};
        return false;
      }
      g.moves++;
      if(outside)
        g.outside.push_back(int(i));
      if(each)
        each(g.board.pos, outside);
    }
  }
  return true;
}

// Calls f(q, r, v) for each cell of p, in the order of the rules board's rows: its site hex q,r and its value.
template<class F>
void forEachCell(const Pos& p, F&& f) {
  const bp::Box& box = p.cell.box();
  if(box.any)
    for(int y = box.y0; y <= box.y1; y++)
      for(int x = box.x0; x <= box.x1; x++)
        if(const int8_t v = p.cell[y * bp::RULES_LEN + x])
          f(siteQ(p, x), siteR(p, y), v);
}

// A cell in strategyjam.org's form: [owner, kind], owner 0 for Blue, 1 for Red.
json cellJson(int8_t v) { return {bp::cell_owner(v), KINDS[bp::cell_type(v)]}; }

// The board's cells by their hexes, in strategyjam.org's form: {"q,r": [owner, kind]}.
json cellsJson(const Pos& p) {
  json cells = json::object();
  forEachCell(p, [&](int q, int r, int8_t v) { cells[bp::cell_text(q, r)] = cellJson(v); });
  return cells;
}

// Each player's three blueprints in their own frames (the anchor cell at 0,0), in strategyjam.org's form: for each
// player a list of three blueprints, each {"q,r": kind}.
json blueprintsJson(const Pos& p) {
  json players = json::array();
  for(int player = 0; player < 2; player++) {
    json hand = json::array();
    for(int k = 0; k < bp::HAND; k++) {
      const bp::Blueprint& b = p.hand[player][k];
      json cells = json::object();
      for(int i = 0; i < b.n; i++)
        cells[bp::cell_text(b.q[i], b.r[i])] = KINDS[b.t[i]];
      hand.push_back(std::move(cells));
    }
    players.push_back(std::move(hand));
  }
  return players;
}

json outcome(const Pos& p) {
  if(!GameRules::terminal(p))
    return nullptr;
  const int d = p.scores[0] - p.scores[1];
  return {{"result", d > 0 ? "b+" + std::to_string(d) : d < 0 ? "r+" + std::to_string(-d) : string("0")}};
}

// The mover's view: its middle's hex and its orientation. The view's hex (x, y), 0 <= x, y < 2 * radius + 1, is the
// board's hex of the middle plus R^orientation (x - radius, y - radius), R the turn by 60 degrees, (q, r) -> (-r,
// q + r). The ownership's index y * (2 * radius + 1) + x is that hex.
json viewJson(const Pos& p) {
  const bp::View v = bp::view_of(p);
  return {{"q", siteQ(p, v.wx)}, {"r", siteR(p, v.wy)}, {"orientation", v.o}, {"radius", bp::VIEW_RADIUS}};
}

// The views of p's cover (RootViews::cover), each in viewJson's form: "" is p's own view.
json viewsJson(const Pos& p, const vector<string>& views) {
  json out = json::array();
  for(const string& view : views) {
    Pos& v = scratch(2);
    v = p;
    string why;
    GameRules::set_view(v, view, why);
    out.push_back(viewJson(v));
  }
  return out;
}

// The fields of a position that a board shows: describe's, without the legal moves and the view.
json boardFields(const Pos& p) {
  return {
    {"position", GameRules::position_text(p)},
    {"mover", playerName(GameRules::next_player(p))},
    {"movesLeft", GameRules::turn_moves_left(p)},
    {"moveCount", GameRules::moves_played(p)},
    {"scores", {p.scores[0], p.scores[1]}},
    {"cells", cellsJson(p)},
    {"blueprints", blueprintsJson(p)},
    {"outcome", outcome(p)},
  };
}

// What the move from `before` to `after` did: the hexes that it filled (its footprint, the placement's own cells),
// the opponent's cells that it captured, as they were before it, and the points that it scored. An edit and a pass
// fill no hex.
void moveEffect(const Pos& before, const Pos& after, json& out) {
  vector<string> footprint;
  json captured = json::object();
  auto empty = [](const Pos& p, int q, int r) { return bp::at(p, rulesX(p, q), rulesY(p, r)) == 0; };
  forEachCell(before, [&](int q, int r, int8_t v) {
    if(empty(after, q, r))
      captured[bp::cell_text(q, r)] = cellJson(v);
  });
  forEachCell(after, [&](int q, int r, int8_t) {
    if(empty(before, q, r))
      footprint.push_back(bp::cell_text(q, r));
  });
  const int mover = bp::turn_of(before.ply);
  out["footprint"] = footprint;
  out["captured"] = std::move(captured);
  out["points"] = after.scores[mover] - before.scores[mover];
}

// Every legal move of p in the site's strings, sorted: each placement that the site's rule allows and the rules board
// holds, and each edit (each pick with each of its drops). outsideView gets the placements that no action of the view
// names, which the search never gives.
void legalMoves(const Pos& p, vector<string>& legal, vector<string>& outsideView) {
  legal.clear();
  outsideView.clear();
  if(GameRules::terminal(p))
    return;
  vector<int> actions, drops;
  GameRules::legal_actions(p, actions);
  for(int a : actions) {
    const string text = siteText(p, a);
    if(!isPick(p, a)) {
      legal.push_back(text);
      continue;
    }
    Pos& picked = scratch(2);
    picked = p;
    GameRules::play(picked, a);
    GameRules::legal_actions(picked, drops);
    for(int d : drops)
      legal.push_back(text + ">" + siteText(picked, d));
  }
  std::sort(legal.begin(), legal.end());
  legal.erase(std::unique(legal.begin(), legal.end()), legal.end());
  forEachSitePlacement(p, [&](int k, int x, int y, int span) {
    if(span > bp::RULES_LEN)
      return;
    string text = placementText(p, k, x, y);
    if(!std::binary_search(legal.begin(), legal.end(), text))
      outsideView.push_back(std::move(text));
  });
  // The placements outside the view come once each: one anchor a blueprint (forEachSitePlacement).
  std::sort(outsideView.begin(), outsideView.end());
  const size_t inView = legal.size();
  legal.insert(legal.end(), outsideView.begin(), outsideView.end());
  std::inplace_merge(legal.begin(), legal.begin() + inView, legal.end());
}

json describe(const json& q) {
  std::unique_ptr<Game> g(new Game());
  Fail f;
  if(!buildPosition(q, *g, f))
    return errorReply(q["id"], f);
  const Pos& p = g->board.pos;
  json reply = boardFields(p);
  vector<string> legal, outside;
  legalMoves(p, legal, outside);
  reply["id"] = q["id"];
  reply["view"] = viewJson(p);
  reply["views"] = viewsJson(p, GameRules::terminal(p) ? vector<string>{""} : RootViews::cover(g->board));
  reply["legal"] = legal;
  reply["legalOutsideView"] = outside;
  reply["movesOutsideView"] = g->outside;
  return reply;
}

// Every position of a game in one request: "position" and "moves" as describe's, played once. "from" (default 0) is
// the first position in the reply: 0 is the position before the moves, k the position after k moves. Each position
// has the board's fields (boardFields) and what the move to it did (moveEffect: "footprint", "captured", "points"),
// with "outsideView", whether that move played outside the view. Those four are null for the position before the
// moves.
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
  std::unique_ptr<Pos> before(new Pos());
  int k = 0;
  const auto each = [&](const Pos& p, bool outside) {
    if(k >= from) {
      json r = boardFields(p);
      if(k == 0) {
        for(const char* key : {"footprint", "captured", "points", "outsideView"})
          r[key] = nullptr;
      }
      else {
        moveEffect(*before, p, r);
        r["outsideView"] = outside;
      }
      positions.push_back(std::move(r));
    }
    *before = p;
    k++;
  };
  std::unique_ptr<Game> g(new Game());
  Fail f;
  if(!buildPosition(q, *g, f, each))
    return errorReply(q["id"], f);
  return {{"id", q["id"]}, {"positions", std::move(positions)}};
}

json play(const json& q) {
  std::unique_ptr<Game> g(new Game());
  Fail f;
  if(!buildPosition(q, *g, f))
    return errorReply(q["id"], f);
  const json* m = field(q, "move");
  if(m == nullptr || !m->is_string())
    return errorReply(q["id"], {"move", "play needs a move"});
  std::unique_ptr<Pos> before(new Pos());
  *before = g->board.pos;
  bool outside = false;
  string why;
  const Played played = playSiteMove(*g, m->get<string>(), outside, why);
  if(played == Played::LIMIT)
    return errorReply(q["id"], {"move", why});
  if(played == Played::ILLEGAL)
    return {{"id", q["id"]}, {"legal", false}, {"why", why}};
  json reply = {{"id", q["id"]}, {"legal", true}, {"move", m->get<string>()}};
  moveEffect(*before, g->board.pos, reply);
  reply["outsideView"] = outside;
  reply["position"] = GameRules::position_text(g->board.pos);
  reply["outcome"] = outcome(g->board.pos);
  return reply;
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

json queryVersion(const json& q) {
  // The outputs that Blueprints' networks train. The lead (scoreLead) trains since bpj2's restart with the lead's
  // targets, and a release lists it once its network passes the lead's check (games/blueprints/README.md, "The browser
  // build").
  const json trained = json::array({"winrate", "scoreSelfplay", "scoreStdev", "policy", "ownership"});
  return {
    {"id", q["id"]}, {"protocol", PROTOCOL}, {"build", P.build}, {"search", searchName()},
    {"model", E.nnEval != nullptr ? modelInfo() : json(nullptr)}, {"rules", RULES}, {"trained", trained},
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
  delete E.nnEval;
  E.nnEval = nullptr;
  delete E.cfg;
  string file;
  if(path != nullptr) {
    E.modelSha = sha256(fileBytes);
    E.modelBytes = fileBytes.size();
    file = path->get<string>();
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
  E.rootViews = E.cfg->contains("rootViews") && E.cfg->getBool("rootViews");
  E.rootViewSettings = RootViews::loadSettings(*E.cfg);
  // As many evaluations at once as search threads; a batch size named in the settings is taken as it is.
  const Setup::MaxBatchSizeRequest batch = E.cfg->contains("nnMaxBatchSize")
    ? Setup::MaxBatchSizeRequest::explicitSize(E.cfg->getInt("nnMaxBatchSize", 1, 65536))
    : Setup::MaxBatchSizeRequest::fromConcurrency();
  Rand seedRand("engine");
  E.nnEval = Setup::initializeNNEvaluator(
    "engine", file, "", *E.cfg, *E.logger, seedRand, std::max(1, E.params.numThreads), GameRules::MAX_BOARD_LEN,
    GameRules::MAX_BOARD_LEN, batch, true, false, Setup::SETUP_FOR_ANALYSIS
  );
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

// A move string of KataGo's protocol (the engine's action text, gamerules.h) -> the engine's action, or -1.
int actionOf(const json& m) {
  Loc loc;
  if(!m.is_string() || !Location::tryOfString(m.get<string>(), GameRules::MAX_BOARD_LEN, GameRules::MAX_BOARD_LEN, loc))
    return -1;
  return Board::actionOfMoveLoc(loc);
}

void dropGoFields(json& j) {
  for(const char* key : {"scoreMean", "noResultValue", "rawNoResultProb"})
    j.erase(key);
}

// The query's "seed" as text (default 0): a string as it is, a non-negative integer in decimal (analyze checks it).
string seedText(const json& q) {
  const json* s = field(q, "seed");
  return s == nullptr ? "0" : s->is_string() ? s->get<string>() : std::to_string(s->get<uint64_t>());
}

// The drop of the highest prior after the pick, a root action of a search whose root is g in `view` ("" its own): from
// the policy of the pick's node in `tree`, the search's root node, or, for a pick that the search did not expand (a
// symmetric twin's) or with no tree (the root's views: each search's tree goes with the next search), from one
// evaluation of the position after it.
int bestDrop(const Game& g, const string& view, const SearchNode* tree, int pick) {
  const Loc loc = Board::moveLocOfAction(pick);
  const NNOutput* out = nullptr;
  std::shared_ptr<NNOutput> evaluated;
  if(tree != nullptr) {
    ConstSearchNodeChildrenReference children = tree->getChildren();
    for(int i = 0; i < children.getCapacity(); i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == nullptr)
        break;
      if(children[i].getMoveLoc() == loc)
        out = child->getNNOutput();
    }
  }
  if(out == nullptr) {
    std::unique_ptr<Game> after(new Game(g));
    string why;
    RootViews::setBoardView(after->board, view, why);
    playAction(*after, pick);
    NNResultBuf buf;
    buf.symmetryFromHash = true;
    MiscNNInputParams params;
    params.diversityMember = E.search->searchParams.diversityMember;  //AZdb's latent: the search's member's drop
    E.nnEval->evaluate(after->board, after->hist, after->pla(), params, buf, false, false);
    evaluated = buf.result;
    out = evaluated.get();
  }
  const PolicyView policy = out->policy();
  int best = -1;
  float bestProb = -1;
  for(int i = 0; i < policy.n; i++)
    if(policy.legalPos(i) != NNPos::PASS_POS && policy.probAt(i) > bestProb) {
      best = policy.legalPos(i);
      bestProb = policy.probAt(i);
    }
  return best;
}

// A candidate's line of the engine's actions from the root, g in `view` ("" its own), with its first action the
// candidate, as the site's moves: each pick joined with the drop after it. A pick that ends the line takes the best
// drop (bestDrop, with the search's tree) when it is the candidate, and the line stops before it otherwise. ends: the
// index of each site move's last action, for KataGo's numbers by action (pvVisits). The callers cut the line. Give it
// no bound: GCC 15.2 at -O3 took one caller's max(pvLen, 1) as another caller's constant 1.
void siteLine(const Game& g, const string& view, const SearchNode* tree, const vector<int>& actions,
              vector<string>& moves, vector<size_t>& ends) {
  moves.clear();
  ends.clear();
  Pos& cur = scratch(2);
  cur = g.board.pos;
  if(!view.empty()) {
    string why;
    GameRules::set_view(cur, view, why);
  }
  for(size_t i = 0; i < actions.size(); i++) {
    const int a = actions[i];
    if(a < 0 || a == GameRules::PASS_ACTION)
      break;
    string text = siteText(cur, a);
    if(isPick(cur, a)) {
      const bool inLine = i + 1 < actions.size() && actions[i + 1] >= 0;
      const int drop = inLine ? actions[i + 1] : i == 0 ? bestDrop(g, view, tree, a) : -1;
      if(drop < 0)
        break;
      GameRules::play(cur, a);
      text += ">" + siteText(cur, drop);
      GameRules::play(cur, drop);
      if(inLine)
        i++;
    }
    else
      GameRules::play(cur, a);
    moves.push_back(text);
    ends.push_back(i);
  }
}

// An analysis: its query, its game, the root's views, and its search in slices (design 04, "Search in slices"): where
// it is. The playouts are run as runWholeSearch runs them (search.cpp), only paused between slices, so the slices of a
// search add up to one whole search of the same visits. With more views, the slices run the passes and the checks
// (RootViews::Searches) one after another, each with its share of the query's visits.
struct Running {
  json query;
  std::unique_ptr<Game> game;
  // The cover of the root's views (RootViews::cover) when the query searches them, else the rule's view alone, {""}.
  // One view: one search, in that view. More: their passes and checks, in searches.
  vector<string> views;
  std::unique_ptr<RootViews::Searches> searches;
  // The network's evaluation of the root in the rule's view, for the policy and the ownership of a search in another
  // view or in more views (with includePolicy or includeOwnership).
  std::shared_ptr<NNOutput> ruleEval;
  int64_t visits = 0;  // the query's
  SearchThread* thread = nullptr;  // the search in slices, or none when every search is done
  int64_t maxVisits = 0, nonPlayoutVisits = 0, playouts = 0;  // its visits
  int64_t sliceVisits = 0;  // 0: by time
  double sliceSeconds = 0;

  // The rule's view alone: the search and the reply without the root's views.
  bool ruleViewAlone() const { return views.size() == 1 && views[0].empty(); }
  // Its end ends its search in slices, and the score centre of the root's views (RootViews::Searches::done).
  ~Running() {
    if(thread != nullptr) {
      E.search->endManualSearch(*thread);
      delete thread;
    }
    E.search->fixedScoreCenter = std::numeric_limits<double>::quiet_NaN();
  }
};
Running* RUN = nullptr;

// The query's analysisPVLen (default 15): the site's moves of a line.
int analysisPVLen(const json& q) {
  int pvLen = 15;
  if(const json* l = field(q, "analysisPVLen"))
    if(l->is_number_integer())
      pvLen = std::max(0, l->get<int>());
  return pvLen;
}

// What each search reports (Search::getAnalysisJson): Black's values, and two actions for each site move of the line
// and one more for its first edit's drop.
RootViews::Report analysisReport(const json& q) {
  RootViews::Report report;
  report.perspective = P_BLACK;
  report.analysisPVLen = 2 * analysisPVLen(q) + 1;
  report.preventEncore = false;
  report.includePVVisits = flag(q, "includePVVisits");
  return report;
}

// A moveInfo of a search whose root is g in `view` ("" its own), in the site's moves: its move and its line (siteLine,
// with the search's tree), and KataGo's numbers by action (pvVisits, pvEdgeVisits) by site move. Its first action, or
// -1 when it names no site move.
int siteMoveInfo(const Game& g, const string& view, const SearchNode* tree, int pvLen, json& mi) {
  vector<int> actions;
  for(const json& m : mi["pv"])
    actions.push_back(actionOf(m));
  if(actions.empty())
    actions.push_back(actionOf(mi["move"]));
  vector<string> moves;
  vector<size_t> ends;
  siteLine(g, view, tree, actions, moves, ends);
  if(moves.empty())
    return -1;
  mi["move"] = moves[0];
  const size_t n = std::min(moves.size(), size_t(pvLen));
  mi["pv"] = vector<string>(moves.begin(), moves.begin() + n);
  for(const char* key : {"pvVisits", "pvEdgeVisits"})
    if(mi.contains(key)) {
      json byMove = json::array();
      for(size_t i = 0; i < n; i++)
        byMove.push_back(mi[key][ends[i]]);
      mi[key] = std::move(byMove);
    }
  dropGoFields(mi);
  return actions[0];
}

// The policy and the ownership of the reply (includePolicy, includeOwnership) from root, the network's evaluation of
// g's root in its own view (the reply's "view").
void rootMaps(const json& q, const Game& g, const NNOutput* root, json& reply) {
  if(flag(q, "includePolicy") && root != nullptr) {
    // The root's network policy over every legal action, highest first: a placement by its site string, and a pick by
    // its own string "I:q,r", the first half of an edit, with the pick's prior (the network gives an edit's prior in
    // two steps, the pick's and then the drop's at the pick's node). A network of the inputs version 13 gives no prior
    // to an action of a blueprint equal to a lower-numbered one, which its search does not list: the lower one's
    // action has that prior (NNInputs' NetActionMap), and the policy leaves the other out.
    static float probs[NNPos::MAX_NN_POLICY_SIZE];
    root->policy().toDense(probs);
    vector<int> legal;
    GameRules::legal_actions(g.board.pos, legal);
    vector<std::pair<double, string>> rows;
    for(int a : legal)
      if(probs[a] >= 0)
        rows.push_back({probs[a], siteText(g.board.pos, a)});
    std::stable_sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    json policy = json::array();
    for(const auto& [prob, name] : rows)
      policy.push_back({name, Global::roundDynamic(prob, 8)});
    reply["policy"] = std::move(policy);
  }
  if(flag(q, "includeOwnership") && root != nullptr && root->whiteOwnerMap != nullptr) {
    // The network's ownership at the root, over the view's hexes (index y * 67 + x, "view"), Blue's side: +1 where a
    // cell of Blue's ends the game, -1 for Red's. Rounded as KataGo rounds the ownership, and never -0.
    json ownership = json::array();
    for(int y = 0; y < bp::VIEW; y++)
      for(int x = 0; x < bp::VIEW; x++)
        ownership.push_back(Global::roundStatic(-root->whiteOwnerMap[NNPos::xyToPos(x, y, root->nnXLen)], 1000000.0) + 0.0);
    reply["ownership"] = std::move(ownership);
  }
}

// The analysis reply around rootInfo and moveInfos, in the site's moves: the protocol's fields, and the cover of the
// root's views when it is not the rule's view alone. The policy and the ownership are the network's at the root in the
// rule's view: the search's root, else the evaluation of the root that a search in another view or in more views took.
json replyOf(const Running& r, bool during, json rootInfo, json moveInfos) {
  const json& q = r.query;
  const Game& g = *r.game;
  dropGoFields(rootInfo);
  rootInfo["currentPlayer"] = playerName(g.pla() == P_BLACK ? 0 : 1);
  if(!r.ruleViewAlone())
    rootInfo["views"] = viewsJson(g.board.pos, r.views);
  json reply = {
    {"id", q["id"]}, {"isDuringSearch", during}, {"turnNumber", g.moves}, {"engine", engineTag()},
    {"rootInfo", std::move(rootInfo)}, {"moveInfos", std::move(moveInfos)}, {"movesOutsideView", g.outside},
  };
  const SearchNode* tree = E.search->rootNode;
  rootMaps(q, g, !r.ruleViewAlone() ? r.ruleEval.get() : tree != nullptr ? tree->getNNOutput() : nullptr, reply);
  reply["view"] = viewJson(g.board.pos);
  return reply;
}

// The analysis reply of one search from its tree as it is now (during a search in slices, or after it): the search of
// the rule's view, or of the one view of the root's views (r.views) that holds every legal placement.
json searchReply(const Running& r, bool during) {
  const json& q = r.query;
  const Game& g = *r.game;
  const string& view = r.views[0];
  const SearchNode* tree = E.search->rootNode;
  const int pvLen = analysisPVLen(q);
  json out;
  if(!RootViews::analysisJson(*E.search, analysisReport(q), out))
    return errorReply(q["id"], {"", "the search gave no analysis"});
  json moveInfos = json::array();
  std::map<int, string> named;  // each candidate's action -> its site move
  for(json& mi : out["moveInfos"]) {
    const int a = siteMoveInfo(g, view, tree, pvLen, mi);
    if(a < 0)
      continue;
    named[a] = mi["move"].get<string>();
    moveInfos.push_back(std::move(mi));
  }
  // The site's move of a root action, as its candidate names it: a pick joined with its best drop.
  auto siteMove = [&](int a) -> json {
    auto it = named.find(a);
    if(it != named.end())
      return it->second;
    vector<string> moves;
    vector<size_t> ends;
    siteLine(g, view, tree, {a}, moves, ends);
    return moves.empty() ? json(nullptr) : json(moves[0]);
  };
  // A symmetric twin names the candidate that was searched (design 04, "Symmetric twins").
  for(json& mi : moveInfos)
    if(mi.contains("isSymmetryOf"))
      mi["isSymmetryOf"] = siteMove(actionOf(mi["isSymmetryOf"]));
  json rootInfo = std::move(out["rootInfo"]);
  if(!view.empty())
    rootInfo["searches"] = 1;
  json reply = replyOf(r, during, std::move(rootInfo), std::move(moveInfos));
  if(flag(q, "chooseMove") && !during) {
    // The move KataGo's GTP engine would play (Search::getChosenMoveLoc), drawn with the chosen-move temperature and
    // seeded by the query's "seed" (default 0) and the position, never by earlier requests. A pick names its edit,
    // as its candidate does.
    Rand rand("chosenMove:" + seedText(q) + ":" + g.board.pos_hash.toString());
    const Loc loc = E.search->getChosenMoveLoc(rand);
    reply["chosenMove"] = loc == Board::NULL_LOC ? json(nullptr) : siteMove(Board::actionOfMoveLoc(loc));
  }
  return reply;
}

// The analysis reply of the root's views (RootViews::Searches) from the searches done and the one in slices once it
// has played (a search that just began has its root only): the candidates of every view, by their checks' values and
// then their passes', and the root's values, the best candidate's. The rootInfo has no raw values of the network: each
// search has its own root.
json viewsReply(const Running& r, bool during) {
  const json& q = r.query;
  const Game& g = *r.game;
  json res;
  RootViews::Choice choice;
  // A pass of 1 visit (a small maxVisits) searches its root only, and finds no candidate: the reply then has the first
  // pass's policy move, with no visits (RootViews::Searches::result). It has none before the first pass's root has its
  // evaluation.
  if(!r.searches->result(res, choice, r.thread != nullptr && r.playouts > 0 ? E.search : nullptr))
    return errorReply(q["id"], {"", "the search gave no analysis"});
  const int pvLen = analysisPVLen(q);
  // Each candidate's line from the root in the view that names its move ("view", else the root's own). No tree: each
  // search's tree went with the next search.
  json moveInfos = json::array();
  for(json& mi : res["moveInfos"]) {
    const string view = mi.contains("view") ? mi["view"].get<string>() : string();
    if(siteMoveInfo(g, view, nullptr, pvLen, mi) < 0)
      continue;
    mi.erase("view");
    moveInfos.push_back(std::move(mi));
  }
  json reply = replyOf(r, during, std::move(res["rootInfo"]), std::move(moveInfos));
  // The best check or pass: the search's choice, with no temperature (RootViews::search, as GTP's genmove).
  if(flag(q, "chooseMove") && !during)
    reply["chosenMove"] = reply["moveInfos"].empty() ? json(nullptr) : reply["moveInfos"][0]["move"];
  return reply;
}

json analysisReply(const Running& r, bool during) {
  return r.searches != nullptr ? viewsReply(r, during) : searchReply(r, during);
}

void endRunning() {
  delete RUN;
  RUN = nullptr;
}

// The visits of a search of the root's views with its share of the query's: runWholeSearch's with that searchFactor.
int64_t shareOfVisits(int64_t visits, double share) {
  return (int64_t)std::ceil(std::min((double)((int64_t)1 << 62), visits * share));
}

// Begins a search in slices of E.search's root, with its share of the query's visits (1: all of them).
void beginSlices(Running& r, double share) {
  r.maxVisits = shareOfVisits(r.visits, share);
  r.playouts = 0;
  E.search->beginSearch(false);
  r.nonPlayoutVisits = E.search->getRootVisits();
  r.thread = new SearchThread(0, *E.search);
}

// Begins the next search of the root's views (searches->next()) in slices. False when every search is done.
bool beginNextView(Running& r) {
  if(!r.searches->next())
    return false;
  E.search->setPosition(r.searches->pla(), r.searches->board(), r.searches->hist());
  beginSlices(r, r.searches->share());
  return true;
}

// Ends the search in slices at its visits. With the root's views, gives it to searches and begins the next one, if
// any. False when the search gave no analysis.
bool endSlices(Running& r) {
  E.search->endManualSearch(*r.thread);
  delete r.thread;
  r.thread = nullptr;
  if(r.searches == nullptr)
    return true;
  if(!r.searches->done(*E.search, Board::NULL_LOC))
    return false;
  beginNextView(r);
  return true;
}

// One slice of the running search; its reply. When the search reaches its visits, it ends: with the root's views, the
// slice goes on into the next search, and the last search's end ends the analysis.
json runSlice() {
  Running& r = *RUN;
  ClockTimer timer;
  int64_t done = 0;
  while(r.thread != nullptr) {
    if(r.playouts + r.nonPlayoutVisits >= r.maxVisits) {
      if(!endSlices(r)) {
        const json reply = errorReply(r.query["id"], {"", "the search gave no analysis"});
        endRunning();
        return reply;
      }
      continue;  // the next search can reach its visits with its root's
    }
    if(r.sliceVisits > 0 ? done >= r.sliceVisits : (done >= 1 && timer.getSeconds() >= r.sliceSeconds))
      break;
    const double left = double(r.maxVisits - r.playouts - r.nonPlayoutVisits);
    if(E.search->runSinglePlayout(*r.thread, left)) {
      r.playouts++;
      done++;
    }
  }
  const bool during = r.thread != nullptr;
  const json reply = analysisReply(r, during);
  if(!during)
    endRunning();
  return reply;
}

json analyze(const json& q) {
  if(E.search == nullptr)
    return errorReply(q["id"], {"", "no network loaded (load_model)"});
  std::unique_ptr<Game> g(new Game());
  Fail f;
  if(!buildPosition(q, *g, f))
    return errorReply(q["id"], f);
  if(g->board.isGameOver())
    return errorReply(q["id"], {"position", "the game is over"});
  if(flag(q, "includeMovesOwnership"))
    return errorReply(q["id"], {"includeMovesOwnership", "Blueprints' build gives the root's ownership only: each "
                                                         "candidate's position has its own view"});
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
  // The search of the root's views (RootViews): the query's field, by default the settings' rootViews.
  bool rootViews = E.rootViews;
  if(const json* v = field(q, "rootViews")) {
    if(!v->is_boolean())
      return errorReply(q["id"], {"rootViews",
                                  "rootViews is true or false: whether the search covers the root's views"});
    rootViews = v->get<bool>();
  }
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
  endRunning();  // one search at a time: a new query ends the one before
  std::unique_ptr<Running> r(new Running());
  r->query = q;
  r->views = rootViews ? RootViews::cover(g->board) : vector<string>{""};
  r->visits = visits;
  r->sliceVisits = sliceVisits;
  r->sliceSeconds = sliceSeconds;
  const bool ruleView = r->ruleViewAlone();
  p.maxVisits = visits;
  p.maxPlayouts = (int64_t)1 << 50;
  p.maxTime = maxTime;
  E.search->setParams(p);
  // The root's ownership needs its evaluation with the ownership head's map. A search in another view or in more
  // views takes the policy and the ownership from the root's evaluation in the rule's view, as the search of the
  // rule's view has it at its root (Search::computeRootNNEvaluation).
  E.search->setAlwaysIncludeOwnerMap(ruleView && flag(q, "includeOwnership"));
  if(!ruleView && (flag(q, "includeOwnership") || flag(q, "includePolicy"))) {
    E.search->setPosition(g->pla(), g->board, g->hist);
    NNResultBuf buf;
    E.search->computeRootNNEvaluation(buf, flag(q, "includeOwnership"));
    r->ruleEval = buf.result;
  }
  if(r->views.size() == 1) {
    // One search: in the rule's view, or in the one view that holds every legal placement.
    const Board* board = &g->board;
    std::unique_ptr<Board> viewed;
    if(!ruleView) {
      viewed.reset(new Board(g->board));
      string why;
      RootViews::setBoardView(*viewed, r->views[0], why);
      board = viewed.get();
    }
    E.search->setPosition(g->pla(), *board, g->hist);
  }
  // Root noise draws from the search thread's random numbers, seeded with the query's "seed" in place of the engine's
  // and the count of searches reset, so the same request draws the same noise whatever came before (design 04,
  // "Reproducible analysis"). Each search of the root's views counts one more search, and so has its own seed.
  E.search->randSeed = "analysis:" + seedText(q);
  E.search->numSearchesBegun = 0;
  r->game = std::move(g);
  // More views: the passes and the checks (RootViews::Searches), each with its share of the visits and of maxTime.
  if(r->views.size() > 1)
    r->searches.reset(new RootViews::Searches(
      r->game->board, r->game->hist, r->game->pla(), r->views, E.rootViewSettings, analysisReport(q)));
  if(slice == nullptr) {
    if(r->searches == nullptr) {
      E.search->runWholeSearch(r->game->pla());
      return analysisReply(*r, false);
    }
    bool suc = true;
    while(suc && r->searches->next()) {
      E.search->setPosition(r->searches->pla(), r->searches->board(), r->searches->hist());
      E.search->runWholeSearch(nullptr, nullptr, false, TimeControls(), r->searches->share());
      suc = r->searches->done(*E.search, Board::NULL_LOC);
    }
    return suc ? analysisReply(*r, false) : errorReply(q["id"], {"", "the search gave no analysis"});
  }
  RUN = r.release();
  if(RUN->searches == nullptr)
    beginSlices(*RUN, 1.0);
  else
    beginNextView(*RUN);
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
  json reply = analysisReply(*RUN, false);
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

void BlueprintsProtocol::setup(const string& configText, const string& build) {
  P.baseConfig = configText;
  P.build = build;
}

char* BlueprintsProtocol::modelBuffer(size_t len) {
  E.pending.assign(std::max<size_t>(len, 1), 0);
  return E.pending.data();
}

string BlueprintsProtocol::handle(const string& request) {
  return handleJson(request).dump();
}
