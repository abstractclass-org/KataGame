#include "../program/rootviews.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>

#include "../core/global.h"
#include "../core/test.h"
#include "../neuralnet/nninputs.h"

using namespace std;
using json = nlohmann::json;

RootViews::Settings RootViews::loadSettings(ConfigParser& cfg) {
  Settings settings;
  if(cfg.contains("rootViewsPassShare"))
    settings.passShare = cfg.getDouble("rootViewsPassShare", 0.01, 0.99);
  if(cfg.contains("rootViewsCandidates"))
    settings.candidates = cfg.getInt("rootViewsCandidates", 1, 100);
  if(cfg.contains("rootViewsFinalists"))
    settings.finalists = cfg.getInt("rootViewsFinalists", 1, 100);
  return settings;
}

vector<string> RootViews::cover(const Board& board) {
  vector<string> views;
  GameInterface::viewCover<GameRules>(board.pos, views);
  return views;
}

bool RootViews::setBoardView(Board& board, const string& view, string& why) {
  if(view.empty())
    return true;
  GameRules::Position p = board.pos;
  if(!GameInterface::setView<GameRules>(p, view, why))
    return false;
  board.setPosition(p);
  return true;
}

void RootViews::restartHistory(const Board& board, BoardHistory& hist) {
  const int64_t turnNumber = hist.getCurrentTurnNumber();
  hist.clear(board, hist.presumedNextMovePla, hist.rules, hist.encorePhase);
  hist.setInitialTurnNumber(turnNumber);
}

//The child position of the move loc on board, by its hash: the same position whatever view named the move.
static Hash128 childHash(const Board& board, Loc loc) {
  GameRules::Position p = board.pos;
  GameRules::play(p, Board::actionOfMoveLoc(loc));
  return GameRules::hash(p);
}

RootViews::Searches::Searches(
  const Board& rootBoard, const BoardHistory& rootHist, Player rootPla, const vector<string>& views,
  const Settings& settings, const Report& report
)
  : rootBoard_(rootBoard), rootHist_(rootHist), rootPla_(rootPla), views_(views), settings_(settings), report_(report)
{
  //Every search reports its values from one side: the report's perspective as the root has it. A check's root has
  //another player to move.
  report_.perspective =
    report.perspective == P_BLACK || (report.perspective != P_WHITE && rootPla == P_BLACK) ? P_BLACK : P_WHITE;
  moverSign_ = rootPla == report_.perspective ? 1.0 : -1.0;
  testAssert(!views.empty());
  boards_.reserve(views.size());
  for(const string& view : views) {
    boards_.push_back(rootBoard);
    string why;
    bool suc = setBoardView(boards_.back(), view, why);
    testAssert(suc);
  }
  //Each root move of the root's own view, by its child position, so that a move that the root's view names keeps its
  //text there. A move that it cannot name keeps its text in its pass's view, and the moveInfo gives that view
  //("view").
  if(views.size() > 1 || !views[0].empty()) {
    vector<int> actions;
    GameRules::legal_actions(rootBoard.pos, actions);
    for(int action : actions)
      ownMoves_[childHash(rootBoard, Board::moveLocOfAction(action))] = Board::moveLocOfAction(action);
  }
}

bool RootViews::Searches::next() {
  current_ = -1;
  const int numViews = (int)views_.size();
  if(nextSearch_ < numViews) {
    //A pass: the root in one view, with an even share of the passes' share (one view: the whole budget).
    current_ = nextSearch_++;
    board_ = boards_[current_];
    hist_ = rootHist_;
    pla_ = rootPla_;
    share_ = numViews == 1 ? 1.0 : settings_.passShare / numViews;
    return true;
  }
  if(numViews == 1 || results_.candidates.empty())
    return false;
  //The checks of the finalists, the candidates with the best values of the passes: a search of each one's child
  //position.
  if(order_.empty()) {
    const vector<Candidate>& candidates = results_.candidates;
    order_.resize(candidates.size());
    for(size_t i = 0; i < order_.size(); i++)
      order_[i] = i;
    std::stable_sort(order_.begin(), order_.end(), [&](size_t a, size_t b) {
      return candidates[a].value > candidates[b].value;
    });
    numFinalists_ = std::min<size_t>(std::max(1, settings_.finalists), order_.size());
  }
  while(nextSearch_ < numViews + (int)numFinalists_) {
    const int f = nextSearch_++ - numViews;
    //The move in the root's own view when that view names it. Else in its pass's view, and then the child's history
    //starts at the child (restartHistory), as GTP's game after such a move.
    const Candidate& c = results_.candidates[order_[f]];
    const bool own = c.own != Board::NULL_LOC;
    board_ = own ? rootBoard_ : boards_[c.pass];
    hist_ = rootHist_;
    hist_.makeBoardMoveAssumeLegal(board_, own ? c.own : c.loc, rootPla_, NULL);
    if(hist_.isGameFinished)
      continue;  //the pass's value is the result's
    if(!own)
      restartHistory(board_, hist_);
    pla_ = hist_.presumedNextMovePla;
    share_ = (1.0 - settings_.passShare) / numFinalists_;
    current_ = numViews + f;
    return true;
  }
  return false;
}

bool RootViews::Searches::done(Search& search, Loc chosen) {
  testAssert(current_ >= 0);
  json out;
  const bool suc = analysisJson(search, report_, out);
  if(suc) {
    if(current_ == 0 && views_.size() > 1)
      search.fixedScoreCenter = search.recentScoreCenter;
    record(results_, out, chosen);
  }
  current_ = -1;
  return suc;
}

bool RootViews::Searches::result(json& ret, Choice& choice, const Search* running) const {
  if(running == nullptr || current_ < 0)
    return respond(results_, ret, choice);
  Results results = results_;
  json out;
  if(analysisJson(*running, report_, out))
    record(results, out, Board::NULL_LOC);
  return respond(results, ret, choice);
}

bool RootViews::analysisJson(const Search& search, const Report& report, json& out) {
  return search.getAnalysisJson(
    report.perspective, report.analysisPVLen, report.preventEncore, report.includePolicy, report.includeOwnership,
    report.includeOwnershipStdev, report.includeMovesOwnership, report.includeMovesOwnershipStdev,
    report.includePVVisits, report.includeNoResultValue, out);
}

//The analysis out of the current search into results.
void RootViews::Searches::record(Results& results, json& out, Loc chosen) const {
  results.numSearches++;
  results.totalVisits += out["rootInfo"]["visits"].get<int64_t>();
  const int numViews = (int)views_.size();
  if(numViews == 1) {
    results.single = std::move(out);
    results.chosen = chosen;
    return;
  }
  if(current_ < numViews) {
    //A pass: its best moves are candidates, and a move that two passes find is one candidate, by its child position.
    const int i = current_;
    int taken = 0;
    for(const json& info : out["moveInfos"]) {
      Loc loc;
      if(taken >= settings_.candidates)
        break;
      if(info.contains("isSymmetryOf") || !Location::tryOfString(info["move"].get<string>(), boards_[i], loc))
        continue;
      taken++;
      const Hash128 child = childHash(boards_[i], loc);
      const double value = moverSign_ * info["utility"].get<double>();
      const int64_t visits = info["visits"].get<int64_t>();
      auto found = results.byChild.find(child);
      if(found == results.byChild.end()) {
        results.byChild[child] = results.candidates.size();
        results.candidates.push_back(Candidate{i, loc, ownMove(child), info, value, visits, json()});
        continue;
      }
      Candidate& c = results.candidates[found->second];
      c.value = (c.value * c.visits + value * visits) / std::max<int64_t>(1, c.visits + visits);
      if(visits > c.info["visits"].get<int64_t>()) {
        c.pass = i;
        c.loc = loc;
        c.info = info;
      }
      c.visits += visits;
    }
    out["view"] = views_[i];
    results.passes.push_back(std::move(out));
    return;
  }
  //A check: its values replace its candidate's, and its line follows the candidate's move.
  Candidate& c = results.candidates[order_[current_ - numViews]];
  c.check = out["rootInfo"];
  c.value = moverSign_ * c.check["utility"].get<double>();
  if(!out["moveInfos"].empty()) {
    const json& best = out["moveInfos"][0];
    json pv = json::array();
    pv.push_back(c.info["move"]);
    for(const json& move : best["pv"])
      pv.push_back(move);
    c.info["pv"] = pv;
    //The numbers by move of the line (includePVVisits): the candidate's own in its pass, then the check's.
    for(const char* key : {"pvVisits", "pvEdgeVisits"}) {
      if(!c.info.contains(key) || c.info[key].empty() || !best.contains(key))
        continue;
      json byMove = json::array();
      byMove.push_back(c.info[key][0]);
      for(const json& n : best[key])
        byMove.push_back(n);
      c.info[key] = std::move(byMove);
    }
  }
}

//The move of the root's own view with that child, or NULL_LOC.
Loc RootViews::Searches::ownMove(const Hash128& child) const {
  auto own = ownMoves_.find(child);
  return own == ownMoves_.end() ? Board::NULL_LOC : own->second;
}

//Names a moveInfo's move by its move of the root's own view, or gives it the view that names it.
void RootViews::Searches::nameMove(json& info, Loc own, const string& view) const {
  if(own == Board::NULL_LOC) {
    info["view"] = view;
    return;
  }
  const string text = Location::toString(own, rootBoard_);
  info["move"] = text;
  if(info.contains("pv") && !info["pv"].empty())
    info["pv"][0] = text;
}

//The response from results: a cover of one view gives its search's analysis. With more, the finalists by their checks'
//values, then the other candidates by their passes' values (before the checks, every candidate by its passes' values).
//A check's values replace its candidate's, and the best candidate's are the root's.
bool RootViews::Searches::respond(const Results& results, json& ret, Choice& choice) const {
  //The choice: the move of the root's own view when it names the move, else the move in its view.
  auto setChoice = [&choice](Loc loc, Loc own, const string& view) {
    choice.loc = own != Board::NULL_LOC ? own : loc;
    choice.view = own != Board::NULL_LOC ? string() : view;
  };
  if(views_.size() == 1) {
    if(results.single.is_null())
      return false;
    ret = results.single;
    const Board& board = boards_[0];
    const Loc chosen = results.chosen;
    setChoice(chosen, Board::NULL_LOC, views_[0]);
    if(!views_[0].empty()) {
      for(json& info : ret["moveInfos"]) {
        Loc loc;
        if(Location::tryOfString(info["move"].get<string>(), board, loc))
          nameMove(info, ownMove(childHash(board, loc)), views_[0]);
      }
      ret["view"] = views_[0];  //the frame of the ownership and the policy
      if(chosen != Board::NULL_LOC)
        setChoice(chosen, ownMove(childHash(board, chosen)), views_[0]);
    }
    ret["rootInfo"]["views"] = views_;
    ret["rootInfo"]["searches"] = results.numSearches;
    return true;
  }
  const vector<Candidate>& candidates = results.candidates;
  if(candidates.empty())
    return false;
  auto byValue = [&](size_t a, size_t b) { return candidates[a].value > candidates[b].value; };
  vector<size_t> order = order_;
  if(order.empty()) {
    order.resize(candidates.size());
    for(size_t i = 0; i < order.size(); i++)
      order[i] = i;
    std::stable_sort(order.begin(), order.end(), byValue);
  }
  else
    std::stable_sort(order.begin(), order.begin() + numFinalists_, byValue);

  const char* const valueKeys[] = {"winrate", "scoreSelfplay", "scoreLead", "scoreStdev", "utility"};
  json moveInfos = json::array();
  json checks = json::array();
  for(size_t i = 0; i < order.size(); i++) {
    const Candidate& c = candidates[order[i]];
    json info = c.info;
    info["visits"] = c.visits;
    info["order"] = (int)i;
    if(!c.check.is_null()) {
      for(const char* key : valueKeys)
        info[key] = c.check[key];
      info["scoreMean"] = c.check["scoreLead"];
      info["checkVisits"] = c.check["visits"];
    }
    nameMove(info, c.own, views_[c.pass]);
    if(!c.check.is_null()) {
      json check;
      check["move"] = info["move"];
      if(info.contains("view"))
        check["view"] = info["view"];
      check["rootInfo"] = c.check;
      checks.push_back(std::move(check));
    }
    moveInfos.push_back(std::move(info));
  }
  const Candidate& top = candidates[order[0]];
  setChoice(top.loc, top.own, views_[top.pass]);
  const json& best = moveInfos[0];
  json rootInfo;
  rootInfo["visits"] = results.totalVisits;
  for(const char* key : valueKeys)
    rootInfo[key] = best[key];
  Hash128 thisHash = rootBoard_.getSitHashWithSimpleKo(rootPla_), symHash = thisHash;
  for(int symmetry : SymmetryHelpers::ALLOWED) {
    const Hash128 hash = SymmetryHelpers::getSymBoard(rootBoard_, symmetry).getSitHashWithSimpleKo(rootPla_);
    if(hash < symHash)
      symHash = hash;
  }
  rootInfo["thisHash"] = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
  rootInfo["symHash"] = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
  rootInfo["currentPlayer"] = PlayerIO::playerToStringShort(rootPla_);
  rootInfo["views"] = views_;
  rootInfo["searches"] = results.numSearches;
  ret["moveInfos"] = std::move(moveInfos);
  ret["rootInfo"] = std::move(rootInfo);
  ret["passes"] = results.passes;
  ret["checks"] = std::move(checks);
  return true;
}

bool RootViews::search(
  AsyncBot* bot, const Board& rootBoard, const BoardHistory& rootHist, Player rootPla,
  const vector<string>& views, const TimeControls& tc, double searchFactor,
  const Settings& settings, const Report& report,
  const std::function<void()>& onSearchBegun, const std::function<bool()>& stopped,
  json& ret, Choice& choice
) {
  //Each search, with its share of the limits (genMoveSynchronous's searchFactor): the first takes onSearchBegun, and a
  //later one stops at once when stopped() is true.
  Search* search = bot->getSearchStopAndWait();
  struct CenterReset {
    Search* search;
    ~CenterReset() { search->fixedScoreCenter = std::numeric_limits<double>::quiet_NaN(); }
  } centerReset{search};
  auto isStopped = [&stopped]() noexcept {
    return stopped && stopped();
  };
  std::function<void()> stopIfEnded = [&]() noexcept {
    if(isStopped())
      bot->stopWithoutWait();
  };
  //The clock of each search: the root's limit for one move (TimeControls::getTime, before the lag buffer, which each
  //search applies) is each search's. A check's root is the next position, and its own limit can be larger: when the
  //root's move has more of its turn to come, Fischer's and absolute time give it at most its share of the time left.
  TimeControls clock = tc;
  if(!tc.isEffectivelyUnlimitedTime()) {
    double minTime, recommendedTime, maxTime;
    tc.getTime(rootBoard, rootHist, 0.0, minTime, recommendedTime, maxTime);
    clock.maxTimePerMove = maxTime;
  }
  std::unique_ptr<Searches> searches(new Searches(rootBoard, rootHist, rootPla, views, settings, report));
  for(int n = 0; searches->next(); n++) {
    bot->setPosition(searches->pla(), searches->board(), searches->hist());
    const Loc chosen = bot->genMoveSynchronous(
      searches->pla(), clock, searches->share() * searchFactor, n == 0 ? onSearchBegun : stopIfEnded);
    if(views.size() > 1 && isStopped())
      return false;
    if(!searches->done(*search, chosen))
      return false;
  }
  return searches->result(ret, choice);
}
