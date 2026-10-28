#include "../program/rootviews.h"

#include <algorithm>
#include <limits>
#include <map>

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

bool RootViews::search(
  AsyncBot* bot, const Board& rootBoard, const BoardHistory& rootHist, Player rootPla,
  const vector<string>& views, const TimeControls& tc, double searchFactor,
  const Settings& settings, const Report& report,
  const std::function<void()>& onSearchBegun, const std::function<bool()>& stopped,
  json& ret, Pick& pick
) {
  //Every search reports its values from one side: the report's perspective as the root has it. A check's root has
  //another player to move.
  const Player perspective =
    report.perspective == P_BLACK || (report.perspective != P_WHITE && rootPla == P_BLACK) ? P_BLACK : P_WHITE;
  const double moverSign = rootPla == perspective ? 1.0 : -1.0;
  const int numViews = (int)views.size();
  testAssert(numViews >= 1);
  auto inView = [&rootBoard](const string& view) {
    Board board = rootBoard;
    string why;
    bool suc = setBoardView(board, view, why);
    testAssert(suc);
    return board;
  };

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
  //The clock of each search. A check's root is the next position. When the root's move has more of its turn to come,
  //Fischer's and absolute time give it at most its share of the time left (TimeControls::getTime), and a check of the
  //turn's next move would not have that limit: each search takes it as its limit for one move.
  TimeControls clock = tc;
  const int turnMovesLeft = GameRules::turn_moves_left(rootBoard.pos);
  if(!tc.isEffectivelyUnlimitedTime() && (tc.increment > 0 || tc.numPeriodsLeftIncludingCurrent <= 0) &&
     turnMovesLeft > 1)
    clock.maxTimePerMove = std::min(clock.maxTimePerMove, tc.mainTimeLeft / turnMovesLeft);
  int numSearches = 0;
  int64_t totalVisits = 0;
  Loc chosen = Board::NULL_LOC;  //the move that the last search chose
  auto runSearch = [&](const Board& board, const BoardHistory& hist, Player pla, double share, json& out) {
    bot->setPosition(pla, board, hist);
    chosen = bot->genMoveSynchronous(pla, clock, share * searchFactor, numSearches == 0 ? onSearchBegun : stopIfEnded);
    numSearches++;
    if(numViews > 1 && isStopped())
      return false;
    out = json();
    if(!bot->getSearch()->getAnalysisJson(
         perspective, report.analysisPVLen, report.preventEncore, report.includePolicy, report.includeOwnership,
         report.includeOwnershipStdev, report.includeMovesOwnership, report.includeMovesOwnershipStdev,
         report.includePVVisits, report.includeNoResultValue, out))
      return false;
    totalVisits += out["rootInfo"]["visits"].get<int64_t>();
    return true;
  };

  //Each root move of the root's own view, by its child position, so that a move that the root's view names keeps its
  //text there. A move that it cannot name keeps its text in its pass's view, and the moveInfo gives that view
  //("view").
  auto childHash = [](const Board& board, Loc loc) {
    GameRules::Position p = board.pos;
    GameRules::play(p, Board::actionOfMoveLoc(loc));
    return GameRules::hash(p);
  };
  std::map<Hash128, Loc> ownMoves;
  if(numViews > 1 || !views[0].empty()) {
    vector<int> actions;
    GameRules::legal_actions(rootBoard.pos, actions);
    for(int action : actions)
      ownMoves[childHash(rootBoard, Board::moveLocOfAction(action))] = Board::moveLocOfAction(action);
  }
  //The move of the root's own view with that child, or NULL_LOC.
  auto ownMove = [&](const Hash128& child) {
    auto own = ownMoves.find(child);
    return own == ownMoves.end() ? Board::NULL_LOC : own->second;
  };
  //Names a moveInfo's move by its move of the root's own view, or gives it the view that names it.
  auto nameMove = [&](json& info, Loc own, const string& view) {
    if(own == Board::NULL_LOC) {
      info["view"] = view;
      return;
    }
    const string text = Location::toString(own, rootBoard);
    info["move"] = text;
    if(info.contains("pv") && !info["pv"].empty())
      info["pv"][0] = text;
  };
  //The pick: the move of the root's own view when it names the move, else the move in its view.
  auto setPick = [&pick](Loc loc, Loc own, const string& view) {
    pick.loc = own != Board::NULL_LOC ? own : loc;
    pick.view = own != Board::NULL_LOC ? string() : view;
  };

  if(numViews == 1) {
    const Board board = inView(views[0]);
    if(!runSearch(board, rootHist, rootPla, 1.0, ret))
      return false;
    setPick(chosen, Board::NULL_LOC, views[0]);
    if(!views[0].empty()) {
      for(json& info : ret["moveInfos"]) {
        Loc loc;
        if(Location::tryOfString(info["move"].get<string>(), board, loc))
          nameMove(info, ownMove(childHash(board, loc)), views[0]);
      }
      ret["view"] = views[0];  //the frame of the ownership and the policy
      if(chosen != Board::NULL_LOC)
        setPick(chosen, ownMove(childHash(board, chosen)), views[0]);
    }
    ret["rootInfo"]["views"] = views;
    ret["rootInfo"]["searches"] = numSearches;
    return true;
  }

  //The passes, and their candidates.
  struct Candidate {
    int pass;
    Loc loc;
    Loc own;  //the move in the root's own view, or NULL_LOC
    json info;  //the moveInfo of the pass where it has the most visits
    double value;  //its utility from the root mover's side, the mean of its passes' weighted by their visits
    int64_t visits;  //its visits in the passes
    json check;  //the rootInfo of its check, or null
  };
  vector<Candidate> candidates;
  std::map<Hash128, size_t> byChild;
  vector<Board> boards;
  boards.reserve(numViews);
  json passes = json::array();
  for(int i = 0; i < numViews; i++) {
    boards.push_back(inView(views[i]));
    json out;
    if(!runSearch(boards[i], rootHist, rootPla, settings.passShare / numViews, out))
      return false;
    if(i == 0)
      search->fixedScoreCenter = search->recentScoreCenter;
    int taken = 0;
    for(const json& info : out["moveInfos"]) {
      Loc loc;
      if(taken >= settings.candidates)
        break;
      if(info.contains("isSymmetryOf") || !Location::tryOfString(info["move"].get<string>(), boards[i], loc))
        continue;
      taken++;
      const Hash128 child = childHash(boards[i], loc);
      const double value = moverSign * info["utility"].get<double>();
      const int64_t visits = info["visits"].get<int64_t>();
      auto found = byChild.find(child);
      if(found == byChild.end()) {
        byChild[child] = candidates.size();
        candidates.push_back(Candidate{i, loc, ownMove(child), info, value, visits, json()});
        continue;
      }
      Candidate& c = candidates[found->second];
      c.value = (c.value * c.visits + value * visits) / std::max<int64_t>(1, c.visits + visits);
      if(visits > c.info["visits"].get<int64_t>()) {
        c.pass = i;
        c.loc = loc;
        c.info = info;
      }
      c.visits += visits;
    }
    out["view"] = views[i];
    passes.push_back(std::move(out));
  }
  if(candidates.empty())
    return false;

  //The checks of the finalists: a search of each one's child position.
  vector<size_t> order(candidates.size());
  for(size_t i = 0; i < order.size(); i++)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return candidates[a].value > candidates[b].value; });
  const size_t numFinalists = std::min<size_t>(std::max(1, settings.finalists), order.size());
  json checks = json::array();
  for(size_t f = 0; f < numFinalists; f++) {
    //The move in the root's own view when that view names it. Else in its pass's view, and then the child's history
    //starts at the child: a move in a view of its own cannot replay from the history's start (the search's graph hash
    //replays it). The game's position keeps what its inputs read of the earlier moves.
    Candidate& c = candidates[order[f]];
    const bool own = c.own != Board::NULL_LOC;
    Board board = own ? rootBoard : boards[c.pass];
    BoardHistory hist = rootHist;
    hist.makeBoardMoveAssumeLegal(board, own ? c.own : c.loc, rootPla, NULL);
    if(hist.isGameFinished)
      continue;  //the pass's value is the result's
    if(!own)
      hist.clear(board, hist.presumedNextMovePla, hist.rules, hist.encorePhase);
    json out;
    if(!runSearch(board, hist, hist.presumedNextMovePla, (1.0 - settings.passShare) / numFinalists, out))
      return false;
    c.check = out["rootInfo"];
    c.value = moverSign * c.check["utility"].get<double>();
    if(!out["moveInfos"].empty()) {
      json pv = json::array();
      pv.push_back(c.info["move"]);
      for(const json& move : out["moveInfos"][0]["pv"])
        pv.push_back(move);
      c.info["pv"] = pv;
    }
  }
  std::stable_sort(order.begin(), order.begin() + numFinalists,
                   [&](size_t a, size_t b) { return candidates[a].value > candidates[b].value; });

  //The response: the finalists by their checks' values, then the other candidates by their passes' values. A check's
  //values replace its candidate's, and the best candidate's are the root's.
  const char* const valueKeys[] = {"winrate", "scoreSelfplay", "scoreLead", "scoreStdev", "utility"};
  json moveInfos = json::array();
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
    nameMove(info, c.own, views[c.pass]);
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
  const Candidate& picked = candidates[order[0]];
  setPick(picked.loc, picked.own, views[picked.pass]);
  const json& best = moveInfos[0];
  json rootInfo;
  rootInfo["visits"] = totalVisits;
  for(const char* key : valueKeys)
    rootInfo[key] = best[key];
  Hash128 thisHash = rootBoard.getSitHashWithSimpleKo(rootPla), symHash = thisHash;
  for(int symmetry : SymmetryHelpers::ALLOWED) {
    const Hash128 hash = SymmetryHelpers::getSymBoard(rootBoard, symmetry).getSitHashWithSimpleKo(rootPla);
    if(hash < symHash)
      symHash = hash;
  }
  rootInfo["thisHash"] = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
  rootInfo["symHash"] = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
  rootInfo["currentPlayer"] = PlayerIO::playerToStringShort(rootPla);
  rootInfo["views"] = views;
  rootInfo["searches"] = numSearches;
  ret["moveInfos"] = std::move(moveInfos);
  ret["rootInfo"] = std::move(rootInfo);
  ret["passes"] = std::move(passes);
  ret["checks"] = std::move(checks);
  return true;
}
