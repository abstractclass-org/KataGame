// KataGame fork (abscls-csmy): the search of the root's views (program/rootviews.h) with few visits, on the build's
// game, which must have views of its own (GameInterface::hasViews: Blueprints). Each position of POSITIONS has a cover
// of several views (n). With maxVisits 1, 2, 3 and 2n, each pass has 1 visit and searches its root only, so no pass
// finds a candidate. The checks there:
//   - the search on an AsyncBot (RootViews::search: GTP's genmove and the analysis engine) gives a response and a move
//     that is legal at the root in its view. The response's one candidate is that move: the best legal move of the
//     first pass's root by the network's policy (the network's own evaluation of that root, here), with no visits and
//     the first pass's root values. There is no check, and the searches are the n passes of 1 visit;
//   - with includeMovesOwnership, the candidate has the root's ownership, as KataGo gives a move with no visits;
//   - the searches run with the manual interface of Search, as Blueprints' engine protocol runs them in slices: no
//     response before the first pass's first playout, the policy move after it, and the same move at the end.
// Above the threshold, with maxVisits 2n + 1, 4n and 8n, the passes find candidates: the response has checks and a
// legal move. --print writes each response as one line, so that the responses of a build from before the policy's
// move (this file built against its objects) and of this one compare byte for byte above the threshold.
//
//   testrootviews MODEL POSITIONS [--print]
//
// MODEL is a network of the build's game, and POSITIONS a file of lines "LABEL STATE", each STATE a position's whole
// state (GameInterface::stateText), such as tests/data/blueprints_root_views.txt. The test builds with
// tests/enclosure/build.sh BUILD_DIR, with TESTS=testrootviews. Each search runs on one thread with fixed seeds. Exit
// code 0 when every check holds.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/global.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../core/test.h"
#include "../../neuralnet/nneval.h"
#include "../../program/playutils.h"
#include "../../program/rootviews.h"
#include "../../search/asyncbot.h"
#include "../../search/search.h"

using namespace std;
using json = nlohmann::json;

static int failures = 0;

static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 30)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

//The child position of the move loc on board, by its hash: the same position whatever view named the move.
static Hash128 childHash(const Board& board, Loc loc) {
  GameRules::Position p = board.pos;
  GameRules::play(p, Board::actionOfMoveLoc(loc));
  return GameRules::hash(p);
}

//The root in a view of its cover.
static Board inView(const Board& board, const string& view) {
  Board viewed = board;
  string why;
  bool suc = RootViews::setBoardView(viewed, view, why);
  testAssert(suc);
  return viewed;
}

static SearchParams params(int64_t visits) {
  SearchParams p;
  p.maxVisits = visits;
  p.numThreads = 1;
  return p;
}

//The best legal move of the board's root by the network's own evaluation, as the search evaluates its root: KataGo's
//policy move with no temperature (PlayUtils::chooseRandomPolicyMove), apart from the code under test.
static Loc policyMove(NNEvaluator* nnEval, const Board& board, const BoardHistory& hist, Player pla) {
  NNResultBuf buf;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = params(1).drawEquivalentWinsForWhite;
  nnEval->evaluate(board, hist, pla, nnInputParams, buf, false, false);
  Rand rand("testrootviews");  //no draw at temperature 0
  return PlayUtils::chooseRandomPolicyMove(buf.result.get(), board, hist, pla, rand, 0.0, true, Board::NULL_LOC);
}

//A root's position and its cover.
struct Root {
  string label;
  Board board;
  BoardHistory hist;
  Player pla;
  vector<string> views;
};

//The response of the search of the root's views on an AsyncBot with maxVisits visits, and its move: false when it gave
//none.
static bool searchViews(
  NNEvaluator* nnEval, Logger& logger, const Root& root, int64_t visits, const RootViews::Report& report,
  json& ret, RootViews::Choice& choice
) {
  AsyncBot bot(params(visits), nnEval, &logger, "testrootviews " + root.label + " " + Global::int64ToString(visits));
  bot.setAlwaysIncludeOwnerMap(report.includeMovesOwnership || report.includeMovesOwnershipStdev);
  return RootViews::search(
    &bot, root.board, root.hist, root.pla, root.views, TimeControls(), 1.0, RootViews::Settings(), report,
    std::function<void()>(), std::function<bool()>(), ret, choice);
}

//The move's text in its view, as GTP writes a move of another view (MOVE/VIEW).
static string moveText(const Root& root, const RootViews::Choice& choice) {
  const string text = Location::toString(choice.loc, inView(root.board, choice.view));
  return choice.view.empty() ? text : text + "/" + choice.view;
}

int main(int argc, char** argv) {
  const bool print = argc == 4 && string(argv[3]) == "--print";
  if(argc != 3 && !print) {
    fprintf(stderr, "usage: testrootviews MODEL POSITIONS [--print]\n");
    return 2;
  }
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  if(!GameInterface::hasViews<GameRules>()) {
    fprintf(stderr, "the build's game has no views of its own (GameInterface::hasViews)\n");
    return 2;
  }

  //NHWC inputs, which every backend takes (Eigen takes no other). Every evaluation has the symmetry 0.
  map<string, string> kvs = {{"cudaUseNHWC", "true"}};
  ConfigParser cfg(kvs);
  Logger logger(nullptr, false, true, false, false);
  const int S = NNPos::MAX_BOARD_LEN;
  const bool exact = NNPos::fillsNNBoard(Board::DEFAULT_X_SIZE, Board::DEFAULT_Y_SIZE);
  NNEvaluator* nnEval = new NNEvaluator(
    "test", argv[1], "", &logger, 8, S, S, exact, true, 16, 4, false, "", enabled_t::False, 1, {0},
    "testrootviews", false, 0, cfg);
  nnEval->spawnServerThreads();

  vector<Root> roots;
  {
    ifstream in(argv[2]);
    if(!in) {
      fprintf(stderr, "cannot read %s\n", argv[2]);
      return 2;
    }
    string line;
    while(getline(in, line)) {
      line = Global::trim(line);
      if(line.empty() || line[0] == '#')
        continue;
      const size_t space = line.find(' ');
      GameRules::Position p;
      string why;
      if(space == string::npos || !GameInterface::parseState<GameRules>(line.substr(space + 1), p, why)) {
        check(false, "a line of POSITIONS is not LABEL STATE: " + why + ": " + line.substr(0, 40));
        continue;
      }
      Root root;
      root.label = line.substr(0, space);
      root.board.setPosition(p);
      root.pla = root.board.nextPla();
      root.hist = BoardHistory(root.board, root.pla, Rules(), 0);
      root.views = RootViews::cover(root.board);
      check(root.views.size() > 1, root.label + ": a cover of several views");
      roots.push_back(root);
    }
  }
  check(!roots.empty(), "POSITIONS has positions");

  for(const Root& root : roots) {
    const int64_t n = (int64_t)root.views.size();
    const string& label = root.label;
    //The policy's move of the first pass's root, by its child position.
    const Board first = inView(root.board, root.views[0]);
    const Loc expected = policyMove(nnEval, first, root.hist, root.pla);
    check(expected != Board::NULL_LOC, label + ": the first view holds a legal move");
    const Hash128 expectedChild = childHash(first, expected);
    RootViews::Report report;
    report.perspective = P_WHITE;
    report.includePVVisits = true;

    //Few visits: every pass has 1 visit.
    for(int64_t visits : {(int64_t)1, (int64_t)2, (int64_t)3, 2 * n}) {
      const string where = label + " (" + Global::int64ToString(n) + " views), maxVisits " +
                           Global::int64ToString(visits);
      json ret;
      RootViews::Choice choice;
      const bool suc = searchViews(nnEval, logger, root, visits, report, ret, choice);
      if(print)
        printf("%s %s %s\n", where.c_str(), suc ? moveText(root, choice).c_str() : "none",
               suc ? ret.dump().c_str() : "");
      check(suc && choice.loc != Board::NULL_LOC, where + ": a response and a move");
      if(!suc || choice.loc == Board::NULL_LOC)
        continue;
      const Board chosen = inView(root.board, choice.view);
      check(root.hist.isLegal(chosen, choice.loc, root.pla), where + ": the move is legal in its view");
      check(childHash(chosen, choice.loc) == expectedChild, where + ": the move is the first pass's policy move");
      const json& infos = ret["moveInfos"];
      check(infos.size() == 1 && infos[0]["visits"] == 0 && infos[0]["edgeVisits"] == 0, where + ": one candidate, "
            "with no visits");
      check(infos.size() == 1 && infos[0]["pv"].size() == 1 && infos[0]["pv"][0] == infos[0]["move"] &&
              infos[0]["pvVisits"] == json::array({0}) && infos[0]["prior"].get<double>() > 0.0,
            where + ": its line is the move alone, with its prior");
      check(ret["checks"].empty() && ret["rootInfo"]["searches"] == n && ret["passes"].size() == (size_t)n,
            where + ": no check, and n passes");
      check(ret["rootInfo"]["visits"] == n, where + ": 1 visit a pass");
      const json& firstRoot = ret["passes"][0]["rootInfo"];
      for(const char* key : {"winrate", "scoreLead", "scoreSelfplay", "scoreStdev", "utility"})
        check(infos.size() == 1 && infos[0][key] == firstRoot[key] && ret["rootInfo"][key] == firstRoot[key],
              where + ": the first pass's root's " + key);
      check(infos.size() == 1 && infos[0]["lcb"] == firstRoot["winrate"], where + ": KataGo's bound of no visits");
    }

    //The root's ownership, with includeMovesOwnership.
    {
      RootViews::Report owned = report;
      owned.includeMovesOwnership = true;
      owned.includeMovesOwnershipStdev = true;
      json ret;
      RootViews::Choice choice;
      const bool suc = searchViews(nnEval, logger, root, 1, owned, ret, choice);
      const size_t points = (size_t)root.board.x_size * root.board.y_size;
      check(suc && ret["moveInfos"].size() == 1 && ret["moveInfos"][0]["ownership"].size() == points &&
              ret["moveInfos"][0]["ownershipStdev"].size() == points,
            label + ": with includeMovesOwnership, the candidate has an ownership");
    }

    //The searches with the manual interface of Search (Blueprints' engine protocol): each search's visits are
    //ceil(maxVisits x its share), as RootViews::search's searchFactor makes them.
    {
      const int64_t visits = 2;
      Search search(params(visits), nnEval, &logger, "testrootviews manual " + label);
      RootViews::Searches searches(root.board, root.hist, root.pla, root.views, RootViews::Settings(), report);
      json ret;
      RootViews::Choice choice;
      check(!searches.result(ret, choice), label + ": no response before a search");
      bool firstPlayout = true;
      int numSearches = 0;
      while(searches.next()) {
        numSearches++;
        search.setPosition(searches.pla(), searches.board(), searches.hist());
        search.beginSearch(false);
        SearchThread thread(0, search);
        const int64_t limit = (int64_t)std::ceil(visits * searches.share());
        int64_t done = search.getRootVisits();
        if(firstPlayout)
          check(!searches.result(ret, choice, &search), label + ": no response before the first playout");
        while(done < limit) {
          if(search.runSinglePlayout(thread, double(limit - done)))
            done++;
          if(firstPlayout) {
            firstPlayout = false;
            check(searches.result(ret, choice, &search) && choice.loc != Board::NULL_LOC &&
                    childHash(inView(root.board, choice.view), choice.loc) == expectedChild,
                  label + ": during the first pass, the policy move");
          }
        }
        search.endManualSearch(thread);
        check(searches.done(search, Board::NULL_LOC), label + ": each search gives an analysis");
      }
      check(numSearches == n, label + ": the passes alone");
      check(searches.result(ret, choice) && choice.loc != Board::NULL_LOC &&
              childHash(inView(root.board, choice.view), choice.loc) == expectedChild,
            label + ": at the end, the policy move");
    }

    //Above the threshold: the passes find candidates, and the checks run.
    for(int64_t visits : {2 * n + 1, 4 * n, 8 * n}) {
      const string where = label + " (" + Global::int64ToString(n) + " views), maxVisits " +
                           Global::int64ToString(visits);
      json ret;
      RootViews::Choice choice;
      const bool suc = searchViews(nnEval, logger, root, visits, report, ret, choice);
      if(print)
        printf("%s %s %s\n", where.c_str(), suc ? moveText(root, choice).c_str() : "none",
               suc ? ret.dump().c_str() : "");
      check(suc && choice.loc != Board::NULL_LOC, where + ": a response and a move");
      if(!suc || choice.loc == Board::NULL_LOC)
        continue;
      check(root.hist.isLegal(inView(root.board, choice.view), choice.loc, root.pla),
            where + ": the move is legal in its view");
      check(!ret["checks"].empty() && ret["moveInfos"][0]["visits"].get<int64_t>() > 0,
            where + ": candidates of the passes, and checks");
    }
    fprintf(stderr, "%s: %d views, the policy's move %s\n", label.c_str(), (int)n,
            Location::toString(expected, first).c_str());
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  if(failures > 0) {
    fprintf(stderr, "testrootviews: %d failures\n", failures);
    return 1;
  }
  fprintf(stderr, "testrootviews: OK\n");
  return 0;
}
