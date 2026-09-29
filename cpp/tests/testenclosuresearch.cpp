// Enclosure fork: the search under two-edge turns (plan step 3, ENCLOSURE.md "Turn order in the search").
//
//   katago runenclosuretests
//
// Uses KataGo's no-network evaluator (debugSkipNeuralNet): random policy logits and values, with exact
// results at the end of the game. Build with asserts on (a Debug build) so a move used as a point fails loudly.

#include "../tests/tests.h"

#include <set>

#include "../game/graphhash.h"
#include "../neuralnet/nninputs.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestSearchCommon;

namespace {

//Every node's player to move is the history's, every child's move is legal for it, and every child's player
//to move is the history's after the move (the 1-2-2 order, never alternation).
void checkTree(const SearchNode* node, Board board, BoardHistory hist, set<const SearchNode*>& seen, int& nodes, int depth) {
  if(!seen.insert(node).second || depth > 6)
    return;
  nodes++;
  testAssert(node->nextPla == hist.presumedNextMovePla);
  ConstSearchNodeChildrenReference children = node->getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLoc();
    testAssert(moveLoc >= Board::MOVE_LOC_BASE && moveLoc < Board::NUM_MOVE_LOCS);
    testAssert(hist.isLegal(board, moveLoc, node->nextPla));
    Board b = board;
    BoardHistory h = hist;
    h.makeBoardMoveAssumeLegal(b, moveLoc, node->nextPla, NULL);
    testAssert(child->nextPla == h.presumedNextMovePla);
    checkTree(child, b, h, seen, nodes, depth + 1);
  }
}

Loc mostVisitedChild(const Search* search, const SearchNode* node) {
  ConstSearchNodeChildrenReference children = node->getChildren();
  Loc best = Board::NULL_LOC;
  int64_t bestVisits = -1;
  for(int i = 0; i < children.getCapacity(); i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    int64_t visits = children[i].getEdgeVisits();
    if(visits > bestVisits) {
      bestVisits = visits;
      best = children[i].getMoveLoc();
    }
  }
  (void)search;
  return best;
}

int64_t childVisits(const SearchNode* node, Loc moveLoc) {
  ConstSearchNodeChildrenReference children = node->getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    if(children[i].getMoveLoc() == moveLoc)
      return child->stats.visits.load();
  }
  return 0;
}

Loc moveOf(int x, int y, int dx, int dy) {
  return Board::moveLocOfAction((y * enclosure::SIZE + x) * enclosure::NOFF + enclosure::OFF_INDEX[dx + 3][dy + 3]);
}

SearchParams testParams(int64_t maxVisits) {
  SearchParams params;
  params.maxVisits = maxVisits;
  params.useGraphSearch = true;
  params.subtreeValueBiasFactor = 0.30;  //As in self-play: exercises the edge patterns
  params.staticScoreUtilityFactor = 0.0;  //Win/loss alone decides (owner, option (b))
  params.dynamicScoreUtilityFactor = 0.0;
  return params;
}

}  // namespace

void Tests::runEnclosureSearchTests() {
  cout << "Running Enclosure search tests" << endl;
  NeuralNet::globalInitialize();
  Logger logger(nullptr, false, false, false);
  logger.addOStream(cout);
  NNEvaluator* nnEval = startNNEval("/dev/null", logger, "", 19, 19, 0, true, false, false, true, false);

  //1. Tree invariants at every point of the turn order: blue's single opening edge, a turn's first and
  //second edges, and the last edges of the game.
  {
    Rand rand("enclosure search tests: random games");
    set<int> stops = {0, 1, 2, 3, 30, 31, 116, 117, 118, 119};
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    for(int n = 0; n < enclosure::MAX_MOVES; n++) {
      if(stops.count(n)) {
        Search search(testParams(1500), nnEval, &logger, "enclosure invariants " + Global::intToString(n));
        Player pla = hist.presumedNextMovePla;
        search.setPosition(pla, board, hist);
        search.runWholeSearch(pla);
        set<const SearchNode*> seen;
        int nodes = 0;
        checkTree(search.rootNode, board, hist, seen, nodes, 0);
        //The principal variation replays with the history's players.
        vector<Loc> pv, scratchLocs;
        vector<int64_t> visitsBuf, edgeVisitsBuf;
        vector<double> scratchValues;
        search.appendPV(pv, visitsBuf, edgeVisitsBuf, scratchLocs, scratchValues, search.rootNode, 10);
        Board b = board;
        BoardHistory h = hist;
        for(Loc loc : pv) {
          testAssert(h.isLegal(b, loc, h.presumedNextMovePla));
          h.makeBoardMoveAssumeLegal(b, loc, h.presumedNextMovePla, NULL);
        }
        cout << "edge " << n + 1 << " (" << PlayerIO::playerToString(pla) << ", " << board.pos.moves_left
             << " left in the turn): " << nodes << " nodes checked, PV " << pv.size() << " moves" << endl;
      }
      vector<Loc> legal;
      board.getLegalMoveLocs(legal);
      Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
      hist.makeBoardMoveAssumeLegal(board, loc, hist.presumedNextMovePla, NULL);
    }
    testAssert(hist.isGameFinished);
  }

  //2. Transposed turns share a node: the same two edges in either order give the same graph hash.
  {
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    hist.makeBoardMoveAssumeLegal(board, moveOf(3, 9, 0, 3), P_BLACK, NULL);  //Blue's opening edge; red has two
    Loc a = moveOf(15, 9, 0, -3);
    Loc b = moveOf(18, 9, 0, 3);
    int repBound = SearchParams().graphSearchRepBound;
    auto hashAfter = [&](Loc first, Loc second) {
      Board bd = board;
      BoardHistory h = hist;
      Hash128 g = GraphHash::getGraphHashFromScratch(h, h.presumedNextMovePla, repBound, 0.5);
      testAssert(h.isLegal(bd, first, P_WHITE));
      h.makeBoardMoveAssumeLegal(bd, first, P_WHITE, NULL);
      g = GraphHash::getGraphHash(g, h, h.presumedNextMovePla, repBound, 0.5);
      testAssert(h.presumedNextMovePla == P_WHITE);
      testAssert(h.isLegal(bd, second, P_WHITE));
      h.makeBoardMoveAssumeLegal(bd, second, P_WHITE, NULL);
      g = GraphHash::getGraphHash(g, h, h.presumedNextMovePla, repBound, 0.5);
      testAssert(h.presumedNextMovePla == P_BLACK);
      return g;
    };
    testAssert(hashAfter(a, b) == hashAfter(b, a));
    testAssert(hashAfter(a, b) != hashAfter(a, moveOf(18, 9, 0, -3)));
    cout << "Transposed turns share a graph hash" << endl;
  }

  //3. Two edges, one mover. Red (P_WHITE) plays edges 118 and 119, then Blue's single edge 120 ends the game.
  //Red has two sides of a 3x3 square; only the other two sides together close it (area 9, scored after both
  //remaining turns: 18 against Blue's 16). Any other pair loses: the one-edge triangle gives 4.5, the other
  //quadrilaterals 7.5 (15 against 16). Blue's two edges are out of Red's reach (Red cannot leave Blue without a
  //move) and Blue cannot reach Red's old edges; Red's new ones are protected. Checked exhaustively on the
  //Enclosure repository's Python engine: 2 of Red's 45,601 two-edge turns win, all others lose.
  //A search that negated values across Red's own turn would avoid the win.
  {
    enclosure::Position p;
    p.edges[0] = {enclosure::make_edge(0, 16, 3, 16), enclosure::make_edge(0, 18, 3, 18)};
    p.edges[1].push_back(enclosure::make_edge(5, 5, 8, 5));
    p.edges[1].push_back(enclosure::make_edge(8, 5, 8, 8));
    p.scores[0] = 16.0;
    p.scores[1] = 0.0;
    p.player = 1;
    p.moves_left = 2;
    p.move_count = 117;
    p.area_ok[0] = p.area_ok[1] = false;
    Board board;
    board.setPosition(p);
    BoardHistory hist(board, P_WHITE, Rules(), 0);
    testAssert(hist.presumedNextMovePla == P_WHITE);

    auto closesSquare = [](const Board& b) {
      return enclosure::contains(b.pos.edges[1], enclosure::make_edge(5, 8, 8, 8)) &&
             enclosure::contains(b.pos.edges[1], enclosure::make_edge(5, 5, 5, 8));
    };

    //With random priors, finding 1 winning pair of ~40,000 by visits alone is luck. So the root considers only
    //20 first edges (KataGo's allowMoves): both winning ones and 18 losing ones, among them the one-edge
    //triangle. Red's second edge, the node under test, keeps every legal move.
    vector<Loc> legalFirst;
    board.getLegalMoveLocs(legalFirst);
    vector<Loc> allowed = {moveOf(8, 8, -3, 0), moveOf(5, 5, 0, 3), moveOf(5, 5, 3, 3)};
    for(Loc loc : legalFirst)
      if(allowed.size() < 20 && std::find(allowed.begin(), allowed.end(), loc) == allowed.end())
        allowed.push_back(loc);
    for(Loc loc : allowed)
      testAssert(hist.isLegal(board, loc, P_WHITE));
    vector<int> avoidWhite(Board::NUM_MOVE_LOCS, 1);  //Banned at the root only (depth 1)
    for(Loc loc : allowed)
      avoidWhite[loc] = 0;

    //The random priors are noise here; flattened, the search tries every second edge instead of waiting for
    //thousands of visits before a low-prior one beats first-play urgency (every tried line is a loss).
    SearchParams params = testParams(100000);
    params.nnPolicyTemperature = 100.0f;
    Search search(params, nnEval, &logger, "enclosure two edges one mover");
    search.setPosition(P_WHITE, board, hist);
    search.setAvoidMoveUntilByLoc(vector<int>(), avoidWhite);
    search.runWholeSearch(P_WHITE);
    ReportedSearchValues values = search.getRootValuesRequireSuccess();
    Loc first = mostVisitedChild(&search, search.rootNode);
    Board b1 = board;
    BoardHistory h1 = hist;
    h1.makeBoardMoveAssumeLegal(b1, first, P_WHITE, NULL);
    cout << "Red's first edge " << Location::toString(first, board) << ", red win value " << values.winValue << endl;
    testAssert(enclosure::contains(b1.pos.edges[1], enclosure::make_edge(5, 8, 8, 8)) ||
               enclosure::contains(b1.pos.edges[1], enclosure::make_edge(5, 5, 5, 8)));
    testAssert(values.winValue > 0.9);  //winValue is White's, and Red is White

    //Tree reuse between Red's two edges: the root becomes that child, still Red to move, with its visits.
    int64_t visitsBefore = childVisits(search.rootNode, first);
    testAssert(search.makeMove(first, P_WHITE));
    testAssert(search.rootNode != NULL);
    testAssert(search.rootNode->nextPla == P_WHITE);
    testAssert(search.rootNode->stats.visits.load() >= visitsBefore);
    cout << "Tree reuse: " << search.rootNode->stats.visits.load() << " visits kept (child had " << visitsBefore << ")" << endl;

    //Red's second edge: every legal move (this clears the tree; its reuse was checked above).
    search.setAvoidMoveUntilByLoc(vector<int>(), vector<int>());
    search.runWholeSearch(P_WHITE);
    Loc second = mostVisitedChild(&search, search.rootNode);
    Board b2 = b1;
    BoardHistory h2 = h1;
    h2.makeBoardMoveAssumeLegal(b2, second, P_WHITE, NULL);
    ReportedSearchValues values2 = search.getRootValuesRequireSuccess();
    cout << "Red's second edge " << Location::toString(second, b1) << ", red win value " << values2.winValue << endl;
    testAssert(closesSquare(b2));
    testAssert(h2.presumedNextMovePla == P_BLACK);
    testAssert(values2.winValue > 0.9);
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  cout << "Enclosure search tests passed" << endl;
}
