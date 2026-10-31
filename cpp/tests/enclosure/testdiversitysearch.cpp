// KataGame fork (2026-10-29): AZdb's diversity term in the search (Search::diversityWeight, program/azdb.h), on the
// build's game, which must give AZdb's statistics (GameInterface: Diversity, as Blueprints does). The engine
// repository's games/blueprints/docs/research/diversity-players.md, "The search". The checks:
//   - member 0 with the term's settings (λ and a state) searches as the search without them: the same visits, values
//     and moves (w = 0);
//   - D on chains, where each depth allows one move, so that the searches with and without the term have the same
//     visits: at each node, the utility with the term less the utility without it is w s_P times the node's reward,
//     plus the average of w s_P times its head's return and its child's difference. The rewards are those of P's moves
//     only, and equal the rows' rewards of the same moves (AZdb::rowTargets). The head's return is output A where P
//     moves and output B where the opponent moves, and 0 at the game's end. The root has no reward. For P = each
//     player;
//   - a transposition: P's two moves in both orders (Blueprints: two edits, which change the hands only) reach one node
//     of the search's graph through other nodes, whose rewards differ. Each parent reads that node, and its value holds
//     its own reward once and no earlier one: KataGo's utility of its evaluation plus w s_P times its reward and its
//     head's return;
//   - on the game's last move, where every move ends the game, a member with w > 0 prefers the move with the larger
//     reward to the move that the search without the term prefers, and with the reward's sign turned the other move;
//   - a search of member 3 after a search of member 2 on the same evaluator reads member 3's evaluations only;
//   - self-play's kind of search with the term (every weighting that reads the utilities): at each node of its graph,
//     the node constant is w s_P times the reward of the move into it, and the values are finite;
//   - the reward's scale k = 100 (State's rewardScale) multiplies the rows' rewards and targets (AZdb::rowTargets) and
//     the search's node rewards alike: k times those of k = 1. On a chain, D holds with the scaled rewards;
//   - the refusals (no state, a model without the head, the eval cache), and the tree's clear when P changes.
//
//   testdiversitysearch MODEL MODEL_WITHOUT_HEAD
//
// MODEL is a network of the build's game of the inputs version 14 with the intrinsic value head, and
// MODEL_WITHOUT_HEAD one of the same version without it. Random networks do (python/enclosure_nn_check.py make
// -game GAME -inputs-version 14 [-intrinsic-head], then export_model_pytorch.py). The test builds with
// tests/enclosure/build.sh BUILD_DIR, with TESTS=testdiversitysearch, on a backend that computes the head (Eigen or
// CUDA). Its positions come from random games with fixed seeds, and each search runs on one thread. Exit code 0 when
// every check holds.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../core/rand.h"
#include "../../core/test.h"
#include "../../game/graphhash.h"
#include "../../neuralnet/nneval.h"
#include "../../program/azdb.h"
#include "../../search/search.h"
#include "../../search/searchnode.h"

using namespace std;
using State = AZdb::State<GameEncoding>;

static int failures = 0;

static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 30)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

static bool near(double a, double b, double tolerance = 1e-9) {
  return std::fabs(a - b) <= tolerance * std::max(1.0, std::fabs(a) + std::fabs(b));
}

//A position of a game.
struct Position {
  Board board;
  BoardHistory hist;
  Player pla() const { return hist.presumedNextMovePla; }
  bool legal(Loc loc) const { return hist.isLegal(board, loc, pla()); }
  Position after(Loc loc) const {
    Position p = *this;
    p.hist.makeBoardMoveAssumeLegal(p.board, loc, pla(), NULL);
    return p;
  }
  //The action that led here, and whether it ends a move (not a pick that waits for its drop).
  Loc lastLoc() const { return hist.moveHistory.back().loc; }
  Player lastPla() const { return hist.moveHistory.back().pla; }
  bool endsMove() const { return !GameInterface::continuesMove<GameRules>(board.pos); }
};

//A random game from the start of the game's standard board to its end, the position after each action.
static vector<Position> randomGame(Rand& rand) {
  Board board;
  vector<Position> game = {Position{board, BoardHistory(board, P_BLACK, Rules(), 0)}};
  vector<Loc> legal;
  while(!game.back().hist.isGameFinished) {
    game.back().board.getLegalMoveLocs(legal);
    testAssert(!legal.empty());
    game.push_back(game.back().after(legal[rand.nextUInt((uint32_t)legal.size())]));
  }
  return game;
}

//Each statistic's mean and standard deviation over the game's positions, from both sides: a normalizer that keeps
//most of φ inside its clip.
static void normalization(const vector<Position>& game, vector<double>& mean, vector<double>& sd) {
  const int N = State::N;
  vector<double> x(N), sum(N, 0.0), squares(N, 0.0);
  int count = 0;
  for(const Position& p: game) {
    for(int player = 0; player < 2; player++) {
      GameInterface::diversityStatisticsOf<GameEncoding>(p.board.pos, player, x.data());
      for(int s = 0; s < N; s++) {
        sum[s] += x[s];
        squares[s] += x[s] * x[s];
      }
      count++;
    }
  }
  mean.assign(N, 0.0);
  sd.assign(N, 0.0);
  for(int s = 0; s < N; s++) {
    mean[s] = sum[s] / count;
    sd[s] = std::sqrt(std::max(0.0, squares[s] / count - mean[s] * mean[s]));
  }
}

//The test's member, with w > 0.
static const int MEMBER = 2;

//w = (1 - λ) / λ, and w s_P: the weight of the term with the sign of P in White's frame.
static double weightOf(double lambda) { return (1.0 - lambda) / lambda; }
static double signedWeight(Player P, double lambda) { return P == P_WHITE ? weightOf(lambda) : -weightOf(lambda); }

//A state where MEMBER's ψ is v and every other member's is 0, with ℓ0 = 2 |v| and the reward's scale k. MEMBER's
//nearest member is then member 0 (the lowest of equal distances), so its reward is k (1 - 1/8) / MOVES φ · v.
static shared_ptr<const State> memberState(
  const vector<double>& v, const vector<double>& mean, const vector<double>& sd, double k = 1.0
) {
  const int N = State::N;
  vector<double> psi((size_t)AZdb::NUM_MEMBERS * N, 0.0);
  double norm2 = 0.0;
  for(int s = 0; s < N; s++) {
    psi[(size_t)MEMBER * N + s] = v[s];
    norm2 += v[s] * v[s];
  }
  return make_shared<const State>(psi.data(), mean.data(), sd.data(), 2.0 * std::sqrt(norm2), k);
}

//φ of the move into p from the side of its mover, when it ends a move.
static bool phiOfMove(const State& state, const Position& p, vector<double>& phi) {
  vector<double> x(State::N);
  phi.assign(State::N, 0.0);
  return state.phiOfMove(p.board.pos, Board::indexOfPlayer(p.lastPla()), x.data(), phi.data());
}

//The member's reward of the move into p, as the search counts it for P: P's moves that end a move, else 0.
static double rewardFor(const State& state, int member, const Position& p, Player P) {
  vector<double> phi;
  if(p.hist.moveHistory.empty() || p.lastPla() != P || !phiOfMove(state, p, phi))
    return 0.0;
  return state.reward(member, phi.data());
}

static SearchParams params(int64_t visits, int member, double lambda) {
  SearchParams p;
  p.maxVisits = visits;
  p.numThreads = 1;
  p.useGraphSearch = true;
  p.valueWeightExponent = 0.0;  //plain averages, for the chains' arithmetic
  p.diversityMember = member;
  p.diversityLambda = lambda;
  return p;
}

//Self-play's kind of search (configs/training/selfplay_blueprints.cfg): every weighting that reads the utilities.
static SearchParams selfplayParams(int64_t visits, int member, double lambda) {
  SearchParams p = params(visits, member, lambda);
  p.dynamicScoreUtilityFactor = 0.4;
  p.cpuctUtilityStdevScale = 0.85;
  p.valueWeightExponent = 0.5;
  p.useNoisePruning = true;
  p.subtreeValueBiasFactor = 0.3;
  p.subtreeValueBiasWeightExponent = 0.8;
  p.fpuParentWeightByVisitedPolicy = true;
  p.fpuParentWeightByVisitedPolicyPow = 2.0;
  p.useUncertainty = true;
  return p;
}

//The moves that a search may play: allowed[i] = (move, the first depth from the root that may play it). The search
//avoids every other move (Search::setAvoidMoveUntilByLoc), and its root policy is rescaled over the root's allowed
//moves, so that each gets visits.
static vector<int> onlyMoves(const vector<pair<Loc, int>>& allowed) {
  vector<int> until(Board::NUM_MOVE_LOCS, 1 << 20);
  for(const pair<Loc, int>& a: allowed)
    until[a.first] = a.second;
  return until;
}

//The moves of a chain along line: line[j + 1]'s action at depth j.
static vector<pair<Loc, int>> chainMoves(const vector<Position>& line) {
  vector<pair<Loc, int>> allowed;
  for(size_t j = 0; j + 1 < line.size(); j++)
    allowed.push_back({line[j + 1].lastLoc(), (int)j});
  return allowed;
}

//Whether a search that may play the moves of allowed has, from depth 1 on along line, line[j + 1]'s action as its only
//legal move at each depth j, and none at line's end: so that its nodes from depth 1 on form a chain. A game's actions
//can name other moves at other positions (Blueprints' actions are in the mover's view), so an action of an earlier
//depth can be legal again.
static bool chainBelow(const vector<Position>& line, const vector<pair<Loc, int>>& allowed) {
  for(size_t j = 1; j < line.size(); j++) {
    const bool end = j + 1 == line.size();
    if(line[j].hist.isGameFinished)
      return end;
    int legal = 0;
    bool next = false;
    for(const pair<Loc, int>& a: allowed)
      if(a.second <= (int)j && line[j].legal(a.first)) {
        legal++;
        next = next || (!end && a.first == line[j + 1].lastLoc());
      }
    if(legal != (end ? 0 : 1) || (!end && !next))
      return false;
  }
  return true;
}

//untilOther, when given, is the avoided moves of the root's opponent, and until those of the root's player.
static unique_ptr<Search> newSearch(
  NNEvaluator* nnEval, Logger& logger, const SearchParams& p, const Position& root,
  const shared_ptr<const State>& state,
  const vector<int>& until, const vector<int>& untilOther = vector<int>()
) {
  unique_ptr<Search> search = make_unique<Search>(p, nnEval, &logger, "testdiversitysearch");
  search->setPosition(root.pla(), root.board, root.hist);
  search->setDiversityState(state);
  if(!until.empty()) {
    const vector<int>& other = untilOther.empty() ? until : untilOther;
    search->setAvoidMoveUntilByLoc(root.pla() == P_BLACK ? until : other, root.pla() == P_BLACK ? other : until);
    search->setAvoidMoveUntilRescaleRoot(true);
  }
  return search;
}

static unique_ptr<Search> runSearch(
  NNEvaluator* nnEval, Logger& logger, const SearchParams& p, const Position& root,
  const shared_ptr<const State>& state,
  const vector<int>& until = vector<int>(), const vector<int>& untilOther = vector<int>()
) {
  unique_ptr<Search> search = newSearch(nnEval, logger, p, root, state, until, untilOther);
  search->runWholeSearch(root.pla());
  return search;
}

struct Child {
  const SearchNode* node;
  Loc loc;
  int64_t edgeVisits;
};

static vector<Child> childrenOf(const SearchNode& node) {
  vector<Child> out;
  ConstSearchNodeChildrenReference children = node.getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    out.push_back({child, children[i].getMoveLoc(), children[i].getEdgeVisits()});
  }
  return out;
}

static double utilityOf(const SearchNode& node) { return node.stats.utilityAvg.load(); }
static int64_t visitsOf(const SearchNode& node) { return node.stats.visits.load(); }

//Each node of the search's graph once, with its position.
static void forEachNode(const Search& search, const function<void(const SearchNode&, const Position&)>& f) {
  set<const SearchNode*> seen;
  vector<pair<const SearchNode*, Position>> stack = {{search.rootNode, Position{search.rootBoard, search.rootHistory}}};
  while(!stack.empty()) {
    pair<const SearchNode*, Position> top = std::move(stack.back());
    stack.pop_back();
    if(!seen.insert(top.first).second)
      continue;
    f(*top.first, top.second);
    for(const Child& c: childrenOf(*top.first))
      stack.push_back({c.node, top.second.after(c.loc)});
  }
}

static string where(const string& label, int depth) { return label + " depth " + Global::intToString(depth); }

//D on a chain: line[0] is the root and line[j] the position after j actions (chainBelow holds). Depth j allows only
//action j, so that each node has one child, and the searches with and without the term (λ = 1, the same member, so
//the same evaluations) have the same visits. At the chain's end, a node that does not end the game repeats its own
//evaluation.
static void chainTest(
  NNEvaluator* nnEval, Logger& logger, const vector<Position>& line, int member, double lambda,
  const shared_ptr<const State>& state, int64_t visits, const string& label
) {
  const int k = (int)line.size() - 1;
  const vector<pair<Loc, int>> allowed = chainMoves(line);
  testAssert(k >= 1 && chainBelow(line, allowed));
  const vector<int> until = onlyMoves(allowed);
  unique_ptr<Search> with = runSearch(nnEval, logger, params(visits, member, lambda), line[0], state, until);
  unique_ptr<Search> without = runSearch(nnEval, logger, params(visits, member, 1.0), line[0], state, until);

  const Player P = line[0].pla();
  const double wP = signedWeight(P, lambda);
  check(with->diversityWeight == weightOf(lambda) && without->diversityWeight == 0.0, label + ": the weights");

  //The nodes of the chains, the root first.
  vector<const SearchNode*> nodes = {with->rootNode}, nodes0 = {without->rootNode};
  for(int j = 0; j < k; j++) {
    vector<Child> c = childrenOf(*nodes[j]), c0 = childrenOf(*nodes0[j]);
    if(c.size() != 1 || c0.size() != 1 || c[0].loc != allowed[j].first || c0[0].loc != allowed[j].first ||
       c[0].edgeVisits != visitsOf(*c[0].node)) {
      check(false, where(label + ": not a chain at", j));
      return;
    }
    nodes.push_back(c[0].node);
    nodes0.push_back(c0[0].node);
  }
  check(childrenOf(*nodes[k]).empty() && visitsOf(*nodes[k]) >= 1, label + ": the end is a leaf with visits");

  //The rewards: the search's node constants are the rows' rewards of the same moves (rowTargets: ownReward), as
  //members of the same member on both sides.
  vector<double> reward(k + 1, 0.0);
  for(int j = 1; j <= k; j++)
    reward[j] = rewardFor(*state, member, line[j], P);
  AZdb::Targets targets;
  const int members[2] = {member, member};
  AZdb::rowTargets<GameEncoding>(
    [&](int j) -> const GameRules::Position& { return line[j].board.pos; }, k, line[k].hist.isGameFinished, members,
    *state, NULL, NULL, targets);
  int rewarded = 0;
  for(int j = 1; j <= k; j++) {
    const double rowReward = line[j].lastPla() == P ? targets.ownReward[j - 1] : 0.0;
    check(near(reward[j], rowReward, 1e-6), where(label + ": the row's reward at", j));
    check(near(nodes[j]->diversityReward, wP * reward[j]), where(label + ": the node's reward at", j));
    check(nodes0[j]->diversityReward == 0.0, where(label + ": no reward without the term at", j));
    if(reward[j] != 0.0)
      rewarded++;
  }
  check(nodes[0]->diversityReward == 0.0, label + ": the root has no reward");
  check(rewarded > 0, label + ": some move has a reward");

  //D from the end back to the root.
  auto headOf = [&](int j) {
    const NNOutput* nnOutput = nodes[j]->getNNOutput();
    check(nnOutput != NULL && nnOutput->hasIntrinsic, where(label + ": the head's outputs at", j));
    if(nnOutput == NULL)
      return 0.0;
    return (double)(nodes[j]->nextPla == P ? nnOutput->intrinsicA : nnOutput->intrinsicB);
  };
  vector<double> expected(k + 1);
  expected[k] = wP * reward[k] + (line[k].hist.isGameFinished ? 0.0 : wP * headOf(k));
  for(int j = k - 1; j >= 0; j--) {
    const SearchNode& child = *nodes[j + 1];
    const double childWeight = child.stats.getChildWeight(visitsOf(child));
    expected[j] = wP * reward[j] + (wP * headOf(j) + childWeight * expected[j + 1]) / (1.0 + childWeight);
  }
  for(int j = 0; j <= k; j++) {
    check(visitsOf(*nodes[j]) == visitsOf(*nodes0[j]), where(label + ": the same visits at", j));
    const double measured = utilityOf(*nodes[j]) - utilityOf(*nodes0[j]);
    check(near(measured, expected[j]), where(label + ": D at", j) + " (" + Global::doubleToString(measured) + " vs " +
          Global::doubleToString(expected[j]) + ")");
  }
}

//The positions of a game from its action from to its end.
static vector<Position> lineOf(const vector<Position>& game, int from) {
  return vector<Position>(game.begin() + from, game.end());
}

//The key of the search's graph for a position: the game's repetition bound is always exceeded
//(Board::simpleRepetitionBoundGt), so GraphHash::getGraphHash keys each node by its state alone.
static Hash128 graphKey(const Position& p) {
  return GraphHash::getStateHash(p.hist, p.pla(), SearchParams().drawEquivalentWinsForWhite);
}

//The moves at p that the search can play: those of the network's policy. A game's encoding can keep one of several
//actions that give one position (Blueprints' does), though the rules allow each.
static vector<Loc> searchableMoves(NNEvaluator* nnEval, const Position& p) {
  NNResultBuf buf;
  nnEval->evaluate(p.board, p.hist, p.pla(), MiscNNInputParams(), buf, false, false);
  vector<Loc> legal, out;
  p.board.getLegalMoveLocs(legal);
  for(Loc loc: legal)
    if(buf.result->policy()[buf.result->getPos(loc, p.board)] >= 0.0f)
      out.push_back(loc);
  return out;
}

static bool searchable(NNEvaluator* nnEval, const Position& p, Loc loc) {
  const vector<Loc> moves = searchableMoves(nnEval, p);
  return std::find(moves.begin(), moves.end(), loc) != moves.end();
}

//A random move at p of two actions that the search can play: one that does not end a move, then one that ends it
//(Blueprints: an edit, a pick and its drop). Empty when none turns up.
static vector<Loc> twoActionMove(NNEvaluator* nnEval, const Position& p, Rand& rand) {
  vector<Loc> firsts = searchableMoves(nnEval, p);
  for(int tries = 0; tries < 300 && !firsts.empty(); tries++) {
    const size_t i = rand.nextUInt((uint32_t)firsts.size());
    const Loc first = firsts[i];
    firsts[i] = firsts.back();
    firsts.pop_back();
    const Position q = p.after(first);
    if(q.endsMove() || q.hist.isGameFinished)
      continue;
    const vector<Loc> seconds = searchableMoves(nnEval, q);
    for(int tries2 = 0; tries2 < 20 && !seconds.empty(); tries2++) {
      const Loc second = seconds[rand.nextUInt((uint32_t)seconds.size())];
      if(q.after(second).endsMove())
        return {first, second};
    }
  }
  return vector<Loc>();
}

//The positions along actions from p, or none when one of them is not legal.
static vector<Position> play(const Position& p, const vector<Loc>& actions) {
  vector<Position> line = {p};
  for(Loc loc: actions) {
    if(line.back().hist.isGameFinished || !line.back().legal(loc))
      return vector<Position>();
    line.push_back(line.back().after(loc));
  }
  return line;
}

//A transposition: a position R of the game where P's two moves of two actions each (twoActionMove), m1 then m2 and m2
//then m1, reach one node of the search's graph through other nodes (Blueprints: two edits of other blueprints, which
//change the hands only). one and two are the positions along both orders, whose every action the search can play.
//False when the game has none.
static bool commutingMoves(
  NNEvaluator* nnEval, const vector<Position>& game, Rand& rand, vector<Position>& one, vector<Position>& two
) {
  for(size_t i = 10; i + 10 < game.size(); i++) {
    const Position& R = game[i];
    if(!R.endsMove())
      continue;
    const Player P = R.pla();
    for(int tries = 0; tries < 5; tries++) {
      const vector<Loc> m1 = twoActionMove(nnEval, R, rand);
      if(m1.empty())
        break;
      const vector<Position> after1 = play(R, m1);
      if(after1.back().pla() != P || after1.back().hist.isGameFinished)
        continue;
      const vector<Loc> m2 = twoActionMove(nnEval, after1.back(), rand);
      if(m2.empty())
        continue;
      vector<Loc> order1 = m1, order2 = m2;
      order1.insert(order1.end(), m2.begin(), m2.end());
      order2.insert(order2.end(), m1.begin(), m1.end());
      one = play(R, order1);
      two = play(R, order2);
      if(one.size() != 5 || two.size() != 5 || two[1].endsMove() || !two[2].endsMove() || two[3].endsMove() ||
         !two[4].endsMove() || two[4].hist.isGameFinished || graphKey(one[4]) != graphKey(two[4]) ||
         graphKey(one[2]) == graphKey(two[2]))
        continue;
      bool all = true;
      for(int j = 0; j < 4 && all; j++)
        all = searchable(nnEval, one[j], one[j + 1].lastLoc()) && searchable(nnEval, two[j], two[j + 1].lastLoc());
      if(all)
        return true;
    }
  }
  return false;
}

//The utility of p's network evaluation alone, as the search without the term counts a leaf: a search of one visit at p.
static double nnUtility(NNEvaluator* nnEval, Logger& logger, const Position& p, int member) {
  return utilityOf(*runSearch(nnEval, logger, params(1, member, 1.0), p, nullptr)->rootNode);
}

int main(int argc, char** argv) {
  if(argc != 3) {
    fprintf(stderr, "usage: testdiversitysearch MODEL MODEL_WITHOUT_HEAD\n");
    return 2;
  }
  Board::initHash();
  ScoreValue::initTables();
  NeuralNet::globalInitialize();
  if(State::N == 0) {
    fprintf(stderr, "the build's game gives no statistics for AZdb (GameInterface: Diversity)\n");
    return 2;
  }

  //NHWC inputs, which every backend takes (Eigen takes no other).
  map<string, string> kvs = {{"cudaUseNHWC", "true"}};
  ConfigParser cfg(kvs);
  Logger logger(nullptr, false, true, false, false);
  const int S = NNPos::MAX_BOARD_LEN;
  auto newEvaluator = [&](const string& file) {
    const bool exact = NNPos::fillsNNBoard(Board::DEFAULT_X_SIZE, Board::DEFAULT_Y_SIZE);
    NNEvaluator* nnEval = new NNEvaluator(
      "test", file, "", &logger, 8, S, S, exact, true, 16, 4, false, "", enabled_t::False, 1, {0},
      "testdiversitysearch", false, 0, cfg);
    nnEval->spawnServerThreads();
    return nnEval;
  };
  NNEvaluator* nnEval = newEvaluator(argv[1]);
  NNEvaluator* plainEval = newEvaluator(argv[2]);
  if(!nnEval->computesIntrinsicHead() || plainEval->hasIntrinsicHead()) {
    fprintf(stderr, "MODEL must have the intrinsic value head on a backend that computes it, MODEL_WITHOUT_HEAD not\n");
    return 2;
  }

  //A random game whose last 5 actions are whole moves (no edit) and make chains, so that the chains of its last 3 and 5
  //actions start with each player, and which has a transposition (commutingMoves).
  vector<Position> game, one, two;
  int seed = 0;
  bool found = false;
  for(; seed < 50 && !found; seed++) {
    Rand rand("testdiversitysearch game " + Global::intToString(seed));
    game = randomGame(rand);
    const int G = (int)game.size() - 1;
    if(G < 16)
      continue;
    bool whole = true;
    for(int j = G - 4; j <= G; j++)
      whole = whole && game[j].endsMove();
    const vector<Position> last3 = lineOf(game, G - 3), last5 = lineOf(game, G - 5);
    found = whole && last3[0].pla() != last5[0].pla() && chainBelow(last3, chainMoves(last3)) &&
      chainBelow(last5, chainMoves(last5)) && commutingMoves(nnEval, game, rand, one, two);
  }
  if(!found) {
    fprintf(stderr, "no random game of 50 has the chains and the transposition that the test needs\n");
    return 1;
  }
  const int G = (int)game.size() - 1;
  printf("game: seed %d, %d actions\n", seed - 1, G);

  vector<double> mean, sd;
  normalization(game, mean, sd);
  vector<double> v(State::N);
  for(int s = 0; s < State::N; s++)
    v[s] = std::sin(1.0 + 0.7 * s);
  const shared_ptr<const State> state = memberState(v, mean, sd);
  const double lambda = 0.9;
  const int member = MEMBER;

  //Member 0 with the term's settings searches as KataGo's search (no member, no state).
  {
    int compared = 0;
    for(int at: {G / 8, G / 2, G - 12}) {
      const Position& root = game[at];
      unique_ptr<Search> kataGo = runSearch(nnEval, logger, selfplayParams(200, 0, 1.0), root, nullptr);
      unique_ptr<Search> member0 = runSearch(nnEval, logger, selfplayParams(200, 0, lambda), root, state);
      check(member0->diversityWeight == 0.0, "member 0 has w = 0");
      vector<Child> c = childrenOf(*kataGo->rootNode), c0 = childrenOf(*member0->rootNode);
      check(c.size() == c0.size() && c.size() > 1, "member 0: the same children");
      for(size_t i = 0; i < c.size() && i < c0.size(); i++) {
        const string ofChild = " at child " + Global::intToString((int)i);
        check(c[i].loc == c0[i].loc && c[i].edgeVisits == c0[i].edgeVisits &&
              visitsOf(*c[i].node) == visitsOf(*c0[i].node), "member 0: the same visits" + ofChild);
        check(utilityOf(*c[i].node) == utilityOf(*c0[i].node), "member 0: the same utility" + ofChild);
      }
      check(utilityOf(*kataGo->rootNode) == utilityOf(*member0->rootNode), "member 0: the same root utility");
      check(kataGo->getChosenMoveLoc() == member0->getChosenMoveLoc(), "member 0: the same move");
      compared += (int)c.size();
    }
    printf("member 0: 3 positions, %d root children with the same visits, utilities and moves\n", compared);
  }

  //D on chains to the game's end, for P = each player: its last 3 and its last 5 actions.
  {
    const Player Pa = game[G - 3].pla(), Pb = game[G - 5].pla();
    chainTest(nnEval, logger, lineOf(game, G - 3), member, lambda, state, 8, "chain of 3");
    chainTest(nnEval, logger, lineOf(game, G - 5), member, lambda, state, 12, "chain of 5");
    printf("chains: D at each node, with P %s on the chain of 3 and %s on the chain of 5\n",
           PlayerIO::playerToString(Pa).c_str(), PlayerIO::playerToString(Pb).c_str());
  }

  //A transposition: P's moves m1 then m2, and m2 then m1, reach one node N. The search may play only their actions,
  //from the root on, and the opponent nothing, so that N is a leaf.
  {
    const Position& R = one[0];
    const Player P = R.pla();
    vector<pair<Loc, int>> allowed;
    for(int j = 1; j <= 4; j++) {
      allowed.push_back({one[j].lastLoc(), 0});
      allowed.push_back({two[j].lastLoc(), 0});
    }
    //A flat root policy, so that both orders get visits whatever their priors.
    SearchParams flat = params(60, member, lambda);
    flat.rootPolicyTemperature = flat.rootPolicyTemperatureEarly = 1000.0;
    flat.rootFpuReductionMax = 0.0;
    unique_ptr<Search> graph = runSearch(nnEval, logger, flat, R, state, onlyMoves(allowed), onlyMoves({}));
    //The nodes along each order, NULL from where the search did not go, and how deep it went.
    auto follow = [&](const vector<Position>& line, int& depth) {
      vector<const SearchNode*> nodes = {graph->rootNode};
      depth = 0;
      for(int j = 1; j <= 4; j++) {
        const SearchNode* next = NULL;
        if(nodes.back() != NULL)
          for(const Child& c: childrenOf(*nodes.back()))
            if(c.loc == line[j].lastLoc())
              next = c.node;
        nodes.push_back(next);
        if(next != NULL)
          depth = j;
      }
      return nodes;
    };
    int depth1, depth2;
    const vector<const SearchNode*> nodes1 = follow(one, depth1), nodes2 = follow(two, depth2);
    const bool shared = nodes1[4] != NULL && nodes1[4] == nodes2[4];
    check(shared, "transposition: both orders reach one node (depths " + Global::intToString(depth1) + " and " +
          Global::intToString(depth2) + " of 4, root children " +
          Global::intToString((int)childrenOf(*graph->rootNode).size()) + ")");
    for(int j = 1; j < 4; j++)
      check(nodes1[j] != NULL && nodes2[j] != NULL && nodes1[j] != nodes2[j],
            where("transposition: other nodes at", j));
    const double wP = signedWeight(P, lambda);
    double sum1 = 0.0, sum2 = 0.0;
    for(int j = 1; j <= 4; j++) {
      const double r1 = rewardFor(*state, member, one[j], P), r2 = rewardFor(*state, member, two[j], P);
      sum1 += r1;
      sum2 += r2;
      if(nodes1[j] != NULL && nodes2[j] != NULL) {
        check(near(nodes1[j]->diversityReward, wP * r1), where("transposition: a reward on the first order at", j));
        check(near(nodes2[j]->diversityReward, wP * r2), where("transposition: a reward on the second order at", j));
      }
    }
    check(sum1 != sum2, "transposition: the orders' sums of rewards differ");
    if(shared) {
      //N's value holds its own reward once and none of either order's earlier rewards: KataGo's utility of its
      //evaluation, plus w s_P times its reward and the head's return after it (output B: the opponent moves).
      const SearchNode& N = *nodes1[4];
      const NNOutput* out = N.getNNOutput();
      const double rN = rewardFor(*state, member, one[4], P);
      check(childrenOf(N).empty() && out != NULL && N.nextPla != P && rN != 0.0, "transposition: the node is a leaf");
      if(out != NULL) {
        const double expected = nnUtility(nnEval, logger, one[4], member) + wP * (rN + out->intrinsicB);
        check(near(utilityOf(N), expected), "transposition: the node's value (" + Global::doubleToString(utilityOf(N)) +
              " vs " + Global::doubleToString(expected) + ")");
      }
      printf("transposition: at action %d, two orders of P's moves meet in one node of %lld visits; the orders' "
             "rewards before it sum to %.6f and %.6f\n", (int)R.hist.moveHistory.size(), (long long)visitsOf(N),
             sum1 - rN, sum2 - rN);
    }
  }

  //The game's last move: each move that ends it is a leaf, whose utility is KataGo's plus w s_P times its reward.
  {
    const Position& root = game[G - 1];
    const Player P = root.pla();
    const double wP = signedWeight(P, lambda);
    vector<Loc> legal;
    root.board.getLegalMoveLocs(legal);
    vector<Loc> moves;
    vector<vector<double>> phis;
    for(Loc loc: legal) {
      const Position p = root.after(loc);
      vector<double> phi;
      if(p.hist.isGameFinished && phiOfMove(*state, p, phi)) {
        moves.push_back(loc);
        phis.push_back(phi);
      }
      if(moves.size() >= 200)
        break;
    }
    //The two moves whose φ differ most.
    size_t i1 = 0, i2 = 0;
    double most = -1.0;
    for(size_t i = 0; i < moves.size(); i++)
      for(size_t j = i + 1; j < moves.size(); j++) {
        double d2 = 0.0;
        for(int s = 0; s < State::N; s++)
          d2 += (phis[i][s] - phis[j][s]) * (phis[i][s] - phis[j][s]);
        if(d2 > most) {
          most = d2;
          i1 = i;
          i2 = j;
        }
      }
    check(moves.size() >= 2 && most > 0.0, "last move: two moves with different φ");
    if(moves.size() >= 2 && most > 0.0) {
      const vector<int> until = onlyMoves({{moves[i1], 0}, {moves[i2], 0}});
      unique_ptr<Search> kataGo = runSearch(nnEval, logger, params(40, member, 1.0), root, nullptr, until);
      const Loc preferred = kataGo->getChosenMoveLoc();
      const size_t ip = preferred == moves[i1] ? i1 : i2, io = preferred == moves[i1] ? i2 : i1;
      check(preferred == moves[i1] || preferred == moves[i2], "last move: the search chooses an allowed move");
      //ψ along φ(other) - φ(preferred), scaled so that w times the difference of the two rewards is 20.
      vector<double> dir(State::N);
      double norm2 = 0.0;
      for(int s = 0; s < State::N; s++) {
        dir[s] = phis[io][s] - phis[ip][s];
        norm2 += dir[s] * dir[s];
      }
      const double factor = (1.0 - 0.125) / GameInterface::diversityMoves<GameEncoding>();
      const double scale = 20.0 / (weightOf(lambda) * factor * norm2);
      vector<double> plus(State::N), minus(State::N);
      for(int s = 0; s < State::N; s++) {
        plus[s] = scale * dir[s];
        minus[s] = -scale * dir[s];
      }
      for(int sign: {1, -1}) {
        const shared_ptr<const State> towards = memberState(sign > 0 ? plus : minus, mean, sd);
        const double rp = rewardFor(*towards, member, root.after(moves[ip]), P);
        const double ro = rewardFor(*towards, member, root.after(moves[io]), P);
        check(near((ro - rp) * weightOf(lambda), 20.0 * sign, 1e-6), "last move: the rewards differ as scaled");
        unique_ptr<Search> diverse = runSearch(nnEval, logger, params(40, member, lambda), root, towards, until);
        const Loc chosen = diverse->getChosenMoveLoc();
        check(chosen == (sign > 0 ? moves[io] : moves[ip]),
              "last move: the member prefers the move with the larger reward");
        //Each leaf's utility is KataGo's plus w s_P times its reward.
        int leaves = 0;
        for(const Child& d: childrenOf(*diverse->rootNode))
          for(const Child& k: childrenOf(*kataGo->rootNode))
            if(d.loc == k.loc) {
              check(near(utilityOf(*d.node) - utilityOf(*k.node), wP * (d.loc == moves[ip] ? rp : ro), 1e-6),
                    "last move: a leaf's utility");
              leaves++;
            }
        check(leaves == 2, "last move: both leaves searched");
        printf("last move: rewards %+.3f and %+.3f, KataGo's search plays the first, the member the %s\n", rp, ro,
               chosen == moves[io] ? "second" : "first");
      }
    }
  }

  //The cache: a search of member 3 after a search of member 2 on the same evaluator reads member 3's evaluations.
  {
    const Position& root = game[G / 2];
    unique_ptr<Search> search2 = runSearch(nnEval, logger, params(40, 2, lambda), root, state);
    unique_ptr<Search> search3 = runSearch(nnEval, logger, params(40, 3, lambda), root, state);
    int nodes = 0;
    forEachNode(*search3, [&](const SearchNode& node, const Position& p) {
      const NNOutput* got = node.getNNOutput();
      if(got == NULL)
        return;
      MiscNNInputParams fresh;
      fresh.diversityMember = 3;
      NNResultBuf buf;
      nnEval->evaluate(p.board, p.hist, p.pla(), fresh, buf, true, false);
      const NNOutput& want = *buf.result;
      check(got->whiteWinProb == want.whiteWinProb && got->whiteScoreMean == want.whiteScoreMean &&
            got->intrinsicA == want.intrinsicA && got->intrinsicB == want.intrinsicB, "cache: member 3's evaluation");
      nodes++;
    });
    const NNOutput& r2 = *search2->rootNode->getNNOutput();
    const NNOutput& r3 = *search3->rootNode->getNNOutput();
    check(r2.intrinsicA != r3.intrinsicA && r2.whiteWinProb != r3.whiteWinProb,
          "cache: the members' evaluations differ");
    check(nodes >= 10, "cache: enough nodes");
    printf("cache: %d nodes of member 3's search hold member 3's evaluations\n", nodes);
  }

  //Self-play's kind of search with the term: at each node of its graph, of either player's moves, the node constant is
  //w s_P times the reward of the move into it (none at the root), and the values are finite.
  {
    int nodes = 0, rewarded = 0;
    for(int at: {G / 8, G / 2, G - 12}) {
      const Position& root = game[at];
      const Player P = root.pla();
      unique_ptr<Search> search = runSearch(nnEval, logger, selfplayParams(200, MEMBER, lambda), root, state);
      check(visitsOf(*search->rootNode) >= 200, "self-play's search: its visits");
      forEachNode(*search, [&](const SearchNode& node, const Position& p) {
        const bool isRoot = &node == search->rootNode;
        const double want = isRoot ? 0.0 : signedWeight(P, lambda) * rewardFor(*state, MEMBER, p, P);
        check(near(node.diversityReward, want), "self-play's search: a node's reward");
        check(std::isfinite(utilityOf(node)), "self-play's search: a finite utility");
        nodes++;
        if(node.diversityReward != 0.0)
          rewarded++;
      });
    }
    check(rewarded > 0, "self-play's search: some node has a reward");
    printf("self-play's search: %d nodes of 3 searches, %d with a reward, each the reward of its move\n", nodes,
           rewarded);
  }

  //The reward's scale k (the run's diversityRewardScale): a state with k = 100 and the same ψ, μ, σ and ℓ0. The rows'
  //rewards and targets, and the search's node rewards, are k times those of k = 1, through the one reward
  //(AZdb::State::reward).
  {
    const double k = 100.0;
    const shared_ptr<const State> scaled = memberState(v, mean, sd, k);
    for(int i = 0; i < AZdb::NUM_MEMBERS; i++)
      check(near(scaled->factor[i], k * state->factor[i]), "the reward's scale: a member's factor");
    check(state->factor[member] > 0.0, "the reward's scale: the member's factor at k = 1");

    //The rows of the whole game, both players the member, with the roots' outputs at 0: each target is a sum of
    //rewards. The rewards are floats, so their sums are compared to 1e-4.
    const vector<float> zeros(G, 0.0f);
    const int members[2] = {member, member};
    auto positionOf = [&](int j) -> const GameRules::Position& { return game[j].board.pos; };
    AZdb::Targets plain, times;
    AZdb::rowTargets<GameEncoding>(positionOf, G, true, members, *state, zeros.data(), zeros.data(), plain);
    AZdb::rowTargets<GameEncoding>(positionOf, G, true, members, *scaled, zeros.data(), zeros.data(), times);
    int rewards = 0, targets = 0;
    for(int j = 0; j < G; j++) {
      const string at = "the reward's scale: row " + Global::intToString(j);
      check(near(times.ownReward[j], k * plain.ownReward[j], 1e-6) &&
            near(times.otherReward[j], k * plain.otherReward[j], 1e-6), at + ": its rewards");
      check(near(times.aTarget[j], k * plain.aTarget[j], 1e-4) && near(times.bTarget[j], k * plain.bTarget[j], 1e-4),
            at + ": its targets");
      check(times.aWeight[j] == plain.aWeight[j] && times.bWeight[j] == plain.bWeight[j], at + ": its weights");
      rewards += plain.ownReward[j] != 0.0f ? 1 : 0;
      targets += plain.aWeight[j] == 1.0f && plain.aTarget[j] != 0.0f ? 1 : 0;
    }
    check(rewards > 0 && targets > 0, "the reward's scale: rows with rewards and targets");

    //The search: on a chain, the node constants are the scaled rows' rewards, and D holds (chainTest). In self-play's
    //kind of search, each node constant is k times w s_P times the reward of the move into it at k = 1.
    chainTest(nnEval, logger, lineOf(game, G - 5), member, lambda, scaled, 12, "chain of 5 at k = 100");
    const Position& root = game[G / 2];
    const Player P = root.pla();
    unique_ptr<Search> search = runSearch(nnEval, logger, selfplayParams(200, member, lambda), root, scaled);
    int nodes = 0, rewarded = 0;
    forEachNode(*search, [&](const SearchNode& node, const Position& p) {
      const bool isRoot = &node == search->rootNode;
      const double want = isRoot ? 0.0 : k * signedWeight(P, lambda) * rewardFor(*state, member, p, P);
      check(near(node.diversityReward, want), "the reward's scale: a node's reward in self-play's search");
      check(std::isfinite(utilityOf(node)), "the reward's scale: a finite utility");
      nodes++;
      if(node.diversityReward != 0.0)
        rewarded++;
    });
    check(rewarded > 0, "the reward's scale: some node has a reward");
    printf("the reward's scale: k = %g gives %d rows (%d with a reward, %d with a target) k times their rewards and "
           "targets, a chain of 5 its D, and %d nodes of self-play's search (%d with a reward) k times their "
           "rewards\n", k, G, rewards, targets, nodes, rewarded);
  }

  //The refusals.
  {
    auto refuses = [&](NNEvaluator* eval, const SearchParams& p, const shared_ptr<const State>& s, const string& word,
                       const string& label) {
      unique_ptr<Search> search = newSearch(eval, logger, p, game[G / 2], s, vector<int>());
      try {
        search->runWholeSearch(game[G / 2].pla());
        check(false, label + ": no refusal");
      }
      catch(const StringError& e) {
        check(string(e.what()).find(word) != string::npos, label + ": " + e.what());
      }
    };
    refuses(nnEval, params(10, member, lambda), nullptr, "state", "no state");
    refuses(plainEval, params(10, member, lambda), state, "intrinsic value head", "a model without the head");
    SearchParams cached = params(10, member, lambda);
    cached.useEvalCache = true;
    refuses(nnEval, cached, state, "useEvalCache", "the eval cache");
    refuses(nnEval, params(10, AZdb::NUM_MEMBERS, lambda), state, "diversityMember", "a member outside the team");
    //w = 0 needs neither the head nor a state.
    runSearch(plainEval, logger, params(10, member, 1.0), game[G / 2], nullptr);
    runSearch(plainEval, logger, params(10, 0, lambda), game[G / 2], nullptr);
    printf("refusals: no state, no head, the eval cache, a member outside the team\n");
  }

  //The tree: kept between P's moves (the new root without a reward), cleared when P changes, unless w = 0.
  {
    int start = -1;
    for(int i = G / 4; i < G - 10 && start < 0; i++)
      if(game[i].pla() == P_WHITE && game[i - 1].pla() != P_WHITE)
        start = i;
    check(start >= 0, "tree: a turn's start");
    for(double lam: {lambda, 1.0}) {
      const string label = lam < 1.0 ? "tree with the term" : "tree without the term";
      unique_ptr<Search> search = runSearch(nnEval, logger, params(40, member, lam), game[start], state);
      int kept = 0;
      bool cleared = false;
      for(int moves = 0; moves < 6; moves++) {
        Loc best = Board::NULL_LOC;
        int64_t bestVisits = -1;
        for(const Child& c: childrenOf(*search->rootNode))
          if(c.edgeVisits > bestVisits) {
            best = c.loc;
            bestVisits = c.edgeVisits;
          }
        const Player before = search->rootPla;
        search->makeMove(best, before);
        if(search->rootHistory.isGameFinished)
          break;
        const Player now = search->rootPla;
        search->runWholeSearch(now);
        if(now == before) {
          check(search->lastSearchNumPlayouts < 40, label + ": kept within the turn");
          check(search->rootNode->diversityReward == 0.0, label + ": the new root has no reward");
          kept++;
        }
        else {
          if(lam < 1.0)
            check(search->lastSearchNumPlayouts == 40, label + ": cleared when P changes");
          else
            check(search->lastSearchNumPlayouts < 40, label + ": kept when P changes");
          cleared = true;
          break;
        }
      }
      check(kept >= 1 && cleared, label + ": a move within the turn and a change of P");
    }
    printf("tree: kept within P's turn, cleared when P changes with the term only\n");
  }

  delete nnEval;
  delete plainEval;
  NeuralNet::globalCleanup();
  if(failures > 0) {
    printf("testdiversitysearch: %d failures\n", failures);
    return 1;
  }
  printf("testdiversitysearch: OK\n");
  return 0;
}
