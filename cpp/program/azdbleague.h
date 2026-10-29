#ifndef PROGRAM_AZDBLEAGUE_H_
#define PROGRAM_AZDBLEAGUE_H_

//AZdb in self-play (Zahavy et al., "Diversifying AI: Towards Creative Chess with AlphaZero", arXiv 2308.09175; the
//engine repository's games/blueprints/docs/research/diversity-players.md, "The matchmaker and one trainer"): the
//league of the members. It holds the matchmaker, PSRO-NASH as the paper's Appendix B publishes it, with the table of
//results that the matchmaker reads, and the estimate of ψ with the statistics' normalization. The game threads share
//one league, and a file keeps it across self-play's restarts. Each game takes a copy of ψ, μ and σ at its start
//(snapshot), and gives its sides to the league at its end (finishGame).
//
//The file reads the game's interface only (program/azdb.h), so that a game's Python binding tests it too.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "../core/global.h"
#include "../core/rand.h"
#include "../external/nlohmann_json/json.hpp"
#include "azdb.h"

namespace AZdb {

//ψ's moving average at each of the chooser's finished sides, ψ <- PSI_ALPHA ψ + (1 - PSI_ALPHA) (the side's mean φ),
//as DOMiNO's estimate (Zahavy et al., arXiv 2205.13521, section 4). The design's α, which the statistics'
//normalization takes too (Blueprints: Diversity::ALPHA).
constexpr double PSI_ALPHA = 0.9995;
//The version of the league's file.
constexpr int LEAGUE_FILE_VERSION = 1;

//The table of results that the matchmaker reads (Appendix B: "we track the outcome of matchups in a payoff
//matrix"). The paper does not say how it keeps the table, so it is kept as AlphaStar's league keeps its own (Vinyals
//et al. 2019, the Payoff of its pseudocode): for each pair of the first player's member and the second player's
//member, decayed counts of games, of the first player's wins and of draws. A game of a pair multiplies the pair's
//counts by DECAY, then adds itself. The first player's score against the second is (wins + draws / 2) / games, and
//0.5 for a pair without games. The first player is the rules' player 0 (KataGo's Black, Blueprints' Blue), the
//paper's White: the rows of its table.
struct Payoff {
  static constexpr double DECAY = 0.99;
  double games[NUM_MEMBERS][NUM_MEMBERS] = {};
  double wins[NUM_MEMBERS][NUM_MEMBERS] = {};
  double draws[NUM_MEMBERS][NUM_MEMBERS] = {};

  //A game of the member first, as the first player, against second: the first player's score, 1 for a win, 0.5 for
  //a draw, 0 for a loss.
  void add(int first, int second, double firstScore) {
    games[first][second] = DECAY * games[first][second] + 1.0;
    wins[first][second] = DECAY * wins[first][second] + (firstScore == 1.0 ? 1.0 : 0.0);
    draws[first][second] = DECAY * draws[first][second] + (firstScore == 0.5 ? 1.0 : 0.0);
  }
  double score(int first, int second) const {
    if(games[first][second] == 0.0)
      return 0.5;
    return (wins[first][second] + 0.5 * draws[first][second]) / games[first][second];
  }
};

namespace nash {
  //Solves the n x n system m x = b (row-major m, overwritten) by Gaussian elimination with partial pivoting. False
  //when a pivot is below tol (a singular system).
  inline bool solve(std::vector<double>& m, std::vector<double>& b, int n, double tol) {
    for(int c = 0; c < n; c++) {
      int pivot = c;
      for(int r = c + 1; r < n; r++)
        if(std::fabs(m[r * n + c]) > std::fabs(m[pivot * n + c]))
          pivot = r;
      if(!(std::fabs(m[pivot * n + c]) > tol))
        return false;
      if(pivot != c) {
        for(int k = 0; k < n; k++)
          std::swap(m[c * n + k], m[pivot * n + k]);
        std::swap(b[c], b[pivot]);
      }
      for(int r = c + 1; r < n; r++) {
        const double f = m[r * n + c] / m[c * n + c];
        if(f == 0.0)
          continue;
        for(int k = c; k < n; k++)
          m[r * n + k] -= f * m[c * n + k];
        b[r] -= f * b[c];
      }
    }
    for(int r = n - 1; r >= 0; r--) {
      double s = b[r];
      for(int k = r + 1; k < n; k++)
        s -= m[r * n + k] * b[k];
      b[r] = s / m[r * n + r];
    }
    return true;
  }

  //The guaranteed payoff of the row player's mixture x: its least payoff over the columns.
  inline double guaranteed(const std::vector<double>& x, const double* a, int rows, int cols) {
    double least = 0.0;
    for(int c = 0; c < cols; c++) {
      double s = 0.0;
      for(int r = 0; r < rows; r++)
        s += x[r] * a[r * cols + c];
      if(c == 0 || s < least)
        least = s;
    }
    return least;
  }

  //The row player's optimal strategies that are extreme points of its set of optimal strategies, and the game's
  //value: the Shapley-Snow kernels (Shapley and Snow 1950). Each square submatrix M of rows R and columns C whose
  //system x^T M = v 1, sum(x) = 1 has one solution gives a candidate x on R. The value is the largest guaranteed
  //payoff of the candidates with no negative weight, and the optimal ones are those that guarantee it.
  inline std::vector<std::vector<double>> optimalVertices(const double* a, int rows, int cols, double tol,
                                                          double& value) {
    std::vector<std::vector<double>> candidates;
    std::vector<double> guarantees;
    const int most = std::min(rows, cols);
    for(unsigned rowSet = 1; rowSet < (1u << rows); rowSet++) {
      std::vector<int> rs;
      for(int r = 0; r < rows; r++)
        if(rowSet & (1u << r))
          rs.push_back(r);
      const int s = (int)rs.size();
      if(s > most)
        continue;
      for(unsigned colSet = 1; colSet < (1u << cols); colSet++) {
        std::vector<int> cs;
        for(int c = 0; c < cols; c++)
          if(colSet & (1u << c))
            cs.push_back(c);
        if((int)cs.size() != s)
          continue;
        //Unknowns x_r (r in rs) and v: for each c in cs, sum_r a[r][c] x_r - v = 0; sum_r x_r = 1.
        const int n = s + 1;
        std::vector<double> m((size_t)n * n, 0.0), b(n, 0.0);
        for(int k = 0; k < s; k++) {
          for(int j = 0; j < s; j++)
            m[k * n + j] = a[rs[j] * cols + cs[k]];
          m[k * n + s] = -1.0;
        }
        for(int j = 0; j < s; j++)
          m[s * n + j] = 1.0;
        b[s] = 1.0;
        if(!solve(m, b, n, 1e-12))
          continue;
        std::vector<double> x(rows, 0.0);
        bool nonNegative = true;
        for(int j = 0; j < s; j++) {
          if(b[j] < -tol)
            nonNegative = false;
          x[rs[j]] = std::max(0.0, b[j]);
        }
        if(!nonNegative)
          continue;
        double sum = 0.0;
        for(double w : x)
          sum += w;
        for(double& w : x)
          w /= sum;
        candidates.push_back(x);
        guarantees.push_back(guaranteed(x, a, rows, cols));
      }
    }
    //Each row with any column is a kernel of size 1, whose system always has its solution: the pure strategy.
    if(guarantees.empty())
      throw StringError("maxentNash: no kernel of the game has a solution");
    value = guarantees[0];
    for(double g : guarantees)
      value = std::max(value, g);
    std::vector<std::vector<double>> vertices;
    for(size_t k = 0; k < candidates.size(); k++) {
      if(guarantees[k] < value - tol)
        continue;
      bool seen = false;
      for(const std::vector<double>& v : vertices) {
        double d = 0.0;
        for(int r = 0; r < rows; r++)
          d = std::max(d, std::fabs(v[r] - candidates[k][r]));
        if(d <= tol)
          seen = true;
      }
      if(!seen)
        vertices.push_back(candidates[k]);
    }
    return vertices;
  }

  //The point of largest entropy in the row player's set of optimal strategies, the convex hull of vertices (the game's
  //value is value). The point lies in the affine hull of the vertices, where it maximizes the entropy subject to the
  //columns' constraints that are not tight on the whole set: sum_r x_r a[r][c] >= value. A Newton method on the
  //coordinates of the affine hull, with a logarithmic barrier for those constraints whose weight falls to 1e-14,
  //from the vertices' mean, a point of the set's relative interior. The entropy keeps each weight of the support
  //positive.
  inline std::vector<double> maxentOver(const std::vector<std::vector<double>>& vertices, const double* a, int rows,
                                        int cols, double value, double tol) {
    const int k = (int)vertices.size();
    std::vector<double> x0(rows, 0.0);
    for(const std::vector<double>& v : vertices)
      for(int r = 0; r < rows; r++)
        x0[r] += v[r] / k;
    //An orthonormal basis of the directions from x0 to the vertices.
    std::vector<std::vector<double>> basis;
    for(const std::vector<double>& v : vertices) {
      std::vector<double> d(rows);
      for(int r = 0; r < rows; r++)
        d[r] = v[r] - x0[r];
      for(int pass = 0; pass < 2; pass++)
        for(const std::vector<double>& u : basis) {
          double p = 0.0;
          for(int r = 0; r < rows; r++)
            p += d[r] * u[r];
          for(int r = 0; r < rows; r++)
            d[r] -= p * u[r];
        }
      double norm = 0.0;
      for(double e : d)
        norm += e * e;
      norm = std::sqrt(norm);
      if(norm > 1e-9) {
        for(double& e : d)
          e /= norm;
        basis.push_back(d);
      }
    }
    const int dim = (int)basis.size();
    if(dim == 0)
      return x0;
    std::vector<int> support;
    for(int r = 0; r < rows; r++)
      if(x0[r] > 1e-12)
        support.push_back(r);
    auto slack = [&](const std::vector<double>& x, int c) {
      double s = -value;
      for(int r = 0; r < rows; r++)
        s += x[r] * a[r * cols + c];
      return s;
    };
    //The columns whose constraint has room at x0: the others hold with equality on the whole set.
    std::vector<int> loose;
    for(int c = 0; c < cols; c++)
      if(slack(x0, c) > tol)
        loose.push_back(c);
    //g[c][i]: the change of column c's slack along basis[i].
    std::vector<std::vector<double>> g(loose.size(), std::vector<double>(dim, 0.0));
    for(size_t j = 0; j < loose.size(); j++)
      for(int i = 0; i < dim; i++)
        for(int r = 0; r < rows; r++)
          g[j][i] += basis[i][r] * a[r * cols + loose[j]];

    std::vector<double> z(dim, 0.0), x(rows), xt(rows), zt(dim);
    auto pointOf = [&](const std::vector<double>& zz, std::vector<double>& out) {
      out = x0;
      for(int i = 0; i < dim; i++)
        for(int r = 0; r < rows; r++)
          out[r] += zz[i] * basis[i][r];
    };
    auto feasible = [&](const std::vector<double>& xx) {
      for(int r : support)
        if(!(xx[r] > 0.0))
          return false;
      for(int c : loose)
        if(!(slack(xx, c) > 0.0))
          return false;
      return true;
    };
    auto objective = [&](const std::vector<double>& xx, double tau) {
      double f = 0.0;
      for(int r : support)
        f -= xx[r] * std::log(xx[r]);
      for(int c : loose)
        f += tau * std::log(slack(xx, c));
      return f;
    };
    for(double tau = 1e-2; tau > 1e-15; tau *= 0.1) {
      for(int iter = 0; iter < 200; iter++) {
        pointOf(z, x);
        std::vector<double> grad(dim, 0.0), hess((size_t)dim * dim, 0.0);
        for(int r : support) {
          const double gr = -std::log(x[r]) - 1.0;
          for(int i = 0; i < dim; i++) {
            grad[i] += basis[i][r] * gr;
            for(int l = 0; l < dim; l++)
              hess[i * dim + l] += basis[i][r] * basis[l][r] / x[r];
          }
        }
        for(size_t j = 0; j < loose.size(); j++) {
          const double s = slack(x, loose[j]);
          for(int i = 0; i < dim; i++) {
            grad[i] += tau * g[j][i] / s;
            for(int l = 0; l < dim; l++)
              hess[i * dim + l] += tau * g[j][i] * g[j][l] / (s * s);
          }
        }
        //hess holds minus the Hessian, positive definite: the Newton step solves hess step = grad.
        std::vector<double> step = grad;
        if(!solve(hess, step, dim, 1e-300))
          break;
        double decrement = 0.0;
        for(int i = 0; i < dim; i++)
          decrement += grad[i] * step[i];
        if(!(decrement > 1e-20))
          break;
        const double f0 = objective(x, tau);
        double t = 1.0;
        bool moved = false;
        for(; t > 1e-20; t *= 0.5) {
          for(int i = 0; i < dim; i++)
            zt[i] = z[i] + t * step[i];
          pointOf(zt, xt);
          if(feasible(xt) && objective(xt, tau) >= f0 + 0.25 * t * decrement) {
            moved = true;
            break;
          }
        }
        if(!moved)
          break;
        z = zt;
      }
    }
    pointOf(z, x);
    double sum = 0.0;
    for(double& w : x) {
      w = std::max(0.0, w);
      sum += w;
    }
    for(double& w : x)
      w /= sum;
    return x;
  }

  //The row player's maximum-entropy optimal strategy (maxentNash) of the game a: rows x cols numbers, row-major, the
  //row player's payoff. At most 8 rows and 8 columns: the kernels are every pair of a set of rows and a set of
  //columns.
  inline std::vector<double> maxent(const double* a, int rows, int cols) {
    if(rows < 1 || cols < 1 || rows > 8 || cols > 8)
      throw StringError("maxentNash: a game of 1 to 8 rows and columns");
    double scale = 1.0;
    for(int i = 0; i < rows * cols; i++) {
      if(!std::isfinite(a[i]))
        throw StringError("maxentNash: a payoff that is not finite");
      scale = std::max(scale, std::fabs(a[i]));
    }
    const double tol = 1e-9 * scale;
    double value;
    const std::vector<std::vector<double>> vertices = optimalVertices(a, rows, cols, tol, value);
    return maxentOver(vertices, a, rows, cols, value, tol);
  }
}

//The maximum-entropy Nash equilibrium of a zero-sum game in a matrix: among the equilibria, the strategies of the
//largest entropy, which are unique (Balduzzi et al. 2018, "Re-evaluating Evaluation", Proposition 4: the natural
//choice of a Nash equilibrium of a zero-sum game). payoff holds rows x cols numbers, row-major, the row player's
//payoff, which the row player maximizes and the column player minimizes. rowMix gets the row player's strategy (rows
//numbers) and colMix the column player's (cols numbers). A table of equal results makes every strategy optimal, so
//both strategies are uniform. At most 8 rows and 8 columns.
inline void maxentNash(const double* payoff, int rows, int cols, double* rowMix, double* colMix) {
  const std::vector<double> row = nash::maxent(payoff, rows, cols);
  std::copy(row.begin(), row.end(), rowMix);
  //The column player's: the row player's of the game -payoff^T.
  std::vector<double> minus((size_t)cols * rows);
  for(int r = 0; r < rows; r++)
    for(int c = 0; c < cols; c++)
      minus[c * rows + r] = -payoff[r * cols + c];
  const std::vector<double> col = nash::maxent(minus.data(), cols, rows);
  std::copy(col.begin(), col.end(), colMix);
}

//A game's members, drawn by the matchmaker.
struct Match {
  //The chooser, member i, and its player: the rules' player 0 (the first player) or 1.
  int chooser = 0;
  int chooserPlayer = 0;
  //The member of each of the rules' players.
  int members[2] = {0, 0};
  //Whether the game starts from a state of RGSC's buffer, vanilla or swing (restartMatch).
  bool restart = false;
};

//The match of a game that starts from a state of RGSC's buffer, either buffer (a departure from both papers: the
//engine repository's games/blueprints/docs/research/diversity-players.md, "The matchmaker and one trainer"): member 0
//against itself, with no draw (self-play takes it in place of League::draw). Member 0 is the chooser at both sides, so
//its rows train at both. The league takes nothing from it (League::finishGame).
inline Match restartMatch() {
  Match match;
  match.restart = true;
  return match;
}

//The mixture over the members 0 to chooser that the chooser meets as the rules' player chooserPlayer: the other
//player's maximum-entropy Nash mixture of the table over the members 0 to chooser (maxentNash). The table's rows are
//the first player's, so the chooser as the first player meets the second player's mixture (the row player's of
//-table^T), and as the second player the first player's (the row player's of the table).
inline std::vector<double> opponentMixture(const Payoff& payoff, int chooser, int chooserPlayer) {
  const int n = chooser + 1;
  //The opponent's payoff, the opponent in the rows.
  std::vector<double> opponentPayoff((size_t)n * n);
  for(int r = 0; r < n; r++)
    for(int c = 0; c < n; c++)
      opponentPayoff[r * n + c] = chooserPlayer == 0 ? -payoff.score(c, r) : payoff.score(r, c);
  return nash::maxent(opponentPayoff.data(), n, n);
}

//PSRO-NASH's interaction graphs (Appendix B, Fig. 10), one for each player of the chooser: graphs[p][i][j] is the
//probability that the chooser i, as the rules' player p, plays against the member j. For j <= i, it is the other
//player's maximum-entropy Nash mixture of the table over the members 0 to i (opponentMixture); for j > i it is 0, so
//the graphs are lower triangular and member 0 plays only against itself. The current member is in the population it
//answers, as in the paper ("player 0 ... essentially engages in standard AZ self-play").
inline void interactionGraphs(const Payoff& payoff, double graphs[2][NUM_MEMBERS][NUM_MEMBERS]) {
  for(int p = 0; p < 2; p++)
    for(int i = 0; i < NUM_MEMBERS; i++) {
      const std::vector<double> mixture = opponentMixture(payoff, i, p);
      for(int j = 0; j < NUM_MEMBERS; j++)
        graphs[p][i][j] = j <= i ? mixture[j] : 0.0;
    }
}

//PSRO-NASH's draw of a game (Appendix B, Eq. 10): P(i, j, t) = P(j | i, t) P_unif(i) P_unif(t). The chooser i is
//uniform over the members, its player t is uniform, and its opponent j comes from the interaction graph of t
//(opponentMixture).
inline Match drawMatch(const Payoff& payoff, Rand& rand) {
  Match match;
  match.chooser = (int)rand.nextUInt((uint32_t)NUM_MEMBERS);
  match.chooserPlayer = (int)rand.nextUInt(2);
  const std::vector<double> mixture = opponentMixture(payoff, match.chooser, match.chooserPlayer);
  match.members[match.chooserPlayer] = match.chooser;
  match.members[1 - match.chooserPlayer] = (int)rand.nextUInt(mixture.data(), mixture.size());
  return match;
}

//The statistics of a game's sides: side[p] gets, for each move of the rules' player p, the statistics of the
//position after it from p's side (GameInterface: Diversity, before the normalization), N numbers each, one position
//after another. A move's position is the one after the action that ends it, as for its reward (program/azdb.h).
//positions(j) is the game's position after j actions (const Position&), for 0 <= j <= numActions.
template<class E, class Positions>
void sideStatistics(const Positions& positions, int numActions, std::vector<double> side[2]) {
  using R = typename E::Rules;
  constexpr int N = GameInterface::diversityStatistics<E>();
  side[0].clear();
  side[1].clear();
  std::vector<double> x(N);
  for(int j = 0; j < numActions; j++) {
    if(GameInterface::continuesMove<R>(positions(j + 1)))
      continue;
    const int p = R::next_player(positions(j));
    GameInterface::diversityStatisticsOf<E>(positions(j + 1), p, x.data());
    side[p].insert(side[p].end(), x.begin(), x.end());
  }
}

//The league of a run's members in self-play: the table of results and the estimate of ψ with the statistics'
//normalization, shared by the game threads (every method takes the league's lock), and kept in a file across
//restarts (save, load). ψ of every member starts at 0, the mean of the normalized statistics over the seed's
//positions, and μ and σ start at the seed's.
//
//At each game's end (finishGame), with the game's own copy of ψ, μ and σ (snapshot, taken at its start):
//- the chooser's ψ moves toward the mean φ of its player's positions, at each of the chooser's sides (both in a game
//  against itself): ψ <- PSI_ALPHA ψ + (1 - PSI_ALPHA) mean φ, φ the statistics normalized with the game's copy.
//  These are the positions of the rows that train;
//- μ and σ move at each side of the game, every member's (the normalizer's update);
//- the table adds the game's result, when the game has one.
//A game that starts from a state of RGSC's buffer (restartMatch) changes none of them: ψ, μ and σ come from games
//from the normal start only, and the table holds the results of the matchmaker's games. It counts as a game.
//The matchmaker draws among every member of the latent (NUM_MEMBERS).
template<class E> class League {
 public:
  static constexpr int N = GameInterface::diversityStatistics<E>();

  //l0: the distance at which a member's reward is 0. seedMean, seedSd: each statistic's mean and standard deviation
  //over the seed's positions.
  League(double l0_, const double* seedMean, const double* seedSd) : l0(l0_), normalizer(seedMean, seedSd) {
    for(int i = 0; i < NUM_MEMBERS; i++)
      psi[i].assign(N, 0.0);
  }

  //The game's copy of the members' state, for its searches and its rows.
  std::shared_ptr<const State<E>> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<double> rows((size_t)NUM_MEMBERS * N), mean(N), sd(N);
    for(int i = 0; i < NUM_MEMBERS; i++)
      std::copy(psi[i].begin(), psi[i].end(), rows.begin() + (size_t)i * N);
    for(int s = 0; s < N; s++) {
      mean[s] = normalizer.mean(s);
      sd[s] = normalizer.sd(s);
    }
    return std::make_shared<const State<E>>(rows.data(), mean.data(), sd.data(), l0);
  }

  //A game's members, by PSRO-NASH (drawMatch) on the table as it is. A game that starts from a state of RGSC's buffer
  //takes restartMatch instead.
  Match draw(Rand& rand) const {
    Payoff table;
    {
      std::lock_guard<std::mutex> lock(mutex);
      table = payoff;
    }
    return drawMatch(table, rand);
  }

  //A finished game: its match, its sides' statistics (sideStatistics), the game's copy of the state (snapshot), and
  //the first player's score (1 a win, 0.5 a draw, 0 a loss; NaN for a game without a result, which the table does
  //not take). A restart (Match::restart) only counts. Returns the league's count of finished games, this one
  //included.
  int64_t finishGame(const Match& match, const std::vector<double> side[2], const State<E>& gameState,
                     double firstScore) {
    if(match.restart) {
      std::lock_guard<std::mutex> lock(mutex);
      return ++games;
    }
    std::vector<double> phi(N), meanPhi[2];
    int counts[2];
    for(int p = 0; p < 2; p++) {
      counts[p] = N == 0 ? 0 : (int)(side[p].size() / N);
      if(match.members[p] != match.chooser || counts[p] == 0)
        continue;
      meanPhi[p].assign(N, 0.0);
      for(int k = 0; k < counts[p]; k++) {
        gameState.phi(side[p].data() + (size_t)k * N, phi.data());
        for(int s = 0; s < N; s++)
          meanPhi[p][s] += phi[s] / counts[p];
      }
    }
    std::lock_guard<std::mutex> lock(mutex);
    for(int p = 0; p < 2; p++) {
      if(counts[p] > 0)
        normalizer.update(side[p].data(), counts[p]);
      if(meanPhi[p].empty())
        continue;
      std::vector<double>& estimate = psi[match.chooser];
      for(int s = 0; s < N; s++)
        estimate[s] = PSI_ALPHA * estimate[s] + (1.0 - PSI_ALPHA) * meanPhi[p][s];
      sides[match.chooser]++;
    }
    if(std::isfinite(firstScore))
      payoff.add(match.members[0], match.members[1], firstScore);
    return ++games;
  }

  //Saves the league to path (a temporary file, then a rename), under the league's lock, as RGSC's buffer saves.
  void save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex);
    nlohmann::json j;
    j["version"] = LEAGUE_FILE_VERSION;
    j["members"] = NUM_MEMBERS;
    j["statistics"] = N;
    j["l0"] = l0;
    j["games"] = games;
    j["sides"] = std::vector<int64_t>(sides, sides + NUM_MEMBERS);
    j["psi"] = nlohmann::json::array();
    for(int i = 0; i < NUM_MEMBERS; i++)
      j["psi"].push_back(psi[i]);
    std::vector<double> mean(N), sd(N);
    for(int s = 0; s < N; s++) {
      mean[s] = normalizer.mean(s);
      sd[s] = normalizer.sd(s);
    }
    j["mean"] = mean;
    j["sd"] = sd;
    nlohmann::json table;
    table["decay"] = Payoff::DECAY;
    table["games"] = nlohmann::json::array();
    table["wins"] = nlohmann::json::array();
    table["draws"] = nlohmann::json::array();
    for(int r = 0; r < NUM_MEMBERS; r++) {
      table["games"].push_back(std::vector<double>(payoff.games[r], payoff.games[r] + NUM_MEMBERS));
      table["wins"].push_back(std::vector<double>(payoff.wins[r], payoff.wins[r] + NUM_MEMBERS));
      table["draws"].push_back(std::vector<double>(payoff.draws[r], payoff.draws[r] + NUM_MEMBERS));
    }
    j["payoff"] = table;
    const std::string tmp = path + ".tmp";
    {
      std::ofstream out(tmp);
      if(!out)
        throw StringError("AZdb: cannot write " + tmp);
      out << j.dump() << "\n";
      if(!out)
        throw StringError("AZdb: cannot write " + tmp);
    }
    if(std::rename(tmp.c_str(), path.c_str()) != 0)
      throw StringError("AZdb: cannot rename " + tmp + " to " + path);
  }

  //Loads the league from path, saved by save. False when the file does not exist: the league stays as it starts.
  //A file of another version or of other counts is refused (an error), so that a run never restarts ψ by mistake.
  //l0 stays the caller's.
  bool load(const std::string& path) {
    std::ifstream in(path);
    if(!in)
      return false;
    nlohmann::json j;
    try {
      in >> j;
    }
    catch(const nlohmann::json::exception& e) {
      throw StringError("AZdb: " + path + " is not a league's file: " + e.what());
    }
    auto refuse = [&](const std::string& why) { throw StringError("AZdb: " + path + ": " + why); };
    try {
      if(j.at("version").get<int>() != LEAGUE_FILE_VERSION)
        refuse("version " + j.at("version").dump() + ", not " + Global::intToString(LEAGUE_FILE_VERSION));
      if(j.at("members").get<int>() != NUM_MEMBERS || j.at("statistics").get<int>() != N)
        refuse("a league of " + j.at("members").dump() + " members and " + j.at("statistics").dump() +
               " statistics, not " + Global::intToString(NUM_MEMBERS) + " and " + Global::intToString(N));
      std::vector<std::vector<double>> psiRows = j.at("psi").get<std::vector<std::vector<double>>>();
      std::vector<double> mean = j.at("mean").get<std::vector<double>>();
      std::vector<double> sd = j.at("sd").get<std::vector<double>>();
      std::vector<int64_t> sideCounts = j.at("sides").get<std::vector<int64_t>>();
      const nlohmann::json& table = j.at("payoff");
      std::vector<std::vector<double>> tGames = table.at("games").get<std::vector<std::vector<double>>>();
      std::vector<std::vector<double>> tWins = table.at("wins").get<std::vector<std::vector<double>>>();
      std::vector<std::vector<double>> tDraws = table.at("draws").get<std::vector<std::vector<double>>>();
      bool shaped = (int)psiRows.size() == NUM_MEMBERS && (int)mean.size() == N && (int)sd.size() == N &&
        (int)sideCounts.size() == NUM_MEMBERS && (int)tGames.size() == NUM_MEMBERS &&
        (int)tWins.size() == NUM_MEMBERS && (int)tDraws.size() == NUM_MEMBERS;
      for(int i = 0; shaped && i < NUM_MEMBERS; i++)
        shaped = (int)psiRows[i].size() == N && (int)tGames[i].size() == NUM_MEMBERS &&
          (int)tWins[i].size() == NUM_MEMBERS && (int)tDraws[i].size() == NUM_MEMBERS;
      if(!shaped)
        refuse("its arrays do not have the league's shapes");
      std::lock_guard<std::mutex> lock(mutex);
      for(int i = 0; i < NUM_MEMBERS; i++) {
        psi[i] = psiRows[i];
        sides[i] = sideCounts[i];
        for(int k = 0; k < NUM_MEMBERS; k++) {
          payoff.games[i][k] = tGames[i][k];
          payoff.wins[i][k] = tWins[i][k];
          payoff.draws[i][k] = tDraws[i][k];
        }
      }
      normalizer = GameInterface::DiversityNormalizer<E>(mean.data(), sd.data());
      games = j.at("games").get<int64_t>();
    }
    catch(const nlohmann::json::exception& e) {
      refuse(std::string("a key is missing or of another type: ") + e.what());
    }
    return true;
  }

  //Each member's distance to its nearest other member, ℓ_i = ||ψ^i - ψ^j*||, from ψ as it is (the nearest members
  //of State).
  std::vector<double> distances() const {
    const std::shared_ptr<const State<E>> state = snapshot();
    std::vector<double> out(NUM_MEMBERS, 0.0);
    for(int i = 0; i < NUM_MEMBERS; i++) {
      const std::vector<double>& mine = state->psi[i];
      const std::vector<double>& theirs = state->psi[state->nearest[i]];
      double d2 = 0.0;
      for(int s = 0; s < N; s++)
        d2 += (mine[s] - theirs[s]) * (mine[s] - theirs[s]);
      out[i] = std::sqrt(d2);
    }
    return out;
  }

  //A one-line summary for the log: the games, each member's finished sides and distance to its nearest member, and
  //the interaction graphs of each player of the chooser.
  std::string summary() const {
    Payoff table;
    int64_t gameCount, sideCounts[NUM_MEMBERS];
    {
      std::lock_guard<std::mutex> lock(mutex);
      table = payoff;
      gameCount = games;
      std::copy(sides, sides + NUM_MEMBERS, sideCounts);
    }
    const std::vector<double> d = distances();
    double graphs[2][NUM_MEMBERS][NUM_MEMBERS];
    interactionGraphs(table, graphs);
    std::ostringstream out;
    out << "AZdb: " << gameCount << " games; sides and distances by member:";
    for(int i = 0; i < NUM_MEMBERS; i++)
      out << " " << sideCounts[i] << "/" << Global::strprintf("%.4f", d[i]);
    out << " (l0 " << l0 << "); graphs as the first player and as the second:";
    for(int p = 0; p < 2; p++) {
      out << " [";
      for(int i = 0; i < NUM_MEMBERS; i++) {
        out << (i == 0 ? "" : " ") << "(";
        for(int j = 0; j <= i; j++)
          out << (j == 0 ? "" : " ") << Global::strprintf("%.2f", graphs[p][i][j]);
        out << ")";
      }
      out << "]";
    }
    return out.str();
  }

  //What the league holds, for the tests and the analysis. mean and sd are the normalizer's own, where a snapshot
  //rebuilds them from the mean and the standard deviation.
  std::vector<double> psiOf(int member) const {
    std::lock_guard<std::mutex> lock(mutex);
    return psi[member];
  }
  double mean(int s) const {
    std::lock_guard<std::mutex> lock(mutex);
    return normalizer.mean(s);
  }
  double sd(int s) const {
    std::lock_guard<std::mutex> lock(mutex);
    return normalizer.sd(s);
  }
  Payoff table() const {
    std::lock_guard<std::mutex> lock(mutex);
    return payoff;
  }
  int64_t sidesOf(int member) const {
    std::lock_guard<std::mutex> lock(mutex);
    return sides[member];
  }
  int64_t gamesFinished() const {
    std::lock_guard<std::mutex> lock(mutex);
    return games;
  }

  const double l0;

 private:
  mutable std::mutex mutex;
  std::vector<double> psi[NUM_MEMBERS];
  GameInterface::DiversityNormalizer<E> normalizer;
  Payoff payoff;
  int64_t sides[NUM_MEMBERS] = {};
  int64_t games = 0;
};

//The seed of the statistics' normalization: each statistic's mean and standard deviation over a reference's positions,
//from a file of tools/diversity_stats.py's seed (the engine repository's games/blueprints/runs/diversity_seed.json):
//{"statistics": [{"mean": ..., "sd": ...}, ...]}, one entry for each statistic in their order.
inline void readSeed(const std::string& path, int n, std::vector<double>& mean, std::vector<double>& sd) {
  std::ifstream in(path);
  if(!in)
    throw StringError("AZdb: cannot read the seed " + path);
  try {
    nlohmann::json j;
    in >> j;
    const nlohmann::json& statistics = j.at("statistics");
    if((int)statistics.size() != n)
      throw StringError("AZdb: the seed " + path + " holds " + Global::intToString((int)statistics.size()) +
                        " statistics, not " + Global::intToString(n));
    mean.assign(n, 0.0);
    sd.assign(n, 0.0);
    for(int s = 0; s < n; s++) {
      mean[s] = statistics[s].at("mean").get<double>();
      sd[s] = statistics[s].at("sd").get<double>();
    }
  }
  catch(const nlohmann::json::exception& e) {
    throw StringError("AZdb: the seed " + path + " is not a seed's file: " + e.what());
  }
}

//A finished game's line for the analysis (one JSON object): the match, the komi, the first player's score (null
//without a result), and for each of the rules' players the count of its positions and the mean of each statistic over
//them (before the normalization; null without positions).
template<class E>
nlohmann::json gameRecord(const Match& match, double komi, double firstScore, const std::vector<double> side[2]) {
  constexpr int N = GameInterface::diversityStatistics<E>();
  nlohmann::json j;
  j["members"] = {match.members[0], match.members[1]};
  j["chooser"] = match.chooser;
  j["chooser_player"] = match.chooserPlayer;
  j["komi"] = komi;
  j["first_score"] = std::isfinite(firstScore) ? nlohmann::json(firstScore) : nlohmann::json(nullptr);
  j["positions"] = nlohmann::json::array();
  j["statistics"] = nlohmann::json::array();
  for(int p = 0; p < 2; p++) {
    const int count = N == 0 ? 0 : (int)(side[p].size() / N);
    j["positions"].push_back(count);
    if(count == 0) {
      j["statistics"].push_back(nullptr);
      continue;
    }
    std::vector<double> mean(N, 0.0);
    for(int k = 0; k < count; k++)
      for(int s = 0; s < N; s++)
        mean[s] += side[p][(size_t)k * N + s] / count;
    j["statistics"].push_back(mean);
  }
  return j;
}

}  // namespace AZdb

#endif  // PROGRAM_AZDBLEAGUE_H_
