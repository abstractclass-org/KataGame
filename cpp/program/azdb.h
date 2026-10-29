#ifndef PROGRAM_AZDB_H_
#define PROGRAM_AZDB_H_

//AZdb's members (Zahavy et al., "Diversifying AI: Towards Creative Chess with AlphaZero", arXiv 2308.09175): a team of
//NUM_MEMBERS policies in one network, which reads the member as a one-hot latent (NNInputs V14). Each member i >= 1
//gets an intrinsic reward for reaching positions that differ from the other members' positions. Member 0 keeps the
//true objective, and its reward is 0. The design is the engine repository's
//games/blueprints/docs/research/diversity-players.md ("The latent and the members", "The intrinsic value head and its
//targets").
//
//This file holds what the training rows and the search read, for every game: a game's copy of the members' state
//(State), the reward of a move, and the targets of the network's intrinsic value head (rowTargets). The search's
//diversity term reads the same rewards ("The search": Search::computeDiversityReward). The game gives the statistics
//of a position (GameInterface: Diversity). A game without them gives every member the reward 0. The file reads the
//game's interface only, so that a game's Python binding tests it too.

#include <cmath>
#include <vector>

#include "../game/gameinterface.h"

namespace AZdb {

//Member 0 and the diversity members 1 to 4 (the paper's fast configuration).
constexpr int NUM_MEMBERS = 5;
//The span of the intrinsic value head's targets: a row's targets read the rewards up to the start of its mover's
//RETURN_TURNS-th turn after the row's turn.
constexpr int RETURN_TURNS = 5;

//One game's copy of the members' state, taken at the game's start: psi[i], member i's estimate of ψ (the mean of φ
//over its positions, in the normalized statistics), the normalizer of the statistics (each statistic's mean and
//standard deviation), and ℓ0, the distance at which a member's reward is 0.
template<class E> struct State {
  static constexpr int N = GameInterface::diversityStatistics<E>();
  std::vector<double> psi[NUM_MEMBERS];
  GameInterface::DiversityNormalizer<E> normalizer;
  double l0;
  //nearest[i]: member i's nearest other member j*, the lowest one of equal distances. factor[i]: its reward's factor
  //(1 - (ℓ_i / ℓ0)^3) / MOVES, with ℓ_i = ||ψ^i - ψ^j*||. 0 for member 0, and for every member when ℓ0 is not
  //positive.
  int nearest[NUM_MEMBERS];
  double factor[NUM_MEMBERS];

  //psiRows: NUM_MEMBERS rows of N numbers. mean, sd: N numbers each.
  State(const double* psiRows, const double* mean, const double* sd, double l0_) : normalizer(mean, sd), l0(l0_) {
    for(int i = 0; i < NUM_MEMBERS; i++)
      psi[i].assign(psiRows + (size_t)i * N, psiRows + (size_t)(i + 1) * N);
    for(int i = 0; i < NUM_MEMBERS; i++) {
      double best = -1.0;
      nearest[i] = -1;
      for(int j = 0; j < NUM_MEMBERS; j++) {
        if(j == i)
          continue;
        double d2 = 0.0;
        for(int s = 0; s < N; s++)
          d2 += (psi[i][s] - psi[j][s]) * (psi[i][s] - psi[j][s]);
        if(best < 0.0 || d2 < best) {
          best = d2;
          nearest[i] = j;
        }
      }
      const double ratio = std::sqrt(best) / l0;
      factor[i] = (i == 0 || !(l0 > 0.0)) ? 0.0 : (1.0 - ratio * ratio * ratio) / GameInterface::diversityMoves<E>();
    }
  }

  //φ: the normalized statistics x (N numbers each).
  void phi(const double* x, double* out) const { normalizer.z(x, out); }

  //φ of the position after an action of the player mover, from the mover's side, into out (x: scratch), when the
  //action ends a move. False for an action that does not end a move (a pick that waits for its drop): a move's reward
  //is at the action that ends it. The rows' rewards (rowTargets) and the search's (Search::computeDiversityReward)
  //both come from here, so that they cannot differ.
  template<class Position> bool phiOfMove(const Position& after, int mover, double* x, double* out) const {
    if(GameInterface::continuesMove<typename E::Rules>(after))
      return false;
    GameInterface::diversityStatisticsOf<E>(after, mover, x);
    phi(x, out);
    return true;
  }

  //Member i's reward of a move whose position after it has the normalized statistics phi, from the side of the
  //player who moved: (1 - (ℓ_i / ℓ0)^3) φ · (ψ^i - ψ^j*) / MOVES, and 0 for member 0.
  double reward(int member, const double* phi) const {
    if(factor[member] == 0.0)
      return 0.0;
    const std::vector<double>& mine = psi[member];
    const std::vector<double>& theirs = psi[nearest[member]];
    double dot = 0.0;
    for(int s = 0; s < N; s++)
      dot += phi[s] * (mine[s] - theirs[s]);
    return factor[member] * dot;
  }
};

//The rewards and the targets of a game's rows (rowTargets), one entry for each action.
struct Targets {
  //The reward of the row's move as its player's member, and as the other player's member: of the move that the row's
  //action ends, from the mover's side. 0 for an action that does not end a move (a pick that waits for its drop).
  std::vector<float> ownReward;
  std::vector<float> otherReward;
  //The targets of the intrinsic value head's outputs A and B, and their weights (0 or 1).
  std::vector<float> aTarget;
  std::vector<float> aWeight;
  std::vector<float> bTarget;
  std::vector<float> bWeight;
};

//The rows' rewards and targets of a game of numActions actions. positions(j) is the game's position after j actions
//(const Position&), for 0 <= j <= numActions, and the row of action j is at positions(j). finished: whether the game
//ends at positions(numActions); otherwise its record stops before the end. members[player] is the member of each of
//the rules' players. state is the game's copy of the members' state. rootA[j] and rootB[j] are the outputs A and B of
//the network's evaluation at the root of the search of positions(j), with the latent of the mover's member: NaN where
//the network has no intrinsic head, and NULL for none at all.
//
//A move's reward is from the side of the player who made it: φ of the position after the action that ends the move.
//A turn is a run of actions of one player. The row of action j, of player P with member i and opponent O:
//- A: the rewards (as member i) of P's moves from the row up to the start of P's RETURN_TURNS-th turn after the
//  row's turn, then rootA there, or 0 when the game ends before. Weight 1 for i >= 1, and 0 when that root has no
//  output or the record stops before the span's end.
//- B: the rewards of O's moves in the same span, from O's side as member i, then rootB at the same position. Weight
//  1 where O's member is i too, with the same exceptions: B is the other player's return as the same member.
//Member 0's rows have weight 0 for both outputs.
template<class E, class Positions>
void rowTargets(const Positions& positions, int numActions, bool finished, const int* members, const State<E>& state,
                const float* rootA, const float* rootB, Targets& out) {
  using R = typename E::Rules;
  const int n = numActions;
  for(std::vector<float>* v: {&out.ownReward, &out.otherReward, &out.aTarget, &out.aWeight, &out.bTarget,
                              &out.bWeight})
    v->assign(n, 0.0f);
  std::vector<int> mover(n), turn(n);
  for(int j = 0; j < n; j++) {
    mover[j] = R::next_player(positions(j));
    turn[j] = j == 0 ? 0 : turn[j - 1] + (mover[j] != mover[j - 1] ? 1 : 0);
  }

  std::vector<double> x(State<E>::N), phi(State<E>::N);
  for(int j = 0; j < n; j++) {
    const int p = mover[j];
    if((members[p] == 0 && members[1 - p] == 0) || !state.phiOfMove(positions(j + 1), p, x.data(), phi.data()))
      continue;
    out.ownReward[j] = (float)state.reward(members[p], phi.data());
    out.otherReward[j] = (float)state.reward(members[1 - p], phi.data());
  }

  const float nan = std::nanf("");
  for(int j = 0; j < n; j++) {
    const int p = mover[j];
    const int member = members[p];
    if(member == 0)
      continue;
    int k = j + 1;
    while(k < n && turn[k] < turn[j] + 2 * RETURN_TURNS)
      k++;
    float bootA, bootB;
    if(k < n) {
      bootA = rootA != NULL ? rootA[k] : nan;
      bootB = rootB != NULL ? rootB[k] : nan;
    }
    else if(finished)
      bootA = bootB = 0.0f;
    else
      continue;
    double a = 0.0, b = 0.0;
    for(int t = j; t < k; t++) {
      if(mover[t] == p)
        a += out.ownReward[t];
      else
        b += out.otherReward[t];
    }
    if(std::isfinite(bootA)) {
      out.aTarget[j] = (float)(a + bootA);
      out.aWeight[j] = 1.0f;
    }
    if(members[1 - p] == member && std::isfinite(bootB)) {
      out.bTarget[j] = (float)(b + bootB);
      out.bWeight[j] = 1.0f;
    }
  }
}

}  // namespace AZdb

#endif  // PROGRAM_AZDB_H_
