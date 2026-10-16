#pragma once
// Enclosure's network encoding part of the game interface (game/gameinterface.h). The inputs are input version V8's
// (ENCLOSURE.md, "Network inputs"; NNInputs::fillRowV8 in neuralnet/nninputs.cpp), without the on-board plane, which
// the front writes: the observation's edge planes (engine.h) at both ends of each edge, then the feature planes
// (features.h). The symmetries act on positions, not on input planes (ENCLOSURE.md, "Symmetries"): these members are
// SymmetryHelpers' getSymPosition and SymActionMap (nninputs.cpp), and the input hash is BoardHistory's history hash
// (boardhistory.cpp).

#include <cstdint>
#include <vector>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "engine.h"
#include "features.h"
#include "gamerules.h"

namespace enclosure {

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  // The observation's 4 * NOFF edge planes (each player's edges, then each player's protected or just-placed edges),
  // then the feature planes (9 per player, then the history planes).
  static constexpr int EDGE_PLANES = 4 * NOFF;
  static constexpr int NUM_SPATIAL_INPUTS = EDGE_PLANES + FEATURE_PLANES;
  // The observation's scalars: the player to move, the edges left in the turn / 2, the edge count / 120, and both
  // scores / (18 * 18 * 61).
  static constexpr int NUM_GLOBAL_INPUTS = SCALARS;

  static void sym_point(int& x, int& y, int symmetry) { GameInterface::symPoint(x, y, symmetry, SIZE, SIZE); }

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    const int symmetry = options.symmetry;
    // Each edge is set at its lower end, in the plane of its offset, and at its upper end, in the opposite offset's
    // plane of the same group. Under a symmetry, the image edge is anchored as make_edge anchors it.
    int32_t idx[MAX_OBS_IDX];
    float scalars[SCALARS];
    const int n = p.observation(idx, scalars);
    for(int i = 0; i < n; i++) {
      const int pos = idx[i] / PLANES, group = idx[i] % PLANES / NOFF;
      int x = pos % SIZE, y = pos / SIZE, k = idx[i] % PLANES % NOFF;
      if(symmetry != 0) {
        const Edge e = sym_edge(make_edge(x, y, x + OFFS[k].dx, y + OFFS[k].dy), symmetry);
        x = e.ax;
        y = e.ay;
        k = OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3];
      }
      set(x, y, group * NOFF + k);
      const int dx = OFFS[k].dx, dy = OFFS[k].dy;
      set(x + dx, y + dy, group * NOFF + OFF_INDEX[3 - dx][3 - dy]);
    }
    // Without history, the history planes (the last HISTORY_TURNS feature planes) are left out.
    static thread_local std::vector<uint8_t> planes(N_POINTS * FEATURE_PLANES);
    feature_planes(p, planes.data());
    const int numFeaturePlanes = options.history ? FEATURE_PLANES : 2 * PER_PLAYER;
    for(int y = 0; y < SIZE; y++)
      for(int x = 0; x < SIZE; x++) {
        const uint8_t* q = planes.data() + (y * SIZE + x) * FEATURE_PLANES;
        int sx = x, sy = y;
        sym_point(sx, sy, symmetry);
        for(int f = 0; f < numFeaturePlanes; f++)
          if(q[f])
            set(sx, sy, EDGE_PLANES + f);
      }
  }

  // Komi enters as a starting score: Red receiving komi K is Red starting K points ahead (scores only add up, and the
  // winner is the sign of the final margin). It goes into the score input of the side it favours.
  static void global_inputs(const Position& p, const GameInterface::InputOptions& options, float* out) {
    int32_t idx[MAX_OBS_IDX];
    p.observation(idx, out);
    if(options.komi > 0)
      out[4] = float((p.scores[1] + options.komi) / (18 * 18 * 61));
    else if(options.komi < 0)
      out[3] = float((p.scores[0] - options.komi) / (18 * 18 * 61));
  }

  // The edges of the earlier turns that the history planes show (Position::history), mixed by turn.
  static Hash128 input_hash(const Position& p) {
    Hash128 h;
    for(int t = 0; t < HISTORY_TURNS; t++) {
      for(int i = 0; i < p.history_n[t]; i++) {
        uint64_t key = (uint64_t)edge_action(p.history[t][i]) * 8 + (uint64_t)t;
        h.hash0 ^= Hash::splitMix64(key ^ 0xD6E8FEB86659FD93ULL);
        h.hash1 ^= Hash::murmurMix(key + 0xA0761D6478BD642FULL);
      }
    }
    return h;
  }

  // A move is an edge from its source point, (y * 19 + x) * NOFF + offset: one channel per offset.
  static constexpr int POLICY_CHANNELS = NOFF;
  static void action_point(int action, int& x, int& y, int& channel) {
    const int cell = action / NOFF;
    channel = action % NOFF;
    x = cell % SIZE;
    y = cell / SIZE;
  }
  static int point_action(int x, int y, int channel) { return (y * SIZE + x) * NOFF + channel; }

  // All 8: the rules respect each, and the network trains on each (ENCLOSURE.md, "Symmetries").
  static constexpr unsigned SYMMETRIES = 0xFFu;

  static Edge sym_edge(const Edge& e, int symmetry) {
    int ax = e.ax, ay = e.ay, bx = e.bx, by = e.by;
    sym_point(ax, ay, symmetry);
    sym_point(bx, by, symmetry);
    return make_edge(ax, ay, bx, by);
  }
  // Every edge (both players', protected, placed, history) mapped. The scores and the turn state do not change.
  static Position sym_position(const Position& p, int symmetry) {
    Position s = p;
    auto map = [&](std::vector<Edge>& v) {
      for(Edge& e : v)
        e = sym_edge(e, symmetry);
    };
    map(s.edges[0]);
    map(s.edges[1]);
    map(s.protected_);
    map(s.placed);
    for(int h = 0; h < HISTORY_TURNS; h++)
      for(int i = 0; i < s.history_n[h]; i++)
        s.history[h][i] = sym_edge(s.history[h][i], symmetry);
    s.area_ok[0] = s.area_ok[1] = false;
    return s;
  }
  // Between two of the mover's nodes, only one direction is legal: from the lexicographically first end. So the image
  // is listed from its far end exactly when that end is one of the mover's nodes and comes first.
  struct SymActionMap {
    int symmetry;
    bool node[SIZE][SIZE];
    SymActionMap(const Position& p, int sym) : symmetry(sym), node() {
      for(const Edge& e : p.edges[p.player])
        node[e.ax][e.ay] = node[e.bx][e.by] = true;
    }
    int operator()(int action) const {
      if(action == PASS || symmetry == 0)
        return action;
      int ax, ay, k;
      action_point(action, ax, ay, k);
      int bx = ax + OFFS[k].dx, by = ay + OFFS[k].dy;
      const bool farNode = node[bx][by];
      sym_point(ax, ay, symmetry);
      sym_point(bx, by, symmetry);
      const bool fromB = farNode && (bx < ax || (bx == ax && by < ay));
      if(fromB)
        return point_action(bx, by, OFF_INDEX[ax - bx + 3][ay - by + 3]);
      return point_action(ax, ay, OFF_INDEX[bx - ax + 3][by - ay + 3]);
    }
  };

  // C0: the future occupancy (the mean over the turn ends after the row of +1 inside the mover's regions, -1 inside
  // the opponent's). C1: the edges that get cut. C2-3: the nodes 8 and 32 edges later. C4: the region occupancy 4
  // rounds later. C5-6: the territory income. C7: the next mover's nodes (ENCLOSURE.md, "Spatial targets").
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP | GameInterface::TARGET_SEKI |
                                      GameInterface::TARGET_FUTURE_PIECES | GameInterface::TARGET_SCORING |
                                      GameInterface::TARGET_INCOME | GameInterface::TARGET_NEXT_PIECES;

  // The width of the score bins against KataGo's bins of 1 point, from the measured margins (ENCLOSURE.md, "Score
  // scale").
  static constexpr double SCORE_SCALE = 12.0;
  // The largest total score, sqrt(18 * 18 * 61), about 141 points (ENCLOSURE.md, "Score utility").
  static constexpr double SCORE_UNIT = 140.5845;
  // Every game has MAX_MOVES edges (Board::TIME_SCALE; games/enclosure/docs/design/08-time-horizons.md in the engine
  // repository).
  static constexpr double GAME_LENGTH = MAX_MOVES;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Enclosure's network encoding part");

}  // namespace enclosure
