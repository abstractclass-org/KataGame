#pragma once
// Blueprints' statistics of a position for AZdb's diversity reward (the engine repository's
// games/blueprints/docs/research/diversity-players.md, "The features of a position"), and their running
// normalization. The reference is the engine repository's games/blueprints/tools/diversity_stats.py: the same
// statistics in the same order, the same transforms, and the same side. Only the Python binding calls this code so far
// (games/blueprints/binding/extras.h): no row, input or search reads it.
//
// The statistics are from the side of a player P (AZdb takes the player who made the move), against its opponent O.
// Distances are in the plane of the hex grid, where the centres of two touching hexes are 1 apart. A hex distance
// counts the steps between two hexes.
// - Cells (6): P's normal cells, stars and walls on the board, then O's, as log(1 + n).
// - Spread (5): the root mean square distance of P's cells from their centroid, the same of P's stars, the same for
//   O, and the distance between the centroid of P's stars and the centroid of O's cells (0 without either), as
//   log(1 + d).
// - Closed areas (4): the hexes of the area closed by P's walls (GameEncoding::for_each_closed), and P's stars inside
//   it, then the same for O, as log(1 + n).
// - Reach (6): the shares of P's stars at a hex distance of REACH_NEAR, of 2 to REACH_CONV, and of REACH_FAR or more
//   from O's nearest cell, then the same of O's stars from P's cells. 0 without stars. Without the other player's
//   cells, every star is at REACH_FAR or more. A grid of buckets of the other player's cells finds the nearest cell.
// - Threats (6): P's stars that some placement of O captures, O's stars that some placement of P captures (as
//   log(1 + n)), the points of P's best placement and of O's (squash), and P's and O's legal placements (as
//   log(1 + n)). Each player's placements are those of its own view when it moves (player_view), on the board as it
//   is, with no pick waiting: the placements that the engine may play. A capture counts every enemy star next to the
//   stamped cells, outside the view too, and a placement's points are the stars it takes times the player's stars.
// - Hands (30): for each of P's blueprints in the hand's canonical order (HandOrder), its cells, stars, walls, length
//   and width, then the same for O. The length is the largest distance between the centres of two of its cells, + 1,
//   and the width the least extent of the centres across any direction (the minimum width of their convex hull), + 1,
//   each rounded to 4 decimals as the reference rounds them. An empty blueprint gives 0 for all five.
//
// The normalization (DiversityNormalizer): z = (x - mean) / max(sd, SIGMA_FLOOR), clipped to [-CLIP, CLIP], times
// 1 / sqrt(the size of its group). The mean and the standard deviation are running estimates (Welford's method),
// seeded from the statistics of a reference's positions (games/blueprints/runs/diversity_seed.json).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gameencoding.h"

namespace blueprints {

struct Diversity {
  static constexpr int NUM_STATISTICS = 57;
  // The groups in the order of the statistics: cells, spread, closed areas, reach, threats, hands.
  static constexpr int NUM_GROUPS = 6;
  static constexpr int GROUP_SIZE[NUM_GROUPS] = {6, 5, 4, 6, 6, 30};
  static constexpr int CELLS = 0, SPREAD = 6, CLOSED = 11, REACH = 15, THREATS = 21, HANDS = 27;
  static_assert(SPREAD == CELLS + 6 && CLOSED == SPREAD + 5 && REACH == CLOSED + 4 && THREATS == REACH + 6 &&
                  HANDS == THREATS + 6 && NUM_STATISTICS == HANDS + 30,
                "the groups follow one another");
  // The reach's bounds in hexes (the design's "The statistics"): touching, within the network's convolutions, and
  // beyond the view's radius.
  static constexpr int REACH_NEAR = 1, REACH_CONV = 13, REACH_FAR = VIEW_RADIUS + 1;
  static constexpr double SIGMA_FLOOR = 0.1, CLIP = 3.0;

  // A point of the rules board.
  struct Point {
    int16_t x, y;
  };

  // The cells of the board, by player: all of them and the stars, the counts of each type, and the box of the
  // player's cells.
  struct Scan {
    int count[2][3];
    std::vector<Point> cells[2], stars[2];
    Box own[2];
  };
  static void scan(const Position& p, Scan& s) {
    for(int player = 0; player < 2; player++) {
      std::fill(s.count[player], s.count[player] + 3, 0);
      s.cells[player].clear();
      s.stars[player].clear();
      s.own[player] = Box();
    }
    const Box& used = p.cell.box();
    if(!used.any)
      return;
    for(int y = used.y0; y <= used.y1; y++) {
      const int8_t* row = p.cell.row(y);
      for(int x = used.x0; x <= used.x1; x++)
        if(const int8_t v = row[x]) {
          const int owner = cell_owner(v), type = cell_type(v);
          s.count[owner][type]++;
          s.cells[owner].push_back({int16_t(x), int16_t(y)});
          if(type == STAR)
            s.stars[owner].push_back({int16_t(x), int16_t(y)});
          s.own[owner].add(x, y);
        }
    }
  }

  // A point's centre in the plane, from the rules board's middle: (q + r / 2, r sqrt(3) / 2).
  static void centre(const Point& c, double& u, double& v) {
    const int q = c.x - RULES_RADIUS, r = c.y - RULES_RADIUS;
    u = q + r / 2.0;
    v = r * std::sqrt(3.0) / 2.0;
  }
  static void centroid(const std::vector<Point>& points, double& u, double& v) {
    u = v = 0.0;
    for(const Point& c : points) {
      double a, b;
      centre(c, a, b);
      u += a;
      v += b;
    }
    u /= double(points.size());
    v /= double(points.size());
  }
  // The root mean square distance of the points' centres from their centroid, 0 for none.
  static double rms(const std::vector<Point>& points) {
    if(points.empty())
      return 0.0;
    double mu, mv, sum = 0.0;
    centroid(points, mu, mv);
    for(const Point& c : points) {
      double a, b;
      centre(c, a, b);
      sum += (a - mu) * (a - mu) + (b - mv) * (b - mv);
    }
    return std::sqrt(sum / double(points.size()));
  }

  // A player's cells in buckets of BUCKET x BUCKET points of the rules board, for the hex distance from a point to the
  // nearest of them (nearest).
  struct Buckets {
    static constexpr int SHIFT = 3, BUCKET = 1 << SHIFT;
    int x0 = 0, y0 = 0, w = 0, h = 0;
    std::vector<int> start;     // bucket b's cells: cells[start[b]] to cells[start[b + 1] - 1]
    std::vector<Point> points;  // the cells, bucket by bucket

    void build(const std::vector<Point>& cells, const Box& box) {
      w = h = 0;
      points.clear();
      if(cells.empty())
        return;
      x0 = box.x0;
      y0 = box.y0;
      w = ((box.x1 - box.x0) >> SHIFT) + 1;
      h = ((box.y1 - box.y0) >> SHIFT) + 1;
      start.assign(size_t(w) * h + 1, 0);
      for(const Point& c : cells)
        start[bucket(c) + 1]++;
      for(size_t b = 1; b < start.size(); b++)
        start[b] += start[b - 1];
      points.resize(cells.size());
      static thread_local std::vector<int> next;
      next.assign(start.begin(), start.end() - 1);
      for(const Point& c : cells)
        points[next[bucket(c)]++] = c;
    }
    int bucket(const Point& c) const { return ((c.y - y0) >> SHIFT) * w + ((c.x - x0) >> SHIFT); }
    // The bucket's column (or row) of a coordinate u from the grid's corner u0, also off the grid: rounded down.
    static int index(int u, int u0) { return u >= u0 ? (u - u0) >> SHIFT : -(((u0 - u) + BUCKET - 1) >> SHIFT); }

    // The hex distance from (x, y) to the nearest cell, capped at `cap`: cap without cells. It may return any distance
    // of a cell up to `enough` once it finds one. The buckets go by rings around the point's bucket: a cell of ring
    // k >= 1 is at least (k - 1) * BUCKET + 1 steps away on an axis, and a hex distance is at least either axis's.
    int nearest(int x, int y, int enough, int cap) const {
      int best = cap;
      if(w == 0)
        return best;
      const int bx = index(x, x0), by = index(y, y0);
      for(int k = 0; k == 0 || (k - 1) * BUCKET + 1 < best; k++) {
        for(int dy = -k; dy <= k; dy++) {
          const int cy = by + dy;
          if(cy < 0 || cy >= h)
            continue;
          const int step = dy == -k || dy == k ? 1 : 2 * k;
          for(int dx = -k; dx <= k; dx += step) {
            const int cx = bx + dx;
            if(cx < 0 || cx >= w)
              continue;
            const int b = cy * w + cx;
            for(int i = start[b]; i < start[b + 1]; i++) {
              const int dq = points[i].x - x, dr = points[i].y - y;
              const int d = std::max({std::abs(dq), std::abs(dr), std::abs(dq + dr)});
              if(d < best) {
                best = d;
                if(best <= enough)
                  return best;
              }
            }
          }
        }
      }
      return best;
    }
  };

  // The shares of the stars at a hex distance of REACH_NEAR, of 2 to REACH_CONV, and of REACH_FAR or more from the
  // nearest cell of `other` (theirs: that player's cells in buckets).
  static void reach(const Position& p, const std::vector<Point>& stars, int other, const Buckets& theirs,
                    double* out) {
    static_assert(REACH_NEAR == 1, "the stars at REACH_NEAR are those next to a cell (next_to_own)");
    int near = 0, conv = 0, far = 0;
    for(const Point& s : stars) {
      if(next_to_own(p, s.x, s.y, other)) {
        near++;
        continue;
      }
      const int d = theirs.nearest(s.x, s.y, REACH_CONV, REACH_FAR);
      conv += d <= REACH_CONV;
      far += d >= REACH_FAR;
    }
    const double n = double(stars.size());
    out[0] = stars.empty() ? 0.0 : near / n;
    out[1] = stars.empty() ? 0.0 : conv / n;
    out[2] = stars.empty() ? 0.0 : far / n;
  }

  static bool has_bit(const ViewRow& row, int x) { return (x < 64 ? row.lo >> x : row.hi >> (x - 64)) & 1; }
  static int bits(const ViewRow& row) { return __builtin_popcountll(row.lo) + __builtin_popcountll(row.hi); }

  // A player's threats in the view where it places (player_view: its window, in p's orientation), as the reference
  // reads them from a copy of the position with that player to move: the other player's stars that some placement
  // captures, the most stars that one placement captures, and the legal placements (each blueprint at each anchor of
  // the view where it fits, equal blueprints each). The copy is the one-line position, whose board parse_position
  // moves to the rules board's middle, where every position's board already is (the start's, and after each
  // placement), so its windows are p's. theirStars: the other player's stars. A placement captures the other
  // player's cells next to its cells (taken_by), so a star is captured by the anchors at the star less each offset of
  // the points next to the blueprint's cells. Those anchors are in the view's frame, as are the blueprint (ViewHand's)
  // and the star's point.
  struct Threats {
    int captured, mostStars, placements;
  };
  static Threats threats(const Position& p, int player, const std::vector<Point>& theirStars) {
    const View view = player_view(p, player);
    Threats out{0, 0, 0};
    ViewRows rows;
    view_rows(p, view, rows);
    // The other player's stars that can lie next to a cell of the view: their points of the view and one around it.
    static thread_local std::vector<Point> stars;
    static thread_local std::vector<uint8_t> taken;
    stars.clear();
    for(const Point& s : theirStars) {
      int x, y;
      view.view_point(s.x, s.y, x, y);
      if(x >= -1 && x <= VIEW && y >= -1 && y <= VIEW)
        stars.push_back({int16_t(x), int16_t(y)});
    }
    taken.assign(stars.size(), 0);
    // The stars that each anchor's placement captures, for the anchors that capture any (touched).
    static thread_local std::vector<uint8_t> count(VIEW_AREA, 0);
    static thread_local std::vector<int> touched;
    for(int k = 0; k < HAND; k++) {
      Blueprint b;
      int mq, mr, noQ = 0, noR = 0;
      view_blueprint(p.hand[player][k], view.o, false, noQ, noR, b, mq, mr);
      if(b.n == 0)
        continue;
      ViewRow fits[VIEW];
      for(int y = 0; y < VIEW; y++) {
        fits[y] = fit_row(b, rows.empty, rows.frontier[player], y);
        out.placements += bits(fits[y]);
      }
      if(stars.empty())
        continue;
      // The offsets of the points next to the blueprint's cells that are not its cells, each once.
      int8_t nq[6 * MAX_CELLS], nr[6 * MAX_CELLS];
      int nn = 0;
      for(int i = 0; i < b.n; i++)
        for(const auto& d : DIRS) {
          const int q = b.q[i] + d[0], r = b.r[i] + d[1];
          if(b.find(q, r) >= 0)
            continue;
          int m = 0;
          while(m < nn && !(nq[m] == q && nr[m] == r))
            m++;
          if(m == nn) {
            nq[nn] = int8_t(q);
            nr[nn] = int8_t(r);
            nn++;
          }
        }
      touched.clear();
      for(size_t s = 0; s < stars.size(); s++)
        for(int m = 0; m < nn; m++) {
          const int ax = stars[s].x - nq[m], ay = stars[s].y - nr[m];
          if(!on_view(ax, ay) || !has_bit(fits[ay], ax))
            continue;
          taken[s] = 1;
          uint8_t& c = count[ay * VIEW + ax];
          if(c++ == 0)
            touched.push_back(ay * VIEW + ax);
          out.mostStars = std::max(out.mostStars, int(c));
        }
      for(int i : touched)
        count[i] = 0;
    }
    out.captured = int(std::count(taken.begin(), taken.end(), uint8_t(1)));
    return out;
  }

  // A score on the inputs' log scale (GameEncoding::squash) in double precision.
  static double squash(double points) {
    const double a = std::log10(1.0 + std::fabs(points) / 10.0) / 2.0;
    return points < 0 ? -a : a;
  }

  // A blueprint's length and width, each + 1 and rounded to 4 decimals (games/blueprints/dashboard.py,
  // blueprint_sizes): the hull of its cells' centres, by the monotone chain on their axial coordinates (a shear of the
  // centres, so the hull's corners are the same and the test is exact), then the largest distance between two
  // corners, and the least, over the hull's sides, of the farthest corner from the side's line (0 for fewer than 3
  // corners). Rounded as Python's round(x, 4), half to even.
  static void sizes(const Blueprint& b, double& length, double& width) {
    struct Axial {
      int q, r;
      bool operator<(const Axial& o) const { return q != o.q ? q < o.q : r < o.r; }
      bool operator==(const Axial& o) const { return q == o.q && r == o.r; }
    };
    Axial points[MAX_CELLS];
    for(int i = 0; i < b.n; i++)
      points[i] = {b.q[i], b.r[i]};
    std::sort(points, points + b.n);
    const int n = int(std::unique(points, points + b.n) - points);
    Axial hull[2 * MAX_CELLS];
    int corners = 0;
    if(n < 3) {
      std::copy(points, points + n, hull);
      corners = n;
    }
    else {
      // One half of the chain: the points in their order, less the last corner.
      auto half = [&](int from, int to, int step) {
        const int first = corners;
        for(int i = from; i != to; i += step) {
          const Axial& c = points[i];
          while(corners - first >= 2) {
            const Axial &a = hull[corners - 2], &o = hull[corners - 1];
            if((o.q - a.q) * (c.r - a.r) - (o.r - a.r) * (c.q - a.q) > 0)
              break;
            corners--;
          }
          hull[corners++] = c;
        }
        corners--;
      };
      half(0, n, 1);
      half(n - 1, -1, -1);
    }
    double u[2 * MAX_CELLS], v[2 * MAX_CELLS];
    for(int i = 0; i < corners; i++) {
      u[i] = hull[i].q + hull[i].r / 2.0;
      v[i] = hull[i].r * std::sqrt(3.0) / 2.0;
    }
    // The squared distance of two corners: the square root of the largest is the largest distance.
    auto square = [&](int i, int j) { return (u[i] - u[j]) * (u[i] - u[j]) + (v[i] - v[j]) * (v[i] - v[j]); };
    auto dist = [&](int i, int j) { return std::sqrt(square(i, j)); };
    double most = 0.0;
    for(int i = 0; i < corners; i++)
      for(int j = i + 1; j < corners; j++)
        most = std::max(most, square(i, j));
    length = std::sqrt(most);
    width = 0.0;
    if(corners >= 3) {
      width = INFINITY;
      for(int i = 0; i < corners; i++) {
        const int j = (i + 1) % corners;
        double farthest = 0.0;
        for(int c = 0; c < corners; c++)
          farthest = std::max(farthest, std::fabs((u[j] - u[i]) * (v[c] - v[i]) - (v[j] - v[i]) * (u[c] - u[i])));
        width = std::min(width, farthest / dist(i, j));
      }
    }
    length = std::nearbyint((length + 1.0) * 1e4) / 1e4;
    width = std::nearbyint((width + 1.0) * 1e4) / 1e4;
  }

  // The statistics of position p from the side of player P (the order above), each after its transform: out holds
  // NUM_STATISTICS numbers. p may be at the game's end. A pick waiting for its drop counts as the blueprints are.
  static void statistics(const Position& p, int player, double* out) {
    const int opponent = 1 - player;
    static thread_local Scan s;
    scan(p, s);
    for(int side = 0; side < 2; side++)
      for(int type = 0; type < 3; type++)
        out[CELLS + 3 * side + type] = std::log1p(double(s.count[side == 0 ? player : opponent][type]));

    const std::vector<Point>&mine = s.cells[player], &theirs = s.cells[opponent];
    double between = 0.0;
    if(!s.stars[player].empty() && !theirs.empty()) {
      double au, av, bu, bv;
      centroid(s.stars[player], au, av);
      centroid(theirs, bu, bv);
      between = std::hypot(au - bu, av - bv);
    }
    const double spread[5] = {rms(mine), rms(s.stars[player]), rms(theirs), rms(s.stars[opponent]), between};
    for(int i = 0; i < 5; i++)
      out[SPREAD + i] = std::log1p(spread[i]);

    for(int side = 0; side < 2; side++) {
      const int who = side == 0 ? player : opponent;
      const int8_t star = cell_value(who, STAR);
      int hexes = 0, stars = 0;
      GameEncoding::for_each_closed(p, who, [&](int x, int y) {
        hexes++;
        stars += p.cell[y * RULES_LEN + x] == star;
      });
      out[CLOSED + 2 * side] = std::log1p(double(hexes));
      out[CLOSED + 2 * side + 1] = std::log1p(double(stars));
    }

    static thread_local Buckets buckets[2];
    for(int who = 0; who < 2; who++)
      buckets[who].build(s.cells[who], s.own[who]);
    reach(p, s.stars[player], opponent, buckets[opponent], out + REACH);
    reach(p, s.stars[opponent], player, buckets[player], out + REACH + 3);

    const Threats my = threats(p, player, s.stars[opponent]);
    const Threats their = threats(p, opponent, s.stars[player]);
    out[THREATS + 0] = std::log1p(double(their.captured));
    out[THREATS + 1] = std::log1p(double(my.captured));
    out[THREATS + 2] = squash(double(my.mostStars) * p.stars[player]);
    out[THREATS + 3] = squash(double(their.mostStars) * p.stars[opponent]);
    out[THREATS + 4] = std::log1p(double(my.placements));
    out[THREATS + 5] = std::log1p(double(their.placements));

    const HandOrder order(p);
    for(int side = 0; side < 2; side++) {
      const int who = side == 0 ? player : opponent;
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[who][k];
        double* f = out + HANDS + 15 * side + 5 * order.slot[who][k];
        if(b.n == 0) {
          std::fill(f, f + 5, 0.0);
          continue;
        }
        f[0] = b.n;
        f[1] = double(std::count(b.t, b.t + b.n, STAR));
        f[2] = double(std::count(b.t, b.t + b.n, WALL));
        sizes(b, f[3], f[4]);
      }
    }
  }

  // Each statistic's weight, 1 / sqrt(the size of its group), so that each group adds the same variance.
  static double weight(int i) {
    int g = 0, end = GROUP_SIZE[0];
    while(i >= end)
      end += GROUP_SIZE[++g];
    return 1.0 / std::sqrt(double(GROUP_SIZE[g]));
  }
};

// The running standardization of the statistics (Diversity, "The normalization"): each statistic's mean and variance
// over the positions so far, by Welford's method, from a seed of `count` positions with the given means and standard
// deviations. The standard deviation is the population's (the variance divides by the count), as the reference's.
struct DiversityNormalizer {
  static constexpr int N = Diversity::NUM_STATISTICS;
  double count = 0.0;
  double mean[N] = {}, m2[N] = {};  // m2: the sum of the squared differences from the mean
  double weight[N] = {};

  DiversityNormalizer(const double* seedMean, const double* seedSd, double seedCount) : count(seedCount) {
    for(int i = 0; i < N; i++) {
      mean[i] = seedMean[i];
      m2[i] = seedSd[i] * seedSd[i] * seedCount;
      weight[i] = Diversity::weight(i);
    }
  }
  // One more position's statistics.
  void update(const double* x) {
    count += 1.0;
    for(int i = 0; i < N; i++) {
      const double delta = x[i] - mean[i];
      mean[i] += delta / count;
      m2[i] += delta * (x[i] - mean[i]);
    }
  }
  double sd(int i) const { return count > 0.0 ? std::sqrt(m2[i] / count) : 0.0; }
  // The normalized statistics: (x - mean) / max(sd, SIGMA_FLOOR), clipped to [-CLIP, CLIP], times the group's weight.
  void z(const double* x, double* out) const {
    for(int i = 0; i < N; i++) {
      const double v = (x[i] - mean[i]) / std::max(sd(i), Diversity::SIGMA_FLOOR);
      out[i] = std::max(-Diversity::CLIP, std::min(Diversity::CLIP, v)) * weight[i];
    }
  }
};

}  // namespace blueprints
