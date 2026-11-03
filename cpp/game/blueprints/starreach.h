#pragma once
// Blueprints: the opponent's placements to reach each star, the measure A of the engine repository's
// games/blueprints/tools/star_reach.py (games/blueprints/docs/decisions/port.md, "the opponent's moves to reach a
// star"), capped at StarReach::CAP placements: a bounded kernel for a candidate input. No input plane reads it, and no
// inputs version changes.
//
// For a player P and each star s of P: k(s), the least placements of the opponent O, in a chain from O's cells now,
// whose last placement touches s and so captures it. It ignores P's replies and O's edits.
// - R_0 is O's cells. Step k + 1 makes every placement of O that is legal against the board of step k, and R_{k+1} is
//   R_k with every hex that those placements fill.
// - A placement: one of O's blueprints as O's hand holds it (each shape once: the cells, not their types), with its
//   (0, 0) cell on the anchor hex and no turn, every hex that it fills free, and one of them touching a hex of R.
// - Free at step k: an empty hex, or a normal cell or a star of P that a hex filled at a step up to k touches (a
//   placement captured it). A hex that a step filled stays free: R holds the placements of every chain at once, and
//   one chain does not fill the hexes of another. P's walls and O's cells of now are never free.
// - k(s) is the first step at which a hex that the step filled touches s.
// The values:
// - k from 1 to CAP.
// - BEYOND: CAP steps do not reach s, and R still grows.
// - NEVER: a step up to CAP adds nothing to R, so no later step reaches s.
// - UNKNOWN: the work budget runs out before s has one of the values above.
// - UNAVAILABLE: O's pick waits for its drop (O's picked blueprint lacks a cell), for each star of P.
//
// The board has no edge: each player's reach works on a box of the cells with a margin of empty hexes, CAP * (D + 1) +
// 1 on each side, D the most hexes between two cells of a blueprint of O. A placement that touches R reaches at most
// D + 1 hexes past it, so every placement of the CAP steps lies in the box: neither the view's edge nor the rules
// board's border stops a chain. P's own blueprints do not widen the box.
//
// The cost is bounded:
// - Packed bit rows: the box's rows of the rules board (y), each a row of 64-bit words along x, with a guard word of 0
//   on each side, so that a shift along x by less than 64 hexes reads 0 beyond the row. A shift along y is another
//   row. The hex (x, y) touches (x +- 1, y), (x, y - 1), (x + 1, y - 1), (x - 1, y + 1) and (x, y + 1) (DIRS).
// - Long passes: a shape's anchors take one pass over a run of whole rows for each of its cells (whether the cell
//   touches R, whether it lands on a hex that is not free, the fill), a loop that the compiler vectorizes. A shift
//   moves a row's bits into its guard words at most, and no placement of the CAP steps reaches the margin, so the guard
//   words stay 0. Rows of 2 to 6 words make short loops a row, whose overhead costs more than the words they skip.
// - Reusable scratch storage: the planes of one thread, all 0 between calls (each call clears what it wrote).
// - Each blueprint shape once: a hand's equal shapes are one.
// - Only the new placements of each step: a placement that is legal at step k + 1 and not at step k has a cell that
//   touches a hex that step k filled, or a cell on a cell that step k captured, which touches such a hex. So step k + 1
//   computes the anchors in the rows around the box of the hexes that step k filled. A placement that was legal before
//   filled its hexes then.
// - The early stop: when each star has its value, after CAP steps, or when a step fills nothing.
// - A deterministic budget of work units (StarReach::BUDGET): a unit is one word of the run of a shape's cell in a
//   step. A step whose units would take the total past the budget does not run, and the stars without a value are
//   UNKNOWN.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gamerules.h"

namespace blueprints {

struct StarReach {
  static constexpr int CAP = 6;
  static constexpr int8_t UNAVAILABLE = -2, UNKNOWN = -1, BEYOND = CAP + 1, NEVER = CAP + 2;
  // The work units of one player's reach, before its stars become UNKNOWN: about twice the most of the positions that
  // checked the kernel (games/blueprints/tools/star_reach_check.py), where long blueprints widen the box's margin.
  static constexpr int64_t BUDGET = 1000000;
  struct Star {
    int16_t x, y;  // a point of the rules board
    int8_t k;
  };
  // Each player's stars, in the order of the rules board's rows, and their values.
  std::vector<Star> stars[2];
  int64_t work[2] = {0, 0};  // the units of each player's reach
};

// A blueprint's shape for the reach: its cells (q, r) without their types, in the site's frame (Blueprint::normalize)
// and sorted by r then q (sorted_cells). A placement fills the same hexes in any frame, and in this one every anchor is
// a cell of the shape: 0 <= r <= rmax <= diameter and |q| <= diameter, so that every anchor of the CAP steps lies in
// the box. The rules keep a blueprint connected, so |q| <= MAX_CELLS - 1 < 64, the most that a shift of a row takes. A
// test's hand need not be connected: the binding refuses a cell with |q| or |r| over 20, so |q| <= 40 here.
static_assert(MAX_CELLS - 1 < 64, "a shape's cell is less than 64 hexes from its (0, 0) cell along x");
struct ReachShape {
  int8_t n = 0;
  int8_t q[MAX_CELLS] = {}, r[MAX_CELLS] = {};
  int rmax = 0;
  int diameter = 0;  // the most hexes between two cells

  explicit ReachShape(const Blueprint& b) {
    const Blueprint s = sorted_cells(b);
    n = s.n;
    for(int i = 0; i < n; i++) {
      q[i] = int8_t(s.q[i] - s.q[0]);
      r[i] = int8_t(s.r[i] - s.r[0]);
      for(int j = 0; j < i; j++) {
        const int dq = q[i] - q[j], dr = r[i] - r[j];
        diameter = std::max(diameter, (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2);
      }
    }
    rmax = n > 0 ? r[n - 1] : 0;
  }
  bool operator==(const ReachShape& o) const {
    return n == o.n && std::equal(q, q + n, o.q) && std::equal(r, r + n, o.r);
  }
};

// The other player's blueprints that reach a player's stars: count blueprints from `blueprints` (none: the reach of
// that player's stars is not computed). Not available: the player's stars are UNAVAILABLE.
struct ReachHand {
  const Blueprint* blueprints = nullptr;
  int count = 0;
  bool available = true;
};

class StarReachKernel {
 public:
  // The reach of each player's stars by hands[player].
  void compute(const Position& p, const ReachHand hands[2], int64_t budget, StarReach& out) {
    collect(p);
    for(int player = 0; player < 2; player++) {
      const ReachHand& hand = hands[player];
      std::vector<StarReach::Star>& stars = out.stars[player];
      stars.clear();
      out.work[player] = 0;
      if(!hand.blueprints)
        continue;
      for(int32_t point : cellPoints)
        if(cell_owner(p.cell[point]) == player && cell_type(p.cell[point]) == STAR)
          stars.push_back({int16_t(point % RULES_LEN), int16_t(point / RULES_LEN),
                           hand.available ? StarReach::UNKNOWN : StarReach::UNAVAILABLE});
      if(!hand.available || stars.empty())
        continue;
      shapes.clear();
      for(int i = 0; i < hand.count; i++)
        if(hand.blueprints[i].n > 0) {
          const ReachShape s(hand.blueprints[i]);
          if(std::find(shapes.begin(), shapes.end(), s) == shapes.end())
            shapes.push_back(s);
        }
      out.work[player] = reach(p, player, budget, stars);
    }
  }

 private:
  struct Window {  // the rows ay0..ay1 of a shape's anchors in a step
    int ay0, ay1;
  };

  std::vector<int32_t> cellPoints;  // the points of the rules board that hold a cell, in the order of its rows
  Box cells;                        // their box on the rules board
  std::vector<ReachShape> shapes;   // the other player's shapes, each once
  std::vector<Window> windows;
  // The box: the hex (x, y) of the box is the point (x + x0, y + y0) of the rules board, the bit x & 63 of the word
  // x >> 6 of row y. A plane holds its rows in turn, each a guard word and `words` words (stride words), then a last
  // guard word, after a word of padding and before another. A row's guard word is the right guard of the row before:
  // a shift by less than 64 hexes reads only the next word on each side. The guard and padding words stay 0.
  int x0 = 0, y0 = 0, rows = 0, words = 0, stride = 0;
  Box cellsInBox;  // the box of the cells, in the box
  Box theirs;      // the box of the other player's cells, in the box
  // The planes of a reach: the player's cells that a placement can capture (not walls), R, the hexes that touch R, the
  // hexes that are not free, the hexes that a step fills, and the anchors of a shape's placements.
  std::vector<uint64_t> capturable, region, touching, blocked, filled, anchors;
  std::vector<uint64_t> column;      // a step's new hexes, each word's OR over the rows
  std::vector<uint64_t> blockedRun;  // a shape's anchors in a window's rows where a cell lands on a hex not free

  // Row y from its guard word, and from its first word.
  uint64_t* flat(std::vector<uint64_t>& plane, int y) { return plane.data() + 1 + size_t(y) * size_t(stride); }
  uint64_t* row(std::vector<uint64_t>& plane, int y) { return flat(plane, y) + 1; }
  void set(std::vector<uint64_t>& plane, int x, int y) { row(plane, y)[x >> 6] |= uint64_t(1) << (x & 63); }
  bool test(std::vector<uint64_t>& plane, int x, int y) { return row(plane, y)[x >> 6] >> (x & 63) & 1; }

  // The bits x + s of a row for x in word w, 0 beyond the row (|s| < 64).
  static uint64_t shifted(const uint64_t* r, int w, int s) {
    if(s > 0)
      return (r[w] >> s) | (r[w + 1] << (64 - s));
    if(s < 0)
      return (r[w] << -s) | (r[w - 1] >> (64 + s));
    return r[w];
  }
  // out[j] |= the bits x + s of the run src for x in word j, for j in 0..n-1, which reads src[-1] and src[n] too
  // (|s| < 64). One loop over the run, so that the compiler vectorizes it.
  static void or_shifted(uint64_t* __restrict out, const uint64_t* __restrict src, int n, int s) {
    if(s > 0) {
      for(int j = 0; j < n; j++)
        out[j] |= (src[j] >> s) | (src[j + 1] << (64 - s));
    } else if(s < 0) {
      for(int j = 0; j < n; j++)
        out[j] |= (src[j] << -s) | (src[j - 1] >> (64 + s));
    } else {
      for(int j = 0; j < n; j++)
        out[j] |= src[j];
    }
  }
  int word_of(int x) const { return std::clamp(x, 0, words * 64 - 1) >> 6; }

  // The points of the rules board that hold a cell, and their box.
  void collect(const Position& p) {
    cellPoints.clear();
    cells = Box();
    const Box& used = p.cell.box();
    if(!used.any)
      return;
    for(int y = used.y0; y <= used.y1; y++) {
      const int8_t* line = p.cell.row(y);
      int x = used.x0;
      for(; x + 8 <= used.x1 + 1; x += 8) {
        uint64_t eight;
        std::memcpy(&eight, line + x, 8);
        if(eight)
          for(int i = 0; i < 8; i++)
            if(line[x + i]) {
              cellPoints.push_back(y * RULES_LEN + x + i);
              cells.add(x + i, y);
            }
      }
      for(; x <= used.x1; x++)
        if(line[x]) {
          cellPoints.push_back(y * RULES_LEN + x);
          cells.add(x, y);
        }
    }
  }

  // Lays out the box of the cells with the margin, and sets the planes of the board for the player's reach: R_0 (the
  // other player's cells), the hexes that are not free (every cell), and the player's cells that are not walls.
  void layout(const Position& p, int player, int margin) {
    x0 = cells.x0 - margin;
    y0 = cells.y0 - margin;
    rows = cells.y1 - cells.y0 + 1 + 2 * margin;
    words = (cells.x1 - cells.x0 + 1 + 2 * margin + 63) / 64;
    stride = words + 1;
    const size_t size = size_t(rows) * size_t(stride) + 3;
    for(std::vector<uint64_t>* plane : {&capturable, &region, &touching, &blocked, &filled, &anchors})
      if(plane->size() < size)
        plane->resize(size, 0);  // all 0 between calls, so a new layout reads 0 too
    column.resize(size_t(words));
    blockedRun.resize(size);
    cellsInBox = Box();
    cellsInBox.add(margin, margin);
    cellsInBox.add(margin + cells.x1 - cells.x0, margin + cells.y1 - cells.y0);
    theirs = Box();
    for(int32_t point : cellPoints) {
      const int8_t v = p.cell[point];
      const int x = point % RULES_LEN - x0, y = point / RULES_LEN - y0;
      set(blocked, x, y);
      if(cell_owner(v) != player) {
        set(region, x, y);
        theirs.add(x, y);
      } else if(cell_type(v) != WALL) {
        set(capturable, x, y);
      }
    }
  }

  // Clears a plane's words that hold the box's hexes.
  void clear(std::vector<uint64_t>& plane, const Box& r) {
    if(!r.any)
      return;
    const int w0 = word_of(r.x0), w1 = word_of(r.x1);
    for(int y = std::max<int>(r.y0, 0); y <= std::min<int>(r.y1, rows - 1); y++)
      std::memset(row(plane, y) + w0, 0, sizeof(uint64_t) * size_t(w1 - w0 + 1));
  }
  // Clears the box's whole rows of a plane, their guard words on both sides too.
  void clear_rows(std::vector<uint64_t>& plane, const Box& r) {
    if(r.any)
      std::memset(flat(plane, r.y0), 0, sizeof(uint64_t) * (size_t(r.y1 - r.y0 + 1) * size_t(stride) + 1));
  }

  // f(y, w, near): for each word w of each row y around the box (1 hex more on each side), the hexes of the word that
  // touch a hex of the plane src. Outside the box, src is 0.
  template<class F>
  void for_each_near(std::vector<uint64_t>& src, const Box& r, F&& f) {
    const int w0 = word_of(r.x0 - 1), w1 = word_of(r.x1 + 1);
    for(int y = std::max(r.y0 - 1, 0); y <= std::min(r.y1 + 1, rows - 1); y++) {
      const uint64_t* mid = row(src, y);
      const uint64_t* up = y > 0 ? row(src, y - 1) : nullptr;
      const uint64_t* down = y + 1 < rows ? row(src, y + 1) : nullptr;
      for(int w = w0; w <= w1; w++) {
        uint64_t near = shifted(mid, w, -1) | shifted(mid, w, 1);
        if(up)
          near |= up[w] | shifted(up, w, 1);
        if(down)
          near |= down[w] | shifted(down, w, -1);
        f(y, w, near);
      }
    }
  }

  // The reach of the player's stars (their points of the rules board and their values, UNKNOWN), by `shapes`.
  // Returns the work units.
  int64_t reach(const Position& p, int player, int64_t budget, std::vector<StarReach::Star>& stars) {
    int diameter = 0;
    for(const ReachShape& s : shapes)
      diameter = std::max(diameter, s.diameter);
    layout(p, player, StarReach::CAP * (diameter + 1) + 1);
    int left = int(stars.size());
    int64_t work = 0;
    int8_t unreached = StarReach::BEYOND;  // the value of the stars that no step reaches
    Box dirty = cellsInBox;                // the hexes whose words the reach wrote
    Box gained = theirs;                   // the hexes that R gained at the step before
    if(shapes.empty() || !gained.any) {
      unreached = StarReach::NEVER;
      left = 0;
    } else {
      for_each_near(region, gained, [&](int y, int w, uint64_t near) { row(touching, y)[w] |= near; });
      dirty.add(gained.x0 - 1, gained.y0 - 1);
      dirty.add(gained.x1 + 1, gained.y1 + 1);
    }
    for(int k = 1; k <= StarReach::CAP && left > 0; k++) {
      // The rows of each shape's anchors whose placements have a cell that touches `gained`, the shape's cells in the
      // box.
      windows.resize(shapes.size());
      int64_t units = 0;
      for(size_t i = 0; i < shapes.size(); i++) {
        const ReachShape& s = shapes[i];
        Window& win = windows[i];
        win.ay0 = std::max(gained.y0 - 1 - s.rmax, 0);
        win.ay1 = std::min(gained.y1 + 1, rows - 1 - s.rmax);
        if(win.ay1 >= win.ay0)
          units += int64_t(win.ay1 - win.ay0 + 1) * stride * s.n;
      }
      if(work + units > budget) {
        unreached = StarReach::UNKNOWN;
        break;
      }
      work += units;
      Box fill;  // the rows that this step's placements fill
      for(size_t i = 0; i < shapes.size(); i++)
        place(shapes[i], windows[i], fill);
      // The new hexes of R: the filled hexes that R does not hold. Their box is the next step's `gained`.
      Box added;
      if(fill.any) {
        std::fill(column.begin(), column.end(), 0);
        int firstRow = -1, lastRow = -1;
        for(int y = fill.y0; y <= fill.y1; y++) {
          uint64_t* f = row(filled, y);
          uint64_t* rg = row(region, y);
          uint64_t any = 0;
          for(int w = 0; w < words; w++) {
            const uint64_t v = f[w] & ~rg[w];
            f[w] = v;
            rg[w] |= v;
            column[size_t(w)] |= v;
            any |= v;
          }
          if(any) {
            firstRow = firstRow < 0 ? y : firstRow;
            lastRow = y;
          }
        }
        if(firstRow >= 0) {
          int first = 0, last = words - 1;
          while(!column[size_t(first)])
            first++;
          while(!column[size_t(last)])
            last--;
          added.add(first * 64 + __builtin_ctzll(column[size_t(first)]), firstRow);
          added.add(last * 64 + 63 - __builtin_clzll(column[size_t(last)]), lastRow);
        }
      }
      if(!added.any) {
        clear_rows(filled, fill);
        unreached = StarReach::NEVER;
        break;
      }
      // The hexes that touch the new ones: they touch R now, and the player's cells there are captured, so free.
      for_each_near(filled, added, [&](int y, int w, uint64_t near) {
        row(touching, y)[w] |= near;
        row(blocked, y)[w] &= ~(row(capturable, y)[w] & near);
      });
      for(StarReach::Star& star : stars) {
        if(star.k != StarReach::UNKNOWN)
          continue;
        const int x = star.x - x0, y = star.y - y0;
        // The star is a cell, at least the margin from the box's edge, so the six hexes that touch it are in the box.
        if(test(filled, x - 1, y) || test(filled, x + 1, y) || test(filled, x, y - 1) || test(filled, x + 1, y - 1) ||
           test(filled, x - 1, y + 1) || test(filled, x, y + 1)) {
          star.k = int8_t(k);
          left--;
        }
      }
      clear_rows(filled, fill);
      dirty.add(added.x0 - 1, added.y0 - 1);
      dirty.add(added.x1 + 1, added.y1 + 1);
      gained = added;
    }
    for(StarReach::Star& s : stars)
      if(s.k == StarReach::UNKNOWN)
        s.k = unreached;
    clear(region, dirty);
    clear(touching, dirty);
    clear(blocked, dirty);
    clear(capturable, cellsInBox);
    return work;
  }

  // The legal placements of a shape with their anchors in the window's rows: their hexes go into `filled`, and fill
  // grows to hold the rows written. Each pass runs over the window's rows as one run of words, guard words included:
  // a shift by less than 64 hexes moves a row's bits into its guard words at most, which stay 0 (no placement of the
  // CAP steps reaches the box's margin).
  void place(const ReachShape& s, const Window& win, Box& fill) {
    if(win.ay1 < win.ay0)
      return;
    const int n = (win.ay1 - win.ay0 + 1) * stride;
    uint64_t* const a = flat(anchors, win.ay0);
    uint64_t* const b = blockedRun.data();
    // The anchors where a cell touches R, then those where no cell lands on a hex that is not free.
    for(int i = 0; i < s.n; i++)
      or_shifted(a, flat(touching, win.ay0 + s.r[i]), n, s.q[i]);
    std::fill(b, b + n, 0);
    for(int i = 0; i < s.n; i++)
      or_shifted(b, flat(blocked, win.ay0 + s.r[i]), n, s.q[i]);
    uint64_t any = 0;
    for(int j = 0; j < n; j++) {
      a[j] &= ~b[j];
      any |= a[j];
    }
    if(any) {
      for(int i = 0; i < s.n; i++)
        or_shifted(flat(filled, win.ay0 + s.r[i]), a, n, -s.q[i]);
      fill.add(0, win.ay0);
      fill.add(0, win.ay1 + s.rmax);
    }
    std::memset(a, 0, sizeof(uint64_t) * size_t(n));
  }
};

// The kernel of the calling thread, with its scratch planes.
inline StarReachKernel& star_reach_kernel() {
  static thread_local StarReachKernel kernel;
  return kernel;
}

// Each player's stars and their reach by the other player's hand. While a pick waits, the reach of the stars of the
// player who does not pick is UNAVAILABLE: the picker's blueprint lacks the picked cell.
inline void star_reach(const Position& p, StarReach& out, int64_t budget = StarReach::BUDGET) {
  const int picker = p.picked >= 0 ? turn_of(p.ply) : -1;
  const ReachHand hands[2] = {{p.hand[1], HAND, picker != 1}, {p.hand[0], HAND, picker != 0}};
  star_reach_kernel().compute(p, hands, budget, out);
}

// One player's stars and their reach by the given blueprints of the other player (a test's hand: any number, each in
// the site's frame or not). The other player's entry of out stays empty.
inline void star_reach(const Position& p, int player, const Blueprint* hand, int count, StarReach& out,
                       int64_t budget = StarReach::BUDGET) {
  ReachHand hands[2];
  hands[player] = {hand, count, true};
  star_reach_kernel().compute(p, hands, budget, out);
}

}  // namespace blueprints
