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
//   (0, 0) cell on the anchor hex and no turn, every hex that it fills free, and one of them next to a hex of R.
// - Free at step k: an empty hex, or a normal cell or a star of P that a hex filled at a step up to k touches (a
//   placement captured it). A hex that a step filled stays free: R holds the placements of every chain at once, and
//   one chain does not fill the hexes of another. P's walls and O's cells of now are never free.
// - k(s) is the first step at which a hex that the step filled touches s.
// The values: k from 1 to CAP; BEYOND when CAP steps do not reach s and R still grows; NEVER when a step up to CAP adds
// nothing to R, so that no later step reaches s; UNKNOWN when the work budget runs out before s has one of those
// values; UNAVAILABLE for each star of P while O's pick waits for its drop (O's picked blueprint lacks a cell).
//
// The board has no edge: the kernel works on a box of the cells with a margin of empty hexes, CAP * (D + 1) + 1 on
// each side, D the most hexes between two cells of a blueprint. A placement that touches R reaches at most D + 1 hexes
// past it, so every placement of the CAP steps lies in the box: neither the view's edge nor the rules board's border
// stops a chain.
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
// - Only the new placements of each step: a placement that is legal at step k + 1 and not at step k has a cell next to
//   a hex that step k filled, or on a cell that step k captured, which touches such a hex. So step k + 1 computes the
//   anchors in the rows around the box of the hexes that step k filled. A placement that was legal before filled its
//   hexes then.
// - The early stop: when each star has its value, after CAP steps, or when a step fills nothing.
// - A deterministic budget of work units (StarReach::BUDGET): a unit is one word of the run of a shape's cell in a
//   step. A step whose units would take the total past the budget does not run, and the stars without a value are
//   UNKNOWN.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gamerules.h"

namespace blueprints {

struct StarReach {
  static constexpr int CAP = 6;
  static constexpr int8_t UNAVAILABLE = -2, UNKNOWN = -1, BEYOND = CAP + 1, NEVER = CAP + 2;
  // The work units of one player's reach, before its stars become UNKNOWN: about twice the most of the positions that
  // checked the kernel (games/blueprints/tools/star_reach_check.py): at most 81,270 in bpj5's self-play and 507,837
  // in people's games, where long blueprints widen the box's margin.
  static constexpr int64_t BUDGET = 1000000;
  struct Star {
    int16_t x, y;  // a point of the rules board
    int8_t k;
  };
  // Each player's stars, in the order of the rules board's rows, and their values.
  std::vector<Star> stars[2];
  int64_t work[2] = {0, 0};  // the units of each player's reach
};

// A blueprint's shape for the reach: its cells (q, r) from its (0, 0) cell, without their types, sorted by r then q.
struct ReachShape {
  int8_t n = 0;
  int8_t q[MAX_CELLS] = {}, r[MAX_CELLS] = {};
  int qmin = 0, qmax = 0, rmin = 0, rmax = 0;
  int diameter = 0;  // the most hexes between two cells

  explicit ReachShape(const Blueprint& b) {
    int order[MAX_CELLS];
    for(int i = 0; i < b.n; i++)
      order[i] = i;
    std::sort(order, order + b.n, [&b](int i, int j) { return b.r[i] != b.r[j] ? b.r[i] < b.r[j] : b.q[i] < b.q[j]; });
    n = b.n;
    for(int i = 0; i < n; i++) {
      q[i] = b.q[order[i]];
      r[i] = b.r[order[i]];
    }
    for(int i = 0; i < n; i++) {
      qmin = i ? std::min<int>(qmin, q[i]) : q[i];
      qmax = i ? std::max<int>(qmax, q[i]) : q[i];
      rmin = i ? std::min<int>(rmin, r[i]) : r[i];
      rmax = i ? std::max<int>(rmax, r[i]) : r[i];
      for(int j = 0; j < i; j++) {
        const int dq = q[i] - q[j], dr = r[i] - r[j];
        diameter = std::max(diameter, (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2);
      }
    }
  }
  bool operator==(const ReachShape& o) const {
    return n == o.n && std::equal(q, q + n, o.q) && std::equal(r, r + n, o.r);
  }
};

class StarReachKernel {
 public:
  // The reach of each player's stars whose `hands[player]` is not null: by the other player's blueprints
  // `hands[player][0 .. counts[player] - 1]`. available[player] false: the player's stars are UNAVAILABLE.
  void compute(const Position& p, const Blueprint* const hands[2], const int counts[2], const bool available[2],
               int64_t budget, StarReach& out) {
    for(int player = 0; player < 2; player++) {
      out.stars[player].clear();
      out.work[player] = 0;
      shapes[player].clear();
      if(hands[player])
        for(int i = 0; i < counts[player]; i++)
          if(hands[player][i].n > 0) {
            const ReachShape s(hands[player][i]);
            if(std::find(shapes[player].begin(), shapes[player].end(), s) == shapes[player].end())
              shapes[player].push_back(s);
          }
    }
    if(!scan(p, hands))
      return;
    for(int player = 0; player < 2; player++) {
      if(!hands[player])
        continue;
      for(const Cell& c : starsOf[player])
        out.stars[player].push_back({int16_t(c.x + x0), int16_t(c.y + y0), StarReach::UNKNOWN});
      if(!available[player])
        for(StarReach::Star& s : out.stars[player])
          s.k = StarReach::UNAVAILABLE;
      else if(!starsOf[player].empty())
        out.work[player] = reach(player, budget, out.stars[player]);
    }
    unscan();
  }

 private:
  struct Cell {
    int x, y;  // a hex of the box
  };
  struct Rect {  // hexes of the box: rows y0..y1, x0..x1
    int y0 = 0, y1 = -1, x0 = 0, x1 = -1;
    bool empty() const { return y1 < y0 || x1 < x0; }
    void add(int x, int y) {
      if(empty()) {
        x0 = x1 = x;
        y0 = y1 = y;
      } else {
        x0 = std::min(x0, x);
        x1 = std::max(x1, x);
        y0 = std::min(y0, y);
        y1 = std::max(y1, y);
      }
    }
    void add(const Rect& o) {
      if(!o.empty()) {
        add(o.x0, o.y0);
        add(o.x1, o.y1);
      }
    }
  };

  struct Window {  // the rows ay0..ay1 of a shape's anchors in a step
    int ay0, ay1;
  };

  std::vector<ReachShape> shapes[2];
  std::vector<Window> windows;
  // The box: the hex (x, y) of the box is the point (x + x0, y + y0) of the rules board, the bit x & 63 of the word
  // x >> 6 of row y. A plane holds its rows in turn, each a left guard word, `words` words and a right guard word
  // (stride words), after a word of padding and before another, all 0, so that a pass over rows reads 0 beyond them.
  int x0 = 0, y0 = 0, rows = 0, words = 0, stride = 0;
  // The board's planes, for both players: the cells, each player's cells, and each player's cells that are not walls.
  std::vector<uint64_t> occupied, cellsOf[2], soft[2];
  // A reach's planes: R, the hexes that touch R, the hexes that are not free, the hexes that a step fills, and the
  // anchors of a shape's placements.
  std::vector<uint64_t> region, touching, blocked, filled, anchors;
  std::vector<Cell> starsOf[2];
  Rect cellsRect, cellsRectOf[2];
  std::vector<uint64_t> column;      // a step's new hexes, each word's OR over the rows
  std::vector<uint64_t> blockedRun;  // a shape's anchors in a window's rows where a cell lands on a hex not free

  // Row y from its left guard word, and from its first word.
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

  // Lays out the box around the cells and sets the board's planes. False when the board has no cell.
  bool scan(const Position& p, const Blueprint* const hands[2]) {
    static thread_local std::vector<int32_t> found;  // the points of the rules board that hold a cell
    found.clear();
    const Box& used = p.cell.box();
    if(!used.any)
      return false;
    Box box;
    for(int y = used.y0; y <= used.y1; y++) {
      const int8_t* line = p.cell.row(y);
      int x = used.x0;
      for(; x + 8 <= used.x1 + 1; x += 8) {
        uint64_t eight;
        std::memcpy(&eight, line + x, 8);
        if(eight)
          for(int i = 0; i < 8; i++)
            if(line[x + i]) {
              found.push_back(y * RULES_LEN + x + i);
              box.add(x + i, y);
            }
      }
      for(; x <= used.x1; x++)
        if(line[x]) {
          found.push_back(y * RULES_LEN + x);
          box.add(x, y);
        }
    }
    if(!box.any)
      return false;
    int diameter = 0;
    for(int player = 0; player < 2; player++)
      if(hands[player])
        for(const ReachShape& s : shapes[player])
          diameter = std::max(diameter, s.diameter);
    const int margin = StarReach::CAP * (diameter + 1) + 1;
    x0 = box.x0 - margin;
    y0 = box.y0 - margin;
    rows = box.y1 - box.y0 + 1 + 2 * margin;
    words = (box.x1 - box.x0 + 1 + 2 * margin + 63) / 64;
    stride = words + 2;
    const size_t size = size_t(rows) * size_t(stride) + 2;
    for(std::vector<uint64_t>* plane : {&occupied, &cellsOf[0], &cellsOf[1], &soft[0], &soft[1], &region, &touching,
                                        &blocked, &filled, &anchors})
      if(plane->size() < size)
        plane->resize(size, 0);  // all 0 between calls, so a new layout reads 0 too
    column.resize(size_t(words));
    blockedRun.resize(size);
    cellsRect = Rect();
    for(int player = 0; player < 2; player++) {
      cellsRectOf[player] = Rect();
      starsOf[player].clear();
    }
    for(int32_t point : found) {
      const int8_t v = p.cell[point];
      const int x = point % RULES_LEN - x0, y = point / RULES_LEN - y0, owner = cell_owner(v), type = cell_type(v);
      set(occupied, x, y);
      set(cellsOf[owner], x, y);
      if(type != WALL)
        set(soft[owner], x, y);
      if(type == STAR)
        starsOf[owner].push_back({x, y});
      cellsRect.add(x, y);
      cellsRectOf[owner].add(x, y);
    }
    return true;
  }

  // Clears the board's planes.
  void clear(std::vector<uint64_t>& plane, const Rect& r) {
    if(r.empty())
      return;
    const int w0 = word_of(r.x0), w1 = word_of(r.x1);
    for(int y = std::max(r.y0, 0); y <= std::min(r.y1, rows - 1); y++)
      std::memset(row(plane, y) + w0, 0, sizeof(uint64_t) * size_t(w1 - w0 + 1));
  }
  // Clears the rows first..last of a plane, their guard words too.
  void clear_rows(std::vector<uint64_t>& plane, int first, int last) {
    std::memset(flat(plane, first), 0, sizeof(uint64_t) * size_t(last - first + 1) * size_t(stride));
  }
  void unscan() {
    for(std::vector<uint64_t>* plane : {&occupied, &cellsOf[0], &cellsOf[1], &soft[0], &soft[1]})
      clear(*plane, cellsRect);
  }

  // f(y, w, near): for each word w of each row y around the rectangle (1 hex more on each side), the hexes of the word
  // that touch a hex of the plane src. Outside the rectangle, src is 0.
  template<class F>
  void for_each_near(std::vector<uint64_t>& src, const Rect& r, F&& f) {
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

  // The reach of the player's stars (stars: their values, in the order of starsOf), by the other player's shapes.
  // Returns the work units.
  int64_t reach(int player, int64_t budget, std::vector<StarReach::Star>& stars) {
    const int other = 1 - player;
    const std::vector<ReachShape>& shapesOf = shapes[player];
    const std::vector<Cell>& starCells = starsOf[player];
    int left = int(starCells.size());
    int64_t work = 0;
    Rect news = cellsRectOf[other];
    if(news.empty() || shapesOf.empty()) {
      for(StarReach::Star& s : stars)
        s.k = StarReach::NEVER;
      return 0;
    }
    // R_0: the other player's cells. The hexes not free: every cell. The hexes that touch R_0.
    const int cw0 = word_of(cellsRect.x0), cw1 = word_of(cellsRect.x1);
    for(int y = cellsRect.y0; y <= cellsRect.y1; y++)
      for(int w = cw0; w <= cw1; w++) {
        row(region, y)[w] = row(cellsOf[other], y)[w];
        row(blocked, y)[w] = row(occupied, y)[w];
      }
    for_each_near(cellsOf[other], news, [&](int y, int w, uint64_t near) { row(touching, y)[w] |= near; });
    Rect dirty = cellsRect;
    dirty.add(news.x0 - 1, news.y0 - 1);
    dirty.add(news.x1 + 1, news.y1 + 1);
    int8_t rest = StarReach::BEYOND;
    for(int k = 1; k <= StarReach::CAP && left > 0; k++) {
      // The rows of each shape's anchors whose placements have a cell next to `news`, the shape's cells in the box.
      windows.resize(shapesOf.size());
      int64_t units = 0;
      for(size_t i = 0; i < shapesOf.size(); i++) {
        const ReachShape& s = shapesOf[i];
        Window& win = windows[i];
        win.ay0 = std::max(news.y0 - 1 - s.rmax, -s.rmin);
        win.ay1 = std::min(news.y1 + 1 - s.rmin, rows - 1 - s.rmax);
        if(win.ay1 >= win.ay0)
          units += int64_t(win.ay1 - win.ay0 + 1) * stride * s.n;
      }
      if(work + units > budget) {
        rest = StarReach::UNKNOWN;
        break;
      }
      work += units;
      Rect fill;  // the rows that this step's placements fill
      for(size_t i = 0; i < shapesOf.size(); i++)
        place(shapesOf[i], windows[i], fill);
      // The new hexes of R: the filled hexes that R does not hold. Their box is the next step's `news`.
      Rect added;
      if(!fill.empty()) {
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
          added = Rect{firstRow, lastRow, first * 64 + __builtin_ctzll(column[size_t(first)]),
                       last * 64 + 63 - __builtin_clzll(column[size_t(last)])};
        }
      }
      if(added.empty()) {
        clear_rows(filled, fill.y0, fill.y1);
        rest = StarReach::NEVER;
        break;
      }
      // The hexes next to the new ones: they touch R now, and the player's cells there are captured, so free.
      for_each_near(filled, added, [&](int y, int w, uint64_t near) {
        row(touching, y)[w] |= near;
        row(blocked, y)[w] &= ~(row(soft[player], y)[w] & near);
      });
      for(size_t i = 0; i < starCells.size(); i++) {
        StarReach::Star& star = stars[i];
        if(star.k != StarReach::UNKNOWN)
          continue;
        const int x = starCells[i].x, y = starCells[i].y;
        // The star is in the box's margin, so its neighbours are in the box.
        if(test(filled, x - 1, y) || test(filled, x + 1, y) || test(filled, x, y - 1) || test(filled, x + 1, y - 1) ||
           test(filled, x - 1, y + 1) || test(filled, x, y + 1)) {
          star.k = int8_t(k);
          left--;
        }
      }
      clear_rows(filled, fill.y0, fill.y1);
      dirty.add(added.x0 - 1, added.y0 - 1);
      dirty.add(added.x1 + 1, added.y1 + 1);
      news = added;
    }
    for(StarReach::Star& s : stars)
      if(s.k == StarReach::UNKNOWN)
        s.k = rest;
    clear(region, dirty);
    clear(touching, dirty);
    clear(blocked, dirty);
    return work;
  }

  // The legal placements of a shape with their anchors in the window's rows: their hexes go into `filled`, and fill
  // grows to hold the rows written. Each pass runs over the window's rows as one run of words, guard words included:
  // a shift by less than 64 hexes moves a row's bits into its guard words at most, which stay 0 (no placement of the
  // CAP steps reaches the box's margin).
  void place(const ReachShape& s, const Window& win, Rect& fill) {
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
      fill.add(0, win.ay0 + s.rmin);
      fill.add(0, win.ay1 + s.rmax);
    }
    std::memset(a, 0, sizeof(uint64_t) * size_t(n));
  }
};

// Each player's stars and their reach by the other player's hand. While a pick waits, the reach of the stars of the
// player who does not pick is UNAVAILABLE: the picker's blueprint lacks the picked cell.
inline void star_reach(const Position& p, StarReach& out, int64_t budget = StarReach::BUDGET) {
  static thread_local StarReachKernel kernel;
  const Blueprint* hands[2] = {p.hand[1], p.hand[0]};
  const int counts[2] = {HAND, HAND};
  const int picker = p.picked >= 0 ? turn_of(p.ply) : -1;
  const bool available[2] = {picker != 1, picker != 0};
  kernel.compute(p, hands, counts, available, budget, out);
}

// One player's stars and their reach by the given blueprints of the other player (a test's hand: any number, each in
// the site's frame or not). The other player's entry of out stays empty.
inline void star_reach(const Position& p, int player, const Blueprint* hand, int count, StarReach& out,
                       int64_t budget = StarReach::BUDGET) {
  static thread_local StarReachKernel kernel;
  const Blueprint* hands[2] = {nullptr, nullptr};
  int counts[2] = {0, 0};
  hands[player] = hand;
  counts[player] = count;
  const bool available[2] = {true, true};
  kernel.compute(p, hands, counts, available, budget, out);
}

}  // namespace blueprints
