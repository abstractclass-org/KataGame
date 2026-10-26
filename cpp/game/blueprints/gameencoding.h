#pragma once
// Blueprints' network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/blueprints/README.md describes the inputs, the targets and the score.
//
// - The inputs are the view: the VIEW x VIEW points of the rules board around the mover's window, in the view's
//   orientation (gamerules.h, View). After the front's on-board plane, from the mover's side: each player's cells by
//   type, each player's frontier (the empty points next to its cells), each player's cells open to a stamp (not walls,
//   next to an empty point), and for each blueprint of each player the anchors in the view where it fits, where it
//   takes at least 1, 2, 3 and 4 enemy stars, and where it scores at least 25, 100 and 400 points (the stars that it
//   takes times the player's stars on the board), and each player's cells that some placement of the other player
//   takes. Then each player's blueprints, drawn at the hand's points (gamerules.h, HAND_X and HAND_Y), where the
//   policy's picks and drops are, and the picked cell, a plane for each blueprint. Then the history: the points that
//   each of the last 4 actions stamped and took.
// - The global inputs: the mover's seat, its moves left in the turn, the game's progress, the stars on the board, the
//   effective scores (with the komi), the pick, each player's cells and stars outside the view, and the mover's
//   effective margin.
// - The komi K is points added to the second player's final margin (KataGo's White), for the whole game. The rules'
//   scores stay the earned ones, and the front's history adds K to the result (BoardHistory::endAndScoreGameNow). The
//   inputs read K as a starting score of the side that it favours, the effective scores S0 + max(-K, 0) and S1 +
//   max(K, 0). No rule reads a score, so a position at komi K plays as the position with those scores at komi 0.
// - The policy: 6 channels a point, the actions of gamerules.h (placements, then the cells of the hand: picks, or
//   drops after a pick).
// - The view's orientation (gamerules.h): the inputs, the policy and the spatial targets are in the view's frame, and
//   the hand's planes show the blueprints in the view's frame (ViewHand). So a position in orientation o gives the
//   inputs of the position turned by R^back(o) around the window's middle in orientation 0.
// - The symmetries: the 4 of KataGo's 8 that keep the hex neighbours (DIRS) on the axial square: the identity, the
//   turn by 180 degrees (3) and two reflections (4, 7). The image maps the rules board around its middle. The window's
//   rule and the recentering round toward zero, so the image's window is the window's image, and with the image's
//   orientation (sym_orientation) a point of the view moves as a pixel. A blueprint of the view's frame moves cell by
//   cell, then goes back to its site frame (its lowest cell at 0,0): so the hand's points, the anchors of the
//   placements and the picks and drops move as pixels and then by each blueprint's shift. The inputs at a symmetry are
//   the inputs of the position's image (sym_position). With the view's 6 turns, the 4 give the hex grid's 12.
// - The training targets, on the row's view: the final board as ownership (C0), the cells captured before the game's
//   end (C1), the stars 4 and 16 moves later (C2-3), where the points are earned (C4), the cells that go in the rest
//   of the mover's turn and in the opponent's next turn (C5-6), and in the global target G76 which player moves
//   next, for the trainer's symmetries (no loss reads it). The value targets and
//   the score targets are KataGo's, from the history's result: the final margin with the komi, where a margin of 0
//   is half a win.

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "gamerules.h"

namespace blueprints {

// The linear part of an allowed symmetry on axial offsets, as GameInterface::symPoint moves the board's points around
// its middle: flip x (q to -q), flip y (r to -r), then transpose (q and r swap).
inline void sym_offset(int& q, int& r, int symmetry) {
  if(symmetry & 0x2)
    q = -q;
  if(symmetry & 0x1)
    r = -r;
  if(symmetry & 0x4)
    std::swap(q, r);
}

// The shift that puts the lowest cell of the image of b's cells (with the cell (extraQ, extraR) when withExtra) at
// (0, 0): the site's frame of the image. (0, 0) for no cell.
inline void frame_shift(const Blueprint& b, int symmetry, bool withExtra, int extraQ, int extraR, int& mq, int& mr) {
  lowest_image(b, [symmetry](int& q, int& r) { sym_offset(q, r, symmetry); }, withExtra, extraQ, extraR, mq, mr);
}

// The orientation of a symmetry's image, so that the image's view is the image of the view: R^o' is the symmetry's
// linear part S times R^o times S's inverse. That is R^o for the identity and the turn by 180 degrees, which commute
// with the turns, and R^back(o) for the reflections (the linear part's determinant -1: an odd number of its flips and
// its transpose).
inline int sym_orientation(int o, int symmetry) { return __builtin_parity(unsigned(symmetry) & 7u) ? back(o) : o; }

// A rules point's image: its pixel on the rules board, as symPoint gives it.
inline void sym_rules(int& x, int& y, int symmetry) { GameInterface::symPoint(x, y, symmetry, RULES_LEN, RULES_LEN); }
inline int32_t sym_rules_index(int32_t index, int symmetry) {
  int x = index % RULES_LEN, y = index / RULES_LEN;
  sym_rules(x, y, symmetry);
  return y * RULES_LEN + x;
}

// How far from the view the inputs read the rules board: a placement lies in the view, and the cells next to its
// cells, or next to the frontier's points, one point beyond.
constexpr int INPUT_REACH = 2;

// The rules board's columns x0 to x1 - 1 and rows y0 to y1 - 1 (on the board) that hold every rules point of the
// position's view and of the points within `reach` of it: the box of the turned corners of the view and its margin.
inline void view_box(const Position& p, int reach, int& x0, int& x1, int& y0, int& y1) {
  const View view = view_of(p);
  const int l = VIEW_RADIUS + reach;
  int q0 = 0, q1 = 0, r0 = 0, r1 = 0;
  for(int corner = 0; corner < 4; corner++) {
    int q = corner & 1 ? l : -l, r = corner & 2 ? l : -l;
    turn(q, r, view.o);
    q0 = std::min(q0, q);
    q1 = std::max(q1, q);
    r0 = std::min(r0, r);
    r1 = std::max(r1, r);
  }
  x0 = std::max(0, view.wx + q0);
  x1 = std::min(RULES_LEN, view.wx + q1 + 1);
  y0 = std::max(0, view.wy + r0);
  y1 = std::min(RULES_LEN, view.wy + r1 + 1);
}

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  // The planes. Each pair of bases names the mover's planes first, then the opponent's.
  static constexpr int MINE = 0, THEIRS = 3;                 // + type
  static constexpr int MY_FRONTIER = 6, THEIR_FRONTIER = 7;
  static constexpr int MY_OPEN = 8, THEIR_OPEN = 9;
  // A player's placements, + level * HAND + k for blueprint k: where it fits (level 0), where it takes at least 1 to
  // STAR_LEVELS enemy stars (levels 1 to 4), and where it scores at least POINT_LEVELS[i] (levels 5 + i).
  static constexpr int STAR_LEVELS = 4;
  static constexpr int NUM_POINT_LEVELS = 3;
  static constexpr int POINT_LEVELS[NUM_POINT_LEVELS] = {25, 100, 400};
  static constexpr int PLACE_PLANES = (1 + STAR_LEVELS + NUM_POINT_LEVELS) * HAND;  // 24
  static constexpr int MY_PLACES = 10, THEIR_PLACES = 34;
  static constexpr int MINE_TAKEN = 58, THEIRS_TAKEN = 59;   // cells that some placement of the other player takes
  static constexpr int MY_HAND = 60, THEIR_HAND = 69;        // + k: cells; + 3 + k: stars; + 6 + k: walls
  // + k: the picked cell of blueprint k. A plane for each blueprint, so that the planes alone give the blueprint's
  // frame (the trainer's augmentation reads no global input).
  static constexpr int PICKED = 78;
  static constexpr int HISTORY_PLANE = 81, NUM_HISTORY_PLANES = 2 * HISTORY;  // action h: + 2h stamped, + 2h + 1 taken
  static constexpr int NUM_SPATIAL_INPUTS = HISTORY_PLANE + NUM_HISTORY_PLANES;  // 89
  static_assert(THEIR_PLACES == MY_PLACES + PLACE_PLANES && MINE_TAKEN == THEIR_PLACES + PLACE_PLANES &&
                MY_HAND == THEIRS_TAKEN + 1 && PICKED == THEIR_HAND + 3 * HAND && HISTORY_PLANE == PICKED + HAND,
                "Blueprints' planes follow one another");
  static constexpr int NUM_GLOBAL_INPUTS = 22;
  static constexpr int MARGIN_INPUT = 21;  // the mover's effective margin / MARGIN_UNIT, not squashed
  // Of the order of the site's margins (median 523 points, 90th percentile 2,101) and of self-play's komis (up to
  // 3,000 points in a new game), so that the margin's input is of the order of 1.
  static constexpr double MARGIN_UNIT = 1000.0;

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    if(options.symmetry == 0)
      write_inputs(p, options.history, set);
    else
      write_inputs(near_image(p, options.symmetry), options.history, set);
  }

  template<class Set>
  static void write_inputs(const Position& p, bool history, Set& set) {
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    const View view = view_of(p);
    auto cell_at = [&](int x, int y) {
      int rx, ry;
      view.rules_point(x, y, rx, ry);
      return at(p, rx, ry);
    };
    if(view.o == 0) {
      // The rules board's rows, as they are: the view's point (0, 0) is the rules point (ox, oy).
      const int ox = view.wx - VIEW_RADIUS, oy = view.wy - VIEW_RADIUS;
      for(int y = 0; y < VIEW; y++) {
        const int ry = y + oy;
        if(ry < 0 || ry >= RULES_LEN)
          continue;
        const int8_t* row = p.cell.row(ry);
        for(int x = std::max(0, -ox); x < std::min(VIEW, RULES_LEN - ox); x++)
          if(const int8_t v = row[x + ox])
            set(x, y, (cell_owner(v) == mover ? MINE : THEIRS) + cell_type(v));
      }
    }
    else
      for(int y = 0; y < VIEW; y++)
        for(int x = 0; x < VIEW; x++)
          if(const int8_t v = cell_at(x, y))
            set(x, y, (cell_owner(v) == mover ? MINE : THEIRS) + cell_type(v));
    // The blueprints in the view's frame.
    ViewHand hands[2];
    for(int player = 0; player < 2; player++)
      view_hand(p, player, hands[player]);
    // The open cells, the frontiers and the placements, from the view's rows of bits (gamerules.h, view_rows).
    ViewRows rows;
    view_rows(p, view, rows);
    auto set_rows = [&](const ViewRow* r, int plane) {
      for(int y = 0; y < VIEW; y++)
        for_each_bit(r[y], [&](int x) { set(x, y, plane); });
    };
    set_rows(rows.open[mover], MY_OPEN);
    set_rows(rows.open[opponent], THEIR_OPEN);
    set_rows(rows.frontier[mover], MY_FRONTIER);
    set_rows(rows.frontier[opponent], THEIR_FRONTIER);
    // The placements of each player in the view, on the board as it is: the mover's when no pick waits for its drop,
    // and the opponent's as if it moved now (its threats, which score with its own stars).
    if(!GameRules::terminal(p)) {
      // What a placement of each player captures (taken_by), on the view's points and one point around it (padded:
      // x + 1, y + 1): 1 on each of the opponent's cells that is not a wall, 2 on the opponent's star. A placement's
      // cells lie in the view, so the cells next to them lie within one point of it.
      constexpr int PADDED = VIEW + 2;
      int8_t take[2][PADDED * PADDED] = {};
      for(int j = 0; j < PADDED; j++)
        for(int i = 0; i < PADDED; i++) {
          const int8_t v = cell_at(i - 1, j - 1);
          if(v && cell_type(v) != WALL)
            take[1 - cell_owner(v)][j * PADDED + i] = int8_t(cell_type(v) == STAR ? 2 : 1);
        }
      for(int player : {mover, opponent}) {
        if(player == mover && p.picked >= 0)
          continue;
        const int places = player == mover ? MY_PLACES : THEIR_PLACES;
        const int taken = player == mover ? THEIRS_TAKEN : MINE_TAKEN;
        for(int k = 0; k < HAND; k++) {
          const Blueprint& b = hands[player].b[k];
          // The points next to b's cells, each once, from b's (0, 0) cell: a placement captures the opponent's cells
          // there (taken_by).
          int8_t nq[6 * MAX_CELLS], nr[6 * MAX_CELLS];
          int nn = 0;
          for(int i = 0; i < b.n; i++)
            for(const auto& d : DIRS) {
              const int q = b.q[i] + d[0], r = b.r[i] + d[1];
              int m = 0;
              while(m < nn && !(nq[m] == q && nr[m] == r))
                m++;
              if(m == nn) {
                nq[nn] = int8_t(q);
                nr[nn] = int8_t(r);
                nn++;
              }
            }
          for_each_fit(b, rows.empty, rows.frontier[player], [&](int x, int y) {
            set(x, y, places + k);
            int stars = 0;
            for(int m = 0; m < nn; m++) {
              const int tx = x + nq[m], ty = y + nr[m];
              const int8_t t = take[player][(ty + 1) * PADDED + tx + 1];
              if(!t)
                continue;
              stars += t == 2;
              if(on_view(tx, ty))
                set(tx, ty, taken);
            }
            for(int level = 1; level <= std::min(stars, STAR_LEVELS); level++)
              set(x, y, places + level * HAND + k);
            const int score = stars * p.stars[player];
            for(int i = 0; i < NUM_POINT_LEVELS; i++)
              if(score >= POINT_LEVELS[i])
                set(x, y, places + (1 + STAR_LEVELS + i) * HAND + k);
          });
        }
      }
    }
    for(int player = 0; player < 2; player++) {
      const int base = player == mover ? MY_HAND : THEIR_HAND;
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = hands[player].b[k];
        for(int i = 0; i < b.n; i++) {
          const int x = b.q[i] + HAND_X, y = b.r[i] + HAND_Y;
          set(x, y, base + k);
          if(b.t[i] == STAR)
            set(x, y, base + HAND + k);
          if(b.t[i] == WALL)
            set(x, y, base + 2 * HAND + k);
        }
      }
    }
    if(p.picked >= 0)
      set(hands[mover].pickQ + HAND_X, hands[mover].pickR + HAND_Y, PICKED + p.picked);
    if(history)
      for(int h = 0; h < HISTORY; h++) {
        const Footprint& f = p.history[h];
        auto mark = [&](int32_t point, int plane) {
          int x, y;
          view.view_point(point % RULES_LEN, point / RULES_LEN, x, y);
          if(on_view(x, y))
            set(x, y, plane);
        };
        for(int i = 0; i < f.nStamped; i++)
          mark(f.stamped[i], HISTORY_PLANE + 2 * h);
        for(int i = 0; i < f.nTaken; i++)
          mark(f.taken[i], HISTORY_PLANE + 2 * h + 1);
      }
  }

  // A score on a log scale, log10(1 + |points| / 10) / 2 with the score's sign: 100 points give 0.52, 1,000 give 1.00
  // and 10,000 give 1.50. The scores grow with the stars on the board.
  static float squash(double points) {
    const double a = std::log10(1.0 + std::fabs(points) / 10.0) / 2.0;
    return float(points < 0 ? -a : a);
  }
  // A player's effective score: its earned score, and the komi K when K favours it (K > 0 favours the second player).
  static double effective_score(const Position& p, int player, double komi) {
    return p.scores[player] + std::max(player == 1 ? komi : -komi, 0.0);
  }
  // The same at every symmetry: no input names a point. The scores are the effective ones, and the margin's input
  // gives the mover's effective margin linearly, where the squash flattens large margins. The cells and stars outside
  // the view count what the planes do not show.
  static void global_inputs(const Position& p, const GameInterface::InputOptions& options, float* out) {
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    for(int i = 0; i < NUM_GLOBAL_INPUTS; i++)
      out[i] = 0.0f;
    const double mine = effective_score(p, mover, options.komi), theirs = effective_score(p, opponent, options.komi);
    out[0] = mover == 0 ? 1.0f : 0.0f;
    out[1] = GameRules::turn_moves_left(p) >= 2 ? 1.0f : 0.0f;
    out[2] = float(p.ply) / PLIES;
    out[3] = float(PLIES - p.ply) / PLIES;
    out[4] = float(p.stars[mover]) / 50.0f;
    out[5] = float(p.stars[opponent]) / 50.0f;
    out[6] = squash(mine);
    out[7] = squash(theirs);
    out[8] = squash(mine - theirs);
    if(p.picked >= 0) {
      out[9] = 1.0f;
      out[10 + p.picked] = 1.0f;
      out[13 + p.pickType] = 1.0f;
      out[16] = p.pickConnected ? 1.0f : 0.0f;
    }
    const View view = view_of(p);
    int cellsIn[2] = {0, 0}, starsIn[2] = {0, 0};
    auto count = [&](int8_t v) {
      if(v) {
        cellsIn[cell_owner(v)]++;
        starsIn[cell_owner(v)] += cell_type(v) == STAR;
      }
    };
    if(view.o == 0) {
      const int ox = view.wx - VIEW_RADIUS, oy = view.wy - VIEW_RADIUS;
      for(int y = std::max(0, oy); y < std::min(RULES_LEN, oy + VIEW); y++)
        for(int x = std::max(0, ox); x < std::min(RULES_LEN, ox + VIEW); x++)
          count(p.cell[y * RULES_LEN + x]);
    }
    else
      for(int y = 0; y < VIEW; y++)
        for(int x = 0; x < VIEW; x++) {
          int rx, ry;
          view.rules_point(x, y, rx, ry);
          count(at(p, rx, ry));
        }
    out[17] = float(p.cells[mover] - cellsIn[mover]) / 100.0f;
    out[18] = float(p.cells[opponent] - cellsIn[opponent]) / 100.0f;
    out[19] = float(p.stars[mover] - starsIn[mover]) / 50.0f;
    out[20] = float(p.stars[opponent] - starsIn[opponent]) / 50.0f;
    out[MARGIN_INPUT] = float((mine - theirs) / MARGIN_UNIT);
  }
  // What the inputs read beyond hash(): the history's points.
  static Hash128 input_hash(const Position& p) {
    Hash128 h;
    for(int t = 0; t < HISTORY; t++) {
      const Footprint& f = p.history[t];
      for(int i = 0; i < f.nStamped; i++)
        h ^= key(KEY_HISTORY + uint64_t(t) * 2 * RULES_AREA + uint64_t(f.stamped[i]), 0);
      for(int i = 0; i < f.nTaken; i++)
        h ^= key(KEY_HISTORY + uint64_t(t) * 2 * RULES_AREA + RULES_AREA + uint64_t(f.taken[i]), 0);
    }
    return h;
  }

  static constexpr int POLICY_CHANNELS = CHANNELS;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action_x(action);
    y = action_y(action);
    channel = action_channel(action);
  }
  static int point_action(int x, int y, int channel) { return action_at(x, y, channel); }

  static constexpr unsigned SYMMETRIES = (1u << 0) | (1u << 3) | (1u << 4) | (1u << 7);

  // The image of everything but the cells on the board: the scores, the counts, the ply and the pick as they are, each
  // blueprint's cells at their images in the site's frame (the rest of a picked blueprint and the picked cell keep the
  // frame of the whole blueprint's image, as the rest keeps the whole blueprint's frame until the drop), the history's
  // points, the recent placements' boxes, the offset and the view's orientation (sym_orientation), and a view that a
  // caller set: its middle's pixel and the rule's orientation's image.
  static void sym_records(const Position& p, int symmetry, Position& s) {
    std::copy(p.scores, p.scores + 2, s.scores);
    std::copy(p.stars, p.stars + 2, s.stars);
    std::copy(p.cells, p.cells + 2, s.cells);
    s.ply = p.ply;
    s.orientation = int8_t(sym_orientation(p.orientation, symmetry));
    int vx = p.viewX, vy = p.viewY;
    sym_rules(vx, vy, symmetry);
    s.viewSet = p.viewSet;
    s.viewX = int16_t(vx);
    s.viewY = int16_t(vy);
    s.ruleOrientation = int8_t(sym_orientation(p.ruleOrientation, symmetry));
    s.picked = p.picked;
    s.pickType = p.pickType;
    s.pickConnected = p.pickConnected;
    const int mover = GameRules::next_player(p);
    int pickQ = p.pickQ, pickR = p.pickR;  // in the image's frame of the picked blueprint (map_blueprint)
    for(int player = 0; player < 2; player++)
      for(int k = 0; k < HAND; k++) {
        int mq, mr;
        map_blueprint(p.hand[player][k], [symmetry](int& q, int& r) { sym_offset(q, r, symmetry); },
                      player == mover && k == p.picked, pickQ, pickR, s.hand[player][k], mq, mr);
      }
    s.pickQ = int8_t(pickQ);
    s.pickR = int8_t(pickR);
    for(int h = 0; h < HISTORY; h++) {
      s.history[h] = p.history[h];
      Footprint& f = s.history[h];
      for(int i = 0; i < f.nStamped; i++)
        f.stamped[i] = sym_rules_index(f.stamped[i], symmetry);
      for(int i = 0; i < f.nTaken; i++)
        f.taken[i] = sym_rules_index(f.taken[i], symmetry);
    }
    for(int player = 0; player < 2; player++)
      for(int i = 0; i < RECENT; i++) {
        const Box& b = p.recent[player][i];
        Box& image = s.recent[player][i];
        image = Box();
        if(!b.any)
          continue;
        int ax = b.x0, ay = b.y0, bx = b.x1, by = b.y1;
        sym_rules(ax, ay, symmetry);
        sym_rules(bx, by, symmetry);
        image.add(ax, ay);
        image.add(bx, by);
      }
    int oq = p.offsetQ, orr = p.offsetR;
    sym_offset(oq, orr, symmetry);
    s.offsetQ = int16_t(oq);
    s.offsetR = int16_t(orr);
  }

  // Writes the images of p's cells in the rules board's columns x0 to x1 - 1 and rows y0 to y1 - 1 into s, whose cells
  // are empty and whose offset is the image's (sym_records): each at its pixel's image around the board's middle. Then
  // s's cells' hash, by the image's site coordinates. Only the part of those columns and rows in the box that holds
  // p's cells (Cells::box) can hold a cell.
  static void sym_cells(const Position& p, int symmetry, int x0, int x1, int y0, int y1, Position& s) {
    s.cellHash = Hash128();
    const Box& used = p.cell.box();
    if(!used.any)
      return;
    x0 = std::max<int>(x0, used.x0);
    x1 = std::min<int>(x1, used.x1 + 1);
    y0 = std::max<int>(y0, used.y0);
    y1 = std::min<int>(y1, used.y1 + 1);
    for(int y = y0; y < y1; y++)
      for(int x = x0; x < x1; x++) {
        const int8_t v = p.cell[y * RULES_LEN + x];
        if(!v)
          continue;
        int sx = x, sy = y;
        sym_rules(sx, sy, symmetry);
        s.cell.set(sy * RULES_LEN + sx, v);
        s.cellHash ^= key(site_key(sx - RULES_RADIUS + s.offsetQ, sy - RULES_RADIUS + s.offsetR), uint64_t(v));
      }
  }

  // The image: the records (sym_records) and every cell (sym_cells).
  static Position sym_position(const Position& p, int symmetry) {
    if(symmetry == 0)
      return p;
    Position s;
    sym_records(p, symmetry, s);
    sym_cells(p, symmetry, 0, RULES_LEN, 0, RULES_LEN, s);
    return s;
  }

  // The image as the inputs read it: the records, and the cells within INPUT_REACH of the view only. The counts of
  // cells and stars stay the whole board's.
  static const Position& near_image(const Position& p, int symmetry) {
    static thread_local std::vector<Position> buffer(1);
    Position& s = buffer[0];
    s.cell.clear();
    sym_records(p, symmetry, s);
    int x0, x1, y0, y1;
    view_box(p, INPUT_REACH, x0, x1, y0, y1);
    sym_cells(p, symmetry, x0, x1, y0, y1, s);
    return s;
  }

  // A legal action's image, an action of sym_position: the image's view is the view's image, a pixel map around the
  // view's middle. A placement's anchor is the lowest point of its stamped cells' images, which is the anchor's pixel
  // moved by the shift (frame_shift) of its blueprint in the view's frame (ViewHand). A pick's or a drop's hand point
  // moves by the linear part, then by its blueprint's shift. The trainer's module maps both the same way.
  struct SymActionMap {
    int symmetry;
    int picked;
    ViewHand hand;  // the mover's, in the view's frame
    SymActionMap(const Position& p, int sym) : symmetry(sym), picked(p.picked) {
      view_hand(p, GameRules::next_player(p), hand);
    }
    int operator()(int action) const {
      if(symmetry == 0 || action == GameRules::PASS_ACTION)
        return action;
      const int x = action_x(action), y = action_y(action), channel = action_channel(action);
      int mq, mr;
      if(channel < HAND_CHANNEL) {
        int ax = x, ay = y;
        GameInterface::symPoint(ax, ay, symmetry, VIEW, VIEW);
        frame_shift(hand.b[channel - PLACE], symmetry, false, 0, 0, mq, mr);
        return action_at(ax + mq, ay + mr, channel);
      }
      const int k = channel - HAND_CHANNEL;  // a pick from blueprint k, or a drop into it after a pick
      frame_shift(hand.b[k], symmetry, k == picked, hand.pickQ, hand.pickR, mq, mr);
      int q = x - HAND_X, r = y - HAND_Y;
      sym_offset(q, r, symmetry);
      return action_at(q - mq + HAND_X, r - mr + HAND_Y, channel);
    }
  };

  // The training targets, all on the row's view: a later position's cells move by their site coordinates into the
  // row's view (in_row_view), and a cell off the row's view counts for no point. The horizons count the site's moves
  // (the ply, where a pick and its drop are one move), not actions.
  // - C0, the final board as ownership: +1 on the mover's cells, -1 on the opponent's, 0 on an empty point.
  // - C1, the cells captured before the game's end, as classes: on each cell of the row's view that is not a wall, +1
  //   when a capture takes it and it is the mover's, -1 when it is the opponent's, 0 when it stays to the end. A cell
  //   counts as captured even if a cell comes to its site again later. Class 2 on the other points (the empty points
  //   and the walls, which no capture takes), which the loss leaves out. Its flag C68.
  // - C2-3, the stars FUTURE_MOVES[0] and [1] moves later: +1 on the mover's stars, -1 on the opponent's, 0 elsewhere,
  //   in the first position with no pick waiting and at least that many moves more than the row (the end, when the
  //   game ends sooner). A star there counts even if a star of the same player stood there and was taken before.
  // - C4, where the points are earned: on each site, the points that the captures of the stars there earn from the
  //   row on (for_each_earning: the capturer's stars on the board before its placement), the mover's less the
  //   opponent's, / EARNED_UNIT and clipped to [-1, 1]. The points of the sites off the view, and the clip, make C4's
  //   sum differ from the later margin. Its weight C34.
  // - C5, the opponent's cells that go in the rest of the mover's turn: 1 on each of the opponent's cells of the row's
  //   view, not a wall, that a placement takes before the opponent moves, else 0. C6, the mover's cells that go in the
  //   opponent's next turn: the same on the mover's cells, in that turn. A cell goes when its site no longer holds it,
  //   even if a cell of the same player comes there later. A turn that the game's end cuts off takes no cell. The
  //   other points hold 0, and the trainer's loss reads only these cells (katago/train/games/blueprints.py,
  //   income_mask).
  // - G76, which player makes the row's next move, for the trainer's symmetries of the second policy target: +1 the
  //   row's mover, -1 the opponent, 0 at the end. A pick and a placement with a move to come leave the move to the
  //   mover. G77, the version of these targets, TARGETS_VERSION: 1 added C2-3, C5-6 and G76, and 2 added C1 and C4.
  //   The rows before version 1 (version 0) hold the next mover in C1 on every point, which the trainer reads in
  //   them, and no C2-3 or C5-6. The rows of version 1 hold 0 in C1. Neither has C1 or C4 as a target: C68 and C34
  //   are 0.
  // - G78, the orientation of the row's view, and G79, the orientation of the next position's view (the row's when
  //   none follows): the second policy target is a policy of the next position's view. When the two differ, that
  //   policy is in a turned frame that the row's inputs do not give, so the trainer gives it the weight 0
  //   (katago/train/games/blueprints.py, mask_targets). Both are 0 in the rows from before the orientations, as their
  //   views were.
  // A row whose positions stop before a horizon or a turn's end (a game that hit the turn limit) has no C2-3 or C5-6:
  // their weight C33 and their flag C69 are 0. A row whose positions stop before the game's end has no C1 or C4.
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP | GameInterface::TARGET_SEKI |
                                      GameInterface::TARGET_FUTURE_PIECES | GameInterface::TARGET_SCORING |
                                      GameInterface::TARGET_INCOME;
  static constexpr unsigned OWN_TARGETS = TARGETS;  // C2-3 too: the front's would read each position's own view
  // The horizons of C2-3 in the site's moves, from bpj's design review (games/blueprints/docs/decisions/training.md,
  // "bpj2's targets"). KataGo's are 8 and 32 moves of Go.
  static constexpr int FUTURE_MOVES[2] = {4, 16};
  // The points on a site at which C4 reaches 1, from the same review. The highest level of the placements' points
  // (POINT_LEVELS) is 400 too.
  static constexpr double EARNED_UNIT = 400.0;
  static constexpr int NEXT_MOVER_TARGET = 76, TARGETS_VERSION_TARGET = 77;  // global targets (TargetValues)
  static constexpr int ORIENTATION_TARGET = 78, NEXT_ORIENTATION_TARGET = 79;
  static constexpr int TARGETS_VERSION = 2;
  static_assert(NEXT_MOVER_TARGET >= GameInterface::TargetValues::FIRST_GLOBAL &&
                NEXT_ORIENTATION_TARGET < GameInterface::TargetValues::FIRST_GLOBAL +
                                          GameInterface::TargetValues::NUM_GLOBALS,
                "Blueprints' global targets are among those that a game writes");

  // Calls f(x, y, v) for each point (x, y) of the row's view, the view of `now`, with v the cell of `later` at the
  // same site (0 when it holds none).
  template<class F>
  static void in_row_view(const Position& now, const View& view, const Position& later, F&& f) {
    // A point of later's rules board less (dq, dr) is the point of now's board at the same site.
    const int dq = later.offsetQ - now.offsetQ, dr = later.offsetR - now.offsetR;
    for(int y = 0; y < VIEW; y++)
      for(int x = 0; x < VIEW; x++) {
        int rx, ry;
        view.rules_point(x, y, rx, ry);
        f(x, y, at(later, rx - dq, ry - dr));
      }
  }

  // Calls f(j, q, r, v) for each cell v that the action after positions(j) captures, for from <= j < numPositions - 1,
  // (q, r) its site. The action's footprint, positions(j + 1).history[0], gives the points that its placement filled
  // (they hold cells, so they stay on the board when it recenters), and the cells that it captured are those that
  // taken_by gives for these points on positions(j). The footprint's own list of captured points loses those that the
  // recentering moves off the board. A pick, a drop and a pass fill no point.
  template<class Positions, class F>
  static void for_each_capture(const Positions& positions, int numPositions, int from, F&& f) {
    for(int j = from; j + 1 < numPositions; j++) {
      const Position& before = positions(j);
      const Position& after = positions(j + 1);
      const Footprint& footprint = after.history[0];
      if(footprint.nStamped == 0)
        continue;
      // The filled points as a blueprint on before's board, from the first of them (taken_by reads the cells' points
      // only): a point of after's board is the point of before's board at the same site.
      const int ax = footprint.stamped[0] % RULES_LEN, ay = footprint.stamped[0] / RULES_LEN;
      Blueprint filled;
      for(int i = 0; i < footprint.nStamped; i++)
        filled.add(footprint.stamped[i] % RULES_LEN - ax, footprint.stamped[i] / RULES_LEN - ay, PLAIN);
      int32_t taken[MAX_TAKEN];
      const int n = taken_by(before, GameRules::next_player(before), filled, ax + after.offsetQ - before.offsetQ,
                             ay + after.offsetR - before.offsetR, taken);
      for(int i = 0; i < n; i++)
        f(j, taken[i] % RULES_LEN - RULES_RADIUS + before.offsetQ, taken[i] / RULES_LEN - RULES_RADIUS + before.offsetR,
          before.cell[taken[i]]);
    }
  }

  // Where the points are earned from positions(from) on, C4's ledger before the view and the clip: calls f(q, r,
  // points) for each star that a capture takes, (q, r) its site, with the points that it earns, the capturer's stars
  // on the board before its placement (as stamp scores), positive when the capturer is `player` and negative when it
  // is the other player. Over every site, the points add up to player's later points less the other player's.
  template<class Positions, class F>
  static void for_each_earning(const Positions& positions, int numPositions, int from, int player, F&& f) {
    for_each_capture(positions, numPositions, from, [&](int j, int q, int r, int8_t v) {
      if(cell_type(v) != STAR)
        return;
      const int capturer = GameRules::next_player(positions(j));
      const int points = positions(j).stars[capturer];
      f(q, r, capturer == player ? points : -points);
    });
  }

  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int row, int mover, float* values) {
    const Position& now = positions(row);
    const View view = view_of(now);
    float* globals = values + GameInterface::TargetValues::NUM_CHANNELS * VIEW_AREA;
    const int firstGlobal = GameInterface::TargetValues::FIRST_GLOBAL;
    unsigned slots = GameInterface::TARGET_OWNERSHIP;

    if(row + 1 < numPositions && !GameRules::terminal(positions(row + 1)))
      globals[NEXT_MOVER_TARGET - firstGlobal] = GameRules::next_player(positions(row + 1)) == mover ? 1.0f : -1.0f;
    globals[TARGETS_VERSION_TARGET - firstGlobal] = float(TARGETS_VERSION);
    globals[ORIENTATION_TARGET - firstGlobal] = float(now.orientation);
    globals[NEXT_ORIENTATION_TARGET - firstGlobal] =
      float(row + 1 < numPositions ? positions(row + 1).orientation : now.orientation);

    in_row_view(now, view, positions(numPositions - 1), [&](int x, int y, int8_t v) {
      if(v)
        values[y * VIEW + x] = cell_owner(v) == mover ? 1.0f : -1.0f;
    });

    // The horizons: -1 when the positions stop before one.
    int horizon[2];
    for(int h = 0; h < 2; h++) {
      const int ply = std::min(now.ply + FUTURE_MOVES[h], PLIES);
      horizon[h] = -1;
      for(int j = row + 1; j < numPositions && horizon[h] < 0; j++)
        if(positions(j).picked < 0 && positions(j).ply >= ply)
          horizon[h] = j;
    }
    if(horizon[0] >= 0 && horizon[1] >= 0) {
      for(int h = 0; h < 2; h++)
        in_row_view(now, view, positions(horizon[h]), [&](int x, int y, int8_t v) {
          if(v && cell_type(v) == STAR)
            values[(2 + h) * VIEW_AREA + y * VIEW + x] = cell_owner(v) == mover ? 1.0f : -1.0f;
        });
      slots |= GameInterface::TARGET_FUTURE_PIECES;
    }

    // The end of a player's turn after position `from`: the first position where the player no longer moves, or the
    // game has ended. -1 when the positions stop before it.
    auto turn_end = [&](int from, int player) {
      for(int j = from + 1; j < numPositions; j++)
        if(GameRules::terminal(positions(j)) || GameRules::next_player(positions(j)) != player)
          return j;
      return -1;
    };
    const int moverEnd = turn_end(row, mover);
    const int opponentEnd =
      moverEnd < 0 || GameRules::terminal(positions(moverEnd)) ? moverEnd : turn_end(moverEnd, 1 - mover);
    if(opponentEnd >= 0) {
      // Each cell of the row's view that can go (not a wall), and the first position where its site no longer holds
      // it (-1: none up to the opponent's turn's end).
      int8_t start[VIEW_AREA] = {};
      int gone[VIEW_AREA];
      std::fill(gone, gone + VIEW_AREA, -1);
      in_row_view(now, view, now, [&](int x, int y, int8_t v) {
        if(v && cell_type(v) != WALL)
          start[y * VIEW + x] = v;
      });
      for(int j = row + 1; j <= opponentEnd; j++)
        in_row_view(now, view, positions(j), [&](int x, int y, int8_t v) {
          const int i = y * VIEW + x;
          if(start[i] && gone[i] < 0 && v != start[i])
            gone[i] = j;
        });
      for(int i = 0; i < VIEW_AREA; i++) {
        if(!start[i] || gone[i] < 0)
          continue;
        if(cell_owner(start[i]) != mover && gone[i] <= moverEnd)
          values[5 * VIEW_AREA + i] = 1.0f;
        if(cell_owner(start[i]) == mover && gone[i] > moverEnd)
          values[6 * VIEW_AREA + i] = 1.0f;
      }
      slots |= GameInterface::TARGET_INCOME;
    }

    // The captures up to the game's end: none in a record that stops before it.
    if(GameRules::terminal(positions(numPositions - 1))) {
      float* captured = values + VIEW_AREA;  // C1
      in_row_view(now, view, now, [&](int x, int y, int8_t v) {
        captured[y * VIEW + x] = v && cell_type(v) != WALL ? 0.0f : 2.0f;
      });
      // The point (x, y) of the row's view of the site (q, r), on the view or not.
      auto site_point = [&](int q, int r, int& x, int& y) {
        view.view_point(q - now.offsetQ + RULES_RADIUS, r - now.offsetR + RULES_RADIUS, x, y);
      };
      for_each_capture(positions, numPositions, row, [&](int, int q, int r, int8_t v) {
        int x, y;
        site_point(q, r, x, y);
        if(!on_view(x, y))
          return;
        // A cell of the row is the first that a capture takes from its site: a cell goes only by a capture.
        float& c = captured[y * VIEW + x];
        if(c == 0.0f)
          c = cell_owner(v) == mover ? 1.0f : -1.0f;
      });
      float* earned = values + 4 * VIEW_AREA;  // C4: the points first, then / EARNED_UNIT and clipped
      for_each_earning(positions, numPositions, row, mover, [&](int q, int r, int points) {
        int x, y;
        site_point(q, r, x, y);
        if(on_view(x, y))
          earned[y * VIEW + x] += float(points);
      });
      for(int i = 0; i < VIEW_AREA; i++)
        earned[i] = std::max(-1.0f, std::min(1.0f, earned[i] / float(EARNED_UNIT)));
      slots |= GameInterface::TARGET_SEKI | GameInterface::TARGET_SCORING;
    }
    return slots;
  }

  // A capture late in a game scores tens of stars times tens of stars: margins of hundreds of points, against Go's
  // tens. So KataGo's constants in points scale by 12, as Enclosure's, and the search's unit is 200 points.
  static constexpr double SCORE_SCALE = 12.0;
  static constexpr double SCORE_UNIT = 200.0;
  // A game is 120 of the site's moves (moves_played counts them; a pick and its drop are one).
  static constexpr double GAME_LENGTH = PLIES;
  // The view is 67 x 67 so that it holds the window, not because a decision reaches that far: KataGo's look-ahead
  // horizons read Go's 361, about 64, 20 and 6 actions in a game of about 140. And the komi sampler reads 361, so that
  // it draws a komi in the points that the config gives (GAMES.md, the encoding part's optional members).
  static constexpr double LOOKAHEAD_AREA = 19 * 19;
  static constexpr double KOMI_AREA = 19 * 19;
  // The even-komi search's bracket: offsets of 12, 24, 48 and so on around its guess, up to 3,000 points, the reach of
  // the widest komi of a new game in self-play (3 standard deviations of 1,000 points). KataGo's offsets of 1 to 32
  // points are less than 3 of Go's points at this SCORE_SCALE, and Blueprints' margins are hundreds of points. Beyond
  // 3,000 points from the guess, the search finds no even komi (GameInterface::komiBracketEndIsEven).
  static constexpr double KOMI_BRACKET_STEP = SCORE_SCALE;
  static constexpr double KOMI_BRACKET_LIMIT = 3000.0;
  // No smoothing of the lead: KataGo's smooths over the parity of Go's area scoring, where the result changes every 2
  // points of komi. Blueprints' margins are whole points of both parities (in the site's finished games, 426 even and
  // 311 odd), so each half point of komi can change the result, and the search's even komi is the lead.
  static constexpr double KOMI_LEAD_STEP = 0.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Blueprints' network encoding part");

}  // namespace blueprints
