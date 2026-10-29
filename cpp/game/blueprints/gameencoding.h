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
//   each of the last 4 actions stamped and took, or under the inputs version 11 the points that the moves of each of
//   the last 4 turns stamped and took: the turn in progress, then the 3 whole turns before it.
// - The global inputs: the mover's seat, its moves left in the turn, the game's progress, the stars on the board, the
//   effective scores (with the komi), the pick, each player's cells and stars outside the view, and the mover's
//   effective margin.
// - The extended inputs, which the inputs version 10 adds (GameInterface; bpj3): the mover's cells that the
//   opponent's placements in its own view take, the hand's points where a pick or a drop is legal now, and the
//   minimap (each player's cells by type over the whole board, in bins of 5 x 5 points, and the bins of each player's
//   view). Then each player's last 2 placements' boxes, the window's rule, as global inputs.
// - The turn inputs, which the inputs version 11 adds after those (GameInterface): for each move of each of the last 4
//   turns, 1 when it was an edit. A move of a turn that stamped nothing and was not an edit was a pass, or is still to
//   come.
// - The planes of closed areas, which the inputs version 12 adds after the extended planes (GameInterface): the points
//   of the view in the areas closed by the mover's walls and by the opponent's walls, where the other player can never
//   place (for_each_closed).
// - The hand's canonical order, which the inputs version 13 reads (GameInterface, CanonicalActionMap): the same inputs
//   as version 12, with each hand's blueprints in an order of their cells (HandOrder) in place of their numbers'. So
//   are the policy's channels, and of equal blueprints the search lists the lowest-numbered one's actions only.
// - The komi K is points added to the second player's final margin (KataGo's White), for the whole game. The rules'
//   scores stay the earned ones, and the front's history adds K to the result (BoardHistory::endAndScoreGameNow). The
//   inputs read K as a starting score of the side that it favours, the effective scores S0 + max(-K, 0) and S1 +
//   max(K, 0). No rule reads a score, so a position at komi K plays as the position with those scores at komi 0.
// - The policy: 6 channels a point, the actions of gamerules.h (placements, then the cells of the hand: picks, or
//   drops after a pick), each blueprint's at its slot of the hand's order under the inputs version 13.
// - The view's orientation (gamerules.h): the inputs, the policy and the spatial targets are in the view's frame, and
//   the hand's planes show the blueprints in the view's frame (ViewHand). So a position in orientation o gives the
//   inputs of the position turned by R^back(o) around the window's middle in orientation 0.
// - The symmetries: the 4 of KataGo's 8 that keep the hex neighbours (DIRS) on the axial square: the identity, the
//   turn by 180 degrees (3) and two reflections (4, 7). The image maps the rules board around its middle. The window's
//   rule and the recentering round toward zero, so the image's window is the window's image, and with the image's
//   orientation (sym_orientation) a point of the view moves as a pixel. A blueprint of the view's frame moves cell by
//   cell, then goes back to its site frame (its lowest cell at 0,0): so the hand's points, the anchors of the
//   placements and the picks and drops move as pixels and then by each blueprint's shift. The inputs at a symmetry are
//   the inputs of the position's image (sym_position). With the view's 6 turns, the 4 give the hex grid's 12. The
//   window boxes' corners move as offsets (box_inputs).
// - The training targets, on the row's view: the final board as ownership (C0), the cells captured before the game's
//   end (C1), the stars 4 and 16 moves later (C2-3), where the points are earned (C4), the cells that go in the rest
//   of the mover's turn and in the opponent's next turn (C5-6), and in the global target G76 which player moves
//   next, for the trainer's symmetries (no loss reads it). The value targets and
//   the score targets are KataGo's, from the history's result: the final margin with the komi, where a margin of 0
//   is half a win.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
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

// A blueprint's cells as codes (r * 64 + q + 32) * 3 + type, sorted: the least r, then the least q, first, as
// sorted_cells orders them. In the site's frame (|q|, r <= 20), two blueprints have the same codes when they are equal.
// Returns the number of cells.
inline int cell_codes(const Blueprint& b, int16_t* out) {
  for(int i = 0; i < b.n; i++)
    out[i] = int16_t((b.r[i] * 64 + b.q[i] + 32) * 3 + b.t[i]);
  std::sort(out, out + b.n);
  return b.n;
}

// A blueprint's shape: of the codes of its 12 images under the hex grid's symmetries, each in the site's frame, the
// least in the order of a dictionary. The 12 are the view's 6 turns (turn), each with and without the reflection that
// swaps q and r. Two blueprints have the same shape when one is an image of the other.
inline void shape_codes(const Blueprint& b, int16_t* out) {
  int16_t codes[MAX_CELLS];
  for(int image = 0; image < 2 * ORIENTATIONS; image++) {
    const bool reflect = image >= ORIENTATIONS;
    const int o = image % ORIENTATIONS;
    Blueprint mapped;
    int mq, mr, noQ = 0, noR = 0;
    map_blueprint(b, [reflect, o](int& q, int& r) {
      if(reflect)
        std::swap(q, r);
      turn(q, r, o);
    }, false, noQ, noR, mapped, mq, mr);
    cell_codes(mapped, codes);
    if(image == 0 || std::lexicographical_compare(codes, codes + b.n, out, out + b.n))
      std::copy(codes, codes + b.n, out);
  }
}

// A player's blueprints as the hand's order reads them: the mover's blueprint whose cell a pick took is the whole
// blueprint, the picked cell back in it (its rest keeps the whole blueprint's frame until the drop).
inline void whole_hand(const Position& p, int player, Blueprint* hand) {
  std::copy(p.hand[player], p.hand[player] + HAND, hand);
  if(player == turn_of(p.ply) && p.picked >= 0)
    hand[p.picked].add(p.pickQ, p.pickR, p.pickType);
}

// The hand's canonical order, which the inputs version 13 reads (InputOptions::canonical). No rule reads a blueprint's
// number, so a hand's order is a label, and in this order two positions whose hands differ only in their order give the
// same inputs. A player's blueprint a comes before blueprint b when a has more cells, else more stars, else more walls,
// else the lesser shape (shape_codes). The mover's blueprint with a pick waiting counts as the whole blueprint
// (whole_hand). The key does not depend on the board's frame: a view of any orientation, and the image at any of the
// game's symmetries, give the position's order, so the trainer's symmetries keep each blueprint's slot. Of equal
// blueprints, the one with the pick comes first, then the lower number: their lowest-numbered one, whose actions the
// search lists (CanonicalActionMap), has the lowest slot. Blueprints that are images of each other but not equal tie on
// the key (in people's games, 0.7% of the hands): they keep the order of their equal blueprints' lowest numbers,
// whatever the pick. So a pick that the search lists keeps the order of the position before it, and the row's second
// policy target, in the next position's frame (trainingwrite.cpp), has the row's slots.
struct HandOrder {
  int8_t slot[2][HAND];  // the slot of each player's blueprint k in the order

  explicit HandOrder(const Position& p) {
    for(int player = 0; player < 2; player++) {
      Blueprint hand[HAND];
      whole_hand(p, player, hand);
      int count[HAND][3];
      for(int k = 0; k < HAND; k++) {
        count[k][0] = hand[k].n;
        count[k][1] = int(std::count(hand[k].t, hand[k].t + hand[k].n, STAR));
        count[k][2] = int(std::count(hand[k].t, hand[k].t + hand[k].n, WALL));
      }
      // The codes, when two blueprints' counts tie, and the shapes, when their codes differ too.
      int16_t codes[HAND][MAX_CELLS], shapes[HAND][MAX_CELLS];
      bool coded[HAND] = {}, shaped[HAND] = {};
      auto equal = [&](int a, int b) {
        if(!std::equal(count[a], count[a] + 3, count[b]))
          return false;
        for(int k : {a, b})
          if(!coded[k]) {
            cell_codes(hand[k], codes[k]);
            coded[k] = true;
          }
        return std::equal(codes[a], codes[a] + hand[a].n, codes[b]);
      };
      // Each blueprint's lowest-numbered equal blueprint, itself when none comes before it.
      int first[HAND];
      for(int k = 0; k < HAND; k++) {
        first[k] = k;
        for(int j = 0; j < k; j++)
          if(equal(j, k)) {
            first[k] = j;
            break;
          }
      }
      const int picked = player == turn_of(p.ply) ? p.picked : -1;
      // The key: the counts, the shape, the lowest number of the equal blueprints, the pick, the number.
      auto before = [&](int a, int b) {
        if(!std::equal(count[a], count[a] + 3, count[b]))
          return std::lexicographical_compare(count[b], count[b] + 3, count[a], count[a] + 3);
        if(first[a] == first[b])  // equal: the one with the pick first, then the lower number
          return picked == a || picked == b ? picked == a : a < b;
        const int n = hand[a].n;
        for(int k : {a, b})
          if(!shaped[k]) {
            shape_codes(hand[k], shapes[k]);
            shaped[k] = true;
          }
        if(!std::equal(shapes[a], shapes[a] + n, shapes[b]))
          return std::lexicographical_compare(shapes[a], shapes[a] + n, shapes[b], shapes[b] + n);
        return first[a] < first[b];  // images of each other, whatever the pick
      };
      int order[HAND] = {0, 1, 2};
      static_assert(HAND == 3, "the order of three blueprints");
      if(before(order[1], order[0]))
        std::swap(order[0], order[1]);
      if(before(order[2], order[1]))
        std::swap(order[1], order[2]);
      if(before(order[1], order[0]))
        std::swap(order[0], order[1]);
      for(int i = 0; i < HAND; i++)
        slot[player][order[i]] = int8_t(i);
    }
  }
};

// For each of the mover's blueprints k, the lowest-numbered blueprint equal to it (k itself when none comes before
// it). A pick waiting for its drop leaves its blueprint out: a drop into either of two equal blueprints gives the same
// position up to the hand's order, and a drop into the picked blueprint gives a position of its own.
inline void first_equal(const Position& p, int8_t* first) {
  const Blueprint* hand = p.hand[turn_of(p.ply)];
  int16_t codes[HAND][MAX_CELLS];
  for(int k = 0; k < HAND; k++) {
    cell_codes(hand[k], codes[k]);
    first[k] = int8_t(k);
    if(k == p.picked)
      continue;
    for(int j = 0; j < k; j++)
      if(j != p.picked && hand[j].n == hand[k].n && std::equal(codes[j], codes[j] + hand[k].n, codes[k])) {
        first[k] = int8_t(j);
        break;
      }
  }
}

// A box's image: the box of its corners' images on the rules board (sym_rules).
inline Box sym_box(const Box& b, int symmetry) {
  Box image;
  if(!b.any)
    return image;
  int ax = b.x0, ay = b.y0, bx = b.x1, by = b.y1;
  sym_rules(ax, ay, symmetry);
  sym_rules(bx, by, symmetry);
  image.add(ax, ay);
  image.add(bx, by);
  return image;
}

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

struct Diversity;  // diversity.h, which this file includes after GameEncoding

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;
  // AZdb's statistics of a position (GameInterface: Diversity): diversity.h's, which reads this encoding's inputs.
  using Diversity = blueprints::Diversity;

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
  // Action h, or turn h under the inputs version 11 (0 the turn in progress): + 2h stamped, + 2h + 1 taken.
  static constexpr int HISTORY_PLANE = 81, NUM_HISTORY_PLANES = 2 * HISTORY;
  static constexpr int NUM_SPATIAL_INPUTS = HISTORY_PLANE + NUM_HISTORY_PLANES;  // 89
  static_assert(THEIR_PLACES == MY_PLACES + PLACE_PLANES && MINE_TAKEN == THEIR_PLACES + PLACE_PLANES &&
                MY_HAND == THEIRS_TAKEN + 1 && PICKED == THEIR_HAND + 3 * HAND && HISTORY_PLANE == PICKED + HAND,
                "Blueprints' planes follow one another");
  static constexpr int NUM_GLOBAL_INPUTS = 22;
  static constexpr int MARGIN_INPUT = 21;  // the mover's effective margin / MARGIN_UNIT, not squashed
  // Of the order of people's margins (median 409 points, 90th percentile 2,060) and of self-play's komis (up to
  // 3,000 points in a new game), so that the margin's input is of the order of 1.
  static constexpr double MARGIN_UNIT = 1000.0;

  // The extended inputs (GameInterface: the inputs version 10, for bpj3), after the planes and the global inputs
  // above. Each pair of planes names the mover's first, then the opponent's.
  static constexpr int MINE_THREATENED = NUM_SPATIAL_INPUTS;  // the mover's cells that the opponent's placements take
  static constexpr int LEGAL_HAND = 90;   // + k: the hand's points of blueprint k where a pick or a drop is legal now
  static constexpr int MINIMAP = 93;      // + 3 * side + type (side 0 the mover's, 1 the opponent's): cells in a bin
  static constexpr int MY_WINDOW = 99, THEIR_WINDOW = 100;  // the minimap's bins that each player's view meets
  static constexpr int NUM_EXTENDED_SPATIAL_INPUTS = THEIR_WINDOW + 1 - NUM_SPATIAL_INPUTS;  // 12
  static_assert(LEGAL_HAND == MINE_THREATENED + 1 && MINIMAP == LEGAL_HAND + HAND && MY_WINDOW == MINIMAP + 6,
                "Blueprints' extended planes follow one another");
  // The window boxes, + BOX_FIELDS * (RECENT * side + i) for box i (the newest first) of the mover (side 0) or the
  // opponent: 1 when the box is there, then its corners' x0, x1, y0 and y1 / VIEW (box_inputs).
  static constexpr int BOX_INPUT = NUM_GLOBAL_INPUTS, BOX_FIELDS = 5;
  static constexpr int NUM_EXTENDED_GLOBAL_INPUTS = 2 * RECENT * BOX_FIELDS;  // 20
  // The turn inputs (GameInterface: the inputs version 11, with the history planes of the turns), after the extended
  // global inputs: + TURN_MOVES * h + m, 1 when move m of turn h (Position::turns) was an edit.
  static constexpr int TURN_INPUT = BOX_INPUT + NUM_EXTENDED_GLOBAL_INPUTS;
  static constexpr int NUM_TURN_GLOBAL_INPUTS = HISTORY * TURN_MOVES;  // 8
  // The planes of closed areas (GameInterface: the inputs version 12), after the extended planes: the hexes of the
  // area closed by the mover's walls, then by the opponent's walls (for_each_closed).
  static constexpr int MY_CLOSED = THEIR_WINDOW + 1, THEIR_CLOSED = MY_CLOSED + 1;  // 101, 102
  static constexpr int NUM_AREA_SPATIAL_INPUTS = 2;
  // The minimap's bins: 5 x 5 points, 67 x 67 of them around the minimap's middle (|u|, |v| <= MINIMAP_REACH). The
  // points within 127 of the middle are 51 x 51 of them, in the middle. A point beyond MINIMAP_REACH is in the bin at
  // the edge (minimap_bin).
  static constexpr int MINIMAP_BIN = 5;
  static constexpr int MINIMAP_REACH = (VIEW * MINIMAP_BIN - 1) / 2;  // 167
  static_assert(VIEW * MINIMAP_BIN % 2 == 1 && (MINIMAP_REACH - 127) % MINIMAP_BIN == 0,
                "the minimap's bins are 5 x 5 points, and 51 x 51 of them hold the 255 x 255 points in the middle");

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    const Position& image = options.symmetry == 0 ? p : near_image(p, options.symmetry);
    // The hand's order of the inputs version 13: the image's is the position's (HandOrder).
    std::optional<HandOrder> order;
    if(options.canonical)
      order.emplace(p);
    const HandOrder* slots = order ? &*order : nullptr;
    write_inputs(image, options, slots, set);
    // The threats, the minimap and the closed areas read cells far from the view, which the image near the view does
    // not hold. They move as pixels (the trainer maps them so), so they come from p, at the image's points.
    auto pixel = [&](int x, int y, int plane) {
      GameInterface::symPoint(x, y, options.symmetry, VIEW, VIEW);
      set(x, y, plane);
    };
    if(options.extended) {
      write_threats(p, pixel);
      write_legal_hand(image, slots, set);
      write_minimap(p, pixel);
    }
    if(options.areas)
      write_closed_areas(p, pixel);
  }

  // A player's blueprint k's slot in the planes: k, or its slot of the hand's order (order, the inputs version 13).
  static int slot_of(const HandOrder* order, int player, int k) { return order ? order->slot[player][k] : k; }

  // The inputs but the extended ones. The history planes show p's last actions, or with options.turnHistory its last
  // turns, or nothing without options.history. Each blueprint's planes are at its slot (slot_of).
  template<class Set>
  static void write_inputs(const Position& p, const GameInterface::InputOptions& options, const HandOrder* order,
                           Set& set) {
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
          const int s = slot_of(order, player, k);
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
            set(x, y, places + s);
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
              set(x, y, places + level * HAND + s);
            const int score = stars * p.stars[player];
            for(int i = 0; i < NUM_POINT_LEVELS; i++)
              if(score >= POINT_LEVELS[i])
                set(x, y, places + (1 + STAR_LEVELS + i) * HAND + s);
          });
        }
      }
    }
    for(int player = 0; player < 2; player++) {
      const int base = player == mover ? MY_HAND : THEIR_HAND;
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = hands[player].b[k];
        const int s = slot_of(order, player, k);
        for(int i = 0; i < b.n; i++) {
          const int x = b.q[i] + HAND_X, y = b.r[i] + HAND_Y;
          set(x, y, base + s);
          if(b.t[i] == STAR)
            set(x, y, base + HAND + s);
          if(b.t[i] == WALL)
            set(x, y, base + 2 * HAND + s);
        }
      }
    }
    if(p.picked >= 0)
      set(hands[mover].pickQ + HAND_X, hands[mover].pickR + HAND_Y, PICKED + slot_of(order, mover, p.picked));
    // The actions' footprints or the turns, the newest first.
    auto write_history = [&](const auto* history) {
      for(int h = 0; h < HISTORY; h++) {
        const auto& f = history[h];
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
    };
    if(options.history && options.turnHistory)
      write_history(p.turns);
    else if(options.history)
      write_history(p.history);
  }

  // The opponent's threats where it places (MINE_THREATENED): the mover's cells that some placement of the opponent in
  // its own view (player_view, in the position's orientation) takes, on the board as it is, at their points of the
  // mover's view. The planes of the opponent's placements show its placements in the mover's view instead. In rows of
  // bits of the opponent's view and one point around it (padded: row y + 1, bit x + 1, as view_rows): the points that
  // the placements stamp (each fit_row moved by each cell of its blueprint), the points next to them, and of those
  // the mover's cells that a capture takes (not walls). Such a cell in the view is next to an empty point, so it is
  // open (ViewRows::open), and the points around the view are read one by one.
  template<class Set>
  static void write_threats(const Position& p, Set& set) {
    if(GameRules::terminal(p))
      return;
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    const View theirs = player_view(p, opponent), view = view_of(p);
    ViewRows rows;
    view_rows(p, theirs, rows);
    ViewHand hand;
    view_hand(p, opponent, hand);
    // The padded rows at index j + 1, so that each padded row j has a row on each side.
    constexpr int PADDED = VIEW + 2;
    ViewRow takeable[PADDED] = {}, stamped[PADDED + 2] = {};
    for(int y = 0; y < VIEW; y++)
      takeable[y + 1] = rows.open[mover][y] << 1;
    auto read = [&](int x, int y) {
      int rx, ry;
      theirs.rules_point(x, y, rx, ry);
      const int8_t v = at(p, rx, ry);
      if(cell_owner(v) == mover && cell_type(v) != WALL)
        takeable[y + 1] |= ViewRow::bit(x + 1);
    };
    for(int x = -1; x <= VIEW; x++) {
      read(x, -1);
      read(x, VIEW);
    }
    for(int y = 0; y < VIEW; y++) {
      read(-1, y);
      read(VIEW, y);
    }
    for(int k = 0; k < HAND; k++) {
      const Blueprint& b = hand.b[k];
      if(b.n == 0)
        continue;
      for(int y = 0; y < VIEW; y++) {
        const ViewRow fits = fit_row(b, rows.empty, rows.frontier[opponent], y);
        if(!fits)
          continue;
        for(int i = 0; i < b.n; i++) {
          const int s = b.q[i] + 1;  // the cell's bit, from its fit's anchor, in a padded row
          stamped[y + b.r[i] + 2] |= s >= 0 ? fits << s : fits >> -s;
        }
      }
    }
    for(int j = 0; j < PADDED; j++) {
      // The points next to a stamped point: stamped[j + 1] is the padded row j.
      const ViewRow hit = next_to_rows(stamped, j + 1) & takeable[j];
      for_each_bit(hit, [&](int i) {
        int rx, ry, x, y;
        theirs.rules_point(i - 1, j - 1, rx, ry);
        view.view_point(rx, ry, x, y);
        if(on_view(x, y))
          set(x, y, MINE_THREATENED);
      });
    }
  }

  // The legal hand actions (LEGAL_HAND + k): the hand's points of the mover's blueprint k (in the view's frame,
  // ViewHand) where a pick is legal now, or a drop after a pick, as legal_actions gives them (for_each_hand_action), at
  // the blueprint's slot (slot_of). The rules' legal actions: an equal blueprint's too, which the search does not list.
  template<class Set>
  static void write_legal_hand(const Position& p, const HandOrder* order, Set& set) {
    if(GameRules::terminal(p))
      return;
    const int mover = GameRules::next_player(p);
    ViewHand view;
    view_hand(p, mover, view);
    for_each_hand_action(p, view, [&](int k, int x, int y) { set(x, y, LEGAL_HAND + slot_of(order, mover, k)); });
  }

  // The minimap's bin (0 to VIEW - 1) of a point u of the minimap's frame, from its middle: the bins beyond its reach
  // hold the points beyond the edge.
  static int minimap_bin(int u) {
    const int a = u + MINIMAP_REACH;
    return a < 0 ? 0 : std::min(VIEW - 1, a / MINIMAP_BIN);
  }

  // The minimap (MINIMAP to THEIR_WINDOW), the whole board at a coarse scale, in the view's frame: the cells turned by
  // R^back(o) around the view's middle (view_middle), as the turned position (the view's in orientation 0) holds them,
  // then moved so that their box is in the middle, as the rules board recenters (center_box). So in orientation 0 the
  // bins are 5 x 5 points of the rules board around its middle (the board recenters after each placement). A
  // bin of planes MINIMAP + 3 * side + type holds a cell of that player (side 0 the mover) and type. MY_WINDOW and
  // THEIR_WINDOW: the bins that each player's view meets, a square of the turned frame: the mover's view (view_of: its
  // window, or a view that a caller set), and the view where the opponent places (player_view). The planes of the
  // position's image at a symmetry are the planes' image, a pixel map (spatial_inputs): the bins of the rules board map
  // to bins, and the box's middle rounds toward zero, as the recentering does.
  template<class Set>
  static void write_minimap(const Position& p, Set& set) {
    const int mover = GameRules::next_player(p);
    const View view = view_of(p);
    const int turnBack = back(view.o);
    // The turned point (u, v) of a rules point, from the rules board's middle.
    auto turned = [&](int x, int y, int& u, int& v) {
      int q = x - view.wx, r = y - view.wy;
      turn(q, r, turnBack);
      u = q + view.wx - RULES_RADIUS;
      v = r + view.wy - RULES_RADIUS;
    };
    struct Cell {
      int16_t u, v;
      int8_t plane;
    };
    static thread_local std::vector<Cell> cells;
    cells.clear();
    Box box;  // the turned cells' box
    const Box& used = p.cell.box();
    if(used.any)
      for(int y = used.y0; y <= used.y1; y++) {
        const int8_t* row = p.cell.row(y);
        for(int x = used.x0; x <= used.x1; x++)
          if(const int8_t c = row[x]) {
            int u, v;
            turned(x, y, u, v);
            cells.push_back({int16_t(u), int16_t(v), int8_t(3 * (cell_owner(c) != mover) + cell_type(c))});
            box.add(u, v);
          }
      }
    // The recentering's shift: the middle of the box, rounded toward zero (half_off_middle).
    const int du = box.any ? (box.x0 + box.x1) / 2 : 0, dv = box.any ? (box.y0 + box.y1) / 2 : 0;
    // Each bin's planes as bits (bit plane - MINIMAP), and the bins with any.
    static thread_local std::vector<uint8_t> marks(VIEW_AREA);
    static thread_local std::vector<int> marked;
    std::fill(marks.begin(), marks.end(), uint8_t(0));
    marked.clear();
    auto mark = [&](int bx, int by, int plane) {
      uint8_t& m = marks[by * VIEW + bx];
      if(!m)
        marked.push_back(by * VIEW + bx);
      m = uint8_t(m | 1u << (plane - MINIMAP));
    };
    for(const Cell& c : cells)
      mark(minimap_bin(c.u - du), minimap_bin(c.v - dv), MINIMAP + c.plane);
    for(int side = 0; side < 2; side++) {
      const View sideView = side == 0 ? view : player_view(p, 1 - mover);
      int u, v;
      turned(sideView.wx, sideView.wy, u, v);
      for(int by = minimap_bin(v - dv - VIEW_RADIUS); by <= minimap_bin(v - dv + VIEW_RADIUS); by++)
        for(int bx = minimap_bin(u - du - VIEW_RADIUS); bx <= minimap_bin(u - du + VIEW_RADIUS); bx++)
          mark(bx, by, side == 0 ? MY_WINDOW : THEIR_WINDOW);
    }
    for(int i : marked)
      for(int plane = MINIMAP; plane <= THEIR_WINDOW; plane++)
        if(marks[i] >> (plane - MINIMAP) & 1)
          set(i % VIEW, i / VIEW, plane);
  }

  // The area closed by a player's walls, on the rules board: f(x, y) for each of its points. Its components are the
  // finite connected components (the hex grid's 6 neighbours, DIRS) of the points that hold no wall of the player,
  // each with no cell of the other player (whose walls count as its cells). The other player can never place there: a
  // placement fills empty points of one component and touches the placer's cells, and no capture removes a wall. So no
  // cell inside can be captured either. The area holds the components' empty points and the player's cells that are
  // not walls. A component that reaches the rules board's border is open: the site's grid has no edge.
  //
  // A component that meets the edge of the box of the player's walls reaches beyond it, where no point holds a wall,
  // so it is open. In the box, each row's runs of points that hold no wall of the player join the runs of the row
  // before that they touch (a union-find of runs). A run is open on the box's edge, or with a cell of the other player.
  template<class F>
  static void for_each_closed(const Position& p, int player, F&& f) {
    const int8_t wall = cell_value(player, WALL);
    // The other player's cells, of any type: the values theirs to theirs + 2 (cell_value).
    const uint8_t theirs = uint8_t(cell_value(1 - player, PLAIN));
    const Box walls = wall_box(p, player);
    // A closed point has a wall on each side: the box spans 3 columns and 3 rows at least.
    if(!walls.any || walls.x1 - walls.x0 < 2 || walls.y1 - walls.y0 < 2)
      return;
    // A run: the points a to b of row y, a node of the union-find (parent), open at its root.
    struct Run {
      int16_t y, a, b;
      int parent;
      bool open;
    };
    static thread_local std::vector<Run> runs;
    runs.clear();
    auto root = [](int i) {
      while(runs[i].parent != i) {
        runs[i].parent = runs[runs[i].parent].parent;
        i = runs[i].parent;
      }
      return i;
    };
    int before = 0, start = 0;  // the runs of the row before: before to start - 1
    for(int y = walls.y0; y <= walls.y1; y++) {
      const int8_t* row = p.cell.row(y);
      start = int(runs.size());
      for(int x = walls.x0; x <= walls.x1;) {
        if(row[x] == wall) {
          x++;
          continue;
        }
        Run run{int16_t(y), int16_t(x), 0, int(runs.size()), y == walls.y0 || y == walls.y1 || x == walls.x0};
        for(; x <= walls.x1 && row[x] != wall; x++)
          run.open |= uint8_t(uint8_t(row[x]) - theirs) < 3;
        run.b = int16_t(x - 1);
        run.open |= run.b == walls.x1;
        runs.push_back(run);
      }
      // The point (x, y) touches (x, y - 1) and (x + 1, y - 1): the run a..b touches the points a..b + 1 of the row
      // before. Both rows' runs are in order and apart, so one pass joins each pair that touches.
      for(int i = start, j = before; i < int(runs.size()) && j < start;) {
        if(runs[j].b < runs[i].a)
          j++;
        else if(runs[j].a > runs[i].b + 1)
          i++;
        else {
          const int ri = root(i), rj = root(j);
          if(ri != rj) {
            runs[rj].parent = ri;
            runs[ri].open = runs[ri].open || runs[rj].open;
          }
          if(runs[j].b < runs[i].b + 1)
            j++;
          else
            i++;
        }
      }
      before = start;
    }
    for(int i = 0; i < int(runs.size()); i++)
      if(!runs[root(i)].open)
        for(int x = runs[i].a; x <= runs[i].b; x++)
          f(x, int(runs[i].y));
  }

  // The box of the player's walls on the rules board: each row's first wall (memchr) and last wall. On people's boards,
  // this takes half the time of a scan that compares each point of the cells' box.
  static Box wall_box(const Position& p, int player) {
    Box box;
    const Box& used = p.cell.box();
    if(!used.any)
      return box;
    const int8_t wall = cell_value(player, WALL);
    const int width = used.x1 - used.x0 + 1;
    for(int y = used.y0; y <= used.y1; y++) {
      const int8_t* row = p.cell.row(y) + used.x0;
      const void* first = std::memchr(row, wall, size_t(width));
      if(!first)
        continue;
      int last = width - 1;
      while(row[last] != wall)
        last--;
      box.add(used.x0 + int(static_cast<const int8_t*>(first) - row), y);
      box.add(used.x0 + last, y);
    }
    return box;
  }

  // The closed areas (MY_CLOSED, THEIR_CLOSED): the points of the view in the areas closed by the mover's walls and by
  // the opponent's walls (for_each_closed), in the view's orientation.
  template<class Set>
  static void write_closed_areas(const Position& p, Set& set) {
    const int mover = GameRules::next_player(p);
    const View view = view_of(p);
    for(int side = 0; side < 2; side++)
      for_each_closed(p, side == 0 ? mover : 1 - mover, [&](int rx, int ry) {
        int x, y;
        view.view_point(rx, ry, x, y);
        if(on_view(x, y))
          set(x, y, side == 0 ? MY_CLOSED : THEIR_CLOSED);
      });
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
  // The same at every symmetry, but the window boxes: no other input names a point. The scores are the effective ones,
  // and the margin's input gives the mover's effective margin linearly, where the squash flattens large margins. The
  // cells and stars outside the view count what the planes do not show. With options.extended, the window boxes
  // follow (box_inputs). With options.turnHistory, then the turn inputs: the edit flags of each turn's moves
  // (TURN_INPUT), with the history only (options.history), as the history planes.
  static void global_inputs(const Position& p, const GameInterface::InputOptions& options, float* out) {
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    for(int i = 0; i < NUM_GLOBAL_INPUTS; i++)
      out[i] = 0.0f;
    if(options.extended)
      box_inputs(p, options.symmetry, out + BOX_INPUT);
    if(options.turnHistory) {
      float* edits = out + (options.extended ? TURN_INPUT : NUM_GLOBAL_INPUTS);
      for(int h = 0; h < HISTORY; h++)
        for(int m = 0; m < TURN_MOVES; m++)
          edits[TURN_MOVES * h + m] = options.history && p.turns[h].edit[m] ? 1.0f : 0.0f;
    }
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
      out[10 + (options.canonical ? HandOrder(p).slot[mover][p.picked] : p.picked)] = 1.0f;
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
  // The window boxes of the image at the symmetry, the state of the window's rule (window_middle): each player's last
  // RECENT placements' boxes, the newest first, the mover's first, as BOX_FIELDS values each (BOX_INPUT): 1 when the
  // box is there, then the box's corners (x0, y0) and (x1, y1) from the view's middle (view_middle: the mover's
  // window's middle, or the middle of a view that a caller set), in the view's frame (turned by R^back(o)), / VIEW: the
  // first corner's x and the second's, then the first corner's y and the second's. In orientation 0 they are the box's
  // least and most x and y. A turned box is not a box, so its two corners stand for it. A symmetry maps the corners as
  // offsets (sym_offset), because both middles map as pixels, and the turn by 180 degrees and the second reflection (3
  // and 7) swap them: the trainer's map (katago/train/games/blueprints.py). Absent boxes hold 0.
  static void box_inputs(const Position& p, int symmetry, float* out) {
    const int mover = GameRules::next_player(p);
    Box recent[2][RECENT];
    for(int player = 0; player < 2; player++)
      for(int i = 0; i < RECENT; i++)
        recent[player][i] = symmetry == 0 ? p.recent[player][i] : sym_box(p.recent[player][i], symmetry);
    // The image's view middle: the view's middle maps as a pixel, a set view's (sym_records) and the mover's window's
    // (half_off_middle commutes with the symmetries).
    int wx, wy;
    view_middle(p, wx, wy);
    sym_rules(wx, wy, symmetry);
    const int turnBack = back(sym_orientation(p.orientation, symmetry));
    for(int side = 0; side < 2; side++)
      for(int i = 0; i < RECENT; i++) {
        const Box& b = recent[side == 0 ? mover : 1 - mover][i];
        float* f = out + BOX_FIELDS * (RECENT * side + i);
        std::fill(f, f + BOX_FIELDS, 0.0f);
        if(!b.any)
          continue;
        int q0 = b.x0 - wx, r0 = b.y0 - wy, q1 = b.x1 - wx, r1 = b.y1 - wy;
        turn(q0, r0, turnBack);
        turn(q1, r1, turnBack);
        f[0] = 1.0f;
        f[1] = float(q0) / VIEW;
        f[2] = float(q1) / VIEW;
        f[3] = float(r0) / VIEW;
        f[4] = float(r1) / VIEW;
      }
  }
  // What the inputs read beyond hash(): the points of the history of the actions and of the turns, and the turns' edit
  // flags. Both histories, so that the network's cache and RGSC's states are exact under each inputs version.
  static Hash128 input_hash(const Position& p) {
    Hash128 h;
    // An action's or a turn's points by the keys of its place in its ring (gamerules.h, KEY_HISTORY): from `near` a
    // point within KEY_RADIUS of the rules board's middle by its number there, a taken point after the stamped points,
    // and from `far` a point beyond by its site_key from the middle, a taken point at far + 1.
    auto add_points = [&h](uint64_t near, uint64_t far, const auto& f) {
      auto point_key = [near, far](int32_t index, bool taken) {
        const int q = index % RULES_LEN - RULES_RADIUS, r = index / RULES_LEN - RULES_RADIUS;
        if(std::abs(q) <= KEY_RADIUS && std::abs(r) <= KEY_RADIUS)
          return key(near + (taken ? KEY_AREA : 0) + uint64_t((r + KEY_RADIUS) * KEY_LEN + q + KEY_RADIUS), 0);
        return key(far + (taken ? 1 : 0), site_key(q, r));
      };
      for(int i = 0; i < f.nStamped; i++)
        h ^= point_key(f.stamped[i], false);
      for(int i = 0; i < f.nTaken; i++)
        h ^= point_key(f.taken[i], true);
    };
    for(int t = 0; t < HISTORY; t++) {
      const uint64_t ring = uint64_t(t) * 2;
      add_points(KEY_HISTORY + ring * KEY_AREA, KEY_FAR + ring * 2, p.history[t]);
      add_points(KEY_TURNS + ring * KEY_AREA, KEY_FAR + ring * 2 + 2, p.turns[t]);
      for(int m = 0; m < TURN_MOVES; m++)
        if(p.turns[t].edit[m])
          h ^= key(KEY_TURN_EDIT + uint64_t(t * TURN_MOVES + m), 0);
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

  // The network's frame of the mover's actions under the inputs version 13 (GameInterface): an action's blueprint at
  // its slot of the hand's order (HandOrder), its point as it is. Of equal blueprints, the search lists the actions of
  // the lowest-numbered one only (first_equal): a placement of the other one is the same placement, and its pick or a
  // drop into it gives the same position up to the hand's order. Its actions map to the listed one's policy index, so
  // that equal blueprints give one edge, with one prior and its visits. Equal blueprints have the same frame, so the
  // listed action has the same point (listed_action). The rules still allow both, as the site does.
  struct CanonicalActionMap {
    int8_t policyChannel[CHANNELS];  // each channel's channel in the network's frame
    int8_t listedChannel[CHANNELS];  // the channel of the listed action that an action of each channel stands for

    explicit CanonicalActionMap(const Position& p) {
      const HandOrder order(p);
      int8_t first[HAND];
      first_equal(p, first);
      const int mover = GameRules::next_player(p);
      for(int c = 0; c < CHANNELS; c++) {
        const int base = c < HAND_CHANNEL ? PLACE : HAND_CHANNEL, k = first[c - base];
        listedChannel[c] = int8_t(base + k);
        policyChannel[c] = int8_t(base + order.slot[mover][k]);
      }
    }
    int operator()(int action) const {
      if(action == GameRules::PASS_ACTION)
        return action;
      const int c = action_channel(action);
      return action - c + policyChannel[c];
    }
    bool listed(int action) const { return listed_action(action) == action; }
    // The listed action that an action stands for: itself, or the same point of the lowest-numbered equal blueprint.
    int listed_action(int action) const {
      if(action == GameRules::PASS_ACTION)
        return action;
      const int c = action_channel(action);
      return action - c + listedChannel[c];
    }
  };

  static constexpr unsigned SYMMETRIES = (1u << 0) | (1u << 3) | (1u << 4) | (1u << 7);

  // The image of everything but the cells on the board: the scores, the counts, the ply and the pick as they are, each
  // blueprint's cells at their images in the site's frame (the rest of a picked blueprint and the picked cell keep the
  // frame of the whole blueprint's image, as the rest keeps the whole blueprint's frame until the drop), the points of
  // the actions' and the turns' history, the recent placements' boxes, the offset and the view's orientation
  // (sym_orientation), and a view that a caller set: its middle's pixel and the rule's orientation's image.
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
    auto sym_points = [symmetry](int32_t* points, int n) {
      for(int i = 0; i < n; i++)
        points[i] = sym_rules_index(points[i], symmetry);
    };
    for(int h = 0; h < HISTORY; h++) {
      s.history[h] = p.history[h];
      sym_points(s.history[h].stamped, s.history[h].nStamped);
      sym_points(s.history[h].taken, s.history[h].nTaken);
      s.turns[h] = p.turns[h];
      sym_points(s.turns[h].stamped, s.turns[h].nStamped);
      sym_points(s.turns[h].taken, s.turns[h].nTaken);
    }
    for(int player = 0; player < 2; player++)
      for(int i = 0; i < RECENT; i++)
        s.recent[player][i] = sym_box(p.recent[player][i], symmetry);
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

  // The game's score spread against Go's (GAMES.md, "The score scale"): the median |margin| of people's games, 409
  // points, over KataGo's score multiplier of 20 (the engine repository's games/blueprints/README.md, "The network").
  static constexpr double SCORE_SCALE = 20.0;
  // The search's score utility unit: KataGo's square root of Go's board area, 19 points, in this game's points. The
  // square root of the game's largest score (Enclosure's unit) does not apply: Blueprints' scores have no bound.
  static constexpr double SCORE_UNIT = 19.0 * SCORE_SCALE;
  // A game is 120 of the site's moves (moves_played counts them; a pick and its drop are one).
  // KataGo's game-phase constants count the front's moves, which are actions: the history's moves, the searches and
  // the rows. An edit is two actions, a pick and its drop. So the game's mean length is in actions: 132, the mean of
  // bpj3's self-play games from the empty board (1,642 games of 7 networks: 132.0 to 132.7 for each network, 120 to 146
  // from the shortest game to the longest). Go's own games vary at least as much (GAMES.md, "The game's length").
  static constexpr double GAME_LENGTH = 132.0;
  // The view is 67 x 67 so that it holds the window, not because a decision reaches that far: KataGo's look-ahead
  // horizons read Go's 361, about 64, 20 and 6 actions in a game of about 132. And the komi sampler reads 361, so that
  // it draws a komi in the points that the config gives (GAMES.md, the encoding part's optional members).
  static constexpr double LOOKAHEAD_AREA = 19 * 19;
  static constexpr double KOMI_AREA = 19 * 19;
  // The even-komi search's bracket: offsets of 20, 40, 80 and so on around its guess, up to 3,000 points, the reach of
  // the widest komi of a new game in self-play (3 standard deviations of 1,000 points). KataGo's offsets of 1 to 32
  // points are less than 2 of Go's points at this SCORE_SCALE, and Blueprints' margins are hundreds of points. Beyond
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

// GameEncoding::Diversity, complete for whoever includes this file.
#include "diversity.h"
