#pragma once
// Blueprints' network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/blueprints/README.md describes the inputs, the targets and the score.
//
// - The inputs are the view: the VIEW x VIEW points of the rules board around the mover's window (gamerules.h,
//   view_origin). After the front's on-board plane, from the mover's side: each player's cells by type, each player's
//   frontier (the empty points next to its cells), each player's cells open to a stamp (not walls, next to an empty
//   point), and for each blueprint of each player the anchors in the view where it fits, where it takes at least 1,
//   2, 3 and 4 enemy stars, and where it scores at least 25, 100 and 400 points (the stars that it takes times the
//   player's stars on the board), and each player's cells that some placement of the other player takes. Then each
//   player's blueprints, drawn at the hand's points (gamerules.h, HAND_X and HAND_Y), where the policy's picks and
//   drops are, and the picked cell, a plane for each blueprint. Then the history: the points that each of the last 4
//   actions stamped and took.
// - The global inputs: the mover's seat, its moves left in the turn, the game's progress, the stars on the board, the
//   effective scores (with the komi), the pick, each player's cells and stars outside the view, and the mover's
//   effective margin.
// - The komi K is points added to the second player's final margin (KataGo's White), for the whole game. The rules'
//   scores stay the earned ones, and the front's history adds K to the result (BoardHistory::endAndScoreGameNow). The
//   inputs read K as a starting score of the side that it favours, the effective scores S0 + max(-K, 0) and S1 +
//   max(K, 0). No rule reads a score, so a position at komi K plays as the position with those scores at komi 0.
// - The policy: 6 channels a point, the actions of gamerules.h (placements, then the cells of the hand: picks, or
//   drops after a pick).
// - The symmetries: the 4 of KataGo's 8 that keep the hex neighbours (DIRS) on the axial square: the identity, the
//   turn by 180 degrees (3) and two reflections (4, 7). The image maps the rules board around its middle. The window's
//   rule and the recentering round toward zero, so the image's window is the window's image, and a point of the view
//   moves as a pixel. A blueprint moves cell by cell, then goes back to the site's frame (its lowest cell at 0,0): so
//   the hand's points, the anchors of the placements and the picks and drops move as pixels and then by each
//   blueprint's shift. The inputs at a symmetry are the inputs of the position's image (sym_position).
// - The training targets: the final board as ownership on the row's view (C0), and in C1 which player moves next, for
//   the trainer's symmetries (no loss reads it). The value targets and the score targets are KataGo's, from the
//   history's result: the final margin with the komi, where a margin of 0 is half a win.

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
  bool any = false;
  mq = mr = 0;
  auto consider = [&](int q, int r) {
    sym_offset(q, r, symmetry);
    if(!any || r < mr || (r == mr && q < mq)) {
      mq = q;
      mr = r;
      any = true;
    }
  };
  for(int i = 0; i < b.n; i++)
    consider(b.q[i], b.r[i]);
  if(withExtra)
    consider(extraQ, extraR);
}

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
    int ox, oy;
    view_origin(p, ox, oy);
    // Each player's frontier in the view, as rules points (frontier_near gives the same), for the placements below.
    static thread_local std::vector<int32_t> frontier[2];
    frontier[0].clear();
    frontier[1].clear();
    for(int y = 0; y < VIEW; y++)
      for(int x = 0; x < VIEW; x++) {
        const int rx = x + ox, ry = y + oy;
        if(!on_rules(rx, ry))
          continue;
        const int8_t v = p.cell[ry * RULES_LEN + rx];
        if(v) {
          const int owner = cell_owner(v), type = cell_type(v);
          set(x, y, (owner == mover ? MINE : THEIRS) + type);
          if(type != WALL) {
            bool open = false;
            for(const auto& d : DIRS)
              open = open || (on_rules(rx + d[0], ry + d[1]) && !p.cell[(ry + d[1]) * RULES_LEN + rx + d[0]]);
            if(open)
              set(x, y, owner == mover ? MY_OPEN : THEIR_OPEN);
          }
        }
        else
          for(int player : {mover, opponent})
            if(next_to_own(p, rx, ry, player)) {
              set(x, y, player == mover ? MY_FRONTIER : THEIR_FRONTIER);
              frontier[player].push_back(ry * RULES_LEN + rx);
            }
      }
    // The placements of each player in the view, on the board as it is: the mover's when no pick waits for its drop,
    // and the opponent's as if it moved now (its threats, which score with its own stars).
    if(!GameRules::terminal(p)) {
      for(int player : {mover, opponent}) {
        if(player == mover && p.picked >= 0)
          continue;
        const int places = player == mover ? MY_PLACES : THEIR_PLACES;
        const int taken = player == mover ? THEIRS_TAKEN : MINE_TAKEN;
        for(int k = 0; k < HAND; k++) {
          const Blueprint& b = p.hand[player][k];
          for_each_placement(p, player, b, ox, oy, frontier[player], [&](int x, int y) {
            set(x, y, places + k);
            int32_t points[MAX_TAKEN];
            const int n = taken_by(p, player, b, x + ox, y + oy, points);
            int stars = 0;
            for(int i = 0; i < n; i++) {
              stars += cell_type(p.cell[points[i]]) == STAR;
              const int tx = points[i] % RULES_LEN - ox, ty = points[i] / RULES_LEN - oy;
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
        const Blueprint& b = p.hand[player][k];
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
      set(p.pickQ + HAND_X, p.pickR + HAND_Y, PICKED + p.picked);
    if(history)
      for(int h = 0; h < HISTORY; h++) {
        const Footprint& f = p.history[h];
        auto mark = [&](int32_t point, int plane) {
          const int x = point % RULES_LEN - ox, y = point / RULES_LEN - oy;
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
    int ox, oy;
    view_origin(p, ox, oy);
    int cellsIn[2] = {0, 0}, starsIn[2] = {0, 0};
    for(int y = std::max(0, oy); y < std::min(RULES_LEN, oy + VIEW); y++)
      for(int x = std::max(0, ox); x < std::min(RULES_LEN, ox + VIEW); x++) {
        const int8_t v = p.cell[y * RULES_LEN + x];
        if(v) {
          cellsIn[cell_owner(v)]++;
          starsIn[cell_owner(v)] += cell_type(v) == STAR;
        }
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
  // points, the recent placements' boxes and the offset.
  static void sym_records(const Position& p, int symmetry, Position& s) {
    std::copy(p.scores, p.scores + 2, s.scores);
    std::copy(p.stars, p.stars + 2, s.stars);
    std::copy(p.cells, p.cells + 2, s.cells);
    s.ply = p.ply;
    s.picked = p.picked;
    s.pickQ = p.pickQ;
    s.pickR = p.pickR;
    s.pickType = p.pickType;
    s.pickConnected = p.pickConnected;
    const int mover = GameRules::next_player(p);
    for(int player = 0; player < 2; player++)
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[player][k];
        const bool rest = player == mover && k == p.picked;
        int mq, mr;
        frame_shift(b, symmetry, rest, p.pickQ, p.pickR, mq, mr);
        Blueprint& image = s.hand[player][k];
        image = b;
        for(int i = 0; i < b.n; i++) {
          int q = b.q[i], r = b.r[i];
          sym_offset(q, r, symmetry);
          image.q[i] = int8_t(q - mq);
          image.r[i] = int8_t(r - mr);
        }
        if(rest) {
          int q = p.pickQ, r = p.pickR;
          sym_offset(q, r, symmetry);
          s.pickQ = int8_t(q - mq);
          s.pickR = int8_t(r - mr);
        }
      }
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
  // s's cells' hash, by the image's site coordinates.
  static void sym_cells(const Position& p, int symmetry, int x0, int x1, int y0, int y1, Position& s) {
    s.cellHash = Hash128();
    for(int y = y0; y < y1; y++)
      for(int x = x0; x < x1; x++) {
        const int8_t v = p.cell[y * RULES_LEN + x];
        if(!v)
          continue;
        int sx = x, sy = y;
        sym_rules(sx, sy, symmetry);
        s.cell[sy * RULES_LEN + sx] = v;
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
    std::fill(s.cell, s.cell + RULES_AREA, int8_t(0));
    sym_records(p, symmetry, s);
    int ox, oy;
    view_origin(p, ox, oy);
    sym_cells(p, symmetry, std::max(0, ox - INPUT_REACH), std::min(RULES_LEN, ox + VIEW + INPUT_REACH),
              std::max(0, oy - INPUT_REACH), std::min(RULES_LEN, oy + VIEW + INPUT_REACH), s);
    return s;
  }

  // A legal action's image, an action of sym_position: a placement's anchor is the lowest point of its stamped cells'
  // images, which is the anchor's pixel moved by the blueprint's shift (frame_shift), in the image's view (the window's
  // image). A pick's or a drop's hand point moves by the linear part, then by its blueprint's shift. The trainer's
  // module maps both the same way.
  struct SymActionMap {
    int symmetry;
    Blueprint hand[HAND];  // the mover's
    int picked, pickQ, pickR;
    int ox, oy, imageOx, imageOy;  // the view's origin on the rules board, and the image's
    SymActionMap(const Position& p, int sym)
      : symmetry(sym), picked(p.picked), pickQ(p.pickQ), pickR(p.pickR) {
      for(int k = 0; k < HAND; k++)
        hand[k] = p.hand[GameRules::next_player(p)][k];
      view_origin(p, ox, oy);
      int wx = ox + VIEW_RADIUS, wy = oy + VIEW_RADIUS;
      sym_rules(wx, wy, sym);
      imageOx = wx - VIEW_RADIUS;
      imageOy = wy - VIEW_RADIUS;
    }
    int operator()(int action) const {
      if(symmetry == 0 || action == GameRules::PASS_ACTION)
        return action;
      const int x = action_x(action), y = action_y(action), channel = action_channel(action);
      int mq, mr;
      if(channel < HAND_CHANNEL) {
        int ax = x + ox, ay = y + oy;
        sym_rules(ax, ay, symmetry);
        frame_shift(hand[channel - PLACE], symmetry, false, 0, 0, mq, mr);
        return action_at(ax + mq - imageOx, ay + mr - imageOy, channel);
      }
      const int k = channel - HAND_CHANNEL;  // a pick from blueprint k, or a drop into it after a pick
      frame_shift(hand[k], symmetry, k == picked, pickQ, pickR, mq, mr);
      int q = x - HAND_X, r = y - HAND_Y;
      sym_offset(q, r, symmetry);
      return action_at(q - mq + HAND_X, r - mr + HAND_Y, channel);
    }
  };

  // C0, the final board as ownership, on the row's view: +1 on the mover's cells, -1 on the opponent's, 0 on an empty
  // point. The final cells move by their site coordinates into the row's rules board, then into its view, and a final
  // cell off the row's view counts for no point.
  // C1 is no target: the slot carries which player makes the row's next move, for the trainer's symmetries of the
  // second policy target (katago/train/games/blueprints.py): +1 on every point when it is the row's mover, -1 when it
  // is the opponent, 0 at the end. The front stores the slot as it is, and its flag stays 0 (the returned slots hold
  // no TARGET_SEKI), so no loss reads it. A pick and a placement with a move to come leave the move to the mover.
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP;
  static constexpr int NEXT_MOVER_SLOT = 1;
  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int row, int mover, float* values) {
    const Position& end = positions(numPositions - 1);
    const Position& now = positions(row);
    int ox, oy;
    view_origin(now, ox, oy);
    const int dx = end.offsetQ - now.offsetQ - ox, dy = end.offsetR - now.offsetR - oy;
    for(int i = 0; i < VIEW_AREA; i++)
      values[i] = 0.0f;
    float next = 0.0f;
    if(row + 1 < numPositions && !GameRules::terminal(positions(row + 1)))
      next = GameRules::next_player(positions(row + 1)) == mover ? 1.0f : -1.0f;
    for(int i = 0; i < VIEW_AREA; i++)
      values[NEXT_MOVER_SLOT * VIEW_AREA + i] = next;
    for(int y = 0; y < RULES_LEN; y++)
      for(int x = 0; x < RULES_LEN; x++) {
        const int owner = cell_owner(end.cell[y * RULES_LEN + x]);
        if(owner >= 0 && on_view(x + dx, y + dy))
          values[(y + dy) * VIEW + x + dx] = owner == mover ? 1.0f : -1.0f;
      }
    return GameInterface::TARGET_OWNERSHIP;
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
  // self-play's widest komi (3 standard deviations of 1,000 points). KataGo's offsets of 1 to 32 points are less than 3
  // of Go's points at this SCORE_SCALE, and Blueprints' margins are hundreds of points.
  static constexpr double KOMI_BRACKET_STEP = SCORE_SCALE;
  static constexpr double KOMI_BRACKET_LIMIT = 3000.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Blueprints' network encoding part");

}  // namespace blueprints
