// Blueprints' moves in strategyjam.org's strings, and the engine's actions (the engine repository's
// games/blueprints/README.md, "The port" and "The browser build"): a placement "K@q,r" (blueprint K's anchor cell on
// the board's hex q,r), a pick or a drop "K:q,r" (a cell of blueprint K in its own frame), the pick and the drop of an
// edit joined by ">", and "pass". Blueprints' engine protocol (command/blueprintsprotocol.cpp) and its practice bot in
// the browser (the engine repository's games/blueprints/wasm/practice.cpp) read and write the site's moves with these.
#ifndef COMMAND_BLUEPRINTSSITE_H_
#define COMMAND_BLUEPRINTSSITE_H_

#include <algorithm>
#include <string>

#include "../game/blueprints/gamerules.h"

namespace BlueprintsSite {
  // The site's hex of the rules board's point (x, y).
  inline int siteQ(const blueprints::Position& p, int x) { return x - blueprints::RULES_RADIUS + p.offsetQ; }
  inline int siteR(const blueprints::Position& p, int y) { return y - blueprints::RULES_RADIUS + p.offsetR; }
  // The rules board's point of the site's hex (q, r), on the board or off it.
  inline int rulesX(const blueprints::Position& p, int q) { return q - p.offsetQ + blueprints::RULES_RADIUS; }
  inline int rulesY(const blueprints::Position& p, int r) { return r - p.offsetR + blueprints::RULES_RADIUS; }

  // The site's placement "K@q,r" of blueprint k with its anchor cell on the rules point (x, y).
  inline std::string placementText(const blueprints::Position& p, int k, int x, int y) {
    return std::to_string(k + 1) + "@" + blueprints::cell_text(siteQ(p, x), siteR(p, y));
  }

  // The site's string of the engine's action at p, pass excepted: a placement names its anchor hex in the board's
  // coordinates, a pick or a drop its cell in the blueprint's frame (the view's turns undone, gamerules.h ViewHand).
  inline std::string siteText(const blueprints::Position& p, int action) {
    namespace bp = blueprints;
    const int x = bp::action_x(action), y = bp::action_y(action), channel = bp::action_channel(action);
    bp::ViewHand hand;
    bp::view_hand(p, bp::turn_of(p.ply), hand);
    if(channel < bp::HAND_CHANNEL) {
      const int k = channel - bp::PLACE;
      int rx, ry;
      bp::view_of(p).rules_point(x - hand.mq[k], y - hand.mr[k], rx, ry);
      return placementText(p, k, rx, ry);
    }
    const int k = channel - bp::HAND_CHANNEL;
    int q, r;
    hand.site_cell(k, x, y, q, r);
    return std::to_string(k + 1) + ":" + bp::cell_text(q, r);
  }

  // Whether the action is a pick: a cell of the hand when no pick waits for its drop.
  inline bool isPick(const blueprints::Position& p, int action) {
    return action != blueprints::GameRules::PASS_ACTION &&
           blueprints::action_channel(action) >= blueprints::HAND_CHANNEL && p.picked < 0;
  }

  // Whether the mover's blueprint k fits with its anchor cell on the rules point (x, y), on the board or off it, by the
  // site's rule on its grid with no edge: every hex of its cells empty (a hex off the rules board is empty, as bp::at
  // reads it), and one of them touching a cell of the mover.
  inline bool siteFits(const blueprints::Position& p, int k, int x, int y) {
    namespace bp = blueprints;
    const int mover = bp::turn_of(p.ply);
    const bp::Blueprint& b = p.hand[mover][k];
    if(b.n == 0)
      return false;
    bool touches = false;
    for(int i = 0; i < b.n; i++) {
      const int cx = x + b.q[i], cy = y + b.r[i];
      if(bp::at(p, cx, cy))
        return false;
      touches = touches || bp::next_to_own(p, cx, cy, mover);
    }
    return touches;
  }

  // The hexes that the cells (their box, bp::cells_box) and blueprint b with its anchor cell on (x, y) span, in q or in
  // r, whichever is more. The rules board holds at most RULES_LEN.
  inline int spanWith(blueprints::Box box, const blueprints::Blueprint& b, int x, int y) {
    for(int i = 0; i < b.n; i++)
      box.add(x + b.q[i], y + b.r[i]);
    return std::max(box.x1 - box.x0, box.y1 - box.y0) + 1;
  }

  inline std::string limitText(int span) {
    return "the cells would span " + std::to_string(span) + " hexes, and this build's rules board holds " +
           std::to_string(blueprints::RULES_LEN) + " x " + std::to_string(blueprints::RULES_LEN);
  }

  // A site's cell string "K:q,r" (blueprint K from 1, the cell q,r in its frame) or placement "K@q,r" (its anchor
  // hex): K, q and r. False for other text.
  inline bool parseSiteMove(const std::string& text, char sep, int& k, int& q, int& r) {
    if(text.size() < 5 || text[1] != sep || text[0] < '1' || text[0] > '0' + blueprints::HAND)
      return false;
    k = text[0] - '1';
    return blueprints::parse_cell(text.substr(2), q, r);
  }

  // The engine's action of the site's pick or drop "K:q,r" at p (its cell in blueprint K's frame), when it is legal and
  // names that cell; else -1.
  inline int handAction(const blueprints::Position& p, const std::string& text) {
    namespace bp = blueprints;
    int k, q, r;
    if(!parseSiteMove(text, ':', k, q, r))
      return -1;
    bp::ViewHand hand;
    bp::view_hand(p, bp::turn_of(p.ply), hand);
    int x, y;
    hand.hand_point(k, q, r, x, y);
    if(!bp::on_view(x, y))
      return -1;
    const int action = bp::action_at(x, y, bp::HAND_CHANNEL + k);
    return bp::GameRules::is_legal(p, action) && siteText(p, action) == text ? action : -1;
  }

  // The engine's action of the site's placement `text` at p, blueprint k with its anchor cell on the site's hex (q, r),
  // in p's view: the action that names it, when its cells all lie in the view and it is legal; else -1.
  inline int placementAction(const blueprints::Position& p, int k, int q, int r, const std::string& text) {
    namespace bp = blueprints;
    // The view's hex of the anchor cell's image: the anchor cell's hex in the view, plus the blueprint's shift in the
    // view's frame (gamerules.h, play).
    bp::ViewHand hand;
    bp::view_hand(p, bp::turn_of(p.ply), hand);
    int vx, vy;
    bp::view_of(p).view_point(rulesX(p, q), rulesY(p, r), vx, vy);
    vx += hand.mq[k];
    vy += hand.mr[k];
    if(!bp::on_view(vx, vy))
      return -1;
    const int action = bp::action_at(vx, vy, bp::PLACE + k);
    return bp::GameRules::is_legal(p, action) && siteText(p, action) == text ? action : -1;
  }
}

#endif  // COMMAND_BLUEPRINTSSITE_H_
