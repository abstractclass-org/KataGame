#ifndef GAME_BOTINTERFACE_H_
#define GAME_BOTINTERFACE_H_

/*
 * The bot part of the game interface: what a game supplies to the alpha-beta search (search/alphabeta.h), beside its
 * rules part (game/gameinterface.h, GAMES.md).
 *
 * A game's bot part is struct GameBot in game/<name>/gamebot.h, which ends with checkBot<GameBot>(). The Python
 * binding includes it when the file exists (game/chosenbot.h). katago does not compile it: the alpha-beta search is a
 * bot of its own, and KataGo's search does not call it.
 *
 * The bot part (struct B):
 *   Rules                   the game's rules part (its GameRules).
 *   evaluate(p)             double: a heuristic value of a position that is not terminal, from the view of the player
 *                           to move (Rules::next_player(p)). Higher is better for that player. The search keeps it
 *                           within +-WIN_VALUE / 2 (search/alphabeta.h), below every won or lost end.
 *   order_actions(p, actions)
 *                           optional, void: reorders the legal actions (std::vector<int>&), the most promising first.
 *                           Equal actions keep their order (a stable sort): the search's random order of ties is that
 *                           order. The search then puts the transposition table's move first. Without it, the actions
 *                           keep the search's order: ascending, or random with a seed.
 */

#include <type_traits>
#include <utility>
#include <vector>

#include "gameinterface.h"

namespace GameInterface {

namespace bot {
  using encoding::HasRules;  // the encoding part's check: a Rules with a Position
  template<class G> using P = encoding::P<G>;
  template<class G, class = void> struct HasEvaluate : std::false_type {};
  template<class G> struct HasEvaluate<G, std::enable_if_t<std::is_convertible<
    decltype(G::evaluate(std::declval<const P<G>&>())), double>::value>> : std::true_type {};
  template<class G, class = void> struct HasOrderActions : std::false_type {};
  template<class G> struct HasOrderActions<G, std::void_t<
    decltype(G::order_actions(std::declval<const P<G>&>(), std::declval<std::vector<int>&>()))>> : std::true_type {};
}

template<class B> constexpr bool checkBot() {
  using namespace bot;
  static_assert(HasRules<B>::value, "game interface, bot part: the game lacks Rules (its rules part, with a Position)");
  static_assert(HasEvaluate<B>::value,
                "game interface, bot part: the game lacks evaluate(const Position&) -> double");
  return true;
}

//The game's move ordering when it has one (order_actions, optional). Without it, the actions keep the search's order.
template<class B> void orderActions(const typename B::Rules::Position& p, std::vector<int>& actions) {
  if constexpr(bot::HasOrderActions<B>::value)
    B::order_actions(p, actions);
}

}  // namespace GameInterface

#endif  // GAME_BOTINTERFACE_H_
