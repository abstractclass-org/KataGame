#ifndef GAME_CHOSENGAME_H_
#define GAME_CHOSENGAME_H_

//The game that this build plays, both parts of its interface (game/gameinterface.h): GameRules (game/chosenrules.h
//chooses the game) and GameEncoding, the network encoding part.

#include "chosenrules.h"

#include KATAGAME_GAME_HEADER(KATAGAME_GAME, gameencoding.h)

using GameEncoding = chosengame::GameEncoding;

//katago's build gives the network's board, MAX_BOARD_LEN of the game's trainer module
//(python/katago/train/games/<game>.py), where the module has it as a number, and 0 without one (CMakeLists.txt).
#ifdef KATAGAME_NETWORK_BOARD_LEN
static_assert(KATAGAME_NETWORK_BOARD_LEN == 0 || KATAGAME_NETWORK_BOARD_LEN == GameRules::MAX_BOARD_LEN,
              "the network's board (MAX_BOARD_LEN of the trainer's module) is the game's largest board");
#endif

#endif  // GAME_CHOSENGAME_H_
