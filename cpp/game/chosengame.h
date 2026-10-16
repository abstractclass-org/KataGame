#ifndef GAME_CHOSENGAME_H_
#define GAME_CHOSENGAME_H_

//The game that this build plays, both parts of its interface (game/gameinterface.h): GameRules (game/chosenrules.h
//chooses the game) and GameEncoding, the network encoding part.

#include "chosenrules.h"

#include KATAGAME_GAME_HEADER(KATAGAME_GAME, gameencoding.h)

using GameEncoding = chosengame::GameEncoding;

#endif  // GAME_CHOSENGAME_H_
