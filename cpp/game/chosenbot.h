#ifndef GAME_CHOSENBOT_H_
#define GAME_CHOSENBOT_H_

//The bot part of the game that this build plays (game/botinterface.h): GameBot, beside its rules part
//(game/chosenrules.h chooses the game). The Python binding includes it when the game has game/<name>/gamebot.h.

#include "chosenrules.h"

#include KATAGAME_GAME_HEADER(KATAGAME_GAME, gamebot.h)

using GameBot = chosengame::GameBot;

#endif  // GAME_CHOSENBOT_H_
