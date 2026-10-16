#ifndef GAME_CHOSENRULES_H_
#define GAME_CHOSENRULES_H_

//The rules part of the game that this build plays (game/gameinterface.h), alone: the Python binding builds against
//it. game/chosengame.h adds the network encoding part.
//
//KATAGAME_GAME names the game: its folder in game/ and its namespace. The CMake option GAME and the Python binding's
//build define it. A build that defines nothing (the WebAssembly module, the fork's standalone tests) gets Enclosure.
//
//`chosengame` is the chosen game's namespace. GameRules is its rules part.

#ifndef KATAGAME_GAME
#define KATAGAME_GAME enclosure
#endif

#define KATAGAME_STRING(x) #x
#define KATAGAME_GAME_HEADER(game, file) KATAGAME_STRING(game/file)

#include KATAGAME_GAME_HEADER(KATAGAME_GAME, gamerules.h)
namespace chosengame = KATAGAME_GAME;

using GameRules = chosengame::GameRules;

#endif  // GAME_CHOSENRULES_H_
