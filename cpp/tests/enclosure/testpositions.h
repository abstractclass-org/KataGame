#ifndef TESTS_ENCLOSURE_TESTPOSITIONS_H_
#define TESTS_ENCLOSURE_TESTPOSITIONS_H_

//The positions of a test's file (tests/data/<game>_positions.txt), for testnnoutputs and testdiversityinputs. A line is
//  position LABEL N ACTION ...   the game's actions from the start of its standard board
//  state LABEL KOMI TEXT         a state of the game (GameInterface::stateText), at that komi, on any board
//Other lines are comments. A position whose actions are not all legal, or that is the end of its game, is left out.

#include <istream>
#include <sstream>
#include <string>

#include "../../core/global.h"
#include "../../game/boardhistory.h"

//Reads the next position of the file. False at its end. A state that does not parse throws.
inline bool nextTestPosition(std::istream& in, std::string& label, Board& board, BoardHistory& hist) {
  std::string line;
  while(std::getline(in, line)) {
    std::istringstream head(line);
    std::string word;
    if(!(head >> word >> label))
      continue;
    Rules rules;
    if(word == "state") {
      std::string text, why;
      double komi;
      GameRules::Position p;
      if(!(head >> komi) || !std::getline(head >> std::ws, text) ||
         !GameInterface::parseState<GameRules>(Global::trim(text), p, why))
        throw StringError("the state " + label + " does not parse: " + why);
      rules.komi = (float)komi;
      board = Board();
      board.setPosition(p);
      hist = BoardHistory(board, board.nextPla(), rules, 0);
    }
    else {
      int n;
      if(word != "position" || !(head >> n))
        continue;
      board = Board();
      hist = BoardHistory(board, P_BLACK, rules, 0);
      bool legal = true;
      for(int i = 0; i < n && legal; i++) {
        int action;
        head >> action;
        const Loc loc = Board::moveLocOfAction(action);
        const Player pla = hist.presumedNextMovePla;
        legal = hist.isLegal(board, loc, pla);
        if(legal)
          hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      }
      if(!legal)
        continue;
    }
    if(!hist.isGameFinished)
      return true;
  }
  return false;
}

#endif  // TESTS_ENCLOSURE_TESTPOSITIONS_H_
