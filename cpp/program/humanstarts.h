#ifndef PROGRAM_HUMANSTARTS_H_
#define PROGRAM_HUMANSTARTS_H_

//KataGame fork (2026-11-11, abscls-7ald): self-play's starts from positions of people's games. A share of the games
//(humanStartProb) starts from a state of a file (humanStartsFile), drawn uniformly over the file's states. The file
//holds the states as the game's texts (GameInterface::stateText), so only a game that gives one (Blueprints) has
//them. The engine repository's games/blueprints/tools/human_starts.py writes the file from strong people's games,
//and its games/blueprints/README.md, "Human starts", gives the method.
//
//A start is RGSC's exact state (program/rgsc.h) with a context of its own: the komi 0, as in the people's game, and no
//asymmetric playouts, frozen (Sgf::PositionSample::frozenContext), so that no komi is drawn and none compensates, and
//the game's result teaches the position's own value. The state keeps its turn number (the actions before it), which
//the move's temperature reads. Then the game's own draws at a start (GameInterface::drawStart): Blueprints draws the
//view's orientation under viewOrientation = randomGame, as for a game from the standard start.
//
//Self-play marks the game as RGSC marks its restarts: its rows' C74 holds Rgsc::SOURCE_HUMAN, AZdb's games file its
//rgsc_source, and its SGF record gtype=humanpos. It is member 0 against itself, and AZdb's league takes nothing from
//it, as from RGSC's restarts. It offers RGSC's buffer no candidate, and it takes no fork.
//
//The file's lines are JSON objects. The first is the header: {"format": "human-starts", "version": 1, ...}, with the
//source's description for the dashboard (the other fields). Each other line is a start: {"state": TEXT, "turn": N, ...}
//(the tool adds the game and the site's move it comes from). The load checks each state: its text gives a state back
//(GameInterface::parseState, under the game's settings, such as the rule of the view's orientation), and that state's
//text is the line's own. A state that fails is rejected and counted, as is a state already in the file. The draw takes
//the states that the load kept.
//
//With humanStartProb 0 (the default), self-play draws nothing and runs as without this file.

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../dataio/sgf.h"
#include "../program/rgsc.h"

namespace HumanStarts {

  //The format of the file's header line, and its version.
  extern const char* FORMAT;
  constexpr int VERSION = 1;

  struct Settings {
    //The share of self-play's games that start from the file. 0: none.
    double prob = 0.0;
    std::string file;
    static Settings load(ConfigParser& cfg);
    bool enabled() const { return prob > 0.0; }
  };

  //The context of every start: the komi 0 and no asymmetric playouts.
  Rgsc::Context context();

  class Starts {
   public:
    //Loads the file. Throws when it cannot be read, its header is not one of this format and version, or no state is
    //kept.
    void load(const std::string& file);
    //Loads the lines of a stream, the same way (the file's, or a test's), a line at a time.
    void read(std::istream& in);

    int64_t rejected() const { return numRejected; }
    int64_t duplicates() const { return numDuplicates; }
    //The states that the load kept, in the file's order: each state's text, key (Rgsc::stateKey with context()) and
    //turn number.
    const std::vector<Rgsc::Entry>& states() const { return entries; }
    //The load's summary for the log.
    std::string summary() const;

    //Draws a start uniformly over the states: entry gets its state, and start the game's exact start from it
    //(Rgsc::startOf), with the game's draws at its start (GameInterface::drawStart). False, with the reason, when the
    //state does not come back with its key.
    bool draw(Rand& rand, int xSize, int ySize, Rgsc::Entry& entry, Sgf::PositionSample& start, std::string& why)
      const;

   private:
    std::vector<Rgsc::Entry> entries;
    int64_t numRejected = 0;
    int64_t numDuplicates = 0;
    //The first reasons of the rejections, for the summary.
    std::vector<std::string> firstRejections;
  };
}

#endif  // PROGRAM_HUMANSTARTS_H_
