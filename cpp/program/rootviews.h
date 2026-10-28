#ifndef PROGRAM_ROOTVIEWS_H_
#define PROGRAM_ROOTVIEWS_H_

#include <functional>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../search/asyncbot.h"
#include "../external/nlohmann_json/json.hpp"

//KataGame fork (abscls-il4e.3, abscls-il4e.4): the root views, for a game whose network sees a part of a larger board
//(Blueprints: a view of 67 x 67 points of its rules board). The game gives views that hold every legal move of the root
//between them (GameInterface::viewCover). With one view, the search is the one of that view: the usual search when the
//rules' view holds every move. With more, each view gets a search of the root in that view (a pass) with a share of the
//visits. Each pass's best moves are the candidates, and a move that two passes find is one candidate (the same child
//position). The candidates with the best child values from the root mover's side then get a search of their child
//position each (a check), with the rest of the visits, and the move is the check with the best value. The passes and
//the checks take the first pass's score centre, so that their utilities have one scale. Their visits are not compared.
//The analysis engine's rootViews (command/analysis.cpp) and GTP's genmove (command/gtp.cpp) both search so.
namespace RootViews {
  struct Settings {
    double passShare = 0.5;  //rootViewsPassShare: the passes' share of the visits, the playouts and the time
    int candidates = 2;      //rootViewsCandidates: each pass's best moves that become candidates
    int finalists = 3;       //rootViewsFinalists: the candidates that get a check
  };
  //The config's rootViewsPassShare, rootViewsCandidates and rootViewsFinalists, each the default above when absent.
  Settings loadSettings(ConfigParser& cfg);

  //The views of a cover of the board's position (GameInterface::viewCover): {""} when the rules' view holds every
  //legal move.
  std::vector<std::string> cover(const Board& board);

  //Sets the game's view on the board (GameInterface::setView), or returns false with the reason. "" leaves it as it is.
  bool setBoardView(Board& board, const std::string& view, std::string& why);

  //What each search reports: the arguments of Search::getAnalysisJson.
  struct Report {
    Player perspective = C_EMPTY;
    int analysisPVLen = 15;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
  };

  //The move that the search chose: a move of the root's own view (view ""), or a move that only another view names,
  //with that view (loc is then the move in it).
  struct Pick {
    Loc loc = Board::NULL_LOC;
    std::string view;
  };

  //The search of the root's views (above), of the position's cover views: writes the analysis engine's response in
  //ret and the chosen move in pick, or returns false when a search was stopped or had no result. Each search takes its
  //share of the search's limits and of the clock tc times searchFactor (genMoveSynchronous's searchFactor), within the
  //clock's limit for the root's move, so the searches together take the limits and the clock's time of one search.
  //The first search calls onSearchBegun, and a later one stops at once when stopped() is true (an empty stopped:
  //never). The bot's position is the last search's at the end.
  bool search(
    AsyncBot* bot, const Board& rootBoard, const BoardHistory& rootHist, Player rootPla,
    const std::vector<std::string>& views, const TimeControls& tc, double searchFactor,
    const Settings& settings, const Report& report,
    const std::function<void()>& onSearchBegun, const std::function<bool()>& stopped,
    nlohmann::json& ret, Pick& pick
  );
}

#endif  // PROGRAM_ROOTVIEWS_H_
