#ifndef PROGRAM_ROOTVIEWS_H_
#define PROGRAM_ROOTVIEWS_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../search/asyncbot.h"
#include "../external/nlohmann_json/json.hpp"

//KataGame fork (abscls-il4e.3, abscls-il4e.4, abscls-il4e.9): the root views, for a game whose network sees a part of a
//larger board (Blueprints: a view of 67 x 67 points of its rules board). The game gives views that hold every legal
//move of the root between them (GameInterface::viewCover). With one view, the search is the one of that view: the usual
//search when the rules' view holds every move. With more, each view gets a search of the root in that view (a pass)
//with a share of the visits. Each pass's best moves are the candidates, and a move that two passes find is one
//candidate (the same child position). The candidates with the best child values from the root mover's side then get a
//search of their child position each (a check), with the rest of the visits, and the move is the check with the best
//value. The passes and the checks take the first pass's score centre, so that their utilities have one scale. Their
//visits are not compared. When no pass finds a candidate (abscls-csmy: a pass of 1 visit searches its root only), the
//move is the best legal move of the first pass's root by the network's policy, with no check. The analysis engine's
//rootViews (command/analysis.cpp), GTP's genmove (command/gtp.cpp) and Blueprints' engine protocol
//(command/blueprintsprotocol.cpp) all search so.
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

  //Starts the history again at the board's position, at the history's turn number, after a move in a view of its own:
  //the search's graph hash replays the history's moves from its start, where such a move would name another action.
  //The game's position keeps what its inputs read of the earlier moves.
  void restartHistory(const Board& board, BoardHistory& hist);

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
  //The analysis engine's response of the search as it is now: Search::getAnalysisJson with the report's arguments.
  bool analysisJson(const Search& search, const Report& report, nlohmann::json& out);

  //The move that the search chose: a move of the root's own view (view ""), or a move that only another view names,
  //with that view (loc is then the move in it). KataGame fork (abscls-il4e.11): pass, the pass that gave the move (its
  //view the cover's view of that index), and passLoc, the move in that pass's view, for self-play, which names each
  //move in the view of its search (Play::runGame).
  struct Choice {
    Loc loc = Board::NULL_LOC;
    std::string view;
    int pass = -1;
    Loc passLoc = Board::NULL_LOC;
  };

  //The searches of the root's views (above), one at a time, for a caller that runs each search: search() below runs
  //them on an AsyncBot, and Blueprints' engine protocol (command/blueprintsprotocol.cpp) runs them in slices. next()
  //gives each search's root and its share of the limits, done() takes the search at its end, and result() gives the
  //response at any time. It holds boards of the game's whole board: keep it on the heap.
  class Searches {
  public:
    Searches(
      const Board& rootBoard, const BoardHistory& rootHist, Player rootPla, const std::vector<std::string>& views,
      const Settings& settings, const Report& report
    );
    //The next search, or false when every search is done: its root is then board(), hist() and pla(), and its share
    //of the limits share() (genMoveSynchronous's searchFactor). The passes come first, then the checks.
    bool next();
    const Board& board() const { return board_; }
    //The root in the view of pass i: the cover's view i (KataGame fork, abscls-il4e.11: self-play's root of a move).
    const Board& viewBoard(int i) const { return boards_[i]; }
    const BoardHistory& hist() const { return hist_; }
    Player pla() const { return pla_; }
    double share() const { return share_; }
    //The search that next() gave, at its end: its analysis, and chosen, the move that it chose (genMoveSynchronous),
    //which a cover of one view gives as the choice. After the first pass of more views, every later search keeps its
    //score centre (Search::fixedScoreCenter): the caller sets it back to NAN after the last. False when the search
    //gave no analysis.
    bool done(Search& search, Loc chosen);
    //The response, the analysis engine's, and the chosen move, from the searches done and from running, the search
    //that next() gave, as it is now (nullptr: none): a reply during the search, or the best check or pass so far when
    //it stops. With no candidate, the first pass's policy move (policyMove below). False before the first pass's root
    //has its evaluation.
    bool result(nlohmann::json& ret, Choice& choice, const Search* running = nullptr) const;

  private:
    struct Candidate {
      int pass;
      Loc loc;
      Loc own;  //the move in the root's own view, or NULL_LOC
      nlohmann::json info;  //the moveInfo of the pass where it has the most visits
      double value;  //its utility from the root mover's side, the mean of its passes' weighted by their visits
      int64_t visits;  //its visits in the passes
      nlohmann::json check;  //the rootInfo of its check, or null
    };
    //What the searches gave.
    struct Results {
      std::vector<Candidate> candidates;
      std::map<Hash128, size_t> byChild;
      //When the first pass gives no candidate: the best legal move of its root by the network's policy, as a
      //candidate of no visits, the move while no search gives a candidate (respond takes it in place of candidates).
      std::vector<Candidate> policyMove;
      nlohmann::json passes = nlohmann::json::array();
      nlohmann::json single;  //the analysis of a cover of one view
      Loc chosen = Board::NULL_LOC;  //and its chosen move
      int numSearches = 0;
      int64_t totalVisits = 0;
    };
    void record(Results& results, const Search& search, nlohmann::json& out, Loc chosen) const;
    bool respond(const Results& results, nlohmann::json& ret, Choice& choice) const;
    Loc ownMove(const Hash128& child) const;
    void nameMove(nlohmann::json& info, Loc own, const std::string& view) const;

    Board rootBoard_;
    BoardHistory rootHist_;
    Player rootPla_;
    std::vector<std::string> views_;
    Settings settings_;
    Report report_;  //its perspective the side of every search's values
    double moverSign_;  //+1 when it is the root mover's, else -1
    std::map<Hash128, Loc> ownMoves_;  //each move of the root's own view by its child position
    std::vector<Board> boards_;  //the root in each view
    std::vector<size_t> order_;  //the candidates by their passes' values, once the passes are done
    size_t numFinalists_ = 0;  //the first ones of order_ get a check
    int nextSearch_ = 0;  //the passes 0 to views_.size() - 1, then views_.size() + each finalist
    int current_ = -1;  //the search that next() gave and done() did not take yet, or -1
    Board board_;
    BoardHistory hist_;
    Player pla_ = C_EMPTY;
    double share_ = 1.0;
    Results results_;
  };

  //The search of the root's views (above), of the position's cover views: writes the analysis engine's response in
  //ret and the chosen move in choice, or returns false when a search was stopped or had no result. Each search takes
  //its share of the search's limits and of the clock tc times searchFactor (genMoveSynchronous's searchFactor), within
  //the clock's limit for the root's move, so the searches together take the limits and the clock's time of one search.
  //The first search calls onSearchBegun, and a later one stops at once when stopped() is true (an empty stopped:
  //never). The bot's position is the last search's at the end.
  bool search(
    AsyncBot* bot, const Board& rootBoard, const BoardHistory& rootHist, Player rootPla,
    const std::vector<std::string>& views, const TimeControls& tc, double searchFactor,
    const Settings& settings, const Report& report,
    const std::function<void()>& onSearchBegun, const std::function<bool()>& stopped,
    nlohmann::json& ret, Choice& choice
  );
}

#endif  // PROGRAM_ROOTVIEWS_H_
