#ifndef TESTS_ENCLOSURE_TPSBOARD_H_
#define TESTS_ENCLOSURE_TPSBOARD_H_

//A position of Tak for the tests, read from its one-line text (TPS, then the plies since the last placement or
//flattening) with none of the game's code: the stacks, the roads, the end of the game, the legal moves as policy
//indices and the squares that a move in PTN drops on. testgameends and testlegalrows compare Tak's build with it.

#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Tps {

static const int DX[4] = {0, 0, 1, -1}, DY[4] = {1, -1, 0, 0};  //+, -, > and <, in the moves' order
static const char* const DIRECTIONS = "+-><";

//A spread's drops, from its first step on: every list of drops of a pickup of up to so many pieces.
inline void dropLists(int left, std::vector<int>& drops, std::vector<std::vector<int>>& out) {
  for(int c = 1; c <= left; c++) {
    drops.push_back(c);
    out.push_back(drops);
    dropLists(left - c, drops, out);
    drops.pop_back();
  }
}

//The policy channel of a spread: 3 + 4 * pattern + direction. A pattern of a pickup of k pieces is 2^(k-1) - 1 plus a
//bit 2^(i-1) for each step before the last that ends after the i-th piece.
inline int spreadChannel(const std::vector<int>& drops, int direction) {
  int pieces = 0, code = 0;
  for(size_t s = 0; s < drops.size(); s++) {
    pieces += drops[s];
    if(s + 1 < drops.size())
      code += 1 << (pieces - 1);
  }
  return 3 + 4 * ((1 << (pieces - 1)) - 1 + code) + direction;
}

//The squares that a move drops on, as y * n + x, in order.
inline std::vector<int> dropSquares(const std::string& ptn, int n) {
  const size_t at = ptn.find_first_of(DIRECTIONS);
  const std::string head = ptn.substr(0, at);
  const int x = head[head.size() - 2] - 'a', y = head[head.size() - 1] - '1';
  if(at == std::string::npos)
    return {y * n + x};
  const int d = (int)std::string(DIRECTIONS).find(ptn[at]);
  const size_t steps = at + 1 < ptn.size() ? ptn.size() - at - 1 : 1;
  std::vector<int> squares;
  for(size_t s = 1; s <= steps; s++)
    squares.push_back((y + DY[d] * (int)s) * n + x + DX[d] * (int)s);
  return squares;
}

struct Board {
  struct Stack {
    std::vector<int> pieces;  //players, from the bottom up
    char kind = 'F';          //the top piece's: F, S or C
  };
  int n = 0, turn = 0, ply = 0, clock = 0;
  std::vector<Stack> stacks;  //y * n + x

  explicit Board(const std::string& text) {
    std::istringstream in(text);
    std::string rows, cell;
    int player, move;
    in >> rows >> player >> move >> clock;
    turn = player - 1;
    ply = 2 * (move - 1) + turn;
    for(char c : rows)
      n += c == '/';
    n++;
    stacks.resize(n * n);
    std::istringstream byRow(rows);
    for(int y = n - 1; y >= 0; y--) {
      std::string row;
      std::getline(byRow, row, '/');
      std::istringstream cells(row);
      int x = 0;
      while(std::getline(cells, cell, ',')) {
        if(cell[0] == 'x') {
          x += cell.size() > 1 ? std::atoi(cell.c_str() + 1) : 1;
          continue;
        }
        for(char c : cell) {
          if(c == 'S' || c == 'C')
            stacks[y * n + x].kind = c;
          else
            stacks[y * n + x].pieces.push_back(c - '1');
        }
        x++;
      }
    }
  }

  int owner(int i) const { return stacks[i].pieces.empty() ? -1 : stacks[i].pieces.back(); }
  bool flat(int i, int player) const { return owner(i) == player && stacks[i].kind == 'F'; }
  int flats(int player) const {
    int count = 0;
    for(int i = 0; i < n * n; i++)
      count += flat(i, player);
    return count;
  }
  //The stones and the capstones off the board.
  std::pair<int, int> reserves(int player) const {
    static const int STONES[9] = {0, 0, 0, 10, 15, 21, 30, 40, 50}, CAPS[9] = {0, 0, 0, 0, 0, 1, 1, 2, 2};
    int stones = STONES[n], caps = CAPS[n];
    for(const Stack& s : stacks)
      for(size_t h = 0; h < s.pieces.size(); h++)
        if(s.pieces[h] == player)
          (s.kind == 'C' && h + 1 == s.pieces.size() ? caps : stones)--;
    return {stones, caps};
  }
  //The squares of the player's complete roads: of each group of its flats and capstones on top that joins opposite
  //edges.
  std::set<int> road(int player) const {
    std::set<int> squares, left;
    for(int i = 0; i < n * n; i++)
      if(owner(i) == player && stacks[i].kind != 'S')
        left.insert(i);
    while(!left.empty()) {
      std::vector<int> todo = {*left.begin()};
      left.erase(left.begin());
      std::set<int> group;
      bool west = false, east = false, south = false, north = false;
      while(!todo.empty()) {
        const int i = todo.back(), x = i % n, y = i / n;
        todo.pop_back();
        group.insert(i);
        west = west || x == 0;
        east = east || x == n - 1;
        south = south || y == 0;
        north = north || y == n - 1;
        for(int d = 0; d < 4; d++) {
          const int nx = x + DX[d], ny = y + DY[d];
          if(nx >= 0 && ny >= 0 && nx < n && ny < n && left.erase(ny * n + nx) > 0)
            todo.push_back(ny * n + nx);
        }
      }
      if((west && east) || (south && north))
        squares.insert(group.begin(), group.end());
    }
    return squares;
  }
  enum How { ONGOING, ROAD, FLATS, FIFTY };
  //How the game ended, and the winner of a road (the player who moved last, when both have one).
  How ending(int& winner) const {
    const bool roads[2] = {!road(0).empty(), !road(1).empty()};
    winner = roads[0] && roads[1] ? 1 - turn : roads[0] ? 0 : roads[1] ? 1 : -1;
    if(winner >= 0)
      return ROAD;
    bool full = true;
    for(int i = 0; i < n * n; i++)
      full = full && owner(i) >= 0;
    for(int player = 0; player < 2; player++) {
      const std::pair<int, int> r = reserves(player);
      full = full || r.first + r.second == 0;
    }
    return full ? FLATS : clock >= 50 ? FIFTY : ONGOING;
  }
  //The legal moves as indices of the policy of a network's board of that side.
  std::set<int> legalMoves(int side) const {
    std::set<int> legal;
    int winner;
    if(ending(winner) != ONGOING)
      return legal;
    const int channels = 3 + 4 * ((1 << side) - 1);
    const std::pair<int, int> reserve = reserves(turn);
    for(int i = 0; i < n * n; i++) {
      const int x = i % n, y = i / n, first = (y * side + x) * channels;
      if(owner(i) < 0) {
        if(ply < 2 || reserve.first > 0)
          legal.insert(first);
        if(ply >= 2 && reserve.first > 0)
          legal.insert(first + 1);
        if(ply >= 2 && reserve.second > 0)
          legal.insert(first + 2);
        continue;
      }
      if(ply < 2 || owner(i) != turn)
        continue;
      std::vector<int> drops;
      std::vector<std::vector<int>> lists;
      const int height = (int)stacks[i].pieces.size();
      dropLists(height < n ? height : n, drops, lists);
      for(int d = 0; d < 4; d++)
        for(const std::vector<int>& list : lists) {
          bool ok = true;
          for(size_t s = 0; s < list.size() && ok; s++) {
            const int sx = x + DX[d] * (int)(s + 1), sy = y + DY[d] * (int)(s + 1);
            ok = sx >= 0 && sy >= 0 && sx < n && sy < n;
            if(!ok || owner(sy * n + sx) < 0)
              continue;
            const char kind = stacks[sy * n + sx].kind;
            //A capstone alone, as the last step, flattens a wall. Nothing else covers a wall or a capstone.
            ok = kind == 'F' || (kind == 'S' && stacks[i].kind == 'C' && s + 1 == list.size() && list[s] == 1);
          }
          if(ok)
            legal.insert(first + spreadChannel(list, d));
        }
    }
    return legal;
  }
};

}  // namespace Tps

#endif  // TESTS_ENCLOSURE_TPSBOARD_H_
