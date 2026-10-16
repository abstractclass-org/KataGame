// Enclosure fork: the set bits of a std::bitset in order. libstdc++ has the fast _Find_first/_Find_next (GNU
// extensions); libc++ (the WebAssembly build, games/enclosure/wasm) has neither, so it scans bit by bit. Walking a whole set in
// order is linear in its size either way.
#pragma once

#include <bitset>
#include <cstddef>

template <std::size_t N>
inline std::size_t bitsFirst(const std::bitset<N>& b) {
#ifdef __GLIBCXX__
  return b._Find_first();
#else
  std::size_t i = 0;
  while(i < N && !b[i])
    i++;
  return i;
#endif
}

template <std::size_t N>
inline std::size_t bitsNext(const std::bitset<N>& b, std::size_t prev) {
#ifdef __GLIBCXX__
  return b._Find_next(prev);
#else
  std::size_t i = prev + 1;
  while(i < N && !b[i])
    i++;
  return i;
#endif
}
