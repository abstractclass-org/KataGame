#!/bin/bash
# Builds the fork's standalone Enclosure tests (testgamelayer, testsymmetry, testnninputs, testlocalpattern,
# testincome, testnnoutputs, testrgsc, testnotation, testgameinterface; or $TESTS) against a KataGo build's objects, swapping KataGo's main for each test's. In WSL:
#   bash tests/enclosure/build.sh [BUILD_DIR (default ~/ws/abstractclass-org/engines/builds/debug)] [OUT_DIR (default ~/ws/abstractclass-org/engines/v2scratch/enctests)]
# then e.g. ~/ws/abstractclass-org/engines/v2scratch/enctests/testnninputs tests/data/enclosure_inputs.txt (from katagame/cpp).
set -e
B=${1:-$HOME/ws/abstractclass-org/engines/builds/debug}; OUT=${2:-$HOME/ws/abstractclass-org/engines/v2scratch/enctests}
SRC=$(cd "$(dirname "$0")/../.." && pwd); mkdir -p "$OUT"
python3 "$SRC/tests/checkgamelayer.py" "$SRC"  # no code outside the game layer names a game
cd "$B"
flags=$(grep -E "^CXX_FLAGS" CMakeFiles/katago.dir/flags.make | cut -d= -f2-)
defs=$(grep -E "^CXX_DEFINES" CMakeFiles/katago.dir/flags.make | cut -d= -f2-)
incs=$(grep -E "^CXX_INCLUDES" CMakeFiles/katago.dir/flags.make | cut -d= -f2-)
link=$(cat CMakeFiles/katago.dir/link.txt)
for t in ${TESTS:-testgamelayer testsymmetry testnninputs testlocalpattern testincome testnnoutputs testrgsc testnotation testgameinterface}; do
  nice -n 19 g++ $defs $incs $flags -c "$SRC/tests/enclosure/$t.cpp" -o "$OUT/$t.o"
  cmd=$(echo "$link" | sed -E "s# CMakeFiles/katago.dir/main.cpp.o # $OUT/$t.o #; s# -o katago # -o $OUT/$t #; s# CMakeFiles/katago.dir/(command|tests)/[A-Za-z0-9_]+.cpp.o##g")
  nice -n 19 bash -c "$cmd"
  echo "built $t"
done
