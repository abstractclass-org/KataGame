#!/usr/bin/env python3
"""KataGame: the test of tests/checkgamelayer.py, on temporary trees with uses that the test plants.

    python3 tests/checkgamelayer_test.py        # from katagame/cpp

Each tree holds the folders of three games (Enclosure, Connect-4 and a made-up game with a short name, hex), the
interface's two headers (copied from this tree, which the check reads for the interface's members), and one file that
a test fills. A use of a game's namespace or a header of its folder fails, for each game. A word that holds a short
name (std::hex, hexagon, cargo) passes. The check's own run on this tree must pass too. Python's standard library only.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

CPP = pathlib.Path(__file__).resolve().parent.parent
CHECK = CPP / "tests/checkgamelayer.py"
GAMES = ("enclosure", "connect4", "hex")


class CheckGameLayer(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cpp = pathlib.Path(tmp.name) / "cpp"
        for game in GAMES:
            (self.cpp / "game" / game).mkdir(parents=True)
            (self.cpp / "game" / game / "gamerules.h").write_text("")
        for header in ("gameinterface.h", "botinterface.h"):
            shutil.copy(CPP / "game" / header, self.cpp / "game" / header)
        (self.cpp / "command").mkdir()

    def run_check(self):
        done = subprocess.run([sys.executable, str(CHECK), str(self.cpp)], capture_output=True, text=True)
        return done.returncode, done.stderr

    def plant(self, code, name="command/planted.cpp"):
        (self.cpp / name).write_text(code, encoding="utf-8")
        return self.run_check()

    def test_the_check_passes_on_this_tree(self):
        done = subprocess.run([sys.executable, str(CHECK), str(CPP)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_tree_with_no_planted_use_passes(self):
        self.assertEqual(self.run_check(), (0, ""))

    def test_each_use_of_a_game_fails_for_each_game(self):
        for game in GAMES:
            for code in (f"auto b = {game}::Board();\n", f"namespace {game} {{\n", f"namespace x = {game};\n",
                         f"using namespace {game};\n", f"using namespace {game}::detail;\n", f"::{game}::init();\n",
                         f"if (a && {game}::ok) {{}}\n", f'#include "game/{game}/rules.h"\n',
                         f"#include <{game}/rules.h>\n", f'#  include "{game}/rules.h"\n'):
                with self.subTest(game=game, code=code):
                    status, errors = self.plant(code)
                    self.assertEqual(status, 1, code)
                    self.assertIn("planted.cpp:1:", errors)
                    self.assertIn("1 references to a game", errors)

    def test_the_words_that_hold_a_short_name_pass(self):
        code = ("#include <iomanip>\n#include <hexagon/grid.h>\n#include <cargo/hexdigest.h>\n"
                "std::cout << std::hex << 255 << std::ios::hex;\n"
                "auto a = hexagon::Grid(); auto b = cargo::load(); auto c = ago::x; auto d = good::y;\n"
                "namespace hexagon { namespace x = cargo; }\n"
                "auto e = app::hex::f();\n"
                "// Hex, hex and Enclosure fork, in a comment: \"hex\" and \"enclosure\" in a string\n"
                'const char* GAME = "hex"; int hexBoard = 0; auto h = Hex::Board();\n')
        self.assertEqual(self.plant(code), (0, ""))

    def test_the_failure_names_the_namespace(self):
        status, errors = self.plant("auto b = hex::Board();\n")
        self.assertEqual(status, 1)
        self.assertIn("names the namespace hex", errors)

    def test_a_games_own_file_names_its_own_game_only(self):
        (self.cpp / "game/hex/ownfiles.txt").write_text("command/hexcommand.cpp  # the game's command\n")
        self.assertEqual(self.plant("auto b = hex::Board();\n", "command/hexcommand.cpp"), (0, ""))
        status, errors = self.plant("auto b = enclosure::Board();\n", "command/hexcommand.cpp")
        self.assertEqual(status, 1)
        self.assertIn("names the namespace enclosure", errors)

    def test_a_list_that_names_a_missing_file_fails(self):
        (self.cpp / "game/hex/ownfiles.txt").write_text("command/gone.cpp\n")
        status, errors = self.run_check()
        self.assertEqual(status, 1)
        self.assertIn("game/hex/ownfiles.txt: command/gone.cpp does not exist", errors)


if __name__ == "__main__":
    unittest.main()
