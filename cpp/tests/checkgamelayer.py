#!/usr/bin/env python3
"""KataGame: the check that no code outside the game layer names a game (GAMES.md, "The game layer").

    python3 tests/checkgamelayer.py [CPP_DIR]

CPP_DIR is katagame/cpp (default: the parent of this file's folder). Each build of katago runs the check
(CMakeLists.txt, the target checkgamelayer), and so does tests/enclosure/build.sh. It needs Python's standard library
only. When it fails, it lists each reference and exits with 1.

The game layer is each game's folder, game/<game>/ (a folder of game/ with gamerules.h), and the headers that choose the
game (game/chosen*.h). Every other C++ file of CPP_DIR, except those of external/, reaches a game only through the
interface (GameRules, GameEncoding, GameBot). The check fails when such a file
- names a game's namespace: <game>::, or namespace <game> (using namespace <game> and an alias of it too);
- includes a header of a game's folder (an #include path with <game>/ in it), or a header of a game's own code outside
  the game layer (game/<game>/ownfiles.txt), which only that game's binary compiles;
- names the chosen game's namespace (chosengame::) beyond the interface's aliases;
- names a member of GameRules, GameEncoding or GameBot that the interface does not have (GameEncoding::sym_edge is
  Enclosure's own): the interface's members are the names that its checks detect (G::<member> in
  game/gameinterface.h and game/botinterface.h);
- names an optional member directly, beside its helper (a game may lack it): the optional members are those that a
  helper of game/gameinterface.h or game/botinterface.h calls after its if constexpr on the member's detector;
- reads a field of the game's position (Board::pos): board.pos.edges names Enclosure's position.
A game's own code outside the game layer (game/<game>/ownfiles.txt) may name its own game, and no other.
"""

import os
import re
import sys
from pathlib import Path

EXTENSIONS = {".cpp", ".cc", ".c", ".h", ".hpp", ".inc", ".cu"}
SKIPPED_FOLDERS = {"external", "CMakeFiles"}


class Rule:
    """A pattern that names a game, what a match finds (str.format of the match's groups), and its keywords: a line
    that holds none of them cannot match, so the pattern does not run on it."""

    def __init__(self, keywords, pattern, finding):
        self.keywords, self.regex, self.finding = keywords, re.compile(pattern), finding

    def check(self, line):
        if any(w in line for w in self.keywords):
            m = self.regex.search(line)
            if m:
                return self.finding.format(*m.groups())
        return None


def game_rules(game, own):
    """The rules that find one game's names. own: the game's own files outside the game layer."""
    g = re.escape(game)
    rules = [
        Rule([f"{game}::"], rf"\b{g}::", f"names the namespace {game}"),
        Rule(["namespace"], rf"\bnamespace\s+(?:\w+\s*=\s*)?{g}\b", f"names the namespace {game}"),
        Rule(["include"], rf"#\s*include\s*[\"<](?:[^\">]*/)?{g}/", f"includes a header of game/{game}/"),
    ]
    headers = sorted({Path(e).name for e in own if os.path.splitext(e)[1] in {".h", ".hpp", ".inc"}})
    if headers:
        rules.append(Rule(["include"], r"#\s*include\s*[\"<](?:[^\">]*/)?(%s)[\">]" % "|".join(map(re.escape, headers)),
                          f"includes {{0}}, which game/{game}/ownfiles.txt lists: only {game}'s binary has its code"))
    return rules


def chosen_game_rules(cpp):
    """The rules that find the names of whichever game the build chooses."""
    text = "".join((cpp / "game" / h).read_text(encoding="utf-8") for h in ("gameinterface.h", "botinterface.h"))
    members = set(re.findall(r"\bG::(?:template\s+)?(\w+)", text))
    optional = set(re.findall(
        r"if constexpr\(\w+::Has\w+<\w+>::value\)\s*\{?\s*(?:\(void\)\w+;\s*)*(?:return\s+)?\w+::(\w+)", text))
    part = r"\b(GameRules|GameEncoding|GameBot)::"
    keywords = ["GameRules::", "GameEncoding::", "GameBot::"]
    return [
        Rule(["chosengame::"], r"\bchosengame::",
             "names the chosen game's namespace: use GameRules, GameEncoding or GameBot"),
        Rule(keywords, part + r"(?!(?:%s)\b)(\w+)" % "|".join(sorted(members)),
             "names {0}::{1}, which is not a member of the game interface"),
        Rule(keywords, part + r"(%s)\b" % "|".join(sorted(optional)),
             "names the optional member {0}::{1}: call its helper in game/gameinterface.h or game/botinterface.h"),
        Rule(["pos."], r"(?:\.|->)pos\.[A-Za-z_]",
             "reads a field of the game's position: pass it to the interface's members"),
    ]


def own_files(cpp, game):
    """The game's own files and folders outside the game layer (a folder ends with /), from game/<game>/ownfiles.txt."""
    path = cpp / "game" / game / "ownfiles.txt"
    if not path.exists():
        return []
    entries = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [e for e in entries if e]


def owner_of(path, owners):
    for game, entries in owners.items():
        for e in entries:
            if path == e or (e.endswith("/") and path.startswith(e)):
                return game
    return None


def main():
    cpp = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent).resolve()
    games = sorted(p.parent.name for p in (cpp / "game").glob("*/gamerules.h"))
    if not games:
        print(f"checkgamelayer: no game in {cpp / 'game'}", file=sys.stderr)
        return 1
    failures = []
    owners = {game: own_files(cpp, game) for game in games}
    for game, entries in owners.items():
        failures += [f"game/{game}/ownfiles.txt: {e} does not exist" for e in entries if not (cpp / e).exists()]
    layer = re.compile(r"game/(%s)/|game/chosen[a-z]*\.h$" % "|".join(map(re.escape, games)))
    # The rules for a file of each owner (None: a file of no game's own).
    rules = {game: game_rules(game, owners[game]) for game in games}
    rules_of = {owner: [r for game in games if game != owner for r in rules[game]] for owner in games}
    rules_of[None] = [r for game in games for r in rules[game]] + chosen_game_rules(cpp)

    checked = 0
    for root, folders, files in os.walk(cpp):
        folders[:] = sorted(f for f in folders if f not in SKIPPED_FOLDERS)
        for name in sorted(files):
            if os.path.splitext(name)[1] not in EXTENSIONS:
                continue
            path = Path(root, name).relative_to(cpp).as_posix()
            if layer.match(path):
                continue
            rules = rules_of[owner_of(path, owners)]
            checked += 1
            text = (cpp / path).read_bytes().decode("utf-8", errors="replace")
            if not any(w in text for rule in rules for w in rule.keywords):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                finding = next(filter(None, (rule.check(line) for rule in rules)), None)
                if finding:
                    failures.append(f"{path}:{n}: {finding}: {line.strip()}")

    if failures:
        print(f"checkgamelayer: {len(failures)} references to a game outside the game layer (game/<game>/, "
              "game/chosen*.h). Reach the game through GameRules, GameEncoding or GameBot, or list a file of the "
              "game's own in game/<game>/ownfiles.txt:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1
    print(f"checkgamelayer: no code outside the game layer names a game ({checked} files; games: {', '.join(games)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
