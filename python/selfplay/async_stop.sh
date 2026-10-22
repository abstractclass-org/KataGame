#!/bin/bash
# Stops async_start.sh's parts (their systemd scopes, by name; each process gets SIGTERM: self-play exits
# cleanly, a trainer loses only the samples since its last save). Usage: async_stop.sh NAMEPREFIX [PART...]
# With no PART, every running scope of the run (NAMEPREFIX-*.scope): its parts, whatever their names, frozen too.
# A frozen scope (systemctl --user freeze: the engine repository's GPU guard, a benchmark's pause) refuses a stop:
# systemd's unit_stop() refuses every FreezerState but running ("Cannot stop frozen unit", systemd 259). A thaw
# first would let the part run again, and the GPU guard freezes a part because the GPU is in a bad state. So a
# frozen scope gets SIGKILL, which reaches frozen processes (cgroup v2), and the part never runs again. Self-play
# then loses what its clean exit saves: its rows since its last file, and RGSC's buffer since its last save (every
# 50 games). No file is left half written: each part writes a .tmp and renames it.
# One line for each part: "stopped NAME", "killed NAME (frozen)", "NAME not running" (its scope was inactive or
# failed), or "NAME not stopped (...)", and then the exit code is 1.
NAMEPREFIX="$1"; shift
if [[ $# -gt 0 ]]; then
    PARTS="$*"
else
    PARTS=$(systemctl --user list-units "$NAMEPREFIX-*.scope" --state=active --no-legend | awk '{print $1}' \
        | sed -e "s/^$NAMEPREFIX-//" -e 's/\.scope$//')
fi
# A scope is gone when it is inactive or failed (a scope that is not loaded is inactive). One that stops is
# "deactivating" until its processes end.
gone() { [[ "$(systemctl --user show -p ActiveState --value "$1")" =~ ^(inactive|failed)$ ]]; }
# A freeze that comes during the stop makes systemd refuse it: then the next try kills the frozen scope.
stop_part() {
    local name="$1" u="$1.scope" try i err=""
    gone "$u" && { echo "$name not running"; return 0; }
    for try in 1 2 3; do
        if [[ "$(systemctl --user show -p FreezerState --value "$u")" != running ]]; then
            err=$(systemctl --user kill --signal=SIGKILL "$u" 2>&1)
            for i in $(seq 50); do
                gone "$u" && { echo "killed $name (frozen)"; return 0; }
                sleep 0.2
            done
        else
            err=$(systemctl --user stop "$u" 2>&1)
            gone "$u" && { echo "stopped $name"; return 0; }
        fi
    done
    echo "$name not stopped ($(systemctl --user show -p ActiveState --value "$u")," \
        "$(systemctl --user show -p FreezerState --value "$u")): ${err:-it stays active}"
    return 1
}
rc=0
for p in $PARTS; do
    stop_part "$NAMEPREFIX-$p" || rc=1
done
exit $rc
