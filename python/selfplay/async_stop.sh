#!/bin/bash
# Stops async_start.sh's parts (their systemd scopes, by name; each process gets SIGTERM: self-play exits
# cleanly, a trainer loses only the samples since its last save). Usage: async_stop.sh NAMEPREFIX [PART...]
# With no PART, every running scope of the run (NAMEPREFIX-*.scope): its parts, whatever their names, frozen too.
# systemd refuses to stop a frozen scope (systemctl --user freeze: the engine repository's GPU guard, a benchmark's
# pause). A thaw first would let the part run again on a GPU in a bad state, so a frozen scope gets SIGKILL, which
# reaches frozen processes (cgroup v2). Self-play then loses its rows since its last file and RGSC's buffer since its
# last save. Each part writes a .tmp and renames it, so the kill leaves no file half written.
# One line for each part: "stopped NAME", "killed NAME (frozen)", "NAME not running" (its scope was inactive or
# failed), or "NAME not stopped (...)", and then the exit code is 1.
{
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
# A freeze that comes during the stop makes systemd refuse it: then the next try kills the frozen scope. The kill
# waits up to 30 s for the scope to end, and is not tried again: a second SIGKILL does nothing.
stop_part() {
    local name="$1" u="$1.scope" try err=""
    gone "$u" && { echo "$name not running"; return 0; }
    for try in 1 2 3; do
        if [[ "$(systemctl --user show -p FreezerState --value "$u")" != running ]]; then
            err=$(timeout 30 systemctl --user kill --wait --signal=SIGKILL "$u" 2>&1)
            gone "$u" && { echo "killed $name (frozen)"; return 0; }
            break
        fi
        err=$(systemctl --user stop "$u" 2>&1)
        gone "$u" && { echo "stopped $name"; return 0; }
    done
    echo "$name not stopped ($(systemctl --user show -p ActiveState -p FreezerState "$u" | paste -sd' ')):" \
        "${err:-it stays active}"
    return 1
}
rc=0
for p in $PARTS; do
    stop_part "$NAMEPREFIX-$p" || rc=1
done
exit $rc
}
