#!/bin/bash
# Stops enclosure_async.sh's parts (their systemd scopes, by name; each process gets SIGTERM: self-play exits
# cleanly, a trainer loses only the samples since its last save). Usage: enclosure_async_stop.sh NAMEPREFIX [PART...]
# With no PART, every running scope of the run (NAMEPREFIX-*.scope): its parts, whatever their names.
NAMEPREFIX="$1"; shift
if [[ $# -gt 0 ]]; then
    PARTS="$*"
else
    PARTS=$(systemctl --user list-units "$NAMEPREFIX-*.scope" --state=active --no-legend | awk '{print $1}' \
        | sed -e "s/^$NAMEPREFIX-//" -e 's/\.scope$//')
fi
for p in $PARTS; do
    systemctl --user stop "$NAMEPREFIX-$p.scope" 2>/dev/null && echo "stopped $NAMEPREFIX-$p" || echo "$NAMEPREFIX-$p not running"
done
