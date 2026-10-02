#!/bin/bash
# Stops enclosure_async.sh's parts (their systemd scopes, by name; each process gets SIGTERM: self-play exits
# cleanly, a trainer loses only the samples since its last save). Usage: enclosure_async_stop.sh NAMEPREFIX [PART...]
NAMEPREFIX="$1"; shift
PARTS="${*:-selfplay shuffle export train train-next}"
for p in $PARTS; do
    systemctl --user stop "$NAMEPREFIX-$p.scope" 2>/dev/null && echo "stopped $NAMEPREFIX-$p" || echo "$NAMEPREFIX-$p not running"
done
