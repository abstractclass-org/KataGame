#!/bin/bash
# Self-play of a pilot (PILOT=1, async_parts.sh), the self-play part of async_start.sh. Until the first export (a
# network in MODELSDIR), self-play plays the random network with GAMES games in flight, the run's CPUs: the random
# network needs no evaluation, so each game is a runnable thread, and more games than CPUs only queue for them. At the
# first network it stops that self-play (SIGTERM: it writes its rows and exits) and starts it again with the run's own
# games in flight. With a network already in MODELSDIR, it starts self-play with the run's settings at once.
#   pilot_selfplay.sh MODELSDIR GAMES OVERRIDES KATAGO selfplay ARGS...
#   (OVERRIDES: self-play's -override-config, empty for none)
# A stop of the part (SIGTERM) stops self-play and does not start it again.
{
models=$1 games=$2 overrides=$3; shift 3
has_network() { [[ -n "$(ls -A "$models" 2>/dev/null)" ]]; }
if ! has_network; then
    "$@" -override-config "${overrides:+$overrides,}numGameThreads=$games" &
    pid=$! stopped=0
    trap 'stopped=1; kill -TERM "$pid" 2>/dev/null' TERM INT
    echo "pilot: $games games in flight until the first network in $models"
    while kill -0 "$pid" 2>/dev/null && ! has_network; do sleep 5 & wait $!; done
    if kill -0 "$pid" 2>/dev/null; then
        echo "pilot: the first network is in $models: self-play starts again with the run's games in flight"
        kill -TERM "$pid"
    fi
    wait "$pid"; rc=$?
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done  # a wait that a signal ended
    trap - TERM INT
    if [[ $stopped == 1 ]] || ! has_network; then exit "$rc"; fi
fi
exec "$@" ${overrides:+-override-config "$overrides"}
}
