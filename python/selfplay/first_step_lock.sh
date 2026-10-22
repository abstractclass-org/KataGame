#!/bin/bash
# Runs a trainer (COMMAND: train.sh and its arguments) so that the trainers of a run make their first training steps
# one at a time. async_start.sh starts each trainer through it, with one lock for the run. It takes the lock LOCK
# (flock) before COMMAND starts, and frees it when COMMAND logs its first training metrics (train.py's
# "nsamp = ...", after its first 100 batches) or when COMMAND ends. Another trainer waits for the lock before it
# starts, so the GPU memory that a trainer takes in its first steps is in use before the next trainer starts.
# Why (the engine repository's games/enclosure/docs/decisions/running-v2a.md, "The transformer's trainer beside
# self-play: GPU memory"): on 2026-10-18 three trainers began to train within 19 seconds of each other, beside
# self-play. The GPU's memory went from 7.9 to 15.4 GB in a minute, and the GPU driver reset 75 s later.
# A trainer that holds the lock and waits for data holds it until it trains: in a new run, until the first shuffle.
# COMMAND's output (stdout and stderr) goes to this script's stdout. NAME names the trainer in its notes, and LOCK
# names the trainer that holds it.
# Usage: first_step_lock.sh LOCK NAME COMMAND...
# The body is one { ... } block: bash reads all of it before running any (async_start.sh's header).
{
LOCK="$1" NAME="$2"
shift 2
exec 9>> "$LOCK"
if ! flock -n 9; then
    echo "=== $(date '+%F %T') $NAME waits for the first steps of $(cat "$LOCK" 2>/dev/null) ==="
    flock 9
fi
echo "$NAME (process $$)" > "$LOCK"
echo "=== $(date '+%F %T') $NAME starts; the run's next trainer starts after its first training metrics ==="
# COMMAND does not get the lock's descriptor: only this script's reader frees the lock (flock -u frees it for every
# descriptor of the open file), and the script's end frees it too.
"$@" 9>&- 2>&1 | {
    while IFS= read -r line; do
        printf '%s\n' "$line"
        if [[ "$line" == *"nsamp = "* ]]; then
            flock -u 9
            echo "=== $(date '+%F %T') $NAME logged its first training metrics: the lock is free ==="
            exec cat
        fi
    done
}
exit "${PIPESTATUS[0]}"
}
