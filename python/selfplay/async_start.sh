#!/bin/bash -eu
# KataGo's asynchronous setup (SelfplayTraining.md: self-play, shuffler and exporter, training, no gatekeeper), all on
# this machine, for a run of any game: GAME names it (KataGame's GAMES.md). It replaced the synchronous loop, whose
# self-play waited through every shuffle, training and export and the end-of-cycle drain.
# Departures from SelfplayTraining.md, both for one machine:
#   - the shuffle runs every SHUFFLE_EVERY seconds (600), not every 20: upstream's 20 s assumes the shuffler has its
#     own machine ("as many threads as you have cores"); here it would take 8 cores from self-play all the time;
#   - each part runs in its own memory cap (systemd scope), so one running out of memory never takes the others.
# The game: the binary KATAGO_BIN must play it (`katago version` names its game), the self-play config is
# cpp/configs/training/selfplay_GAME.cfg (SELFPLAY_CONFIG), the load check reads cpp/configs/analysis_GAME.cfg
# (ANALYSIS_CONFIG), and the trainers train the game's model (train.py -game) on its largest board (-pos-len: the
# trainer's module, python/katago/train/games/GAME.py).
# Usage: GAME=NAME KATAGO_BIN=BINARY TRAINERS="b10c128 b15c192" bash katagame/python/selfplay/async_start.sh \
#            NAMEPREFIX BASEDIR
#   (the trainers and the parts' resources: async_parts.sh; other settings in the environment: SELFPLAY_OVERRIDES,
#   BATCHSIZE, ...). The engine repository's pipeline/scripts/start.sh sets them from a run's settings.
# Stop: bash katagame/python/selfplay/async_stop.sh NAMEPREFIX [PART...]
# The body is one { ... } block (as pipeline/scripts/*.sh): bash reads all of it before running any, so an edit while
# a copy starts cannot corrupt what it runs (2026-10-06 19:50: a restart read the file mid-edit, hit a syntax error
# and left self-play down for 48 minutes).
{
set -eu
if [[ $# -lt 2 ]]; then
    echo "Usage: GAME=NAME KATAGO_BIN=BINARY TRAINERS=\"NAME ...\" $0 NAMEPREFIX BASEDIR"
    exit 0
fi
NAMEPREFIX="$1"
BASEDIR="$(realpath -m "$2")"
. "$(dirname "$0")"/async_parts.sh
TRAINERS="${TRAINERS:?set TRAINERS, the trainers, the one for self-play first (async_parts.sh)}"
GAME="${GAME:?set GAME, the game of the run (a folder of cpp/game/)}"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
KATAGO="$REPO"/katagame
KATAGO_BIN="${KATAGO_BIN:?set KATAGO_BIN, a binary of the game of the run}"
export PATH="$REPO/.venv-katago/bin:$PATH"
[[ -x "$KATAGO_BIN" ]] || { echo "no binary at $KATAGO_BIN (KATAGO_BIN)"; exit 1; }

# The binary must play the run's game. A binary from before KataGame's version named the game cannot be checked here:
# the load check below and self-play's load of each network still refuse a network of another game.
BIN_GAME=$("$KATAGO_BIN" version | sed -n 's/^Game: //p')
if [[ -z "$BIN_GAME" ]]; then
    echo "Note: $KATAGO_BIN does not name its game (built before its version did); the run plays $GAME"
elif [[ "$BIN_GAME" != "$GAME" ]]; then
    echo "$KATAGO_BIN plays $BIN_GAME, and the run plays $GAME: set KATAGO_BIN to a build of $GAME"
    exit 1
fi
SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$KATAGO/cpp/configs/training/selfplay_$GAME.cfg}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-$KATAGO/cpp/configs/analysis_$GAME.cfg}"
[[ -f "$SELFPLAY_CONFIG" ]] || { echo "no self-play config $SELFPLAY_CONFIG for $GAME (SELFPLAY_CONFIG)"; exit 1; }
ONLY="${ONLY:-}"  # e.g. ONLY=shuffle: start only these parts (one that stopped), leaving the others running
# The game's largest board, the trainers' -pos-len (train.sh reads POS_LEN), from the trainer's module of the game.
# Only when a trainer starts: the module imports torch, and a restart of self-play alone (its watchdog) need not wait.
if [[ -z "$ONLY" || " $ONLY " == *" train-"* ]]; then
    POS_LEN=$(cd "$KATAGO"/python && python3 -c "from katago.train import games; print(games.get('$GAME').MAX_BOARD_LEN)") \
        || { echo "no trainer's module for $GAME (python/katago/train/games/$GAME.py)"; exit 1; }
    export POS_LEN
fi

LOGSDIR="$BASEDIR"/logs
SCRATCHDIR="$BASEDIR"/shufflescratch
mkdir -p "$BASEDIR" "$LOGSDIR" "$SCRATCHDIR" "$BASEDIR"/selfplay

# The shuffle's defaults: upstream's synchronous_loop.sh's, scaled for Enclosure's rows in the run v2a.
# 4 shuffle workers, not upstream's 8: a worker holds about 1.7 GB of Enclosure's rows (shuffle.py's estimate, made
# for Go's rows, says 72 MB), and here the shuffle runs beside self-play (2026-10-03: 8 workers ran out of 8 GB).
# An Enclosure row is 186,167 bytes (two 17,329-entry policy targets and three q-value targets), Go's 7,675: so a
# sharding worker takes 3,300 rows (upstream 80,000) and a shuffled file holds 2,900 (upstream 70,000). A game with
# smaller rows can set upstream's values in its run's settings (NUM_THREADS_FOR_SHUFFLING, ROWS_PER_SHUFFLE_FILE,
# SHUFFLE_WORKER_GROUP_ROWS).
NUM_THREADS_FOR_SHUFFLING="${NUM_THREADS_FOR_SHUFFLING:-4}"
NUM_TRAIN_SAMPLES_PER_EPOCH="${NUM_TRAIN_SAMPLES_PER_EPOCH:-100000}"
MAX_TRAIN_PER_DATA="${MAX_TRAIN_PER_DATA:-8}"
NUM_TRAIN_SAMPLES_PER_SWA="${NUM_TRAIN_SAMPLES_PER_SWA:-80000}"
BATCHSIZE="${BATCHSIZE:-128}"
SHUFFLE_MINROWS="${SHUFFLE_MINROWS:-100000}"
MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-500000}"
TAPER_WINDOW_SCALE="${TAPER_WINDOW_SCALE:-50000}"
SHUFFLE_KEEPROWS="${SHUFFLE_KEEPROWS:-600000}"
ROWS_PER_SHUFFLE_FILE="${ROWS_PER_SHUFFLE_FILE:-2900}"
SHUFFLE_WORKER_GROUP_ROWS="${SHUFFLE_WORKER_GROUP_ROWS:-3300}"
# As the synchronous loop of v2a: validation (VALIDATE=1: 1% of the self-play files held out, train.py reports
# metrics_val.json) and extra train.py flags for every trainer (TRAIN_EXTRA_ARGS, e.g. -use-fp16); one trainer's
# own flags are TRAIN_ARGS_NAME (2026-10-06: b10c128's learning-rate drop, -lr-scale, while b15c192 trains at the
# full rate).
VALIDATE="${VALIDATE:-0}"
if [[ "$VALIDATE" == 1 ]]; then SHUFFLE_ENV=""; else SHUFFLE_ENV="SKIP_VALIDATE=1"; fi
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
SHUFFLE_EVERY="${SHUFFLE_EVERY:-600}"
EXPORT_EVERY="${EXPORT_EVERY:-10}"  # upstream's
CPUS="${CPUS:-0-15}"
# Each part's memory cap, CPU weight and nice value: part_limits (async_parts.sh). A lower CPU weight for the
# parts beside self-play lets self-play keep the CPUs while they wait their turn (2026-10-03: the shuffle took its
# full share from self-play whenever it ran).

# As upstream: freeze the scripts, binary and config of this run.
DATED_ARCHIVE="$BASEDIR"/scripts/dated/$(date "+%Y%m%d-%H%M%S")
mkdir -p "$DATED_ARCHIVE"/bin
cp "$KATAGO"/python/*.py "$KATAGO"/python/selfplay/*.py "$KATAGO"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/katago "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/muon "$DATED_ARCHIVE"
cp "$KATAGO_BIN" "$DATED_ARCHIVE"/bin/katago
cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
# The run's overrides of selfplay.cfg (SELFPLAY_OVERRIDES), which the dashboard applies to show the settings in force.
printf '%s\n' "${SELFPLAY_OVERRIDES:-}" > "$DATED_ARCHIVE"/selfplay_overrides.txt
git -C "$REPO" show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt || true
git -C "$REPO" diff --no-color > "$DATED_ARCHIVE"/diff.txt || true
# KataGame is a submodule: the parent's diff shows only its commit, so record its own changes too.
git -C "$KATAGO" diff --no-color > "$DATED_ARCHIVE"/diff-katago.txt || true
cd "$DATED_ARCHIVE"
# As the synchronous loop of v2a: the archived binary must read what the archived export script writes; stop before
# starting anything if it cannot.
CHECKPOINT="$BASEDIR"/train/"$(selfplay_trainer)"/checkpoint.ckpt
if [[ -f "$CHECKPOINT" ]]
then
    LOADCHECK="$BASEDIR"/loadcheck
    rm -rf "$LOADCHECK"; mkdir -p "$LOADCHECK"
    if python3 -W ignore ./export_model_pytorch.py -checkpoint "$CHECKPOINT" -export-dir "$LOADCHECK" -model-name loadcheck            -filename-prefix model -use-swa > "$LOADCHECK"/export.log 2>&1        && gzip "$LOADCHECK"/model.bin        && ./bin/katago analysis -model "$LOADCHECK"/model.bin.gz -config "$ANALYSIS_CONFIG"            < /dev/null > "$LOADCHECK"/load.log 2>&1
    then
        echo "Load check passed: the archived binary reads the archived export script's model"
        rm -rf "$LOADCHECK"
    else
        echo "Load check FAILED: the archived binary cannot read the archived export script's model; stopping"
        tail -n 5 "$LOADCHECK"/export.log "$LOADCHECK"/load.log
        exit 1
    fi
fi

STARTED=()
part() {  # NAME COMMAND...: one part, detached, in its own scope with its limits (part_limits), logging to logs/NAME.log
    local name="$1" mem weight nice; shift
    if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then return; fi
    read -r mem weight nice < <(part_limits "$name")
    [[ -n "$nice" ]] || { echo "no limits for $name; stopping"; exit 1; }
    STARTED+=("$name")
    systemctl --user reset-failed "${NAMEPREFIX}-$name.scope" 2>/dev/null || true
    echo "=== $(date '+%F %T') start $name ===" >> "$LOGSDIR/$name.log"
    setsid nohup systemd-run --user --scope -q --unit="${NAMEPREFIX}-$name" -p MemoryMax="$mem" -p MemorySwapMax=0 -p CPUWeight="$weight" \
        taskset -c "$CPUS" nice -n "$nice" "$@" >> "$LOGSDIR/$name.log" 2>&1 < /dev/null &
    disown
}

part selfplay ./bin/katago selfplay -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models \
    -config "$DATED_ARCHIVE"/selfplay.cfg ${SELFPLAY_OVERRIDES:+-override-config "$SELFPLAY_OVERRIDES"}

part shuffle bash -c "while true; do env $SHUFFLE_ENV ./shuffle.sh '$BASEDIR' '$SCRATCHDIR' $NUM_THREADS_FOR_SHUFFLING \
    -min-rows $SHUFFLE_MINROWS -keep-target-rows $SHUFFLE_KEEPROWS -taper-window-scale $TAPER_WINDOW_SCALE \
    -approx-rows-per-out-file $ROWS_PER_SHUFFLE_FILE -worker-group-size $SHUFFLE_WORKER_GROUP_ROWS \
    | tee -a '$BASEDIR'/logs/outshuffle.txt; sleep $SHUFFLE_EVERY; done"

part export bash -c "while true; do ./export_model_for_selfplay.sh '$NAMEPREFIX' '$BASEDIR' 0 \
    >> '$BASEDIR'/logs/outexport.txt 2>&1; sleep $EXPORT_EVERY; done"

TRAIN_ARGS="-game $GAME -samples-per-epoch $NUM_TRAIN_SAMPLES_PER_EPOCH -swa-period-samples $NUM_TRAIN_SAMPLES_PER_SWA -no-repeat-files \
    -max-train-bucket-per-new-data $MAX_TRAIN_PER_DATA -max-train-bucket-size $MAX_TRAIN_SAMPLES_PER_CYCLE $TRAIN_EXTRA_ARGS"
# The trainers: the first trains the self-play network (main: exports to models/), the others alongside (extra).
mode=main
for t in $TRAINERS; do
    # shellcheck disable=SC2046  # a trainer's own flags are a list
    part "train-$t" ./train.sh "$BASEDIR" "$t" "$(trainer_kind "$t")" "$BATCHSIZE" "$mode" $TRAIN_ARGS $(trainer_args "$t")
    mode=extra
done
sleep 2
echo "started from $DATED_ARCHIVE; logs in $LOGSDIR (${STARTED[*]:-no part})"
exit
}
