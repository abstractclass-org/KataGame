#!/bin/bash -eu
# Enclosure: KataGo's asynchronous setup (SelfplayTraining.md: self-play, shuffler and exporter, training, no
# gatekeeper), all on this machine, with enclosure_loop.sh's Enclosure settings (row sizes, threads, the next size
# alongside). Owner, 2026-10-02: replaces the synchronous loop, whose self-play waited through every shuffle,
# training and export and the end-of-cycle drain.
# Departures from SelfplayTraining.md, both for one machine:
#   - the shuffle runs every SHUFFLE_EVERY seconds (600), not every 20: upstream's 20 s assumes the shuffler has its
#     own machine ("as many threads as you have cores"); here it would take 8 cores from self-play all the time;
#   - each part runs in its own memory cap (systemd scope), so one running out of memory never takes the others.
# Usage: TRAINERS="b10c128 b15c192" bash katago/python/selfplay/enclosure_async.sh NAMEPREFIX BASEDIR
#   (the trainers and the parts' resources: enclosure_parts.sh; other settings in the environment, as
#   enclosure_loop.sh: SELFPLAY_OVERRIDES, BATCHSIZE, ...)
# Stop: bash katago/python/selfplay/enclosure_async_stop.sh NAMEPREFIX [PART...]
# The body is one { ... } block (as v2/scripts/*.sh): bash reads all of it before running any, so an edit while
# a copy starts cannot corrupt what it runs (2026-10-06 19:50: a restart read the file mid-edit, hit a syntax error
# and left self-play down for 48 minutes).
{
set -eu
if [[ $# -lt 2 ]]; then
    echo "Usage: TRAINERS=\"NAME ...\" $0 NAMEPREFIX BASEDIR"
    exit 0
fi
NAMEPREFIX="$1"
BASEDIR="$(realpath -m "$2")"
. "$(dirname "$0")"/enclosure_parts.sh
TRAINERS="${TRAINERS:?set TRAINERS, the trainers, the one for self-play first (enclosure_parts.sh)}"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
KATAGO="$REPO"/katago
KATAGO_BIN="${KATAGO_BIN:-$HOME/ws/game-ai/enclosure/builds/cuda/katago}"
export PATH="$REPO/.venv-katago/bin:$PATH"

LOGSDIR="$BASEDIR"/logs
SCRATCHDIR="$BASEDIR"/shufflescratch
mkdir -p "$BASEDIR" "$LOGSDIR" "$SCRATCHDIR" "$BASEDIR"/selfplay

# enclosure_loop.sh's values (upstream's, Enclosure's row size).
# 4 shuffle workers, not the loop's 8: a worker holds about 1.7 GB of Enclosure rows (shuffle.py's estimate, made
# for Go's rows, says 72 MB), and here the shuffle runs beside self-play (2026-10-03: 8 workers ran out of 8 GB).
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
# As enclosure_loop.sh (2026-10-05): validation (VALIDATE=1: 1% of the self-play files held out, train.py reports
# metrics_val.json) and extra train.py flags for every trainer (TRAIN_EXTRA_ARGS, e.g. -use-fp16); one trainer's
# own flags are TRAIN_ARGS_NAME (2026-10-06: b10c128's learning-rate drop, -lr-scale, while b15c192 trains at the
# full rate).
VALIDATE="${VALIDATE:-0}"
if [[ "$VALIDATE" == 1 ]]; then SHUFFLE_ENV=""; else SHUFFLE_ENV="SKIP_VALIDATE=1"; fi
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$KATAGO/cpp/configs/training/selfplay_enclosure.cfg}"
SHUFFLE_EVERY="${SHUFFLE_EVERY:-600}"
EXPORT_EVERY="${EXPORT_EVERY:-10}"  # upstream's
CPUS="${CPUS:-0-15}"
# Each part's memory cap, CPU weight and nice value: part_limits (enclosure_parts.sh). A lower CPU weight for the
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
cd "$DATED_ARCHIVE"
# As enclosure_loop.sh (owner, 2026-10-04): the archived binary must read what the archived export script writes;
# stop before starting anything if it cannot.
CHECKPOINT="$BASEDIR"/train/"$(selfplay_trainer)"/checkpoint.ckpt
if [[ -f "$CHECKPOINT" ]]
then
    LOADCHECK="$BASEDIR"/loadcheck
    rm -rf "$LOADCHECK"; mkdir -p "$LOADCHECK"
    if python3 -W ignore ./export_model_pytorch.py -checkpoint "$CHECKPOINT" -export-dir "$LOADCHECK" -model-name loadcheck            -filename-prefix model -use-swa > "$LOADCHECK"/export.log 2>&1        && gzip "$LOADCHECK"/model.bin        && ./bin/katago analysis -model "$LOADCHECK"/model.bin.gz -config "$KATAGO"/cpp/configs/analysis_enclosure.cfg            < /dev/null > "$LOADCHECK"/load.log 2>&1
    then
        echo "Load check passed: the archived binary reads the archived export script's model"
        rm -rf "$LOADCHECK"
    else
        echo "Load check FAILED: the archived binary cannot read the archived export script's model; stopping"
        tail -n 5 "$LOADCHECK"/export.log "$LOADCHECK"/load.log
        exit 1
    fi
fi

ONLY="${ONLY:-}"  # e.g. ONLY=shuffle: start only these parts (one that stopped), leaving the others running
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

TRAIN_ARGS="-samples-per-epoch $NUM_TRAIN_SAMPLES_PER_EPOCH -swa-period-samples $NUM_TRAIN_SAMPLES_PER_SWA -no-repeat-files \
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
