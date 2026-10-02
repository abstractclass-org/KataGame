#!/bin/bash -eu
# Enclosure: KataGo's asynchronous setup (SelfplayTraining.md: self-play, shuffler and exporter, training, no
# gatekeeper), all on this machine, with enclosure_loop.sh's Enclosure settings (row sizes, threads, the next size
# alongside). Owner, 2026-10-02: replaces the synchronous loop, whose self-play waited through every shuffle,
# training and export and the end-of-cycle drain.
# Departures from SelfplayTraining.md, both for one machine:
#   - the shuffle runs every SHUFFLE_EVERY seconds (600), not every 20: upstream's 20 s assumes the shuffler has its
#     own machine ("as many threads as you have cores"); here it would take 8 cores from self-play all the time;
#   - each part runs in its own memory cap (systemd scope), so one running out of memory never takes the others.
# Usage: bash katago/python/selfplay/enclosure_async.sh NAMEPREFIX BASEDIR TRAININGNAME MODELKIND
#   (environment as enclosure_loop.sh: SELFPLAY_OVERRIDES, NEXT_TRAININGNAME, NEXT_MODELKIND, ...)
# Stop: bash katago/python/selfplay/enclosure_async_stop.sh NAMEPREFIX [PART...]
set -eu
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND"
    exit 0
fi
NAMEPREFIX="$1"
BASEDIR="$(realpath -m "$2")"
TRAININGNAME="$3"
MODELKIND="$4"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
KATAGO="$REPO"/katago
KATAGO_BIN="${KATAGO_BIN:-$HOME/katago-build-cuda/katago}"
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
SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$KATAGO/cpp/configs/training/selfplay_enclosure.cfg}"
NEXT_TRAININGNAME="${NEXT_TRAININGNAME:-}"
NEXT_MODELKIND="${NEXT_MODELKIND:-$NEXT_TRAININGNAME}"
SHUFFLE_EVERY="${SHUFFLE_EVERY:-600}"
EXPORT_EVERY="${EXPORT_EVERY:-10}"  # upstream's
CPUS="${CPUS:-0-15}"
# Memory caps (GB) per part; self-play at 512 games peaked at 20.9 GB.
MEM_SELFPLAY="${MEM_SELFPLAY:-24G}"
MEM_SHUFFLE="${MEM_SHUFFLE:-10G}"
MEM_TRAIN="${MEM_TRAIN:-6G}"

# As upstream: freeze the scripts, binary and config of this run.
DATED_ARCHIVE="$BASEDIR"/scripts/dated/$(date "+%Y%m%d-%H%M%S")
mkdir -p "$DATED_ARCHIVE"/bin
cp "$KATAGO"/python/*.py "$KATAGO"/python/selfplay/*.py "$KATAGO"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/katago "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/muon "$DATED_ARCHIVE"
cp "$KATAGO_BIN" "$DATED_ARCHIVE"/bin/katago
cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
git -C "$REPO" show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt || true
cd "$DATED_ARCHIVE"

ONLY="${ONLY:-}"  # e.g. ONLY=shuffle: start only these parts (one that stopped), leaving the others running
part() {  # NAME MEMORY NICE COMMAND...: one part, detached, in its own memory cap, logging to logs/NAME.log
    local name="$1" mem="$2" nice="$3"; shift 3
    if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then return; fi
    systemctl --user reset-failed "${NAMEPREFIX}-$name.scope" 2>/dev/null || true
    echo "=== $(date '+%F %T') start $name ===" >> "$LOGSDIR/$name.log"
    setsid nohup systemd-run --user --scope -q --unit="${NAMEPREFIX}-$name" -p MemoryMax="$mem" -p MemorySwapMax=0 \
        taskset -c "$CPUS" nice -n "$nice" "$@" >> "$LOGSDIR/$name.log" 2>&1 < /dev/null &
    disown
}

part selfplay "$MEM_SELFPLAY" 10 ./bin/katago selfplay -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models \
    -config "$DATED_ARCHIVE"/selfplay.cfg ${SELFPLAY_OVERRIDES:+-override-config "$SELFPLAY_OVERRIDES"}

part shuffle "$MEM_SHUFFLE" 15 bash -c "while true; do SKIP_VALIDATE=1 ./shuffle.sh '$BASEDIR' '$SCRATCHDIR' $NUM_THREADS_FOR_SHUFFLING \
    -min-rows $SHUFFLE_MINROWS -keep-target-rows $SHUFFLE_KEEPROWS -taper-window-scale $TAPER_WINDOW_SCALE \
    -approx-rows-per-out-file $ROWS_PER_SHUFFLE_FILE -worker-group-size $SHUFFLE_WORKER_GROUP_ROWS \
    | tee -a '$BASEDIR'/logs/outshuffle.txt; sleep $SHUFFLE_EVERY; done"

part export 4G 10 bash -c "while true; do ./export_model_for_selfplay.sh '$NAMEPREFIX' '$BASEDIR' 0 \
    >> '$BASEDIR'/logs/outexport.txt 2>&1; sleep $EXPORT_EVERY; done"

TRAIN_ARGS="-samples-per-epoch $NUM_TRAIN_SAMPLES_PER_EPOCH -swa-period-samples $NUM_TRAIN_SAMPLES_PER_SWA -no-repeat-files \
    -max-train-bucket-per-new-data $MAX_TRAIN_PER_DATA -max-train-bucket-size $MAX_TRAIN_SAMPLES_PER_CYCLE"
part train "$MEM_TRAIN" 10 ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main $TRAIN_ARGS
if [[ -n "$NEXT_TRAININGNAME" ]]; then
    part train-next "$MEM_TRAIN" 10 ./train.sh "$BASEDIR" "$NEXT_TRAININGNAME" "$NEXT_MODELKIND" "$BATCHSIZE" extra $TRAIN_ARGS
fi
sleep 2
echo "started from $DATED_ARCHIVE; logs in $LOGSDIR (selfplay, shuffle, export, train, train-next)"
