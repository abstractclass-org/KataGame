#!/bin/bash -eu
# Enclosure: KataGo's synchronous_loop.sh (self-play -> shuffle -> train -> export, no gating) for this
# repository. Only these differ from upstream:
#   - KataGo lives in katago/ of this repository (upstream assumes the repository root);
#   - Python is the project's .venv-katago; the binary is ~/katago-build-cuda/katago (override KATAGO_BIN);
#   - the self-play config is configs/training/selfplay_enclosure.cfg;
#   - the shuffle's row counts are upstream's scaled to the same bytes per worker: an Enclosure row is
#     186,167 bytes uncompressed (measured: two 17,329-entry policy targets and three q-value targets)
#     against Go's 7,675 (shuffle.py's constants), 24.3 times as large. So each sharding worker takes
#     3,300 rows (upstream 80,000, -worker-group-size) and shuffled files hold 2,900 rows (upstream
#     70,000). With upstream's counts the first real shuffle (2026-10-02, 100,000 rows) ran WSL out of
#     memory; shuffle.py's own memory estimate uses Go's row size and is too low by the same factor.
#   - optionally, the next larger network trains on the same data each cycle (NEXT_TRAININGNAME,
#     NEXT_MODELKIND), in train.sh's 'extra' mode: exported to models_extra/, never used by self-play. As
#     KataGo's run: "concurrently training the next larger size on the same data and switching when its
#     average loss caught up to the smaller size" (Wu 2019, section 2). The switch is made by hand.
# Every other setting is upstream's; each can be overridden from the environment (e.g. for a smoke cycle).
#   bash katago/python/selfplay/enclosure_loop.sh NAMEPREFIX BASEDIR TRAININGNAME MODELKIND [CYCLES]
#   e.g. bash katago/python/selfplay/enclosure_loop.sh v2 ~/v2runs/v2a b6c96 b6c96
set -eu  # upstream's shebang flags (ignored when run as "bash enclosure_loop.sh"); no pipefail, as upstream:
# the shuffle exits 1 until -min-rows rows exist, and upstream relies on its pipe into tee to carry on
{
if [[ $# -lt 4 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND [CYCLES]"
    exit 0
fi
NAMEPREFIX="$1"
BASEDIR="$(realpath -m "$2")"
TRAININGNAME="$3"
MODELKIND="$4"
CYCLES="${5:-0}"  # 0: until stopped

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
KATAGO="$REPO"/katago
KATAGO_BIN="${KATAGO_BIN:-$HOME/katago-build-cuda/katago}"
export PATH="$REPO/.venv-katago/bin:$PATH"  # the scripts call python3

LOGSDIR="$BASEDIR"/logs
SCRATCHDIR="$BASEDIR"/shufflescratch
mkdir -p "$BASEDIR" "$LOGSDIR" "$SCRATCHDIR" "$BASEDIR"/selfplay

# Upstream's values (synchronous_loop.sh), overridable.
NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-500}"
NUM_THREADS_FOR_SHUFFLING="${NUM_THREADS_FOR_SHUFFLING:-8}"
NUM_TRAIN_SAMPLES_PER_EPOCH="${NUM_TRAIN_SAMPLES_PER_EPOCH:-100000}"
MAX_TRAIN_PER_DATA="${MAX_TRAIN_PER_DATA:-8}"
NUM_TRAIN_SAMPLES_PER_SWA="${NUM_TRAIN_SAMPLES_PER_SWA:-80000}"
BATCHSIZE="${BATCHSIZE:-128}"
SHUFFLE_MINROWS="${SHUFFLE_MINROWS:-100000}"
MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-500000}"
TAPER_WINDOW_SCALE="${TAPER_WINDOW_SCALE:-50000}"
SHUFFLE_KEEPROWS="${SHUFFLE_KEEPROWS:-600000}"
# Enclosure's row size (see above).
ROWS_PER_SHUFFLE_FILE="${ROWS_PER_SHUFFLE_FILE:-2900}"
SHUFFLE_WORKER_GROUP_ROWS="${SHUFFLE_WORKER_GROUP_ROWS:-3300}"
SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$KATAGO/cpp/configs/training/selfplay_enclosure.cfg}"
NEXT_TRAININGNAME="${NEXT_TRAININGNAME:-}"  # e.g. b10c128 beside a b6c96 run; empty: none
NEXT_MODELKIND="${NEXT_MODELKIND:-$NEXT_TRAININGNAME}"

# As upstream: freeze the scripts, binary and config of this run.
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"/bin
cp "$KATAGO"/python/*.py "$KATAGO"/python/selfplay/*.py "$KATAGO"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/katago "$DATED_ARCHIVE"
cp -r "$KATAGO"/python/muon "$DATED_ARCHIVE"
cp "$KATAGO_BIN" "$DATED_ARCHIVE"/bin/katago
cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
# A record only (as upstream): WSL's git has no Git LFS, so a diff touching the LFS model may fail.
git -C "$REPO" show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt || true
git -C "$REPO" diff --no-color > "$DATED_ARCHIVE"/diff.txt || true
git -C "$REPO" diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt || true

cd "$DATED_ARCHIVE"
set -x
cycle=0
while [[ "$CYCLES" -eq 0 || "$cycle" -lt "$CYCLES" ]]
do
    cycle=$((cycle + 1))
    echo "Selfplay"
    time ./bin/katago selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$DATED_ARCHIVE"/selfplay.cfg ${SELFPLAY_OVERRIDES:+-override-config "$SELFPLAY_OVERRIDES"} | tee -a "$BASEDIR"/selfplay/stdout.txt
    echo "Shuffle"
    (
        time SKIP_VALIDATE=1 ./shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" -min-rows "$SHUFFLE_MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS" -taper-window-scale "$TAPER_WINDOW_SCALE" -approx-rows-per-out-file "$ROWS_PER_SHUFFLE_FILE" -worker-group-size "$SHUFFLE_WORKER_GROUP_ROWS" | tee -a "$BASEDIR"/logs/outshuffle.txt
    )
    echo "Train"
    time ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -no-repeat-files -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE"
    if [[ -n "$NEXT_TRAININGNAME" ]]
    then
        echo "Train next size"
        time ./train.sh "$BASEDIR" "$NEXT_TRAININGNAME" "$NEXT_MODELKIND" "$BATCHSIZE" extra -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -no-repeat-files -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE"
    fi
    echo "Export"
    (
        time ./export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" 0 | tee -a "$BASEDIR"/logs/outexport.txt
    )
done
exit 0
}
