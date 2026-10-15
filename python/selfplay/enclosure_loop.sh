#!/bin/bash -eu
# Enclosure: KataGo's synchronous_loop.sh (self-play -> shuffle -> train -> export, no gating) for this
# repository. Retired for v2a on 2026-10-05 in favour of the asynchronous setup (enclosure_async.sh, run by
# pipeline/scripts/start.sh); kept for a run that wants the loop. Only these differ from upstream:
#   - KataGo lives in katago/ of this repository (upstream assumes the repository root);
#   - Python is the project's .venv-katago; the binary is ~/ws/game-ai/enclosure/builds/cuda/katago (override KATAGO_BIN);
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
#   e.g. bash katago/python/selfplay/enclosure_loop.sh v2 ~/ws/game-ai/enclosure/v2runs/v2a b6c96 b6c96
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
KATAGO_BIN="${KATAGO_BIN:-$HOME/ws/game-ai/enclosure/builds/cuda/katago}"
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
# Until the first trained model exists, the window is random play only, which shuffle.py caps at -min-rows; an
# epoch is NUM_TRAIN_SAMPLES_PER_EPOCH (100,000) and must fit in it after the 1% validation split and each
# file's partial batch, which Enclosure's small files (scaled for memory) make about 3%. With 100,000 the first
# training never starts (2026-10-02). So the bootstrap window is larger (owner, 2026-10-02); upstream's again
# from the first model on.
BOOTSTRAP_MINROWS="${BOOTSTRAP_MINROWS:-110000}"
MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-500000}"
TAPER_WINDOW_SCALE="${TAPER_WINDOW_SCALE:-50000}"
SHUFFLE_KEEPROWS="${SHUFFLE_KEEPROWS:-600000}"
# Enclosure's row size (see above).
ROWS_PER_SHUFFLE_FILE="${ROWS_PER_SHUFFLE_FILE:-2900}"
SHUFFLE_WORKER_GROUP_ROWS="${SHUFFLE_WORKER_GROUP_ROWS:-3300}"
# Validation (owner, 2026-10-05): VALIDATE=1 runs shuffle.sh without SKIP_VALIDATE, so 1% of the self-play files
# (by path hash) are held out and train.py reports metrics_val.json after each epoch. Off by default, as upstream's
# synchronous_loop.sh.
VALIDATE="${VALIDATE:-0}"
if [[ "$VALIDATE" == 1 ]]; then SHUFFLE_ENV=""; else SHUFFLE_ENV="SKIP_VALIDATE=1"; fi
SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$KATAGO/cpp/configs/training/selfplay_enclosure.cfg}"
NEXT_TRAININGNAME="${NEXT_TRAININGNAME:-}"  # e.g. b10c128 beside a b6c96 run; empty: none
NEXT_MODELKIND="${NEXT_MODELKIND:-$NEXT_TRAININGNAME}"
# Extra train.py flags for both sizes, e.g. -use-fp16 (upstream SelfplayTraining.md: "may make the training faster").
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

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
# Enclosure (owner, 2026-10-04): the archived binary must read what the archived export script writes. The archive
# takes the working tree as it is, so on 2026-10-04 it held a new export format (version 17) that its binary could
# not read; nothing failed until the first export, 70 minutes later, and then every self-play died at once while
# the loop kept shuffling and training on old data (1.5 h). So export the latest checkpoint with the archived script
# and load it with the archived binary now, and stop before self-play if that fails.
CHECKPOINT="$BASEDIR"/train/"$TRAININGNAME"/checkpoint.ckpt
if [[ -f "$CHECKPOINT" ]]
then
    LOADCHECK="$BASEDIR"/loadcheck
    rm -rf "$LOADCHECK"; mkdir -p "$LOADCHECK"
    if python3 -W ignore ./export_model_pytorch.py -checkpoint "$CHECKPOINT" -export-dir "$LOADCHECK" -model-name loadcheck \
           -filename-prefix model -use-swa > "$LOADCHECK"/export.log 2>&1 \
       && gzip "$LOADCHECK"/model.bin \
       && ./bin/katago analysis -model "$LOADCHECK"/model.bin.gz -config "$KATAGO"/cpp/configs/analysis_enclosure.cfg \
           < /dev/null > "$LOADCHECK"/load.log 2>&1
    then
        echo "Load check passed: the archived binary reads the archived export script's model"
        rm -rf "$LOADCHECK"
    else
        echo "Load check FAILED: the archived binary cannot read the archived export script's model; stopping"
        tail -n 5 "$LOADCHECK"/export.log "$LOADCHECK"/load.log
        exit 1
    fi
fi
set -x
cycle=0
while [[ "$CYCLES" -eq 0 || "$cycle" -lt "$CYCLES" ]]
do
    cycle=$((cycle + 1))
    # Enclosure (owner, 2026-10-03): refresh the start positions (our lost games, games/enclosure/tools/export_start_poses.py) before
    # each cycle's self-play, which reads them at its start (hintPosesDir in SELFPLAY_OVERRIDES).
    if [[ -n "${START_POSES_CMD:-}" ]]; then
        (cd "$REPO" && eval "$START_POSES_CMD") || echo "start positions: export failed, the previous file stays"
    fi
    echo "Selfplay"
    time ./bin/katago selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$DATED_ARCHIVE"/selfplay.cfg ${SELFPLAY_OVERRIDES:+-override-config "$SELFPLAY_OVERRIDES"} | tee -a "$BASEDIR"/selfplay/stdout.txt
    echo "Shuffle"
    MINROWS="$SHUFFLE_MINROWS"
    if [[ -z "$(ls -A "$BASEDIR"/models 2>/dev/null)" ]]; then MINROWS="$BOOTSTRAP_MINROWS"; echo "No model yet: -min-rows $MINROWS"; fi
    (
        time env $SHUFFLE_ENV ./shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" -min-rows "$MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS" -taper-window-scale "$TAPER_WINDOW_SCALE" -approx-rows-per-out-file "$ROWS_PER_SHUFFLE_FILE" -worker-group-size "$SHUFFLE_WORKER_GROUP_ROWS" | tee -a "$BASEDIR"/logs/outshuffle.txt
    )
    echo "Train"
    time ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -no-repeat-files -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE" $TRAIN_EXTRA_ARGS
    if [[ -n "$NEXT_TRAININGNAME" ]]
    then
        echo "Train next size"
        time ./train.sh "$BASEDIR" "$NEXT_TRAININGNAME" "$NEXT_MODELKIND" "$BATCHSIZE" extra -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -no-repeat-files -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE" $TRAIN_EXTRA_ARGS
    fi
    echo "Export"
    (
        time ./export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" 0 | tee -a "$BASEDIR"/logs/outexport.txt
    )
done
exit 0
}
