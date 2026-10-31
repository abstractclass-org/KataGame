# A run's trainers, the resources of its parts and the defaults of its levers, for async_start.sh and the engine
# repository's pipeline/scripts/lib.sh (source this file). Every game's runs use it.
#
# Trainers: TRAINERS="b10c128 b15c192", each a folder train/NAME. The first trains the self-play network (train.sh's
# main mode: it exports to models/); the others train alongside on the same data (extra mode: models_extra/, never
# used by self-play). A size switch is a reorder. Each trainer's architecture is MODELKIND_NAME (default NAME), its
# batch BATCHSIZE_NAME (default BATCHSIZE), its own train.py flags TRAIN_ARGS_NAME (beside TRAIN_EXTRA_ARGS, which every
# trainer gets), and its part train-NAME.
#
# Resources: each part runs in its own systemd scope with a memory cap (mem=), a CPU weight (cpu=, systemd's CPUWeight:
# the kernel shares the CPUs between scopes by weight, not by nice) and a nice value (nice=). LIMITS_GROUP sets them
# for a group of parts (selfplay, shuffle, export, train: every trainer) and LIMITS_PART for one part (LIMITS_train_b10c128
# for train-b10c128), each over the defaults below, e.g. LIMITS_train="mem=16G cpu=20".

# Defaults, measured in the run v2a (Enclosure's rows, 186 KB each): self-play at 512 games peaked at 20.9 GB; a
# trainer at 6 GB was killed (2026-10-03, at 5.9 GB). A game with smaller rows can set lower caps in its run's settings.
PART_DEFAULTS_selfplay="mem=24G cpu=100 nice=10"
PART_DEFAULTS_shuffle="mem=10G cpu=100 nice=15"
PART_DEFAULTS_export="mem=4G cpu=100 nice=10"
PART_DEFAULTS_train="mem=8G cpu=100 nice=10"

trainer_kind() {  # NAME: its architecture
    local v="MODELKIND_$1"
    echo "${!v:-$1}"
}

trainer_batch() {  # NAME: its batch (after run_defaults)
    local v="BATCHSIZE_$1"
    echo "${!v:-$BATCHSIZE}"
}

trainer_args() {  # NAME: its own train.py flags
    local v="TRAIN_ARGS_$1"
    echo "${!v:-}"
}

selfplay_trainer() {  # the trainer whose network self-play plays
    local t
    for t in ${TRAINERS:-}; do echo "$t"; return; done
}

part_limits() {  # PART: "MEM CPUWEIGHT NICE" from the defaults, LIMITS_GROUP and LIMITS_PART
    local part="$1" group="${1%%-*}" mem="" cpu="" nice="" spec w
    local d="PART_DEFAULTS_$group" g="LIMITS_$group" p="LIMITS_${1//-/_}"
    for spec in "${!d:-}" "${!g:-}" "${!p:-}"; do
        for w in $spec; do
            case "$w" in
                mem=*) mem="${w#mem=}" ;;
                cpu=*) cpu="${w#cpu=}" ;;
                nice=*) nice="${w#nice=}" ;;
                *) echo "part_limits: '$w' in the limits of $part is not mem=, cpu= or nice=" >&2; return 1 ;;
            esac
        done
    done
    if [[ -z "$mem" || -z "$cpu" || -z "$nice" ]]; then
        echo "part_limits: no defaults for the part $part (group $group)" >&2; return 1
    fi
    echo "$mem $cpu $nice"
}

# --- The levers' defaults ---------------------------------------------------------------------------------------------
# run_defaults sets the default of each lever that async_start.sh reads, where the run's settings and the environment
# leave it unset, so a value that they set wins. game_defaults sets the defaults that need the game's trainer's module
# (python/katago/train/games/GAME.py), which imports torch. async_start.sh calls both, and so does the dry run of the
# engine repository's pipeline/scripts/start.sh, which prints RUN_LEVERS in force. The values are upstream's
# (synchronous_loop.sh, SelfplayTraining.md), but for these, which fit a run of any game:
# - The shuffle's row counts hold upstream's bytes for the game's rows. A sharding worker takes
#   SHUFFLE_WORKER_GROUP_ROWS rows (upstream's 80,000) and a shuffled file holds ROWS_PER_SHUFFLE_FILE (70,000): each is
#   upstream's count times Go's row bytes (7,675, KataGo's shuffle.py) over the game's, rounded to 100, and at most
#   upstream's count. The game's row bytes come from its encoding (katago/train/games: row_bytes). Enclosure's rows are
#   187,258 bytes, 24.4 times Go's, so it gets 2,900 and 3,300 (2026-10-02: with upstream's counts, the first shuffle
#   of 100,000 rows ran WSL out of memory; shuffle.py's estimate of its memory uses Go's rows). A game with rows smaller
#   than Go's keeps upstream's counts: the scale bounds a worker's memory, and larger counts would only coarsen the
#   shuffle and the training (Connect-4's rows of 1,470 bytes would give files of 365,000 rows: the window's 100,000
#   rows in one file and one worker, and an epoch of at least a whole file, as train.py takes each file whole).
# - 4 shuffle workers, not upstream's 8: a worker holds about 1.7 GB of Enclosure's rows, and the shuffle runs beside
#   self-play (2026-10-03: 8 workers ran out of 8 GB). At most the run's CPUs.
# - Each part's thread pools within the run's CPUs (CPUS), when the run has fewer CPUs than the machine. The libraries'
#   own defaults count the machine's cores: a run on 2 CPUs kept about 8 runnable threads of Eigen's pool alone
#   (2026-10-18). Eigen's network threads in self-play (numEigenThreadsPerModel, unless the self-play config sets it)
#   take one thread a CPU of the run: each evaluates its own batch. torch and numpy in each Python part
#   (OMP_NUM_THREADS, MKL_NUM_THREADS) take one thread: their pool waits for all its threads at each operation, and the
#   parts share the run's CPUs. On 2 CPUs beside 2 busy threads, Connect-4's trainer trained 106 samples a second with
#   2 threads and 261 with 1 (665 and 482 alone, 2026-10-19). With the machine's CPUs, nothing is set.
# - The shuffle every SHUFFLE_EVERY seconds (600), not upstream's 20 (async_start.sh). The trainer reads each shuffled
#   file once (-no-repeat-files), so one shuffle gives it at most the window's rows: while the window holds few epochs,
#   a long SHUFFLE_EVERY limits the training, not MAX_TRAIN_PER_DATA. At 600 s, Blueprints' bpj3 trained one epoch of
#   16,000 samples after each shuffle in its first hour, while its bucket allowed 56,000 (2026-10-27).
# - SHUFFLE_EXTRA_ARGS: more shuffle.py flags, for both shuffles (training and validation). -exclude-qvalues leaves out
#   the rows' Q-value targets: only a network that predicts Q values reads them (model version 16, or 17 with
#   predict_q_values), and they are 161,610 of a Blueprints row's 365,356 bytes.
# - PILOT=1, for a short run, such as a new game's first run of 30 minutes, so that it trains on its networks' own rows.
#   Until the first export, self-play plays as many games as the run has CPUs (pilot_selfplay.sh): it plays a random
#   network, which needs no evaluation, so each game is a runnable thread. A network's rows go to a file every 1,000
#   rows or fewer (maxRowsPerTrainFile, upstream's 10,000 for Go's rows; self-play writes a file only when it has that
#   many rows, at a network switch, and at its stop). The shuffle runs every 60 s, and a trainer that waits for data
#   looks again after 30 s (WAIT_FOR_DATA_SECONDS, train.py's -wait-for-data-seconds: 300 by default). A long run keeps
#   its settings. A run that goes on past its first hours restarts with PILOT=0: each shuffle reads the whole window
#   again, and with Blueprints' window of 163,631 rows the pilot's shuffle ran about 5 minutes in every 6 on 4 of the
#   run's 8 CPUs, beside self-play (bpj2, 2026-10-27).
RUN_LEVERS="PILOT CPUS NUM_THREADS_FOR_SHUFFLING NUM_TRAIN_SAMPLES_PER_EPOCH MAX_TRAIN_PER_DATA
NUM_TRAIN_SAMPLES_PER_SWA BATCHSIZE SHUFFLE_MINROWS MAX_TRAIN_SAMPLES_PER_CYCLE TAPER_WINDOW_SCALE SHUFFLE_KEEPROWS
ROWS_PER_SHUFFLE_FILE SHUFFLE_WORKER_GROUP_ROWS SHUFFLE_EXTRA_ARGS VALIDATE TRAIN_EXTRA_ARGS SHUFFLE_EVERY EXPORT_EVERY
WAIT_FOR_DATA_SECONDS OMP_NUM_THREADS MKL_NUM_THREADS SELFPLAY_CONFIG ANALYSIS_CONFIG SELFPLAY_IN_FORCE POS_LEN
ROW_BYTES"
ASYNC_KATAGO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # KataGame

cpu_count() {  # LIST: the CPUs in a CPU list of taskset's form (0-15, 12-15, 0,2,4-7)
    local n=0 r IFS=,
    for r in $1; do
        [[ "$r" =~ ^([0-9]+)(-([0-9]+))?$ ]] || { echo "cpu_count: '$1' is not a list of CPUs" >&2; return 1; }
        n=$(( n + ${BASH_REMATCH[3]:-${BASH_REMATCH[1]}} - BASH_REMATCH[1] + 1 ))
    done
    echo "$n"
}

config_value() {  # KEY: its value in the self-play config, or nothing
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\([^#[:space:]]*\).*/\1/p" "$SELFPLAY_CONFIG" 2>/dev/null \
        | tail -1
}

run_defaults() {  # needs GAME
    local cpus rows
    : "${CPUS:=0-15}" "${PILOT:=0}"
    [[ "$PILOT" == [01] ]] || { echo "PILOT is 0 (a long run) or 1 (a pilot), not '$PILOT'" >&2; return 1; }
    cpus=$(cpu_count "$CPUS") || return 1
    : "${SELFPLAY_CONFIG:=$ASYNC_KATAGO/cpp/configs/training/selfplay_$GAME.cfg}"
    : "${ANALYSIS_CONFIG:=$ASYNC_KATAGO/cpp/configs/analysis_$GAME.cfg}"
    : "${NUM_THREADS_FOR_SHUFFLING:=$(( cpus < 4 ? cpus : 4 ))}"
    : "${NUM_TRAIN_SAMPLES_PER_EPOCH:=100000}" "${MAX_TRAIN_PER_DATA:=8}" "${NUM_TRAIN_SAMPLES_PER_SWA:=80000}"
    : "${BATCHSIZE:=128}" "${SHUFFLE_MINROWS:=100000}" "${MAX_TRAIN_SAMPLES_PER_CYCLE:=500000}"
    : "${TAPER_WINDOW_SCALE:=50000}" "${SHUFFLE_KEEPROWS:=600000}" "${VALIDATE:=0}" "${TRAIN_EXTRA_ARGS:=}"
    : "${SHUFFLE_EXTRA_ARGS:=}"
    : "${EXPORT_EVERY:=10}"  # upstream's
    # Self-play's -override-config: the defaults of its config keys, then the run's own (SELFPLAY_OVERRIDES), which win.
    SELFPLAY_IN_FORCE=""
    if (( cpus < $(getconf _NPROCESSORS_ONLN) )); then
        : "${OMP_NUM_THREADS:=1}" "${MKL_NUM_THREADS:=1}"
        export OMP_NUM_THREADS MKL_NUM_THREADS
        [[ -n "$(config_value numEigenThreadsPerModel)" ]] || selfplay_key "numEigenThreadsPerModel=$cpus"
    fi
    if [[ "$PILOT" == 1 ]]; then
        : "${SHUFFLE_EVERY:=60}" "${WAIT_FOR_DATA_SECONDS:=30}"
        rows=$(config_value maxRowsPerTrainFile)
        if [[ -z "$rows" || "$rows" -gt 1000 ]]; then selfplay_key maxRowsPerTrainFile=1000; fi
    else
        : "${SHUFFLE_EVERY:=600}"
    fi
    selfplay_key "${SELFPLAY_OVERRIDES:-}"
}

selfplay_key() {  # KEY=VALUE[,...]: added to SELFPLAY_IN_FORCE (nothing for an empty argument)
    SELFPLAY_IN_FORCE="$SELFPLAY_IN_FORCE${SELFPLAY_IN_FORCE:+${1:+,}}$1"
}

train_flags() {  # the train.py flags of every trainer, from the levers (after run_defaults)
    echo "-game $GAME -samples-per-epoch $NUM_TRAIN_SAMPLES_PER_EPOCH -swa-period-samples $NUM_TRAIN_SAMPLES_PER_SWA" \
        "-no-repeat-files -max-train-bucket-per-new-data $MAX_TRAIN_PER_DATA -max-train-bucket-size" \
        "$MAX_TRAIN_SAMPLES_PER_CYCLE${WAIT_FOR_DATA_SECONDS:+ -wait-for-data-seconds $WAIT_FOR_DATA_SECONDS}" \
        "$TRAIN_EXTRA_ARGS"
}

scaled_rows() {  # UPSTREAM GO_BYTES BYTES: upstream's count of Go's rows, scaled to the same bytes of rows of BYTES
    local n=$(( ($1 * $2 + 50 * $3) / (100 * $3) * 100 ))
    (( n < 100 )) && n=100
    (( n > $1 )) && n=$1
    echo "$n"
}

# game_defaults (needs GAME): the game's largest board (POS_LEN, the trainers' -pos-len), its rows' bytes (ROW_BYTES)
# and the shuffle's counts. It reads them with the engine repository's .venv-katago (async_start.sh's python), or with
# python3 from PATH when that venv is missing. When a game's module exists but does not load, it says which python ran
# and the error's last line: a python with no torch (for example in a worktree without the venv) fails there.
game_defaults() {
    local facts go python err venv
    venv="$(cd "$ASYNC_KATAGO/.." && pwd)/.venv-katago"
    python="$venv/bin/python3"
    [[ -x "$python" ]] || python=python3
    err=$(mktemp) || return 1
    if ! facts=$(cd "$ASYNC_KATAGO"/python && "$python" -c "from katago.train import games; g = games.get('$GAME'); \
print(g.MAX_BOARD_LEN, games.row_bytes(g), games.GO_ROW_BYTES)" 2>"$err"); then
        if [[ ! -e "$ASYNC_KATAGO/python/katago/train/games/$GAME.py" ]]; then
            echo "no trainer's module for $GAME (python/katago/train/games/$GAME.py)" >&2
        else
            [[ "$python" == python3 ]] && echo "no $venv: python3 from PATH ran ($(command -v python3))" >&2
            echo "$python could not load the trainer's module for $GAME: $(tail -1 "$err")" >&2
        fi
        rm -f "$err"
        return 1
    fi
    cat "$err" >&2
    rm -f "$err"
    read -r POS_LEN ROW_BYTES go <<< "$facts"
    : "${ROWS_PER_SHUFFLE_FILE:=$(scaled_rows 70000 "$go" "$ROW_BYTES")}"
    : "${SHUFFLE_WORKER_GROUP_ROWS:=$(scaled_rows 80000 "$go" "$ROW_BYTES")}"
    export POS_LEN
}
