# A V2 run's trainers and the resources of its parts, for enclosure_async.sh and pipeline/scripts/lib.sh (source this file).
#
# Trainers: TRAINERS="b10c128 b15c192", each a folder train/NAME. The first trains the self-play network (train.sh's
# main mode: it exports to models/); the others train alongside on the same data (extra mode: models_extra/, never
# used by self-play). A size switch is a reorder. Each trainer's architecture is MODELKIND_NAME (default NAME), its own
# train.py flags TRAIN_ARGS_NAME (beside TRAIN_EXTRA_ARGS, which every trainer gets), and its part train-NAME.
#
# Resources: each part runs in its own systemd scope with a memory cap (mem=), a CPU weight (cpu=, systemd's CPUWeight:
# the kernel shares the CPUs between scopes by weight, not by nice) and a nice value (nice=). LIMITS_GROUP sets them
# for a group of parts (selfplay, shuffle, export, train: every trainer) and LIMITS_PART for one part (LIMITS_train_b10c128
# for train-b10c128), each over the defaults below, e.g. LIMITS_train="mem=16G cpu=20".

# Defaults: self-play at 512 games peaked at 20.9 GB; a trainer at 6 GB was killed (2026-10-03, at 5.9 GB).
PART_DEFAULTS_selfplay="mem=24G cpu=100 nice=10"
PART_DEFAULTS_shuffle="mem=10G cpu=100 nice=15"
PART_DEFAULTS_export="mem=4G cpu=100 nice=10"
PART_DEFAULTS_train="mem=8G cpu=100 nice=10"

trainer_kind() {  # NAME: its architecture
    local v="MODELKIND_$1"
    echo "${!v:-$1}"
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
