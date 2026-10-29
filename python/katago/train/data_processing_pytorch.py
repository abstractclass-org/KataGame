import logging
import os

import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

from ..train import games
from ..train import modelconfigs

# Needs to be kept in sync with GLOBAL_TARGET_NUM_CHANNELS in trainingwrite.cpp C++ code among other places.
# Data format version 2 files (recorded in channel 63 of each row) had only 64 channels; they are zero-padded
# up to this width when loading, which correctly encodes "not reanalyzed" for the version 3 channels. KataGame fork
# (2026-10-29): the rows of the inputs version 14 add AZdb's 4 targets of the intrinsic value head (C80-83, C++
# GLOBAL_TARGET_NUM_CHANNELS_V14). The rows of earlier versions have 80 channels, zero-padded here: their weights C81
# and C83 are 0.
GLOBAL_TARGETS_NC_CHANNELS = 84

def pad_global_targets_nc(globalTargetsNC: np.ndarray) -> np.ndarray:
    """Zero-pad older-format globalTargetsNC rows up to the current channel count."""
    num_channels = globalTargetsNC.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return globalTargetsNC
    assert num_channels < GLOBAL_TARGETS_NC_CHANNELS, f"globalTargetsNC has {num_channels} channels, more than the expected {GLOBAL_TARGETS_NC_CHANNELS}"
    padded = np.zeros((globalTargetsNC.shape[0], GLOBAL_TARGETS_NC_CHANNELS), dtype=globalTargetsNC.dtype)
    padded[:, :num_channels] = globalTargetsNC
    return padded

# Enclosure fork (2026-10-13): the C++ V9 rows add the playout doubling advantage as 2 global inputs after the game's
# (Enclosure: 5, modelconfigs.get_inputs_version). Rows written before have the game's alone and are zero-padded, as
# rows with no advantage are. A model with the game's inputs alone reads the first ones of a V9 row: it has no inputs
# for the advantage. KataGame fork (2026-10-27): it fits any array on dimension 1, the planes too
# (fit_binary_input_nchw).
def fit_global_input_nc(globalInputNC: np.ndarray, num_global_features: int) -> np.ndarray:
    num_channels = globalInputNC.shape[1]
    if num_channels == num_global_features:
        return globalInputNC
    if num_channels > num_global_features:
        return globalInputNC[:, :num_global_features]
    padded = np.zeros((globalInputNC.shape[0], num_global_features, *globalInputNC.shape[2:]), dtype=globalInputNC.dtype)
    padded[:, :num_channels] = globalInputNC
    return padded

# KataGame fork (2026-10-27): the rows of the inputs version 12 are version 11's with the game's planes of closed areas
# after them (games.area_spatial_inputs). A model of version 12 reads a row of version 11 with those planes at 0, and a
# model of version 11 reads the first planes of a row of version 12, as fit_global_input_nc fits the global inputs. So
# a run switches versions with a converted checkpoint (add_extended_inputs.py) while its window holds rows of both. The
# converted weights of those planes start at 0, and a plane at 0 gives its weights no gradient, so the rows of version
# 11 train them not at all. Any other count of planes stays, and the reader refuses it (the shuffle refuses the files
# of other versions too: shuffle.pad_planes_to_widest). The planes are on dimension 1, packed or not.
def fit_binary_input_nchw(binaryInputNCHWPacked: np.ndarray, num_bin_features: int, game) -> np.ndarray:
    areas = games.area_spatial_inputs(game)
    if areas == 0 or abs(binaryInputNCHWPacked.shape[1] - num_bin_features) != areas:
        return binaryInputNCHWPacked
    return fit_global_input_nc(binaryInputNCHWPacked, num_bin_features)

# Enclosure fork (owner, 2026-10-03): valueTargetsNCHW gained C5-6 (the territory income), and C7 (2026-10-14: the
# next mover's nodes, for the symmetries of policy channel 1, not a target); rows written before have 5 or 7
# channels and are zero-padded (their C69 and C75 flags are 0, so the padding is never read).
VALUE_TARGETS_NCHW_CHANNELS = 8

def pad_value_targets_nchw(valueTargetsNCHW: np.ndarray) -> np.ndarray:
    """Zero-pad older-format valueTargetsNCHW rows up to the current channel count."""
    num_channels = valueTargetsNCHW.shape[1]
    if num_channels == VALUE_TARGETS_NCHW_CHANNELS:
        return valueTargetsNCHW
    assert num_channels < VALUE_TARGETS_NCHW_CHANNELS, f"valueTargetsNCHW has {num_channels} channels, more than the expected {VALUE_TARGETS_NCHW_CHANNELS}"
    padded = np.zeros((valueTargetsNCHW.shape[0], VALUE_TARGETS_NCHW_CHANNELS, *valueTargetsNCHW.shape[2:]), dtype=valueTargetsNCHW.dtype)
    padded[:, :num_channels] = valueTargetsNCHW
    return padded

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    include_meta: bool,
    model_config: modelconfigs.ModelConfig,
    prefetch_depth: int = 1,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    inputs_version = modelconfigs.get_inputs_version(model_config)
    # Enclosure fork: the history dropout and the symmetries are the game's (apply_history_dropout, katago/train/games).
    game = modelconfigs.get_game(model_config)
    symmetries = games.symmetries(game)

    # Version 16 always predicts q values; version 17+ does so only when configured.
    include_qvalues = model_config["version"] == 16 or (
        model_config["version"] >= 17 and bool(model_config.get("predict_q_values"))
    )

    def load_npz_file(npz_file):
        # Select only THIS rank's rows up front, while the arrays are still in
        # their compact on-disk dtypes (packed bits / int8 / int16), so the
        # expensive unpackbits + float32 expansion runs on 1/world_size of the
        # data rather than the whole shard in every rank.
        with np.load(npz_file) as npz:
            num_samples = npz["globalInputNC"].shape[0]
            num_whole_steps = num_samples // (batch_size * world_size)
            used = num_whole_steps * world_size * batch_size

            def select_rank_rows(arr):
                # Keep only the rows this rank will consume:
                # reshape the used prefix to (steps, world_size, batch, ...) and
                # take this rank's slice. For world_size>1 the trailing reshape
                # forces a compact 1/world_size-size copy and lets the full
                # decompressed array be freed immediately.
                # Drop any trailing suffix that doesn't match the overall world batch size.
                arr = arr[:used]
                rest = arr.shape[1:]
                arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
                arr = arr[:, rank]
                return arr.reshape(num_whole_steps * batch_size, *rest)

            binaryInputNCHWPacked = fit_binary_input_nchw(select_rank_rows(npz["binaryInputNCHWPacked"]), num_bin_features, game)
            globalInputNC = select_rank_rows(npz["globalInputNC"])
            # KataGame fork (2026-10-29): a model of the inputs version 14 reads the rows of its version only. A row of
            # an earlier version holds no member's latent, which fit_global_input_nc would leave at 0, no member's:
            # AZdb's run trains on its own rows only (shuffle.refuse_mixed_global_targets).
            if inputs_version >= 14 and globalInputNC.shape[1] < num_global_features:
                raise ValueError(f"{npz_file}: rows of {globalInputNC.shape[1]} global inputs, without the latent of "
                                 f"the inputs version 14 ({num_global_features})")
            globalInputNC = fit_global_input_nc(globalInputNC, num_global_features)
            policyTargetsNCMove = select_rank_rows(npz["policyTargetsNCMove"]).astype(np.float32)
            globalTargetsNC = pad_global_targets_nc(select_rank_rows(npz["globalTargetsNC"]))
            scoreDistrN = select_rank_rows(npz["scoreDistrN"]).astype(np.float32)
            valueTargetsNCHW = pad_value_targets_nchw(select_rank_rows(npz["valueTargetsNCHW"])).astype(np.float32)
            if include_meta:
                metadataInputNC = select_rank_rows(npz["metadataInputNC"]).astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = select_rank_rows(npz["qValueTargetsNCMove"]).astype(np.float32)
            else:
                qValueTargetsNCMove = None
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        if ENCLOSURE_ZERO_INPUT_CHANNELS:
            binaryInputNCHW[:, ENCLOSURE_ZERO_INPUT_CHANNELS] = 0.0
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    # Prefetch up to prefetch_depth files *ahead* of the one currently being
    # consumed, so the GPU does not stall at a file boundary waiting on disk +
    # decompress + unpackbits for the next shard.
    # Each in-flight file holds its full expanded (float32) arrays in RAM,
    # so memory scales linearly with prefetch_depth
    # (times world_size, since every rank loads each file).
    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        # Keep a queue of in-flight loads: the head is the file being consumed,
        # and up to prefetch_depth more are loading/loaded behind it.
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1

        while pending:
            future = pending.popleft()
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            # The arrays already hold only this rank's rows (selected in load_npz_file),
            # so the first dim is num_whole_steps * batch_size.
            num_whole_steps = binaryInputNCHW.shape[0] // batch_size

            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            # Top the pipeline back up so prefetch_depth files stay in flight.
            if next_index < len(npz_files):
                logging.info(f"Preloading {npz_files[next_index]} while processing this file")
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1

            for n in range(num_whole_steps):
                start = n * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                # KataGame fork: the targets that the game's symmetries cannot map get weight 0, in every batch.
                batch_globalTargetsNC = games.mask_targets(game, batch_binaryInputNCHW, batch_globalInputNC, batch_valueTargetsNCHW, batch_globalTargetsNC)
                batch_binaryInputNCHW, batch_globalInputNC = apply_history_dropout(game, batch_binaryInputNCHW, batch_globalInputNC, inputs_version)

                if randomize_symmetries:
                    # Enclosure fork: one of the game's allowed symmetries, applied by the game (katago/train/games).
                    # The policy targets map with the row before the symmetry (Enclosure: an edge moves as an edge,
                    # with each channel's player's nodes, the next mover's from valueTargetsNCHW C7).
                    symm = symmetries[int(rand.integers(0,len(symmetries)))]
                    batch_policyTargetsNCMove = game.apply_symmetry_policy_targets(batch_policyTargetsNCMove, symm, batch_binaryInputNCHW, batch_globalInputNC, batch_valueTargetsNCHW, batch_globalTargetsNC)
                    if include_qvalues:
                        batch_qValueTargetsNCMove = game.apply_symmetry_policy(batch_qValueTargetsNCMove, symm, batch_binaryInputNCHW, batch_globalInputNC)
                    batch_binaryInputNCHW = game.apply_symmetry_inputs(batch_binaryInputNCHW, symm)
                    # KataGame fork (2026-10-26): the extended global inputs of the inputs version 10 can name points.
                    batch_globalInputNC = games.apply_symmetry_global_inputs(game, batch_globalInputNC, symm)
                    batch_valueTargetsNCHW = games.apply_symmetry_points(batch_valueTargetsNCHW, symm)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


# Enclosure fork: KataGo's random history masking (apply_history_matrices, the same draws) on the game's history
# planes (katago/train/games: Enclosure's feature planes marking the edges placed 2-5 turns ago), newest first. History
# item k is kept only if none of k+1 independent draws stops it (each with probability 0.02), so older history is
# dropped more often, and never a newer item without the older ones. The C++ evaluation without history
# (maxHistory <= 0) drops them all. A game without history planes draws nothing. KataGame fork (2026-10-27): in the rows
# of the inputs version 11 the items are the game's turns (NUM_HISTORY_TURNS), one draw each, and a turn goes with its
# planes and its turn inputs (games.turn_global_inputs), as KataGo's pass inputs go with their moves. The rows of the
# other versions keep a draw for each plane.
# Enclosure fork, for input-plane comparisons (2026-10-02): ENCLOSURE_ZERO_INPUT_CHANNELS="200,201,209,210" blanks
# those spatial input channels in every row read, training and validation alike (feature planes start at 193; per
# player 9: node, enclosed, solid, reach 1-3, unreached, held ground, its openings). Unset: no change.
ENCLOSURE_ZERO_INPUT_CHANNELS = [int(c) for c in os.environ.get("ENCLOSURE_ZERO_INPUT_CHANNELS", "").split(",") if c]

def apply_history_dropout(game, batch_binaryInputNCHW, batch_globalInputNC, inputs_version):
    num_history = game.NUM_HISTORY_PLANES
    if num_history == 0:
        return batch_binaryInputNCHW, batch_globalInputNC
    start = 1 + game.HISTORY_PLANE  # after KataGo's on-board plane
    assert start + num_history <= 1 + game.NUM_SPATIAL_INPUTS
    n = batch_binaryInputNCHW.shape[0]
    # The history's items: each plane, or under the inputs version 11 (the model's, as the reader fits the rows' global
    # inputs to it) each turn with its planes and its turn inputs, which follow the game's extended global inputs: the
    # last global inputs, but for version 14's latent after them.
    turns = games.turn_global_inputs(game) if inputs_version >= 11 else 0
    items = game.NUM_HISTORY_TURNS if turns else num_history
    should_stop_history = torch.rand((n, items), device=batch_binaryInputNCHW.device) >= 0.98
    include_history = (torch.cumsum(should_stop_history, axis=1, dtype=torch.float32) <= 0.1).to(batch_binaryInputNCHW.dtype)
    out = batch_binaryInputNCHW.clone()
    planes = include_history.repeat_interleave(num_history // items, dim=1)  # each item's draw on its planes
    out[:, start:start + num_history] *= planes.view(n, num_history, 1, 1)
    if not turns:
        return out, batch_globalInputNC
    out_global = batch_globalInputNC.clone()
    first = games.num_global_inputs(game, 10)
    out_global[:, first:first + turns] *= include_history.repeat_interleave(turns // items, dim=1).to(batch_globalInputNC.dtype)
    return out, out_global

def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC
