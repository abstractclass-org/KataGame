"""The trainer's reader of rows as it was before it read the arrays in their dtypes on disk (KataGame 83843fca,
katago/train/data_processing_pytorch.read_npz_training_data): the reference of test_npz_reader_batches.py, which checks
that the reader gives its batches, bit for bit. It reads each file's planes to float32 in its loading thread, and copies
each batch's arrays in float32 to the device. The game's work on each batch on the device is the reader's own
(data_processing_pytorch.finish_batch), so a change of the game's work changes both readers. Do not change the rest: it
is the old reader's load and copy, not a reader to use."""
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from katago.train import data_processing_pytorch as dpp
from katago.train import games, modelconfigs
from katago.train.data_processing_pytorch import (fit_binary_input_nchw, fit_global_input_nc, pad_global_targets_nc,
                                                  pad_value_targets_nchw)


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
        if dpp.ENCLOSURE_ZERO_INPUT_CHANNELS:
            binaryInputNCHW[:, dpp.ENCLOSURE_ZERO_INPUT_CHANNELS] = 0.0
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
                batch_metadataInputNC = None
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                batch_qValueTargetsNCMove = None
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                # The game's work on the batch, the reader's own (finish_batch).
                symm = symmetries[int(rand.integers(0,len(symmetries)))] if randomize_symmetries else None
                yield dpp.finish_batch(game, inputs_version, symm, batch_binaryInputNCHW, batch_globalInputNC,
                                       batch_policyTargetsNCMove, batch_globalTargetsNC, batch_scoreDistrN,
                                       batch_valueTargetsNCHW, batch_metadataInputNC, batch_qValueTargetsNCMove)
