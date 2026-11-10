#!/usr/bin/python3
"""KataGame fork (2026-11-10, the engine repository's abscls-jtbd): the rows that the CUDA backend's FP8 trunk
calibrates on (ENCLOSURE_FP8_TRUNK=1 with ENCLOSURE_FP8_CALIBRATION=OUT; readFp8CalibrationRows in
katagame/cpp/neuralnet/cudaandrocmbackend.inc), from a run's self-play rows:

    python fp8_calibration_rows.py OUT NPZ [NPZ ...] -size LEN [-rows 256] [-skip 0]

It takes about ROWS / (the number of NPZ) rows from each NPZ, evenly spaced over the file's rows after its first SKIP
(a self-play file holds its games one after the other, so the rows come from many games and every part of a game), and
writes them as the backend reads them: a line "nnrowsbits ROWS CHANNELS LEN GLOBALS", then each row's spatial inputs
packed as the NPZ holds them (binaryInputNCHWPacked: for each channel its LEN x LEN entries, 8 to a byte, the first in
the high bit), then every row's global inputs (globalInputNC, float32, little-endian). NumPy only.
"""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="the rows file to write")
    parser.add_argument("npz", nargs="+", help="self-play rows' files (.npz)")
    parser.add_argument("-size", type=int, required=True, help="the network's board length (LEN x LEN)")
    parser.add_argument("-rows", type=int, default=256, help="the rows to take in all (default 256)")
    parser.add_argument("-skip", type=int, default=0, help="the rows to leave out at the start of each file")
    args = parser.parse_args()
    if args.rows < 1 or args.skip < 0 or args.size < 1:
        parser.error("-rows and -size must be at least 1, and -skip at least 0")

    area = args.size * args.size
    per = -(-args.rows // len(args.npz))
    spatial, global_ = [], []
    for path in args.npz:
        # Only the taken rows stay in memory, one file's arrays at a time.
        with np.load(path) as z:
            packed = z["binaryInputNCHWPacked"]
            if packed.shape[2] != (area + 7) // 8:
                sys.exit(f"{path}: {packed.shape[2]} bytes a channel, a board of {args.size} needs {(area + 7) // 8}")
            n = packed.shape[0] - args.skip
            if n < 1:
                sys.exit(f"{path}: {packed.shape[0]} rows, none after the first {args.skip}")
            take = np.unique(np.linspace(args.skip, packed.shape[0] - 1, min(per, n)).round().astype(int))
            spatial.append(packed[take])
            del packed
            global_.append(z["globalInputNC"][take].astype("<f4"))
    spatial = np.ascontiguousarray(np.concatenate(spatial)[:args.rows], dtype=np.uint8)
    global_ = np.ascontiguousarray(np.concatenate(global_)[:args.rows])
    rows, channels = spatial.shape[0], spatial.shape[1]
    with open(args.out, "wb") as f:
        f.write(f"nnrowsbits {rows} {channels} {args.size} {global_.shape[1]}\n".encode("ascii"))
        f.write(spatial.tobytes())
        f.write(global_.tobytes())
    print(f"wrote {rows} rows of {channels} planes and {global_.shape[1]} global inputs from {len(args.npz)} files")


if __name__ == "__main__":
    main()
