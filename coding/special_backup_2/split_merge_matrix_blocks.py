#!/usr/bin/env python3
"""
split_merge_matrix_blocks.py

Utility for improving matrix -> VTP -> hologram -> decode quality.

Problem:
    In the current hologram pipeline, each matrix is placed into one FFT tile.
    If a matrix is larger than its tile, the encoder crops its FFT spectrum.
    This is why FFN1/FFN2 matrices can decode poorly while 768x768 Q/K/V
    matrices decode almost perfectly.

Solution:
    Split large matrices into smaller blocks, for example 768x768.
    Then each block becomes its own matrix layer for the hologram pipeline.
    After decoding, merge the decoded blocks back into the original matrices.

Typical BERT example:
    FFN1: 768 x 3072  -> four 768 x 768 blocks
    FFN2: 3072 x 768  -> four 768 x 768 blocks
    Q/K/V/attention: 768 x 768 -> one block

Commands:

1. Split original matrices:
    python split_merge_matrix_blocks.py split original_matrices split_blocks --block_rows 768 --block_cols 768

2. Run your hologram roundtrip on split_blocks:
    python matrix_group_holography_pipeline.py roundtrip split_blocks roundtrip_blocks --holo_height 3840 --holo_width 3840 --block_gap 2

3. Merge decoded blocks:
    python split_merge_matrix_blocks.py merge roundtrip_blocks/decoded_csv merged_decoded split_blocks/block_metadata.json

4. Compare:
    python compare_decoded_csv_similarity.py original_matrices merged_decoded
"""

import argparse
import json
from pathlib import Path

import numpy as np


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy"}


def load_matrix(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mat = np.load(path)
    else:
        try:
            mat = np.loadtxt(path, delimiter=",")
        except ValueError:
            mat = np.loadtxt(path)

    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix: {path}")

    return np.asarray(mat, dtype=np.float64)


def save_csv(path, mat):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, delimiter=",", fmt="%.10g")


def normalize_decoded_stem(path):
    """
    Decoder usually creates names like:
        FFN1_01__block_r0000_c0000_decoded.csv

    This returns:
        FFN1_01__block_r0000_c0000
    """
    stem = Path(path).stem
    if stem.endswith("_decoded"):
        stem = stem[:-len("_decoded")]
    return stem


def split_folder(input_folder, output_folder, block_rows=768, block_cols=768):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not files:
        raise FileNotFoundError(f"No matrix files found in {input_folder}")

    metadata = {
        "version": 1,
        "block_rows": int(block_rows),
        "block_cols": int(block_cols),
        "matrices": []
    }

    total_blocks = 0

    for path in files:
        mat = load_matrix(path)
        rows, cols = mat.shape
        base = path.stem

        matrix_info = {
            "original_file": path.name,
            "base": base,
            "shape": [int(rows), int(cols)],
            "blocks": []
        }

        for r0 in range(0, rows, block_rows):
            for c0 in range(0, cols, block_cols):
                block = mat[r0:min(r0 + block_rows, rows), c0:min(c0 + block_cols, cols)]

                block_name = (
                    f"{base}__block_r{r0:04d}_c{c0:04d}"
                    f"__orig_{rows}x{cols}.csv"
                )

                save_csv(output_folder / block_name, block)

                matrix_info["blocks"].append({
                    "block_file": block_name,
                    "block_stem": Path(block_name).stem,
                    "r0": int(r0),
                    "c0": int(c0),
                    "rows": int(block.shape[0]),
                    "cols": int(block.shape[1])
                })

                total_blocks += 1

        metadata["matrices"].append(matrix_info)

        print(
            f"Split {path.name}: shape={rows}x{cols}, "
            f"blocks={len(matrix_info['blocks'])}"
        )

    meta_path = output_folder / "block_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DONE: split matrices into blocks")
    print("=" * 72)
    print(f"Input folder:     {input_folder}")
    print(f"Output folder:    {output_folder}")
    print(f"Metadata:         {meta_path}")
    print(f"Total blocks:     {total_blocks}")
    print(f"Block size:       {block_rows} x {block_cols}")


def merge_folder(decoded_blocks_folder, output_folder, metadata_json):
    decoded_blocks_folder = Path(decoded_blocks_folder)
    output_folder = Path(output_folder)
    metadata_json = Path(metadata_json)

    if not decoded_blocks_folder.is_dir():
        raise FileNotFoundError(f"Decoded blocks folder not found: {decoded_blocks_folder}")

    if not metadata_json.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_json}")

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

    decoded_files = sorted([
        p for p in decoded_blocks_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    ])

    decoded_map = {
        normalize_decoded_stem(p): p
        for p in decoded_files
    }

    output_folder.mkdir(parents=True, exist_ok=True)

    for matrix_info in metadata["matrices"]:
        rows, cols = matrix_info["shape"]
        recon = np.zeros((rows, cols), dtype=np.float64)

        missing = []

        for block_info in matrix_info["blocks"]:
            block_stem = block_info["block_stem"]

            if block_stem not in decoded_map:
                missing.append(block_stem)
                continue

            block = load_matrix(decoded_map[block_stem])

            r0 = int(block_info["r0"])
            c0 = int(block_info["c0"])
            br = int(block_info["rows"])
            bc = int(block_info["cols"])

            if block.shape != (br, bc):
                raise ValueError(
                    f"Block shape mismatch for {decoded_map[block_stem].name}: "
                    f"expected {(br, bc)}, got {block.shape}"
                )

            recon[r0:r0 + br, c0:c0 + bc] = block

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} decoded blocks for {matrix_info['original_file']}. "
                f"First missing: {missing[0]}"
            )

        out_name = Path(matrix_info["original_file"]).stem + "_decoded.csv"
        save_csv(output_folder / out_name, recon)

        print(
            f"Merged {matrix_info['original_file']} -> {out_name}, "
            f"shape={rows}x{cols}, blocks={len(matrix_info['blocks'])}"
        )

    print("=" * 72)
    print("DONE: merged decoded blocks")
    print("=" * 72)
    print(f"Decoded blocks folder: {decoded_blocks_folder}")
    print(f"Output folder:         {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Split large matrix CSVs into blocks before hologram encoding, then merge decoded blocks."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="Split original matrices into smaller CSV blocks")
    split.add_argument("input_folder")
    split.add_argument("output_folder")
    split.add_argument("--block_rows", type=int, default=768)
    split.add_argument("--block_cols", type=int, default=768)

    merge = sub.add_parser("merge", help="Merge decoded matrix blocks back into full matrices")
    merge.add_argument("decoded_blocks_folder")
    merge.add_argument("output_folder")
    merge.add_argument("metadata_json")

    args = parser.parse_args()

    if args.command == "split":
        split_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
        )
    elif args.command == "merge":
        merge_folder(
            decoded_blocks_folder=args.decoded_blocks_folder,
            output_folder=args.output_folder,
            metadata_json=args.metadata_json,
        )


if __name__ == "__main__":
    main()
