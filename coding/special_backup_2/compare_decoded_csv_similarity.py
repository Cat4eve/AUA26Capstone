#!/usr/bin/env python3
"""
compare_decoded_csv_similarity.py

Compare original CSV matrix layers with decoded/reconstructed CSV matrix layers.

It reports:
- MAE
- RMSE
- Max absolute error
- Relative Frobenius error
- Similarity percentage

Formula:
    relative_error = ||original - decoded|| / ||original||
    similarity %   = 100 * (1 - relative_error)

Usage:
    python compare_decoded_csv_similarity.py original_folder decoded_folder

Example:
    python compare_decoded_csv_similarity.py test_matrices roundtrip_output/decoded_csv

Optional:
    python compare_decoded_csv_similarity.py test_matrices roundtrip_output/decoded_csv --output similarity_report.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_matrix(path):
    """
    Load a CSV matrix.
    """
    try:
        mat = np.loadtxt(path, delimiter=",")
    except ValueError:
        mat = np.loadtxt(path)

    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    return np.asarray(mat, dtype=np.float64)


def normalized_name(path):
    """
    Normalize names for matching.

    Example:
        layer_0_small.csv
        layer_0_small_decoded.csv

    Both become:
        layer_0_small
    """
    stem = Path(path).stem

    if stem.endswith("_decoded"):
        stem = stem[:-len("_decoded")]

    return stem


def collect_csv_files(folder):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    ])

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    return files


def pair_files(original_folder, decoded_folder, match_by_name=True):
    original_files = collect_csv_files(original_folder)
    decoded_files = collect_csv_files(decoded_folder)

    if match_by_name:
        original_map = {normalized_name(p): p for p in original_files}
        decoded_map = {normalized_name(p): p for p in decoded_files}

        common_names = sorted(set(original_map.keys()) & set(decoded_map.keys()))

        if common_names:
            pairs = [(original_map[name], decoded_map[name]) for name in common_names]

            missing_decoded = sorted(set(original_map.keys()) - set(decoded_map.keys()))
            missing_original = sorted(set(decoded_map.keys()) - set(original_map.keys()))

            if missing_decoded:
                print("Warning: these original files had no decoded match:")
                for name in missing_decoded:
                    print(f"  {name}")

            if missing_original:
                print("Warning: these decoded files had no original match:")
                for name in missing_original:
                    print(f"  {name}")

            return pairs

    # Fallback: pair by sorted order.
    if len(original_files) != len(decoded_files):
        raise ValueError(
            f"Different number of CSV files: "
            f"{len(original_files)} original vs {len(decoded_files)} decoded. "
            f"Use matching names or fix the folders."
        )

    return list(zip(original_files, decoded_files))


def compare_matrices(original_folder, decoded_folder, output_csv="similarity_report.csv", match_by_name=True):
    pairs = pair_files(original_folder, decoded_folder, match_by_name=match_by_name)

    results = []

    total_squared_error = 0.0
    total_original_energy = 0.0
    total_elements = 0

    for original_path, decoded_path in pairs:
        original = read_csv_matrix(original_path)
        decoded = read_csv_matrix(decoded_path)

        if original.shape != decoded.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"  original: {original_path.name}, shape={original.shape}\n"
                f"  decoded:  {decoded_path.name}, shape={decoded.shape}"
            )

        diff = original - decoded

        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_error = float(np.max(np.abs(diff)))

        frobenius_error = float(np.linalg.norm(diff))
        original_norm = float(np.linalg.norm(original))

        if original_norm < 1e-12:
            relative_error = 0.0 if frobenius_error < 1e-12 else 1.0
        else:
            relative_error = frobenius_error / original_norm

        similarity_percent = max(0.0, 100.0 * (1.0 - relative_error))

        total_squared_error += float(np.sum(diff ** 2))
        total_original_energy += float(np.sum(original ** 2))
        total_elements += int(original.size)

        results.append({
            "original_file": original_path.name,
            "decoded_file": decoded_path.name,
            "shape": str(original.shape),
            "MAE": mae,
            "RMSE": rmse,
            "max_abs_error": max_error,
            "relative_error": relative_error,
            "similarity_percent": similarity_percent,
        })

    if total_original_energy < 1e-12:
        overall_relative_error = 0.0 if total_squared_error < 1e-12 else 1.0
    else:
        overall_relative_error = float(np.sqrt(total_squared_error) / np.sqrt(total_original_energy))

    overall_similarity_percent = max(0.0, 100.0 * (1.0 - overall_relative_error))

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("=" * 70)
    print("LAYER-BY-LAYER SIMILARITY")
    print("=" * 70)

    for row in results:
        print(
            f"{row['original_file']}  vs  {row['decoded_file']}\n"
            f"  Similarity:      {row['similarity_percent']:.6f}%\n"
            f"  Relative error:  {row['relative_error']:.10f}\n"
            f"  RMSE:            {row['RMSE']:.10f}\n"
            f"  MAE:             {row['MAE']:.10f}\n"
            f"  Max abs error:   {row['max_abs_error']:.10f}\n"
        )

    print("=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)
    print(f"Overall similarity:      {overall_similarity_percent:.6f}%")
    print(f"Overall relative error:  {overall_relative_error:.10f}")
    print(f"Total compared values:   {total_elements}")
    print(f"Saved report to:         {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare original CSV matrices with decoded CSV matrices."
    )

    parser.add_argument(
        "original_folder",
        help="Folder containing original CSV matrix files.",
    )

    parser.add_argument(
        "decoded_folder",
        help="Folder containing decoded/reconstructed CSV matrix files.",
    )

    parser.add_argument(
        "--output",
        default="similarity_report.csv",
        help="Output CSV report path. Default: similarity_report.csv",
    )

    parser.add_argument(
        "--no-name-match",
        action="store_true",
        help="Do not match files by name. Instead compare sorted original CSVs with sorted decoded CSVs.",
    )

    args = parser.parse_args()

    compare_matrices(
        original_folder=args.original_folder,
        decoded_folder=args.decoded_folder,
        output_csv=args.output,
        match_by_name=not args.no_name_match,
    )


if __name__ == "__main__":
    main()
