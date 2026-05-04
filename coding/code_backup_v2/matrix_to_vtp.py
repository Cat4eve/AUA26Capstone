#!/usr/bin/env python3
"""
CSV/TXT/NPY matrix layers -> ASCII VTP point cloud.

This version does not require the vtk Python package. It writes a valid
VTK XML PolyData (.vtp) file in ASCII format. Besides the 3D point positions,
it also stores point-data arrays needed by the encoder:

- original_value: the matrix value before z-stacking
- layer_index: which input matrix/layer the point came from
- row_index: matrix row
- col_index: matrix column

Those arrays are part of the intermediate VTP representation only. The final
encoded object produced by vtp_to_hologram.py is still a fixed set of 2D images.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def load_matrix(file_path: str) -> np.ndarray:
    """Load one 2D matrix from CSV, TXT, or NPY."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        matrix = np.load(file_path)
    elif ext in {".csv", ".txt"}:
        try:
            matrix = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            matrix = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Input file must contain a 2D matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Input matrix contains NaN or infinite values")
    return matrix


def _array_text(values: Iterable[object]) -> str:
    return " ".join(str(v) for v in values)


def _float_array_text(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.17g}" for v in values)


def matrices_to_points(
    matrices: Sequence[np.ndarray],
    names: Sequence[str] | None = None,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    z_spacing: float = 2.0,
) -> dict:
    """Convert matrix stack into point and metadata arrays."""
    if not matrices:
        raise ValueError("At least one matrix is required")

    points: List[Tuple[float, float, float]] = []
    original_values: List[float] = []
    layer_indices: List[int] = []
    row_indices: List[int] = []
    col_indices: List[int] = []

    layer_names = list(names or [f"layer_{i}" for i in range(len(matrices))])
    layer_shapes = []

    for layer_idx, matrix in enumerate(matrices):
        rows, cols = matrix.shape
        layer_shapes.append((rows, cols))
        for row in range(rows):
            for col in range(cols):
                value = float(matrix[row, col])
                x = col * x_spacing
                y = row * y_spacing
                z = value + layer_idx * z_spacing
                points.append((x, y, z))
                original_values.append(value)
                layer_indices.append(layer_idx)
                row_indices.append(row)
                col_indices.append(col)

    return {
        "points": points,
        "original_values": original_values,
        "layer_indices": layer_indices,
        "row_indices": row_indices,
        "col_indices": col_indices,
        "layer_names": layer_names,
        "layer_shapes": layer_shapes,
    }


def write_ascii_vtp(
    output_path: str,
    point_data: dict,
    x_spacing: float,
    y_spacing: float,
    z_spacing: float,
) -> None:
    """Write point cloud to an ASCII .vtp file."""
    points = point_data["points"]
    n = len(points)
    connectivity = list(range(n))
    offsets = list(range(1, n + 1))
    point_flat = [coord for point in points for coord in point]

    layer_names_text = "|".join(html.escape(name) for name in point_data["layer_names"])
    shapes_text = "|".join(f"{rows},{cols}" for rows, cols in point_data["layer_shapes"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <PolyData x_spacing="{x_spacing:.17g}" y_spacing="{y_spacing:.17g}" z_spacing="{z_spacing:.17g}" layer_names="{layer_names_text}" layer_shapes="{shapes_text}">\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <PointData Scalars="original_value">\n')
        f.write(f'        <DataArray type="Float64" Name="original_value" format="ascii">{_float_array_text(point_data["original_values"])}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="layer_index" format="ascii">{_array_text(point_data["layer_indices"])}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="row_index" format="ascii">{_array_text(point_data["row_indices"])}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="col_index" format="ascii">{_array_text(point_data["col_indices"])}</DataArray>\n')
        f.write('      </PointData>\n')
        f.write('      <CellData/>\n')
        f.write('      <Points>\n')
        f.write(f'        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{_float_array_text(point_flat)}</DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Verts>\n')
        f.write(f'        <DataArray type="Int32" Name="connectivity" format="ascii">{_array_text(connectivity)}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="offsets" format="ascii">{_array_text(offsets)}</DataArray>\n')
        f.write('      </Verts>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


def matrix_to_vtp(
    matrices: Sequence[np.ndarray],
    output_path: str,
    names: Sequence[str] | None = None,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    z_spacing: float = 2.0,
) -> None:
    """Public helper used by the command-line interface and other scripts."""
    point_data = matrices_to_points(
        matrices,
        names=names,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        z_spacing=z_spacing,
    )
    write_ascii_vtp(output_path, point_data, x_spacing, y_spacing, z_spacing)


def discover_matrix_files(input_folder: str) -> List[Path]:
    supported_extensions = {".csv", ".txt", ".npy"}
    input_path = Path(input_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    )
    if not files:
        raise FileNotFoundError(f"No CSV/TXT/NPY files found in: {input_folder}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert all CSV/TXT/NPY files in a folder into an ASCII VTP point cloud."
    )
    parser.add_argument("input_folder", help="Path to folder containing CSV/TXT/NPY files")
    parser.add_argument("output_file", help="Path to output .vtp file")
    parser.add_argument("--x_spacing", type=float, default=1.0, help="Spacing between columns")
    parser.add_argument("--y_spacing", type=float, default=1.0, help="Spacing between rows")
    parser.add_argument("--z_spacing", type=float, default=2.0, help="Height offset between stacked layers")
    args = parser.parse_args()

    matrix_files = discover_matrix_files(args.input_folder)
    matrices = []
    names = []
    for file_path in matrix_files:
        matrix = load_matrix(str(file_path))
        matrices.append(matrix)
        names.append(file_path.stem)
        print(f"Loaded {file_path.name}: shape={matrix.shape}, min={matrix.min():.6g}, max={matrix.max():.6g}")

    matrix_to_vtp(
        matrices,
        args.output_file,
        names=names,
        x_spacing=args.x_spacing,
        y_spacing=args.y_spacing,
        z_spacing=args.z_spacing,
    )
    print(f"Saved VTP point cloud: {args.output_file}")
    print(f"Layers: {len(matrices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
