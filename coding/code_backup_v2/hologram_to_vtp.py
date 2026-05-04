#!/usr/bin/env python3
"""
Decode fixed 2D hologram images back to CSV layers and optional VTP.

This decoder uses only the fixed 2D output images produced by vtp_to_hologram.py:

- <base>_real.png
- <base>_imag.png
- <base>_meta.png

It does not load a sidecar containing original matrices, original VTP points, or
layer values. The layer values are recovered from the 2D complex hologram field.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

UINT16_MAX = 65535


def infer_paths(input_hologram: str) -> Dict[str, str]:
    path = Path(input_hologram)
    base = str(path.with_suffix(""))
    return {
        "base": base,
        "preview": input_hologram,
        "real": f"{base}_real.png",
        "imag": f"{base}_imag.png",
        "meta": f"{base}_meta.png",
    }


def load_uint16_png(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required 2D hologram image not found: {path}")
    return np.array(Image.open(path), dtype=np.uint16)


def uint16_to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float64) / (UINT16_MAX / 2.0)) - 1.0


def load_metadata_png(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required 2D metadata image not found: {path}")
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8).reshape(-1)
    if arr.size < 4:
        raise ValueError("Metadata image is too small")
    length = int.from_bytes(bytes(arr[:4]), byteorder="big", signed=False)
    if length <= 0 or 4 + length > arr.size:
        raise ValueError("Metadata image has invalid JSON length")
    raw = bytes(arr[4:4 + length])
    metadata = json.loads(raw.decode("utf-8"))
    if metadata.get("format") != "2d_complex_fourier_hologram_v1":
        raise ValueError("Unsupported hologram metadata format")
    return metadata


def load_complex_field(real_path: str, imag_path: str, metadata: Dict[str, object]) -> np.ndarray:
    real_img = load_uint16_png(real_path)
    imag_img = load_uint16_png(imag_path)
    if real_img.shape != imag_img.shape:
        raise ValueError(f"Real and imaginary images have different shapes: {real_img.shape} vs {imag_img.shape}")
    expected = (int(metadata["hologram_rows"]), int(metadata["hologram_cols"]))
    if real_img.shape != expected:
        raise ValueError(f"Image shape {real_img.shape} does not match metadata shape {expected}")
    field_scaled = uint16_to_float(real_img) + 1j * uint16_to_float(imag_img)
    return field_scaled * float(metadata["field_scale"])


def decode_field_to_layers(field: np.ndarray, metadata: Dict[str, object]) -> List[Tuple[str, np.ndarray]]:
    """Recover matrices from the 2D complex field."""
    spectrum = np.fft.fftshift(np.fft.fft2(field, norm="ortho"))
    decoded: List[Tuple[str, np.ndarray]] = []

    for idx, layer_meta in enumerate(metadata["layers"]):
        rows = int(layer_meta["rows"])
        cols = int(layer_meta["cols"])
        r0 = int(layer_meta["tile_r0"])
        c0 = int(layer_meta["tile_c0"])
        tile = spectrum[r0:r0 + rows, c0:c0 + cols]
        norm_layer = np.fft.ifft2(np.fft.ifftshift(tile), norm="ortho").real

        # Tiny quantization and FFT round-off errors may create values just outside [0, 1].
        norm_layer = np.clip(norm_layer, 0.0, 1.0)
        mn = float(layer_meta["value_min"])
        mx = float(layer_meta["value_max"])
        if abs(mx - mn) < 1e-12:
            layer = np.full((rows, cols), mn, dtype=np.float64)
        else:
            layer = norm_layer * (mx - mn) + mn
        decoded.append((str(layer_meta.get("name", f"layer_{idx}")), layer))
    return decoded


def safe_filename(name: str, fallback: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch if ch in allowed else "_" for ch in name).strip("._")
    return cleaned or fallback


def save_layers_as_csv(layers: Sequence[Tuple[str, np.ndarray]], output_folder: str, decimals: int = 10) -> List[str]:
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    paths = []
    fmt = f"%.{decimals}g"
    for idx, (name, layer) in enumerate(layers):
        filename = f"{idx:03d}_{safe_filename(name, f'layer_{idx}')}.csv"
        path = str(Path(output_folder) / filename)
        np.savetxt(path, layer, delimiter=",", fmt=fmt)
        paths.append(path)
        print(f"Saved CSV layer {idx}: {path} shape={layer.shape}")
    return paths


def _array_text(values: Iterable[object]) -> str:
    return " ".join(str(v) for v in values)


def _float_array_text(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.17g}" for v in values)


def layers_to_ascii_vtp(
    layers: Sequence[Tuple[str, np.ndarray]],
    output_path: str,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    z_spacing: float = 2.0,
) -> None:
    """Write reconstructed layers to ASCII VTP without depending on vtk."""
    points = []
    original_values = []
    layer_indices = []
    row_indices = []
    col_indices = []
    layer_names = []
    layer_shapes = []

    for layer_idx, (name, matrix) in enumerate(layers):
        matrix = np.asarray(matrix, dtype=np.float64)
        rows, cols = matrix.shape
        layer_names.append(name)
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

    n = len(points)
    connectivity = list(range(n))
    offsets = list(range(1, n + 1))
    point_flat = [coord for point in points for coord in point]
    layer_names_text = "|".join(html.escape(name) for name in layer_names)
    shapes_text = "|".join(f"{rows},{cols}" for rows, cols in layer_shapes)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <PolyData x_spacing="{x_spacing:.17g}" y_spacing="{y_spacing:.17g}" z_spacing="{z_spacing:.17g}" layer_names="{layer_names_text}" layer_shapes="{shapes_text}">\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('      <PointData Scalars="original_value">\n')
        f.write(f'        <DataArray type="Float64" Name="original_value" format="ascii">{_float_array_text(original_values)}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="layer_index" format="ascii">{_array_text(layer_indices)}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="row_index" format="ascii">{_array_text(row_indices)}</DataArray>\n')
        f.write(f'        <DataArray type="Int32" Name="col_index" format="ascii">{_array_text(col_indices)}</DataArray>\n')
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
    print(f"Saved reconstructed VTP: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode fixed 2D hologram images back to matrix layers and optional VTP."
    )
    parser.add_argument("input_hologram", help="Phase-preview path used during encoding, e.g. hologram.png")
    parser.add_argument("output_folder", help="Output folder for decoded CSV layers")
    parser.add_argument("--real", default=None, help="Override path to <base>_real.png")
    parser.add_argument("--imag", default=None, help="Override path to <base>_imag.png")
    parser.add_argument("--meta", default=None, help="Override path to <base>_meta.png")
    parser.add_argument("--vtp", default=None, help="Optional reconstructed .vtp output")
    parser.add_argument("--x_spacing", type=float, default=1.0, help="VTP output x spacing")
    parser.add_argument("--y_spacing", type=float, default=1.0, help="VTP output y spacing")
    parser.add_argument("--z_spacing", type=float, default=2.0, help="VTP output layer z spacing")
    parser.add_argument("--decimals", type=int, default=10, help="Significant digits for CSV output")
    args = parser.parse_args()

    paths = infer_paths(args.input_hologram)
    real_path = args.real or paths["real"]
    imag_path = args.imag or paths["imag"]
    meta_path = args.meta or paths["meta"]

    metadata = load_metadata_png(meta_path)
    field = load_complex_field(real_path, imag_path, metadata)
    layers = decode_field_to_layers(field, metadata)
    csv_paths = save_layers_as_csv(layers, args.output_folder, decimals=args.decimals)

    if args.vtp:
        layers_to_ascii_vtp(
            layers,
            args.vtp,
            x_spacing=args.x_spacing,
            y_spacing=args.y_spacing,
            z_spacing=args.z_spacing,
        )

    print("Decoding completed from fixed 2D images only.")
    print(f"Decoded layers: {len(layers)}")
    print(f"CSV files: {', '.join(os.path.basename(p) for p in csv_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
