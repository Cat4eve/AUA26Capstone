#!/usr/bin/env python3
"""
VTP point cloud -> fixed 2D hologram images.

Important design rule for the capstone idea:
The decoder does NOT receive the original matrix layers, original point cloud, or a
3D sidecar. The encoded object is a fixed set of 2D images:

1. <base>_real.png : 16-bit real part of a complex 2D hologram field
2. <base>_imag.png : 16-bit imaginary part of the same 2D field
3. <base>_meta.png : small 2D image containing only decoding metadata

The metadata image stores shapes, normalization ranges, and Fourier tile positions.
It does not store the matrix values. Matrix values are encoded in the 2D complex
field through Fourier-domain multiplexing.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

UINT16_MAX = 65535


def _numbers(text: str, dtype=float) -> np.ndarray:
    if text is None or not text.strip():
        return np.array([], dtype=dtype)
    return np.fromstring(text, sep=" ", dtype=dtype)


def read_ascii_vtp(vtp_path: str) -> Dict[str, object]:
    """Read the ASCII VTP format written by matrix_to_vtp.py."""
    tree = ET.parse(vtp_path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("Invalid VTP: missing Piece element")

    polydata = root.find(".//PolyData")
    attrs = polydata.attrib if polydata is not None else {}

    arrays = {}
    for data_array in root.findall(".//PointData/DataArray"):
        name = data_array.attrib.get("Name")
        if not name:
            continue
        typ = data_array.attrib.get("type", "Float64")
        dtype = int if typ.lower().startswith("int") else float
        arrays[name] = _numbers(data_array.text or "", dtype=dtype)

    points_array = root.find(".//Points/DataArray")
    if points_array is None:
        raise ValueError("Invalid VTP: missing Points/DataArray")
    points_flat = _numbers(points_array.text or "", dtype=float)
    if points_flat.size % 3 != 0:
        raise ValueError("Invalid VTP: point coordinate count is not divisible by 3")
    points = points_flat.reshape((-1, 3))

    layer_names = []
    if attrs.get("layer_names"):
        layer_names = [html.unescape(x) for x in attrs["layer_names"].split("|") if x]

    layer_shapes = []
    if attrs.get("layer_shapes"):
        for item in attrs["layer_shapes"].split("|"):
            if not item:
                continue
            rows, cols = item.split(",")
            layer_shapes.append((int(rows), int(cols)))

    return {
        "points": points,
        "arrays": arrays,
        "x_spacing": float(attrs.get("x_spacing", 1.0)),
        "y_spacing": float(attrs.get("y_spacing", 1.0)),
        "z_spacing": float(attrs.get("z_spacing", 2.0)),
        "layer_names": layer_names,
        "layer_shapes": layer_shapes,
    }


def vtp_to_matrices(vtp: Dict[str, object]) -> Tuple[List[np.ndarray], List[str]]:
    """Rebuild ordered matrix layers from VTP point data arrays."""
    arrays = vtp["arrays"]
    required = ["original_value", "layer_index", "row_index", "col_index"]
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(
            "This VTP does not contain the arrays needed for layered encoding: "
            + ", ".join(missing)
            + ". Recreate it with the fixed matrix_to_vtp.py."
        )

    values = np.asarray(arrays["original_value"], dtype=np.float64)
    layer_index = np.asarray(arrays["layer_index"], dtype=np.int64)
    row_index = np.asarray(arrays["row_index"], dtype=np.int64)
    col_index = np.asarray(arrays["col_index"], dtype=np.int64)

    if not (len(values) == len(layer_index) == len(row_index) == len(col_index)):
        raise ValueError("VTP point-data arrays have inconsistent lengths")

    layer_count = int(layer_index.max()) + 1 if len(layer_index) else 0
    if layer_count <= 0:
        raise ValueError("No layers found in VTP")

    matrices: List[np.ndarray] = []
    for layer in range(layer_count):
        mask = layer_index == layer
        if not np.any(mask):
            raise ValueError(f"Layer {layer} has no points")
        rows = int(row_index[mask].max()) + 1
        cols = int(col_index[mask].max()) + 1
        mat = np.zeros((rows, cols), dtype=np.float64)
        mat[row_index[mask], col_index[mask]] = values[mask]
        matrices.append(mat)

    names = list(vtp.get("layer_names") or [])
    if len(names) != layer_count:
        names = [f"layer_{i}" for i in range(layer_count)]
    return matrices, names


def choose_tile_grid(layer_count: int) -> Tuple[int, int]:
    """Choose a near-square grid for frequency tiles."""
    grid_cols = int(math.ceil(math.sqrt(layer_count)))
    grid_rows = int(math.ceil(layer_count / grid_cols))
    return grid_rows, grid_cols


def normalize_layer(layer: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize one matrix to [0, 1], returning normalized layer and min/max."""
    mn = float(np.min(layer))
    mx = float(np.max(layer))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError("Layer contains NaN or infinite values")
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(layer, dtype=np.float64), mn, mx
    return (layer - mn) / (mx - mn), mn, mx


def encode_layers_to_field(matrices: List[np.ndarray], names: List[str], output_base: str) -> Dict[str, object]:
    """
    Encode all layers into one complex 2D field using Fourier-domain multiplexing.

    Each layer is transformed with FFT and placed into a non-overlapping tile in a
    larger 2D spectrum. The 2D hologram field is the inverse FFT of that spectrum.
    The decoder FFTs the 2D field, extracts each tile, and inverse-FFTs the tile.
    """
    if not matrices:
        raise ValueError("No matrices to encode")

    tile_rows = max(int(m.shape[0]) for m in matrices)
    tile_cols = max(int(m.shape[1]) for m in matrices)
    layer_count = len(matrices)
    grid_rows, grid_cols = choose_tile_grid(layer_count)

    hologram_rows = tile_rows * grid_rows
    hologram_cols = tile_cols * grid_cols
    spectrum = np.zeros((hologram_rows, hologram_cols), dtype=np.complex128)

    layers_meta = []
    for layer_id, matrix in enumerate(matrices):
        norm_layer, value_min, value_max = normalize_layer(matrix)
        rows, cols = norm_layer.shape
        layer_fft = np.fft.fftshift(np.fft.fft2(norm_layer, norm="ortho"))

        tile_r = layer_id // grid_cols
        tile_c = layer_id % grid_cols
        r0 = tile_r * tile_rows
        c0 = tile_c * tile_cols
        spectrum[r0:r0 + rows, c0:c0 + cols] = layer_fft

        layers_meta.append({
            "name": names[layer_id] if layer_id < len(names) else f"layer_{layer_id}",
            "rows": rows,
            "cols": cols,
            "value_min": value_min,
            "value_max": value_max,
            "tile_r0": r0,
            "tile_c0": c0,
        })

    field = np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho")
    scale = float(max(np.max(np.abs(field.real)), np.max(np.abs(field.imag)), 1e-12))
    field_scaled = field / scale

    metadata = {
        "format": "2d_complex_fourier_hologram_v1",
        "description": "Matrix layers are stored only in a fixed set of 2D images: real, imag, and metadata.",
        "hologram_rows": hologram_rows,
        "hologram_cols": hologram_cols,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "field_scale": scale,
        "layers": layers_meta,
        "used_images": {
            "real": f"{output_base}_real.png",
            "imag": f"{output_base}_imag.png",
            "metadata": f"{output_base}_meta.png",
        },
    }
    return {"field_scaled": field_scaled, "metadata": metadata}


def complex_array_to_uint16(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map real/imag values in [-1, 1] to uint16 images."""
    real = np.clip(arr.real, -1.0, 1.0)
    imag = np.clip(arr.imag, -1.0, 1.0)
    real_u16 = np.rint((real + 1.0) * (UINT16_MAX / 2.0)).astype(np.uint16)
    imag_u16 = np.rint((imag + 1.0) * (UINT16_MAX / 2.0)).astype(np.uint16)
    return real_u16, imag_u16


def save_uint16_png(path: str, array: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint16)).save(path)


def save_phase_preview(path: str, field_scaled: np.ndarray) -> None:
    phase = np.angle(field_scaled)
    preview = np.rint((phase + np.pi) / (2.0 * np.pi) * 255.0).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)


def save_metadata_png(path: str, metadata: Dict[str, object]) -> None:
    """Store UTF-8 JSON metadata bytes in a small grayscale 2D PNG."""
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    length = len(raw).to_bytes(4, byteorder="big", signed=False)
    payload = np.frombuffer(length + raw, dtype=np.uint8)
    width = 1024
    height = int(math.ceil(payload.size / width))
    padded = np.zeros(width * height, dtype=np.uint8)
    padded[:payload.size] = payload
    image = padded.reshape((height, width))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def output_paths(output_preview: str) -> Dict[str, str]:
    path = Path(output_preview)
    base = str(path.with_suffix(""))
    return {
        "base": base,
        "preview": output_preview,
        "real": f"{base}_real.png",
        "imag": f"{base}_imag.png",
        "meta": f"{base}_meta.png",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode VTP matrix layers into fixed 2D hologram images."
    )
    parser.add_argument("input_vtp", help="Input .vtp produced by matrix_to_vtp.py")
    parser.add_argument("output_hologram", help="Output phase-preview image path, e.g. hologram.png")
    parser.add_argument("--method", choices=["fourier"], default="fourier", help="2D encoding method")
    parser.add_argument("--write_npz", action="store_true", help="Optional debug export of field; not needed for decoding")
    args = parser.parse_args()

    if not os.path.exists(args.input_vtp):
        raise FileNotFoundError(f"Input VTP file not found: {args.input_vtp}")

    paths = output_paths(args.output_hologram)
    vtp = read_ascii_vtp(args.input_vtp)
    matrices, names = vtp_to_matrices(vtp)

    encoded = encode_layers_to_field(matrices, names, paths["base"])
    field_scaled = encoded["field_scaled"]
    metadata = encoded["metadata"]
    real_u16, imag_u16 = complex_array_to_uint16(field_scaled)

    save_uint16_png(paths["real"], real_u16)
    save_uint16_png(paths["imag"], imag_u16)
    save_metadata_png(paths["meta"], metadata)
    save_phase_preview(paths["preview"], field_scaled)

    if args.write_npz:
        np.savez_compressed(f"{paths['base']}_debug_field.npz", field_scaled=field_scaled)

    print("Saved 2D hologram encoding:")
    print(f"  phase preview: {paths['preview']}")
    print(f"  real image:    {paths['real']}")
    print(f"  imag image:    {paths['imag']}")
    print(f"  metadata img:  {paths['meta']}")
    print(f"Layers encoded: {len(matrices)}")
    print(f"Hologram field shape: {field_scaled.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
