#!/usr/bin/env python3
"""
hemisphere_gabor_llm_csv_folder_radial.py

Research prototype for the capstone idea:
folder of CSV arrays -> one packed square matrix -> plane VTP -> hemisphere VTP ->
inline Gabor-style hologram -> approximate reconstruction -> metrics.

This variant implements your requested radial displacement idea:
- each matrix cell has a base location on a perfect hemisphere,
- the normalized value in [-1, 1] determines how far the point is displaced
  along the local radial direction,
- the hologram is generated from these displaced 3D points.

Decoding is approximate: for each base hemisphere location, the code performs a
small radial search and chooses the displacement that best matches the hologram.
This estimates the encoded normalized value, which is then denormalized back to
its original numeric range.

Important: this is still a research prototype, not a production LLM compressor.
A single intensity-only Gabor hologram is lossy and has twin-image ambiguity.
Use the resulting metrics as experimental evidence rather than guaranteed recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------

def safe_name(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "array"


def read_array_file(path: Path) -> List[Tuple[str, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return [(safe_name(path.name), np.asarray(arr, dtype=np.float64))]
    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return [(safe_name(f"{path.stem}_{k}"), np.asarray(z[k], dtype=np.float64)) for k in z.files]
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
        return [(safe_name(path.name), np.asarray(arr, dtype=np.float64))]
    raise ValueError(f"Unsupported input file type: {path}")


def force_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], int(np.prod(arr.shape[1:])))


def _append_loaded_array(
    arrays: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    seen: Dict[str, int],
    name: str,
    arr: np.ndarray,
) -> None:
    base = safe_name(name)
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count:
        base = f"{base}_{count}"
    arrays.append((base, force_2d(arr), tuple(int(x) for x in arr.shape)))


def discover_csv_files(folder: str, recursive: bool = False) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--input-folder must be a directory: {root}")
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(root.glob(pattern), key=lambda p: str(p.relative_to(root)).lower())
    if not files:
        scope = "recursively" if recursive else "directly"
        raise ValueError(f"No .csv files found {scope} inside input folder: {root}")
    return files


def load_csv_folder(folder: str, recursive: bool = False) -> List[Tuple[str, np.ndarray, Tuple[int, ...]]]:
    arrays: List[Tuple[str, np.ndarray, Tuple[int, ...]]] = []
    seen: Dict[str, int] = {}
    for path in discover_csv_files(folder, recursive=recursive):
        for name, arr in read_array_file(path):
            _append_loaded_array(arrays, seen, name, arr)
    if not arrays:
        raise ValueError("No CSV arrays were loaded.")
    return arrays


# -----------------------------------------------------------------------------
# Packing arrays into one square matrix
# -----------------------------------------------------------------------------

@dataclass
class Placement:
    name: str
    original_shape: Tuple[int, ...]
    stored_shape: Tuple[int, int]
    row: int
    col: int
    size: int


def try_shelf_pack(shapes: List[Tuple[int, int, int]], side: int) -> Optional[Dict[int, Tuple[int, int]]]:
    x = 0
    y = 0
    shelf_h = 0
    positions: Dict[int, Tuple[int, int]] = {}
    for idx, h, w in shapes:
        if h > side or w > side:
            return None
        if x + w > side:
            y += shelf_h
            x = 0
            shelf_h = 0
        if y + h > side:
            return None
        positions[idx] = (y, x)
        x += w
        shelf_h = max(shelf_h, h)
    return positions


def pack_arrays_square(
    arrays: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[Placement]]:
    total_area = int(sum(a.size for _, a, _ in arrays))
    max_dim = int(max(max(a.shape) for _, a, _ in arrays))
    side = max(int(math.ceil(math.sqrt(total_area))), max_dim, 1)
    shapes = [(i, int(arr.shape[0]), int(arr.shape[1])) for i, (_, arr, _) in enumerate(arrays)]
    sorted_shapes = sorted(shapes, key=lambda t: (t[1], t[2]), reverse=True)
    positions = None
    while positions is None:
        positions = try_shelf_pack(sorted_shapes, side)
        if positions is None:
            side += max(1, int(math.ceil(side * 0.05)))

    matrix = np.full((side, side), pad_value, dtype=np.float64)
    valid_mask = np.zeros((side, side), dtype=bool)
    placements: List[Placement] = []
    for i, (name, arr, original_shape) in enumerate(arrays):
        r, c = positions[i]
        h, w = arr.shape
        matrix[r:r+h, c:c+w] = arr
        valid_mask[r:r+h, c:c+w] = True
        placements.append(
            Placement(name=name, original_shape=original_shape, stored_shape=(int(h), int(w)), row=int(r), col=int(c), size=int(arr.size))
        )
    return matrix, valid_mask, placements


# -----------------------------------------------------------------------------
# Square -> disk -> hemisphere geometry
# -----------------------------------------------------------------------------

def requested_hologram_size(side: int, min_holo_size: int = 64) -> int:
    return max(int(min_holo_size), int(math.ceil(side / (2.0 * math.sqrt(math.pi)))))


def square_to_concentric_disk(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dx = np.zeros_like(u)
    dy = np.zeros_like(v)
    mask_zero = (np.abs(u) < EPS) & (np.abs(v) < EPS)
    mask = ~mask_zero
    a = u[mask]
    b = v[mask]
    cond = np.abs(a) > np.abs(b)
    r = np.empty_like(a)
    theta = np.empty_like(a)
    r[cond] = a[cond]
    theta[cond] = (math.pi / 4.0) * (b[cond] / (a[cond] + EPS))
    r[~cond] = b[~cond]
    theta[~cond] = (math.pi / 2.0) - (math.pi / 4.0) * (a[~cond] / (b[~cond] + EPS))
    dx[mask] = r * np.cos(theta)
    dy[mask] = r * np.sin(theta)
    return dx, dy


def hemisphere_base_points_from_indices(indices: np.ndarray, side: int, radius: float) -> np.ndarray:
    rows = indices // side
    cols = indices % side
    u = ((cols.astype(np.float64) + 0.5) / side) * 2.0 - 1.0
    v = ((rows.astype(np.float64) + 0.5) / side) * 2.0 - 1.0
    dx, dy = square_to_concentric_disk(u, v)
    dz = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy * dy))
    return np.column_stack([radius * dx, radius * dy, radius * dz])


def radial_unit_vectors(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points / (np.linalg.norm(points, axis=1, keepdims=True) + EPS)


def displace_points_radially(base_points: np.ndarray, signed_values: np.ndarray, radius: float, value_height_scale: float) -> np.ndarray:
    normals = radial_unit_vectors(base_points)
    offsets = (np.asarray(signed_values, dtype=np.float64) * float(value_height_scale) * float(radius))[:, None]
    return base_points + normals * offsets


def normalize_values_signed(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if abs(vmax - vmin) < EPS:
        return np.zeros_like(values, dtype=np.float64)
    return 2.0 * (values.astype(np.float64) - vmin) / (vmax - vmin) - 1.0


def denormalize_values_signed(norm_values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (norm_values + 1.0) * 0.5 * (vmax - vmin) + vmin


# -----------------------------------------------------------------------------
# VTP writing without VTK dependency
# -----------------------------------------------------------------------------

def format_float_list(arr: np.ndarray) -> str:
    return " ".join(f"{float(x):.8g}" for x in np.asarray(arr).reshape(-1))


def format_int_list(arr: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in np.asarray(arr).reshape(-1))


def write_vtp_points(path: Path, points: np.ndarray, point_data: Optional[Dict[str, np.ndarray]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])
    point_data = point_data or {}
    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        if point_data:
            first_name = next(iter(point_data.keys()))
            f.write(f'      <PointData Scalars="{first_name}">\n')
            for name, data in point_data.items():
                data = np.asarray(data)
                dtype = "Float64" if np.issubdtype(data.dtype, np.floating) else "Int32"
                f.write(f'        <DataArray type="{dtype}" Name="{name}" format="ascii">\n')
                f.write("          " + (format_float_list(data) if dtype == "Float64" else format_int_list(data)) + "\n")
                f.write('        </DataArray>\n')
            f.write('      </PointData>\n')
        else:
            f.write('      <PointData/>\n')
        f.write('      <CellData/>\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + format_float_list(points.reshape(-1, 3)) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + format_int_list(connectivity) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + format_int_list(offsets) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


# -----------------------------------------------------------------------------
# Hologram encoding / decoding
# -----------------------------------------------------------------------------

def make_detector_grid(holo_size: int, half_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-half_width, half_width, holo_size, dtype=np.float64)
    ys = np.linspace(-half_width, half_width, holo_size, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, xs, ys


def gabor_inline_hologram(
    points: np.ndarray,
    holo_size: int,
    half_width: float,
    wavelength: float,
    reference_amplitude: float,
    object_strength: float,
    chunk_points: int,
) -> np.ndarray:
    """
    Inline Gabor-style hologram using displaced points.
    Here the numeric value is encoded primarily into geometry (radial displacement),
    so each point uses the same object amplitude.
    """
    points = np.asarray(points, dtype=np.float64)
    k = 2.0 * math.pi / wavelength
    X, Y, _, _ = make_detector_grid(holo_size, half_width)
    Z = np.zeros_like(X)
    obj = np.zeros_like(X, dtype=np.complex128)

    chunk_points = max(1, int(chunk_points))
    for start in range(0, points.shape[0], chunk_points):
        end = min(start + chunk_points, points.shape[0])
        p = points[start:end]
        dx = X[None, :, :] - p[:, 0, None, None]
        dy = Y[None, :, :] - p[:, 1, None, None]
        dz = Z[None, :, :] - p[:, 2, None, None]
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + EPS
        obj += np.sum(object_strength * np.exp(1j * k * r) / r, axis=0)

    ref = reference_amplitude + 0j
    intensity = np.abs(ref + obj) ** 2
    return intensity.astype(np.float64)


def save_hologram_png(path: Path, intensity: np.ndarray) -> Tuple[float, float]:
    imin = float(np.min(intensity))
    imax = float(np.max(intensity))
    if abs(imax - imin) < EPS:
        scaled = np.zeros_like(intensity, dtype=np.uint8)
    else:
        scaled = np.round(255.0 * np.clip((intensity - imin) / (imax - imin), 0.0, 1.0)).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)
    return imin, imax


def load_hologram_from_png(path: Path, imin: float, imax: float) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return gray * (imax - imin) + imin


def matched_filter_response_for_points(
    signal: np.ndarray,
    points: np.ndarray,
    half_width: float,
    wavelength: float,
) -> np.ndarray:
    holo_size = int(signal.shape[0])
    k = 2.0 * math.pi / wavelength
    X, Y, _, _ = make_detector_grid(holo_size, half_width)
    Z = np.zeros_like(X)
    responses = np.zeros(points.shape[0], dtype=np.complex128)
    for i in range(points.shape[0]):
        p = points[i]
        dx = X - p[0]
        dy = Y - p[1]
        dz = Z - p[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + EPS
        responses[i] = np.sum(signal * np.exp(-1j * k * r) / r)
    return responses


def decode_radial_displacements(
    intensity: np.ndarray,
    base_points: np.ndarray,
    radius: float,
    value_height_scale: float,
    half_width: float,
    wavelength: float,
    reference_amplitude: float,
    search_steps: int,
    point_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each base hemisphere location, search along the radial direction and
    choose the displacement that gives the strongest matched-filter response.

    Returns:
      best_alphas: estimated signed normalized values in [-1, 1]
      best_scores: matched-filter response magnitudes for diagnostics
    """
    signal = intensity.astype(np.float64) - reference_amplitude ** 2
    signal = signal - np.mean(signal)

    base_points = np.asarray(base_points, dtype=np.float64)
    normals = radial_unit_vectors(base_points)
    search_steps = max(3, int(search_steps))
    if search_steps % 2 == 0:
        search_steps += 1
    alphas = np.linspace(-1.0, 1.0, search_steps, dtype=np.float64)

    best_alpha = np.zeros(base_points.shape[0], dtype=np.float64)
    best_score = np.full(base_points.shape[0], -np.inf, dtype=np.float64)

    point_chunk_size = max(1, int(point_chunk_size))
    for start in range(0, base_points.shape[0], point_chunk_size):
        end = min(start + point_chunk_size, base_points.shape[0])
        bp = base_points[start:end]
        nn = normals[start:end]

        local_best_score = np.full(bp.shape[0], -np.inf, dtype=np.float64)
        local_best_alpha = np.zeros(bp.shape[0], dtype=np.float64)

        for alpha in alphas:
            candidate_points = bp + nn * (alpha * value_height_scale * radius)
            resp = matched_filter_response_for_points(signal, candidate_points, half_width, wavelength)
            score = np.abs(resp)
            mask = score > local_best_score
            local_best_score[mask] = score[mask]
            local_best_alpha[mask] = alpha

        best_alpha[start:end] = local_best_alpha
        best_score[start:end] = local_best_score

    return best_alpha, best_score


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def array_metrics(original: np.ndarray, decoded: np.ndarray) -> Dict[str, float]:
    original = np.asarray(original, dtype=np.float64).reshape(-1)
    decoded = np.asarray(decoded, dtype=np.float64).reshape(-1)
    finite = np.isfinite(decoded) & np.isfinite(original)
    coverage = float(np.mean(finite)) if original.size else 0.0
    if not np.any(finite):
        return {
            "coverage_percent": 0.0,
            "n_compared": 0,
            "similarity_percent": 0.0,
            "relative_error": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "max_abs_error": float("nan"),
            "cosine_similarity": float("nan"),
            "psnr_db": float("nan"),
        }
    a = original[finite]
    b = decoded[finite]
    diff = b - a
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(a) + EPS))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    cos = float(np.dot(a, b) / ((np.linalg.norm(a) + EPS) * (np.linalg.norm(b) + EPS)))
    similarity = float(max(0.0, (1.0 - rel) * 100.0))
    data_range = float(np.max(a) - np.min(a))
    psnr = float("inf") if rmse < EPS else float(20.0 * math.log10((data_range + EPS) / rmse))
    return {
        "coverage_percent": 100.0 * coverage,
        "n_compared": int(a.size),
        "similarity_percent": similarity,
        "relative_error": rel,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "cosine_similarity": cos,
        "psnr_db": psnr,
    }


def unpack_arrays_from_matrix(matrix: np.ndarray, placements: List[Placement]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for p in placements:
        h, w = p.stored_shape
        block = matrix[p.row:p.row+h, p.col:p.col+w]
        try:
            arr = block.reshape(p.original_shape)
        except ValueError:
            arr = block.copy()
        out[p.name] = arr
    return out


def save_array_outputs(out_dir: Path, arrays: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
        if arr.ndim <= 2:
            np.savetxt(out_dir / f"{name}.csv", np.asarray(arr), delimiter=",")


# -----------------------------------------------------------------------------
# Pipeline commands
# -----------------------------------------------------------------------------

def metadata_path(out: Path) -> Path:
    return out / "metadata.json"


def load_metadata(out: Path) -> Dict:
    with metadata_path(out).open("r", encoding="utf-8") as f:
        return json.load(f)


def placements_from_metadata(meta: Dict) -> List[Placement]:
    return [Placement(name=p["name"], original_shape=tuple(p["original_shape"]), stored_shape=tuple(p["stored_shape"]), row=int(p["row"]), col=int(p["col"]), size=int(p["size"])) for p in meta["placements"]]


def command_encode(args: argparse.Namespace) -> None:
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arrays = load_csv_folder(args.input_folder, recursive=bool(getattr(args, "recursive_csv", False)))
    input_files = [str(p) for p in discover_csv_files(args.input_folder, recursive=bool(getattr(args, "recursive_csv", False)))]

    packed, valid_mask, placements = pack_arrays_square(arrays, pad_value=args.pad_value)
    side = int(packed.shape[0])
    valid_linear = np.flatnonzero(valid_mask.reshape(-1))
    valid_values = packed.reshape(-1)[valid_linear]
    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))

    holo_size = requested_hologram_size(side, args.min_holo_size) if args.holo_size == "auto" else int(args.holo_size)
    radius = float(side / (4.0 * math.sqrt(math.pi)))
    half_width = radius

    rng = np.random.default_rng(int(args.seed))
    if int(args.max_points) > 0 and valid_linear.size > int(args.max_points):
        chosen = rng.choice(valid_linear, size=int(args.max_points), replace=False)
        chosen.sort()
    else:
        chosen = valid_linear

    chosen_values = packed.reshape(-1)[chosen]
    signed_values = normalize_values_signed(chosen_values, vmin, vmax)
    base_points = hemisphere_base_points_from_indices(chosen, side, radius)
    points = displace_points_radially(base_points, signed_values, radius, float(args.value_height_scale))

    if args.save_vtp:
        rows = chosen // side
        cols = chosen % side
        plane_points = np.column_stack([rows.astype(np.float64), cols.astype(np.float64), chosen_values + 1.0])
        write_vtp_points(out / "plane_points_value_plus_1.vtp", plane_points, point_data={"matrix_value": chosen_values, "linear_index": chosen.astype(np.int32)})
        write_vtp_points(
            out / "hemisphere_points_radial.vtp",
            points,
            point_data={
                "matrix_value": chosen_values,
                "signed_value": signed_values,
                "linear_index": chosen.astype(np.int32),
            },
        )
        write_vtp_points(
            out / "hemisphere_base_points.vtp",
            base_points,
            point_data={
                "matrix_value": chosen_values,
                "signed_value": signed_values,
                "linear_index": chosen.astype(np.int32),
            },
        )

    intensity = gabor_inline_hologram(
        points=points,
        holo_size=holo_size,
        half_width=half_width,
        wavelength=float(args.wavelength),
        reference_amplitude=float(args.reference_amplitude),
        object_strength=float(args.object_strength),
        chunk_points=int(args.chunk_points),
    )

    hmin, hmax = save_hologram_png(out / "gabor_hologram.png", intensity)
    if args.save_float_hologram:
        np.save(out / "gabor_hologram_float.npy", intensity)

    np.save(out / "packed_matrix.npy", packed)
    np.save(out / "valid_mask.npy", valid_mask)
    np.save(out / "encoded_indices.npy", chosen.astype(np.int64))

    meta = {
        "version": "0.2-radial",
        "pipeline": "CSV folder -> shelf-packed square -> plane VTP -> radially displaced hemisphere -> inline Gabor hologram",
        "input_folder": str(Path(args.input_folder)),
        "input_files": input_files,
        "side": side,
        "hologram_size": holo_size,
        "hologram_size_rule": "ceil(side / (2*sqrt(pi))) when --holo-size auto",
        "hemisphere_radius_matrix_units": radius,
        "detector_half_width_matrix_units": half_width,
        "wavelength": float(args.wavelength),
        "reference_amplitude": float(args.reference_amplitude),
        "object_strength": float(args.object_strength),
        "pad_value": float(args.pad_value),
        "value_min": vmin,
        "value_max": vmax,
        "value_height_scale": float(args.value_height_scale),
        "radial_search_steps": int(args.radial_search_steps),
        "total_valid_values": int(valid_linear.size),
        "encoded_values": int(chosen.size),
        "encoded_coverage_percent": float(100.0 * chosen.size / max(valid_linear.size, 1)),
        "hologram_png_min": hmin,
        "hologram_png_max": hmax,
        "placements": [asdict(p) for p in placements],
        "notes": [
            "Plane VTP uses x=row, y=column, z=value+1.",
            "Hemisphere encoding uses base hemisphere coordinates from row/column and radial displacement from signed normalized values.",
            "The hologram is generated from the displaced points themselves, not only from a visualized offset.",
            "Decoding estimates values by radial search along each hemisphere normal.",
            "A single intensity-only inline Gabor hologram remains lossy and ambiguous; results are experimental.",
        ],
    }
    with metadata_path(out).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    print(f"[encode] packed square: {side} x {side}")
    print(f"[encode] valid values: {valid_linear.size:,}; encoded values: {chosen.size:,} ({meta['encoded_coverage_percent']:.2f}%)")
    print(f"[encode] hologram: {holo_size} x {holo_size} -> {out / 'gabor_hologram.png'}")
    print(f"[encode] value_height_scale: {float(args.value_height_scale):.6g}")
    print(f"[encode] metadata: {metadata_path(out)}")
    print(f"[encode] done in {dt:.2f}s")


def command_decode(args: argparse.Namespace) -> None:
    t0 = time.time()
    out = Path(args.out)
    meta = load_metadata(out)
    side = int(meta["side"])
    radius = float(meta["hemisphere_radius_matrix_units"])
    half_width = float(meta["detector_half_width_matrix_units"])
    encoded_indices = np.load(out / "encoded_indices.npy")

    if bool(args.use_float_hologram) and (out / "gabor_hologram_float.npy").exists():
        intensity = np.load(out / "gabor_hologram_float.npy")
    else:
        intensity = load_hologram_from_png(out / "gabor_hologram.png", float(meta["hologram_png_min"]), float(meta["hologram_png_max"]))

    base_points = hemisphere_base_points_from_indices(encoded_indices, side, radius)
    signed_est, radial_scores = decode_radial_displacements(
        intensity=intensity,
        base_points=base_points,
        radius=radius,
        value_height_scale=float(meta["value_height_scale"]),
        half_width=half_width,
        wavelength=float(meta["wavelength"]),
        reference_amplitude=float(meta["reference_amplitude"]),
        search_steps=int(args.radial_search_steps) if int(args.radial_search_steps) > 0 else int(meta.get("radial_search_steps", 33)),
        point_chunk_size=int(args.chunk_points),
    )
    decoded_values = denormalize_values_signed(signed_est, float(meta["value_min"]), float(meta["value_max"]))

    decoded_matrix = np.full((side, side), np.nan, dtype=np.float64)
    decoded_matrix.reshape(-1)[encoded_indices] = decoded_values
    np.save(out / "decoded_matrix.npy", decoded_matrix)
    np.save(out / "decoded_signed_values.npy", signed_est)
    np.save(out / "decode_radial_scores.npy", radial_scores)

    placements = placements_from_metadata(meta)
    decoded_arrays = unpack_arrays_from_matrix(decoded_matrix, placements)
    save_array_outputs(out / "decoded_arrays", decoded_arrays)

    dt = time.time() - t0
    print(f"[decode] decoded values: {encoded_indices.size:,}")
    print(f"[decode] radial search steps: {int(args.radial_search_steps) if int(args.radial_search_steps) > 0 else int(meta.get('radial_search_steps', 33))}")
    print(f"[decode] decoded matrix: {out / 'decoded_matrix.npy'}")
    print(f"[decode] decoded arrays: {out / 'decoded_arrays'}")
    print(f"[decode] done in {dt:.2f}s")


def command_metrics(args: argparse.Namespace) -> None:
    out = Path(args.out)
    meta = load_metadata(out)
    packed = np.load(out / "packed_matrix.npy")
    decoded = np.load(out / "decoded_matrix.npy")
    placements = placements_from_metadata(meta)
    original_arrays = unpack_arrays_from_matrix(packed, placements)
    decoded_arrays = unpack_arrays_from_matrix(decoded, placements)

    results: Dict[str, Dict[str, float]] = {}
    results["__packed_matrix__"] = array_metrics(packed, decoded)
    for name in original_arrays:
        results[name] = array_metrics(original_arrays[name], decoded_arrays[name])

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["array"] + list(next(iter(results.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, vals in results.items():
            row = {"array": name}
            row.update(vals)
            writer.writerow(row)

    print(f"[metrics] wrote {out / 'metrics.json'} and {out / 'metrics.csv'}")
    for name, vals in results.items():
        print(
            f"  {name}: coverage={vals['coverage_percent']:.2f}% | "
            f"similarity={vals['similarity_percent']:.4f}% | "
            f"rel_err={vals['relative_error']:.6g} | "
            f"rmse={vals['rmse']:.6g} | "
            f"cos={vals['cosine_similarity']:.6g}"
        )


def command_run_all(args: argparse.Namespace) -> None:
    command_encode(args)
    decode_args = argparse.Namespace(
        out=args.out,
        chunk_points=args.decode_chunk_points,
        use_float_hologram=args.use_float_hologram,
        radial_search_steps=args.radial_search_steps,
    )
    command_decode(decode_args)
    command_metrics(argparse.Namespace(out=args.out))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def add_common_encode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-folder", required=True, help="Folder containing CSV files. All *.csv files are loaded and sorted by filename")
    p.add_argument("--recursive-csv", action="store_true", help="Also read CSV files in subfolders")
    p.add_argument("--out", required=True, help="Output experiment directory")
    p.add_argument("--holo-size", default="auto", help="Hologram side in pixels, or 'auto' for side/(2*sqrt(pi))")
    p.add_argument("--min-holo-size", type=int, default=64, help="Minimum auto hologram side")
    p.add_argument("--max-points", type=int, default=0, help="Max matrix points to encode; 0 means all valid points")
    p.add_argument("--seed", type=int, default=0, help="Random seed for point sampling")
    p.add_argument("--wavelength", type=float, default=1.0, help="Simulation wavelength in matrix units")
    p.add_argument("--reference-amplitude", type=float, default=1.0, help="Reference wave amplitude")
    p.add_argument("--object-strength", type=float, default=0.02, help="Object wave strength")
    p.add_argument("--chunk-points", type=int, default=128, help="Point chunk size for encoding")
    p.add_argument("--pad-value", type=float, default=0.0, help="Padding value in packed square")
    p.add_argument("--value-height-scale", type=float, default=0.15, help="Radial displacement scale as a fraction of hemisphere radius")
    p.add_argument("--radial-search-steps", type=int, default=33, help="Number of radial candidates in decode search; odd values are best")
    p.add_argument("--save-float-hologram", action="store_true", help="Also save raw float hologram .npy")
    p.add_argument("--no-vtp", dest="save_vtp", action="store_false", help="Skip VTP outputs")
    p.set_defaults(save_vtp=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hemisphere + radial displacement + Gabor hologram prototype")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode", help="Pack arrays, write VTPs, and encode hologram")
    add_common_encode_args(p_enc)
    p_enc.set_defaults(func=command_encode)

    p_dec = sub.add_parser("decode", help="Decode hologram back to approximate arrays")
    p_dec.add_argument("--out", required=True, help="Experiment directory created by encode")
    p_dec.add_argument("--chunk-points", type=int, default=32, help="Number of points processed together in decode")
    p_dec.add_argument("--radial-search-steps", type=int, default=0, help="Override radial search steps from metadata; 0 means use metadata")
    p_dec.add_argument("--use-float-hologram", action="store_true", help="Decode from raw float .npy if saved")
    p_dec.set_defaults(func=command_decode)

    p_met = sub.add_parser("metrics", help="Compare original packed arrays with decoded arrays")
    p_met.add_argument("--out", required=True, help="Experiment directory")
    p_met.set_defaults(func=command_metrics)

    p_all = sub.add_parser("run-all", help="Run encode, decode, and metrics in one command")
    add_common_encode_args(p_all)
    p_all.add_argument("--decode-chunk-points", type=int, default=32, help="Number of points processed together in decode")
    p_all.add_argument("--use-float-hologram", action="store_true", help="Decode from raw float .npy if --save-float-hologram was used")
    p_all.set_defaults(func=command_run_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
