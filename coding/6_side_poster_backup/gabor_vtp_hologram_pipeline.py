#!/usr/bin/env python3
"""
Gabor-style VTP point-cloud hologram compressor / decompressor.

Pipeline:
  VTP point cloud -> complex hologram image pairs -> reconstructed VTP -> metrics report

The encoded representation is image-based:
  - amplitude_*.png : 16-bit amplitude image
  - phase_*.png     : 16-bit wrapped phase image
  - metadata.json   : required reconstruction metadata

This is an experimental lossy geometry codec inspired by digital holography.
It is not a true physical optical bench simulation, but it keeps the important
ideas for this project:
  1) a 3D object is encoded into 2D complex hologram planes,
  2) both amplitude and phase are saved,
  3) several rotated views can be saved to reveal hidden sides,
  4) decoding back-propagates the complex field through depth slices.

macOS setup:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install numpy scipy pillow vtk scikit-learn

Examples:
  python gabor_vtp_hologram_pipeline.py run-all cube.vtp experiment_cube \
    --holo-size 1024 --depth-bins 96 --views cube6

  python gabor_vtp_hologram_pipeline.py encode input.vtp encoded_dir \
    --holo-size 1024 --depth-bins 96 --views circle8

  python gabor_vtp_hologram_pipeline.py decode encoded_dir reconstructed.vtp

  python gabor_vtp_hologram_pipeline.py report input.vtp reconstructed.vtp encoded_dir/report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

try:
    import vtk
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: vtk\n"
        "Install it with: pip install vtk\n"
    ) from exc


EPS = 1e-12


# -----------------------------------------------------------------------------
# VTP I/O
# -----------------------------------------------------------------------------

def read_vtp_points(path: str | Path) -> np.ndarray:
    """Read point coordinates from a .vtp file as an (N, 3) float64 array."""
    path = str(path)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in VTP file: {path}")

    arr = np.empty((pts.GetNumberOfPoints(), 3), dtype=np.float64)
    for i in range(pts.GetNumberOfPoints()):
        arr[i] = pts.GetPoint(i)
    return arr


def write_vtp_points(path: str | Path, points: np.ndarray) -> None:
    """Write an (N, 3) point cloud to .vtp as vertices."""
    path = str(path)
    points = np.asarray(points, dtype=np.float64)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()

    verts = vtk.vtkCellArray()
    for p in points:
        pid = vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(verts)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    ok = writer.Write()
    if not ok:
        raise IOError(f"Failed to write VTP file: {path}")


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

@dataclass
class ViewMetadata:
    name: str
    rotation_world_to_view: List[List[float]]
    amplitude_png: str
    phase_png: str
    amp_log_min: float
    amp_log_max: float


@dataclass
class CodecMetadata:
    codec_name: str
    version: str
    original_file: str
    original_point_count: int
    original_bounds_min: List[float]
    original_bounds_max: List[float]
    normalization_center: List[float]
    normalization_scale: float
    holo_size: int
    depth_bins: int
    views_mode: str
    wavelength: float
    pixel_pitch: float
    extent: float
    z_base: float
    z_span: float
    splat_sigma: float
    amplitude_bit_depth: int
    phase_bit_depth: int
    views: List[ViewMetadata]


# -----------------------------------------------------------------------------
# Geometry normalization and views
# -----------------------------------------------------------------------------

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Normalize to a centered unit-scale object.

    The largest side of the original bounding box maps approximately to length 2,
    so the normalized object usually fits inside [-1, 1]^3.
    """
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    side_lengths = bounds_max - bounds_min
    scale = float(np.max(side_lengths) / 2.0)
    if scale < EPS:
        scale = 1.0
    normalized = (points - center) / scale
    return normalized, center, scale, bounds_min, bounds_max


def denormalize_points(points_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return points_norm * scale + center


def rotation_from_forward(forward: Sequence[float], up_hint: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Create a world->view rotation matrix.

    Rows are [right, up, forward]. In view coordinates, z is optical depth.
    """
    f = np.asarray(forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + EPS)
    up = np.asarray(up_hint, dtype=np.float64)
    up = up / (np.linalg.norm(up) + EPS)

    # If forward is nearly parallel to up, choose another up vector.
    if abs(float(np.dot(f, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(up, f)
    right = right / (np.linalg.norm(right) + EPS)
    true_up = np.cross(f, right)
    true_up = true_up / (np.linalg.norm(true_up) + EPS)

    return np.vstack([right, true_up, f])


def make_view_rotations(mode: str) -> List[Tuple[str, np.ndarray]]:
    """
    Supported view modes:
      single  : one front view
      cube6   : +Z, -Z, +X, -X, +Y, -Y directions
      circleN : N views around vertical z-axis, e.g. circle8, circle12
    """
    mode = mode.lower().strip()

    if mode == "single":
        return [("front_z", rotation_from_forward((0, 0, 1), (0, 1, 0)))]

    if mode == "cube6":
        directions = [
            ("front_z", (0, 0, 1), (0, 1, 0)),
            ("back_z", (0, 0, -1), (0, 1, 0)),
            ("right_x", (1, 0, 0), (0, 0, 1)),
            ("left_x", (-1, 0, 0), (0, 0, 1)),
            ("top_y", (0, 1, 0), (0, 0, 1)),
            ("bottom_y", (0, -1, 0), (0, 0, 1)),
        ]
        return [(name, rotation_from_forward(direction, up)) for name, direction, up in directions]

    if mode.startswith("circle"):
        try:
            n = int(mode.replace("circle", ""))
        except ValueError:
            raise ValueError("Circle view mode must look like circle8 or circle12")
        if n < 2:
            raise ValueError("circleN requires N >= 2")
        out = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            direction = (math.cos(a), math.sin(a), 0.0)
            out.append((f"circle_{i:02d}", rotation_from_forward(direction, (0, 0, 1))))
        return out

    raise ValueError("Unknown --views mode. Use: single, cube6, circle8, circle12, etc.")


# -----------------------------------------------------------------------------
# Hologram math
# -----------------------------------------------------------------------------

def angular_spectrum_propagate(field: np.ndarray, z: float, wavelength: float, pixel_pitch: float) -> np.ndarray:
    """
    Angular spectrum propagation.

    field: complex field at one plane
    z: propagation distance. Positive = forward, negative = backward.
    wavelength and pixel_pitch are in arbitrary consistent units.
    """
    n = field.shape[0]
    if field.shape[0] != field.shape[1]:
        raise ValueError("Only square hologram arrays are supported")

    fx = np.fft.fftfreq(n, d=pixel_pitch)
    fy = np.fft.fftfreq(n, d=pixel_pitch)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    inside = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    # Evanescent components are suppressed instead of amplified.
    kz = (2.0 * np.pi / wavelength) * np.sqrt(np.maximum(inside, 0.0))
    transfer = np.exp(1j * z * kz)
    transfer[inside < 0.0] = 0.0

    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def points_to_plane(points_xy: np.ndarray, weights: np.ndarray, n: int, extent: float, splat_sigma: float) -> np.ndarray:
    """Rasterize points into a complex object-plane field image."""
    img = np.zeros((n, n), dtype=np.complex128)
    if points_xy.size == 0:
        return img

    # Map x,y in [-extent, extent] to pixel coordinates.
    u = ((points_xy[:, 0] + extent) / (2.0 * extent) * (n - 1)).round().astype(np.int64)
    v = ((points_xy[:, 1] + extent) / (2.0 * extent) * (n - 1)).round().astype(np.int64)
    mask = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    u = u[mask]
    v = v[mask]
    weights = weights[mask]
    np.add.at(img, (v, u), weights)

    if splat_sigma > 0:
        real = gaussian_filter(img.real, splat_sigma, mode="constant")
        imag = gaussian_filter(img.imag, splat_sigma, mode="constant")
        img = real + 1j * imag

    return img


def encode_view_field(
    points_view: np.ndarray,
    n: int,
    depth_bins: int,
    wavelength: float,
    pixel_pitch: float,
    extent: float,
    z_base: float,
    z_span: float,
    splat_sigma: float,
) -> np.ndarray:
    """
    Encode one rotated view into one complex hologram field.

    The normalized view z coordinate [-1, 1] is mapped to physical propagation
    distances [z_base, z_base + z_span].
    """
    z_norm = np.clip(points_view[:, 2], -1.0, 1.0)
    z_phys = z_base + ((z_norm + 1.0) / 2.0) * z_span
    bin_edges = np.linspace(z_base, z_base + z_span, depth_bins + 1)
    bin_ids = np.clip(np.digitize(z_phys, bin_edges) - 1, 0, depth_bins - 1)

    holo = np.zeros((n, n), dtype=np.complex128)

    # Deterministic weak random phase reduces grid artifacts and mimics diffuse object scattering.
    rng = np.random.default_rng(12345)
    random_phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=len(points_view)))

    for b in range(depth_bins):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        z_mid = 0.5 * (bin_edges[b] + bin_edges[b + 1])
        plane = points_to_plane(points_view[idx, :2], random_phase[idx], n, extent, splat_sigma)
        holo += angular_spectrum_propagate(plane, z_mid, wavelength, pixel_pitch)

    return holo


def complex_field_to_pngs(field: np.ndarray, amp_path: Path, phase_path: Path) -> Tuple[float, float]:
    """
    Save complex field as two 16-bit PNGs and return amplitude log min/max.

    amplitude image stores normalized log(1 + abs(field)).
    phase image stores wrapped phase from [-pi, pi] to [0, 65535].
    """
    amp_log = np.log1p(np.abs(field)).astype(np.float64)
    amp_min = float(amp_log.min())
    amp_max = float(amp_log.max())
    amp_norm = (amp_log - amp_min) / (amp_max - amp_min + EPS)
    amp_u16 = np.clip(np.round(amp_norm * 65535.0), 0, 65535).astype(np.uint16)

    phase = np.angle(field)
    phase_norm = (phase + np.pi) / (2.0 * np.pi)
    phase_u16 = np.clip(np.round(phase_norm * 65535.0), 0, 65535).astype(np.uint16)

    Image.fromarray(amp_u16).save(amp_path)
    Image.fromarray(phase_u16).save(phase_path)
    return amp_min, amp_max


def pngs_to_complex_field(amp_path: Path, phase_path: Path, amp_log_min: float, amp_log_max: float) -> np.ndarray:
    amp_u16 = np.array(Image.open(amp_path), dtype=np.float64)
    phase_u16 = np.array(Image.open(phase_path), dtype=np.float64)

    amp_norm = amp_u16 / 65535.0
    amp_log = amp_norm * (amp_log_max - amp_log_min) + amp_log_min
    amp = np.expm1(amp_log)

    phase = (phase_u16 / 65535.0) * (2.0 * np.pi) - np.pi
    return amp * np.exp(1j * phase)


# -----------------------------------------------------------------------------
# Encoding / decoding
# -----------------------------------------------------------------------------

def encode_vtp(args: argparse.Namespace) -> None:
    input_path = Path(args.input_vtp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = read_vtp_points(input_path)
    points_norm, center, scale, bounds_min, bounds_max = normalize_points(points)
    views = make_view_rotations(args.views)

    view_metas: List[ViewMetadata] = []
    for view_name, R in views:
        print(f"[encode] view {view_name}: building complex hologram")
        points_view = points_norm @ R.T
        field = encode_view_field(
            points_view=points_view,
            n=args.holo_size,
            depth_bins=args.depth_bins,
            wavelength=args.wavelength,
            pixel_pitch=args.pixel_pitch,
            extent=args.extent,
            z_base=args.z_base,
            z_span=args.z_span,
            splat_sigma=args.splat_sigma,
        )

        amp_name = f"amplitude_{view_name}.png"
        phase_name = f"phase_{view_name}.png"
        amp_min, amp_max = complex_field_to_pngs(field, out_dir / amp_name, out_dir / phase_name)

        view_metas.append(
            ViewMetadata(
                name=view_name,
                rotation_world_to_view=R.tolist(),
                amplitude_png=amp_name,
                phase_png=phase_name,
                amp_log_min=amp_min,
                amp_log_max=amp_max,
            )
        )

    meta = CodecMetadata(
        codec_name="gabor_style_vtp_point_cloud_hologram",
        version="1.0",
        original_file=str(input_path),
        original_point_count=int(points.shape[0]),
        original_bounds_min=bounds_min.tolist(),
        original_bounds_max=bounds_max.tolist(),
        normalization_center=center.tolist(),
        normalization_scale=float(scale),
        holo_size=int(args.holo_size),
        depth_bins=int(args.depth_bins),
        views_mode=str(args.views),
        wavelength=float(args.wavelength),
        pixel_pitch=float(args.pixel_pitch),
        extent=float(args.extent),
        z_base=float(args.z_base),
        z_span=float(args.z_span),
        splat_sigma=float(args.splat_sigma),
        amplitude_bit_depth=16,
        phase_bit_depth=16,
        views=view_metas,
    )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[encode] saved encoded hologram package to: {out_dir}")


def local_maxima_mask(img: np.ndarray, footprint: int = 3) -> np.ndarray:
    if footprint <= 1:
        return np.ones_like(img, dtype=bool)
    return img == maximum_filter(img, size=footprint, mode="nearest")


def decode_one_view_candidates(
    field: np.ndarray,
    meta: Dict,
    view_meta: Dict,
    target_candidates: int,
    threshold_percentile: float,
    local_maxima: bool,
) -> np.ndarray:
    """
    Back-propagate one hologram through depth slices and extract candidate points.

    Returns normalized coordinates in world orientation, not original scale.
    """
    n = int(meta["holo_size"])
    depth_bins = int(meta["depth_bins"])
    wavelength = float(meta["wavelength"])
    pixel_pitch = float(meta["pixel_pitch"])
    extent = float(meta["extent"])
    z_base = float(meta["z_base"])
    z_span = float(meta["z_span"])

    R = np.asarray(view_meta["rotation_world_to_view"], dtype=np.float64)
    R_inv = R.T

    per_bin = max(1, int(math.ceil(target_candidates / depth_bins)))
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)

    candidates_view: List[np.ndarray] = []

    for b in range(depth_bins):
        frac = (b + 0.5) / depth_bins
        z_phys = z_base + frac * z_span
        z_norm = (frac * 2.0) - 1.0

        recon_plane = angular_spectrum_propagate(field, -z_phys, wavelength, pixel_pitch)
        intensity = np.abs(recon_plane) ** 2

        if local_maxima:
            mask = local_maxima_mask(intensity, footprint=3)
        else:
            mask = np.ones_like(intensity, dtype=bool)

        cutoff = np.percentile(intensity, threshold_percentile)
        mask &= intensity >= cutoff
        valid = np.flatnonzero(mask.ravel())
        if valid.size == 0:
            continue

        vals = intensity.ravel()[valid]
        if valid.size > per_bin:
            chosen_rel = np.argpartition(vals, -per_bin)[-per_bin:]
            chosen = valid[chosen_rel]
        else:
            chosen = valid

        vv, uu = np.unravel_index(chosen, intensity.shape)
        xv = xs[uu]
        yv = ys[vv]
        zv = np.full_like(xv, z_norm, dtype=np.float64)
        pts_view = np.column_stack([xv, yv, zv])
        pts_world_norm = pts_view @ R_inv.T
        candidates_view.append(pts_world_norm)

    if not candidates_view:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(candidates_view)


def voxel_downsample(points: np.ndarray, target_count: int, voxel_size: float) -> np.ndarray:
    """Simple centroid voxel downsampling, then trim/pad to target_count."""
    if points.shape[0] == 0:
        return points

    if voxel_size <= 0:
        out = points
    else:
        keys = np.floor(points / voxel_size).astype(np.int64)
        buckets: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        for key, p in zip(map(tuple, keys), points):
            buckets.setdefault(key, []).append(p)
        out = np.array([np.mean(bucket, axis=0) for bucket in buckets.values()], dtype=np.float64)

    if out.shape[0] > target_count:
        # Deterministic spread: choose evenly along a lexicographic ordering.
        order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
        idx = np.linspace(0, len(order) - 1, target_count).round().astype(int)
        out = out[order[idx]]
    elif out.shape[0] < target_count and out.shape[0] > 0:
        rng = np.random.default_rng(7)
        extra_idx = rng.choice(out.shape[0], size=target_count - out.shape[0], replace=True)
        noise = rng.normal(scale=voxel_size * 0.05 if voxel_size > 0 else 1e-4, size=(len(extra_idx), 3))
        out = np.vstack([out, out[extra_idx] + noise])

    return out


def decode_hologram(args: argparse.Namespace) -> None:
    encoded_dir = Path(args.encoded_dir)
    with open(encoded_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_count = int(args.target_points or meta["original_point_count"])
    decode_oversample = float(args.decode_oversample)
    target_candidates_each_view = max(1, int(math.ceil(target_count * decode_oversample)))

    all_candidates: List[np.ndarray] = []
    for view_meta in meta["views"]:
        print(f"[decode] view {view_meta['name']}: back-propagating through depth")
        field = pngs_to_complex_field(
            encoded_dir / view_meta["amplitude_png"],
            encoded_dir / view_meta["phase_png"],
            float(view_meta["amp_log_min"]),
            float(view_meta["amp_log_max"]),
        )
        cand = decode_one_view_candidates(
            field=field,
            meta=meta,
            view_meta=view_meta,
            target_candidates=target_candidates_each_view,
            threshold_percentile=float(args.threshold_percentile),
            local_maxima=bool(args.local_maxima),
        )
        all_candidates.append(cand)

    if not all_candidates:
        raise RuntimeError("Decoding produced no candidate points. Try lower --threshold-percentile.")

    candidates_norm = np.vstack([c for c in all_candidates if c.shape[0] > 0])

    # Keep points inside a slightly expanded normalized cube. This removes obvious artifacts.
    clip = float(args.clip_extent)
    keep = np.all(np.abs(candidates_norm) <= clip, axis=1)
    candidates_norm = candidates_norm[keep]
    if candidates_norm.shape[0] == 0:
        raise RuntimeError("All candidates were clipped. Increase --clip-extent.")

    # Fuse multi-view candidates.
    voxel_size = float(args.voxel_size)
    if voxel_size <= 0:
        # Reasonable default in normalized coordinates.
        voxel_size = 2.0 / max(32, int(round(meta["holo_size"] / 8)))

    fused_norm = voxel_downsample(candidates_norm, target_count=target_count, voxel_size=voxel_size)

    center = np.asarray(meta["normalization_center"], dtype=np.float64)
    scale = float(meta["normalization_scale"])
    reconstructed = denormalize_points(fused_norm, center, scale)

    output_path = Path(args.output_vtp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtp_points(output_path, reconstructed)

    print(f"[decode] candidates before fusion: {len(candidates_norm)}")
    print(f"[decode] reconstructed points:     {len(reconstructed)}")
    print(f"[decode] saved reconstructed VTP:  {output_path}")


# -----------------------------------------------------------------------------
# Metrics and report
# -----------------------------------------------------------------------------

def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += file_size(Path(root) / name)
    return total


def nearest_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    tree_rec = cKDTree(reconstructed)
    tree_org = cKDTree(original)

    d_org_to_rec, _ = tree_rec.query(original, k=1)
    d_rec_to_org, _ = tree_org.query(reconstructed, k=1)

    bounds_min = original.min(axis=0)
    bounds_max = original.max(axis=0)
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    if diag < EPS:
        diag = 1.0

    chamfer = float(np.mean(d_org_to_rec ** 2) + np.mean(d_rec_to_org ** 2))
    rmse_org_to_rec = float(np.sqrt(np.mean(d_org_to_rec ** 2)))
    mae_org_to_rec = float(np.mean(d_org_to_rec))
    hausdorff_95 = float(max(np.percentile(d_org_to_rec, 95), np.percentile(d_rec_to_org, 95)))
    hausdorff_max = float(max(np.max(d_org_to_rec), np.max(d_rec_to_org)))

    coverage_1 = float(np.mean(d_org_to_rec <= 0.01 * diag) * 100.0)
    coverage_2 = float(np.mean(d_org_to_rec <= 0.02 * diag) * 100.0)
    coverage_5 = float(np.mean(d_org_to_rec <= 0.05 * diag) * 100.0)

    return {
        "bbox_diagonal": diag,
        "chamfer_l2_squared": chamfer,
        "rmse_original_to_reconstructed": rmse_org_to_rec,
        "mae_original_to_reconstructed": mae_org_to_rec,
        "hausdorff_95": hausdorff_95,
        "hausdorff_max": hausdorff_max,
        "normalized_rmse_percent_of_bbox_diag": float((rmse_org_to_rec / diag) * 100.0),
        "coverage_within_1_percent_bbox_diag_percent": coverage_1,
        "coverage_within_2_percent_bbox_diag_percent": coverage_2,
        "coverage_within_5_percent_bbox_diag_percent": coverage_5,
    }


def build_report(args: argparse.Namespace) -> None:
    original_path = Path(args.original_vtp)
    reconstructed_path = Path(args.reconstructed_vtp)
    report_prefix = Path(args.report_prefix)
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    original = read_vtp_points(original_path)
    reconstructed = read_vtp_points(reconstructed_path)
    metrics = nearest_metrics(original, reconstructed)

    encoded_dir = Path(args.encoded_dir) if args.encoded_dir else None
    encoded_size = directory_size(encoded_dir) if encoded_dir and encoded_dir.exists() else None
    original_size = file_size(original_path)
    reconstructed_size = file_size(reconstructed_path)

    out = {
        "files": {
            "original_vtp": str(original_path),
            "reconstructed_vtp": str(reconstructed_path),
            "encoded_dir": str(encoded_dir) if encoded_dir else None,
        },
        "point_counts": {
            "original": int(original.shape[0]),
            "reconstructed": int(reconstructed.shape[0]),
        },
        "file_sizes_bytes": {
            "original_vtp": int(original_size),
            "encoded_dir_total": int(encoded_size) if encoded_size is not None else None,
            "reconstructed_vtp": int(reconstructed_size),
        },
        "compression": {},
        "geometry_metrics": metrics,
    }

    if encoded_size and encoded_size > 0:
        out["compression"] = {
            "compression_ratio_original_vtp_over_encoded": float(original_size / encoded_size),
            "encoded_size_percent_of_original_vtp": float(encoded_size / original_size * 100.0) if original_size > 0 else None,
        }

    json_path = report_prefix.with_suffix(".json")
    txt_path = report_prefix.with_suffix(".txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append("=" * 72)
    lines.append("GABOR-STYLE VTP HOLOGRAM RECONSTRUCTION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("FILES")
    lines.append(f"Original VTP:       {original_path}")
    lines.append(f"Reconstructed VTP:  {reconstructed_path}")
    if encoded_dir:
        lines.append(f"Encoded directory:  {encoded_dir}")
    lines.append("")
    lines.append("POINT COUNTS")
    lines.append(f"Original points:      {original.shape[0]:,}")
    lines.append(f"Reconstructed points: {reconstructed.shape[0]:,}")
    lines.append("")
    lines.append("FILE SIZES")
    lines.append(f"Original VTP size:       {original_size:,} bytes")
    if encoded_size is not None:
        lines.append(f"Encoded total size:      {encoded_size:,} bytes")
    lines.append(f"Reconstructed VTP size:  {reconstructed_size:,} bytes")
    if encoded_size and encoded_size > 0:
        lines.append(f"Compression ratio:       {original_size / encoded_size:.4f}x")
        lines.append(f"Encoded vs original:     {encoded_size / original_size * 100.0:.2f}%")
    lines.append("")
    lines.append("GEOMETRY METRICS")
    lines.append(f"BBox diagonal:                    {metrics['bbox_diagonal']:.10f}")
    lines.append(f"Chamfer L2 squared:               {metrics['chamfer_l2_squared']:.10f}")
    lines.append(f"RMSE original -> reconstructed:   {metrics['rmse_original_to_reconstructed']:.10f}")
    lines.append(f"MAE original -> reconstructed:    {metrics['mae_original_to_reconstructed']:.10f}")
    lines.append(f"Hausdorff 95%:                    {metrics['hausdorff_95']:.10f}")
    lines.append(f"Hausdorff max:                    {metrics['hausdorff_max']:.10f}")
    lines.append(f"Normalized RMSE (% bbox diag):    {metrics['normalized_rmse_percent_of_bbox_diag']:.6f}%")
    lines.append(f"Coverage within 1% bbox diag:     {metrics['coverage_within_1_percent_bbox_diag_percent']:.2f}%")
    lines.append(f"Coverage within 2% bbox diag:     {metrics['coverage_within_2_percent_bbox_diag_percent']:.2f}%")
    lines.append(f"Coverage within 5% bbox diag:     {metrics['coverage_within_5_percent_bbox_diag_percent']:.2f}%")
    lines.append("")
    lines.append("NOTES")
    lines.append("- Point order is not preserved; this compares geometry using nearest neighbors.")
    lines.append("- A single view is usually weak for 3D objects because back-side information is hidden.")
    lines.append("- More views improve reconstruction but increase encoded image size.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[report] saved: {json_path}")
    print(f"[report] saved: {txt_path}")


# -----------------------------------------------------------------------------
# run-all command
# -----------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)
    encoded_dir = experiment_dir / "encoded_hologram"
    reconstructed_vtp = experiment_dir / "reconstructed.vtp"
    report_prefix = experiment_dir / "report" / "metrics"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    encode_args = argparse.Namespace(
        input_vtp=args.input_vtp,
        output_dir=str(encoded_dir),
        holo_size=args.holo_size,
        depth_bins=args.depth_bins,
        views=args.views,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        extent=args.extent,
        z_base=args.z_base,
        z_span=args.z_span,
        splat_sigma=args.splat_sigma,
    )
    encode_vtp(encode_args)

    decode_args = argparse.Namespace(
        encoded_dir=str(encoded_dir),
        output_vtp=str(reconstructed_vtp),
        target_points=None,
        decode_oversample=args.decode_oversample,
        threshold_percentile=args.threshold_percentile,
        local_maxima=args.local_maxima,
        clip_extent=args.clip_extent,
        voxel_size=args.voxel_size,
    )
    decode_hologram(decode_args)

    report_args = argparse.Namespace(
        original_vtp=args.input_vtp,
        reconstructed_vtp=str(reconstructed_vtp),
        encoded_dir=str(encoded_dir),
        report_prefix=str(report_prefix),
    )
    build_report(report_args)

    print(f"\n[run-all] done. Experiment directory: {experiment_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def add_encoding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--holo-size", type=int, default=512, help="Square hologram size in pixels. Use 512, 1024, 2048, etc.")
    parser.add_argument("--depth-bins", type=int, default=64, help="Number of depth slices used in encoding/decoding.")
    parser.add_argument("--views", type=str, default="cube6", help="single, cube6, circle8, circle12, etc.")
    parser.add_argument("--wavelength", type=float, default=0.05, help="Simulation wavelength in arbitrary units.")
    parser.add_argument("--pixel-pitch", type=float, default=0.01, help="Pixel pitch in same arbitrary unit system.")
    parser.add_argument("--extent", type=float, default=1.25, help="Half-width of encoded normalized x/y area.")
    parser.add_argument("--z-base", type=float, default=0.35, help="Nearest propagation distance.")
    parser.add_argument("--z-span", type=float, default=1.50, help="Depth span after z-base.")
    parser.add_argument("--splat-sigma", type=float, default=0.75, help="Gaussian point splat radius in pixels.")


def add_decoding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-points", type=int, default=None, help="Output point count. Default = original point count from metadata.")
    parser.add_argument("--decode-oversample", type=float, default=2.0, help="Candidate multiplier before fusion/downsampling.")
    parser.add_argument("--threshold-percentile", type=float, default=99.5, help="Keep bright reconstruction peaks above this percentile per depth slice.")
    parser.add_argument("--local-maxima", action="store_true", help="Keep only local maxima during peak extraction.")
    parser.add_argument("--clip-extent", type=float, default=1.35, help="Clip normalized candidates outside [-clip, clip]^3.")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel fusion size in normalized coordinates. 0 = automatic.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gabor-style hologram VTP point-cloud codec")
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="Encode VTP into amplitude/phase hologram images + metadata")
    p_encode.add_argument("input_vtp")
    p_encode.add_argument("output_dir")
    add_encoding_options(p_encode)
    p_encode.set_defaults(func=encode_vtp)

    p_decode = sub.add_parser("decode", help="Decode hologram package back to reconstructed VTP")
    p_decode.add_argument("encoded_dir")
    p_decode.add_argument("output_vtp")
    add_decoding_options(p_decode)
    p_decode.set_defaults(func=decode_hologram)

    p_report = sub.add_parser("report", help="Compare original and reconstructed VTP files")
    p_report.add_argument("original_vtp")
    p_report.add_argument("reconstructed_vtp")
    p_report.add_argument("report_prefix", help="Output prefix, e.g. experiment/report/metrics")
    p_report.add_argument("--encoded-dir", default=None, help="Encoded directory for compression size reporting")
    p_report.set_defaults(func=build_report)

    p_run = sub.add_parser("run-all", help="Encode, decode, and report in one command")
    p_run.add_argument("input_vtp")
    p_run.add_argument("experiment_dir")
    add_encoding_options(p_run)
    add_decoding_options(p_run)
    p_run.set_defaults(func=run_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
