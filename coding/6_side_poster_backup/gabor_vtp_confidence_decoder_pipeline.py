#!/usr/bin/env python3
"""
Gabor-style VTP hologram codec with improved confidence-volume decoding.

This script keeps the same storage idea as the previous version:
  VTP point cloud -> amplitude PNG + phase PNG per view + metadata.json

The main change is decoding:
  OLD decoder: back-propagate each view, pick bright peaks, directly output points.
  NEW decoder: back-propagate each view, vote into a shared 3D confidence volume,
               require multi-view support, remove weak/isolated evidence, then extract only clean high-confidence voxels into a reconstructed point cloud.

By default, this version is quality-first: it does NOT force the reconstructed
point cloud to have exactly the same point count as the original. If the decoder
finds fewer reliable points, it outputs fewer points instead of inventing noisy
ones. Exact point-count matching can still be enabled with --force-target-points.

No AI is used.

macOS setup:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install numpy scipy pillow vtk

Recommended run:
  python gabor_vtp_confidence_decoder_pipeline.py run-all cube_dense_large_500.vtp experiment_confidence \
    --holo-size 1024 \
    --depth-bins 96 \
    --views cube6 \
    --volume-resolution 256 \
    --min-view-support 2 \
    --support-radius 1 \
    --decode-oversample 1.2 \
    --threshold-percentile 99.2 \
    --connected-filter

Optional exact-count mode, not recommended for quality-first tests:
  add --force-target-points

Faster run:
  python gabor_vtp_confidence_decoder_pipeline.py run-all cube_dense_large_500.vtp experiment_confidence_fast \
    --holo-size 512 \
    --depth-bins 64 \
    --views cube6 \
    --volume-resolution 192 \
    --min-view-support 2 \
    --support-radius 1 \
    --decode-oversample 1.0 \
    --threshold-percentile 99.1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.spatial import cKDTree

try:
    import vtk
except ImportError as exc:
    raise SystemExit("Missing dependency: vtk. Install with: pip install vtk") from exc

EPS = 1e-12


# -----------------------------------------------------------------------------
# VTP I/O
# -----------------------------------------------------------------------------

def read_vtp_points(path: str | Path) -> np.ndarray:
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
    writer.EncodeAppendedDataOn()
    ok = writer.Write()
    if not ok:
        raise IOError(f"Could not write VTP: {path}")


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
# Normalization and views
# -----------------------------------------------------------------------------

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    bmin = points.min(axis=0)
    bmax = points.max(axis=0)
    center = (bmin + bmax) / 2.0
    scale = float(np.max(bmax - bmin) / 2.0)
    if scale < EPS:
        scale = 1.0
    return (points - center) / scale, center, scale, bmin, bmax


def denormalize_points(points_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return points_norm * scale + center


def rotation_from_forward(forward: Sequence[float], up_hint: Sequence[float]) -> np.ndarray:
    f = np.asarray(forward, dtype=np.float64)
    f /= np.linalg.norm(f) + EPS
    up = np.asarray(up_hint, dtype=np.float64)
    up /= np.linalg.norm(up) + EPS

    if abs(float(np.dot(f, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(up, f)
    right /= np.linalg.norm(right) + EPS
    true_up = np.cross(f, right)
    true_up /= np.linalg.norm(true_up) + EPS

    # Rows are view axes: [right, up, forward].
    # p_view = p_world @ R.T
    return np.vstack([right, true_up, f])


def make_view_rotations(mode: str) -> List[Tuple[str, np.ndarray]]:
    mode = mode.lower().strip()
    if mode == "single":
        return [("front_z", rotation_from_forward((0, 0, 1), (0, 1, 0)))]

    if mode == "cube6":
        items = [
            ("front_z", (0, 0, 1), (0, 1, 0)),
            ("back_z", (0, 0, -1), (0, 1, 0)),
            ("right_x", (1, 0, 0), (0, 0, 1)),
            ("left_x", (-1, 0, 0), (0, 0, 1)),
            ("top_y", (0, 1, 0), (0, 0, 1)),
            ("bottom_y", (0, -1, 0), (0, 0, 1)),
        ]
        return [(name, rotation_from_forward(direction, up)) for name, direction, up in items]

    if mode.startswith("circle"):
        try:
            n = int(mode.replace("circle", ""))
        except ValueError:
            raise ValueError("Circle mode must look like circle8, circle12, etc.")
        if n < 2:
            raise ValueError("circleN requires N >= 2")
        out = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            out.append((f"circle_{i:02d}", rotation_from_forward((math.cos(a), math.sin(a), 0), (0, 0, 1))))
        return out

    raise ValueError("Unknown --views. Use single, cube6, circle8, circle12, etc.")


# -----------------------------------------------------------------------------
# Wave propagation and image storage
# -----------------------------------------------------------------------------

def angular_spectrum_propagate(field: np.ndarray, z: float, wavelength: float, pixel_pitch: float) -> np.ndarray:
    n = field.shape[0]
    fx = np.fft.fftfreq(n, d=pixel_pitch)
    fy = np.fft.fftfreq(n, d=pixel_pitch)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    inside = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    kz = (2.0 * np.pi / wavelength) * np.sqrt(np.maximum(inside, 0.0))
    transfer = np.exp(1j * z * kz)
    transfer[inside < 0] = 0.0
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def points_to_plane(points_xy: np.ndarray, weights: np.ndarray, n: int, extent: float, splat_sigma: float) -> np.ndarray:
    img = np.zeros((n, n), dtype=np.complex128)
    if len(points_xy) == 0:
        return img

    u = np.round((points_xy[:, 0] + extent) / (2.0 * extent) * (n - 1)).astype(np.int64)
    v = np.round((points_xy[:, 1] + extent) / (2.0 * extent) * (n - 1)).astype(np.int64)
    mask = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    np.add.at(img, (v[mask], u[mask]), weights[mask])

    if splat_sigma > 0:
        img = gaussian_filter(img.real, splat_sigma, mode="constant") + 1j * gaussian_filter(img.imag, splat_sigma, mode="constant")
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
    z_norm = np.clip(points_view[:, 2], -1.0, 1.0)
    z_phys = z_base + ((z_norm + 1.0) / 2.0) * z_span
    edges = np.linspace(z_base, z_base + z_span, depth_bins + 1)
    bins = np.clip(np.digitize(z_phys, edges) - 1, 0, depth_bins - 1)

    rng = np.random.default_rng(12345)
    random_phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=len(points_view)))
    holo = np.zeros((n, n), dtype=np.complex128)

    for b in range(depth_bins):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        z_mid = 0.5 * (edges[b] + edges[b + 1])
        plane = points_to_plane(points_view[idx, :2], random_phase[idx], n, extent, splat_sigma)
        holo += angular_spectrum_propagate(plane, z_mid, wavelength, pixel_pitch)

    return holo


def complex_field_to_pngs(field: np.ndarray, amp_path: Path, phase_path: Path) -> Tuple[float, float]:
    amp_log = np.log1p(np.abs(field)).astype(np.float64)
    amin = float(amp_log.min())
    amax = float(amp_log.max())
    amp_norm = (amp_log - amin) / (amax - amin + EPS)
    amp_u16 = np.clip(np.round(amp_norm * 65535.0), 0, 65535).astype(np.uint16)

    phase_norm = (np.angle(field) + np.pi) / (2.0 * np.pi)
    phase_u16 = np.clip(np.round(phase_norm * 65535.0), 0, 65535).astype(np.uint16)

    Image.fromarray(amp_u16).save(amp_path)
    Image.fromarray(phase_u16).save(phase_path)
    return amin, amax


def pngs_to_complex_field(amp_path: Path, phase_path: Path, amp_log_min: float, amp_log_max: float) -> np.ndarray:
    amp_u16 = np.asarray(Image.open(amp_path), dtype=np.float64)
    phase_u16 = np.asarray(Image.open(phase_path), dtype=np.float64)

    amp_norm = amp_u16 / 65535.0
    amp_log = amp_norm * (amp_log_max - amp_log_min) + amp_log_min
    amp = np.expm1(amp_log)
    phase = (phase_u16 / 65535.0) * (2.0 * np.pi) - np.pi
    return amp * np.exp(1j * phase)


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

def encode_vtp(args: argparse.Namespace) -> None:
    input_path = Path(args.input_vtp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = read_vtp_points(input_path)
    points_norm, center, scale, bmin, bmax = normalize_points(points)
    view_defs = make_view_rotations(args.views)
    view_metas: List[ViewMetadata] = []

    for name, R in view_defs:
        print(f"[encode] {name}")
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
        amp_name = f"amplitude_{name}.png"
        phase_name = f"phase_{name}.png"
        amin, amax = complex_field_to_pngs(field, out_dir / amp_name, out_dir / phase_name)
        view_metas.append(ViewMetadata(name, R.tolist(), amp_name, phase_name, amin, amax))

    meta = CodecMetadata(
        codec_name="gabor_style_vtp_confidence_decoder",
        version="2.0",
        original_file=str(input_path),
        original_point_count=int(points.shape[0]),
        original_bounds_min=bmin.tolist(),
        original_bounds_max=bmax.tolist(),
        normalization_center=center.tolist(),
        normalization_scale=float(scale),
        holo_size=int(args.holo_size),
        depth_bins=int(args.depth_bins),
        views_mode=args.views,
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
    print(f"[encode] saved encoded package: {out_dir}")


# -----------------------------------------------------------------------------
# Improved confidence-volume decoding
# -----------------------------------------------------------------------------

def select_top_pixels(intensity: np.ndarray, percentile: float, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat indices, yx coords, and normalized weights for strongest pixels."""
    flat = intensity.ravel()
    cutoff = np.percentile(flat, percentile)
    valid = np.flatnonzero(flat >= cutoff)
    if valid.size == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    if valid.size > top_k:
        vals = flat[valid]
        chosen_rel = np.argpartition(vals, -top_k)[-top_k:]
        chosen = valid[chosen_rel]
    else:
        chosen = valid

    vals = flat[chosen].astype(np.float64)
    vals = vals / (vals.max() + EPS)
    yy, xx = np.unravel_index(chosen, intensity.shape)
    return chosen, np.column_stack([yy, xx]), vals.astype(np.float32)


def add_points_to_volume(
    view_conf: np.ndarray,
    points_world_norm: np.ndarray,
    weights: np.ndarray,
    clip_extent: float,
) -> None:
    res = view_conf.shape[0]
    p = points_world_norm
    ix = np.round((p[:, 0] + clip_extent) / (2.0 * clip_extent) * (res - 1)).astype(np.int64)
    iy = np.round((p[:, 1] + clip_extent) / (2.0 * clip_extent) * (res - 1)).astype(np.int64)
    iz = np.round((p[:, 2] + clip_extent) / (2.0 * clip_extent) * (res - 1)).astype(np.int64)
    mask = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) & (iz >= 0) & (iz < res)
    if np.any(mask):
        # Volume axis order is z, y, x.
        np.add.at(view_conf, (iz[mask], iy[mask], ix[mask]), weights[mask])


def connected_component_filter(mask: np.ndarray, max_components: int, min_component_voxels: int) -> np.ndarray:
    if not np.any(mask):
        return mask
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = label(mask, structure=structure)
    if count == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep_labels = np.argsort(sizes)[::-1]
    keep_labels = [int(x) for x in keep_labels if sizes[x] >= min_component_voxels][:max_components]
    if not keep_labels:
        return mask
    return np.isin(labels, keep_labels)


def confidence_volume_to_points(
    confidence: np.ndarray,
    target_count: int,
    clip_extent: float,
    confidence_gamma: float,
    jitter_fraction: float,
    force_target_points: bool,
    max_output_points: Optional[int],
    min_output_points: int,
) -> np.ndarray:
    """
    Convert a filtered confidence volume into reconstructed points.

    Default behavior is quality-first:
      - output one point per surviving confident voxel,
      - do not force the original point count,
      - optionally cap output with --max-output-points.

    If --force-target-points is used, the old behavior is restored:
      - strongest/resampled confident voxels are forced to target_count.
    """
    flat = confidence.ravel()
    valid = np.flatnonzero(flat > 0)
    if valid.size == 0:
        raise RuntimeError("No valid voxels found in confidence volume.")

    weights = flat[valid].astype(np.float64)
    weights = np.power(weights / (weights.max() + EPS), confidence_gamma)

    rng = np.random.default_rng(2026)

    if force_target_points:
        # Legacy mode: force exact point count. Useful for some comparisons,
        # but it can create weak/off-surface points when confidence is low.
        if valid.size >= target_count:
            chosen_rel = np.argpartition(weights, -target_count)[-target_count:]
            chosen = valid[chosen_rel]
        else:
            p = weights / (weights.sum() + EPS)
            extra = rng.choice(valid, size=target_count - valid.size, replace=True, p=p)
            chosen = np.concatenate([valid, extra])
    else:
        # Quality-first mode: use only confident voxels.
        chosen = valid

        # Optional cap: keep strongest confident voxels only.
        if max_output_points is not None and max_output_points > 0 and chosen.size > max_output_points:
            chosen_weights = flat[chosen].astype(np.float64)
            keep_rel = np.argpartition(chosen_weights, -max_output_points)[-max_output_points:]
            chosen = chosen[keep_rel]

        # Optional minimum: only if the user wants a denser preview. This still
        # does not match the original point count unless requested explicitly.
        if min_output_points > 0 and chosen.size < min_output_points:
            chosen_weights = flat[chosen].astype(np.float64)
            p = chosen_weights / (chosen_weights.sum() + EPS)
            extra = rng.choice(chosen, size=min_output_points - chosen.size, replace=True, p=p)
            chosen = np.concatenate([chosen, extra])

    res = confidence.shape[0]
    iz, iy, ix = np.unravel_index(chosen, confidence.shape)
    step = 2.0 * clip_extent / max(1, res - 1)
    x = -clip_extent + ix.astype(np.float64) * step
    y = -clip_extent + iy.astype(np.float64) * step
    z = -clip_extent + iz.astype(np.float64) * step
    pts = np.column_stack([x, y, z])

    # Jitter is now opt-in. It can make visual clouds less voxel-like, but it can
    # hurt strict metrics, so default CLI value is 0.0.
    if jitter_fraction > 0:
        pts += rng.uniform(-0.5, 0.5, size=pts.shape) * step * jitter_fraction

    return pts


def decode_confidence(args: argparse.Namespace) -> None:
    encoded_dir = Path(args.encoded_dir)
    with open(encoded_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_points = int(args.target_points or meta["original_point_count"])
    views = meta["views"]
    nviews = len(views)
    depth_bins = int(meta["depth_bins"])
    holo_size = int(meta["holo_size"])
    res = int(args.volume_resolution)

    # Candidate budget controls computation, not storage.
    top_k_per_slice = int(args.top_k_per_slice or math.ceil(target_points * args.decode_oversample / max(1, nviews * depth_bins)))
    top_k_per_slice = max(1, top_k_per_slice)

    print(f"[decode] reference point budget: {target_points:,}")
    print(f"[decode] volume resolution:    {res}^3")
    print(f"[decode] top-k per depth view: {top_k_per_slice:,}")
    print(f"[decode] min view support:     {args.min_view_support}")

    confidence = np.zeros((res, res, res), dtype=np.float32)
    support = np.zeros((res, res, res), dtype=np.uint8)

    xs = np.linspace(-float(meta["extent"]), float(meta["extent"]), holo_size)
    ys = np.linspace(-float(meta["extent"]), float(meta["extent"]), holo_size)

    for view in views:
        print(f"[decode] voting from view: {view['name']}")
        field = pngs_to_complex_field(
            encoded_dir / view["amplitude_png"],
            encoded_dir / view["phase_png"],
            float(view["amp_log_min"]),
            float(view["amp_log_max"]),
        )
        R = np.asarray(view["rotation_world_to_view"], dtype=np.float64)
        view_conf = np.zeros_like(confidence)

        for b in range(depth_bins):
            frac = (b + 0.5) / depth_bins
            z_phys = float(meta["z_base"]) + frac * float(meta["z_span"])
            z_norm = 2.0 * frac - 1.0

            recon = angular_spectrum_propagate(field, -z_phys, float(meta["wavelength"]), float(meta["pixel_pitch"]))
            intensity = (np.abs(recon) ** 2).astype(np.float32)
            if args.decoder_smooth_sigma > 0:
                intensity = gaussian_filter(intensity, args.decoder_smooth_sigma)

            _, yx, weights = select_top_pixels(intensity, args.threshold_percentile, top_k_per_slice)
            if yx.size == 0:
                continue

            y_idx = yx[:, 0]
            x_idx = yx[:, 1]
            pts_view = np.column_stack([
                xs[x_idx],
                ys[y_idx],
                np.full(len(x_idx), z_norm, dtype=np.float64),
            ])

            # Since p_view = p_world @ R.T, inverse is p_world = p_view @ R.
            pts_world_norm = pts_view @ R
            add_points_to_volume(view_conf, pts_world_norm, weights, args.clip_extent)

        # Normalize each view before combining so one view cannot dominate all others.
        vmax = float(view_conf.max())
        if vmax > 0:
            view_conf /= vmax

        # Support is per-view binary evidence, optionally dilated by radius.
        view_mask = view_conf > 0
        if args.support_radius > 0:
            size = 2 * int(args.support_radius) + 1
            view_mask = maximum_filter(view_mask, size=size)

        confidence += view_conf
        support += view_mask.astype(np.uint8)

    # Multi-view consistency gate.
    min_support = int(args.min_view_support)
    valid_mask = support >= min_support
    if np.count_nonzero(valid_mask) < 100:
        print("[decode] WARNING: too few multi-view voxels. Falling back to support >= 1.")
        valid_mask = support >= 1

    combined = confidence.copy()
    combined[~valid_mask] = 0.0

    # Remove low global confidence values.
    nonzero = combined[combined > 0]
    if nonzero.size == 0:
        raise RuntimeError("No confidence survived support filtering.")
    cutoff = np.percentile(nonzero, args.volume_confidence_percentile)
    combined[combined < cutoff] = 0.0

    if args.connected_filter:
        mask = combined > 0
        mask = connected_component_filter(
            mask,
            max_components=args.max_components,
            min_component_voxels=args.min_component_voxels,
        )
        combined[~mask] = 0.0

    valid_voxels = int(np.count_nonzero(combined > 0))
    print(f"[decode] surviving confident voxels: {valid_voxels:,}")

    points_norm = confidence_volume_to_points(
        combined,
        target_count=target_points,
        clip_extent=float(args.clip_extent),
        confidence_gamma=float(args.confidence_gamma),
        jitter_fraction=float(args.jitter_fraction),
        force_target_points=bool(args.force_target_points),
        max_output_points=args.max_output_points,
        min_output_points=int(args.min_output_points),
    )

    center = np.asarray(meta["normalization_center"], dtype=np.float64)
    scale = float(meta["normalization_scale"])
    points = denormalize_points(points_norm, center, scale)

    output_vtp = Path(args.output_vtp)
    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    write_vtp_points(output_vtp, points)

    # Save decoding diagnostics.
    diag = {
        "reference_target_points": target_points,
        "output_points": int(points.shape[0]),
        "force_target_points": bool(args.force_target_points),
        "volume_resolution": res,
        "top_k_per_slice": top_k_per_slice,
        "min_view_support": min_support,
        "support_radius": int(args.support_radius),
        "volume_confidence_percentile": float(args.volume_confidence_percentile),
        "surviving_confident_voxels": valid_voxels,
        "connected_filter": bool(args.connected_filter),
    }
    with open(output_vtp.with_suffix(".decode_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"[decode] saved reconstructed VTP: {output_vtp}")


# -----------------------------------------------------------------------------
# Strict metrics
# -----------------------------------------------------------------------------

def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += file_size(Path(root) / name)
    return total


def distance_summary(d: np.ndarray, bbox_diag: float) -> Dict[str, float]:
    return {
        "mean": float(np.mean(d)),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
        "max": float(np.max(d)),
        "rmse_percent_bbox_diag": float(np.sqrt(np.mean(d ** 2)) / bbox_diag * 100.0),
        "max_percent_bbox_diag": float(np.max(d) / bbox_diag * 100.0),
    }


def estimate_spacing(points: np.ndarray, sample_size: int = 20000) -> float:
    if len(points) < 3:
        return 0.0
    rng = np.random.default_rng(123)
    idx = rng.choice(len(points), size=min(sample_size, len(points)), replace=False)
    sample = points[idx]
    tree = cKDTree(points)
    d, _ = tree.query(sample, k=2)
    return float(np.median(d[:, 1]))


def precision_recall_f1(d_o2r: np.ndarray, d_r2o: np.ndarray, thresholds: List[float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for t in thresholds:
        recall = float(np.mean(d_o2r <= t))
        precision = float(np.mean(d_r2o <= t))
        f1 = float((2 * precision * recall) / (precision + recall + EPS))
        out[f"threshold_{t:.10g}"] = {
            "threshold": float(t),
            "precision_reconstructed_to_original_percent": precision * 100.0,
            "recall_original_to_reconstructed_percent": recall * 100.0,
            "f1_percent": f1 * 100.0,
        }
    return out


def voxel_occupancy(points: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, res: int) -> np.ndarray:
    span = np.maximum(bmax - bmin, EPS)
    idx = np.floor((points - bmin) / span * res).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    flat = idx[:, 0] + res * idx[:, 1] + (res * res) * idx[:, 2]
    return np.unique(flat)


def voxel_iou(original: np.ndarray, reconstructed: np.ndarray, resolutions: List[int]) -> Dict[str, float]:
    bmin = np.minimum(original.min(axis=0), reconstructed.min(axis=0))
    bmax = np.maximum(original.max(axis=0), reconstructed.max(axis=0))
    out = {}
    for res in resolutions:
        a = voxel_occupancy(original, bmin, bmax, res)
        b = voxel_occupancy(reconstructed, bmin, bmax, res)
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = np.union1d(a, b).size
        out[f"voxel_iou_{res}"] = float(inter / max(1, union) * 100.0)
    return out


def cube_surface_metrics(points: np.ndarray, original: np.ndarray) -> Dict[str, float]:
    """
    Useful when the object is known to be a normalized axis-aligned cube-like object.
    Measures distance to nearest original bounding-box face and outside-box rate.
    """
    bmin = original.min(axis=0)
    bmax = original.max(axis=0)
    inside = np.all((points >= bmin) & (points <= bmax), axis=1)
    outside_rate = float((1.0 - np.mean(inside)) * 100.0)

    # Distance to closest of six bounding-box planes.
    face_d = np.column_stack([
        np.abs(points[:, 0] - bmin[0]),
        np.abs(points[:, 0] - bmax[0]),
        np.abs(points[:, 1] - bmin[1]),
        np.abs(points[:, 1] - bmax[1]),
        np.abs(points[:, 2] - bmin[2]),
        np.abs(points[:, 2] - bmax[2]),
    ])
    d = face_d.min(axis=1)
    return {
        "outside_original_bbox_percent": outside_rate,
        "distance_to_nearest_cube_face_mean": float(np.mean(d)),
        "distance_to_nearest_cube_face_p95": float(np.percentile(d, 95)),
        "distance_to_nearest_cube_face_p99": float(np.percentile(d, 99)),
        "distance_to_nearest_cube_face_max": float(np.max(d)),
    }


def build_report(args: argparse.Namespace) -> None:
    original_path = Path(args.original_vtp)
    reconstructed_path = Path(args.reconstructed_vtp)
    report_prefix = Path(args.report_prefix)
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    original = read_vtp_points(original_path)
    reconstructed = read_vtp_points(reconstructed_path)

    bmin = original.min(axis=0)
    bmax = original.max(axis=0)
    bbox_diag = float(np.linalg.norm(bmax - bmin)) or 1.0

    tree_r = cKDTree(reconstructed)
    tree_o = cKDTree(original)
    d_o2r, _ = tree_r.query(original, k=1)
    d_r2o, _ = tree_o.query(reconstructed, k=1)

    spacing = estimate_spacing(original)
    thresholds = [0.0025 * bbox_diag, 0.005 * bbox_diag, 0.01 * bbox_diag, 0.02 * bbox_diag]
    if spacing > 0:
        thresholds.extend([spacing, 2 * spacing, 5 * spacing, 10 * spacing])
    thresholds = sorted(set(float(t) for t in thresholds if t > 0))

    encoded_dir = Path(args.encoded_dir) if args.encoded_dir else None
    original_size = file_size(original_path)
    reconstructed_size = file_size(reconstructed_path)
    encoded_size = directory_size(encoded_dir) if encoded_dir and encoded_dir.exists() else None

    report = {
        "files": {
            "original_vtp": str(original_path),
            "reconstructed_vtp": str(reconstructed_path),
            "encoded_dir": str(encoded_dir) if encoded_dir else None,
        },
        "point_counts": {
            "original": int(len(original)),
            "reconstructed": int(len(reconstructed)),
        },
        "file_sizes_bytes": {
            "original_vtp": int(original_size),
            "reconstructed_vtp": int(reconstructed_size),
            "encoded_dir_total": int(encoded_size) if encoded_size is not None else None,
        },
        "compression": {},
        "geometry": {
            "bbox_diagonal": bbox_diag,
            "estimated_original_spacing": spacing,
            "original_to_reconstructed": distance_summary(d_o2r, bbox_diag),
            "reconstructed_to_original": distance_summary(d_r2o, bbox_diag),
            "chamfer_l2_squared": float(np.mean(d_o2r ** 2) + np.mean(d_r2o ** 2)),
            "hausdorff_max_symmetric": float(max(np.max(d_o2r), np.max(d_r2o))),
            "hausdorff_95_symmetric": float(max(np.percentile(d_o2r, 95), np.percentile(d_r2o, 95))),
            "precision_recall_f1": precision_recall_f1(d_o2r, d_r2o, thresholds),
            "voxel_iou": voxel_iou(original, reconstructed, args.voxel_iou_resolutions),
        },
    }

    if args.cube_surface_metrics:
        report["geometry"]["cube_surface_metrics"] = cube_surface_metrics(reconstructed, original)

    if encoded_size and encoded_size > 0:
        report["compression"] = {
            "compression_ratio_original_vtp_over_encoded": float(original_size / encoded_size),
            "encoded_size_percent_of_original_vtp": float(encoded_size / original_size * 100.0) if original_size else None,
        }

    json_path = report_prefix.with_suffix(".json")
    txt_path = report_prefix.with_suffix(".txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append("=" * 72)
    lines.append("STRICT GABOR VTP CONFIDENCE-DECODER REPORT")
    lines.append("=" * 72)
    lines.append(f"Original:      {original_path}")
    lines.append(f"Reconstructed: {reconstructed_path}")
    if encoded_dir:
        lines.append(f"Encoded dir:   {encoded_dir}")
    lines.append("")
    lines.append(f"Original points:      {len(original):,}")
    lines.append(f"Reconstructed points: {len(reconstructed):,}")
    lines.append(f"Original size:        {original_size:,} bytes")
    if encoded_size is not None:
        lines.append(f"Encoded size:         {encoded_size:,} bytes")
        lines.append(f"Compression ratio:    {original_size / encoded_size:.4f}x")
    lines.append(f"Reconstructed size:   {reconstructed_size:,} bytes")
    lines.append("")
    lines.append(f"BBox diagonal:        {bbox_diag:.10f}")
    lines.append(f"Estimated spacing:    {spacing:.10f}")
    lines.append("")
    lines.append("NEAREST-NEIGHBOR DISTANCES")
    lines.append(f"O -> R RMSE:          {report['geometry']['original_to_reconstructed']['rmse']:.10f}")
    lines.append(f"R -> O RMSE:          {report['geometry']['reconstructed_to_original']['rmse']:.10f}")
    lines.append(f"O -> R p95:           {report['geometry']['original_to_reconstructed']['p95']:.10f}")
    lines.append(f"R -> O p95:           {report['geometry']['reconstructed_to_original']['p95']:.10f}")
    lines.append(f"Symmetric Hausdorff95:{report['geometry']['hausdorff_95_symmetric']:.10f}")
    lines.append(f"Symmetric HausdorffMax:{report['geometry']['hausdorff_max_symmetric']:.10f}")
    lines.append("")
    lines.append("PRECISION / RECALL / F1")
    for item in report["geometry"]["precision_recall_f1"].values():
        lines.append(
            f"t={item['threshold']:.10f} | "
            f"precision={item['precision_reconstructed_to_original_percent']:.2f}% | "
            f"recall={item['recall_original_to_reconstructed_percent']:.2f}% | "
            f"F1={item['f1_percent']:.2f}%"
        )
    lines.append("")
    lines.append("VOXEL IOU")
    for k, v in report["geometry"]["voxel_iou"].items():
        lines.append(f"{k}: {v:.2f}%")
    if args.cube_surface_metrics:
        lines.append("")
        lines.append("CUBE-SURFACE METRICS")
        for k, v in report["geometry"]["cube_surface_metrics"].items():
            lines.append(f"{k}: {v:.10f}" if "percent" not in k else f"{k}: {v:.2f}%")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[report] saved {json_path}")
    print(f"[report] saved {txt_path}")


# -----------------------------------------------------------------------------
# run-all
# -----------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> None:
    exp = Path(args.experiment_dir)
    encoded = exp / "encoded_hologram"
    recon = exp / "reconstructed.vtp"
    report_prefix = exp / "report" / "strict_metrics"
    exp.mkdir(parents=True, exist_ok=True)

    encode_args = argparse.Namespace(
        input_vtp=args.input_vtp,
        output_dir=str(encoded),
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
        encoded_dir=str(encoded),
        output_vtp=str(recon),
        target_points=args.target_points,
        volume_resolution=args.volume_resolution,
        min_view_support=args.min_view_support,
        support_radius=args.support_radius,
        threshold_percentile=args.threshold_percentile,
        volume_confidence_percentile=args.volume_confidence_percentile,
        decode_oversample=args.decode_oversample,
        top_k_per_slice=args.top_k_per_slice,
        decoder_smooth_sigma=args.decoder_smooth_sigma,
        clip_extent=args.clip_extent,
        confidence_gamma=args.confidence_gamma,
        jitter_fraction=args.jitter_fraction,
        force_target_points=args.force_target_points,
        max_output_points=args.max_output_points,
        min_output_points=args.min_output_points,
        connected_filter=args.connected_filter,
        max_components=args.max_components,
        min_component_voxels=args.min_component_voxels,
    )
    decode_confidence(decode_args)

    report_args = argparse.Namespace(
        original_vtp=args.input_vtp,
        reconstructed_vtp=str(recon),
        encoded_dir=str(encoded),
        report_prefix=str(report_prefix),
        voxel_iou_resolutions=args.voxel_iou_resolutions,
        cube_surface_metrics=args.cube_surface_metrics,
    )
    build_report(report_args)
    print(f"\n[run-all] done: {exp}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def add_encoding_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--holo-size", type=int, default=1024)
    p.add_argument("--depth-bins", type=int, default=96)
    p.add_argument("--views", type=str, default="cube6")
    p.add_argument("--wavelength", type=float, default=0.05)
    p.add_argument("--pixel-pitch", type=float, default=0.01)
    p.add_argument("--extent", type=float, default=1.25)
    p.add_argument("--z-base", type=float, default=0.35)
    p.add_argument("--z-span", type=float, default=1.50)
    p.add_argument("--splat-sigma", type=float, default=0.75)


def add_decoding_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target-points", type=int, default=None)
    p.add_argument("--volume-resolution", type=int, default=256, help="3D confidence grid size. 192 or 256 are practical; 384 is heavier.")
    p.add_argument("--min-view-support", type=int, default=2, help="Voxel must be supported by at least this many views.")
    p.add_argument("--support-radius", type=int, default=1, help="Dilate per-view support by this voxel radius.")
    p.add_argument("--threshold-percentile", type=float, default=99.2, help="Per-depth-slice pixel threshold before top-k.")
    p.add_argument("--volume-confidence-percentile", type=float, default=30.0, help="Remove weak surviving volume voxels below this percentile.")
    p.add_argument("--decode-oversample", type=float, default=1.2, help="Candidate budget multiplier before volume fusion.")
    p.add_argument("--top-k-per-slice", type=int, default=None, help="Override automatic top-k pixels per depth slice.")
    p.add_argument("--decoder-smooth-sigma", type=float, default=0.0, help="Optional Gaussian smoothing of back-propagated intensity.")
    p.add_argument("--clip-extent", type=float, default=1.35)
    p.add_argument("--confidence-gamma", type=float, default=0.75, help="Lower = more uniform sampling; higher = more strongest-voxel sampling.")
    p.add_argument("--jitter-fraction", type=float, default=0.0, help="Optional jitter inside selected voxels. Default 0.0 gives stricter geometry.")
    p.add_argument("--force-target-points", action="store_true", help="Force output to original/target point count. Default is quality-first variable output.")
    p.add_argument("--max-output-points", type=int, default=None, help="Optional cap for variable-output mode; keeps strongest confident voxels.")
    p.add_argument("--min-output-points", type=int, default=0, help="Optional minimum output count for visualization; does not force original count.")
    p.add_argument("--connected-filter", action="store_true", help="Keep largest connected confidence components. More accurate but heavier.")
    p.add_argument("--max-components", type=int, default=3)
    p.add_argument("--min-component-voxels", type=int, default=32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gabor-style VTP hologram codec with confidence-volume decoder")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("encode")
    p.add_argument("input_vtp")
    p.add_argument("output_dir")
    add_encoding_options(p)
    p.set_defaults(func=encode_vtp)

    p = sub.add_parser("decode")
    p.add_argument("encoded_dir")
    p.add_argument("output_vtp")
    add_decoding_options(p)
    p.set_defaults(func=decode_confidence)

    p = sub.add_parser("report")
    p.add_argument("original_vtp")
    p.add_argument("reconstructed_vtp")
    p.add_argument("report_prefix")
    p.add_argument("--encoded-dir", default=None)
    p.add_argument("--voxel-iou-resolutions", type=int, nargs="+", default=[128, 256])
    p.add_argument("--cube-surface-metrics", action="store_true")
    p.set_defaults(func=build_report)

    p = sub.add_parser("run-all")
    p.add_argument("input_vtp")
    p.add_argument("experiment_dir")
    add_encoding_options(p)
    add_decoding_options(p)
    p.add_argument("--voxel-iou-resolutions", type=int, nargs="+", default=[128, 256])
    p.add_argument("--cube-surface-metrics", action="store_true")
    p.set_defaults(func=run_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
