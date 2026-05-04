#!/usr/bin/env python3
"""
encode_iwch.py

IWCH = Importance/structure-aware Weighted Compressive Hologram prototype.

Practical encoder for mixed-size matrix layers:
    matrices -> blockwise low-rank SVD + sparse Fourier residual coefficients
             -> fixed-size 2D complex hologram field
             -> PNG images: output.png, output_real.png, output_imag.png, output_meta.png

This is a research prototype, not a physical optics simulator. The 2D field is
made diffraction-like by spreading packed coefficients through an inverse FFT.
"""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import zlib

import numpy as np
from PIL import Image


def load_matrix(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in {".csv", ".txt"}:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:
            arr = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported matrix file: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Matrix must be 2D: {path}")
    return np.asarray(arr, dtype=np.float64)


def list_matrix_files(folder: Path):
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".npy"})
    if not files:
        raise FileNotFoundError(f"No CSV/TXT/NPY matrices found in {folder}")
    return files


def indices_to_b64(indices: np.ndarray) -> str:
    indices = np.asarray(indices, dtype=np.uint32)
    return base64.b64encode(indices.tobytes()).decode("ascii")


def quantize_uint16(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-15:
        q = np.zeros(arr.shape, dtype=np.uint16)
    else:
        q = np.round((arr - mn) / (mx - mn) * 65535.0)
        q = np.clip(q, 0, 65535).astype(np.uint16)
    return q, mn, mx


def save_uint16_png(arr: np.ndarray, path: str):
    Image.fromarray(np.asarray(arr, dtype=np.uint16), mode="I;16").save(path)


def save_phase_preview(field: np.ndarray, path: str):
    phase = np.angle(field)
    img = np.round((phase + np.pi) / (2.0 * np.pi) * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def save_meta_png(meta: dict, path: str, width: int = 1024):
    raw_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw_json, level=9)
    header = len(compressed).to_bytes(4, byteorder="little", signed=False)
    payload = header + compressed
    height = math.ceil(len(payload) / width)
    arr = np.zeros((height, width), dtype=np.uint8)
    arr.ravel()[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def auto_hologram_size(scalar_count: int, min_side: int = 64):
    complex_count = math.ceil(scalar_count / 2)
    side = int(math.ceil(math.sqrt(complex_count)))
    side = max(min_side, side)
    return side, side


def compress_block(block: np.ndarray, rank: int, residual_k: int):
    br, bc = block.shape
    mean = float(np.mean(block))
    std = float(np.std(block))
    if std < 1e-12:
        std = 1.0
    X = (block - mean) / std

    r = int(max(0, min(rank, br, bc)))
    if r > 0:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]
        low = (U * S) @ Vt
    else:
        U = np.empty((br, 0), dtype=np.float64)
        S = np.empty((0,), dtype=np.float64)
        Vt = np.empty((0, bc), dtype=np.float64)
        low = np.zeros_like(X)

    R = X - low
    F = np.fft.fft2(R, norm="ortho")
    flat = F.ravel()
    k = int(max(0, min(residual_k, flat.size)))
    if k > 0:
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        idx = np.sort(idx.astype(np.uint32))
        vals = flat[idx]
    else:
        idx = np.empty((0,), dtype=np.uint32)
        vals = np.empty((0,), dtype=np.complex128)

    coeffs = []
    coeffs.append(mean)
    coeffs.append(std)
    coeffs.extend(U.ravel(order="C"))
    coeffs.extend(S.ravel(order="C"))
    coeffs.extend(Vt.ravel(order="C"))
    coeffs.extend(np.real(vals).ravel(order="C"))
    coeffs.extend(np.imag(vals).ravel(order="C"))

    bmeta = {
        "br": int(br),
        "bc": int(bc),
        "rank": int(r),
        "residual_k": int(k),
        "residual_indices_b64": indices_to_b64(idx),
        "coeff_count": int(len(coeffs)),
    }
    return coeffs, bmeta


def pack_coefficients_to_field(scalars: np.ndarray, height: int, width: int, seed: int):
    scalar_count = int(scalars.size)
    coeff_mean = float(np.mean(scalars)) if scalar_count else 0.0
    coeff_std = float(np.std(scalars)) if scalar_count else 1.0
    if coeff_std < 1e-12:
        coeff_std = 1.0
    scalars_n = (scalars - coeff_mean) / coeff_std

    if scalar_count % 2 == 1:
        scalars_n = np.concatenate([scalars_n, np.zeros(1, dtype=np.float64)])
    complex_coeffs = scalars_n[0::2] + 1j * scalars_n[1::2]
    complex_count = int(complex_coeffs.size)
    capacity = int(height * width)
    if complex_count > capacity:
        needed = int(math.ceil(math.sqrt(complex_count)))
        raise ValueError(
            f"Hologram too small: need {complex_count} complex pixels, capacity is {capacity}. "
            f"Try --holo_height {needed} --holo_width {needed}, or reduce rank/residual_k."
        )

    canvas = np.zeros((height, width), dtype=np.complex128)
    rng = np.random.default_rng(seed)
    positions = rng.permutation(capacity)[:complex_count]
    canvas.ravel()[positions] = complex_coeffs

    # Diffraction-like spreading. Decoding uses FFT2 to recover the sparse coefficient canvas.
    field = np.fft.ifft2(canvas, norm="ortho")
    return field, complex_count, coeff_mean, coeff_std


def main():
    ap = argparse.ArgumentParser(description="Encode mixed-size matrix layers into fixed-size 2D hologram images.")
    ap.add_argument("input_folder", help="Folder with CSV/TXT/NPY matrix layers")
    ap.add_argument("output", help="Output base or preview path, e.g. output.png")
    ap.add_argument("--rank", type=int, default=8, help="SVD rank per block")
    ap.add_argument("--residual_k", type=int, default=32, help="Sparse Fourier residual coefficients per block")
    ap.add_argument("--block_rows", type=int, default=128, help="Block height")
    ap.add_argument("--block_cols", type=int, default=128, help="Block width")
    ap.add_argument("--holo_height", type=int, default=0, help="Fixed hologram height; 0 = auto minimum")
    ap.add_argument("--holo_width", type=int, default=0, help="Fixed hologram width; 0 = auto minimum")
    ap.add_argument("--seed", type=int, default=12345, help="Deterministic coefficient spreading seed")
    args = ap.parse_args()

    folder = Path(args.input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    matrix_files = list_matrix_files(folder)

    all_coeffs = []
    layers_meta = []
    total_input_values = 0

    for li, path in enumerate(matrix_files):
        W = load_matrix(path)
        rows, cols = W.shape
        total_input_values += int(W.size)
        layer_meta = {"name": path.name, "rows": int(rows), "cols": int(cols), "blocks": []}
        print(f"Layer {li}: {path.name}, shape={rows}x{cols}")

        for r0 in range(0, rows, args.block_rows):
            r1 = min(r0 + args.block_rows, rows)
            for c0 in range(0, cols, args.block_cols):
                c1 = min(c0 + args.block_cols, cols)
                block = W[r0:r1, c0:c1]
                offset = len(all_coeffs)
                coeffs, bmeta = compress_block(block, args.rank, args.residual_k)
                bmeta.update({"r0": int(r0), "c0": int(c0), "offset": int(offset)})
                all_coeffs.extend(float(x) for x in coeffs)
                layer_meta["blocks"].append(bmeta)

        print(f"  blocks={len(layer_meta['blocks'])}")
        layers_meta.append(layer_meta)

    scalars = np.asarray(all_coeffs, dtype=np.float64)
    if args.holo_height <= 0 or args.holo_width <= 0:
        holo_h, holo_w = auto_hologram_size(len(scalars))
    else:
        holo_h, holo_w = int(args.holo_height), int(args.holo_width)

    field, complex_count, coeff_mean, coeff_std = pack_coefficients_to_field(scalars, holo_h, holo_w, args.seed)
    real_q, real_min, real_max = quantize_uint16(np.real(field))
    imag_q, imag_min, imag_max = quantize_uint16(np.imag(field))

    out = Path(args.output)
    if out.suffix:
        base = str(out.with_suffix(""))
        preview = str(out)
    else:
        base = str(out)
        preview = base + ".png"

    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"

    save_phase_preview(field, preview)
    save_uint16_png(real_q, real_path)
    save_uint16_png(imag_q, imag_path)

    meta = {
        "version": 1,
        "encoding": "IWCH_blockwise_svd_sparse_residual_fft_hologram",
        "rank": int(args.rank),
        "residual_k": int(args.residual_k),
        "block_rows": int(args.block_rows),
        "block_cols": int(args.block_cols),
        "seed": int(args.seed),
        "holo_height": int(holo_h),
        "holo_width": int(holo_w),
        "total_input_values": int(total_input_values),
        "scalar_count": int(len(scalars)),
        "complex_count": int(complex_count),
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "real_min": real_min,
        "real_max": real_max,
        "imag_min": imag_min,
        "imag_max": imag_max,
        "layers": layers_meta,
    }
    save_meta_png(meta, meta_path)

    raw_original_bytes = total_input_values * 4
    raw_hologram_bytes = holo_h * holo_w * 2 * 2
    print("\nSaved:")
    print(f"  {preview}")
    print(f"  {real_path}")
    print(f"  {imag_path}")
    print(f"  {meta_path}")
    print("\nStorage estimate, excluding metadata image compression overhead:")
    print(f"  original float32 values: {total_input_values:,} values ≈ {raw_original_bytes/1e6:.2f} MB")
    print(f"  hologram real+imag:     {holo_h}x{holo_w} complex pixels ≈ {raw_hologram_bytes/1e6:.2f} MB")
    print(f"  estimated compression:  {raw_original_bytes / max(raw_hologram_bytes,1):.2f}x")
    print(f"  coefficient scalar ratio: {len(scalars) / max(total_input_values,1):.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
