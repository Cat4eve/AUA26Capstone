#!/usr/bin/env python3
"""
VTP <-> hologram codec for simple VTP point clouds.

This script is designed for the kind of VTP files used in ParaView/OpenHolo-style
experiments: a VTK XML PolyData file containing 3D points.

Pipeline:
    VTP point cloud
        -> normalize XYZ coordinates
        -> rasterize points into a complex 2D object field
           amplitude = point density/occupancy
           phase     = normalized depth z
        -> 2D FFT gives the hologram-like complex field
        -> save:
              preview phase PNG
              real uint16 PNG
              imag uint16 PNG
              metadata PNG

Decoding:
    real/imag PNG + metadata PNG
        -> recover complex hologram
        -> inverse FFT
        -> recover object-field amplitude and phase
        -> threshold amplitude to points
        -> phase gives reconstructed z
        -> write reconstructed VTP

Important limitation:
    This practical codec is strongest for surfaces/height-fields where each XY
    pixel has one main depth. If many different z-values overlap at the same XY
    location, the phase values interfere/average. For exact arbitrary 3D point
    clouds, a more advanced multi-depth/angular-spectrum reconstruction is needed.
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


# -----------------------------
# Basic VTP reading/writing
# -----------------------------

def _parse_floats(text):
    return np.fromstring((text or "").strip(), sep=" ", dtype=np.float64)


def read_vtp_points(vtp_path):
    """
    Read XYZ points from a .vtp PolyData file.

    Returns:
        points: ndarray of shape (N, 3)
    """
    tree = ET.parse(vtp_path)
    root = tree.getroot()

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("Invalid VTP: missing Piece element")

    points_node = piece.find("Points")
    if points_node is None:
        raise ValueError("Invalid VTP: missing Points element")

    arr = points_node.find("DataArray")
    if arr is None or not (arr.text or "").strip():
        raise ValueError("Invalid VTP: missing point coordinates")

    ncomp = int(arr.attrib.get("NumberOfComponents", "3"))
    raw = _parse_floats(arr.text)
    if ncomp != 3:
        raise ValueError(f"Expected 3 coordinate components, got {ncomp}")
    if raw.size % 3 != 0:
        raise ValueError("Point coordinate array length is not divisible by 3")

    points = raw.reshape(-1, 3)
    if points.size == 0:
        raise ValueError("No points found in VTP")
    return points


def write_vtp_points(output_path, points, strength=None, phase=None):
    """
    Write points as a VTP PolyData vertex cloud.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    poly = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(n),
        NumberOfVerts=str(n),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )

    pdata = ET.SubElement(piece, "PointData", Scalars="z_value")

    def add_pdata(name, values, dtype="Float64"):
        arr = ET.SubElement(
            pdata,
            "DataArray",
            type=dtype,
            Name=name,
            NumberOfComponents="1",
            format="ascii",
        )
        arr.text = " ".join(f"{float(v):.17g}" for v in values)

    if n:
        add_pdata("z_value", points[:, 2])
    if strength is not None:
        add_pdata("reconstruction_strength", np.asarray(strength)[:n])
    if phase is not None:
        add_pdata("reconstruction_phase", np.asarray(phase)[:n])

    pts_node = ET.SubElement(piece, "Points")
    pts_arr = ET.SubElement(
        pts_node,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    pts_arr.text = " ".join(f"{x:.17g} {y:.17g} {z:.17g}" for x, y, z in points)

    verts = ET.SubElement(piece, "Verts")
    conn = ET.SubElement(verts, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(i) for i in range(n))
    offs = ET.SubElement(verts, "DataArray", type="Int32", Name="offsets", format="ascii")
    offs.text = " ".join(str(i + 1) for i in range(n))

    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


# -----------------------------
# PNG + metadata helpers
# -----------------------------

def quantize_to_uint16(arr):
    arr = np.asarray(arr, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-15:
        q = np.zeros(arr.shape, dtype=np.uint16)
    else:
        q = np.round((arr - mn) / (mx - mn) * 65535.0).astype(np.uint16)
    return q, mn, mx


def dequantize_uint16(q, mn, mx):
    q = np.asarray(q, dtype=np.float64)
    if mx - mn < 1e-15:
        return np.full(q.shape, mn, dtype=np.float64)
    return q / 65535.0 * (mx - mn) + mn


def save_uint16_png(arr, path):
    Image.fromarray(np.asarray(arr, dtype=np.uint16), mode="I;16").save(path)


def load_uint16_png(path):
    return np.array(Image.open(path), dtype=np.uint16)


def save_phase_preview(field, path):
    phase = np.angle(field)
    img = np.round((phase + np.pi) / (2.0 * np.pi) * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def encode_metadata_png(metadata, path, width=512):
    payload = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header = len(payload).to_bytes(4, byteorder="little", signed=False)
    buf = header + payload

    height = math.ceil(len(buf) / width)
    arr = np.zeros((height, width), dtype=np.uint8)
    flat = arr.ravel()
    flat[: len(buf)] = np.frombuffer(buf, dtype=np.uint8)

    Image.fromarray(arr, mode="L").save(path)


def decode_metadata_png(path):
    arr = np.array(Image.open(path), dtype=np.uint8).ravel()
    if arr.size < 4:
        raise ValueError("Metadata PNG too small")

    length = int.from_bytes(bytes(arr[:4].tolist()), byteorder="little", signed=False)
    payload = bytes(arr[4 : 4 + length].tolist())
    return json.loads(payload.decode("utf-8"))


# -----------------------------
# Hologram-style encoding
# -----------------------------

def normalize_points(points):
    """
    Normalize XYZ to [0, 1] using per-axis min/max.
    """
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    spans[spans < 1e-15] = 1.0
    norm = (points - mins) / spans
    return norm, mins, maxs


def rasterize_points_to_complex_field(points_norm, height, width, accumulation="mean"):
    """
    Convert normalized 3D points into a complex 2D object field.

    x,y choose pixel position.
    z becomes phase, so 3D depth is encoded into the 2D complex field.

    If multiple points land in the same pixel:
        mean mode averages the complex phasors.
        max mode keeps the strongest single phasor.
    """
    x = np.clip(np.round(points_norm[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.round(points_norm[:, 1] * (height - 1)).astype(np.int64), 0, height - 1)
    z = np.clip(points_norm[:, 2], 0.0, 1.0)

    # Phase depth encoding:
    # z=0 -> -pi, z=1 -> +pi
    phase = (z * 2.0 * np.pi) - np.pi
    phasor = np.exp(1j * phase)

    field = np.zeros((height, width), dtype=np.complex128)
    counts = np.zeros((height, width), dtype=np.float64)

    if accumulation == "mean":
        np.add.at(field, (y, x), phasor)
        np.add.at(counts, (y, x), 1.0)
        mask = counts > 0
        field[mask] /= counts[mask]
    elif accumulation == "sum":
        np.add.at(field, (y, x), phasor)
        np.add.at(counts, (y, x), 1.0)
        if counts.max() > 0:
            field /= counts.max()
    else:
        raise ValueError("accumulation must be 'mean' or 'sum'")

    return field, counts


def encode_vtp_to_hologram(input_vtp, output_image, holo_height=1024, holo_width=1024, accumulation="mean"):
    points = read_vtp_points(input_vtp)
    norm, mins, maxs = normalize_points(points)

    object_field, counts = rasterize_points_to_complex_field(
        norm, holo_height, holo_width, accumulation=accumulation
    )

    # Hologram-like transformation:
    # The 2D FFT stores the complex object field in a frequency/diffraction-like plane.
    hologram = np.fft.fft2(object_field)

    real_img, real_min, real_max = quantize_to_uint16(np.real(hologram))
    imag_img, imag_min, imag_max = quantize_to_uint16(np.imag(hologram))

    base = str(Path(output_image).with_suffix(""))
    preview_path = output_image
    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"
    density_path = base + "_density.png"

    save_phase_preview(hologram, preview_path)
    save_uint16_png(real_img, real_path)
    save_uint16_png(imag_img, imag_path)

    density = np.clip(counts / max(float(counts.max()), 1.0), 0.0, 1.0)
    Image.fromarray(np.round(density * 255.0).astype(np.uint8), mode="L").save(density_path)

    meta = {
        "version": 1,
        "encoding": "vtp_pointcloud_phase_depth_fft",
        "description": "XYZ VTP point cloud -> complex object field with z as phase -> FFT hologram",
        "source_vtp": os.path.basename(str(input_vtp)),
        "original_point_count": int(len(points)),
        "holo_height": int(holo_height),
        "holo_width": int(holo_width),
        "xyz_min": [float(v) for v in mins],
        "xyz_max": [float(v) for v in maxs],
        "accumulation": accumulation,
        "real_min": real_min,
        "real_max": real_max,
        "imag_min": imag_min,
        "imag_max": imag_max,
        "companion_files": {
            "preview": os.path.basename(preview_path),
            "real": os.path.basename(real_path),
            "imag": os.path.basename(imag_path),
            "meta": os.path.basename(meta_path),
            "density": os.path.basename(density_path),
        },
    }
    encode_metadata_png(meta, meta_path)

    print("Encoded VTP point cloud into hologram images")
    print(f"Input VTP:       {input_vtp}")
    print(f"Points:          {len(points)}")
    print(f"Hologram size:   {holo_height} x {holo_width}")
    print(f"Preview phase:   {preview_path}")
    print(f"Real image:      {real_path}")
    print(f"Imag image:      {imag_path}")
    print(f"Metadata image:  {meta_path}")
    print(f"Density preview: {density_path}")


# -----------------------------
# Hologram-style decoding
# -----------------------------

def recover_points_from_object_field(object_field, meta, threshold=0.20, max_points=None):
    amp = np.abs(object_field)
    phase = np.angle(object_field)

    if amp.max() < 1e-15:
        raise ValueError("Decoded object field is empty")

    # Use relative threshold. This keeps pixels that likely represent encoded points.
    mask = amp >= (float(threshold) * float(amp.max()))

    ys, xs = np.where(mask)
    strengths = amp[ys, xs]
    phases = phase[ys, xs]

    if max_points is not None and len(xs) > max_points:
        order = np.argsort(strengths)[::-1][:max_points]
        xs = xs[order]
        ys = ys[order]
        strengths = strengths[order]
        phases = phases[order]

    h = int(meta["holo_height"])
    w = int(meta["holo_width"])

    x_norm = xs.astype(np.float64) / max(w - 1, 1)
    y_norm = ys.astype(np.float64) / max(h - 1, 1)

    # Inverse of phase = z*2*pi - pi
    z_norm = (phases + np.pi) / (2.0 * np.pi)
    z_norm = np.clip(z_norm, 0.0, 1.0)

    mins = np.array(meta["xyz_min"], dtype=np.float64)
    maxs = np.array(meta["xyz_max"], dtype=np.float64)
    spans = maxs - mins
    spans[spans < 1e-15] = 1.0

    points_norm = np.column_stack([x_norm, y_norm, z_norm])
    points = points_norm * spans + mins
    return points, strengths, phases


def decode_hologram_to_vtp(input_hologram, output_vtp, threshold=0.20, max_points=None):
    base = str(Path(input_hologram).with_suffix(""))
    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"

    for p in [real_path, imag_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required companion file: {p}")

    meta = decode_metadata_png(meta_path)
    if meta.get("encoding") != "vtp_pointcloud_phase_depth_fft":
        raise ValueError(
            "Metadata encoding does not match this decoder. "
            f"Found: {meta.get('encoding')}"
        )

    real = dequantize_uint16(load_uint16_png(real_path), meta["real_min"], meta["real_max"])
    imag = dequantize_uint16(load_uint16_png(imag_path), meta["imag_min"], meta["imag_max"])
    hologram = real + 1j * imag

    object_field = np.fft.ifft2(hologram)
    points, strengths, phases = recover_points_from_object_field(
        object_field, meta, threshold=threshold, max_points=max_points
    )

    write_vtp_points(output_vtp, points, strength=strengths, phase=phases)

    print("Decoded hologram images into reconstructed VTP")
    print(f"Input hologram:  {input_hologram}")
    print(f"Output VTP:      {output_vtp}")
    print(f"Recovered points:{len(points)}")
    print(f"Threshold:       {threshold}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode/decode VTP point clouds through a hologram-like FFT transform.")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode VTP point cloud to hologram PNG files")
    enc.add_argument("input_vtp")
    enc.add_argument("output_hologram", help="Preview hologram PNG path, e.g. sphere_holo.png")
    enc.add_argument("--holo_height", type=int, default=1024)
    enc.add_argument("--holo_width", type=int, default=1024)
    enc.add_argument("--accumulation", choices=["mean", "sum"], default="mean")

    dec = sub.add_parser("decode", help="Decode hologram PNG files back to VTP point cloud")
    dec.add_argument("input_hologram", help="Preview hologram PNG path used during encoding")
    dec.add_argument("output_vtp")
    dec.add_argument("--threshold", type=float, default=0.20, help="Relative amplitude threshold, e.g. 0.10-0.35")
    dec.add_argument("--max_points", type=int, default=None, help="Optional cap on reconstructed points")

    rt = sub.add_parser("roundtrip", help="Encode then immediately decode")
    rt.add_argument("input_vtp")
    rt.add_argument("output_hologram")
    rt.add_argument("output_vtp")
    rt.add_argument("--holo_height", type=int, default=1024)
    rt.add_argument("--holo_width", type=int, default=1024)
    rt.add_argument("--threshold", type=float, default=0.20)
    rt.add_argument("--max_points", type=int, default=None)
    rt.add_argument("--accumulation", choices=["mean", "sum"], default="mean")

    args = parser.parse_args()

    if args.command == "encode":
        encode_vtp_to_hologram(
            args.input_vtp,
            args.output_hologram,
            args.holo_height,
            args.holo_width,
            args.accumulation,
        )
    elif args.command == "decode":
        decode_hologram_to_vtp(
            args.input_hologram,
            args.output_vtp,
            args.threshold,
            args.max_points,
        )
    elif args.command == "roundtrip":
        encode_vtp_to_hologram(
            args.input_vtp,
            args.output_hologram,
            args.holo_height,
            args.holo_width,
            args.accumulation,
        )
        decode_hologram_to_vtp(
            args.output_hologram,
            args.output_vtp,
            args.threshold,
            args.max_points,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
