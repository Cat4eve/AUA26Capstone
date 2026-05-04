#!/usr/bin/env python3
"""
decode_iwch.py

Decodes IWCH hologram images back to approximate matrix layers.
Required files for base output.png:
    output_real.png
    output_imag.png
    output_meta.png
"""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from PIL import Image


def load_uint16_png(path: str) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def load_meta_png(path: str) -> dict:
    arr = np.asarray(Image.open(path), dtype=np.uint8).ravel()
    if arr.size < 4:
        raise ValueError("Metadata image too small")
    n = int.from_bytes(bytes(arr[:4].tolist()), byteorder="little", signed=False)
    comp = bytes(arr[4:4+n].tolist())
    raw = zlib.decompress(comp)
    return json.loads(raw.decode("utf-8"))


def dequantize_uint16(q: np.ndarray, mn: float, mx: float):
    q = np.asarray(q, dtype=np.float64)
    if mx - mn < 1e-15:
        return np.full(q.shape, mn, dtype=np.float64)
    return q / 65535.0 * (mx - mn) + mn


def indices_from_b64(text: str) -> np.ndarray:
    raw = base64.b64decode(text.encode("ascii"))
    return np.frombuffer(raw, dtype=np.uint32).astype(np.int64)


def recover_scalars(input_preview: str):
    p = Path(input_preview)
    base = str(p.with_suffix("")) if p.suffix else str(p)
    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"

    for path in [real_path, imag_path, meta_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    meta = load_meta_png(meta_path)
    real = dequantize_uint16(load_uint16_png(real_path), meta["real_min"], meta["real_max"])
    imag = dequantize_uint16(load_uint16_png(imag_path), meta["imag_min"], meta["imag_max"])
    field = real + 1j * imag

    canvas = np.fft.fft2(field, norm="ortho")
    capacity = int(meta["holo_height"] * meta["holo_width"])
    complex_count = int(meta["complex_count"])
    rng = np.random.default_rng(int(meta["seed"]))
    positions = rng.permutation(capacity)[:complex_count]
    coeffs = canvas.ravel()[positions]

    scalars_n = np.empty(complex_count * 2, dtype=np.float64)
    scalars_n[0::2] = np.real(coeffs)
    scalars_n[1::2] = np.imag(coeffs)
    scalars_n = scalars_n[:int(meta["scalar_count"])]

    scalars = scalars_n * float(meta["coeff_std"]) + float(meta["coeff_mean"])
    return scalars, meta


def decode_block(scalars: np.ndarray, block_meta: dict) -> np.ndarray:
    br = int(block_meta["br"])
    bc = int(block_meta["bc"])
    rank = int(block_meta["rank"])
    k = int(block_meta["residual_k"])
    offset = int(block_meta["offset"])
    coeff_count = int(block_meta["coeff_count"])

    c = scalars[offset:offset+coeff_count]
    ptr = 0
    mean = float(c[ptr]); ptr += 1
    std = float(c[ptr]); ptr += 1

    u_count = br * rank
    U = c[ptr:ptr+u_count].reshape((br, rank), order="C") if rank else np.empty((br, 0))
    ptr += u_count

    S = c[ptr:ptr+rank]
    ptr += rank

    vt_count = rank * bc
    Vt = c[ptr:ptr+vt_count].reshape((rank, bc), order="C") if rank else np.empty((0, bc))
    ptr += vt_count

    res_real = c[ptr:ptr+k]
    ptr += k
    res_imag = c[ptr:ptr+k]
    ptr += k

    if rank:
        low = (U * S) @ Vt
    else:
        low = np.zeros((br, bc), dtype=np.float64)

    residual_fft = np.zeros(br * bc, dtype=np.complex128)
    if k:
        idx = indices_from_b64(block_meta["residual_indices_b64"])
        residual_fft[idx] = res_real + 1j * res_imag
    residual = np.fft.ifft2(residual_fft.reshape((br, bc)), norm="ortho").real

    block_n = low + residual
    return block_n * std + mean


def reconstruct_layers(scalars: np.ndarray, meta: dict):
    layers = []
    names = []
    for layer_meta in meta["layers"]:
        rows = int(layer_meta["rows"])
        cols = int(layer_meta["cols"])
        W = np.zeros((rows, cols), dtype=np.float64)
        for b in layer_meta["blocks"]:
            r0 = int(b["r0"])
            c0 = int(b["c0"])
            block = decode_block(scalars, b)
            br, bc = block.shape
            W[r0:r0+br, c0:c0+bc] = block
        layers.append(W)
        names.append(layer_meta.get("name", f"layer_{len(layers)-1}.csv"))
    return layers, names


def save_csv_layers(layers, names, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    paths = []
    for i, W in enumerate(layers):
        name = names[i] if i < len(names) else f"layer_{i}.csv"
        if not name.lower().endswith(".csv"):
            name = f"{Path(name).stem}.csv"
        out = os.path.join(output_folder, name)
        np.savetxt(out, W, delimiter=",", fmt="%.10g")
        paths.append(out)
    return paths


def write_vtp(layers, names, output_path, x_spacing=1.0, y_spacing=1.0, z_spacing=2.0):
    points = []
    values = []
    layer_ids = []
    row_ids = []
    col_ids = []
    conn = []
    offs = []
    pid = 0
    for li, W in enumerate(layers):
        rows, cols = W.shape
        for i in range(rows):
            for j in range(cols):
                x = j * x_spacing
                y = i * y_spacing
                z = float(W[i, j]) + li * z_spacing
                points.append((x, y, z))
                values.append(float(W[i, j]))
                layer_ids.append(li); row_ids.append(i); col_ids.append(j)
                conn.append(pid); offs.append(pid+1); pid += 1

    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    poly = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(poly, "Piece", NumberOfPoints=str(len(points)), NumberOfVerts=str(len(points)), NumberOfLines="0", NumberOfStrips="0", NumberOfPolys="0")
    pd = ET.SubElement(piece, "PointData", Scalars="original_value")
    def arr(name, vals, typ="Float64"):
        a = ET.SubElement(pd, "DataArray", type=typ, Name=name, NumberOfComponents="1", format="ascii")
        a.text = " ".join(str(v) for v in vals)
    arr("original_value", [f"{v:.17g}" for v in values], "Float64")
    arr("layer_index", layer_ids, "Int32")
    arr("row_index", row_ids, "Int32")
    arr("col_index", col_ids, "Int32")
    fd = ET.SubElement(piece, "FieldData")
    na = ET.SubElement(fd, "DataArray", type="String", Name="layer_names", NumberOfTuples=str(len(names)), format="ascii")
    na.text = "\n".join(names)
    sa = ET.SubElement(fd, "DataArray", type="String", Name="layer_shapes", NumberOfTuples=str(len(layers)), format="ascii")
    sa.text = "\n".join(f"{W.shape[0]},{W.shape[1]}" for W in layers)
    ET.SubElement(piece, "CellData")
    pts = ET.SubElement(piece, "Points")
    pa = ET.SubElement(pts, "DataArray", type="Float64", NumberOfComponents="3", format="ascii")
    pa.text = " ".join(f"{x:.17g} {y:.17g} {z:.17g}" for x,y,z in points)
    verts = ET.SubElement(piece, "Verts")
    ca = ET.SubElement(verts, "DataArray", type="Int32", Name="connectivity", format="ascii")
    ca.text = " ".join(str(v) for v in conn)
    oa = ET.SubElement(verts, "DataArray", type="Int32", Name="offsets", format="ascii")
    oa.text = " ".join(str(v) for v in offs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description="Decode IWCH hologram images back to approximate matrices.")
    ap.add_argument("input", help="Base preview image, e.g. output.png")
    ap.add_argument("output_folder", help="Folder for reconstructed CSV layers")
    ap.add_argument("--vtp", default=None, help="Optional reconstructed VTP output path")
    ap.add_argument("--x_spacing", type=float, default=1.0)
    ap.add_argument("--y_spacing", type=float, default=1.0)
    ap.add_argument("--z_spacing", type=float, default=2.0)
    args = ap.parse_args()

    scalars, meta = recover_scalars(args.input)
    layers, names = reconstruct_layers(scalars, meta)
    paths = save_csv_layers(layers, names, args.output_folder)
    print(f"Decoded {len(paths)} layer(s) to {args.output_folder}")
    if args.vtp:
        write_vtp(layers, names, args.vtp, args.x_spacing, args.y_spacing, args.z_spacing)
        print(f"Saved reconstructed VTP: {args.vtp}")
    print(f"Hologram size: {meta['holo_height']}x{meta['holo_width']}")
    print(f"Encoding: {meta.get('encoding')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
