#!/usr/bin/env python3
"""
matrix_group_holography_pipeline.py

Complete test pipeline:

    group of matrix files
        -> grouped VTP point cloud
        -> hologram PNG files
        -> decoded matrix CSV files
        -> reconstructed grouped VTP point cloud

This is matrix-aware. The grouped VTP keeps:
    original_value, layer_index, row_index, col_index

The hologram encoder reads those arrays from the VTP, encodes the matrix values,
and the decoder reconstructs matrix CSVs and a grouped VTP again.

Geometry rule for VTP:
    x = col_index * x_spacing + layer_x_offset
    y = row_index * y_spacing + layer_y_offset
    z = matrix[row, col]

So height is the matrix value, and matrices are placed near each other
in the same x-y plane with a minimum block offset/gap of at least 2.
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


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy"}


# ============================================================
# Matrix loading/saving
# ============================================================

def load_matrix(file_path):
    file_path = str(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        mat = np.load(file_path)
    elif ext in {".csv", ".txt"}:
        try:
            mat = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            mat = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if mat.ndim != 2:
        raise ValueError(f"Input must contain a 2D matrix: {file_path}")

    return np.asarray(mat, dtype=np.float64)


def load_matrices_from_folder(input_folder):
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No matrix files found in {folder}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    matrices = []
    names = []

    for p in files:
        mat = load_matrix(p)
        matrices.append(mat)
        names.append(p.name)
        print(f"Loaded {p.name}: shape={mat.shape}, min={mat.min():.6g}, max={mat.max():.6g}")

    return matrices, names


def save_matrix_csvs(matrices, names, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    paths = []
    for idx, mat in enumerate(matrices):
        if idx < len(names):
            name = Path(names[idx]).stem + "_decoded.csv"
        else:
            name = f"layer_{idx}_decoded.csv"

        path = output_folder / name
        np.savetxt(path, mat, delimiter=",", fmt="%.10g")
        paths.append(path)

    return paths


# ============================================================
# VTP grouped writer
# ============================================================

def matrix_span(mat, x_spacing, y_spacing):
    rows, cols = mat.shape
    x_span = max(cols - 1, 0) * x_spacing
    y_span = max(rows - 1, 0) * y_spacing
    return x_span, y_span


def compute_layer_offsets(
    matrices,
    x_spacing=1.0,
    y_spacing=1.0,
    block_gap=2.0,
    layout="grid",
    grid_cols=None,
):
    block_gap = max(float(block_gap), 2.0)

    spans = [matrix_span(m, x_spacing, y_spacing) for m in matrices]
    n = len(matrices)
    offsets = []

    if layout == "row":
        current_x = 0.0
        for x_span, _ in spans:
            offsets.append((current_x, 0.0))
            current_x += x_span + block_gap

    elif layout == "column":
        current_y = 0.0
        for _, y_span in spans:
            offsets.append((0.0, current_y))
            current_y += y_span + block_gap

    elif layout == "grid":
        if grid_cols is None:
            grid_cols = math.ceil(math.sqrt(n))
        grid_cols = max(1, int(grid_cols))

        max_x_span = max(x for x, y in spans)
        max_y_span = max(y for x, y in spans)

        cell_w = max_x_span + block_gap
        cell_h = max_y_span + block_gap

        for idx in range(n):
            r = idx // grid_cols
            c = idx % grid_cols
            offsets.append((c * cell_w, r * cell_h))

    else:
        raise ValueError("layout must be one of: grid, row, column")

    return offsets, block_gap


def write_grouped_vtp(
    matrices,
    names,
    output_path,
    x_spacing=1.0,
    y_spacing=1.0,
    block_gap=2.0,
    layout="grid",
    grid_cols=None,
):
    offsets_xy, actual_gap = compute_layer_offsets(
        matrices=matrices,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        block_gap=block_gap,
        layout=layout,
        grid_cols=grid_cols,
    )

    points = []
    connectivity = []
    vertex_offsets = []

    original_value = []
    layer_index = []
    row_index = []
    col_index = []

    layer_x_offset = []
    layer_y_offset = []
    local_x = []
    local_y = []

    point_id = 0

    for l, mat in enumerate(matrices):
        rows, cols = mat.shape
        ox, oy = offsets_xy[l]

        for i in range(rows):
            y_local = i * y_spacing
            y = y_local + oy

            for j in range(cols):
                x_local = j * x_spacing
                x = x_local + ox
                value = float(mat[i, j])
                z = value

                points.append((x, y, z))
                connectivity.append(point_id)
                vertex_offsets.append(point_id + 1)

                original_value.append(value)
                layer_index.append(l)
                row_index.append(i)
                col_index.append(j)

                layer_x_offset.append(ox)
                layer_y_offset.append(oy)
                local_x.append(x_local)
                local_y.append(y_local)

                point_id += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    poly = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfVerts=str(len(points)),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )

    pdata = ET.SubElement(piece, "PointData", Scalars="original_value")

    def add_pdata(name, values, dtype="Float64"):
        arr = ET.SubElement(
            pdata,
            "DataArray",
            type=dtype,
            Name=name,
            NumberOfComponents="1",
            format="ascii",
        )
        if dtype.startswith("Int"):
            arr.text = " ".join(str(int(v)) for v in values)
        else:
            arr.text = " ".join(f"{float(v):.17g}" for v in values)

    add_pdata("original_value", original_value, "Float64")
    add_pdata("layer_index", layer_index, "Int32")
    add_pdata("row_index", row_index, "Int32")
    add_pdata("col_index", col_index, "Int32")

    add_pdata("layer_x_offset", layer_x_offset, "Float64")
    add_pdata("layer_y_offset", layer_y_offset, "Float64")
    add_pdata("local_x", local_x, "Float64")
    add_pdata("local_y", local_y, "Float64")

    field = ET.SubElement(piece, "FieldData")

    def add_field_string(name, text, tuples="1"):
        arr = ET.SubElement(
            field,
            "DataArray",
            type="String",
            Name=name,
            NumberOfTuples=str(tuples),
            format="ascii",
        )
        arr.text = text

    add_field_string("layer_names", "\n".join(names), len(names))
    add_field_string(
        "layer_shapes",
        "\n".join(f"{m.shape[0]},{m.shape[1]}" for m in matrices),
        len(matrices),
    )
    add_field_string("matrix_to_vtp_mode", "grouped_xy_offsets_z_equals_value")
    add_field_string("spacing", f"x_spacing={x_spacing},y_spacing={y_spacing},block_gap={actual_gap}")
    add_field_string("layout", f"layout={layout},grid_cols={grid_cols if grid_cols is not None else 'auto'}")
    add_field_string(
        "layer_offsets_xy",
        "\n".join(f"{ox:.17g},{oy:.17g}" for ox, oy in offsets_xy),
        len(offsets_xy),
    )

    cell = ET.SubElement(piece, "CellData")
    ET.SubElement(cell, "DataArray", type="Int32", Name="dummy", format="ascii").text = " "

    pts = ET.SubElement(piece, "Points")
    pts_arr = ET.SubElement(
        pts,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    pts_arr.text = " ".join(f"{x:.17g} {y:.17g} {z:.17g}" for x, y, z in points)

    verts = ET.SubElement(piece, "Verts")
    conn = ET.SubElement(verts, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(v) for v in connectivity)
    offs = ET.SubElement(verts, "DataArray", type="Int32", Name="offsets", format="ascii")
    offs.text = " ".join(str(v) for v in vertex_offsets)

    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)

    return {
        "path": str(output_path),
        "points": len(points),
        "layers": len(matrices),
        "offsets_xy": offsets_xy,
        "block_gap": actual_gap,
    }


# ============================================================
# VTP matrix reader
# ============================================================

def parse_ints(text):
    return np.fromstring((text or "").strip(), sep=" ", dtype=np.int64)


def parse_floats(text):
    return np.fromstring((text or "").strip(), sep=" ", dtype=np.float64)


def read_vtp_matrices(vtp_path):
    tree = ET.parse(vtp_path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("Invalid VTP: missing Piece element")

    point_data = piece.find("PointData")
    if point_data is None:
        raise ValueError("Invalid VTP: missing PointData")

    arrays = {}
    for arr in point_data.findall("DataArray"):
        name = arr.attrib.get("Name", "")
        dtype = arr.attrib.get("type", "")
        if dtype.startswith("Int"):
            arrays[name] = parse_ints(arr.text)
        else:
            arrays[name] = parse_floats(arr.text)

    required = {"original_value", "layer_index", "row_index", "col_index"}
    if not required.issubset(arrays):
        missing = sorted(required - set(arrays))
        raise ValueError(
            "This encoder expects a matrix-aware VTP. "
            f"Missing arrays: {missing}"
        )

    vals = arrays["original_value"]
    layers = arrays["layer_index"]
    rows = arrays["row_index"]
    cols = arrays["col_index"]

    names = []
    field_data = piece.find("FieldData")
    if field_data is not None:
        for arr in field_data.findall("DataArray"):
            if arr.attrib.get("Name") == "layer_names" and arr.text:
                names = [line.strip() for line in arr.text.splitlines() if line.strip()]

    matrices = []
    max_layer = int(layers.max()) if layers.size else -1

    for l in range(max_layer + 1):
        mask = layers == l
        if not np.any(mask):
            continue

        rmax = int(rows[mask].max()) + 1
        cmax = int(cols[mask].max()) + 1
        mat = np.zeros((rmax, cmax), dtype=np.float64)
        mat[rows[mask], cols[mask]] = vals[mask]
        matrices.append(mat)

    if not names or len(names) != len(matrices):
        names = [f"layer_{i}.csv" for i in range(len(matrices))]

    return matrices, names


# ============================================================
# Hologram PNG helpers
# ============================================================

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
    img = np.round((phase + np.pi) / (2 * np.pi) * 255.0).astype(np.uint8)
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
        raise ValueError("Metadata image too small")

    length = int.from_bytes(bytes(arr[:4].tolist()), byteorder="little", signed=False)
    payload = bytes(arr[4 : 4 + length].tolist())
    return json.loads(payload.decode("utf-8"))


# ============================================================
# Matrix-aware hologram encoding/decoding
# ============================================================

def choose_grid(num_layers):
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols


def center_crop(arr, out_h, out_w):
    h, w = arr.shape
    out_h = min(out_h, h)
    out_w = min(out_w, w)
    r0 = (h - out_h) // 2
    c0 = (w - out_w) // 2
    return arr[r0 : r0 + out_h, c0 : c0 + out_w]


def paste_center(full_shape, tile):
    out = np.zeros(full_shape, dtype=np.complex128)
    h, w = full_shape
    th, tw = tile.shape

    th = min(th, h)
    tw = min(tw, w)

    r0 = (h - th) // 2
    c0 = (w - tw) // 2
    out[r0 : r0 + th, c0 : c0 + tw] = tile[:th, :tw]
    return out


def encode_layers_to_hologram_field(matrices, holo_height, holo_width):
    if not matrices:
        raise ValueError("No matrices to encode")

    layer_count = len(matrices)
    grid_rows, grid_cols = choose_grid(layer_count)

    cell_h = holo_height // grid_rows
    cell_w = holo_width // grid_cols

    if cell_h < 2 or cell_w < 2:
        raise ValueError("Hologram size too small for the number of layers")

    spectrum_canvas = np.zeros((holo_height, holo_width), dtype=np.complex128)
    layer_meta = []

    for idx, mat in enumerate(matrices):
        mat = np.asarray(mat, dtype=np.float64)
        m, n = mat.shape

        mn = float(mat.min())
        mx = float(mat.max())

        if mx - mn < 1e-15:
            norm = np.zeros_like(mat)
        else:
            norm = (mat - mn) / (mx - mn)

        F = np.fft.fftshift(np.fft.fft2(norm))
        tile = center_crop(F, cell_h, cell_w)

        gr = idx // grid_cols
        gc = idx % grid_cols
        canvas_r0 = gr * cell_h
        canvas_c0 = gc * cell_w

        spectrum_canvas[
            canvas_r0 : canvas_r0 + tile.shape[0],
            canvas_c0 : canvas_c0 + tile.shape[1],
        ] = tile

        layer_meta.append({
            "index": int(idx),
            "orig_rows": int(m),
            "orig_cols": int(n),
            "min": mn,
            "max": mx,
            "tile_row": int(gr),
            "tile_col": int(gc),
            "cell_h": int(cell_h),
            "cell_w": int(cell_w),
            "canvas_r0": int(canvas_r0),
            "canvas_c0": int(canvas_c0),
            "tile_h": int(tile.shape[0]),
            "tile_w": int(tile.shape[1]),
        })

    field = np.fft.ifft2(spectrum_canvas)

    meta = {
        "version": 3,
        "encoding": "matrix_group_vtp_low_frequency_fft_tiling",
        "layer_count": int(layer_count),
        "holo_height": int(holo_height),
        "holo_width": int(holo_width),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "layers": layer_meta,
    }

    return field, meta


def encode_vtp_to_hologram(input_vtp, output_hologram, holo_height=1024, holo_width=1024):
    matrices, names = read_vtp_matrices(input_vtp)

    field, meta = encode_layers_to_hologram_field(matrices, holo_height, holo_width)
    meta["layer_names"] = names
    meta["source_vtp"] = os.path.basename(str(input_vtp))

    real_img, real_min, real_max = quantize_to_uint16(np.real(field))
    imag_img, imag_min, imag_max = quantize_to_uint16(np.imag(field))

    meta["real_min"] = real_min
    meta["real_max"] = real_max
    meta["imag_min"] = imag_min
    meta["imag_max"] = imag_max

    base = str(Path(output_hologram).with_suffix(""))
    preview_path = output_hologram
    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"

    save_phase_preview(field, preview_path)
    save_uint16_png(real_img, real_path)
    save_uint16_png(imag_img, imag_path)
    encode_metadata_png(meta, meta_path)

    print("Encoded grouped VTP into hologram")
    print(f"Input VTP:      {input_vtp}")
    print(f"Preview image:  {preview_path}")
    print(f"Real image:     {real_path}")
    print(f"Imag image:     {imag_path}")
    print(f"Metadata image: {meta_path}")
    print(f"Layers:         {len(matrices)}")
    print(f"Hologram size:  {holo_height} x {holo_width}")

    return {
        "preview": preview_path,
        "real": real_path,
        "imag": imag_path,
        "meta": meta_path,
    }


def decode_hologram_to_layers(input_hologram):
    base = str(Path(input_hologram).with_suffix(""))
    real_path = base + "_real.png"
    imag_path = base + "_imag.png"
    meta_path = base + "_meta.png"

    for p in [real_path, imag_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required companion file: {p}")

    meta = decode_metadata_png(meta_path)
    real = dequantize_uint16(load_uint16_png(real_path), meta["real_min"], meta["real_max"])
    imag = dequantize_uint16(load_uint16_png(imag_path), meta["imag_min"], meta["imag_max"])

    field = real + 1j * imag
    spectrum_canvas = np.fft.fft2(field)

    layers = []
    for layer_meta in meta["layers"]:
        m = int(layer_meta["orig_rows"])
        n = int(layer_meta["orig_cols"])

        canvas_r0 = int(layer_meta["canvas_r0"])
        canvas_c0 = int(layer_meta["canvas_c0"])
        th = int(layer_meta["tile_h"])
        tw = int(layer_meta["tile_w"])

        tile = spectrum_canvas[canvas_r0 : canvas_r0 + th, canvas_c0 : canvas_c0 + tw]
        full_spec = paste_center((m, n), tile)

        recon_norm = np.real(np.fft.ifft2(np.fft.ifftshift(full_spec)))
        recon_norm = np.clip(recon_norm, 0.0, 1.0)

        mn = float(layer_meta["min"])
        mx = float(layer_meta["max"])

        if mx - mn < 1e-15:
            recon = np.full((m, n), mn, dtype=np.float64)
        else:
            recon = recon_norm * (mx - mn) + mn

        layers.append(recon)

    names = meta.get("layer_names", [f"layer_{i}.csv" for i in range(len(layers))])
    return layers, names, meta


def decode_hologram_to_outputs(
    input_hologram,
    output_folder,
    output_vtp=None,
    x_spacing=1.0,
    y_spacing=1.0,
    block_gap=2.0,
    layout="grid",
    grid_cols=None,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    layers, names, meta = decode_hologram_to_layers(input_hologram)
    csv_dir = output_folder / "decoded_csv"
    csv_paths = save_matrix_csvs(layers, names, csv_dir)

    if output_vtp is None:
        output_vtp = output_folder / "decoded_grouped.vtp"

    write_grouped_vtp(
        matrices=layers,
        names=[Path(p).name for p in csv_paths],
        output_path=output_vtp,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        block_gap=block_gap,
        layout=layout,
        grid_cols=grid_cols,
    )

    print("Decoded hologram into matrices and grouped VTP")
    print(f"Decoded CSV folder: {csv_dir}")
    print(f"Decoded VTP:        {output_vtp}")

    return layers, names, meta


# ============================================================
# Metrics
# ============================================================

def compare_layers(original_layers, decoded_layers):
    rows = []
    for idx, (orig, dec) in enumerate(zip(original_layers, decoded_layers)):
        orig = np.asarray(orig, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)

        if orig.shape != dec.shape:
            rows.append({
                "layer": idx,
                "shape": f"{orig.shape} vs {dec.shape}",
                "mae": None,
                "rmse": None,
                "max_abs": None,
                "relative_rmse": None,
            })
            continue

        diff = dec - orig
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        denom = float(np.sqrt(np.mean(orig ** 2)))
        relative_rmse = float(rmse / denom) if denom > 1e-15 else 0.0

        rows.append({
            "layer": idx,
            "shape": str(orig.shape),
            "mae": mae,
            "rmse": rmse,
            "max_abs": max_abs,
            "relative_rmse": relative_rmse,
        })

    return rows


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("layer,shape,mae,rmse,max_abs,relative_rmse\n")
        for row in metrics:
            f.write(
                f"{row['layer']},{row['shape']},"
                f"{row['mae']},{row['rmse']},{row['max_abs']},{row['relative_rmse']}\n"
            )


# ============================================================
# Test data
# ============================================================

def create_test_matrices(output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    a = np.array([
        [0.0, 0.2, 0.4, 0.2],
        [0.1, 0.7, 1.0, 0.5],
        [0.0, 0.3, 0.6, 0.2],
    ])

    b = np.array([
        [1, 2, 3],
        [2, 4, 2],
        [3, 2, 1],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)

    x = np.linspace(-1.5, 1.5, 7)
    y = np.linspace(-1.5, 1.5, 7)
    xx, yy = np.meshgrid(x, y)
    c = np.exp(-(xx**2 + yy**2))

    np.savetxt(output_folder / "layer_0_small.csv", a, delimiter=",", fmt="%.6f")
    np.savetxt(output_folder / "layer_1_tall.csv", b, delimiter=",", fmt="%.6f")
    np.savetxt(output_folder / "layer_2_bell.csv", c, delimiter=",", fmt="%.6f")

    return output_folder


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete matrix group -> VTP -> hologram -> decoded matrices/VTP pipeline."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_vtp = sub.add_parser("matrices-to-vtp", help="Convert matrix folder to grouped VTP")
    p_vtp.add_argument("input_folder")
    p_vtp.add_argument("output_vtp")
    p_vtp.add_argument("--x_spacing", type=float, default=1.0)
    p_vtp.add_argument("--y_spacing", type=float, default=1.0)
    p_vtp.add_argument("--block_gap", type=float, default=2.0)
    p_vtp.add_argument("--layout", choices=["grid", "row", "column"], default="grid")
    p_vtp.add_argument("--grid_cols", type=int, default=None)

    p_enc = sub.add_parser("encode", help="Encode grouped matrix-aware VTP to hologram")
    p_enc.add_argument("input_vtp")
    p_enc.add_argument("output_hologram")
    p_enc.add_argument("--holo_height", type=int, default=1024)
    p_enc.add_argument("--holo_width", type=int, default=1024)

    p_dec = sub.add_parser("decode", help="Decode hologram to CSV layers and grouped VTP")
    p_dec.add_argument("input_hologram")
    p_dec.add_argument("output_folder")
    p_dec.add_argument("--output_vtp", default=None)
    p_dec.add_argument("--x_spacing", type=float, default=1.0)
    p_dec.add_argument("--y_spacing", type=float, default=1.0)
    p_dec.add_argument("--block_gap", type=float, default=2.0)
    p_dec.add_argument("--layout", choices=["grid", "row", "column"], default="grid")
    p_dec.add_argument("--grid_cols", type=int, default=None)

    p_rt = sub.add_parser("roundtrip", help="Run full pipeline from matrix folder to decoded outputs")
    p_rt.add_argument("input_folder")
    p_rt.add_argument("output_folder")
    p_rt.add_argument("--holo_height", type=int, default=1024)
    p_rt.add_argument("--holo_width", type=int, default=1024)
    p_rt.add_argument("--x_spacing", type=float, default=1.0)
    p_rt.add_argument("--y_spacing", type=float, default=1.0)
    p_rt.add_argument("--block_gap", type=float, default=2.0)
    p_rt.add_argument("--layout", choices=["grid", "row", "column"], default="grid")
    p_rt.add_argument("--grid_cols", type=int, default=None)

    p_test = sub.add_parser("make-test-data", help="Create sample matrices for testing")
    p_test.add_argument("output_folder")

    args = parser.parse_args()

    if args.command == "make-test-data":
        create_test_matrices(args.output_folder)
        print(f"Created test matrices in: {args.output_folder}")
        return

    if args.command == "matrices-to-vtp":
        matrices, names = load_matrices_from_folder(args.input_folder)
        info = write_grouped_vtp(
            matrices,
            names,
            args.output_vtp,
            x_spacing=args.x_spacing,
            y_spacing=args.y_spacing,
            block_gap=args.block_gap,
            layout=args.layout,
            grid_cols=args.grid_cols,
        )
        print("=" * 72)
        print("DONE: matrices -> grouped VTP")
        print("=" * 72)
        print(f"Output VTP: {info['path']}")
        print(f"Layers:     {info['layers']}")
        print(f"Points:     {info['points']}")
        return

    if args.command == "encode":
        encode_vtp_to_hologram(
            args.input_vtp,
            args.output_hologram,
            holo_height=args.holo_height,
            holo_width=args.holo_width,
        )
        return

    if args.command == "decode":
        decode_hologram_to_outputs(
            args.input_hologram,
            args.output_folder,
            output_vtp=args.output_vtp,
            x_spacing=args.x_spacing,
            y_spacing=args.y_spacing,
            block_gap=args.block_gap,
            layout=args.layout,
            grid_cols=args.grid_cols,
        )
        return

    if args.command == "roundtrip":
        output_folder = Path(args.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        original_vtp = output_folder / "original_grouped.vtp"
        hologram = output_folder / "encoded_hologram.png"
        decoded_vtp = output_folder / "decoded_grouped.vtp"

        matrices, names = load_matrices_from_folder(args.input_folder)

        print("=" * 72)
        print("STEP 1: matrices -> grouped VTP")
        print("=" * 72)
        write_grouped_vtp(
            matrices,
            names,
            original_vtp,
            x_spacing=args.x_spacing,
            y_spacing=args.y_spacing,
            block_gap=args.block_gap,
            layout=args.layout,
            grid_cols=args.grid_cols,
        )

        print("=" * 72)
        print("STEP 2: grouped VTP -> hologram")
        print("=" * 72)
        encode_vtp_to_hologram(
            original_vtp,
            hologram,
            holo_height=args.holo_height,
            holo_width=args.holo_width,
        )

        print("=" * 72)
        print("STEP 3: hologram -> decoded matrices + grouped VTP")
        print("=" * 72)
        decoded_layers, decoded_names, meta = decode_hologram_to_outputs(
            hologram,
            output_folder,
            output_vtp=decoded_vtp,
            x_spacing=args.x_spacing,
            y_spacing=args.y_spacing,
            block_gap=args.block_gap,
            layout=args.layout,
            grid_cols=args.grid_cols,
        )

        metrics = compare_layers(matrices, decoded_layers)
        metrics_path = output_folder / "reconstruction_metrics.csv"
        save_metrics(metrics, metrics_path)

        print("=" * 72)
        print("DONE: full roundtrip")
        print("=" * 72)
        print(f"Original grouped VTP: {original_vtp}")
        print(f"Hologram preview:     {hologram}")
        print(f"Hologram real:        {str(hologram.with_suffix(''))}_real.png")
        print(f"Hologram imag:        {str(hologram.with_suffix(''))}_imag.png")
        print(f"Hologram metadata:    {str(hologram.with_suffix(''))}_meta.png")
        print(f"Decoded VTP:          {decoded_vtp}")
        print(f"Decoded CSV folder:   {output_folder / 'decoded_csv'}")
        print(f"Metrics CSV:          {metrics_path}")

        print("\nReconstruction metrics:")
        for row in metrics:
            print(
                f"  layer {row['layer']} shape={row['shape']} "
                f"MAE={row['mae']:.6g} RMSE={row['rmse']:.6g} "
                f"max_abs={row['max_abs']:.6g} rel_RMSE={row['relative_rmse']:.6g}"
            )
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
