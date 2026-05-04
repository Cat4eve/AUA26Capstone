#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


def parse_ints(text):
    return np.fromstring((text or '').strip(), sep=' ', dtype=np.int64)


def parse_floats(text):
    return np.fromstring((text or '').strip(), sep=' ', dtype=np.float64)


def read_vtp_matrices(vtp_path):
    tree = ET.parse(vtp_path)
    root = tree.getroot()
    piece = root.find('.//Piece')
    if piece is None:
        raise ValueError('Invalid VTP: missing Piece element')

    point_data = piece.find('PointData')
    if point_data is None:
        raise ValueError('Invalid VTP: missing PointData')

    arrays = {}
    for arr in point_data.findall('DataArray'):
        name = arr.attrib.get('Name', '')
        if arr.attrib.get('type', '').startswith('Int'):
            arrays[name] = parse_ints(arr.text)
        else:
            arrays[name] = parse_floats(arr.text)

    required = {'original_value', 'layer_index', 'row_index', 'col_index'}
    if not required.issubset(arrays):
        missing = sorted(required - set(arrays))
        raise ValueError(f'VTP missing required arrays: {missing}')

    vals = arrays['original_value']
    layers = arrays['layer_index']
    rows = arrays['row_index']
    cols = arrays['col_index']

    matrices = []
    names = []
    field_data = piece.find('FieldData')
    if field_data is not None:
        for arr in field_data.findall('DataArray'):
            if arr.attrib.get('Name') == 'layer_names' and arr.text:
                names = [line.strip() for line in arr.text.splitlines() if line.strip()]

    for l in range(int(layers.max()) + 1 if layers.size else 0):
        mask = layers == l
        if not np.any(mask):
            continue
        rmax = int(rows[mask].max()) + 1
        cmax = int(cols[mask].max()) + 1
        mat = np.zeros((rmax, cmax), dtype=np.float64)
        mat[rows[mask], cols[mask]] = vals[mask]
        matrices.append(mat)

    if not names:
        names = [f'layer_{i}.csv' for i in range(len(matrices))]
    return matrices, names


def quantize_to_uint16(arr):
    arr = np.asarray(arr, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-15:
        q = np.zeros(arr.shape, dtype=np.uint16)
    else:
        q = np.round((arr - mn) / (mx - mn) * 65535.0).astype(np.uint16)
    return q, mn, mx


def save_uint16_png(arr, path):
    Image.fromarray(np.asarray(arr, dtype=np.uint16), mode='I;16').save(path)


def save_preview_phase(field, path):
    phase = np.angle(field)
    img = np.round((phase + np.pi) / (2 * np.pi) * 255.0).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path)


def encode_metadata_png(metadata, path, width=256):
    payload = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    header = len(payload).to_bytes(4, byteorder='little', signed=False)
    buf = header + payload
    height = math.ceil(len(buf) / width)
    arr = np.zeros((height, width), dtype=np.uint8)
    flat = arr.ravel()
    flat[:len(buf)] = np.frombuffer(buf, dtype=np.uint8)
    Image.fromarray(arr, mode='L').save(path)


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
    return arr[r0:r0 + out_h, c0:c0 + out_w]


def encode_layers_to_field(matrices, holo_h, holo_w):
    if not matrices:
        raise ValueError('No layers found')

    layer_count = len(matrices)
    grid_rows, grid_cols = choose_grid(layer_count)
    tile_h = holo_h // grid_rows
    tile_w = holo_w // grid_cols

    if tile_h < 2 or tile_w < 2:
        raise ValueError('Hologram size too small for number of layers')

    spectrum_canvas = np.zeros((holo_h, holo_w), dtype=np.complex128)
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
        tile = center_crop(F, tile_h, tile_w)

        gr = idx // grid_cols
        gc = idx % grid_cols
        r0 = gr * tile_h
        c0 = gc * tile_w
        spectrum_canvas[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] = tile

        layer_meta.append({
            'index': idx,
            'orig_rows': int(m),
            'orig_cols': int(n),
            'min': mn,
            'max': mx,
            'tile_row': int(gr),
            'tile_col': int(gc),
            'tile_h': int(tile.shape[0]),
            'tile_w': int(tile.shape[1]),
        })

    field = np.fft.ifft2(spectrum_canvas)

    meta = {
        'version': 1,
        'encoding': 'fixed_size_low_frequency_fft_tiling',
        'layer_count': layer_count,
        'holo_h': int(holo_h),
        'holo_w': int(holo_w),
        'grid_rows': int(grid_rows),
        'grid_cols': int(grid_cols),
        'layers': layer_meta,
    }
    return field, meta


def main():
    parser = argparse.ArgumentParser(description='Encode VTP point cloud into a smaller fixed-size 2D hologram field.')
    parser.add_argument('input_vtp')
    parser.add_argument('output_image')
    parser.add_argument('--holo_height', type=int, default=1024, help='Fixed hologram height')
    parser.add_argument('--holo_width', type=int, default=1024, help='Fixed hologram width')
    args = parser.parse_args()

    if not os.path.exists(args.input_vtp):
        raise FileNotFoundError(f'Input VTP not found: {args.input_vtp}')

    matrices, names = read_vtp_matrices(args.input_vtp)
    field, meta = encode_layers_to_field(matrices, args.holo_height, args.holo_width)
    meta['layer_names'] = names

    real_img, real_min, real_max = quantize_to_uint16(np.real(field))
    imag_img, imag_min, imag_max = quantize_to_uint16(np.imag(field))
    meta['real_min'] = real_min
    meta['real_max'] = real_max
    meta['imag_min'] = imag_min
    meta['imag_max'] = imag_max

    base = str(Path(args.output_image).with_suffix(''))
    preview_path = args.output_image
    real_path = base + '_real.png'
    imag_path = base + '_imag.png'
    meta_path = base + '_meta.png'

    save_preview_phase(field, preview_path)
    save_uint16_png(real_img, real_path)
    save_uint16_png(imag_img, imag_path)
    encode_metadata_png(meta, meta_path)

    print(f'Hologram preview: {preview_path}')
    print(f'Real image:       {real_path}')
    print(f'Imag image:       {imag_path}')
    print(f'Metadata image:   {meta_path}')
    print(f'Original layers:  {len(matrices)}')
    print(f'Fixed hologram:   {args.holo_height} x {args.holo_width}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
