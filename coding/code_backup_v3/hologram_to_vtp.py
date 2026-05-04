#!/usr/bin/env python3
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


def load_uint16_png(path):
    return np.array(Image.open(path), dtype=np.uint16)


def save_matrix_csvs(layers, names, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    paths = []
    for idx, layer in enumerate(layers):
        name = names[idx] if idx < len(names) else f'layer_{idx}.csv'
        if not name.lower().endswith('.csv'):
            name = f'{Path(name).stem}.csv'
        path = os.path.join(output_folder, name)
        np.savetxt(path, layer, delimiter=',', fmt='%.10g')
        paths.append(path)
    return paths


def decode_metadata_png(path):
    arr = np.array(Image.open(path), dtype=np.uint8).ravel()
    if arr.size < 4:
        raise ValueError('Metadata image too small')
    length = int.from_bytes(bytes(arr[:4].tolist()), byteorder='little', signed=False)
    payload = bytes(arr[4:4 + length].tolist())
    return json.loads(payload.decode('utf-8'))


def dequantize_uint16(q, mn, mx):
    q = np.asarray(q, dtype=np.float64)
    if mx - mn < 1e-15:
        return np.full(q.shape, mn, dtype=np.float64)
    return q / 65535.0 * (mx - mn) + mn


def paste_center(full_shape, tile):
    out = np.zeros(full_shape, dtype=np.complex128)
    h, w = full_shape
    th, tw = tile.shape
    th = min(th, h)
    tw = min(tw, w)
    r0 = (h - th) // 2
    c0 = (w - tw) // 2
    out[r0:r0 + th, c0:c0 + tw] = tile[:th, :tw]
    return out


def write_vtp(matrices, names, output_path, x_spacing=1.0, y_spacing=1.0, z_spacing=2.0):
    points = []
    connectivity = []
    offsets = []
    original_value = []
    layer_index = []
    row_index = []
    col_index = []
    point_id = 0

    for l, mat in enumerate(matrices):
        rows, cols = mat.shape
        for i in range(rows):
            y = i * y_spacing
            for j in range(cols):
                x = j * x_spacing
                z = float(mat[i, j]) + l * z_spacing
                points.append((x, y, z))
                connectivity.append(point_id)
                offsets.append(point_id + 1)
                original_value.append(float(mat[i, j]))
                layer_index.append(l)
                row_index.append(i)
                col_index.append(j)
                point_id += 1

    root = ET.Element('VTKFile', type='PolyData', version='0.1', byte_order='LittleEndian')
    poly = ET.SubElement(root, 'PolyData')
    piece = ET.SubElement(poly, 'Piece', NumberOfPoints=str(len(points)), NumberOfVerts=str(len(points)),
                          NumberOfLines='0', NumberOfStrips='0', NumberOfPolys='0')
    pdata = ET.SubElement(piece, 'PointData', Scalars='original_value')

    def add_pdata(name, text, dtype='Float64'):
        arr = ET.SubElement(pdata, 'DataArray', type=dtype, Name=name,
                            NumberOfComponents='1', format='ascii')
        arr.text = text

    add_pdata('original_value', ' '.join(f'{v:.17g}' for v in original_value))
    add_pdata('layer_index', ' '.join(str(v) for v in layer_index), dtype='Int32')
    add_pdata('row_index', ' '.join(str(v) for v in row_index), dtype='Int32')
    add_pdata('col_index', ' '.join(str(v) for v in col_index), dtype='Int32')

    field = ET.SubElement(piece, 'FieldData')
    ET.SubElement(field, 'DataArray', type='String', Name='layer_names',
                  NumberOfTuples=str(len(names)), format='ascii').text = '\n'.join(names)
    ET.SubElement(field, 'DataArray', type='String', Name='layer_shapes',
                  NumberOfTuples=str(len(matrices)), format='ascii').text = '\n'.join(
        f'{m.shape[0]},{m.shape[1]}' for m in matrices
    )
    ET.SubElement(field, 'DataArray', type='String', Name='spacing',
                  NumberOfTuples='1', format='ascii').text = f'{x_spacing},{y_spacing},{z_spacing}'

    cell = ET.SubElement(piece, 'CellData')
    ET.SubElement(cell, 'DataArray', type='Int32', Name='dummy', format='ascii').text = ' '

    pts = ET.SubElement(piece, 'Points')
    ET.SubElement(pts, 'DataArray', type='Float64', NumberOfComponents='3', format='ascii').text = ' '.join(
        f'{x:.17g} {y:.17g} {z:.17g}' for x, y, z in points
    )

    verts = ET.SubElement(piece, 'Verts')
    ET.SubElement(verts, 'DataArray', type='Int32', Name='connectivity', format='ascii').text = ' '.join(
        str(v) for v in connectivity
    )
    ET.SubElement(verts, 'DataArray', type='Int32', Name='offsets', format='ascii').text = ' '.join(
        str(v) for v in offsets
    )

    ET.ElementTree(root).write(output_path, encoding='utf-8', xml_declaration=True)


def decode_layers(base_image_path):
    base = str(Path(base_image_path).with_suffix(''))
    real_path = base + '_real.png'
    imag_path = base + '_imag.png'
    meta_path = base + '_meta.png'

    for p in (real_path, imag_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing required file: {p}')

    meta = decode_metadata_png(meta_path)
    real = dequantize_uint16(load_uint16_png(real_path), meta['real_min'], meta['real_max'])
    imag = dequantize_uint16(load_uint16_png(imag_path), meta['imag_min'], meta['imag_max'])

    field = real + 1j * imag
    spectrum = np.fft.fft2(field)

    layers = []
    names = meta.get('layer_names', [])
    for layer_meta in meta['layers']:
        m = int(layer_meta['orig_rows'])
        n = int(layer_meta['orig_cols'])
        gr = int(layer_meta['tile_row'])
        gc = int(layer_meta['tile_col'])
        th = int(layer_meta['tile_h'])
        tw = int(layer_meta['tile_w'])

        r0 = gr * th
        c0 = gc * tw
        tile = spectrum[r0:r0 + th, c0:c0 + tw]

        full_spec = paste_center((m, n), tile)
        recon_norm = np.real(np.fft.ifft2(np.fft.ifftshift(full_spec)))
        recon_norm = np.clip(recon_norm, 0.0, 1.0)

        mn = float(layer_meta['min'])
        mx = float(layer_meta['max'])
        if mx - mn < 1e-15:
            recon = np.full((m, n), mn, dtype=np.float64)
        else:
            recon = recon_norm * (mx - mn) + mn
        layers.append(recon)

    return layers, names, meta


def main():
    parser = argparse.ArgumentParser(description='Decode fixed-size hologram images back to approximate matrix layers.')
    parser.add_argument('input_hologram', help='Base hologram preview image, e.g. output.png')
    parser.add_argument('output_folder', help='Folder to save reconstructed CSV layers')
    parser.add_argument('--vtp', default=None, help='Optional reconstructed VTP output path')
    parser.add_argument('--x_spacing', type=float, default=1.0)
    parser.add_argument('--y_spacing', type=float, default=1.0)
    parser.add_argument('--z_spacing', type=float, default=2.0)
    args = parser.parse_args()

    if not os.path.exists(args.input_hologram):
        raise FileNotFoundError(f'Input hologram not found: {args.input_hologram}')

    layers, names, meta = decode_layers(args.input_hologram)
    paths = save_matrix_csvs(layers, names, args.output_folder)

    if args.vtp:
        write_vtp(layers, [Path(p).name for p in paths],
                  args.vtp, args.x_spacing, args.y_spacing, args.z_spacing)

    print(f'Decoded {len(layers)} layer(s)')
    print(f'CSV output folder: {args.output_folder}')
    if args.vtp:
        print(f'Reconstructed VTP: {args.vtp}')
    print(f'Encoding mode: {meta.get("encoding", "unknown")}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
