#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def load_matrix(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        mat = np.load(file_path)
    elif ext in ('.csv', '.txt'):
        try:
            mat = np.loadtxt(file_path, delimiter=',')
        except ValueError:
            mat = np.loadtxt(file_path)
    else:
        raise ValueError(f'Unsupported file format: {ext}')
    if mat.ndim != 2:
        raise ValueError('Input must contain a 2D matrix')
    return np.asarray(mat, dtype=np.float64)


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

    def add_pdata(name, ncomp, text, dtype='Float64'):
        arr = ET.SubElement(pdata, 'DataArray', type=dtype, Name=name,
                            NumberOfComponents=str(ncomp), format='ascii')
        arr.text = text

    add_pdata('original_value', 1, ' '.join(f'{v:.17g}' for v in original_value))
    add_pdata('layer_index', 1, ' '.join(str(v) for v in layer_index), dtype='Int32')
    add_pdata('row_index', 1, ' '.join(str(v) for v in row_index), dtype='Int32')
    add_pdata('col_index', 1, ' '.join(str(v) for v in col_index), dtype='Int32')

    field = ET.SubElement(piece, 'FieldData')
    names_arr = ET.SubElement(field, 'DataArray', type='String', Name='layer_names',
                              NumberOfTuples=str(len(names)), format='ascii')
    names_arr.text = '\n'.join(names)

    shape_arr = ET.SubElement(field, 'DataArray', type='String', Name='layer_shapes',
                              NumberOfTuples=str(len(matrices)), format='ascii')
    shape_arr.text = '\n'.join(f'{mat.shape[0]},{mat.shape[1]}' for mat in matrices)

    spacing_arr = ET.SubElement(field, 'DataArray', type='String', Name='spacing',
                                NumberOfTuples='1', format='ascii')
    spacing_arr.text = f'{x_spacing},{y_spacing},{z_spacing}'

    cell = ET.SubElement(piece, 'CellData')
    ET.SubElement(cell, 'DataArray', type='Int32', Name='dummy', format='ascii').text = ' '

    pts = ET.SubElement(piece, 'Points')
    pts_arr = ET.SubElement(pts, 'DataArray', type='Float64', NumberOfComponents='3', format='ascii')
    pts_arr.text = ' '.join(f'{x:.17g} {y:.17g} {z:.17g}' for x, y, z in points)

    verts = ET.SubElement(piece, 'Verts')
    conn = ET.SubElement(verts, 'DataArray', type='Int32', Name='connectivity', format='ascii')
    conn.text = ' '.join(str(v) for v in connectivity)
    offs = ET.SubElement(verts, 'DataArray', type='Int32', Name='offsets', format='ascii')
    offs.text = ' '.join(str(v) for v in offsets)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description='Convert matrices in a folder into a VTP point cloud.')
    parser.add_argument('input_folder')
    parser.add_argument('output_file')
    parser.add_argument('--x_spacing', type=float, default=1.0)
    parser.add_argument('--y_spacing', type=float, default=1.0)
    parser.add_argument('--z_spacing', type=float, default=2.0)
    args = parser.parse_args()

    folder = Path(args.input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Input folder not found: {folder}')

    files = sorted([p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.csv', '.txt', '.npy'}])
    if not files:
        raise FileNotFoundError(f'No matrix files found in {folder}')

    matrices = []
    names = []
    for p in files:
        matrices.append(load_matrix(str(p)))
        names.append(p.name)
        print(f'Loaded {p.name} shape={matrices[-1].shape}')

    write_vtp(matrices, names, args.output_file, args.x_spacing, args.y_spacing, args.z_spacing)
    print(f'Wrote VTP: {args.output_file}')
    print(f'Points: {sum(m.size for m in matrices)}')


if __name__ == '__main__':
    main()
