#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Full pipeline: matrix files -> VTP -> compressed fixed-size hologram images')
    parser.add_argument('input_folder', help='Folder containing CSV/TXT/NPY matrix layers')
    parser.add_argument('output_hologram', help='Main hologram preview image path, e.g. output.png')
    parser.add_argument('--vtp_file', default=None, help='Intermediate VTP path (default: <output>.vtp)')
    parser.add_argument('--x_spacing', type=float, default=1.0)
    parser.add_argument('--y_spacing', type=float, default=1.0)
    parser.add_argument('--z_spacing', type=float, default=2.0)
    parser.add_argument('--holo_height', type=int, default=1024, help='Fixed hologram height')
    parser.add_argument('--holo_width', type=int, default=1024, help='Fixed hologram width')
    parser.add_argument('--keep_intermediate', action='store_true', help='Keep the intermediate VTP file')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f'Input folder not found: {input_folder}')

    output_base = str(Path(args.output_hologram).with_suffix(''))
    vtp_file = args.vtp_file or f'{output_base}.vtp'

    here = Path(__file__).resolve().parent
    matrix_to_vtp = here / 'matrix_to_vtp.py'
    vtp_to_holo = here / 'vtp_to_hologram.py'

    step1 = [
        sys.executable, str(matrix_to_vtp), str(input_folder), vtp_file,
        '--x_spacing', str(args.x_spacing),
        '--y_spacing', str(args.y_spacing),
        '--z_spacing', str(args.z_spacing),
    ]

    step2 = [
        sys.executable, str(vtp_to_holo), vtp_file, args.output_hologram,
        '--holo_height', str(args.holo_height),
        '--holo_width', str(args.holo_width),
    ]

    print('=' * 72)
    print('STEP 1: matrices -> VTP')
    print('=' * 72)
    r1 = subprocess.run(step1)
    if r1.returncode != 0:
        raise RuntimeError('Step 1 failed')

    print('=' * 72)
    print('STEP 2: VTP -> compressed hologram images')
    print('=' * 72)
    r2 = subprocess.run(step2)
    if r2.returncode != 0:
        raise RuntimeError('Step 2 failed')

    if (not args.keep_intermediate) and os.path.exists(vtp_file):
        os.remove(vtp_file)
        print(f'Removed intermediate VTP: {vtp_file}')

    print('=' * 72)
    print('DONE')
    print('=' * 72)
    print(f'Main hologram preview: {args.output_hologram}')
    print(f'Companion images: {output_base}_real.png, {output_base}_imag.png, {output_base}_meta.png')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
