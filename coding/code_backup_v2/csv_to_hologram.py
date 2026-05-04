#!/usr/bin/env python3
"""
Complete 2D-only encoding pipeline:

CSV/TXT/NPY matrix folder -> intermediate VTP -> fixed 2D hologram images

The final encoded form is a fixed number of 2D images:
- <base>_real.png
- <base>_imag.png
- <base>_meta.png
plus <base>.png as a human-viewable phase preview.

No original matrix values or original point cloud are saved for decoding.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def run_command(cmd: list[str], title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(" ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode CSV/TXT/NPY matrix layers into fixed 2D hologram images."
    )
    parser.add_argument("input_folder", help="Path to folder containing CSV/TXT/NPY matrix layers")
    parser.add_argument("output_hologram", help="Output phase-preview path, e.g. hologram.png")
    parser.add_argument("--vtp_file", default=None, help="Intermediate VTP path")
    parser.add_argument("--x_spacing", type=float, default=1.0, help="Spacing between columns in VTP")
    parser.add_argument("--y_spacing", type=float, default=1.0, help="Spacing between rows in VTP")
    parser.add_argument("--z_spacing", type=float, default=2.0, help="Layer z offset in VTP")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep the intermediate VTP file")
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    output_base = str(Path(args.output_hologram).with_suffix(""))
    vtp_file = args.vtp_file or f"{output_base}.vtp"

    print("CSV -> VTP -> fixed 2D hologram images")
    print(f"Input folder: {args.input_folder}")
    print(f"Output preview: {args.output_hologram}")
    print(f"Intermediate VTP: {vtp_file}")

    run_command(
        [
            sys.executable,
            script_path("matrix_to_vtp.py"),
            args.input_folder,
            vtp_file,
            "--x_spacing", str(args.x_spacing),
            "--y_spacing", str(args.y_spacing),
            "--z_spacing", str(args.z_spacing),
        ],
        "STEP 1: Matrix folder -> VTP",
    )

    run_command(
        [
            sys.executable,
            script_path("vtp_to_hologram.py"),
            vtp_file,
            args.output_hologram,
        ],
        "STEP 2: VTP -> fixed 2D hologram images",
    )

    if not args.keep_intermediate and os.path.exists(vtp_file):
        os.remove(vtp_file)
        print(f"Removed intermediate VTP: {vtp_file}")

    print("\nEncoding completed.")
    print("Decoder input should be the preview path, and it will automatically find:")
    print(f"  {output_base}_real.png")
    print(f"  {output_base}_imag.png")
    print(f"  {output_base}_meta.png")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
