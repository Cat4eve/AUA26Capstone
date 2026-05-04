#!/usr/bin/env python3
"""
Complete pipeline: CSV files → VTP → Hologram diffraction image
Converts CSV/TXT/NPY matrices to 3D point cloud and generates hologram
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import vtk
import matplotlib.pyplot as plt
from pathlib import Path
from numba import typed, types
from concurrent.futures import ThreadPoolExecutor
import threading

# OpenHolo Python package
# pip install ophpy
from ophpy import PointCloud


def load_matrix(file_path):
    """Load a matrix from CSV, TXT, or NPY file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        matrix = np.load(file_path)
    elif ext in [".csv", ".txt"]:
        try:
            matrix = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            matrix = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if matrix.ndim != 2:
        raise ValueError("Input file must contain a 2D matrix.")

    return matrix


def matrix_to_vtp(matrices, output_path, x_spacing=1.0, y_spacing=1.0, z_spacing=1.0, stack_direction="z"):
    """
    Convert one or more matrices into a VTP point cloud.
    
    Args:
        matrices: Single matrix or list of matrices
        output_path: Path to output .vtp file
        x_spacing: Spacing between columns
        y_spacing: Spacing between rows
        z_spacing: Spacing between stacked matrices
        stack_direction: "z" (layer in z), "x" (layer in x), or "y" (layer in y)
    """
    if not isinstance(matrices, list):
        matrices = [matrices]
    
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    height_array = vtk.vtkDoubleArray()
    height_array.SetName("height")

    for layer_idx, matrix in enumerate(matrices):
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                x = j * x_spacing
                y = i * y_spacing
                z = float(matrix[i, j])
                
                # Apply stacking offset based on direction
                if stack_direction == "z":
                    z += layer_idx * z_spacing
                elif stack_direction == "x":
                    x += layer_idx * z_spacing
                elif stack_direction == "y":
                    y += layer_idx * z_spacing

                pid = points.InsertNextPoint(x, y, z)

                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, pid)
                vertices.InsertNextCell(vertex)

                height_array.InsertNextValue(z)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().AddArray(height_array)
    polydata.GetPointData().SetScalars(height_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()


def read_vtp_points(vtp_path):
    """Read 3D points from VTP file."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()

    points = reader.GetOutput().GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        raise ValueError("No points found in the VTP file.")

    return np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())], dtype=np.float64)


def normalize_points(points, target_xy_span=1.0, target_z_span=0.25):
    """
    Normalize point cloud into a smaller coordinate box.
    This helps avoid huge raw coordinates causing poor hologram scaling.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)

    pts = points - mins

    # Scale XY together
    xy_scale = target_xy_span / max(spans[0], spans[1], 1e-9)
    pts[:, 0] *= xy_scale
    pts[:, 1] *= xy_scale

    # Scale Z separately
    z_scale = target_z_span / max(spans[2], 1e-9)
    pts[:, 2] *= z_scale

    # Center XY around origin
    pts[:, 0] -= 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    pts[:, 1] -= 0.5 * (pts[:, 1].min() + pts[:, 1].max())

    return pts


def points_to_numba_list(points):
    """Convert numpy point array to numba-compatible format for ophpy (single channel only)."""
    plydata_list = typed.List()
    for i in range(len(points)):
        raw = typed.List()
        # Add x, y, z coordinates
        raw.append(types.float64(points[i, 0]))
        raw.append(types.float64(points[i, 1]))
        raw.append(types.float64(points[i, 2]))
        # Single channel intensity to avoid redundant multi-channel computation
        raw.append(types.float64(255))
        plydata_list.append(raw)
    return plydata_list


def generate_hologram_from_points(
    points,
    output_image,
    method="RS",
    angle_y=1.0,
    save_phase_preview=None
):
    """
    Generate grayscale hologram directly from 3D points.
    """
    # Convert points to numba-compatible format
    plydata_list = points_to_numba_list(points)
    
    # Select computation method
    compute_func = PointCloud.FresnelIntegral if method == "Fresnel" else PointCloud.RSIntegral
    
    # Compute hologram (single channel)
    holo_field = compute_func(plydata_list, 'red', angleY=angle_y)

    # Convert to phase and normalize to 0-255 range
    phase = np.angle(holo_field)
    holo_image = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    
    # Save as grayscale image
    plt.figure(figsize=(8, 8))
    plt.imshow(holo_image, cmap="gray")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_image, dpi=100, bbox_inches='tight')
    plt.close()

    if save_phase_preview:
        plt.figure(figsize=(6, 6))
        plt.imshow(phase, cmap="gray")
        plt.title("Hologram phase")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_phase_preview, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV folder → VTP → Hologram diffraction image"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing CSV/TXT/NPY files"
    )
    parser.add_argument(
        "output_hologram",
        help="Output hologram image path (.png or .bmp)"
    )
    parser.add_argument(
        "--vtp_file",
        default=None,
        help="Intermediate VTP file path (default: <output_hologram>.vtp)"
    )
    parser.add_argument(
        "--x_spacing",
        type=float,
        default=1.0,
        help="Spacing between columns (default: 1.0)"
    )
    parser.add_argument(
        "--y_spacing",
        type=float,
        default=1.0,
        help="Spacing between rows (default: 1.0)"
    )
    parser.add_argument(
        "--z_spacing",
        type=float,
        default=2.0,
        help="Height offset between stacked layers (default: 2.0)"
    )
    parser.add_argument(
        "--method",
        choices=["RS", "Fresnel"],
        default="RS",
        help="Hologram propagation method (default: RS)"
    )
    parser.add_argument(
        "--angle_y",
        type=float,
        default=1.0,
        help="Reference angleY for hologram (default: 1.0)"
    )
    parser.add_argument(
        "--target_xy_span",
        type=float,
        default=1.0,
        help="Normalized XY span (default: 1.0)"
    )
    parser.add_argument(
        "--target_z_span",
        type=float,
        default=0.25,
        help="Normalized Z span (default: 0.25)"
    )
    parser.add_argument(
        "--phase_preview",
        default=None,
        help="Optional phase preview image path"
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate VTP file"
    )

    args = parser.parse_args()

    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        print(f"ERROR: Input folder not found: {args.input_folder}")
        sys.exit(1)

    # Set up intermediate file paths
    output_base = os.path.splitext(args.output_hologram)[0]
    vtp_file = args.vtp_file if args.vtp_file else f"{output_base}.vtp"

    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  CSV → VTP → HOLOGRAM PIPELINE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    print(f"\nInput folder: {args.input_folder}")
    print(f"Output hologram: {args.output_hologram}")
    print(f"Intermediate VTP: {vtp_file}")

    results = {"step1": None, "step2": None}
    errors = []
    lock = threading.Lock()

    def run_step1():
        """Run CSV to VTP conversion"""
        try:
            step1_cmd = [
                sys.executable,
                "matrix_to_vtp.py",
                str(args.input_folder),
                vtp_file,
                "--x_spacing", str(args.x_spacing),
                "--y_spacing", str(args.y_spacing),
                "--z_spacing", str(args.z_spacing),
            ]
            print(f"\n{'='*70}")
            print("STEP 1: CSV files → VTP point cloud")
            print(f"{'='*70}")
            result = subprocess.run(step1_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                with lock:
                    errors.append(f"Step 1 failed:\n{result.stderr}")
                return False
            print(result.stdout)
            with lock:
                results["step1"] = True
            return True
        except subprocess.TimeoutExpired:
            with lock:
                errors.append("Step 1 timed out (1 hour limit)")
            return False
        except Exception as e:
            with lock:
                errors.append(f"Step 1 error: {e}")
            return False

    def run_step2():
        """Run VTP to Hologram conversion (waits for step 1)"""
        try:
            # Wait for step 1 to complete
            for _ in range(600):  # Max 10 minutes waiting
                if results["step1"] is not None:
                    break
                threading.Event().wait(1)

            if results["step1"] is False:
                return False

            if not os.path.exists(vtp_file):
                with lock:
                    errors.append("Step 1 VTP file not found before Step 2")
                return False

            step2_cmd = [
                sys.executable,
                "vtp_to_hologram.py",
                vtp_file,
                args.output_hologram,
                "--method", args.method,
                "--angle_y", str(args.angle_y),
                "--target_xy_span", str(args.target_xy_span),
                "--target_z_span", str(args.target_z_span),
            ]

            if args.phase_preview:
                step2_cmd.extend(["--phase_preview", args.phase_preview])

            print(f"\n{'='*70}")
            print("STEP 2: VTP point cloud → Hologram diffraction image")
            print(f"{'='*70}")
            result = subprocess.run(step2_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                with lock:
                    errors.append(f"Step 2 failed:\n{result.stderr}")
                return False
            print(result.stdout)
            with lock:
                results["step2"] = True
            return True
        except subprocess.TimeoutExpired:
            with lock:
                errors.append("Step 2 timed out (1 hour limit)")
            return False
        except Exception as e:
            with lock:
                errors.append(f"Step 2 error: {e}")
            return False

    # Run both steps in parallel threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(run_step1)
        future2 = executor.submit(run_step2)

        step1_success = future1.result()
        step2_success = future2.result()

    # Check for errors
    if errors:
        for error in errors:
            print(f"\n❌ {error}", file=sys.stderr)
        sys.exit(1)

    if not (step1_success and step2_success):
        print("\n❌ Pipeline failed", file=sys.stderr)
        sys.exit(1)

    # Clean up intermediate files if requested
    if not args.keep_intermediate:
        print(f"\n{'='*70}")
        print("CLEANUP: Removing intermediate VTP file")
        print(f"{'='*70}")
        try:
            if os.path.exists(vtp_file):
                os.remove(vtp_file)
                print(f"✓ Removed: {vtp_file}")
        except Exception as e:
            print(f"Warning: Could not remove intermediate file: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n✓ Hologram saved to: {args.output_hologram}")
    if args.phase_preview:
        print(f"✓ Phase preview saved to: {args.phase_preview}")
    if args.keep_intermediate:
        print(f"✓ Intermediate VTP kept: {vtp_file}")
    else:
        print(f"✓ No intermediate files (clean workflow)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
