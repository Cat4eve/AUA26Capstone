import argparse
import os
import sys
import numpy as np
import vtk
import matplotlib.pyplot as plt
from numba import typed, types

# OpenHolo Python package
# pip install ophpy
from ophpy import PointCloud


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
    Generate grayscale hologram directly from 3D points (no intermediate PLY needed).
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
        description="Convert VTP point cloud to a grayscale hologram diffraction image."
    )
    parser.add_argument("input_vtp", help="Input .vtp point cloud")
    parser.add_argument("output_image", help="Output hologram image (.png or .bmp)")
    parser.add_argument("--method", default="RS", choices=["RS", "Fresnel"], help="Propagation method")
    parser.add_argument("--angle_y", type=float, default=1.0, help="Reference angleY for hologram generation")
    parser.add_argument("--target_xy_span", type=float, default=1.0, help="Normalized XY span")
    parser.add_argument("--target_z_span", type=float, default=0.25, help="Normalized Z span")
    parser.add_argument("--phase_preview", default=None, help="Optional phase preview image path")
    args = parser.parse_args()

    if not os.path.exists(args.input_vtp):
        raise FileNotFoundError(f"Input file not found: {args.input_vtp}")

    pts = read_vtp_points(args.input_vtp)
    pts = normalize_points(
        pts,
        target_xy_span=args.target_xy_span,
        target_z_span=args.target_z_span
    )

    generate_hologram_from_points(
        points=pts,
        output_image=args.output_image,
        method=args.method,
        angle_y=args.angle_y,
        save_phase_preview=args.phase_preview
    )

    print(f"Hologram image saved to: {args.output_image}")
    if args.phase_preview:
        print(f"Phase preview saved to:  {args.phase_preview}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)