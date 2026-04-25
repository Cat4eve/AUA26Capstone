#!/usr/bin/env python3
"""
Hologram Decoder: Convert holographic diffraction image (PNG) back to 3D point cloud (VTP)

Uses angular spectrum method for inverse wave propagation:
1. Load PNG hologram → extract phase
2. Inverse propagate using FFT-based angular spectrum
3. Reconstruct 3D volume → sample peaks as point cloud
4. Save as VTP file
"""

import argparse
import os
import numpy as np
import vtk
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.ndimage import center_of_mass
import warnings
warnings.filterwarnings('ignore')


def load_hologram_png(png_path):
    """
    Load hologram PNG and extract phase information.
    
    Args:
        png_path: Path to hologram PNG (grayscale 0-255)
        
    Returns:
        phase: Phase array in range [-π, +π]
    """
    img = Image.open(png_path).convert('L')  # Grayscale
    img_array = np.array(img, dtype=np.float32)
    
    # Convert 0-255 back to phase [-π, +π]
    phase = (img_array / 255.0) * (2 * np.pi) - np.pi
    
    print(f"✓ Loaded hologram: {png_path}")
    print(f"  Shape: {phase.shape}, Phase range: [{phase.min():.3f}, {phase.max():.3f}]")
    
    return phase


def phase_to_complex_field(phase, amplitude=1.0):
    """
    Convert phase to complex field.
    
    Args:
        phase: Phase array in [-π, +π]
        amplitude: Uniform amplitude (default 1.0)
        
    Returns:
        Complex field: amplitude * e^(i*phase)
    """
    complex_field = amplitude * np.exp(1j * phase)
    return complex_field


def angular_spectrum_propagation(complex_field, distance, wavelength, pixel_pitch):
    """
    Inverse propagate hologram using angular spectrum method (FFT-based).
    
    Args:
        complex_field: Complex amplitude field on hologram plane
        distance: Propagation distance (negative for backward/reconstruction)
        wavelength: Wavelength in meters
        pixel_pitch: Pixel size in meters
        
    Returns:
        reconstructed_field: Complex field at propagation distance
    """
    height, width = complex_field.shape
    
    # Create frequency grids
    fx = np.fft.fftfreq(width, pixel_pitch)
    fy = np.fft.fftfreq(height, pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)
    
    # Angular spectrum of input
    spectrum = np.fft.fft2(complex_field)
    
    # Propagation kernel (Fresnel diffraction)
    # K = e^(i*2π*sqrt(1/λ² - (fx²+fy²)) * z)
    f_squared = FX**2 + FY**2
    
    # Avoid evanescent waves (high frequencies)
    valid_mask = (f_squared <= (1.0 / wavelength)**2)
    
    # Propagation phase
    prop_phase = np.zeros_like(f_squared)
    prop_phase[valid_mask] = 2 * np.pi * np.sqrt(
        (1.0 / wavelength)**2 - f_squared[valid_mask]
    ) * distance
    prop_phase[~valid_mask] = 0  # Evanescent waves decay
    
    # Apply propagation kernel
    propagation_kernel = np.exp(1j * prop_phase)
    spectrum_propagated = spectrum * propagation_kernel
    
    # Inverse FFT to get reconstructed field
    reconstructed_field = np.fft.ifft2(spectrum_propagated)
    
    return reconstructed_field


def reconstruct_3d_volume(phase, distance_range=(0.001, 0.1), num_planes=50):
    """
    Reconstruct 3D volume from hologram by propagating at multiple distances.
    
    Args:
        phase: Hologram phase array
        distance_range: (min_dist, max_dist) in meters
        num_planes: Number of z-planes to propagate
        
    Returns:
        volume: 3D intensity array [z, y, x]
        z_distances: Distance values for each z-plane
    """
    # Hologram parameters (ophpy defaults)
    wavelength = 632.8e-9  # Red laser ~633nm
    pixel_pitch = 3.6e-6   # 3.6 micrometers
    
    complex_field = phase_to_complex_field(phase, amplitude=1.0)
    
    z_distances = np.linspace(distance_range[0], distance_range[1], num_planes)
    volume = np.zeros((num_planes, phase.shape[0], phase.shape[1]), dtype=np.float32)
    
    print(f"\nRecovering 3D volume ({num_planes} z-planes)...")
    for i, z_dist in enumerate(z_distances):
        if (i + 1) % max(1, num_planes // 10) == 0:
            print(f"  Plane {i+1}/{num_planes} @ z={z_dist*1000:.2f}mm")
        
        # Propagate backward (negative distance)
        reconstructed = angular_spectrum_propagation(
            complex_field,
            distance=-z_dist,
            wavelength=wavelength,
            pixel_pitch=pixel_pitch
        )
        
        # Intensity = |field|²
        volume[i] = np.abs(reconstructed)**2
    
    print(f"✓ Volume shape: {volume.shape}")
    
    return volume, z_distances


def extract_point_cloud(volume, z_distances, threshold_percentile=85, min_size=5):
    """
    Extract point cloud from reconstructed volume by finding local maxima.
    
    Args:
        volume: 3D intensity array
        z_distances: Z coordinates for each plane
        threshold_percentile: Intensity threshold (85th percentile)
        min_size: Minimum cluster size to keep
        
    Returns:
        points: Nx3 array of (x, y, z) coordinates
    """
    # Normalize volume
    volume_norm = volume / (volume.max() + 1e-10)
    
    # Threshold: keep brightest voxels
    threshold = np.percentile(volume_norm, threshold_percentile)
    binary_volume = volume_norm > threshold
    
    print(f"\nExtracting point cloud (threshold: {threshold_percentile}th percentile = {threshold:.4f})...")
    
    # Label connected components
    labeled_volume, num_features = ndimage.label(binary_volume)
    print(f"  Found {num_features} connected regions")
    
    # Extract center of mass for each region (cluster)
    points = []
    pixel_pitch = 3.6e-6  # meters
    
    for region_id in range(1, num_features + 1):
        # Get voxels in this region
        region_coords = np.where(labeled_volume == region_id)
        region_size = len(region_coords[0])
        
        # Filter small regions
        if region_size < min_size:
            continue
        
        # Compute center of mass
        z_idx, y_idx, x_idx = center_of_mass(binary_volume, labeled_volume, region_id)
        
        # Convert to physical coordinates
        x = x_idx * pixel_pitch * 1000  # mm
        y = y_idx * pixel_pitch * 1000  # mm
        z = z_distances[int(z_idx)] * 1000  # mm
        
        # Weight by intensity at center
        intensity = volume_norm[int(z_idx), int(y_idx), int(x_idx)]
        
        points.append([x, y, z, intensity, region_size])
    
    points = np.array(points)
    print(f"✓ Extracted {len(points)} point clusters")
    
    if len(points) == 0:
        print("  WARNING: No points extracted! Try lowering threshold_percentile")
        return None
    
    return points


def normalize_points(points, target_scale=100.0):
    """
    Normalize point cloud to reasonable coordinates.
    
    Args:
        points: Nx5 array [x, y, z, intensity, size]
        target_scale: Scale to normalize to
        
    Returns:
        normalized: Nx3 array [x, y, z] normalized and centered
    """
    xyz = points[:, :3]
    
    # Center
    center = xyz.mean(axis=0)
    xyz = xyz - center
    
    # Scale
    max_dist = np.max(np.linalg.norm(xyz, axis=1))
    if max_dist > 0:
        xyz = xyz * (target_scale / max_dist)
    
    return xyz


def points_to_vtp(points, output_path):
    """
    Convert point cloud to VTP format.
    
    Args:
        points: Nx3 array of (x, y, z) coordinates
        output_path: Output .vtp file path
    """
    vtp_points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    intensity_array = vtk.vtkDoubleArray()
    intensity_array.SetName("intensity")
    
    for i, point in enumerate(points):
        pid = vtp_points.InsertNextPoint(point[0], point[1], point[2])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, pid)
        vertices.InsertNextCell(vertex)
        intensity_array.InsertNextValue(1.0)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtp_points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().AddArray(intensity_array)
    polydata.GetPointData().SetScalars(intensity_array)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"✓ Saved to VTP: {output_path}")


def save_reconstructed_preview(volume, z_distances, output_path):
    """
    Save 2D slices of reconstructed volume as preview.
    
    Args:
        volume: 3D intensity array
        z_distances: Z coordinates
        output_path: Output image path
    """
    # Normalize
    vol_norm = volume / (volume.max() + 1e-10)
    
    # Extract slices at different depths
    num_slices = 4
    slice_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, z_idx in enumerate(slice_indices):
        ax = axes[idx]
        slice_data = vol_norm[z_idx]
        ax.imshow(slice_data, cmap='hot')
        ax.set_title(f'Reconstructed plane z={z_distances[z_idx]*1000:.2f}mm')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved preview: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode hologram PNG diffraction image back to 3D point cloud VTP file"
    )
    parser.add_argument("input_hologram", help="Input hologram PNG image")
    parser.add_argument("output_vtp", help="Output VTP 3D point cloud file")
    parser.add_argument(
        "--distance_min",
        type=float,
        default=0.001,
        help="Minimum propagation distance in meters (default: 0.001 = 1mm)"
    )
    parser.add_argument(
        "--distance_max",
        type=float,
        default=0.1,
        help="Maximum propagation distance in meters (default: 0.1 = 100mm)"
    )
    parser.add_argument(
        "--num_planes",
        type=int,
        default=50,
        help="Number of z-planes to reconstruct (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=85,
        help="Intensity threshold percentile (0-100, higher=stricter, default: 85)"
    )
    parser.add_argument(
        "--preview",
        default=None,
        help="Optional: save reconstructed volume preview image"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_hologram):
        raise FileNotFoundError(f"Hologram file not found: {args.input_hologram}")
    
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  HOLOGRAM → 3D POINT CLOUD DECODER".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Step 1: Load hologram
    phase = load_hologram_png(args.input_hologram)
    
    # Step 2: Reconstruct 3D volume
    volume, z_distances = reconstruct_3d_volume(
        phase,
        distance_range=(args.distance_min, args.distance_max),
        num_planes=args.num_planes
    )
    
    # Step 3: Extract point cloud
    points_raw = extract_point_cloud(
        volume,
        z_distances,
        threshold_percentile=args.threshold,
        min_size=5
    )
    
    if points_raw is None:
        print("\n❌ Reconstruction failed: No points extracted")
        return 1
    
    # Step 4: Normalize
    points_normalized = normalize_points(points_raw[:, :3], target_scale=100.0)
    
    # Step 5: Save as VTP
    points_to_vtp(points_normalized, args.output_vtp)
    
    # Step 6: Save preview if requested
    if args.preview:
        save_reconstructed_preview(volume, z_distances, args.preview)
    
    print(f"\n{'='*70}")
    print("DECODING COMPLETED!")
    print(f"{'='*70}")
    print(f"✓ Hologram decoded: {args.input_hologram}")
    print(f"✓ Point cloud saved: {args.output_vtp}")
    print(f"✓ Total points: {len(points_normalized)}")
    if args.preview:
        print(f"✓ Preview saved: {args.preview}")


if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code:
            exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDecoding interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc()
        exit(1)
