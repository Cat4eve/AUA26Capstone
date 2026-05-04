#!/usr/bin/env python3
"""
Hologram Decoder: Decode holographic diffraction image (PNG) back to layered 2D matrices

Uses angular spectrum method for inverse wave propagation:
1. Load PNG hologram → extract phase
2. Inverse propagate using FFT-based angular spectrum
3. Reconstruct 3D volume at multiple z-planes
4. Extract layers from z-planes → downsample to specified dimensions
5. Save each layer as CSV file (with optional VTP output)

Layer Configuration File Format:
    num_layers: 3
    
    layer 0
    rows: 128
    cols: 128
    
    layer 1
    rows: 128
    cols: 128
    
    layer 2
    rows: 128
    cols: 128

Dependencies: pip install numpy vtk matplotlib Pillow scipy
"""

# Standard library imports
import argparse
import os
import sys
import traceback
import warnings

# Third-party imports
import numpy as np
import vtk
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.ndimage import center_of_mass
from scipy.ndimage import zoom

# Suppress warnings
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
        z_distances: Distance values for each z-plane (in meters)
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


def parse_layer_config(config_path):
    """
    Parse layer configuration file.
    
    Format:
        num_layers: 3
        x_spacing: 1.0
        y_spacing: 1.0
        z_spacing: 0.002
        
        layer 0
        rows: 128
        cols: 128
        
        layer 1
        rows: 128
        cols: 128
        ...
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Dict with 'layers', 'x_spacing', 'y_spacing', 'z_spacing'
    """
    layers = []
    config = {
        'x_spacing': 1.0,
        'y_spacing': 1.0,
        'z_spacing': 0.002,
        'layers': []
    }
    
    with open(config_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    num_layers = None
    layer_dict = {}
    for line in lines:
        if line.startswith('num_layers:'):
            num_layers = int(line.split(':')[1])
        elif line.startswith('x_spacing:'):
            config['x_spacing'] = float(line.split(':')[1])
        elif line.startswith('y_spacing:'):
            config['y_spacing'] = float(line.split(':')[1])
        elif line.startswith('z_spacing:'):
            config['z_spacing'] = float(line.split(':')[1])
        elif line.startswith('layer'):
            layer_dict = {}
        elif line.startswith('rows:'):
            layer_dict['rows'] = int(line.split(':')[1])
        elif line.startswith('cols:'):
            layer_dict['cols'] = int(line.split(':')[1])
            layers.append(layer_dict)
    
    if num_layers and len(layers) != num_layers:
        print(f"Warning: Expected {num_layers} layers but found {len(layers)}")
    
    config['layers'] = layers
    
    print(f"✓ Parsed {len(layers)} layer configurations")
    print(f"  Spacing: x={config['x_spacing']}, y={config['y_spacing']}, z={config['z_spacing']}")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {layer['rows']}×{layer['cols']}")
    
    return config


def generate_template_config(output_path, num_layers=3):
    """
    Generate a template configuration file.
    
    Args:
        output_path: Path where to save the template
        num_layers: Number of layers in the template
    """
    with open(output_path, 'w') as f:
        f.write("# Hologram Layer Configuration File\n")
        f.write("# Specify the number of layers, spacing, and dimensions for each layer\n\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"x_spacing: 1.0\n")
        f.write(f"y_spacing: 1.0\n")
        f.write(f"z_spacing: 0.002\n\n")
        
        for i in range(num_layers):
            f.write(f"layer {i}\n")
            f.write(f"rows: 8\n")
            f.write(f"cols: 9\n\n")
    
    print(f"✓ Generated template config: {output_path}")


def extract_layers(volume, z_distances, z_spacing=0.002, layer_config=None, axis='z'):
    """
    Extract 2D layers from 3D volume at specified z-distances.
    
    Args:
        volume: 3D intensity array [z, y, x]
        z_distances: Distance values for each z-plane (in meters)
        z_spacing: Expected spacing between layers (in meters)
        layer_config: Dict with 'layers' key containing list of dicts with 'rows' and 'cols'
        axis: 'z' (default), 'y', or 'x' - which axis to slice along
        
    Returns:
        layers: List of 2D numpy arrays (one per layer)
    """
    # Normalize volume
    volume_norm = volume / (volume.max() + 1e-10)
    
    print(f"\nExtracting layers (axis={axis})...")
    
    if layer_config is None:
        # Simple extraction: use peaks in intensity
        print(f"  No layer config provided - extracting peaks along {axis}-axis")
        
        if axis == 'z':
            max_intensity = np.max(volume_norm, axis=(1, 2))
            axis_idx = 0
        elif axis == 'y':
            max_intensity = np.max(volume_norm, axis=(0, 2))
            axis_idx = 1
        elif axis == 'x':
            max_intensity = np.max(volume_norm, axis=(0, 1))
            axis_idx = 2
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(max_intensity, distance=5)
        
        if len(peaks) == 0:
            num_planes = min(10, max_intensity.shape[0] // 5)
            peaks = np.linspace(0, max_intensity.shape[0]-1, num_planes, dtype=int)
        
        if axis == 'z':
            layers = [volume_norm[i] for i in peaks]
        elif axis == 'y':
            layers = [volume_norm[:, i, :] for i in peaks]
        elif axis == 'x':
            layers = [volume_norm[:, :, i] for i in peaks]
        
        print(f"  Found {len(layers)} layers")
        
    else:
        # Use layer config to determine where to extract
        layer_configs = layer_config['layers']
        num_layers = len(layer_configs)
        
        # Calculate z-positions for layers: start at first distance + layer_index * z_spacing
        z_start = z_distances[0]
        z_positions = [z_start + i * z_spacing for i in range(num_layers)]
        
        # Map layer positions to volume indices
        z_indices = []
        for z_pos in z_positions:
            idx = np.argmin(np.abs(z_distances - z_pos))
            z_indices.append(idx)
        
        # Extract layers and resize to specified dimensions
        layers = []
        print(f"  Extracting {num_layers} layers and resizing to specified dimensions...")
        
        for i, (z_idx, cfg) in enumerate(zip(z_indices, layer_configs)):
            # Get the 2D slice
            layer_data = volume_norm[z_idx]
            
            # Resize to target dimensions
            target_rows = cfg['rows']
            target_cols = cfg['cols']
            
            # Use zoom for resizing
            zoom_factors = (target_rows / layer_data.shape[0], target_cols / layer_data.shape[1])
            resized_layer = zoom(layer_data, zoom_factors, order=1)
            
            # Ensure exact size
            if resized_layer.shape != (target_rows, target_cols):
                # Pad or crop to exact dimensions
                padded = np.zeros((target_rows, target_cols))
                r, c = min(resized_layer.shape[0], target_rows), min(resized_layer.shape[1], target_cols)
                padded[:r, :c] = resized_layer[:r, :c]
                resized_layer = padded
            
            # Denormalize from [0, 1] to [0, 255] for consistency with original data
            denormalized = (resized_layer * 255).astype(np.uint8).astype(np.float32)
            
            layers.append(denormalized)
            print(f"    Layer {i}: extracted from z={z_distances[z_idx]*1000:.2f}mm, "
                  f"resized to {target_rows}x{target_cols}, denormalized to [0, 255]")
    
    return layers


def save_layers_as_csv(layers, output_folder, base_name="layer"):
    """
    Save extracted layers as CSV files (as uint8 integers 0-255).
    
    Args:
        layers: List of 2D numpy arrays
        output_folder: Folder to save CSV files
        base_name: Base name for output files (e.g., "layer_0.csv", "layer_1.csv")
        
    Returns:
        file_paths: List of saved file paths
    """
    os.makedirs(output_folder, exist_ok=True)
    
    file_paths = []
    print(f"\nSaving layers as CSV files...")
    
    for i, layer in enumerate(layers):
        # Ensure values are uint8 (0-255 integers)
        if layer.max() <= 1.0:
            # If normalized, denormalize
            layer_uint8 = (layer * 255).astype(np.uint8)
        else:
            # Already in 0-255 range
            layer_uint8 = layer.astype(np.uint8)
        
        file_path = os.path.join(output_folder, f"{base_name}_{i}.csv")
        np.savetxt(file_path, layer_uint8, delimiter=',', fmt='%d')
        file_paths.append(file_path)
        print(f"  Saved: {file_path} ({layer_uint8.shape[0]}×{layer_uint8.shape[1]})")
    
    return file_paths


def layers_to_vtp(layers, output_path, x_spacing=1.0, y_spacing=1.0, z_spacing=1.0):
    """
    Convert layered 2D matrices to VTP point cloud.
    
    Args:
        layers: List of 2D numpy arrays
        output_path: Output VTP file path
        x_spacing: Spacing between columns
        y_spacing: Spacing between rows
        z_spacing: Spacing between stacked layers
    """
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    height_array = vtk.vtkDoubleArray()
    height_array.SetName("height")
    
    print(f"\nConverting layers to VTP point cloud...")
    
    for layer_idx, layer in enumerate(layers):
        rows, cols = layer.shape
        
        for i in range(rows):
            for j in range(cols):
                # Replicate the coordinate generation from matrix_to_vtp.py
                # X = Column index * x_spacing
                # Y = Row index * y_spacing
                # Z = Matrix value + Layer index * z_spacing
                x = j * x_spacing
                y = i * y_spacing
                z = float(layer[i, j]) + layer_idx * z_spacing
                
                pid = points.InsertNextPoint(x, y, z)
                
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, pid)
                vertices.InsertNextCell(vertex)
                
                height_array.InsertNextValue(z) # Store the final Z coordinate as scalar, mirroring matrix_to_vtp.py
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().AddArray(height_array)
    polydata.GetPointData().SetScalars(height_array)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"✓ Saved VTP: {output_path}")


def save_reconstructed_preview(volume, z_distances, output_path):
    """
    Save 2D slices of reconstructed volume as preview.
    
    Args:
        volume: 3D intensity array
        z_distances: Z coordinates (in meters)
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
        description="Decode hologram PNG back to layered 2D matrices and optional VTP point cloud"
    )
    parser.add_argument("input_hologram", nargs='?', default=None, help="Input hologram PNG image")
    parser.add_argument("output_folder", nargs='?', default=None, help="Output folder for layer CSV files")
    parser.add_argument(
        "--config",
        default=None,
        help="Layer configuration file (auto-generates if not provided)"
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Generate a template configuration file at this path and exit"
    )
    parser.add_argument(
        "--vtp",
        default=None,
        help="Optional: save reconstructed point cloud as VTP file"
    )
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
        "--z_spacing",
        type=float,
        default=0.002,
        help="Expected spacing between layers in meters (default: 0.002 = 2mm)"
    )
    parser.add_argument(
        "--vtp_x_spacing",
        type=float,
        default=1.0,
        help="Spacing between columns for VTP output (default: 1.0, matches matrix_to_vtp.py)"
    )
    parser.add_argument(
        "--vtp_y_spacing",
        type=float,
        default=1.0,
        help="Spacing between rows for VTP output (default: 1.0, matches matrix_to_vtp.py)"
    )
    parser.add_argument(
        "--vtp_z_spacing",
        type=float,
        default=2.0,
        help="Height offset between stacked layers for VTP output (default: 2.0, matches matrix_to_vtp.py)"
    )
    parser.add_argument(
        "--preview",
        default=None,
        help="Optional: save reconstructed volume preview image"
    )
    parser.add_argument(
        "--axis",
        choices=['x', 'y', 'z'],
        default='z',
        help="Axis along which to extract layers: 'x', 'y', or 'z' (default: z)"
    )
    
    args = parser.parse_args()
    
    # Handle template generation
    if args.template:
        generate_template_config(args.template, num_layers=3)
        return 0
    
    # Check required arguments
    if not args.input_hologram or not args.output_folder:
        parser.error("input_hologram and output_folder are required (unless using --template)")
    
    if not os.path.exists(args.input_hologram):
        raise FileNotFoundError(f"Hologram file not found: {args.input_hologram}")
    
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  HOLOGRAM TO LAYERED DECODER".center(68) + "║")
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
    
    # Step 3: Load or parse layer configuration
    layer_config = None
    if args.config and os.path.exists(args.config):
        layer_config = parse_layer_config(args.config)
    else:
        if args.config:
            print(f"Warning: Config file not found at {args.config}, using automatic detection")
    
    # Step 4: Extract layers from volume
    layers = extract_layers(volume, z_distances, z_spacing=args.z_spacing, layer_config=layer_config, axis=args.axis)
    
    # Step 5: Save layers as CSV
    os.makedirs(args.output_folder, exist_ok=True)
    layer_files = save_layers_as_csv(layers, args.output_folder)
    
    # Step 6: Optional VTP output (use spacing values from config if available)
    if args.vtp:
        vtp_x_spacing = args.vtp_x_spacing
        vtp_y_spacing = args.vtp_y_spacing
        vtp_z_spacing = args.vtp_z_spacing
        
        if layer_config:
            vtp_x_spacing = layer_config.get('x_spacing', args.vtp_x_spacing)
            vtp_y_spacing = layer_config.get('y_spacing', args.vtp_y_spacing)
            vtp_z_spacing = layer_config.get('z_spacing', args.vtp_z_spacing)
        
        layers_to_vtp(layers, args.vtp, x_spacing=vtp_x_spacing, y_spacing=vtp_y_spacing, z_spacing=vtp_z_spacing)
    
    # Step 7: Optional preview
    if args.preview:
        save_reconstructed_preview(volume, z_distances, args.preview)
    
    print(f"\n{'='*70}")
    print("DECODING COMPLETED!")
    print(f"{'='*70}")
    print(f"✓ Hologram decoded: {args.input_hologram}")
    print(f"✓ Layers saved to: {args.output_folder}")
    print(f"✓ Total layers: {len(layers)}")
    print(f"✓ Layer files: {', '.join([os.path.basename(f) for f in layer_files])}")
    if args.vtp:
        print(f"✓ VTP point cloud: {args.vtp}")
    if args.preview:
        print(f"✓ Preview image: {args.preview}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code:
            exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDecoding interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        exit(1)
