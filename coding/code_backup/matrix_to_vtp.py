import argparse
import os
import numpy as np
import vtk
from pathlib import Path


def load_matrix(file_path):
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert all CSV files in a folder into a stacked VTP point cloud. Each CSV layer is offset by z_spacing in the z-direction."
    )
    parser.add_argument("input_folder", help="Path to folder containing CSV/TXT/NPY files")
    parser.add_argument("output_file", help="Path to output .vtp file")
    parser.add_argument("--x_spacing", type=float, default=1.0, help="Spacing between columns")
    parser.add_argument("--y_spacing", type=float, default=1.0, help="Spacing between rows")
    parser.add_argument("--z_spacing", type=float, default=2.0, help="Height offset between stacked layers (default: 2.0)")

    args = parser.parse_args()

    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    # Find all CSV, TXT, and NPY files
    supported_extensions = {".csv", ".txt", ".npy"}
    matrix_files = sorted([
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])

    if not matrix_files:
        raise FileNotFoundError(f"No CSV/TXT/NPY files found in: {args.input_folder}")

    # Load all matrices
    matrices = []
    for file_path in matrix_files:
        try:
            matrix = load_matrix(str(file_path))
            matrices.append(matrix)
            print(f"Loaded: {file_path.name} (shape: {matrix.shape})")
        except Exception as e:
            print(f"Warning: Skipped {file_path.name} - {e}")

    if not matrices:
        raise ValueError("No valid matrices could be loaded from the folder.")

    # Convert to VTP with z-stacking
    matrix_to_vtp(
        matrices if len(matrices) > 1 else matrices[0],
        args.output_file,
        x_spacing=args.x_spacing,
        y_spacing=args.y_spacing,
        z_spacing=args.z_spacing,
        stack_direction="z"
    )

    print(f"\nStacked {len(matrices)} layer(s) into VTP point cloud: {args.output_file}")
    print(f"Z-spacing between layers: {args.z_spacing}")


if __name__ == "__main__":
    main()