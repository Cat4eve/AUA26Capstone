import os
import glob
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import re

def calculate_metrics(truth, pred):
    """Calculate comparison metrics between two matrices."""
    # Ensure same shape
    if truth.shape != pred.shape:
        return None

    mse = np.mean((truth - pred) ** 2)
    mae = np.mean(np.abs(truth - pred))
    rmse = np.sqrt(mse)
    
    # Normalize RMSE by the range of truth values (typically 0-255)
    truth_range = np.max(truth) - np.min(truth)
    nrmse = rmse / truth_range if truth_range > 0 else rmse

    # Vectorized comparison
    truth_f = truth.flatten()
    pred_f = pred.flatten()
    
    # Cosine Similarity: 1 is identical, 0 is orthogonal
    cos_sim = 1 - cosine(truth_f, pred_f) if np.any(truth_f) and np.any(pred_f) else 0.0
    
    # Pearson Correlation: 1 is perfect linear correlation
    correlation = np.corrcoef(truth_f, pred_f)[0, 1] if np.std(truth_f) > 0 and np.std(pred_f) > 0 else 0.0

    return {
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "Cosine_Similarity": cos_sim,
        "Correlation": correlation
    }

def natural_sort_key(s):
    """Key for natural alphanumeric sorting (e.g., layer_2 before layer_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main():
    parser = argparse.ArgumentParser(description="Compare original BERT layers with decoded layers.")
    parser.add_argument("source_dir", help="Path to folder with original CSVs")
    parser.add_argument("decoded_dir", help="Path to folder with decoded CSVs")
    args = parser.parse_args()

    # Use natural sort to ensure Layer 2 comes before Layer 10
    source_files = sorted(glob.glob(os.path.join(args.source_dir, "*.csv")), key=natural_sort_key)
    decoded_files = sorted(glob.glob(os.path.join(args.decoded_dir, "*.csv")), key=natural_sort_key)

    if not source_files:
        print(f"DEBUG: No .csv files found in: {os.path.abspath(args.source_dir)}")
    if not decoded_files:
        print(f"DEBUG: No .csv files found in: {os.path.abspath(args.decoded_dir)}")

    if not source_files or not decoded_files:
        print("Error: One or both directories are empty or contain no CSV files.")
        return

    print(f"Comparing {len(source_files)} source files with {len(decoded_files)} decoded files...\n")
    print(f"{'Layer File':<20} | {'RMSE (0-255)':<12} | {'Cosine Sim':<12} | {'Correlation':<12}")
    print("-" * 75)

    all_metrics = []
    
    for i in range(min(len(source_files), len(decoded_files))):
        s_path = source_files[i]
        d_path = decoded_files[i]
        
        try:
            # Load CSVs (assuming comma delimiter as per previous scripts)
            truth = np.loadtxt(s_path, delimiter=',')
            pred = np.loadtxt(d_path, delimiter=',')
            
            m = calculate_metrics(truth, pred)
            if m:
                fname = os.path.basename(s_path)
                print(f"{fname:<20} | {m['RMSE']:<12.4f} | {m['Cosine_Similarity']:<12.4f} | {m['Correlation']:<12.4f}")
                all_metrics.append(m)
            else:
                print(f"Error: Shape mismatch for {os.path.basename(s_path)}: {truth.shape} vs {pred.shape}")
        except Exception as e:
            print(f"Error processing {os.path.basename(s_path)}: {e}")

    if all_metrics:
        avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
        avg_cos = np.mean([m['Cosine_Similarity'] for m in all_metrics])
        avg_corr = np.mean([m['Correlation'] for m in all_metrics])
        avg_nrmse = np.mean([m['NRMSE'] for m in all_metrics])

        print("-" * 75)
        print(f"{'AVERAGE':<20} | {avg_rmse:<12.4f} | {avg_cos:<12.4f} | {avg_corr:<12.4f}")
        
        print(f"\nStatistical Estimation:")
        print(f"- Structural Accuracy: {avg_cos * 100:.2f}%")
        print(f"- Normalized Error:    {avg_nrmse * 100:.2f}%")
        
        if avg_cos > 0.98:
            print("\nEstimation: Excellent. The decoded layers are mathematically near-identical.")
        elif avg_cos > 0.90:
            print("\nEstimation: Strong. Reconstruction is highly successful with minimal diffraction noise.")
        elif avg_cos > 0.75:
            print("\nEstimation: Moderate. The weights are recognizable, but interference patterns are degrading the values.")
        else:
            print("\nEstimation: Weak. High distortion. Check if 'num_planes' or 'z_spacing' matches the encoding settings.")

if __name__ == "__main__":
    main()
