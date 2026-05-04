#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load_matrix(path: Path):
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(np.float64)
    try:
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        return np.loadtxt(path, dtype=np.float64)


def cosine(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a,b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def metrics(A, B):
    D = A - B
    mse = float(np.mean(D**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(D)))
    rel = float(rmse / (np.sqrt(np.mean(A**2)) + 1e-12))
    acc = max(0.0, 1.0 - rel) * 100.0
    return mse, rmse, mae, rel, cosine(A, B), acc


def main():
    ap = argparse.ArgumentParser(description="Compare original and reconstructed matrix folders.")
    ap.add_argument("original_folder")
    ap.add_argument("decoded_folder")
    args = ap.parse_args()

    orig_dir = Path(args.original_folder)
    dec_dir = Path(args.decoded_folder)
    files = sorted(p for p in orig_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".npy"})
    if not files:
        raise FileNotFoundError("No original matrix files found")

    all_a = []
    all_b = []
    print("file,mse,rmse,mae,relative_rmse,cosine_similarity,accuracy_percent")
    for p in files:
        candidates = [dec_dir / p.name, dec_dir / f"{p.stem}.csv"]
        q = next((x for x in candidates if x.exists()), None)
        if q is None:
            print(f"{p.name},MISSING,,,,,")
            continue
        A = load_matrix(p)
        B = load_matrix(q)
        if A.shape != B.shape:
            print(f"{p.name},SHAPE_MISMATCH original={A.shape} decoded={B.shape},,,,,")
            continue
        vals = metrics(A, B)
        print(f"{p.name},{vals[0]:.8g},{vals[1]:.8g},{vals[2]:.8g},{vals[3]:.8g},{vals[4]:.8g},{vals[5]:.4f}")
        all_a.append(A.ravel())
        all_b.append(B.ravel())

    if all_a:
        A = np.concatenate(all_a)
        B = np.concatenate(all_b)
        vals = metrics(A, B)
        print("\nOverall")
        print(f"MSE:               {vals[0]:.8g}")
        print(f"RMSE:              {vals[1]:.8g}")
        print(f"MAE:               {vals[2]:.8g}")
        print(f"Relative RMSE:     {vals[3]:.8g}")
        print(f"Cosine similarity: {vals[4]:.8g}")
        print(f"Accuracy percent:  {vals[5]:.4f}")


if __name__ == "__main__":
    main()
