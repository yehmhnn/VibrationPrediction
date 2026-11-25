import glob
import os
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal


def preprocess(output_path=None, downsample=True, target_fs=10000):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "ottawa")
    data_dir = os.path.abspath(data_dir)
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))

    if not mat_files:
        raise FileNotFoundError(
            "No Ottawa .mat files found. Please place .mat files in datasets/ottawa/."
        )

    print(f"Found {len(mat_files)} Ottawa .mat files. Extracting Channel_1...")
    signals = []
    for file in mat_files:
        values = scipy.io.loadmat(file)
        if "Channel_1" not in values:
            raise KeyError(f"Channel_1 not found in {file}")
        signal = values["Channel_1"].reshape(-1)
        signals.append(signal.astype(np.float64))

    full_signal = np.concatenate(signals)
    full_signal = np.nan_to_num(full_signal, nan=0.0, posinf=0.0, neginf=0.0)

    if downsample:
        print(
            (
                "Downsampling Ottawa signal to 10 kHz for consistency with the existing"
                " plotting code..."
            )
        )
        full_signal = scipy.signal.resample_poly(full_signal, up=1, down=20)

    mean = np.mean(full_signal)
    std = np.std(full_signal)
    full_signal = (full_signal - mean) / (std + 1e-8)
    full_signal = full_signal.astype(np.float32)

    if output_path is None:
        output_path = os.path.join(data_dir, "ottawa_processed_signal.npy")

    np.save(output_path, full_signal)
    print(
        f"Saved preprocessed Ottawa signal to {output_path}. Length = {len(full_signal)}"
    )
    return output_path


if __name__ == "__main__":
    preprocess()
