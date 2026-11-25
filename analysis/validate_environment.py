"Bounded offline numerical check; no model imports, training or reserved recordings."

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vibration_analysis_mpl")
)
import numpy as np
import scipy
import pywt
import skimage
import matplotlib
from scipy.io import loadmat
from scipy.signal import welch
from skimage.restoration import denoise_wavelet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Output already exists; use a fresh destination")
    root = Path(__file__).resolve().parents[1]
    manifest = root / "evidence/datasets/cwru_multi_record_B/manifest.csv"
    row = next(r for r in csv.DictReader(manifest.open()) if r["record_id"] == "105_0")
    assert row["split"] == "train" and int(row["load_id"]) == 0
    digest = hashlib.sha256(a.input.read_bytes()).hexdigest()
    assert digest == row["source_sha256"], "Expected authorized 105_0 input hash"
    x = np.asarray(
        loadmat(a.input, variable_names=[row["source_key"]])[row["source_key"]],
        dtype=np.float64,
    ).ravel()
    assert len(x) == int(row["num_samples"]) and np.isfinite(x).all()
    params = dict(
        fs=12000,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        nfft=8192,
        detrend="constant",
        scaling="density",
        average="mean",
        return_onesided=True,
    )
    f, P = welch(x, **params)
    power = float(P.sum() * (f[1] - f[0]))
    old = next(
        r
        for r in csv.DictReader(
            (root / "analysis/studyII_welch_20261028/record_summary.csv").open()
        )
        if r["record_id"] == "105_0"
    )
    assert np.isclose(
        power, float(old["welch_power_original_units2"]), rtol=1e-12, atol=1e-14
    )
    result = {
        "record_id": "105_0",
        "input_sha256": digest,
        "samples": len(x),
        "welch_power": power,
        "reserved_signal_access": False,
        "models_executed": False,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywt": pywt.__version__,
            "skimage": skimage.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "modes": {},
    }
    rows = list(
        csv.DictReader(
            (root / "analysis/studyII_wavelet_denoising_20261028/metrics.csv").open()
        )
    )
    for mode in ["soft", "hard"]:
        y = denoise_wavelet(
            x,
            wavelet="db4",
            wavelet_levels=3,
            sigma=None,
            method="BayesShrink",
            mode=mode,
            rescale_sigma=True,
            convert2ycbcr=False,
            channel_axis=None,
        )
        assert y.shape == x.shape and np.isfinite(y).all()
        old = next(r for r in rows if r["record_id"] == "105_0" and r["method"] == mode)
        rms = float(np.sqrt(np.mean(y * y)))
        assert np.isclose(rms, float(old["denoised_rms"]), rtol=1e-12, atol=1e-14)
        z = x - y
        error = float(
            abs(np.mean(x * x) - np.mean(y * y) - np.mean(z * z) - 2 * np.mean(y * z))
        )
        assert error < 1e-12
        result["modes"][mode] = {
            "rms": rms,
            "energy_identity_error": error,
            "saved_metrics_match": True,
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        (
            "PASS: one authorized training recording; Welch and both wavelet modes "
            "match saved metrics"
        )
    )


if __name__ == "__main__":
    main()
