"""Compare fixed wavelet reconstructions and spectra on full recordings."""

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(__import__("tempfile").gettempdir(), "vibration_analysis_mpl"),
)
from pathlib import Path
import csv
import json
import inspect
import hashlib
import platform
import numpy as np
import scipy
import pywt
import skimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis
from skimage.restoration import denoise_wavelet
import skimage.restoration._denoise as implementation

output_dir = Path(__file__).resolve().parent
project_root = output_dir.parents[1]
welch_dir = project_root / "analysis/studyII_welch_20261028"


def sha(path):
    """Hash file contents for provenance checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvout(name, rows):
    """Write rows in the field order of the first row."""
    with (output_dir / name).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


manifest = project_root / "evidence/datasets/cwru_multi_record_B/manifest.csv"
rows = [
    r for r in csv.DictReader(manifest.open()) if r["split"] in ("train", "validation")
]
expected = {
    "105_0",
    "106_1",
    "107_2",
    "118_0",
    "119_1",
    "120_2",
    "130@6_0",
    "131@6_1",
    "132@6_2",
}
assert {r["record_id"] for r in rows} == expected and all(
    int(r["load_id"]) < 3 for r in rows
)
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
settings = dict(
    sigma=None,
    wavelet="db4",
    wavelet_levels=3,
    convert2ycbcr=False,
    method="BayesShrink",
    rescale_sigma=True,
    channel_axis=None,
)
config = dict(
    versions={
        k: detail_mean_square
        for k, detail_mean_square in [
            ("python", platform.python_version()),
            ("numpy", np.__version__),
            ("scipy", scipy.__version__),
            ("PyWavelets", pywt.__version__),
            ("scikit-image", skimage.__version__),
            ("matplotlib", matplotlib.__version__),
        ]
    },
    signature=str(inspect.signature(denoise_wavelet)),
    settings=settings,
    threshold_modes=["soft", "hard"],
    boundary="symmetric, inherited PyWavelets default verified by inspection",
    welch=params,
    main_time_interval_seconds=[1, 1.05],
    impulse_window=(
        "20 ms centered on original maximum absolute deviation from mean, "
        "clipped at endpoints"
    ),
    source_hashes={str(manifest.relative_to(project_root)): sha(manifest)},
    units=(
        "original MAT numerical signal units; "
        "physical calibration not independently verified"
    ),
    scope=(
        "offline full-record signal analysis; "
        "not leakage-safe forecasting preprocessing"
    ),
)
prior_file_hashes = {
    str(path.relative_to(project_root)): sha(path)
    for path in welch_dir.rglob("*")
    if path.is_file()
}
(output_dir / "installed_api_source.txt").write_text(
    inspect.getsource(denoise_wavelet)
    + "\n"
    + inspect.getsource(implementation._wavelet_threshold)
    + "\n"
    + inspect.getsource(implementation._sigma_est_dwt)
    + "\n"
    + inspect.getsource(implementation._bayes_thresh)
)
assert "mode='soft'" in str(inspect.signature(denoise_wavelet))
summary = []
bands = []
peaks = []
thresholds = []
checks = []
spectral_rows = []
plt.rcParams.update(
    {
        "font.size": 9,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = {"original": "#30343b", "soft": "#D55E00", "hard": "#0072B2"}


def rms(signal):
    """Return the root mean square amplitude."""
    return float(np.sqrt(np.mean(signal * signal)))


def flat(spectrum):
    """Compute spectral flatness with a floor for the logarithm."""
    return float(
        np.exp(np.mean(np.log(np.maximum(spectrum, np.finfo(float).tiny))))
        / np.mean(spectrum)
    )


def save(fig, name):
    """Save PNG and PDF copies and close the figure."""
    fig.savefig(output_dir / (name + ".png"), bbox_inches="tight")
    fig.savefig(output_dir / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


for row in rows:
    record_id = row["record_id"]
    path = welch_dir / "inputs" / Path(row["source_path"]).name
    assert sha(path) == row["source_sha256"]
    config["source_hashes"][str(path.relative_to(project_root))] = sha(path)
    mat_data = loadmat(path)
    signal = np.asarray(mat_data[row["source_key"]], dtype=np.float64).ravel()
    assert (
        len(signal) == int(row["num_samples"])
        and int(row["sampling_rate_hz"]) == 12000
        and np.isfinite(signal).all()
    )
    frequency_hz, original_psd = welch(signal, **params)
    bin_width_hz = frequency_hz[1] - frequency_hz[0]
    power = original_psd.sum() * bin_width_hz
    # Match original PSD against prior saved computation without rerunning it.
    prior = next(
        r
        for r in csv.DictReader((welch_dir / "record_summary.csv").open())
        if r["record_id"] == record_id
    )
    assert np.isclose(power, float(prior["welch_power_original_units2"]), rtol=1e-12)

    peak_bins, _ = find_peaks(original_psd, distance=int(np.ceil(50 / bin_width_hz)))
    peak_bins = sorted(peak_bins, key=lambda i: original_psd[i], reverse=True)[:5]
    # Reconstruct threshold diagnostics from the inspected implementation.
    # These diagnostics do not feed the denoising call.
    wavelet_coefficients = pywt.wavedecn(signal, "db4", mode="symmetric", level=3)
    detail = wavelet_coefficients[-1]["d"]
    nonzero_detail = detail[detail != 0]
    sigma = float(np.median(np.abs(nonzero_detail)) / scipy.stats.norm.ppf(0.75))
    for j, c in enumerate(wavelet_coefficients[1:]):
        d = c["d"]
        detail_mean_square = np.mean(d * d)
        threshold = sigma**2 / np.sqrt(
            max(detail_mean_square - sigma**2, np.finfo(d.dtype).eps)
        )
        thresholds.append(
            dict(
                record_id=record_id,
                detail_level=3 - j,
                sigma_estimate=sigma,
                threshold=threshold,
                coefficient_count=len(d),
                fraction_abs_coefficients_below_or_equal_threshold=np.mean(
                    np.abs(d) <= threshold
                ),
            )
        )
    signals = {"original": signal}

    signal_spectra = {"original": original_psd}
    residual_signals = {}
    residual_spectra = {}
    for mode in ["soft", "hard"]:
        signal_values = denoise_wavelet(signal, mode=mode, **settings)
        assert signal_values.shape == signal.shape and np.isfinite(signal_values).all()
        residual = signal - signal_values
        _, denoised_psd = welch(signal_values, **params)
        _, residual_psd = welch(residual, **params)
        signals[mode] = signal_values
        signal_spectra[mode] = denoised_psd
        residual_signals[mode] = residual
        residual_spectra[mode] = residual_psd
        cross_term = 2 * np.mean(signal_values * residual)
        energy_identity_error = (
            np.mean(signal * signal)
            - np.mean(signal_values * signal_values)
            - np.mean(residual * residual)
            - cross_term
        )
        band_2to4khz = (frequency_hz >= 2000) & (frequency_hz < 4000)
        # Fixed peak-exclusion mask for an explicitly descriptive background proxy.
        excluded_peak_bins, _ = find_peaks(
            original_psd, distance=int(np.ceil(20 / bin_width_hz))
        )
        excluded_peak_bins = sorted(
            excluded_peak_bins, key=lambda i: original_psd[i], reverse=True
        )[:20]
        background_mask = band_2to4khz.copy()
        for i in excluded_peak_bins:
            background_mask &= np.abs(frequency_hz - frequency_hz[i]) > 10
        strongest_original_bin = peak_bins[0]
        strongest_denoised_bin = int(np.argmax(denoised_psd[1:]) + 1)
        strongest_residual_bin = int(np.argmax(residual_psd[1:]) + 1)
        metrics_row = dict(
            record_id=record_id,
            fault_type=row["fault_type"],
            split=row["split"],
            load_hp=int(row["load_id"]),
            method=mode,
            sigma_estimate=sigma,
            original_rms=rms(signal),
            denoised_rms=rms(signal_values),
            residual_rms=rms(residual),
            original_welch_power=power,
            denoised_welch_power=denoised_psd.sum() * bin_width_hz,
            relative_power_change=denoised_psd.sum() / original_psd.sum() - 1,
            residual_welch_power=residual_psd.sum() * bin_width_hz,
            residual_power_over_original=residual_psd.sum() / original_psd.sum(),
            original_2to4khz_power=original_psd[band_2to4khz].sum() * bin_width_hz,
            denoised_2to4khz_power=denoised_psd[band_2to4khz].sum() * bin_width_hz,
            relative_2to4khz_power_change=denoised_psd[band_2to4khz].sum()
            / original_psd[band_2to4khz].sum()
            - 1,
            residual_2to4khz_fraction=residual_psd[band_2to4khz].sum()
            / residual_psd.sum(),
            original_peak_hz=frequency_hz[strongest_original_bin],
            denoised_peak_hz=frequency_hz[strongest_denoised_bin],
            original_peak_same_bin_psd_ratio=denoised_psd[strongest_original_bin]
            / original_psd[strongest_original_bin],
            residual_peak_hz=frequency_hz[strongest_residual_bin],
            residual_flatness_0to6khz=flat(residual_psd[1:]),
            residual_flatness_2to4khz=flat(residual_psd[band_2to4khz]),
            background_proxy_original_median=original_psd[background_mask].mean()
            if False
            else np.median(original_psd[background_mask]),
            background_proxy_denoised_median=np.median(denoised_psd[background_mask]),
            background_proxy_ratio=np.median(denoised_psd[background_mask])
            / np.median(original_psd[background_mask]),
            original_crest_factor=np.max(np.abs(signal - signal.mean())) / signal.std(),
            denoised_crest_factor=np.max(np.abs(signal_values - signal_values.mean()))
            / signal_values.std(),
            original_excess_kurtosis=kurtosis(signal),
            denoised_excess_kurtosis=kurtosis(signal_values),
            original_max_deviation_event_retention=abs(
                signal_values[np.argmax(abs(signal - signal.mean()))]
                - signal_values.mean()
            )
            / np.max(abs(signal - signal.mean())),
            twice_mean_denoised_residual_product=cross_term,
            time_energy_identity_error=energy_identity_error,
        )
        summary.append(metrics_row)
        for lo in range(0, 6000, 1000):
            mask = (frequency_hz >= lo) & (
                (frequency_hz < lo + 1000) if lo < 5000 else frequency_hz <= 6000
            )
            bands.append(
                dict(
                    record_id=record_id,
                    method=mode,
                    band_low_hz=lo,
                    band_high_hz=lo + 1000,
                    original_power=original_psd[mask].sum() * bin_width_hz,
                    denoised_power=denoised_psd[mask].sum() * bin_width_hz,
                    residual_power=residual_psd[mask].sum() * bin_width_hz,
                    original_fraction=original_psd[mask].sum() / original_psd.sum(),
                    denoised_fraction=denoised_psd[mask].sum() / denoised_psd.sum(),
                    residual_fraction=residual_psd[mask].sum() / residual_psd.sum(),
                )
            )
        for rank, i in enumerate(peak_bins, 1):
            sl = slice(max(1, i - 2), min(len(frequency_hz), i + 3))
            local = max(1, i - 2) + np.argmax(denoised_psd[sl])
            peaks.append(
                dict(
                    record_id=record_id,
                    method=mode,
                    original_peak_rank=rank,
                    original_peak_hz=frequency_hz[i],
                    original_psd=original_psd[i],
                    denoised_same_bin_psd=denoised_psd[i],
                    same_bin_ratio=denoised_psd[i] / original_psd[i],
                    local_max_hz=frequency_hz[local],
                    local_max_ratio=denoised_psd[local] / original_psd[i],
                    residual_same_bin_psd=residual_psd[i],
                )
            )
        checks.append(
            dict(
                record_id=record_id,
                method=mode,
                length_preserved=len(signal_values) == len(signal),
                finite=True,
                energy_identity_absolute_error=abs(energy_identity_error),
            )
        )
    for i, hz in enumerate(frequency_hz):
        spectral_rows.append(
            dict(
                record_id=record_id,
                frequency_hz=hz,
                original=original_psd[i],
                soft=signal_spectra["soft"][i],
                hard=signal_spectra["hard"][i],
                soft_residual=residual_spectra["soft"][i],
                hard_residual=residual_spectra["hard"][i],
            )
        )
    np.savez_compressed(
        output_dir / (record_id + "_signals.npz"),
        original=signal,
        soft=signals["soft"],
        hard=signals["hard"],
        soft_residual=residual_signals["soft"],
        hard_residual=residual_signals["hard"],
        sampling_rate_hz=12000,
    )

    fig, ax = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    time_seconds = np.arange(len(signal)) / 12000
    overview_interval = (time_seconds >= 1) & (time_seconds < 1.05)
    event_index = np.argmax(abs(signal - signal.mean()))
    start = max(0, min(event_index - 120, len(signal) - 240))
    event_interval = slice(start, start + 240)
    for key, signal_values in signals.items():
        ax[0, 0].plot(
            time_seconds[overview_interval],
            signal_values[overview_interval],
            color=colors[key],
            lw=0.8,
            label=key,
        )
        ax[2, 0].plot(
            time_seconds[event_interval],
            signal_values[event_interval],
            color=colors[key],
            lw=0.8,
            label=key,
        )
        ax[0, 1].semilogy(
            frequency_hz[1:],
            signal_spectra[key][1:],
            color=colors[key],
            lw=0.8,
            label=key,
        )
        ax[2, 1].semilogy(
            frequency_hz, signal_spectra[key], color=colors[key], lw=0.8, label=key
        )
    for mode in ["soft", "hard"]:
        ax[1, 0].plot(
            time_seconds[overview_interval],
            residual_signals[mode][overview_interval],
            color=colors[mode],
            lw=0.8,
            label=mode + " residual",
        )
        ax[1, 1].semilogy(
            frequency_hz[1:],
            residual_spectra[mode][1:],
            color=colors[mode],
            lw=0.8,
            label=mode + " residual",
        )
    lim = (
        max(
            abs(signal[overview_interval]).max(),
            *(
                abs(signal_values[overview_interval]).max()
                for signal_values in signals.values()
            ),
        )
        * 1.08
    )
    ax[0, 0].set(
        ylim=(-lim, lim), title="Common interval: original and reconstructions"
    )
    ax[1, 0].set(ylim=(-lim, lim), title="Residuals: same amplitude scale as row above")
    ax[2, 0].set_title("20 ms around largest original absolute deviation")
    for a in ax[:, 0]:
        a.set(xlabel="Time (s)", ylabel="Source signal units")
    for a in ax[:, 1]:
        a.set(xlabel="Frequency (Hz)", ylabel="PSD (source units²/Hz)")
    ax[0, 1].set(xlim=(0, 6000), title="Welch PSD: original and reconstructions")
    ax[1, 1].set(xlim=(0, 6000), title="Residual Welch PSD")
    ax[2, 1].set(xlim=(2000, 4000), title="Preservation of the 2–4 kHz structure")
    for a in [ax[0, 1], ax[1, 1]]:
        a.axvspan(2000, 4000, color="#c7d6e3", alpha=0.15)
    for a in ax.flat:
        a.grid(alpha=0.18)
        a.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{record_id} | {row['fault_type']} | {row['load_id']} hp {row['split']} | "
        "db4 / 3 levels / BayesShrink",
        fontsize=13,
    )
    save(fig, record_id + "_comparison")
    print(record_id, "completed", flush=True)
csvout("metrics.csv", summary)
csvout("band_power.csv", bands)
csvout("peak_changes.csv", peaks)
csvout("threshold_diagnostics.csv", thresholds)
csvout("welch_spectra.csv", spectral_rows)
config["analysis_script_sha256"] = sha(Path(__file__))
config["checks"] = checks
config["prior_analysis_unchanged"] = all(
    sha(project_root / k) == detail_mean_square
    for k, detail_mean_square in prior_file_hashes.items()
)
assert config["prior_analysis_unchanged"]
assert max(c["energy_identity_absolute_error"] for c in checks) < 1e-12
(output_dir / "config_and_provenance.json").write_text(
    json.dumps(config, indent=2) + "\n"
)
fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
names = [r["record_id"] for r in rows]
pos = np.arange(9)
for mode, shift in [("soft", -0.18), ("hard", 0.18)]:
    mode_metrics = [
        metrics_row for metrics_row in summary if metrics_row["method"] == mode
    ]
    for a, key, scale, title in [
        (ax[0], "relative_power_change", 100, "Change in total Welch power (%)"),
        (ax[1], "relative_2to4khz_power_change", 100, "Change in 2–4 kHz power (%)"),
        (
            ax[2],
            "original_peak_same_bin_psd_ratio",
            100,
            "Original strongest peak PSD retained (%)",
        ),
    ]:
        a.barh(
            pos + shift,
            [metrics_row[key] * scale for metrics_row in mode_metrics],
            height=0.34,
            color=colors[mode],
            label=mode,
        )
        a.set(
            title=title,
            yticks=pos,
            yticklabels=names if a == ax[0] else [],
            ylim=(8.6, -0.6),
        )
        a.grid(axis="x", alpha=0.2)
ax[0].legend()
fig.suptitle("Fixed db4/BayesShrink denoising · offline full recordings", fontsize=14)
save(fig, "denoising_overview")
