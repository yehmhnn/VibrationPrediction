from pathlib import Path
import csv
import json
import hashlib
import numpy as np

O = Path(__file__).resolve().parent


def read(n):
    return list(csv.DictReader((O / n).open()))


a = read("metrics.csv")
bands = read("band_power.csv")
conf = json.loads((O / "config_and_provenance.json").read_text())
header = "| Record | Method | RMS original → denoised | Residual RMS | Welch power original → denoised | Power change | 2–4 kHz power change | Strongest original peak retained |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
table = header
for r in a:
    table += f"| {r['record_id']} | {r['method']} | {float(r['original_rms']):.4f} → {float(r['denoised_rms']):.4f} | {float(r['residual_rms']):.4f} | {float(r['original_welch_power']):.5f} → {float(r['denoised_welch_power']):.5f} | {float(r['relative_power_change']) * 100:.2f}% | {float(r['relative_2to4khz_power_change']) * 100:.2f}% | {float(r['original_peak_same_bin_psd_ratio']) * 100:.3f}% |\n"
md = (
    "# Numerical summaries\n\n"
    + table
    + "\nRMS in source signal units; Welch power in source units². Peak retention is PSD at the original strongest-peak bin, not a validated signal-fidelity score.\n\n## Band fractions: original → denoised (%)\n\n| Record | Method | 0–1 kHz | 1–2 kHz | 2–3 kHz | 3–4 kHz | 4–5 kHz | 5–6 kHz |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
)
for r in a:
    bb = [
        b
        for b in bands
        if b["record_id"] == r["record_id"] and b["method"] == r["method"]
    ]
    md += (
        "| "
        + r["record_id"]
        + " | "
        + r["method"]
        + " | "
        + " | ".join(
            f"{float(b['original_fraction']) * 100:.3f} → {float(b['denoised_fraction']) * 100:.3f}"
            for b in bb
        )
        + " |\n"
    )
md += "\nResidual power, fractions and RMS: metrics.csv and band_power.csv. Top-five original peak comparisons, with same-bin and ±2-bin local maximum matching: peak_changes.csv. Automatic sigma and per-level thresholds: threshold_diagnostics.csv.\n"
(O / "summary_tables.md").write_text(md)
text = (
    """# Study II wavelet thresholding: offline signal-analysis experiment

2026-10-28. **Main finding:** fixed db4/BayesShrink thresholding removes substantial structured vibration, especially in ball-fault recordings. The current results do not justify treating the reconstructions as clean ground truth or adopting this configuration for forecasting.

## Verified setup and consequential choices

The prior `analysis/studyII_welch_20261028/REPORT.md`, `summary_tables.md`, scripts and figures were reviewed. Exact assignments were read from the repository's `VibrationPrediction_Report_TwoStudies_20261016/evidence/datasets/processed/cwru_multi_record_B/manifest.csv`, filtering before signal access. Inner race: 105_0/106_1 training, 107_2 validation. Ball: 118_0/119_1 training, 120_2 validation. Outer race (@6): 130@6_0/131@6_1 training, 132@6_2 validation. Training loads are 0/1 hp, validation 2 hp. Only these nine previously copied MAT files were opened, and all source hashes match the manifest. No reserved 3 hp signal was accessed.

Signals are loaded as float64 from the manifest-designated `X*_DE_time` channels, in their original numerical acceleration scale. Physical calibration/units were not independently verified, so figures use “source signal units,” not an assumed calibrated unit. No dataset/model normalization, amplitude rescaling, resampling, or record concatenation is applied. Sampling remains 12,000 Hz and all signal lengths are preserved, including odd lengths.

The original local interpreter lacked scikit-image and PyWavelets. An isolated temporary environment was created at `<TEMP_DIR>/studyII_wavelet_env`; the forecasting environment was not changed. Installed experiment versions: **scikit-image 0.24.0, PyWavelets 1.6.0, SciPy 1.13.1, NumPy 2.0.2**. The installed signature and source were inspected and saved in `installed_api_source.txt`; both modes are supported directly by `skimage.restoration.denoise_wavelet`. No substitute denoiser was needed.

Fixed invocation for every record: `wavelet='db4'`, `wavelet_levels=3`, `sigma=None`, `method='BayesShrink'`, `mode='soft'` or `'hard'`, `rescale_sigma=True`, `convert2ycbcr=False`, `channel_axis=None`. Inputs are floating point; no clipping to an image display range is applied. The installed implementation calls PyWavelets with its symmetric boundary extension and crops reconstruction to the original extent. Hard uses the same library-estimated subband thresholds as soft; it is an API-supported comparison, not a claim of Bayes optimality for hard shrinkage.

**Level tradeoff:** three levels are explicit rather than the length-dependent library default. This gives nominal D1/D2/D3 bands of 3–6/1.5–3/0.75–1.5 kHz, leaves the coarse approximation unthresholded, and limits coarse-scale modification. These are approximate filter bands, not sharp passbands. The choice was fixed before results, with no parameter search. Three levels do not protect 2–4 kHz: that region overlaps thresholded details. Symmetric boundaries and a decimated DWT introduce edge and shift sensitivity.

**Automatic estimation tradeoff:** the library estimates sigma as median absolute nonzero finest-detail coefficient divided by the Gaussian 75th-percentile constant. It then estimates a separate BayesShrink threshold per detail level. Sigma and thresholds vary by record because the same adaptive rule is applied; they were not individually tuned or chosen using forecasting scores. Genuine vibration in the finest band can inflate the estimate. This is an assumption-driven noise estimate, not a measurement of sensor noise. Full-record estimation is suitable only for this offline analysis.

Welch parameters are unchanged from the verified prior analysis: Hann window, fs=12000, nperseg=nfft=8192, noverlap=4096, constant detrending, one-sided density scaling, mean averaging, no smoothing. Frequency spacing is 1.46484375 Hz, with 28 overlapping segments per recording. Original integrated powers reproduce the prior CSV values within 1e−12 relative tolerance.

## Numerical definitions

For original x, reconstruction y and residual r=x−y: RMS(z)=sqrt(mean(z²)). Integrated Welch power W(z)=sum_k PSD_z[k] × Δf. Relative power change=(W(y)−W(x))/W(x). Bands are [0,1000), …, [5000,6000] Hz, with nonoverlapping bin masks; band fraction is band power divided by that signal's own total Welch power. Residual power and fractions use the residual's own PSD. Comparing fractions alone can hide a large loss of absolute power.

Top-five original peaks are local maxima separated by at least 50 Hz, ranked by PSD. Their reconstructed PSD is measured at the identical bin and at the local maximum within ±2 bins (±2.93 Hz). The strongest reconstructed peak is also reported independently. Peak selection is descriptive, not mechanical component identification. We additionally report centered crest factor, excess kurtosis and amplitude retained at the original largest absolute-deviation sample; these are distortion indicators, not certified impulse preservation.

A “background proxy” is median PSD over 2–4 kHz after excluding ±10 Hz around the original spectrum's 20 strongest maxima separated by at least 20 Hz. This remaining spectrum still contains structured components and cannot be equated to noise. Residual spectral flatness is geometric/arithmetic mean PSD, excluding DC for the full band; 1 denotes a flat estimated spectrum, but colored noise need not be flat. Neither proxy proves a noise model.

**Energy accounting:** mean(x²)=mean(y²)+mean(r²)+2mean(yr). The cross term is exported and the identity checked to absolute error below 1e−12. Consequently, W(x)−W(y) is not residual power and neither quantity is “removed noise energy.” Welch detrending/windowing also distinguishes spectral estimates from exact full-record mean square. No SNR, PSNR or clean-signal accuracy is reported because no clean reference exists.

## Results

"""
    + table
    + """
Full precision, residual powers, band fractions, peak shifts, event retention and threshold diagnostics are in the accompanying CSVs and `summary_tables.md`.

**Inner race — observed:** soft reduces total power by 49.3–56.3% and 2–4 kHz power by 50.9–57.7%; the strongest original peak retains only 21.2–42.3% of its PSD. For the two training records, the strongest reconstructed frequency changes to 2778.8 and 2841.8 Hz. Hard reduces total power by 4.9–8.0%, retaining 69.3–88.6% of the strongest original-peak PSD. Soft retains about 78–83% of amplitude at the original extreme-deviation event, hard about 99–100%. Hard is less destructive here but is not verified noise-only removal.

**Ball — observed:** soft reduces total power by 94.5–97.1% and 2–4 kHz power by 98.9–99.98%. Hard reduces total power by 79.9–96.6% and 2–4 kHz power by 84.9–99.6%. The dominant peak moves from approximately 3.4 kHz to 360–419 Hz for both modes. In validation 120_2, both reconstructions retain only about 7.2% of the amplitude at the original extreme event; the comparison figure visibly loses rapid oscillation and that event. Roughly 92–96% of ball residual spectral power lies at 2–4 kHz. This is strong evidence of removal of the observed dominant vibration structure, not proof that it was noise. Automatic sigma is 0.147–0.165, comparable to or greater than original RMS 0.139–0.147: the Gaussian finest-detail assumption is particularly consequential here.

**Outer race — observed:** soft reduces total power by 20.3–23.7%, retaining 57.7–79.1% of original strongest-peak PSD. Hard changes total power by only −0.8 to −1.2% and 2–4 kHz power by −0.6 to −0.8%, retaining 93.7–100.8% of the strongest original-peak PSD. A single PSD bin can increase after nonlinear reconstruction despite lower total power. Extreme-event amplitude retention is about 92–93% for soft and 97–99% for hard. Major bursts are visually retained in 132@6_2, while soft attenuates their oscillatory content. This does not establish preservation of every transient or frequency.

**Background and residuals — observed:** soft residuals concentrate about 71% (inner race), 94–96% (ball), and 77–81% (outer race) of their power at 2–4 kHz. Full-band residual flatness ranges about 0.009–0.111 for soft and 0.010–0.379 for hard, and residual PSDs have peaks and shaped envelopes. Soft lowers the defined background proxy to 62–73% of original for inner race and 0.1–12.3% for ball, but outer-race values are 92–102%. Hard often increases that proxy (about 101–152% except validation ball at 9.7%). Thresholding therefore does not uniformly lower a broadband background; reconstruction may redistribute power or produce artifacts. A structured residual can include colored noise, but these results do not support identifying the entire residual as broadband noise.

## Direct answers and interpretation

1. **What is removed?** Wavelet detail coefficients below adaptive thresholds, plus shrinkage of retained coefficients for soft. In the signal domain this includes substantial dominant-band power, narrow peaks and some large waveform events—not just visually small fluctuations.
2. **Is the residual broadband noise?** Not established. It has broad support but also strong non-flat spectral structure, especially the original ball 2–4 kHz content. Broad support alone is not whiteness or absence of physical signal.
3. **Are important structures preserved?** Unevenly. Hard largely preserves the inspected outer-race peaks/bursts; soft attenuates them. Both methods severely alter ball waveforms. Spectral amplitude and event preservation provide partial evidence, not physical validation of which features matter.
4. **Are effects consistent across train/validation?** The qualitative fault-dependent pattern occurs at all loads: ball is most attenuated and outer-race hard least. Magnitudes differ; 120_2 is more heavily suppressed by hard thresholding than either ball training recording. One validation acquisition per fault cannot establish robustness or causality.
5. **Enough evidence for forecasting?** Not enough to adopt these settings or call the targets clean. There is enough evidence to motivate a small signal-preservation study of the automatic sigma assumption, using training recordings and synthetic known-noise controls. Forecasting should wait until a defensible fixed candidate preserves the relevant structures. These findings neither explain nor solve TSFlow/TimesNet overfitting; no forecasting performance was consulted.

## Future-information leakage: three distinct uses

**A — offline analysis:** complete-record DWT and full-record sigma estimation use samples on both sides of a time point. That is explicitly what was done here, independently for each authorized recording. These reconstructions must not be silently substituted for online model inputs.

**B — forecasting inputs:** for each forecast origin, extract the 512 past samples first, then denoise that context alone with fixed parameters and past-only sigma estimation, or use a documented streaming procedure whose right boundary never crosses the origin. Do not denoise a complete recording and then split it into contexts. Even a fixed sigma does not fix future use by the transform. Boundary padding must use only available context; all training/validation feature transformations must follow the same rule. This can alter offline results and needs separate validation. No state, parameter estimate or normalization may cross split boundaries.

**C — estimated clean targets:** an offline denoised target is an estimated alternative label, not ground truth. If constructing such a future benchmark, keep transforms inside split/record boundaries, document any future samples used for label construction, and ensure target construction cannot alter inputs or conditioning. Pick a common target definition before comparing systems. To test input denoising, train/evaluate raw-input and context-denoised-input models against the **same raw targets** first. An alternative estimated-clean-target experiment must use the same frozen estimated targets for every competing model and report raw-target error as a separate endpoint. Never compare raw-target RMSE from one model with denoised-target RMSE from another as evidence of improvement. A target transformation that makes signals easier is not proof of better raw-vibration prediction.

## Proposed next steps — not executed

Preserve this fixed experiment, including its unfavorable results. First test the denoiser on training-only segments with known synthetic additive noise overlaid on observed signals, while acknowledging those observed signals are not clean. Compare reconstruction against the pre-injection reference and examine distortion of peaks/events; this isolates recovery of added noise, not real sensor noise. If available, acquire a physically justified noise reference rather than letting vibration-rich finest details define noise uncritically. Any weaker sigma rule or level change should be a separate preregistered analysis, not per-validation tuning. Do not launch a large threshold sweep or retrain models on the current outputs.

## Reproducibility, checks and files

`analyze.py` is the only signal-processing script; `write_report.py` renders tables and this report. `config_and_provenance.json` records the API, all parameters, versions, allowed input hashes, length/finiteness and energy checks, and confirmation that prior PSD-analysis files are unchanged. `installed_api_source.txt` records exact installed behavior. `requirements-analysis.txt` and `environment_freeze.txt` document dependencies; the temporary environment can be recreated without changing forecasting dependencies.

Nine `*_comparison.png/.pdf` figures contain the common 1.000–1.050 s waveform, residual on the same amplitude scale, full PSDs, residual PSDs, a deterministic 20-ms largest-event view, and 2–4 kHz PSD detail. `denoising_overview.png/.pdf` summarizes power/peak changes. Figures use source units and legends. `*_signals.npz` preserve originals, reconstructions and residuals at 12 kHz within this directory only. `metrics.csv`, `band_power.csv`, `peak_changes.csv`, `threshold_diagnostics.csv`, `welch_spectra.csv` and `summary_tables.md` provide numerical results. No code/configuration/checkpoint/result in the forecasting experiments was changed.

Run from project root using the isolated interpreter: `<TEMP_DIR>/studyII_wavelet_env/bin/python -B analysis/studyII_wavelet_denoising_20261028/analyze.py`, then the same interpreter with `write_report.py`. These commands replace only this analysis's generated outputs. Input files are read from the preceding Welch analysis and must retain manifest hashes.

Documentation: [scikit-image denoise_wavelet API](https://scikit-image.org/docs/0.24.x/api/skimage.restoration.html#skimage.restoration.denoise_wavelet) and [PyWavelets multilevel DWT](https://pywavelets.readthedocs.io/en/stable/ref/dwt-discrete-wavelet-transform.html). Installed source inspection, rather than latest online defaults, governs this experiment.
"""
)
(O / "REPORT.md").write_text(text)
