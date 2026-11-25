from pathlib import Path
import csv
import json

O = Path(__file__).resolve().parent
ROOT = O.parents[1]


def read(n):
    return list(csv.DictReader((O / n).open()))


s = read("record_summary.csv")
b = read("band_power.csv")
rm = read("validation_rmse_context.csv")
cmp = read("spectral_comparisons.csv")
tc = read("training_shape_comparisons.csv")
table = "| Fault / recording | Role / hp | RMS | Welch power | Strongest peak (Hz) | Power in 2–4 kHz |\n|---|---|---:|---:|---:|---:|\n"
for r in s:
    band = sum(
        float(t["fraction_total"])
        for t in b
        if t["record_id"] == r["record_id"] and t["band_low_hz"] in ("2000", "3000")
    )
    table += f"| {r['record_id']} | {r['split']} / {r['load_hp']} | {float(r['rms_original_units']):.4f} | {float(r['welch_power_original_units2']):.5f} | {float(r['dominant_peak_hz']):.1f} | {band * 100:.2f}% |\n"
rt = "| Validation recording | Zero RMSE | TSFlow RMSE / zero ratio | TimesNet RMSE / zero ratio |\n|---|---:|---:|---:|\n"
for rid in ["107_2", "120_2", "132@6_2"]:
    rr = [r for r in rm if r["record_id"] == rid]
    rt += (
        f"| {rid} | {float(rr[0]['zero_forecast_rmse_model_units']):.4f} | "
        + " | ".join(
            f"{float(r['selected_checkpoint_rmse_model_units']):.4f} / {float(r['rmse_divided_by_zero']):.3f}"
            for r in rr
        )
        + " |\n"
    )
text = (
    """# Study II: training–validation Welch PSD comparison

2026-10-28. Descriptive analysis only; no training or model inference. Observed means directly calculated or read from saved results; possible explanations are hypotheses, not causal findings.

## Research summary for discussion

**Observed.** The nine Study II recordings share broad spectral regions but differ substantially in the positions and relative heights of narrow peaks. Approximately 77–78% of inner-race, 90–93% of ball, and 97–98% of outer-race spectral power lies at 2–4 kHz. Validation does not introduce a new dominant broad band. Ball validation is more concentrated at 3–4 kHz; outer-race validation has lower total power than both of its training recordings. Outer-race amplitude nevertheless remains substantially larger than inner-race and ball amplitude, which matters when comparing unadjusted RMSE under one shared normalization.

**Possible explanations.** Load, RPM and acquisition identity change together. Peak shifts and altered relative peak strengths could affect a model fitted to training waveforms. Higher target amplitude can also contribute to larger absolute errors. These are plausible considerations, not evidence that spectral shift caused the observed errors or explained overfitting. Broad spectral similarity does not establish similar phase evolution or forecasting difficulty.

**Unresolved.** Whether short-time spectral variation, phase/timing errors, or model limitations account for performance remains unknown. Fine-bin spectral distance is not uniquely larger for the high-error outer-race record, and spectral differences also exist between training recordings. A small context-only Daubechies analysis is reasonable before considering any forecasting intervention. Reserved 3 hp data remain untouched.

## Scope, provenance and method

Assignments are taken directly from `VibrationPrediction_Report_TwoStudies_20261016/evidence/datasets/processed/cwru_multi_record_B/manifest.csv`, filtering to `train` and `validation` **before opening signals**:

- Inner race: training 105_0 / 106_1; validation 107_2.
- Ball: training 118_0 / 119_1; validation 120_2.
- Outer race, @6 position: training 130@6_0 / 131@6_1; validation 132@6_2.

All are 0.007-inch faults, drive-end channels, sampled at 12 kHz. Nine original MAT files were copied from the server into `inputs/`; every file SHA-256 matches the repository manifest. No reserved 3 hp signal was copied, opened or analyzed. The existing manifest includes test assignments, but those signals were excluded. Existing datasets, normalizers, models, configurations, logs and report source were not modified.

The verified settings in `.../data/spectral_method.json` and the implementation in `VibrationPrediction_Report_Revised/analysis/spectral_revision.py` are retained: periodic Hann window, nperseg=nfft=8192, noverlap=4096, per-segment constant detrending, one-sided density scaling, mean averaging, no smoothing. Bin spacing is 1.46484375 Hz; segment duration is 0.683 s. All nine complete recordings supply 28 overlapping segments. This extends the estimator, not the old Study I chronological split or its normalization. Full-record PSD describes acquisition-level structure, whereas forecasting uses 42.67-ms contexts and 8-ms targets.

**Absolute scale:** PSD is computed from original MAT channel values, in source signal units²/Hz. Physical calibration/units were not independently established, so plots do not assert calibrated g²/Hz. Model-scale PSD is also exported: divide raw PSD by the squared Study II training SD, 0.41000491194742045². The shared training mean is 0.010496160688842994; constant detrending removes constant offsets from PSD. No per-record variance scaling is applied to absolute plots.

**Shape:** divide each PSD by its own integrated power, giving unit-area density in 1/Hz. This removes amplitude scale but retains narrow-peak shifts. Integration uses sum(PSD bins) × bin spacing; six disjoint 1-kHz bands partition every bin, including DC and Nyquist. RMS is computed directly from the full time signal; AC RMS, mean square, and variance comparisons are exported separately. Welch power is an estimate of detrended variance, not exactly total mean square. Its relative difference from full-record variance is at most 0.42% here; this is a consistency check, not an estimator confidence interval. No statistical significance tests are claimed from 28 overlapping segments or from single acquisitions.

## Observations: amplitude, power and peaks

"""
    + table
    + """
RMS uses source units; Welch power uses source units². Peak frequencies are strongest estimated local maxima, not identified mechanical fault frequencies. `dominant_peaks.csv` lists five maxima per recording, selected with at least 50-Hz separation to avoid simply listing adjacent bins. It is a descriptive selection rule and can omit closer genuine components.

- **Inner race:** integrated validation power is 4.6–5.6% above the individual training recordings. The strongest peaks are 3587.4, 3539.1 and 2862.3 Hz at 0/1/2 hp. Validation also has a peak at 3490.7 Hz, so the change of strongest peak is not evidence that the 3.5-kHz family disappears. The 3–4-kHz fraction increases from 38.8/40.9% to 42.0%, while 2–3 kHz decreases from 38.6/37.2% to 35.5%. A validation peak at 1445.8 Hz is among its top five.
- **Ball:** validation power is 12.2–13.4% higher. Strongest peaks are 3446.8/3477.5/3432.1 Hz; the validation peak PSD at 3432.1 Hz is approximately 3.1 times the 1-hp record's strongest peak PSD. Its 3–4-kHz share rises from 71.4/73.9% to 78.2%, while 2–3 kHz falls from about 18.9% to 14.6%. This is redistribution within an already strong band.
- **Outer race:** validation power is 27.3% below 0 hp and 7.3% below 1 hp. Strongest peaks are 3336.9/2869.6/3357.4 Hz. Other prominent validation peaks occur at 2833.0 and 3462.9 Hz. The 3–4-kHz share, 69.9%, lies between the training values 73.1% and 68.1%; the 2–3-kHz share, 27.2%, likewise lies between 25.0% and 29.2%. The 5–6-kHz share is 1.19%, versus 0.77/1.18% in training: small in absolute share and close to 1 hp.

**Interpretation limit:** The recordings differ at individual frequency bins, and validation contains prominent lines at locations not equally prominent in training. Because of leakage, noise and finite resolution, this does not establish wholly absent/present physical components. No “novel frequency” detection threshold or component matching model was preregistered. There is no evidence here of a new validation-only dominant broad band.

` spectral_comparisons.csv` reports power ratios and unit-area total-variation/Jensen–Shannon distances at the original bin resolution. Validation versus the equal-record mean training PSD has Jensen–Shannon distance 0.830 (inner race), 0.644 (ball), and 0.780 (outer race), base 2. Training-versus-training values are already 0.845, 0.631 and 0.789. These distances are sensitive to narrow-peak shifts and resolution; the mean-reference comparison is not directly equivalent to a two-record comparison. They are descriptive, not evidence that validation is statistically out of distribution or that outer race has uniquely large shift. Optional ratio CSVs mask DC and reference PSD below 1e−8 of its maximum; no valid ratios are clipped or smoothed.

## Relation to saved validation RMSE

The following uses the **original Study II pooled-validation-selected checkpoints**, TSFlow epoch 8 and TimesNet epoch 1, from their `multi_record_B_*_objective_v1/result.json` files. These are not separately selected per-record minima or later follow-up models. No model evaluation was rerun. The zero forecast is calculated on exactly the same L512/H96, stride96 target grid using the existing global training normalization.

"""
    + rt
    + """
**Observed:** Outer-race errors are largest in absolute normalized units, but its zero-prediction RMSE is also largest. TSFlow is approximately equal to zero on this record; TimesNet is about 11.9% worse. None of these six saved model/record combinations beats zero on this target grid. The high outer-race error therefore must be discussed with target amplitude, rather than interpreted solely as a larger spectral distribution shift. Validation outer-race power is not unusually elevated relative to its own training records.

**Possible explanations, not established:** narrow-peak relocation, phase errors, transient timing or fitting limitations could matter even with similar broad-band envelopes. PSD discards phase and averages time variation. There are only three validation acquisitions; fault, load, RPM and record identity are confounded. These data cannot establish a causal PSD–RMSE relationship or explain training overfitting. Validation data also selected the checkpoints, so these are development-set diagnostics, not independent generalization evidence. Error-spectrum or phase analyses would require saved predictions and targets aligned to the same checkpoint; PSD alone is insufficient.

## Small controlled Daubechies follow-up — proposed only

First perform a **no-training diagnostic** with a fixed db4, three-level DWT on the existing 512-sample contexts separately, without concatenating records. Use only past-context samples and a fixed boundary mode, e.g. symmetric; record boundary sensitivity and reconstruction accuracy. Approximate detail-band labels at 12 kHz would be 3–6, 1.5–3 and 0.75–1.5 kHz, with approximation below 0.75 kHz; these are nominal bands, not ideal brick-wall filters. Report absolute and relative coefficient energy and its variation across contexts/recordings. Do not interpret overlapping contexts as independent acquisitions. Do not denoise, discard bands, or select a wavelet family using validation RMSE at this stage.

If that diagnostic reveals repeatable localized structure worth testing, preregister one intervention: append the four db4 context-energy features to a fixed conditioning pathway while predicting the unchanged raw normalized waveform. Compare against a capacity-matched constant-feature control using three matched seeds, fixed split and equal stopping/budget policies, for one model initially (six fresh runs). Choose feature scaling from training data only; select checkpoints on validation only; report pooled and per-record validation RMSE, zero-relative RMSE, and mean ± sample SD across seeds. Keep reserved 3 hp data closed. Improvement would support only that intervention under this split, not a causal explanation of the current failure. This analysis implements no wavelet transform or retraining.

Method reference for the proposed transform and edge handling: [PyWavelets DWT documentation](https://pywavelets.readthedocs.io/en/stable/ref/dwt-discrete-wavelet-transform.html).

## Deliverables and reproducibility

- `spectral_overview.png/.pdf`: three fault rows, absolute and unit-area columns; common y scales within each column.
- `psd_inner_race_fault`, `psd_ball_fault`, `psd_outer_race_fault`: individual overlay figures, PNG/PDF.
- `band_power_and_amplitude.png/.pdf`: broad-band fractions, RMS and integrated power.
- `record_summary.csv`, `dominant_peaks.csv`, `band_power.csv`: numerical summaries.
- `welch_spectra.csv`: all nine complete PSDs, original/model units and unit-area shape.
- `spectral_comparisons.csv`, `training_shape_comparisons.csv`, `validation_vs_training_*.csv`: descriptive differences.
- `validation_rmse_context.csv`: saved selected-checkpoint scores and common-grid zero references.
- `provenance_and_checks.json`: input SHA-256, settings, library versions and numerical checks.
- `analyze.py`, `write_report.py`, `execution.log`, `fetch_inputs.sh`: reproducible analysis and exact nine-file retrieval recipe; no credentials stored.

From the project root, rerun only this dedicated analysis with `python3 -B analysis/studyII_welch_20261028/analyze.py` and then `python3 -B analysis/studyII_welch_20261028/write_report.py`. Existing inputs must match their recorded source hashes. These commands replace only generated outputs in this analysis directory.
"""
)
(O / "REPORT.md").write_text(
    text.replace("` spectral_comparisons.csv`", "`spectral_comparisons.csv`")
)
# Human-readable compact companion, same computed quantities.
(O / "summary_tables.md").write_text(
    "# Study II spectral summary\n\n"
    + table
    + "\n## Saved validation-selected performance\n\n"
    + rt
    + "\nUnits and interpretation: see REPORT.md.\n"
)
