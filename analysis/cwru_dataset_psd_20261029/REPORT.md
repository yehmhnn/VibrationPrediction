> Submission copy: privacy-redacted historical report. Original source hash is retained in staging provenance; some detailed outputs remain private. Reproduce only in a fresh workspace using analysis/reproduce.py.

# Verified CWRU dataset statistics and spectral assets

2026-10-29. No training, inference, raw recording access or reserved-signal access was performed. The figures redraw existing verified PSD values. All statistics distinguish raw samples from windows. Reserved-record statistics are verified against manifest/code metadata, not newly measured from reserved arrays.

## Study I: 105_0 chronological split

Original N=121265 samples, duration N/fs=10.105416667 s at12000Hz (sample support convention N/fs, not (N−1)/fs). Raw count is recorded in the StudyII manifest and was previously checked in the allowed 105_0 MAT analysis. Retained samples=118865; discarded gap samples=2400 (1.979136602% of original). The retained-data fractions are exactly60/20/20. Relative to the entire original recording, use the distinct percentages below.

| Partition | Indices [start,end) | Raw samples | Duration (s) | % original | % retained | Stride | Windows |
|---|---|---|---|---|---|---|---|
| train | [0,71319) | 71319 | 5.943250000 | 58.812518039 | 60 | 32 | 2210 |
| validation | [72519,96292) | 23773 | 1.981083333 | 19.604172680 | 20 | 96 | 242 |
| test | [97492,121265) | 23773 | 1.981083333 | 19.604172680 | 20 | 96 | 242 |


Two gaps: [71319,72519) and [96292,97492), each1200 samples/0.1s. Not random sample splitting. For chronological preprocessing, available=N−2gap; train=int(0.6available); validation=(available−train)//2; test=remainder.

Primary L512/H96 spans42.666667ms context and8ms forecast. Window count=max(0,floor((n−L−H)/stride)+1). Window starts are partition-local0,stride,...; no window crosses a partition or recording boundary. Contexts overlap; nonoverlapping evaluation target blocks do not make errors statistically independent.

The242 common test origins are **all** valid H96 stride96 starts: context start=97492+96j and forecast start=98004+96j, j=0,...,241. Last context start120628; last forecast start121140; final target interval[121140,121236), leaving29 trailing samples. No subsampling was used to create242; the same grid applies to both persisted model outputs. Waveform example61 is a separately chosen visualization, not the grid definition. Full H96 metric uses242×96=23232 target elements.

Historical independently trained horizons use evaluation strideH and have different window counts; the242 claim is specifically H96/fixed-H96 prefixes. From the same count formula:

| H | Training stride32 windows | Validation strideH windows | Test strideH windows |
|---|---|---|---|
| 24 | 2212 | 969 | 969 |
| 48 | 2212 | 484 | 484 |
| 72 | 2211 | 323 | 323 |
| 96 | 2210 | 242 | 242 |


Normalization: (x−mean)/std, fitted only to raw chronological training samples[0,71319), float64 mean/population SD(ddof0), stored float32. Mean=0.014212163956835134, SD=0.2907071138920723. It is not the global StudyII normalization. TSFlow adapter uses identity scaling on normalized arrays; TimesNet additionally applies past-context internal normalization/de-normalization as part of its architecture.

Evidence:
- `VibrationPrediction_Report_Revised/source_artifacts/datasets/processed/cwru_105_0_chronological/metadata.json` and `manifest.csv`: exact intervals, gap, normalizer, partition counts.
- `VibrationPrediction_Report_Revised/source_artifacts/scripts/preprocess_cwru_within_record.py`: chronological_boundaries and mean/std fit.
- `VibrationPrediction_Report_Revised/source_artifacts/scripts/sequence_data.py`: ManifestSequenceDataset.window_index, range(0,last_start+1,stride), context/target slicing; `tsflow_data.py` reuses this grid.
- `VibrationPrediction_Report_TwoStudies_20261016/evidence/logs/convergence/single_105_0_{tsflow,timesnet}_objective_v2/config.json`: L512,H96,strides32/96,num_origins2210/242/242 (also verified in the previous audit).
- `VibrationPrediction_Report_Revised/data/reconstructed/inference_provenance.json`:242 origins, stride96, exact_target_match=true, historical TSFlow100/TimesNet30 checkpoint family. No new historical test evaluation was made.

## Study II: recording-disjoint assignment

| MAT filename / ID | Fault | hp | RPM | Hz | Raw samples | Duration (s) | Windows | Role |
|---|---|---|---|---|---|---|---|---|
| 105_0.mat | inner race | 0 | 1797 | 12000 | 121265 | 10.105417 | 3771 | train |
| 106_1.mat | inner race | 1 | 1772 | 12000 | 121991 | 10.165917 | 3794 | train |
| 107_2.mat | inner race | 2 | 1748 | 12000 | 122136 | 10.178000 | 1266 | validation |
| 108_3.mat | inner race | 3 | 1721 | 12000 | 122917 | 10.243083 | 1275 | reserved |
| 118_0.mat | ball | 0 | 1796 | 12000 | 122571 | 10.214250 | 3812 | train |
| 119_1.mat | ball | 1 | 1772 | 12000 | 121410 | 10.117500 | 3776 | train |
| 120_2.mat | ball | 2 | 1748 | 12000 | 121556 | 10.129667 | 1260 | validation |
| 121_3.mat | ball | 3 | 1722 | 12000 | 121556 | 10.129667 | 1260 | reserved |
| 130@6_0.mat | outer race | 0 | 1796 | 12000 | 121991 | 10.165917 | 3794 | train |
| 131@6_1.mat | outer race | 1 | 1773 | 12000 | 122426 | 10.202167 | 3807 | train |
| 132@6_2.mat | outer race | 2 | 1750 | 12000 | 121410 | 10.117500 | 1259 | validation |
| 133@6_3.mat | outer race | 3 | 1725 | 12000 | 122571 | 10.214250 | 1271 | reserved |


All listed IDs correspond to same-named MAT files. All have0.007-inch defects; outer-race position is@6. Window counts are computed from each manifest sample count using L512/H96, stride32 for train and96 for validation/reserved. **Reserved window counts are potential loader windows, not evaluated forecasts.** The code and metadata call that split `test`; this report labels it reserved to distinguish assignment from evaluation. Both final original StudyII model result.json files explicitly record test_evaluated=false. Thus the3hp recordings were reserved and not evaluated for those final models; this does not erase any separate historical test inspection under other protocols.

| Role | Raw sample total | Usable windows |
|---|---|---|
| train | 731654 | 22754 |
| validation | 365102 | 3785 |
| reserved | 367044 | 3806 |


StudyII normalization uses six training recordings only, sample-weighted global mean0.010496160688842994 and population SD0.41000491194742045, from731654 training samples. Validation and reserved signals do not fit the normalizer. Full records are independently standardized; no cross-record windows are formed.

Evidence:
- `VibrationPrediction_Report_TwoStudies_20261016/evidence/datasets/processed/cwru_multi_record_B/manifest.csv`: filename,channel,fault,load_id,rpm,sampling_rate_hz,num_samples,source intervals,split. RPM/count derive from MAT checks; fs/load/fault derive from acquisition metadata, not a timestamp field in the MAT.
- Same-directory `metadata.json`: split_record_ids,windowing L512/H96/strides32/96; `normalization.json`: training IDs/sample count, mean/std and uses_validation_or_test=false.
- Server `./scripts/cwru_protocols.py`, read-only inspection: GROUPS exact IDs/counts/RPM and split='train' if load<2 else 'validation' if load==2 else 'test'.
- Server `./scripts/preprocess_cwru_protocol_b.py`, read-only inspection: checks MAT channel/RPM/count against assignments; standardizes full records; reports windows with (n−608)//stride+1. It was inspected, not executed.
- `VibrationPrediction_Report_TwoStudies_20261016/evidence/scripts/run_convergence_experiment.py`: L512/H96 and loader strides32/96; both `multi_record_B_*_objective_v1/config.json`: matching loader counts; corresponding `result.json`: test_evaluated=false.
- `analysis/studyII_welch_20261028/record_summary.csv` and `provenance_and_checks.json`: prior SHA-256-verified actual measurements for the nine nonreserved MAT files. Reserved counts/RPM were not remeasured in this task; they are metadata-verified.

## hp, RPM, and operating conditions

**hp** means horsepower, the motor-load condition recorded for the test (0hp does not mean zero shaft rotation). **RPM** means revolutions per minute of the motor shaft. Use recording-specific stored values, not the approximate speed associated with a load row. E.g.1hp outer race=1773RPM while1hp inner race/ball=1772RPM;2hp outer race=1750RPM versus1748RPM in the other two. These values are in the manifest and source GROUPS.

Shaft-speed changes can shift rotation-related spectral lines and repetition rates; load can affect bearing excitation and amplitudes. These are possible mechanisms, not a causal finding from this study: load,RPM and acquisition identity change together. Model transfer from0/1hp recordings to2hp is therefore a joint operating-condition/recording change. CWRU describes accelerometer acquisition and recorded speed/load in its [apparatus documentation](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) and [data-variable guide](https://engineering.case.edu/bearingdatacenter/download-data-file). Physical acceleration calibration in the local artifacts remains unknown; plots retain source numerical units rather than assuming calibrated g.

## Figures and verified observations — post-report exploration

FigureA `assets/cwru_fault_psd.pdf`: individual0hp training recordings105_0(inner race,1797RPM),118_0(ball,1796RPM),130@6_0(outer race,1796RPM). Same nominal load, almost identical but explicitly unequal RPM. This is not an average or a universal fault-class signature.

FigureB `assets/cwru_load_psd.pdf`: one panel/fault, overlaying exactly0/1hp training and2hp validation; actual stored RPM in every legend. Common log-y scale and original-unit PSD preserve absolute power differences; no per-record variance normalization. Both figures span0–6000Hz and lightly shade2–4kHz. No3hp curve is present.

Reused source: `analysis/studyII_welch_20261028/welch_spectra.csv` (psd_original_units2_per_hz), originally computed using fs12000,Hann,nperseg8192,noverlap4096,nfft8192,constant detrend,density scaling,mean averaging,one-sided PSD, no smoothing. Original equivalent `analysis/studyII_welch_20261028/spectral_overview.pdf` was inspected; its data are reused to create the specifically requested layouts, not recomputed or altered. Frequency spacing1.46484375Hz;28 Welch segments/record. Existing band/RMS tables supply the observations below.

### Dataset-slide bullets

- CWRU drive-end vibration is sampled at12kHz;512 past samples predict96 future samples (42.67ms→8ms).
- StudyI uses chronological regions of one105_0 acquisition, separated by100ms gaps.
- StudyII trains on six0/1hp recordings and validates on three2hp recordings across three fault types.
- Three3hp recordings remain reserved; raw sample counts and forecasting-window counts are different quantities.

### Spectral-slide bullets

- Across all nine analyzed recordings, most power lies at2–4kHz: inner race77.4–78.1%, ball90.3–92.8%, outer race97.2–98.0%.
- Peak positions/heights vary with acquisition; similar broad bands do not imply identical spectral shape or forecasting difficulty.
- Outer-race2hp total power is lower than both outer-race training recordings; its high forecasting RMSE is not explained by unusually elevated validation power.

Numerical sources: `analysis/studyII_welch_20261028/band_power.csv` (sum fraction_total for2000–3000 and3000–4000Hz); `record_summary.csv` (integrated Welch power), `dominant_peaks.csv` (peak frequencies). Observations are descriptive; no causal or denoising-improvement claim is made.

### Overleaf inclusion

```latex
\includegraphics[width=0.98\linewidth]{assets/cwru_fault_psd.pdf}
\includegraphics[width=0.98\linewidth]{assets/cwru_load_psd.pdf}
```

Use on separate frames and label “Post-report exploratory PSD analysis.” No Beamer slides were generated.

## Outputs and verification

Created: `assets/cwru_fault_psd.pdf`, `assets/cwru_load_psd.pdf`, correspondingPNG previews, plus this analysis directory's `prepare.py`, `REPORT.md`, `studyI_split.csv`, `studyII_recordings.csv`, and `provenance.json`. No original artifact was modified. `prepare.py` reproduces tables and figures solely from metadata and saved PSD CSV values. Source SHA-256 values and plotting versions are in provenance.json. Duration/sample fractions and window counts are derived arithmetically from the cited file fields; reserved signal arrays remain unopened.
