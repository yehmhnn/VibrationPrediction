# Vibration Prediction Implementation Plan

## Research Goal

Evaluate deterministic and probabilistic forecasting of raw bearing-vibration waveforms under:

1. an approximately cyclostationary, constant-speed regime (CWRU); and
2. a nonstationary, time-varying-speed regime (Ottawa).

The bearing data are proxy signals for studying future feedforward laser stabilization. This project does not claim closed-loop laser stabilization without actuator, latency, transfer-function, and control experiments.

## Scientific Rules

- Preserve every physical recording and its metadata as a separate record.
- Never construct a window across recording or data-split boundaries.
- Split recordings or time regions before constructing windows.
- Estimate normalization parameters from training data only.
- Never combine signals with different sampling rates without explicit anti-alias filtering and resampling.
- Report lookback and horizon in samples and milliseconds.
- Compare learned models with simple forecasting baselines.
- Use validation data for model selection and test data only for final evaluation.
- Record random seeds, package versions, hardware, and runtime.

## Workspace Layout

```text
VibrationPrediction/
├── datasets/
│   ├── cwru/                         # Raw CWRU .mat files
│   ├── ottawa/                       # Raw Ottawa .mat files
│   └── processed/                    # Record-aware processed datasets
├── scripts/
│   ├── preprocess_cwru.py
│   ├── preprocess_ottawa.py
│   ├── sequence_data.py
│   ├── timesnet.py
│   ├── train_timesnet_smoke.py
│   ├── run_experiments.py
│   ├── profile_gp.py
│   ├── generate_figures.py
│   └── generate_report.py
├── figures/                           # Generated scientific figures
├── logs/                              # Metrics and runtime records
├── reports/                           # Generated reports
├── pyproject.toml                     # Pixi environments and dependencies
└── pixi.lock                          # Reproducible dependency lock (to create)
```

Each processed dataset contains one `.npy` file per physical record, a `manifest.csv` with record-level metadata and split assignments, and a `metadata.json` with dataset-wide preprocessing information.

## Environment and GPU Strategy

`TSFlow/.pixi` belongs to the TSFlow project and must not be moved. Moving it would break path assumptions and would couple the clean implementation to the old repository. `VibrationPrediction` must maintain its own Pixi environment and lock file.

Planned environments:

- `default`: NVIDIA GPU-enabled PyTorch environment;
- `cpu`: CPU fallback for preprocessing, tests, and systems without a working NVIDIA driver.

Before a GPU experiment, verify both:

```bash
nvidia-smi
pixi run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA packages alone do not prove GPU access. The NVIDIA driver must be available and compatible. At the latest check, `nvidia-smi` could not communicate with the driver, so GPU execution is currently unverified.

## TSFlow Cleanup and Integration Status

TSFlow remains a sibling model-library repository; it will be imported by this experiment project rather than copied into it.

### Protected work — completed

The six modified tracked TSFlow files are preserved at Git reference:

```text
refs/backup/pre-cleanup-2026-08-13
2d37940fbd6e3b00cdeba86677f3574a1d722765
```

### Retain and migrate — reviewed

- `custom_dataloader.py` and `eval_cwru*.py` contain useful TSFlow calling and sampling logic, but depend on the scientifically invalid concatenated CWRU array. Reimplement this functionality as a manifest-based adapter in `VibrationPrediction`.
- `configs_local/cwru.yaml` and `cwru_smoke.yaml` are historical starting configurations. Preserve them for provenance, then replace fake frequency and unjustified window settings.
- Six checkpoint-bearing runs under `TSFlow/logs/` are historical results. Preserve them but do not report their metrics as final results.
- `.aim/` contains experiment tracking state and should remain until the historical runs are exported.

### Generated cleanup — partially completed

Project-source `__pycache__/` directories were removed. Caches inside `.pixi/` are part of the managed environment and were intentionally left untouched.

### Pending cleanup

- Export an inventory of historical runs and identify the checkpoints worth retaining.
- Archive or remove incomplete empty log runs.
- Migrate useful download and dataset-selection utilities into this project.
- Centralize raw datasets only after replacing hard-coded paths.
- Decide whether VBL and SUBF must be kept to reproduce dataset screening; together they occupy about 30 GB.

### Root-script organization — completed

Loose TSFlow scripts were moved without data loss into documented categories under `TSFlow/archive/`:

- `cwru_legacy_2026/`: scientifically obsolete concatenated-data preprocessing and evaluation;
- `dataset_screening_2026/`: early VBL/SUBF/CWRU/Ottawa screening utilities;
- `mendeley_exploration_2026/`: Ottawa API/download discovery and inspection.

The historical CWRU bypass and compatibility loader have now been removed from active TSFlow code and archived. TSFlow-specific experiment data logic lives in this project.

### Manifest-based TSFlow integration — completed

- `VibrationPrediction` declares sibling `../TSFlow` as an editable local dependency.
- `scripts/tsflow_data.py` adapts the same record-aware windows used by TimesNet to TSFlow's batch contract.
- `scripts/train_flow_smoke.py` verifies GP feature extraction, Flow Matching loss, and backward gradients.
- The CPU smoke test passes on records `105_0` and `106_1` with `L=96`, `H=48`.
- TSFlow's S4 PyKeOps operation now selects CPU or GPU according to the tensor device instead of forcing GPU.
- The unavailable upstream multivariate backbone now produces a clear `NotImplementedError`; supported univariate behavior remains explicit.

GPU execution remains unverified because the NVIDIA driver is not currently accessible.

## Phase 1 — Dataset Audit and Representation

### 1.1 CWRU inventory — completed

- Identified 161 raw files across 12 kHz and 48 kHz acquisition families.
- Selected `12k_Drive_End_Bearing_Fault_Data/IR/007` for the first homogeneous baseline.
- Selected `DE_time` as the prediction channel.
- Preserved load and RPM metadata.

### 1.2 Record-aware CWRU preprocessing — completed

Output: `datasets/processed/cwru_12k_de_ir007/`

- Train: records `105_0`, `106_1`.
- Validation: record `107_2`.
- Test: record `108_3`.
- Sampling rate: 12 kHz.
- Normalization: training-record mean and standard deviation only.
- Interpretation: test performance measures cross-load/RPM generalization.

### 1.3 Record-aware window loader — completed

`scripts/sequence_data.py` reads manifests and creates windows independently inside each recording and split. `run_experiments.py` and `train_timesnet_smoke.py` use separate train, validation, and test loaders.

### 1.4 Spectral inspection — completed for current data

`scripts/generate_figures.py` computes record-wise Welch spectra using the true sampling rates:

- CWRU: 12 kHz;
- Ottawa raw data: 200 kHz.

The generated plot is exploratory and does not constitute a forecasting result.

### 1.5 Raw-waveform predictability diagnostic — completed for current CWRU split

`scripts/analyze_predictability.py` analyzes each physical record separately and saves
`logs/predictability_analysis.csv` and `figures/predictability_analysis.png`.

- Welch spectra contain strong high-frequency resonances, mostly with periods of roughly
  3--9 samples, but spectral power alone does not establish phase predictability.
- The strongest record-level autocorrelation peaks occur mainly at much longer lags
  (roughly 371--1,677 samples) and change with load/RPM.
- The initial 192-sample lookback cannot observe most of those long repeat intervals.
- Short-lag correlations exist, but their best lag is not stable across records; this
  helps explain why a training-estimated fixed seasonal period and TimesNet both tend
  toward the near-zero/mean solution on the held-out load.

Next, test longer physically motivated lookbacks and a leakage-safe linear
autoregressive baseline before deciding whether to replace raw samples with envelope,
RMS, or frequency-band targets.

### 1.6 Ottawa preprocessing — next

- Preserve each `.mat` recording separately.
- Use `Channel_1` as the vibration target.
- Preserve `Channel_2` encoder data for speed extraction and future conditioning.
- Resample each recording independently from 200 kHz to an initially proposed 40 kHz.
- Validate anti-alias filtering and quantify retained spectral energy.
- Initially create leakage-safe chronological regions within `H-A-1`.
- Preferably download trials `H-A-2` and `H-A-3` so entire trials can form train, validation, and test splits.
- Generate `manifest.csv` and `metadata.json` in a new processed directory.

## Phase 2 — Baselines and Evaluation Protocol

### 2.1 Simple baselines — completed for the first CWRU split

Implement and evaluate:

- zero/mean prediction;
- last-value persistence;
- seasonal persistence based on an estimated dominant period;
- autoregressive forecasting;
- optionally a small MLP or 1D CNN.

These establish whether TimesNet adds value beyond simple periodic prediction.

`scripts/run_baselines.py` evaluates the held-out record with target stride equal to the horizon and saves `logs/baseline_results.csv`. With $L=192$ and $H\in\{48,96,192,336\}$, zero and lookback-mean forecasts currently outperform last-value and training-estimated seasonal persistence. These results are documented in `20260618.tex` and are limited to record `108_3`.

A ridge-regularized linear autoregressive baseline is now fitted using training records
only, without crossing record boundaries. `scripts/tune_autoregression.py` compares 18
order/regularization configurations on the validation record at $L=512,H=96$ and
selects AR(32), ridge 100 by validation RMSE. On the untouched test record it improves
over the mean forecast at every evaluated horizon; at $H=96$, MAE decreases from
0.767208 to 0.719477 and correlation increases from -0.011764 to 0.301371. The search
and final results are stored in `logs/ar_validation_search.csv` and
`logs/baseline_results_L512_ar_selected.csv`. Larger AR orders overfit and do not
benefit from the longer nominal lookback.

The script also evaluates every lead time $h=1,\ldots,336$ on one common set of maximum-horizon forecast origins. It saves `logs/baseline_lead_time.csv` and `figures/baseline_lead_time.png`; the resulting curves and their non-monotonic interpretation are included in the report.

### 2.2 Metrics — pending

At minimum record:

- MAE;
- RMSE;
- normalized error with an explicit definition;
- correlation or coefficient of determination;
- frequency-domain or coherence error;
- phase error near dominant frequencies;
- inference latency, training time, parameter count, and peak memory.

Probabilistic models additionally require CRPS and empirical interval coverage. Report metrics per record and aggregated across records.

### 2.3 Experimental reproducibility — partially completed

- A default random seed is present.
- Add multiple seeds and report mean and standard deviation.
- Save configurations and hardware/package information with every run.
- Add tests for split isolation, window boundaries, normalization, shapes, and sampling-rate metadata.

## Phase 3 — TimesNet Baseline

### 3.1 Smoke-test infrastructure — completed

The smoke script uses separate training and validation subsets. A one-epoch CPU run
completed successfully with finite gradients and the expected output shape.

### 3.2 Reference-aligned TimesNet implementation — completed

The compact univariate implementation was corrected against the official TimesNet
paper/code, including:

- FFT-bin-to-period conversion;
- batch handling of selected periods;
- input-dependent FFT-amplitude aggregation;
- temporal embedding and projection;
- forecasting head so temporal phase is not erased by global average pooling.

Output shapes and backward gradients were checked for all planned horizons.

### 3.3 CWRU experiments — in progress

- Preliminary `L=192`, `H=96`, seed 42 experiments completed for one and five epochs.
- Both selected the first-epoch checkpoint (test MAE 0.770493, RMSE 1.076533),
  slightly worse than the mean baseline; predictions were nearly flat and uncorrelated
  with the target. This diagnostic is not yet a final multi-seed result.
- A matching longer-lookback run (`L=512`, `H=96`, five epochs, seed 42) also selected
  its first epoch and obtained test MAE 0.826039, RMSE 1.122022, and correlation
  0.039152. This is worse than validation-selected AR(32) (MAE 0.719477, RMSE
  1.023612, correlation 0.301371). The TimesNet forecast retained some variation
  (prediction standard deviation 0.3716 versus target 1.0733) but did not track phase.
- The longer TimesNet run used training stride 256 for CPU feasibility: about 950
  windows and 30 optimizer steps per epoch, compared with 30,400 one-step AR fitting
  examples. Treat the comparison as preliminary until denser neural-model training
  and multiple seeds can be run, preferably with working GPU access.
- Host CUDA access was verified outside the restricted execution sandbox: PyTorch
  2.5.1 (CUDA 12.6 build) uses the NVIDIA RTX 3090 successfully. The earlier NVML
  failures were caused by sandbox device isolation, not a broken host driver.
- A denser GPU run with training stride 32 (`L=512`, `H=96`, five epochs, seed 42)
  used about 7,600 windows and 237 optimizer steps per epoch. It again selected epoch
  one and worsened to test MAE 0.855246, RMSE 1.167840, and correlation 0.063970.
  Prediction standard deviation increased to 0.5340, but remained below the target's
  1.0733 and did not track phase. Training took 112.595 seconds on the shared GPU.
  Denser training therefore did not resolve cross-load generalization for this seed.
- The same dense GPU protocol was repeated with seeds 7 and 123. Across seeds
  42, 7, and 123, test MAE is 0.868097 +/- 0.011765 and RMSE is
  1.181220 +/- 0.012102 (sample standard deviation). All three runs are worse than
  validation-selected AR(32) and the lookback-mean forecast. Seeds 7 and 123 selected
  epoch five, while seed 42 selected epoch one, so the conclusion is not solely an
  early-stopping artifact. These controlled results are now documented in
  `20260618.tex`; broader horizons, targets, and datasets remain pending.
- Then sweep justified physical durations rather than arbitrary sample counts.
- Train on training records, select using validation, evaluate test once.
- Compare with every simple baseline.

### 3.4 Within-record predictability control — completed at $L=512,H=96$

- `scripts/preprocess_cwru_within_record.py` creates chronological train,
  validation, and test regions from record `105_0`, separated by two 1200-sample
  (100 ms) gaps and normalized using the training region only.
- The exact source intervals are train `[0,71319)`, validation `[72519,96292)`,
  and test `[97492,121265)`; all retain load 0 and 1797 RPM.
- Validation selection over the same 18 AR candidates chooses AR(512), ridge 100.
  It obtains test MAE 0.589386, RMSE 0.819400, and correlation 0.598331.
- Three TimesNet seeds obtain test MAE 0.576763 +/- 0.004529 and RMSE
  0.796375 +/- 0.003839. Seed-42 correlation is 0.626528. All three seeds beat
  AR and the mean baseline under the fixed operating condition.
- Comparing this result with the cross-load/RPM experiment shows that local raw
  waveform prediction is feasible, while transfer across operating conditions is
  the current bottleneck. Next evaluate shorter physical horizons, then investigate
  explicit RPM/rotational-phase conditioning.
- The within-record horizon sweep is completed for 1, 2, 4, and 8 ms using separately
  trained TimesNet models and validation-selected AR settings. AR(512), ridge 100 is
  selected at every horizon and has MAE 0.4146, 0.4834, 0.5394, and 0.5894. Three-seed
  TimesNet mean MAE is 0.4487, 0.5167, 0.5559, and 0.5768. AR is strongest at 1--4 ms;
  TimesNet is strongest at 8 ms. Both outperform the mean baseline at every horizon.
  Results and the accuracy--horizon figure are stored in
  `logs/horizon_sweep_summary_within_105_0.csv` and
  `figures/horizon_sweep_comparison_within_105_0.png` and documented in the report.

### 3.5 Cross-record RPM conditioning — completed at $L=512,H=96$

- `ManifestRPMSequenceDataset` reads RPM from each manifest record and normalizes it
  using the training records only (mean 1784.5, standard deviation 12.5 RPM).
- TimesNet embeds the scalar RPM through a small MLP and adds it to the hidden temporal
  representation; the raw-waveform target and all other training settings are unchanged.
- Across seeds 42, 7, and 123, conditioning improves MAE from
  0.868097 +/- 0.011765 to 0.805225 +/- 0.024627 and RMSE from
  1.181220 +/- 0.012102 to 1.116809 +/- 0.024747.
- The improvement is consistent with RPM carrying useful information, but the model
  remains worse than the mean baseline and AR(32). Validation/test require substantial
  extrapolation beyond two training RPM values, and RPM is confounded with load.
- Next obtain more operating points or implement rotational-angle/phase alignment;
  do not infer that scalar RPM conditioning is generally ineffective from this split.

### 3.4 Ottawa experiments — pending

Run only after Ottawa preprocessing and split validation. Evaluate both vibration-only and, later, vibration-plus-speed conditioning.

## Phase 4 — Conditional Flow Matching

### 4.1 Model implementation — pending

- Define conditioning on lookback observations and optional speed.
- Define target forecast paths and vector-field network.
- Implement training objective and ODE inference.
- Evaluate multiple samples per input.

### 4.2 GP prior study — pending

- Justify kernel choice and hyperparameters.
- Distinguish covariance construction, factorization, cached sampling, and per-batch costs.
- Measure quadratic memory and dense factorization scaling.
- Compare isotropic Gaussian and GP priors.
- Consider structured stationary-kernel/FFT alternatives.

`profile_gp.py` currently provides only an isolated dense covariance/Cholesky prototype; it is not evidence of Flow Matching training complexity yet.

## Phase 5 — Figures, Results, and Report

### 5.1 Figures — partially completed

Add:

- representative raw waveform plots;
- Ottawa spectrograms demonstrating changing frequency content;
- spectra separated by recording/condition;
- true-versus-predicted forecasts;
- error versus physical horizon;
- runtime and memory scaling;
- probabilistic intervals and calibration plots.

### 5.2 Report generation — pending

- Align CSV schemas between experiment and report scripts.
- Populate tables only from saved experiment outputs.
- Include dataset provenance, split policy, preprocessing, hardware, seeds, and uncertainty.
- Revise `20260618.tex` after methods and results are verified.

## Cleanup Policy

Safe generated files such as `__pycache__/` and `.pyc` may be removed at any time. The old concatenated arrays:

- `datasets/cwru/cwru_processed_signal.npy`;
- `datasets/ottawa/ottawa_processed_signal.npy`

are obsolete for scientific experiments but should be removed only after all active scripts and documentation no longer reference them. Raw datasets should remain until the planned cross-condition studies are complete.

## Immediate Next Actions

1. Complete and lock the local Pixi CPU/GPU environments; verify actual GPU access.
2. Implement record-aware Ottawa preprocessing at 40 kHz with encoder preservation.
3. Add simple forecasting baselines and split-integrity tests.
4. Correct and validate TimesNet before running full experiments.
