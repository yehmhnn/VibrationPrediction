# Reproduction guide

Study I uses one105_0 acquisition split chronologically with two1200-sample gaps and train-only normalization. Study II uses three fault types, six0–1hp training/three2hp validation/three3hp reserved recordings. L512/H96, strides32/96, no cross-record windows, and global training-only normalization are unchanged. E1 uses chronological regions within six0–1hp recordings, a1200-sample gap and a separate training-region normalizer. Its RMSE is not directly comparable to Study II.

Use [the experiment map](experiments/EXPERIMENT_MAP.md) for source/configuration/result paths and [results](experiments/RESULTS.md) for interpretation. Configurations document completed executions; they are not universally portable command files. Historical hashes attest original private files; redaction changes current byte hashes without changing numerical results/settings. See [staging provenance](provenance/README.md).

## Forecasting installation

1. Reconstruct the patched sibling TSFlow dependency using [the pinned instructions](../third_party/tsflow/README.md). Redistribution rights remain unresolved.
2. In a new workspace with that sibling, use the retained linux-64 Pixi lockfile (`pixi install --locked`). Root requirements.txt is not an exact replacement. Do not silently re-solve dependencies.
3. Check driver/PyTorch compatibility before separately authorized GPU work. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python for deterministic CUDA.

Existing-environment imports resolve TSFlow through the original editable installation. A clean forecasting installation and full GPU reproduction were **not** performed.

## Data and original executions

Raw MAT/processed NPY data are excluded. Obtain CWRU12kHz drive-end0.007inch recordings separately; evidence/datasets supplies channel names, assignments and source_sha256. Redacted source_path fields are portable/historical locations, not bundled files. Verify input hashes before use.

scripts/preprocess_cwru_within_record.py prepares Study I. scripts/preprocess_cwru_protocol_b.py and cwru_protocols.py implement Study II. Inspect their CLI and supply a fresh output directory; preprocess_cwru.py defines a different historical protocol. Protocol B preprocessing prepares all assigned records including reserved arrays; it is not a no-reserved-access validation command.

The original convergence runner can open reserved arrays even without --evaluate-test. Do not run it under strict no-reserved-access requirements. Its behavior was not redesigned here. After data/compute authorization, the historical command form is:

```sh
python scripts/run_convergence_experiment.py --protocol PROTOCOL --model MODEL --max-epochs BUDGET --dataset-dir DATASET_DIRECTORY --output-dir NEW_OUTPUT_DIRECTORY --device cuda
```

Replace placeholders from the matched saved configuration, including stopping/objective-evaluation options; defaults alone do not establish exact reproduction. Objective-matched budgets: Study I TSFlow200/TimesNet150; Study II TSFlow150/TimesNet100. Actual epochs/selection are in saved results. Do not add --evaluate-test merely to reproduce validation histories.

## E1–E6

All eight configurations, implementation and compact histories/results are preserved. The archive has redacted paths and historical sealed hashes; it is **not a portable executable batch**. Do not run it. A portable launcher needs explicit data/output roots, filtered manifests/normalizers, a newly sealed layout and reserved-data-exclusion checks. These require a separately reviewed implementation; none was added.

E2/E6 constitute dropout0.3 seeds42/7/123. E5 supplies baseline seeds7/123; seed42 baseline is reused. E3/E4 weight-decay comparisons each have one seed. Stop policies differ from baseline criteria; compare realized budgets and checkpoint rules explicitly.

## Offline environment and safe commands

```sh
python3.11 -m venv ../analysis-env
../analysis-env/bin/python -m pip install -r analysis/studyII_wavelet_denoising_20261028/requirements-analysis.txt
../analysis-env/bin/python -B analysis/reproduce.py metadata --work-dir ../metadata-reproduction
../analysis-env/bin/python -B analysis/reproduce.py welch --input-dir ALLOWED_MAT_DIRECTORY --work-dir ../welch-reproduction
../analysis-env/bin/python -B analysis/reproduce.py wavelet --input-dir ALLOWED_MAT_DIRECTORY --work-dir ../wavelet-reproduction
```

Work directories must be absent and outside the submitted repository. `--check-only` validates required input hashes/destination without executing analyses. Only nine manifest-selected train/validation MAT names are admitted. The wrapper copies preserved code/evidence into a fresh workspace; submitted reports/tables remain unchanged. Original entry points overwrite their own outputs if invoked directly, so use the wrapper.

The separate Python3.11 environment installed NumPy2.0.2, SciPy1.13.1, matplotlib3.9.4, PyWavelets1.6.0 and scikit-image0.24.0. A bounded105_0 training-record check matched saved Welch power and soft/hard denoised RMS at rtol1e-12/atol1e-14. This does not certify all-record reproduction or forecasting behavior. No reserved signal was opened. See [analysis inventory](../analysis/README.md).
