# CWRU vibration forecasting — Master's Practical

[Repository guide](docs/REPOSITORY_GUIDE.md)

This project compares **TSFlow conditional Flow Matching** and a **TimesNet direct forecasting implementation** for high-frequency bearing vibration. It investigates forecast horizon, optimization behavior and transfer across recordings/operating conditions. This is a submission candidate, not a claim of publication clearance.

## Experiments and verified findings

- **Study I:** chronological regions of recording105_0, with exclusion gaps and training-only normalization. Historical horizon/budget experiments and later objective-matched convergence runs remain separate.
- **Study II:** CWRU12kHz drive-end vibration, three fault types; six0–1hp training recordings, three2hp validation recordings, three3hp records reserved. Context512, horizon96; train-only global normalization; training/validation strides32/96.
- **E1–E6:** same-condition chronological diagnostic, TimesNet dropout, weight decay for each model, and matched TimesNet dropout/baseline seeds42,7,123. Eight fresh follow-up runs reused original seed42 baselines.

Original Study II selected validation RMSE was0.952 for TSFlow and1.024 for TimesNet; final RMSE was1.063/1.086. Neither original run met the practical stabilization criterion. Stronger TimesNet weight decay worsened selected validation RMSE; TSFlow weight decay improved it in one seed. Dropout did not improve all three seeds. These are validation/checkpoint-selection findings, not held-out model superiority or proof that overfitting was solved. [Results and paired-seed statistics](docs/experiments/RESULTS.md) retain the full scientific qualifications.

TSFlow Flow Matching objective and TimesNet forecasting MSE have incomparable scales. Forecast RMSE is the common endpoint. Load, RPM and recording identity are confounded. Offline wavelet denoising removed substantial structured vibration, especially for ball faults, and was not adopted as a forecasting fix.

## Where to start

| Path | Purpose |
|---|---|
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Environments, protocols, entry points and execution limits |
| [Experiment map](docs/experiments/EXPERIMENT_MAP.md) |23 completed runs mapped to source/configuration/evidence |
| scripts/ | Original forecasting models, adapters, preprocessing and evaluation |
| scripts/preserved/ | Privacy-redacted historical overnight implementation; do not launch |
| configs/ | Saved Study I/II and E1–E6 configurations |
| evidence/ | Compact saved histories/results, manifests and historical diagnostics |
| [Offline analyses](analysis/README.md) | Welch, wavelet and dataset/spectral code, reports and selected figures |
| [TSFlow dependency](third_party/tsflow/README.md) | Pinned upstream commit and exact patches |
| [Research history](docs/experiments/HISTORY.md) | Preliminary and exploratory results distinguished from final studies |
| [Publication review](docs/PUBLICATION_REVIEW.md) | Licensing/privacy gates and GitLab/GitHub recommendations |

## Installation and reproduction

Forecasting uses the retained linux-64 Pixi lockfile and a patched sibling TSFlow checkout at ff89317; requirements.txt alone is insufficient. Follow the dependency reconstruction guide before `pixi install --locked` in a separate workspace. Full clean-machine/GPU reproduction was not tested. Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts.

Offline analyses use a separate environment. From repository root:

```sh
python3.11 -m venv ../analysis-env
../analysis-env/bin/python -m pip install -r analysis/studyII_wavelet_denoising_20261028/requirements-analysis.txt
../analysis-env/bin/python -B analysis/reproduce.py metadata --work-dir ../metadata-reproduction
```

The wrapper requires a fresh destination outside the repository. Signal analyses additionally need the nine train/validation MAT files, verified against their manifest hashes. No datasets, checkpoints, optimizer states, virtual environments or large private logs are bundled. The retained Welch spectrum table is needed to reproduce the dataset spectral figures; larger denoising intermediates and redundant figures remain private.

## Limitations and readiness

One fixed validation split, dependent windows, three seeds for TimesNet dropout only, single-seed weight-decay comparisons and historical baseline reuse limit interpretation. No new reserved3hp forecasts are included. Some older exploratory protocols had different test usage and must not be confused with final Study II.

Privacy edits affect staged copies only; original research artifacts and snapshots are unchanged. Historical hash manifests attest original bytes; current staging hashes and every transformation are recorded separately. No Git repository was initialized and no remote was accessed or changed. **Publication is blocked pending licensing/ownership and repository-access review.** University GitLab is the first intended destination; its visibility is not assumed private.
