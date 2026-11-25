# Controlled results and follow-up diagnostics

Computed from saved result.json and history.csv only. No new forecasting scores were generated. RMSE is pooled over normalized target elements; arithmetic mean of per-record RMSE is a different quantity. Selected validation RMSE is a checkpoint-selection statistic, not an unbiased estimate of held-out superiority. Raw epoch minima can differ from the recorded selection when min_delta is used; both are preserved in PHASE2B_RESULTS.json.

| Run | Selected epoch | Selected validation RMSE | Final validation RMSE | Epochs |
|---|---:|---:|---:|---:|
| multi_record_B_timesnet_objective_v1 | 1 | 1.024 | 1.086 | 100 |
| multi_record_B_tsflow_objective_v1 | 8 | 0.952 | 1.063 | 150 |
| E1 | 70 | 0.876 | 0.914 | 100 |
| E2 | 1 | 1.011 | 1.050 | 100 |
| E3 | 71 | 1.059 | 1.118 | 100 |
| E4 | 97 | 0.916 | 0.929 | 150 |
| E5_seed7 | 54 | 1.038 | 1.090 | 100 |
| E5_seed123 | 90 | 1.006 | 1.070 | 100 |
| E6_seed7 | 9 | 1.048 | 1.070 | 100 |
| E6_seed123 | 88 | 0.965 | 1.005 | 100 |

## Matched TimesNet dropout comparison

Paired difference is dropout 0.3 minus baseline 0.1; negative favors dropout on this fixed validation split. Seed42 baseline is reused from Study II.

| Seed | Baseline | Dropout0.3 | Paired difference |
|---|---:|---:|---:|
| 42 | 1.024 | 1.011 | -0.013 |
| 7 | 1.038 | 1.048 | +0.010 |
| 123 | 1.006 | 0.965 | -0.041 |

Across three seeds, selected RMSE mean ± sample SD (ddof1): baseline **1.023 ± 0.016**, dropout **1.008 ± 0.042**; paired difference **-0.015 ± 0.026**. This is descriptive seed variability, not statistical significance or variability across independent acquisitions.

## Interpretation and controls

E1: same-condition chronological regions within the six 0–1hp records, 1200-sample exclusion gap and separately fitted training-region normalization; 15894 training /2177 validation windows. This tests a different generalization question from Study II and its RMSE is not directly comparable with Study II because targets and normalization differ. Online training MSE averages changing weights with dropout enabled; evaluation-mode validation MSE uses fixed post-epoch weights without dropout. Their gap is informative but is not a controlled evaluation-mode train-vs-validation estimate.

E2: TimesNet dropout 0.3 seed 42, otherwise matched Study II settings. E3: TimesNet weight_decay 0.1 seed 42, baseline dropout0.1. E4: TSFlow weight_decay 0.1 seed 42. E5: baseline TimesNet seeds 7/123. E6: predetermined dropout 0.3 seeds 7/123, not selected adaptively from the overnight results. Baseline weight decay is0.01. Full fixed plans/configurations are under configs/regularization/overnight_20261020_v1/; original baseline configurations remain in configs/study_ii/.

TimesNet stronger weight decay worsened selected validation RMSE relative to its seed 42 baseline; TSFlow stronger weight decay improved it in one seed. Dropout helped selected RMSE in two of three seeds, with no uniform improvement. Neither finding establishes held-out generalization or solves overfitting. Checkpoint timing is seed-dependent (see table); optimal checkpoints do not consistently occur early. Historical baseline reuse and different stopping policies are caveats; realized budgets should be compared, not silently equalized.

TSFlow trains a Flow Matching velocity objective on the context-plus-future path; TimesNet trains forecasting MSE. Their objective magnitudes cannot be compared. Forecast RMSE is the common validation endpoint. Single-seed comparisons have no run-level SD; epoch-level SD and descriptive differences across three dependent fault recordings are not run-level uncertainty.

Study II's falling training objectives without sustained validation improvement are consistent with fitting the training distribution and a generalization gap; overfitting and distribution shift are not uniquely separable here. Inner-race diagnostics compare seen0/1hp with unseen2hp using models trained on all three faults. Numerical checks found no confirmed bug in the tested calculations, not proof of correctness for every path. Welch spectra and offline wavelets are descriptive follow-up analyses, not demonstrated fixes or causes. Load, RPM and recording identity are confounded. Overlapping windows and one validation acquisition per fault limit independence. No new reserved3hp evaluation is included.

## Exact evidence

See PHASE2B_RESULTS.json for each original result/history path and the corresponding byte-identical evidence/experiments copies. Diagnostic originals and copies are mapped in the Phase2B manifest. The Phase2A EXPERIMENT_MAP identifies Study I's distinct historical horizons and later objective-matched convergence runs. Historical test scores must not be reassigned to later checkpoints.
