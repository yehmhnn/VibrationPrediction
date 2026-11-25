# Repository guide

Start with the [verified results](experiments/RESULTS.md), then use the [experiment map](experiments/EXPERIMENT_MAP.md) to trace a result to its runner, configuration and saved evidence. The [reproduction guide](REPRODUCIBILITY.md) describes environments, authorized inputs and execution limits.

| Location | Role |
|---|---|
| `scripts/` | Dataset preparation, record-local windows, TimesNet and TSFlow integration, training, metrics and tests. `preserved/` holds the historical E1–E6 execution record. |
| `configs/study_i/` | Configurations for chronological within-record experiments. |
| `configs/study_ii/` | Configurations for recording-disjoint Protocol B. |
| `configs/regularization/` | E1–E6 plan, seed-specific configurations and historical checksums. |
| `evidence/datasets/` | Dataset manifests, splits and normalization metadata. |
| `evidence/experiments/` | Saved histories and results for Study I, Study II and follow-ups. |
| `evidence/diagnostics/`, `historical/`, `overnight/` | Numerical verification, exploratory comparisons and campaign records. |
| `analysis/` | Welch PSD, offline wavelet analysis and dataset/spectral reports. Use `reproduce.py` to run in a fresh directory. |
| `analysis/evidence/` | Unique historical inputs for the analyses, including source snapshots that differ from the maintained scripts. |
| `third_party/tsflow/` | Upstream pin, patches and dependency reconstruction instructions. |
| `docs/provenance/` | Historical validation and the current [file inventory](provenance/STAGING_MANIFEST.json). |

The offline analyses read common dataset metadata from `evidence/datasets/`, completed Protocol B results from `evidence/experiments/study_ii/`, and saved configurations from `configs/`. The inventory records mappings from historical duplicate paths to their retained files. Saved reports and historical hashes retain their original reference context. The full Welch PSD table is retained as input for spectral redraw. Different formats, such as PDF and PNG, serve distinct uses even when they depict the same figure.

The [publication review](PUBLICATION_REVIEW.md) records GitLab access and redistribution requirements. Repository preparation and publication approval are separate steps.
