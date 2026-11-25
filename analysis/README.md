# Offline signal analyses

| Directory | Analysis | Saved evidence |
|---|---|---|
| studyII_welch_20261028 | Nine train/validation records; Hann8192/overlap4096 Welch at12kHz | Report, summaries, original PSD table, overview figures |
| studyII_wavelet_denoising_20261028 | Fixed db4,3-level BayesShrink soft/hard | Report, metrics/bands/peaks/thresholds, overview |
| cwru_dataset_psd_20261029 | Metadata tables and saved-PSD redraw | Report, split/record tables, two figures |

Reports are privacy-redacted derivatives; original bytes remain private with recorded hashes. Three pre-adaptation scripts under original_sources/ preserve history and are not recommended entry points. Historical output_hashes/provenance refer to original outputs, including intentionally omitted files; they are not current staged-file inventories.

## Safe reproduction

Follow [the guide](../docs/REPRODUCIBILITY.md) and use reproduce.py with a fresh external work directory. The full original-unit Welch CSV is retained because the metadata redraw requires4097 frequency bins for each of nine records. The larger denoising spectrum table and detailed waveform PDFs are omitted; compact metrics/overview remain. They can be regenerated separately in a fresh workspace.

No MAT inputs, signal NPZs or server retrieval scripts are distributed. Obtain exactly105_0,106_1,107_2,118_0,119_1,120_2,130@6_0,131@6_1,132@6_2 with .mat extension and verify manifest hashes. Reserved3hp records are not required or admitted by the wrapper. Metadata describes their assignment without opening signals.

`validate_environment.py --input ALLOWED_105_0_MAT --output NEW_JSON` checks one authorized training record without models, rejects existing output and verifies input hash before loading. The separate environment/numerical check passed. Metadata redraw was checked in a fresh workspace; full nine-record wavelet/Welch regeneration was not performed during preparation.

## Interpretation

Spectra do not isolate load/RPM/recording causes. Thresholding removed substantial structured vibration, especially for ball faults; reconstructions are not clean ground truth. Full-record DWT/sigma estimation uses future samples and is not leakage-safe forecasting preprocessing. No model was trained on these reconstructions and no forecasting improvement is claimed.
