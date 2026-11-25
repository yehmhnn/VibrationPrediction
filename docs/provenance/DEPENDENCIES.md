# Dependency provenance

Forecasting is pinned by root pyproject.toml/pixi.lock and the patched sibling TSFlow dependency described in third_party/tsflow/README.md. requirements.txt is incomplete for exact environment reproduction. No forecasting dependency was installed or re-solved in Phase2C. Safe-module imports use the existing research environment and editable dependency, not a clean clone.

Offline analysis installed separately under Python3.11 with NumPy2.0.2, SciPy1.13.1, matplotlib3.9.4, PyWavelets1.6.0 and scikit-image0.24.0. requirements-analysis.txt specifies the direct pins; the final freeze is included as PHASE2C_ANALYSIS_FREEZE.txt. No environment binary is distributed. Historical environment_freeze.txt records the earlier Mac execution, not this new Linux environment.

The bounded105_0 Welch/wavelet check reproduced saved numerical values; metadata redraw used only saved PSD/metadata and matched saved dataset tables. No model forward/backward pass, GPU training or reserved-test inference was performed. Full clean-machine forecasting and all-record analysis reproduction remain unverified.

THIRD_PARTY_METADATA.json records historical installed distribution metadata, not a complete transitive SBOM or a redistribution grant. License/ownership and target-repository access gates remain explicit in docs/PUBLICATION_REVIEW.md.
