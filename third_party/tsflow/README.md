# TSFlow dependency pin, patches and rights

Upstream reference: https://github.com/marcelkollovieh/TSFlow.git
Full pin: `ff893176a41a1037f63efae674aab71480eee7d0` (`ff89317`).

Attribution from pinned README: Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn and Stephan Günnemann, *Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting*, ICLR2025. Checkout inspection was read-only; no remote was queried.

## Exact changes

- patches/model_compatibility.patch: select Cauchy CPU/GPU backend from tensor device in s4.py; remove unavailable BackboneModelMultivariate import and explicitly reject unsupported multivariate use in tsflow_cond.py. The practical uses univariate forecasting.
- patches/environment.patch: exact local pyproject.toml/pixi.lock changes, separate from model-source changes.
- patched_file_sha256.json: expected resulting four file hashes.

Patches were not privacy-edited. Reconstruction applied both to exported pinned baseline files and compared all four hashes. This is not a clean installation or model evaluation.

## Reconstruct in a NEW workspace — instructions only

Do not apply patches to the existing research checkout. Once access/rights are authorized:

```sh
git clone https://github.com/marcelkollovieh/TSFlow.git TSFlow
cd TSFlow
git checkout --detach ff893176a41a1037f63efae674aab71480eee7d0
git apply --check ../VibrationPrediction/third_party/tsflow/patches/model_compatibility.patch
git apply ../VibrationPrediction/third_party/tsflow/patches/model_compatibility.patch
git apply --check ../VibrationPrediction/third_party/tsflow/patches/environment.patch
git apply ../VibrationPrediction/third_party/tsflow/patches/environment.patch
```

Compare all resulting SHA-256 values with patched_file_sha256.json before environment setup. The forecasting editable ../TSFlow dependency must resolve to this sibling. Upstream pin alone is insufficient.

## Redistribution blocker

No tracked LICENSE/COPYING/NOTICE exists at the inspected commit; its README identifies no grant. Public availability/citation do not establish permission. Patches contain upstream context and require rights review before either GitLab or GitHub publication. They are staged privately for review, not as a redistribution grant. No upstream branch was modified; the entire checkout, datasets, checkpoints and unrelated untracked files are not bundled.
