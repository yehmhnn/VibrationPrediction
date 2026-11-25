# Provenance layers

Historical Phase2A/2B manifests, validation records, source/output hashes and source_sha256 fields attest original research files. Staged text has privacy redactions; old hashes are not assertions about current derivative bytes. Original experimental numbers, splits, hyperparameters and fit/evaluation functions were not silently changed.

STAGING_MANIFEST.json identifies original relative sources and hashes. STAGING_MANIFEST.json inventories current files with source/staged hashes, transformation categories and exclusions. Its own digest is recorded externally to avoid recursion. PHASE2C_VALIDATION.json records checks actually performed. Detailed path substitutions, original-state checks, installation logs and local validation workspaces remain outside the repository.

Raw data/checkpoints/logs and protected snapshots remain private and unchanged. Exclusion is not deletion. Environments/caches are excluded by policy. The retained original Welch CSV is required for metadata spectral redraw; larger denoising spectra and redundant figures remain private intermediates.
