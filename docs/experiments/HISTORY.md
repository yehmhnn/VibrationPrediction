# Research history and evidence boundaries

The Phase2A EXPERIMENT_MAP.json inventories completed experiment families with exact original artifacts and available source hashes. Some early executions have no execution-time source hash; the present source is not proof of the precise historical code version.

- Study I: single105_0 chronological split with gaps and train-only normalization. Historical5/30/60/100-epoch TSFlow runs, TimesNet30 and separately trained horizons24/48/72/96 differ from later objective-matched convergence runs. Horizon-dependent origin grids differ; fixed-H96 prefixes answer a distinct lead-distance question. The later TSFlow run stopped at137 under a max200 budget, selecting127; its original saved result reports no test evaluation. Historical test metrics cannot be assigned to it.
- Study II: primary controlled recording-disjoint three-fault comparison, six0–1hp training/three2hp validation records, three3hp records reserved. Original TSFlow150 and TimesNet100 realized budgets and checkpoint-selection/stability settings are retained, not made artificially identical.
- E1–E6: fixed follow-up plan, eight fresh runs, historical seed 42 baselines reused. See RESULTS.md and configs/regularization/overnight_20261020_v1/. The runner copies are preserved archives, not portable launchers.
- RPM-conditioned models, autoregression/zero references, early protocols, pilots and failed attempts are historical exploratory/infrastructure evidence. They are not interchangeable with Study II. Scripts run_baselines.py, tune_autoregression.py and summarize_rpm_conditioning.py remain where they were; original logs/*.csv and PAPER_EXPERIMENT_PLAN.md retain their historical meaning.
- Wavelet/Welch/CWRU spectral work is post-report offline exploration. No denoised model was trained; unfavorable distortion results remain included.

Nothing in this inventory authorizes deletion. No experiments were rerun in Phase2B; the report and Git history were untouched.

## Submission locations

Compact historical CSVs are in evidence/historical/, not logs/. They include older protocols with historical test use and are explicitly separate from final Study II. No new reserved-test scores were computed. Original plans under docs/history/ are historical documents, not current executable instructions.
