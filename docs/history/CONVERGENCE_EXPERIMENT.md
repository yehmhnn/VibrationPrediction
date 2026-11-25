# Primary convergence experiment preparation

Implementation only: no CWRU training, evaluation, or figure generation was run.
Original training scripts, historical logs/checkpoints/datasets, TSFlow source,
and the preserved revised report are unchanged. Convergence infrastructure is
extended for the approved dual-protocol design below. New runner reuses existing model construction,
record-aware loaders, L512/H96, seed42, strides32/96, batch32, AdamW lr0.001,
and TSFlow eight samples/eight Euler steps/frequency H. No resume support.

## Files

- scripts/convergence_utils.py: losses, validation RNG isolation, CSV and assessment.
- scripts/run_convergence_experiment.py: fresh-only run for either model.
- scripts/plot_convergence_losses.py: logarithmic objective-matched loss plots.
- scripts/test_convergence_logging.py: synthetic checks only.

## Objective definitions and measurement

TSFlow training and validation both call _extract_features followed by p_losses.
x1 is the normalized observed context concatenated with future target; x0 is
observed context plus a GP future sample conditioned only on the past. Features
contain past context, GP conditional future mean, and observation mask. p_losses
computes mean squared velocity error over the entire context+future path, as in
the original driver. It is not a forecast MSE, and it is not restricted to the
future portion. No target definition or TSFlow source was changed.

Epoch training loss is weighted by path element count across batches during
optimization, rather than the old equal-batch average. Validation uses the same
objective with held-out targets, model.eval(), torch.no_grad(), and no optimizer.
Four Monte Carlo passes by default use fixed seeds420001..420004, fixed order,
and batch16. Fixed seeds are reused each epoch; Python, NumPy, CPU Torch, and
selected-device CUDA RNG are restored afterward. This reduces evaluation noise
and prevents validation draws from perturbing training RNG. Draw count and seed
are configurable. Fixed finite draws do not eliminate Monte Carlo uncertainty.

TimesNet training_forecast_MSE and validation_forecast_MSE are element-weighted
mean squared normalized waveform error. Training is measured during optimization
with training mode/dropout; validation is an evaluation-mode post-epoch pass.
They use the same mathematical objective but are not losses of identical model
states/modes. No duplicate post-epoch training pass is added. Validation RMSE is
sqrt(validation MSE), and MAE is element-weighted absolute forecast error.

Each epoch logs the seven requested fields plus cumulative optimization_seconds,
per-epoch validation_seconds, and assessment. optimizer_step counts actual
optimizer updates. elapsed_training_seconds includes optimization, validation,
and checkpoint/logging work after initialization; optimization_seconds measures
training batches separately. CUDA timings synchronize at interval boundaries.
No test values are used in stopping or selection.

## Proposed practical stabilization criterion (configurable, not validated)

Checkpoint selection always uses the strictly lowest validation forecast RMSE.
Stopping tolerance never changes that selection rule.

Default assessment parameters:

- min_delta=0.001 normalized RMSE: meaningful improvement from the last anchor.
- patience=10 validation evaluations without that improvement.
- min_assessment_epochs=20; stability_window=10, needing two complete windows.
- Last-window RMSE range / mean <=0.02.
- Absolute fitted linear loss change over the window / mean <=0.02, separately
  for training and validation objective histories.
- Mean validation/training objective ratio must not grow by >0.05 relative to
  the preceding window. This is a widening-gap diagnostic, not proof of overfit;
  online training and evaluation validation have different modes/model states.

Default stop_policy=stabilization stops only when patience and all stability
checks hold. patience policy may stop without stabilization; none disables
stopping but keeps assessment. Reaching max_epochs is maximum_epoch_budget,
never convergence. Results explicitly record
mathematical_convergence_demonstrated=false. Practical stabilization remains a
heuristic about the measured trajectory, not a mathematical guarantee. Thresholds
must be chosen before running and inspected scientifically afterward; a noisy
curve may never pass them. The full budget has no default and needs authorization.

## Later launch commands (NOT executed)

Use the existing environment; do not replace the placeholders until budgets
and fresh output names are approved. Shell variables below intentionally fail
if no budget has been supplied. Existing output directories are rejected.

```sh
cd .
PY=../TSFlow/.pixi/envs/default/bin/python
"$PY" -B scripts/run_convergence_experiment.py --protocol single_105_0 --model tsflow \
  --max-epochs "${TSFLOW_APPROVED_MAX_EPOCHS:?Set approved TSFlow budget}" \
  --output-dir ./logs/convergence/tsflow_L512_H96_seed42_objective_v1 \
  --device cuda
"$PY" -B scripts/run_convergence_experiment.py --protocol single_105_0 --model timesnet \
  --max-epochs "${TIMESNET_APPROVED_MAX_EPOCHS:?Set approved TimesNet budget}" \
  --output-dir ./logs/convergence/timesnet_L512_H96_seed42_objective_v1 \
  --device cuda
```

Test evaluation is off by default. --evaluate-test requests one final pass at
the selected checkpoint, with the existing test split, stride96, and evaluation
seed20042. It is not needed to assess convergence. No historical run is resumed.

## Later plotting commands (NOT executed)

```sh
"$PY" -B scripts/plot_convergence_losses.py --protocol single_105_0 --model tsflow \
  --history logs/convergence/tsflow_L512_H96_seed42_objective_v1/history.csv \
  --output tsflow_objective_losses.pdf
"$PY" -B scripts/plot_convergence_losses.py --protocol single_105_0 --model timesnet \
  --history logs/convergence/timesnet_L512_H96_seed42_objective_v1/history.csv \
  --output timesnet_objective_losses.pdf
```

Plots use logarithmic Y axes and never combine velocity loss with forecast RMSE.
They reject missing epochs, missing fields, nonfinite/nonpositive values, and
existing output files. No values are clipped or floored. Historical histories
lack required fields and are intentionally rejected. Full precision is stored;
report display can later round to three decimals.

## Validation

Run only synthetic checks:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B scripts/test_convergence_logging.py
```

Checks include synthetic plateau/improvement/noise/divergence sequences, mode
and RNG restoration, unchanged parameters/gradients during validation, unequal
batch weighting, distinct synthetic manifest splits, two-epoch toy-model CSVs,
and actual TSFlow/TimesNet objective calculations on generated tensors. Tests
use temporary directories; they never open CWRU signals or persist real results.
CPU model checks do not establish CUDA portability or scientific convergence.
TimesNet retains deterministic-algorithm enforcement from its former Lightning
setup; TSFlow retains its original algorithm policy. Fixed draws give reproducible
evaluation within a compatible device/environment, not a cross-device guarantee.
RNG isolation intentionally separates validation from future training randomness;
these fresh runs should not be treated as bit-identical historical continuations.

Validation completed: five synthetic test methods passed in 20.669 seconds,
including actual-model CPU objective checks. No optimizer updates were applied
to the actual TSFlow/TimesNet models; two-epoch integration uses toy models only.

# Approved dual-protocol extension

This supersedes earlier single-protocol-only launch examples. Infrastructure and
preprocessing only; no GPU pilot or real-data training/inference has run.
The runner/config distinguishes single_105_0 and multi_record_B. Legacy default
remains single_105_0; later commands below always specify the protocol explicitly.

## Protocol B data

Output: datasets/processed/cwru_multi_record_B/

Train:105_0,106_1,118_0,119_1,130@6_0,131@6_1.
Validation:107_2,120_2,132@6_2. Test:108_3,121_3,133@6_3.
All IR007,B007,OR007@6 at12kHz on drive-end sensor. Loads0/1 train,2 validation,
3 test; RPM covaries. Fault types and diameter are seen in training.
Curated acquisition mapping: scripts/cwru_protocols.py and
https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data.
MAT contents verify unique DE channel, RPM, finite values and sample counts.
Rate/fault/load/position are acquisition metadata, not MAT-stored fields.

Reproduce only into a NEW directory:

```sh
python3 -B scripts/preprocess_cwru_protocol_b.py \
  --output-dir datasets/processed/cwru_multi_record_B_reproduction
```

All records use training global mean0.010496160688842994 and population
std0.41000491194742045. These use731654 samples from six training files only.
normalization.json stores formula, estimator/ddof, training IDs/count, source
hashes and uses_validation_or_test=false. metadata.json records protocol, splits,
window settings, normalization, processed-array hashes and manifest hash.

manifest.csv contains record_id,parent_record_id,source_relative_path,source_path,
source_key,condition,fault_type,fault_diameter_inch,load_id,rpm,sampling_rate_hz,
num_samples,split,sensor,outer_race_position,source_start,source_end_exclusive,
duration_seconds,relative_path,normalization_provenance,normalization_sha256,
source_sha256,metadata_authority,metadata_verification. Array/provenance paths
are relative to this directory; source MAT paths are absolute.

L512/H96 and strides32/96 remain unchanged. Totals:
train731654 samples/60.971167s/22754 windows; validation365102/30.425167s/3785;
test367044/30.587s/3806. Batch32 gives712 updates/epoch vs70 for single105_0.
Uniform-window sampling weights by window count; record lengths are similar.
Never concatenate records for windows. IdentifiedDataset preserves IDs/origins
without exposing them as model features.

## Evaluation outputs

MAE=sum absolute errors/count; MSE=sum squared errors/count; RMSE=sqrt(MSE).
Per-record outputs include MAE/MSE/RMSE, window/element counts, fault/load/RPM.
Pooled metrics weight all elements. Equal-record MAE/RMSE are arithmetic means;
mean record RMSE is not sqrt(mean record MSE). Per-fault metrics pool elements
belonging to that fault. Recording details use float64 reduction; top-level
pooled checkpoint metrics retain previous float32 reductions, so negligible
rounding differences are possible.

history.csv preserves objective/timing fields. validation_records.jsonl records
per-epoch record/fault/pooled/equal-record summaries; assessment_history.jsonl
retains every stabilization diagnostic. result.json retains selected-validation
record metrics and optional selected-checkpoint test/reference outputs.

--evaluate-test enables one post-selection pass, saving targets.npy,
predictive_mean.npy,record_ids.npy,forecast_origins.npy and prefix metrics24/48/72/96.
FullH96 pooled prefix reproduces the main metric; no independent horizon fits.
--in-condition-reference uses a separate stride96 training-record loader ONLY
after selection: reused training data, not independent evidence. Neither flag
is enabled in pilot/convergence examples. Request final evaluations separately.
No test/reference values enter optimization, normalization, stopping or selection.

Architecture, learning rate, training batch32, seed42, solver/sample counts,
evaluation batches16/32 and single-record origins/data remain unchanged. Earlier
source is backed up under scripts/infrastructure_backups/protocol_B_20261009/.

## Later GPU pilot (NOT executed)

Propose a one-epoch smoke pilot per model on Protocol B to check CUDA execution,
finite losses, IDs/counts and timings. This is not convergence evidence. Explicitly
approve and set GPU_PILOT_APPROVED_MAX_EPOCHS before launching; no default is set.

```sh
cd .
CONVERGENCE_PY=../TSFlow/.pixi/envs/default/bin/python
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol multi_record_B --model tsflow \
  --max-epochs "${GPU_PILOT_APPROVED_MAX_EPOCHS:?Set approved pilot budget}" \
  --output-dir logs/convergence/multi_record_B_tsflow_gpu_pilot_v1 --device cuda
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol multi_record_B --model timesnet \
  --max-epochs "${GPU_PILOT_APPROVED_MAX_EPOCHS:?Set approved pilot budget}" \
  --output-dir logs/convergence/multi_record_B_timesnet_gpu_pilot_v1 --device cuda
```

## Later four convergence studies (NOT executed)

Approve budgets individually; output paths must not already exist. No test or
in-condition reference is enabled. These are four separate fresh training runs.

```sh
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol single_105_0 --model tsflow \
  --max-epochs "${SINGLE_TSFLOW_APPROVED_MAX_EPOCHS:?Set approved budget}" \
  --output-dir logs/convergence/single_105_0_tsflow_objective_v2 --device cuda
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol single_105_0 --model timesnet \
  --max-epochs "${SINGLE_TIMESNET_APPROVED_MAX_EPOCHS:?Set approved budget}" \
  --output-dir logs/convergence/single_105_0_timesnet_objective_v2 --device cuda
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol multi_record_B --model tsflow \
  --max-epochs "${MULTI_TSFLOW_APPROVED_MAX_EPOCHS:?Set approved budget}" \
  --output-dir logs/convergence/multi_record_B_tsflow_objective_v1 --device cuda
"$CONVERGENCE_PY" -B scripts/run_convergence_experiment.py \
  --protocol multi_record_B --model timesnet \
  --max-epochs "${MULTI_TIMESNET_APPROVED_MAX_EPOCHS:?Set approved budget}" \
  --output-dir logs/convergence/multi_record_B_timesnet_objective_v1 --device cuda
```

Plot later with scripts/plot_convergence_losses.py --protocol PROTOCOL --model MODEL
--history RUN/history.csv --output NEW.pdf. Both protocols use matching objective
losses and logY. Plotting rejects mismatched config/model/protocol, missing epochs,
nonfinite/nonpositive loss and existing figure paths. No final figures generated.

Validation: five ProtocolB tests passed in1.588s and five original convergence tests
passed in3.931s. The first suite includes real-data integrity/loading ONLY, no real
model evaluation. Actual-model objective checks use generated tensors. Toy-model
fits and metric tests use temporary synthetic data. Original single-region arrays,
manifests and normalization remain intact. Budget/stopping diagnostics remain
heuristics; CUDA behavior is pending the authorized pilot. No new baseline scores
exist for ProtocolB; see PAPER_EXPERIMENT_PLAN.md for later baseline work.
