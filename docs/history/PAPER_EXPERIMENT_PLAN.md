# Paper-oriented experiment plan

Status: infrastructure and approved Protocol B preprocessing only. No new real-data
model training, pilot, inference, final figures, or report edits have been performed.
Numerical model results and claims of stabilization remain pending authorized runs.

## Original research questions (unchanged)

RQ1: How effectively can TSFlow and TimesNet forecast high-frequency mechanical vibration signals?

RQ2: How does forecasting performance vary with prediction horizon, and how do the models differ in training and prediction behavior?

RQ3: To what extent does forecasting performance generalize across different operating conditions?

## Evidence roles

- single_105_0 is Preliminary Single-Recording Experiments. Preserve existing
  forecasting, baselines, independent H24/48/72/96 models, fixed-H96 prefixes,
  waveforms, and histories. These establish local feasibility, not multi-recording
  generalization. Retrain both methods separately, when approved, to obtain
  complete objective-matched convergence evidence.
- multi_record_B is the main RQ1/RQ3 experiment. Train105_0/106_1,
  118_0/119_1,130@6_0/131@6_1. Validate107_2/120_2/132@6_2;
  test108_3/121_3/133@6_3. All12kHz drive-end, diameter0.007inch;
  inner-race, ball, centered outer-race faults. This tests joint held-out
  load/RPM conditions for seen fault families and diameter.
- RQ2 uses four separate convergence studies (two models x two protocols),
  existing preliminary separate-horizon evidence, and new main fixed-H96
  cumulative prefixes24/48/72/96. Prefixes require one trainedH96 model per
  method, not four additional fits. They are not independent-horizon results.

## Proposed paper structure

1. Introduction: motivation, scope, the three RQs.
2. Dataset and Experimental Setup
   2.1 Preliminary single-recording protocol
   2.2 Main multi-recording Protocol B
   2.3 Preprocessing and leakage prevention
3. Methods
   3.1 TSFlow
   3.2 TimesNet
   3.3 Baselines and metrics
   3.4 Convergence assessment
4. Results
   4.1 Preliminary single-recording forecasting
   4.2 Single-recording convergence and horizon analysis
   4.3 Multi-recording convergence
   4.4 Multi-recording forecasting comparison
   4.5 Operating-condition generalization
   4.6 Qualitative examples and failure cases
5. Discussion and Limitations
6. Conclusion

Keep preliminary results compact; combine related panels rather than repeat
entire metric tables. No existing LaTeX structure is changed in this task.

## Planned tables and figures

| Item | RQ | Required output | Existing sufficient? | New training? | Role / interpretation limit |
|---|---|---|---|---|---|
| Protocol/split table and diagram | All | Both manifests, metadata, normalization | Yes for setup | No | Preliminary vs main scopes explicit; no random window split |
| Optional vibration diversity panel | RQ1/RQ3 context | Signals; common Welch parameters | Sources available | No | Dataset description; spectral differences are not independence |
| Preliminary forecast/baseline table | RQ1 | Existing105_0 JSON/CSV | Yes | No | Historical budgets; no converged-model claim |
| Single-recording log-loss panels | RQ2 | Complete train/validation FM and MSE histories | No; old fields missing | Both models on single105_0 | Preliminary; FM is not forecast MSE |
| Preliminary separate-horizon table/panel | RQ2 | Existing separately trained H24/48/72/96 results | Yes | No | Different tasks/budgets/origins; not main prefixes |
| Main log-loss panels | RQ2 | ProtocolB histories and assessment diagnostics | No | Both models on ProtocolB | Practical stabilization only if supported |
| Optional validation RMSE vs steps/time | RQ2 | Histories, optimization/validation timing | No | Same fits | Predeclared target/stabilization; equal epochs are not equal compute |
| Main forecasting table/RMSE panel | RQ1 | Selected test scores, per-record/pooled/equal-record outputs | No | Both models on ProtocolB | Formal comparison conditional on both practically stabilizing |
| Main baseline rows | RQ1 | Protocol-specific Zero and validation-selected AR evidence | No | No neural training; baseline fitting/evaluation later | Old105_0 scores cannot be reused; adapters may need extension |
| Operating-condition/fault table | RQ3 | Best validation load2 record metrics; selected test load3 metrics | No | Same ProtocolB fits | Joint load/RPM shift, not isolated causal effect; records differ |
| Optional in-condition reference | RQ3 context | Post-selection training-record evaluation, stride96 | No | Same fit; evaluation later | Reused training data, not independent evidence; never affects selection/stopping |
| Main fixed-model prefix panel/table | RQ2 | One H96 model/method; identical targets, IDs/origins; prefixes24/48/72/96 | No | Same ProtocolB fits only | Cumulative averages, not per-lead errors or independently fitted models |
| Main qualitative/failure examples | RQ1/RQ2/RQ3 | Actual/predicted arrays with IDs/origins | No | Same ProtocolB fits | Predeclare selection per fault; examples cannot establish mechanisms |

## Fair comparison and reporting

Use identical splits, normalized arrays, origins, seed42, L512/H96, strides32/96,
batch32, existing architectures/learning rates. TSFlow retains eight generated
samples/eight Euler steps. Evaluation batches remain16 for TSFlow and32 for
TimesNet, as before; origins and metric definitions match. Fit normalization
only on training records. Never concatenate signals for windows.

Select checkpoints using pooled validation forecasting RMSE for both methods.
Publish per-record counts/errors, per-fault pooled errors, pooled metrics over all
target elements, and arithmetic equal-record averages of MAE/RMSE. Mean record
RMSE is neither pooled RMSE nor sqrt(mean record MSE). Similar record lengths
make uniform-window training approximately record-balanced; amplitude differences
can still dominate squared error. Keep per-fault results visible.

After both methods practically stabilize, perform one authorized selected-model
test evaluation. Test/reference are off by default and never affect selection or
stopping. Best-validation summaries supply the load2 comparison. Save full precision
and display report metrics to three decimals. Avoid significance claims from one
neural seed and only three held-out test recordings.

## Convergence, budgets, and speed

Budgets require explicit approval; none is assigned here. Two protocols require
four fresh runs, never resumes. Retain documented stabilization parameters and
all diagnostics; inspect per-fault validation trajectories as well as pooled
curves. Budget limits or exhausted patience do not demonstrate convergence.
If either model fails practical stabilization, report budget-limited descriptive
performance rather than a converged-model ranking. Do not assume ProtocolB
improves scores or favors either model.

Single105_0:2210 windows/70 updates per epoch;242 validation/test origins.
ProtocolB:22754 windows/712 updates per epoch;3785 validation/3806 test origins.
Main workload is about10.3x training windows and15.6x validation origins, before
added FM validation passes. Separate optimization, validation, elapsed, and final
evaluation timings. No runtime estimate is a measured result.

## Limits and completion decisions

ProtocolB does not establish unseen fault types, diameters, machines, sensors,
or independent bearing specimens. Load/RPM covary. Training/validation objective
modes differ; finite FM draws retain uncertainty. Frequency H is inherited,
not an established vibration period. Some recordings were already inspected;
do not call the test set untouched. Decide with the tutor whether repeated seeds
or more recordings are needed after the bounded main study, not automatically.
