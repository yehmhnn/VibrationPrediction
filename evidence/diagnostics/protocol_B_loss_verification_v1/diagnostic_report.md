# Protocol B loss verification

Validation-only CPU diagnostics. Full numerical details: diagnostic_results.json.

- A_repeatability: PASS
- B_loss_calculation: PASS
- C_context_future: PARTIAL
- D_batch_dependence: PASS
- E_parameter_sensitivity: PASS

Test C lacks a later checkpoint; only the retained epoch-8 checkpoint can be decomposed.
