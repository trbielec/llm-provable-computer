# Phase 0 Test Baseline

**Date:** 2026-04-06
**Branch:** issue-29-investigation
**Base commit:** HEAD at branch creation from `main` (dbcee8e)
**Overall status: PASS**

All three test invocations completed with zero failures across every test suite.

---

## 1. `cargo test`

**Command:** `cargo test`
**Result: PASS**

| Metric | Value |
|--------|-------|
| Passed | 426 |
| Failed | 0 |
| Ignored | 13 |
| Total wall time (sum of suite times) | ~254.75s |

### Suite-by-suite breakdown

| Test binary / suite | Passed | Failed | Ignored | Time |
|---------------------|--------|--------|---------|------|
| lib (unittests) | 229 | 0 | 4 | 36.06s |
| bin/tvm (unittests) | 0 | 0 | 0 | 0.00s |
| tests/approximate.rs | 4 | 0 | 0 | 0.00s |
| tests/assembly.rs | 3 | 0 | 0 | 0.00s |
| tests/audit_repro.rs | 1 | 0 | 0 | 0.00s |
| tests/batch_sweep.rs | 3 | 0 | 0 | 8.54s |
| tests/burn_model.rs | 0 | 0 | 0 | 0.00s |
| tests/cdmn_backend_tvm.rs | 1 | 0 | 0 | 0.00s |
| tests/cdmn_decision.rs | 5 | 0 | 0 | 0.00s |
| tests/cdmn_decision_stark.rs | 1 | 0 | 0 | 25.94s |
| tests/cdmn_decision_stark_direct.rs | 3 | 0 | 1 | 13.07s |
| tests/cdmn_examples.rs | 1 | 0 | 0 | 0.00s |
| tests/cdmn_parser.rs | 5 | 0 | 0 | 0.00s |
| tests/cdmn_semantics.rs | 2 | 0 | 0 | 0.00s |
| tests/cli.rs | 13 | 0 | 0 | 2.80s |
| tests/commit_audit.rs | 2 | 0 | 0 | 0.12s |
| tests/dmn.rs | 12 | 0 | 0 | 0.00s |
| tests/dmn_decision_stark_direct.rs | 11 | 0 | 0 | 28.40s |
| tests/e2e.rs | 41 | 0 | 0 | 0.34s |
| tests/heterogeneous_batch.rs | 1 | 0 | 0 | 0.03s |
| tests/hull_cache.rs | 13 | 0 | 0 | 0.02s |
| tests/interpreter.rs | 4 | 0 | 0 | 0.10s |
| tests/loan_sweep.rs | 4 | 0 | 0 | 2.05s |
| tests/milestone1_full.rs | 0 | 0 | 0 | 0.00s |
| tests/onnx_export.rs | 0 | 0 | 0 | 0.00s |
| tests/property_tests.rs | 9 | 0 | 0 | 3.14s |
| tests/python_validation.rs | 0 | 0 | 0 | 0.00s |
| tests/recursive_stark.rs | 2 | 0 | 0 | 124.66s |
| tests/runtime.rs | 34 | 0 | 0 | 0.13s |
| tests/stress.rs | 10 | 0 | 0 | 8.89s |
| tests/transformer_stark.rs | 7 | 0 | 7 | 0.04s |
| tests/transformer_trace.rs | 4 | 0 | 0 | 0.00s |
| tests/vanillastark_smoke.rs | 1 | 0 | 0 | 0.42s |
| doc-tests | 0 | 0 | 1 | 0.00s |

### Ignored tests (cargo test)

These tests are marked `#[ignore]` with explicit reasons and are excluded from the default suite:

1. `cdmn::stark_direct::tests::medication_policy_compact_trace_satisfies_transition_constraints` — deep AIR/trace regression for compact quantified medication policy
2. `poet_proof::tests::test_poet_proof_simple` — Poet forward-pass proofs are still too slow for the default test suite
3. `proof::tests::lab_policy_round_trips_through_stark_proof` — expensive derived-expression STARK regression
4. `proof::tests::quantified_medication_policy_round_trips_through_stark_proof` — expensive quantified STARK regression
5. `cdmn_decision_stark_direct::medication_policy_direct_stark_round_trips_success_and_failure` — quantified medication direct proof now passes, but still too slow for the default debug suite
6. `transformer_stark::forward_pass_cost_estimate_matches_staged_cost_estimate` — shared transformer cost estimation remains expensive in the default suite
7. `transformer_stark::forward_pass_proof_round_trips_for_average_hard_execution` — covered by the proof-regression CI job; too slow for the default local suite
8. `transformer_stark::staged_skeleton_cost_estimate_reports_trace_shape` — shared transformer cost estimation remains expensive in the default suite
9. `transformer_stark::staged_skeleton_profiled_proof_round_trips_for_average_hard_execution` — shared transformer profiled proof remains expensive in the default suite
10. `transformer_stark::staged_skeleton_proof_round_trips_compare_branch_execution` — shared transformer compare proof remains expensive in the default suite
11. `transformer_stark::staged_skeleton_proof_round_trips_for_average_hard_execution` — covered by the proof-regression CI job; too slow for the default local suite
12. `transformer_stark::staged_skeleton_proof_round_trips_overflowing_raw_accumulator` — shared transformer overflow proof remains expensive in the default suite
13. `doc-test: src/batch.rs - batch::BatchRunner (line 26)` — doc-test ignored

### Notable timing

- `tests/recursive_stark.rs`: 124.66s — both tests exceeded the 60-second warning threshold but completed successfully
  - `aggregate_proof_round_trips_for_single_inner_proof`
  - `aggregate_proof_rejects_tampered_claim_digest`

---

## 2. `cargo test --all-features`

**Command:** `cargo test --all-features`
**Result: PASS**

Features activated: `burn-model`, `onnx-export`, `metal-batch`, `full`, and all transitive feature flags.

| Metric | Value |
|--------|-------|
| Passed | 453 |
| Failed | 0 |
| Ignored | 13 |
| Total wall time (sum of suite times) | ~315.37s |

### Suite-by-suite breakdown (delta from default suite)

All suites from `cargo test` ran and passed. Additional tests activated by features:

| Test suite | Default | All-features | Delta |
|------------|---------|--------------|-------|
| lib (unittests) | 229 | 233 | +4 (`burn_model`, `onnx_export` unit tests unlocked) |
| tests/batch_sweep.rs | 3 | 6 | +3 (metal sweep tests: `metal_sweep_is_odd_256`, `metal_sweep_double_256`, `metal_sweep_is_odd_exhaustive_16bit`) |
| tests/burn_model.rs | 0 | 4 | +4 (burn model tests unlocked) |
| tests/cli.rs | 13 | 16 | +3 (burn/ONNX CLI tests unlocked) |
| tests/heterogeneous_batch.rs | 1 | 2 | +1 (`test_heterogeneous_batch_metal`) |
| tests/loan_sweep.rs | 4 | 6 | +2 (metal loan sweep tests) |
| tests/milestone1_full.rs | 0 | 3 | +3 (full milestone1 workflow tests unlocked) |
| tests/onnx_export.rs | 0 | 4 | +4 (ONNX export tests unlocked) |
| tests/property_tests.rs | 9 | 10 | +1 (`burn_matches_native_on_random_average_hard_programs`) |
| tests/python_validation.rs | 0 | 2 | +2 (Python validation tests unlocked) |
| tests/recursive_stark.rs | 2 | 2 | 0 (160.37s — same two tests, both completed) |

### Notable timing

- `tests/recursive_stark.rs`: 160.37s — same two long-running tests as in the default suite, both passed

---

## 3. `cargo test --features metal-batch`

**Command:** `cargo test --features metal-batch`
**Result: PASS**

| Metric | Value |
|--------|-------|
| Passed | 432 |
| Failed | 0 |
| Ignored | 13 |
| Total wall time (sum of suite times) | ~256.26s |

### Suite-by-suite breakdown (vs. default suite)

| Test suite | Default | metal-batch | Delta |
|------------|---------|-------------|-------|
| lib (unittests) | 229 | 229 | 0 |
| tests/batch_sweep.rs | 3 | 6 | +3 (metal sweep tests: same three as all-features) |
| tests/heterogeneous_batch.rs | 1 | 2 | +1 (`test_heterogeneous_batch_metal`) |
| tests/loan_sweep.rs | 4 | 6 | +2 (metal loan sweep tests) |
| All other suites | same | same | 0 |

### Notable timing

- `tests/recursive_stark.rs`: 125.57s — both tests exceeded the 60-second warning threshold but completed successfully

---

## Summary

| Command | Passed | Failed | Ignored | Status |
|---------|--------|--------|---------|--------|
| `cargo test` | 426 | 0 | 13 | PASS |
| `cargo test --all-features` | 453 | 0 | 13 | PASS |
| `cargo test --features metal-batch` | 432 | 0 | 13 | PASS |

**Overall: PASS**

Zero failures across all three invocations. The 13 ignored tests are consistently the same across all runs — all carry explicit `#[ignore]` annotations with documented reasons (too slow for the default suite or gated to CI-only proof-regression jobs). No unexpected warnings or error-level output was observed.

This report serves as the frozen baseline for the `issue-29-investigation` branch against which post-change test results should be compared.
