//! Exhaustive input-sweep tests for `loan_approval.tvm`.
//!
//! Sweeps `credit_score = 0..=1023` across three financial scenarios, each
//! targeting a different denial path in the short-circuit evaluation:
//!
//! | Scenario | income  | annual_debt | Expected denials                  |
//! |----------|---------|-------------|-----------------------------------|
//! | Default  | 75,000  | 24,000      | credit_score < 620                |
//! | Low income | 40,000 | 12,000     | credit_score ≥ 620 (income check) |
//! | High DTI | 75,000  | 35,000      | credit_score ≥ 620 (DTI check)    |
//!
//! Each test verifies results against the native interpreter (zero mismatches).

use std::sync::Arc;

use llm_provable_computer::{
    batch::BatchRunner, MachineState, NativeInterpreter, ProgramCompiler, TransformerVmConfig,
};

const SRC: &str = include_str!("../programs/loan_approval.tvm");
const MAX_STEPS: usize = 64;
// Memory layout: [income, annual_debt, scratch×3]
const MEM_SIZE: usize = 5;

fn compile() -> Arc<llm_provable_computer::model::TransformerVm> {
    Arc::new(
        ProgramCompiler
            .compile_source(SRC, TransformerVmConfig::default())
            .expect("compile loan_approval.tvm"),
    )
}

/// Build initial states sweeping credit_score = 0..n with fixed income/debt.
fn make_states(n: usize, income: i64, annual_debt: i64) -> Vec<MachineState> {
    (0..n)
        .map(|credit_score| {
            let mut mem = vec![0i64; MEM_SIZE];
            mem[0] = income;
            mem[1] = annual_debt;
            let mut s = MachineState::with_memory(mem);
            s.acc = credit_score as i64;
            s
        })
        .collect()
}

/// Run each initial state through the native interpreter; return final ACC values.
fn native_reference(
    model: &llm_provable_computer::model::TransformerVm,
    states: &[MachineState],
) -> Vec<i64> {
    states
        .iter()
        .map(|init| {
            let mut interp = NativeInterpreter::new_with_state(
                model.program().clone(),
                model.config().attention_mode.clone(),
                init.clone(),
                MAX_STEPS,
            );
            interp.run().expect("native run").final_state.acc
        })
        .collect()
}

// ── Scenario 1: default parameters (income=75000, debt=24000) ────────────────
//
// Only criterion 1 (credit_score ≥ 620) creates a decision boundary.
// Criteria 2 and 3 both pass for every input.
//
// Expected:
//   credit_score  0..619  → 0 (denied at credit check,   ~6 steps)
//   credit_score 620..850 → 1 (approved, all checks pass, ~21 steps)

#[test]
fn default_scenario_credit_boundary() {
    let model = compile();
    let n = 851; // full credit score range 0..=850
    let states = make_states(n, 75_000, 24_000);
    let reference = native_reference(&model, &states);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        assert!(r.result.is_ok(), "run {} errored", r.run_idx);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);

        let got = r.final_state.acc;
        let expected = reference[r.run_idx];
        assert_eq!(
            got, expected,
            "native mismatch: credit_score={} expected={} got={}",
            r.run_idx, expected, got,
        );

        let credit_score = r.run_idx as i64;
        let expected_decision = if credit_score >= 620 { 1 } else { 0 };
        assert_eq!(
            got, expected_decision,
            "wrong decision: credit_score={} expected={} got={}",
            credit_score, expected_decision, got,
        );
    }

    let approved = results.iter().filter(|r| r.final_state.acc == 1).count();
    assert_eq!(approved, 231, "expected 231 approvals (620..=850), got {approved}");
}

// ── Scenario 2: income below floor (income=40000, debt=12000) ────────────────
//
// Criterion 1 passes for credit_score ≥ 620, then criterion 2 fails (income
// 40,000 < 50,000 floor).  Every input is denied; only the denial path differs.
//
// Expected:
//   credit_score  0..619  → 0 (denied at credit check,  ~6 steps)
//   credit_score 620..850 → 0 (denied at income check, ~10 steps)

#[test]
fn low_income_scenario_all_denied() {
    let model = compile();
    let n = 851;
    let states = make_states(n, 40_000, 12_000);
    let reference = native_reference(&model, &states);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        assert!(r.result.is_ok(), "run {} errored", r.run_idx);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
        assert_eq!(
            r.final_state.acc, reference[r.run_idx],
            "native mismatch at credit_score={}",
            r.run_idx,
        );
        assert_eq!(
            r.final_state.acc, 0,
            "credit_score={} should be denied (income too low)",
            r.run_idx,
        );
    }
}

// ── Scenario 3: DTI too high (income=75000, debt=35000) ──────────────────────
//
// income × 43 = 3,225,000   debt × 100 = 3,500,000   → DTI = 46.7% > 43%
//
// Criterion 1 passes for credit_score ≥ 620, criterion 2 passes (income ok),
// then criterion 3 fails.  All credit-qualified applicants hit the DTI denial path.
//
// Expected:
//   credit_score  0..619  → 0 (denied at credit check, ~6 steps)
//   credit_score 620..850 → 0 (denied at DTI check,   ~21 steps)

#[test]
fn high_dti_scenario_all_denied() {
    let model = compile();
    let n = 851;
    let states = make_states(n, 75_000, 35_000);
    let reference = native_reference(&model, &states);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        assert!(r.result.is_ok(), "run {} errored", r.run_idx);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
        assert_eq!(
            r.final_state.acc, reference[r.run_idx],
            "native mismatch at credit_score={}",
            r.run_idx,
        );
        assert_eq!(
            r.final_state.acc, 0,
            "credit_score={} should be denied (DTI too high)",
            r.run_idx,
        );
    }
}

// ── DTI boundary: debt that puts ratio exactly at 43% ────────────────────────
//
// income=100,000   →   income × 43 = 4,300,000
// debt=43,000      →   debt × 100  = 4,300,000   →  DTI = exactly 43%
//
// The program uses strict inequality (< 43%), so exactly-at-limit is denied.

#[test]
fn dti_exact_boundary_denied() {
    let model = compile();
    // Use a single qualified credit score; focus on the DTI boundary
    let states = make_states(851, 100_000, 43_000);
    let reference = native_reference(&model, &states);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        assert_eq!(r.final_state.acc, reference[r.run_idx]);
        // All should be denied: credit-low ones at check 1, rest at DTI (exactly 43%)
        assert_eq!(
            r.final_state.acc, 0,
            "credit_score={}: DTI exactly 43% should be denied",
            r.run_idx,
        );
    }
}

// ── Metal GPU variants ────────────────────────────────────────────────────────

#[cfg(all(feature = "metal-batch", target_os = "macos"))]
mod metal {
    use super::*;
    use llm_provable_computer::metal_batch::MetalBatchRunner;

    fn metal_runner() -> MetalBatchRunner {
        MetalBatchRunner::new(compile(), MAX_STEPS).expect("Metal init")
    }

    #[test]
    fn metal_default_scenario_matches_native() {
        let model = compile();
        let n = 851;
        let states = make_states(n, 75_000, 24_000);
        let reference = native_reference(&model, &states);

        let results = metal_runner().run(states);
        for r in &results {
            assert!(r.result.is_ok());
            assert!(r.final_state.halted);
            assert_eq!(
                r.final_state.acc, reference[r.run_idx],
                "Metal/native mismatch at credit_score={}",
                r.run_idx,
            );
        }
        let approved = results.iter().filter(|r| r.final_state.acc == 1).count();
        assert_eq!(approved, 231);
    }

    #[test]
    fn metal_high_dti_all_denied() {
        let model = compile();
        let states = make_states(851, 75_000, 35_000);
        let reference = native_reference(&model, &states);

        let results = metal_runner().run(states);
        for r in &results {
            assert_eq!(r.final_state.acc, reference[r.run_idx]);
            assert_eq!(r.final_state.acc, 0);
        }
    }
}
