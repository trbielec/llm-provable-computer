//! Input-sweep tests for the batch runners.
//!
//! Each test compiles a small program that uses the initial ACC value as its
//! sole input, sweeps ACC = 0..N through both the CPU [`BatchRunner`] (and
//! optionally the Metal [`MetalBatchRunner`]), then verifies every result
//! against the [`NativeInterpreter`] running from the same initial state.
//!
//! These tests prove that:
//! 1. The batch runner produces correct outputs across the full input range.
//! 2. The transformer and native execution paths agree for all inputs.
//! 3. The Metal GPU path agrees with the CPU path (when `metal-batch` is
//!    enabled on macOS).

use std::sync::Arc;

use llm_provable_computer::{
    batch::BatchRunner, MachineState, NativeInterpreter, ProgramCompiler, TransformerVmConfig,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn compile(source: &str) -> Arc<llm_provable_computer::model::TransformerVm> {
    Arc::new(
        ProgramCompiler
            .compile_source(source, TransformerVmConfig::default())
            .expect("compile"),
    )
}

/// Build `n` initial states with `acc = 0, 1, …, n-1` and the program's
/// default memory.
fn make_states(model: &llm_provable_computer::model::TransformerVm, n: usize) -> Vec<MachineState> {
    let base = model.program().initial_memory().to_vec();
    (0..n)
        .map(|i| {
            let mut s = MachineState::with_memory(base.clone());
            s.acc = i as i64;
            s
        })
        .collect()
}

/// Run every initial state through the native interpreter and return the
/// final ACC values as the reference.
fn native_sweep(
    model: &llm_provable_computer::model::TransformerVm,
    initial_states: &[MachineState],
    max_steps: usize,
) -> Vec<i64> {
    initial_states
        .iter()
        .map(|init| {
            let mut interp = NativeInterpreter::new_with_state(
                model.program().clone(),
                model.config().attention_mode.clone(),
                init.clone(),
                max_steps,
            );
            interp.run().expect("native run").final_state.acc
        })
        .collect()
}

// ── CPU batch sweep tests ─────────────────────────────────────────────────────

const MAX_STEPS: usize = 512;

/// Sweep through `double.tvm` (acc → 2 * acc) with 256 inputs.
#[test]
fn cpu_sweep_double_256() {
    let src = include_str!("../programs/double.tvm");
    let model = compile(src);
    let states = make_states(&model, 256);
    let native_acc = native_sweep(&model, &states, MAX_STEPS);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        let expected = native_acc[r.run_idx];
        assert!(r.result.is_ok(), "run {} failed: {:?}", r.run_idx, r.result);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
        assert_eq!(
            r.final_state.acc, expected,
            "run {} (input {}): expected acc={}, got {}",
            r.run_idx, r.run_idx, expected, r.final_state.acc,
        );
        // Explicit check: double of input
        assert_eq!(
            r.final_state.acc,
            2 * r.run_idx as i64,
            "run {}: 2 * {} should equal {}",
            r.run_idx,
            r.run_idx,
            2 * r.run_idx,
        );
    }
}

/// Sweep through `is_odd.tvm` (acc → acc & 1) with 256 inputs.
#[test]
fn cpu_sweep_is_odd_256() {
    let src = include_str!("../programs/is_odd.tvm");
    let model = compile(src);
    let states = make_states(&model, 256);
    let native_acc = native_sweep(&model, &states, MAX_STEPS);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    for r in &results {
        let expected = native_acc[r.run_idx];
        assert!(r.result.is_ok(), "run {} failed", r.run_idx);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
        assert_eq!(
            r.final_state.acc, expected,
            "run {} (input {}): expected acc={}, got {}",
            r.run_idx, r.run_idx, expected, r.final_state.acc,
        );
        // Explicit check: low bit of input
        let expected_bit = (r.run_idx as i64) & 1;
        assert_eq!(
            r.final_state.acc, expected_bit,
            "run {}: {} & 1 should equal {}",
            r.run_idx, r.run_idx, expected_bit,
        );
    }
}

/// Exhaustive 16-bit sweep of `is_odd.tvm` — all 65,536 inputs.
///
/// Verifies parity for every possible 16-bit value.
#[test]
fn cpu_sweep_is_odd_exhaustive_16bit() {
    let n = 1usize << 16;
    let src = include_str!("../programs/is_odd.tvm");
    let model = compile(src);
    let states = make_states(&model, n);

    let runner = BatchRunner::new(Arc::clone(&model), MAX_STEPS);
    let results = runner.run(states);

    assert_eq!(results.len(), n);
    for r in &results {
        assert!(r.result.is_ok(), "run {} failed", r.run_idx);
        assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
        let expected = (r.run_idx as i64) & 1;
        assert_eq!(
            r.final_state.acc, expected,
            "parity wrong for input {}",
            r.run_idx
        );
    }
}

// ── Metal GPU batch sweep tests ───────────────────────────────────────────────

#[cfg(all(feature = "metal-batch", target_os = "macos"))]
mod metal_sweep {
    use super::*;
    use llm_provable_computer::metal_batch::MetalBatchRunner;

    /// Metal sweep of `double.tvm` with 256 inputs.
    #[test]
    fn metal_sweep_double_256() {
        let src = include_str!("../programs/double.tvm");
        let model = compile(src);
        let states = make_states(&model, 256);
        let native_acc = native_sweep(&model, &states, MAX_STEPS);

        let runner = MetalBatchRunner::new(Arc::clone(&model), MAX_STEPS).expect("Metal init");
        let results = runner.run(states);

        for r in &results {
            assert!(r.result.is_ok(), "run {} failed", r.run_idx);
            assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
            assert_eq!(
                r.final_state.acc, native_acc[r.run_idx],
                "Metal/native mismatch on run {} (input {})",
                r.run_idx, r.run_idx,
            );
            assert_eq!(r.final_state.acc, 2 * r.run_idx as i64);
        }
    }

    /// Metal sweep of `is_odd.tvm` with 256 inputs.
    #[test]
    fn metal_sweep_is_odd_256() {
        let src = include_str!("../programs/is_odd.tvm");
        let model = compile(src);
        let states = make_states(&model, 256);
        let native_acc = native_sweep(&model, &states, MAX_STEPS);

        let runner = MetalBatchRunner::new(Arc::clone(&model), MAX_STEPS).expect("Metal init");
        let results = runner.run(states);

        for r in &results {
            assert!(r.result.is_ok(), "run {} failed", r.run_idx);
            assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
            assert_eq!(
                r.final_state.acc, native_acc[r.run_idx],
                "Metal/native mismatch on run {} (input {})",
                r.run_idx, r.run_idx,
            );
            assert_eq!(r.final_state.acc, (r.run_idx as i64) & 1);
        }
    }

    /// Exhaustive 16-bit Metal sweep of `is_odd.tvm` — 65,536 inputs.
    #[test]
    fn metal_sweep_is_odd_exhaustive_16bit() {
        let n = 1usize << 16;
        let src = include_str!("../programs/is_odd.tvm");
        let model = compile(src);
        let states = make_states(&model, n);

        let runner = MetalBatchRunner::new(Arc::clone(&model), MAX_STEPS).expect("Metal init");
        let results = runner.run(states);

        assert_eq!(results.len(), n);
        for r in &results {
            assert!(r.result.is_ok(), "run {} failed", r.run_idx);
            assert!(r.final_state.halted, "run {} did not halt", r.run_idx);
            assert_eq!(
                r.final_state.acc,
                (r.run_idx as i64) & 1,
                "parity wrong for input {}",
                r.run_idx
            );
        }
    }
}
