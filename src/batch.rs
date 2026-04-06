//! CPU-parallel batch execution for data-parallel program testing (Phase 1, issue #19).
//!
//! Runs N independent copies of the same compiled program — each with its own
//! [`MachineState`] and [`AddressedMemory`] — using `rayon` thread-level
//! parallelism. Results are returned in input order.
//!
//! This is a pure-CPU baseline. For GPU batch execution on Apple Silicon, see
//! the `metal-batch` feature and [`crate::metal_batch`].

use std::sync::Arc;

use rayon::prelude::*;

use crate::engine::ExecutionResult;
use crate::error::Result;
use crate::model::TransformerVm;
use crate::runtime::ExecutionRuntime;
use crate::state::MachineState;

// ── public types ──────────────────────────────────────────────────────────────

/// Runs N copies of a compiled program with independent initial states in
/// parallel across all available CPU cores.
///
/// # Example
/// ```rust,ignore
/// let model = Arc::new(TransformerVm::new(config, program)?);
/// let runner = BatchRunner::new(model, 512);
/// let states = (0i64..256).map(|acc| {
///     let mut s = MachineState::with_memory(initial_memory.clone());
///     s.acc = acc;
///     s
/// }).collect();
/// let results = runner.run(states);
/// ```
pub struct BatchRunner {
    model: Arc<TransformerVm>,
    max_steps: usize,
}

/// The per-run outcome returned by [`BatchRunner::run`].
#[derive(Debug)]
pub struct BatchRunResult {
    /// Index in the caller's `initial_states` slice.
    pub run_idx: usize,
    /// `Ok` if the run completed without a VM error.
    pub result: Result<ExecutionResult>,
    /// Final [`MachineState`] — present even when `result` is `Err`.
    pub final_state: MachineState,
}

// ── BatchRunner ───────────────────────────────────────────────────────────────

impl BatchRunner {
    /// Create a batch runner for `model`, limiting each run to `max_steps`.
    pub fn new(model: Arc<TransformerVm>, max_steps: usize) -> Self {
        Self { model, max_steps }
    }

    /// Execute the program for every initial state, returning results in input
    /// order. Panics if `rayon` cannot spawn threads (should not happen in
    /// practice).
    pub fn run(&self, initial_states: Vec<MachineState>) -> Vec<BatchRunResult> {
        initial_states
            .into_par_iter()
            .enumerate()
            .map(
                |(run_idx, initial_state)| match self.run_one(initial_state.clone()) {
                    Ok((exec_result, final_state)) => BatchRunResult {
                        run_idx,
                        result: Ok(exec_result),
                        final_state,
                    },
                    Err(err) => BatchRunResult {
                        run_idx,
                        result: Err(err),
                        final_state: initial_state,
                    },
                },
            )
            .collect()
    }

    fn run_one(&self, initial_state: MachineState) -> Result<(ExecutionResult, MachineState)> {
        // Clone the Arc-wrapped model (cheap) and create an independent runtime.
        let mut runtime =
            ExecutionRuntime::new_with_state((*self.model).clone(), initial_state, self.max_steps);
        let result = runtime.run()?;
        Ok((result, runtime.state().clone()))
    }
}
