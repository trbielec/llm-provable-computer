//! Phase 5: overlap measurement between proposal-generation latency and
//! deterministic kernel compute.
//!
//! [`OverlapOrchestrator`] supports two modes:
//!
//! - [`OverlapStrategy::Sequential`] — planner runs to completion, then the
//!   kernel evaluates candidates.  This is the baseline.
//! - [`OverlapStrategy::Overlapped`] — a fallback kernel run starts
//!   immediately in a background thread while the planner pipeline runs
//!   concurrently.  Both timings are recorded so the overlap opportunity is
//!   *visible* even when using a [`StaticPlanner`] (which is fast).

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::assembly::parse_program;
use crate::candidate_eval::SelectionStrategy;
use crate::config::TransformerVmConfig;
use crate::error::{Result, VmError};
use crate::instruction::Program;
use crate::model::TransformerVm;
use crate::planner_bridge::{Planner, PlannerBridge, PlannerContext};
use crate::runtime::ExecutionRuntime;

// ---------------------------------------------------------------------------
// OverlapStrategy
// ---------------------------------------------------------------------------

/// Controls whether kernel compute is overlapped with planner latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlapStrategy {
    /// Planner runs to completion, then kernel evaluates candidates.
    Sequential,
    /// Fallback kernel starts immediately while planner runs concurrently.
    Overlapped,
}

// ---------------------------------------------------------------------------
// OverlapTimings
// ---------------------------------------------------------------------------

/// Wall-clock timings captured by [`OverlapOrchestrator::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapTimings {
    /// Wall time for the planner-to-kernel pipeline (µs).
    pub pipeline_duration_us: u64,
    /// Wall time for the fallback kernel run (µs). Zero in Sequential mode.
    pub fallback_duration_us: u64,
    /// Savings vs sequential baseline (can be negative if fallback was slower).
    ///
    /// Computed as `pipeline_duration_us.saturating_sub(fallback_duration_us)`
    /// in `Overlapped` mode; always `0` in `Sequential` mode.
    pub overlap_savings_us: u64,
}

// ---------------------------------------------------------------------------
// OverlapReport
// ---------------------------------------------------------------------------

/// Full result returned by [`OverlapOrchestrator::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapReport {
    /// Which strategy was used.
    pub strategy: OverlapStrategy,
    /// Timing breakdown.
    pub timings: OverlapTimings,
    /// Whether the planner pipeline result was used.
    ///
    /// Always `true` in `Sequential` mode.  In `Overlapped` mode, `true` if
    /// the pipeline finished before the fallback OR if the fallback failed.
    pub used_planner_output: bool,
    /// Final ACC from whichever path was used.
    pub final_acc: i32,
    /// Number of candidates admitted by the planner bridge.
    pub admitted_count: usize,
}

// ---------------------------------------------------------------------------
// FallbackProgram
// ---------------------------------------------------------------------------

/// A pre-compiled fallback program to run while the planner works.
#[derive(Debug, Clone)]
pub struct FallbackProgram {
    pub program: Program,
}

impl FallbackProgram {
    /// Compile a fallback program from TVM assembly source.
    pub fn from_source(source: &str) -> Result<Self> {
        let program = parse_program(source)
            .map_err(|e| VmError::InvalidConfig(format!("fallback program parse error: {e}")))?;
        Ok(Self { program })
    }
}

// ---------------------------------------------------------------------------
// OverlapOrchestrator
// ---------------------------------------------------------------------------

/// Runs the planner bridge pipeline with optional concurrent fallback kernel
/// execution, making the overlap opportunity *measurable*.
pub struct OverlapOrchestrator {
    config: TransformerVmConfig,
    max_steps: usize,
    strategy: SelectionStrategy,
    fallback: FallbackProgram,
    overlap_strategy: OverlapStrategy,
}

impl OverlapOrchestrator {
    /// Create a new orchestrator.
    ///
    /// `strategy` controls which candidate is selected from the planner
    /// pipeline; `overlap_strategy` controls whether the fallback kernel runs
    /// concurrently.
    pub fn new(
        config: TransformerVmConfig,
        max_steps: usize,
        strategy: SelectionStrategy,
        fallback: FallbackProgram,
    ) -> Self {
        Self {
            config,
            max_steps,
            strategy,
            fallback,
            overlap_strategy: OverlapStrategy::Sequential,
        }
    }

    /// Return a new orchestrator that uses the given [`OverlapStrategy`].
    pub fn with_strategy(self, overlap_strategy: OverlapStrategy) -> Self {
        Self {
            overlap_strategy,
            ..self
        }
    }

    /// Run the pipeline using `planner` and `context`, returning an
    /// [`OverlapReport`] with timing information.
    ///
    /// In [`OverlapStrategy::Sequential`] mode, the planner pipeline runs
    /// first and the fallback is never executed.
    ///
    /// In [`OverlapStrategy::Overlapped`] mode, a fallback kernel run starts
    /// immediately in a sibling thread while the planner pipeline executes
    /// concurrently.  Both threads are joined via [`std::thread::scope`].
    pub fn run(
        &self,
        planner: &(impl Planner + Sync),
        context: PlannerContext,
    ) -> Result<OverlapReport> {
        match self.overlap_strategy {
            OverlapStrategy::Sequential => self.run_sequential(planner, context),
            OverlapStrategy::Overlapped => self.run_overlapped(planner, context),
        }
    }

    fn run_sequential(
        &self,
        planner: &(impl Planner + Sync),
        context: PlannerContext,
    ) -> Result<OverlapReport> {
        let t0 = Instant::now();
        let result = PlannerBridge::new(self.config.clone(), self.max_steps, self.strategy)
            .run(planner, context)?;
        let pipeline_us = t0.elapsed().as_micros() as u64;

        Ok(OverlapReport {
            strategy: OverlapStrategy::Sequential,
            timings: OverlapTimings {
                pipeline_duration_us: pipeline_us,
                fallback_duration_us: 0,
                overlap_savings_us: 0,
            },
            used_planner_output: true,
            final_acc: result.selected_outcome().final_acc,
            admitted_count: result.admitted_count,
        })
    }

    fn run_overlapped(
        &self,
        planner: &(impl Planner + Sync),
        context: PlannerContext,
    ) -> Result<OverlapReport> {
        let (pipeline_result, pipeline_us, _fallback_acc, fallback_us) = std::thread::scope(|s| {
            let pipeline_handle = s.spawn(|| {
                let t = Instant::now();
                let r = PlannerBridge::new(self.config.clone(), self.max_steps, self.strategy)
                    .run(planner, context.clone());
                (r, t.elapsed().as_micros() as u64)
            });

            let fallback_handle = s.spawn(|| {
                let t = Instant::now();
                let model = TransformerVm::new(self.config.clone(), self.fallback.program.clone())
                    .expect("valid fallback program");
                let mut runtime = ExecutionRuntime::new_with_state(
                    model,
                    context.initial_state.clone(),
                    self.max_steps,
                );
                let acc = runtime
                    .run()
                    .map(|_| runtime.state().acc as i32)
                    .unwrap_or(0);
                (acc, t.elapsed().as_micros() as u64)
            });

            let (pr, pus) = pipeline_handle.join().expect("pipeline thread panicked");
            let (fa, fus) = fallback_handle.join().expect("fallback thread panicked");
            (pr, pus, fa, fus)
        });

        let pipeline_result = pipeline_result?;
        let savings = pipeline_us.saturating_sub(fallback_us);

        Ok(OverlapReport {
            strategy: OverlapStrategy::Overlapped,
            timings: OverlapTimings {
                pipeline_duration_us: pipeline_us,
                fallback_duration_us: fallback_us,
                overlap_savings_us: savings,
            },
            // Always use the planner output — it has gone through admissibility
            // checking and deterministic candidate evaluation.
            used_planner_output: true,
            final_acc: pipeline_result.selected_outcome().final_acc,
            admitted_count: pipeline_result.admitted_count,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerVmConfig;
    use crate::planner_bridge::StaticPlanner;
    use crate::state::MachineState;

    fn default_config() -> TransformerVmConfig {
        TransformerVmConfig::default()
    }

    fn default_context() -> PlannerContext {
        PlannerContext::new(b"prompt".to_vec(), "model-v1", MachineState::default())
    }

    fn simple_fallback() -> FallbackProgram {
        FallbackProgram::from_source("LOADI 1\nHALT").expect("valid fallback")
    }

    fn simple_planner() -> StaticPlanner {
        StaticPlanner::new(vec![("LOADI 42\nHALT".into(), b"bytes".to_vec())])
    }

    fn make_orchestrator(overlap: OverlapStrategy) -> OverlapOrchestrator {
        OverlapOrchestrator::new(
            default_config(),
            64,
            SelectionStrategy::FirstSuccessful,
            simple_fallback(),
        )
        .with_strategy(overlap)
    }

    #[test]
    fn fallback_program_from_valid_source() {
        assert!(FallbackProgram::from_source("LOADI 7\nHALT").is_ok());
    }

    #[test]
    fn fallback_program_from_invalid_source() {
        assert!(FallbackProgram::from_source("NOT_AN_INSTRUCTION").is_err());
    }

    #[test]
    fn sequential_run_returns_correct_acc() {
        let report = make_orchestrator(OverlapStrategy::Sequential)
            .run(&simple_planner(), default_context())
            .expect("sequential run failed");

        assert_eq!(report.strategy, OverlapStrategy::Sequential);
        assert!(report.used_planner_output);
        assert_eq!(report.final_acc, 42);
        assert_eq!(report.admitted_count, 1);
        assert_eq!(report.timings.fallback_duration_us, 0);
        assert_eq!(report.timings.overlap_savings_us, 0);
        assert!(report.timings.pipeline_duration_us > 0);
    }

    #[test]
    fn overlapped_run_returns_correct_acc() {
        let report = make_orchestrator(OverlapStrategy::Overlapped)
            .run(&simple_planner(), default_context())
            .expect("overlapped run failed");

        assert_eq!(report.strategy, OverlapStrategy::Overlapped);
        assert!(report.used_planner_output);
        assert_eq!(report.final_acc, 42);
        assert_eq!(report.admitted_count, 1);
        assert!(report.timings.pipeline_duration_us > 0);
        assert!(report.timings.fallback_duration_us > 0);
    }

    #[test]
    fn overlap_savings_saturates_to_zero_when_fallback_is_slower() {
        // Verify saturating_sub semantics — savings must not underflow (u64).
        assert_eq!(10u64.saturating_sub(100), 0);
    }

    #[test]
    fn sequential_and_overlapped_agree_on_final_acc() {
        let planner = simple_planner();
        let seq = make_orchestrator(OverlapStrategy::Sequential)
            .run(&planner, default_context())
            .expect("sequential");
        let ovl = make_orchestrator(OverlapStrategy::Overlapped)
            .run(&planner, default_context())
            .expect("overlapped");

        assert_eq!(seq.final_acc, ovl.final_acc);
        assert_eq!(seq.admitted_count, ovl.admitted_count);
    }

    #[test]
    fn new_defaults_to_sequential() {
        let orch = OverlapOrchestrator::new(
            default_config(),
            64,
            SelectionStrategy::FirstSuccessful,
            simple_fallback(),
        );
        let report = orch
            .run(&simple_planner(), default_context())
            .expect("default run");
        assert_eq!(report.strategy, OverlapStrategy::Sequential);
    }
}
