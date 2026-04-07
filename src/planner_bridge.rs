//! Orchestration layer that ties together planning, admissibility filtering,
//! and candidate evaluation into a single pipeline.
//!
//! The central entry point is [`PlannerBridge::run`], which accepts any
//! [`Planner`] implementation and returns a [`PlannerBridgeResult`] containing
//! an [`EvaluationReport`] over all admitted candidates.

use serde::{Deserialize, Serialize};

use crate::assembly::parse_program;
use crate::candidate_eval::{
    Candidate, CandidateEvaluator, CandidateOutcome, EvaluationReport, SelectionStrategy,
};
use crate::config::TransformerVmConfig;
use crate::effect_root::EffectBoundary;
use crate::error::{Result, VmError};
use crate::instruction::{Instruction, Program};
use crate::proposal_receipt::{DecodeMetadata, ProposalReceipt};
use crate::state::MachineState;

// ---------------------------------------------------------------------------
// PlannerContext
// ---------------------------------------------------------------------------

/// Input to the planner — describes what to plan for.
///
/// Note: [`MachineState`] does not implement `Serialize`/`Deserialize`, so
/// `PlannerContext` derives only `Debug` and `Clone`.
#[derive(Debug, Clone)]
pub struct PlannerContext {
    pub prompt: Vec<u8>,
    pub model_id: String,
    pub initial_state: MachineState,
    pub effect_boundary: Option<EffectBoundary>,
}

impl PlannerContext {
    /// Create a new context with no effect boundary.
    pub fn new(
        prompt: impl Into<Vec<u8>>,
        model_id: impl Into<String>,
        initial_state: MachineState,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            model_id: model_id.into(),
            initial_state,
            effect_boundary: None,
        }
    }

    /// Attach an [`EffectBoundary`] to this context.
    pub fn with_effect_boundary(self, boundary: EffectBoundary) -> Self {
        Self {
            effect_boundary: Some(boundary),
            ..self
        }
    }
}

// ---------------------------------------------------------------------------
// ProposedCandidate
// ---------------------------------------------------------------------------

/// A raw proposal from the planner, before admissibility checking.
#[derive(Debug, Clone)]
pub struct ProposedCandidate {
    /// TVM assembly source.
    pub source: String,
    /// Raw planner output bytes (used for receipt hashing).
    pub output_bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Planner trait
// ---------------------------------------------------------------------------

/// The integration seam between the outside world (LLM, rule engine, stub,
/// …) and the deterministic kernel.
///
/// Implementations must be `Send + Sync` so they can be used across rayon
/// worker threads.
pub trait Planner: Send + Sync {
    /// Propose zero or more candidate programs given `context`.
    fn propose(&self, context: &PlannerContext) -> Result<Vec<ProposedCandidate>>;
}

// ---------------------------------------------------------------------------
// AdmissibilityResult
// ---------------------------------------------------------------------------

/// Outcome of checking a single raw proposal against the deterministic kernel.
#[derive(Debug, Clone)]
pub enum AdmissibilityResult {
    /// The source compiled cleanly and uses only supported instructions.
    Admissible(Program),
    /// The source was rejected; the string contains a human-readable reason.
    Rejected(String),
}

impl AdmissibilityResult {
    /// Returns `true` iff the result is [`AdmissibilityResult::Admissible`].
    pub fn is_admissible(&self) -> bool {
        matches!(self, AdmissibilityResult::Admissible(_))
    }

    /// Consume the result and return the inner [`Program`], if admissible.
    pub fn into_program(self) -> Option<Program> {
        match self {
            AdmissibilityResult::Admissible(p) => Some(p),
            AdmissibilityResult::Rejected(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AdmissibilityFilter
// ---------------------------------------------------------------------------

/// Validates that a raw assembly source compiles and only uses instructions
/// supported by the deterministic kernel.
///
/// Currently rejects:
/// - sources that fail [`parse_program`],
/// - empty programs (zero instructions), and
/// - programs that contain `CMP`/`CMPM` instructions.
pub struct AdmissibilityFilter;

impl AdmissibilityFilter {
    /// Create a new filter (stateless; allocation is trivial).
    pub fn new() -> Self {
        Self
    }

    /// Check a single source string for admissibility.
    pub fn check(&self, source: &str) -> AdmissibilityResult {
        let program = match parse_program(source) {
            Ok(p) => p,
            Err(e) => return AdmissibilityResult::Rejected(format!("parse error: {e}")),
        };

        if program.instructions().is_empty() {
            return AdmissibilityResult::Rejected("program has no instructions".into());
        }

        for instr in program.instructions() {
            match instr {
                Instruction::CmpImmediate(_) => {
                    return AdmissibilityResult::Rejected("unsupported instruction: CMP".into());
                }
                Instruction::CmpMemory(_) => {
                    return AdmissibilityResult::Rejected("unsupported instruction: CMPM".into());
                }
                _ => {}
            }
        }

        AdmissibilityResult::Admissible(program)
    }
}

impl Default for AdmissibilityFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PlannerBridgeResult
// ---------------------------------------------------------------------------

/// Result returned by [`PlannerBridge::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerBridgeResult {
    /// Total number of raw proposals from the planner.
    pub proposed_count: usize,
    /// Number of proposals that passed the admissibility filter.
    pub admitted_count: usize,
    /// Number of proposals that were rejected by the filter.
    pub rejected_count: usize,
    /// Full evaluation report from [`CandidateEvaluator`].
    pub report: EvaluationReport,
}

impl PlannerBridgeResult {
    /// Return a reference to the winning [`CandidateOutcome`].
    pub fn selected_outcome(&self) -> &CandidateOutcome {
        self.report.selected()
    }
}

// ---------------------------------------------------------------------------
// PlannerBridge
// ---------------------------------------------------------------------------

/// Orchestrates the full pipeline: planner → admissibility filter →
/// [`CandidateEvaluator`].
///
/// `PlannerBridge` is not `Clone` because it holds a [`TransformerVmConfig`]
/// by value and its [`Planner`] argument is a trait object.
pub struct PlannerBridge {
    config: TransformerVmConfig,
    max_steps: usize,
    strategy: SelectionStrategy,
    filter: AdmissibilityFilter,
}

impl PlannerBridge {
    /// Create a new bridge.
    pub fn new(config: TransformerVmConfig, max_steps: usize, strategy: SelectionStrategy) -> Self {
        Self {
            config,
            max_steps,
            strategy,
            filter: AdmissibilityFilter::new(),
        }
    }

    /// Run the full pipeline.
    ///
    /// Returns `Err` only if the planner itself returns an error or if zero
    /// candidates pass the admissibility filter.
    pub fn run(
        &self,
        planner: &dyn Planner,
        context: PlannerContext,
    ) -> Result<PlannerBridgeResult> {
        // Step 1: ask the planner for proposals.
        let proposals = planner.propose(&context)?;
        let proposed_count = proposals.len();

        // Step 2 & 3: filter and build Candidates.
        let mut admitted_count = 0usize;
        let mut rejected_count = 0usize;
        let mut candidates: Vec<Candidate> = Vec::new();

        for candidate in proposals {
            match self.filter.check(&candidate.source) {
                AdmissibilityResult::Admissible(program) => {
                    let receipt = ProposalReceipt::new(
                        &context.prompt,
                        &context.model_id,
                        &candidate.output_bytes,
                        DecodeMetadata {
                            temperature: 0.0,
                            top_p: 1.0,
                            max_tokens: 0,
                            seed: None,
                        },
                        None,
                    );
                    candidates.push(Candidate::with_receipt(program, receipt));
                    admitted_count += 1;
                }
                AdmissibilityResult::Rejected(_) => {
                    rejected_count += 1;
                }
            }
        }

        // Step 4: require at least one admitted candidate.
        if candidates.is_empty() {
            return Err(VmError::InvalidConfig(
                "no admissible candidates after filtering".into(),
            ));
        }

        // Step 5 & 6: evaluate and return.
        let evaluator = CandidateEvaluator::new(self.config.clone(), self.max_steps, self.strategy);
        let report = evaluator.evaluate(candidates, context.initial_state)?;

        Ok(PlannerBridgeResult {
            proposed_count,
            admitted_count,
            rejected_count,
            report,
        })
    }
}

// ---------------------------------------------------------------------------
// StaticPlanner
// ---------------------------------------------------------------------------

/// A test double that always proposes a fixed list of (source, output_bytes)
/// pairs.
pub struct StaticPlanner {
    proposals: Vec<(String, Vec<u8>)>,
}

impl StaticPlanner {
    /// Create a `StaticPlanner` from a list of `(source, output_bytes)` pairs.
    pub fn new(proposals: Vec<(String, Vec<u8>)>) -> Self {
        Self { proposals }
    }
}

impl Planner for StaticPlanner {
    fn propose(&self, _context: &PlannerContext) -> Result<Vec<ProposedCandidate>> {
        Ok(self
            .proposals
            .iter()
            .map(|(src, out)| ProposedCandidate {
                source: src.clone(),
                output_bytes: out.clone(),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerVmConfig;
    use crate::state::MachineState;

    fn default_config() -> TransformerVmConfig {
        TransformerVmConfig::default()
    }

    fn default_state() -> MachineState {
        MachineState::default()
    }

    // ---- AdmissibilityFilter -----------------------------------------------

    #[test]
    fn filter_accepts_valid_program() {
        let filter = AdmissibilityFilter::new();
        let result = filter.check("LOADI 1\nHALT");
        assert!(result.is_admissible());
        assert!(result.into_program().is_some());
    }

    #[test]
    fn filter_rejects_parse_error() {
        let filter = AdmissibilityFilter::new();
        let result = filter.check("NOT_AN_INSTRUCTION");
        assert!(!result.is_admissible());
    }

    #[test]
    fn filter_rejects_cmp_instruction() {
        let filter = AdmissibilityFilter::new();
        let result = filter.check("LOADI 1\nCMP 0\nHALT");
        assert!(!result.is_admissible());
        match result {
            AdmissibilityResult::Rejected(reason) => {
                assert!(reason.contains("CMP"), "expected CMP in reason: {reason}");
            }
            _ => panic!("expected Rejected"),
        }
    }

    #[test]
    fn filter_rejects_cmpm_instruction() {
        let filter = AdmissibilityFilter::new();
        let result = filter.check("LOADI 1\nCMPM 0\nHALT");
        assert!(!result.is_admissible());
    }

    // ---- StaticPlanner + PlannerBridge ------------------------------------

    #[test]
    fn bridge_runs_single_valid_program() {
        let planner = StaticPlanner::new(vec![("LOADI 42\nHALT".into(), b"output-bytes".to_vec())]);
        let bridge = PlannerBridge::new(default_config(), 64, SelectionStrategy::FirstSuccessful);
        let context = PlannerContext::new(b"prompt".to_vec(), "model-v1", default_state());
        let result = bridge.run(&planner, context).expect("bridge run failed");
        assert_eq!(result.proposed_count, 1);
        assert_eq!(result.admitted_count, 1);
        assert_eq!(result.rejected_count, 0);
        assert!(result.selected_outcome().receipt.is_some());
    }

    #[test]
    fn bridge_filters_rejected_candidates() {
        let planner = StaticPlanner::new(vec![
            ("LOADI 1\nCMP 0\nHALT".into(), b"bad".to_vec()),
            ("LOADI 7\nHALT".into(), b"good".to_vec()),
        ]);
        let bridge =
            PlannerBridge::new(default_config(), 64, SelectionStrategy::HighestAccumulator);
        let context = PlannerContext::new(b"p".to_vec(), "m", default_state());
        let result = bridge.run(&planner, context).expect("bridge run failed");
        assert_eq!(result.proposed_count, 2);
        assert_eq!(result.admitted_count, 1);
        assert_eq!(result.rejected_count, 1);
    }

    #[test]
    fn bridge_errors_when_all_rejected() {
        let planner = StaticPlanner::new(vec![("LOADI 1\nCMP 0\nHALT".into(), b"bad".to_vec())]);
        let bridge = PlannerBridge::new(default_config(), 64, SelectionStrategy::FirstSuccessful);
        let context = PlannerContext::new(b"p".to_vec(), "m", default_state());
        let err = bridge.run(&planner, context).unwrap_err();
        assert!(
            err.to_string().contains("no admissible"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn planner_context_with_effect_boundary() {
        use crate::effect_root::{EffectBoundary, EffectLog, StateRoot};
        let pre = StateRoot::empty();
        let log = EffectLog::new();
        let boundary = EffectBoundary::new(pre, log);
        let ctx =
            PlannerContext::new(b"p".to_vec(), "m", default_state()).with_effect_boundary(boundary);
        assert!(ctx.effect_boundary.is_some());
    }

    #[test]
    fn planner_bridge_result_selected_outcome_delegates_to_report() {
        let planner = StaticPlanner::new(vec![("LOADI 1\nHALT".into(), b"bytes".to_vec())]);
        let bridge = PlannerBridge::new(default_config(), 64, SelectionStrategy::FirstSuccessful);
        let context = PlannerContext::new(b"p".to_vec(), "m", default_state());
        let result = bridge.run(&planner, context).unwrap();
        // selected_outcome() must equal report.selected()
        assert_eq!(
            result.selected_outcome().candidate_idx,
            result.report.selected().candidate_idx,
        );
    }
}
