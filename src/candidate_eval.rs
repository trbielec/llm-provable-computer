//! Evaluate many candidate programs against a shared initial state in parallel,
//! then select a winner deterministically.
//!
//! This is orthogonal to [`crate::batch::BatchRunner`], which runs one program
//! against many initial states. [`CandidateEvaluator`] runs many programs
//! against one initial state.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::TransformerVmConfig;
use crate::error::{Result, VmError};
use crate::instruction::Program;
use crate::model::TransformerVm;
use crate::proposal_receipt::ProposalReceipt;
use crate::runtime::ExecutionRuntime;
use crate::state::MachineState;

// ---------------------------------------------------------------------------
// Candidate
// ---------------------------------------------------------------------------

/// A program together with an optional upstream [`ProposalReceipt`] that
/// attests to its provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub program: Program,
    pub receipt: Option<ProposalReceipt>,
}

impl Candidate {
    /// Create a candidate with no receipt attached.
    pub fn new(program: Program) -> Self {
        Self {
            program,
            receipt: None,
        }
    }

    /// Create a candidate with a pre-existing receipt.
    pub fn with_receipt(program: Program, receipt: ProposalReceipt) -> Self {
        Self {
            program,
            receipt: Some(receipt),
        }
    }
}

// ---------------------------------------------------------------------------
// SelectionStrategy
// ---------------------------------------------------------------------------

/// How the winner is chosen from the evaluated outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Pick the candidate whose final `acc` value is numerically greatest.
    ///
    /// Ties are broken by lower `candidate_idx`. If no candidate succeeded,
    /// falls back to index 0.
    HighestAccumulator,
    /// Pick the first candidate (lowest index) that completed without a VM
    /// error.
    ///
    /// If no candidate succeeded, falls back to index 0.
    FirstSuccessful,
}

// ---------------------------------------------------------------------------
// CandidateOutcome
// ---------------------------------------------------------------------------

/// The per-candidate result produced by [`CandidateEvaluator::evaluate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateOutcome {
    /// Position of this candidate in the original slice.
    pub candidate_idx: usize,
    /// Value of the accumulator register at the end of the run.
    pub final_acc: i32,
    /// Whether the VM reached a `HALT` instruction.
    pub halted: bool,
    /// `true` iff the run returned `Ok` **and** the VM halted.
    pub succeeded: bool,
    /// The receipt carried by this candidate, if any.
    pub receipt: Option<ProposalReceipt>,
}

// ---------------------------------------------------------------------------
// EvaluationReport
// ---------------------------------------------------------------------------

/// Summary returned by [`CandidateEvaluator::evaluate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// One entry per candidate, in original order.
    pub outcomes: Vec<CandidateOutcome>,
    /// Index into `outcomes` that was selected as the winner.
    pub selected_idx: usize,
    /// Strategy that was used to pick the winner.
    pub strategy: SelectionStrategy,
    /// Total number of candidates evaluated.
    pub candidate_count: usize,
}

impl EvaluationReport {
    /// Return a reference to the winning [`CandidateOutcome`].
    pub fn selected(&self) -> &CandidateOutcome {
        &self.outcomes[self.selected_idx]
    }
}

// ---------------------------------------------------------------------------
// CandidateEvaluator
// ---------------------------------------------------------------------------

/// Runs many candidate programs against a single initial [`MachineState`] in
/// parallel, then picks a winner deterministically according to a
/// [`SelectionStrategy`].
pub struct CandidateEvaluator {
    config: TransformerVmConfig,
    max_steps: usize,
    strategy: SelectionStrategy,
}

impl CandidateEvaluator {
    /// Create a new evaluator.
    ///
    /// - `config` is cloned for each parallel run.
    /// - `max_steps` caps execution per candidate.
    /// - `strategy` determines how the winner is selected.
    pub fn new(config: TransformerVmConfig, max_steps: usize, strategy: SelectionStrategy) -> Self {
        Self {
            config,
            max_steps,
            strategy,
        }
    }

    /// Run all candidates in parallel from `initial_state`, then select a
    /// winner.
    ///
    /// Returns `Err` only if `candidates` is empty.
    pub fn evaluate(
        &self,
        candidates: Vec<Candidate>,
        initial_state: MachineState,
    ) -> Result<EvaluationReport> {
        if candidates.is_empty() {
            return Err(VmError::InvalidConfig(
                "candidates must not be empty".into(),
            ));
        }

        let candidate_count = candidates.len();

        let outcomes: Vec<CandidateOutcome> = candidates
            .into_par_iter()
            .enumerate()
            .map(|(idx, candidate)| {
                // TransformerVm::new returns Result; the config is caller-
                // controlled so construction errors here are configuration
                // bugs — treat them as fatal for this candidate.
                let model = TransformerVm::new(self.config.clone(), candidate.program)
                    .expect("TransformerVm construction failed with caller-supplied config");
                let mut runtime =
                    ExecutionRuntime::new_with_state(model, initial_state.clone(), self.max_steps);
                let run_result = runtime.run();
                let final_state = runtime.state().clone();
                CandidateOutcome {
                    candidate_idx: idx,
                    final_acc: final_state.acc as i32,
                    halted: final_state.halted,
                    succeeded: run_result.is_ok() && final_state.halted,
                    receipt: candidate.receipt,
                }
            })
            .collect();

        let selected_idx = select_winner(&outcomes, self.strategy);

        Ok(EvaluationReport {
            outcomes,
            selected_idx,
            strategy: self.strategy,
            candidate_count,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal selection logic
// ---------------------------------------------------------------------------

/// Deterministically pick a winner index from the outcomes slice.
fn select_winner(outcomes: &[CandidateOutcome], strategy: SelectionStrategy) -> usize {
    match strategy {
        SelectionStrategy::HighestAccumulator => {
            // Among succeeded outcomes, pick highest final_acc; break ties by
            // lowest candidate_idx (guaranteed by enumerate order + stable
            // max-finding below).
            outcomes
                .iter()
                .filter(|o| o.succeeded)
                .max_by(|a, b| {
                    a.final_acc
                        .cmp(&b.final_acc)
                        // lower idx wins ties
                        .then(b.candidate_idx.cmp(&a.candidate_idx))
                })
                .map(|o| o.candidate_idx)
                // Fall back to index 0 when no candidate succeeded.
                .unwrap_or(0)
        }
        SelectionStrategy::FirstSuccessful => {
            // Lowest candidate_idx where succeeded == true.
            outcomes
                .iter()
                .find(|o| o.succeeded)
                .map(|o| o.candidate_idx)
                // Fall back to index 0.
                .unwrap_or(0)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcomes(data: &[(bool, i32)]) -> Vec<CandidateOutcome> {
        data.iter()
            .enumerate()
            .map(|(idx, &(succeeded, final_acc))| CandidateOutcome {
                candidate_idx: idx,
                final_acc,
                halted: succeeded,
                succeeded,
                receipt: None,
            })
            .collect()
    }

    #[test]
    fn highest_accumulator_picks_max() {
        let outcomes = make_outcomes(&[(true, 10), (true, 30), (true, 20)]);
        assert_eq!(
            select_winner(&outcomes, SelectionStrategy::HighestAccumulator),
            1
        );
    }

    #[test]
    fn highest_accumulator_tie_lower_idx_wins() {
        let outcomes = make_outcomes(&[(true, 42), (true, 42)]);
        assert_eq!(
            select_winner(&outcomes, SelectionStrategy::HighestAccumulator),
            0
        );
    }

    #[test]
    fn highest_accumulator_no_success_falls_back_to_zero() {
        let outcomes = make_outcomes(&[(false, 99), (false, 1)]);
        assert_eq!(
            select_winner(&outcomes, SelectionStrategy::HighestAccumulator),
            0
        );
    }

    #[test]
    fn first_successful_picks_lowest_idx() {
        let outcomes = make_outcomes(&[(false, 0), (true, 5), (true, 10)]);
        assert_eq!(
            select_winner(&outcomes, SelectionStrategy::FirstSuccessful),
            1
        );
    }

    #[test]
    fn first_successful_no_success_falls_back_to_zero() {
        let outcomes = make_outcomes(&[(false, 0), (false, 0)]);
        assert_eq!(
            select_winner(&outcomes, SelectionStrategy::FirstSuccessful),
            0
        );
    }

    #[test]
    fn evaluation_report_selected_returns_correct_outcome() {
        let outcomes = make_outcomes(&[(true, 5), (true, 10)]);
        let report = EvaluationReport {
            selected_idx: 1,
            candidate_count: outcomes.len(),
            strategy: SelectionStrategy::HighestAccumulator,
            outcomes,
        };
        assert_eq!(report.selected().candidate_idx, 1);
        assert_eq!(report.selected().final_acc, 10);
    }
}
