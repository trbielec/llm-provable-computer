use llm_provable_computer::assembly::parse_program;
use llm_provable_computer::candidate_eval::{Candidate, CandidateEvaluator, SelectionStrategy};
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::proposal_receipt::{DecodeMetadata, ProposalReceipt};
use llm_provable_computer::state::MachineState;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn default_evaluator(strategy: SelectionStrategy) -> CandidateEvaluator {
    CandidateEvaluator::new(TransformerVmConfig::default(), 64, strategy)
}

fn halting_candidate(src: &str) -> Candidate {
    let program = parse_program(src).expect("valid program");
    Candidate::new(program)
}

fn make_receipt() -> ProposalReceipt {
    ProposalReceipt::new(
        b"prompt",
        "model",
        b"output",
        DecodeMetadata {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 10,
            seed: None,
        },
        None,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. empty candidates returns error
#[test]
fn empty_candidates_returns_error() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let result = evaluator.evaluate(vec![], state);
    assert!(
        result.is_err(),
        "evaluate with no candidates must return Err"
    );
}

/// 2. single candidate is always selected regardless of strategy
#[test]
fn single_candidate_is_always_selected() {
    let state = MachineState::default();

    for strategy in [
        SelectionStrategy::HighestAccumulator,
        SelectionStrategy::FirstSuccessful,
    ] {
        let evaluator = default_evaluator(strategy);
        let candidates = vec![halting_candidate("LOADI 5\nHALT")];
        let report = evaluator
            .evaluate(candidates, state.clone())
            .expect("evaluation must succeed");
        assert_eq!(
            report.selected_idx, 0,
            "single candidate must always be selected (strategy={strategy:?})"
        );
    }
}

/// 3. HighestAccumulator selects the candidate with the highest ACC
#[test]
fn highest_accumulator_selects_max_acc() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let candidates = vec![
        halting_candidate("LOADI 5\nHALT"),
        halting_candidate("LOADI 10\nHALT"),
        halting_candidate("LOADI 1\nHALT"),
    ];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    assert_eq!(
        report.selected_idx, 1,
        "candidate at index 1 (ACC=10) should be selected"
    );
}

/// 4. FirstSuccessful selects index zero when all candidates succeed
#[test]
fn first_successful_selects_index_zero_when_all_succeed() {
    let evaluator = default_evaluator(SelectionStrategy::FirstSuccessful);
    let state = MachineState::default();
    let candidates = vec![
        halting_candidate("LOADI 5\nHALT"),
        halting_candidate("LOADI 10\nHALT"),
        halting_candidate("LOADI 1\nHALT"),
    ];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    assert_eq!(
        report.selected_idx, 0,
        "first successful candidate (index 0) should be selected"
    );
}

/// 5. report contains all outcomes
#[test]
fn report_contains_all_outcomes() {
    let k = 4;
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let candidates = (0..k)
        .map(|_| halting_candidate("LOADI 7\nHALT"))
        .collect::<Vec<_>>();
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    assert_eq!(
        report.outcomes.len(),
        k,
        "outcomes vector length must equal number of candidates"
    );
    assert_eq!(
        report.candidate_count, k,
        "candidate_count must equal number of candidates"
    );
}

/// 6. outcome.succeeded and outcome.halted are true for a halting program
#[test]
fn outcome_succeeded_is_true_for_halting_program() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let candidates = vec![halting_candidate("LOADI 42\nHALT")];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    let outcome = report.selected();
    assert!(
        outcome.succeeded,
        "a halting program should produce succeeded=true"
    );
    assert!(
        outcome.halted,
        "a halting program should produce halted=true"
    );
}

/// 7. EvaluationReport serializes to JSON and round-trips correctly
#[test]
fn report_serializes_to_json() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let candidates = vec![
        halting_candidate("LOADI 3\nHALT"),
        halting_candidate("LOADI 7\nHALT"),
    ];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");

    let json = serde_json::to_string(&report).expect("serialization must succeed");
    let deserialized: llm_provable_computer::candidate_eval::EvaluationReport =
        serde_json::from_str(&json).expect("deserialization must succeed");

    assert_eq!(
        deserialized.selected_idx, report.selected_idx,
        "selected_idx must survive JSON round-trip"
    );
    assert_eq!(
        deserialized.candidate_count, report.candidate_count,
        "candidate_count must survive JSON round-trip"
    );
}

/// 8. Candidate::with_receipt preserves the receipt in the outcome
#[test]
fn candidate_with_receipt_preserves_receipt_in_outcome() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let program = parse_program("LOADI 1\nHALT").expect("valid program");
    let receipt = make_receipt();
    let candidates = vec![Candidate::with_receipt(program, receipt)];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    let outcome = report.selected();
    assert!(
        outcome.receipt.is_some(),
        "receipt attached to candidate must appear in outcome"
    );
}

/// 9. Candidate::new (no receipt) yields outcome.receipt == None
#[test]
fn candidate_without_receipt_has_none_in_outcome() {
    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);
    let state = MachineState::default();
    let candidates = vec![halting_candidate("LOADI 1\nHALT")];
    let report = evaluator
        .evaluate(candidates, state)
        .expect("evaluation must succeed");
    let outcome = report.selected();
    assert!(
        outcome.receipt.is_none(),
        "candidate without a receipt must yield outcome.receipt == None"
    );
}

/// 10. Evaluation is deterministic: same inputs produce the same selected_idx
#[test]
fn evaluation_is_deterministic() {
    let state = MachineState::default();
    let build_candidates = || {
        vec![
            halting_candidate("LOADI 5\nHALT"),
            halting_candidate("LOADI 10\nHALT"),
            halting_candidate("LOADI 1\nHALT"),
        ]
    };

    let evaluator = default_evaluator(SelectionStrategy::HighestAccumulator);

    let report_a = evaluator
        .evaluate(build_candidates(), state.clone())
        .expect("first evaluation must succeed");
    let report_b = evaluator
        .evaluate(build_candidates(), state.clone())
        .expect("second evaluation must succeed");

    assert_eq!(
        report_a.selected_idx, report_b.selected_idx,
        "repeated evaluations with identical inputs must select the same candidate"
    );
}
