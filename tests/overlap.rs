//! Integration tests for the `overlap` module.
//!
//! All tests are self-contained; no shared fixtures. The implementation under
//! test lives in `src/overlap.rs` and is re-exported from the crate root.

use llm_provable_computer::candidate_eval::SelectionStrategy;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::overlap::{FallbackProgram, OverlapOrchestrator, OverlapStrategy};
use llm_provable_computer::planner_bridge::{PlannerContext, StaticPlanner};
use llm_provable_computer::state::MachineState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_context() -> PlannerContext {
    PlannerContext::new(
        b"test-prompt".to_vec(),
        "test-model".to_string(),
        MachineState::default(),
    )
}

fn make_fallback() -> FallbackProgram {
    FallbackProgram::from_source("LOADI 0\nHALT").expect("fallback program must parse")
}

fn make_orchestrator(strategy: OverlapStrategy) -> OverlapOrchestrator {
    OverlapOrchestrator::new(
        TransformerVmConfig::default(),
        64,
        SelectionStrategy::HighestAccumulator,
        make_fallback(),
    )
    .with_strategy(strategy)
}

fn single_candidate_planner() -> StaticPlanner {
    StaticPlanner::new(vec![("LOADI 7\nHALT".to_string(), b"out".to_vec())])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. Sequential mode produces a report with the expected field values.
#[test]
fn sequential_mode_produces_report() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = single_candidate_planner();
    let report = orchestrator
        .run(&planner, make_context())
        .expect("sequential run must succeed");
    assert_eq!(
        report.strategy,
        OverlapStrategy::Sequential,
        "report.strategy must be Sequential"
    );
    assert!(
        report.used_planner_output,
        "report.used_planner_output must be true when planner succeeds"
    );
    assert_eq!(
        report.timings.fallback_duration_us, 0,
        "fallback_duration_us must be 0 in Sequential mode"
    );
}

/// 2. Overlapped mode produces a report with fallback timing populated.
#[test]
fn overlapped_mode_produces_report() {
    let orchestrator = make_orchestrator(OverlapStrategy::Overlapped);
    let planner = single_candidate_planner();
    let report = orchestrator
        .run(&planner, make_context())
        .expect("overlapped run must succeed");
    assert_eq!(
        report.strategy,
        OverlapStrategy::Overlapped,
        "report.strategy must be Overlapped"
    );
    assert!(
        report.timings.fallback_duration_us > 0,
        "fallback_duration_us must be non-zero in Overlapped mode"
    );
}

/// 3. Sequential HighestAccumulator mode selects the candidate with the highest ACC.
#[test]
fn sequential_final_acc_matches_highest() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = StaticPlanner::new(vec![
        ("LOADI 3\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 9\nHALT".to_string(), b"b".to_vec()),
    ]);
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed");
    assert_eq!(
        report.final_acc, 9,
        "HighestAccumulator must select ACC=9 in Sequential mode"
    );
}

/// 4. Overlapped HighestAccumulator mode also selects the candidate with the highest ACC.
#[test]
fn overlapped_final_acc_matches_highest() {
    let orchestrator = make_orchestrator(OverlapStrategy::Overlapped);
    let planner = StaticPlanner::new(vec![
        ("LOADI 3\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 9\nHALT".to_string(), b"b".to_vec()),
    ]);
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed");
    assert_eq!(
        report.final_acc, 9,
        "HighestAccumulator must select ACC=9 in Overlapped mode"
    );
}

/// 5. admitted_count reflects only valid candidates (invalid sources are rejected).
#[test]
fn admitted_count_reflects_valid_candidates() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = StaticPlanner::new(vec![
        ("LOADI 1\nHALT".to_string(), b"v1".to_vec()),
        ("NOT VALID SOURCE !!!".to_string(), b"bad".to_vec()),
        ("LOADI 2\nHALT".to_string(), b"v2".to_vec()),
    ]);
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed with 2 valid candidates");
    assert_eq!(
        report.admitted_count, 2,
        "admitted_count must equal the number of valid candidates (2)"
    );
}

/// 6. OverlapReport serializes to JSON and round-trips correctly.
#[test]
fn report_serializes_to_json() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = single_candidate_planner();
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed");

    let json = serde_json::to_string(&report).expect("OverlapReport must serialize to JSON");
    let deserialized: llm_provable_computer::overlap::OverlapReport =
        serde_json::from_str(&json).expect("JSON must deserialize back to OverlapReport");

    assert_eq!(
        deserialized.strategy, report.strategy,
        "strategy must survive JSON round-trip"
    );
    assert_eq!(
        deserialized.final_acc, report.final_acc,
        "final_acc must survive JSON round-trip"
    );
}

/// 7. overlap_savings_us is zero in Sequential mode.
#[test]
fn overlap_savings_is_zero_in_sequential_mode() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = single_candidate_planner();
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed");
    assert_eq!(
        report.timings.overlap_savings_us, 0,
        "overlap_savings_us must be 0 in Sequential mode"
    );
}

/// 8. pipeline_duration_us is always greater than zero (wall time is always positive).
#[test]
fn pipeline_duration_is_nonzero() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = single_candidate_planner();
    let report = orchestrator
        .run(&planner, make_context())
        .expect("run must succeed");
    assert!(
        report.timings.pipeline_duration_us > 0,
        "pipeline_duration_us must be greater than zero"
    );
}

/// 9. When every candidate source is invalid, run() returns Err.
#[test]
fn all_rejected_returns_error() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = StaticPlanner::new(vec![
        ("NOT VALID SOURCE !!!".to_string(), b"bad1".to_vec()),
        ("ALSO INVALID ???".to_string(), b"bad2".to_vec()),
    ]);
    let result = orchestrator.run(&planner, make_context());
    assert!(
        result.is_err(),
        "run must return Err when zero candidates are admitted"
    );
}

/// 10. Running the orchestrator twice with identical inputs produces identical results.
#[test]
fn determinism_across_runs() {
    let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
    let planner = StaticPlanner::new(vec![
        ("LOADI 5\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 10\nHALT".to_string(), b"b".to_vec()),
        ("LOADI 1\nHALT".to_string(), b"c".to_vec()),
    ]);

    let report_a = orchestrator
        .run(&planner, make_context())
        .expect("first run must succeed");
    let report_b = orchestrator
        .run(&planner, make_context())
        .expect("second run must succeed");

    assert_eq!(
        report_a.final_acc, report_b.final_acc,
        "final_acc must be identical across repeated runs with the same inputs"
    );
    assert_eq!(
        report_a.admitted_count, report_b.admitted_count,
        "admitted_count must be identical across repeated runs with the same inputs"
    );
}
