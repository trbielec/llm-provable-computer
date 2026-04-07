//! Integration tests for the `planner_bridge` module.
//!
//! All tests are self-contained; no shared fixtures. The implementation under
//! test lives in `src/planner_bridge.rs` and is re-exported from the crate root.

use llm_provable_computer::assembly::parse_program;
use llm_provable_computer::candidate_eval::SelectionStrategy;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::planner_bridge::{
    AdmissibilityFilter, Planner, PlannerBridge, PlannerContext, StaticPlanner,
};
use llm_provable_computer::state::MachineState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_bridge(strategy: SelectionStrategy) -> PlannerBridge {
    PlannerBridge::new(TransformerVmConfig::default(), 64, strategy)
}

fn make_context() -> PlannerContext {
    PlannerContext::new(
        b"test".to_vec(),
        "test-model".to_string(),
        MachineState::default(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. AdmissibilityFilter accepts a valid program.
#[test]
fn admissibility_filter_accepts_valid_program() {
    let filter = AdmissibilityFilter::new();
    let result = filter.check("LOADI 5\nHALT");
    assert!(
        result.is_admissible(),
        "a well-formed halting program must be Admissible"
    );
}

/// 2. AdmissibilityFilter rejects unparseable source.
#[test]
fn admissibility_filter_rejects_invalid_syntax() {
    let filter = AdmissibilityFilter::new();
    let result = filter.check("NOT VALID SOURCE !!!");
    assert!(
        !result.is_admissible(),
        "source that cannot be parsed must be Rejected"
    );
}

/// 3. AdmissibilityFilter rejects a program containing a CMP instruction.
///
/// Whether parse_program itself rejects CMP or the filter catches it, the
/// overall admissibility result must be Rejected.
#[test]
fn admissibility_filter_rejects_cmp_instruction() {
    let filter = AdmissibilityFilter::new();
    let result = filter.check("LOADI 5\nCMP 0\nHALT");
    assert!(
        !result.is_admissible(),
        "a program containing CMP must not be admissible"
    );
}

/// 4. StaticPlanner proposes exactly the candidates it was constructed with.
#[test]
fn static_planner_proposes_given_candidates() {
    let proposals = vec![
        ("LOADI 1\nHALT".to_string(), b"out1".to_vec()),
        ("LOADI 2\nHALT".to_string(), b"out2".to_vec()),
        ("LOADI 3\nHALT".to_string(), b"out3".to_vec()),
    ];
    let planner = StaticPlanner::new(proposals);
    let context = make_context();
    let candidates = planner
        .propose(&context)
        .expect("StaticPlanner::propose must not fail");
    assert_eq!(
        candidates.len(),
        3,
        "StaticPlanner must propose exactly 3 candidates"
    );
}

/// 5. Bridge runs a single valid candidate successfully.
#[test]
fn bridge_runs_single_valid_candidate() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![("LOADI 7\nHALT".to_string(), b"out".to_vec())]);
    let result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed with one valid candidate");
    assert_eq!(
        result.admitted_count, 1,
        "one valid candidate must be admitted"
    );
    assert_eq!(
        result.report.selected_idx, 0,
        "only candidate must be at index 0"
    );
}

/// 6. HighestAccumulator strategy selects the candidate with the highest ACC.
#[test]
fn bridge_selects_highest_acc_from_multiple_candidates() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![
        ("LOADI 5\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 20\nHALT".to_string(), b"b".to_vec()),
        ("LOADI 3\nHALT".to_string(), b"c".to_vec()),
    ]);
    let result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed");
    let outcome = result.selected_outcome();
    assert_eq!(
        outcome.final_acc, 20,
        "HighestAccumulator must select the candidate with ACC=20"
    );
}

/// 7. Bridge counts proposed, admitted, and rejected candidates correctly.
#[test]
fn bridge_counts_proposed_and_admitted_correctly() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    // Two valid, one invalid.
    let planner = StaticPlanner::new(vec![
        ("LOADI 1\nHALT".to_string(), b"v1".to_vec()),
        ("NOT VALID SOURCE !!!".to_string(), b"bad".to_vec()),
        ("LOADI 2\nHALT".to_string(), b"v2".to_vec()),
    ]);
    let result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed with at least one valid candidate");
    assert_eq!(
        result.proposed_count, 3,
        "proposed_count must equal total proposals"
    );
    assert!(
        result.admitted_count < result.proposed_count,
        "admitted_count must be less than proposed_count when some sources are invalid"
    );
    assert_eq!(
        result.rejected_count,
        result.proposed_count - result.admitted_count,
        "rejected_count must equal proposed_count minus admitted_count"
    );
}

/// 8. Bridge returns Err when every candidate is rejected by the admissibility filter.
#[test]
fn bridge_returns_error_when_all_rejected() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![
        ("NOT VALID SOURCE !!!".to_string(), b"bad1".to_vec()),
        ("ALSO INVALID ???".to_string(), b"bad2".to_vec()),
    ]);
    let result = bridge.run(&planner, make_context());
    assert!(
        result.is_err(),
        "bridge must return Err when zero candidates are admitted"
    );
}

/// 9. selected_outcome() references the same candidate as report.selected_idx.
#[test]
fn bridge_result_selected_outcome_matches_report() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![
        ("LOADI 10\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 30\nHALT".to_string(), b"b".to_vec()),
    ]);
    let result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed");
    assert_eq!(
        result.selected_outcome().candidate_idx,
        result.report.selected_idx,
        "selected_outcome().candidate_idx must equal report.selected_idx"
    );
}

/// 10. PlannerBridgeResult serializes to JSON and round-trips correctly.
#[test]
fn bridge_result_serializes_to_json() {
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![
        ("LOADI 5\nHALT".to_string(), b"x".to_vec()),
        ("LOADI 15\nHALT".to_string(), b"y".to_vec()),
    ]);
    let result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed");

    let json = serde_json::to_string(&result).expect("PlannerBridgeResult must serialize to JSON");
    let deserialized: llm_provable_computer::planner_bridge::PlannerBridgeResult =
        serde_json::from_str(&json).expect("JSON must deserialize back to PlannerBridgeResult");

    assert_eq!(
        deserialized.report.selected_idx, result.report.selected_idx,
        "selected_idx must survive JSON round-trip"
    );
    assert_eq!(
        deserialized.admitted_count, result.admitted_count,
        "admitted_count must survive JSON round-trip"
    );
}

/// 11. Context with an EffectBoundary runs cleanly through the bridge.
#[test]
fn context_with_effect_boundary_runs_cleanly() {
    use llm_provable_computer::effect_root::{EffectBoundary, EffectLog, StateRoot};

    let boundary = EffectBoundary::new(StateRoot::empty(), EffectLog::new());
    let context = PlannerContext::new(
        b"test".to_vec(),
        "test-model".to_string(),
        MachineState::default(),
    )
    .with_effect_boundary(boundary);

    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![("LOADI 9\nHALT".to_string(), b"out".to_vec())]);
    let result = bridge.run(&planner, context);
    assert!(
        result.is_ok(),
        "bridge must run cleanly when context carries an EffectBoundary"
    );
}

/// 12. Running a program directly via ExecutionRuntime produces the same final
///     ACC as the bridge's selected outcome for the same single-candidate input.
///     This confirms the bridge does not alter execution semantics.
#[test]
fn bridge_default_path_unchanged() {
    use llm_provable_computer::compiler::ProgramCompiler;
    use llm_provable_computer::runtime::ExecutionRuntime;

    const SOURCE: &str = "LOADI 42\nHALT";

    // Run directly.
    let program = parse_program(SOURCE).expect("parse must succeed");
    let model = ProgramCompiler
        .compile_program(program, TransformerVmConfig::default())
        .expect("compile must succeed");
    let mut runtime = ExecutionRuntime::new(model, 64);
    let direct_result = runtime.run().expect("direct execution must succeed");
    let direct_acc = direct_result.final_state.acc;

    // Run through bridge.
    let bridge = make_bridge(SelectionStrategy::HighestAccumulator);
    let planner = StaticPlanner::new(vec![(SOURCE.to_string(), b"out".to_vec())]);
    let bridge_result = bridge
        .run(&planner, make_context())
        .expect("bridge must succeed");
    let bridge_acc = bridge_result.selected_outcome().final_acc;

    assert_eq!(
        direct_acc, bridge_acc as i64,
        "bridge must not alter execution semantics: direct ACC ({direct_acc}) must equal bridge ACC ({bridge_acc})"
    );
}
