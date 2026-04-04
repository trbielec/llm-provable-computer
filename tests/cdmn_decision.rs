use llm_provable_computer::cdmn::QuantifierKind;
use llm_provable_computer::{
    compile_cdmn_table_to_decision_program, evaluate_cdmn_decision_program, CdmnTable,
    DecisionBooleanExpr,
};

fn load_table(path: &str) -> CdmnTable {
    let source = std::fs::read_to_string(path).expect("fixture");
    serde_json::from_str(&source).expect("parse cdmn json")
}

#[test]
fn medication_decision_program_tracks_referenced_inputs() {
    let table = load_table("programs/medication_safety.cdmn.json");
    let program = compile_cdmn_table_to_decision_program(&table).expect("compile decision");

    assert_eq!(program.input_cells, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert!(program.policy_commitment().expect("commitment").len() >= 32);
}

#[test]
fn medication_decision_program_preserves_quantified_ir() {
    let table = load_table("programs/medication_safety.cdmn.json");
    let program = compile_cdmn_table_to_decision_program(&table).expect("compile decision");

    match &program.constraints[0].expression {
        DecisionBooleanExpr::Quantified {
            kind,
            collection,
            predicate,
        } => {
            assert_eq!(*kind, QuantifierKind::ForAll);
            assert_eq!(collection.max_items, 3);
            assert!(matches!(
                predicate.as_ref(),
                DecisionBooleanExpr::Comparison { .. }
            ));
        }
        other => panic!("expected preserved quantified IR, found {other:?}"),
    }
}

#[test]
fn medication_decision_program_reports_success_and_commitments() {
    let table = load_table("programs/medication_safety.cdmn.json");
    let program = compile_cdmn_table_to_decision_program(&table).expect("compile decision");
    let initial_memory = vec![
        500, // max_dose
        11,  // required_class
        12,  // backup_class
        99,  // prohibited_class
        2,   // medication count
        400, // med0 dose
        11,  // med0 class
        450, // med1 dose
        12,  // med1 class
        0, 0,
    ];

    let evaluation = evaluate_cdmn_decision_program(&program, &initial_memory).expect("evaluate");
    assert_eq!(evaluation.status, 1);
    assert_eq!(evaluation.reason_code, 0);
    assert_eq!(
        evaluation.policy_commitment,
        program.policy_commitment().expect("policy commitment")
    );
    assert!(evaluation.input_commitment.len() >= 32);
}

#[test]
fn medication_decision_program_reports_primary_and_insufficient_failures() {
    let table = load_table("programs/medication_safety.cdmn.json");
    let program = compile_cdmn_table_to_decision_program(&table).expect("compile decision");

    let mut dose_failure_memory = vec![0; 11];
    dose_failure_memory[0] = 500;
    dose_failure_memory[1] = 11;
    dose_failure_memory[2] = 12;
    dose_failure_memory[3] = 99;
    dose_failure_memory[4] = 2;
    dose_failure_memory[5] = 600;
    dose_failure_memory[6] = 7;
    dose_failure_memory[7] = 450;
    dose_failure_memory[8] = 12;

    let dose_failure =
        evaluate_cdmn_decision_program(&program, &dose_failure_memory).expect("evaluate");
    assert_eq!(dose_failure.status, 0);
    assert_eq!(dose_failure.reason_code, 101);
    assert_eq!(
        program
            .reason_for_code(dose_failure.reason_code)
            .map(|reason| reason.label.as_str()),
        Some("dose_exceeds_max")
    );

    let mut count_failure_memory = dose_failure_memory.clone();
    count_failure_memory[4] = 4;
    let count_failure =
        evaluate_cdmn_decision_program(&program, &count_failure_memory).expect("evaluate");
    assert_eq!(count_failure.status, 0);
    assert_eq!(count_failure.reason_code, 104);
    assert_eq!(
        program
            .reason_for_code(count_failure.reason_code)
            .map(|reason| reason.label.as_str()),
        Some("dose_exceeds_max_insufficient_data")
    );
}

#[test]
fn lab_vitals_decision_program_handles_definitions_and_ranges() {
    let table = load_table("programs/lab_vitals_guardrail.cdmn.json");
    let program = compile_cdmn_table_to_decision_program(&table).expect("compile decision");

    let success_memory = vec![8_000, 4_000, 11_000, 120, 80, 30, 60];
    let success = evaluate_cdmn_decision_program(&program, &success_memory).expect("evaluate");
    assert_eq!(success.status, 1);
    assert_eq!(success.reason_code, 0);

    let failure_memory = vec![8_000, 4_000, 11_000, 170, 90, 30, 60];
    let failure = evaluate_cdmn_decision_program(&program, &failure_memory).expect("evaluate");
    assert_eq!(failure.status, 0);
    assert_eq!(failure.reason_code, 302);
    assert_eq!(
        program
            .reason_for_code(failure.reason_code)
            .map(|reason| reason.label.as_str()),
        Some("pulse_pressure_out_of_range")
    );
}
