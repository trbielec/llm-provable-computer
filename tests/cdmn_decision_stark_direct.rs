use llm_provable_computer::{
    prove_cdmn_decision_stark_direct, prove_cdmn_decision_stark_direct_profiled,
    verify_cdmn_decision_stark_direct, CdmnTable, Collection, CollectionField, Constraint,
    Variable,
};

fn load_table(path: &str) -> CdmnTable {
    let source = std::fs::read_to_string(path).expect("fixture");
    serde_json::from_str(&source).expect("parse cdmn json")
}

#[test]
#[ignore = "quantified medication direct proof now passes, but it is still too slow for the default debug suite"]
fn medication_policy_direct_stark_round_trips_success_and_failure() {
    let table = load_table("programs/medication_safety.cdmn.json");

    let success_memory = vec![
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
    let success_proof = prove_cdmn_decision_stark_direct(&table, &success_memory).expect("prove");
    assert!(verify_cdmn_decision_stark_direct(&success_proof).expect("verify"));
    assert_eq!(success_proof.claim.status, 1);
    assert_eq!(success_proof.claim.reason_code, 0);

    let failure_memory = vec![
        500, // max_dose
        11,  // required_class
        12,  // backup_class
        99,  // prohibited_class
        2,   // medication count
        600, // med0 dose
        7,   // med0 class
        450, // med1 dose
        12,  // med1 class
        0, 0,
    ];
    let failure_proof = prove_cdmn_decision_stark_direct(&table, &failure_memory).expect("prove");
    assert!(verify_cdmn_decision_stark_direct(&failure_proof).expect("verify"));
    assert_eq!(failure_proof.claim.status, 0);
    assert_eq!(failure_proof.claim.reason_code, 101);
}

#[test]
fn small_quantified_policy_direct_stark_smoke() {
    let table = CdmnTable {
        name: "Small Quantified Smoke".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![Variable {
            name: "max_dose".to_string(),
            memory: 0,
            presence_memory: None,
            unknown_memory: None,
        }],
        definitions: vec![],
        collections: vec![Collection {
            name: "medications".to_string(),
            count_memory: 1,
            base_memory: 2,
            stride: 1,
            max_items: 1,
            fields: vec![CollectionField {
                name: "dose".to_string(),
                offset: 0,
                presence_offset: None,
                unknown_offset: None,
            }],
        }],
        constraints: vec![Constraint {
            id: "Q1".to_string(),
            description: "all doses must be within limit".to_string(),
            logic: "forall medication in medications (medication.dose <= max_dose)".to_string(),
            reason_code: Some(501),
            reason_label: Some("dose_exceeds_limit".to_string()),
            insufficient_data_reason_code: None,
            insufficient_data_reason_label: None,
        }],
    };

    let success_proof =
        prove_cdmn_decision_stark_direct(&table, &[500, 1, 400]).expect("prove success");
    assert!(verify_cdmn_decision_stark_direct(&success_proof).expect("verify success"));
    assert_eq!(success_proof.claim.status, 1);
    assert_eq!(success_proof.claim.reason_code, 0);

    let failure_proof =
        prove_cdmn_decision_stark_direct(&table, &[500, 1, 600]).expect("prove failure");
    assert!(verify_cdmn_decision_stark_direct(&failure_proof).expect("verify failure"));
    assert_eq!(failure_proof.claim.status, 0);
    assert_eq!(failure_proof.claim.reason_code, 501);
}

#[test]
fn direct_stark_profile_reports_stage_timings() {
    let table = load_table("programs/lab_vitals_guardrail.cdmn.json");
    let memory = vec![8_000, 4_000, 11_000, 170, 90, 30, 60];

    let report =
        prove_cdmn_decision_stark_direct_profiled(&table, &memory).expect("profiled prove");
    assert!(verify_cdmn_decision_stark_direct(&report.proof).expect("verify"));
    assert!(report.profile.total_micros > 0);
    assert!(report.profile.stark.total_micros > 0);
    assert!(report.profile.stark.trace_interpolation_micros > 0);
    assert!(report.profile.stark.fri_prove_micros > 0);
}

#[test]
fn lab_vitals_policy_direct_stark_round_trips_definition_and_between_logic() {
    let table = load_table("programs/lab_vitals_guardrail.cdmn.json");

    let memory = vec![8_000, 4_000, 11_000, 170, 90, 30, 60];
    let proof = prove_cdmn_decision_stark_direct(&table, &memory).expect("prove");
    assert!(verify_cdmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(proof.claim.status, 0);
    assert_eq!(proof.claim.reason_code, 302);
}
