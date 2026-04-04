use llm_provable_computer::{
    prove_cdmn_decision_stark_compat, verify_cdmn_decision_stark_compat, CdmnTable, Constraint,
    Variable,
};

fn small_table() -> CdmnTable {
    CdmnTable {
        name: "Small Decision".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![
            Variable {
                name: "dose".to_string(),
                memory: 0,
                presence_memory: None,
                unknown_memory: None,
            },
            Variable {
                name: "max_dose".to_string(),
                memory: 1,
                presence_memory: None,
                unknown_memory: None,
            },
        ],
        definitions: vec![],
        collections: vec![],
        constraints: vec![Constraint {
            id: "S1".to_string(),
            description: "Dose must not exceed maximum".to_string(),
            logic: "dose <= max_dose".to_string(),
            reason_code: Some(11),
            reason_label: Some("dose_above_limit".to_string()),
            insufficient_data_reason_code: None,
            insufficient_data_reason_label: None,
        }],
    }
}

#[test]
fn cdmn_stark_compat_proof_round_trips_small_policy() {
    let table = small_table();
    let proof = prove_cdmn_decision_stark_compat(&table, &[700, 500], 128).expect("prove");
    assert_eq!(proof.claim.status, 0);
    assert_eq!(proof.claim.reason_code, 11);
    assert!(
        verify_cdmn_decision_stark_compat(&proof).expect("verify"),
        "compat proof should verify"
    );
}
