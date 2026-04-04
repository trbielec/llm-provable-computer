use llm_provable_computer::{
    compile_cdmn_table_with_metadata, parse_program, CdmnTable, Collection, CollectionField,
    Constraint, Variable,
};

fn medication_policy_table() -> CdmnTable {
    CdmnTable {
        name: "Medication Safety".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![
            Variable {
                name: "max_dose".to_string(),
                memory: 0,
                presence_memory: None,
                unknown_memory: None,
            },
            Variable {
                name: "required_class".to_string(),
                memory: 1,
                presence_memory: None,
                unknown_memory: None,
            },
            Variable {
                name: "backup_class".to_string(),
                memory: 2,
                presence_memory: None,
                unknown_memory: None,
            },
            Variable {
                name: "prohibited_class".to_string(),
                memory: 3,
                presence_memory: None,
                unknown_memory: None,
            },
        ],
        definitions: vec![],
        collections: vec![Collection {
            name: "medications".to_string(),
            count_memory: 4,
            base_memory: 5,
            stride: 2,
            max_items: 3,
            fields: vec![
                CollectionField {
                    name: "dose".to_string(),
                    offset: 0,
                    presence_offset: None,
                    unknown_offset: None,
                },
                CollectionField {
                    name: "class_code".to_string(),
                    offset: 1,
                    presence_offset: None,
                    unknown_offset: None,
                },
            ],
        }],
        constraints: vec![
            Constraint {
                id: "M1".to_string(),
                description: "Each medication must respect the max dose".to_string(),
                logic: "forall medication in medications (medication.dose <= max_dose)"
                    .to_string(),
                reason_code: Some(101),
                reason_label: Some("dose_exceeds_max".to_string()),
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            },
            Constraint {
                id: "M2".to_string(),
                description: "At least one required or backup class must be present".to_string(),
                logic: "exists medication in medications ((medication.class_code == required_class) or (medication.class_code == backup_class))".to_string(),
                reason_code: Some(102),
                reason_label: Some("required_or_backup_class_missing".to_string()),
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            },
            Constraint {
                id: "M3".to_string(),
                description: "No medication may use a prohibited class".to_string(),
                logic: "forall medication in medications (not (medication.class_code == prohibited_class))".to_string(),
                reason_code: Some(103),
                reason_label: Some("prohibited_class_present".to_string()),
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            },
        ],
    }
}

#[test]
fn medication_policy_compiles_with_real_ordered_branches() {
    let compiled =
        compile_cdmn_table_with_metadata(&medication_policy_table()).expect("compile cdmn");
    assert!(
        compiled.assembly.contains("JS"),
        "ordered comparisons should lower to sign-aware branches"
    );
    let program = parse_program(&compiled.assembly).expect("parse");
    assert!(
        program.len() <= usize::from(u8::MAX) + 1,
        "cDMN lowering must stay within the current VM program-counter bound"
    );
}
