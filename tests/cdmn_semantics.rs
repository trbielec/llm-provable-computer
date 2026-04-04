use llm_provable_computer::{
    compile_cdmn_table_with_metadata, parse_program, Attention2DMode, CdmnTable, Constraint,
    NativeInterpreter, Variable,
};

#[test]
fn missing_comparison_data_emits_insufficient_data_reason() {
    let table = CdmnTable {
        name: "Missing Data".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![Variable {
            name: "max_dose".to_string(),
            memory: 0,
            presence_memory: Some(1),
            unknown_memory: Some(2),
        }],
        definitions: vec![],
        collections: vec![],
        constraints: vec![Constraint {
            id: "M1".to_string(),
            description: "Dose threshold".to_string(),
            logic: "max_dose >= 1".to_string(),
            reason_code: Some(101),
            reason_label: Some("dose_threshold_failed".to_string()),
            insufficient_data_reason_code: Some(201),
            insufficient_data_reason_label: Some("dose_threshold_missing".to_string()),
        }],
    };

    let compiled = compile_cdmn_table_with_metadata(&table).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let initial_memory = vec![0; program.memory_size()];
    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 256);
    let result = runtime.run().expect("run");

    assert!(result.halted);
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(result.final_state.memory[status_idx], 0);
    assert_eq!(result.final_state.memory[reason_idx], 201);
    assert_eq!(
        compiled
            .reason_for_code(result.final_state.memory[reason_idx])
            .map(|reason| reason.label.as_str()),
        Some("dose_threshold_missing")
    );
}

#[test]
fn explicit_availability_predicate_uses_primary_reason() {
    let table = CdmnTable {
        name: "Availability Predicate".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![Variable {
            name: "lab_value".to_string(),
            memory: 0,
            presence_memory: Some(1),
            unknown_memory: Some(2),
        }],
        definitions: vec![],
        collections: vec![],
        constraints: vec![Constraint {
            id: "A1".to_string(),
            description: "Lab value must be present".to_string(),
            logic: "present(lab_value)".to_string(),
            reason_code: Some(301),
            reason_label: Some("lab_value_missing".to_string()),
            insufficient_data_reason_code: None,
            insufficient_data_reason_label: None,
        }],
    };

    let compiled = compile_cdmn_table_with_metadata(&table).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let initial_memory = vec![0; program.memory_size()];
    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 256);
    let result = runtime.run().expect("run");

    assert!(result.halted);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(result.final_state.memory[reason_idx], 301);
}
