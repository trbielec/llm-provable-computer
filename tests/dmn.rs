use llm_provable_computer::{
    compile_dmn_table_with_metadata, parse_program, Attention2DMode, CompiledDmnProgram,
    DmnAssignment, DmnCollectOperator, DmnDefaultRule, DmnHitPolicy, DmnInput, DmnOutput, DmnRule,
    DmnTable, MachineState, NativeInterpreter,
};

fn input(name: &str, memory: u8) -> DmnInput {
    DmnInput {
        name: name.to_string(),
        memory,
        presence_memory: None,
        unknown_memory: None,
    }
}

fn input_with_presence(name: &str, memory: u8, presence_memory: u8) -> DmnInput {
    DmnInput {
        name: name.to_string(),
        memory,
        presence_memory: Some(presence_memory),
        unknown_memory: None,
    }
}

fn output(name: &str, memory: u8) -> DmnOutput {
    DmnOutput {
        name: name.to_string(),
        memory,
    }
}

fn assignment(output: &str, value: i64) -> DmnAssignment {
    DmnAssignment {
        output: output.to_string(),
        value,
    }
}

fn matched_output(
    compiled: &CompiledDmnProgram,
    state: &MachineState,
    row_index: u8,
    output: &str,
) -> i64 {
    let address = compiled
        .output
        .matched_outputs
        .matched_output_register(row_index, output)
        .expect("matched output address");
    state.memory[usize::from(address)]
}

fn run_table(
    table: &DmnTable,
    memory_updates: &[(usize, i64)],
) -> (CompiledDmnProgram, MachineState) {
    let compiled = compile_dmn_table_with_metadata(table).expect("compile dmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    for &(address, value) in memory_updates {
        initial_memory[address] = value;
    }
    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 512);
    let result = runtime.run().expect("run");
    assert!(result.halted);
    (compiled, result.final_state)
}

fn loan_table() -> DmnTable {
    DmnTable {
        name: "Loan Approval Decision Table".to_string(),
        hit_policy: DmnHitPolicy::First,
        collect_operator: None,
        inputs: vec![input("credit_score", 0), input("income", 1)],
        outputs: vec![output("status", 2), output("rule_id", 3)],
        rules: vec![
            DmnRule {
                id: 1,
                description: "Prime applicant approval".to_string(),
                priority: None,
                when: "credit_score >= 750 and income >= 25000".to_string(),
                assignments: vec![assignment("status", 1), assignment("rule_id", 1)],
                reason_code: None,
                reason_label: Some("prime_approval".to_string()),
            },
            DmnRule {
                id: 2,
                description: "Standard applicant approval".to_string(),
                priority: None,
                when: "credit_score >= 650 and income >= 30000".to_string(),
                assignments: vec![assignment("status", 1), assignment("rule_id", 2)],
                reason_code: None,
                reason_label: Some("standard_approval".to_string()),
            },
        ],
        default_rule: Some(DmnDefaultRule {
            id: 99,
            description: "Default reject".to_string(),
            priority: None,
            assignments: vec![assignment("status", 0), assignment("rule_id", 99)],
            reason_code: None,
            reason_label: Some("default_reject".to_string()),
        }),
    }
}

#[test]
fn dmn_first_hit_policy_selects_first_matching_rule() {
    let (compiled, state) = run_table(&loan_table(), &[(0, 780), (1, 40_000)]);
    assert_eq!(state.memory[2], 1);
    assert_eq!(state.memory[3], 1);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        1
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        1
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        1
    );
}

#[test]
fn dmn_first_hit_policy_falls_through_to_later_rule() {
    let (_, state) = run_table(&loan_table(), &[(0, 700), (1, 35_000)]);
    assert_eq!((state.memory[2], state.memory[3]), (1, 2));
}

#[test]
fn dmn_default_rule_handles_non_matches() {
    let (compiled, state) = run_table(&loan_table(), &[(0, 620), (1, 20_000)]);
    assert_eq!((state.memory[2], state.memory[3]), (0, 99));
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        1
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        99
    );
}

#[test]
fn dmn_unique_hit_policy_rejects_multiple_matches() {
    let table = DmnTable {
        name: "Unique".to_string(),
        hit_policy: DmnHitPolicy::Unique,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 10,
                description: "Baseline approval".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("baseline_approval".to_string()),
            },
            DmnRule {
                id: 11,
                description: "Elevated approval".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 2)],
                reason_code: None,
                reason_label: Some("elevated_approval".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 0, "user outputs are cleared on failure");
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        0
    );
    let reason_code = state.memory[usize::from(compiled.output.reason_register)];
    assert_eq!(
        compiled
            .reason_for_code(reason_code)
            .map(|reason| reason.label.as_str()),
        Some("hit_policy_violation")
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        2
    );
}

#[test]
fn dmn_any_hit_policy_accepts_identical_outputs() {
    let table = DmnTable {
        name: "Any".to_string(),
        hit_policy: DmnHitPolicy::Any,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 20,
                description: "Approved by threshold A".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("threshold_a".to_string()),
            },
            DmnRule {
                id: 21,
                description: "Approved by threshold B".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("threshold_b".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 1);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        1
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        2
    );
}

#[test]
fn dmn_collect_sum_aggregates_matching_rules() {
    let table = DmnTable {
        name: "Collect Sum".to_string(),
        hit_policy: DmnHitPolicy::Collect,
        collect_operator: Some(DmnCollectOperator::Sum),
        inputs: vec![input("score", 0)],
        outputs: vec![output("risk_points", 1)],
        rules: vec![
            DmnRule {
                id: 30,
                description: "Base points".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("risk_points", 5)],
                reason_code: None,
                reason_label: Some("base_points".to_string()),
            },
            DmnRule {
                id: 31,
                description: "Extra points".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("risk_points", 7)],
                reason_code: None,
                reason_label: Some("extra_points".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 12);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        1
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        2
    );
}

#[test]
fn dmn_collect_count_counts_matching_rules() {
    let table = DmnTable {
        name: "Collect Count".to_string(),
        hit_policy: DmnHitPolicy::Collect,
        collect_operator: Some(DmnCollectOperator::Count),
        inputs: vec![input("score", 0)],
        outputs: vec![output("match_count", 1)],
        rules: vec![
            DmnRule {
                id: 32,
                description: "Threshold A".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("match_count", 0)],
                reason_code: None,
                reason_label: Some("threshold_a".to_string()),
            },
            DmnRule {
                id: 33,
                description: "Threshold B".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("match_count", 0)],
                reason_code: None,
                reason_label: Some("threshold_b".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 2);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        1
    );
}

#[test]
fn dmn_reports_insufficient_data_instead_of_silent_fallthrough() {
    let table = DmnTable {
        name: "Insufficient Data".to_string(),
        hit_policy: DmnHitPolicy::First,
        collect_operator: None,
        inputs: vec![input_with_presence("lab_value", 0, 1)],
        outputs: vec![output("decision", 2)],
        rules: vec![DmnRule {
            id: 40,
            description: "Lab value gate".to_string(),
            priority: None,
            when: "lab_value >= 10".to_string(),
            assignments: vec![assignment("decision", 1)],
            reason_code: None,
            reason_label: Some("lab_gate_passed".to_string()),
        }],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[]);
    assert_eq!(state.memory[2], 0);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        0
    );
    let reason_code = state.memory[usize::from(compiled.output.reason_register)];
    assert_eq!(
        compiled
            .reason_for_code(reason_code)
            .map(|reason| reason.label.as_str()),
        Some("insufficient_data")
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        40
    );
}

#[test]
fn dmn_no_default_no_match_is_explicit_failure() {
    let table = DmnTable {
        name: "No Default".to_string(),
        hit_policy: DmnHitPolicy::First,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![DmnRule {
            id: 50,
            description: "Strict approval".to_string(),
            priority: None,
            when: "score >= 100".to_string(),
            assignments: vec![assignment("decision", 1)],
            reason_code: None,
            reason_label: Some("strict_approval".to_string()),
        }],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 10)]);
    assert_eq!(state.memory[1], 0);
    assert_eq!(
        state.memory[usize::from(compiled.output.decision_status_register)],
        0
    );
    let reason_code = state.memory[usize::from(compiled.output.reason_register)];
    assert_eq!(
        compiled
            .reason_for_code(reason_code)
            .map(|reason| reason.label.as_str()),
        Some("no_matching_rule")
    );
}

#[test]
fn dmn_priority_hit_policy_selects_highest_priority_match() {
    let table = DmnTable {
        name: "Priority".to_string(),
        hit_policy: DmnHitPolicy::Priority,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 60,
                description: "Lower priority approval".to_string(),
                priority: Some(10),
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("lower_priority".to_string()),
            },
            DmnRule {
                id: 61,
                description: "Higher priority approval".to_string(),
                priority: Some(100),
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 2)],
                reason_code: None,
                reason_label: Some("higher_priority".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 2);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        61
    );
    assert_eq!(matched_output(&compiled, &state, 0, "decision"), 2);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        1
    );
}

#[test]
fn dmn_rule_order_emits_all_matching_rows_in_source_order() {
    let table = DmnTable {
        name: "Rule Order".to_string(),
        hit_policy: DmnHitPolicy::RuleOrder,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 70,
                description: "First matching band".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 10)],
                reason_code: None,
                reason_label: Some("first_band".to_string()),
            },
            DmnRule {
                id: 71,
                description: "Second matching band".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 20)],
                reason_code: None,
                reason_label: Some("second_band".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 10);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        70
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        2
    );
    assert_eq!(matched_output(&compiled, &state, 0, "decision"), 10);
    assert_eq!(matched_output(&compiled, &state, 1, "decision"), 20);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_register(0).unwrap())],
        70
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_register(1).unwrap())],
        71
    );
}

#[test]
fn dmn_output_order_emits_rows_in_priority_order() {
    let table = DmnTable {
        name: "Output Order".to_string(),
        hit_policy: DmnHitPolicy::OutputOrder,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 80,
                description: "Lower ranked match".to_string(),
                priority: Some(5),
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 10)],
                reason_code: None,
                reason_label: Some("lower_ranked".to_string()),
            },
            DmnRule {
                id: 81,
                description: "Higher ranked match".to_string(),
                priority: Some(50),
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 20)],
                reason_code: None,
                reason_label: Some("higher_ranked".to_string()),
            },
        ],
        default_rule: None,
    };
    let (compiled, state) = run_table(&table, &[(0, 25)]);
    assert_eq!(state.memory[1], 20);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.selected_rule_id_register)],
        81
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_count_register)],
        2
    );
    assert_eq!(matched_output(&compiled, &state, 0, "decision"), 20);
    assert_eq!(matched_output(&compiled, &state, 1, "decision"), 10);
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_register(0).unwrap())],
        81
    );
    assert_eq!(
        state.memory[usize::from(compiled.output.trace.matched_rule_register(1).unwrap())],
        80
    );
}
