use llm_provable_computer::{
    compile_dmn_table_to_decision_program, estimate_compiled_dmn_decision_stark_direct_cost,
    evaluate_compiled_dmn_decision_program, evaluate_dmn_table_decision,
    prove_compiled_dmn_decision_stark_direct, prove_compiled_dmn_decision_stark_direct_profiled,
    prove_dmn_decision_stark_direct, verify_compiled_dmn_decision_stark_direct,
    verify_dmn_decision_stark_direct, DmnAssignment, DmnCollectOperator,
    DmnDecisionStarkDirectBackend, DmnDefaultRule, DmnHitPolicy, DmnInput, DmnOutput, DmnRule,
    DmnTable, VanillaStarkProofOptions,
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

fn matched_output_row_value(
    evaluation: &llm_provable_computer::DmnDecisionEvaluation,
    row_index: usize,
    output: &str,
) -> i64 {
    *evaluation
        .matched_output_rows
        .get(row_index)
        .and_then(|row| row.get(output))
        .expect("matched output value")
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
fn dmn_direct_first_hit_proof_round_trips() {
    let table = loan_table();
    let memory = vec![780, 40_000, 0, 0];
    let evaluation = evaluate_dmn_table_decision(&table, &memory).expect("evaluate");
    assert_eq!(evaluation.decision_status, 1);
    assert_eq!(evaluation.reason_code, 1);
    assert_eq!(evaluation.selected_rule_id, 1);
    assert_eq!(evaluation.outputs.get("status"), Some(&1));
    assert_eq!(evaluation.outputs.get("rule_id"), Some(&1));
    assert_eq!(evaluation.matched_rule_trace, vec![1]);
    assert_eq!(evaluation.matched_output_rows.len(), 1);

    let proof = prove_dmn_decision_stark_direct(&table, &memory).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(
        proof.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert_eq!(proof.claim.decision_status, 1);
    assert_eq!(proof.claim.selected_rule_id, 1);
    assert_eq!(proof.claim.outputs.get("status"), Some(&1));
}

#[test]
fn compiled_dmn_decision_program_matches_table_evaluation() {
    let table = loan_table();
    let memory = vec![700, 35_000, 0, 0];
    let compiled = compile_dmn_table_to_decision_program(&table).expect("compile");
    let table_evaluation = evaluate_dmn_table_decision(&table, &memory).expect("table evaluate");
    let compiled_evaluation =
        evaluate_compiled_dmn_decision_program(&compiled, &memory).expect("compiled evaluate");
    assert_eq!(compiled_evaluation, table_evaluation);
}

#[test]
fn compiled_dmn_direct_proof_round_trips() {
    let compiled = compile_dmn_table_to_decision_program(&loan_table()).expect("compile");
    let proof =
        prove_compiled_dmn_decision_stark_direct(&compiled, &[780, 40_000, 0, 0]).expect("prove");
    assert!(verify_compiled_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(
        proof.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert_eq!(proof.claim.outputs.get("status"), Some(&1));
    assert_eq!(proof.claim.selected_rule_id, 1);
}

#[test]
fn compiled_dmn_direct_profiled_report_and_cost_estimate_are_available() {
    let compiled = compile_dmn_table_to_decision_program(&loan_table()).expect("compile");
    let memory = [780, 40_000, 0, 0];
    let estimate = estimate_compiled_dmn_decision_stark_direct_cost(
        &compiled,
        &memory,
        VanillaStarkProofOptions::default(),
    )
    .expect("estimate");
    assert_eq!(
        estimate.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert!(estimate.selection.trace_rows >= 2);
    assert_eq!(estimate.output_count, 2);
    assert!(estimate.evaluated_rule_count >= 1);

    let report = prove_compiled_dmn_decision_stark_direct_profiled(&compiled, &memory)
        .expect("profiled prove");
    assert!(verify_compiled_dmn_decision_stark_direct(&report.proof).expect("verify"));
    assert_eq!(
        report.proof.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert_eq!(
        report.profile.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert!(report.profile.total_micros > 0);
    assert!(report.profile.selection.total_micros > 0);
}

#[test]
fn dmn_direct_priority_proof_chooses_highest_priority_match() {
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

    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(
        proof.backend,
        DmnDecisionStarkDirectBackend::NativeAttestedSelection
    );
    assert_eq!(proof.claim.decision_status, 1);
    assert_eq!(proof.claim.reason_code, 61);
    assert_eq!(proof.claim.selected_rule_id, 61);
    assert_eq!(proof.claim.matched_rule_count, 1);
    assert_eq!(proof.claim.matched_rule_trace, vec![61]);
    assert_eq!(proof.claim.outputs.get("decision"), Some(&2));
}

#[test]
fn dmn_direct_proof_preserves_insufficient_data_outcome() {
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

    let proof = prove_dmn_decision_stark_direct(&table, &[0, 0, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(
        proof.backend,
        DmnDecisionStarkDirectBackend::DynamicRuleProfile
    );
    assert_eq!(proof.claim.decision_status, 0);
    assert_eq!(proof.claim.selected_rule_id, 40);
    assert_eq!(proof.claim.matched_rule_count, 0);
    assert!(proof.claim.matched_rule_trace.is_empty());
    assert_eq!(proof.claim.outputs.get("decision"), Some(&0));
}

#[test]
fn dmn_direct_unique_proof_preserves_hit_policy_violation() {
    let table = DmnTable {
        name: "Unique".to_string(),
        hit_policy: DmnHitPolicy::Unique,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1), output("band", 2)],
        rules: vec![
            DmnRule {
                id: 10,
                description: "Baseline approval".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 1), assignment("band", 10)],
                reason_code: None,
                reason_label: Some("baseline".to_string()),
            },
            DmnRule {
                id: 11,
                description: "Elevated approval".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 1), assignment("band", 20)],
                reason_code: None,
                reason_label: Some("elevated".to_string()),
            },
        ],
        default_rule: None,
    };

    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(proof.claim.decision_status, 0);
    assert_eq!(proof.claim.selected_rule_id, 11);
    assert_eq!(proof.claim.matched_rule_count, 2);
    assert_eq!(proof.claim.matched_rule_trace, vec![10, 11]);
    assert_eq!(proof.claim.outputs.get("decision"), Some(&0));
    assert_eq!(proof.claim.outputs.get("band"), Some(&0));
}

#[test]
fn dmn_direct_any_proof_accepts_identical_outputs() {
    let table = DmnTable {
        name: "Any".to_string(),
        hit_policy: DmnHitPolicy::Any,
        collect_operator: None,
        inputs: vec![input("score", 0)],
        outputs: vec![output("decision", 1)],
        rules: vec![
            DmnRule {
                id: 20,
                description: "Threshold A".to_string(),
                priority: None,
                when: "score >= 10".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("threshold_a".to_string()),
            },
            DmnRule {
                id: 21,
                description: "Threshold B".to_string(),
                priority: None,
                when: "score >= 20".to_string(),
                assignments: vec![assignment("decision", 1)],
                reason_code: None,
                reason_label: Some("threshold_b".to_string()),
            },
        ],
        default_rule: None,
    };

    let evaluation = evaluate_dmn_table_decision(&table, &[25, 0]).expect("evaluate");
    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(evaluation.decision_status, 1);
    assert_eq!(evaluation.selected_rule_id, 20);
    assert_eq!(evaluation.matched_rule_count, 2);
    assert_eq!(evaluation.matched_rule_trace, vec![20, 21]);
    assert_eq!(proof.claim.outputs.get("decision"), Some(&1));
}

#[test]
fn dmn_direct_collect_sum_proof_aggregates_matches() {
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

    let evaluation = evaluate_dmn_table_decision(&table, &[25, 0]).expect("evaluate");
    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(evaluation.outputs.get("risk_points"), Some(&12));
    assert_eq!(evaluation.selected_rule_id, 30);
    assert_eq!(evaluation.matched_rule_trace, vec![30, 31]);
    assert_eq!(matched_output_row_value(&evaluation, 0, "risk_points"), 5);
    assert_eq!(matched_output_row_value(&evaluation, 1, "risk_points"), 7);
}

#[test]
fn dmn_direct_rule_order_proof_emits_rows_in_source_order() {
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

    let evaluation = evaluate_dmn_table_decision(&table, &[25, 0]).expect("evaluate");
    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(evaluation.outputs.get("decision"), Some(&10));
    assert_eq!(evaluation.selected_rule_id, 70);
    assert_eq!(evaluation.matched_rule_trace, vec![70, 71]);
    assert_eq!(matched_output_row_value(&evaluation, 0, "decision"), 10);
    assert_eq!(matched_output_row_value(&evaluation, 1, "decision"), 20);
}

#[test]
fn dmn_direct_output_order_proof_emits_rows_in_priority_order() {
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

    let evaluation = evaluate_dmn_table_decision(&table, &[25, 0]).expect("evaluate");
    let proof = prove_dmn_decision_stark_direct(&table, &[25, 0]).expect("prove");
    assert!(verify_dmn_decision_stark_direct(&proof).expect("verify"));
    assert_eq!(evaluation.outputs.get("decision"), Some(&20));
    assert_eq!(evaluation.selected_rule_id, 81);
    assert_eq!(evaluation.matched_rule_trace, vec![81, 80]);
    assert_eq!(matched_output_row_value(&evaluation, 0, "decision"), 20);
    assert_eq!(matched_output_row_value(&evaluation, 1, "decision"), 10);
}
