use crate::cdmn::backend_tvm::LogicCompiler;
use crate::dmn::{
    CompiledDmnProgram, DmnCollectOperator, DmnHitPolicy, DmnMatchedOutputLayout, DmnOutputLayout,
    DmnReasonMetadata, DmnTraceLayout, RuleTraceMetadata,
};
use crate::dmn_ir::{
    DecisionLayout, DecisionTablePlan, ResolvedAssignment, ResolvedDefaultRule, ResolvedOutput,
    ResolvedRule,
};
use crate::error::Result;
use crate::state::Word;

pub(crate) fn compile_decision_table_to_tvm(
    plan: &DecisionTablePlan,
) -> Result<CompiledDmnProgram> {
    let mut compiler = LogicCompiler::new(plan.scratch_start);
    let mut body = Vec::new();

    emit_zero_outputs(&mut body, &plan.outputs);
    emit_zero_result_rows(&mut body, &plan.layout);
    emit_store_immediate(&mut body, plan.layout.decision_status_register, 0);
    emit_store_immediate(&mut body, plan.layout.reason_register, 0);
    emit_store_immediate(&mut body, plan.layout.selected_rule_id_register, 0);
    emit_store_immediate(&mut body, plan.layout.matched_rule_count_register, 0);
    for index in 0..plan.layout.max_matched_rules {
        emit_store_immediate(&mut body, plan.layout.matched_rule_trace_start + index, 0);
    }

    match plan.hit_policy {
        DmnHitPolicy::First => compile_first_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::Unique => compile_unique_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::Any => compile_any_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::Collect => compile_collect_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::Priority => compile_priority_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::RuleOrder => compile_rule_order_hit_policy(plan, &mut compiler, &mut body)?,
        DmnHitPolicy::OutputOrder => {
            compile_output_order_hit_policy(plan, &mut compiler, &mut body)?
        }
    }

    let trace_end = usize::from(plan.layout.matched_rule_trace_start)
        + usize::from(plan.layout.max_matched_rules);
    let result_rows_end = usize::from(plan.layout.matched_output_rows_start)
        + usize::from(plan.layout.max_matched_rules)
            * usize::from(plan.layout.matched_output_row_stride);
    let memory_size = compiler
        .high_watermark()
        .max(plan.scratch_start.saturating_add(1))
        .max(u8::try_from(trace_end).unwrap_or(u8::MAX))
        .max(u8::try_from(result_rows_end).unwrap_or(u8::MAX));
    let mut assembly = String::new();
    assembly.push_str("# Auto-compiled TVM assembly from DMN decision table\n");
    assembly.push_str(&format!(".memory {}\n\n", usize::from(memory_size)));
    for line in body {
        assembly.push_str(&line);
        assembly.push('\n');
    }

    let mut reasons = plan
        .rules
        .iter()
        .map(|rule| DmnReasonMetadata {
            code: rule.metadata.reason_code,
            label: rule.metadata.label.clone(),
            description: rule.metadata.description.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(default_rule) = &plan.default_rule {
        reasons.push(DmnReasonMetadata {
            code: default_rule.metadata.reason_code,
            label: default_rule.metadata.label.clone(),
            description: default_rule.metadata.description.clone(),
        });
    }
    reasons.push(DmnReasonMetadata {
        code: plan.system_reasons.no_match.code,
        label: plan.system_reasons.no_match.label.clone(),
        description: plan.system_reasons.no_match.description.clone(),
    });
    reasons.push(DmnReasonMetadata {
        code: plan.system_reasons.insufficient_data.code,
        label: plan.system_reasons.insufficient_data.label.clone(),
        description: plan.system_reasons.insufficient_data.description.clone(),
    });
    reasons.push(DmnReasonMetadata {
        code: plan.system_reasons.hit_policy_violation.code,
        label: plan.system_reasons.hit_policy_violation.label.clone(),
        description: plan.system_reasons.hit_policy_violation.description.clone(),
    });

    let mut rules = plan
        .rules
        .iter()
        .map(|rule| RuleTraceMetadata {
            rule_id: rule.metadata.rule_id,
            reason_code: rule.metadata.reason_code,
            label: rule.metadata.label.clone(),
            description: rule.metadata.description.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(default_rule) = &plan.default_rule {
        rules.push(RuleTraceMetadata {
            rule_id: default_rule.metadata.rule_id,
            reason_code: default_rule.metadata.reason_code,
            label: default_rule.metadata.label.clone(),
            description: default_rule.metadata.description.clone(),
        });
    }

    let output_offsets = plan
        .outputs
        .iter()
        .enumerate()
        .map(|(offset, output)| {
            (
                output.name.clone(),
                u8::try_from(offset).expect("validated DMN output offsets fit in u8"),
            )
        })
        .collect();
    Ok(CompiledDmnProgram {
        assembly,
        output: DmnOutputLayout {
            outputs: plan
                .outputs
                .iter()
                .map(|output| (output.name.clone(), output.memory))
                .collect(),
            decision_status_register: plan.layout.decision_status_register,
            reason_register: plan.layout.reason_register,
            trace: DmnTraceLayout {
                selected_rule_id_register: plan.layout.selected_rule_id_register,
                matched_rule_count_register: plan.layout.matched_rule_count_register,
                matched_rule_trace_start: plan.layout.matched_rule_trace_start,
                max_matched_rules: plan.layout.max_matched_rules,
            },
            matched_outputs: DmnMatchedOutputLayout {
                rows_start: plan.layout.matched_output_rows_start,
                row_stride: plan.layout.matched_output_row_stride,
                max_rows: plan.layout.max_matched_rules,
                output_offsets,
            },
        },
        rules,
        reasons,
    })
}

fn compile_first_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let ordered_rules = plan.rules.iter().collect::<Vec<_>>();
    compile_first_like(plan, compiler, asm, &ordered_rules)
}

fn compile_priority_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let ordered_rules = rules_by_priority(plan);
    compile_first_like(plan, compiler, asm, &ordered_rules)
}

fn compile_first_like(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
    ordered_rules: &[&ResolvedRule],
) -> Result<()> {
    for (index, rule) in ordered_rules.iter().enumerate() {
        let next_label = if index + 1 < ordered_rules.len() {
            ordered_rules[index + 1].label.clone()
        } else if let Some(default_rule) = &plan.default_rule {
            default_rule.label.clone()
        } else {
            "dmn_no_match".to_string()
        };
        let insufficient_label = format!("{}_insufficient", rule.label);
        emit_rule_header(asm, rule);
        asm.push(format!("{}:", rule.label));
        compiler.compile_boolean_expr(
            &rule.condition,
            asm,
            &next_label,
            Some(&insufficient_label),
        )?;
        emit_append_match(asm, &plan.layout, rule.id, &rule.assignments, &rule.label);
        emit_assignments(asm, &rule.assignments);
        emit_success_meta(asm, &plan.layout, rule.metadata.reason_code, rule.id);
        asm.push("  HALT".to_string());
        emit_system_failure_block(
            asm,
            &insufficient_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.insufficient_data.code,
            rule.id,
        );
    }

    emit_default_or_no_match(plan, asm);
    Ok(())
}

fn compile_unique_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let finalize_label = "dmn_unique_finalize";
    for (index, rule) in plan.rules.iter().enumerate() {
        let next_label = format!("{}_next", rule.label);
        let first_match_label = format!("{}_first_match", rule.label);
        let insufficient_label = format!("{}_insufficient", rule.label);
        let violation_label = format!("{}_violation", rule.label);
        emit_rule_header(asm, rule);
        asm.push(format!("{}:", rule.label));
        compiler.compile_boolean_expr(
            &rule.condition,
            asm,
            &next_label,
            Some(&insufficient_label),
        )?;
        emit_load(asm, plan.layout.matched_rule_count_register);
        asm.push(format!("  JZ {first_match_label}"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_repeat", rule.label),
        );
        asm.push(format!("  JMP {violation_label}"));
        asm.push(format!("{first_match_label}:"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_first", rule.label),
        );
        emit_assignments(asm, &rule.assignments);
        emit_store_immediate(asm, plan.layout.reason_register, rule.metadata.reason_code);
        emit_store_immediate(asm, plan.layout.selected_rule_id_register, rule.id);
        asm.push(format!("{next_label}:"));
        if index + 1 < plan.rules.len() {
            asm.push(format!("  JMP {}", plan.rules[index + 1].label));
        } else {
            asm.push(format!("  JMP {finalize_label}"));
        }
        emit_system_failure_block(
            asm,
            &insufficient_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.insufficient_data.code,
            rule.id,
        );
        emit_system_failure_block(
            asm,
            &violation_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.hit_policy_violation.code,
            rule.id,
        );
    }

    emit_finalize_non_first_policy(plan, asm, finalize_label);
    Ok(())
}

fn compile_any_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let finalize_label = "dmn_any_finalize";
    for (index, rule) in plan.rules.iter().enumerate() {
        let next_label = format!("{}_next", rule.label);
        let first_match_label = format!("{}_first_match", rule.label);
        let insufficient_label = format!("{}_insufficient", rule.label);
        let violation_label = format!("{}_violation", rule.label);
        let aligned_label = format!("{}_aligned", rule.label);
        emit_rule_header(asm, rule);
        asm.push(format!("{}:", rule.label));
        compiler.compile_boolean_expr(
            &rule.condition,
            asm,
            &next_label,
            Some(&insufficient_label),
        )?;
        emit_load(asm, plan.layout.matched_rule_count_register);
        asm.push(format!("  JZ {first_match_label}"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_repeat", rule.label),
        );
        emit_compare_assignments_to_outputs(asm, &rule.assignments, &violation_label);
        asm.push(format!("  JMP {aligned_label}"));
        asm.push(format!("{first_match_label}:"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_first", rule.label),
        );
        emit_assignments(asm, &rule.assignments);
        emit_store_immediate(asm, plan.layout.reason_register, rule.metadata.reason_code);
        emit_store_immediate(asm, plan.layout.selected_rule_id_register, rule.id);
        asm.push(format!("{aligned_label}:"));
        asm.push(format!("{next_label}:"));
        if index + 1 < plan.rules.len() {
            asm.push(format!("  JMP {}", plan.rules[index + 1].label));
        } else {
            asm.push(format!("  JMP {finalize_label}"));
        }
        emit_system_failure_block(
            asm,
            &insufficient_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.insufficient_data.code,
            rule.id,
        );
        emit_system_failure_block(
            asm,
            &violation_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.hit_policy_violation.code,
            rule.id,
        );
    }

    emit_finalize_non_first_policy(plan, asm, finalize_label);
    Ok(())
}

fn compile_collect_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let finalize_label = "dmn_collect_finalize";
    let operator = plan
        .collect_operator
        .expect("validated collect policy must carry a collect operator");
    for (index, rule) in plan.rules.iter().enumerate() {
        let next_label = format!("{}_next", rule.label);
        let first_match_label = format!("{}_first_match", rule.label);
        let aggregated_label = format!("{}_aggregated", rule.label);
        let insufficient_label = format!("{}_insufficient", rule.label);
        emit_rule_header(asm, rule);
        asm.push(format!("{}:", rule.label));
        compiler.compile_boolean_expr(
            &rule.condition,
            asm,
            &next_label,
            Some(&insufficient_label),
        )?;
        emit_load(asm, plan.layout.matched_rule_count_register);
        asm.push(format!("  JZ {first_match_label}"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_repeat", rule.label),
        );
        emit_collect_assignment(asm, &rule.assignments, operator, false, &rule.label)?;
        asm.push(format!("  JMP {aggregated_label}"));
        asm.push(format!("{first_match_label}:"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_first", rule.label),
        );
        emit_collect_assignment(asm, &rule.assignments, operator, true, &rule.label)?;
        emit_store_immediate(asm, plan.layout.reason_register, rule.metadata.reason_code);
        emit_store_immediate(asm, plan.layout.selected_rule_id_register, rule.id);
        asm.push(format!("{aggregated_label}:"));
        asm.push(format!("{next_label}:"));
        if index + 1 < plan.rules.len() {
            asm.push(format!("  JMP {}", plan.rules[index + 1].label));
        } else {
            asm.push(format!("  JMP {finalize_label}"));
        }
        emit_system_failure_block(
            asm,
            &insufficient_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.insufficient_data.code,
            rule.id,
        );
    }

    emit_finalize_non_first_policy(plan, asm, finalize_label);
    Ok(())
}

fn compile_rule_order_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let ordered_rules = plan.rules.iter().collect::<Vec<_>>();
    compile_multi_hit_ordered(
        plan,
        compiler,
        asm,
        &ordered_rules,
        "dmn_rule_order_finalize",
    )
}

fn compile_output_order_hit_policy(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
) -> Result<()> {
    let ordered_rules = rules_by_priority(plan);
    compile_multi_hit_ordered(
        plan,
        compiler,
        asm,
        &ordered_rules,
        "dmn_output_order_finalize",
    )
}

fn compile_multi_hit_ordered(
    plan: &DecisionTablePlan,
    compiler: &mut LogicCompiler,
    asm: &mut Vec<String>,
    ordered_rules: &[&ResolvedRule],
    finalize_label: &str,
) -> Result<()> {
    for (index, rule) in ordered_rules.iter().enumerate() {
        let next_label = if index + 1 < ordered_rules.len() {
            ordered_rules[index + 1].label.clone()
        } else {
            finalize_label.to_string()
        };
        let first_match_label = format!("{}_first_match", rule.label);
        let insufficient_label = format!("{}_insufficient", rule.label);
        emit_rule_header(asm, rule);
        asm.push(format!("{}:", rule.label));
        compiler.compile_boolean_expr(
            &rule.condition,
            asm,
            &next_label,
            Some(&insufficient_label),
        )?;
        emit_load(asm, plan.layout.matched_rule_count_register);
        asm.push(format!("  JZ {first_match_label}"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_repeat", rule.label),
        );
        asm.push(format!("  JMP {next_label}"));
        asm.push(format!("{first_match_label}:"));
        emit_append_match(
            asm,
            &plan.layout,
            rule.id,
            &rule.assignments,
            &format!("{}_first", rule.label),
        );
        emit_assignments(asm, &rule.assignments);
        emit_store_immediate(asm, plan.layout.reason_register, rule.metadata.reason_code);
        emit_store_immediate(asm, plan.layout.selected_rule_id_register, rule.id);
        asm.push(format!("  JMP {next_label}"));
        emit_system_failure_block(
            asm,
            &insufficient_label,
            &plan.outputs,
            &plan.layout,
            plan.system_reasons.insufficient_data.code,
            rule.id,
        );
    }

    emit_finalize_non_first_policy(plan, asm, finalize_label);
    Ok(())
}

fn emit_finalize_non_first_policy(plan: &DecisionTablePlan, asm: &mut Vec<String>, label: &str) {
    asm.push(format!("{label}:"));
    emit_load(asm, plan.layout.matched_rule_count_register);
    if let Some(default_rule) = &plan.default_rule {
        asm.push(format!("  JZ {}", default_rule.label));
        emit_store_immediate(asm, plan.layout.decision_status_register, 1);
        asm.push("  HALT".to_string());
        emit_default_block(asm, &plan.layout, default_rule);
    } else {
        asm.push("  JZ dmn_no_match".to_string());
        emit_store_immediate(asm, plan.layout.decision_status_register, 1);
        asm.push("  HALT".to_string());
        emit_no_match_block(plan, asm);
    }
}

fn emit_default_or_no_match(plan: &DecisionTablePlan, asm: &mut Vec<String>) {
    if let Some(default_rule) = &plan.default_rule {
        emit_default_block(asm, &plan.layout, default_rule);
    } else {
        emit_no_match_block(plan, asm);
    }
}

fn emit_default_block(
    asm: &mut Vec<String>,
    layout: &DecisionLayout,
    default_rule: &ResolvedDefaultRule,
) {
    asm.push(format!("{}:", default_rule.label));
    emit_append_match(
        asm,
        layout,
        default_rule.id,
        &default_rule.assignments,
        &default_rule.label,
    );
    emit_assignments(asm, &default_rule.assignments);
    emit_success_meta(
        asm,
        layout,
        default_rule.metadata.reason_code,
        default_rule.id,
    );
    asm.push("  HALT".to_string());
}

fn emit_no_match_block(plan: &DecisionTablePlan, asm: &mut Vec<String>) {
    emit_system_failure_block(
        asm,
        "dmn_no_match",
        &plan.outputs,
        &plan.layout,
        plan.system_reasons.no_match.code,
        0,
    );
}

fn emit_system_failure_block(
    asm: &mut Vec<String>,
    label: &str,
    outputs: &[ResolvedOutput],
    layout: &DecisionLayout,
    reason_code: Word,
    selected_rule_id: Word,
) {
    asm.push(format!("{label}:"));
    emit_zero_outputs(asm, outputs);
    emit_store_immediate(asm, layout.decision_status_register, 0);
    emit_store_immediate(asm, layout.reason_register, reason_code);
    emit_store_immediate(asm, layout.selected_rule_id_register, selected_rule_id);
    asm.push("  HALT".to_string());
}

fn rules_by_priority(plan: &DecisionTablePlan) -> Vec<&ResolvedRule> {
    let mut ordered = plan.rules.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by(|(left_index, left_rule), (right_index, right_rule)| {
        right_rule
            .priority
            .expect("validated priority-based policies require rule priority")
            .cmp(
                &left_rule
                    .priority
                    .expect("validated priority-based policies require rule priority"),
            )
            .then_with(|| left_index.cmp(right_index))
    });
    ordered.into_iter().map(|(_, rule)| rule).collect()
}

fn emit_rule_header(asm: &mut Vec<String>, rule: &ResolvedRule) {
    asm.push(format!(
        "  # Rule {}{}",
        rule.id,
        if rule.description.is_empty() {
            String::new()
        } else {
            format!(": {}", rule.description)
        }
    ));
}

fn emit_assignments(asm: &mut Vec<String>, assignments: &[ResolvedAssignment]) {
    for assignment in assignments {
        emit_store_immediate(asm, assignment.memory, assignment.value);
    }
}

fn emit_compare_assignments_to_outputs(
    asm: &mut Vec<String>,
    assignments: &[ResolvedAssignment],
    fail_label: &str,
) {
    for assignment in assignments {
        emit_load(asm, assignment.memory);
        asm.push(format!("  SUB {}", assignment.value));
        asm.push(format!("  JNZ {fail_label}"));
    }
}

fn emit_collect_assignment(
    asm: &mut Vec<String>,
    assignments: &[ResolvedAssignment],
    operator: DmnCollectOperator,
    first_match: bool,
    label_prefix: &str,
) -> Result<()> {
    let assignment = assignments.first();
    match operator {
        DmnCollectOperator::Count => {
            if let Some(assignment) = assignment {
                if first_match {
                    emit_store_immediate(asm, assignment.memory, 1);
                } else {
                    emit_load(asm, assignment.memory);
                    asm.push("  ADD 1".to_string());
                    emit_store(asm, assignment.memory);
                }
            }
        }
        DmnCollectOperator::Sum => {
            let assignment = assignment.expect("validated collect sum requires an assignment");
            if first_match {
                emit_store_immediate(asm, assignment.memory, assignment.value);
            } else {
                emit_load(asm, assignment.memory);
                asm.push(format!("  ADD {}", assignment.value));
                emit_store(asm, assignment.memory);
            }
        }
        DmnCollectOperator::Min => {
            let assignment = assignment.expect("validated collect min requires an assignment");
            if first_match {
                emit_store_immediate(asm, assignment.memory, assignment.value);
            } else {
                let keep_label = format!("{label_prefix}_collect_min_keep_{}", assignment.memory);
                let done_label = format!("{label_prefix}_collect_min_done_{}", assignment.memory);
                emit_load(asm, assignment.memory);
                asm.push(format!("  SUB {}", assignment.value));
                asm.push(format!("  JS {keep_label}"));
                asm.push(format!("  JZ {keep_label}"));
                emit_store_immediate(asm, assignment.memory, assignment.value);
                asm.push(format!("  JMP {done_label}"));
                asm.push(format!("{keep_label}:"));
                asm.push(format!("{done_label}:"));
            }
        }
        DmnCollectOperator::Max => {
            let assignment = assignment.expect("validated collect max requires an assignment");
            if first_match {
                emit_store_immediate(asm, assignment.memory, assignment.value);
            } else {
                let keep_label = format!("{label_prefix}_collect_max_keep_{}", assignment.memory);
                let done_label = format!("{label_prefix}_collect_max_done_{}", assignment.memory);
                emit_load(asm, assignment.memory);
                asm.push(format!("  SUB {}", assignment.value));
                asm.push(format!("  JNS {keep_label}"));
                emit_store_immediate(asm, assignment.memory, assignment.value);
                asm.push(format!("  JMP {done_label}"));
                asm.push(format!("{keep_label}:"));
                asm.push(format!("{done_label}:"));
            }
        }
    }
    Ok(())
}

fn emit_success_meta(
    asm: &mut Vec<String>,
    layout: &DecisionLayout,
    reason_code: Word,
    rule_id: Word,
) {
    emit_store_immediate(asm, layout.decision_status_register, 1);
    emit_store_immediate(asm, layout.reason_register, reason_code);
    emit_store_immediate(asm, layout.selected_rule_id_register, rule_id);
}

fn emit_append_match(
    asm: &mut Vec<String>,
    layout: &DecisionLayout,
    rule_id: Word,
    assignments: &[ResolvedAssignment],
    label_prefix: &str,
) {
    let done_label = format!("{label_prefix}_match_done");
    for index in 0..layout.max_matched_rules {
        let slot_label = format!("{label_prefix}_match_slot_{index}");
        emit_load(asm, layout.matched_rule_count_register);
        asm.push(format!("  SUB {}", i64::from(index)));
        asm.push(format!("  JZ {slot_label}"));
        if index + 1 == layout.max_matched_rules {
            asm.push(format!("{slot_label}:"));
            emit_store_match_slot(asm, layout, assignments, index, rule_id);
            asm.push(format!("  JMP {done_label}"));
        } else {
            asm.push(format!("  JMP {label_prefix}_match_check_{}", index + 1));
            asm.push(format!("{slot_label}:"));
            emit_store_match_slot(asm, layout, assignments, index, rule_id);
            asm.push(format!("  JMP {done_label}"));
            asm.push(format!("{label_prefix}_match_check_{}:", index + 1));
        }
    }
    asm.push(format!("{done_label}:"));
    emit_load(asm, layout.matched_rule_count_register);
    asm.push("  ADD 1".to_string());
    emit_store(asm, layout.matched_rule_count_register);
}

fn emit_store_match_slot(
    asm: &mut Vec<String>,
    layout: &DecisionLayout,
    assignments: &[ResolvedAssignment],
    index: u8,
    rule_id: Word,
) {
    emit_store_immediate(asm, layout.matched_rule_trace_start + index, rule_id);
    for assignment in assignments {
        let address = match_row_address(layout, index, assignment.row_offset);
        emit_store_immediate(asm, address, assignment.value);
    }
}

fn match_row_address(layout: &DecisionLayout, row_index: u8, row_offset: u8) -> u8 {
    let address = usize::from(layout.matched_output_rows_start)
        + usize::from(row_index) * usize::from(layout.matched_output_row_stride)
        + usize::from(row_offset);
    u8::try_from(address).expect("validated DMN row layout must fit in u8 memory")
}

fn emit_zero_outputs(asm: &mut Vec<String>, outputs: &[ResolvedOutput]) {
    for output in outputs {
        emit_store_immediate(asm, output.memory, 0);
    }
}

fn emit_zero_result_rows(asm: &mut Vec<String>, layout: &DecisionLayout) {
    for row_index in 0..layout.max_matched_rules {
        for row_offset in 0..layout.matched_output_row_stride {
            emit_store_immediate(asm, match_row_address(layout, row_index, row_offset), 0);
        }
    }
}

fn emit_store_immediate(asm: &mut Vec<String>, address: u8, value: Word) {
    asm.push(format!("  LOADI {value}"));
    emit_store(asm, address);
}

fn emit_load(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  LOAD {address}"));
}

fn emit_store(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  STORE {address}"));
}
