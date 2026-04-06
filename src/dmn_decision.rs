use std::collections::{BTreeMap, BTreeSet};

use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};

use crate::cdmn::proof_eval::evaluate_boolean_expression;
use crate::cdmn::{
    DecisionArithmeticExpr, DecisionAvailability, DecisionAvailabilitySource, DecisionBooleanExpr,
    DecisionValueRef, DecisionValueSource,
};
use crate::dmn::{
    build_decision_table_plan, DmnAssignment, DmnCollectOperator, DmnHitPolicy, DmnReasonMetadata,
    DmnTable,
};
use crate::dmn_ir::{DecisionTablePlan, ResolvedAssignment, ResolvedDefaultRule, ResolvedRule};
use crate::error::{Result, VmError};
use crate::state::Word;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionProgram {
    pub name: String,
    pub hit_policy: DmnHitPolicy,
    pub collect_operator: Option<DmnCollectOperator>,
    pub outputs: Vec<String>,
    pub rules: Vec<CompiledDmnDecisionRule>,
    pub default_rule: Option<CompiledDmnDecisionDefaultRule>,
    pub input_cells: Vec<u8>,
    pub reasons: Vec<DmnReasonMetadata>,
    pub system_reasons: DmnSystemReasonSet,
}

impl CompiledDmnDecisionProgram {
    pub fn reason_for_code(&self, code: Word) -> Option<&DmnReasonMetadata> {
        self.reasons.iter().find(|reason| reason.code == code)
    }

    pub fn policy_commitment(&self) -> Result<Vec<u8>> {
        hash_json(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionRule {
    pub id: Word,
    pub description: String,
    pub priority: Option<Word>,
    pub condition: DecisionBooleanExpr,
    pub assignments: Vec<DmnAssignment>,
    pub reason: DmnReasonMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionDefaultRule {
    pub id: Word,
    pub description: String,
    pub priority: Option<Word>,
    pub assignments: Vec<DmnAssignment>,
    pub reason: DmnReasonMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnSystemReasonSet {
    pub no_match: DmnReasonMetadata,
    pub insufficient_data: DmnReasonMetadata,
    pub hit_policy_violation: DmnReasonMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnDecisionEvaluation {
    pub outputs: BTreeMap<String, Word>,
    pub decision_status: Word,
    pub reason_code: Word,
    pub selected_rule_id: Word,
    pub matched_rule_count: Word,
    pub matched_rule_trace: Vec<Word>,
    pub matched_output_rows: Vec<BTreeMap<String, Word>>,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuleCheck {
    Match,
    NoMatch,
    InsufficientData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalOutcome {
    Success,
    Default,
    NoMatch,
    InsufficientData { failing_index: usize },
    HitPolicyViolation { violating_index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvaluatedRule {
    pub(crate) rule_index: usize,
    pub(crate) outcome: RuleCheck,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DmnDecisionEvaluationDetail {
    pub(crate) evaluation: DmnDecisionEvaluation,
    pub(crate) evaluated_rules: Vec<EvaluatedRule>,
    pub(crate) matched_rule_indexes: Vec<usize>,
    pub(crate) terminal: TerminalOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecisionProfile {
    evaluated_rules: Vec<EvaluatedRule>,
    matched_rule_indexes: Vec<usize>,
    terminal: TerminalOutcome,
    outputs: BTreeMap<String, Word>,
    decision_status: Word,
    reason_code: Word,
    selected_rule_id: Word,
}

pub fn compile_dmn_table_to_decision_program(
    table: &DmnTable,
) -> Result<CompiledDmnDecisionProgram> {
    let plan = build_decision_table_plan(table)?;
    lower_plan_to_decision_program(table, &plan)
}

pub fn evaluate_compiled_dmn_decision_program(
    program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<DmnDecisionEvaluation> {
    Ok(evaluate_compiled_dmn_decision_program_detailed(program, initial_memory)?.evaluation)
}

pub fn evaluate_dmn_table_decision(
    table: &DmnTable,
    initial_memory: &[Word],
) -> Result<DmnDecisionEvaluation> {
    let program = compile_dmn_table_to_decision_program(table)?;
    evaluate_compiled_dmn_decision_program(&program, initial_memory)
}

pub(crate) fn evaluate_compiled_dmn_decision_program_detailed(
    program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<DmnDecisionEvaluationDetail> {
    let projected_inputs = project_inputs(program, initial_memory)?;
    let input_commitment = hash_json(&projected_inputs)?;
    let policy_commitment = program.policy_commitment()?;
    let ordered_indexes = ordered_rule_indexes(program);
    let profile = evaluate_program_profile(program, initial_memory, &ordered_indexes)?;
    let evaluation = build_evaluation(program, policy_commitment, input_commitment, &profile);
    Ok(DmnDecisionEvaluationDetail {
        evaluation,
        evaluated_rules: profile.evaluated_rules,
        matched_rule_indexes: profile.matched_rule_indexes,
        terminal: profile.terminal,
    })
}

pub(crate) fn project_inputs(
    program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<Vec<(u8, Word)>> {
    program
        .input_cells
        .iter()
        .map(|&address| {
            initial_memory
                .get(usize::from(address))
                .copied()
                .map(|value| (address, value))
                .ok_or(VmError::MemoryOutOfBounds {
                    addr: usize::from(address),
                    size: initial_memory.len(),
                })
        })
        .collect()
}

pub(crate) fn materialize_projected_inputs(
    program: &CompiledDmnDecisionProgram,
    projected_inputs: &[(u8, Word)],
) -> Result<Vec<Word>> {
    let highest = program
        .input_cells
        .iter()
        .copied()
        .map(usize::from)
        .max()
        .unwrap_or(0);
    let mut memory = if program.input_cells.is_empty() {
        Vec::new()
    } else {
        vec![0; highest + 1]
    };
    let lookup = projected_inputs.iter().copied().collect::<BTreeMap<_, _>>();
    for &address in &program.input_cells {
        memory[usize::from(address)] = *lookup.get(&address).ok_or_else(|| {
            VmError::InvalidConfig(format!(
                "missing projected DMN input cell {address} for direct verification"
            ))
        })?;
    }
    Ok(memory)
}

fn lower_plan_to_decision_program(
    table: &DmnTable,
    plan: &DecisionTablePlan,
) -> Result<CompiledDmnDecisionProgram> {
    let outputs = table
        .outputs
        .iter()
        .map(|output| output.name.clone())
        .collect::<Vec<_>>();
    let rules = plan
        .rules
        .iter()
        .map(lower_rule)
        .collect::<Result<Vec<_>>>()?;
    let default_rule = plan
        .default_rule
        .as_ref()
        .map(lower_default_rule)
        .transpose()?;
    let reasons = collect_reasons(plan, &default_rule);
    Ok(CompiledDmnDecisionProgram {
        name: table.name.clone(),
        hit_policy: table.hit_policy,
        collect_operator: table.collect_operator,
        outputs,
        rules,
        default_rule,
        input_cells: collect_input_cells(table),
        reasons,
        system_reasons: DmnSystemReasonSet {
            no_match: lower_system_reason(
                plan.system_reasons.no_match.code,
                &plan.system_reasons.no_match.label,
                &plan.system_reasons.no_match.description,
            ),
            insufficient_data: lower_system_reason(
                plan.system_reasons.insufficient_data.code,
                &plan.system_reasons.insufficient_data.label,
                &plan.system_reasons.insufficient_data.description,
            ),
            hit_policy_violation: lower_system_reason(
                plan.system_reasons.hit_policy_violation.code,
                &plan.system_reasons.hit_policy_violation.label,
                &plan.system_reasons.hit_policy_violation.description,
            ),
        },
    })
}

fn lower_rule(rule: &ResolvedRule) -> Result<CompiledDmnDecisionRule> {
    Ok(CompiledDmnDecisionRule {
        id: rule.id,
        description: rule.description.clone(),
        priority: rule.priority,
        condition: lower_boolean(&rule.condition)?,
        assignments: lower_assignments(&rule.assignments),
        reason: lower_system_reason(
            rule.metadata.reason_code,
            &rule.metadata.label,
            &rule.metadata.description,
        ),
    })
}

fn lower_default_rule(rule: &ResolvedDefaultRule) -> Result<CompiledDmnDecisionDefaultRule> {
    Ok(CompiledDmnDecisionDefaultRule {
        id: rule.id,
        description: rule.description.clone(),
        priority: rule.priority,
        assignments: lower_assignments(&rule.assignments),
        reason: lower_system_reason(
            rule.metadata.reason_code,
            &rule.metadata.label,
            &rule.metadata.description,
        ),
    })
}

fn lower_assignments(assignments: &[ResolvedAssignment]) -> Vec<DmnAssignment> {
    assignments
        .iter()
        .map(|assignment| DmnAssignment {
            output: assignment.output.clone(),
            value: assignment.value,
        })
        .collect()
}

fn collect_reasons(
    plan: &DecisionTablePlan,
    default_rule: &Option<CompiledDmnDecisionDefaultRule>,
) -> Vec<DmnReasonMetadata> {
    let mut reasons = plan
        .rules
        .iter()
        .map(|rule| {
            lower_system_reason(
                rule.metadata.reason_code,
                &rule.metadata.label,
                &rule.metadata.description,
            )
        })
        .collect::<Vec<_>>();
    if let Some(default_rule) = default_rule {
        reasons.push(default_rule.reason.clone());
    }
    reasons.push(lower_system_reason(
        plan.system_reasons.no_match.code,
        &plan.system_reasons.no_match.label,
        &plan.system_reasons.no_match.description,
    ));
    reasons.push(lower_system_reason(
        plan.system_reasons.insufficient_data.code,
        &plan.system_reasons.insufficient_data.label,
        &plan.system_reasons.insufficient_data.description,
    ));
    reasons.push(lower_system_reason(
        plan.system_reasons.hit_policy_violation.code,
        &plan.system_reasons.hit_policy_violation.label,
        &plan.system_reasons.hit_policy_violation.description,
    ));
    reasons
}

fn lower_system_reason(code: Word, label: &str, description: &str) -> DmnReasonMetadata {
    DmnReasonMetadata {
        code,
        label: label.to_string(),
        description: description.to_string(),
    }
}

fn collect_input_cells(table: &DmnTable) -> Vec<u8> {
    let mut input_cells = BTreeSet::new();
    for input in &table.inputs {
        input_cells.insert(input.memory);
        if let Some(address) = input.presence_memory {
            input_cells.insert(address);
        }
        if let Some(address) = input.unknown_memory {
            input_cells.insert(address);
        }
    }
    input_cells.into_iter().collect()
}

fn lower_boolean(expr: &crate::cdmn::ir::BooleanIr) -> Result<DecisionBooleanExpr> {
    Ok(match expr {
        crate::cdmn::ir::BooleanIr::Comparison { lhs, op, rhs } => {
            DecisionBooleanExpr::Comparison {
                lhs: lower_arithmetic(lhs)?,
                op: *op,
                rhs: lower_arithmetic(rhs)?,
            }
        }
        crate::cdmn::ir::BooleanIr::Availability { kind, value } => {
            DecisionBooleanExpr::Availability {
                kind: *kind,
                value: lower_arithmetic(value)?,
            }
        }
        crate::cdmn::ir::BooleanIr::Between {
            value,
            lower,
            upper,
        } => DecisionBooleanExpr::Between {
            value: lower_arithmetic(value)?,
            lower: lower_arithmetic(lower)?,
            upper: lower_arithmetic(upper)?,
        },
        crate::cdmn::ir::BooleanIr::And(left, right) => DecisionBooleanExpr::And(
            Box::new(lower_boolean(left)?),
            Box::new(lower_boolean(right)?),
        ),
        crate::cdmn::ir::BooleanIr::Or(left, right) => DecisionBooleanExpr::Or(
            Box::new(lower_boolean(left)?),
            Box::new(lower_boolean(right)?),
        ),
        crate::cdmn::ir::BooleanIr::Not(inner) => {
            DecisionBooleanExpr::Not(Box::new(lower_boolean(inner)?))
        }
        crate::cdmn::ir::BooleanIr::Quantified { .. } => {
            return Err(VmError::InvalidConfig(
                "DMN direct decision programs currently support only scalar DMN inputs; quantified rule conditions over collections are not supported. Use cDMN for collection logic.".to_string(),
            ));
        }
    })
}

fn lower_arithmetic(expr: &crate::cdmn::ir::ArithmeticIr) -> Result<DecisionArithmeticExpr> {
    Ok(match expr {
        crate::cdmn::ir::ArithmeticIr::Number(value) => DecisionArithmeticExpr::Number(*value),
        crate::cdmn::ir::ArithmeticIr::Value(value_ref) => {
            DecisionArithmeticExpr::Value(lower_value_ref(value_ref)?)
        }
        crate::cdmn::ir::ArithmeticIr::Binary { op, left, right } => {
            DecisionArithmeticExpr::Binary {
                op: *op,
                left: Box::new(lower_arithmetic(left)?),
                right: Box::new(lower_arithmetic(right)?),
            }
        }
    })
}

fn lower_value_ref(value_ref: &crate::cdmn::ir::ValueRefIr) -> Result<DecisionValueRef> {
    let source = match value_ref.source {
        crate::cdmn::ir::ValueSource::Cell(address) => DecisionValueSource::InputCell(address),
        crate::cdmn::ir::ValueSource::BoundField { .. } => {
            return Err(VmError::InvalidConfig(
                "DMN direct decision programs currently support only scalar DMN inputs; bound collection fields are not supported. Use cDMN for collection logic.".to_string(),
            ));
        }
    };
    Ok(DecisionValueRef {
        source,
        availability: lower_availability(&value_ref.availability)?,
    })
}

fn lower_availability(
    availability: &crate::cdmn::ir::AvailabilityIr,
) -> Result<DecisionAvailability> {
    Ok(match availability {
        crate::cdmn::ir::AvailabilityIr::Flags { presence, unknown } => {
            DecisionAvailability::Flags {
                presence: presence.map(lower_availability_source).transpose()?,
                unknown: unknown.map(lower_availability_source).transpose()?,
            }
        }
        crate::cdmn::ir::AvailabilityIr::Status(source) => {
            DecisionAvailability::Status(lower_availability_source(*source)?)
        }
    })
}

fn lower_availability_source(
    source: crate::cdmn::ir::AvailabilitySource,
) -> Result<DecisionAvailabilitySource> {
    Ok(match source {
        crate::cdmn::ir::AvailabilitySource::Cell(address) => {
            DecisionAvailabilitySource::InputCell(address)
        }
        crate::cdmn::ir::AvailabilitySource::BoundField { .. } => {
            return Err(VmError::InvalidConfig(
                "DMN direct decision programs currently support only scalar DMN inputs; bound availability fields are not supported. Use cDMN for collection logic.".to_string(),
            ));
        }
    })
}

pub(crate) fn ordered_rule_indexes(program: &CompiledDmnDecisionProgram) -> Vec<usize> {
    match program.hit_policy {
        DmnHitPolicy::First
        | DmnHitPolicy::Unique
        | DmnHitPolicy::Any
        | DmnHitPolicy::Collect
        | DmnHitPolicy::RuleOrder => (0..program.rules.len()).collect(),
        DmnHitPolicy::Priority | DmnHitPolicy::OutputOrder => {
            let mut indexes = (0..program.rules.len()).collect::<Vec<_>>();
            indexes.sort_by(|left, right| {
                program.rules[*right]
                    .priority
                    .expect("validated priority tables carry rule priorities")
                    .cmp(
                        &program.rules[*left]
                            .priority
                            .expect("validated priority tables carry rule priorities"),
                    )
                    .then_with(|| left.cmp(right))
            });
            indexes
        }
    }
}

fn evaluate_program_profile(
    program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    ordered_indexes: &[usize],
) -> Result<DecisionProfile> {
    let mut evaluated_rules = Vec::new();
    let mut matched_rule_indexes = Vec::new();
    let mut outputs = zero_outputs(program);
    let mut selected_rule_id = 0;
    let mut reason_code = 0;

    for &rule_index in ordered_indexes {
        let outcome = evaluate_rule_condition(&program.rules[rule_index], initial_memory)?;
        evaluated_rules.push(EvaluatedRule {
            rule_index,
            outcome,
        });
        match outcome {
            RuleCheck::NoMatch => {}
            RuleCheck::InsufficientData => {
                return Ok(DecisionProfile {
                    evaluated_rules,
                    matched_rule_indexes,
                    terminal: TerminalOutcome::InsufficientData {
                        failing_index: rule_index,
                    },
                    outputs: zero_outputs(program),
                    decision_status: 0,
                    reason_code: program.system_reasons.insufficient_data.code,
                    selected_rule_id: program.rules[rule_index].id,
                });
            }
            RuleCheck::Match => {
                let rule = &program.rules[rule_index];
                let rule_outputs = outputs_from_assignments(program, &rule.assignments);
                matched_rule_indexes.push(rule_index);
                match program.hit_policy {
                    DmnHitPolicy::First | DmnHitPolicy::Priority => {
                        return Ok(DecisionProfile {
                            evaluated_rules,
                            matched_rule_indexes,
                            terminal: TerminalOutcome::Success,
                            outputs: rule_outputs,
                            decision_status: 1,
                            reason_code: rule.reason.code,
                            selected_rule_id: rule.id,
                        });
                    }
                    DmnHitPolicy::Unique => {
                        if matched_rule_indexes.len() == 1 {
                            outputs = rule_outputs;
                            selected_rule_id = rule.id;
                            reason_code = rule.reason.code;
                        } else {
                            return Ok(DecisionProfile {
                                evaluated_rules,
                                matched_rule_indexes,
                                terminal: TerminalOutcome::HitPolicyViolation {
                                    violating_index: rule_index,
                                },
                                outputs: zero_outputs(program),
                                decision_status: 0,
                                reason_code: program.system_reasons.hit_policy_violation.code,
                                selected_rule_id: rule.id,
                            });
                        }
                    }
                    DmnHitPolicy::Any => {
                        if matched_rule_indexes.len() == 1 {
                            outputs = rule_outputs;
                            selected_rule_id = rule.id;
                            reason_code = rule.reason.code;
                        } else if outputs != rule_outputs {
                            return Ok(DecisionProfile {
                                evaluated_rules,
                                matched_rule_indexes,
                                terminal: TerminalOutcome::HitPolicyViolation {
                                    violating_index: rule_index,
                                },
                                outputs: zero_outputs(program),
                                decision_status: 0,
                                reason_code: program.system_reasons.hit_policy_violation.code,
                                selected_rule_id: rule.id,
                            });
                        }
                    }
                    DmnHitPolicy::Collect => {
                        if matched_rule_indexes.len() == 1 {
                            outputs = apply_collect_assignment(
                                zero_outputs(program),
                                &rule_outputs,
                                program.collect_operator,
                                true,
                            )?;
                            selected_rule_id = rule.id;
                            reason_code = rule.reason.code;
                        } else {
                            outputs = apply_collect_assignment(
                                outputs,
                                &rule_outputs,
                                program.collect_operator,
                                false,
                            )?;
                        }
                    }
                    DmnHitPolicy::RuleOrder | DmnHitPolicy::OutputOrder => {
                        if matched_rule_indexes.len() == 1 {
                            outputs = rule_outputs;
                            selected_rule_id = rule.id;
                            reason_code = rule.reason.code;
                        }
                    }
                }
            }
        }
    }

    if !matched_rule_indexes.is_empty() {
        Ok(DecisionProfile {
            evaluated_rules,
            matched_rule_indexes,
            terminal: TerminalOutcome::Success,
            outputs,
            decision_status: 1,
            reason_code,
            selected_rule_id,
        })
    } else if let Some(default_rule) = &program.default_rule {
        Ok(DecisionProfile {
            evaluated_rules,
            matched_rule_indexes: Vec::new(),
            terminal: TerminalOutcome::Default,
            outputs: outputs_from_assignments(program, &default_rule.assignments),
            decision_status: 1,
            reason_code: default_rule.reason.code,
            selected_rule_id: default_rule.id,
        })
    } else {
        Ok(DecisionProfile {
            evaluated_rules,
            matched_rule_indexes: Vec::new(),
            terminal: TerminalOutcome::NoMatch,
            outputs: zero_outputs(program),
            decision_status: 0,
            reason_code: program.system_reasons.no_match.code,
            selected_rule_id: 0,
        })
    }
}

fn evaluate_rule_condition(
    rule: &CompiledDmnDecisionRule,
    initial_memory: &[Word],
) -> Result<RuleCheck> {
    let (value, status) = evaluate_boolean_expression(&rule.condition, initial_memory)?;
    if status == 0 {
        if value {
            Ok(RuleCheck::Match)
        } else {
            Ok(RuleCheck::NoMatch)
        }
    } else {
        Ok(RuleCheck::InsufficientData)
    }
}

fn build_evaluation(
    program: &CompiledDmnDecisionProgram,
    policy_commitment: Vec<u8>,
    input_commitment: Vec<u8>,
    profile: &DecisionProfile,
) -> DmnDecisionEvaluation {
    let matched_rule_trace = match profile.terminal {
        TerminalOutcome::Default => {
            vec![program.default_rule.as_ref().expect("default rule").id]
        }
        _ => profile
            .matched_rule_indexes
            .iter()
            .map(|&index| program.rules[index].id)
            .collect(),
    };
    let matched_output_rows = match profile.terminal {
        TerminalOutcome::Default => vec![outputs_from_assignments(
            program,
            &program
                .default_rule
                .as_ref()
                .expect("default rule")
                .assignments,
        )],
        _ => profile
            .matched_rule_indexes
            .iter()
            .map(|&index| outputs_from_assignments(program, &program.rules[index].assignments))
            .collect(),
    };
    DmnDecisionEvaluation {
        outputs: profile.outputs.clone(),
        decision_status: profile.decision_status,
        reason_code: profile.reason_code,
        selected_rule_id: profile.selected_rule_id,
        matched_rule_count: i64::try_from(matched_rule_trace.len()).expect("match count fits i64"),
        matched_rule_trace,
        matched_output_rows,
        policy_commitment,
        input_commitment,
    }
}

fn outputs_from_assignments(
    program: &CompiledDmnDecisionProgram,
    assignments: &[DmnAssignment],
) -> BTreeMap<String, Word> {
    let mut outputs = zero_outputs(program);
    for assignment in assignments {
        outputs.insert(assignment.output.clone(), assignment.value);
    }
    outputs
}

fn zero_outputs(program: &CompiledDmnDecisionProgram) -> BTreeMap<String, Word> {
    program
        .outputs
        .iter()
        .map(|output| (output.clone(), 0))
        .collect()
}

fn apply_collect_assignment(
    mut current_outputs: BTreeMap<String, Word>,
    rule_outputs: &BTreeMap<String, Word>,
    operator: Option<DmnCollectOperator>,
    first_match: bool,
) -> Result<BTreeMap<String, Word>> {
    let output_name = rule_outputs
        .keys()
        .next()
        .ok_or_else(|| VmError::InvalidConfig("collect DMN rule is missing an output".to_string()))?
        .clone();
    let rule_value = *rule_outputs.get(&output_name).ok_or_else(|| {
        VmError::InvalidConfig("collect DMN rule output missing value".to_string())
    })?;
    let current_value = *current_outputs.get(&output_name).unwrap_or(&0);
    let aggregated = match operator.expect("validated collect tables carry an operator") {
        DmnCollectOperator::Count => {
            if first_match {
                1
            } else {
                current_value + 1
            }
        }
        DmnCollectOperator::Sum => {
            if first_match {
                rule_value
            } else {
                current_value + rule_value
            }
        }
        DmnCollectOperator::Min => {
            if first_match {
                rule_value
            } else {
                current_value.min(rule_value)
            }
        }
        DmnCollectOperator::Max => {
            if first_match {
                rule_value
            } else {
                current_value.max(rule_value)
            }
        }
    };
    current_outputs.insert(output_name, aggregated);
    Ok(current_outputs)
}

fn hash_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| VmError::Serialization(format!("json serialization error: {error}")))?;
    let mut hasher = Blake2b512::new();
    hasher.update(encoded);
    Ok(hasher.finalize().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdmn::ir::{
        ArithmeticIr, AvailabilityIr, AvailabilitySource, BindingId, BooleanIr,
        QuantifiedCollection, ValueRefIr, ValueSource,
    };
    use crate::cdmn::{ComparisonOp, QuantifierKind};

    #[test]
    fn dmn_lowering_rejects_quantified_rule_conditions_with_guidance() {
        let expr = BooleanIr::Quantified {
            kind: QuantifierKind::ForAll,
            collection: QuantifiedCollection {
                binding: BindingId(0),
                name: "items".to_string(),
                count_memory: 0,
                base_memory: 1,
                stride: 1,
                max_items: 1,
            },
            predicate: Box::new(BooleanIr::Comparison {
                lhs: ArithmeticIr::Number(1),
                op: ComparisonOp::Eq,
                rhs: ArithmeticIr::Number(1),
            }),
        };

        let err = lower_boolean(&expr).unwrap_err();
        match err {
            VmError::InvalidConfig(message) => {
                assert!(message.contains("scalar DMN inputs"));
                assert!(message.contains("quantified rule conditions"));
                assert!(message.contains("Use cDMN"));
            }
            other => panic!("expected invalid-config error, found {other:?}"),
        }
    }

    #[test]
    fn dmn_lowering_rejects_bound_collection_fields_with_guidance() {
        let value_ref = ValueRefIr {
            source: ValueSource::BoundField {
                binding: BindingId(0),
                offset: 0,
            },
            availability: AvailabilityIr::Flags {
                presence: None,
                unknown: None,
            },
        };

        let err = lower_value_ref(&value_ref).unwrap_err();
        match err {
            VmError::InvalidConfig(message) => {
                assert!(message.contains("scalar DMN inputs"));
                assert!(message.contains("bound collection fields"));
                assert!(message.contains("Use cDMN"));
            }
            other => panic!("expected invalid-config error, found {other:?}"),
        }
    }

    #[test]
    fn dmn_lowering_rejects_bound_availability_fields_with_guidance() {
        let err = lower_availability_source(AvailabilitySource::BoundField {
            binding: BindingId(0),
            offset: 1,
        })
        .unwrap_err();
        match err {
            VmError::InvalidConfig(message) => {
                assert!(message.contains("scalar DMN inputs"));
                assert!(message.contains("bound availability fields"));
                assert!(message.contains("Use cDMN"));
            }
            other => panic!("expected invalid-config error, found {other:?}"),
        }
    }
}
