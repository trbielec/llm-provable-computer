use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::cdmn::schema::{CdmnTable, Constraint as CdmnConstraint, Variable as CdmnVariable};
use crate::cdmn::semantics::{build_compilation_plan, next_output_register};
use crate::dmn_backend_tvm::compile_decision_table_to_tvm;
use crate::dmn_ir::{
    DecisionLayout, DecisionTablePlan, ResolvedAssignment, ResolvedDefaultRule, ResolvedOutput,
    ResolvedRule, RuleMetadata, SystemReason, SystemReasonSet,
};
use crate::error::{Result, VmError};
use crate::state::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DmnHitPolicy {
    #[default]
    First,
    Unique,
    Any,
    Collect,
    Priority,
    RuleOrder,
    OutputOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DmnCollectOperator {
    Sum,
    Min,
    Max,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnInput {
    pub name: String,
    pub memory: u8,
    #[serde(default)]
    pub presence_memory: Option<u8>,
    #[serde(default)]
    pub unknown_memory: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnOutput {
    pub name: String,
    pub memory: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnAssignment {
    pub output: String,
    pub value: Word,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnRule {
    pub id: Word,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: Option<Word>,
    pub when: String,
    #[serde(default)]
    pub assignments: Vec<DmnAssignment>,
    #[serde(default)]
    pub reason_code: Option<Word>,
    #[serde(default)]
    pub reason_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnDefaultRule {
    pub id: Word,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: Option<Word>,
    #[serde(default)]
    pub assignments: Vec<DmnAssignment>,
    #[serde(default)]
    pub reason_code: Option<Word>,
    #[serde(default)]
    pub reason_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnTable {
    pub name: String,
    #[serde(default)]
    pub hit_policy: DmnHitPolicy,
    #[serde(default)]
    pub collect_operator: Option<DmnCollectOperator>,
    pub inputs: Vec<DmnInput>,
    pub outputs: Vec<DmnOutput>,
    pub rules: Vec<DmnRule>,
    #[serde(default)]
    pub default_rule: Option<DmnDefaultRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleTraceMetadata {
    pub rule_id: Word,
    pub reason_code: Word,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnReasonMetadata {
    pub code: Word,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnTraceLayout {
    pub selected_rule_id_register: u8,
    pub matched_rule_count_register: u8,
    pub matched_rule_trace_start: u8,
    pub max_matched_rules: u8,
}

impl DmnTraceLayout {
    pub fn matched_rule_register(&self, index: u8) -> Option<u8> {
        if index < self.max_matched_rules {
            self.matched_rule_trace_start.checked_add(index)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnMatchedOutputLayout {
    pub rows_start: u8,
    pub row_stride: u8,
    pub max_rows: u8,
    pub output_offsets: HashMap<String, u8>,
}

impl DmnMatchedOutputLayout {
    pub fn matched_output_register(&self, row_index: u8, output: &str) -> Option<u8> {
        let output_offset = self.output_offsets.get(output).copied()?;
        if row_index >= self.max_rows {
            return None;
        }
        let row_base =
            usize::from(self.rows_start) + usize::from(row_index) * usize::from(self.row_stride);
        let address = row_base + usize::from(output_offset);
        u8::try_from(address).ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnOutputLayout {
    pub outputs: HashMap<String, u8>,
    pub decision_status_register: u8,
    pub reason_register: u8,
    pub trace: DmnTraceLayout,
    pub matched_outputs: DmnMatchedOutputLayout,
}

impl DmnOutputLayout {
    pub fn memory_for(&self, output: &str) -> Option<u8> {
        self.outputs.get(output).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDmnProgram {
    pub assembly: String,
    pub output: DmnOutputLayout,
    pub rules: Vec<RuleTraceMetadata>,
    pub reasons: Vec<DmnReasonMetadata>,
}

impl CompiledDmnProgram {
    pub fn rule_for_id(&self, rule_id: Word) -> Option<&RuleTraceMetadata> {
        self.rules.iter().find(|rule| rule.rule_id == rule_id)
    }

    pub fn reason_for_code(&self, code: Word) -> Option<&DmnReasonMetadata> {
        self.reasons.iter().find(|reason| reason.code == code)
    }
}

pub fn compile_dmn_table(table: &DmnTable) -> Result<String> {
    Ok(compile_dmn_table_with_metadata(table)?.assembly)
}

pub fn compile_dmn_table_with_metadata(table: &DmnTable) -> Result<CompiledDmnProgram> {
    let plan = build_decision_table_plan(table)?;
    compile_decision_table_to_tvm(&plan)
}

pub(crate) fn build_decision_table_plan(table: &DmnTable) -> Result<DecisionTablePlan> {
    validate_dmn_table(table)?;
    let output_map = table
        .outputs
        .iter()
        .map(|output| (output.name.clone(), output.memory))
        .collect::<HashMap<_, _>>();
    let output_offsets = table
        .outputs
        .iter()
        .enumerate()
        .map(|(offset, output)| {
            let offset = u8::try_from(offset)
                .map_err(|_| dmn_error("too many DMN outputs for row layout".to_string()))?;
            Ok((output.name.clone(), offset))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let synthetic = CdmnTable {
        name: table.name.clone(),
        table_type: "DMN".to_string(),
        variables: table
            .inputs
            .iter()
            .map(|input| CdmnVariable {
                name: input.name.clone(),
                memory: input.memory,
                presence_memory: input.presence_memory,
                unknown_memory: input.unknown_memory,
            })
            .collect(),
        definitions: Vec::new(),
        collections: Vec::new(),
        constraints: table
            .rules
            .iter()
            .map(|rule| CdmnConstraint {
                id: format!("rule_{}", rule.id),
                description: rule.description.clone(),
                logic: rule.when.clone(),
                reason_code: rule.reason_code,
                reason_label: rule.reason_label.clone(),
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            })
            .collect(),
    };
    let condition_plan = build_compilation_plan(&synthetic)?;
    let layout = allocate_decision_layout(table)?;
    let system_reasons = plan_system_reasons(table);

    let rules = table
        .rules
        .iter()
        .zip(condition_plan.constraints.iter())
        .enumerate()
        .map(|(index, (rule, condition))| {
            Ok(ResolvedRule {
                id: rule.id,
                description: rule.description.clone(),
                priority: rule.priority,
                condition: condition.expression.clone(),
                assignments: resolve_assignments(&rule.assignments, &output_map, &output_offsets)?,
                label: format!("dmn_rule_{index}"),
                metadata: RuleMetadata {
                    rule_id: rule.id,
                    reason_code: rule.reason_code.unwrap_or(rule.id),
                    label: rule
                        .reason_label
                        .clone()
                        .unwrap_or_else(|| derive_label(&rule.description, rule.id)),
                    description: rule.description.clone(),
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let default_rule = table
        .default_rule
        .as_ref()
        .map(|rule| -> Result<ResolvedDefaultRule> {
            Ok(ResolvedDefaultRule {
                id: rule.id,
                description: rule.description.clone(),
                priority: rule.priority,
                assignments: resolve_assignments(&rule.assignments, &output_map, &output_offsets)?,
                label: "dmn_default_rule".to_string(),
                metadata: RuleMetadata {
                    rule_id: rule.id,
                    reason_code: rule.reason_code.unwrap_or(rule.id),
                    label: rule
                        .reason_label
                        .clone()
                        .unwrap_or_else(|| derive_label(&rule.description, rule.id)),
                    description: rule.description.clone(),
                },
            })
        })
        .transpose()?;

    Ok(DecisionTablePlan {
        hit_policy: table.hit_policy,
        collect_operator: table.collect_operator,
        scratch_start: layout.matched_outputs.rows_start.saturating_add(
            layout
                .matched_outputs
                .row_stride
                .saturating_mul(layout.matched_outputs.max_rows),
        ),
        layout: DecisionLayout {
            decision_status_register: layout.decision_status_register,
            reason_register: layout.reason_register,
            selected_rule_id_register: layout.trace.selected_rule_id_register,
            matched_rule_count_register: layout.trace.matched_rule_count_register,
            matched_rule_trace_start: layout.trace.matched_rule_trace_start,
            max_matched_rules: layout.trace.max_matched_rules,
            matched_output_rows_start: layout.matched_outputs.rows_start,
            matched_output_row_stride: layout.matched_outputs.row_stride,
        },
        rules,
        default_rule,
        outputs: table
            .outputs
            .iter()
            .map(|output| ResolvedOutput {
                name: output.name.clone(),
                memory: output.memory,
            })
            .collect(),
        system_reasons,
    })
}

fn resolve_assignments(
    assignments: &[DmnAssignment],
    outputs: &HashMap<String, u8>,
    output_offsets: &HashMap<String, u8>,
) -> Result<Vec<ResolvedAssignment>> {
    let mut seen = HashSet::new();
    assignments
        .iter()
        .map(|assignment| {
            if !seen.insert(assignment.output.clone()) {
                return Err(dmn_error(format!(
                    "duplicate assignment for DMN output `{}`",
                    assignment.output
                )));
            }
            let memory = outputs.get(&assignment.output).copied().ok_or_else(|| {
                dmn_error(format!(
                    "unknown DMN output `{}` in rule assignment",
                    assignment.output
                ))
            })?;
            Ok(ResolvedAssignment {
                output: assignment.output.clone(),
                memory,
                row_offset: *output_offsets.get(&assignment.output).ok_or_else(|| {
                    dmn_error(format!(
                        "unknown DMN output `{}` in row assignment layout",
                        assignment.output
                    ))
                })?,
                value: assignment.value,
            })
        })
        .collect()
}

fn validate_dmn_table(table: &DmnTable) -> Result<()> {
    if table.rules.is_empty() {
        return Err(dmn_error(
            "DMN table must contain at least one rule".to_string(),
        ));
    }
    if table.outputs.is_empty() {
        return Err(dmn_error(
            "DMN table must declare at least one output".to_string(),
        ));
    }

    match table.hit_policy {
        DmnHitPolicy::Collect => {
            if table.collect_operator.is_none() {
                return Err(dmn_error(
                    "DMN collect hit policy requires `collect_operator`".to_string(),
                ));
            }
        }
        DmnHitPolicy::Priority | DmnHitPolicy::OutputOrder => {
            if table.collect_operator.is_some() {
                return Err(dmn_error(
                    "`collect_operator` is only valid with the `collect` hit policy".to_string(),
                ));
            }
            for rule in &table.rules {
                if rule.priority.is_none() {
                    return Err(dmn_error(format!(
                        "DMN `{}` hit policy requires every rule to declare `priority`",
                        match table.hit_policy {
                            DmnHitPolicy::Priority => "priority",
                            DmnHitPolicy::OutputOrder => "output_order",
                            _ => unreachable!(),
                        }
                    )));
                }
            }
        }
        _ => {
            if table.collect_operator.is_some() {
                return Err(dmn_error(
                    "`collect_operator` is only valid with the `collect` hit policy".to_string(),
                ));
            }
        }
    }

    let mut names = HashSet::new();
    for input in &table.inputs {
        if !names.insert(input.name.clone()) {
            return Err(dmn_error(format!(
                "duplicate DMN input name `{}`",
                input.name
            )));
        }
    }

    let mut outputs = HashSet::new();
    for output in &table.outputs {
        if !outputs.insert(output.name.clone()) {
            return Err(dmn_error(format!(
                "duplicate DMN output name `{}`",
                output.name
            )));
        }
    }

    let mut rule_ids = HashSet::new();
    for rule in &table.rules {
        if !rule_ids.insert(rule.id) {
            return Err(dmn_error(format!("duplicate DMN rule id `{}`", rule.id)));
        }
    }
    if let Some(default_rule) = &table.default_rule {
        if !rule_ids.insert(default_rule.id) {
            return Err(dmn_error(format!(
                "default DMN rule id `{}` duplicates an existing rule id",
                default_rule.id
            )));
        }
    }

    let mut reason_codes = HashSet::new();
    for rule in &table.rules {
        let code = rule.reason_code.unwrap_or(rule.id);
        if !reason_codes.insert(code) {
            return Err(dmn_error(format!(
                "duplicate DMN reason code `{}` across rules/defaults",
                code
            )));
        }
    }
    if let Some(default_rule) = &table.default_rule {
        let code = default_rule.reason_code.unwrap_or(default_rule.id);
        if !reason_codes.insert(code) {
            return Err(dmn_error(format!(
                "duplicate DMN reason code `{}` across rules/defaults",
                code
            )));
        }
    }

    validate_memory_layout(table)?;

    match table.hit_policy {
        DmnHitPolicy::Collect => validate_collect_assignments(table)?,
        DmnHitPolicy::First
        | DmnHitPolicy::Unique
        | DmnHitPolicy::Any
        | DmnHitPolicy::Priority
        | DmnHitPolicy::RuleOrder
        | DmnHitPolicy::OutputOrder => {
            for rule in &table.rules {
                validate_assignments_cover_outputs(&rule.assignments, &outputs)?;
            }
            if let Some(default_rule) = &table.default_rule {
                validate_assignments_cover_outputs(&default_rule.assignments, &outputs)?;
            }
        }
    }

    Ok(())
}

fn validate_collect_assignments(table: &DmnTable) -> Result<()> {
    if table.outputs.len() != 1 {
        return Err(dmn_error(
            "DMN collect hit policy currently supports exactly one business output".to_string(),
        ));
    }
    let output_name = &table.outputs[0].name;
    for rule in &table.rules {
        validate_collect_rule_assignments(&rule.assignments, output_name, table.collect_operator)?;
    }
    if let Some(default_rule) = &table.default_rule {
        validate_collect_rule_assignments(
            &default_rule.assignments,
            output_name,
            table.collect_operator,
        )?;
    }
    Ok(())
}

fn validate_collect_rule_assignments(
    assignments: &[DmnAssignment],
    output_name: &str,
    operator: Option<DmnCollectOperator>,
) -> Result<()> {
    match operator {
        Some(DmnCollectOperator::Count) => {
            if assignments.len() != 1 {
                return Err(dmn_error(
                    "DMN collect-count rules currently require exactly one assignment per rule"
                        .to_string(),
                ));
            }
            if assignments[0].output != output_name {
                return Err(dmn_error(format!(
                    "DMN collect-count assignment must target `{output_name}`"
                )));
            }
        }
        Some(DmnCollectOperator::Sum | DmnCollectOperator::Min | DmnCollectOperator::Max) => {
            if assignments.len() != 1 {
                return Err(dmn_error(
                    "DMN collect rules currently require exactly one assignment per rule"
                        .to_string(),
                ));
            }
            if assignments[0].output != output_name {
                return Err(dmn_error(format!(
                    "DMN collect assignment must target `{output_name}`"
                )));
            }
        }
        None => {
            return Err(dmn_error(
                "DMN collect hit policy requires `collect_operator`".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_assignments_cover_outputs(
    assignments: &[DmnAssignment],
    outputs: &HashSet<String>,
) -> Result<()> {
    let assigned = assignments
        .iter()
        .map(|assignment| assignment.output.clone())
        .collect::<HashSet<_>>();
    if &assigned != outputs {
        let missing = outputs
            .difference(&assigned)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let extra = assigned
            .difference(outputs)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(dmn_error(format!(
            "DMN rule assignments must cover outputs exactly (missing: [{}], extra: [{}])",
            missing, extra
        )));
    }
    Ok(())
}

fn validate_memory_layout(table: &DmnTable) -> Result<()> {
    let mut occupied = HashSet::new();
    for input in &table.inputs {
        for (address, label) in [
            (Some(input.memory), format!("input `{}`", input.name)),
            (
                input.presence_memory,
                format!("presence flag for input `{}`", input.name),
            ),
            (
                input.unknown_memory,
                format!("unknown flag for input `{}`", input.name),
            ),
        ] {
            if let Some(address) = address {
                if !occupied.insert(address) {
                    return Err(dmn_error(format!(
                        "DMN declared memory address {} overlaps at {}",
                        address, label
                    )));
                }
            }
        }
    }
    for output in &table.outputs {
        if !occupied.insert(output.memory) {
            return Err(dmn_error(format!(
                "DMN output `{}` overlaps an existing declared memory address {}",
                output.name, output.memory
            )));
        }
    }
    Ok(())
}

fn allocate_decision_layout(table: &DmnTable) -> Result<DmnOutputLayout> {
    let declared_max = max_declared_memory_cell(table)?;
    let decision_status_register = next_output_register(declared_max)?;
    let reason_register = next_output_register(decision_status_register)?;
    let selected_rule_id_register = next_output_register(reason_register)?;
    let matched_rule_count_register = next_output_register(selected_rule_id_register)?;
    let matched_rule_trace_start = next_output_register(matched_rule_count_register)?;
    let max_matched_rules = u8::try_from(
        table.rules.len() + usize::from(table.default_rule.is_some()),
    )
    .map_err(|_| dmn_error("too many DMN rules for the u8 trace register layout".to_string()))?;
    let mut cursor = matched_rule_trace_start;
    for _ in 0..usize::from(max_matched_rules) {
        cursor = next_output_register(cursor)?;
    }
    let matched_output_rows_start = cursor;
    let row_stride = u8::try_from(table.outputs.len())
        .map_err(|_| dmn_error("too many DMN outputs for the u8 row layout".to_string()))?;
    let output_offsets = table
        .outputs
        .iter()
        .enumerate()
        .map(|(offset, output)| {
            let offset = u8::try_from(offset)
                .map_err(|_| dmn_error("too many DMN outputs for row offsets".to_string()))?;
            Ok((output.name.clone(), offset))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    if row_stride > 0 {
        for _ in 0..usize::from(max_matched_rules) * usize::from(row_stride) {
            cursor = next_output_register(cursor)?;
        }
    }
    Ok(DmnOutputLayout {
        outputs: table
            .outputs
            .iter()
            .map(|output| (output.name.clone(), output.memory))
            .collect(),
        decision_status_register,
        reason_register,
        trace: DmnTraceLayout {
            selected_rule_id_register,
            matched_rule_count_register,
            matched_rule_trace_start,
            max_matched_rules,
        },
        matched_outputs: DmnMatchedOutputLayout {
            rows_start: matched_output_rows_start,
            row_stride,
            max_rows: max_matched_rules,
            output_offsets,
        },
    })
}

fn max_declared_memory_cell(table: &DmnTable) -> Result<u8> {
    let mut max_cell = 0usize;
    for input in &table.inputs {
        max_cell = max_cell
            .max(usize::from(input.memory))
            .max(input.presence_memory.map(usize::from).unwrap_or(0))
            .max(input.unknown_memory.map(usize::from).unwrap_or(0));
    }
    for output in &table.outputs {
        max_cell = max_cell.max(usize::from(output.memory));
    }
    u8::try_from(max_cell).map_err(|_| {
        dmn_error("declared DMN memory layout exceeds the u8 address space".to_string())
    })
}

fn plan_system_reasons(table: &DmnTable) -> SystemReasonSet {
    let mut used = table
        .rules
        .iter()
        .filter_map(|rule| rule.reason_code)
        .chain(
            table
                .default_rule
                .iter()
                .filter_map(|rule| rule.reason_code),
        )
        .collect::<HashSet<_>>();
    let mut next_code = -1;
    let mut reserve = |label: &str, description: &str| -> SystemReason {
        while used.contains(&next_code) {
            next_code -= 1;
        }
        let code = next_code;
        used.insert(code);
        next_code -= 1;
        SystemReason {
            code,
            label: label.to_string(),
            description: description.to_string(),
        }
    };
    SystemReasonSet {
        no_match: reserve(
            "no_matching_rule",
            "No DMN rule matched and no default rule applied",
        ),
        insufficient_data: reserve(
            "insufficient_data",
            "Insufficient data prevented DMN rule evaluation",
        ),
        hit_policy_violation: reserve(
            "hit_policy_violation",
            "Multiple matching DMN rules violated the configured hit policy",
        ),
    }
}

fn derive_label(description: &str, rule_id: Word) -> String {
    let source = if description.is_empty() {
        format!("rule_{rule_id}")
    } else {
        description.to_string()
    };
    let mut label = String::new();
    let mut last_was_underscore = false;
    for ch in source.chars() {
        if ch.is_ascii_alphanumeric() {
            label.push(ch.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !label.is_empty() && !last_was_underscore {
            label.push('_');
            last_was_underscore = true;
        }
    }
    while label.ends_with('_') {
        label.pop();
    }
    if label.is_empty() {
        format!("rule_{rule_id}")
    } else {
        label
    }
}

fn dmn_error(message: String) -> VmError {
    VmError::InvalidConfig(message)
}
