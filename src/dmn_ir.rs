use crate::cdmn::ir::BooleanIr;
use crate::dmn::{DmnCollectOperator, DmnHitPolicy};
use crate::state::Word;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionTablePlan {
    pub(crate) hit_policy: DmnHitPolicy,
    pub(crate) collect_operator: Option<DmnCollectOperator>,
    pub(crate) scratch_start: u8,
    pub(crate) layout: DecisionLayout,
    pub(crate) rules: Vec<ResolvedRule>,
    pub(crate) default_rule: Option<ResolvedDefaultRule>,
    pub(crate) outputs: Vec<ResolvedOutput>,
    pub(crate) system_reasons: SystemReasonSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionLayout {
    pub(crate) decision_status_register: u8,
    pub(crate) reason_register: u8,
    pub(crate) selected_rule_id_register: u8,
    pub(crate) matched_rule_count_register: u8,
    pub(crate) matched_rule_trace_start: u8,
    pub(crate) max_matched_rules: u8,
    pub(crate) matched_output_rows_start: u8,
    pub(crate) matched_output_row_stride: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedRule {
    pub(crate) id: Word,
    pub(crate) description: String,
    pub(crate) priority: Option<Word>,
    pub(crate) condition: BooleanIr,
    pub(crate) assignments: Vec<ResolvedAssignment>,
    pub(crate) label: String,
    pub(crate) metadata: RuleMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedDefaultRule {
    pub(crate) id: Word,
    pub(crate) description: String,
    pub(crate) priority: Option<Word>,
    pub(crate) assignments: Vec<ResolvedAssignment>,
    pub(crate) label: String,
    pub(crate) metadata: RuleMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedAssignment {
    pub(crate) output: String,
    pub(crate) memory: u8,
    pub(crate) row_offset: u8,
    pub(crate) value: Word,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedOutput {
    pub(crate) name: String,
    pub(crate) memory: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuleMetadata {
    pub(crate) rule_id: Word,
    pub(crate) reason_code: Word,
    pub(crate) label: String,
    pub(crate) description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemReason {
    pub(crate) code: Word,
    pub(crate) label: String,
    pub(crate) description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemReasonSet {
    pub(crate) no_match: SystemReason,
    pub(crate) insufficient_data: SystemReason,
    pub(crate) hit_policy_violation: SystemReason,
}
