use std::collections::HashSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cdmn::ComparisonOp;
use crate::cdmn::{
    estimate_compiled_cdmn_decision_stark_direct_cost, evaluate_cdmn_decision_program,
    prove_compiled_cdmn_decision_stark_direct_profiled_with_options,
    verify_cdmn_decision_stark_direct, CdmnDecisionStarkDirectCostEstimate,
    CdmnDecisionStarkDirectProof, CdmnDecisionStarkDirectProvingProfile,
    CompiledCdmnDecisionProgram, DecisionArithmeticExpr, DecisionAvailability, DecisionBooleanExpr,
    DecisionConstraint, DecisionFailureReason, DecisionValueRef, DecisionValueSource,
    ReasonMetadata,
};
use crate::dmn::{DmnAssignment, DmnHitPolicy, DmnTable};
use crate::dmn_decision::{
    compile_dmn_table_to_decision_program, evaluate_compiled_dmn_decision_program_detailed,
    materialize_projected_inputs, ordered_rule_indexes, project_inputs, CompiledDmnDecisionProgram,
    DmnDecisionEvaluation, DmnDecisionEvaluationDetail, RuleCheck, TerminalOutcome,
};
use crate::error::Result;
use crate::proof::VanillaStarkProofOptions;
use crate::state::Word;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnDecisionStarkDirectClaim {
    pub table: DmnTable,
    pub projected_inputs: Vec<(u8, Word)>,
    pub outputs: std::collections::BTreeMap<String, Word>,
    pub decision_status: Word,
    pub reason_code: Word,
    pub selected_rule_id: Word,
    pub matched_rule_count: Word,
    pub matched_rule_trace: Vec<Word>,
    pub matched_output_rows: Vec<std::collections::BTreeMap<String, Word>>,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DmnDecisionStarkDirectProof {
    pub backend: DmnDecisionStarkDirectBackend,
    pub claim: DmnDecisionStarkDirectClaim,
    pub selection_proof: CdmnDecisionStarkDirectProof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DmnDecisionStarkDirectBackend {
    DynamicRuleProfile,
    NativeAttestedSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionStarkDirectClaim {
    pub decision_program: CompiledDmnDecisionProgram,
    pub projected_inputs: Vec<(u8, Word)>,
    pub outputs: std::collections::BTreeMap<String, Word>,
    pub decision_status: Word,
    pub reason_code: Word,
    pub selected_rule_id: Word,
    pub matched_rule_count: Word,
    pub matched_rule_trace: Vec<Word>,
    pub matched_output_rows: Vec<std::collections::BTreeMap<String, Word>>,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionStarkDirectProof {
    pub backend: DmnDecisionStarkDirectBackend,
    pub claim: CompiledDmnDecisionStarkDirectClaim,
    pub selection_proof: CdmnDecisionStarkDirectProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnDecisionStarkDirectCostEstimate {
    pub backend: DmnDecisionStarkDirectBackend,
    pub selection: CdmnDecisionStarkDirectCostEstimate,
    pub evaluated_rule_count: usize,
    pub matched_rule_count: usize,
    pub output_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmnDecisionStarkDirectProvingProfile {
    pub backend: DmnDecisionStarkDirectBackend,
    pub selection: CdmnDecisionStarkDirectProvingProfile,
    pub evaluated_rule_count: usize,
    pub matched_rule_count: usize,
    pub total_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DmnDecisionStarkDirectProvingReport {
    pub proof: DmnDecisionStarkDirectProof,
    pub profile: DmnDecisionStarkDirectProvingProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledDmnDecisionStarkDirectProvingReport {
    pub proof: CompiledDmnDecisionStarkDirectProof,
    pub profile: DmnDecisionStarkDirectProvingProfile,
}

#[derive(Debug, Clone)]
struct SelectionBackendPreparation {
    backend: DmnDecisionStarkDirectBackend,
    program: CompiledCdmnDecisionProgram,
    memory: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeAttestationLayout {
    decision_status: u8,
    reason_code: u8,
    selected_rule_id: u8,
    matched_rule_count: u8,
    output_cells: Vec<(String, u8)>,
}

pub fn prove_dmn_decision_stark_direct(
    table: &DmnTable,
    initial_memory: &[Word],
) -> Result<DmnDecisionStarkDirectProof> {
    prove_dmn_decision_stark_direct_with_options(
        table,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_dmn_decision_stark_direct_with_options(
    table: &DmnTable,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<DmnDecisionStarkDirectProof> {
    Ok(
        prove_dmn_decision_stark_direct_profiled_with_options(table, initial_memory, options)?
            .proof,
    )
}

pub fn prove_dmn_decision_stark_direct_profiled(
    table: &DmnTable,
    initial_memory: &[Word],
) -> Result<DmnDecisionStarkDirectProvingReport> {
    prove_dmn_decision_stark_direct_profiled_with_options(
        table,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_dmn_decision_stark_direct_profiled_with_options(
    table: &DmnTable,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<DmnDecisionStarkDirectProvingReport> {
    let decision_program = compile_dmn_table_to_decision_program(table)?;
    let compiled_report = prove_compiled_dmn_decision_stark_direct_profiled_with_options(
        &decision_program,
        initial_memory,
        options,
    )?;
    Ok(DmnDecisionStarkDirectProvingReport {
        proof: DmnDecisionStarkDirectProof {
            backend: compiled_report.proof.backend,
            claim: DmnDecisionStarkDirectClaim {
                table: table.clone(),
                projected_inputs: compiled_report.proof.claim.projected_inputs.clone(),
                outputs: compiled_report.proof.claim.outputs.clone(),
                decision_status: compiled_report.proof.claim.decision_status,
                reason_code: compiled_report.proof.claim.reason_code,
                selected_rule_id: compiled_report.proof.claim.selected_rule_id,
                matched_rule_count: compiled_report.proof.claim.matched_rule_count,
                matched_rule_trace: compiled_report.proof.claim.matched_rule_trace.clone(),
                matched_output_rows: compiled_report.proof.claim.matched_output_rows.clone(),
                policy_commitment: compiled_report.proof.claim.policy_commitment.clone(),
                input_commitment: compiled_report.proof.claim.input_commitment.clone(),
            },
            selection_proof: compiled_report.proof.selection_proof,
        },
        profile: compiled_report.profile,
    })
}

pub fn verify_dmn_decision_stark_direct(proof: &DmnDecisionStarkDirectProof) -> Result<bool> {
    let decision_program = compile_dmn_table_to_decision_program(&proof.claim.table)?;
    let compiled_proof = CompiledDmnDecisionStarkDirectProof {
        backend: proof.backend,
        claim: CompiledDmnDecisionStarkDirectClaim {
            decision_program,
            projected_inputs: proof.claim.projected_inputs.clone(),
            outputs: proof.claim.outputs.clone(),
            decision_status: proof.claim.decision_status,
            reason_code: proof.claim.reason_code,
            selected_rule_id: proof.claim.selected_rule_id,
            matched_rule_count: proof.claim.matched_rule_count,
            matched_rule_trace: proof.claim.matched_rule_trace.clone(),
            matched_output_rows: proof.claim.matched_output_rows.clone(),
            policy_commitment: proof.claim.policy_commitment.clone(),
            input_commitment: proof.claim.input_commitment.clone(),
        },
        selection_proof: proof.selection_proof.clone(),
    };
    verify_compiled_dmn_decision_stark_direct(&compiled_proof)
}

pub fn prove_compiled_dmn_decision_stark_direct(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<CompiledDmnDecisionStarkDirectProof> {
    prove_compiled_dmn_decision_stark_direct_with_options(
        decision_program,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_compiled_dmn_decision_stark_direct_profiled(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<CompiledDmnDecisionStarkDirectProvingReport> {
    prove_compiled_dmn_decision_stark_direct_profiled_with_options(
        decision_program,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_compiled_dmn_decision_stark_direct_with_options(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CompiledDmnDecisionStarkDirectProof> {
    Ok(
        prove_compiled_dmn_decision_stark_direct_profiled_with_options(
            decision_program,
            initial_memory,
            options,
        )?
        .proof,
    )
}

pub fn prove_compiled_dmn_decision_stark_direct_profiled_with_options(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CompiledDmnDecisionStarkDirectProvingReport> {
    let total_start = Instant::now();
    let detail = evaluate_compiled_dmn_decision_program_detailed(decision_program, initial_memory)?;
    let projected_inputs = project_inputs(decision_program, initial_memory)?;
    let selection = prepare_selection_backend(decision_program, initial_memory, &detail)?;
    let selection_report = prove_compiled_cdmn_decision_stark_direct_profiled_with_options(
        &selection.program,
        &selection.memory,
        options,
    )?;

    Ok(CompiledDmnDecisionStarkDirectProvingReport {
        proof: CompiledDmnDecisionStarkDirectProof {
            backend: selection.backend,
            claim: build_compiled_claim(decision_program, projected_inputs, &detail.evaluation),
            selection_proof: selection_report.proof,
        },
        profile: DmnDecisionStarkDirectProvingProfile {
            backend: selection.backend,
            selection: selection_report.profile,
            evaluated_rule_count: detail.evaluated_rules.len(),
            matched_rule_count: detail.matched_rule_indexes.len(),
            total_micros: elapsed_micros(total_start),
        },
    })
}

pub fn estimate_compiled_dmn_decision_stark_direct_cost(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<DmnDecisionStarkDirectCostEstimate> {
    let detail = evaluate_compiled_dmn_decision_program_detailed(decision_program, initial_memory)?;
    let selection = prepare_selection_backend(decision_program, initial_memory, &detail)?;
    Ok(DmnDecisionStarkDirectCostEstimate {
        backend: selection.backend,
        selection: estimate_compiled_cdmn_decision_stark_direct_cost(&selection.program, options)?,
        evaluated_rule_count: detail.evaluated_rules.len(),
        matched_rule_count: detail.matched_rule_indexes.len(),
        output_count: decision_program.outputs.len(),
    })
}

pub fn estimate_dmn_decision_stark_direct_cost(
    table: &DmnTable,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<DmnDecisionStarkDirectCostEstimate> {
    let decision_program = compile_dmn_table_to_decision_program(table)?;
    estimate_compiled_dmn_decision_stark_direct_cost(&decision_program, initial_memory, options)
}

pub fn verify_compiled_dmn_decision_stark_direct(
    proof: &CompiledDmnDecisionStarkDirectProof,
) -> Result<bool> {
    let initial_memory =
        materialize_projected_inputs(&proof.claim.decision_program, &proof.claim.projected_inputs)?;
    let detail = evaluate_compiled_dmn_decision_program_detailed(
        &proof.claim.decision_program,
        &initial_memory,
    )?;
    if detail.evaluation != claim_evaluation(&proof.claim) {
        return Ok(false);
    }

    let expected_selection =
        prepare_selection_backend(&proof.claim.decision_program, &initial_memory, &detail)?;
    let expected_selection_evaluation =
        evaluate_cdmn_decision_program(&expected_selection.program, &expected_selection.memory)?;
    let expected_projected_inputs =
        project_selection_inputs(&expected_selection.program, &expected_selection.memory)?;

    if proof.backend != expected_selection.backend
        || proof.selection_proof.claim.decision_program != expected_selection.program
        || proof.selection_proof.claim.projected_inputs != expected_projected_inputs
        || proof.selection_proof.claim.status != expected_selection_evaluation.status
        || proof.selection_proof.claim.reason_code != expected_selection_evaluation.reason_code
    {
        return Ok(false);
    }

    verify_cdmn_decision_stark_direct(&proof.selection_proof)
}

fn build_compiled_claim(
    decision_program: &CompiledDmnDecisionProgram,
    projected_inputs: Vec<(u8, Word)>,
    evaluation: &DmnDecisionEvaluation,
) -> CompiledDmnDecisionStarkDirectClaim {
    CompiledDmnDecisionStarkDirectClaim {
        decision_program: decision_program.clone(),
        projected_inputs,
        outputs: evaluation.outputs.clone(),
        decision_status: evaluation.decision_status,
        reason_code: evaluation.reason_code,
        selected_rule_id: evaluation.selected_rule_id,
        matched_rule_count: evaluation.matched_rule_count,
        matched_rule_trace: evaluation.matched_rule_trace.clone(),
        matched_output_rows: evaluation.matched_output_rows.clone(),
        policy_commitment: evaluation.policy_commitment.clone(),
        input_commitment: evaluation.input_commitment.clone(),
    }
}

fn claim_evaluation(claim: &CompiledDmnDecisionStarkDirectClaim) -> DmnDecisionEvaluation {
    DmnDecisionEvaluation {
        outputs: claim.outputs.clone(),
        decision_status: claim.decision_status,
        reason_code: claim.reason_code,
        selected_rule_id: claim.selected_rule_id,
        matched_rule_count: claim.matched_rule_count,
        matched_rule_trace: claim.matched_rule_trace.clone(),
        matched_output_rows: claim.matched_output_rows.clone(),
        policy_commitment: claim.policy_commitment.clone(),
        input_commitment: claim.input_commitment.clone(),
    }
}

fn prepare_selection_backend(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    detail: &DmnDecisionEvaluationDetail,
) -> Result<SelectionBackendPreparation> {
    if should_use_native_attested_backend(decision_program, detail) {
        let layout = allocate_native_attestation_layout(decision_program)?;
        let program = build_native_attested_selection_program(decision_program, &layout)?;
        let memory = build_native_attestation_memory(
            decision_program,
            initial_memory,
            &detail.evaluation,
            &layout,
        )?;
        Ok(SelectionBackendPreparation {
            backend: DmnDecisionStarkDirectBackend::NativeAttestedSelection,
            program,
            memory,
        })
    } else {
        Ok(SelectionBackendPreparation {
            backend: DmnDecisionStarkDirectBackend::DynamicRuleProfile,
            program: build_dynamic_selection_decision_program(decision_program, detail)?,
            memory: initial_memory.to_vec(),
        })
    }
}

fn should_use_native_attested_backend(
    decision_program: &CompiledDmnDecisionProgram,
    detail: &DmnDecisionEvaluationDetail,
) -> bool {
    matches!(
        decision_program.hit_policy,
        DmnHitPolicy::First | DmnHitPolicy::Priority
    ) && decision_program.outputs.len() <= 1
        && !matches!(detail.terminal, TerminalOutcome::InsufficientData { .. })
}

fn build_dynamic_selection_decision_program(
    decision_program: &CompiledDmnDecisionProgram,
    detail: &DmnDecisionEvaluationDetail,
) -> Result<CompiledCdmnDecisionProgram> {
    let mut used_reason_codes = decision_program
        .reasons
        .iter()
        .map(|reason| reason.code)
        .collect::<HashSet<_>>();
    used_reason_codes.insert(decision_program.system_reasons.no_match.code);
    used_reason_codes.insert(decision_program.system_reasons.insufficient_data.code);
    used_reason_codes.insert(decision_program.system_reasons.hit_policy_violation.code);
    let mut next_reason_code = selection_reason_seed(&used_reason_codes);
    let mut reserve_reason = |label: String, description: String| -> DecisionFailureReason {
        while used_reason_codes.contains(&next_reason_code) {
            next_reason_code -= 1;
        }
        let code = next_reason_code;
        used_reason_codes.insert(code);
        next_reason_code -= 1;
        DecisionFailureReason {
            code,
            label,
            description,
        }
    };

    let mut constraints = Vec::new();
    let mut reasons = Vec::new();
    for (order, evaluated_rule) in detail.evaluated_rules.iter().enumerate() {
        let rule = &decision_program.rules[evaluated_rule.rule_index];
        let (expression, reason_label, reason_description) = match evaluated_rule.outcome {
            RuleCheck::Match => (
                rule.condition.clone(),
                format!("rule_{}_must_match", rule.id),
                format!("DMN direct proof expected rule {} to match", rule.id),
            ),
            RuleCheck::NoMatch => (
                DecisionBooleanExpr::Not(Box::new(rule.condition.clone())),
                format!("rule_{}_must_not_match", rule.id),
                format!("DMN direct proof expected rule {} not to match", rule.id),
            ),
            RuleCheck::InsufficientData => (
                rule.condition.clone(),
                format!("rule_{}_unexpected_false", rule.id),
                format!(
                    "DMN direct proof expected rule {} to be insufficient-data gated",
                    rule.id
                ),
            ),
        };
        let reason = reserve_reason(reason_label, reason_description);
        let insufficient_data_reason = DecisionFailureReason {
            code: decision_program.system_reasons.insufficient_data.code,
            label: decision_program
                .system_reasons
                .insufficient_data
                .label
                .clone(),
            description: decision_program
                .system_reasons
                .insufficient_data
                .description
                .clone(),
        };
        constraints.push(DecisionConstraint {
            id: format!("dmn_rule_profile_{}_{}", order, rule.id),
            description: format!(
                "DMN direct proof rule-profile assertion for rule {}",
                rule.id
            ),
            bound_checks: Vec::new(),
            expression,
            reason: reason.clone(),
            insufficient_data_reason: insufficient_data_reason.clone(),
        });
        reasons.push(ReasonMetadata {
            code: reason.code,
            constraint_id: format!("dmn_rule_profile_{}_{}", order, rule.id),
            label: reason.label,
            description: reason.description,
        });
        reasons.push(ReasonMetadata {
            code: insufficient_data_reason.code,
            constraint_id: format!("dmn_rule_profile_{}_{}", order, rule.id),
            label: insufficient_data_reason.label,
            description: insufficient_data_reason.description,
        });
        if matches!(evaluated_rule.outcome, RuleCheck::InsufficientData) {
            break;
        }
    }

    Ok(CompiledCdmnDecisionProgram {
        name: format!("{} DMN Direct Selection Proof", decision_program.name),
        definitions: Vec::new(),
        constraints,
        input_cells: decision_program.input_cells.clone(),
        reasons,
    })
}

fn build_native_attested_selection_program(
    decision_program: &CompiledDmnDecisionProgram,
    layout: &NativeAttestationLayout,
) -> Result<CompiledCdmnDecisionProgram> {
    let mut used_reason_codes = decision_program
        .reasons
        .iter()
        .map(|reason| reason.code)
        .collect::<HashSet<_>>();
    used_reason_codes.insert(decision_program.system_reasons.no_match.code);
    used_reason_codes.insert(decision_program.system_reasons.insufficient_data.code);
    used_reason_codes.insert(decision_program.system_reasons.hit_policy_violation.code);
    let mut next_reason_code = selection_reason_seed(&used_reason_codes);
    let mut reserve_reason = |label: String, description: String| -> DecisionFailureReason {
        while used_reason_codes.contains(&next_reason_code) {
            next_reason_code -= 1;
        }
        let code = next_reason_code;
        used_reason_codes.insert(code);
        next_reason_code -= 1;
        DecisionFailureReason {
            code,
            label,
            description,
        }
    };

    let ordered_indexes = ordered_rule_indexes(decision_program);
    let terminal_predicates = build_terminal_predicates(decision_program, &ordered_indexes);
    let mut constraints = Vec::new();
    let mut reasons = Vec::new();

    let coverage_reason = reserve_reason(
        "native_attested_terminal_coverage".to_string(),
        "DMN native attested proof expected exactly one First/Priority terminal branch".to_string(),
    );
    let coverage_id = "dmn_native_terminal_coverage".to_string();
    constraints.push(DecisionConstraint {
        id: coverage_id.clone(),
        description: "DMN native attested proof requires some terminal branch to hold".to_string(),
        bound_checks: Vec::new(),
        expression: or_all(
            terminal_predicates
                .iter()
                .map(|(_, predicate)| predicate.clone()),
        )
        .unwrap_or(DecisionBooleanExpr::Constant(false)),
        reason: coverage_reason.clone(),
        insufficient_data_reason: native_insufficient_reason(decision_program),
    });
    reasons.push(reason_metadata(&coverage_id, &coverage_reason));
    reasons.push(reason_metadata(
        &coverage_id,
        &native_insufficient_reason(decision_program),
    ));

    for (label, predicate) in terminal_predicates {
        let expected = expected_terminal_attestation(decision_program, &label);
        let reason = reserve_reason(
            format!("native_attested_{label}"),
            format!("DMN native attested proof expected terminal `{label}` to match its attested outputs"),
        );
        let constraint_id = format!("dmn_native_attested_{label}");
        constraints.push(DecisionConstraint {
            id: constraint_id.clone(),
            description: format!("DMN native attested proof validates terminal `{label}`"),
            bound_checks: Vec::new(),
            expression: implies(predicate, attestation_matches(layout, &expected)),
            reason: reason.clone(),
            insufficient_data_reason: native_insufficient_reason(decision_program),
        });
        reasons.push(reason_metadata(&constraint_id, &reason));
        reasons.push(reason_metadata(
            &constraint_id,
            &native_insufficient_reason(decision_program),
        ));
    }

    Ok(CompiledCdmnDecisionProgram {
        name: format!(
            "{} DMN Native Attested Selection Proof",
            decision_program.name
        ),
        definitions: Vec::new(),
        constraints,
        input_cells: native_attestation_input_cells(decision_program, layout),
        reasons,
    })
}

fn build_terminal_predicates(
    decision_program: &CompiledDmnDecisionProgram,
    ordered_indexes: &[usize],
) -> Vec<(String, DecisionBooleanExpr)> {
    let mut predicates = Vec::new();
    let mut prior_not_matches = Vec::new();
    for &rule_index in ordered_indexes {
        let rule = &decision_program.rules[rule_index];
        let predicate = and_all(
            prior_not_matches
                .iter()
                .cloned()
                .chain(std::iter::once(rule.condition.clone())),
        )
        .unwrap_or_else(|| rule.condition.clone());
        predicates.push((format!("rule_{}", rule.id), predicate));
        prior_not_matches.push(DecisionBooleanExpr::Not(Box::new(rule.condition.clone())));
    }

    let no_match_predicate =
        and_all(prior_not_matches).unwrap_or(DecisionBooleanExpr::Constant(true));
    if let Some(default_rule) = &decision_program.default_rule {
        predicates.push((format!("default_{}", default_rule.id), no_match_predicate));
    } else {
        predicates.push(("no_match".to_string(), no_match_predicate));
    }
    predicates
}

fn expected_terminal_attestation(
    decision_program: &CompiledDmnDecisionProgram,
    label: &str,
) -> NativeAttestationValues {
    if let Some(rule_id) = label
        .strip_prefix("rule_")
        .and_then(|value| value.parse::<Word>().ok())
    {
        let rule = decision_program
            .rules
            .iter()
            .find(|rule| rule.id == rule_id)
            .expect("terminal rule label matches an existing rule");
        NativeAttestationValues {
            decision_status: 1,
            reason_code: rule.reason.code,
            selected_rule_id: rule.id,
            matched_rule_count: 1,
            outputs: outputs_for_assignments(decision_program, &rule.assignments),
        }
    } else if let Some(default_rule) = &decision_program.default_rule {
        NativeAttestationValues {
            decision_status: 1,
            reason_code: default_rule.reason.code,
            selected_rule_id: default_rule.id,
            matched_rule_count: 1,
            outputs: outputs_for_assignments(decision_program, &default_rule.assignments),
        }
    } else {
        NativeAttestationValues {
            decision_status: 0,
            reason_code: decision_program.system_reasons.no_match.code,
            selected_rule_id: 0,
            matched_rule_count: 0,
            outputs: zero_outputs(decision_program),
        }
    }
}

fn native_attestation_input_cells(
    decision_program: &CompiledDmnDecisionProgram,
    layout: &NativeAttestationLayout,
) -> Vec<u8> {
    let mut cells = decision_program.input_cells.clone();
    cells.push(layout.decision_status);
    cells.push(layout.reason_code);
    cells.push(layout.selected_rule_id);
    cells.push(layout.matched_rule_count);
    cells.extend(layout.output_cells.iter().map(|(_, address)| *address));
    cells.sort_unstable();
    cells.dedup();
    cells
}

fn allocate_native_attestation_layout(
    decision_program: &CompiledDmnDecisionProgram,
) -> Result<NativeAttestationLayout> {
    let mut next = decision_program
        .input_cells
        .iter()
        .copied()
        .max()
        .map(usize::from)
        .unwrap_or(0);
    if !decision_program.input_cells.is_empty() {
        next += 1;
    }
    let mut allocate = || -> Result<u8> {
        let address = u8::try_from(next).map_err(|_| {
            crate::error::VmError::InvalidConfig(
                "DMN native attestation layout exceeds the u8 address space".to_string(),
            )
        })?;
        next += 1;
        Ok(address)
    };

    let decision_status = allocate()?;
    let reason_code = allocate()?;
    let selected_rule_id = allocate()?;
    let matched_rule_count = allocate()?;
    let mut output_cells = Vec::new();
    for output in &decision_program.outputs {
        output_cells.push((output.clone(), allocate()?));
    }

    Ok(NativeAttestationLayout {
        decision_status,
        reason_code,
        selected_rule_id,
        matched_rule_count,
        output_cells,
    })
}

fn build_native_attestation_memory(
    decision_program: &CompiledDmnDecisionProgram,
    initial_memory: &[Word],
    evaluation: &DmnDecisionEvaluation,
    layout: &NativeAttestationLayout,
) -> Result<Vec<Word>> {
    let projected_inputs = project_inputs(decision_program, initial_memory)?;
    let highest = native_attestation_input_cells(decision_program, layout)
        .into_iter()
        .map(usize::from)
        .max()
        .unwrap_or(0);
    let mut memory = vec![0; highest + 1];
    for (address, value) in projected_inputs {
        memory[usize::from(address)] = value;
    }
    memory[usize::from(layout.decision_status)] = evaluation.decision_status;
    memory[usize::from(layout.reason_code)] = evaluation.reason_code;
    memory[usize::from(layout.selected_rule_id)] = evaluation.selected_rule_id;
    memory[usize::from(layout.matched_rule_count)] = evaluation.matched_rule_count;
    for (output, address) in &layout.output_cells {
        memory[usize::from(*address)] = *evaluation.outputs.get(output).unwrap_or(&0);
    }
    Ok(memory)
}

fn attestation_matches(
    layout: &NativeAttestationLayout,
    expected: &NativeAttestationValues,
) -> DecisionBooleanExpr {
    let mut checks = vec![
        equality_expr(layout.decision_status, expected.decision_status),
        equality_expr(layout.reason_code, expected.reason_code),
        equality_expr(layout.selected_rule_id, expected.selected_rule_id),
        equality_expr(layout.matched_rule_count, expected.matched_rule_count),
    ];
    for (output, address) in &layout.output_cells {
        checks.push(equality_expr(
            *address,
            *expected.outputs.get(output).unwrap_or(&0),
        ));
    }
    and_all(checks).unwrap_or(DecisionBooleanExpr::Constant(true))
}

fn equality_expr(address: u8, expected: Word) -> DecisionBooleanExpr {
    DecisionBooleanExpr::Comparison {
        lhs: DecisionArithmeticExpr::Value(DecisionValueRef {
            source: DecisionValueSource::InputCell(address),
            availability: DecisionAvailability::Always,
        }),
        op: ComparisonOp::Eq,
        rhs: DecisionArithmeticExpr::Number(expected),
    }
}

fn and_all<I>(expressions: I) -> Option<DecisionBooleanExpr>
where
    I: IntoIterator<Item = DecisionBooleanExpr>,
{
    expressions
        .into_iter()
        .reduce(|left, right| DecisionBooleanExpr::And(Box::new(left), Box::new(right)))
}

fn or_all<I>(expressions: I) -> Option<DecisionBooleanExpr>
where
    I: IntoIterator<Item = DecisionBooleanExpr>,
{
    expressions
        .into_iter()
        .reduce(|left, right| DecisionBooleanExpr::Or(Box::new(left), Box::new(right)))
}

fn implies(
    antecedent: DecisionBooleanExpr,
    consequent: DecisionBooleanExpr,
) -> DecisionBooleanExpr {
    DecisionBooleanExpr::Or(
        Box::new(DecisionBooleanExpr::Not(Box::new(antecedent))),
        Box::new(consequent),
    )
}

fn outputs_for_assignments(
    decision_program: &CompiledDmnDecisionProgram,
    assignments: &[DmnAssignment],
) -> std::collections::BTreeMap<String, Word> {
    let mut outputs = zero_outputs(decision_program);
    for assignment in assignments {
        outputs.insert(assignment.output.clone(), assignment.value);
    }
    outputs
}

fn zero_outputs(
    decision_program: &CompiledDmnDecisionProgram,
) -> std::collections::BTreeMap<String, Word> {
    decision_program
        .outputs
        .iter()
        .map(|output| (output.clone(), 0))
        .collect()
}

fn project_selection_inputs(
    program: &CompiledCdmnDecisionProgram,
    memory: &[Word],
) -> Result<Vec<(u8, Word)>> {
    program
        .input_cells
        .iter()
        .map(|&address| {
            memory
                .get(usize::from(address))
                .copied()
                .map(|value| (address, value))
                .ok_or(crate::error::VmError::MemoryOutOfBounds {
                    addr: usize::from(address),
                    size: memory.len(),
                })
        })
        .collect()
}

fn native_insufficient_reason(
    decision_program: &CompiledDmnDecisionProgram,
) -> DecisionFailureReason {
    DecisionFailureReason {
        code: decision_program.system_reasons.insufficient_data.code,
        label: decision_program
            .system_reasons
            .insufficient_data
            .label
            .clone(),
        description: decision_program
            .system_reasons
            .insufficient_data
            .description
            .clone(),
    }
}

fn reason_metadata(constraint_id: &str, reason: &DecisionFailureReason) -> ReasonMetadata {
    ReasonMetadata {
        code: reason.code,
        constraint_id: constraint_id.to_string(),
        label: reason.label.clone(),
        description: reason.description.clone(),
    }
}

#[derive(Debug, Clone)]
struct NativeAttestationValues {
    decision_status: Word,
    reason_code: Word,
    selected_rule_id: Word,
    matched_rule_count: Word,
    outputs: std::collections::BTreeMap<String, Word>,
}

fn selection_reason_seed(used: &HashSet<Word>) -> Word {
    let mut candidate = -1;
    while used.contains(&candidate) {
        candidate -= 1;
    }
    candidate
}

fn elapsed_micros(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}
