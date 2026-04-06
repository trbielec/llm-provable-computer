use crate::approximate::{
    canonicalize_values, compare_approximate, dequantize_signed, ApproximateComparisonResult,
    ApproximateVerificationConfig,
};
use crate::config::TransformerVmConfig;
use crate::error::{Result, VmError};
use crate::freivalds::FreivaldsChallenge;
use crate::instruction::Program;
use crate::merkle_trace::MerkleizedTrace;
use crate::model::{
    SharedTransformerExecutionWitness, SharedTransformerStepWitness, TransformerVm,
};
use crate::state::MachineState;
use crate::transformer_trace::{
    build_shared_transformer_staged_trace, SharedTransformerFeedForwardChunkTraceRow,
    SharedTransformerStagedTrace, SharedTransformerTraceRow, SharedTransformerTransitionTraceRow,
};
use crate::vanillastark::field::FieldElement;
use crate::vanillastark::merkle::{Merkle, MerkleTree};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub program_hash: Vec<u8>,
    pub initial_state_hash: Vec<u8>,
    pub weights_root: Vec<u8>,
    pub final_state: MachineState,
    pub trace_root: Vec<u8>,
    pub numeric_trace_root: Vec<u8>,
    pub step_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditTarget {
    Steps(Vec<usize>),
    Range(usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequest {
    pub target: AuditTarget,
    pub challenge: Option<FreivaldsChallenge>,
    #[serde(default)]
    pub approximate_config: Option<ApproximateVerificationConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenedFeedForwardWeights {
    pub gate: Vec<Vec<i128>>,
    pub gate_bias: Vec<i128>,
    pub value: Vec<Vec<i128>>,
    pub value_bias: Vec<i128>,
    pub out: Vec<Vec<i128>>,
    pub out_bias: Vec<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum AuditableNumericTraceRow {
    InputState {
        step_number: usize,
    },
    Attention {
        step_number: usize,
        input_vector: Vec<i128>,
    },
    FeedForwardChunk {
        step_number: usize,
        chunk_index: usize,
        hidden_start: usize,
        gate_values: Vec<i128>,
        value_values: Vec<i128>,
        hidden_values: Vec<i128>,
    },
    Transition {
        step_number: usize,
        ff_output: Vec<i128>,
    },
    OutputState {
        step_number: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenedWeight {
    pub layer_idx: usize,
    pub instruction_idx: usize,
    pub weights: OpenedFeedForwardWeights,
    pub merkle_path: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenedStep {
    pub index: usize,
    pub row: SharedTransformerTraceRow,
    pub merkle_path: Vec<Vec<u8>>,
    pub freivalds_witness: Option<Vec<FieldElement>>,
    pub opened_weight: Option<OpenedWeight>,
    pub approximate_payload: Option<AuditableNumericTraceRow>,
    pub approximate_merkle_path: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResponse {
    pub opened_steps: Vec<OpenedStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApproximateAuditVerificationReport {
    pub exact_commitment_checks_passed: bool,
    pub approximate_comparisons: Vec<ApproximateComparisonResult>,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ReceiptProgramBinding<'a> {
    program: &'a Program,
    config: &'a TransformerVmConfig,
}

pub struct WeightCommitment {
    pub tree: MerkleTree,
}

struct NumericTraceCommitment {
    tree: MerkleTree,
}

impl WeightCommitment {
    pub fn new(vm: &TransformerVm) -> Result<Self> {
        let mut leaves = Vec::new();
        for pc in 0..vm.program().len() {
            let (compiled, _layer) = vm.compiled_instruction(pc as u8)?;
            let opened_weights = convert_weights_to_numeric(&compiled.ff_weights)?;
            leaves.push(hash_bytes(&serialize_opened_weights(&opened_weights)));
        }

        if leaves.is_empty() {
            leaves.push(vec![0u8; 64]);
        }

        let padded_len = leaves.len().next_power_of_two();
        while leaves.len() < padded_len {
            leaves.push(vec![0u8; 64]);
        }

        Ok(Self {
            tree: MerkleTree::from_hashed_leaves(&leaves),
        })
    }

    pub fn root(&self) -> Vec<u8> {
        self.tree.root().to_vec()
    }

    pub fn open(&self, index: usize) -> Vec<Vec<u8>> {
        self.tree.open(index)
    }
}

impl NumericTraceCommitment {
    fn new(rows: &[AuditableNumericTraceRow]) -> Self {
        let mut leaves: Vec<Vec<u8>> = rows
            .iter()
            .map(|row| hash_bytes(&serialize_numeric_trace_row(row)))
            .collect();

        if leaves.is_empty() {
            leaves.push(vec![0u8; 64]);
        }

        let padded_len = leaves.len().next_power_of_two();
        while leaves.len() < padded_len {
            leaves.push(vec![0u8; 64]);
        }

        Self {
            tree: MerkleTree::from_hashed_leaves(&leaves),
        }
    }

    fn root(&self) -> Vec<u8> {
        self.tree.root().to_vec()
    }

    fn open(&self, index: usize) -> Vec<Vec<u8>> {
        self.tree.open(index)
    }
}

fn hash_bytes(data: &[u8]) -> Vec<u8> {
    use blake2::{Blake2b512, Digest};
    let mut hasher = Blake2b512::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

fn hash_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| VmError::Serialization(format!("json serialization error: {error}")))?;
    Ok(hash_bytes(&encoded))
}

pub fn receipt_program_hash(vm: &TransformerVm) -> Result<Vec<u8>> {
    hash_json(&ReceiptProgramBinding {
        program: vm.program(),
        config: vm.config(),
    })
}

pub fn receipt_initial_state_hash(initial_state: &MachineState) -> Result<Vec<u8>> {
    hash_json(initial_state)
}

pub fn verify_receipt_bindings(
    receipt: &ExecutionReceipt,
    vm: &TransformerVm,
    initial_state: &MachineState,
) -> Result<bool> {
    Ok(receipt.program_hash == receipt_program_hash(vm)?
        && receipt.initial_state_hash == receipt_initial_state_hash(initial_state)?)
}

pub fn create_receipt(
    trace: &SharedTransformerStagedTrace,
    program_hash: Vec<u8>,
    initial_state_hash: Vec<u8>,
    weights_root: Vec<u8>,
    numeric_trace_root: Vec<u8>,
    final_state: MachineState,
) -> Result<(ExecutionReceipt, MerkleizedTrace)> {
    let m_trace = MerkleizedTrace::new(&trace.rows)?;
    let receipt = ExecutionReceipt {
        program_hash,
        initial_state_hash,
        weights_root,
        final_state,
        trace_root: m_trace.root(),
        numeric_trace_root,
        step_count: trace.rows.len(),
    };
    Ok((receipt, m_trace))
}

pub fn commit_execution(
    vm: &TransformerVm,
    witness: &SharedTransformerExecutionWitness,
) -> Result<(ExecutionReceipt, MerkleizedTrace, WeightCommitment)> {
    let staged_trace = build_shared_transformer_staged_trace(witness, vm.config().ff_dim)?;
    let numeric_rows = build_auditable_numeric_rows(&staged_trace, witness)?;
    let numeric_trace_commitment = NumericTraceCommitment::new(&numeric_rows);
    let weight_commitment = WeightCommitment::new(vm)?;

    let program_hash = receipt_program_hash(vm)?;
    let initial_state_hash = receipt_initial_state_hash(&witness.initial_state)?;

    let (receipt, m_trace) = create_receipt(
        &staged_trace,
        program_hash,
        initial_state_hash,
        weight_commitment.root(),
        numeric_trace_commitment.root(),
        witness.final_state.clone(),
    )?;

    Ok((receipt, m_trace, weight_commitment))
}

pub fn audit_execution(
    vm: &TransformerVm,
    m_trace: &MerkleizedTrace,
    weight_commitment: &WeightCommitment,
    witness: &SharedTransformerExecutionWitness,
    staged_trace: &SharedTransformerStagedTrace,
    request: &AuditRequest,
) -> Result<AuditResponse> {
    let numeric_rows = request
        .approximate_config
        .as_ref()
        .map(|_| build_auditable_numeric_rows(staged_trace, witness))
        .transpose()?;
    let numeric_commitment = numeric_rows
        .as_ref()
        .map(|rows| NumericTraceCommitment::new(rows));

    let indices: Vec<usize> = match &request.target {
        AuditTarget::Steps(steps) => steps.clone(),
        AuditTarget::Range(start, end) => (*start..*end).collect(),
    };

    let mut opened_steps = Vec::new();
    for index in indices {
        if index >= staged_trace.rows.len() {
            continue;
        }

        let row = &staged_trace.rows[index];
        let merkle_path = m_trace.open(index);
        let mut opened_weight = None;
        let freivalds_witness = None;

        if let SharedTransformerTraceRow::Transition(t_row) = row {
            let instruction_idx = t_row.transition.pc as usize;
            let (compiled, layer_idx) = vm.compiled_instruction(t_row.transition.pc as u8)?;
            opened_weight = Some(OpenedWeight {
                layer_idx,
                instruction_idx,
                weights: convert_weights_to_numeric(&compiled.ff_weights)?,
                merkle_path: weight_commitment.open(instruction_idx),
            });
        }

        let approximate_payload = numeric_rows.as_ref().map(|rows| rows[index].clone());
        let approximate_merkle_path = numeric_commitment
            .as_ref()
            .map(|commitment| commitment.open(index));

        opened_steps.push(OpenedStep {
            index,
            row: row.clone(),
            merkle_path,
            freivalds_witness,
            opened_weight,
            approximate_payload,
            approximate_merkle_path,
        });
    }

    Ok(AuditResponse { opened_steps })
}

pub fn verify_audit(
    receipt: &ExecutionReceipt,
    response: &AuditResponse,
    challenge: Option<&FreivaldsChallenge>,
) -> bool {
    for step in &response.opened_steps {
        if !MerkleizedTrace::verify(
            &receipt.trace_root,
            step.index,
            &step.merkle_path,
            &step.row,
        ) {
            return false;
        }

        if let Some(ow) = &step.opened_weight {
            let weight_bytes = serialize_opened_weights(&ow.weights);
            if !Merkle::verify_(
                &receipt.weights_root,
                ow.instruction_idx,
                &ow.merkle_path,
                &hash_bytes(&weight_bytes),
            ) {
                return false;
            }
        }

        if let (Some(payload), Some(path)) =
            (&step.approximate_payload, &step.approximate_merkle_path)
        {
            let payload_bytes = serialize_numeric_trace_row(payload);
            if !Merkle::verify_(
                &receipt.numeric_trace_root,
                step.index,
                path,
                &hash_bytes(&payload_bytes),
            ) {
                return false;
            }
        }

        if let (Some(_challenge), Some(_vt_w)) = (challenge, &step.freivalds_witness) {
            // The exact Freivalds audit path remains optional and exact-only.
        }
    }
    true
}

pub fn verify_audit_with_config(
    receipt: &ExecutionReceipt,
    response: &AuditResponse,
    replay_witness: &SharedTransformerExecutionWitness,
    config: &ApproximateVerificationConfig,
    challenge: Option<&FreivaldsChallenge>,
) -> Result<ApproximateAuditVerificationReport> {
    config.validate().map_err(|error| {
        VmError::InvalidConfig(format!("invalid approximate audit configuration: {error}"))
    })?;

    let exact_commitment_checks_passed = verify_audit(receipt, response, challenge);
    if !exact_commitment_checks_passed {
        return Ok(ApproximateAuditVerificationReport {
            exact_commitment_checks_passed: false,
            approximate_comparisons: Vec::new(),
            passed: false,
        });
    }

    let mut approximate_comparisons = Vec::new();
    for step in &response.opened_steps {
        let expected = step.approximate_payload.as_ref().ok_or_else(|| {
            VmError::InvalidConfig(format!(
                "missing approximate payload for audited row {}",
                step.index
            ))
        })?;

        let actual = build_auditable_numeric_row(&step.row, replay_witness)?;

        approximate_comparisons.push(compare_numeric_rows(expected, actual, config));
    }

    let passed = approximate_comparisons
        .iter()
        .all(|comparison| comparison.passed);
    Ok(ApproximateAuditVerificationReport {
        exact_commitment_checks_passed,
        approximate_comparisons,
        passed,
    })
}

pub fn build_auditable_numeric_rows(
    staged_trace: &SharedTransformerStagedTrace,
    witness: &SharedTransformerExecutionWitness,
) -> Result<Vec<AuditableNumericTraceRow>> {
    staged_trace
        .rows
        .iter()
        .map(|row| build_auditable_numeric_row(row, witness))
        .collect()
}

fn build_auditable_numeric_row(
    row: &SharedTransformerTraceRow,
    witness: &SharedTransformerExecutionWitness,
) -> Result<AuditableNumericTraceRow> {
    match row {
        SharedTransformerTraceRow::InputState(row) => Ok(AuditableNumericTraceRow::InputState {
            step_number: row.step_number,
        }),
        SharedTransformerTraceRow::Attention(row) => {
            let step = witness_step(witness, row.step_number)?;
            Ok(AuditableNumericTraceRow::Attention {
                step_number: row.step_number,
                input_vector: quantize_named_slice(
                    row.step_number,
                    "input_vector",
                    &step.input_vector,
                )?,
            })
        }
        SharedTransformerTraceRow::FeedForwardChunk(row) => {
            build_feed_forward_numeric_row(row, witness)
        }
        SharedTransformerTraceRow::Transition(row) => build_transition_numeric_row(row, witness),
        SharedTransformerTraceRow::OutputState(row) => Ok(AuditableNumericTraceRow::OutputState {
            step_number: row.step_number,
        }),
    }
}

fn build_feed_forward_numeric_row(
    row: &SharedTransformerFeedForwardChunkTraceRow,
    witness: &SharedTransformerExecutionWitness,
) -> Result<AuditableNumericTraceRow> {
    let step = witness_step(witness, row.step_number)?;
    let end = row.hidden_start + row.hidden_values.len();

    Ok(AuditableNumericTraceRow::FeedForwardChunk {
        step_number: row.step_number,
        chunk_index: row.chunk_index,
        hidden_start: row.hidden_start,
        gate_values: quantize_named_slice(
            row.step_number,
            "ff_gate",
            &step.ff_gate[row.hidden_start..end],
        )?,
        value_values: quantize_named_slice(
            row.step_number,
            "ff_value",
            &step.ff_value[row.hidden_start..end],
        )?,
        hidden_values: quantize_named_slice(
            row.step_number,
            "ff_hidden",
            &step.ff_hidden[row.hidden_start..end],
        )?,
    })
}

fn build_transition_numeric_row(
    row: &SharedTransformerTransitionTraceRow,
    witness: &SharedTransformerExecutionWitness,
) -> Result<AuditableNumericTraceRow> {
    let step = witness_step(witness, row.step_number)?;
    Ok(AuditableNumericTraceRow::Transition {
        step_number: row.step_number,
        ff_output: quantize_named_slice(row.step_number, "ff_output", &step.ff_output)?,
    })
}

fn witness_step<'a>(
    witness: &'a SharedTransformerExecutionWitness,
    step_number: usize,
) -> Result<&'a SharedTransformerStepWitness> {
    witness
        .steps
        .iter()
        .find(|step| step.step_number == step_number)
        .ok_or_else(|| {
            VmError::InvalidConfig(format!(
                "missing witness step {step_number} required for audit payload construction"
            ))
        })
}

fn quantize_named_slice(step_number: usize, name: &str, values: &[f64]) -> Result<Vec<i128>> {
    canonicalize_values(values).map_err(|error| {
        VmError::UnsupportedProof(format!(
            "failed to encode {name} at step {step_number} into the audit numeric domain: {error}"
        ))
    })
}

fn convert_weights_to_numeric(
    weights: &crate::model::FeedForwardWeights,
) -> Result<OpenedFeedForwardWeights> {
    Ok(OpenedFeedForwardWeights {
        gate: quantize_matrix(
            "gate",
            &weights.gate.data,
            weights.gate.rows,
            weights.gate.cols,
        )?,
        gate_bias: quantize_named_slice(0, "gate_bias", &weights.gate_bias)?,
        value: quantize_matrix(
            "value",
            &weights.value.data,
            weights.value.rows,
            weights.value.cols,
        )?,
        value_bias: quantize_named_slice(0, "value_bias", &weights.value_bias)?,
        out: quantize_matrix("out", &weights.out.data, weights.out.rows, weights.out.cols)?,
        out_bias: quantize_named_slice(0, "out_bias", &weights.out_bias)?,
    })
}

fn quantize_matrix(name: &str, data: &[f64], rows: usize, cols: usize) -> Result<Vec<Vec<i128>>> {
    let values = canonicalize_values(data).map_err(|error| {
        VmError::UnsupportedProof(format!(
            "failed to encode {name} weights into the audit numeric domain: {error}"
        ))
    })?;

    Ok(values
        .chunks(cols)
        .take(rows)
        .map(|row| row.to_vec())
        .collect())
}

fn serialize_opened_weights(weights: &OpenedFeedForwardWeights) -> Vec<u8> {
    let mut bytes = Vec::new();
    serialize_matrix(&mut bytes, &weights.gate);
    serialize_vector(&mut bytes, &weights.gate_bias);
    serialize_matrix(&mut bytes, &weights.value);
    serialize_vector(&mut bytes, &weights.value_bias);
    serialize_matrix(&mut bytes, &weights.out);
    serialize_vector(&mut bytes, &weights.out_bias);
    bytes
}

fn serialize_numeric_trace_row(row: &AuditableNumericTraceRow) -> Vec<u8> {
    let mut bytes = Vec::new();
    match row {
        AuditableNumericTraceRow::InputState { step_number } => {
            bytes.push(0);
            serialize_usize(&mut bytes, *step_number);
        }
        AuditableNumericTraceRow::Attention {
            step_number,
            input_vector,
        } => {
            bytes.push(1);
            serialize_usize(&mut bytes, *step_number);
            serialize_vector(&mut bytes, input_vector);
        }
        AuditableNumericTraceRow::FeedForwardChunk {
            step_number,
            chunk_index,
            hidden_start,
            gate_values,
            value_values,
            hidden_values,
        } => {
            bytes.push(2);
            serialize_usize(&mut bytes, *step_number);
            serialize_usize(&mut bytes, *chunk_index);
            serialize_usize(&mut bytes, *hidden_start);
            serialize_vector(&mut bytes, gate_values);
            serialize_vector(&mut bytes, value_values);
            serialize_vector(&mut bytes, hidden_values);
        }
        AuditableNumericTraceRow::Transition {
            step_number,
            ff_output,
        } => {
            bytes.push(3);
            serialize_usize(&mut bytes, *step_number);
            serialize_vector(&mut bytes, ff_output);
        }
        AuditableNumericTraceRow::OutputState { step_number } => {
            bytes.push(4);
            serialize_usize(&mut bytes, *step_number);
        }
    }
    bytes
}

fn serialize_matrix(bytes: &mut Vec<u8>, matrix: &[Vec<i128>]) {
    serialize_usize(bytes, matrix.len());
    let cols = matrix.first().map_or(0, Vec::len);
    serialize_usize(bytes, cols);
    for row in matrix {
        serialize_vector(bytes, row);
    }
}

fn serialize_vector(bytes: &mut Vec<u8>, values: &[i128]) {
    serialize_usize(bytes, values.len());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn serialize_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn compare_numeric_rows(
    expected: &AuditableNumericTraceRow,
    actual: AuditableNumericTraceRow,
    config: &ApproximateVerificationConfig,
) -> ApproximateComparisonResult {
    if !rows_are_compatible(expected, &actual) {
        let total_values = flattened_numeric_values(expected)
            .len()
            .max(flattened_numeric_values(&actual).len());
        return ApproximateComparisonResult {
            total_values,
            deviating_values: total_values,
            max_absolute_error: f64::INFINITY,
            max_relative_error: f64::INFINITY,
            passed: false,
        };
    }

    let expected_values = flattened_numeric_values(expected);
    let actual_values = flattened_numeric_values(&actual);
    compare_approximate(&expected_values, &actual_values, config)
}

fn rows_are_compatible(
    expected: &AuditableNumericTraceRow,
    actual: &AuditableNumericTraceRow,
) -> bool {
    match (expected, actual) {
        (
            AuditableNumericTraceRow::InputState { step_number: left },
            AuditableNumericTraceRow::InputState { step_number: right },
        ) => left == right,
        (
            AuditableNumericTraceRow::Attention {
                step_number: left_step,
                input_vector: left_values,
            },
            AuditableNumericTraceRow::Attention {
                step_number: right_step,
                input_vector: right_values,
            },
        ) => left_step == right_step && left_values.len() == right_values.len(),
        (
            AuditableNumericTraceRow::FeedForwardChunk {
                step_number: left_step,
                chunk_index: left_chunk,
                hidden_start: left_start,
                gate_values: left_gate,
                value_values: left_value,
                hidden_values: left_hidden,
            },
            AuditableNumericTraceRow::FeedForwardChunk {
                step_number: right_step,
                chunk_index: right_chunk,
                hidden_start: right_start,
                gate_values: right_gate,
                value_values: right_value,
                hidden_values: right_hidden,
            },
        ) => {
            left_step == right_step
                && left_chunk == right_chunk
                && left_start == right_start
                && left_gate.len() == right_gate.len()
                && left_value.len() == right_value.len()
                && left_hidden.len() == right_hidden.len()
        }
        (
            AuditableNumericTraceRow::Transition {
                step_number: left_step,
                ff_output: left_output,
            },
            AuditableNumericTraceRow::Transition {
                step_number: right_step,
                ff_output: right_output,
            },
        ) => left_step == right_step && left_output.len() == right_output.len(),
        (
            AuditableNumericTraceRow::OutputState { step_number: left },
            AuditableNumericTraceRow::OutputState { step_number: right },
        ) => left == right,
        _ => false,
    }
}

fn flattened_numeric_values(row: &AuditableNumericTraceRow) -> Vec<f64> {
    match row {
        AuditableNumericTraceRow::InputState { .. } => Vec::new(),
        AuditableNumericTraceRow::Attention { input_vector, .. } => input_vector
            .iter()
            .copied()
            .map(dequantize_signed)
            .collect(),
        AuditableNumericTraceRow::FeedForwardChunk {
            gate_values,
            value_values,
            hidden_values,
            ..
        } => gate_values
            .iter()
            .chain(value_values.iter())
            .chain(hidden_values.iter())
            .copied()
            .map(dequantize_signed)
            .collect(),
        AuditableNumericTraceRow::Transition { ff_output, .. } => {
            ff_output.iter().copied().map(dequantize_signed).collect()
        }
        AuditableNumericTraceRow::OutputState { .. } => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerVmConfig;
    use crate::instruction::{Instruction, Program};
    use crate::model::{DispatchInfo, SharedTransformerTransitionWitness};

    fn sample_witness(config: &TransformerVmConfig) -> SharedTransformerExecutionWitness {
        let initial_state = MachineState::new(1);
        SharedTransformerExecutionWitness {
            initial_state: initial_state.clone(),
            steps: vec![SharedTransformerStepWitness {
                step_number: 0,
                pre_state: initial_state.clone(),
                input_token: vec![0.0; config.d_model],
                decoded_state: initial_state.clone(),
                dispatch: DispatchInfo {
                    instruction: Instruction::Halt,
                    layer_idx: 0,
                },
                operand: 0,
                input_vector: vec![1.0; 9 + 2 * crate::state::WORD_BITS],
                ff_gate: vec![2.0; 256],
                ff_value: vec![3.0; 256],
                ff_hidden: vec![6.0; 256],
                ff_output: vec![4.0; 6],
                transition: SharedTransformerTransitionWitness {
                    pc: 0,
                    acc: 0,
                    sp: 0,
                    zero_flag: true,
                    carry_flag: false,
                    halted: true,
                    memory_write: None,
                },
                post_write_state: initial_state.clone(),
                output_token: vec![0.0; config.d_model],
                final_state: initial_state.clone(),
            }],
            final_state: initial_state,
        }
    }

    #[test]
    fn approximate_report_requires_matching_row_shape() {
        let left = AuditableNumericTraceRow::Attention {
            step_number: 0,
            input_vector: vec![1, 2],
        };
        let right = AuditableNumericTraceRow::Transition {
            step_number: 0,
            ff_output: vec![1, 2],
        };
        let config = ApproximateVerificationConfig {
            absolute_epsilon: 0.1,
            relative_epsilon: 0.1,
            max_deviating_fraction: 0.0,
        };

        let result = compare_numeric_rows(&left, right, &config);
        assert!(!result.passed);
    }

    #[test]
    fn commit_execution_populates_numeric_trace_root() {
        let config = TransformerVmConfig {
            num_layers: 1,
            ff_dim: 256,
            ..TransformerVmConfig::default()
        };
        let program = Program::new(vec![Instruction::Halt], 1);
        let vm = TransformerVm::new(config.clone(), program).expect("model");
        let witness = sample_witness(&config);

        let (receipt, _, _) = commit_execution(&vm, &witness).expect("commit");
        assert!(!receipt.numeric_trace_root.is_empty());
    }
}
