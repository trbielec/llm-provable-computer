use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::config::{Attention2DMode, TransformerVmConfig};
use crate::error::{Result, VmError};
use crate::instruction::Program;
use crate::memory::AddressedMemory;
use crate::model::{
    build_input_vector, checked_transition_u8, transition_from_output, validate_stack_pointer,
    CarryBlend, CompiledInstruction, DispatchInfo, SharedTransformerTransitionWitness,
    TransformerVm, TransitionControls, INPUT_DIM, OUTPUT_DIM,
};
use crate::proof::VanillaStarkProofOptions;
use crate::runtime::ExecutionRuntime;
use crate::state::{decode_state, encode_state, MachineState, Word, WORD_BITS};
use crate::transformer_trace::{
    build_shared_transformer_staged_trace, SharedTransformerFeedForwardChunkTraceRow,
    SharedTransformerOutputTraceRow, SharedTransformerStagedTrace, SharedTransformerTraceRow,
    SharedTransformerTransitionTraceRow,
};
use crate::vanillastark::field::FieldElement;
use crate::vanillastark::multivariate::MPolynomial;
use crate::vanillastark::stark::{Stark, StarkProofProfile};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerTraceValidationReport {
    pub validated_steps: usize,
    pub row_count: usize,
    pub ff_chunk_size: usize,
}

pub fn validate_shared_transformer_staged_trace(
    model: &TransformerVm,
    trace: &SharedTransformerStagedTrace,
) -> Result<SharedTransformerTraceValidationReport> {
    let mut cursor = 0usize;
    let mut expected_step_number = 1usize;
    let mut expected_pre_state =
        MachineState::with_memory(model.program().initial_memory().to_vec());

    while cursor < trace.rows.len() {
        let input_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::InputState(row)) => row,
            Some(other) => {
                return Err(mismatch(
                    expected_step_number,
                    format!("expected input-state row, got {:?}", other.stage()),
                ))
            }
            None => unreachable!("cursor is bounded by rows.len()"),
        };
        if input_row.step_number != expected_step_number {
            return Err(mismatch(
                expected_step_number,
                format!(
                    "unexpected step number {}; expected {}",
                    input_row.step_number, expected_step_number
                ),
            ));
        }
        if input_row.pre_state != expected_pre_state {
            return Err(mismatch(
                expected_step_number,
                "pre-state does not chain from the previous staged step",
            ));
        }

        validate_input_row(model, input_row)?;
        cursor += 1;

        let attention_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::Attention(row)) => row,
            Some(other) => {
                return Err(mismatch(
                    expected_step_number,
                    format!("expected attention row, got {:?}", other.stage()),
                ))
            }
            None => {
                return Err(mismatch(
                    expected_step_number,
                    "trace ended before attention row",
                ))
            }
        };
        validate_attention_row(input_row.decoded_state.clone(), attention_row)?;
        cursor += 1;

        let mut ff_chunks = Vec::new();
        while let Some(SharedTransformerTraceRow::FeedForwardChunk(row)) = trace.rows.get(cursor) {
            ff_chunks.push(row.clone());
            cursor += 1;
        }
        validate_feed_forward_rows(expected_step_number, trace.ff_chunk_size, &ff_chunks)?;

        let transition_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::Transition(row)) => row,
            Some(other) => {
                return Err(mismatch(
                    expected_step_number,
                    format!("expected transition row, got {:?}", other.stage()),
                ))
            }
            None => {
                return Err(mismatch(
                    expected_step_number,
                    "trace ended before transition row",
                ))
            }
        };
        cursor += 1;

        let output_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::OutputState(row)) => row,
            Some(other) => {
                return Err(mismatch(
                    expected_step_number,
                    format!("expected output-state row, got {:?}", other.stage()),
                ))
            }
            None => {
                return Err(mismatch(
                    expected_step_number,
                    "trace ended before output-state row",
                ))
            }
        };
        cursor += 1;

        let next_state = validate_transition_and_output_rows(
            model,
            input_row.dispatch,
            input_row.decoded_state.clone(),
            attention_row.operand,
            ff_chunks,
            transition_row,
            output_row,
        )?;
        expected_pre_state = next_state;
        expected_step_number += 1;
    }

    Ok(SharedTransformerTraceValidationReport {
        validated_steps: expected_step_number - 1,
        row_count: trace.rows.len(),
        ff_chunk_size: trace.ff_chunk_size,
    })
}

fn validate_input_row(
    model: &TransformerVm,
    row: &crate::transformer_trace::SharedTransformerInputTraceRow,
) -> Result<()> {
    let encoded = encode_state(&row.pre_state, model.config().d_model)?;
    let expected_token = encoded
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_token_to_i8(row.step_number, "input_token", index, value))
        .collect::<Result<Vec<_>>>()?;
    if expected_token != row.input_token {
        return Err(mismatch(
            row.step_number,
            "input token does not match shared encode_state semantics",
        ));
    }

    let decoded = decode_state(&encoded, row.pre_state.memory.clone())?;
    if decoded != row.decoded_state {
        return Err(mismatch(
            row.step_number,
            "decoded input state does not match shared decode_state semantics",
        ));
    }

    let dispatch = model.dispatch_info(decoded.pc)?;
    if dispatch != row.dispatch {
        return Err(mismatch(
            row.step_number,
            "dispatch info does not match shared dispatch_info semantics",
        ));
    }

    Ok(())
}

fn validate_attention_row(
    decoded_state: MachineState,
    row: &crate::transformer_trace::SharedTransformerAttentionTraceRow,
) -> Result<()> {
    let expected = build_input_vector(&decoded_state, row.operand);
    if expected.len() != INPUT_DIM || row.input_vector.len() != INPUT_DIM {
        return Err(mismatch(
            row.step_number,
            format!(
                "attention input vector width mismatch: expected {}, got {}",
                INPUT_DIM,
                row.input_vector.len()
            ),
        ));
    }
    let expected_words = expected
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_scalar_to_word(row.step_number, "input_vector", index, value))
        .collect::<Result<Vec<_>>>()?;
    if expected_words != row.input_vector {
        return Err(mismatch(
            row.step_number,
            "attention row does not match shared build_input_vector semantics",
        ));
    }
    Ok(())
}

fn validate_feed_forward_rows(
    step_number: usize,
    ff_chunk_size: usize,
    rows: &[SharedTransformerFeedForwardChunkTraceRow],
) -> Result<()> {
    if rows.is_empty() {
        return Err(mismatch(
            step_number,
            "expected at least one feed-forward chunk row",
        ));
    }

    for (expected_chunk_index, row) in rows.iter().enumerate() {
        if row.step_number != step_number {
            return Err(mismatch(
                step_number,
                format!(
                    "feed-forward chunk row has mismatched step number {}",
                    row.step_number
                ),
            ));
        }
        if row.chunk_index != expected_chunk_index {
            return Err(mismatch(
                step_number,
                format!(
                    "feed-forward chunk index {} is out of order; expected {}",
                    row.chunk_index, expected_chunk_index
                ),
            ));
        }
        if row.hidden_start != expected_chunk_index * ff_chunk_size {
            return Err(mismatch(
                step_number,
                format!(
                    "feed-forward chunk {} starts at {}, expected {}",
                    row.chunk_index,
                    row.hidden_start,
                    expected_chunk_index * ff_chunk_size
                ),
            ));
        }
    }

    Ok(())
}

fn validate_transition_and_output_rows(
    model: &TransformerVm,
    dispatch: DispatchInfo,
    decoded_state: MachineState,
    operand: Word,
    ff_chunks: Vec<SharedTransformerFeedForwardChunkTraceRow>,
    transition_row: &SharedTransformerTransitionTraceRow,
    output_row: &SharedTransformerOutputTraceRow,
) -> Result<MachineState> {
    let (compiled, layer_idx) = model.compiled_instruction(decoded_state.pc)?;
    if layer_idx != dispatch.layer_idx {
        return Err(mismatch(
            transition_row.step_number,
            "compiled instruction layer does not match dispatch layer",
        ));
    }

    let input_vector = build_input_vector(&decoded_state, operand);
    let evaluation = compiled.ff_weights.evaluate_with_hidden(&input_vector);

    let expected_hidden = evaluation
        .hidden
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            exact_scalar_to_word(transition_row.step_number, "ff_hidden", index, value)
        })
        .collect::<Result<Vec<_>>>()?;
    let actual_hidden = flatten_hidden_chunks(&ff_chunks);
    if expected_hidden != actual_hidden {
        return Err(mismatch(
            transition_row.step_number,
            "feed-forward chunk rows do not match shared FF hidden activations",
        ));
    }
    let expected_gate = evaluation
        .gate
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            exact_scalar_to_word(transition_row.step_number, "ff_gate", index, value)
        })
        .collect::<Result<Vec<_>>>()?;
    let actual_gate = flatten_gate_chunks(&ff_chunks);
    if expected_gate != actual_gate {
        return Err(mismatch(
            transition_row.step_number,
            "feed-forward chunk rows do not match shared FF gate activations",
        ));
    }
    let expected_value = evaluation
        .value
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            exact_scalar_to_word(transition_row.step_number, "ff_value", index, value)
        })
        .collect::<Result<Vec<_>>>()?;
    let actual_value = flatten_value_chunks(&ff_chunks);
    if expected_value != actual_value {
        return Err(mismatch(
            transition_row.step_number,
            "feed-forward chunk rows do not match shared FF value activations",
        ));
    }

    let expected_output = evaluation
        .output
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            exact_trace_scalar_to_i128(transition_row.step_number, "ff_output", index, value)
        })
        .collect::<Result<Vec<_>>>()?;
    if transition_row.ff_output.len() != OUTPUT_DIM || transition_row.ff_output != expected_output {
        return Err(mismatch(
            transition_row.step_number,
            "transition row FF output does not match shared FF output semantics",
        ));
    }

    let expected_transition: SharedTransformerTransitionWitness = transition_from_output(
        &decoded_state,
        operand,
        &compiled.controls,
        &evaluation.output,
    )
    .into();
    if transition_row.transition != expected_transition {
        return Err(mismatch(
            transition_row.step_number,
            "transition row does not match shared transition materialization semantics",
        ));
    }

    let expected_post_write_state = apply_transition(
        transition_row.step_number,
        &decoded_state,
        &transition_row.transition,
    )?;
    if transition_row.post_write_state != expected_post_write_state {
        return Err(mismatch(
            transition_row.step_number,
            "post-write state does not match shared writeback semantics",
        ));
    }

    let encoded_output = encode_state(&expected_post_write_state, model.config().d_model)?;
    let expected_output_token = encoded_output
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            exact_token_to_i8(output_row.step_number, "output_token", index, value)
        })
        .collect::<Result<Vec<_>>>()?;
    if output_row.output_token != expected_output_token {
        return Err(mismatch(
            output_row.step_number,
            "output token does not match shared encode_state semantics",
        ));
    }

    let expected_final_state =
        decode_state(&encoded_output, expected_post_write_state.memory.clone())?;
    if output_row.final_state != expected_final_state {
        return Err(mismatch(
            output_row.step_number,
            "final state does not match shared decode_state semantics",
        ));
    }

    Ok(expected_final_state)
}

fn flatten_hidden_chunks(rows: &[SharedTransformerFeedForwardChunkTraceRow]) -> Vec<Word> {
    let mut values = Vec::new();
    for row in rows {
        values.extend_from_slice(&row.hidden_values);
    }
    values
}

fn flatten_gate_chunks(rows: &[SharedTransformerFeedForwardChunkTraceRow]) -> Vec<Word> {
    let mut values = Vec::new();
    for row in rows {
        values.extend_from_slice(&row.gate_values);
    }
    values
}

fn flatten_value_chunks(rows: &[SharedTransformerFeedForwardChunkTraceRow]) -> Vec<Word> {
    let mut values = Vec::new();
    for row in rows {
        values.extend_from_slice(&row.value_values);
    }
    values
}

fn apply_transition(
    step_number: usize,
    pre_state: &MachineState,
    transition: &SharedTransformerTransitionWitness,
) -> Result<MachineState> {
    let mut memory = AddressedMemory::from_initial(&pre_state.memory);
    if let Some((address, value)) = transition.memory_write {
        let address = checked_transition_u8("memory address", address)?;
        memory.store(address, value, step_number)?;
    }

    let next_pc = checked_transition_u8("pc", transition.pc)?;
    let next_sp = checked_transition_u8("sp", transition.sp)?;
    validate_stack_pointer(next_sp, memory.len())?;

    Ok(MachineState {
        pc: next_pc,
        acc: transition.acc,
        sp: next_sp,
        zero_flag: transition.zero_flag,
        carry_flag: transition.carry_flag,
        halted: transition.halted,
        memory: memory.snapshot(),
    })
}

fn exact_scalar_to_word(
    step_number: usize,
    location: &str,
    index: usize,
    value: f64,
) -> Result<Word> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer exact-domain check failed; {location}[{index}] at step {step_number} was {value}"
        )));
    }
    if value < Word::MIN as f64 || value > Word::MAX as f64 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer exact-domain value out of range; {location}[{index}] at step {step_number} was {value}"
        )));
    }
    Ok(value as Word)
}

fn exact_token_to_i8(step_number: usize, location: &str, index: usize, value: f32) -> Result<i8> {
    match value {
        -1.0 => Ok(-1),
        0.0 => Ok(0),
        1.0 => Ok(1),
        _ => Err(VmError::UnsupportedProof(format!(
            "shared transformer token exact-domain check failed; {location}[{index}] at step {step_number} was {value}"
        ))),
    }
}

fn mismatch(step: usize, message: impl Into<String>) -> VmError {
    VmError::ExecutionMismatch {
        step,
        message: message.into(),
    }
}

const SKELETON_STAGE: usize = 0;
const SKELETON_STEP: usize = 1;
const SKELETON_LAYER: usize = 2;
const SKELETON_PRE_PC: usize = 3;
const SKELETON_PRE_ACC: usize = 4;
const SKELETON_PRE_SP: usize = 5;
const SKELETON_PRE_ZERO: usize = 6;
const SKELETON_PRE_CARRY: usize = 7;
const SKELETON_PRE_HALTED: usize = 8;
const SKELETON_OPERAND: usize = 9;
const SKELETON_PRE_ACC_BITS_START: usize = 10;
const SKELETON_OPERAND_BITS_START: usize = SKELETON_PRE_ACC_BITS_START + WORD_BITS;
const SKELETON_POST_PC: usize = SKELETON_OPERAND_BITS_START + WORD_BITS;
const SKELETON_POST_ACC: usize = SKELETON_POST_PC + 1;
const SKELETON_POST_SP: usize = SKELETON_POST_ACC + 1;
const SKELETON_POST_ZERO: usize = SKELETON_POST_SP + 1;
const SKELETON_POST_CARRY: usize = SKELETON_POST_ZERO + 1;
const SKELETON_POST_HALTED: usize = SKELETON_POST_CARRY + 1;
const SKELETON_POST_ACC_INV: usize = SKELETON_POST_HALTED + 1;
const SKELETON_POST_ACC_BITS_START: usize = SKELETON_POST_ACC_INV + 1;
const SKELETON_FF_OUTPUT_START: usize = SKELETON_POST_ACC_BITS_START + WORD_BITS;
const SKELETON_MEM_WRITE_ENABLE: usize = SKELETON_FF_OUTPUT_START + 6; // Using literal 6 as OUTPUT_DIM is 6
const SKELETON_MEM_WRITE_ADDR: usize = SKELETON_MEM_WRITE_ENABLE + 1;
const SKELETON_MEM_WRITE_VALUE: usize = SKELETON_MEM_WRITE_ADDR + 1;
const SKELETON_BASE_REGISTER_COUNT: usize = SKELETON_MEM_WRITE_VALUE + 1;
const SKELETON_DIGEST_BETA: i128 = 1_315_423_911;
const SKELETON_LINEAR_FEATURE_CHUNK_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedTransformerStagedSkeletonClaim {
    pub program: Program,
    pub config: TransformerVmConfig,
    pub ff_chunk_size: usize,
    pub steps: usize,
    pub final_state: MachineState,
    pub first_row_digest: u128,
    pub trace_digest: u128,
    pub options: VanillaStarkProofOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedTransformerStagedSkeletonProof {
    pub claim: SharedTransformerStagedSkeletonClaim,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerStagedSkeletonCostEstimate {
    pub steps: usize,
    pub trace_rows: usize,
    pub register_count: usize,
    pub memory_size: usize,
    pub d_model: usize,
    pub ff_chunk_rows: usize,
    pub linear_feature_chunk_rows: usize,
    pub hidden_chunk_width: usize,
    pub memory_column_count: usize,
    pub token_column_count: usize,
    pub input_feature_column_count: usize,
    pub chunk_value_column_count: usize,
    pub output_accumulator_column_count: usize,
    pub linear_accumulator_column_count: usize,
    pub output_slot_accumulator_column_count: usize,
    pub compare_column_count: usize,
    pub raw_acc_column_count: usize,
    pub transition_constraint_count: usize,
    pub transition_degree_bound: usize,
    pub num_randomizers: usize,
    pub omicron_domain_length: usize,
    pub fri_domain_length: usize,
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub security_level: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerStagedSkeletonProvingProfile {
    pub witness_build_micros: u64,
    pub staged_trace_build_micros: u64,
    pub trace_validation_micros: u64,
    pub skeleton_trace_build_micros: u64,
    pub air_build_micros: u64,
    pub boundary_build_micros: u64,
    pub stark: StarkProofProfile,
    pub total_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedTransformerStagedSkeletonProvingReport {
    pub proof: SharedTransformerStagedSkeletonProof,
    pub profile: SharedTransformerStagedSkeletonProvingProfile,
}

pub type SharedTransformerForwardPassClaim = SharedTransformerStagedSkeletonClaim;
pub type SharedTransformerForwardPassProof = SharedTransformerStagedSkeletonProof;
pub type SharedTransformerForwardPassCostEstimate = SharedTransformerStagedSkeletonCostEstimate;
pub type SharedTransformerForwardPassProvingProfile = SharedTransformerStagedSkeletonProvingProfile;
pub type SharedTransformerForwardPassProvingReport = SharedTransformerStagedSkeletonProvingReport;

#[derive(Debug, Clone)]
struct ExactMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i128>,
}

#[derive(Debug, Clone)]
struct ExactFeedForwardWeights {
    gate: ExactMatrix,
    gate_bias: Vec<i128>,
    value: ExactMatrix,
    value_bias: Vec<i128>,
    out: ExactMatrix,
    out_bias: Vec<i128>,
}

#[derive(Debug, Clone)]
struct ExactCompiledInstruction {
    ff_weights: ExactFeedForwardWeights,
    controls: ExactTransitionControls,
}

#[derive(Debug, Clone)]
struct ExactTransitionControls {
    zero: ExactBoolBlend,
    carry: ExactCarryBlend,
    halted: ExactBoolBlend,
}

#[derive(Debug, Clone)]
struct ExactBoolBlend {
    prev_weight: i128,
    result_weight: i128,
    constant: i128,
}

#[derive(Debug, Clone)]
struct ExactCarryBlend {
    prev_weight: i128,
    overflow_weight: i128,
    less_than_weight: i128,
    constant: i128,
    rhs_constant: Word,
    rhs_uses_operand: bool,
}

struct SharedTransformerStagedSkeletonLayout {
    memory_size: usize,
    d_model: usize,
    hidden_chunk_width: usize,
}

impl SharedTransformerStagedSkeletonLayout {
    fn new(memory_size: usize, d_model: usize, hidden_chunk_width: usize) -> Self {
        Self {
            memory_size,
            d_model,
            hidden_chunk_width,
        }
    }

    fn pre_mem(&self, index: usize) -> usize {
        SKELETON_BASE_REGISTER_COUNT + index
    }

    fn post_mem(&self, index: usize) -> usize {
        SKELETON_BASE_REGISTER_COUNT + self.memory_size + index
    }

    fn chunk_index(&self) -> usize {
        SKELETON_BASE_REGISTER_COUNT + 2 * self.memory_size
    }

    fn hidden_start(&self) -> usize {
        self.chunk_index() + 1
    }

    fn output_accumulator(&self, index: usize) -> usize {
        self.hidden_start() + 1 + index
    }

    fn gate_linear_accumulator(&self, slot: usize) -> usize {
        self.output_accumulator(OUTPUT_DIM) + slot
    }

    fn value_linear_accumulator(&self, slot: usize) -> usize {
        self.gate_linear_accumulator(self.hidden_chunk_width) + slot
    }

    fn raw_acc_bit(&self, index: usize) -> usize {
        self.value_linear_accumulator(self.hidden_chunk_width) + index
    }

    fn raw_acc_in_range(&self) -> usize {
        self.raw_acc_bit(128)
    }

    fn raw_acc_range_inv(&self) -> usize {
        self.raw_acc_in_range() + 1
    }

    fn digest(&self) -> usize {
        self.raw_acc_range_inv() + 1
    }

    fn register_count(&self) -> usize {
        self.digest() + 1
    }
}

struct SharedTransformerStagedSkeletonAir {
    program: Program,
    config: TransformerVmConfig,
    ff_chunk_rows: usize,
    linear_feature_chunk_rows: usize,
    rows_per_step: usize,
    hidden_chunk_width: usize,
    layout: SharedTransformerStagedSkeletonLayout,
    compiled_instructions: Vec<ExactCompiledInstruction>,
}

impl SharedTransformerStagedSkeletonAir {
    fn new(program: Program, config: TransformerVmConfig, ff_chunk_size: usize) -> Result<Self> {
        let ff_chunk_rows = config.ff_dim.div_ceil(ff_chunk_size);
        let linear_feature_chunk_rows = INPUT_DIM.div_ceil(SKELETON_LINEAR_FEATURE_CHUNK_SIZE);
        let rows_per_step = ff_chunk_rows * linear_feature_chunk_rows + 4;
        let hidden_chunk_width = ff_chunk_size.min(config.ff_dim);
        let layout = SharedTransformerStagedSkeletonLayout::new(
            program.memory_size(),
            config.d_model,
            hidden_chunk_width,
        );
        let model = TransformerVm::new(config.clone(), program.clone())?;
        let mut compiled_instructions = Vec::with_capacity(program.len());
        for pc in 0..program.len() {
            let (compiled, _) = model.compiled_instruction(pc as u8)?;
            compiled_instructions.push(ExactCompiledInstruction::from_compiled(compiled)?);
        }
        Ok(Self {
            program,
            config,
            ff_chunk_rows,
            linear_feature_chunk_rows,
            rows_per_step,
            hidden_chunk_width,
            layout,
            compiled_instructions,
        })
    }

    fn register_count(&self) -> usize {
        self.layout.register_count()
    }

    fn chunk_stage_index(&self, chunk_index: usize, feature_chunk: usize) -> usize {
        2 + chunk_index * self.linear_feature_chunk_rows + feature_chunk
    }

    fn transition_stage_index(&self) -> usize {
        2 + self.ff_chunk_rows * self.linear_feature_chunk_rows
    }

    fn transition_degree_bound(&self) -> usize {
        self.transition_constraints()
            .iter()
            .flat_map(|constraint| constraint.dictionary.keys())
            .map(|powers| powers.iter().sum())
            .max()
            .unwrap_or(1)
            .max(2)
    }

    fn boundary_constraints(
        &self,
        claim: &SharedTransformerStagedSkeletonClaim,
    ) -> Vec<(usize, usize, FieldElement)> {
        let mut boundary = Vec::new();
        let initial = MachineState::with_memory(self.program.initial_memory().to_vec());
        let last_row = claim.steps * self.rows_per_step - 1;

        boundary.push((0, SKELETON_STAGE, field_from_i128(0)));
        boundary.push((0, SKELETON_STEP, field_from_i128(1)));
        push_state_boundary(&mut boundary, 0, &initial, true);
        push_memory_boundary(&mut boundary, 0, &initial.memory, &self.layout, true);
        boundary.push((
            0,
            self.layout.digest(),
            field_from_u128(claim.first_row_digest),
        ));

        boundary.push((
            last_row,
            SKELETON_STAGE,
            field_from_i128((self.rows_per_step - 1) as i128),
        ));
        boundary.push((
            last_row,
            SKELETON_STEP,
            field_from_i128(claim.steps as i128),
        ));
        push_state_boundary(&mut boundary, last_row, &claim.final_state, false);
        push_memory_boundary(
            &mut boundary,
            last_row,
            &claim.final_state.memory,
            &self.layout,
            false,
        );
        boundary.push((
            last_row,
            self.layout.digest(),
            field_from_u128(claim.trace_digest),
        ));

        boundary
    }

    fn transition_constraints(&self) -> Vec<MPolynomial> {
        let register_count = self.register_count();
        let variables = MPolynomial::variables(1 + 2 * register_count);
        let current = variables[1..1 + register_count].to_vec();
        let next = variables[1 + register_count..].to_vec();
        let one = mp_constant(1);
        let current_stage = current[SKELETON_STAGE].clone();
        let current_step = current[SKELETON_STEP].clone();
        let current_layer = current[SKELETON_LAYER].clone();
        let current_pre_pc = current[SKELETON_PRE_PC].clone();
        let current_pre_acc = current[SKELETON_PRE_ACC].clone();
        let current_pre_sp = current[SKELETON_PRE_SP].clone();
        let current_pre_zero = current[SKELETON_PRE_ZERO].clone();
        let current_pre_carry = current[SKELETON_PRE_CARRY].clone();
        let current_pre_halted = current[SKELETON_PRE_HALTED].clone();
        let current_operand = current[SKELETON_OPERAND].clone();
        let current_pre_acc_bits: Vec<MPolynomial> = (0..WORD_BITS)
            .map(|bit| current[SKELETON_PRE_ACC_BITS_START + bit].clone())
            .collect();
        let current_operand_bits: Vec<MPolynomial> = (0..WORD_BITS)
            .map(|bit| current[SKELETON_OPERAND_BITS_START + bit].clone())
            .collect();
        let current_post_pc = current[SKELETON_POST_PC].clone();
        let current_post_acc = current[SKELETON_POST_ACC].clone();
        let current_post_sp = current[SKELETON_POST_SP].clone();
        let current_post_zero = current[SKELETON_POST_ZERO].clone();
        let current_post_carry = current[SKELETON_POST_CARRY].clone();
        let current_post_halted = current[SKELETON_POST_HALTED].clone();
        let current_post_acc_inv = current[SKELETON_POST_ACC_INV].clone();
        let current_chunk_index = current[self.layout.chunk_index()].clone();
        let current_hidden_start = current[self.layout.hidden_start()].clone();
        let current_output_accumulators: Vec<MPolynomial> = (0..OUTPUT_DIM)
            .map(|index| current[self.layout.output_accumulator(index)].clone())
            .collect();
        let current_gate_linear_accumulators: Vec<MPolynomial> = (0..self.hidden_chunk_width)
            .map(|slot| current[self.layout.gate_linear_accumulator(slot)].clone())
            .collect();
        let current_value_linear_accumulators: Vec<MPolynomial> = (0..self.hidden_chunk_width)
            .map(|slot| current[self.layout.value_linear_accumulator(slot)].clone())
            .collect();
        let current_raw_acc_bits: Vec<MPolynomial> = (0..128)
            .map(|index| current[self.layout.raw_acc_bit(index)].clone())
            .collect();
        let current_raw_acc_in_range = current[self.layout.raw_acc_in_range()].clone();
        let current_raw_acc_range_inv = current[self.layout.raw_acc_range_inv()].clone();
        let current_mem_write_enable = current[SKELETON_MEM_WRITE_ENABLE].clone();
        let current_mem_write_addr = current[SKELETON_MEM_WRITE_ADDR].clone();
        let current_mem_write_value = current[SKELETON_MEM_WRITE_VALUE].clone();
        let current_digest = current[self.layout.digest()].clone();
        let current_pre_memory: Vec<MPolynomial> = (0..self.layout.memory_size)
            .map(|index| current[self.layout.pre_mem(index)].clone())
            .collect();
        let current_post_memory: Vec<MPolynomial> = (0..self.layout.memory_size)
            .map(|index| current[self.layout.post_mem(index)].clone())
            .collect();

        let next_stage = next[SKELETON_STAGE].clone();
        let next_step = next[SKELETON_STEP].clone();
        let next_pre_pc = next[SKELETON_PRE_PC].clone();
        let next_pre_acc = next[SKELETON_PRE_ACC].clone();
        let next_pre_sp = next[SKELETON_PRE_SP].clone();
        let next_pre_zero = next[SKELETON_PRE_ZERO].clone();
        let next_pre_carry = next[SKELETON_PRE_CARRY].clone();
        let next_pre_halted = next[SKELETON_PRE_HALTED].clone();
        let next_gate_linear_accumulators: Vec<MPolynomial> = (0..self.hidden_chunk_width)
            .map(|slot| next[self.layout.gate_linear_accumulator(slot)].clone())
            .collect();
        let next_value_linear_accumulators: Vec<MPolynomial> = (0..self.hidden_chunk_width)
            .map(|slot| next[self.layout.value_linear_accumulator(slot)].clone())
            .collect();
        let next_output_accumulators: Vec<MPolynomial> = (0..OUTPUT_DIM)
            .map(|index| next[self.layout.output_accumulator(index)].clone())
            .collect();
        let next_digest = next[self.layout.digest()].clone();

        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let last_selector = lagrange_selector(
            &current_stage,
            &stage_domain,
            (self.rows_per_step - 1) as i128,
        );
        let non_last_selector = one.clone() - last_selector.clone();
        let input_selector = lagrange_selector(&current_stage, &stage_domain, 0);
        let attention_selector = lagrange_selector(&current_stage, &stage_domain, 1);
        let first_chunk_selector = lagrange_selector(&current_stage, &stage_domain, 2);
        let transition_selector = lagrange_selector(
            &current_stage,
            &stage_domain,
            self.transition_stage_index() as i128,
        );
        let chunk_selector = self.feed_forward_chunk_selector(&current_stage);
        let mut first_feature_selector = MPolynomial::zero();
        let mut final_feature_selector = MPolynomial::zero();
        let mut non_final_feature_selector = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            let first_selector = lagrange_selector(
                &current_stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, 0) as i128,
            );
            first_feature_selector = first_feature_selector + first_selector.clone();
            let final_selector = lagrange_selector(
                &current_stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, self.linear_feature_chunk_rows - 1) as i128,
            );
            final_feature_selector = final_feature_selector + final_selector.clone();
            for feature_chunk in 0..(self.linear_feature_chunk_rows - 1) {
                non_final_feature_selector = non_final_feature_selector
                    + lagrange_selector(
                        &current_stage,
                        &stage_domain,
                        self.chunk_stage_index(chunk_index, feature_chunk) as i128,
                    );
            }
        }
        let expected_input_features = self.input_feature_expressions(
            &current_pre_pc,
            &current_pre_acc,
            &current_pre_sp,
            &current_pre_zero,
            &current_pre_carry,
            &current_pre_halted,
            &current_operand,
            &current_pre_acc_bits,
            &current_operand_bits,
        );
        let pc_selectors = self.pc_selectors(&current_pre_pc);
        let compare_rhs_bits =
            self.compare_rhs_bit_expressions(&pc_selectors, &current_operand_bits);
        let expected_chunk_index = self.chunk_index_expression(&current_stage);
        let expected_hidden_start = self.hidden_start_expression(&current_stage);
        let derived_post_acc_zero =
            one.clone() - current_post_acc.clone() * current_post_acc_inv.clone();
        let expected_output_bias: Vec<MPolynomial> = (0..OUTPUT_DIM)
            .map(|index| self.output_bias_expression(index, &pc_selectors))
            .collect();
        let expected_zero =
            self.zero_flag_expression(&pc_selectors, &current_pre_zero, &derived_post_acc_zero);
        let expected_halted = self.halted_flag_expression(&pc_selectors, &current_pre_halted);
        let expected_compare_lt = self.compare_lt_expression(
            &current_pre_acc_bits,
            &compare_rhs_bits,
            &current_raw_acc_bits[..WORD_BITS],
        );
        let expected_overflow = mp_constant(1) - current_raw_acc_in_range.clone();
        let expected_carry = self.carry_flag_expression(
            &pc_selectors,
            &current_pre_carry,
            &expected_overflow,
            &expected_compare_lt,
        );
        let expected_post_acc =
            self.saturated_raw_acc_expression(&current_raw_acc_bits, &current_raw_acc_in_range);

        let mut constraints = vec![
            domain_zerofier(&current_stage, self.rows_per_step),
            domain_zerofier(&current_layer, self.config.num_layers),
            domain_zerofier(&current_pre_pc, self.program.len()),
            domain_zerofier(&current_post_pc, self.program.len()),
            domain_zerofier(&current_pre_sp, self.layout.memory_size + 1),
            domain_zerofier(&current_post_sp, self.layout.memory_size + 1),
            current_pre_zero.clone() * (current_pre_zero.clone() - one.clone()),
            current_pre_carry.clone() * (current_pre_carry.clone() - one.clone()),
            current_pre_halted.clone() * (current_pre_halted.clone() - one.clone()),
            current_post_zero.clone() * (current_post_zero.clone() - one.clone()),
            current_post_carry.clone() * (current_post_carry.clone() - one.clone()),
            current_post_halted.clone() * (current_post_halted.clone() - one.clone()),
            current_mem_write_enable.clone() * (current_mem_write_enable.clone() - one.clone()),
            next_stage.clone()
                - (current_stage.clone() + one.clone()
                    - last_selector.clone() * mp_constant(self.rows_per_step as i128)),
            next_step - (current_step.clone() + last_selector.clone()),
            current_layer - self.layer_expression(&current_pre_pc),
            current_operand.clone()
                - self.operand_expression(&current_pre_pc, &current_pre_sp, &current_pre_memory),
            current_post_pc.clone() - current_output_accumulators[0].clone(),
            current_post_acc.clone() - expected_post_acc,
            current_post_sp.clone() - current_output_accumulators[2].clone(),
            current_mem_write_enable.clone() - current_output_accumulators[3].clone(),
            current_mem_write_addr.clone() - current_output_accumulators[4].clone(),
            current_mem_write_value.clone() - current_output_accumulators[5].clone(),
            current_chunk_index.clone() - expected_chunk_index,
            current_hidden_start.clone() - expected_hidden_start,
            current_post_zero.clone() - expected_zero,
            current_post_halted.clone() - expected_halted,
            current_post_carry.clone() - expected_carry,
            derived_post_acc_zero.clone() * (derived_post_acc_zero.clone() - one.clone()),
            current_post_acc.clone() * derived_post_acc_zero.clone(),
            current_post_acc_inv.clone() * derived_post_acc_zero,
            next_digest
                - (current_digest * mp_constant(SKELETON_DIGEST_BETA)
                    + self.row_hash_expression(&next)),
        ];

        for bit in current_pre_acc_bits
            .iter()
            .chain(current_operand_bits.iter())
        {
            constraints.push(bit.clone() * (bit.clone() - one.clone()));
        }
        for value in &current_raw_acc_bits {
            constraints.push(value.clone() * (value.clone() - one.clone()));
        }
        constraints.push(
            current_raw_acc_in_range.clone() * (current_raw_acc_in_range.clone() - one.clone()),
        );
        constraints.push(current_pre_acc - twos_complement_expression(&current_pre_acc_bits));
        constraints.push(current_operand - twos_complement_expression(&current_operand_bits));
        constraints.push(
            current_output_accumulators[1].clone()
                - twos_complement_i128_expression(&current_raw_acc_bits),
        );

        for slot in 0..self.hidden_chunk_width {
            let first_gate_increment = self.gate_linear_feature_chunk_expression(
                slot,
                0,
                &current_stage,
                &pc_selectors,
                &expected_input_features,
            );
            let first_value_increment = self.value_linear_feature_chunk_expression(
                slot,
                0,
                &current_stage,
                &pc_selectors,
                &expected_input_features,
            );
            constraints.push(
                first_feature_selector.clone()
                    * (current_gate_linear_accumulators[slot].clone()
                        - self.gate_bias_expression(slot, &current_stage, &pc_selectors)
                        - first_gate_increment),
            );
            constraints.push(
                first_feature_selector.clone()
                    * (current_value_linear_accumulators[slot].clone()
                        - self.value_bias_expression(slot, &current_stage, &pc_selectors)
                        - first_value_increment),
            );
            for feature_chunk in 1..self.linear_feature_chunk_rows {
                let previous_feature_selector = MPolynomial::zero()
                    + (0..self.ff_chunk_rows)
                        .map(|chunk_index| {
                            lagrange_selector(
                                &current_stage,
                                &stage_domain,
                                self.chunk_stage_index(chunk_index, feature_chunk - 1) as i128,
                            )
                        })
                        .fold(MPolynomial::zero(), |acc, selector| acc + selector);
                let next_gate_increment = self.gate_linear_feature_chunk_expression(
                    slot,
                    feature_chunk,
                    &next_stage,
                    &pc_selectors,
                    &expected_input_features,
                );
                let next_value_increment = self.value_linear_feature_chunk_expression(
                    slot,
                    feature_chunk,
                    &next_stage,
                    &pc_selectors,
                    &expected_input_features,
                );
                constraints.push(
                    previous_feature_selector.clone()
                        * (next_gate_linear_accumulators[slot].clone()
                            - current_gate_linear_accumulators[slot].clone()
                            - next_gate_increment),
                );
                constraints.push(
                    previous_feature_selector
                        * (next_value_linear_accumulators[slot].clone()
                            - current_value_linear_accumulators[slot].clone()
                            - next_value_increment),
                );
            }
            constraints.push(
                (mp_constant(1) - chunk_selector.clone())
                    * current_gate_linear_accumulators[slot].clone(),
            );
            constraints.push(
                (mp_constant(1) - chunk_selector.clone())
                    * current_value_linear_accumulators[slot].clone(),
            );
        }
        for output_idx in 0..OUTPUT_DIM {
            constraints.push(
                input_selector.clone()
                    * (current_output_accumulators[output_idx].clone()
                        - expected_output_bias[output_idx].clone()),
            );
            constraints.push(
                attention_selector.clone()
                    * (current_output_accumulators[output_idx].clone()
                        - expected_output_bias[output_idx].clone()),
            );
            constraints.push(
                first_chunk_selector.clone()
                    * (current_output_accumulators[output_idx].clone()
                        - expected_output_bias[output_idx].clone()),
            );
            let chunk_contribution = self.chunk_output_contribution_expression(
                output_idx,
                &current_stage,
                &pc_selectors,
                &current_gate_linear_accumulators,
                &current_value_linear_accumulators,
            );
            constraints.push(
                non_final_feature_selector.clone()
                    * (next_output_accumulators[output_idx].clone()
                        - current_output_accumulators[output_idx].clone()),
            );
            constraints.push(
                final_feature_selector.clone()
                    * (next_output_accumulators[output_idx].clone()
                        - current_output_accumulators[output_idx].clone()
                        - chunk_contribution),
            );
            constraints.push(
                transition_selector.clone()
                    * (next_output_accumulators[output_idx].clone()
                        - current_output_accumulators[output_idx].clone()),
            );
        }
        let mut raw_acc_mismatch = MPolynomial::zero();
        for bit in 64..128 {
            let diff = current_raw_acc_bits[bit].clone() - current_raw_acc_bits[63].clone();
            raw_acc_mismatch = raw_acc_mismatch + diff.clone() * diff;
        }
        constraints.push(raw_acc_mismatch.clone() * current_raw_acc_in_range.clone());
        constraints.push(
            raw_acc_mismatch * current_raw_acc_range_inv
                - (one.clone() - current_raw_acc_in_range.clone()),
        );

        if self.layout.memory_size == 0 {
            constraints.push(current_mem_write_enable);
        } else {
            constraints.push(
                current_mem_write_enable.clone()
                    * domain_zerofier(&current_mem_write_addr, self.layout.memory_size),
            );

            let write_selectors = self.write_selectors(&current_mem_write_addr);
            for index in 0..self.layout.memory_size {
                let selector = write_selectors[index].clone();
                let expected = current_pre_memory[index].clone()
                    + current_mem_write_enable.clone()
                        * selector
                        * (current_mem_write_value.clone() - current_pre_memory[index].clone());
                constraints.push(current_post_memory[index].clone() - expected);
            }
        }

        for &index in &self.payload_columns() {
            constraints
                .push(non_last_selector.clone() * (next[index].clone() - current[index].clone()));
        }

        constraints.extend([
            last_selector.clone() * (next_pre_pc - current_post_pc),
            last_selector.clone() * (next_pre_acc - current_post_acc),
            last_selector.clone() * (next_pre_sp - current_post_sp),
            last_selector.clone() * (next_pre_zero - current_post_zero),
            last_selector.clone() * (next_pre_carry - current_post_carry),
            last_selector.clone() * (next_pre_halted - current_post_halted),
        ]);

        for index in 0..self.layout.memory_size {
            constraints.push(
                last_selector.clone()
                    * (next[self.layout.pre_mem(index)].clone()
                        - current[self.layout.post_mem(index)].clone()),
            );
        }

        constraints
    }

    fn payload_columns(&self) -> Vec<usize> {
        let mut columns = Vec::new();
        for index in SKELETON_LAYER..self.layout.chunk_index() {
            columns.push(index);
        }
        for index in self.layout.raw_acc_bit(0)..self.layout.digest() {
            columns.push(index);
        }
        columns
    }

    fn layer_expression(&self, pc: &MPolynomial) -> MPolynomial {
        let chunk_size = self.program.len().div_ceil(self.config.num_layers);
        let domain: Vec<i128> = (0..self.program.len()).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for pc_value in 0..self.program.len() {
            let layer = (pc_value / chunk_size).min(self.config.num_layers - 1);
            expression = expression
                + lagrange_selector(pc, &domain, pc_value as i128) * mp_constant(layer as i128);
        }
        expression
    }

    fn operand_expression(
        &self,
        pc: &MPolynomial,
        sp: &MPolynomial,
        memory: &[MPolynomial],
    ) -> MPolynomial {
        let domain: Vec<i128> = (0..self.program.len()).map(|value| value as i128).collect();
        let stack_read = self.stack_read_expression(sp, memory);
        let mut expression = MPolynomial::zero();
        for (pc_value, instruction) in self.program.instructions().iter().enumerate() {
            let selector = lagrange_selector(pc, &domain, pc_value as i128);
            let operand = match memory_read_for_instruction(*instruction) {
                SharedTransformerMemoryRead::None => MPolynomial::zero(),
                SharedTransformerMemoryRead::Direct(address) => memory[address as usize].clone(),
                SharedTransformerMemoryRead::StackTop => stack_read.clone(),
            };
            expression = expression + selector * operand;
        }
        expression
    }

    fn stack_read_expression(&self, sp: &MPolynomial, memory: &[MPolynomial]) -> MPolynomial {
        let domain: Vec<i128> = (0..=self.layout.memory_size)
            .map(|value| value as i128)
            .collect();
        let mut expression = MPolynomial::zero();
        for (index, cell) in memory.iter().enumerate() {
            expression = expression + lagrange_selector(sp, &domain, index as i128) * cell.clone();
        }
        expression
    }

    fn write_selectors(&self, addr: &MPolynomial) -> Vec<MPolynomial> {
        let domain: Vec<i128> = (0..self.layout.memory_size)
            .map(|value| value as i128)
            .collect();
        (0..self.layout.memory_size)
            .map(|index| lagrange_selector(addr, &domain, index as i128))
            .collect()
    }

    fn pc_selectors(&self, pc: &MPolynomial) -> Vec<MPolynomial> {
        let domain: Vec<i128> = (0..self.program.len()).map(|value| value as i128).collect();
        domain
            .iter()
            .map(|&point| lagrange_selector(pc, &domain, point))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn input_feature_expressions(
        &self,
        pc: &MPolynomial,
        acc: &MPolynomial,
        sp: &MPolynomial,
        zero: &MPolynomial,
        carry: &MPolynomial,
        halted: &MPolynomial,
        operand: &MPolynomial,
        acc_bits: &[MPolynomial],
        operand_bits: &[MPolynomial],
    ) -> Vec<MPolynomial> {
        let mut input = vec![MPolynomial::zero(); INPUT_DIM];
        input[0] = mp_constant(1);
        input[1] = pc.clone();
        input[2] = self.pc_next_expression(pc);
        input[3] = acc.clone();
        input[4] = zero.clone();
        input[5] = carry.clone();
        input[6] = halted.clone();
        input[7] = sp.clone();
        input[8] = operand.clone();

        for bit in 0..WORD_BITS {
            input[9 + bit] = acc_bits[bit].clone();
            input[9 + WORD_BITS + bit] = operand_bits[bit].clone();
        }

        input
    }

    fn pc_next_expression(&self, pc: &MPolynomial) -> MPolynomial {
        let domain: Vec<i128> = (0..self.program.len()).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for pc_value in 0..self.program.len() {
            let next_pc = (pc_value as u8).wrapping_add(1);
            expression = expression
                + lagrange_selector(pc, &domain, pc_value as i128)
                    * mp_constant(i128::from(next_pc));
        }
        expression
    }

    fn chunk_index_expression(&self, stage: &MPolynomial) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            for feature_chunk in 0..self.linear_feature_chunk_rows {
                let selector = lagrange_selector(
                    stage,
                    &stage_domain,
                    self.chunk_stage_index(chunk_index, feature_chunk) as i128,
                );
                expression = expression + selector * mp_constant(chunk_index as i128);
            }
        }
        expression
    }

    fn hidden_start_expression(&self, stage: &MPolynomial) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            for feature_chunk in 0..self.linear_feature_chunk_rows {
                let selector = lagrange_selector(
                    stage,
                    &stage_domain,
                    self.chunk_stage_index(chunk_index, feature_chunk) as i128,
                );
                expression = expression
                    + selector * mp_constant((chunk_index * self.hidden_chunk_width) as i128);
            }
        }
        expression
    }

    fn gate_bias_expression(
        &self,
        slot: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
    ) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            let hidden_index = chunk_index * self.hidden_chunk_width + slot;
            let selector = lagrange_selector(
                stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, 0) as i128,
            );
            if hidden_index >= self.config.ff_dim {
                continue;
            }
            let mut bias = MPolynomial::zero();
            for (pc_idx, pc_selector) in pc_selectors.iter().enumerate() {
                let value = self.compiled_instructions[pc_idx].ff_weights.gate_bias[hidden_index];
                bias = bias + pc_selector.clone() * mp_constant(value);
            }
            expression = expression + selector * bias;
        }
        expression
    }

    fn value_bias_expression(
        &self,
        slot: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
    ) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            let hidden_index = chunk_index * self.hidden_chunk_width + slot;
            let selector = lagrange_selector(
                stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, 0) as i128,
            );
            if hidden_index >= self.config.ff_dim {
                continue;
            }
            let mut bias = MPolynomial::zero();
            for (pc_idx, pc_selector) in pc_selectors.iter().enumerate() {
                let value = self.compiled_instructions[pc_idx].ff_weights.value_bias[hidden_index];
                bias = bias + pc_selector.clone() * mp_constant(value);
            }
            expression = expression + selector * bias;
        }
        expression
    }

    fn gate_linear_feature_chunk_expression(
        &self,
        slot: usize,
        feature_chunk: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
        input_features: &[MPolynomial],
    ) -> MPolynomial {
        self.linear_feature_chunk_expression(
            slot,
            feature_chunk,
            stage,
            pc_selectors,
            input_features,
            true,
        )
    }

    fn value_linear_feature_chunk_expression(
        &self,
        slot: usize,
        feature_chunk: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
        input_features: &[MPolynomial],
    ) -> MPolynomial {
        self.linear_feature_chunk_expression(
            slot,
            feature_chunk,
            stage,
            pc_selectors,
            input_features,
            false,
        )
    }

    fn linear_feature_chunk_expression(
        &self,
        slot: usize,
        feature_chunk: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
        input_features: &[MPolynomial],
        is_gate: bool,
    ) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let feature_start = feature_chunk * SKELETON_LINEAR_FEATURE_CHUNK_SIZE;
        let feature_end = ((feature_chunk + 1) * SKELETON_LINEAR_FEATURE_CHUNK_SIZE).min(INPUT_DIM);
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            let hidden_index = chunk_index * self.hidden_chunk_width + slot;
            let selector = lagrange_selector(
                stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, feature_chunk) as i128,
            );
            if hidden_index >= self.config.ff_dim {
                continue;
            }
            let mut linear = MPolynomial::zero();
            for (pc_idx, pc_selector) in pc_selectors.iter().enumerate() {
                let weights = &self.compiled_instructions[pc_idx].ff_weights;
                let mut chunk_sum = MPolynomial::zero();
                for (feature_index, input_feature) in input_features
                    .iter()
                    .enumerate()
                    .take(feature_end)
                    .skip(feature_start)
                {
                    let coefficient = if is_gate {
                        weights.gate.get(hidden_index, feature_index)
                    } else {
                        weights.value.get(hidden_index, feature_index)
                    };
                    if coefficient == 0 {
                        continue;
                    }
                    chunk_sum = chunk_sum + input_feature.clone() * mp_constant(coefficient);
                }
                linear = linear + pc_selector.clone() * chunk_sum;
            }
            expression = expression + selector * linear;
        }
        expression
    }

    fn output_bias_expression(
        &self,
        output_idx: usize,
        pc_selectors: &[MPolynomial],
    ) -> MPolynomial {
        let mut expression = MPolynomial::zero();
        for (pc_idx, selector) in pc_selectors.iter().enumerate() {
            let bias = self.compiled_instructions[pc_idx].ff_weights.out_bias[output_idx];
            expression = expression + selector.clone() * mp_constant(bias);
        }
        expression
    }

    fn chunk_output_contribution_expression(
        &self,
        output_idx: usize,
        stage: &MPolynomial,
        pc_selectors: &[MPolynomial],
        gate_linear_accumulators: &[MPolynomial],
        value_linear_accumulators: &[MPolynomial],
    ) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut expression = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            let selector = lagrange_selector(
                stage,
                &stage_domain,
                self.chunk_stage_index(chunk_index, self.linear_feature_chunk_rows - 1) as i128,
            );
            let mut chunk_contribution = MPolynomial::zero();
            for slot in 0..self.hidden_chunk_width {
                let hidden_index = chunk_index * self.hidden_chunk_width + slot;
                if hidden_index >= self.config.ff_dim {
                    continue;
                }
                let mut coefficient = MPolynomial::zero();
                for (pc_idx, pc_selector) in pc_selectors.iter().enumerate() {
                    let out_coeff = self.compiled_instructions[pc_idx]
                        .ff_weights
                        .out
                        .get(output_idx, hidden_index);
                    if out_coeff == 0 {
                        continue;
                    }
                    coefficient = coefficient + pc_selector.clone() * mp_constant(out_coeff);
                }
                chunk_contribution = chunk_contribution
                    + coefficient
                        * gate_linear_accumulators[slot].clone()
                        * value_linear_accumulators[slot].clone();
            }
            expression = expression + selector * chunk_contribution;
        }
        expression
    }

    fn feed_forward_chunk_selector(&self, stage: &MPolynomial) -> MPolynomial {
        let stage_domain: Vec<i128> = (0..self.rows_per_step).map(|value| value as i128).collect();
        let mut selector = MPolynomial::zero();
        for chunk_index in 0..self.ff_chunk_rows {
            for feature_chunk in 0..self.linear_feature_chunk_rows {
                selector = selector
                    + lagrange_selector(
                        stage,
                        &stage_domain,
                        self.chunk_stage_index(chunk_index, feature_chunk) as i128,
                    );
            }
        }
        selector
    }

    fn zero_flag_expression(
        &self,
        pc_selectors: &[MPolynomial],
        pre_zero: &MPolynomial,
        result_zero: &MPolynomial,
    ) -> MPolynomial {
        let mut expression = MPolynomial::zero();
        for (pc_idx, selector) in pc_selectors.iter().enumerate() {
            let controls = &self.compiled_instructions[pc_idx].controls.zero;
            expression = expression + selector.clone() * controls.expression(pre_zero, result_zero);
        }
        expression
    }

    fn halted_flag_expression(
        &self,
        pc_selectors: &[MPolynomial],
        pre_halted: &MPolynomial,
    ) -> MPolynomial {
        let false_result = MPolynomial::zero();
        let mut expression = MPolynomial::zero();
        for (pc_idx, selector) in pc_selectors.iter().enumerate() {
            let controls = &self.compiled_instructions[pc_idx].controls.halted;
            expression =
                expression + selector.clone() * controls.expression(pre_halted, &false_result);
        }
        expression
    }

    fn compare_rhs_bit_expressions(
        &self,
        pc_selectors: &[MPolynomial],
        operand_bits: &[MPolynomial],
    ) -> Vec<MPolynomial> {
        let mut bits = vec![MPolynomial::zero(); WORD_BITS];
        for bit in 0..WORD_BITS {
            for (pc_idx, selector) in pc_selectors.iter().enumerate() {
                bits[bit] = bits[bit].clone()
                    + selector.clone()
                        * self.compiled_instructions[pc_idx]
                            .controls
                            .carry
                            .rhs_bit_expression(bit, &operand_bits[bit]);
            }
        }
        bits
    }

    fn compare_lt_expression(
        &self,
        left_bits: &[MPolynomial],
        right_bits: &[MPolynomial],
        raw_acc_low_bits: &[MPolynomial],
    ) -> MPolynomial {
        let left_sign = left_bits[WORD_BITS - 1].clone();
        let right_sign = right_bits[WORD_BITS - 1].clone();
        let diff_sign = raw_acc_low_bits[WORD_BITS - 1].clone();
        let sign_xor = left_sign.clone() + right_sign.clone()
            - mp_constant(2) * left_sign.clone() * right_sign.clone();
        left_sign.clone() * (mp_constant(1) - right_sign) + (mp_constant(1) - sign_xor) * diff_sign
    }

    fn carry_flag_expression(
        &self,
        pc_selectors: &[MPolynomial],
        pre_carry: &MPolynomial,
        overflow: &MPolynomial,
        compare_lt: &MPolynomial,
    ) -> MPolynomial {
        let mut expression = MPolynomial::zero();
        for (pc_idx, selector) in pc_selectors.iter().enumerate() {
            let controls = &self.compiled_instructions[pc_idx].controls.carry;
            expression = expression
                + selector.clone() * controls.expression(pre_carry, overflow, compare_lt);
        }
        expression
    }

    fn saturated_raw_acc_expression(
        &self,
        raw_acc_bits: &[MPolynomial],
        in_range: &MPolynomial,
    ) -> MPolynomial {
        let sign = raw_acc_bits[127].clone();
        let low_word = twos_complement_expression(&raw_acc_bits[..WORD_BITS]);
        let overflow = mp_constant(1) - in_range.clone();
        let overflow_high = (mp_constant(1) - sign.clone()) * overflow.clone();
        let overflow_low = sign * overflow;
        in_range.clone() * low_word
            + overflow_high * mp_constant(i128::from(Word::MAX))
            + overflow_low * mp_constant(i128::from(Word::MIN))
    }

    fn row_hash_expression(&self, registers: &[MPolynomial]) -> MPolynomial {
        let mut hash = MPolynomial::zero();
        for (index, value) in registers.iter().enumerate() {
            if index == self.layout.digest() {
                continue;
            }
            hash = hash + value.clone() * mp_constant((index + 1) as i128);
        }
        hash
    }
}

impl ExactCompiledInstruction {
    fn from_compiled(compiled: &CompiledInstruction) -> Result<Self> {
        Ok(Self {
            ff_weights: ExactFeedForwardWeights::from_shared(&compiled.ff_weights)?,
            controls: ExactTransitionControls::from_shared(&compiled.controls)?,
        })
    }
}

impl ExactFeedForwardWeights {
    fn from_shared(weights: &crate::model::FeedForwardWeights) -> Result<Self> {
        Ok(Self {
            gate: ExactMatrix::from_shared(&weights.gate)?,
            gate_bias: exact_scalar_slice_to_i128(&weights.gate_bias, "gate_bias")?,
            value: ExactMatrix::from_shared(&weights.value)?,
            value_bias: exact_scalar_slice_to_i128(&weights.value_bias, "value_bias")?,
            out: ExactMatrix::from_shared(&weights.out)?,
            out_bias: exact_scalar_slice_to_i128(&weights.out_bias, "out_bias")?,
        })
    }
}

impl ExactMatrix {
    fn from_shared(matrix: &crate::model::Matrix) -> Result<Self> {
        Ok(Self {
            rows: matrix.rows,
            cols: matrix.cols,
            data: exact_scalar_slice_to_i128(&matrix.data, "matrix")?,
        })
    }

    fn get(&self, row: usize, col: usize) -> i128 {
        self.data[row * self.cols + col]
    }
}

impl ExactTransitionControls {
    fn from_shared(controls: &TransitionControls) -> Result<Self> {
        Ok(Self {
            zero: ExactBoolBlend::from_shared(&controls.zero)?,
            carry: ExactCarryBlend::from_shared(&controls.carry)?,
            halted: ExactBoolBlend::from_shared(&controls.halted)?,
        })
    }
}

impl ExactBoolBlend {
    fn from_shared(blend: &crate::model::BoolBlend) -> Result<Self> {
        Ok(Self {
            prev_weight: exact_scalar_to_i128(blend.prev_weight, "bool_blend", 0)?,
            result_weight: exact_scalar_to_i128(blend.result_weight, "bool_blend", 1)?,
            constant: exact_scalar_to_i128(blend.constant, "bool_blend", 2)?,
        })
    }

    fn expression(&self, prev: &MPolynomial, result: &MPolynomial) -> MPolynomial {
        prev.clone() * mp_constant(self.prev_weight)
            + result.clone() * mp_constant(self.result_weight)
            + mp_constant(self.constant)
    }
}

impl ExactCarryBlend {
    fn from_shared(blend: &CarryBlend) -> Result<Self> {
        let rhs_operand_weight = exact_scalar_to_i128(blend.rhs_operand_weight, "carry_blend", 5)?;
        Ok(Self {
            prev_weight: exact_scalar_to_i128(blend.prev_weight, "carry_blend", 0)?,
            overflow_weight: exact_scalar_to_i128(blend.overflow_weight, "carry_blend", 1)?,
            less_than_weight: exact_scalar_to_i128(blend.less_than_weight, "carry_blend", 2)?,
            constant: exact_scalar_to_i128(blend.constant, "carry_blend", 3)?,
            rhs_constant: exact_scalar_to_i128(blend.rhs_constant, "carry_blend", 4)? as Word,
            rhs_uses_operand: match rhs_operand_weight {
                0 => false,
                1 => true,
                other => {
                    return Err(VmError::UnsupportedProof(format!(
                        "shared transformer staged skeleton proof supports compare rhs operand weights of 0 or 1, got {other}"
                    )))
                }
            },
        })
    }

    fn expression(
        &self,
        prev: &MPolynomial,
        overflow: &MPolynomial,
        less_than: &MPolynomial,
    ) -> MPolynomial {
        prev.clone() * mp_constant(self.prev_weight)
            + overflow.clone() * mp_constant(self.overflow_weight)
            + less_than.clone() * mp_constant(self.less_than_weight)
            + mp_constant(self.constant)
    }

    fn rhs_bit_expression(&self, bit: usize, operand_bit: &MPolynomial) -> MPolynomial {
        let constant_bit = ((self.rhs_constant as u64 >> bit) & 1) as i128;
        if self.rhs_uses_operand {
            operand_bit.clone()
        } else {
            mp_constant(constant_bit)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SharedTransformerMemoryRead {
    None,
    Direct(u8),
    StackTop,
}

fn memory_read_for_instruction(
    instruction: crate::instruction::Instruction,
) -> SharedTransformerMemoryRead {
    use crate::instruction::Instruction;

    match instruction {
        Instruction::Load(address)
        | Instruction::AddMemory(address)
        | Instruction::SubMemory(address)
        | Instruction::MulMemory(address)
        | Instruction::AndMemory(address)
        | Instruction::OrMemory(address)
        | Instruction::XorMemory(address)
        | Instruction::CmpMemory(address) => SharedTransformerMemoryRead::Direct(address),
        Instruction::Pop | Instruction::Ret => SharedTransformerMemoryRead::StackTop,
        _ => SharedTransformerMemoryRead::None,
    }
}

struct SharedTransformerStagedSkeletonTrace {
    trace: Vec<Vec<FieldElement>>,
    first_row_digest: FieldElement,
    final_digest: FieldElement,
}

pub fn prove_shared_transformer_staged_skeleton_stark(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
) -> Result<SharedTransformerStagedSkeletonProof> {
    Ok(
        prove_shared_transformer_staged_skeleton_stark_profiled_with_options(
            model,
            max_steps,
            ff_chunk_size,
            VanillaStarkProofOptions::default(),
        )?
        .proof,
    )
}

pub fn prove_shared_transformer_forward_pass_stark(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
) -> Result<SharedTransformerForwardPassProof> {
    prove_shared_transformer_staged_skeleton_stark(model, max_steps, ff_chunk_size)
}

pub fn prove_shared_transformer_staged_skeleton_stark_with_options(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerStagedSkeletonProof> {
    Ok(
        prove_shared_transformer_staged_skeleton_stark_profiled_with_options(
            model,
            max_steps,
            ff_chunk_size,
            options,
        )?
        .proof,
    )
}

pub fn prove_shared_transformer_forward_pass_stark_with_options(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerForwardPassProof> {
    prove_shared_transformer_staged_skeleton_stark_with_options(
        model,
        max_steps,
        ff_chunk_size,
        options,
    )
}

pub fn prove_shared_transformer_staged_skeleton_stark_profiled(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
) -> Result<SharedTransformerStagedSkeletonProvingReport> {
    prove_shared_transformer_staged_skeleton_stark_profiled_with_options(
        model,
        max_steps,
        ff_chunk_size,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_shared_transformer_forward_pass_stark_profiled(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
) -> Result<SharedTransformerForwardPassProvingReport> {
    prove_shared_transformer_staged_skeleton_stark_profiled(model, max_steps, ff_chunk_size)
}

pub fn estimate_shared_transformer_staged_skeleton_stark_cost(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerStagedSkeletonCostEstimate> {
    validate_shared_transformer_staged_skeleton_support(
        model.program(),
        model.config(),
        ff_chunk_size,
    )?;

    let mut runtime = ExecutionRuntime::new(model.clone(), max_steps);
    let (result, witness) = runtime.run_with_witness()?;
    if !result.halted {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged skeleton proof requires halting execution; stopped after {} steps without HALT",
            result.steps
        )));
    }
    if result.steps == 0 {
        return Err(VmError::UnsupportedProof(
            "shared transformer staged skeleton proof requires at least one executed step"
                .to_string(),
        ));
    }

    let staged_trace = build_shared_transformer_staged_trace(&witness, ff_chunk_size)?;
    validate_shared_transformer_staged_trace(model, &staged_trace)?;
    let air = SharedTransformerStagedSkeletonAir::new(
        model.program().clone(),
        model.config().clone(),
        ff_chunk_size,
    )?;
    let transition_constraints = air.transition_constraints();
    let transition_degree_bound = air.transition_degree_bound();
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        result.steps * air.rows_per_step,
        transition_degree_bound,
    );

    Ok(SharedTransformerStagedSkeletonCostEstimate {
        steps: result.steps,
        trace_rows: result.steps * air.rows_per_step,
        register_count: air.register_count(),
        memory_size: air.layout.memory_size,
        d_model: air.layout.d_model,
        ff_chunk_rows: air.ff_chunk_rows,
        linear_feature_chunk_rows: air.linear_feature_chunk_rows,
        hidden_chunk_width: air.hidden_chunk_width,
        memory_column_count: 2 * air.layout.memory_size,
        token_column_count: 0,
        input_feature_column_count: 0,
        chunk_value_column_count: 2,
        output_accumulator_column_count: OUTPUT_DIM,
        linear_accumulator_column_count: 2 * air.hidden_chunk_width,
        output_slot_accumulator_column_count: 0,
        compare_column_count: 0,
        raw_acc_column_count: 128 + 2,
        transition_constraint_count: transition_constraints.len(),
        transition_degree_bound,
        num_randomizers: stark.num_randomizers,
        omicron_domain_length: stark.omicron_domain.len(),
        fri_domain_length: stark.fri.domain_length,
        expansion_factor: options.expansion_factor,
        num_colinearity_checks: options.num_colinearity_checks,
        security_level: options.security_level,
    })
}

pub fn estimate_shared_transformer_forward_pass_stark_cost(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerForwardPassCostEstimate> {
    estimate_shared_transformer_staged_skeleton_stark_cost(model, max_steps, ff_chunk_size, options)
}

pub fn prove_shared_transformer_staged_skeleton_stark_profiled_with_options(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerStagedSkeletonProvingReport> {
    let total_start = Instant::now();
    validate_shared_transformer_staged_skeleton_support(
        model.program(),
        model.config(),
        ff_chunk_size,
    )?;

    let phase_start = Instant::now();
    let mut runtime = ExecutionRuntime::new(model.clone(), max_steps);
    let (result, witness) = runtime.run_with_witness()?;
    let witness_build_micros = elapsed_micros(phase_start);
    if !result.halted {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged skeleton proof requires halting execution; stopped after {} steps without HALT",
            result.steps
        )));
    }
    if result.steps == 0 {
        return Err(VmError::UnsupportedProof(
            "shared transformer staged skeleton proof requires at least one executed step"
                .to_string(),
        ));
    }

    let phase_start = Instant::now();
    let staged_trace = build_shared_transformer_staged_trace(&witness, ff_chunk_size)?;
    let staged_trace_build_micros = elapsed_micros(phase_start);

    let phase_start = Instant::now();
    validate_shared_transformer_staged_trace(model, &staged_trace)?;
    let trace_validation_micros = elapsed_micros(phase_start);

    let phase_start = Instant::now();
    let field_trace = build_shared_transformer_staged_skeleton_trace(model, &staged_trace)?;
    let skeleton_trace_build_micros = elapsed_micros(phase_start);

    let phase_start = Instant::now();
    let air = SharedTransformerStagedSkeletonAir::new(
        model.program().clone(),
        model.config().clone(),
        ff_chunk_size,
    )?;
    let transition_constraints = air.transition_constraints();
    let transition_degree_bound = air.transition_degree_bound();
    let air_build_micros = elapsed_micros(phase_start);
    let claim = SharedTransformerStagedSkeletonClaim {
        program: model.program().clone(),
        config: model.config().clone(),
        ff_chunk_size,
        steps: result.steps,
        final_state: result.final_state.clone(),
        first_row_digest: field_trace.first_row_digest.value(),
        trace_digest: field_trace.final_digest.value(),
        options: options.clone(),
    };
    let phase_start = Instant::now();
    let boundary = air.boundary_constraints(&claim);
    let boundary_build_micros = elapsed_micros(phase_start);
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        field_trace.trace.len(),
        transition_degree_bound,
    );
    let (proof, stark_profile) =
        stark.prove_with_profile(&field_trace.trace, &transition_constraints, &boundary);

    Ok(SharedTransformerStagedSkeletonProvingReport {
        proof: SharedTransformerStagedSkeletonProof { claim, proof },
        profile: SharedTransformerStagedSkeletonProvingProfile {
            witness_build_micros,
            staged_trace_build_micros,
            trace_validation_micros,
            skeleton_trace_build_micros,
            air_build_micros,
            boundary_build_micros,
            stark: stark_profile,
            total_micros: elapsed_micros(total_start),
        },
    })
}

pub fn prove_shared_transformer_forward_pass_stark_profiled_with_options(
    model: &TransformerVm,
    max_steps: usize,
    ff_chunk_size: usize,
    options: VanillaStarkProofOptions,
) -> Result<SharedTransformerForwardPassProvingReport> {
    prove_shared_transformer_staged_skeleton_stark_profiled_with_options(
        model,
        max_steps,
        ff_chunk_size,
        options,
    )
}

pub fn verify_shared_transformer_staged_skeleton_stark(
    proof: &SharedTransformerStagedSkeletonProof,
) -> Result<bool> {
    validate_shared_transformer_staged_skeleton_support(
        &proof.claim.program,
        &proof.claim.config,
        proof.claim.ff_chunk_size,
    )?;
    validate_shared_transformer_public_state(&proof.claim.program, &proof.claim.final_state)?;

    if proof.claim.steps == 0 {
        return Err(VmError::UnsupportedProof(
            "shared transformer staged skeleton proof claim must contain at least one step"
                .to_string(),
        ));
    }

    let air = SharedTransformerStagedSkeletonAir::new(
        proof.claim.program.clone(),
        proof.claim.config.clone(),
        proof.claim.ff_chunk_size,
    )?;
    let stark = Stark::new(
        proof.claim.options.expansion_factor,
        proof.claim.options.num_colinearity_checks,
        proof.claim.options.security_level,
        air.register_count(),
        proof.claim.steps * air.rows_per_step,
        air.transition_degree_bound(),
    );

    Ok(stark.verify(
        &proof.proof,
        &air.transition_constraints(),
        &air.boundary_constraints(&proof.claim),
    ))
}

pub fn verify_shared_transformer_forward_pass_stark(
    proof: &SharedTransformerForwardPassProof,
) -> Result<bool> {
    verify_shared_transformer_staged_skeleton_stark(proof)
}

fn validate_shared_transformer_staged_skeleton_support(
    program: &Program,
    config: &TransformerVmConfig,
    ff_chunk_size: usize,
) -> Result<()> {
    config.validate()?;
    if program.is_empty() {
        return Err(VmError::UnsupportedProof(
            "cannot prove an empty program".to_string(),
        ));
    }
    if ff_chunk_size == 0 {
        return Err(VmError::InvalidConfig(
            "shared transformer staged skeleton proof requires ff_chunk_size > 0".to_string(),
        ));
    }
    if !matches!(config.attention_mode, Attention2DMode::AverageHard { .. }) {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged skeleton proof currently supports only `average-hard` attention, got `{}`",
            config.attention_mode
        )));
    }
    Ok(())
}

fn validate_shared_transformer_public_state(
    program: &Program,
    final_state: &MachineState,
) -> Result<()> {
    if final_state.memory.len() != program.memory_size() {
        return Err(VmError::InvalidConfig(format!(
            "shared transformer proof final-state memory length {} does not match program memory size {}",
            final_state.memory.len(),
            program.memory_size()
        )));
    }
    Ok(())
}

fn build_shared_transformer_staged_skeleton_trace(
    model: &TransformerVm,
    trace: &SharedTransformerStagedTrace,
) -> Result<SharedTransformerStagedSkeletonTrace> {
    let air = SharedTransformerStagedSkeletonAir::new(
        model.program().clone(),
        model.config().clone(),
        trace.ff_chunk_size,
    )?;
    if trace.rows.is_empty() {
        return Err(VmError::UnsupportedProof(
            "shared transformer staged skeleton trace cannot be empty".to_string(),
        ));
    }

    let mut field_rows = Vec::new();
    let mut cursor = 0usize;
    while cursor < trace.rows.len() {
        let input_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::InputState(row)) => row,
            Some(other) => {
                return Err(VmError::ExecutionMismatch {
                    step: 0,
                    message: format!("expected input-state row, got {:?}", other.stage()),
                })
            }
            None => unreachable!("cursor is bounded by trace.rows.len()"),
        };
        cursor += 1;
        let attention_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::Attention(row)) => row,
            Some(other) => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: format!("expected attention row, got {:?}", other.stage()),
                })
            }
            None => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: "trace ended before attention row".to_string(),
                })
            }
        };
        cursor += 1;
        let mut chunk_rows = Vec::new();
        while let Some(SharedTransformerTraceRow::FeedForwardChunk(chunk)) = trace.rows.get(cursor)
        {
            chunk_rows.push(chunk);
            cursor += 1;
        }
        if chunk_rows.len() != air.ff_chunk_rows {
            return Err(VmError::ExecutionMismatch {
                step: input_row.step_number,
                message: format!(
                    "expected {} feed-forward chunk rows, got {}",
                    air.ff_chunk_rows,
                    chunk_rows.len()
                ),
            });
        }
        let transition_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::Transition(row)) => row,
            Some(other) => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: format!("expected transition row, got {:?}", other.stage()),
                })
            }
            None => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: "trace ended before transition row".to_string(),
                })
            }
        };
        cursor += 1;
        let output_row = match trace.rows.get(cursor) {
            Some(SharedTransformerTraceRow::OutputState(row)) => row,
            Some(other) => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: format!("expected output-state row, got {:?}", other.stage()),
                })
            }
            None => {
                return Err(VmError::ExecutionMismatch {
                    step: input_row.step_number,
                    message: "trace ended before output-state row".to_string(),
                })
            }
        };
        cursor += 1;

        if transition_row.ff_output.len() != OUTPUT_DIM {
            return Err(VmError::ExecutionMismatch {
                step: input_row.step_number,
                message: format!(
                    "transition row ff_output width {} does not match OUTPUT_DIM {OUTPUT_DIM}",
                    transition_row.ff_output.len()
                ),
            });
        }

        let layer_idx = input_row.dispatch.layer_idx;
        let (mem_write_enable, mem_write_addr, mem_write_value) =
            transition_write_fields(&transition_row.transition);

        for stage in 0..air.rows_per_step {
            let mut row = vec![FieldElement::zero(); air.register_count()];
            row[SKELETON_STAGE] = field_from_i128(stage as i128);
            row[SKELETON_STEP] = field_from_i128(input_row.step_number as i128);
            row[SKELETON_LAYER] = field_from_i128(layer_idx as i128);
            fill_state_columns(&mut row, &input_row.pre_state, true);
            fill_word_bits_columns(
                &mut row,
                input_row.pre_state.acc,
                SKELETON_PRE_ACC_BITS_START,
            );
            fill_word_bits_columns(&mut row, attention_row.operand, SKELETON_OPERAND_BITS_START);
            fill_state_columns(&mut row, &output_row.final_state, false);
            row[SKELETON_OPERAND] = field_from_i128(i128::from(attention_row.operand));
            let post_acc = output_row.final_state.acc;
            row[SKELETON_POST_ACC_INV] = field_inverse_or_zero(post_acc);
            let compiled = &air.compiled_instructions[input_row.pre_state.pc as usize];
            let mut output_accumulator = compiled.ff_weights.out_bias.clone();
            let completed_chunks = if stage < 2 {
                0
            } else if stage < air.transition_stage_index() {
                (stage - 2) / air.linear_feature_chunk_rows
            } else {
                chunk_rows.len()
            };
            if completed_chunks > 0 {
                for chunk_row in chunk_rows.iter().take(completed_chunks) {
                    apply_output_chunk_accumulator(
                        &mut output_accumulator,
                        &compiled.ff_weights,
                        chunk_row,
                    )?;
                }
            }
            if (2..air.transition_stage_index()).contains(&stage) {
                let chunk_offset = stage - 2;
                let chunk_index = chunk_offset / air.linear_feature_chunk_rows;
                let feature_chunk = chunk_offset % air.linear_feature_chunk_rows;
                let chunk_row = chunk_rows[chunk_index];
                row[air.layout.chunk_index()] = field_from_i128(chunk_row.chunk_index as i128);
                row[air.layout.hidden_start()] = field_from_i128(chunk_row.hidden_start as i128);
                fill_linear_feature_accumulator_columns(
                    &mut row,
                    &air.layout,
                    compiled,
                    &attention_row.input_vector,
                    chunk_row,
                    feature_chunk,
                )?;
            }
            fill_i128_bits_columns(
                &mut row,
                transition_row.ff_output[1],
                air.layout.raw_acc_bit(0),
            );
            let raw_acc_in_range = raw_acc_in_range_witness(transition_row.ff_output[1]);
            row[air.layout.raw_acc_in_range()] = field_from_bool(raw_acc_in_range);
            row[air.layout.raw_acc_range_inv()] =
                field_inverse_or_zero_i128(raw_acc_mismatch_sum(transition_row.ff_output[1]));
            for (index, value) in output_accumulator.into_iter().enumerate() {
                row[air.layout.output_accumulator(index)] = field_from_i128(value);
            }
            row[SKELETON_MEM_WRITE_ENABLE] = field_from_bool(mem_write_enable);
            row[SKELETON_MEM_WRITE_ADDR] = field_from_i128(mem_write_addr);
            row[SKELETON_MEM_WRITE_VALUE] = field_from_i128(i128::from(mem_write_value));
            fill_memory_columns(&mut row, &input_row.pre_state.memory, &air.layout, true);
            fill_memory_columns(&mut row, &output_row.final_state.memory, &air.layout, false);
            field_rows.push(row);
        }
    }

    let digest_index = air.layout.digest();
    let mut first_row_digest = FieldElement::zero();
    let mut final_digest = FieldElement::zero();
    for (index, row) in field_rows.iter_mut().enumerate() {
        let row_hash = row_hash_field_values(row, digest_index);
        let digest = if index == 0 {
            row_hash
        } else {
            final_digest * mp_field_constant(SKELETON_DIGEST_BETA) + row_hash
        };
        row[digest_index] = digest;
        if index == 0 {
            first_row_digest = digest;
        }
        final_digest = digest;
    }

    Ok(SharedTransformerStagedSkeletonTrace {
        trace: field_rows,
        first_row_digest,
        final_digest,
    })
}

fn transition_write_fields(transition: &SharedTransformerTransitionWitness) -> (bool, i128, Word) {
    match transition.memory_write {
        Some((address, value)) => (true, i128::from(address), value),
        None => (false, 0, 0),
    }
}

fn fill_state_columns(row: &mut [FieldElement], state: &MachineState, is_pre: bool) {
    let (pc, acc, sp, zero, carry, halted) = if is_pre {
        (
            SKELETON_PRE_PC,
            SKELETON_PRE_ACC,
            SKELETON_PRE_SP,
            SKELETON_PRE_ZERO,
            SKELETON_PRE_CARRY,
            SKELETON_PRE_HALTED,
        )
    } else {
        (
            SKELETON_POST_PC,
            SKELETON_POST_ACC,
            SKELETON_POST_SP,
            SKELETON_POST_ZERO,
            SKELETON_POST_CARRY,
            SKELETON_POST_HALTED,
        )
    };
    row[pc] = field_from_i128(i128::from(state.pc));
    row[acc] = field_from_i128(i128::from(state.acc));
    row[sp] = field_from_i128(i128::from(state.sp));
    row[zero] = field_from_bool(state.zero_flag);
    row[carry] = field_from_bool(state.carry_flag);
    row[halted] = field_from_bool(state.halted);
}

fn fill_word_bits_columns(row: &mut [FieldElement], value: Word, start: usize) {
    let bits = value as u64;
    for bit in 0..WORD_BITS {
        row[start + bit] = field_from_bool(((bits >> bit) & 1) == 1);
    }
}

fn fill_i128_bits_columns(row: &mut [FieldElement], value: i128, start: usize) {
    let bits = value as u128;
    for bit in 0..128 {
        row[start + bit] = field_from_bool(((bits >> bit) & 1) == 1);
    }
}

fn fill_memory_columns(
    row: &mut [FieldElement],
    memory: &[Word],
    layout: &SharedTransformerStagedSkeletonLayout,
    is_pre: bool,
) {
    for (index, value) in memory.iter().copied().enumerate() {
        let column = if is_pre {
            layout.pre_mem(index)
        } else {
            layout.post_mem(index)
        };
        row[column] = field_from_i128(i128::from(value));
    }
}

fn fill_linear_feature_accumulator_columns(
    row: &mut [FieldElement],
    layout: &SharedTransformerStagedSkeletonLayout,
    compiled: &ExactCompiledInstruction,
    input_features: &[Word],
    chunk_row: &SharedTransformerFeedForwardChunkTraceRow,
    feature_chunk: usize,
) -> Result<()> {
    for slot in 0..layout.hidden_chunk_width {
        let hidden_index = chunk_row.hidden_start + slot;
        let mut gate_accumulator = if hidden_index < compiled.ff_weights.gate.rows {
            compiled.ff_weights.gate_bias[hidden_index]
        } else {
            0
        };
        let mut value_accumulator = if hidden_index < compiled.ff_weights.value.rows {
            compiled.ff_weights.value_bias[hidden_index]
        } else {
            0
        };

        if hidden_index < compiled.ff_weights.gate.rows {
            for previous_feature_chunk in 0..=feature_chunk {
                let previous_feature_start =
                    previous_feature_chunk * SKELETON_LINEAR_FEATURE_CHUNK_SIZE;
                let previous_feature_end = ((previous_feature_chunk + 1)
                    * SKELETON_LINEAR_FEATURE_CHUNK_SIZE)
                    .min(INPUT_DIM);
                for (feature_index, &feature) in input_features
                    .iter()
                    .enumerate()
                    .take(previous_feature_end)
                    .skip(previous_feature_start)
                {
                    let feature_value = i128::from(feature);
                    let gate_term = compiled
                        .ff_weights
                        .gate
                        .get(hidden_index, feature_index)
                        .checked_mul(feature_value)
                        .ok_or_else(|| {
                            VmError::UnsupportedProof(format!(
                                "shared transformer staged skeleton proof gate linear accumulator overflowed i128 for hidden index {} feature {}",
                                hidden_index, feature_index
                            ))
                        })?;
                    let value_term = compiled
                        .ff_weights
                        .value
                        .get(hidden_index, feature_index)
                        .checked_mul(feature_value)
                        .ok_or_else(|| {
                            VmError::UnsupportedProof(format!(
                                "shared transformer staged skeleton proof value linear accumulator overflowed i128 for hidden index {} feature {}",
                                hidden_index, feature_index
                            ))
                        })?;
                    gate_accumulator = gate_accumulator.checked_add(gate_term).ok_or_else(|| {
                        VmError::UnsupportedProof(format!(
                            "shared transformer staged skeleton proof gate partial sum overflowed i128 for hidden index {} feature chunk {}",
                            hidden_index, previous_feature_chunk
                        ))
                    })?;
                    value_accumulator = value_accumulator.checked_add(value_term).ok_or_else(|| {
                        VmError::UnsupportedProof(format!(
                            "shared transformer staged skeleton proof value partial sum overflowed i128 for hidden index {} feature chunk {}",
                            hidden_index, previous_feature_chunk
                        ))
                    })?;
                }
            }
        }
        row[layout.gate_linear_accumulator(slot)] = field_from_i128(gate_accumulator);
        row[layout.value_linear_accumulator(slot)] = field_from_i128(value_accumulator);
    }

    Ok(())
}

fn apply_output_chunk_accumulator(
    accumulator: &mut [i128],
    weights: &ExactFeedForwardWeights,
    chunk_row: &SharedTransformerFeedForwardChunkTraceRow,
) -> Result<()> {
    for (slot, hidden_value) in chunk_row.hidden_values.iter().copied().enumerate() {
        let hidden_index = chunk_row.hidden_start + slot;
        if hidden_index >= weights.gate.rows {
            continue;
        }
        for (output_idx, accumulator_value) in accumulator.iter_mut().enumerate() {
            let coefficient = weights.out.get(output_idx, hidden_index);
            if coefficient == 0 {
                continue;
            }
            let contribution = coefficient
                .checked_mul(i128::from(hidden_value))
                .ok_or_else(|| {
                    VmError::UnsupportedProof(format!(
                        "shared transformer staged skeleton proof output contribution overflowed i128 for chunk {} hidden index {} output {}",
                        chunk_row.chunk_index, hidden_index, output_idx
                    ))
                })?;
            *accumulator_value = accumulator_value.checked_add(contribution).ok_or_else(|| {
                VmError::UnsupportedProof(format!(
                    "shared transformer staged skeleton proof output accumulator overflowed i128 for chunk {} output {}",
                    chunk_row.chunk_index, output_idx
                ))
            })?;
        }
    }
    Ok(())
}

fn push_state_boundary(
    boundary: &mut Vec<(usize, usize, FieldElement)>,
    row: usize,
    state: &MachineState,
    is_pre: bool,
) {
    let (pc, acc, sp, zero, carry, halted) = if is_pre {
        (
            SKELETON_PRE_PC,
            SKELETON_PRE_ACC,
            SKELETON_PRE_SP,
            SKELETON_PRE_ZERO,
            SKELETON_PRE_CARRY,
            SKELETON_PRE_HALTED,
        )
    } else {
        (
            SKELETON_POST_PC,
            SKELETON_POST_ACC,
            SKELETON_POST_SP,
            SKELETON_POST_ZERO,
            SKELETON_POST_CARRY,
            SKELETON_POST_HALTED,
        )
    };
    boundary.push((row, pc, field_from_i128(i128::from(state.pc))));
    boundary.push((row, acc, field_from_i128(i128::from(state.acc))));
    boundary.push((row, sp, field_from_i128(i128::from(state.sp))));
    boundary.push((row, zero, field_from_bool(state.zero_flag)));
    boundary.push((row, carry, field_from_bool(state.carry_flag)));
    boundary.push((row, halted, field_from_bool(state.halted)));
}

fn push_memory_boundary(
    boundary: &mut Vec<(usize, usize, FieldElement)>,
    row: usize,
    memory: &[Word],
    layout: &SharedTransformerStagedSkeletonLayout,
    is_pre: bool,
) {
    for (index, value) in memory.iter().copied().enumerate() {
        let column = if is_pre {
            layout.pre_mem(index)
        } else {
            layout.post_mem(index)
        };
        boundary.push((row, column, field_from_i128(i128::from(value))));
    }
}

fn row_hash_field_values(row: &[FieldElement], digest_index: usize) -> FieldElement {
    let mut hash = FieldElement::zero();
    for (index, value) in row.iter().copied().enumerate() {
        if index == digest_index {
            continue;
        }
        hash = hash + value * mp_field_constant(index as i128 + 1);
    }
    hash
}

fn twos_complement_expression(bits: &[MPolynomial]) -> MPolynomial {
    let mut expression = MPolynomial::zero();
    for (bit, bit_poly) in bits.iter().enumerate().take(WORD_BITS - 1) {
        expression = expression + bit_poly.clone() * mp_constant(1i128 << bit);
    }
    expression - bits[WORD_BITS - 1].clone() * mp_constant(1i128 << (WORD_BITS - 1))
}

fn twos_complement_i128_expression(bits: &[MPolynomial]) -> MPolynomial {
    let mut expression = MPolynomial::zero();
    for (bit, bit_poly) in bits.iter().enumerate().take(127) {
        expression = expression + bit_poly.clone() * mp_constant(1i128 << bit);
    }
    expression + bits[127].clone() * mp_constant(i128::MIN)
}

fn domain_zerofier(variable: &MPolynomial, size: usize) -> MPolynomial {
    let mut polynomial = mp_constant(1);
    for point in 0..size {
        polynomial = polynomial * (variable.clone() - mp_constant(point as i128));
    }
    polynomial
}

fn lagrange_selector(variable: &MPolynomial, domain: &[i128], point: i128) -> MPolynomial {
    let mut selector = mp_constant(1);
    for &other in domain {
        if other == point {
            continue;
        }
        let denominator = field_from_i128(point - other).inverse();
        selector =
            selector * (variable.clone() - mp_constant(other)) * MPolynomial::constant(denominator);
    }
    selector
}

fn mp_constant(value: i128) -> MPolynomial {
    MPolynomial::constant(field_from_i128(value))
}

fn mp_field_constant(value: i128) -> FieldElement {
    field_from_i128(value)
}

fn field_from_bool(value: bool) -> FieldElement {
    field_from_i128(if value { 1 } else { 0 })
}

fn field_from_u128(value: u128) -> FieldElement {
    FieldElement::new(value)
}

fn field_inverse_or_zero(value: Word) -> FieldElement {
    if value == 0 {
        FieldElement::zero()
    } else {
        field_from_i128(i128::from(value)).inverse()
    }
}

fn field_inverse_or_zero_i128(value: i128) -> FieldElement {
    if value == 0 {
        FieldElement::zero()
    } else {
        field_from_i128(value).inverse()
    }
}

fn elapsed_micros(start: Instant) -> u64 {
    start.elapsed().as_micros().try_into().unwrap_or(u64::MAX)
}

fn exact_scalar_slice_to_i128(values: &[f64], location: &str) -> Result<Vec<i128>> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_scalar_to_i128(value, location, index))
        .collect()
}

fn exact_scalar_to_i128(value: f64, location: &str, index: usize) -> Result<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged skeleton proof requires exact integer coefficients; {location}[{index}] was {value}"
        )));
    }
    if value < i128::MIN as f64 || value > i128::MAX as f64 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged skeleton proof coefficient out of i128 range; {location}[{index}] was {value}"
        )));
    }
    Ok(value as i128)
}

fn exact_trace_scalar_to_i128(
    step_number: usize,
    location: &str,
    index: usize,
    value: f64,
) -> Result<i128> {
    exact_scalar_to_i128(value, location, index).map_err(|err| match err {
        VmError::UnsupportedProof(message) => {
            VmError::UnsupportedProof(format!("{message} at step {step_number}"))
        }
        other => other,
    })
}

fn field_from_i128(value: i128) -> FieldElement {
    if value >= 0 {
        FieldElement::new(value as u128)
    } else {
        -FieldElement::new(value.unsigned_abs())
    }
}

#[allow(dead_code)]
fn compare_less_than_witness(left: Word, right: Word) -> bool {
    left < right
}

fn raw_acc_mismatch_sum(raw_acc: i128) -> i128 {
    let bits = raw_acc as u128;
    let low_sign = ((bits >> 63) & 1) == 1;
    let mut mismatch = 0i128;
    for bit in 64..128 {
        let equal = (((bits >> bit) & 1) == 1) == low_sign;
        if !equal {
            mismatch += 1;
        }
    }
    mismatch
}

fn raw_acc_in_range_witness(raw_acc: i128) -> bool {
    raw_acc_mismatch_sum(raw_acc) == 0
}

#[cfg(test)]
mod tests {
    use super::{
        apply_output_chunk_accumulator, build_shared_transformer_staged_skeleton_trace,
        build_shared_transformer_staged_trace, compare_less_than_witness, raw_acc_in_range_witness,
        raw_acc_mismatch_sum, SharedTransformerStagedSkeletonAir,
    };
    use crate::config::TransformerVmConfig;
    use crate::runtime::ExecutionRuntime;
    use crate::transformer_trace::SharedTransformerTraceRow;
    use crate::ProgramCompiler;

    #[test]
    fn skeleton_trace_populates_feed_forward_chunk_columns() {
        let model = ProgramCompiler
            .compile_source(
                r#"
                .memory 1
                LOADI 5
                HALT
            "#,
                TransformerVmConfig::default(),
            )
            .expect("compile");
        let mut runtime = ExecutionRuntime::new(model.clone(), 8);
        let (_result, witness) = runtime.run_with_witness().expect("run with witness");
        let staged = build_shared_transformer_staged_trace(&witness, 8).expect("staged trace");
        let skeleton =
            build_shared_transformer_staged_skeleton_trace(&model, &staged).expect("skeleton");

        let air = super::SharedTransformerStagedSkeletonAir::new(
            model.program().clone(),
            model.config().clone(),
            staged.ff_chunk_size,
        )
        .expect("air");
        let chunk_row = &skeleton.trace[2];
        let final_feature_row = &skeleton.trace[2 + air.linear_feature_chunk_rows - 1];
        assert_eq!(
            chunk_row[model.config().d_model],
            chunk_row[model.config().d_model]
        );
        assert_eq!(
            chunk_row[air.layout.chunk_index()].value(),
            0,
            "first feed-forward chunk row should carry chunk index 0"
        );
        assert_eq!(
            chunk_row[air.layout.hidden_start()].value(),
            0,
            "first feed-forward chunk row should start at hidden offset 0"
        );
        assert_ne!(
            final_feature_row[air.layout.gate_linear_accumulator(0)].value(),
            0,
            "final feature subrow for the first chunk should populate the shared gate accumulator"
        );
        let input_row = &skeleton.trace[0];
        assert_eq!(
            input_row[air.layout.chunk_index()].value(),
            0,
            "non-chunk rows should keep chunk index cleared"
        );
        assert_eq!(
            input_row[air.layout.gate_linear_accumulator(0)].value(),
            0,
            "non-chunk rows should keep linear-accumulator columns cleared"
        );
        let attention_row = &skeleton.trace[1];
        let compiled = &air.compiled_instructions[0];
        assert_eq!(
            attention_row[air.layout.output_accumulator(0)],
            super::field_from_i128(compiled.ff_weights.out_bias[0]),
            "attention row should carry the shared FF output bias while input features are derived in-AIR"
        );
    }

    #[test]
    fn compare_less_than_witness_handles_sign_boundaries() {
        assert!(compare_less_than_witness(-1, 1));
        assert!(!compare_less_than_witness(7, 7));
        assert!(!compare_less_than_witness(9, -4));
    }

    #[test]
    fn raw_acc_in_range_witness_detects_i64_range_boundary() {
        assert!(raw_acc_in_range_witness(i128::from(i64::MAX)));
        assert_eq!(raw_acc_mismatch_sum(i128::from(i64::MAX)), 0);

        assert!(!raw_acc_in_range_witness(i128::from(i64::MAX) + 1));
        assert!(raw_acc_mismatch_sum(i128::from(i64::MAX) + 1) > 0);
    }

    #[test]
    fn skeleton_trace_stages_output_accumulator_across_feed_forward_chunks() {
        let model = ProgramCompiler
            .compile_source(
                r#"
                .memory 1
                LOADI 5
                HALT
            "#,
                TransformerVmConfig::default(),
            )
            .expect("compile");
        let mut runtime = ExecutionRuntime::new(model.clone(), 8);
        let (_result, witness) = runtime.run_with_witness().expect("run with witness");
        let staged = build_shared_transformer_staged_trace(&witness, 8).expect("staged trace");
        let skeleton =
            build_shared_transformer_staged_skeleton_trace(&model, &staged).expect("skeleton");
        let air = SharedTransformerStagedSkeletonAir::new(
            model.program().clone(),
            model.config().clone(),
            staged.ff_chunk_size,
        )
        .expect("air");

        let compiled = &air.compiled_instructions[0];
        let first_chunk_row = &skeleton.trace[2];
        let first_chunk_final_row = &skeleton.trace[2 + air.linear_feature_chunk_rows - 1];
        for output_idx in 0..super::OUTPUT_DIM {
            assert_eq!(
                first_chunk_row[air.layout.output_accumulator(output_idx)],
                super::field_from_i128(compiled.ff_weights.out_bias[output_idx]),
                "first feature subrow of the first chunk should start from the shared output bias"
            );
        }

        let first_chunk = match &staged.rows[2] {
            SharedTransformerTraceRow::FeedForwardChunk(chunk) => chunk,
            other => panic!(
                "expected first feed-forward chunk row, got {:?}",
                other.stage()
            ),
        };
        let mut expected_after_first_chunk = compiled.ff_weights.out_bias.clone();
        apply_output_chunk_accumulator(
            &mut expected_after_first_chunk,
            &compiled.ff_weights,
            first_chunk,
        )
        .expect("accumulate");

        let second_chunk_row = &skeleton.trace[2 + air.linear_feature_chunk_rows];
        for output_idx in 0..super::OUTPUT_DIM {
            assert_eq!(
                second_chunk_row[air.layout.output_accumulator(output_idx)],
                super::field_from_i128(expected_after_first_chunk[output_idx]),
                "next chunk should carry the running shared FF output accumulation"
            );
            assert_eq!(
                first_chunk_final_row[air.layout.output_accumulator(output_idx)],
                super::field_from_i128(compiled.ff_weights.out_bias[output_idx]),
                "final feature subrow should still expose the pre-chunk running output accumulator"
            );
        }

        let transition_row = &skeleton.trace[air.transition_stage_index()];
        for output_idx in 0..super::OUTPUT_DIM {
            assert_eq!(
                transition_row[air.layout.output_accumulator(output_idx)],
                crate::vanillastark::field::FieldElement::new(
                    witness.steps[0].ff_output[output_idx] as u128
                ),
                "transition row should expose the final shared FF output accumulation"
            );
        }
    }

    #[test]
    fn skeleton_trace_stages_linear_feature_accumulators() {
        let model = ProgramCompiler
            .compile_source(
                r#"
                .memory 1
                LOADI 5
                HALT
            "#,
                TransformerVmConfig::default(),
            )
            .expect("compile");
        let mut runtime = ExecutionRuntime::new(model.clone(), 8);
        let (_result, witness) = runtime.run_with_witness().expect("run with witness");
        let staged = build_shared_transformer_staged_trace(&witness, 8).expect("staged trace");
        let skeleton =
            build_shared_transformer_staged_skeleton_trace(&model, &staged).expect("skeleton");
        let air = SharedTransformerStagedSkeletonAir::new(
            model.program().clone(),
            model.config().clone(),
            staged.ff_chunk_size,
        )
        .expect("air");

        let first_chunk_row = &skeleton.trace[2 + air.linear_feature_chunk_rows - 1];
        assert_ne!(
            first_chunk_row[air.layout.gate_linear_accumulator(0)].value(),
            0,
            "final gate accumulator should remain materialized on the final feature subrow"
        );
        assert_ne!(
            first_chunk_row[air.layout.value_linear_accumulator(0)].value(),
            0,
            "final value accumulator should remain materialized on the final feature subrow"
        );
    }
}
