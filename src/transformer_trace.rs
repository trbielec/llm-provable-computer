use serde::{Deserialize, Serialize};

use crate::error::{Result, VmError};
use crate::model::{
    DispatchInfo, SharedTransformerExecutionWitness, SharedTransformerStepWitness,
    SharedTransformerTransitionWitness,
};
use crate::state::{MachineState, Word};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedTransformerTraceStage {
    InputState,
    Attention,
    FeedForwardChunk,
    Transition,
    OutputState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerInputTraceRow {
    pub step_number: usize,
    pub pre_state: MachineState,
    pub input_token: Vec<i8>,
    pub decoded_state: MachineState,
    pub dispatch: DispatchInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerAttentionTraceRow {
    pub step_number: usize,
    pub operand: Word,
    pub input_vector: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerFeedForwardChunkTraceRow {
    pub step_number: usize,
    pub chunk_index: usize,
    pub hidden_start: usize,
    pub gate_values: Vec<Word>,
    pub value_values: Vec<Word>,
    pub hidden_values: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerTransitionTraceRow {
    pub step_number: usize,
    pub ff_output: Vec<i128>,
    pub transition: SharedTransformerTransitionWitness,
    pub post_write_state: MachineState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerOutputTraceRow {
    pub step_number: usize,
    pub output_token: Vec<i8>,
    pub final_state: MachineState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum SharedTransformerTraceRow {
    InputState(SharedTransformerInputTraceRow),
    Attention(SharedTransformerAttentionTraceRow),
    FeedForwardChunk(SharedTransformerFeedForwardChunkTraceRow),
    Transition(SharedTransformerTransitionTraceRow),
    OutputState(SharedTransformerOutputTraceRow),
}

impl SharedTransformerTraceRow {
    pub fn step_number(&self) -> usize {
        match self {
            SharedTransformerTraceRow::InputState(row) => row.step_number,
            SharedTransformerTraceRow::Attention(row) => row.step_number,
            SharedTransformerTraceRow::FeedForwardChunk(row) => row.step_number,
            SharedTransformerTraceRow::Transition(row) => row.step_number,
            SharedTransformerTraceRow::OutputState(row) => row.step_number,
        }
    }

    pub fn stage(&self) -> SharedTransformerTraceStage {
        match self {
            SharedTransformerTraceRow::InputState(_) => SharedTransformerTraceStage::InputState,
            SharedTransformerTraceRow::Attention(_) => SharedTransformerTraceStage::Attention,
            SharedTransformerTraceRow::FeedForwardChunk(_) => {
                SharedTransformerTraceStage::FeedForwardChunk
            }
            SharedTransformerTraceRow::Transition(_) => SharedTransformerTraceStage::Transition,
            SharedTransformerTraceRow::OutputState(_) => SharedTransformerTraceStage::OutputState,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTransformerStagedTrace {
    pub ff_chunk_size: usize,
    pub rows: Vec<SharedTransformerTraceRow>,
}

impl SharedTransformerStagedTrace {
    pub fn row_count_for_step(&self, step_number: usize) -> usize {
        self.rows
            .iter()
            .filter(|row| row.step_number() == step_number)
            .count()
    }
}

pub fn build_shared_transformer_staged_trace(
    witness: &SharedTransformerExecutionWitness,
    ff_chunk_size: usize,
) -> Result<SharedTransformerStagedTrace> {
    if ff_chunk_size == 0 {
        return Err(VmError::InvalidConfig(
            "shared transformer staged trace requires ff_chunk_size > 0".to_string(),
        ));
    }

    let mut rows = Vec::new();
    for step in &witness.steps {
        rows.push(SharedTransformerTraceRow::InputState(
            build_input_state_row(step)?,
        ));
        rows.push(SharedTransformerTraceRow::Attention(build_attention_row(
            step,
        )?));
        rows.extend(build_feed_forward_chunk_rows(step, ff_chunk_size)?);
        rows.push(SharedTransformerTraceRow::Transition(build_transition_row(
            step,
        )?));
        rows.push(SharedTransformerTraceRow::OutputState(
            build_output_state_row(step)?,
        ));
    }

    Ok(SharedTransformerStagedTrace {
        ff_chunk_size,
        rows,
    })
}

fn build_input_state_row(
    step: &SharedTransformerStepWitness,
) -> Result<SharedTransformerInputTraceRow> {
    Ok(SharedTransformerInputTraceRow {
        step_number: step.step_number,
        pre_state: step.pre_state.clone(),
        input_token: step
            .input_token
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| exact_token_value(step.step_number, "input_token", index, value))
            .collect::<Result<Vec<_>>>()?,
        decoded_state: step.decoded_state.clone(),
        dispatch: step.dispatch,
    })
}

fn build_attention_row(
    step: &SharedTransformerStepWitness,
) -> Result<SharedTransformerAttentionTraceRow> {
    Ok(SharedTransformerAttentionTraceRow {
        step_number: step.step_number,
        operand: step.operand,
        input_vector: step
            .input_vector
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| {
                exact_scalar_to_word(step.step_number, "input_vector", index, value)
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

fn build_feed_forward_chunk_rows(
    step: &SharedTransformerStepWitness,
    ff_chunk_size: usize,
) -> Result<Vec<SharedTransformerTraceRow>> {
    let gate = step
        .ff_gate
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_scalar_to_word(step.step_number, "ff_gate", index, value))
        .collect::<Result<Vec<_>>>()?;
    let value = step
        .ff_value
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_scalar_to_word(step.step_number, "ff_value", index, value))
        .collect::<Result<Vec<_>>>()?;
    let hidden = step
        .ff_hidden
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| exact_scalar_to_word(step.step_number, "ff_hidden", index, value))
        .collect::<Result<Vec<_>>>()?;

    let rows = hidden
        .chunks(ff_chunk_size)
        .enumerate()
        .map(|(chunk_index, hidden_values)| {
            let start = chunk_index * ff_chunk_size;
            let end = start + hidden_values.len();
            SharedTransformerTraceRow::FeedForwardChunk(SharedTransformerFeedForwardChunkTraceRow {
                step_number: step.step_number,
                chunk_index,
                hidden_start: chunk_index * ff_chunk_size,
                gate_values: gate[start..end].to_vec(),
                value_values: value[start..end].to_vec(),
                hidden_values: hidden_values.to_vec(),
            })
        })
        .collect();

    Ok(rows)
}

fn build_transition_row(
    step: &SharedTransformerStepWitness,
) -> Result<SharedTransformerTransitionTraceRow> {
    Ok(SharedTransformerTransitionTraceRow {
        step_number: step.step_number,
        ff_output: step
            .ff_output
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| exact_scalar_to_i128(step.step_number, "ff_output", index, value))
            .collect::<Result<Vec<_>>>()?,
        transition: step.transition.clone(),
        post_write_state: step.post_write_state.clone(),
    })
}

fn build_output_state_row(
    step: &SharedTransformerStepWitness,
) -> Result<SharedTransformerOutputTraceRow> {
    Ok(SharedTransformerOutputTraceRow {
        step_number: step.step_number,
        output_token: step
            .output_token
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| exact_token_value(step.step_number, "output_token", index, value))
            .collect::<Result<Vec<_>>>()?,
        final_state: step.final_state.clone(),
    })
}

fn exact_token_value(step_number: usize, location: &str, index: usize, value: f32) -> Result<i8> {
    match value {
        -1.0 => Ok(-1),
        0.0 => Ok(0),
        1.0 => Ok(1),
        _ => Err(VmError::UnsupportedProof(format!(
            "shared transformer staged trace requires discrete token values; {location}[{index}] at step {step_number} was {value}"
        ))),
    }
}

fn exact_scalar_to_word(
    step_number: usize,
    location: &str,
    index: usize,
    value: f64,
) -> Result<Word> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged trace requires exact integer values; {location}[{index}] at step {step_number} was {value}"
        )));
    }

    if value < Word::MIN as f64 || value > Word::MAX as f64 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged trace value out of i64 range; {location}[{index}] at step {step_number} was {value}"
        )));
    }

    Ok(value as Word)
}

fn exact_scalar_to_i128(
    step_number: usize,
    location: &str,
    index: usize,
    value: f64,
) -> Result<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged trace requires exact integer values; {location}[{index}] at step {step_number} was {value}"
        )));
    }

    if value < i128::MIN as f64 || value > i128::MAX as f64 {
        return Err(VmError::UnsupportedProof(format!(
            "shared transformer staged trace value out of i128 range; {location}[{index}] at step {step_number} was {value}"
        )));
    }

    Ok(value as i128)
}
