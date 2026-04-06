#![allow(dead_code)]

use serde::{Deserialize, Serialize};

use crate::config::{Attention2DMode, TransformerVmConfig};
use crate::error::{Result, VmError};
use crate::instruction::{Instruction, Program};
use crate::memory::AddressedMemory;
use crate::proof::VanillaStarkProofOptions;
use crate::state::{MachineState, Word};
use crate::vanillastark::{FieldElement, MPolynomial, Stark};

pub(crate) const PC: usize = 0;
pub(crate) const ACC: usize = 1;
pub(crate) const SP: usize = 2;
pub(crate) const ZERO: usize = 3;
pub(crate) const CARRY: usize = 4;
pub(crate) const HALTED: usize = 5;

const POET_WORD_BITS: usize = 1;
const TOKEN_LEN: usize = 9 + 2 * POET_WORD_BITS;
const OUTPUT_LEN: usize = 6;

const IN_CONST: usize = 0;
const IN_PC: usize = 1;
const IN_PC_NEXT: usize = 2;
const IN_ACC: usize = 3;
const IN_ZERO: usize = 4;
const IN_CARRY: usize = 5;
const IN_HALTED: usize = 6;
const IN_SP: usize = 7;
const IN_OPERAND: usize = 8;
const UNUSED_BITS_START: usize = 9;

const OUT_NEXT_PC: usize = 0;
const OUT_RAW_ACC: usize = 1;
const OUT_NEXT_SP: usize = 2;
const OUT_MEM_WRITE_ENABLE: usize = 3;
const OUT_MEM_WRITE_ADDR: usize = 4;
const OUT_MEM_WRITE_VALUE: usize = 5;

const STATUS_AVAILABLE: Word = 0;

type Scalar = f64;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) memory_size: usize,
    pub(crate) ff_dim: usize,
}

impl Layout {
    pub fn new(memory_size: usize, ff_dim: usize) -> Self {
        Self {
            memory_size,
            ff_dim,
        }
    }

    pub fn mem_start(&self) -> usize {
        6
    }

    pub fn token_start(&self) -> usize {
        6 + self.memory_size
    }

    pub fn hidden_start(&self) -> usize {
        6 + self.memory_size + TOKEN_LEN
    }

    pub fn output_start(&self) -> usize {
        6 + self.memory_size + TOKEN_LEN + self.ff_dim
    }

    pub fn acc_inv(&self) -> usize {
        6 + self.memory_size + TOKEN_LEN + self.ff_dim + OUTPUT_LEN
    }

    pub fn width(&self) -> usize {
        6 + self.memory_size + TOKEN_LEN + self.ff_dim + OUTPUT_LEN + 1
    }

    pub fn mem(&self, index: usize) -> usize {
        self.mem_start() + index
    }

    pub fn token(&self, index: usize) -> usize {
        self.token_start() + index
    }

    pub fn hidden(&self, index: usize) -> usize {
        self.hidden_start() + index
    }

    pub fn output(&self, index: usize) -> usize {
        self.output_start() + index
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoetExecutionClaim {
    pub program: Program,
    pub config: TransformerVmConfig,
    pub initial_memory: Vec<Word>,
    pub steps: usize,
    pub final_state: MachineState,
    pub options: VanillaStarkProofOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoetExecutionProof {
    pub claim: PoetExecutionClaim,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoetProfileAnalysis {
    pub required_ff_dim: usize,
    pub word_bits: usize,
    pub token_len: usize,
    pub instructions: Vec<PoetInstructionProfile>,
}

impl PoetProfileAnalysis {
    pub fn is_fully_supported(&self) -> bool {
        self.instructions
            .iter()
            .all(|instruction| instruction.unsupported_reason.is_none())
    }

    pub fn required_ff_dim_drivers(&self) -> Vec<&PoetInstructionProfile> {
        self.instructions
            .iter()
            .filter(|instruction| instruction.required_ff_dim == Some(self.required_ff_dim))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoetInstructionProfile {
    pub pc: usize,
    pub instruction: Instruction,
    pub required_ff_dim: Option<usize>,
    pub unsupported_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
enum PoetMemoryRead {
    #[default]
    None,
    Direct(u8),
    StackTop,
}

#[derive(Debug, Clone)]
struct PoetTransition {
    pc: i64,
    acc: Word,
    sp: i64,
    zero_flag: bool,
    carry_flag: bool,
    halted: bool,
    memory_write: Option<(i64, Word)>,
}

#[derive(Debug, Clone)]
struct PoetMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Scalar>,
}

impl PoetMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn add(&mut self, row: usize, col: usize, value: Scalar) {
        self.data[row * self.cols + col] += value;
    }

    fn mul_vec(&self, input: &[Scalar]) -> Vec<Scalar> {
        debug_assert_eq!(self.cols, input.len());
        let mut output = vec![0.0; self.rows];
        for (row_idx, row) in self.data.chunks(self.cols).enumerate() {
            output[row_idx] = row
                .iter()
                .zip(input.iter())
                .map(|(left, right)| left * right)
                .sum();
        }
        output
    }
}

#[derive(Debug, Clone)]
struct PoetFeedForwardWeights {
    gate: PoetMatrix,
    gate_bias: Vec<Scalar>,
    value: PoetMatrix,
    value_bias: Vec<Scalar>,
    out: PoetMatrix,
    out_bias: Vec<Scalar>,
}

impl PoetFeedForwardWeights {
    fn evaluate(&self, input: &[Scalar]) -> Vec<Scalar> {
        let gate = self
            .gate
            .mul_vec(input)
            .into_iter()
            .zip(self.gate_bias.iter())
            .map(|(value, bias)| value + bias)
            .collect::<Vec<_>>();
        let value = self
            .value
            .mul_vec(input)
            .into_iter()
            .zip(self.value_bias.iter())
            .map(|(value, bias)| value + bias)
            .collect::<Vec<_>>();
        let hidden = gate
            .into_iter()
            .zip(value)
            .map(|(left, right)| left * right)
            .collect::<Vec<_>>();

        self.out
            .mul_vec(&hidden)
            .into_iter()
            .zip(self.out_bias.iter())
            .map(|(value, bias)| value + bias)
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
enum HiddenOperand {
    Input(usize),
    Const(Scalar),
}

#[derive(Debug, Clone)]
struct PoetFeedForwardBuilder {
    weights: PoetFeedForwardWeights,
    next_hidden: usize,
}

impl PoetFeedForwardBuilder {
    fn new(ff_dim: usize) -> Self {
        Self {
            weights: PoetFeedForwardWeights {
                gate: PoetMatrix::zeros(ff_dim, TOKEN_LEN),
                gate_bias: vec![0.0; ff_dim],
                value: PoetMatrix::zeros(ff_dim, TOKEN_LEN),
                value_bias: vec![0.0; ff_dim],
                out: PoetMatrix::zeros(OUTPUT_LEN, ff_dim),
                out_bias: vec![0.0; OUTPUT_LEN],
            },
            next_hidden: 0,
        }
    }

    fn emit_linear(&mut self, output_idx: usize, coeff: Scalar, input_idx: usize) -> Result<()> {
        self.emit_term(
            output_idx,
            coeff,
            HiddenOperand::Const(1.0),
            HiddenOperand::Input(input_idx),
        )
    }

    fn emit_product(
        &mut self,
        output_idx: usize,
        coeff: Scalar,
        left: usize,
        right: usize,
    ) -> Result<()> {
        self.emit_term(
            output_idx,
            coeff,
            HiddenOperand::Input(left),
            HiddenOperand::Input(right),
        )
    }

    fn add_output_bias(&mut self, output_idx: usize, value: Scalar) {
        self.weights.out_bias[output_idx] += value;
    }

    fn finalize(self) -> Result<PoetFeedForwardWeights> {
        if self.next_hidden > self.weights.gate.rows {
            return Err(VmError::InvalidConfig(format!(
                "poet compiled instruction requires {} FF hidden units, config only provides {}",
                self.next_hidden, self.weights.gate.rows
            )));
        }
        Ok(self.weights)
    }

    fn emit_term(
        &mut self,
        output_idx: usize,
        coeff: Scalar,
        left: HiddenOperand,
        right: HiddenOperand,
    ) -> Result<()> {
        if self.next_hidden >= self.weights.gate.rows {
            return Err(VmError::InvalidConfig(format!(
                "poet compiled instruction exceeds ff_dim {}; need at least {}",
                self.weights.gate.rows,
                self.next_hidden + 1
            )));
        }

        let hidden_idx = self.next_hidden;
        self.next_hidden += 1;

        match left {
            HiddenOperand::Input(input_idx) => self.weights.gate.add(hidden_idx, input_idx, 1.0),
            HiddenOperand::Const(value) => self.weights.gate_bias[hidden_idx] += value,
        }

        match right {
            HiddenOperand::Input(input_idx) => self.weights.value.add(hidden_idx, input_idx, 1.0),
            HiddenOperand::Const(value) => self.weights.value_bias[hidden_idx] += value,
        }

        self.weights.out.add(output_idx, hidden_idx, coeff);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PoetBoolBlend {
    prev_weight: Scalar,
    result_weight: Scalar,
    constant: Scalar,
}

#[derive(Debug, Clone, Copy, Default)]
struct PoetCarryBlend {
    prev_weight: Scalar,
    overflow_weight: Scalar,
    less_than_weight: Scalar,
    constant: Scalar,
    rhs_constant: Scalar,
    rhs_operand_weight: Scalar,
}

#[derive(Debug, Clone, Copy, Default)]
struct PoetTransitionControls {
    zero: PoetBoolBlend,
    carry: PoetCarryBlend,
    halted: PoetBoolBlend,
}

impl PoetTransitionControls {
    fn preserve() -> Self {
        Self {
            zero: PoetBoolBlend {
                prev_weight: 1.0,
                ..PoetBoolBlend::default()
            },
            carry: PoetCarryBlend {
                prev_weight: 1.0,
                ..PoetCarryBlend::default()
            },
            halted: PoetBoolBlend::default(),
        }
    }

    fn zero_from_result() -> PoetBoolBlend {
        PoetBoolBlend {
            result_weight: 1.0,
            ..PoetBoolBlend::default()
        }
    }

    fn halted_const(value: bool) -> PoetBoolBlend {
        PoetBoolBlend {
            constant: bool_to_scalar(value),
            ..PoetBoolBlend::default()
        }
    }
}

#[derive(Debug, Clone)]
struct PoetCompiledInstruction {
    memory_read: PoetMemoryRead,
    ff_weights: PoetFeedForwardWeights,
    controls: PoetTransitionControls,
}

#[derive(Debug, Clone)]
struct PoetVm {
    config: TransformerVmConfig,
    program: Program,
    compiled: Vec<PoetCompiledInstruction>,
}

impl PoetVm {
    fn new(config: TransformerVmConfig, program: Program) -> Result<Self> {
        if config.ff_dim == 0 {
            return Err(VmError::InvalidConfig(
                "poet ff_dim must be greater than zero".to_string(),
            ));
        }
        let analysis = analyze_poet_program(&program)?;
        if config.ff_dim < analysis.required_ff_dim {
            let drivers = analysis
                .required_ff_dim_drivers()
                .into_iter()
                .map(|instruction| format!("pc {} `{}`", instruction.pc, instruction.instruction))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(VmError::InvalidConfig(format!(
                "poet reduced profile requires ff_dim >= {}, got {} (driven by {})",
                analysis.required_ff_dim, config.ff_dim, drivers
            )));
        }
        if program.memory_size() > usize::from(u8::MAX) {
            return Err(VmError::InvalidConfig(format!(
                "program memory size {} exceeds the encoded stack/address limit of {} cells",
                program.memory_size(),
                u8::MAX
            )));
        }

        let compiler = PoetInstructionCompiler::new(config.ff_dim);
        let compiled = program
            .instructions()
            .iter()
            .copied()
            .map(|instruction| compiler.compile_instruction(instruction))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            config,
            program,
            compiled,
        })
    }

    fn compiled_instruction(&self, pc: u8) -> Result<&PoetCompiledInstruction> {
        self.compiled
            .get(pc as usize)
            .ok_or(VmError::ProgramCounterOutOfBounds {
                pc: pc as usize,
                len: self.compiled.len(),
            })
    }

    fn step(
        &self,
        state: &MachineState,
        memory: &mut AddressedMemory,
        step_number: usize,
    ) -> Result<MachineState> {
        if state.halted {
            return Ok(state.clone());
        }

        let instruction = self.program.instruction_at(state.pc)?;
        validate_stack_pointer(state.sp, memory.len())?;
        validate_stack_precondition(instruction, state.sp, memory.len())?;

        let compiled = self.compiled_instruction(state.pc)?;
        let operand = read_operand(
            memory,
            state,
            compiled.memory_read,
            &self.config.attention_mode,
        )?;
        let input = build_poet_input_vector(state, operand);
        let output = compiled.ff_weights.evaluate(&input);
        let transition = poet_transition_from_output(state, operand, &compiled.controls, &output);

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
}

pub fn inspect_poet_program(program: &Program) -> PoetProfileAnalysis {
    let instructions = program
        .instructions()
        .iter()
        .copied()
        .enumerate()
        .map(|(pc, instruction)| PoetInstructionProfile {
            pc,
            instruction,
            required_ff_dim: poet_instruction_required_ff_dim(instruction),
            unsupported_reason: poet_instruction_unsupported_reason(instruction)
                .map(str::to_string),
        })
        .collect::<Vec<_>>();
    let required_ff_dim = instructions
        .iter()
        .filter_map(|instruction| instruction.required_ff_dim)
        .max()
        .unwrap_or(0);
    PoetProfileAnalysis {
        required_ff_dim,
        word_bits: POET_WORD_BITS,
        token_len: TOKEN_LEN,
        instructions,
    }
}

pub fn analyze_poet_program(program: &Program) -> Result<PoetProfileAnalysis> {
    let analysis = inspect_poet_program(program);
    for instruction in &analysis.instructions {
        if let Some(reason) = &instruction.unsupported_reason {
            return Err(VmError::UnsupportedProof(format!(
                "poet reduced profile does not yet support instruction `{}` at pc {}: {}",
                instruction.instruction, instruction.pc, reason
            )));
        }
    }
    Ok(analysis)
}

fn poet_instruction_unsupported_reason(instruction: Instruction) -> Option<&'static str> {
    match instruction {
        Instruction::AndImmediate(_)
        | Instruction::AndMemory(_)
        | Instruction::OrImmediate(_)
        | Instruction::OrMemory(_)
        | Instruction::XorImmediate(_)
        | Instruction::XorMemory(_) => {
            Some("bitwise instructions are not represented in the reduced low-degree Poet profile")
        }
        Instruction::JumpIfSign(_) | Instruction::JumpIfNotSign(_) => Some(
            "sign-branch instructions require sign-aware state/AIR that the reduced Poet profile does not yet encode",
        ),
        _ => None,
    }
}

fn poet_instruction_required_ff_dim(instruction: Instruction) -> Option<usize> {
    match instruction {
        Instruction::Nop => Some(3),
        Instruction::LoadImmediate(_) => Some(2),
        Instruction::Load(_) => Some(3),
        Instruction::Store(_) => Some(4),
        Instruction::Push => Some(5),
        Instruction::Pop => Some(3),
        Instruction::AddImmediate(_) => Some(3),
        Instruction::AddMemory(_) => Some(4),
        Instruction::SubImmediate(_) => Some(3),
        Instruction::SubMemory(_) => Some(4),
        Instruction::MulImmediate(_) => Some(3),
        Instruction::MulMemory(_) => Some(3),
        Instruction::CmpImmediate(_) => Some(3),
        Instruction::CmpMemory(_) => Some(4),
        Instruction::Call(_) => Some(4),
        Instruction::Ret => Some(3),
        Instruction::Jump(_) => Some(2),
        Instruction::JumpIfZero(_) => Some(5),
        Instruction::JumpIfNotZero(_) => Some(4),
        Instruction::Halt => Some(3),
        Instruction::AndImmediate(_)
        | Instruction::AndMemory(_)
        | Instruction::OrImmediate(_)
        | Instruction::OrMemory(_)
        | Instruction::XorImmediate(_)
        | Instruction::XorMemory(_)
        | Instruction::JumpIfSign(_)
        | Instruction::JumpIfNotSign(_) => None,
    }
}

#[derive(Debug, Clone)]
struct PoetInstructionCompiler {
    ff_dim: usize,
}

impl PoetInstructionCompiler {
    fn new(ff_dim: usize) -> Self {
        Self { ff_dim }
    }

    fn compile_instruction(&self, instruction: Instruction) -> Result<PoetCompiledInstruction> {
        match instruction {
            Instruction::Nop => self.compile_nop(),
            Instruction::LoadImmediate(value) => self.compile_load_immediate(value),
            Instruction::Load(address) => self.compile_load(address),
            Instruction::Store(address) => self.compile_store(address),
            Instruction::Push => self.compile_push(),
            Instruction::Pop => self.compile_pop(),
            Instruction::AddImmediate(value) => self.compile_add_immediate(value),
            Instruction::AddMemory(address) => self.compile_add_memory(address),
            Instruction::SubImmediate(value) => self.compile_sub_immediate(value),
            Instruction::SubMemory(address) => self.compile_sub_memory(address),
            Instruction::MulImmediate(value) => self.compile_mul_immediate(value),
            Instruction::MulMemory(address) => self.compile_mul_memory(address),
            Instruction::CmpImmediate(value) => self.compile_cmp_immediate(value),
            Instruction::CmpMemory(address) => self.compile_cmp_memory(address),
            Instruction::Call(target) => self.compile_call(target),
            Instruction::Ret => self.compile_ret(),
            Instruction::Jump(target) => self.compile_jump(target),
            Instruction::JumpIfZero(target) => self.compile_jump_if_zero(target),
            Instruction::JumpIfNotZero(target) => self.compile_jump_if_not_zero(target),
            Instruction::Halt => self.compile_halt(),
            Instruction::AndImmediate(_)
            | Instruction::AndMemory(_)
            | Instruction::OrImmediate(_)
            | Instruction::OrMemory(_)
            | Instruction::XorImmediate(_)
            | Instruction::XorMemory(_)
            | Instruction::JumpIfSign(_)
            | Instruction::JumpIfNotSign(_) => Err(VmError::UnsupportedProof(
                "poet reduced profile does not yet support bitwise or sign-branch instructions"
                    .to_string(),
            )),
        }
    }

    fn builder(&self) -> PoetFeedForwardBuilder {
        PoetFeedForwardBuilder::new(self.ff_dim)
    }

    fn preserve_sp(&self, builder: &mut PoetFeedForwardBuilder) -> Result<()> {
        builder.emit_linear(OUT_NEXT_SP, 1.0, IN_SP)
    }

    fn compile_nop(&self) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_load_immediate(&self, value: Word) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.add_output_bias(OUT_RAW_ACC, word_to_scalar(value));
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls {
                zero: PoetTransitionControls::zero_from_result(),
                carry: PoetCarryBlend::default(),
                halted: PoetTransitionControls::halted_const(false),
            },
        )
    }

    fn compile_load(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_OPERAND)?;
        self.finish(
            PoetMemoryRead::Direct(address),
            builder,
            PoetTransitionControls {
                zero: PoetTransitionControls::zero_from_result(),
                carry: PoetCarryBlend::default(),
                halted: PoetTransitionControls::halted_const(false),
            },
        )
    }

    fn compile_store(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.add_output_bias(OUT_MEM_WRITE_ENABLE, 1.0);
        builder.add_output_bias(OUT_MEM_WRITE_ADDR, Scalar::from(address));
        builder.emit_linear(OUT_MEM_WRITE_VALUE, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_push(&self) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_NEXT_SP, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_NEXT_SP, -1.0);
        builder.add_output_bias(OUT_MEM_WRITE_ENABLE, 1.0);
        builder.emit_linear(OUT_MEM_WRITE_ADDR, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_MEM_WRITE_ADDR, -1.0);
        builder.emit_linear(OUT_MEM_WRITE_VALUE, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_pop(&self) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_OPERAND)?;
        builder.emit_linear(OUT_NEXT_SP, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_NEXT_SP, 1.0);
        self.finish(
            PoetMemoryRead::StackTop,
            builder,
            PoetTransitionControls {
                zero: PoetTransitionControls::zero_from_result(),
                carry: PoetCarryBlend::default(),
                halted: PoetTransitionControls::halted_const(false),
            },
        )
    }

    fn compile_add_immediate(&self, value: Word) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.add_output_bias(OUT_RAW_ACC, word_to_scalar(value));
        self.finish(PoetMemoryRead::None, builder, arithmetic_controls())
    }

    fn compile_add_memory(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_OPERAND)?;
        self.finish(
            PoetMemoryRead::Direct(address),
            builder,
            arithmetic_controls(),
        )
    }

    fn compile_sub_immediate(&self, value: Word) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.add_output_bias(OUT_RAW_ACC, -word_to_scalar(value));
        self.finish(PoetMemoryRead::None, builder, arithmetic_controls())
    }

    fn compile_sub_memory(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_RAW_ACC, -1.0, IN_OPERAND)?;
        self.finish(
            PoetMemoryRead::Direct(address),
            builder,
            arithmetic_controls(),
        )
    }

    fn compile_mul_immediate(&self, value: Word) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, word_to_scalar(value), IN_ACC)?;
        self.finish(PoetMemoryRead::None, builder, arithmetic_controls())
    }

    fn compile_mul_memory(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_product(OUT_RAW_ACC, 1.0, IN_ACC, IN_OPERAND)?;
        self.finish(
            PoetMemoryRead::Direct(address),
            builder,
            arithmetic_controls(),
        )
    }

    fn compile_cmp_immediate(&self, value: Word) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.add_output_bias(OUT_RAW_ACC, -word_to_scalar(value));
        self.finish(
            PoetMemoryRead::None,
            builder,
            compare_controls(word_to_scalar(value), 0.0),
        )
    }

    fn compile_cmp_memory(&self, address: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_RAW_ACC, -1.0, IN_OPERAND)?;
        self.finish(
            PoetMemoryRead::Direct(address),
            builder,
            compare_controls(0.0, 1.0),
        )
    }

    fn compile_call(&self, target: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        builder.add_output_bias(OUT_NEXT_PC, Scalar::from(target));
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_NEXT_SP, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_NEXT_SP, -1.0);
        builder.add_output_bias(OUT_MEM_WRITE_ENABLE, 1.0);
        builder.emit_linear(OUT_MEM_WRITE_ADDR, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_MEM_WRITE_ADDR, -1.0);
        builder.emit_linear(OUT_MEM_WRITE_VALUE, 1.0, IN_PC_NEXT)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_ret(&self) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_OPERAND)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        builder.emit_linear(OUT_NEXT_SP, 1.0, IN_SP)?;
        builder.add_output_bias(OUT_NEXT_SP, 1.0);
        self.finish(
            PoetMemoryRead::StackTop,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_jump(&self, target: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.add_output_bias(OUT_NEXT_PC, Scalar::from(target));
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_jump_if_zero(&self, target: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC_NEXT)?;
        builder.emit_linear(OUT_NEXT_PC, Scalar::from(target), IN_ZERO)?;
        builder.emit_product(OUT_NEXT_PC, -1.0, IN_ZERO, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_jump_if_not_zero(&self, target: u8) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.add_output_bias(OUT_NEXT_PC, Scalar::from(target));
        builder.emit_linear(OUT_NEXT_PC, -Scalar::from(target), IN_ZERO)?;
        builder.emit_product(OUT_NEXT_PC, 1.0, IN_ZERO, IN_PC_NEXT)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls::preserve(),
        )
    }

    fn compile_halt(&self) -> Result<PoetCompiledInstruction> {
        let mut builder = self.builder();
        self.preserve_sp(&mut builder)?;
        builder.emit_linear(OUT_NEXT_PC, 1.0, IN_PC)?;
        builder.emit_linear(OUT_RAW_ACC, 1.0, IN_ACC)?;
        self.finish(
            PoetMemoryRead::None,
            builder,
            PoetTransitionControls {
                zero: PoetBoolBlend {
                    prev_weight: 1.0,
                    ..PoetBoolBlend::default()
                },
                carry: PoetCarryBlend {
                    prev_weight: 1.0,
                    ..PoetCarryBlend::default()
                },
                halted: PoetTransitionControls::halted_const(true),
            },
        )
    }

    fn finish(
        &self,
        memory_read: PoetMemoryRead,
        builder: PoetFeedForwardBuilder,
        controls: PoetTransitionControls,
    ) -> Result<PoetCompiledInstruction> {
        Ok(PoetCompiledInstruction {
            memory_read,
            ff_weights: builder.finalize()?,
            controls,
        })
    }
}

pub struct PoetVmAir {
    layout: Layout,
    pub claim: PoetExecutionClaim,
}

impl PoetVmAir {
    pub fn new(claim: PoetExecutionClaim) -> Self {
        let layout = Layout::new(claim.program.memory_size(), claim.config.ff_dim);
        Self { layout, claim }
    }

    pub fn boundary_constraints(
        &self,
        steps: usize,
        final_state: &MachineState,
    ) -> Vec<(usize, usize, FieldElement)> {
        let mut boundary = Vec::new();
        let initial_state = initial_state_from_claim(&self.claim);

        boundary.push((0, PC, field_from_i128(initial_state.pc as i128)));
        boundary.push((0, ACC, field_from_i128(initial_state.acc as i128)));
        boundary.push((0, SP, field_from_i128(initial_state.sp as i128)));
        boundary.push((0, ZERO, field_from_i128(initial_state.zero_flag as i128)));
        boundary.push((0, CARRY, field_from_i128(initial_state.carry_flag as i128)));
        boundary.push((0, HALTED, field_from_i128(initial_state.halted as i128)));
        for (i, &val) in self.claim.initial_memory.iter().enumerate() {
            boundary.push((0, self.layout.mem(i), field_from_i128(val as i128)));
        }

        boundary.push((steps, PC, field_from_i128(final_state.pc as i128)));
        boundary.push((steps, ACC, field_from_i128(final_state.acc as i128)));
        boundary.push((steps, SP, field_from_i128(final_state.sp as i128)));
        boundary.push((steps, ZERO, field_from_i128(final_state.zero_flag as i128)));
        boundary.push((
            steps,
            CARRY,
            field_from_i128(final_state.carry_flag as i128),
        ));
        boundary.push((steps, HALTED, field_from_i128(final_state.halted as i128)));
        for (i, &val) in final_state.memory.iter().enumerate() {
            boundary.push((steps, self.layout.mem(i), field_from_i128(val as i128)));
        }

        boundary
    }

    pub fn transition_constraints(&self) -> Vec<MPolynomial> {
        let width = self.layout.width();
        let variables = MPolynomial::variables(1 + 2 * width);
        let current = &variables[1..1 + width];
        let next = &variables[1 + width..];
        let one = mp_constant(1);
        let zero = MPolynomial::zero();
        let current_pc = current[PC].clone();
        let current_acc = current[ACC].clone();
        let current_sp = current[SP].clone();
        let current_zero = current[ZERO].clone();
        let current_carry = current[CARRY].clone();
        let current_halted = current[HALTED].clone();
        let next_pc = next[PC].clone();
        let next_acc = next[ACC].clone();
        let next_sp = next[SP].clone();
        let next_zero = next[ZERO].clone();
        let next_carry = next[CARRY].clone();
        let next_halted = next[HALTED].clone();
        let not_halted = one.clone() - current_halted.clone();

        let mut constraints = Vec::new();
        constraints.push(current_zero.clone() * (current_zero.clone() - one.clone()));
        constraints.push(current_carry.clone() * (current_carry.clone() - one.clone()));
        constraints.push(current_halted.clone() * (current_halted.clone() - one.clone()));

        let vm = PoetVm::new(self.claim.config.clone(), self.claim.program.clone())
            .expect("poet program compiles for AIR");
        let pc_selectors = self.pc_selectors(&current_pc);
        let sp_selectors = sp_selectors(&current_sp, self.layout.memory_size);

        let stack_top = select_memory(&sp_selectors, current, &self.layout);
        let mut expected_output = vec![zero.clone(); OUTPUT_LEN];
        let mut expected_zero_blend = zero.clone();
        let mut expected_carry_blend = zero.clone();
        let mut expected_halted_blend = zero.clone();

        for (pc_idx, compiled) in vm.compiled.iter().enumerate() {
            let selector = pc_selectors[pc_idx].clone();
            let operand = operand_expr(compiled.memory_read, current, &self.layout, &stack_top);
            let gated_selector = not_halted.clone() * selector.clone();

            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_CONST)].clone() - one.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_PC)].clone() - current_pc.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_PC_NEXT)].clone()
                        - (current_pc.clone() + one.clone())),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_ACC)].clone() - current_acc.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_ZERO)].clone() - current_zero.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_CARRY)].clone() - current_carry.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_HALTED)].clone() - current_halted.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_SP)].clone() - current_sp.clone()),
            );
            constraints.push(
                gated_selector.clone()
                    * (current[self.layout.token(IN_OPERAND)].clone() - operand.clone()),
            );
            for bit in UNUSED_BITS_START..TOKEN_LEN {
                constraints.push(gated_selector.clone() * current[self.layout.token(bit)].clone());
            }

            let ff = &compiled.ff_weights;
            for hidden_idx in 0..self.layout.ff_dim {
                let mut gate_sum = mp_constant_from_scalar(ff.gate_bias[hidden_idx]);
                let mut value_sum = mp_constant_from_scalar(ff.value_bias[hidden_idx]);
                for token_idx in 0..TOKEN_LEN {
                    let token = current[self.layout.token(token_idx)].clone();
                    gate_sum = gate_sum
                        + token.clone()
                            * mp_constant_from_scalar(
                                ff.gate.data[hidden_idx * ff.gate.cols + token_idx],
                            );
                    value_sum = value_sum
                        + token
                            * mp_constant_from_scalar(
                                ff.value.data[hidden_idx * ff.value.cols + token_idx],
                            );
                }
                let hidden = current[self.layout.hidden(hidden_idx)].clone();
                constraints.push(gated_selector.clone() * (hidden - gate_sum * value_sum));
            }

            for (output_idx, expected_out) in
                expected_output.iter_mut().enumerate().take(OUTPUT_LEN)
            {
                let mut out_sum = mp_constant_from_scalar(ff.out_bias[output_idx]);
                for hidden_idx in 0..self.layout.ff_dim {
                    out_sum = out_sum
                        + current[self.layout.hidden(hidden_idx)].clone()
                            * mp_constant_from_scalar(
                                ff.out.data[output_idx * ff.out.cols + hidden_idx],
                            );
                }
                *expected_out = expected_out.clone() + selector.clone() * out_sum;
            }

            let controls = compiled.controls;
            expected_zero_blend = expected_zero_blend
                + selector.clone()
                    * (mp_constant_from_scalar(controls.zero.prev_weight) * current_zero.clone()
                        + mp_constant_from_scalar(controls.zero.result_weight) * next_zero.clone()
                        + mp_constant_from_scalar(controls.zero.constant));
            expected_carry_blend = expected_carry_blend
                + selector.clone()
                    * (mp_constant_from_scalar(controls.carry.prev_weight) * current_carry.clone()
                        + mp_constant_from_scalar(controls.carry.constant));
            expected_halted_blend = expected_halted_blend
                + selector.clone()
                    * (mp_constant_from_scalar(controls.halted.prev_weight)
                        * current_halted.clone()
                        + mp_constant_from_scalar(controls.halted.constant));
        }

        constraints
            .push(not_halted.clone() * (next_pc.clone() - expected_output[OUT_NEXT_PC].clone()));
        constraints
            .push(not_halted.clone() * (next_acc.clone() - expected_output[OUT_RAW_ACC].clone()));
        constraints
            .push(not_halted.clone() * (next_sp.clone() - expected_output[OUT_NEXT_SP].clone()));

        let next_acc_inv = next[self.layout.acc_inv()].clone();
        constraints.push(next_acc.clone() * next_acc_inv - (one.clone() - next_zero.clone()));
        constraints.push(next_zero.clone() * next_acc.clone());
        constraints.push(not_halted.clone() * (next_zero.clone() - expected_zero_blend));
        constraints.push(not_halted.clone() * (next_carry.clone() - expected_carry_blend));
        constraints.push(not_halted.clone() * (next_halted.clone() - expected_halted_blend));

        let write_enable = expected_output[OUT_MEM_WRITE_ENABLE].clone();
        let write_addr = expected_output[OUT_MEM_WRITE_ADDR].clone();
        let write_val = expected_output[OUT_MEM_WRITE_VALUE].clone();
        let mem_domain: Vec<i128> = (0..self.layout.memory_size)
            .map(|value| value as i128)
            .collect();
        for index in 0..self.layout.memory_size {
            let is_write_target = lagrange_selector(&write_addr, &mem_domain, index as i128);
            let current_mem = current[self.layout.mem(index)].clone();
            let next_mem = next[self.layout.mem(index)].clone();
            let expected_mem = write_enable.clone() * is_write_target.clone() * write_val.clone()
                + (one.clone() - write_enable.clone() * is_write_target) * current_mem.clone();
            constraints.push(not_halted.clone() * (next_mem.clone() - expected_mem));
            constraints.push(current_halted.clone() * (next_mem - current_mem));
        }

        constraints.push(current_halted.clone() * (next_pc - current_pc));
        constraints.push(current_halted.clone() * (next_acc - current_acc));
        constraints.push(current_halted.clone() * (next_sp - current_sp));
        constraints.push(current_halted.clone() * (next_zero - current_zero));
        constraints.push(current_halted.clone() * (next_carry - current_carry));
        constraints.push(current_halted.clone() * (next_halted - one));

        constraints
    }

    pub fn transition_degree_bound(&self) -> usize {
        self.transition_constraints()
            .iter()
            .flat_map(|constraint| constraint.dictionary.keys())
            .map(|powers| powers.iter().sum())
            .max()
            .unwrap_or(1)
            .max(2)
    }

    fn pc_selectors(&self, pc: &MPolynomial) -> Vec<MPolynomial> {
        let domain: Vec<i128> = (0..self.claim.program.len())
            .map(|value| value as i128)
            .collect();
        domain
            .iter()
            .map(|&point| lagrange_selector(pc, &domain, point))
            .collect()
    }
}

pub fn prove_poet_execution(
    program: &Program,
    initial_memory: &[Word],
    config: &TransformerVmConfig,
    options: VanillaStarkProofOptions,
) -> Result<PoetExecutionProof> {
    let vm = PoetVm::new(config.clone(), program.clone())?;
    let memory_data = if initial_memory.is_empty() {
        program.initial_memory().to_vec()
    } else {
        initial_memory.to_vec()
    };

    let mut memory = AddressedMemory::from_initial(&memory_data);
    let mut state = MachineState::new(program.memory_size());
    state.memory = memory.snapshot();

    let layout = Layout::new(program.memory_size(), config.ff_dim);
    let mut trace = Vec::new();
    let mut step_count = 0;
    while !state.halted && step_count < 10_000 {
        trace.push(generate_trace_row(&vm, &state, &memory, &layout)?);
        state = vm.step(&state, &mut memory, step_count)?;
        step_count += 1;
    }
    trace.push(generate_trace_row(&vm, &state, &memory, &layout)?);

    let claim = PoetExecutionClaim {
        program: program.clone(),
        config: config.clone(),
        initial_memory: memory_data,
        steps: step_count,
        final_state: state,
        options: options.clone(),
    };

    let air = PoetVmAir::new(claim.clone());
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.layout.width(),
        claim.steps + 1,
        air.transition_degree_bound(),
    );

    let boundary = air.boundary_constraints(claim.steps, &claim.final_state);
    let constraints = air.transition_constraints();
    let proof = stark.prove(&trace, &constraints, &boundary);

    Ok(PoetExecutionProof { claim, proof })
}

pub fn verify_poet_execution(proof: &PoetExecutionProof) -> Result<bool> {
    let air = PoetVmAir::new(proof.claim.clone());
    let stark = Stark::new(
        proof.claim.options.expansion_factor,
        proof.claim.options.num_colinearity_checks,
        proof.claim.options.security_level,
        air.layout.width(),
        proof.claim.steps + 1,
        air.transition_degree_bound(),
    );
    let boundary = air.boundary_constraints(proof.claim.steps, &proof.claim.final_state);
    let constraints = air.transition_constraints();
    Ok(stark.verify(&proof.proof, &constraints, &boundary))
}

fn initial_state_from_claim(claim: &PoetExecutionClaim) -> MachineState {
    let mut state = MachineState::new(claim.program.memory_size());
    state.memory = claim.initial_memory.clone();
    state
}

fn generate_trace_row(
    vm: &PoetVm,
    state: &MachineState,
    memory: &AddressedMemory,
    layout: &Layout,
) -> Result<Vec<FieldElement>> {
    let mut row = vec![FieldElement::zero(); layout.width()];

    row[PC] = field_from_i128(state.pc as i128);
    row[ACC] = field_from_i128(state.acc as i128);
    row[SP] = field_from_i128(state.sp as i128);
    row[ZERO] = field_from_i128(state.zero_flag as i128);
    row[CARRY] = field_from_i128(state.carry_flag as i128);
    row[HALTED] = field_from_i128(state.halted as i128);
    for index in 0..layout.memory_size {
        row[layout.mem(index)] = field_from_i128(state.memory[index] as i128);
    }
    row[layout.acc_inv()] = inverse_or_zero(field_from_i128(state.acc as i128));

    if state.halted {
        return Ok(row);
    }

    let compiled = vm.compiled_instruction(state.pc)?;
    let operand = read_operand(
        memory,
        state,
        compiled.memory_read,
        &vm.config.attention_mode,
    )?;
    let input = build_poet_input_vector(state, operand);
    for (index, value) in input.iter().enumerate() {
        row[layout.token(index)] = scalar_to_field(*value);
    }

    let gate = compiled.ff_weights.gate.mul_vec(&input);
    let value = compiled.ff_weights.value.mul_vec(&input);
    let hidden = gate
        .iter()
        .zip(compiled.ff_weights.gate_bias.iter())
        .zip(value.iter())
        .zip(compiled.ff_weights.value_bias.iter())
        .map(|(((gate, gate_bias), value), value_bias)| (gate + gate_bias) * (value + value_bias))
        .collect::<Vec<_>>();
    for (index, entry) in hidden.iter().enumerate() {
        row[layout.hidden(index)] = scalar_to_field(*entry);
    }

    let output = compiled.ff_weights.out.mul_vec(&hidden);
    for output_idx in 0..OUTPUT_LEN {
        row[layout.output(output_idx)] =
            scalar_to_field(output[output_idx] + compiled.ff_weights.out_bias[output_idx]);
    }

    Ok(row)
}

fn build_poet_input_vector(state: &MachineState, operand: Word) -> Vec<Scalar> {
    let mut input = vec![0.0; TOKEN_LEN];
    input[IN_CONST] = 1.0;
    input[IN_PC] = Scalar::from(state.pc);
    input[IN_PC_NEXT] = Scalar::from(state.pc.wrapping_add(1));
    input[IN_ACC] = word_to_scalar(state.acc);
    input[IN_ZERO] = bool_to_scalar(state.zero_flag);
    input[IN_CARRY] = bool_to_scalar(state.carry_flag);
    input[IN_HALTED] = bool_to_scalar(state.halted);
    input[IN_SP] = Scalar::from(state.sp);
    input[IN_OPERAND] = word_to_scalar(operand);
    input
}

fn read_operand(
    memory: &AddressedMemory,
    state: &MachineState,
    memory_read: PoetMemoryRead,
    attention_mode: &Attention2DMode,
) -> Result<Word> {
    match memory_read {
        PoetMemoryRead::None => Ok(0),
        PoetMemoryRead::Direct(address) => memory.load_with_mode(address, attention_mode),
        PoetMemoryRead::StackTop => memory.load_with_mode(state.sp, attention_mode),
    }
}

fn operand_expr(
    memory_read: PoetMemoryRead,
    current: &[MPolynomial],
    layout: &Layout,
    stack_top: &MPolynomial,
) -> MPolynomial {
    match memory_read {
        PoetMemoryRead::None => MPolynomial::zero(),
        PoetMemoryRead::Direct(address) => current[layout.mem(usize::from(address))].clone(),
        PoetMemoryRead::StackTop => stack_top.clone(),
    }
}

fn sp_selectors(sp: &MPolynomial, memory_size: usize) -> Vec<MPolynomial> {
    let domain: Vec<i128> = (0..memory_size).map(|value| value as i128).collect();
    domain
        .iter()
        .map(|&point| lagrange_selector(sp, &domain, point))
        .collect()
}

fn select_memory(
    selectors: &[MPolynomial],
    current: &[MPolynomial],
    layout: &Layout,
) -> MPolynomial {
    selectors
        .iter()
        .enumerate()
        .fold(MPolynomial::zero(), |sum, (index, selector)| {
            sum + selector.clone() * current[layout.mem(index)].clone()
        })
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

fn arithmetic_controls() -> PoetTransitionControls {
    PoetTransitionControls {
        zero: PoetTransitionControls::zero_from_result(),
        carry: PoetCarryBlend {
            overflow_weight: 1.0,
            ..PoetCarryBlend::default()
        },
        halted: PoetTransitionControls::halted_const(false),
    }
}

fn compare_controls(rhs_constant: Scalar, rhs_operand_weight: Scalar) -> PoetTransitionControls {
    PoetTransitionControls {
        zero: PoetTransitionControls::zero_from_result(),
        carry: PoetCarryBlend {
            less_than_weight: 1.0,
            rhs_constant,
            rhs_operand_weight,
            ..PoetCarryBlend::default()
        },
        halted: PoetTransitionControls::halted_const(false),
    }
}

fn poet_transition_from_output(
    state: &MachineState,
    operand: Word,
    controls: &PoetTransitionControls,
    output: &[Scalar],
) -> PoetTransition {
    let raw_acc = output[OUT_RAW_ACC].round();
    let acc = raw_acc as Word;
    let result_zero = acc == 0;
    let overflow =
        !raw_acc.is_finite() || raw_acc < Word::MIN as Scalar || raw_acc > Word::MAX as Scalar;
    let compare_rhs = (controls.carry.rhs_constant
        + controls.carry.rhs_operand_weight * word_to_scalar(operand))
    .round() as Word;
    let less_than = state.acc < compare_rhs;

    PoetTransition {
        pc: output[OUT_NEXT_PC].round() as i64,
        acc,
        sp: output[OUT_NEXT_SP].round() as i64,
        zero_flag: blend_bool(
            controls.zero.prev_weight * bool_to_scalar(state.zero_flag)
                + controls.zero.result_weight * bool_to_scalar(result_zero)
                + controls.zero.constant,
        ),
        carry_flag: blend_bool(
            controls.carry.prev_weight * bool_to_scalar(state.carry_flag)
                + controls.carry.overflow_weight * bool_to_scalar(overflow)
                + controls.carry.less_than_weight * bool_to_scalar(less_than)
                + controls.carry.constant,
        ),
        halted: blend_bool(
            controls.halted.prev_weight * bool_to_scalar(state.halted)
                + controls.halted.result_weight * bool_to_scalar(false)
                + controls.halted.constant,
        ),
        memory_write: (output[OUT_MEM_WRITE_ENABLE] >= 0.5).then(|| {
            (
                output[OUT_MEM_WRITE_ADDR].round() as i64,
                output[OUT_MEM_WRITE_VALUE].round() as Word,
            )
        }),
    }
}

fn word_to_scalar(value: Word) -> Scalar {
    value as Scalar
}

fn bool_to_scalar(value: bool) -> Scalar {
    if value {
        1.0
    } else {
        0.0
    }
}

fn blend_bool(value: Scalar) -> bool {
    value >= 0.5
}

fn checked_transition_u8(field: &'static str, value: i64) -> Result<u8> {
    u8::try_from(value).map_err(|_| VmError::InvalidTransitionField { field, value })
}

fn validate_stack_pointer(sp: u8, memory_size: usize) -> Result<()> {
    if usize::from(sp) > memory_size {
        return Err(VmError::InvalidStackPointer {
            sp: usize::from(sp),
            size: memory_size,
        });
    }
    Ok(())
}

fn validate_stack_precondition(instruction: Instruction, sp: u8, memory_size: usize) -> Result<()> {
    match instruction {
        Instruction::Push | Instruction::Call(_) if sp == 0 => Err(VmError::StackOverflow {
            sp: usize::from(sp),
            size: memory_size,
        }),
        Instruction::Pop | Instruction::Ret if usize::from(sp) >= memory_size => {
            Err(VmError::StackUnderflow {
                sp: usize::from(sp),
                size: memory_size,
            })
        }
        _ => Ok(()),
    }
}

fn scalar_to_field(value: Scalar) -> FieldElement {
    field_from_i128(value.round() as i128)
}

fn field_from_i128(value: i128) -> FieldElement {
    if value >= 0 {
        FieldElement::new(value as u128)
    } else {
        -FieldElement::new(value.unsigned_abs())
    }
}

fn inverse_or_zero(value: FieldElement) -> FieldElement {
    if value.is_zero() {
        FieldElement::zero()
    } else {
        value.inverse()
    }
}

fn mp_constant(value: i128) -> MPolynomial {
    MPolynomial::constant(field_from_i128(value))
}

fn mp_constant_from_scalar(value: Scalar) -> MPolynomial {
    MPolynomial::constant(scalar_to_field(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(ff_dim: usize) -> TransformerVmConfig {
        TransformerVmConfig {
            ff_dim,
            ..TransformerVmConfig::default()
        }
    }

    #[test]
    fn test_poet_proof_skeleton_compiles() {
        assert_eq!(PC, 0);
    }

    #[test]
    fn poet_vm_executes_simple_program_locally() {
        let program = Program::new(vec![Instruction::LoadImmediate(-1), Instruction::Halt], 0);
        let vm = PoetVm::new(test_config(4), program.clone()).expect("poet vm");
        let mut memory = AddressedMemory::from_initial(program.initial_memory());
        let state0 = MachineState::new(program.memory_size());
        let state1 = vm.step(&state0, &mut memory, 0).expect("step 1");
        assert_eq!(state1.acc, -1);
        assert_eq!(state1.pc, 1);
        let state2 = vm.step(&state1, &mut memory, 1).expect("step 2");
        assert!(state2.halted);
        assert_eq!(state2.pc, 1);
    }

    #[test]
    fn poet_vm_rejects_unsupported_bitwise_profile() {
        let program = Program::new(vec![Instruction::XorImmediate(1), Instruction::Halt], 0);
        let err = PoetVm::new(test_config(8), program).unwrap_err();
        assert!(err.to_string().contains("reduced profile"));
    }

    #[test]
    fn poet_program_analysis_reports_required_ff_dim() {
        let program = Program::new(
            vec![
                Instruction::JumpIfZero(4),
                Instruction::Store(0),
                Instruction::Halt,
            ],
            1,
        );
        let analysis = analyze_poet_program(&program).expect("analyze");
        assert_eq!(analysis.required_ff_dim, 5);
        assert!(analysis.is_fully_supported());
        assert_eq!(analysis.word_bits, POET_WORD_BITS);
        assert_eq!(analysis.token_len, TOKEN_LEN);
        let drivers = analysis.required_ff_dim_drivers();
        assert_eq!(drivers.len(), 1);
        assert_eq!(drivers[0].pc, 0);
        assert_eq!(drivers[0].instruction, Instruction::JumpIfZero(4));
    }

    #[test]
    fn inspect_poet_program_reports_unsupported_instruction_details() {
        let program = Program::new(
            vec![
                Instruction::LoadImmediate(1),
                Instruction::XorImmediate(1),
                Instruction::Halt,
            ],
            0,
        );
        let analysis = inspect_poet_program(&program);
        assert_eq!(analysis.required_ff_dim, 3);
        assert!(!analysis.is_fully_supported());
        assert_eq!(analysis.instructions.len(), 3);
        assert_eq!(analysis.instructions[1].pc, 1);
        assert_eq!(
            analysis.instructions[1].instruction,
            Instruction::XorImmediate(1)
        );
        assert_eq!(analysis.instructions[1].required_ff_dim, None);
        assert!(analysis.instructions[1]
            .unsupported_reason
            .as_deref()
            .unwrap()
            .contains("bitwise instructions"));
    }

    #[test]
    fn prove_poet_execution_rejects_unsupported_sign_branch_profile() {
        let program = Program::new(vec![Instruction::JumpIfSign(1), Instruction::Halt], 0);
        let err = prove_poet_execution(
            &program,
            &[],
            &test_config(8),
            VanillaStarkProofOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("sign-aware state/AIR"));
    }

    #[test]
    fn prove_poet_execution_rejects_too_small_local_ff_dim() {
        let program = Program::new(vec![Instruction::JumpIfZero(1), Instruction::Halt], 0);
        let err = prove_poet_execution(
            &program,
            &[],
            &test_config(4),
            VanillaStarkProofOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires ff_dim >= 5"));
        assert!(err.to_string().contains("pc 0 `JZ 1`"));
    }

    #[test]
    #[ignore = "Poet forward-pass proofs are still too slow for the default test suite"]
    fn test_poet_proof_simple() {
        let program = Program::new(vec![Instruction::LoadImmediate(-1), Instruction::Halt], 0);
        let config = test_config(4);
        let options = VanillaStarkProofOptions::default();

        let proof = prove_poet_execution(&program, &[], &config, options).expect("prove");
        let result = verify_poet_execution(&proof).expect("verify");
        assert!(result);
        assert_eq!(proof.claim.final_state.acc, -1);
    }
}
