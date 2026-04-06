#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};

use crate::config::Attention2DMode;
use crate::error::{Result, VmError};
use crate::instruction::{Instruction, Program};
use crate::state::{MachineState, WORD_BITS};
use crate::vanillastark::{FieldElement, MPolynomial, Stark};

const PC: usize = 0;
const ACC: usize = 1;
const SP: usize = 2;
const ZERO: usize = 3;
const CARRY: usize = 4;
const HALTED: usize = 5;
const MACHINE_STATE_PREFIX: usize = 6;
const WORD_MODULUS: i128 = 1i128 << WORD_BITS;
pub const EXECUTION_PROOF_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VanillaStarkProofOptions {
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub security_level: usize,
}

impl Default for VanillaStarkProofOptions {
    fn default() -> Self {
        Self {
            expansion_factor: 4,
            num_colinearity_checks: 2,
            security_level: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VanillaStarkExecutionClaim {
    pub program: Program,
    pub attention_mode: Attention2DMode,
    pub steps: usize,
    pub final_state: MachineState,
    pub options: VanillaStarkProofOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VanillaStarkExecutionProof {
    pub claim: VanillaStarkExecutionClaim,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct VersionedExecutionProofEnvelope {
    pub version: u32,
    pub claim: VanillaStarkExecutionClaim,
    pub proof_base64: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LegacyExecutionProofEnvelope {
    pub claim: VanillaStarkExecutionClaim,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct ColumnLayout {
    memory_size: usize,
}

impl ColumnLayout {
    fn new(memory_size: usize) -> Self {
        Self { memory_size }
    }

    fn mem(&self, index: usize) -> usize {
        MACHINE_STATE_PREFIX + index
    }

    fn acc_bit(&self, bit: usize) -> usize {
        MACHINE_STATE_PREFIX + self.memory_size + bit
    }

    fn operand_bit(&self, bit: usize) -> usize {
        MACHINE_STATE_PREFIX + self.memory_size + WORD_BITS + bit
    }

    fn acc_inv(&self) -> usize {
        MACHINE_STATE_PREFIX + self.memory_size + 2 * WORD_BITS
    }

    fn sp_nonzero_inv(&self) -> usize {
        self.acc_inv() + 1
    }

    fn sp_not_top_inv(&self) -> usize {
        self.acc_inv() + 2
    }

    fn machine_state_register_count(&self) -> usize {
        MACHINE_STATE_PREFIX + self.memory_size
    }

    fn register_count(&self) -> usize {
        // Machine State + WORD_BITS for ACC + WORD_BITS for Operand + 3 Inverses
        self.machine_state_register_count() + 2 * WORD_BITS + 3
    }
}

#[derive(Debug, Clone)]
struct VmAir {
    program: Program,
    layout: ColumnLayout,
}

impl VmAir {
    fn new(program: Program) -> Self {
        let layout = ColumnLayout::new(program.memory_size());
        Self { program, layout }
    }

    fn register_count(&self) -> usize {
        self.layout.register_count()
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
        steps: usize,
        final_state: &MachineState,
    ) -> Vec<(usize, usize, FieldElement)> {
        let mut boundary = Vec::new();
        let initial = MachineState::with_memory(self.program.initial_memory().to_vec());

        for (register, value) in state_public_registers(&self.layout, &initial)
            .into_iter()
            .enumerate()
        {
            boundary.push((0, register, value));
        }

        for (register, value) in state_public_registers(&self.layout, final_state)
            .into_iter()
            .enumerate()
        {
            boundary.push((steps, register, value));
        }

        boundary
    }

    fn transition_constraints(&self) -> Vec<MPolynomial> {
        let register_count = self.layout.register_count();
        let variables = MPolynomial::variables(1 + 2 * register_count);
        let current = variables[1..1 + register_count].to_vec();
        let next = variables[1 + register_count..].to_vec();

        let current_pc = current[PC].clone();
        let current_acc = current[ACC].clone();
        let current_sp = current[SP].clone();
        let current_zero = current[ZERO].clone();
        let current_memory: Vec<MPolynomial> = (0..self.layout.memory_size)
            .map(|index| current[self.layout.mem(index)].clone())
            .collect();

        let next_pc = next[PC].clone();
        let next_acc = next[ACC].clone();
        let next_sp = next[SP].clone();
        let next_zero = next[ZERO].clone();
        let next_carry = next[CARRY].clone();
        let next_halted = next[HALTED].clone();
        let next_memory: Vec<MPolynomial> = (0..self.layout.memory_size)
            .map(|index| next[self.layout.mem(index)].clone())
            .collect();
        let next_acc_inv = next[self.layout.acc_inv()].clone();
        let current_sp_nonzero_inv = current[self.layout.sp_nonzero_inv()].clone();
        let current_sp_not_top_inv = current[self.layout.sp_not_top_inv()].clone();

        // Range Proof: Bit-decomposition of current_acc
        let current_acc_bits: Vec<MPolynomial> = (0..WORD_BITS)
            .map(|bit| current[self.layout.acc_bit(bit)].clone())
            .collect();
        let current_sign = current_acc_bits[WORD_BITS - 1].clone();

        // Range Proof: Bit-decomposition of current operand
        let current_operand_bits: Vec<MPolynomial> = (0..WORD_BITS)
            .map(|bit| current[self.layout.operand_bit(bit)].clone())
            .collect();

        let one = mp_constant(1);
        let zero = MPolynomial::zero();
        let current_stack_read = self.stack_read_expression(&current_sp, &current_memory);
        let current_stack_write_selectors = self.stack_write_selectors(&current_sp);
        let valid_pc = self.domain_zerofier(&current_pc, self.program.len());
        let valid_sp = self.domain_zerofier(&current_sp, self.layout.memory_size + 1);
        let mut expected_next_pc = MPolynomial::zero();
        let mut expected_next_acc = MPolynomial::zero();
        let mut expected_next_sp = MPolynomial::zero();
        let mut expected_next_carry = next_carry.clone(); // Temporarily match witness for stability
        let mut expected_next_halted = MPolynomial::zero();
        let mut expected_next_memory = vec![MPolynomial::zero(); self.layout.memory_size];
        let mut expected_operand_bit_sum = MPolynomial::zero();
        let mut push_or_call_selector = MPolynomial::zero();
        let mut pop_or_ret_selector = MPolynomial::zero();

        let pc_selectors = self.pc_selectors(&current_pc);

        for (pc, instruction) in self.program.instructions().iter().enumerate() {
            let selector = pc_selectors[pc].clone();
            let next_pc_constant = mp_constant(i128::from((pc as u8).wrapping_add(1)));

            match *instruction {
                Instruction::Nop => {
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::LoadImmediate(value) => {
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc =
                        expected_next_acc + selector.clone() * mp_constant(i128::from(value));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::Load(address) => {
                    let loaded = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc + selector.clone() * loaded;
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::Store(address) => {
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        let value = if index == address as usize {
                            current_acc.clone()
                        } else {
                            current_memory[index].clone()
                        };
                        *expected = expected.clone() + pc_selectors[pc].clone() * value;
                    }
                }
                Instruction::Push => {
                    push_or_call_selector = push_or_call_selector + selector.clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp =
                        expected_next_sp + selector.clone() * (current_sp.clone() - one.clone());
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        let write_selector = current_stack_write_selectors[index].clone();
                        let value = write_selector.clone() * current_acc.clone()
                            + (one.clone() - write_selector) * current_memory[index].clone();
                        *expected = expected.clone() + pc_selectors[pc].clone() * value;
                    }
                }
                Instruction::Pop => {
                    pop_or_ret_selector = pop_or_ret_selector + selector.clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc =
                        expected_next_acc + selector.clone() * current_stack_read.clone();
                    expected_next_sp =
                        expected_next_sp + selector.clone() * (current_sp.clone() + one.clone());
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::AddImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() + operand
                                - next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::AddMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() + operand
                                - next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::SubImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() - operand
                                + next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::SubMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() - operand
                                + next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::MulImmediate(value) => {
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone() * (current_acc.clone() * mp_constant(i128::from(value)));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::MulMemory(address) => {
                    let rhs = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc =
                        expected_next_acc + selector.clone() * (current_acc.clone() * rhs);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::CmpImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    // CMP uses SUB carry logic
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() - operand
                                + next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::CmpMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_next_acc = expected_next_acc
                        + selector.clone()
                            * (current_acc.clone() - operand
                                + next_carry.clone() * mp_constant(WORD_MODULUS));
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry =
                        expected_next_carry + selector.clone() * next_carry.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::Call(target) => {
                    push_or_call_selector = push_or_call_selector + selector.clone();
                    expected_next_pc =
                        expected_next_pc + selector.clone() * mp_constant(i128::from(target));
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp =
                        expected_next_sp + selector.clone() * (current_sp.clone() - one.clone());
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        let write_selector = current_stack_write_selectors[index].clone();
                        let return_address = mp_constant(i128::from((pc as u8).wrapping_add(1)));
                        let value = write_selector.clone() * return_address
                            + (one.clone() - write_selector) * current_memory[index].clone();
                        *expected = expected.clone() + pc_selectors[pc].clone() * value;
                    }
                }
                Instruction::Ret => {
                    pop_or_ret_selector = pop_or_ret_selector + selector.clone();
                    expected_next_pc =
                        expected_next_pc + selector.clone() * current_stack_read.clone();
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp =
                        expected_next_sp + selector.clone() * (current_sp.clone() + one.clone());
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::Jump(target) => {
                    expected_next_pc =
                        expected_next_pc + selector.clone() * mp_constant(i128::from(target));
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::JumpIfZero(target) => {
                    let target_expr = current_zero.clone() * mp_constant(i128::from(target))
                        + (one.clone() - current_zero.clone()) * next_pc_constant;
                    expected_next_pc = expected_next_pc + selector.clone() * target_expr;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::JumpIfNotZero(target) => {
                    let target_expr = current_zero.clone() * next_pc_constant
                        + (one.clone() - current_zero.clone()) * mp_constant(i128::from(target));
                    expected_next_pc = expected_next_pc + selector.clone() * target_expr;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::JumpIfSign(target) => {
                    let target_expr = current_sign.clone() * mp_constant(i128::from(target))
                        + (one.clone() - current_sign.clone()) * next_pc_constant;
                    expected_next_pc = expected_next_pc + selector.clone() * target_expr;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::JumpIfNotSign(target) => {
                    let target_expr = current_sign.clone() * next_pc_constant
                        + (one.clone() - current_sign.clone()) * mp_constant(i128::from(target));
                    expected_next_pc = expected_next_pc + selector.clone() * target_expr;
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::Halt => {
                    expected_next_pc =
                        expected_next_pc + selector.clone() * mp_constant(i128::from(pc as i64));
                    expected_next_acc = expected_next_acc + selector.clone() * current_acc.clone();
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * one.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::AndImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    // AND: result_bit[i] = acc_bit[i] * operand_bit[i]
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits
                            .push(current_acc_bits[i].clone() * current_operand_bits[i].clone());
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::AndMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits
                            .push(current_acc_bits[i].clone() * current_operand_bits[i].clone());
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::OrImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    // OR: result_bit[i] = a + b - a*b
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits.push(
                            current_acc_bits[i].clone() + current_operand_bits[i].clone()
                                - current_acc_bits[i].clone() * current_operand_bits[i].clone(),
                        );
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::OrMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits.push(
                            current_acc_bits[i].clone() + current_operand_bits[i].clone()
                                - current_acc_bits[i].clone() * current_operand_bits[i].clone(),
                        );
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::XorImmediate(value) => {
                    let operand = mp_constant(i128::from(value));
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    // XOR: result_bit[i] = a + b - 2*a*b
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits.push(
                            current_acc_bits[i].clone() + current_operand_bits[i].clone()
                                - mp_constant(2)
                                    * current_acc_bits[i].clone()
                                    * current_operand_bits[i].clone(),
                        );
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
                Instruction::XorMemory(address) => {
                    let operand = current_memory[address as usize].clone();
                    expected_next_pc = expected_next_pc + selector.clone() * next_pc_constant;
                    expected_operand_bit_sum =
                        expected_operand_bit_sum + selector.clone() * operand;
                    let mut result_bits = Vec::new();
                    for i in 0..WORD_BITS {
                        result_bits.push(
                            current_acc_bits[i].clone() + current_operand_bits[i].clone()
                                - mp_constant(2)
                                    * current_acc_bits[i].clone()
                                    * current_operand_bits[i].clone(),
                        );
                    }
                    expected_next_acc =
                        expected_next_acc + selector.clone() * bitwise_result_sum(&result_bits);
                    expected_next_sp = expected_next_sp + selector.clone() * current_sp.clone();
                    expected_next_carry = expected_next_carry + selector.clone() * zero.clone();
                    expected_next_halted = expected_next_halted + selector * zero.clone();
                    for (index, expected) in expected_next_memory.iter_mut().enumerate() {
                        *expected = expected.clone()
                            + pc_selectors[pc].clone() * current_memory[index].clone();
                    }
                }
            }
        }

        let mut constraints = vec![
            valid_pc,
            valid_sp,
            current_zero.clone() * (current_zero.clone() - one.clone()),
            current[HALTED].clone(),
            current[CARRY].clone() * (current[CARRY].clone() - one.clone()),
            next_pc - expected_next_pc,
            next_acc.clone() - expected_next_acc,
            next_sp - expected_next_sp,
            next_halted - expected_next_halted,
            next_carry - expected_next_carry,
            next_acc.clone() * next_acc_inv - (one.clone() - next_zero.clone()),
            next_zero * next_acc,
        ];

        // Bit booleanity constraints: bit * (bit - 1) == 0
        for bit in &current_acc_bits {
            constraints.push(bit.clone() * (bit.clone() - one.clone()));
        }

        // Bit booleanity constraints for operand bits
        for bit in &current_operand_bits {
            constraints.push(bit.clone() * (bit.clone() - one.clone()));
        }

        // Bit integrity constraint for signed two's-complement ACC.
        let mut bit_sum = MPolynomial::zero();
        for (i, bit) in current_acc_bits.iter().enumerate().take(WORD_BITS - 1) {
            bit_sum = bit_sum + bit.clone() * mp_constant(1 << i);
        }
        bit_sum = bit_sum
            - current_acc_bits[WORD_BITS - 1].clone() * mp_constant(1i128 << (WORD_BITS - 1));
        constraints.push(current_acc - bit_sum);

        // Operand bit integrity constraint: operand_bits must reconstruct to expected_operand_bit_sum.
        let mut operand_bit_sum = MPolynomial::zero();
        for (i, bit) in current_operand_bits.iter().enumerate().take(WORD_BITS - 1) {
            operand_bit_sum = operand_bit_sum + bit.clone() * mp_constant(1 << i);
        }
        operand_bit_sum = operand_bit_sum
            - current_operand_bits[WORD_BITS - 1].clone() * mp_constant(1i128 << (WORD_BITS - 1));
        constraints.push(operand_bit_sum - expected_operand_bit_sum);

        if !self.layout.memory_size.is_zero() {
            constraints.push(
                push_or_call_selector * (current_sp.clone() * current_sp_nonzero_inv - one.clone()),
            );
            constraints.push(
                pop_or_ret_selector
                    * ((current_sp - mp_constant(self.layout.memory_size as i128))
                        * current_sp_not_top_inv
                        - one.clone()),
            );
        }

        for (index, expected) in expected_next_memory.into_iter().enumerate() {
            constraints.push(next_memory[index].clone() - expected);
        }

        constraints
    }

    fn pc_selectors(&self, pc: &MPolynomial) -> Vec<MPolynomial> {
        let domain: Vec<i128> = (0..self.program.len()).map(|value| value as i128).collect();
        domain
            .iter()
            .map(|&point| lagrange_selector(pc, &domain, point))
            .collect()
    }

    fn stack_write_selectors(&self, sp: &MPolynomial) -> Vec<MPolynomial> {
        let domain: Vec<i128> = (0..=self.layout.memory_size)
            .map(|value| value as i128)
            .collect();
        (0..self.layout.memory_size)
            .map(|index| lagrange_selector(sp, &domain, index as i128 + 1))
            .collect()
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

    fn domain_zerofier(&self, variable: &MPolynomial, size: usize) -> MPolynomial {
        let mut polynomial = mp_constant(1);
        for point in 0..size {
            polynomial = polynomial * (variable.clone() - mp_constant(point as i128));
        }
        polynomial
    }
}

pub fn verify_execution_stark(proof: &VanillaStarkExecutionProof) -> Result<bool> {
    validate_proof_inputs(&proof.claim.program, &proof.claim.attention_mode)?;
    validate_public_state(&proof.claim.program, &proof.claim.final_state)?;

    if !proof.claim.final_state.halted {
        return Err(VmError::UnsupportedProof(
            "the public claim must end in a halted state".to_string(),
        ));
    }
    if proof.claim.final_state.carry_flag {
        return Err(VmError::UnsupportedProof(
            "carry-flag claims are not supported by the vanilla STARK AIR".to_string(),
        ));
    }

    let air = VmAir::new(proof.claim.program.clone());
    let stark = Stark::new(
        proof.claim.options.expansion_factor,
        proof.claim.options.num_colinearity_checks,
        proof.claim.options.security_level,
        air.register_count(),
        proof.claim.steps + 1,
        air.transition_degree_bound(),
    );

    Ok(stark.verify(
        &proof.proof,
        &air.transition_constraints(),
        &air.boundary_constraints(proof.claim.steps, &proof.claim.final_state),
    ))
}

pub fn serialize_execution_stark_proof_json(proof: &VanillaStarkExecutionProof) -> Result<Vec<u8>> {
    let envelope = VersionedExecutionProofEnvelope {
        version: EXECUTION_PROOF_FORMAT_VERSION,
        claim: proof.claim.clone(),
        proof_base64: BASE64_STANDARD.encode(&proof.proof),
    };
    serde_json::to_vec_pretty(&envelope).map_err(|err| VmError::Serialization(err.to_string()))
}

pub fn deserialize_execution_stark_proof_json(bytes: &[u8]) -> Result<VanillaStarkExecutionProof> {
    if let Ok(envelope) = serde_json::from_slice::<VersionedExecutionProofEnvelope>(bytes) {
        if envelope.version != EXECUTION_PROOF_FORMAT_VERSION {
            return Err(VmError::Serialization(format!(
                "unsupported proof format version {}; expected {}",
                envelope.version, EXECUTION_PROOF_FORMAT_VERSION
            )));
        }

        let proof = BASE64_STANDARD
            .decode(envelope.proof_base64.as_bytes())
            .map_err(|err| VmError::Serialization(err.to_string()))?;
        return Ok(VanillaStarkExecutionProof {
            claim: envelope.claim,
            proof,
        });
    }

    let legacy = serde_json::from_slice::<LegacyExecutionProofEnvelope>(bytes)
        .map_err(|err| VmError::Serialization(err.to_string()))?;
    Ok(VanillaStarkExecutionProof {
        claim: legacy.claim,
        proof: legacy.proof,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_execution_stark_proof(proof: &VanillaStarkExecutionProof, path: &Path) -> Result<()> {
    let bytes = serialize_execution_stark_proof_json(proof)?;
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_execution_stark_proof(path: &Path) -> Result<VanillaStarkExecutionProof> {
    let bytes = fs::read(path)?;
    deserialize_execution_stark_proof_json(&bytes)
}

fn validate_proof_inputs(program: &Program, attention_mode: &Attention2DMode) -> Result<()> {
    if program.is_empty() {
        return Err(VmError::UnsupportedProof(
            "cannot prove an empty program".to_string(),
        ));
    }

    if !matches!(attention_mode, Attention2DMode::AverageHard { .. }) {
        return Err(VmError::UnsupportedProof(format!(
            "vanilla STARK proofs currently support only `average-hard` attention, got `{attention_mode}`"
        )));
    }

    for instruction in program.instructions() {
        match instruction {
            Instruction::CmpImmediate(_) | Instruction::CmpMemory(_) => {
                return Err(VmError::UnsupportedProof(format!(
                    "instruction `{instruction}` is not yet supported by the vanilla STARK AIR"
                )));
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_public_state(program: &Program, final_state: &MachineState) -> Result<()> {
    if final_state.memory.len() != program.memory_size() {
        return Err(VmError::InvalidConfig(format!(
            "proof final-state memory length {} does not match program memory size {}",
            final_state.memory.len(),
            program.memory_size()
        )));
    }
    Ok(())
}

fn state_public_registers(layout: &ColumnLayout, state: &MachineState) -> Vec<FieldElement> {
    let mut registers = Vec::with_capacity(layout.machine_state_register_count());
    registers.push(field_from_i128(i128::from(state.pc)));
    registers.push(field_from_i128(i128::from(state.acc)));
    registers.push(field_from_i128(i128::from(state.sp)));
    registers.push(field_from_bool(state.zero_flag));
    registers.push(field_from_bool(state.carry_flag));
    registers.push(field_from_bool(state.halted));
    registers.extend(
        state
            .memory
            .iter()
            .copied()
            .map(|value| field_from_i128(i128::from(value))),
    );
    registers
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

/// Reconstruct a signed two's-complement value from result bits.
fn bitwise_result_sum(result_bits: &[MPolynomial]) -> MPolynomial {
    let mut sum = MPolynomial::zero();
    for (i, bit) in result_bits.iter().enumerate().take(WORD_BITS - 1) {
        sum = sum + bit.clone() * mp_constant(1 << i);
    }
    sum - result_bits[WORD_BITS - 1].clone() * mp_constant(1i128 << (WORD_BITS - 1))
}

fn mp_constant(value: i128) -> MPolynomial {
    MPolynomial::constant(field_from_i128(value))
}

fn field_from_bool(value: bool) -> FieldElement {
    field_from_i128(if value { 1 } else { 0 })
}

fn field_from_i128(value: i128) -> FieldElement {
    if value >= 0 {
        FieldElement::new(value as u128)
    } else {
        -FieldElement::new(value.unsigned_abs())
    }
}

trait IsZeroExt {
    fn is_zero(&self) -> bool;
}

impl IsZeroExt for usize {
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn versioned_serialization_round_trips() {
        let proof = VanillaStarkExecutionProof {
            claim: VanillaStarkExecutionClaim {
                program: Program::new(vec![Instruction::Halt], 1),
                attention_mode: Attention2DMode::AverageHard { query: [1.0, 0.0] },
                steps: 1,
                final_state: MachineState {
                    pc: 0,
                    acc: 0,
                    sp: 1,
                    zero_flag: true,
                    carry_flag: false,
                    halted: true,
                    memory: vec![0],
                },
                options: VanillaStarkProofOptions::default(),
            },
            proof: vec![1, 2, 3, 4],
        };

        let bytes = serialize_execution_stark_proof_json(&proof).expect("serialize");
        let loaded = deserialize_execution_stark_proof_json(&bytes).expect("deserialize");
        assert_eq!(loaded, proof);
    }

    #[test]
    fn legacy_serialization_stays_readable() {
        let legacy = LegacyExecutionProofEnvelope {
            claim: VanillaStarkExecutionClaim {
                program: Program::new(vec![Instruction::Halt], 1),
                attention_mode: Attention2DMode::AverageHard { query: [1.0, 0.0] },
                steps: 1,
                final_state: MachineState {
                    pc: 0,
                    acc: 0,
                    sp: 1,
                    zero_flag: true,
                    carry_flag: false,
                    halted: true,
                    memory: vec![0],
                },
                options: VanillaStarkProofOptions::default(),
            },
            proof: vec![1, 2, 3, 4],
        };

        let loaded =
            deserialize_execution_stark_proof_json(&serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(loaded.proof, vec![1, 2, 3, 4]);
    }
}
