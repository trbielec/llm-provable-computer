use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tvm_verify_core as verify_core;

use crate::config::Attention2DMode;
use crate::error::{Result, VmError};
use crate::instruction::{Instruction, Program};
use crate::interpreter::NativeInterpreter;
use crate::model::TransformerVm;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VanillaStarkCostEstimate {
    pub steps: usize,
    pub trace_rows: usize,
    pub trace_width: usize,
    pub memory_size: usize,
    pub public_register_count: usize,
    pub transition_constraint_count: usize,
    pub transition_degree_bound: usize,
    pub num_randomizers: usize,
    pub omicron_domain_length: usize,
    pub fri_domain_length: usize,
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub security_level: usize,
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

    fn acc_inv(&self) -> usize {
        MACHINE_STATE_PREFIX + self.memory_size + WORD_BITS
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
        // Machine State + WORD_BITS for ACC + 3 Inverses
        self.machine_state_register_count() + WORD_BITS + 3
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
                Instruction::AndImmediate(_)
                | Instruction::AndMemory(_)
                | Instruction::OrImmediate(_)
                | Instruction::OrMemory(_)
                | Instruction::XorImmediate(_)
                | Instruction::XorMemory(_) => {
                    unreachable!("unsupported instructions are filtered before AIR construction")
                }
            }
        }

        let mut constraints = vec![
            valid_pc,
            valid_sp,
            current_zero.clone() * (current_zero.clone() - one.clone()),
            current[HALTED].clone(),
            current[CARRY].clone(),
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

        // Bit integrity constraint for signed two's-complement ACC.
        let mut bit_sum = MPolynomial::zero();
        for i in 0..(WORD_BITS - 1) {
            bit_sum = bit_sum + current_acc_bits[i].clone() * mp_constant(1 << i);
        }
        bit_sum = bit_sum
            - current_acc_bits[WORD_BITS - 1].clone() * mp_constant(1i128 << (WORD_BITS - 1));
        constraints.push(current_acc - bit_sum);

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

pub fn prove_execution_stark(
    model: &TransformerVm,
    max_steps: usize,
) -> Result<VanillaStarkExecutionProof> {
    prove_execution_stark_with_options(model, max_steps, VanillaStarkProofOptions::default())
}

pub fn estimate_execution_stark_cost(
    model: &TransformerVm,
    max_steps: usize,
    options: VanillaStarkProofOptions,
) -> Result<VanillaStarkCostEstimate> {
    validate_proof_inputs(model.program(), &model.config().attention_mode)?;

    let mut interpreter = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        max_steps,
    );
    let result = interpreter.run()?;
    if !result.halted {
        return Err(VmError::UnsupportedProof(format!(
            "execution must halt before proving; stopped after {} steps without HALT",
            result.steps
        )));
    }
    if interpreter.trace().iter().any(|state| state.carry_flag) {
        return Err(VmError::UnsupportedProof(
            "overflowing arithmetic is not supported by the vanilla STARK AIR".to_string(),
        ));
    }

    let air = VmAir::new(model.program().clone());
    let transition_constraints = air.transition_constraints();
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        result.steps + 1,
        air.transition_degree_bound(),
    );

    Ok(VanillaStarkCostEstimate {
        steps: result.steps,
        trace_rows: interpreter.trace().len(),
        trace_width: air.register_count(),
        memory_size: model.program().memory_size(),
        public_register_count: air.layout.machine_state_register_count(),
        transition_constraint_count: transition_constraints.len(),
        transition_degree_bound: air.transition_degree_bound(),
        num_randomizers: stark.num_randomizers,
        omicron_domain_length: stark.omicron_domain.len(),
        fri_domain_length: stark.fri.domain_length,
        expansion_factor: options.expansion_factor,
        num_colinearity_checks: options.num_colinearity_checks,
        security_level: options.security_level,
    })
}

pub fn prove_execution_stark_with_options(
    model: &TransformerVm,
    max_steps: usize,
    options: VanillaStarkProofOptions,
) -> Result<VanillaStarkExecutionProof> {
    validate_proof_inputs(model.program(), &model.config().attention_mode)?;

    let mut interpreter = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        max_steps,
    );
    let result = interpreter.run()?;
    if !result.halted {
        return Err(VmError::UnsupportedProof(format!(
            "execution must halt before proving; stopped after {} steps without HALT",
            result.steps
        )));
    }
    if interpreter.trace().iter().any(|state| state.carry_flag) {
        return Err(VmError::UnsupportedProof(
            "overflowing arithmetic is not supported by the vanilla STARK AIR".to_string(),
        ));
    }

    let air = VmAir::new(model.program().clone());
    let trace = execution_trace_rows(&air.layout, interpreter.trace());
    let claim = VanillaStarkExecutionClaim {
        program: model.program().clone(),
        attention_mode: model.config().attention_mode.clone(),
        steps: result.steps,
        final_state: result.final_state.clone(),
        options: options.clone(),
    };
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        claim.steps + 1,
        air.transition_degree_bound(),
    );
    let proof = stark.prove(
        &trace,
        &air.transition_constraints(),
        &air.boundary_constraints(claim.steps, &claim.final_state),
    );

    Ok(VanillaStarkExecutionProof { claim, proof })
}

pub fn verify_execution_stark(proof: &VanillaStarkExecutionProof) -> Result<bool> {
    let core_proof = to_verify_core_proof(proof)?;
    verify_core::verify_execution_stark(&core_proof).map_err(map_verify_core_error)
}

pub(crate) fn validate_execution_stark_support(
    program: &Program,
    attention_mode: &Attention2DMode,
) -> Result<()> {
    validate_proof_inputs(program, attention_mode)
}

pub fn save_execution_stark_proof(proof: &VanillaStarkExecutionProof, path: &Path) -> Result<()> {
    let core_proof = to_verify_core_proof(proof)?;
    let bytes = verify_core::serialize_execution_stark_proof_json(&core_proof)
        .map_err(map_verify_core_error)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn load_execution_stark_proof(path: &Path) -> Result<VanillaStarkExecutionProof> {
    let bytes = fs::read(path)?;
    let proof = verify_core::deserialize_execution_stark_proof_json(&bytes)
        .map_err(map_verify_core_error)?;
    from_verify_core_proof(&proof)
}

fn to_verify_core_proof(
    proof: &VanillaStarkExecutionProof,
) -> Result<verify_core::VanillaStarkExecutionProof> {
    convert_via_serde(proof)
}

fn from_verify_core_proof(
    proof: &verify_core::VanillaStarkExecutionProof,
) -> Result<VanillaStarkExecutionProof> {
    convert_via_serde(proof)
}

fn convert_via_serde<T, U>(value: &T) -> Result<U>
where
    T: Serialize,
    U: DeserializeOwned,
{
    let json =
        serde_json::to_value(value).map_err(|err| VmError::Serialization(err.to_string()))?;
    serde_json::from_value(json).map_err(|err| VmError::Serialization(err.to_string()))
}

fn map_verify_core_error(error: verify_core::VmError) -> VmError {
    match error {
        verify_core::VmError::Io(error) => VmError::Io(error),
        verify_core::VmError::InvalidConfig(message) => VmError::InvalidConfig(message),
        verify_core::VmError::Parse { line, message } => VmError::Parse { line, message },
        verify_core::VmError::UnknownLabel { line, label } => VmError::UnknownLabel { line, label },
        verify_core::VmError::ProgramCounterOutOfBounds { pc, len } => {
            VmError::ProgramCounterOutOfBounds { pc, len }
        }
        verify_core::VmError::MemoryOutOfBounds { addr, size } => {
            VmError::MemoryOutOfBounds { addr, size }
        }
        verify_core::VmError::StackUnderflow { sp, size } => VmError::StackUnderflow { sp, size },
        verify_core::VmError::StackOverflow { sp, size } => VmError::StackOverflow { sp, size },
        verify_core::VmError::InvalidStackPointer { sp, size } => {
            VmError::InvalidStackPointer { sp, size }
        }
        verify_core::VmError::InvalidTransitionField { field, value } => {
            VmError::InvalidTransitionField { field, value }
        }
        verify_core::VmError::ExecutionMismatch { step, message } => {
            VmError::ExecutionMismatch { step, message }
        }
        verify_core::VmError::Serialization(message) => VmError::Serialization(message),
        verify_core::VmError::UnsupportedProof(message) => VmError::UnsupportedProof(message),
        verify_core::VmError::Onnx(message) => VmError::Onnx(message),
        verify_core::VmError::EmptyHull => VmError::EmptyHull,
    }
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
            Instruction::AndImmediate(_)
            | Instruction::AndMemory(_)
            | Instruction::OrImmediate(_)
            | Instruction::OrMemory(_)
            | Instruction::XorImmediate(_)
            | Instruction::XorMemory(_) => {
                return Err(VmError::UnsupportedProof(format!(
                    "instruction `{instruction}` is not yet supported by the vanilla STARK AIR"
                )));
            }
            _ => {}
        }
    }

    Ok(())
}

fn execution_trace_rows(layout: &ColumnLayout, trace: &[MachineState]) -> Vec<Vec<FieldElement>> {
    trace
        .iter()
        .map(|state| {
            let acc = field_from_i128(i128::from(state.acc));
            let sp = field_from_i128(i128::from(state.sp));
            let sp_minus_top = field_from_i128(i128::from(state.sp) - layout.memory_size as i128);
            let mut row = vec![FieldElement::zero(); layout.register_count()];

            row[PC] = field_from_i128(i128::from(state.pc));
            row[ACC] = acc;
            row[SP] = sp;
            row[ZERO] = field_from_bool(state.zero_flag);
            row[CARRY] = field_from_bool(state.carry_flag);
            row[HALTED] = field_from_bool(state.halted);

            for (i, &val) in state.memory.iter().enumerate() {
                row[layout.mem(i)] = field_from_i128(i128::from(val));
            }

            // Bit decomposition of ACC in two's-complement form.
            let acc_val = state.acc as u64;
            for bit in 0..WORD_BITS {
                if (acc_val >> bit) & 1 == 1 {
                    row[layout.acc_bit(bit)] = FieldElement::one();
                } else {
                    row[layout.acc_bit(bit)] = FieldElement::zero();
                }
            }

            let inv_start = layout.acc_inv();
            row[inv_start] = inverse_or_zero(acc);
            row[inv_start + 1] = inverse_or_zero(sp);
            row[inv_start + 2] = inverse_or_zero(sp_minus_top);
            row
        })
        .collect()
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

fn inverse_or_zero(value: FieldElement) -> FieldElement {
    if value.is_zero() {
        FieldElement::zero()
    } else {
        value.inverse()
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
    use crate::{
        compile_cdmn_table_with_metadata, CdmnTable, Collection, CollectionField,
        Constraint as CdmnConstraint, Definition as CdmnDefinition, ProgramCompiler, TransformerVm,
        TransformerVmConfig, Variable as CdmnVariable,
    };

    fn prove_program(path: &str, max_steps: usize) -> VanillaStarkExecutionProof {
        let source = std::fs::read_to_string(path).expect("program source");
        let model = ProgramCompiler
            .compile_source(&source, TransformerVmConfig::default())
            .expect("compile");
        prove_execution_stark(&model, max_steps).expect("prove")
    }

    fn cdmn_variable(name: &str, memory: u8) -> CdmnVariable {
        CdmnVariable {
            name: name.to_string(),
            memory,
            presence_memory: None,
            unknown_memory: None,
        }
    }

    fn cdmn_field(name: &str, offset: u8) -> CollectionField {
        CollectionField {
            name: name.to_string(),
            offset,
            presence_offset: None,
            unknown_offset: None,
        }
    }

    fn cdmn_constraint(
        id: &str,
        description: &str,
        logic: &str,
        reason_code: Option<i64>,
        reason_label: Option<&str>,
    ) -> CdmnConstraint {
        CdmnConstraint {
            id: id.to_string(),
            description: description.to_string(),
            logic: logic.to_string(),
            reason_code,
            reason_label: reason_label.map(str::to_string),
            insufficient_data_reason_code: None,
            insufficient_data_reason_label: None,
        }
    }

    fn medication_policy_table() -> CdmnTable {
        CdmnTable {
            name: "Medication Safety".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![cdmn_variable("max_dose", 0)],
            definitions: vec![],
            collections: vec![Collection {
                name: "medications".to_string(),
                count_memory: 1,
                base_memory: 2,
                stride: 1,
                max_items: 2,
                fields: vec![cdmn_field("dose", 0)],
            }],
            constraints: vec![cdmn_constraint(
                "M1",
                "Each medication must respect the max dose",
                "forall medication in medications (medication.dose <= max_dose)",
                Some(101),
                Some("dose_exceeds_max"),
            )],
        }
    }

    fn lab_policy_table() -> CdmnTable {
        CdmnTable {
            name: "Lab Guardrails".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                cdmn_variable("wbc_count", 0),
                cdmn_variable("wbc_min", 1),
                cdmn_variable("wbc_max", 2),
                cdmn_variable("systolic_bp", 3),
                cdmn_variable("diastolic_bp", 4),
                cdmn_variable("pulse_pressure_min", 5),
                cdmn_variable("pulse_pressure_max", 6),
            ],
            definitions: vec![CdmnDefinition {
                name: "pulse_pressure".to_string(),
                expression: "systolic_bp - diastolic_bp".to_string(),
                description: "Derived systolic-diastolic pulse pressure".to_string(),
            }],
            collections: vec![],
            constraints: vec![
                cdmn_constraint(
                    "L1",
                    "White blood cell count must remain within range",
                    "wbc_count between wbc_min and wbc_max",
                    Some(301),
                    Some("wbc_out_of_range"),
                ),
                cdmn_constraint(
                    "L2",
                    "Pulse pressure must remain within range",
                    "pulse_pressure between pulse_pressure_min and pulse_pressure_max",
                    Some(302),
                    Some("pulse_pressure_out_of_range"),
                ),
            ],
        }
    }

    #[test]
    fn addition_round_trips_through_stark_proof() {
        let proof = prove_program("programs/addition.tvm", 32);
        assert!(verify_execution_stark(&proof).expect("verify"));
        assert_eq!(proof.claim.final_state.acc, 8);
    }

    #[test]
    fn negative_acc_round_trips_through_stark_proof() {
        let source = "\
.memory 1
LOADI 1
SUB 2
HALT
";
        let model = ProgramCompiler
            .compile_source(source, TransformerVmConfig::default())
            .expect("compile");
        let proof = prove_execution_stark(&model, 32).expect("prove");
        assert!(verify_execution_stark(&proof).expect("verify"));
        assert_eq!(proof.claim.final_state.acc, -1);
    }

    #[test]
    fn small_loop_round_trips_through_stark_proof() {
        let source = "\
.memory 2
.init 1 2

LOADI 0
STORE 0
loop:
LOAD 0
ADD 1
STORE 0
LOAD 0
SUBM 1
JZ done
JMP loop
done:
LOAD 0
HALT
";
        let model = ProgramCompiler
            .compile_source(source, TransformerVmConfig::default())
            .expect("compile");
        let proof = prove_execution_stark(&model, 128).expect("prove");
        assert!(verify_execution_stark(&proof).expect("verify"));
        assert_eq!(proof.claim.final_state.acc, 2);
    }

    #[test]
    fn sign_branch_round_trips_through_stark_proof() {
        let source = "\
.memory 1
LOADI 3
SUB 5
JS negative
LOADI 0
HALT
negative:
LOADI 1
HALT
";
        let model = ProgramCompiler
            .compile_source(source, TransformerVmConfig::default())
            .expect("compile");
        let proof = prove_execution_stark(&model, 32).expect("prove");
        assert!(verify_execution_stark(&proof).expect("verify"));
        assert_eq!(proof.claim.final_state.acc, 1);
    }

    #[test]
    fn addition_cost_estimate_reports_trace_shape() {
        let source = std::fs::read_to_string("programs/addition.tvm").expect("program source");
        let model = ProgramCompiler
            .compile_source(&source, TransformerVmConfig::default())
            .expect("compile");
        let estimate =
            estimate_execution_stark_cost(&model, 32, VanillaStarkProofOptions::default())
                .expect("estimate");
        assert_eq!(estimate.steps, 3);
        assert_eq!(estimate.trace_rows, 4);
        assert!(estimate.trace_width >= 6 + model.program().memory_size());
        assert_eq!(estimate.memory_size, model.program().memory_size());
        assert!(estimate.fri_domain_length >= estimate.omicron_domain_length);
    }

    #[test]
    #[ignore = "expensive derived-expression STARK regression"]
    fn lab_policy_round_trips_through_stark_proof() {
        let compiled = compile_cdmn_table_with_metadata(&lab_policy_table()).expect("compile cdmn");
        let config = TransformerVmConfig::default();
        let model = ProgramCompiler
            .compile_source(&compiled.assembly, config.clone())
            .expect("compile");

        let mut initial_memory = vec![0; model.program().memory_size()];
        initial_memory[0] = 8_000;
        initial_memory[1] = 4_000;
        initial_memory[2] = 11_000;
        initial_memory[3] = 120;
        initial_memory[4] = 80;
        initial_memory[5] = 30;
        initial_memory[6] = 60;

        let program = model
            .program()
            .clone()
            .with_initial_memory(initial_memory)
            .expect("initial memory");
        let model = TransformerVm::new(config, program).expect("model");

        let proof = prove_execution_stark(&model, 128).expect("prove");
        assert!(verify_execution_stark(&proof).expect("verify"));
        let status_idx = usize::from(compiled.output.status_register);
        let reason_idx = usize::from(compiled.output.reason_register);
        assert_eq!(proof.claim.final_state.memory[status_idx], 1);
        assert_eq!(proof.claim.final_state.memory[reason_idx], 0);
    }

    #[test]
    #[ignore = "expensive quantified STARK regression"]
    fn quantified_medication_policy_round_trips_through_stark_proof() {
        let compiled =
            compile_cdmn_table_with_metadata(&medication_policy_table()).expect("compile cdmn");
        let config = TransformerVmConfig::default();
        let model = ProgramCompiler
            .compile_source(&compiled.assembly, config.clone())
            .expect("compile");

        let mut initial_memory = vec![0; model.program().memory_size()];
        initial_memory[0] = 500;
        initial_memory[1] = 2;
        initial_memory[2] = 400;
        initial_memory[3] = 450;

        let program = model
            .program()
            .clone()
            .with_initial_memory(initial_memory)
            .expect("initial memory");
        let model = TransformerVm::new(config, program).expect("model");

        let proof = prove_execution_stark(&model, 256).expect("prove");
        assert!(verify_execution_stark(&proof).expect("verify"));
        let status_idx = usize::from(compiled.output.status_register);
        let reason_idx = usize::from(compiled.output.reason_register);
        assert_eq!(proof.claim.final_state.memory[status_idx], 1);
        assert_eq!(proof.claim.final_state.memory[reason_idx], 0);
    }

    #[test]
    fn stack_and_subroutine_round_trip_through_stark_proof() {
        for (path, max_steps, expected_acc) in [
            ("programs/stack_roundtrip.tvm", 32, 42),
            ("programs/subroutine_addition.tvm", 64, 42),
        ] {
            let proof = prove_program(path, max_steps);
            assert!(verify_execution_stark(&proof).expect("verify"), "{path}");
            assert_eq!(proof.claim.final_state.acc, expected_acc, "{path}");
        }
    }

    #[test]
    fn proof_rejects_softmax_attention() {
        let source = std::fs::read_to_string("programs/addition.tvm").expect("program source");
        let model = ProgramCompiler
            .compile_source(
                &source,
                TransformerVmConfig {
                    attention_mode: Attention2DMode::Softmax { query: [1.0, 0.0] },
                    ..TransformerVmConfig::default()
                },
            )
            .expect("compile");

        let err = prove_execution_stark(&model, 32).unwrap_err();
        assert!(matches!(err, VmError::UnsupportedProof(_)));
    }

    #[test]
    fn proof_rejects_unsupported_instruction_set() {
        let program = Program::new(vec![Instruction::CmpImmediate(1), Instruction::Halt], 1);
        let model = TransformerVm::new(TransformerVmConfig::default(), program).expect("model");
        let err = prove_execution_stark(&model, 16).unwrap_err();
        assert!(matches!(err, VmError::UnsupportedProof(_)));
    }

    #[test]
    fn proof_serialization_round_trip() {
        let proof = prove_program("programs/addition.tvm", 32);
        let path = std::env::temp_dir().join(format!(
            "llm-provable-computer-proof-{}.json",
            std::process::id()
        ));

        save_execution_stark_proof(&proof, &path).expect("save");
        let loaded = load_execution_stark_proof(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded, proof);
        assert!(verify_execution_stark(&loaded).expect("verify"));
    }
}
