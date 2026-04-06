use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VmError};
use crate::state::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Instruction {
    Nop,
    LoadImmediate(Word),
    Load(u8),
    Store(u8),
    Push,
    Pop,
    AddImmediate(Word),
    AddMemory(u8),
    SubImmediate(Word),
    SubMemory(u8),
    MulImmediate(Word),
    MulMemory(u8),
    AndImmediate(Word),
    AndMemory(u8),
    OrImmediate(Word),
    OrMemory(u8),
    XorImmediate(Word),
    XorMemory(u8),
    CmpImmediate(Word),
    CmpMemory(u8),
    Call(u8),
    Ret,
    Jump(u8),
    JumpIfZero(u8),
    JumpIfNotZero(u8),
    JumpIfSign(u8),
    JumpIfNotSign(u8),
    Halt,
}

impl Instruction {
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Instruction::Nop => "NOP",
            Instruction::LoadImmediate(_) => "LOADI",
            Instruction::Load(_) => "LOAD",
            Instruction::Store(_) => "STORE",
            Instruction::Push => "PUSH",
            Instruction::Pop => "POP",
            Instruction::AddImmediate(_) => "ADD",
            Instruction::AddMemory(_) => "ADDM",
            Instruction::SubImmediate(_) => "SUB",
            Instruction::SubMemory(_) => "SUBM",
            Instruction::MulImmediate(_) => "MUL",
            Instruction::MulMemory(_) => "MULM",
            Instruction::AndImmediate(_) => "AND",
            Instruction::AndMemory(_) => "ANDM",
            Instruction::OrImmediate(_) => "OR",
            Instruction::OrMemory(_) => "ORM",
            Instruction::XorImmediate(_) => "XOR",
            Instruction::XorMemory(_) => "XORM",
            Instruction::CmpImmediate(_) => "CMP",
            Instruction::CmpMemory(_) => "CMPM",
            Instruction::Call(_) => "CALL",
            Instruction::Ret => "RET",
            Instruction::Jump(_) => "JMP",
            Instruction::JumpIfZero(_) => "JZ",
            Instruction::JumpIfNotZero(_) => "JNZ",
            Instruction::JumpIfSign(_) => "JS",
            Instruction::JumpIfNotSign(_) => "JNS",
            Instruction::Halt => "HALT",
        }
    }

    pub fn with_offset(&self, offset: u8) -> Result<Self> {
        let add_offset = |target: u8| {
            target.checked_add(offset).ok_or_else(|| VmError::InvalidConfig(format!(
                "jump target {} with offset {} exceeds u8 limit",
                target, offset
            )))
        };

        match *self {
            Instruction::Call(target) => Ok(Instruction::Call(add_offset(target)?)),
            Instruction::Jump(target) => Ok(Instruction::Jump(add_offset(target)?)),
            Instruction::JumpIfZero(target) => Ok(Instruction::JumpIfZero(add_offset(target)?)),
            Instruction::JumpIfNotZero(target) => Ok(Instruction::JumpIfNotZero(add_offset(target)?)),
            Instruction::JumpIfSign(target) => Ok(Instruction::JumpIfSign(add_offset(target)?)),
            Instruction::JumpIfNotSign(target) => Ok(Instruction::JumpIfNotSign(add_offset(target)?)),
            other => Ok(other),
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Nop
            | Instruction::Push
            | Instruction::Pop
            | Instruction::Ret
            | Instruction::Halt => f.write_str(self.mnemonic()),
            Instruction::LoadImmediate(value)
            | Instruction::AddImmediate(value)
            | Instruction::SubImmediate(value)
            | Instruction::MulImmediate(value)
            | Instruction::AndImmediate(value)
            | Instruction::OrImmediate(value)
            | Instruction::XorImmediate(value)
            | Instruction::CmpImmediate(value) => write!(f, "{} {}", self.mnemonic(), value),
            Instruction::Load(address)
            | Instruction::Store(address)
            | Instruction::AddMemory(address)
            | Instruction::SubMemory(address)
            | Instruction::MulMemory(address)
            | Instruction::AndMemory(address)
            | Instruction::OrMemory(address)
            | Instruction::XorMemory(address)
            | Instruction::CmpMemory(address)
            | Instruction::Call(address)
            | Instruction::Jump(address)
            | Instruction::JumpIfZero(address)
            | Instruction::JumpIfNotZero(address)
            | Instruction::JumpIfSign(address)
            | Instruction::JumpIfNotSign(address) => {
                write!(f, "{} {}", self.mnemonic(), address)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    instructions: Vec<Instruction>,
    initial_memory: Vec<Word>,
}

impl Program {
    pub fn new(instructions: Vec<Instruction>, memory_size: usize) -> Self {
        Self {
            instructions,
            initial_memory: vec![0; memory_size],
        }
    }

    pub fn with_initial_memory(mut self, initial_memory: Vec<Word>) -> Result<Self> {
        if initial_memory.len() > usize::from(u8::MAX) {
            return Err(VmError::InvalidConfig(format!(
                "memory size {} exceeds the encoded stack/address limit of {} cells",
                initial_memory.len(),
                u8::MAX
            )));
        }
        if initial_memory.len() != self.initial_memory.len() {
            return Err(VmError::InvalidConfig(format!(
                "initial memory length {} does not match configured memory size {}",
                initial_memory.len(),
                self.initial_memory.len()
            )));
        }
        self.initial_memory = initial_memory;
        Ok(self)
    }

    /// Merges multiple programs into a single shared instruction space.
    /// Returns the combined program and a list of entry points (PCs) for each
    /// original program.
    ///
    /// # Safety and Memory
    /// All programs share the same addressable memory space. Callers must ensure
    /// that merged programs use non-conflicting memory regions or intentionally
    /// share state.
    pub fn merge(programs: &[Self]) -> Result<(Self, Vec<u8>)> {
        if programs.is_empty() {
            return Err(VmError::InvalidConfig(
                "cannot merge zero programs".to_string(),
            ));
        }

        let mut merged_instructions = Vec::new();
        let mut entry_points = Vec::new();
        let mut max_memory_size = 0;

        for p in programs {
            let offset = merged_instructions.len();
            if offset > usize::from(u8::MAX) {
                return Err(VmError::InvalidConfig(format!(
                    "merged program offset {} exceeds u8 limit",
                    offset
                )));
            }
            entry_points.push(offset as u8);

            for instr in p.instructions() {
                merged_instructions.push(instr.with_offset(offset as u8)?);
            }
            max_memory_size = max_memory_size.max(p.memory_size());
        }

        if merged_instructions.len() > usize::from(u8::MAX) + 1 {
            return Err(VmError::InvalidConfig(format!(
                "merged program has {} instructions, exceeding u8 limit",
                merged_instructions.len()
            )));
        }

        // Overlay initial memory: later programs in the list override earlier ones.
        // We only override with non-zero values to prevent default zero-initialized
        // memory from wiping out meaningful values from earlier programs.
        let mut merged_initial_memory = vec![0; max_memory_size];
        for p in programs {
            for (i, &val) in p.initial_memory().iter().enumerate() {
                if val != 0 {
                    merged_initial_memory[i] = val;
                }
            }
        }

        let merged_program = Program::new(merged_instructions, max_memory_size)
            .with_initial_memory(merged_initial_memory)?;

        Ok((merged_program, entry_points))
    }

    pub fn instruction_at(&self, pc: u8) -> Result<Instruction> {
        self.instructions
            .get(pc as usize)
            .copied()
            .ok_or(VmError::ProgramCounterOutOfBounds {
                pc: pc as usize,
                len: self.instructions.len(),
            })
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    pub fn memory_size(&self) -> usize {
        self.initial_memory.len()
    }

    pub fn initial_memory(&self) -> &[Word] {
        &self.initial_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_new_stores_instructions_and_memory() {
        let program = Program::new(vec![Instruction::Nop, Instruction::Halt], 4);
        assert_eq!(program.len(), 2);
        assert!(!program.is_empty());
        assert_eq!(program.memory_size(), 4);
        assert_eq!(program.initial_memory(), &[0, 0, 0, 0]);
    }

    #[test]
    fn program_empty() {
        let program = Program::new(vec![], 2);
        assert!(program.is_empty());
        assert_eq!(program.len(), 0);
    }

    #[test]
    fn instruction_at_returns_correct_instruction() {
        let program = Program::new(
            vec![
                Instruction::Nop,
                Instruction::AddImmediate(5),
                Instruction::Halt,
            ],
            4,
        );
        assert_eq!(program.instruction_at(0).unwrap(), Instruction::Nop);
        assert_eq!(
            program.instruction_at(1).unwrap(),
            Instruction::AddImmediate(5)
        );
        assert_eq!(program.instruction_at(2).unwrap(), Instruction::Halt);
    }

    #[test]
    fn instruction_at_out_of_bounds_returns_error() {
        let program = Program::new(vec![Instruction::Halt], 2);
        let err = program.instruction_at(1).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn with_initial_memory_sets_values() {
        let program = Program::new(vec![Instruction::Halt], 3)
            .with_initial_memory(vec![10, 20, 30])
            .unwrap();
        assert_eq!(program.initial_memory(), &[10, 20, 30]);
    }

    #[test]
    fn with_initial_memory_rejects_size_mismatch() {
        let err = Program::new(vec![Instruction::Halt], 3)
            .with_initial_memory(vec![1, 2])
            .unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn with_initial_memory_rejects_exceeding_u8_max() {
        let err = Program::new(vec![Instruction::Halt], 256)
            .with_initial_memory(vec![0; 256])
            .unwrap_err();
        assert!(err.to_string().contains("encoded stack/address limit"));
    }

    #[test]
    fn mnemonic_returns_correct_string_for_each_variant() {
        let cases: Vec<(Instruction, &str)> = vec![
            (Instruction::Nop, "NOP"),
            (Instruction::LoadImmediate(0), "LOADI"),
            (Instruction::Load(0), "LOAD"),
            (Instruction::Store(0), "STORE"),
            (Instruction::Push, "PUSH"),
            (Instruction::Pop, "POP"),
            (Instruction::AddImmediate(0), "ADD"),
            (Instruction::AddMemory(0), "ADDM"),
            (Instruction::SubImmediate(0), "SUB"),
            (Instruction::SubMemory(0), "SUBM"),
            (Instruction::MulImmediate(0), "MUL"),
            (Instruction::MulMemory(0), "MULM"),
            (Instruction::AndImmediate(0), "AND"),
            (Instruction::AndMemory(0), "ANDM"),
            (Instruction::OrImmediate(0), "OR"),
            (Instruction::OrMemory(0), "ORM"),
            (Instruction::XorImmediate(0), "XOR"),
            (Instruction::XorMemory(0), "XORM"),
            (Instruction::CmpImmediate(0), "CMP"),
            (Instruction::CmpMemory(0), "CMPM"),
            (Instruction::Call(0), "CALL"),
            (Instruction::Ret, "RET"),
            (Instruction::Jump(0), "JMP"),
            (Instruction::JumpIfZero(0), "JZ"),
            (Instruction::JumpIfNotZero(0), "JNZ"),
            (Instruction::JumpIfSign(0), "JS"),
            (Instruction::JumpIfNotSign(0), "JNS"),
            (Instruction::Halt, "HALT"),
        ];
        for (instr, expected) in cases {
            assert_eq!(instr.mnemonic(), expected, "mnemonic wrong for {instr:?}");
        }
    }

    #[test]
    fn display_formats_nullary_instructions() {
        assert_eq!(format!("{}", Instruction::Nop), "NOP");
        assert_eq!(format!("{}", Instruction::Push), "PUSH");
        assert_eq!(format!("{}", Instruction::Pop), "POP");
        assert_eq!(format!("{}", Instruction::Ret), "RET");
        assert_eq!(format!("{}", Instruction::Halt), "HALT");
    }

    #[test]
    fn display_formats_immediate_instructions() {
        assert_eq!(format!("{}", Instruction::LoadImmediate(42)), "LOADI 42");
        assert_eq!(format!("{}", Instruction::AddImmediate(-5)), "ADD -5");
        assert_eq!(format!("{}", Instruction::SubImmediate(100)), "SUB 100");
        assert_eq!(format!("{}", Instruction::MulImmediate(3)), "MUL 3");
    }

    #[test]
    fn display_formats_address_instructions() {
        assert_eq!(format!("{}", Instruction::Load(7)), "LOAD 7");
        assert_eq!(format!("{}", Instruction::Store(3)), "STORE 3");
        assert_eq!(format!("{}", Instruction::Jump(10)), "JMP 10");
        assert_eq!(format!("{}", Instruction::JumpIfZero(5)), "JZ 5");
        assert_eq!(format!("{}", Instruction::JumpIfNotZero(4)), "JNZ 4");
        assert_eq!(format!("{}", Instruction::JumpIfSign(6)), "JS 6");
        assert_eq!(format!("{}", Instruction::JumpIfNotSign(2)), "JNS 2");
        assert_eq!(format!("{}", Instruction::Call(8)), "CALL 8");
    }

    #[test]
    fn test_program_merge_memory_overlay() {
        let prog_a = Program::new(vec![Instruction::Halt], 3)
            .with_initial_memory(vec![5, 0, 3])
            .unwrap();
        let prog_b = Program::new(vec![Instruction::Halt], 2)
            .with_initial_memory(vec![0, 7])
            .unwrap();

        let (merged, _) = Program::merge(&[prog_a, prog_b]).unwrap();
        // Result should have 5 at addr 0 (from A), 7 at addr 1 (from B), 3 at addr 2 (from A)
        // B's 0 at addr 0 should NOT overwrite A's 5.
        assert_eq!(merged.initial_memory(), &[5, 7, 3]);
    }

    #[test]
    fn test_program_merge_offset_overflow_rejected() {
        // Prog C has Jump(10). If offset is 250, target becomes 260 which overflows u8.
        let prog_c = Program::new(vec![Instruction::Jump(10)], 1);
        let instrs_large = vec![Instruction::Nop; 250];
        let prog_large = Program::new(instrs_large, 1);

        let err = Program::merge(&[prog_large, prog_c]).unwrap_err();
        assert!(err.to_string().contains("exceeds u8 limit"));
    }
}
