//! Externally-committed I/O buffer for TVM programs that use `IREAD`/`IWRITE`.
//!
//! [`IoBuffer`] occupies addresses 224–239 in the TVM memory map.  Before
//! execution, caller-supplied input values are injected into those cells via
//! [`IoBuffer::inject_into_memory`].  After execution the written output
//! values can be extracted with [`IoBuffer::extract_from_memory`].
//!
//! The kernel proof (vanilla STARK) then covers the use of those cells exactly
//! like any other memory-mapped access.  An external auditor who holds the
//! committed `input_values` can verify that `initial_state.memory[224..240]`
//! matches — closing the two-layer audit chain.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::memory::AddressedMemory;
use crate::model::TransformerVm;
use crate::proof::{
    prove_execution_stark_full, verify_execution_stark, VanillaStarkExecutionProof,
    VanillaStarkProofOptions,
};
use crate::state::{MachineState, Word};

/// The base address of the I/O buffer region in TVM memory.
pub const IO_BASE_ADDRESS: u8 = 224;

/// The number of I/O slots (keys 0–15).
pub const IO_CAPACITY: usize = 16;

/// Buffer of 16 externally-committed values, mapped to addresses 224–239.
///
/// Programs use `IREAD key` to load a value into ACC and `IWRITE key` to
/// store ACC back.  The buffer must be injected into the program's initial
/// memory before proof generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoBuffer {
    /// 16 word-sized slots (keys 0..=15).
    pub values: Vec<Word>,
}

impl Default for IoBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl IoBuffer {
    /// Create an empty (zero-initialised) buffer.
    pub fn new() -> Self {
        Self {
            values: vec![0; IO_CAPACITY],
        }
    }

    /// Set the value for a key slot.
    ///
    /// # Panics
    /// Panics if `key > 15`.
    pub fn set(&mut self, key: u8, value: Word) {
        assert!(key <= 15, "I/O key {key} exceeds limit 15");
        self.values[key as usize] = value;
    }

    /// Read the value for a key slot.
    ///
    /// # Panics
    /// Panics if `key > 15`.
    pub fn get(&self, key: u8) -> Word {
        assert!(key <= 15, "I/O key {key} exceeds limit 15");
        self.values[key as usize]
    }

    /// Write the buffer values into a memory slice at addresses 224–239.
    ///
    /// `memory` must have at least 240 elements; otherwise the call is a no-op
    /// for out-of-range addresses.
    pub fn inject_into_memory(&self, memory: &mut [Word]) {
        for (i, &val) in self.values.iter().enumerate() {
            let addr = IO_BASE_ADDRESS as usize + i;
            if addr < memory.len() {
                memory[addr] = val;
            }
        }
    }

    /// Extract the current I/O buffer values from a memory snapshot.
    ///
    /// Slots whose addresses are beyond `memory.len()` are read as zero.
    pub fn extract_from_memory(memory: &[Word]) -> Self {
        let mut buf = Self::new();
        for i in 0..IO_CAPACITY {
            let addr = IO_BASE_ADDRESS as usize + i;
            buf.values[i] = memory.get(addr).copied().unwrap_or(0);
        }
        buf
    }
}

/// Full audit trail for a proved I/O-bearing TVM execution.
///
/// Two-layer audit chain:
/// - `kernel_proof` proves that the kernel ran deterministically from
///   `initial_state` (which embeds the I/O inputs at addresses 224–239) to
///   `final_state`.
/// - `input_values` / `output_values` are the committed external I/O values.
///   A verifier can confirm that `kernel_proof.claim.initial_state.memory[224..240]`
///   matches `input_values`, completing the external-witness binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvedIoExecution {
    /// The vanilla STARK proof of kernel correctness.
    pub kernel_proof: VanillaStarkExecutionProof,
    /// Input values that were pre-loaded into the I/O buffer before execution.
    pub input_values: Vec<Word>,
    /// Output values written by `IWRITE` during execution.
    pub output_values: Vec<Word>,
}

/// Prove execution of a program that uses the I/O buffer.
///
/// The `io_input` values are injected into the program's initial memory at
/// addresses 224–239 before the proof is generated.  After execution the
/// output slots are extracted from the final state.
///
/// Programs must declare `.memory 240` (or more) to accommodate the I/O
/// region; otherwise `IREAD`/`IWRITE` will return `MemoryOutOfBounds` at
/// runtime.
pub fn prove_io_execution_stark(
    model: &TransformerVm,
    io_input: &IoBuffer,
    max_steps: usize,
    options: VanillaStarkProofOptions,
) -> Result<ProvedIoExecution> {
    // Build initial memory with I/O inputs pre-loaded.
    let mut initial_memory = model.program().initial_memory().to_vec();
    io_input.inject_into_memory(&mut initial_memory);
    let initial_state = MachineState::with_memory(initial_memory.clone());
    let memory = AddressedMemory::from_initial(&initial_memory);

    // Prove execution using the existing full prover.
    let kernel_proof =
        prove_execution_stark_full(model, memory, initial_state, max_steps, options)?;

    // Extract committed I/O outputs from the final state.
    let output_buf = IoBuffer::extract_from_memory(&kernel_proof.claim.final_state.memory);

    Ok(ProvedIoExecution {
        kernel_proof,
        input_values: io_input.values.clone(),
        output_values: output_buf.values,
    })
}

/// Verify the kernel proof inside a [`ProvedIoExecution`].
///
/// This checks only the STARK proof.  To complete the external-witness audit,
/// the caller should additionally confirm that
/// `result.kernel_proof.claim.initial_state.memory[224..240]` equals
/// `result.input_values`.
pub fn verify_io_execution_stark(result: &ProvedIoExecution) -> Result<bool> {
    verify_execution_stark(&result.kernel_proof)
}

/// Convenience wrapper: verify that the STARK proof is valid **and** that
/// the initial memory's I/O region matches the committed input values.
pub fn verify_io_execution_stark_full(result: &ProvedIoExecution) -> Result<bool> {
    if !verify_execution_stark(&result.kernel_proof)? {
        return Ok(false);
    }
    let initial_mem = &result.kernel_proof.claim.initial_state.memory;
    for (i, &expected) in result.input_values.iter().enumerate() {
        let addr = IO_BASE_ADDRESS as usize + i;
        let actual = initial_mem.get(addr).copied().unwrap_or(0);
        if actual != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Build a [`TransformerVm`] with the default attention mode, used in tests.
#[cfg(test)]
pub(crate) fn default_vm_for_program(source: &str) -> TransformerVm {
    use crate::assembly::parse_program;
    use crate::config::TransformerVmConfig;
    let program = parse_program(source).expect("valid program");
    TransformerVm::new(TransformerVmConfig::default(), program).expect("valid model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_buffer_default_is_zeroed() {
        let buf = IoBuffer::new();
        assert_eq!(buf.values, vec![0i64; 16]);
    }

    #[test]
    fn io_buffer_set_get_roundtrip() {
        let mut buf = IoBuffer::new();
        buf.set(3, 42);
        assert_eq!(buf.get(3), 42);
    }

    #[test]
    fn inject_into_memory_writes_correct_addresses() {
        let mut mem = vec![0i64; 255];
        let mut buf = IoBuffer::new();
        buf.set(0, 99);
        buf.set(15, 7);
        buf.inject_into_memory(&mut mem);
        assert_eq!(mem[224], 99);
        assert_eq!(mem[239], 7);
        assert_eq!(mem[223], 0);
    }

    #[test]
    fn extract_from_memory_reads_correct_addresses() {
        let mut mem = vec![0i64; 255];
        mem[224] = 10;
        mem[230] = 20;
        let buf = IoBuffer::extract_from_memory(&mem);
        assert_eq!(buf.get(0), 10);
        assert_eq!(buf.get(6), 20);
    }

    #[test]
    #[ignore = "STARK proof with 240-cell memory is too slow for the default suite; run with --include-ignored"]
    fn prove_io_execution_stark_simple_iread() {
        let source = r#"
.memory 240
IREAD 0
HALT
"#;
        let model = default_vm_for_program(source);
        let mut input = IoBuffer::new();
        input.set(0, 42);
        let proved =
            prove_io_execution_stark(&model, &input, 32, VanillaStarkProofOptions::default())
                .expect("prove succeeded");
        assert_eq!(proved.kernel_proof.claim.final_state.acc, 42);
        assert_eq!(proved.input_values[0], 42);
    }

    #[test]
    #[ignore = "STARK proof with 240-cell memory is too slow for the default suite; run with --include-ignored"]
    fn prove_io_execution_stark_iwrite_committed() {
        let source = r#"
.memory 240
IREAD 0
ADD 1
IWRITE 1
HALT
"#;
        let model = default_vm_for_program(source);
        let mut input = IoBuffer::new();
        input.set(0, 41);
        let proved =
            prove_io_execution_stark(&model, &input, 32, VanillaStarkProofOptions::default())
                .expect("prove succeeded");
        assert_eq!(proved.kernel_proof.claim.final_state.acc, 42);
        assert_eq!(proved.output_values[1], 42);
    }

    #[test]
    #[ignore = "STARK proof with 240-cell memory is too slow for the default suite; run with --include-ignored"]
    fn verify_io_execution_stark_full_accepts_valid_proof() {
        let source = r#"
.memory 240
IREAD 0
HALT
"#;
        let model = default_vm_for_program(source);
        let mut input = IoBuffer::new();
        input.set(0, 7);
        let proved =
            prove_io_execution_stark(&model, &input, 32, VanillaStarkProofOptions::default())
                .expect("prove succeeded");
        assert!(verify_io_execution_stark_full(&proved).expect("verify succeeded"));
    }
}
