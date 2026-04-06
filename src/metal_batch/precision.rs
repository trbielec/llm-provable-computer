//! Veltkamp split and GPU-facing structs for the Metal batch runner.
//!
//! The Veltkamp split decomposes a `f64` value into an exact `(f32 hi, f32 lo)`
//! pair such that `hi as f64 + lo as f64 ≈ x` with the rounding error captured
//! in `lo`.  For power-of-two values (all the bit-weight entries in the VM weight
//! matrices) `lo == 0.0` exactly, so the compensated GEMM produces identical
//! results to the CPU `f64` path for those terms.

use crate::model::{CompiledInstruction, FeedForwardWeights, Matrix, MemoryRead};

// ── Veltkamp split ────────────────────────────────────────────────────────────

/// Exact decomposition of `x: f64` into `(hi, lo): (f32, f32)`.
///
/// Invariant (in exact arithmetic): `hi as f64 + lo as f64 == x`.
pub fn veltkamp_split(x: f64) -> (f32, f32) {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    (hi, lo)
}

// ── Split weight matrices ─────────────────────────────────────────────────────

/// A weight [`Matrix`] decomposed into `(hi, lo)` f32 planes for the GPU GEMM.
#[derive(Debug, Clone)]
pub struct SplitMatrix {
    pub rows: usize,
    pub cols: usize,
    /// High parts, row-major: `hi[row * cols + col]`.
    pub hi: Vec<f32>,
    /// Low parts (Veltkamp residuals), same layout as `hi`.
    pub lo: Vec<f32>,
}

impl SplitMatrix {
    pub(crate) fn from_matrix(m: &Matrix) -> Self {
        let (hi, lo): (Vec<f32>, Vec<f32>) = m.data.iter().copied().map(veltkamp_split).unzip();
        Self {
            rows: m.rows,
            cols: m.cols,
            hi,
            lo,
        }
    }
}

/// All split weight planes for one compiled instruction, ready for GPU upload.
#[derive(Debug, Clone)]
pub struct SplitWeights {
    pub gate_hi: Vec<f32>,
    pub gate_lo: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub value_hi: Vec<f32>,
    pub value_lo: Vec<f32>,
    pub value_bias: Vec<f32>,
    pub out_hi: Vec<f32>,
    pub out_lo: Vec<f32>,
    pub out_bias: Vec<f32>,
    /// Number of rows in gate/value matrices (== ff_dim).
    pub ff_dim: usize,
    /// Number of columns in gate/value matrices (== INPUT_DIM = 137).
    pub input_dim: usize,
    /// Number of rows in out matrix (== OUTPUT_DIM = 6).
    pub output_dim: usize,
}

impl SplitWeights {
    pub(crate) fn from_ff_weights(fw: &FeedForwardWeights) -> Self {
        let gate = SplitMatrix::from_matrix(&fw.gate);
        let value = SplitMatrix::from_matrix(&fw.value);
        let out = SplitMatrix::from_matrix(&fw.out);
        let ff_dim = gate.rows;
        let input_dim = gate.cols;
        let output_dim = out.rows;
        Self {
            gate_hi: gate.hi,
            gate_lo: gate.lo,
            gate_bias: fw.gate_bias.iter().copied().map(|x| x as f32).collect(),
            value_hi: value.hi,
            value_lo: value.lo,
            value_bias: fw.value_bias.iter().copied().map(|x| x as f32).collect(),
            out_hi: out.hi,
            out_lo: out.lo,
            out_bias: fw.out_bias.iter().copied().map(|x| x as f32).collect(),
            ff_dim,
            input_dim,
            output_dim,
        }
    }
}

// ── GPU-ABI structs ───────────────────────────────────────────────────────────
// All `repr(C)` layouts must match their counterparts in the `.metal` kernels.

/// GPU representation of [`TransitionControls`] + memory-read metadata.
///
/// Must match `GpuControls` in `decode.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuControls {
    pub zero_prev: f32,
    pub zero_result: f32,
    pub zero_const: f32,
    pub carry_prev: f32,
    pub carry_overflow: f32,
    pub carry_less_than: f32,
    pub carry_const: f32,
    pub carry_rhs_const: f32,
    pub carry_rhs_operand: f32,
    pub halted_prev: f32,
    pub halted_result: f32,
    pub halted_const: f32,
    /// 0 = None, 1 = Direct(addr), 2 = StackTop
    pub memory_read_type: u32,
    /// Address for `Direct` reads; ignored otherwise.
    pub memory_read_addr: u32,
    pub _pad: [u32; 2],
}

impl GpuControls {
    pub(crate) fn from_compiled(ci: &CompiledInstruction) -> Self {
        let (mr_type, mr_addr) = match ci.memory_read {
            MemoryRead::None => (0u32, 0u32),
            MemoryRead::Direct(a) => (1u32, u32::from(a)),
            MemoryRead::StackTop => (2u32, 0u32),
        };
        let tc = &ci.controls;
        Self {
            zero_prev: tc.zero.prev_weight as f32,
            zero_result: tc.zero.result_weight as f32,
            zero_const: tc.zero.constant as f32,
            carry_prev: tc.carry.prev_weight as f32,
            carry_overflow: tc.carry.overflow_weight as f32,
            carry_less_than: tc.carry.less_than_weight as f32,
            carry_const: tc.carry.constant as f32,
            carry_rhs_const: tc.carry.rhs_constant as f32,
            carry_rhs_operand: tc.carry.rhs_operand_weight as f32,
            halted_prev: tc.halted.prev_weight as f32,
            halted_result: tc.halted.result_weight as f32,
            halted_const: tc.halted.constant as f32,
            memory_read_type: mr_type,
            memory_read_addr: mr_addr,
            _pad: [0, 0],
        }
    }
}

/// Per-run machine state stored in the GPU state buffer.
///
/// Must match `GpuRunState` in `convert.metal` and `decode.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuRunState {
    pub acc: i64,
    pub pc: u8,
    pub sp: u8,
    /// 1 = `true`, 0 = `false`.
    pub zero_flag: u8,
    pub carry_flag: u8,
    pub halted: u8,
    pub _pad: [u8; 3],
}

impl GpuRunState {
    pub fn from_machine_state(s: &crate::state::MachineState) -> Self {
        Self {
            acc: s.acc,
            pc: s.pc,
            sp: s.sp,
            zero_flag: s.zero_flag as u8,
            carry_flag: s.carry_flag as u8,
            halted: s.halted as u8,
            _pad: [0; 3],
        }
    }

    pub fn to_machine_state(&self, memory: Vec<crate::state::Word>) -> crate::state::MachineState {
        crate::state::MachineState {
            acc: self.acc,
            pc: self.pc,
            sp: self.sp,
            zero_flag: self.zero_flag != 0,
            carry_flag: self.carry_flag != 0,
            halted: self.halted != 0,
            memory,
        }
    }
}

/// Memory-write request emitted by the decode kernel for one run.
///
/// Must match `GpuMemWriteReq` in `decode.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemWriteReq {
    pub value: i64,
    pub addr: u32,
    /// 1 = write pending, 0 = no write.
    pub enabled: u32,
}
