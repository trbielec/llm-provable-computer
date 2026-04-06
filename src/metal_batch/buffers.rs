//! Metal buffer lifecycle for the batch runner.
//!
//! [`GpuWeightBuffers`] holds the read-only, per-instruction weight planes
//! (split into hi/lo pairs) and transition-control constants.  These are
//! uploaded once at [`MetalBatchRunner`][super::MetalBatchRunner] construction
//! time and reused across every step and every batch call.
//!
//! Per-step scratch buffers (`GpuStepBuffers`) are allocated once for a given
//! batch size N and reused across steps.

use metal::{Buffer, Device, MTLResourceOptions};

use crate::model::{TransformerVm, INPUT_DIM, OUTPUT_DIM};

use super::precision::{GpuControls, SplitWeights};

// ── helpers ───────────────────────────────────────────────────────────────────

fn upload<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let byte_len = std::mem::size_of_val(data) as u64;
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_uninit(device: &Device, byte_len: u64) -> Buffer {
    device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
}

fn alloc_slice<T>(device: &Device, count: usize) -> Buffer {
    alloc_uninit(device, (count * std::mem::size_of::<T>()) as u64)
}

// ── GpuWeightBuffers ──────────────────────────────────────────────────────────

/// Read-only GPU buffers for all compiled instructions.
///
/// Layout for each weight plane: `[num_pcs * rows * cols]` elements, where
/// the `num_pcs` axis is the first (outermost), indexed by the run's current
/// PC value.  Unused PC slots are zeroed.
pub struct GpuWeightBuffers {
    // Gate weight planes (ff_dim × INPUT_DIM per instruction).
    pub gate_hi: Buffer,
    pub gate_lo: Buffer,
    pub gate_bias: Buffer,
    // Value weight planes (ff_dim × INPUT_DIM per instruction).
    pub value_hi: Buffer,
    pub value_lo: Buffer,
    pub value_bias: Buffer,
    // Output weight planes (OUTPUT_DIM × ff_dim per instruction).
    pub out_hi: Buffer,
    pub out_lo: Buffer,
    pub out_bias: Buffer,
    // GpuControls for each PC slot.
    pub controls: Buffer,

    pub ff_dim: usize,
    pub num_pcs: usize,
}

impl GpuWeightBuffers {
    /// Build GPU buffers from `model`.
    ///
    /// Each instruction is placed at its PC index; unused PC slots are zero.
    pub fn new(device: &Device, model: &TransformerVm) -> Self {
        let num_pcs = model.program().len();
        let ff_dim = model.config().ff_dim;
        let input_dim = INPUT_DIM;
        let output_dim = OUTPUT_DIM;

        let gate_cells = num_pcs * ff_dim * input_dim;
        let value_cells = num_pcs * ff_dim * input_dim;
        let out_cells = num_pcs * output_dim * ff_dim;
        let bias_g_cells = num_pcs * ff_dim;
        let bias_v_cells = num_pcs * ff_dim;
        let bias_o_cells = num_pcs * output_dim;

        let mut gate_hi = vec![0f32; gate_cells];
        let mut gate_lo = vec![0f32; gate_cells];
        let mut gate_bias = vec![0f32; bias_g_cells];
        let mut value_hi = vec![0f32; value_cells];
        let mut value_lo = vec![0f32; value_cells];
        let mut value_bias = vec![0f32; bias_v_cells];
        let mut out_hi = vec![0f32; out_cells];
        let mut out_lo = vec![0f32; out_cells];
        let mut out_bias = vec![0f32; bias_o_cells];
        let mut controls = vec![
            GpuControls {
                zero_prev: 0.0,
                zero_result: 0.0,
                zero_const: 0.0,
                carry_prev: 0.0,
                carry_overflow: 0.0,
                carry_less_than: 0.0,
                carry_const: 0.0,
                carry_rhs_const: 0.0,
                carry_rhs_operand: 0.0,
                halted_prev: 0.0,
                halted_result: 0.0,
                halted_const: 0.0,
                memory_read_type: 0,
                memory_read_addr: 0,
                _pad: [0, 0],
            };
            num_pcs
        ];

        for (pc, ctl) in controls.iter_mut().enumerate() {
            let Ok((ci, _)) = model.compiled_instruction(pc as u8) else {
                continue;
            };
            let sw = SplitWeights::from_ff_weights(&ci.ff_weights);

            let g_base = pc * ff_dim * input_dim;
            gate_hi[g_base..g_base + ff_dim * input_dim].copy_from_slice(&sw.gate_hi);
            gate_lo[g_base..g_base + ff_dim * input_dim].copy_from_slice(&sw.gate_lo);
            value_hi[g_base..g_base + ff_dim * input_dim].copy_from_slice(&sw.value_hi);
            value_lo[g_base..g_base + ff_dim * input_dim].copy_from_slice(&sw.value_lo);

            let gb_base = pc * ff_dim;
            gate_bias[gb_base..gb_base + ff_dim].copy_from_slice(&sw.gate_bias);
            value_bias[gb_base..gb_base + ff_dim].copy_from_slice(&sw.value_bias);

            let o_base = pc * output_dim * ff_dim;
            out_hi[o_base..o_base + output_dim * ff_dim].copy_from_slice(&sw.out_hi);
            out_lo[o_base..o_base + output_dim * ff_dim].copy_from_slice(&sw.out_lo);

            let ob_base = pc * output_dim;
            out_bias[ob_base..ob_base + output_dim].copy_from_slice(&sw.out_bias);

            *ctl = GpuControls::from_compiled(ci);
        }

        Self {
            gate_hi: upload(device, &gate_hi),
            gate_lo: upload(device, &gate_lo),
            gate_bias: upload(device, &gate_bias),
            value_hi: upload(device, &value_hi),
            value_lo: upload(device, &value_lo),
            value_bias: upload(device, &value_bias),
            out_hi: upload(device, &out_hi),
            out_lo: upload(device, &out_lo),
            out_bias: upload(device, &out_bias),
            controls: upload(device, &controls),
            ff_dim,
            num_pcs,
        }
    }
}

// ── GpuStepBuffers ────────────────────────────────────────────────────────────

/// Per-step scratch buffers, allocated once for a given batch size `n`.
///
/// All buffers use `StorageModeShared` so the CPU can read/write them without
/// an explicit blit.
pub struct GpuStepBuffers {
    pub n: usize,
    // Writable state (ping-pong between steps).
    pub state_a: Buffer, // GpuRunState[n]
    pub state_b: Buffer, // GpuRunState[n]
    // Per-run operand (i64) read from AddressedMemory on CPU.
    pub operands: Buffer, // i64[n]
    // Current PC per run (u8) — kept in sync with state.pc.
    pub pc_buf: Buffer, // u8[n]
    // Input-vector split planes, output of convert kernel.
    pub input_hi: Buffer, // f32[INPUT_DIM × n]
    pub input_lo: Buffer, // f32[INPUT_DIM × n]
    // Intermediate activations.
    pub gate: Buffer,   // f32[ff_dim × n]
    pub value: Buffer,  // f32[ff_dim × n]
    pub hidden: Buffer, // f32[ff_dim × n]
    /// Zero buffer, same size as `hidden`, used as the x_lo plane for the
    /// output GEMM (hidden values are already f32, so their residual is 0).
    pub hidden_lo: Buffer, // f32[ff_dim × n] — zeroed, never written by kernels
    pub output: Buffer, // f32[OUTPUT_DIM × n]
    // Memory-write requests from the decode kernel.
    pub mem_writes: Buffer, // GpuMemWriteReq[n]
}

impl GpuStepBuffers {
    pub fn new(device: &Device, n: usize, ff_dim: usize) -> Self {
        use std::mem::size_of;
        // Allocate and zero hidden_lo once; it is never written to.
        let zeros_ff: Vec<f32> = vec![0f32; ff_dim * n];
        Self {
            n,
            state_a: alloc_uninit(device, (n * 16) as u64), // GpuRunState = 16 bytes
            state_b: alloc_uninit(device, (n * 16) as u64),
            operands: alloc_slice::<i64>(device, n),
            pc_buf: alloc_slice::<u8>(device, n),
            input_hi: alloc_slice::<f32>(device, INPUT_DIM * n),
            input_lo: alloc_slice::<f32>(device, INPUT_DIM * n),
            gate: alloc_slice::<f32>(device, ff_dim * n),
            value: alloc_slice::<f32>(device, ff_dim * n),
            hidden: alloc_slice::<f32>(device, ff_dim * n),
            hidden_lo: upload(device, &zeros_ff),
            output: alloc_slice::<f32>(device, OUTPUT_DIM * n),
            mem_writes: alloc_uninit(
                device,
                (n * size_of::<super::precision::GpuMemWriteReq>()) as u64,
            ),
        }
    }
}
