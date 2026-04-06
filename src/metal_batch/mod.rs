//! Metal GPU batch runner for data-parallel program testing (Phase 2, issue #19).
//!
//! [`MetalBatchRunner`] executes N independent runs of the same program on an
//! Apple Silicon GPU.  Each step dispatches five kernel passes:
//!
//! 1. **`state_to_input_split`** — convert each run's `GpuRunState` + operand
//!    into a Veltkamp-split `(hi, lo)` input vector.
//! 2. **`gemm_ddfp32` × 2** — compensated gate and value GEMMs.
//! 3. **`hadamard`** — element-wise `gate × value → hidden`.
//! 4. **`gemm_ddfp32`** — output GEMM: `W_o × hidden → ff_output`.
//! 5. **`decode_transition`** — apply `transition_from_output`, writing next
//!    state and memory-write requests.
//!
//! Memory reads (attention context) and writes are handled CPU-side because
//! `AddressedMemory`'s hull-backed KV cache is difficult to replicate in MSL.
//!
//! # Precision
//! The compensated GEMM reproduces the CPU `f64` path exactly for values whose
//! magnitude fits within `f32`'s exact integer range (~2^24).  This runner is
//! a **testing-only path**; proof generation must use the CPU transformer.

pub mod buffers;
pub mod precision;

use std::sync::Arc;
use std::time::Instant;

use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Library, MTLSize,
};

use crate::engine::{build_execution_result, ExecutionResult};
use crate::error::{Result, VmError};
use crate::memory::AddressedMemory;
use crate::model::{MemoryRead, TransformerVm, INPUT_DIM, OUTPUT_DIM};
use crate::state::MachineState;

use self::buffers::{GpuStepBuffers, GpuWeightBuffers};
use self::precision::{GpuMemWriteReq, GpuRunState};

// ── MSL source (compiled at runtime) ─────────────────────────────────────────

const CONVERT_MSL: &str = include_str!("kernels/convert.metal");
const GEMM_MSL: &str = include_str!("kernels/gemm_ddfp32.metal");
const DECODE_MSL: &str = include_str!("kernels/decode.metal");

// ── compiled pipelines ────────────────────────────────────────────────────────

struct Pipelines {
    convert: ComputePipelineState,
    gemm: ComputePipelineState,
    hadamard: ComputePipelineState,
    decode: ComputePipelineState,
}

fn compile_lib(device: &Device, src: &str, label: &str) -> Result<Library> {
    device
        .new_library_with_source(src, &CompileOptions::new())
        .map_err(|e| VmError::InvalidConfig(format!("Metal compile ({label}): {e}")))
}

fn make_pipeline(device: &Device, lib: &Library, name: &str) -> Result<ComputePipelineState> {
    let func = lib
        .get_function(name, None)
        .map_err(|e| VmError::InvalidConfig(format!("Metal fn `{name}` not found: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| VmError::InvalidConfig(format!("pipeline `{name}`: {e}")))
}

// ── public types ──────────────────────────────────────────────────────────────

/// The per-run result from [`MetalBatchRunner::run`].
#[derive(Debug)]
pub struct MetalBatchRunResult {
    pub run_idx: usize,
    pub result: Result<ExecutionResult>,
    pub final_state: MachineState,
}

/// GPU batch runner backed by Metal compute kernels.
pub struct MetalBatchRunner {
    model: Arc<TransformerVm>,
    max_steps: usize,
    device: Device,
    queue: metal::CommandQueue,
    pipelines: Pipelines,
    weights: GpuWeightBuffers,
}

impl MetalBatchRunner {
    /// Compile MSL shaders and upload weight matrices.
    pub fn new(model: Arc<TransformerVm>, max_steps: usize) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| VmError::InvalidConfig("No Metal device found".into()))?;
        let queue = device.new_command_queue();

        let lib_conv = compile_lib(&device, CONVERT_MSL, "convert")?;
        let lib_gemm = compile_lib(&device, GEMM_MSL, "gemm")?;
        let lib_decode = compile_lib(&device, DECODE_MSL, "decode")?;

        let pipelines = Pipelines {
            convert: make_pipeline(&device, &lib_conv, "state_to_input_split")?,
            gemm: make_pipeline(&device, &lib_gemm, "gemm_ddfp32")?,
            hadamard: make_pipeline(&device, &lib_gemm, "hadamard")?,
            decode: make_pipeline(&device, &lib_decode, "decode_transition")?,
        };

        let weights = GpuWeightBuffers::new(&device, &model);
        Ok(Self {
            model,
            max_steps,
            device,
            queue,
            pipelines,
            weights,
        })
    }

    /// Run the program for each initial state on the GPU.
    ///
    /// Uses CPU-side stream compaction: after each step, runs that have halted
    /// are retired from the active set by swapping them to the end of the
    /// parallel arrays and shrinking `n_active`.  This reduces GPU thread
    /// utilisation proportionally as runs complete, recovering throughput for
    /// programs with variable termination depth (Phase 3, issue #19).
    pub fn run(&self, initial_states: Vec<MachineState>) -> Vec<MetalBatchRunResult> {
        let n = initial_states.len();
        if n == 0 {
            return Vec::new();
        }

        // Results indexed by original run_idx; filled as runs halt.
        let mut results: Vec<Option<MetalBatchRunResult>> = (0..n).map(|_| None).collect();

        // slot_map[slot] = original run_idx — updated during compaction.
        let mut slot_map: Vec<usize> = (0..n).collect();

        let mut memories: Vec<AddressedMemory> = initial_states
            .iter()
            .map(|s| AddressedMemory::from_initial(&s.memory))
            .collect();
        let mut gpu_states: Vec<GpuRunState> = initial_states
            .iter()
            .map(GpuRunState::from_machine_state)
            .collect();
        let mut step_counts = vec![0usize; n];

        // GPU scratch buffers are sized for the full n; we only dispatch
        // n_active threads per step, leaving trailing slots untouched.
        let scratch = GpuStepBuffers::new(&self.device, n, self.weights.ff_dim);
        let start = Instant::now();

        let mut n_active = n;

        for _ in 0..self.max_steps {
            if n_active == 0 {
                break;
            }

            // ── CPU: memory reads for active slots ────────────────────────────
            let operands: Vec<i64> = (0..n_active)
                .map(|i| {
                    let pc = gpu_states[i].pc;
                    let sp = gpu_states[i].sp;
                    let Ok((ci, _)) = self.model.compiled_instruction(pc) else {
                        return 0i64;
                    };
                    match ci.memory_read {
                        MemoryRead::None => 0i64,
                        MemoryRead::Direct(a) => memories[i].load(a).unwrap_or(0),
                        MemoryRead::StackTop => memories[i].load(sp).unwrap_or(0),
                    }
                })
                .collect();

            // Upload the active prefix only.
            write_slice(&scratch.state_a, &gpu_states[..n_active]);
            write_slice(&scratch.operands, &operands);
            let pc_bytes: Vec<u8> = gpu_states[..n_active].iter().map(|s| s.pc).collect();
            write_slice(&scratch.pc_buf, &pc_bytes);

            // ── GPU pipeline (n_active threads) ──────────────────────────────
            self.dispatch_step(&scratch, n_active);

            // ── CPU: apply results ────────────────────────────────────────────
            // Copy next states into the working arrays before compaction.
            {
                let next_states: &[GpuRunState] = read_slice(&scratch.state_b, n_active);
                let mem_reqs: &[GpuMemWriteReq] = read_slice(&scratch.mem_writes, n_active);
                for i in 0..n_active {
                    gpu_states[i] = next_states[i];
                    step_counts[i] += 1;
                    if mem_reqs[i].enabled != 0 {
                        let _ = memories[i].store(
                            mem_reqs[i].addr as u8,
                            mem_reqs[i].value,
                            step_counts[i],
                        );
                    }
                }
            }

            // ── Stream compaction: retire halted runs ─────────────────────────
            // Scan forward; on halt, swap with the last active slot and shrink.
            // Not incrementing `i` after a swap lets us re-check the swapped-in
            // element in the same position.
            let elapsed = start.elapsed();
            let mut i = 0;
            while i < n_active {
                if gpu_states[i].halted != 0 {
                    let run_idx = slot_map[i];
                    let mem = memories[i].snapshot();
                    let fs = gpu_states[i].to_machine_state(mem);
                    let exec = build_execution_result(&fs, step_counts[i], elapsed);
                    results[run_idx] = Some(MetalBatchRunResult {
                        run_idx,
                        result: Ok(exec),
                        final_state: fs,
                    });
                    // Swap retiring slot with the last active slot.
                    n_active -= 1;
                    if i < n_active {
                        gpu_states.swap(i, n_active);
                        memories.swap(i, n_active);
                        step_counts.swap(i, n_active);
                        slot_map.swap(i, n_active);
                    }
                    // i stays the same: re-examine the swapped-in slot.
                } else {
                    i += 1;
                }
            }
        }

        // Flush runs that exhausted max_steps without halting.
        let elapsed = start.elapsed();
        for i in 0..n_active {
            let run_idx = slot_map[i];
            let mem = memories[i].snapshot();
            let fs = gpu_states[i].to_machine_state(mem);
            let exec = build_execution_result(&fs, step_counts[i], elapsed);
            results[run_idx] = Some(MetalBatchRunResult {
                run_idx,
                result: Ok(exec),
                final_state: fs,
            });
        }

        results.into_iter().map(Option::unwrap).collect()
    }

    // ── GPU kernel dispatch ───────────────────────────────────────────────────

    fn dispatch_step(&self, s: &GpuStepBuffers, n: usize) {
        let ff_dim = self.weights.ff_dim as u64;
        let n64 = n as u64;
        let in_dim = INPUT_DIM as u64;
        let out_dim = OUTPUT_DIM as u64;
        let cmd = self.queue.new_command_buffer();

        // 1. convert → input split planes
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.convert);
            enc.set_buffer(0, Some(&s.state_a), 0);
            enc.set_buffer(1, Some(&s.operands), 0);
            enc.set_buffer(2, Some(&s.input_hi), 0);
            enc.set_buffer(3, Some(&s.input_lo), 0);
            set_u32(enc, 4, n as u32);
            dispatch_2d(enc, &self.pipelines.convert, in_dim, n64);
            enc.end_encoding();
        }

        // 2. gate GEMM
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.gemm);
            enc.set_buffer(0, Some(&self.weights.gate_hi), 0);
            enc.set_buffer(1, Some(&self.weights.gate_lo), 0);
            enc.set_buffer(2, Some(&self.weights.gate_bias), 0);
            enc.set_buffer(3, Some(&s.input_hi), 0);
            enc.set_buffer(4, Some(&s.input_lo), 0);
            enc.set_buffer(5, Some(&s.pc_buf), 0);
            enc.set_buffer(6, Some(&s.gate), 0);
            set_u32(enc, 7, ff_dim as u32);
            set_u32(enc, 8, in_dim as u32);
            set_u32(enc, 9, n as u32);
            dispatch_2d(enc, &self.pipelines.gemm, ff_dim, n64);
            enc.end_encoding();
        }

        // 3. value GEMM
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.gemm);
            enc.set_buffer(0, Some(&self.weights.value_hi), 0);
            enc.set_buffer(1, Some(&self.weights.value_lo), 0);
            enc.set_buffer(2, Some(&self.weights.value_bias), 0);
            enc.set_buffer(3, Some(&s.input_hi), 0);
            enc.set_buffer(4, Some(&s.input_lo), 0);
            enc.set_buffer(5, Some(&s.pc_buf), 0);
            enc.set_buffer(6, Some(&s.value), 0);
            set_u32(enc, 7, ff_dim as u32);
            set_u32(enc, 8, in_dim as u32);
            set_u32(enc, 9, n as u32);
            dispatch_2d(enc, &self.pipelines.gemm, ff_dim, n64);
            enc.end_encoding();
        }

        // 4. hadamard: gate * value → hidden
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.hadamard);
            enc.set_buffer(0, Some(&s.gate), 0);
            enc.set_buffer(1, Some(&s.value), 0);
            enc.set_buffer(2, Some(&s.hidden), 0);
            set_u32(enc, 3, ff_dim as u32);
            set_u32(enc, 4, n as u32);
            dispatch_2d(enc, &self.pipelines.hadamard, ff_dim, n64);
            enc.end_encoding();
        }

        // 5. out GEMM: W_o × hidden → ff_output
        //    hidden is already f32 (product of two f32 values); lo = 0.
        //    We pass hidden_lo (a zeroed buffer) as the x_lo argument.
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.gemm);
            enc.set_buffer(0, Some(&self.weights.out_hi), 0);
            enc.set_buffer(1, Some(&self.weights.out_lo), 0);
            enc.set_buffer(2, Some(&self.weights.out_bias), 0);
            enc.set_buffer(3, Some(&s.hidden), 0); // x_hi
            enc.set_buffer(4, Some(&s.hidden_lo), 0); // x_lo = 0
            enc.set_buffer(5, Some(&s.pc_buf), 0);
            enc.set_buffer(6, Some(&s.output), 0);
            set_u32(enc, 7, out_dim as u32);
            set_u32(enc, 8, ff_dim as u32);
            set_u32(enc, 9, n as u32);
            dispatch_2d(enc, &self.pipelines.gemm, out_dim, n64);
            enc.end_encoding();
        }

        // 6. decode: ff_output → next state + mem_write_reqs
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipelines.decode);
            enc.set_buffer(0, Some(&s.output), 0);
            enc.set_buffer(1, Some(&s.state_a), 0);
            enc.set_buffer(2, Some(&s.state_b), 0);
            enc.set_buffer(3, Some(&s.mem_writes), 0);
            enc.set_buffer(4, Some(&self.weights.controls), 0);
            enc.set_buffer(5, Some(&s.pc_buf), 0);
            enc.set_buffer(6, Some(&s.operands), 0);
            set_u32(enc, 7, n as u32);
            dispatch_1d(enc, &self.pipelines.decode, n64);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();
    }
}

// ── low-level buffer helpers ──────────────────────────────────────────────────

fn write_slice<T: Copy>(buf: &metal::Buffer, data: &[T]) {
    let byte_len = std::mem::size_of_val(data);
    // SAFETY: StorageModeShared buffers are CPU-accessible; sizes match.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            buf.contents().cast::<u8>(),
            byte_len,
        );
    }
}

fn read_slice<T>(buf: &metal::Buffer, count: usize) -> &[T] {
    // SAFETY: wait_until_completed has returned; buffer is CPU-visible.
    unsafe { std::slice::from_raw_parts(buf.contents().cast::<T>(), count) }
}

fn set_u32(enc: &ComputeCommandEncoderRef, idx: u64, value: u32) {
    enc.set_bytes(
        idx,
        std::mem::size_of::<u32>() as u64,
        (&raw const value).cast(),
    );
}

/// Dispatches a 2D grid of (dim0 × dim1) threads using the pipeline's
/// preferred thread-execution width as the threadgroup width.
fn dispatch_2d(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dim0: u64,
    dim1: u64,
) {
    let tw = pipeline.thread_execution_width();
    // Round up grid to a multiple of tw in the x dimension.
    let gx = dim0.div_ceil(tw) * tw;
    let threadgroups = MTLSize {
        width: gx / tw,
        height: dim1,
        depth: 1,
    };
    let per_group = MTLSize {
        width: tw,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(threadgroups, per_group);
}

fn dispatch_1d(enc: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState, dim: u64) {
    let tw = pipeline.thread_execution_width();
    let groups = dim.div_ceil(tw);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    let per_group = MTLSize {
        width: tw,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(threadgroups, per_group);
}
