// convert.metal
// Mirror of model.rs::build_input_vector, executed on GPU.
//
// Converts each run's GpuRunState + i64 operand into a pair of f32 planes
// (hi, lo) that represent the INPUT_DIM-element input vector using the
// Veltkamp split for large integer fields.  The planes are consumed by the
// compensated GEMM kernels in gemm_ddfp32.metal.
//
// Grid shape: (INPUT_DIM, N)
//   dim ∈ [0, INPUT_DIM)  — input-vector dimension
//   run ∈ [0, N)          — batch index

#include <metal_stdlib>
using namespace metal;

// Must match model.rs constants:
constant uint INPUT_DIM       = 137u;   // 9 + 2 * WORD_BITS
constant uint WORD_BITS       = 64u;
constant uint ACC_BITS_START  = 9u;
constant uint OP_BITS_START   = 73u;    // 9 + WORD_BITS

// Must match precision.rs GpuRunState (repr C).
struct GpuRunState {
    long  acc;
    uchar pc;
    uchar sp;
    uchar zero_flag;
    uchar carry_flag;
    uchar halted;
    uchar _pad[3];
};

// Mirror of model.rs bool_to_scalar: true → 1.0, false → 0.0.
static inline float bts(bool v) { return v ? 1.0f : 0.0f; }

// Veltkamp split on GPU: decomposes an i64 into (hi, lo) f32 such that
//   (hi as f64) + (lo as f64) ≈ v
// For |v| ≥ 2^53 (beyond f64 exact-integer range) lo is 0, matching the
// precision the CPU f64 path already loses at that scale.
static inline float2 split_i64(long v) {
    float hi = (float)v;                        // rounds to nearest f32
    if (!isfinite(hi)) return float2(hi, 0.0f);
    if (fabs(hi) >= 9007199254740992.0f)        // 2^53 – f64 exact-integer limit
        return float2(hi, 0.0f);
    long hi_long = (long)hi;
    float lo = (float)(v - hi_long);
    return float2(hi, lo);
}

kernel void state_to_input_split(
    device const GpuRunState* states    [[ buffer(0) ]],
    device const long*        operands  [[ buffer(1) ]],
    device       float*       out_hi    [[ buffer(2) ]],   // [INPUT_DIM × N]
    device       float*       out_lo    [[ buffer(3) ]],   // [INPUT_DIM × N]
    constant     uint&        N         [[ buffer(4) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint dim = gid.x;
    uint run = gid.y;
    if (dim >= INPUT_DIM || run >= N) return;

    GpuRunState s       = states[run];
    long        operand = operands[run];
    float hi = 0.0f, lo = 0.0f;

    if (dim == 0) {
        // IN_CONST = 1.0 (always exact in f32)
        hi = 1.0f;
    } else if (dim == 1) {
        // IN_PC = pc as scalar  (0–255, exact in f32)
        hi = (float)s.pc;
    } else if (dim == 2) {
        // IN_PC_NEXT = pc.wrapping_add(1) as scalar
        hi = (float)((uchar)(s.pc + 1u));
    } else if (dim == 3) {
        // IN_ACC = word_to_scalar(acc) — may be large, use split
        float2 split = split_i64(s.acc);
        hi = split.x; lo = split.y;
    } else if (dim == 4) {
        // IN_ZERO = bool_to_scalar(zero_flag)
        hi = bts(s.zero_flag != 0u);
    } else if (dim == 5) {
        // IN_CARRY
        hi = bts(s.carry_flag != 0u);
    } else if (dim == 6) {
        // IN_HALTED
        hi = bts(s.halted != 0u);
    } else if (dim == 7) {
        // IN_SP = sp as scalar (0–255, exact in f32)
        hi = (float)s.sp;
    } else if (dim == 8) {
        // IN_OPERAND = word_to_scalar(operand) — may be large, use split
        float2 split = split_i64(operand);
        hi = split.x; lo = split.y;
    } else if (dim < ACC_BITS_START + WORD_BITS) {
        // ACC bit: bool_to_scalar((acc >> bit) & 1)  (exact ±0/1 in f32)
        uint bit = dim - ACC_BITS_START;
        hi = bts(((ulong)s.acc >> bit) & 1uL);
    } else {
        // OPERAND bit
        uint bit = dim - OP_BITS_START;
        hi = bts(((ulong)operand >> bit) & 1uL);
    }

    uint idx  = dim * N + run;
    out_hi[idx] = hi;
    out_lo[idx] = lo;
}
