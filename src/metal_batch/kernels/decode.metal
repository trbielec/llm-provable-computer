// decode.metal
// GPU mirror of model.rs::transition_from_output.
//
// For each run this kernel:
//  1. reads the 6-element ff_output produced by the GEMM pipeline,
//  2. applies the exact rounding, clamping, and boolean-blend logic from
//     transition_from_output() in model.rs, and
//  3. writes the resulting GpuRunState and a GpuMemWriteReq to device memory.
//
// Precision note: the output scalars come from a f32 compensated GEMM.
// For values whose magnitude fits within f32's exact integer range (~2^24),
// the results are bit-identical to the CPU f64 path.  The `is_overflow` guard
// reproduces the f64-path boundary at ±2^63.
//
// Grid: N threads

#include <metal_stdlib>
using namespace metal;

// Must match model.rs output-index constants.
constant uint OUT_NEXT_PC          = 0u;
constant uint OUT_RAW_ACC          = 1u;
constant uint OUT_NEXT_SP          = 2u;
constant uint OUT_MEM_WRITE_ENABLE = 3u;
constant uint OUT_MEM_WRITE_ADDR   = 4u;
constant uint OUT_MEM_WRITE_VALUE  = 5u;
constant uint OUTPUT_DIM           = 6u;

// Must match precision.rs repr(C) structs.
struct GpuRunState {
    long  acc;
    uchar pc;
    uchar sp;
    uchar zero_flag;
    uchar carry_flag;
    uchar halted;
    uchar _pad[3];
};

struct GpuControls {
    float zero_prev;
    float zero_result;
    float zero_const;
    float carry_prev;
    float carry_overflow;
    float carry_less_than;
    float carry_const;
    float carry_rhs_const;
    float carry_rhs_operand;
    float halted_prev;
    float halted_result;
    float halted_const;
    uint  memory_read_type;
    uint  memory_read_addr;
    uint  _pad[2];
};

struct GpuMemWriteReq {
    long value;
    uint addr;
    uint enabled;
};

// Mirror of model.rs bool_to_scalar: true → 1.0, false → 0.0.
static inline float bts(bool v) { return v ? 1.0f : 0.0f; }

// Mirror of blend_bool: value >= 0.5.
static inline bool blend(float v) { return v >= 0.5f; }

// Mirror of overflowed_raw_acc_scalar.
// Word::MIN = -(2^63) ≈ -9.223e18, upper bound (exclusive) = 2^63 ≈ +9.223e18.
static inline bool is_overflow(float raw_acc) {
    return !isfinite(raw_acc)
        || raw_acc <  -9.223372036854776e18f    // Word::MIN
        || raw_acc >=  9.223372036854776e18f;   // Word::MAX + 1
}

kernel void decode_transition(
    device const float*        ff_output  [[ buffer(0) ]],  // [OUTPUT_DIM × N]
    device const GpuRunState*  state_in   [[ buffer(1) ]],  // [N]
    device       GpuRunState*  state_out  [[ buffer(2) ]],  // [N]
    device       GpuMemWriteReq* mem_writes [[ buffer(3) ]], // [N]
    device const GpuControls*  controls   [[ buffer(4) ]],  // [num_pcs]
    device const uchar*        pc_buf     [[ buffer(5) ]],  // [N]
    device const long*         operands   [[ buffer(6) ]],  // [N]
    constant     uint&         N          [[ buffer(7) ]],
    uint run [[ thread_position_in_grid ]])
{
    if (run >= N) return;

    GpuRunState s       = state_in[run];
    uint        instr   = (uint)pc_buf[run];
    GpuControls ctl     = controls[instr];
    long        operand = operands[run];

    // Gather the 6 output scalars for this run.
    float out[OUTPUT_DIM];
    for (uint i = 0; i < OUTPUT_DIM; i++)
        out[i] = ff_output[i * N + run];

    // ── mirror of transition_from_output ─────────────────────────────────────

    float raw_acc_f   = rint(out[OUT_RAW_ACC]);     // round-to-nearest (ties to even)
    long  acc         = (long)raw_acc_f;
    bool  result_zero = (acc == 0L);
    bool  overflow    = is_overflow(raw_acc_f);

    // compare_rhs for carry: (rhs_constant + rhs_operand_weight * operand).round()
    float rhs_f      = rint(ctl.carry_rhs_const + ctl.carry_rhs_operand * (float)operand);
    long  compare_rhs = (long)rhs_f;
    bool  less_than  = s.acc < compare_rhs;

    long next_pc = (long)rint(out[OUT_NEXT_PC]);
    long next_sp = (long)rint(out[OUT_NEXT_SP]);

    bool next_zero = blend(
          ctl.zero_prev   * bts(s.zero_flag  != 0u)
        + ctl.zero_result * bts(result_zero)
        + ctl.zero_const);

    bool next_carry = blend(
          ctl.carry_prev      * bts(s.carry_flag != 0u)
        + ctl.carry_overflow  * bts(overflow)
        + ctl.carry_less_than * bts(less_than)
        + ctl.carry_const);

    // halted_result is always multiplied by bool_to_scalar(false) = 0 in the
    // CPU path, so the term drops out.
    bool next_halted = blend(
          ctl.halted_prev   * bts(s.halted != 0u)
        + ctl.halted_const);

    bool mem_enable = out[OUT_MEM_WRITE_ENABLE] >= 0.5f;
    long mem_addr   = (long)rint(out[OUT_MEM_WRITE_ADDR]);
    long mem_val    = (long)rint(out[OUT_MEM_WRITE_VALUE]);

    // ── write next state ─────────────────────────────────────────────────────
    GpuRunState next;
    next.acc        = acc;
    next.pc         = (uchar)(next_pc & 0xFFL);
    next.sp         = (uchar)(next_sp & 0xFFL);
    next.zero_flag  = (uchar)next_zero;
    next.carry_flag = (uchar)next_carry;
    next.halted     = (uchar)next_halted;
    next._pad[0]    = 0u; next._pad[1] = 0u; next._pad[2] = 0u;
    state_out[run]  = next;

    GpuMemWriteReq req;
    req.value   = mem_val;
    req.addr    = (uint)(mem_addr & 0xFFL);
    req.enabled = mem_enable ? 1u : 0u;
    mem_writes[run] = req;
}
