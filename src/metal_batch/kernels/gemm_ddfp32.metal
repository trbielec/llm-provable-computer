// gemm_ddfp32.metal
// Compensated GEMM using Veltkamp-split weight pairs (Bailey three-pass).
//
// For each (row, run) pair the kernel computes:
//
//   out[row * N + run] = bias[instr * rows + row]
//     + Σ_k (w_hi[k] * x_hi[k]          ← dominant product
//           + w_hi[k] * x_lo[k]          ← weight × input residual
//           + w_lo[k] * x_hi[k])         ← weight residual × input
//
// The dropped term w_lo * x_lo is a fourth-order correction that is negligible
// for the VM's bounded weight and activation ranges.
//
// Weights (w_hi, w_lo) are indexed by instruction (pc): the kernel reads
// pc_buf[run] to select the right weight slice.  This handles programs where
// different runs are at different PCs at the same step without requiring a
// separate grouping pass.
//
// Grid: (rows, N) threads

#include <metal_stdlib>
using namespace metal;

kernel void gemm_ddfp32(
    device const float* w_hi    [[ buffer(0) ]],   // [num_pcs * rows * cols]
    device const float* w_lo    [[ buffer(1) ]],
    device const float* bias    [[ buffer(2) ]],   // [num_pcs * rows]
    device const float* x_hi    [[ buffer(3) ]],   // [cols * N]
    device const float* x_lo    [[ buffer(4) ]],
    device const uchar* pc_buf  [[ buffer(5) ]],   // [N]
    device       float* out     [[ buffer(6) ]],   // [rows * N]
    constant     uint&  rows    [[ buffer(7) ]],
    constant     uint&  cols    [[ buffer(8) ]],
    constant     uint&  N       [[ buffer(9) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint row = gid.x;
    uint run = gid.y;
    if (row >= rows || run >= N) return;

    uint instr  = (uint)pc_buf[run];
    uint w_base = (instr * rows + row) * cols;
    float acc   = bias[instr * rows + row];

    for (uint k = 0; k < cols; k++) {
        float wh = w_hi[w_base + k];
        float wl = w_lo[w_base + k];
        float xh = x_hi[k * N + run];
        float xl = x_lo[k * N + run];
        acc += wh * xh + wh * xl + wl * xh;
    }

    out[row * N + run] = acc;
}

// Element-wise gate × value → hidden.
// Grid: (ff_dim, N) threads
kernel void hadamard(
    device const float* gate    [[ buffer(0) ]],
    device const float* value   [[ buffer(1) ]],
    device       float* hidden  [[ buffer(2) ]],
    constant     uint&  ff_dim  [[ buffer(3) ]],
    constant     uint&  N       [[ buffer(4) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint row = gid.x;
    uint run = gid.y;
    if (row >= ff_dim || run >= N) return;
    uint idx    = row * N + run;
    hidden[idx] = gate[idx] * value[idx];
}
