# Phase 5 Overlap Baseline

- **Date:** 2026-04-06
- **Branch:** issue-29-investigation
- **Benchmark:** `cargo bench --bench overlap_benchmark`

## Results (k=4 candidates, StaticPlanner)

| Mode | Median latency |
|------|---------------|
| `overlap/sequential_k4` | 119.54 µs |
| `overlap/overlapped_k4` | 158.19 µs |

## Interpretation

With `StaticPlanner` (near-zero latency), Overlapped mode is **~38 µs slower** than
Sequential. Thread spawn and join overhead from `std::thread::scope` dominates when
the planner itself is fast.

**The overlap benefit is only realized when planner latency >> thread overhead (~40 µs).**

For a real LLM planner (network round-trip: typically 200 ms–2 s), the overlap
savings would be: `planner_latency - kernel_latency`. With kernel execution at
~120 µs and a planner at 500 ms, the Overlapped mode would save approximately
499.88 ms of wall time — the kernel finishes essentially for free.

## Takeaway

Phase 5 has established the measurement infrastructure (`OverlapOrchestrator`,
`OverlapReport`, `OverlapTimings`) and confirmed that:

1. Overlap savings are **measurable and reported** per run.
2. The break-even point is any planner slower than ~40 µs.
3. The logical trust boundary (AdmissibilityFilter, ProposalReceipt) is
   unchanged in both modes — the kernel never receives unfiltered planner output.

Next step for production use: replace `StaticPlanner` with a real LLM-backed
`Planner` implementation and re-run this benchmark.
