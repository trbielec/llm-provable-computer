# Direct STARK Benchmarking

## Purpose

This note is the runbook for measuring the direct cDMN STARK backend separately from the older VM proof path.

## Benchmarks

Use the Criterion bench target in [vanillastark_benchmark.rs](/Users/tombielecki/Documents/Code/llm-provable-computer/benches/vanillastark_benchmark.rs).

Relevant groups:

- `policy_proofs`
  - VM-backed DMN/cDMN proofs
- `direct_policy_proofs`
  - direct cDMN STARK proofs

## Command

```bash
cargo bench --bench vanillastark_benchmark direct_policy_proofs
```

To compare direct and VM-backed policy proofs:

```bash
cargo bench --bench vanillastark_benchmark policy_proofs
```

## What To Record

For each benchmark fixture, capture:

- proof time
- verify time
- proof size
- direct register count or VM trace width
- FRI domain length

For direct proofs, also capture the structural estimate from:

- `estimate_cdmn_decision_stark_direct_cost`
- `estimate_compiled_cdmn_decision_stark_direct_cost`

## Current Interpretation

If direct proofs are still slow while `trace_rows` remains tiny, the bottleneck is probably AIR width and transition-constraint size, not trace length.
That is the expected profile for the current first direct AIR slice.
