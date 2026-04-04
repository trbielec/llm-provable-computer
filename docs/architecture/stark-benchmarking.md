# STARK Benchmarking

## Purpose

This repository now has benchmark coverage for representative proof workloads, not just the Rescue Prime toy path.

The benchmark target is:

- `cargo bench --bench vanillastark_benchmark`

## Representative Policy Fixtures

The benchmark includes three policy-shaped proof fixtures:

- `dmn_loan_first_hit`
- `cdmn_medication_quantified`
- `cdmn_lab_ranges`

These are built from:

- [loan_rules.dmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/loan_rules.dmn.json)
- [medication_safety.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/medication_safety.cdmn.json)
- [lab_vitals_guardrail.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/lab_vitals_guardrail.cdmn.json)

## Metrics

The benchmark labels now include proof-shape metadata from `estimate_execution_stark_cost`:

- trace rows
- trace width
- FRI domain length

The benchmark measures:

- prove time
- verify time
- proof size

## Recommended Commands

Run the full benchmark target:

```bash
cargo bench --bench vanillastark_benchmark
```

Focus on the policy proofs only:

```bash
cargo bench --bench vanillastark_benchmark -- policy_proofs
```

## Notes

- Benchmarks should be interpreted in `--release`, which is what `cargo bench` uses.
- The current vanilla STARK prover is still a reference-style implementation.
- These benchmarks are intended to establish baselines before larger AIR/proof-boundary changes.
