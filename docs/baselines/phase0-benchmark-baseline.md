# Phase 0 Benchmark Baseline

- **Date:** 2026-04-06
- **Branch:** issue-29-investigation
- **Worktree:** /tmp/llm-provable-computer-main

---

## commit_audit_benchmark

```
cargo bench --bench commit_audit_benchmark
```

| Benchmark | Time (median) | Range |
|-----------|---------------|-------|
| commit_and_audit/commit | 1.2802 ms | [1.2791 ms – 1.2815 ms] |
| commit_and_audit/audit_generate | 332.58 µs | [332.25 µs – 333.20 µs] |
| commit_and_audit/audit_verify | 905.61 µs | [904.05 µs – 906.83 µs] |

Status: **PASS**

---

## batch_benchmark

```
cargo bench --bench batch_benchmark
```

| Benchmark | Time (median) | Range |
|-----------|---------------|-------|
| batch_runner/64 | 338.88 µs | [337.78 µs – 340.10 µs] |
| batch_runner/256 | 1.0936 ms | [1.0905 ms – 1.0966 ms] |
| batch_runner/1024 | 3.9942 ms | [3.9819 ms – 4.0143 ms] |
| batch_runner/4096 | 15.306 ms | [15.231 ms – 15.347 ms] |

Status: **PASS**

---

## vanillastark_benchmark

```
cargo bench --bench vanillastark_benchmark -- stark
```

> **Note:** The roadmap specified `-- policy_proofs` but no benchmark with that name exists.
> The correct filter for the STARK proof benchmarks is `stark`.
> See ROADMAP_ISSUE below.

| Benchmark | Time (median) | Range |
|-----------|---------------|-------|
| stark_prove | 12.071 ms | [12.023 ms – 12.121 ms] |
| stark_verify | 824.38 µs | [819.28 µs – 830.68 µs] |
| stark_e2e | 12.878 ms | [12.822 ms – 12.928 ms] |
| stark_proof_size | 11.961 ms | [11.932 ms – 11.990 ms] |

Status: **PASS** (the four STARK benchmarks complete successfully)

> **ROADMAP_ISSUE:** Both `-- policy_proofs` and `-- stark` cause Criterion to panic at the end with
> `assertion failed: n >= 10`. This is a Criterion 0.5.1 edge case: the filter also matches
> a `scaling` group with a single entry, which falls below Criterion's minimum sample count.
> Recommended fix: update the roadmap command to use `-- stark_` or an exact name like
> `-- stark_prove` to avoid matching the scaling group.

---

## transformer_stark proof tests (release)

```
cargo test --release --test transformer_stark
```

| Result | Count |
|--------|-------|
| passed | 7 |
| failed | 0 |
| ignored | 7 |

The 7 ignored tests carry explicit `#[ignore]` annotations and are gated to CI
(too slow for local execution). They were **not** run as part of this baseline.
Running `--include-ignored` locally is not recommended — those tests run for 20+ minutes.

Status: **PASS**

---

## Overall status: PASS

All benchmarks produced valid numbers. One roadmap command needs correction
(see `ROADMAP_ISSUE` above).
