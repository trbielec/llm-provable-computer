# Current Roadmap

## Title

Hybrid proposal lane over the provable TVM microkernel

## Status

- Date: April 6, 2026
- Phase: Phase 2 in progress (external state and effect roots)
- Primary issue: `#29`
- Supporting context: issue `#15` and the supplemental source-map comment on `#29`

## Progress log

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: freeze and isolate | **complete** | Branch `issue-29-investigation` pushed; baselines in `docs/baselines/`; `proof-boundary.md` reconciled; `JS`/`JNS` added to supported lists; roadmap benchmark command corrected |
| Phase 1: `ProposalReceipt` | **complete** | `src/proposal_receipt.rs`; 10 integration tests in `tests/proposal_receipt.rs`; all suites green |
| Phase 2: external state and effect roots | not started | — |
| Phase 3: candidate-parallel evaluation | not started | — |
| Phase 4: planner integration | not started | — |
| Phase 5: serving-side overlap experiments | not started | — |

## Product statement

The next major direction is to add a probabilistic proposal layer, optional external-state receipts, and candidate-parallel evaluation on top of the existing deterministic `TransformerVm` / native-VM proof stack without weakening the claims the repository already makes today.

This work is explicitly a two-way door. The default execution, proof, and CLI behavior must remain stable while the new architecture is investigated behind additive seams.

## Why this exists

The repository already has three strong assets:

1. a small deterministic execution core,
2. a clear proof boundary,
3. scalable batch execution and an implemented commit-and-audit path.

Issue `#29` exists because the repo can likely become more powerful if it combines:

- the current provable VM and shared-forward-pass work,
- a CommitLLM-style upstream receipt boundary,
- a Transformer-OS-style thin-host runtime boundary,
- and a transformer-vm / wasm-transformer style deterministic compiled-compute lane.

The key constraint is that this must not dilute the current claims or destabilize the current proof-oriented system.

## Goals

1. Preserve the current exact claim surface.
2. Add a typed, auditable boundary for probabilistic proposal generation.
3. Add candidate-parallel evaluation using the existing batch machinery.
4. Add an explicit effect/state boundary for external reads and writes.
5. Keep every new capability removable without invasive rollback work.

## Non-goals

1. Do not replace the current native ISA proof path.
2. Do not replace the current shared `TransformerVm` exact-domain proof path.
3. Do not fuse sampled LM internals into the existing proof-critical path.
4. Do not change default runtime semantics, CLI defaults, or benchmark claims during investigation.
5. Do not widen current proof claims in `README.md` until the new path is implemented, measured, and isolated.

## Architectural decision

The repository should evolve toward a layered design:

- `proposal lane`: probabilistic planner / upstream model emits typed proposals and a compact receipt,
- `kernel lane`: existing deterministic VM executes admissible proposals,
- `effect lane`: external state is represented through explicit committed roots and effect logs,
- `parallel lane`: multiple candidate programs or plans are evaluated in parallel,
- `proof lane`: current proof paths continue to attest deterministic execution only.

The existing deterministic runtime remains the kernel. The new work adds orchestration around it.

## Invariants that must remain true

1. `docs/architecture/proof-boundary.md` must remain accurate.
2. Existing proof paths must keep the same semantics unless a phase explicitly says otherwise.
3. Existing `cargo test` behavior must stay green.
4. Existing benchmark baselines must not regress materially without explanation.
5. New code must be behind additive modules, feature flags, or opt-in CLI paths.
6. No README claim may outrun the implemented evidence.

## Two-way-door rules

This roadmap is only valid if the following rules are kept:

1. Use a clean worktree from `main` for the investigation.
2. Keep the current dirty development branch out of the investigation path.
3. Make every phase mergeable and removable independently.
4. Add new modules before editing old core modules.
5. Default behavior must remain unchanged until the final integration phase.
6. Rollback must be possible by reverting a small number of additive commits, not by reconstructing lost semantics.

## Branch and worktree plan

### Base state

Before starting the roadmap implementation:

- local `main` must include `issue-14`,
- local `main` must include `feat/stark-api-carry-bitwise`,
- and the investigation must not be performed in a dirty worktree.

### Operating model

1. Keep the user's current development worktree untouched.
2. Create a clean worktree from `main`.
3. Create a dedicated investigation branch from that clean `main`.

Recommended workflow:

```bash
git fetch origin
git worktree add /tmp/llm-provable-computer-main main
cd /tmp/llm-provable-computer-main
git switch -c issue-29-investigation
```

If a workspace-local worktree is preferred instead of `/tmp`, the same isolation rule applies.

## Required baseline freeze before feature work

The following must be captured before implementation begins:

### Regression baseline

```bash
cargo test
cargo test --all-features
```

If `metal-batch` is available on the target machine:

```bash
cargo test --features metal-batch
```

### Proof baseline

```bash
cargo test --release --test transformer_stark -- --include-ignored
```

### Benchmark baseline

```bash
cargo bench --bench commit_audit_benchmark
cargo bench --bench batch_benchmark
cargo bench --bench vanillastark_benchmark -- stark
```

### Documentation baseline

Before code work begins, confirm that these remain accurate:

- `README.md`
- `CLAUDE.md`
- `docs/architecture/proof-boundary.md`

The baseline results should be saved in a dedicated note before Phase 1 changes begin.

## Prep work required before Phase 1

1. Write down the current proof and trust boundary in one place.
2. Record the benchmark baselines listed above.
3. Define rollback criteria up front.
4. Create an additive namespace for the new work.
5. Decide the initial opt-in surface:
   - feature flag,
   - CLI subcommand,
   - or isolated library module.

## Rollback criteria

The investigation must stop or roll back if any of the following becomes true:

1. The current proof boundary becomes harder to explain.
2. `cargo test` regresses and the failure is not tightly scoped.
3. The default execution path changes semantics.
4. The new receipt or effect design requires invasive edits across the current core.
5. The candidate-parallel path cannot remain opt-in.
6. Benchmark regressions exceed acceptable bounds without a compensating architectural benefit.

## Implementation phases

## Phase 0: freeze and isolate — COMPLETE

Deliverables:

- clean investigation worktree and branch, ✓
- baseline test report, ✓ (`docs/baselines/phase0-test-baseline.md`)
- baseline benchmark report, ✓ (`docs/baselines/phase0-benchmark-baseline.md`)
- this roadmap checked into the repo and referenced from top-level docs. ✓

Exit criteria:

- clean branch exists, ✓ (`issue-29-investigation` on `origin`)
- baseline commands are repeatable, ✓
- default claim language is still precise. ✓ (`proof-boundary.md` reconciled with source)

## Phase 1: `ProposalReceipt` — COMPLETE

Objective:

Add a CommitLLM-inspired typed receipt for upstream proposal generation without changing the current proof kernel.

Scope:

- define a typed `ProposalReceipt`,
- bind prompt/context/model/decode metadata,
- record output hash and optional attestation fields,
- keep verification independent from the deterministic execution proof.

Suggested placement:

- new module under `src/` or a new small crate under `crates/`

Do not:

- mix proposal verification with `src/proof.rs`,
- or treat proposal receipts as proof objects.

Exit criteria:

- proposal receipt can be created, serialized, hashed, and validated,
- no existing execution path changes behavior.

## Phase 2: external state and effect roots

Objective:

Add explicit effect-root plumbing for external reads/writes.

Scope:

- define committed state/effect roots,
- define typed external read/write events,
- bind those events into optimistic receipts or execution metadata,
- avoid implicit host callbacks.

Suggested direction:

- Merkleized effect/state boundary,
- explicit `READ_EXT` / `WRITE_EXT` style semantics at the orchestration layer,
- no raw filesystem cross-attention semantics in the current proof kernel.

Exit criteria:

- external state is explicit, typed, and hash-bound,
- existing deterministic VM semantics remain unchanged by default.

## Phase 3: candidate-parallel evaluation

Objective:

Use the existing batch machinery to evaluate multiple candidate proposals in parallel.

Scope:

- accept `k` candidates,
- run them through the deterministic kernel,
- score or filter them deterministically,
- preserve one selected candidate plus its receipt trail.

Why this phase matters:

This is the first phase that extracts practical value from the roadmap without requiring any proof-boundary rewrite.

Exit criteria:

- candidate-parallel evaluation works on CPU batch first,
- no regression to the current single-program path,
- benchmark deltas are captured.

## Phase 4: planner integration

Objective:

Attach a probabilistic planner to the deterministic kernel through the new receipt boundary.

Scope:

- planner produces typed proposals,
- deterministic kernel accepts only admissible proposals,
- result selection and final execution remain deterministic.

Do not:

- allow free-form planner output to directly mutate kernel state,
- or claim proof coverage for planner internals.

Exit criteria:

- planner integration is fully opt-in,
- the default repo path still runs with no planner attached.

## Phase 5: optional serving-side overlap experiments

Objective:

Experiment with hiding deterministic compute latency behind proposal/prefix work.

Scope:

- measure overlap opportunities,
- keep this at the serving/orchestration layer first,
- avoid baking sampled computation into the exact proof-critical path.

Exit criteria:

- performance gains are measurable,
- the logical trust boundary stays explicit.

## Required regression coverage

The following surfaces must remain healthy throughout the investigation:

### Existing core validation

- `cargo test`
- `cargo test --all-features`

### Existing proof validation

- shared transformer proof regressions,
- commit-and-audit tests,
- direct decision proof regressions,
- interpreter / runtime differential checks.

### Existing batch validation

- heterogeneous batch coverage,
- batch sweep coverage,
- Metal batch coverage where available.

### New coverage to add

1. proposal receipt serialization and hashing,
2. proposal receipt tamper detection,
3. effect-root tamper detection,
4. candidate-parallel determinism,
5. no-regression tests for default CLI behavior.

## Required benchmarking

At minimum, maintain a comparable before/after record for:

1. commit latency,
2. audit generation latency,
3. audit verification latency,
4. batch-run throughput,
5. proof generation and verification on representative policy fixtures.

If planner integration is added, add:

6. candidate fan-out cost,
7. planner-plus-kernel end-to-end latency,
8. overlap vs non-overlap serving latency.

## Required documentation updates per phase

Every phase must update, when applicable:

- `README.md`
- `CLAUDE.md`
- `docs/architecture/proof-boundary.md`
- the benchmark note or report for that phase

No phase is complete if the docs are stale.

## Acceptance criteria for the roadmap as a whole

This roadmap succeeds only if all of the following are true:

1. the current proof claims remain true,
2. the new proposal/effect/batch layers are opt-in,
3. rollback remains cheap,
4. the benchmark story remains defensible,
5. the trust boundary becomes clearer, not murkier,
6. the repo becomes more capable without becoming less auditable.

## Immediate next steps

1. Create the clean investigation branch from merged `main`.
2. Capture the baseline test and benchmark report.
3. Implement `ProposalReceipt` as the first additive unit.
4. Do not edit the existing proof kernel until Phases 1 through 3 are stable.

## Related references

- issue `#15`
- issue `#29`
- `docs/architecture/proof-boundary.md`
- `docs/roadmaps/shared-transformervm-forward-pass-proof-plan.md`
