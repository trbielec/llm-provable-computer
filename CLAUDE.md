<identity>
llm-provable-computer is an implemented Rust workspace for a deterministic transformer-shaped virtual machine and an in-tree vanilla STARK prover over its execution trace. Milestone 1 and Milestone 2 are complete. The issue-29 proposal-lane investigation is also complete, adding an optional orchestration layer (ProposalReceipt, EffectBoundary, CandidateEvaluator, PlannerBridge, OverlapOrchestrator) on top of the existing deterministic kernel without changing any existing execution path. Milestone 3 (STWO integration) is the next major target.
</identity>

<roadmap>
Current roadmap: `docs/roadmaps/current-roadmap.md`
</roadmap>

<stack>
| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| Language | Rust | 2021 | Cargo workspace is checked in |
| Package manager | Cargo | current manifest | `Cargo.toml` and `Cargo.lock` are present |
| CLI | clap | 4.5.38 | `tvm` binary in `src/bin/tvm.rs` |
| Terminal UI | ratatui, crossterm | 0.29.0 / 0.28.1 | Interactive execution viewer |
| Proof stack | In-repo vanilla STARK | local modules | `src/proof.rs` plus `src/vanillastark/` |
| Optional tensor runtime | Burn | 0.20.1 | Gated by `burn-model` |
| Optional export/runtime | ONNX + Tract | 0.2.3 / 0.23.0-dev.2 | Gated by `onnx-export` |
| Optional GPU batch runner | metal | 0.29 | Gated by `metal-batch`, macOS/Apple Silicon only |
| Data-parallel CPU batch | rayon | 1 | Always available; used by `BatchRunner` |
| Serialization | serde, serde_json | 1.0.219 / 1.0.140 | Proof and metadata IO |
| Test tooling | `cargo test`, criterion, proptest | current manifest | Default and feature-gated suites exist |
</stack>

<status>
- Milestone 1: complete.
- Milestone 2: complete for the current proof scope.
- Verified locally on 2026-04-06:
  - `cargo fmt --all --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --all-features`
  - `cargo test --features metal-batch`
- Current vanilla STARK scope:
  - supported: `NOP`, `LOADI`, `LOAD`, `STORE`, `PUSH`, `POP`, `ADD`, `ADDM`, `SUB`, `SUBM`, `MUL`, `MULM`, `AND`, `ANDM`, `OR`, `ORM`, `XOR`, `XORM`, `CALL`, `RET`, `JMP`, `JZ`, `JNZ`, `JS`, `JNS`, `HALT`
  - carry flag: fully constrained; carry-bearing traces (overflow, subtraction borrow) are accepted and carry correctness is verified by the AIR (not a free witness)
  - rejected: softmax and hard-softmax proof paths, `CMP`/`CMPM` compare instructions, non-halted executions
- Commit-and-Audit: receipts bind `program_hash` and `initial_state_hash`; Freivalds algebraic check is fully implemented (not a stub)
- Batch execution: `BatchRunner` (CPU rayon) and `MetalBatchRunner` (Apple Silicon GPU, `metal-batch` feature) both available; heterogeneous batch merges multiple programs into a shared PC space via `Program::merge`
- Benchmarks: `commit_audit_benchmark` reports ~1.28ms commits, ~333µs audit generation, and ~906µs verification (release); `batch_benchmark` shows ~339µs for 64 runs, 1.09ms for 256, 3.99ms for 1,024, and 15.3ms for 4,096 CPU `BatchRunner` executions; `overlap_benchmark` shows ~120µs sequential and ~158µs overlapped for k=4 candidates with `StaticPlanner` (Criterion with 5s measurements).
- Current proof is transparent, not zero-knowledge: the public claim includes the program, attention mode, step count, and final state.
- Proposal pipeline (issue-29, complete): `ProposalReceipt` (typed hash-bound planner receipt), `EffectBoundary` (committed external state/effect log), `CandidateEvaluator` (k-parallel rayon execution + deterministic selection), `PlannerBridge` (Planner trait + AdmissibilityFilter + pipeline orchestration), `OverlapOrchestrator` (Sequential/Overlapped timing measurement). Programs are in .tvm assembly — no WASM. Only `StaticPlanner` is provided; real LLM integration requires a user-supplied `Planner` impl.
</status>

<structure>
Current repository:

```text
Cargo.toml
Cargo.lock
README.md
CLAUDE.md
src/
  assembly.rs           # .tvm parser, directives, labels
  batch.rs              # CPU rayon BatchRunner (data-parallel execution)
  compiler.rs           # ProgramCompiler
  config.rs             # TransformerVmConfig, Attention2DMode
  engine.rs             # Shared execution traits/results
  error.rs              # VmError and Result
  geometry.rs           # Point2D and HullKvCache
  instruction.rs        # Program + ISA
  interpreter.rs        # Native semantic oracle
  memory.rs             # AddressedMemory and write histories
  model.rs              # TransformerVm and compiled transition logic
  runtime.rs            # Transformer execution loop
  state.rs              # MachineState and token encoding
  proof.rs              # Execution-proof plumbing and AIR wiring (classic native-trace STARK)
  transformer_stark.rs  # Shared transformer STARK prover and verifier
  transformer_trace.rs  # Shared transformer trace builder and validator
  merkle_trace.rs       # Merkle tree commitment over transformer trace rows
  commit_and_audit.rs   # Commit-and-audit protocol (Merkle commit + Freivalds audit)
  approximate.rs        # Numerical precision helpers and audit numeric scale
  freivalds.rs          # Freivalds algebraic matrix-multiplication witness check
  poet_proof.rs         # Reduced local transformer-style forward-pass proof
  dmn.rs                # DMN parsing, IR, and rule compilation
  dmn_ir.rs             # DMN IR types (boolean and numeric rule expressions)
  dmn_backend_tvm.rs    # DMN→TVM compilation backend
  dmn_decision.rs       # Direct decision proof types and hash binding
  dmn_direct.rs         # Direct DMN STARK prover
  cdmn/                 # cDMN parser, schema, IR, proof backend, and STARK compat
  proposal_receipt.rs   # ProposalReceipt — typed hash-bound planner output receipt
  effect_root.rs        # EffectBoundary, EffectLog, StateRoot — committed external state
  candidate_eval.rs     # CandidateEvaluator — k-parallel rayon execution + deterministic selection
  planner_bridge.rs     # PlannerBridge, Planner trait, AdmissibilityFilter, StaticPlanner
  overlap.rs            # OverlapOrchestrator — Sequential/Overlapped latency measurement
  verification.rs       # Lockstep engine comparison
  tui.rs                # ratatui execution viewer
  vanillastark/         # Field, polynomial, Merkle, FRI, STARK internals
  metal_batch/          # Apple Silicon GPU batch runner (metal-batch feature)
    mod.rs              # MetalBatchRunner, stream compaction
    buffers.rs          # GpuWeightBuffers, GpuStepBuffers
    precision.rs        # Veltkamp split, GpuRunState, GpuControls
    kernels/            # MSL compute shaders
  burn_model.rs         # Optional Burn model
  burn_runtime.rs       # Optional Burn runtime
  onnx_export.rs        # Optional ONNX exporter
  onnx_runtime.rs       # Optional Tract runtime
  bin/tvm.rs            # CLI
tests/
examples/
benches/
programs/
scripts/
```

</structure>

<commands>
| Task | Command | Notes |
|------|---------|-------|
| Inventory files | `rg --files` | Fastest repo scan |
| Check repo state | `git status --short` | Separate your edits from unrelated files |
| Run default suite | `cargo test` | Core milestone-1 and milestone-2 validation |
| Check formatting | `cargo fmt --all --check` | Matches CI-ready rustfmt output |
| Run strict lint pass | `cargo clippy --all-targets --all-features -- -D warnings` | Keep the full tree warning-free |
| Run full engine suite | `cargo test --all-features` | Burn + ONNX + Python validator + CLI workflow |
| Run a program | `cargo run --bin tvm -- programs/fibonacci.tvm` | Shortcut for `tvm run` |
| Trace execution | `cargo run --bin tvm -- run programs/counter.tvm --trace` | Emits trace and summary |
| Verify transformer vs native | `cargo run --bin tvm -- run programs/fibonacci.tvm --verify-native` | Lockstep comparison |
| Verify all engines | `cargo run --features full --bin tvm -- run programs/fibonacci.tvm --verify-all` | Transformer + native + Burn + ONNX |
| Create a proof | `cargo run --bin tvm -- prove-stark programs/fibonacci.tvm -o /tmp/fib.proof.json` | Uses current vanilla STARK path |
| Verify a proof | `cargo run --bin tvm -- verify-stark /tmp/fib.proof.json` | Re-checks a saved proof |
| CPU batch run | `cargo run --bin tvm -- batch-run programs/fibonacci.tvm --count 65536` | N parallel runs via rayon |
| Heterogeneous batch | `cargo run --bin tvm -- batch-run prog_a.tvm prog_b.tvm --count 65536` | Multiple programs merged into one dispatch |
| Metal GPU batch run | `cargo run --features metal-batch --bin tvm -- batch-run programs/fibonacci.tvm --count 65536 --metal` | Apple Silicon GPU, macOS only |
| Run proposal pipeline example | `cargo run --example proposal_pipeline` | End-to-end: StaticPlanner → AdmissibilityFilter → CandidateEvaluator → OverlapOrchestrator |
| Review doc drift | `git diff -- README.md CLAUDE.md docs/` | Use before finishing doc/context work |
</commands>

<workflows>
  <doc_or_context_change>
  1. Read the relevant source file plus the corresponding code modules or tests.
  2. Prefer current API and test behavior over older RFC language.
  3. Keep implemented-vs-next boundaries explicit.
  4. Review `git diff` for top-level docs before finishing.
  </doc_or_context_change>

  <proof_change>
  1. Read `src/proof.rs` and `src/vanillastark/**`.
  2. Update support and limitation language in `README.md`, `CLAUDE.md`, and matching files under `docs/`.
  3. Re-run at least `cargo test` and, when Burn or ONNX paths are affected, `cargo test --all-features`.
  </proof_change>

  <engine_change>
  1. Read `src/runtime.rs`, `src/interpreter.rs`, and `src/verification.rs`.
  2. If CLI behavior changes, update `README.md` quick-start commands.
  3. If feature-gated runtimes change, update both `README.md` and this file's command/status notes.
  </engine_change>
</workflows>

<boundaries>
  <forbidden>
  - `.git/**`
  - Secret material (`.env*`, keys, tokens, certificates)
  - Destructive cleanup of user-created files without an explicit request
  </forbidden>

  <careful>
  - `CLAUDE.md`, top-level docs, and proof/status wording: keep them synchronized; do not make one-off edits.
  </careful>
</boundaries>

<troubleshooting>
  <known_issues>
  | Symptom | Likely cause | Fix |
  |---------|--------------|-----|
  | `prove-stark` rejects a program | Unsupported instruction or attention mode | Check `src/proof.rs::validate_proof_inputs`; only `CMP`/`CMPM` and non-`average-hard` modes are rejected; carry-bearing and bitwise traces are now accepted |
  | Burn or ONNX commands are unavailable | Missing feature flag | Re-run with `--features burn-model`, `--features onnx-export`, or `--features full` |
  | `--metal` flag not found on `batch-run` | Missing feature flag or non-macOS | Re-run with `--features metal-batch` on macOS/Apple Silicon |
  | Docs mention WASM compilation as current behavior | Stale pre-implementation text | WASM was never implemented; programs are in .tvm assembly only |
  | `PlannerBridge` pipeline has no LLM | `StaticPlanner` is a test double | A real LLM integration requires a user-supplied struct implementing the `Planner` trait in `src/planner_bridge.rs` |
  | An engine mismatch appears during verification | Trace divergence across runtimes | Inspect `ExecutionTraceEntry` output and compare instruction/state pairs |
  </known_issues>
</troubleshooting>
