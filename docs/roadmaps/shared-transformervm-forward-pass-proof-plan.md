# Shared `TransformerVm` Forward-Pass Proof Plan

## Goal

Add a proof path that can honestly support the claim:

> for a given `Program` and `TransformerVmConfig`, the shared `TransformerVm` in `src/model.rs` executed the program correctly, and the proved execution is the shared transformer forward pass rather than a separate semantic surrogate.

This roadmap is about the shared runtime in:

- `src/model.rs`
- `src/runtime.rs`
- `src/memory.rs`
- `src/state.rs`

It is intentionally distinct from:

- the classic native-VM STARK in `src/proof.rs`
- the reduced local Poet path in `src/poet_proof.rs`
- the direct DMN / cDMN proof paths under `src/cdmn/`

## Current State

Today the proof boundary is:

- `src/proof.rs`: prove native VM execution
- `src/verification.rs`: cross-check shared `TransformerVm` against native execution
- `src/poet_proof.rs`: directly prove a reduced local transformer-style profile

The shared `TransformerVm` path in `src/model.rs` is not directly proved.

The relevant shared execution stages are:

1. `encode_state(...)`
2. `decode_state(...)`
3. `dispatch_info(...)`
4. `Attention2D::gather(...)`
5. `GatedFeedForward::apply(...)`
6. transition materialization
7. memory write application
8. output-state encode/decode round-trip

## Final Claim

The target end-state claim should be:

1. The same `InstructionCompiler` and `TransformerVm::new(...)` used by normal execution are the ones being attested.
2. The same `TransformerVm::step(...)` semantics in `src/model.rs` are what the proof covers.
3. The proof binds:
   - `Program`
   - `TransformerVmConfig`
   - initial machine state / memory
   - execution length
   - final machine state / memory
4. A verifier does not need to trust:
   - a separate reduced-profile compiler
   - a separate interpreter
   - a hand-maintained shadow semantics

## Non-Goals

This roadmap is not:

- "make the existing native VM proof faster"
- "replace direct decision proofs for DMN / cDMN"
- "prove every possible backend at once"

It also does not treat softmax-style attention as a day-one requirement. Exact proof of the full shared runtime under `softmax` / `hard-softmax` is materially harder than proving the default deterministic `average-hard` path.

## The Main Technical Obstacles

### 1. The shared runtime is not a tiny AIR target

The shared `TransformerVm` path includes:

- state encoding / decoding
- attention-backed memory reads
- sparse compiled FF evaluation
- boolean thresholding
- transition validation
- memory mutation

This is larger than the current native VM AIR in `src/proof.rs`.

### 2. The shared runtime currently uses `f64`

`src/model.rs` uses `f64`-valued matrices and vectors for the forward pass.

That creates a design choice:

- either prove exact IEEE-754 behavior,
- or prove an exact mathematical subdomain and refactor the shared runtime so it uses that same exact domain.

The second option is far more realistic and should be the default plan.

### 3. Attention / memory is semantically richer than a plain register VM

The forward pass reads memory through `Attention2D::gather(...)`, which ultimately calls `memory.load_with_mode(...)`.

The proof needs to cover the shared memory-read semantics, not just "some read happened."

### 4. The current Poet path is intentionally narrower

`src/poet_proof.rs` already proves a direct transformer-style path, but it:

- uses a reduced profile
- does not prove the full shared runtime
- does not use the shared `TransformerVm` compiler and state layout

It is a useful prototype, not the final solution.

## High-Level Architecture Choice

Build a **shared-forward-pass proof path**, not another shadow runtime.

That means:

1. Refactor the shared runtime into traceable kernels.
2. Emit a canonical execution witness from the shared runtime itself.
3. Prove that witness directly.
4. Only after that, pull compilation into the proof boundary.

The plan should avoid semantic duplication as much as possible.

## Hard Architecture Rules

The following are not optimization ideas. They are mandatory design constraints for the shared-forward-pass prover.

### Rule 1: Use a staged AIR from the start

Do not attempt a final design where one transformer step is proved in a single very wide row.

Reason:

- even reduced-profile forward-pass proving already becomes sluggish when FF internals are symbolically flattened
- the full shared runtime uses `WORD_BITS = 64`
- the shared token and FF widths are too large for a practical one-row design

Implication:

- every proof design in this roadmap assumes a staged row schedule
- "wide one-row first, optimize later" is explicitly out of scope

### Rule 2: Prove an exact discrete kernel, not generic `f64`

Do not define the trusted semantics of the shared proof path as raw IEEE-754 `f64` execution.

Reason:

- the compiled transformer kernels are already structured enough to live in an exact domain
- snapping `f64` outputs into field elements only remains safe if the shared runtime is intentionally constrained to that exact subdomain
- proving full floating-point semantics would add the wrong complexity to this repo

Implication:

- v1 must formalize a discrete transformer kernel for the shared runtime
- the shared runtime and the proof system must speak the same mathematical language

### Rule 3: Witnesses must come from the shared runtime

Do not build the final proof path by re-implementing the shared FF / attention math in a proof-only module.

Reason:

- that creates semantic drift risk
- the code being executed must be the code being proved

Implication:

- witness emission must be extracted from the shared `TransformerVm` kernels in `src/model.rs`
- proof-only recomputation is acceptable only for checking a shared witness, not for defining the semantics

### Rule 4: Coexistence is the intended end-state

Do not plan around replacing every other proof path.

The expected steady state is:

- native VM proof for compact semantic auditability
- direct DMN / cDMN proof for high-level policy logic
- shared transformer proof for exact forward-pass attestation

That is a feature, not a transitional compromise.

## Roadmap

## Milestone 0: Freeze The Claim

Define the exact v1 claim before writing more prover code.

### v1 claim

Prove the full shared `TransformerVm` execution for:

- full shared state width (`Word = i64`, `WORD_BITS = 64`)
- the shared `InstructionCompiler`
- the shared `TransformerVm::step(...)`
- `Attention2DMode::AverageHard`
- the currently supported shared ISA

### v2 claim

Extend the proof to:

- `hard-softmax`
- `softmax`
- any remaining shared-runtime-only behaviors

### Deliverables

- short architecture note defining v1 and v2 claim boundaries
- a test matrix listing which configs and instructions are in scope for v1

## Milestone 1: Refactor The Shared Runtime Into Traceable Kernels

Do not start by writing AIR over the current monolithic `TransformerVm::step(...)`.

Instead, factor the shared runtime into explicit, reusable kernels:

1. state codec kernel
   - `encode_state`
   - `decode_state`
2. dispatch kernel
   - `dispatch_info`
   - layer lookup / bank lookup
3. attention-read kernel
   - `Attention2D::gather`
4. FF kernel
   - `FeedForwardWeights::evaluate`
   - `GatedFeedForward::apply`
5. transition kernel
   - flag blending
   - memory write derivation
   - next-state validation

### Required code changes

- extract trace structs from `src/model.rs`
- add a shared witness-emission path, for example:
  - `SharedTransformerStepWitness`
  - `SharedTransformerExecutionWitness`
- make witness emission call the real shared kernels, not a proof-only copy

### Success criteria

- `TransformerVm::step(...)` and the witness builder produce exactly the same observable result
- witness generation adds no semantic drift
- lockstep tests still pass unchanged

## Milestone 2: Normalize The Arithmetic Domain

This is not a tuning choice. It is the mathematical foundation of the shared-forward-pass proof.

### Required direction

Refactor the shared transformer path so the proved arithmetic is exact and discrete.

Concretely:

- keep the public behavior of the shared runtime the same
- replace "generic `f64` semantics" as the trusted mathematical definition
- define an exact numeric domain for the compiled transformer kernels

For v1, this should be:

- integer / boolean-valued tokens
- integer-valued sparse weights and biases
- exact threshold semantics at `0.5`
- exact two's-complement bit decomposition

### Why this is mandatory

Proving exact IEEE-754 arithmetic for the whole forward pass is a much larger problem than this repo needs.

### Deliverables

- written invariant: every shared compiled forward-pass value in v1 is exact in the chosen domain
- tests showing the shared runtime stays inside that exact domain for supported configs

## Milestone 3: Add A Canonical Shared Forward-Pass Trace

Define the trace the proof will consume.

A single VM step should expose at least:

1. input machine state
2. input token
3. decoded input token
4. dispatch decision
5. attention memory-read witness
6. FF hidden activations
7. FF output vector
8. materialized transition
9. post-write machine state
10. output token
11. decoded output token

### Important rule

This trace is the proof witness for the shared runtime, not for a separate semantics.

### Deliverables

- trace schema
- trace builder
- trace round-trip tests

## Milestone 4: Prove A Single Shared `TransformerVm::step(...)`

Before proving full executions, prove one shared step exactly.

### Public claim

- `Program`
- `TransformerVmConfig`
- pre-state
- pre-memory
- post-state
- post-memory

### Witness obligations

- input token equals `encode_state(pre-state, d_model)`
- decoded input equals the shared decode semantics
- dispatch selects the correct compiled instruction / block
- attention read matches the shared `load_with_mode(...)` semantics
- FF equations match the shared compiled weights
- transition fields match the shared blending logic
- memory write is applied correctly
- output token equals `encode_state(post-state, d_model)`
- decoded output matches the shared decode semantics

### Why start here

This catches nearly all semantic drift before the complexity of multi-step execution and memory histories is added.

## Milestone 5: Replace Wide One-Row Proofs With A Staged AIR

This milestone is the first concrete AIR design, not an optional optimization pass.

Do not try to prove the entire shared step in one very wide row.

Use a staged row schedule such as:

1. state / token row
2. dispatch / attention row
3. FF chunk rows
4. output / transition row
5. writeback / decode row

### Why this is mandatory

The shared runtime has:

- 64-bit word decomposition
- large token width
- nontrivial FF width
- per-step memory semantics

A staged AIR is the only realistic path to acceptable prover costs.

### Deliverables

- row schedule
- register allocation strategy
- stage-local constraints
- stage transition constraints

## Milestone 6: Prove Shared Attention / Memory Semantics

For v1, focus on the exact semantics of `average-hard`.

### Required claim

The proved memory read must match the same value the shared runtime would obtain from:

- address selection
- memory history
- `load_with_mode(..., AverageHard { .. })`

### Recommended proving strategy

Prove the **semantic result** of the shared attention-backed memory read, not every Rust implementation detail of the hull cache.

That means:

- prove the selected read value is the correct `average-hard` result for the shared memory history
- treat the hull cache as an optimization of that semantics

### Later extension

If needed, add a second phase that proves the hull-cache maintenance algorithm itself.

## Milestone 7: Lift Single-Step Proofs To Full Execution Proofs

Once a single step is directly proved, chain steps into a full execution proof.

### Public claim

- `Program`
- `TransformerVmConfig`
- initial state / memory
- final state / memory
- step count

### Required checks

- each step consumes the previous step's post-state
- halted states remain halted
- memory history evolves correctly across steps
- final claim matches the shared `TransformerVm` result

### Success criteria

- the new proof can replace "native STARK + cross-check" for supported transformer examples

## Milestone 8: Pull Compilation Into The Proof Boundary

This is where the claim becomes fully end-to-end.

Right now, even a direct shared-forward-pass proof would still trust off-circuit compilation unless compilation is also attested.

### Required scope

Prove that `TransformerVm::new(config, program)` produces the same:

- `layer_for_pc`
- instruction banks
- compiled instruction weights
- block layout

### Recommended approach

Do this in two stages:

1. **v1.5 compiler commitment**
   - public claim includes a commitment to compiled weights / banks
   - verifier recomputes or spot-checks those weights off-circuit

2. **v2 compiler proof**
   - STARK or auxiliary proof covers compilation itself

### Why stage it

Compilation correctness is a distinct proof problem from forward-pass correctness.

## Milestone 9: Reach Full Shared-ISA Coverage

The final shared-forward-pass claim is incomplete until it covers the same instruction surface as the shared runtime.

Priority order:

1. arithmetic and control flow already covered by the shared model
2. compare / carry-sensitive paths
3. bitwise instructions
4. sign-branch instructions

### Requirement

The coverage target for this milestone is:

- if the shared `TransformerVm` accepts an instruction, the shared-forward-pass proof path should either:
  - prove it, or
  - reject it with an explicit documented unsupported-proof error

## Milestone 10: Add `hard-softmax` And `softmax`

This is a separate research milestone, not part of the v1 default claim.

### Why it is separate

These modes pull the proof path toward:

- non-discrete arithmetic
- interpolation / temperature semantics
- potentially exact exponential or fixed-point approximations

### Recommended order

1. make `average-hard` fully correct and fast
2. define exact semantics for `hard-softmax`
3. only then decide whether exact `softmax` proving is worth the complexity

## Milestone 11: Integrate And Cut Over

Add a new public proof API for the shared transformer path, for example:

- `prove_transformer_execution_stark(...)`
- `verify_transformer_execution_stark(...)`

Add:

- benchmark coverage
- failure-mode tests
- README and proof-boundary updates

### Cutover rule

Do not replace the native VM proof path until the shared-forward-pass prover has:

- parity on core examples
- stable multi-step proofs
- clear unsupported-mode / unsupported-instruction behavior

The likely end-state is not replacement but coexistence:

- native VM proof for small semantic core and auditability
- direct decision proof for DMN / cDMN workloads
- shared transformer proof for exact forward-pass attestation

## Testing Strategy

Each milestone should add three kinds of tests:

1. **parity tests**
   - shared runtime result equals traced/proved result
2. **negative tests**
   - malformed claims fail verification
3. **differential tests**
   - shared transformer proof path and native VM proof path agree on supported programs

Benchmark suites should separately measure:

- trace construction
- AIR width / degree
- proof time
- verification time

## Recommended Module Layout

The likely code shape is:

- `src/transformer_trace.rs`
- `src/transformer_proof_ir.rs`
- `src/transformer_stark.rs`

with `src/model.rs` retaining the runtime and exporting the shared trace hooks.

Do not continue expanding `src/poet_proof.rs` toward this goal. That module should remain a separate, explicit experiment unless and until its internals are replaced by shared kernels.

## Immediate Next Step

The first implementation step should be:

1. freeze the v1 claim as "full shared `TransformerVm`, `average-hard`, full shared state width"
2. add a canonical `SharedTransformerStepWitness` emitted from the real `TransformerVm::step(...)`
3. add parity tests that compare:
   - normal shared execution
   - witness-derived reconstruction
   - native interpreter

That step is small enough to implement safely and large enough to prevent the proof effort from drifting into another reduced-profile shadow runtime.
