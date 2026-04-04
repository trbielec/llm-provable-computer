# STARK Performance Roadmap

## Goal

Reduce proving cost for real DMN/cDMN decisions by an order of magnitude or more.

This roadmap is not about squeezing small wins out of the current reference prover.
It is about changing the proof architecture so richer decision programs stop scaling badly.

## Current Diagnosis

The current vanilla STARK path is expensive for richer cDMN programs because:

- every memory cell becomes a trace column
- the full initial and final memory are public boundary data
- the AIR uses per-program-counter selector logic
- polynomial interpolation, evaluation, and division are mostly naive
- Merkle and FRI are implemented in a clear but not highly optimized style

This is a good research/reference prover, but not a good long-term proving architecture.

## Strategy

Pursue two tracks in sequence:

1. shrink the cost of the generic VM proof system
2. move high-value decision proofs onto a decision-specific proof path

The first track keeps the current system credible.
The second track is where the major step-function improvement comes from.

## Target State

The target architecture should look like:

- `frontend`
  - DMN/cDMN parsers
  - semantic analysis
  - typed decision IR
- `execution backends`
  - native VM
  - transformer-shaped runtime
  - optional ONNX/Burn
- `proof backends`
  - generic VM STARK
  - decision IR STARK

The proof boundary for decision use cases should eventually move from:

- "prove execution of the generic VM"

to:

- "prove execution of typed decision IR over committed inputs"

The generic VM proof should remain as the lower-level compatibility and research path.

## Principles

- Optimize architecture before micro-optimizing code.
- Keep semantics small and explicit.
- Avoid proving more than the application needs.
- Separate decision proving from general-purpose VM proving.
- Bind proofs to committed inputs and outputs, not raw in-memory machine state.

## Workstream 1: Measurement And Proof Cost Visibility

### Goal

Make proving cost measurable before changing architecture.

### Work

- Add Criterion benches for:
  - generic VM proofs on arithmetic programs
  - DMN proofs
  - cDMN proofs with quantifiers
  - cDMN proofs with definitions and ranges
- Record:
  - trace length
  - trace width
  - interpolation time
  - Merkle commit/open time
  - FRI time
  - proof size
  - verify time
- Run proof benchmarks in `--release`
- Add one benchmark dashboard document to the repo

### Exit Criteria

- You can explain where proving time goes for at least three representative programs
- You have a baseline for future speedup claims

## Workstream 2: Replace Naive Polynomial Operations

### Goal

Remove obvious superlinear bottlenecks from the current prover.

### Work

- Replace naive Lagrange interpolation with NTT/FFT-backed interpolation
- Replace point-by-point domain evaluation with NTT-backed evaluation where applicable
- Replace naive polynomial multiplication/division with faster routines
- Reuse evaluation domains and precomputed roots across proofs

### Why This Matters

This is the largest low-level bottleneck still inside the current implementation.
It will help both the generic VM prover and any later specialized prover.

### Expected Gain

- likely `2x` to `10x` on polynomial-heavy prover stages
- not enough by itself for the final target, but necessary groundwork

### Exit Criteria

- the current ignored cDMN proof regressions become materially faster in `--release`

## Workstream 3: Rebuild Merkle/FRI For Real Prover Throughput

### Goal

Stop paying avoidable costs in commitments and query handling.

### Work

- Implement iterative Merkle tree construction with cached internal nodes
- Stop rebuilding trees during `open`
- Parallelize leaf hashing and internal node construction
- Batch query path extraction
- Deduplicate repeated authentication work across FRI rounds where possible
- Review hash choice only after measuring implementation cost

### Expected Gain

- moderate speedup
- more important once polynomial costs are reduced

### Exit Criteria

- Merkle/FRI no longer dominates time on medium-sized decision proofs

## Workstream 4: Shrink The Generic VM AIR

### Goal

Reduce trace width and constraint cost in the generic VM proof path.

### Work

- Replace per-PC selector AIR with an opcode/decode AIR
  - prove the fetched instruction/opcode and operands
  - transition by opcode families rather than one selector per program counter
- Stop exposing full memory in public boundary constraints
  - commit to initial/final memory instead
  - expose only the application-relevant outputs publicly
- Replace full-memory columns with a sparse memory argument
  - memory bus / memory table approach
  - timestamped reads and writes
  - consistency constraint between execution trace and memory trace
- Consider splitting trace tables:
  - CPU trace
  - memory trace
  - auxiliary range/bit trace
- Revisit `ACC` bit-decomposition and flags so they live in narrower auxiliary traces

### Why This Matters

This is the first architectural change that can produce a real step-function improvement.
Today every extra memory cell and every longer compiled program hurts the prover too directly.

### Expected Gain

- potentially `5x` to `20x` depending on memory-heavy decision programs

### Exit Criteria

- generic VM proof cost scales with actual memory activity, not total memory size
- longer compiled cDMN programs do not blow up the AIR linearly in the same way

## Workstream 5: Introduce A Decision-Proof IR

### Goal

Stop proving decision logic as if it were arbitrary machine code.

### Work

- Define a typed proof IR for decision execution:
  - scalar arithmetic
  - booleans
  - comparisons
  - bounded collections
  - quantifiers
  - named definitions
  - reason outcomes
- Lower DMN/cDMN into this proof IR
- Keep this IR separate from TVM assembly
- Preserve explicit failure reasons and insufficient-data outcomes

### Why This Matters

The biggest proving win is to avoid proving the generic VM at all for decision applications.
If the application is structured decision logic, the proof system should know that.

### Expected Gain

- order-of-magnitude improvement potential
- smaller traces
- fewer irrelevant machine-state constraints
- clearer public claims

### Exit Criteria

- DMN/cDMN can compile to either:
  - TVM for execution compatibility
  - decision-proof IR for proof generation

## Workstream 6: Build A Decision-Specific STARK Backend

### Goal

Prove decision semantics directly.

### Work

- Design a specialized AIR for decision IR:
  - expression evaluation rows
  - quantifier iteration rows
  - bounded collection lookup rows
  - reason/status output rows
- Use a compact memory model:
  - committed structured inputs
  - collection tables
  - output registers
- Make public claims application-level:
  - policy version hash
  - input commitment
  - output status
  - reason code
  - optional explanation commitment
- Keep the native interpreter as the semantic oracle during bring-up

### Why This Matters

This is the clearest path to making healthcare and compliance proofs practical.
A medication policy or trial screen should not carry the proving cost of a general-purpose VM.

### Expected Gain

- potentially `10x` to `100x` relative to proving the same logic through the generic VM

### Exit Criteria

- one representative cDMN policy proves significantly faster through the decision backend than through the VM backend

## Workstream 7: Commitments, Inputs, And Public Claims

### Goal

Make proofs smaller and more deployable.

### Work

- Replace raw public initial/final memory with commitments
- Bind the proof to:
  - policy hash
  - structured input commitment
  - result status
  - reason code
  - optional trace/explanation commitment
- Keep full machine state private unless debugging
- Define a stable proof artifact schema for downstream systems

### Why This Matters

Public-memory-heavy claims are expensive and not the right abstraction for real applications.

### Exit Criteria

- proof claims are application-facing, not machine-facing

## Workstream 8: Backend Abstraction And Migration

### Goal

Make the prover replaceable without breaking the language/runtime stack.

### Work

- Introduce a proof backend trait
- Keep the current vanilla STARK as the reference backend
- Add room for a future optimized backend
  - either an in-repo optimized STARK
  - or integration with a mature proof library
- Keep proof generation APIs stable at the application layer

### Why This Matters

Even after architectural cleanup, a production-grade prover may still be best served by a more mature backend implementation.

### Exit Criteria

- the proof system is no longer tightly coupled to the current vanilla implementation

## Recommended Phase Order

### Phase 0

- Workstream 1

### Phase 1

- Workstream 2
- Workstream 3

This should make the current prover less painful and give cleaner measurements.

### Phase 2

- Workstream 4

This is the serious generic-VM architecture pass.

### Phase 3

- Workstream 5
- Workstream 7

At this point the repo has the semantic foundation for application-level proving.

### Phase 4

- Workstream 6

This is the order-of-magnitude step for DMN/cDMN applications.

### Phase 5

- Workstream 8

Do this once the proof architecture is stable enough to support backend substitution.

## Concrete First Milestone

If the goal is to start now on the major path without boiling the ocean, do this first:

1. add release-mode benchmarks for:
   - medication cDMN proof
   - lab/vitals cDMN proof
   - one DMN proof
2. move polynomial interpolation/evaluation onto the existing NTT machinery
3. measure again
4. then begin the generic VM AIR redesign with sparse memory

That sequence gives immediate clarity and prevents redesigning blind.

## Decision Point

After Workstream 4, make an explicit choice:

- if the generic VM path becomes fast enough, keep it as the main prover
- if not, invest fully in the decision-specific AIR and treat the VM prover as a fallback/research path

My expectation is that healthcare/compliance/eligibility use cases will still want the decision-specific prover.

## Risks

- Optimizing the current prover too long delays the real architectural fix.
- A decision-specific AIR can drift from the native semantics unless differential testing is strong.
- Sparse memory arguments add substantial proof-system complexity.
- Public-claim redesign will affect example flows, artifact formats, and downstream tooling.

## Success Criteria

This roadmap succeeds if, for representative DMN/cDMN policies:

- prove time drops by at least `10x`
- proof size is smaller or at least not materially worse
- verify time remains practical
- proofs bind to policy/input/output semantics rather than raw VM memory
- richer cDMN policies no longer need to be ignored in routine proof regression workflows

## Bottom Line

If you want order-of-magnitude improvement, the right move is:

- stop proving so much machine detail
- shrink the generic AIR
- then prove decision semantics directly

That is the path from a strong research prototype to a practical verifiable decision system.
