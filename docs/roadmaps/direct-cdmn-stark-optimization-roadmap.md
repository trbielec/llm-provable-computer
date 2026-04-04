# Direct cDMN STARK Optimization Roadmap

## Goal

Turn the new direct cDMN STARK backend into the default high-value proof path for decision logic.

That means:

- materially faster than the VM-based compatibility proof path
- fast enough for routine regression use on representative policies
- broad enough to cover the cDMN semantics we already support
- structured so DMN can lower into the same backend later

## Current State

The repo now has a first direct cDMN STARK backend in [src/cdmn/stark_direct.rs](/Users/tombielecki/Documents/Code/llm-provable-computer/src/cdmn/stark_direct.rs).

What it already does:

- proves cDMN decisions directly from decision IR rather than through TVM execution
- binds proofs to projected inputs, `policy_commitment`, `input_commitment`, `status`, and `reason_code`
- supports the current arithmetic / boolean / derived-definition / range semantics
- passes the small smoke test and the lab-vitals direct proof regression

What is still weak:

- quantified policies are still expensive enough that the medication direct-proof test is ignored
- the AIR is “direct” but still too register-heavy
- range proofs and comparisons currently allocate many dedicated witness registers
- quantified logic is lowered by boolean unrolling, so proof width grows with policy size
- proof benchmarking is not yet split cleanly between direct-backend stages

This is a meaningful architectural step, but it is still a first direct AIR slice rather than the optimized target architecture.

## Diagnosis

The direct backend is better than the generic VM path because it removes machine-state and opcode overhead.
But it is not yet efficient enough because it still proves too much at once.

The current direct AIR pays for:

- one large registerized circuit containing nearly every intermediate value
- per-node status/value materialization
- full two's-complement bit decomposition on many arithmetic and comparison intermediates
- quantified logic compiled into wide boolean formulas rather than compact iteration traces
- limited witness reuse across repeated subexpressions and repeated constraint shapes

So the next performance gains will not mainly come from tuning FRI again.
They will come from shrinking the decision AIR itself.

## Strategy

Pursue four tracks in order:

1. measure the direct backend properly
2. reduce AIR width with staged evaluation and register reuse
3. replace quantified unrolling with compact iteration traces
4. move shared checks into reusable auxiliary tables / lookup-style arguments

Only after that should the team spend much time on micro-optimizing the direct prover implementation.

## Workstream 1: Direct-Proof Benchmarking

### Goal

Make the direct backend measurable on its own, separate from the VM compatibility path.

### Work

- Add Criterion benches for:
  - small threshold policy
  - definitions + `between`
  - quantified medication policy
  - quantified policy with insufficient-data failure
- Record:
  - circuit register count
  - transition constraint count
  - transition degree bound
  - proof time
  - verify time
  - proof size
- Add per-stage timing for:
  - witness construction
  - trace materialization
  - polynomial stages
  - Merkle/FRI stages
- Add one benchmark comparison table:
  - direct STARK
  - VM compatibility STARK

### Exit Criteria

- There is a repeatable release-mode benchmark report for at least three representative policies.
- It is obvious whether the bottleneck is width, degree, witness construction, or FRI.

## Workstream 2: AIR Width Reduction

### Goal

Stop materializing the whole decision circuit at once.

### Work

- Replace the current “all nodes live at once” layout with staged evaluation frames.
- Introduce a compact execution schedule:
  - load inputs
  - evaluate definitions
  - evaluate one constraint at a time
  - emit final decision state
- Reuse a fixed number of value/status slots instead of allocating one register pair per AST node.
- Reuse a fixed number of range/comparison aux slots instead of dedicating fresh bit lanes per occurrence.
- Separate always-public outputs from internal witness-only registers.

### Why This Matters

The current backend already benefits from only two trace rows, but width is still too large.
The first major direct-backend speedup will come from lowering register count, not row count.

### Expected Gain

- likely `2x` to `5x` on non-quantified policies
- more important as a prerequisite for quantified improvements

### Exit Criteria

- register count scales with maximum live intermediates, not total expression count
- the lab-vitals direct proof is materially faster than today

## Workstream 3: Quantifier-Specific Trace Design

### Goal

Stop proving quantified logic as large unrolled boolean formulas.

### Work

- Introduce explicit iteration rows for bounded collections:
  - collection count row
  - item row
  - predicate result row
  - aggregator state row
- Prove `forall` and `exists` as small state machines:
  - `forall`: accumulator starts `true`, fails on first counterexample
  - `exists`: accumulator starts `false`, succeeds on first witness
- Bind collection field reads to a compact collection table rather than a copied expression tree.
- Reuse the same predicate evaluation frame across collection items.
- Preserve current insufficient-data semantics during iteration:
  - count overflow
  - missing fields
  - unknown fields

### Why This Matters

This is the main reason the medication direct proof is still too expensive.
Quantifiers need their own execution shape.

### Expected Gain

- potentially `5x` to `20x` on quantified policies

### Exit Criteria

- the medication direct-proof regression can run in the default suite or in a near-default “proof-smoke” suite
- quantified proof cost scales mainly with item count, not with unrolled AST size

## Workstream 4: Comparison And Range Aux Tables

### Goal

Shrink repeated arithmetic/range logic in the AIR.

### Work

- Centralize signed comparison witnesses into shared auxiliary traces or lookup-style subarguments.
- Move repeated two's-complement bit decomposition into dedicated narrow auxiliary tables.
- Deduplicate status-range checks and booleanity checks through shared helper columns where possible.
- Consider a dedicated comparison gadget for:
  - equality / inequality
  - ordered compare
  - inclusive `between`

### Why This Matters

The direct backend currently spends too much width proving the same structural facts repeatedly.

### Expected Gain

- moderate speedup on all policies
- larger gain once width reduction and quantifier traces are in place

### Exit Criteria

- comparison-heavy policies add far fewer auxiliary registers per occurrence

## Workstream 5: Public Claims And Explanation Commitments

### Goal

Keep the direct backend application-oriented rather than machine-oriented.

### Work

- Keep the public claim limited to:
  - policy commitment
  - input commitment
  - final status
  - reason code
- Add optional explanation commitments for:
  - matched constraint id
  - failing constraint id
  - selected evidence references later
- Ensure the verifier can validate claims without reconstructing unnecessary internal layout details.

### Why This Matters

A faster backend is not enough if the proof claim drifts back toward machine internals.

### Exit Criteria

- the proof interface remains stable even if the internal AIR layout changes

## Workstream 6: DMN On The Same Direct Backend

### Goal

Make the direct proof backend a general decision substrate rather than a cDMN-only branch.

### Work

- Lower DMN decision-table semantics into the same proof-oriented execution model
- Support current core and non-core hit policies through a direct decision trace
- Preserve insufficient-data behavior and rule-trace outputs
- Share the same proof claim structure used by cDMN

### Why This Matters

The repo should end up with one decision-proof backend, not separate proof worlds for DMN and cDMN.

### Exit Criteria

- one representative DMN table proves through the direct backend
- the VM compatibility path is still available but no longer the preferred proof route for decision tables

## Workstream 7: Direct-Prover Engineering

### Goal

Exploit the low-level prover improvements once the AIR shape is worth optimizing.

### Work

- Add direct-backend-specific benchmark cases to the existing STARK benchmark suite
- Reuse the improved polynomial code paths already added to vanilla STARK
- Reuse cached Merkle/FRI structures where possible
- Profile whether direct proofs are now dominated by:
  - witness generation
  - interpolation/evaluation
  - Merkle/FRI
- Only then consider further prover-specific optimization work

### Why This Matters

Low-level prover work matters, but only after the AIR itself is compact enough.

### Exit Criteria

- direct-proof benchmarks show that prover-engineering work is paying off on the optimized AIR, not masking a bad circuit shape

## Milestones

### Milestone 0: Baseline

- add direct-proof benchmarks
- document current proof times and widths
- keep current lab direct proof in default tests
- keep quantified medication direct proof ignored

### Milestone 1: Width-Controlled Direct AIR

- introduce staged evaluation frames
- reduce live-register count substantially
- keep semantics unchanged

Success looks like:
- lab-vitals direct proof materially faster than current implementation

### Milestone 2: Quantifier Trace

- add compact collection iteration trace
- remove quantifier proof dependence on large unrolled boolean structure

Success looks like:
- medication direct proof no longer needs to be ignored

### Milestone 3: Shared Comparison / Range Gadgets

- move repeated signed-compare/range logic into shared aux traces
- cut repeated bit-decomposition overhead

Success looks like:
- comparison-heavy policies scale sublinearly in auxiliary width

### Milestone 4: DMN Direct Proof Path

- lower DMN onto the same direct backend
- keep VM proof as fallback / compatibility mode

Success looks like:
- direct proof becomes the preferred path for representative DMN and cDMN policies

## Recommended Immediate Next Steps

1. Add direct-backend benchmark instrumentation before more AIR changes.
2. Redesign the direct backend around staged evaluation and register reuse.
3. Build explicit quantifier iteration rows next.

That order matters.
Without measurement, optimization claims are weak.
Without width reduction, quantifier work will still sit on an oversized base.
Without quantifier-specific traces, the most important real policies will remain too expensive.

## Bottom Line

The direct cDMN backend is now architecturally real, but not yet operationally efficient enough.
The main problem is no longer “prove decision logic through a VM.”
The main problem is “the direct AIR still proves too wide a circuit.”

So the roadmap is:

- first measure the direct backend well
- then shrink the circuit width
- then give quantifiers their own compact trace model
- then unify DMN and cDMN on that optimized decision backend
