# Roadmap: Healthcare-Oriented Verifiable CDS

## Goal

Turn the current prototype into a healthcare-oriented, proof-carrying decision system where:

1. a probabilistic model may extract or propose facts,
2. a deterministic DMN/cDMN-style verifier makes the bounded decision,
3. the system emits a machine-checkable proof and a clinician-readable explanation.

## Recommended Target Architecture

The best near-term healthcare architecture is:

1. **ML/LLM extraction layer**
   Convert notes, orders, prior auth requests, or care-plan text into structured facts.
2. **Rule/verifier layer**
   Evaluate those facts against explicit DMN/cDMN logic.
3. **Proof layer**
   Produce a STARK-attested execution artifact.
4. **Human review layer**
   Show the decision, reason codes, evidence used, and any model/rule disagreement.

In Kierner taxonomy terms, this should be built primarily as:

- `MLRB` now

with a possible later move toward:

- `PERML` if you want parallel model and rule recommendations plus an explicit aggregator

## Initial Clinical Positioning

Do not start with diagnosis or treatment recommendation.

Start with structured adjudication and compliance:

- prior authorization / medical necessity
- coverage policy checks
- trial eligibility screening
- contraindication and dosing guardrails
- guideline pathway conformance
- discharge or referral eligibility

These are rule-rich, auditable, and operationally valuable.

## Workstreams

## Workstream 1: Correctness and Claim Alignment

### Objective

Align the repo's public claims with what the code actually proves, stabilize core failures, and remove obvious representation limits before expanding domain scope.

### Tasks

- Fix the attention-mode `Display` / `FromStr` mismatch in `src/config.rs`.
- Fix the failing STARK loop proof path in `src/proof.rs` / `src/vanillastark`.
- Upgrade the TVM from `i16` to `i64` across:
  - machine state
  - interpreter/runtime semantics
  - proof trace encoding
  - AIR constraints
  - ONNX/Burn export paths
- Add numeric-width regression tests that cover:
  - realistic clinical lab values
  - larger financial quantities
  - negative intermediate results
- Update README wording so it distinguishes:
  - what the transformer executes
  - what the native interpreter proves
  - what cross-engine verification establishes
- Add a short architecture note documenting the current proof boundary.

### Exit Criteria

- `cargo test` passes cleanly
- the VM can represent realistic healthcare and financial magnitudes without `i16` overflow assumptions
- README claims are technically precise
- there is one short internal note describing the proof boundary and trust model

## Workstream 2: Decision Language Completeness

### Objective

Move DMN/cDMN from toy examples and hardcoded assembly templates to a semantically meaningful, parser-backed rule layer.

### Tasks

- Replace the example-only compilers with library modules under `src/`.
- Prioritize integrating a real logic parser such as `nom` rather than extending string-pattern assembly generation.
- Introduce an explicit AST for DMN/cDMN logic so parsing, normalization, compilation, and explanation are separate stages.
- Support real comparison semantics:
  - `>`
  - `>=`
  - `<`
  - `<=`
  - `!=`
  - ranges
- Support quantified constraints needed for richer cDMN audits:
  - `forall`
  - `exists`
- Define the collection/domain model those quantifiers operate over:
  - medications
  - diagnoses
  - lab panels
  - prior procedures
  - evidence items
- Add first-class branching support for comparison outcomes.
- Add explicit reason codes and rule-hit outputs.
- Add support for negative outcomes that halt normally and remain provable.
- Add a more faithful DMN subset:
  - inputs
  - outputs
  - ordered rule evaluation
  - default rules
  - hit policy handling
- Add a more faithful cDMN subset:
  - constraint identifiers
  - boolean combinations
  - quantification over collections
  - failure attribution
  - named intermediate expressions

### Likely Repo Changes

- `src/instruction.rs`
- `src/interpreter.rs`
- `src/model.rs`
- `src/proof.rs`
- new modules such as:
  - `src/dmn.rs`
  - `src/cdmn.rs`
  - `src/decision_compiler.rs`
  - `src/logic_parser.rs`
  - `src/logic_ast.rs`

### Exit Criteria

- no more equality-only placeholders for threshold logic
- no more hardcoded string-template logic compilation for core cDMN paths
- quantified constraints can be represented and compiled from parsed logic
- a verifier can prove both:
  - approval/compliance
  - rejection/non-compliance
- rule ID and reason code are included in the public claim or final machine state

## Workstream 3: Proof Architecture Hardening

### Objective

Make the proof system robust enough for real policy programs, not just small arithmetic examples.

### Tasks

- Stabilize control-flow support in the AIR for loops and richer branching.
- Expand proof support beyond the current narrow instruction subset where safe.
- Decide and document the target proof claim:
  - **Option A:** prove native VM semantics and cross-check transformer execution
  - **Option B:** directly prove the transformer trace
- Add proof support for externally supplied structured facts and evidence.
- Add a committed input model so the proof binds to:
  - the program
  - the structured inputs
  - the final decision
  - the rule trace
- Add explicit support for negative verdicts.

### Recommended Near-Term Choice

Use **Option A** first:

- prove native semantics
- keep transformer, Burn, and ONNX as cross-checked execution backends

That is more achievable than directly proving the transformer path today.

### Exit Criteria

- loop-heavy decision programs prove reliably
- proof verification is stable across shipped decision fixtures
- the proof artifact clearly binds to inputs and outcomes

## Workstream 4: Healthcare Data Model and Evidence Binding

### Objective

Create a healthcare-friendly input model that the verifier can consume safely.

### Tasks

- Define a structured patient-fact schema for decision inputs.
- Include support for:
  - missing values
  - unknown values
  - units
  - timestamps
  - source provenance
- Add evidence references so each structured fact can point back to:
  - note spans
  - forms
  - orders
  - lab values
  - coded records
- Support code-based facts where possible:
  - ICD / SNOMED / CPT / RxNorm placeholders or normalized internal IDs
- Add deterministic normalization and validation before verification.

### Example Input Types

- age
- diagnosis codes
- prior therapies tried
- contraindications present
- lab threshold values
- procedure requested
- referral reason
- medication list

### Exit Criteria

- structured decision inputs are versioned and validated
- each decision can produce a fact sheet plus proof
- each fact can optionally include provenance

## Workstream 5: LLM Integration as Fact Extraction, Not Final Decision

### Objective

Use LLMs where they are strong, but keep final decisions in the verifier.

### Tasks

- Define a structured extraction contract:
  - input document(s)
  - extracted fact schema
  - confidence / uncertainty
  - evidence spans
- Keep the LLM out of the final decision path initially.
- Add an evaluation harness comparing:
  - extracted facts
  - verified facts
  - human-reviewed gold labels
- Add disagreement handling:
  - missing evidence
  - conflicting evidence
  - unverifiable claims

### Recommended Pattern

- LLM proposes facts
- verifier decides
- human reviews exceptions

Not:

- LLM decides and verifier tries to rationalize

### Exit Criteria

- at least one extraction-to-verification pipeline works end to end on synthetic or curated data
- unverifiable facts are surfaced explicitly
- the rule engine never silently fills gaps with model guesses

## Workstream 6: Healthcare Pilot Use Cases

### Objective

Build a narrow but credible healthcare pilot before broadening scope.

## Priority Order

### Pilot 1: Prior Authorization / Medical Necessity

Why first:

- highly rule-based
- high administrative cost
- strong need for auditability
- existing policies often resemble DMN/cDMN logic

Example checks:

- diagnosis present
- conservative therapy attempted
- duration threshold met
- contraindications absent
- requested service matches policy

### Pilot 2: Clinical Trial Eligibility

Why second:

- explicit inclusion/exclusion rules
- usually deterministic
- reason codes matter

### Pilot 3: Medication Contraindication / Dosing Guardrails

Why third:

- bounded logic
- clear safety value
- easier to evaluate than open-ended recommendations

## Deliverables Per Pilot

- one structured input schema
- one decision policy
- one DMN/cDMN representation
- one verifier program
- one proof artifact format
- one explanation view with rule hits and evidence

## Workstream 7: Evaluation and Safety

### Objective

Show usefulness with the right metrics.

### Metrics

- semantic correctness of decision vs gold standard
- fraction of decisions with complete proof artifacts
- fraction of denials with explicit reason codes
- fraction of LLM extractions rejected for insufficient evidence
- clinician reviewer agreement
- time saved in adjudication
- false approval / false denial rates

### Safety Principles

- no silent fallback from verifier failure to model approval
- no hidden normalization steps
- preserve negative outcomes as first-class results
- make unverifiable inputs visible
- keep humans in the loop for ambiguous cases

## Concrete Milestones

## Milestone 0: Stabilize the Core

### Scope

- fix failing tests
- fix config parser round-trip
- clarify proof claims

### Success

- zero failing core tests
- one architecture note on proof boundary

## Milestone 1: Make the Decision Layer Real

### Scope

- replace hardcoded assembly templates with parser-backed DMN/cDMN compilation
- proper inequality semantics
- proper DMN/cDMN compilation
- `forall` / `exists` support for cDMN audit constraints
- halt on both success and failure
- reason codes

### Success

- loan and ratio examples no longer rely on equality placeholders
- cDMN compilation is AST-driven rather than string-pattern driven
- quantified audit constraints can be expressed without ad hoc compiler hacks
- new tests cover approvals and denials

## Milestone 2: Prove Real Policy Programs

### Scope

- stabilize STARK support for branching and loops
- support richer verifier programs
- bind proofs to structured inputs

### Success

- end-to-end proof for at least one policy-sized decision program

## Milestone 3: Add Healthcare Input and Evidence

### Scope

- healthcare fact schema
- provenance links
- deterministic normalization

### Success

- one policy can run from structured healthcare facts to proof artifact

## Milestone 4: Integrate LLM Extraction

### Scope

- LLM to structured facts
- fact validation
- verifier downstream

### Success

- one extraction-plus-verification demo on realistic synthetic cases

## Milestone 5: Pilot and Evaluate

### Scope

- prior auth or trial eligibility pilot
- explanation outputs
- evaluation metrics

### Success

- stakeholder-ready demo with:
  - structured input
  - verified decision
  - proof
  - human-readable explanation

## Suggested Near-Term File Strategy

Move from example scripts toward supported modules.

### Keep examples for demos only

- `examples/dmn_compiler.rs`
- `examples/cdmn_logic_compiler.rs`
- `examples/ai_reasoner_audit.rs`
- `examples/verifiable_rag.rs`

### Promote core functionality into `src/`

- DMN/cDMN parser and compiler
- logic parser + AST
- policy execution library
- explanation generation
- structured input schemas
- proof receipt formatting

### Add dedicated tests

- `tests/dmn.rs`
- `tests/cdmn.rs`
- `tests/clinical_policy.rs`
- `tests/proof_policies.rs`

## Recommended First Healthcare Demo

Build:

- a prior authorization verifier for one narrow service

Include:

- structured facts from intake form or note extraction
- explicit inclusion/exclusion rules
- approval and denial paths
- rule-hit IDs
- reason codes
- proof receipt

This is the shortest path from current prototype to something both technically strong and clinically legible.

## Bottom Line

The right strategy is not to turn this into a free-form clinical AI.

The right strategy is to make it a:

- proof-carrying decision and audit layer
- for structured, high-stakes healthcare rules
- downstream of probabilistic extraction systems

That path is concrete, defensible, and well aligned with the part of the Kierner taxonomy that the literature has underexplored most:

- explainability
- trustworthiness
- conformance to explicit domain knowledge
