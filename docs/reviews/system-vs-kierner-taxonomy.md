# System Review: Relation to Kierner-2023 Taxonomy

## Scope

This note reviews the current codebase and explains how the system relates to the taxonomy in:

- Kierner, Kucharski, Kierner (2023), *Taxonomy of hybrid architectures involving rule-based reasoning and machine learning in clinical decision systems: A scoping review*

The focus is the repository as it exists now: a deterministic transformer-shaped VM, a native semantic interpreter, a vanilla STARK proof layer, and prototype DMN/cDMN examples.

## Executive Summary

The current system is not a conventional hybrid CDS in the sense described by Kierner-2023. The decision logic itself is primarily rule-based. The unusual part is that the rules are compiled into a transformer-shaped deterministic execution substrate, then optionally proved with a STARK.

If this repo is used as:

- a rule/verifier layer on top of an upstream LLM or NLP extractor,

then it maps most closely to the paper's `MLRB` pattern:

- ML first produces candidate facts, interpretations, or proposals
- rule-based logic makes the final bounded decision

If the repo is viewed in isolation, it is closest to `REML` only in a structural sense:

- rules are embedded into a neural-style architecture

But that is not a clean fit, because the "ML" part here is not trained predictive ML. It is a compiled, deterministic transformer substrate.

The strongest novelty is not "AI predicts better." The strongest novelty is:

- explicit rule logic
- deterministic transformer execution
- transparent cryptographic attestation

That combination is useful for high-stakes, structured adjudication problems, especially where explainability, policy conformance, and auditability matter more than prediction accuracy.

## What the Current System Actually Is

At the code level, the architecture looks like this:

1. A `.tvm` assembly program is parsed and compiled into a transformer-shaped runtime.
2. A native interpreter exists as a semantic oracle.
3. The transformer runtime and native interpreter can be checked in lockstep.
4. The STARK layer currently proves the native interpreter trace, not the transformer forward pass directly.

Relevant implementation points:

- Transformer compilation into instruction banks: `src/model.rs`
- Transformer runtime execution: `src/runtime.rs`
- Native interpreter semantics: `src/interpreter.rs`
- Cross-engine verification: `src/verification.rs`
- STARK proving and verification: `src/proof.rs`

## Key Findings

### 1. The STARK currently proves native VM semantics, not the transformer execution directly

This is the most important architectural clarification.

The README states:

- "A transformer that computes. A STARK that proves it."
- "program → compile → transformer weights → execute → trace → STARK proof"

But in implementation, `prove_execution_stark()` rebuilds and runs a `NativeInterpreter`:

- `src/proof.rs`, around the `prove_execution_stark_with_options()` implementation

So the current claim should be phrased more precisely as:

- prove the VM semantics with STARK
- separately cross-check that the transformer runtime matches those semantics

That is still useful, but it is weaker than "the transformer forward pass itself is what is being proved."

### 2. The core system is strongest as a proof-carrying rule engine

The rule logic is explicit, deterministic, and inspectable. The transformer substrate is not being used as a statistical learner here. It is being used as:

- a compiled execution target
- a portable runtime form
- a system that can be compared across Rust, Burn, and ONNX backends

That makes the repo much more interesting for:

- bounded policy decisions
- compliance checks
- guideline conformance
- post-hoc verification of model outputs

than for:

- direct diagnosis
- risk prediction
- pure data-driven prognosis

### 3. The DMN/cDMN examples are prototypes, not semantically complete decision engines

The current DMN and cDMN compilers are promising sketches, but not yet correct enough for real policy use.

Examples:

- `examples/dmn_compiler.rs` simplifies threshold checks into equality-like behavior
- `examples/cdmn_logic_compiler.rs` does the same for `>=` constraints
- generated programs such as `programs/loan_rules.tvm` and `programs/budget_verifier.tvm` therefore do not yet encode true DMN/cDMN semantics

This matters because healthcare rules depend heavily on:

- inequalities
- ranges
- temporal logic
- missing-data handling
- prioritized hit policies
- clinician-readable reason codes

The current examples do not yet provide that.

### 4. Negative outcomes are not yet first-class provable results

In the current generated verifier pattern:

- success halts
- failure loops forever

The prover rejects non-halting executions, so the system can prove:

- "this proposal satisfies the constraints"

but it cannot yet cleanly prove:

- "this proposal fails constraint C2"
- "this denial is correct for these reasons"

For healthcare, denials and exclusions are just as important as approvals.

### 5. The proof layer is promising but not fully stable yet

At review time, `cargo test --quiet` reports two failing tests:

- `config::tests::display_round_trips_through_from_str`
- `proof::tests::small_loop_round_trips_through_stark_proof`

The first is a config/parser inconsistency.

The second is more important: a loop-oriented STARK proof test currently fails in the polynomial layer. That means the proof backend is not yet fully stable even for core VM behavior.

### 6. The LLM-facing examples are still conceptual

Examples such as:

- `examples/ai_reasoner_audit.rs`
- `examples/verifiable_rag.rs`

show the intended direction clearly, but they do not yet constitute a real end-to-end "LLM + verified decision" system.

In particular, the RAG example explicitly notes that manually populated memory history is not actually passed through the current prover path.

So the right framing today is:

- conceptually strong
- partially implemented
- not yet a finished verified neuro-symbolic decision pipeline

## Mapping to Kierner-2023

## Closest Fit: `MLRB`

If you put an LLM or NLP system upstream of this verifier, the closest Kierner category is:

- `MLRB`: ML pre-processes input data for rule-based inference

That is because the likely operational architecture is:

1. An LLM extracts or proposes structured facts from unstructured inputs.
2. The DMN/cDMN verifier consumes those facts.
3. The verifier produces the final decision, trace, and proof artifact.

This is a strong fit for:

- note-to-structure extraction
- document review
- prior auth fact extraction
- guideline eligibility extraction

followed by:

- a bounded decision engine

That is a clinically plausible pattern and aligns well with Kierner's observation that explainability and trustworthiness are underdeveloped in the literature.

## Secondary Fit: `PERML` as a future direction

The repo does not currently implement a real parallel ensemble, but it could evolve there.

A future healthcare architecture could keep separate channels for:

- model recommendation
- rule/verifier recommendation
- explicit aggregator logic

That would move the system toward:

- `PERML`: Parallel Ensemble of Rules and ML

This would be especially valuable when:

- the model proposes
- the rule engine constrains
- disagreements are surfaced to a human reviewer

For healthcare, this is often better than collapsing everything into one opaque decision process.

## Weak Fit: `REML`

The system resembles `REML` only superficially because rules are embedded in a transformer-shaped substrate.

But the Kierner paper's `REML` examples mostly involve:

- neuro-fuzzy systems
- knowledge-based neural networks
- rule-informed learned architectures

Your system is different:

- the transformer is compiled, not trained for the decision task
- the execution is deterministic
- the goal is semantic fidelity and proof, not predictive generalization

So calling it `REML` without qualification would be misleading.

## Novelty Assessment

## What is genuinely novel

The most novel part of the repo is the combination of:

- explicit symbolic decision logic
- transformer-shaped deterministic execution
- portable multi-backend verification
- transparent STARK attestation

That is not the main design center of the systems surveyed in Kierner-2023.

In other words, the novelty is less:

- "new hybrid clinical predictor"

and more:

- "proof-carrying decision computation over rule logic, deployable through transformer-like runtimes"

## What is useful right now

Even in its current state, the repo is useful as:

- a research platform for verifiable policy execution
- a prototype for auditable AI output checking
- an architecture testbed for rule-plus-proof systems

It is not yet strong enough to claim:

- production-grade verifiable CDS
- clinically correct DMN/cDMN execution
- direct proof of full transformer execution semantics

## Usefulness for Healthcare

The best healthcare opportunities are not open-ended diagnosis. They are structured, high-stakes decisions where explicit rules are already part of the domain.

### Good early healthcare targets

- Prior authorization / medical necessity adjudication
- Coverage policy conformance
- Clinical guideline eligibility checks
- Trial eligibility screening
- Contraindication and dosing guardrails
- Referral or escalation eligibility checks
- Post-hoc audit of AI-generated care plans

These all have features that match the system well:

- explicit rule sets already exist
- decisions need auditability
- negative outcomes need justification
- there is often pressure to justify automation to regulators, clinicians, and patients

### Poor early targets

- diagnostic prediction from noisy signals
- prognosis/risk scoring as the main system function
- treatment ranking driven mostly by latent statistical inference

Those are better served by conventional learned models, possibly with rule wrappers, but they are not the strongest fit for this repo's current architecture.

## Recommended Strategic Positioning

The strongest positioning for the project is:

- do not present it as "LLM makes decisions"
- present it as "probabilistic systems may propose facts or candidates, but a verifiable rule machine decides and attests"

That is both more defensible technically and more aligned with the Kierner paper's underexplored motivation space:

- explainability
- trustworthiness
- conformance with domain knowledge

## Bottom Line

This system is not well described as just another hybrid rule/ML architecture from the Kierner taxonomy.

Its most useful interpretation is:

- a verifiable rule-execution and audit substrate
- that can sit downstream of probabilistic AI
- and therefore serve as the final decision or compliance layer in high-stakes workflows

That is novel and useful, especially for healthcare, but the current implementation still needs major work before it can support strong claims:

- semantic completeness of DMN/cDMN
- stable STARK proving
- support for negative outcomes
- more precise public claims about what is actually being proved

