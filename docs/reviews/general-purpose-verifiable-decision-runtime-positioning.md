# General-Purpose Verifiable Decision Runtime

## Thesis

The strongest long-term direction for this project is not:

- "implement all of cDMN as the product"

It is:

- build a general-purpose verifiable decision runtime
- treat DMN/cDMN compatibility as a frontend into that runtime

That is a better technical and strategic foundation for healthcare and for other industries.

## Why This Makes Sense

Many high-stakes workflows share the same shape:

- structured inputs
- explicit policy logic
- bounded deterministic outcomes
- reasoned negative decisions
- replayability and auditability
- versioned policy execution

That applies to:

- healthcare
- finance
- insurance
- benefits administration
- regulatory compliance
- identity and trust workflows
- procurement and marketplace governance
- public-sector eligibility

The current repo already points in this direction:

- parser-backed cDMN logic in `src/cdmn.rs`
- deterministic multi-backend execution
- reason codes and output registers
- quantified constraints
- range checks
- named derived expressions
- STARK proofs of the native ISA execution path

## How This Relates To Kierner-2023

The system is not best understood as a standard hybrid predictive model.

It is more accurate to describe it as:

- a verifiable symbolic decision substrate
- that can sit downstream of probabilistic AI
- and serve as the final decision or compliance layer

In Kierner terms:

- closest practical fit: `MLRB`
- possible future fit: `PERML`
- not a clean fit for typical `REML`, because the transformer runtime here is compiled and deterministic rather than task-trained

## Why "Full cDMN" Should Not Be The Product

Full cDMN support is a worthwhile compatibility goal, but it should not be the primary system identity.

If the project is framed as:

- "we implement all of cDMN"

then it risks becoming a standards clone.

If it is framed as:

- "we provide proof-carrying execution for decision logic, with cDMN/DMN as frontends"

then it becomes:

- cross-industry
- backend-agnostic
- easier to position commercially
- easier to evolve technically

## What The Transformer-Shaped Runtime Means In This Framing

The transformer-shaped runtime is not the main source of application value.

The main value is:

- deterministic semantics
- explicit policy logic
- reasoned outcomes
- portable verification
- cryptographic attestation

The transformer-shaped runtime is better understood as:

- a portable execution backend
- a bridge into model-serving infrastructure
- a research and deployment substrate

not as the semantic center of the platform.

## What The Product Should Be

The product should be a small, exact, verifiable decision kernel with:

- typed decision semantics
- multiple frontends
- multiple execution backends
- strong audit artifacts

Frontends could include:

- DMN
- cDMN
- a native JSON/YAML DSL
- programmatic APIs

Backends could include:

- native interpreter
- transformer-shaped runtime
- Burn
- ONNX
- possibly WASM later

## Strongest Use Cases

The best fit is not open-ended prediction.

The best fit is explicit decision computation such as:

- prior authorization and medical necessity
- coverage and benefits adjudication
- trial eligibility
- medication and safety guardrails
- adverse action and denial explanations
- loan and underwriting policy checks
- fraud/compliance gating
- KYC / policy enforcement / escalation routing

These workflows benefit because they need:

- consistent decisions
- explicit reasons
- operational auditability
- resistance to silent model drift
- explainable automation

## Bottom Line

The project should evolve toward:

- a general-purpose verifiable decision runtime

with:

- a small typed core
- cDMN/DMN lowering into that core
- multiple execution backends
- proof artifacts bound to inputs, policy version, outputs, and reasons

That is a stronger long-term direction than making cDMN completeness the entire product story.
