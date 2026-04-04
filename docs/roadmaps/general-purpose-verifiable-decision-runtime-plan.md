# General-Purpose Verifiable Decision Runtime Plan

## Goal

Turn the current prototype into a reusable decision platform with:

- a typed, verifiable core semantics
- DMN/cDMN as compatibility frontends
- multiple execution backends
- proof-carrying outcomes

The core system should be useful across industries, not only healthcare.

## Design Principles

1. The semantic source of truth is a small core IR, not a frontend syntax.
2. Frontends parse and lower into the IR.
3. The native interpreter and proof system target the IR semantics.
4. The transformer-shaped runtime is a backend, not the semantic center.
5. Features enter through typed semantics first, then through frontend compatibility.
6. Compatibility should be profile-based, not all-or-nothing.

## Target Architecture

### Layer 1: Frontends

Supported frontends should eventually include:

- DMN subset
- cDMN subset
- native decision JSON format
- programmatic Rust builder API

Frontend responsibilities:

- parse source files
- build syntax trees
- perform source-level validation
- lower into the typed core IR

Recommended modules:

- `src/frontend/dmn_parser.rs`
- `src/frontend/cdmn_parser.rs`
- `src/frontend/native_schema.rs`

### Layer 2: Semantic Analysis

This layer should:

- resolve names
- typecheck expressions
- validate collections and quantifier scopes
- validate decision-table structure
- validate hit policies
- normalize frontend sugar into canonical forms

Recommended modules:

- `src/semantics/types.rs`
- `src/semantics/typecheck.rs`
- `src/semantics/name_resolution.rs`
- `src/semantics/normalize.rs`

### Layer 3: Core IR

This should be the proof-facing language.

The IR should represent:

- typed scalar values
- records and collections
- named expressions
- boolean predicates
- quantified constraints
- decision tables
- rule outputs
- explicit failure reasons
- deterministic evaluation order

Recommended modules:

- `src/ir/mod.rs`
- `src/ir/types.rs`
- `src/ir/expr.rs`
- `src/ir/decision.rs`
- `src/ir/lowering.rs`

### Layer 4: Execution Backends

Backends should all implement the same IR semantics:

- native interpreter
- TVM lowering
- transformer-shaped runtime
- optional Burn / ONNX export

Recommended modules:

- `src/backend/native_ir.rs`
- `src/backend/tvm_lowering.rs`
- `src/backend/transformer_backend.rs`

### Layer 5: Proof and Attestation

The proof layer should bind:

- core IR program hash
- policy version
- structured inputs
- structured outputs
- reason codes
- optional trace commitments

Near-term proof target:

- native IR / VM semantics

Longer-term target:

- optionally direct transformer-trace proving

## Core Semantic Features

The core IR should support these before trying to claim broad cDMN completeness:

### Types

- `bool`
- `i64`
- fixed-point decimal
- `string` only if strictly needed for frontend identifiers, not proof arithmetic
- `date`
- `timestamp`
- nullable / missing / unknown forms
- lists / bounded collections
- records / structs

### Expressions

- arithmetic
- comparison
- inclusive and exclusive ranges
- boolean composition
- named derived expressions
- quantifiers
- controlled function calls from a standard library

### Decision Semantics

- ordered rule evaluation
- hit policies
- default rules
- explicit pass/fail outcomes
- reason codes
- machine-readable explanation payloads

### Data Semantics

- missing vs unknown
- units
- simple code/value normalization hooks
- explicit collection bounds

## cDMN/DMN Compatibility Strategy

Implement compatibility in profiles:

### Profile A: Verifiable Core

Guaranteed and proof-friendly:

- deterministic expressions
- quantifiers over bounded collections
- ranges
- named derived expressions
- explicit reason outputs

### Profile B: Decision Table Compatibility

Adds:

- ordered rules
- default rules
- core hit policies
- rule tracing

### Profile C: Extended Compatibility

Adds:

- richer standard-library functions
- temporal helpers
- broader import/export support
- selected advanced DMN/cDMN constructs

This keeps claims precise and avoids pretending every spec feature is equally mature.

## Repo Refactor Plan

The current `src/cdmn.rs` is doing too much. Refactor in phases.

### Phase 1: Extract Syntax And Types

Move out:

- syntax structs
- tokens
- parser
- AST types

Target:

- `src/frontend/cdmn_ast.rs`
- `src/frontend/cdmn_parser.rs`

### Phase 2: Introduce Typed IR

Move out:

- semantic lowering
- definition allocation logic
- expression normalization

Target:

- `src/ir/*`
- `src/semantics/*`

### Phase 3: Make TVM A Backend

Current cDMN compilation emits TVM assembly directly.

Replace direct frontend-to-TVM compilation with:

- `frontend -> typed IR -> TVM lowering`

### Phase 4: Bind Proof Claims To IR-Level Artifacts

Instead of only proving final machine state, make the public claim include:

- IR program identity
- input identity
- output identity
- reason identity

## Recommended Milestones

### Milestone 1: Typed Core Skeleton

Deliverables:

- core type system
- IR expression model
- name resolution
- typechecking
- migration of current cDMN parser into frontend modules

Exit criteria:

- current cDMN examples lower into IR
- existing tests still pass

### Milestone 2: Decision-Table Semantics

Deliverables:

- decision-table IR
- ordered evaluation
- default rules
- hit policies
- rule-hit tracing

Exit criteria:

- loan and policy examples run through IR rather than ad hoc assembly generation

### Milestone 3: Temporal And Null Semantics

Deliverables:

- date/timestamp type
- temporal operators
- null/missing/unknown semantics
- proof-friendly lowering rules

Exit criteria:

- healthcare and insurance policies can express recency windows and incomplete evidence

### Milestone 4: Fixed-Point Decimal Semantics

Deliverables:

- decimal representation
- arithmetic rules
- rounding policy
- finance-safe and healthcare-safe comparisons

Exit criteria:

- ratios, money, and dosage calculations avoid float semantics

### Milestone 5: Proof Claim Hardening

Deliverables:

- proof artifact schema for program/input/output/reasons
- explicit IR program commitment
- replay metadata

Exit criteria:

- verifier outputs are replayable and externally checkable at the decision level

### Milestone 6: Compatibility Expansion

Deliverables:

- broader DMN/cDMN import support
- conformance fixtures
- profile documentation

Exit criteria:

- supported subset is precise, tested, and documented

## Cross-Industry Pilots

Build pilots that stress different parts of the core.

### Healthcare

- prior auth
- trial eligibility
- medication safety
- guideline conformance

### Finance

- underwriting guardrails
- adverse action explanations
- affordability and ratio policy checks

### Insurance / Benefits

- coverage determination
- step therapy checks
- eligibility adjudication

### Compliance / Trust

- KYC gating
- sanctions policy checks
- marketplace policy enforcement

## Testing Strategy

You need four test layers:

1. Parser and AST tests
2. Semantic analysis and IR lowering tests
3. Backend equivalence tests
4. Proof regression tests

Include:

- conformance fixtures
- edge-case fixtures
- negative-outcome fixtures
- cross-industry policy samples

## Immediate Next Steps For This Repo

1. Split `src/cdmn.rs` into frontend, semantics, and backend modules.
2. Introduce a first typed IR without changing user-visible behavior.
3. Move current cDMN compilation to `frontend -> IR -> TVM`.
4. Add null/missing semantics before temporal logic.
5. Add decision-table semantics on top of the IR.
6. Bind proofs to IR-level program and outcome identities.

## Bottom Line

The right architecture is:

- verifiable core semantics first
- frontend compatibility second
- backend portability third

That gives the project a general-purpose foundation while still allowing broad DMN/cDMN support over time.
