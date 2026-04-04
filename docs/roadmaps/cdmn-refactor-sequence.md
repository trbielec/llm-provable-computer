# cDMN Refactor Sequence

## Objective

Split the current `src/cdmn.rs` implementation into smaller modules without changing user-visible behavior, then use that structure to introduce a typed IR and broader decision-language support.

This plan is specifically for the current repository state, where `src/cdmn.rs` is still a large mixed module containing:

- serde-facing schema types
- logic AST types
- tokenizer and parser
- semantic validation and allocation helpers
- TVM code generation
- unit tests

## Current Starting Point

As of this plan:

- `src/cdmn.rs` is about 1,500 lines
- `src/lib.rs` re-exports the cDMN surface directly from `src/cdmn.rs`
- cDMN behavior is already used by:
  - unit tests in `src/cdmn.rs`
  - E2E tests in `tests/e2e.rs`
  - proof regressions in `src/proof.rs`
  - example compilers and audit examples

The most important constraint is:

- keep the public API stable while extracting internals

Do not try to introduce full IR and full cDMN compatibility in the same refactor.

## Refactor Rules

1. No user-visible semantic changes in extraction phases.
2. Keep `compile_cdmn_table`, `compile_cdmn_table_with_metadata`, and `parse_logic_expr` stable until the new internal structure is proven.
3. Add or preserve tests before moving code.
4. Make each phase independently releasable.
5. Defer new semantics to phases after the structural split.

## Proposed Intermediate Module Layout

Use sibling files first. That avoids an early `src/cdmn.rs` to `src/cdmn/mod.rs` move and keeps churn lower.

Recommended near-term files:

- `src/cdmn.rs`
- `src/cdmn_schema.rs`
- `src/cdmn_ast.rs`
- `src/cdmn_parser.rs`
- `src/cdmn_semantics.rs`
- `src/cdmn_backend_tvm.rs`

Later, after the split is stable, you can convert to:

- `src/cdmn/mod.rs`
- `src/cdmn/schema.rs`
- `src/cdmn/ast.rs`
- `src/cdmn/parser.rs`
- `src/cdmn/semantics.rs`
- `src/cdmn/backend_tvm.rs`

## Target Responsibilities

### `src/cdmn_schema.rs`

Move only schema and metadata-facing types:

- `Variable`
- `CollectionField`
- `Collection`
- `Definition`
- `Constraint`
- `CdmnOutputLayout`
- `ReasonMetadata`
- `CdmnProgramMetadata`
- `CompiledCdmnProgram`
- `CdmnTable`

Also keep:

- metadata helper methods such as `reason_for_code`

This file should remain serde-focused and mostly free of compiler logic.

### `src/cdmn_ast.rs`

Move syntax and AST-only types:

- `ArithmeticOp`
- `ArithmeticExpr`
- `ComparisonOp`
- `QuantifierKind`
- `BooleanExpr`
- tokenizer `Token`

This file should not know about TVM lowering or memory allocation.

### `src/cdmn_parser.rs`

Move:

- `tokenize`
- `Parser`
- `parse_logic_expr`
- internal arithmetic parser helper
- parser-specific tests

This module should output AST only.

### `src/cdmn_semantics.rs`

Move:

- table-level name validation
- declared memory validation
- definition register allocation
- duplicate-name handling
- other non-backend semantic checks

Near term, this module will still be light. Later it should grow into:

- name resolution
- typechecking
- null/missing semantics
- normalization

### `src/cdmn_backend_tvm.rs`

Move:

- `LogicCompiler`
- `TempAllocator`
- emit helpers
- comparison lowering
- `between` lowering
- quantifier lowering
- definition lowering

This module should accept validated semantic inputs and produce TVM assembly plus output metadata.

### `src/cdmn.rs`

Keep this as a façade during the extraction phases.

Responsibilities:

- wire parser + semantics + TVM backend
- preserve current public API
- re-export core public types

Goal:

- make `src/cdmn.rs` short and orchestration-only before introducing IR

## Concrete Phase Sequence

## Phase 0: Freeze Current Behavior

### Goal

Protect against accidental semantic drift during extraction.

### Work

- Add explicit regression coverage for current shipped cDMN examples:
  - budget
  - medication safety
  - lab/vitals guardrails
- Keep current E2E and proof tests as integration tests
- Add snapshot-like assertions on generated assembly for the current fixtures where helpful

### Files

- `tests/e2e.rs`
- `src/proof.rs`
- `src/cdmn.rs`

### Exit Criteria

- `cargo test --quiet` is green
- current example compilers still build

## Phase 1: Extract Schema Types

### Goal

Remove serde/data types from the parser and backend implementation.

### Work

Create `src/cdmn_schema.rs` and move:

- all public schema structs
- metadata helper methods

Then update:

- `src/cdmn.rs` to import from `crate::cdmn_schema`
- `src/lib.rs` re-exports to export the same public types

### Safety Notes

- do not change field names or serde attributes
- do not change public constructors or helper methods

### Exit Criteria

- zero behavior change
- all imports compile

## Phase 2: Extract AST And Parser

### Goal

Separate syntax concerns from compilation concerns.

### Work

Create:

- `src/cdmn_ast.rs`
- `src/cdmn_parser.rs`

Move:

- AST enums
- tokenizer
- parser
- parser unit tests

Keep `parse_logic_expr` publicly available through `src/cdmn.rs`.

### Safety Notes

- parser error text should remain unchanged where possible
- keep `parse_logic_expr` signature stable

### Exit Criteria

- parser tests remain green
- all existing code uses imported parser functions rather than local copies

## Phase 3: Extract Semantic Prepass

### Goal

Introduce an internal semantic stage before TVM lowering.

### Work

Create `src/cdmn_semantics.rs` and move:

- `validate_table_names`
- `max_declared_memory_cell`
- definition allocation
- other frontend-level validation helpers

Add a small semantic result type, for example:

```rust
pub struct ValidatedCdmnTable {
    pub table: CdmnTable,
    pub allocated_definitions: Vec<AllocatedDefinition>,
    pub scratch_start: u8,
}
```

`compile_cdmn_table_with_metadata` should become:

1. parse constraint expressions
2. validate names and memory layout
3. allocate definitions
4. hand off to TVM backend

### Safety Notes

- still no IR yet
- this is only a prepass, not a redesign

### Exit Criteria

- backend no longer owns table-validation responsibilities

## Phase 4: Extract TVM Backend

### Goal

Make direct TVM assembly lowering a backend module instead of cDMN’s whole identity.

### Work

Create `src/cdmn_backend_tvm.rs` and move:

- `LogicCompiler`
- scratch allocator
- TVM emit helpers
- comparison/range/quantifier lowering
- definition lowering

Add a narrow backend API like:

```rust
pub fn compile_validated_table_to_tvm(
    validated: &ValidatedCdmnTable,
    parsed_constraints: &[ParsedConstraint],
) -> Result<CompiledCdmnProgram>
```

Where `ParsedConstraint` can be a tiny internal struct:

```rust
struct ParsedConstraint {
    id: String,
    description: String,
    expr: BooleanExpr,
    reason_code: Word,
    reason_label: String,
}
```

### Safety Notes

- keep TVM output byte-for-byte stable where possible
- preserve output register layout

### Exit Criteria

- `src/cdmn.rs` is mostly orchestration
- TVM-specific code no longer sits beside parser code

## Phase 5: Move Unit Tests Out Of `src/cdmn.rs`

### Goal

Make module ownership visible in the test layout.

### Work

Move tests into:

- `tests/cdmn_parser.rs`
- `tests/cdmn_semantics.rs`
- `tests/cdmn_backend_tvm.rs`

Keep:

- `tests/e2e.rs` for end-to-end behavior
- `src/proof.rs` proof regressions near the proof layer

### Safety Notes

- keep test names descriptive by layer
- avoid duplicating large table fixtures by adding shared helpers if needed

### Exit Criteria

- unit-test organization mirrors module organization

## Phase 6: Introduce A Minimal Typed IR

### Goal

Start the real architecture transition without breaking the current frontend.

### Work

Add new modules:

- `src/ir/mod.rs`
- `src/ir/types.rs`
- `src/ir/expr.rs`
- `src/ir/decision.rs`

Initial IR scope should be deliberately small:

- integer scalars
- booleans
- named expressions
- comparisons
- `between`
- `and` / `or` / `not`
- bounded quantifiers
- reasoned constraints

Then add lowering from validated cDMN into IR:

- `cDMN -> semantic model -> IR`

The TVM backend should then lower:

- `IR -> TVM`

### Safety Notes

- do not add temporal semantics yet
- do not add decimals yet
- keep current frontends using the same public API

### Exit Criteria

- TVM lowering no longer consumes frontend AST directly

## Phase 7: Add Null/Missing Semantics

### Goal

Support real high-stakes decision inputs where absence matters.

### Work

Add semantic representation for:

- present value
- missing
- unknown

Then define:

- comparison behavior
- quantifier behavior
- range behavior
- reason behavior when data is insufficient

### Why Before Temporal Logic

Healthcare, insurance, and compliance decisions often fail because data is incomplete, not because temporal arithmetic is unavailable.

This is the more urgent semantic gap.

## Phase 8: Add Decision-Table IR

### Goal

Handle DMN-style ordered rules and hit policies on the same core substrate.

### Work

Extend IR for:

- ordered rules
- default rules
- hit policies
- rule trace outputs

Then create:

- `frontend -> decision-table IR -> TVM backend`

### Exit Criteria

- DMN toy compiler examples stop generating assembly directly

## Phase 9: Convert To Directory Modules

### Goal

Only after the structure is stable, replace the flat sibling-file layout with directory modules.

### Work

Move from:

- `src/cdmn.rs`
- `src/cdmn_schema.rs`
- `src/cdmn_parser.rs`
- ...

to:

- `src/cdmn/mod.rs`
- `src/cdmn/schema.rs`
- `src/cdmn/parser.rs`
- ...

### Why This Is Late

This move is cosmetic unless the earlier splits are already working. Do it after behavior is stable.

## Concrete First PR Sequence

If you want the smallest safe sequence of implementation work, do these PRs:

### PR 1

- extract `src/cdmn_schema.rs`
- update `src/lib.rs`
- no behavior change

### PR 2

- extract `src/cdmn_ast.rs`
- extract `src/cdmn_parser.rs`
- move parser tests

### PR 3

- extract `src/cdmn_semantics.rs`
- introduce validated-table helper types

### PR 4

- extract `src/cdmn_backend_tvm.rs`
- preserve current assembly output

### PR 5

- move unit tests into dedicated integration/unit test files

### PR 6

- introduce minimal IR
- lower validated cDMN into IR
- lower IR into TVM

## Success Criteria

The refactor is successful when:

- `src/cdmn.rs` becomes a thin façade
- parser, semantics, and backend code are no longer interleaved
- current cDMN behavior is unchanged
- new semantic work lands in IR and semantic modules instead of growing the façade again
- DMN/cDMN compatibility can expand without entangling proof/runtime code

## Recommended Next Action

Start with PR 1 immediately:

- create `src/cdmn_schema.rs`
- move schema and metadata structs there
- leave all behavior unchanged

That is the lowest-risk first cut and reduces `src/cdmn.rs` meaningfully without forcing a full redesign on day one.
