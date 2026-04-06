# Phase 0 Documentation Baseline

- date: 2026-04-06
- branch: issue-29-investigation

---

## Files Surveyed

### 1. `README.md`

EXISTS: yes

Key claims:

- **Proof boundary**: Classic VM STARK proves native ISA execution. Shared TransformerVm STARK claims to directly prove the `average-hard` exact-domain path. Direct DMN/cDMN STARK proves decision logic. `poet_proof` proves a reduced local transformer-style forward pass.
- **Supported instructions (STARK)**: `NOP`, `LOADI`, `LOAD`, `STORE`, `PUSH`, `POP`, `ADD`/`ADDM`, `SUB`/`SUBM`, `MUL`/`MULM`, `AND`/`ANDM`, `OR`/`ORM`, `XOR`/`XORM`, `CALL`, `RET`, `JMP`, `JZ`, `JNZ`, `HALT`. Carry-bearing traces fully constrained. `CMP`/`CMPM` and non-`average-hard` modes not yet supported.
- **Benchmark numbers**: commit ~1.18ms, audit generation ~318µs, verify ~832µs; batch: 64 runs ~325µs, 256 ~1.04ms, 1,024 ~3.81ms, 4,096 ~14.7ms.
- **Feature flags**: default (transformer + native + TUI + STARK), `burn-model`, `onnx-export`, `metal-batch`, `full`.
- **Structure listed**: includes `commit_and_audit.rs`, `freivalds.rs`, `merkle_trace.rs`, `cdmn/`, `dmn.rs`, `dmn_decision.rs`, `dmn_direct.rs`, `poet_proof.rs`, which are NOT present in `CLAUDE.md`'s structure section (CLAUDE.md structure is narrower/older).
- The README instruction table lists `JumpIfSign` / `JumpIfNotSign` implicitly (not by name; they do not appear in the ISA table at all).

### 2. `CLAUDE.md`

EXISTS: yes

Key claims:

- **Identity**: Milestone 1 and 2 complete; Milestone 3 (STWO) is the next target.
- **Proof boundary**: vanilla STARK scope limited to the listed instructions plus `average-hard` only. Rejected: `CMP`/`CMPM`, non-`average-hard`, non-halted executions.
- **Supported instructions (STARK)**: `NOP`, `LOADI`, `LOAD`, `STORE`, `PUSH`, `POP`, `ADD`, `ADDM`, `SUB`, `SUBM`, `MUL`, `MULM`, `AND`, `ANDM`, `OR`, `ORM`, `XOR`, `XORM`, `CALL`, `RET`, `JMP`, `JZ`, `JNZ`, `HALT`.
- **Benchmark numbers**: same as README (`commit_audit_benchmark`: ~1.18ms commit, ~318µs audit, ~832µs verify; `batch_benchmark`: ~325µs/64, ~1.04ms/256, ~3.81ms/1,024, ~14.7ms/4,096).
- **Feature flags**: consistent with README.
- **Structure**: does NOT list `commit_and_audit.rs`, `freivalds.rs`, `merkle_trace.rs`, `cdmn/`, `dmn.rs`, `dmn_decision.rs`, `dmn_direct.rs`, `poet_proof.rs`. These modules exist in the README structure listing and (from git history) in the actual repository. CLAUDE.md structure block reflects an earlier, narrower snapshot of the codebase.

### 3. `docs/architecture/proof-boundary.md`

EXISTS: yes

Key claims:

- STARK proves native ISA execution trace; `TransformerVm` is not directly proved, only cross-checked via lockstep (`src/verification.rs`).
- Lists four backends: `TransformerVm`, `NativeInterpreter`, `BurnRuntime`, `OnnxRuntime`.
- Does NOT mention: `JumpIfSign`/`JumpIfNotSign`, `CMP`/`CMPM` rejection, direct DMN/cDMN proof path, `poet_proof.rs`, shared-transformer STARK path, or any benchmark numbers.

---

## Source Code Cross-Check

### `src/proof.rs`

`validate_proof_inputs()` (line 955) rejects:
- `CmpImmediate` / `CmpMemory` — returns `UnsupportedProof` error.
- Non-`AverageHard` attention modes — returns `UnsupportedProof` error.
- Empty programs.

All other instructions, including `JumpIfSign` and `JumpIfNotSign`, pass validation and have AIR transition constraints generated (lines 500–516). These two instructions are NOT listed in the STARK-supported instruction set in either README or CLAUDE.md, but they ARE accepted by the prover and have polynomial constraints in the AIR. This is a documentation gap: `JZ` and `JNZ` are listed but `JS` (JumpIfSign) and `JNS` (JumpIfNotSign) are silently omitted from the supported-instruction tables.

### `src/interpreter.rs`

Implements `NativeInterpreter`. Contains handlers for all listed instructions including `JumpIfSign` and `JumpIfNotSign` (lines 276–283). `CmpImmediate` and `CmpMemory` are also handled by the interpreter (lines 239–247) — they execute correctly at the native level; the rejection is only at the STARK proof layer.

---

## Discrepancies

| # | Location | Claim | Reality | Severity |
|---|----------|-------|---------|----------|
| D1 | README.md, "What Is Actually Proved Today" table | Shared transformer STARK "directly proves the full staged shared forward pass" | `src/proof.rs` `prove_execution_stark()` runs a `NativeInterpreter`, not `TransformerVm`. The `proof-boundary.md` explicitly contradicts this README claim. | HIGH |
| D2 | README.md ISA table + CLAUDE.md supported list | `JumpIfSign` / `JumpIfNotSign` not mentioned as STARK-supported | `src/proof.rs` AIR generates constraints for both (lines 500–516); `validate_proof_inputs` does not reject them | MEDIUM |
| D3 | CLAUDE.md `<structure>` block | Does not list `commit_and_audit.rs`, `freivalds.rs`, `merkle_trace.rs`, `cdmn/`, `dmn.rs`, `dmn_decision.rs`, `dmn_direct.rs`, `poet_proof.rs` | README structure section does list all of these; they are present in the codebase | MEDIUM |
| D4 | `docs/architecture/proof-boundary.md` | States "The transformer-shaped runtime is not yet directly proved by the STARK" and focuses only on native-ISA proof path | README claims the `average-hard` exact-domain path now has a direct shared-forward-pass STARK | HIGH (internal inconsistency between docs) |
| D5 | `docs/architecture/proof-boundary.md` | Does not mention `JumpIfSign`/`JumpIfNotSign`, `poet_proof.rs`, direct DMN/cDMN paths | All three exist and affect the proof boundary description | LOW–MEDIUM |

---

## Status of `docs/architecture/proof-boundary.md`

The file EXISTS. However, it describes an older, narrower proof boundary (native-ISA-only STARK, no direct transformer STARK) that is inconsistent with the current README's claim of a "full staged shared forward pass" STARK for the `average-hard` path. It also omits the direct DMN/cDMN proof path and `poet_proof.rs`. The file needs to be updated before Phase 1 to reflect the actual current proof boundary rather than the earlier one it describes.

---

## Overall Status

**NEEDS_UPDATE**

Primary reasons:
1. D1/D4: README and `proof-boundary.md` contradict each other on whether the shared `TransformerVm` forward pass is directly proved by the STARK. `src/proof.rs` supports `proof-boundary.md`'s older claim (native interpreter is what is proved), but README claims a direct shared-forward-pass STARK now exists. This needs resolution before Phase 1.
2. D2: `JumpIfSign` and `JumpIfNotSign` are silently supported by the STARK AIR but absent from every doc's supported-instruction list.
3. D3: CLAUDE.md structure section is stale relative to the actual repository and README.
