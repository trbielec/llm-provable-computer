# Transformer VM

**A transformer that computes. Multiple proof paths for deterministic decision execution.**

This repo has more than one execution path, and they do not all sit inside the same proof boundary.

At a high level:

- human-readable decision logic such as DMN / cDMN can be compiled into the repo's tiny VM language,
- that VM program can run on a native interpreter or on a transformer-shaped runtime,
- the classic vanilla STARK path proves the matching native VM execution,
- the shared `TransformerVm` forward-pass STARK path now directly proves the supported shared transformer execution path,
- newer direct decision provers can prove DMN / cDMN logic without going through the transformer at all,
- a separate `poet_proof` module can directly prove a reduced transformer-style forward pass, but it is not yet the full shared transformer runtime.

Today, the safest one-sentence claim is:

> the repo can prove supported decision execution, and for the supported `average-hard` exact-domain path it can also directly prove the shared `TransformerVm` forward pass itself.

```
human decision logic → compiled executable form → run case → decision + reasons → optional STARK proof
```

No sampling. No stochastic output. Same input, same output, every time.

Based on [*Can LLMs Be Computers?*](https://www.percepta.ai/blog/can-llms-be-computers) by Percepta, which showed that 2D attention over convex hulls turns a transformer into a deterministic machine. This repo implements that system in Rust, checks the transformer against a native semantic oracle, and proves supported executions with transparent STARKs.

---

## Read This First

The same terms mean different things in different papers. In this repo:

- `DMN` = Decision Model and Notation, a rule/table format for decisions.
- `cDMN` = constrained / logical DMN-style decision rules.
- `ISA` = instruction set architecture, the tiny machine language of this repo's VM.
  Example instructions: `LOAD`, `ADD`, `JZ`, `HALT`.
- `VM` = virtual machine, the tiny software computer that executes that ISA.
- `Runtime` = the engine that executes a compiled program.
  Examples: native interpreter, transformer-shaped runtime, direct decision runtime.
- `STARK` = a cryptographic proof that the computation followed the rules correctly.

The important architecture split is:

1. `DMN/cDMN -> VM/ISA program -> native or transformer runtime -> proof/cross-check`
2. `DMN/cDMN -> direct decision program -> direct decision proof`
3. `Program -> reduced local transformer-style forward pass -> reduced Poet proof`

The transformer does **not** emit ISA while running. On the classic path, the ISA program already exists, and the transformer runtime is just one way of executing it.

### What Is Actually Proved Today

| Path | What is executed | What the proof directly covers |
|---|---|---|
| Classic VM STARK | VM program | Native VM execution trace |
| Shared transformer STARK | Shared `TransformerVm` (`average-hard`, exact-domain path) | Full staged shared forward pass |
| Direct DMN/cDMN STARK | Decision program | Decision logic directly |
| `poet_proof` | Reduced local transformer-style program | Reduced local forward pass |

What is **not** yet true repo-wide:

- "every proof path in the repo is the shared `TransformerVm` forward pass in `src/model.rs`"
- "the shared transformer proof currently covers every attention mode or every floating-point behavior"

---

## Why This Works

This section explains the classic transformer/VM path.

Three ideas, stacked.

### 1. Attention Is Memory Access

The memory layout is still ordinary addressed memory. `LOAD 7` means "read address 7", not
"read point `(x, y)` from a 2D grid."

What is different is the read semantics behind that address. Each memory address keeps a
history of writes, and each write is stored as a 2D point `(step, value)`. So a memory read is
really:

> choose one value from this address's write history according to a retrieval policy

The default retrieval policy is "latest write wins." To express that, query with direction
`[1, 0]`:

```
score(q, k) = 1 * step + 0 * value = step
```

The argmax selects the most recent write. Other query directions give different retrieval
policies over the same history:

- `[0, 1]` = prefer the highest value ever written
- `[1, 1]` = trade off recency and value

This is why the 2D convex hull exists. It is not there to provide 2D addressing. It is there to
make history-aware retrieval efficient. In 2D, the maximizer of any linear objective lies on the
**convex hull** of the key set, so the runtime can answer hard reads by searching hull vertices
instead of scanning the full history.

If all you wanted was "read the current contents of address 7," a normal array would be enough.
The hull-backed cache is useful because the model is doing "read from address 7 using the
configured retrieval policy over all prior writes."

At step 1,000,000, each memory read still costs O(log n).

### 2. Feed-Forward Layers Are Instructions

Each compiled instruction becomes a deterministic gate-and-transform operation in the FFN:

```
output = gate(state) * transition(state)
```

The gate activates only for the correct opcode. The transition encodes the instruction semantics. Together they form a compiled, non-learned feed-forward layer that maps one machine state to the next.

### 3. The Trace Is Already a STARK Witness

A STARK proves that a sequence of states satisfies a set of polynomial constraints (an AIR). An execution trace --- a table of `(PC, ACC, SP, flags, memory)` rows where each row follows deterministically from the previous --- is exactly that object.

In the classic implementation, the proved trace is generated by the native ISA interpreter. The transformer, Burn, and ONNX runtimes are separate execution backends that can be checked against the same semantics. The transition constraints **are** the instruction semantics. The boundary constraints **are** the initial and final states.

```
                    ┌──────────────────────────────────────────┐
                    │           Transformer VM Block           │
                    │                                          │
  Machine    ┌──────┴─────┐    ┌─────────────┐    ┌──────────┐ │
  State  ───>│   Encode   │───>│  Attention  │───>│   FFN    │─┼──> Next State
  (d=36)     │  (state →  │    │  (2D heads  │    │ (compiled│ │    (d=36)
             │   token)   │    │  + hull KV) │    │  instr.) │ │
             └────────────┘    └──────┬──────┘    └──────────┘ │
                    │                 │                        │
                    │          ┌──────┴──────┐                 │
                    │          │ HullKvCache │                 │
                    │          │ O(log n)    │                 │
                    │          │ memory read │                 │
                    │          └─────────────┘                 │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                              Execution Trace
                           (= STARK AIR witness)
```

The proof is **transparent** (no trusted setup), **post-quantum** (hash-based, no elliptic curves), and verification is **O(log^2 n)** --- exponentially cheaper than re-execution.

---

# Quick Start

These commands use the classic VM / ISA path.

```bash
git clone https://github.com/AbdelStark/llm-provable-computer && cd llm-provable-computer

# Run a program
cargo run --bin tvm -- programs/fibonacci.tvm

# Full execution trace
cargo run --bin tvm -- run programs/fibonacci.tvm --trace

# --- NEW: Optimistic Verification ---
# Generate a compact execution receipt (Merkle commitment)
cargo run --bin tvm -- commit programs/fibonacci.tvm -o fib.receipt.json

# Audit a specific step of the execution against the receipt
cargo run --bin tvm -- audit programs/fibonacci.tvm fib.receipt.json --step 10

# --- STARK Proofs ---
# Prove execution with a vanilla STARK (more expensive)
cargo run --bin tvm -- prove-stark programs/fibonacci.tvm -o fib.proof.json

# Verify without re-running
cargo run --bin tvm -- verify-stark fib.proof.json
```

# Verify transformer matches native interpreter (lockstep)
cargo run --bin tvm -- run programs/fibonacci.tvm --verify-native

# Interactive TUI
cargo run --bin tvm -- tui programs/fibonacci.tvm

# Tests
cargo test
```

### Example Output

```
program: programs/fibonacci.tvm
engine: transformer
steps: 103
halted: true
acc: 21
zero_flag: false
memory: [13, 21, 21, 7, 7]
elapsed_ms: 9.760
throughput_steps_per_sec: 10553.55
```

---

## Decision Frontends

Human-readable decision policies do not have to be written as assembly by hand.

### DMN / cDMN

- `DMN` and `cDMN` tables can be compiled into `.tvm` assembly for the classic VM path.
- Those compiled `.tvm` programs can then run on the native interpreter or the transformer runtime.
- For newer decision workflows, DMN / cDMN can also compile into direct decision programs that are proved without going through the transformer backend.

So the usual classic flow is:

```text
DMN / cDMN -> TVM / ISA program -> native or transformer runtime -> decision
```

And the newer direct-decision flow is:

```text
DMN / cDMN -> direct decision program -> decision
```

The direct decision flow is usually the clearest proof boundary for decision automation. The transformer path is still valuable as an alternative execution backend and as a bridge to model-serving infrastructure.

---

## The Assembly Language

Programs are `.tvm` files with a compact assembly syntax.

**Fibonacci(8) = 21:**

```asm
.memory 5
.init 0 0
.init 1 1
.init 3 0
.init 4 7

loop:
  LOAD 3
  SUBM 4
  JZ done
  LOAD 0
  ADDM 1
  STORE 2
  LOAD 1
  STORE 0
  LOAD 2
  STORE 1
  LOAD 3
  ADD 1
  STORE 3
  JMP loop

done:
  LOAD 1
  HALT
```

**Recursive factorial (5! = 120):**

```asm
.memory 11

LOADI 5
CALL fact
HALT

fact:
  JZ fact_base
  PUSH
  SUB 1
  CALL fact
  STORE 0
  POP
  MULM 0
  RET

fact_base:
  LOADI 1
  RET
```

<details>
<summary><strong>Full instruction set</strong></summary>

| Instruction | Effect |
|---|---|
| `NOP` | No operation |
| `LOADI imm` | `ACC = imm` |
| `LOAD addr` | `ACC = MEM[addr]` |
| `STORE addr` | `MEM[addr] = ACC` |
| `PUSH` | `SP -= 1; MEM[SP] = ACC` |
| `POP` | `ACC = MEM[SP]; SP += 1` |
| `ADD imm` | `ACC += imm` |
| `ADDM addr` | `ACC += MEM[addr]` |
| `SUB imm` | `ACC -= imm` |
| `SUBM addr` | `ACC -= MEM[addr]` |
| `MUL imm` | `ACC *= imm` |
| `MULM addr` | `ACC *= MEM[addr]` |
| `AND imm` | `ACC &= imm` |
| `ANDM addr` | `ACC &= MEM[addr]` |
| `OR imm` | `ACC \|= imm` |
| `ORM addr` | `ACC \|= MEM[addr]` |
| `XOR imm` | `ACC ^= imm` |
| `XORM addr` | `ACC ^= MEM[addr]` |
| `CMP imm` | `ACC = ACC - imm; carry_flag = ACC < imm` |
| `CMPM addr` | `ACC = ACC - MEM[addr]; carry_flag = ACC < MEM[addr]` |
| `CALL label` | Push return address, jump |
| `RET` | Pop return address, jump |
| `JMP label` | Unconditional jump |
| `JZ label` | Jump if `zero_flag` |
| `JNZ label` | Jump if not `zero_flag` |
| `HALT` | Stop execution |

</details>

---

## Four Execution Engines

The same compiled program runs through four independent backends. The verifier executes them in lockstep and fails on the first divergence.

| Engine | What it is | Purpose |
|--------|-----------|---------|
| **TransformerVm** | Encode-attend-FFN loop with history-aware hull-backed memory reads | The transformer runtime |
| **NativeInterpreter** | Direct ISA semantics, no transformer structure | Semantic oracle |
| **BurnRuntime** | Same compiled weights through [Burn](https://burn.dev) tensors | Tensor-level cross-check |
| **OnnxRuntime** | Exported ONNX models through [Tract](https://github.com/sonos/tract) | Portable-format cross-check |

```bash
# Pick an engine
cargo run --bin tvm -- run programs/fibonacci.tvm --engine native
cargo run --bin tvm -- run programs/fibonacci.tvm --engine transformer
cargo run --features burn-model --bin tvm -- run programs/fibonacci.tvm --engine burn
cargo run --features onnx-export --bin tvm -- run programs/fibonacci.tvm --engine onnx

# Differential verification
cargo run --bin tvm -- run programs/fibonacci.tvm --verify-native
cargo run --features full --bin tvm -- run programs/fibonacci.tvm --verify-all
```

If `--verify-all` passes and `scripts/validate_onnx.py` reproduces the result from exported ONNX files, the computation is not trapped inside custom Rust structs. It is a real, portable transformer computation with independent cross-checks.

### ONNX Export + Python Validation

```bash
# Export ONNX models
cargo run --features onnx-export --bin tvm -- export-onnx programs/fibonacci.tvm -o compiled/fibonacci

# Reproduce in Python with onnxruntime
pip install -r scripts/requirements.txt
python3 scripts/validate_onnx.py compiled/fibonacci \
  --program-name fibonacci --expected-acc 21 --expected-halted true
```

---

## The STARK Proof System

The vanilla STARK prover operates over **F_p** where p = 1 + 407 &middot; 2^119 (a 128-bit prime with a large power-of-two subgroup for NTT). The in-repo implementation includes:

| Component | What it does |
|-----------|-------------|
| `field.rs` | Montgomery-form arithmetic via `ark-ff` |
| `polynomial.rs` | Univariate polynomial ops, Lagrange interpolation |
| `ntt.rs` | Number-theoretic transform for O(n log n) polynomial multiplication |
| `multivariate.rs` | Multivariate polynomial representation for AIR constraints |
| `merkle.rs` | Blake2b Merkle trees for commitment |
| `fri.rs` | FRI protocol for low-degree testing |
| `stark.rs` | STARK prover and verifier |
| `proof.rs` | VM-specific AIR: transition constraints from instruction semantics |

The AIR encodes each supported instruction as polynomial transition constraints over the trace columns `(PC, ACC, SP, zero_flag, carry_flag, halted, MEM[0..n])`. The boundary constraints pin the initial and final machine states.

```bash
# Prove
cargo run --bin tvm -- prove-stark programs/factorial_recursive.tvm -o fact.proof.json

# Verify (does not re-execute the program)
cargo run --bin tvm -- verify-stark fact.proof.json
```

The proof is transparent and public. The claim includes the program, attention mode, step count, and final state. Zero-knowledge hiding is out of scope.

---

## Commit-and-Audit (Optimistic Verification)

For large-scale transformer inference where full STARK proofs are computationally prohibitive, this repo implements a **Commit-and-Audit** protocol (inspired by `lambdaclass/commitllm`). 

This approach provides a high-performance alternative by splitting verification into two phases:

1.  **Commit**: The provider executes the program and issues a compact **Execution Receipt**. This receipt contains Merkle roots committing to the full execution trace and the model weights used. This takes **~12µs**, making it suitable for real-time serving.
2.  **Audit**: A user or auditor can request a proof for specific steps of the execution. The provider opens the Merkle paths for those steps and provides algebraic witnesses (using **Freivalds' algorithm**) to prove matrix-multiplication consistency. Verification takes **~1µs**, allowing for massive-scale statistical auditing to "keep providers honest."

| Feature | Vanilla STARK | Commit-and-Audit |
| :--- | :--- | :--- |
| **Integrity** | Cryptographic (100%) | Statistical (Audit-based) |
| **Overhead** | High (ms to seconds) | Negligible (~12µs) |
| **Proof Size** | Large (JSON) | Tiny (Merkle Root) |
| **Best For** | High-stakes decisions | Routine LLM inference |

```bash
# Generate an optimistic receipt
cargo run --bin tvm -- commit programs/fibonacci.tvm -o fib.receipt.json

# Audit a random step to catch potential tampering
cargo run --bin tvm -- audit programs/fibonacci.tvm fib.receipt.json --step 42
```

### Proof Boundaries In Plain English


---

## Privacy & Explainability

For high-stakes decisions like medical triage or financial underwriting, the Provable Computer provides two advanced security layers:

### 1. Input Privacy (Zero-Knowledge Pattern)
Protect sensitive user data (e.g., income, health records) while still proving compliance.
- **Tech**: **Commit-then-Prove**. The user provides a hashed commitment to their data. The STARK proves the rules were followed using the secret data as a hidden witness.
- **Benefit**: "Prove you are eligible for a loan without revealing your exact income."

### 2. Provable Explainability (Reason-Bound Traces)
Eliminate "reason hallucinations" by cryptographically binding human-readable explanations to the execution logic.
- **Tech**: **Reason-Bound Logic**. Underwriting rules are compiled to write specific "Reason Codes" to memory. The STARK proves that a specific "Deny" decision is mathematically inseparable from its corresponding "Reason Code."
- **Benefit**: "The regulator receives a proof that the denial was strictly due to Debt-to-Income ratios, not a biased or random hallucination."

```bash
# Run the combined Privacy & Explainability demo
cargo run --example explainable_privacy

# Run the end-to-end Underwriting business use case
cargo run --example verifiable_underwriting
```

There are three different proof stories in this repo:

1. **Classic VM proof**

   - The STARK in `src/proof.rs` proves native VM execution.
   - The shared transformer runtime is checked against that VM behavior, but is not what this proof directly proves.

2. **Direct decision proof**

   - The direct DMN / cDMN proof path proves the decision logic more directly.
   - This path does not need the transformer backend.

3. **Poet proof**

   - `src/poet_proof.rs` directly proves a local reduced transformer-style forward pass.
   - This is real, but it is a narrower proof target than the full shared `TransformerVm` in `src/model.rs`.

If you only need to know one thing, it is this:

> "proved decision correctness" is already true for supported paths, and the supported shared `average-hard` path now also has a direct shared-forward-pass STARK. It is still not the claim for every proof path in the repo.

---

## Attention Modes

| Mode | Behavior | Flag |
|------|----------|------|
| `average-hard` | Deterministic argmax over a cell's write history via convex hull | Default |
| `softmax` | Weighted read over a cell's full write history | `--attention-mode softmax` |
| `hard-softmax:T` | Temperature-controlled blend over a cell's write history | `--attention-mode hard-softmax:10` |

These modes do not add new ISA opcodes. They change the retrieval policy used by ordinary
memory-read instructions such as `LOAD`, `ADDM`, `CMPM`, `POP`, and `RET`.

The `average-hard` mode is the only one supported by the STARK proof path. `softmax` and
`hard-softmax` are available for experimentation and comparison.

---

## Feature Flags

| Flag | Enables |
|------|---------|
| *(default)* | Transformer runtime, native interpreter, TUI, STARK prover |
| `burn-model` | Burn tensor execution engine, `--verify-burn` |
| `onnx-export` | ONNX export, Tract execution engine, `--verify-onnx` |
| `full` | All of the above, `--verify-all` |

```bash
cargo test                    # Core suite
cargo test --features full    # Everything
cargo bench                   # Hull + STARK benchmarks
```

## Development Checks

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

The CLI is self-documenting:

```bash
cargo run --bin tvm -- --help
cargo run --bin tvm -- run --help
```

---

## Repository Structure

```
src/
  assembly.rs           # .tvm parser, directives, labels
  compiler.rs           # Assembly source -> Program
  config.rs             # VM configuration, attention modes
  engine.rs             # Execution traits, trace events
  geometry.rs           # Point2D, HullKvCache (convex hull memory)
  model.rs              # 2D attention + compiled FFN blocks
  runtime.rs            # Transformer execution loop
  interpreter.rs        # Native reference interpreter
  commit_and_audit.rs   # Optimistic verification lifecycle
  freivalds.rs          # Fast matrix-vector product verification
  merkle_trace.rs       # Merkleized execution trace
  cdmn/                 # cDMN frontend, semantics, direct decision proof paths
  dmn.rs                # DMN frontend
  dmn_decision.rs       # DMN decision-program compilation/evaluation
  dmn_direct.rs         # DMN direct proof wrappers
  poet_proof.rs         # Reduced local transformer-style proof path
  state.rs              # MachineState encoding / token layout
  memory.rs             # Addressed memory with write histories
  proof.rs              # VM AIR construction, STARK integration
  verification.rs       # Lockstep multi-engine differential verification
  tui.rs                # Interactive terminal viewer
  bin/tvm.rs            # CLI (clap)
  vanillastark/         # Field, polynomial, NTT, Merkle, FRI, STARK
  burn_model.rs         # Burn Module definitions (optional)
  burn_runtime.rs       # Burn execution loop (optional)
  onnx_export.rs        # ONNX graph generation (optional)
  onnx_runtime.rs       # Tract execution runtime (optional)
tests/                  # Unit, integration, property, differential, CLI tests
programs/               # Example .tvm programs
benches/                # Criterion benchmarks
scripts/                # Python ONNX validator
```

~10k lines of Rust. ~3k lines of tests. ~3k lines of STARK internals.

---

## Current Scope

Intentionally narrow. Intentionally correct.

**Implemented:** Compact ISA with arithmetic, memory, stack, and control flow. Transformer execution with hull-backed 2D attention. Four independent execution engines with lockstep differential verification. Vanilla STARK proofs over VM execution traces. Direct DMN / cDMN decision proof paths. A reduced local Poet forward-pass proof path. Optimistic **Commit-and-Audit** protocol for scalable verifiable inference. Interactive TUI. CLI. Benchmarks. Property tests.

**Not implemented:** GPU acceleration. Learned/trained weights. Zero-knowledge proofs. WASM frontend. Production STWO prover integration. Full shared-`TransformerVm` STARK proving as the default path. Full-ISA STARK AIR for all VM and Poet instructions.

The narrowness is the point. The semantics are small enough to inspect, the test suite is strong enough to trust, and the structure is clean enough to prove over.

---

## References

- [Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers) --- Percepta. The original idea.
- [Scalable, Transparent, and Post-Quantum Secure Computational Integrity](https://eprint.iacr.org/2018/046) --- Ben-Sasson et al., 2018. The STARK protocol.
- [Circle STARKs](https://eprint.iacr.org/2024/278) --- Haböck, Levit, Papini, 2024. STARKs over Mersenne prime fields.
- [STWO Prover](https://github.com/starkware-libs/stwo) --- StarkWare's production Circle STARK prover.
- [Anatomy of a STARK](https://aszepieniec.github.io/stark-anatomy/) --- Aszepieniec. The reference implementation this STARK is ported from.

## License

MIT
