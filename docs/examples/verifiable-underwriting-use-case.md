# Verifiable Automated Underwriting: End-to-End Use Case

This example demonstrates how the **Provable Computer** solves the "Integrity Gap" in automated financial decision-making using a hybrid proof architecture.

## The Business Problem
A financial institution uses a Transformer-based model to automate loan underwriting. To satisfy regulatory requirements and maintain customer trust, every decision must be:
1.  **Deterministic**: The same input must always produce the same decision.
2.  **Verifiable**: The bank must be able to prove that the specific rules (e.g., credit score checks, DTI ratios) were applied correctly without tampering.
3.  **Performant**: Verification cannot slow down the real-time approval process for thousands of applications.

## The Hybrid Solution

### 1. Low-Overhead Commitment (Optimistic)
For every application, the system issues an **Execution Receipt**.
- **Tech**: Merkle Tree commitment to the execution trace and model weights.
- **Latency**: ~12µs (negligible).
- **Value**: Provides a cryptographic "anchor" that can be audited later. It proves the bank is *claiming* a specific execution path.

### 2. Statistical Auditing (Spot Checks)
Regulators or internal audit teams can request proofs for random steps of any receipt.
- **Tech**: **Freivalds' Algorithm** for $O(d)$ matrix product verification and Merkle path opening.
- **Latency**: ~1µs.
- **Value**: Catches "lazy" or malicious providers who might use cheaper, non-compliant models or bypass rules to save costs.

### 3. Full STARK Proof (The "Nuclear Option")
For contested decisions or high-value loans, the system generates a full ZK-STARK proof.
- **Tech**: STARK-based proof of the entire Transformer VM execution.
- **Latency**: Seconds.
- **Value**: Provides **100% mathematical certainty** that the decision followed the program instructions bit-for-bit.

## Benchmarks (Empirical)

Based on the `verifiable_underwriting` example:

| Method | Latency (ms) | Throughput (req/s) | Soundness |
| :--- | :--- | :--- | :--- |
| **Raw Execution** | ~9ms | ~110 | Trust-based |
| **Commit-and-Audit** | **~9.3ms** | **~105** | **Statistical** |
| **Vanilla STARK** | ~6,800ms | ~0.15 | **Cryptographic** |

## How to Run
Run the end-to-end demo:
```bash
cargo run --example verifiable_underwriting
```

The output will show the transition from a fast optimistic receipt to a full cryptographic proof, proving the decision: `Denied` (due to credit score < 600 in the example data).
