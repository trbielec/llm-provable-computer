use llm_provable_computer::commit_and_audit::{
    audit_execution, commit_execution, verify_audit, AuditRequest, AuditTarget,
};
use llm_provable_computer::compiler::ProgramCompiler;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::model::TransformerVm;
use llm_provable_computer::proof::{prove_execution_stark, verify_execution_stark};
use llm_provable_computer::runtime::ExecutionRuntime;
use std::time::Instant;

/// BUSINESS USE CASE: Automated Verifiable Loan Underwriting
///
/// This example demonstrates a hybrid proof architecture for a high-stakes financial decision:
/// 1. Low-Overhead Commitment: The provider issues a Merkleized receipt for the full underwriting trace.
/// 2. Statistical Audit: A random check using Freivalds' algorithm ensures matrix consistency.
/// 3. High-Stakes Proof: For the final approved/denied decision, a full STARK proof is generated.
///
/// Note on performance figures: the overhead table below is illustrative only.
/// No benchmark harness currently measures commit latency or throughput at scale.
/// Treat the relative numbers as architectural intent, not measured results.
fn main() -> llm_provable_computer::Result<()> {
    println!("--- Verifiable Automated Underwriting End-to-End ---");

    // 1. Define and Compile the Underwriting Rules (Assembly)
    let assembly = r#"
        .memory 8
        .init 0 50000  ; Income
        .init 1 2000   ; Monthly Debt
        .init 2 720    ; Credit Score
        .init 3 0      ; Final Decision (0=Pending, 1=Approved, 2=Denied)

        ; --- Underwriting Logic ---
        ; Check Credit Score < 600
        LOAD 2
        LOADI 600
        CMP 600
        JZ deny_low_credit

        ; Calculate DTI (Debt-to-Income)
        ; Monthly Debt * 12 / Income < 0.43
        LOAD 1
        LOADI 12
        MUL 12
        STORE 4        ; Annual Debt
        LOAD 0
        LOADI 100
        MUL 100
        STORE 5        ; Income * 100
        LOAD 4
        LOADI 10000
        MUL 10000
        STORE 6        ; Annual Debt * 10000
        ; (Simplified integer ratio check)
        
        ; Final Decision
        LOADI 1
        STORE 3        ; Approved
        HALT

    deny_low_credit:
        LOADI 2
        STORE 3        ; Denied
        HALT
    "#;

    let config = TransformerVmConfig::default();
    let compiled = ProgramCompiler.compile_source(assembly, config.clone())?;
    let vm = TransformerVm::new(config, compiled.program().clone())?;

    println!("Step 1: Executing Underwriting Model...");
    let mut runtime = ExecutionRuntime::new(vm.clone(), 512);
    let start = Instant::now();
    let (result, witness) = runtime.run_with_witness()?;
    let exec_time = start.elapsed();
    println!(
        "Execution Complete. Decision: {}, Steps: {}, Time: {:?}",
        result.final_state.acc, result.steps, exec_time
    );

    // 2. Commit Phase (Optimistic)
    println!("\nStep 2: Issuing Cryptographic Receipt (Commit)...");
    let start = Instant::now();
    let (receipt, m_trace, w_commitment) = commit_execution(&vm, &witness)?;
    let commit_time = start.elapsed();
    println!(
        "Receipt Generated. Trace Root: 0x{}...",
        hex::encode(&receipt.trace_root[..8])
    );
    println!("Commitment Overhead: {:?}", commit_time);

    // 3. Audit Phase (Statistical)
    println!("\nStep 3: Performing Random Audit of Step 5...");
    let request = AuditRequest {
        target: AuditTarget::Steps(vec![5]),
        challenge: None, // In production, this would include a Freivalds challenge
        approximate_config: None,
    };
    let staged_trace =
        llm_provable_computer::transformer_trace::build_shared_transformer_staged_trace(
            &witness,
            vm.config().ff_dim,
        )?;
    let response = audit_execution(
        &vm,
        &m_trace,
        &w_commitment,
        &witness,
        &staged_trace,
        &request,
    )?;

    let is_valid = verify_audit(&receipt, &response, None);
    println!("Audit Step 5 Verified: {}", is_valid);

    // 4. Full STARK Proof (Maximal Soundness)
    println!("\nStep 4: Generating full STARK proof for high-stakes finality...");
    let start = Instant::now();
    let proof = prove_execution_stark(&vm, 512)?;
    let proof_time = start.elapsed();

    let verified = verify_execution_stark(&proof)?;
    println!("STARK Proof Verified: {}", verified);
    println!("STARK Proving Time: {:?}", proof_time);
    println!("Proof Size: {} bytes", proof.proof.len());

    // 5. Value Proposition Summary (measured on this run)
    let exec_ns = exec_time.as_nanos().max(1);
    let commit_ratio = commit_time.as_nanos() as f64 / exec_ns as f64;
    let stark_ratio = proof_time.as_nanos() as f64 / exec_ns as f64;

    println!("\n--- Value Proposition (latency measured on this run, release build) ---");
    println!(
        "| Method           | Soundness    | Latency (measured)    | Use Case                |"
    );
    println!(
        "|------------------|--------------|-----------------------|-------------------------|"
    );
    println!(
        "| Raw Execution    | Trust-based  | {:>10?} (1x)      | Local testing           |",
        exec_time
    );
    println!(
        "| Commit-and-Audit | Statistical  | {:>10?} ({:.0}x)     | Bulk model serving      |",
        commit_time, commit_ratio
    );
    println!(
        "| Vanilla STARK    | Cryptographic| {:>10?} ({:.0}x)  | Final high-stakes audit |",
        proof_time, stark_ratio
    );
    println!();
    println!("Single-run timings on this machine. For stable numbers, run the criterion");
    println!("benchmarks: `cargo bench --bench <name>`.");

    Ok(())
}
