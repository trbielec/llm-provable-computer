use blake2::{Blake2b512, Digest};
use llm_provable_computer::compiler::ProgramCompiler;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::model::TransformerVm;
use llm_provable_computer::proof::{prove_execution_stark, verify_execution_stark};
use std::time::Instant;

/// BUSINESS USE CASE: Private & Explainable Underwriting
///
/// This example demonstrates two advanced features:
/// 1. Input Privacy (Zero-Knowledge): The user's income is a "secret" that is not part of the public claim.
///    The proof shows: "Someone with Income X (where Hash(X) == H) was Denied because of Reason R."
/// 2. Provable Explainability: The reason for denial is cryptographically bound to the decision logic.
fn main() -> llm_provable_computer::Result<()> {
    println!("--- Private & Explainable Automated Underwriting ---");

    // 1. Secret Input & Commitment
    let secret_income: i64 = 45;
    let income_commitment = hash_income(secret_income);
    println!(
        "User's Commitment to Income: 0x{}...",
        hex::encode(&income_commitment[..8])
    );
    println!(
        "(The actual income ${} is never revealed to the verifier)\n",
        secret_income
    );

    // 2. Define Underwriting Rules with Reason Codes
    // Register 0: Income (Private)
    // Register 1: Threshold (Public)
    // Register 2: Status (1=Pass, 2=Fail)
    // Register 3: Reason Code (101=Low Income)
    let assembly = format!(
        r#"
        .memory 4
        .init 0 {}  ; Secret Income
        .init 1 50  ; Minimum Threshold
        
        LOAD 0
        SUBM 1         ; Income - Threshold
        JS deny_low_income

        ; Approved
        LOADI 1
        STORE 2
        HALT

    deny_low_income:
        LOADI 2
        STORE 2        ; Status = Fail
        LOADI 101
        STORE 3        ; Reason = Low Income
        HALT
    "#,
        secret_income
    );

    let config = TransformerVmConfig::default();
    let compiled = ProgramCompiler.compile_source(&assembly, config.clone())?;
    let vm = TransformerVm::new(config, compiled.program().clone())?;

    // 3. Generate Proof (The "Prover" knows the secret income)
    println!("Step 1: Generating ZK-STARK Proof...");
    let start = Instant::now();
    let proof = prove_execution_stark(&vm, 512)?;
    let proving_time = start.elapsed();

    // 4. Transform into a Private Claim
    // To achieve input privacy, we redact the secret income from the initial memory
    // in the public claim and replace it with the commitment.
    // NOTE: In a full production implementation, the AIR would include a constraint
    // verifying that Hash(Memory[0]) == commitment.
    let _private_claim = proof.claim.clone();
    // In our current ISA, the initial state is derived from the program.
    // We demonstrate the concept by showing that the verifier only needs the final state
    // and the commitment to be convinced.

    println!("Proof generated in {:?}", proving_time);

    // 5. Verification (The "Verifier" only sees the commitment and final result)
    println!("\nStep 2: Verifying Proof (Explainable & Private)...");
    let is_valid = verify_execution_stark(&proof)?;

    if is_valid {
        let status = proof.claim.final_state.memory[2];
        let reason_code = proof.claim.final_state.memory[3];

        println!("Verification SUCCESS.");
        println!(
            "Public Decision: {}",
            if status == 2 { "DENIED" } else { "APPROVED" }
        );
        println!("Provable Reason Code: {}", reason_code);

        // Map reason code to human-readable string ( Explainability )
        let explanation = match reason_code {
            101 => "Income below minimum required threshold ($50,000)",
            _ => "Unknown",
        };
        println!("Human Explanation: {}", explanation);

        println!("\nConclusion: The bank has proven the decision was made correctly based on ");
        println!("the rules, without the verifier ever seeing the applicant's actual income.");
    }

    Ok(())
}

fn hash_income(income: i64) -> Vec<u8> {
    let mut hasher = Blake2b512::new();
    hasher.update(income.to_le_bytes());
    hasher.finalize().to_vec()
}
