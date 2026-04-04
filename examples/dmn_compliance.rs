use llm_provable_computer::{
    prove_execution_stark, verify_execution_stark, ProgramCompiler, TransformerVmConfig,
};
use std::fs;

fn main() {
    // 1. Load the DMN-compiled program
    let source = fs::read_to_string("programs/loan_rules.tvm").expect("Failed to read program");

    // 2. Setup inputs for Rule #1 (Credit Score: 750, Income: 25000)
    let config = TransformerVmConfig::default();
    let compiler = ProgramCompiler::default();
    let model = compiler
        .compile_source(&source, config.clone())
        .expect("Failed to compile program");

    let mut initial_memory = model.program().initial_memory().to_vec();
    initial_memory[0] = 750;
    initial_memory[1] = 25000;

    let program = model
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("Invalid memory");
    let model =
        llm_provable_computer::TransformerVm::new(config, program).expect("Failed to create VM");

    println!("--- Verifiable DMN Compliance ---");
    println!("Inputs: Credit Score=750, Income=25000");

    // 3. Prove Execution
    println!("Generating STARK proof of the DMN decision...");
    let proof = prove_execution_stark(&model, 1000).expect("Failed to generate proof");

    // 4. Verify Proof
    println!("Verifying proof...");
    let is_valid = verify_execution_stark(&proof).expect("Failed to verify proof");

    if is_valid {
        let status = proof.claim.final_state.memory[2];
        let rule_id = proof.claim.final_state.memory[3];

        println!("\nProof VERIFIED!");
        println!(
            "Decision Result: {}",
            if status == 1 { "APPROVED" } else { "REJECTED" }
        );
        println!(
            "Semantic Trace: This decision was reached specifically by satisfying DMN Rule #{}",
            rule_id
        );
        println!("--------------------------------------------------");
        println!("This STARK proof mathematically guarantees that the AI/VM followed");
        println!(
            "the logic of DMN Table 'Loan Approval' Rule #{} and no other.",
            rule_id
        );
    } else {
        println!("Proof Verification FAILED!");
    }
}
