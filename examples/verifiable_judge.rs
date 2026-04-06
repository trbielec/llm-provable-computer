use llm_provable_computer::{
    prove_execution_stark, verify_execution_stark, ProgramCompiler, TransformerVmConfig,
};
use std::fs;

fn main() {
    // 1. Load the compliance judge program
    let source =
        fs::read_to_string("programs/compliance_judge.tvm").expect("Failed to read program");

    // 2. Define our inputs (Credit Score: 700, Income: 25000, Debt: 5000)
    let model = ProgramCompiler
        .compile_source(&source, TransformerVmConfig::default())
        .expect("Failed to compile program");

    // Manually set input memory values
    // Memory 0: Credit Score = 700
    // Memory 1: Income = 25000
    // Memory 2: Debt = 5000
    let mut initial_memory = model.program().initial_memory().to_vec();
    initial_memory[0] = 700;
    initial_memory[1] = 25000;
    initial_memory[2] = 5000;

    // Re-create model with specific memory (normally would use .init in .tvm)
    let program = model
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("Invalid memory");
    let model = llm_provable_computer::TransformerVm::new(TransformerVmConfig::default(), program)
        .expect("Failed to create VM");

    println!("--- Verifiable Compliance Judge ---");
    println!("Inputs: Credit Score=700, Income=45000, Debt=5000");

    // 3. Prove Execution
    println!("Generating STARK proof of the compliance decision...");
    let proof = prove_execution_stark(&model, 1000).expect("Failed to generate proof");

    // 4. Verify Proof
    println!("Verifying proof (without re-executing)...");
    let is_valid = verify_execution_stark(&proof).expect("Failed to verify proof");

    if is_valid {
        let result = proof.claim.final_state.memory[3];
        let status = if result == 1 { "APPROVED" } else { "REJECTED" };
        println!("Proof Verified! Decision: {}", status);
        println!("The proof mathematically confirms this decision followed the logic in 'compliance_judge.tvm'.");
    } else {
        println!("Proof Verification FAILED!");
    }
}
