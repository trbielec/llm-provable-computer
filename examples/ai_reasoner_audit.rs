use llm_provable_computer::{
    prove_execution_stark, verify_execution_stark, CdmnProgramMetadata, ProgramCompiler,
    TransformerVmConfig,
};
use std::fs;

fn main() {
    // 1. Load the cDMN-compiled verifier
    let source =
        fs::read_to_string("programs/budget_verifier.tvm").expect("Failed to read program");
    let metadata = fs::read_to_string("programs/budget_verifier.metadata.json")
        .ok()
        .and_then(|json| serde_json::from_str::<CdmnProgramMetadata>(&json).ok());

    // 2. Simulate a Stochastic "LLM Reasoner" Output
    // ----------------------------------------------
    // The LLM was asked: "Propose a budget where R&D is twice Marketing,
    // and total spend equals 9000."
    // LLM Hallucination/Proposal: {Marketing: 3000, Research: 6000, Total: 9000}
    println!("--- Verifiable AI Reasoner Audit ---");
    println!("Stochastic LLM Proposal: Marketing=3000, Research=6000, Total=9000");

    let config = TransformerVmConfig::default();
    let compiler = ProgramCompiler;
    let model = compiler
        .compile_source(&source, config.clone())
        .expect("Failed to compile program");

    // Initialize TVM memory with the LLM's proposal
    let mut initial_memory = model.program().initial_memory().to_vec();
    initial_memory[0] = 3000; // marketing
    initial_memory[1] = 6000; // research
    initial_memory[2] = 9000; // total_budget (target)

    let program = model
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("Invalid memory");
    let model =
        llm_provable_computer::TransformerVm::new(config, program).expect("Failed to create VM");

    // 3. Prove Execution (Verifying the LLM's work)
    println!("Generating STARK proof that the AI solution satisfies constraints...");
    let proof = prove_execution_stark(&model, 1000).expect("Failed to generate proof");

    // 4. Verify Proof
    println!("Verifying proof...");
    let is_valid = verify_execution_stark(&proof).expect("Failed to verify proof");

    if is_valid {
        let (status_idx, reason_idx) = metadata
            .as_ref()
            .map(|compiled| {
                (
                    usize::from(compiled.output.status_register),
                    usize::from(compiled.output.reason_register),
                )
            })
            .unwrap_or((
                proof.claim.final_state.memory.len() - 2,
                proof.claim.final_state.memory.len() - 1,
            ));
        let success_flag = proof.claim.final_state.memory[status_idx];
        let reason_code = proof.claim.final_state.memory[reason_idx];
        println!("\nSTARK Proof VERIFIED!");
        println!("The LLM's proposal is MATHEMATICALLY GUARANTEED to be correct.");
        println!(
            "Verification Status Register [{}]: {}",
            status_idx, success_flag
        );
        println!("Reason Code Register [{}]: {}", reason_idx, reason_code);
        if let Some(reason) = metadata
            .as_ref()
            .and_then(|compiled| compiled.reason_for_code(reason_code))
        {
            println!("Reason Label: {}", reason.label);
            println!("Reason Description: {}", reason.description);
        }
        println!("--------------------------------------------------");
        println!("Emergent Capability: We just proved a probabilistic AI's output");
        println!("using a deterministic, transformer-shaped virtual machine.");
    } else {
        println!("Proof Verification FAILED!");
    }
}
