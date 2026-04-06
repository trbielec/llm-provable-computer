// CONCEPTUAL EXAMPLE — CommitLLM lineage
//
// This example simulates a workflow where an upstream LLM proposes values that are
// then verified by a deterministic TVM constraint program. The "LLM" here is
// a hardcoded stand-in; no real language model is integrated or called.
//
// What this example actually proves:
//   - The TVM executed the budget_verifier program correctly for the given inputs.
//   - The STARK proof attests to the execution trace, not to the LLM's reasoning process.
//
// Future work (not demonstrated here): connecting a live model output to TVM inputs
// such that the audited model boundary is cryptographically bound, not simulated.

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

    // 2. Simulate a Stochastic "LLM Reasoner" Output (SIMULATED — not a real LLM call)
    // ----------------------------------------------------------------------------------
    // In a real integration an upstream model would supply these values at runtime.
    // Here they are hardcoded to illustrate the audit workflow.
    //
    // Simulated proposal: "Propose a budget where R&D is twice Marketing,
    // and total spend equals 9000."
    // Simulated values: {Marketing: 3000, Research: 6000, Total: 9000}
    println!("--- Verifiable AI Reasoner Audit (Conceptual Example) ---");
    println!("Simulated upstream proposal: Marketing=3000, Research=6000, Total=9000");
    println!("Note: the upstream LLM boundary is simulated; no real model is integrated here.");

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
        // What this proves: the TVM executed the budget constraint program correctly for
        // the given input values. It does NOT attest to the LLM's internal reasoning or
        // that the simulated proposal came from a real model.
        println!("The TVM execution of the constraint verifier is cryptographically attested.");
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
        println!("Demonstrated: a deterministic TVM constraint program can audit");
        println!("values supplied by an upstream source (here, a simulated proposal).");
    } else {
        println!("Proof Verification FAILED!");
    }
}
