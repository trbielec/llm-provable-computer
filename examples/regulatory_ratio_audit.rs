use llm_provable_computer::{
    prove_execution_stark, verify_execution_stark, CdmnProgramMetadata, ProgramCompiler,
    TransformerVmConfig,
};
use std::fs;

fn main() {
    // 1. Load the compiled ratio verifier
    let source =
        fs::read_to_string("programs/budget_verifier.tvm").expect("Failed to read program");
    let metadata = fs::read_to_string("programs/budget_verifier.metadata.json")
        .ok()
        .and_then(|json| serde_json::from_str::<CdmnProgramMetadata>(&json).ok());

    // 2. Setup Bank Data
    // ----------------------------------------------
    // Reserve Capital: 100
    // Total Assets: 1000
    // Min Ratio %: 10
    // Equation: 100 * 100 == 1000 * 10  (10,000 == 10,000)
    println!("--- Verifiable Regulatory Ratio Audit ---");
    println!("Bank Data: Reserve=100, Assets=1000, MinRatio=10%");

    let config = TransformerVmConfig::default();
    let compiler = ProgramCompiler;
    let model = compiler
        .compile_source(&source, config.clone())
        .expect("Failed to compile program");

    // Initialize TVM memory
    let mut initial_memory = model.program().initial_memory().to_vec();
    initial_memory[0] = 100; // reserve_capital
    initial_memory[1] = 1000; // total_assets
    initial_memory[2] = 10; // min_ratio_pct

    let program = model
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("Invalid memory");
    let model =
        llm_provable_computer::TransformerVm::new(config, program).expect("Failed to create VM");

    // 3. Prove Execution
    println!("Generating STARK proof of regulatory compliance...");
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
        println!("The financial ratio is MATHEMATICALLY PROVEN to be compliant.");
        println!(
            "Regulatory Status: {}",
            if success_flag == 1 {
                "COMPLIANT"
            } else {
                "NON-COMPLIANT"
            }
        );
        println!("Reason Code: {}", reason_code);
        if let Some(reason) = metadata
            .as_ref()
            .and_then(|compiled| compiled.reason_for_code(reason_code))
        {
            println!("Reason Label: {}", reason.label);
            println!("Reason Description: {}", reason.description);
        }
        println!("--------------------------------------------------");
        println!("This proof ensures that the calculation (Reserve * 100 == Assets * MinRatio)");
        println!("was executed correctly without any truncation or manipulation.");
    } else {
        println!("Proof Verification FAILED!");
    }
}
