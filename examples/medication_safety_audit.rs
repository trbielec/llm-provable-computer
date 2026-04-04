use std::fs;

use llm_provable_computer::{
    compile_cdmn_table_with_metadata, prove_execution_stark, verify_execution_stark, CdmnTable,
    ProgramCompiler, TransformerVm, TransformerVmConfig,
};

fn main() {
    let cdmn_json =
        fs::read_to_string("programs/medication_safety.cdmn.json").expect("Failed to read cDMN");
    let table: CdmnTable = serde_json::from_str(&cdmn_json).expect("Failed to parse cDMN");
    let compiled = compile_cdmn_table_with_metadata(&table).expect("Failed to compile cDMN");

    println!("--- Verifiable Medication Safety Audit ---");
    println!(
        "Case: max dose 500, required class 11, backup class 12, prohibited class 99, medications [(400, 7), (450, 12)]"
    );

    let config = TransformerVmConfig::default();
    let model = ProgramCompiler::default()
        .compile_source(&compiled.assembly, config.clone())
        .expect("Failed to compile program");

    let mut initial_memory = model.program().initial_memory().to_vec();
    initial_memory[0] = 500;
    initial_memory[1] = 11;
    initial_memory[2] = 12;
    initial_memory[3] = 99;
    initial_memory[4] = 2;
    initial_memory[5] = 400;
    initial_memory[6] = 7;
    initial_memory[7] = 450;
    initial_memory[8] = 12;

    let program = model
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("Invalid memory");
    let model = TransformerVm::new(config, program).expect("Failed to create VM");

    println!("Generating STARK proof of the medication safety decision...");
    let proof = prove_execution_stark(&model, 1000).expect("Failed to generate proof");

    println!("Verifying proof...");
    let is_valid = verify_execution_stark(&proof).expect("Failed to verify proof");

    if is_valid {
        let status_idx = usize::from(compiled.output.status_register);
        let reason_idx = usize::from(compiled.output.reason_register);
        let status = proof.claim.final_state.memory[status_idx];
        let reason_code = proof.claim.final_state.memory[reason_idx];
        let reason = compiled.reason_for_code(reason_code);
        println!("\nSTARK Proof VERIFIED!");
        println!(
            "Medication policy status: {}",
            if status == 1 {
                "COMPLIANT"
            } else {
                "NON-COMPLIANT"
            }
        );
        println!("Reason code: {}", reason_code);
        if let Some(reason) = reason {
            println!("Reason label: {}", reason.label);
            println!("Reason description: {}", reason.description);
        }
        println!(
            "The proof confirms the bounded medication list satisfied the quantified cDMN policy, including the required-or-backup class rule and the prohibited-class exclusion."
        );
    } else {
        println!("Proof Verification FAILED!");
    }
}
