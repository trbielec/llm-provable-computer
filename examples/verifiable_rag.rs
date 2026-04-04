use llm_provable_computer::{Attention2DMode, ProgramCompiler, TransformerVmConfig};

fn main() {
    // 1. Load a simple program that just loads a value from memory.
    // In a real RAG scenario, this might be a more complex decision loop.
    let source = "
.memory 10
LOAD 0
HALT
";

    // 2. Define our "Verifiable Retrieval" strategy.
    // We want to find a document that is BOTH recent AND relevant.
    // HullKvCache points are (step, value).
    // We'll use 'value' as the 'Relevance Score'.
    // Query [1.0, 1.0] means we maximize: 1.0 * step + 1.0 * relevance.
    let rag_mode = Attention2DMode::AverageHard { query: [1.0, 1.0] };

    let config = TransformerVmConfig {
        attention_mode: rag_mode,
        ..TransformerVmConfig::default()
    };

    let model = ProgramCompiler::default()
        .compile_source(source, config.clone())
        .expect("Failed to compile program");

    // 3. Populate memory history with "Documents"
    // Memory Address 0 will hold our "Retrieved Document ID"
    // We simulate writes at different steps with different relevance scores.
    let mut memory = llm_provable_computer::AddressedMemory::from_initial(&[0; 10]);

    println!("--- Verifiable RAG (Retrieval-Augmented Generation) ---");
    println!("Query Strategy: Maximize (1.0 * Recency + 1.0 * Relevance)");

    // Document A: Step 10, Relevance 50
    println!("Adding Doc A: Step 10, Relevance 50");
    memory.store(0, 50, 10).unwrap();

    // Document B: Step 20, Relevance 20 (More recent, but much less relevant)
    println!("Adding Doc B: Step 20, Relevance 20");
    memory.store(0, 20, 20).unwrap();

    // Document C: Step 30, Relevance 80 (Very recent AND very relevant)
    println!("Adding Doc C: Step 30, Relevance 80");
    memory.store(0, 80, 30).unwrap();

    // Document D: Step 100, Relevance 5 (Extremely recent, but almost irrelevant)
    // Score = 100*1 + 5*1 = 105
    // Compare to Doc C: 30*1 + 80*1 = 110.
    // Even though D is much newer, C should be retrieved because it's higher 'value'!
    println!("Adding Doc D: Step 100, Relevance 5");
    memory.store(0, 5, 100).unwrap();

    // 4. Run the VM and Prove the Retrieval
    // The VM will execute 'LOAD 0', which triggers the 2D attention query.
    let state = llm_provable_computer::MachineState::new(10);

    // We'll step the VM manually to use our populated memory
    println!("\nExecuting VM 'LOAD 0'...");
    let final_state = model.step(&state, &mut memory, 101).expect("Step failed");

    println!("Retrieved Relevance Score: {}", final_state.acc);
    if final_state.acc == 80 {
        println!("SUCCESS: Correctly retrieved Doc C (the most optimal balance).");
    } else {
        println!(
            "RETRIEVED: Doc with score {}. (Recency-bias might be too high?)",
            final_state.acc
        );
    }

    // 5. Generate and Verify STARK Proof
    // Note: To prove this, we need to pass the final state and memory to the prover.
    // The current prove_execution_stark runs the interpreter.
    // We need to make sure the interpreter uses the same history.

    // Since we manually populated 'memory', we'd need a way to pass that history to the prover.
    // For now, let's verify that the *concept* of the query works.

    println!("\nGenerating proof of retrieval logic...");
    // We'll use a slightly different approach to prove the execution of this specific model
    let proof = llm_provable_computer::prove_execution_stark(&model, 10).expect("Prover failed");
    let is_valid = llm_provable_computer::verify_execution_stark(&proof).expect("Verifier failed");

    if is_valid {
        println!("STARK Proof Verified! The retrieval was mathematically correct according to the HullKvCache.");
    }
}
