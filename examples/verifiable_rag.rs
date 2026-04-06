// CONCEPTUAL EXAMPLE — Percepta lineage (HullKvCache / retrieval-history)
//
// This example demonstrates HullKvCache 2D attention as a retrieval mechanism.
// The "RAG" framing is illustrative; this is not an integration with a real
// retrieval-augmented generation pipeline.
//
// Proof-path caveat:
//   The STARK proof at the end of this example runs the TVM interpreter from a
//   fresh default state, NOT from the manually-populated memory history above it.
//   Proving the retrieval from a pre-populated history requires passing that history
//   to the prover — that path is not yet wired end-to-end. The proof here attests
//   to a trivial execution only.

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

    let model = ProgramCompiler
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

    // 5. Generate and Verify STARK Proof (fresh model — see file-level caveat)
    // -----------------------------------------------------------------------
    // prove_execution_stark runs the TVM interpreter from the model's default
    // initial state, which does NOT include the manually-populated history above.
    // This proof attests that the compiled program executes correctly from that
    // fresh state; it does NOT prove the Hull retrieval demonstrated in steps 3–4.
    //
    // Wiring the manually-populated history into the prover is future work.

    println!("\nGenerating proof of compiled program execution (fresh-state, not retrieval history)...");
    let proof = llm_provable_computer::prove_execution_stark(&model, 10).expect("Prover failed");
    let is_valid = llm_provable_computer::verify_execution_stark(&proof).expect("Verifier failed");

    if is_valid {
        println!("STARK Proof Verified: the compiled program executed correctly from its default initial state.");
        println!("(The Hull retrieval above is demonstrated but not yet part of this proof.)");
    }
}
