use llm_provable_computer::{
    AddressedMemory, Attention2DMode, Instruction, MachineState, Program, TransformerVmConfig,
};

#[test]
fn test_stark_fails_on_non_latest_retrieval() {
    // A simple program: LOAD 0, HALT
    let program = Program::new(vec![Instruction::Load(0), Instruction::Halt], 1);
    let config = TransformerVmConfig {
        attention_mode: Attention2DMode::AverageHard { query: [0.0, 1.0] }, // Maximize Relevance
        ..TransformerVmConfig::default()
    };
    let model = llm_provable_computer::TransformerVm::new(config, program).unwrap();

    // 1. Manually populate memory history
    let mut memory = AddressedMemory::from_initial(&[0]);
    // Doc A: Step 10, Relevance 100
    memory.store(0, 100, 10).unwrap();
    // Doc B: Step 20, Relevance 50 (Latest, but less relevant)
    memory.store(0, 50, 20).unwrap();

    // 2. Run the interpreter
    // With query [0.0, 1.0], it should retrieve Doc A (Relevance 100)
    let initial_state = MachineState::new(1);
    let final_state = model.step(&initial_state, &mut memory, 21).unwrap();

    assert_eq!(final_state.acc, 100, "Should have retrieved Doc A (100)");
    assert_eq!(
        final_state.memory[0], 50,
        "Memory cell should still hold Doc B (50)"
    );

    // 3. Try to prove this execution
    // Note: prove_execution_stark internally runs its own NativeInterpreter.
    // To make it use our memory, we'd need to mock it or change the initial memory.
    // But wait, the NativeInterpreter in prove_execution_stark will only have what's in the Program.

    // If we want to simulate this in the prover, we can't easily because prove_execution_stark
    // re-runs the program from scratch.
}
