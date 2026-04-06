use std::sync::Arc;
use llm_provable_computer::{
    ProgramCompiler, TransformerVmConfig, MachineState,
};
use llm_provable_computer::batch::BatchRunner;

#[test]
fn test_heterogeneous_batch_cpu() {
    // Prog A: 1 + 10 = 11
    let prog_a = r#"
        LOADI 1
        ADD 10
        HALT
    "#;
    // Prog B: 2 * 5 = 10, using a loop to test offset logic
    let prog_b = r#"
        .MEMORY 16
        .INIT 10 5
        .INIT 11 0
        loop:
            LOAD 10
            JZ done
            SUB 1
            STORE 10
            LOAD 11
            ADD 2
            STORE 11
            JMP loop
        done:
            LOAD 11
            HALT
    "#;

    let config = TransformerVmConfig::default();
    let compiler = ProgramCompiler;
    let (model_inner, entry_points) = compiler.compile_sources(&[prog_a, prog_b], config).unwrap();
    let model = Arc::new(model_inner);

    // Initial states: 2 runs for prog_a, 2 runs for prog_b
    let mut initial_states = Vec::new();
    let initial_memory = model.program().initial_memory().to_vec();
    
    // Prog A: starts at entry_points[0], acc should be 11
    for _ in 0..2 {
        let mut s = MachineState::with_memory(initial_memory.clone());
        s.pc = entry_points[0];
        initial_states.push(s);
    }
    
    // Prog B: starts at entry_points[1], acc should be 10
    for _ in 0..2 {
        let mut s = MachineState::with_memory(initial_memory.clone());
        s.pc = entry_points[1];
        initial_states.push(s);
    }

    let runner = BatchRunner::new(model, 100);
    let results = runner.run(initial_states);

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].final_state.acc, 11);
    assert_eq!(results[1].final_state.acc, 11);
    assert_eq!(results[2].final_state.acc, 10);
    assert_eq!(results[3].final_state.acc, 10);
}

#[cfg(all(feature = "metal-batch", target_os = "macos"))]
#[test]
fn test_heterogeneous_batch_metal() {
    use llm_provable_computer::metal_batch::MetalBatchRunner;

    let prog_a = r#"
        LOADI 1
        ADD 10
        HALT
    "#;
    let prog_b = r#"
        .MEMORY 16
        .INIT 10 5
        .INIT 11 0
        loop:
            LOAD 10
            JZ done
            SUB 1
            STORE 10
            LOAD 11
            ADD 2
            STORE 11
            JMP loop
        done:
            LOAD 11
            HALT
    "#;

    let config = TransformerVmConfig::default();
    let compiler = ProgramCompiler;
    let (model_inner, entry_points) = compiler.compile_sources(&[prog_a, prog_b], config).unwrap();
    let model = Arc::new(model_inner);

    let mut initial_states = Vec::new();
    let initial_memory = model.program().initial_memory().to_vec();

    for _ in 0..2 {
        let mut s = MachineState::with_memory(initial_memory.clone());
        s.pc = entry_points[0];
        initial_states.push(s);
    }
    for _ in 0..2 {
        let mut s = MachineState::with_memory(initial_memory.clone());
        s.pc = entry_points[1];
        initial_states.push(s);
    }

    let runner = MetalBatchRunner::new(model, 100).unwrap();
    let results = runner.run(initial_states);

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].final_state.acc, 11);
    assert_eq!(results[1].final_state.acc, 11);
    assert_eq!(results[2].final_state.acc, 10);
    assert_eq!(results[3].final_state.acc, 10);
}
