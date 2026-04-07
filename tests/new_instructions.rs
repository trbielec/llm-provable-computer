//! Integration tests for the new TVM instructions: local variables, structured
//! loops, and externally-committed I/O.

use llm_provable_computer::assembly::parse_program;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::instruction::Instruction;
use llm_provable_computer::interpreter::NativeInterpreter;
#[allow(unused_imports)]
use llm_provable_computer::model::TransformerVm;
use llm_provable_computer::proof::{
    prove_execution_stark, verify_execution_stark, VanillaStarkProofOptions,
};
use llm_provable_computer::state::MachineState;
use llm_provable_computer::{prove_io_execution_stark, verify_io_execution_stark_full, IoBuffer};

// ---------------------------------------------------------------------------
// Assembly parse tests
// ---------------------------------------------------------------------------

#[test]
fn parse_loop_endloop_are_nullary() {
    let prog = parse_program("LOOP\nENDLOOP\nHALT").unwrap();
    assert_eq!(prog.instructions()[0], Instruction::Loop);
    assert_eq!(prog.instructions()[1], Instruction::EndLoop);
}

#[test]
fn parse_brif_with_numeric_target() {
    let prog = parse_program("BRIF 0\nHALT").unwrap();
    assert_eq!(prog.instructions()[0], Instruction::BranchIfNotZero(0));
}

#[test]
fn parse_brif_with_label() {
    let prog = parse_program("loop_top: LOOP\nBRIF loop_top\nHALT").unwrap();
    assert_eq!(prog.instructions()[1], Instruction::BranchIfNotZero(0));
}

#[test]
fn parse_setlocal_getlocal() {
    let prog = parse_program(".memory 255\nSETLOCAL 3\nGETLOCAL 3\nHALT").unwrap();
    assert_eq!(prog.instructions()[0], Instruction::SetLocal(3));
    assert_eq!(prog.instructions()[1], Instruction::GetLocal(3));
}

#[test]
fn parse_iread_iwrite() {
    let prog = parse_program(".memory 240\nIREAD 5\nIWRITE 7\nHALT").unwrap();
    assert_eq!(prog.instructions()[0], Instruction::IRead(5));
    assert_eq!(prog.instructions()[1], Instruction::IWrite(7));
}

#[test]
fn parse_getlocal_slot_out_of_bounds_rejected() {
    let err = parse_program(".memory 255\nGETLOCAL 16\nHALT").unwrap_err();
    assert!(err.to_string().contains("exceeds limit 15"));
}

#[test]
fn parse_setlocal_slot_out_of_bounds_rejected() {
    let err = parse_program(".memory 255\nSETLOCAL 16\nHALT").unwrap_err();
    assert!(err.to_string().contains("exceeds limit 15"));
}

#[test]
fn parse_iread_key_out_of_bounds_rejected() {
    let err = parse_program(".memory 240\nIREAD 16\nHALT").unwrap_err();
    assert!(err.to_string().contains("exceeds limit 15"));
}

#[test]
fn parse_iwrite_key_out_of_bounds_rejected() {
    let err = parse_program(".memory 240\nIWRITE 16\nHALT").unwrap_err();
    assert!(err.to_string().contains("exceeds limit 15"));
}

// ---------------------------------------------------------------------------
// Interpreter execution tests
// ---------------------------------------------------------------------------

fn run_program(source: &str) -> MachineState {
    let program = parse_program(source).expect("parse ok");
    let mut interp = NativeInterpreter::new(
        program,
        llm_provable_computer::config::Attention2DMode::default(),
        128,
    );
    interp.run().expect("run ok");
    interp.state().clone()
}

#[test]
fn setlocal_getlocal_roundtrip() {
    let state = run_program(
        r#"
.memory 255
LOADI 42
SETLOCAL 3
LOADI 0
GETLOCAL 3
HALT
"#,
    );
    assert_eq!(state.acc, 42);
}

#[test]
fn setlocal_stores_at_correct_address() {
    let state = run_program(
        r#"
.memory 255
LOADI 99
SETLOCAL 0
HALT
"#,
    );
    // slot 0 → address 240
    assert_eq!(state.memory[240], 99);
}

#[test]
fn brif_loops_n_times() {
    // Countdown loop: start=3, decrement by 1, stop when zero
    let state = run_program(
        r#"
.memory 4
LOADI 3
loop_start: LOOP
  SUB 1
  BRIF loop_start
ENDLOOP
HALT
"#,
    );
    assert_eq!(state.acc, 0);
}

#[test]
fn brif_falls_through_when_zero() {
    // When ACC=0 at BRIF, fall through; next LOADI 7 should execute
    let state = run_program(
        r#"
.memory 4
LOADI 0
BRIF 0
LOADI 7
HALT
"#,
    );
    assert_eq!(state.acc, 7);
}

#[test]
fn brif_jumps_when_nonzero() {
    // LOADI 5, BRIF past_loadi7, LOADI 7, HALT
    // Because ACC=5≠0, BRIF jumps forward past LOADI 7, so final ACC stays 5.
    let state = run_program(
        r#"
.memory 4
LOADI 5
BRIF skip
LOADI 7
skip: HALT
"#,
    );
    assert_eq!(state.acc, 5);
}

#[test]
fn iread_loads_value_from_io_buffer() {
    let program = parse_program(".memory 240\nIREAD 0\nHALT").unwrap();
    let mut initial_memory = program.initial_memory().to_vec();
    // Pre-seed key 0 (address 224)
    initial_memory[224] = 77;
    let initial_state = MachineState::with_memory(initial_memory);
    let mut interp = NativeInterpreter::new_with_state(
        program,
        llm_provable_computer::config::Attention2DMode::default(),
        initial_state,
        32,
    );
    interp.run().expect("run ok");
    assert_eq!(interp.state().acc, 77);
}

#[test]
fn iwrite_stores_acc_into_io_buffer() {
    let state = run_program(
        r#"
.memory 240
LOADI 55
IWRITE 2
HALT
"#,
    );
    // key 2 → address 226
    assert_eq!(state.memory[226], 55);
}

#[test]
fn iread_iwrite_roundtrip_via_interpreter() {
    let program = parse_program(".memory 240\nIREAD 0\nADD 1\nIWRITE 1\nHALT").unwrap();
    let mut initial_memory = program.initial_memory().to_vec();
    initial_memory[224] = 41; // key 0 = 41
    let initial_state = MachineState::with_memory(initial_memory);
    let mut interp = NativeInterpreter::new_with_state(
        program,
        llm_provable_computer::config::Attention2DMode::default(),
        initial_state,
        32,
    );
    interp.run().expect("run ok");
    assert_eq!(interp.state().acc, 42);
    // key 1 → address 225
    assert_eq!(interp.state().memory[225], 42);
}

// ---------------------------------------------------------------------------
// Proof round-trip tests (vanilla STARK)
// These use small memory programs that fit in the default memory region.
// Local variable and I/O proof tests use larger memory and are marked #[ignore].
// ---------------------------------------------------------------------------

#[test]
fn loop_endloop_nop_proof_round_trips() {
    // LOOP/ENDLOOP are NOP-equivalent so they pass through without special handling.
    let program = parse_program("LOADI 7\nLOOP\nENDLOOP\nHALT").unwrap();
    let model =
        llm_provable_computer::model::TransformerVm::new(TransformerVmConfig::default(), program)
            .unwrap();
    let proof = prove_execution_stark(&model, 32).expect("prove ok");
    assert!(verify_execution_stark(&proof).expect("verify ok"));
    assert_eq!(proof.claim.final_state.acc, 7);
}

#[test]
fn brif_proof_round_trips_countdown() {
    let program = parse_program(
        ".memory 4\nLOADI 3\nloop_start: LOOP\n  SUB 1\n  BRIF loop_start\nENDLOOP\nHALT",
    )
    .unwrap();
    let model =
        llm_provable_computer::model::TransformerVm::new(TransformerVmConfig::default(), program)
            .unwrap();
    let proof = prove_execution_stark(&model, 32).expect("prove ok");
    assert!(verify_execution_stark(&proof).expect("verify ok"));
    assert_eq!(proof.claim.final_state.acc, 0);
}

#[test]
#[ignore = "STARK proof with 255-cell memory is too slow for the default suite; run with --include-ignored"]
fn setlocal_getlocal_proof_round_trips() {
    let program =
        parse_program(".memory 255\nLOADI 42\nSETLOCAL 0\nLOADI 0\nGETLOCAL 0\nHALT").unwrap();
    let model =
        llm_provable_computer::model::TransformerVm::new(TransformerVmConfig::default(), program)
            .unwrap();
    let proof = prove_execution_stark(&model, 32).expect("prove ok");
    assert!(verify_execution_stark(&proof).expect("verify ok"));
    assert_eq!(proof.claim.final_state.acc, 42);
}

#[test]
#[ignore = "STARK proof with 240-cell memory is too slow for the default suite; run with --include-ignored"]
fn iread_iwrite_proof_round_trips() {
    let program = parse_program(".memory 240\nIREAD 0\nADD 1\nIWRITE 1\nHALT").unwrap();
    let model =
        llm_provable_computer::model::TransformerVm::new(TransformerVmConfig::default(), program)
            .unwrap();
    let mut input = IoBuffer::new();
    input.set(0, 41);
    let proved = prove_io_execution_stark(&model, &input, 32, VanillaStarkProofOptions::default())
        .expect("prove ok");
    assert!(verify_io_execution_stark_full(&proved).expect("verify ok"));
    assert_eq!(proved.output_values[1], 42);
}
