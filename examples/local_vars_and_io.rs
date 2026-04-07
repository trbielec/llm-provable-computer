//! End-to-end example: local variables, structured loops, and provable I/O.
//!
//! Run with:
//!   cargo run --example local_vars_and_io
//!
//! This example demonstrates the three new TVM capabilities:
//!
//! 1. **Local variables** — `SETLOCAL n` / `GETLOCAL n` store and retrieve
//!    values in addresses 240–255, independent of the stack.
//!
//! 2. **Structured loops** — `LOOP` / `ENDLOOP` are readable markers;
//!    `BRIF target` branches back (or forward) based on ACC.
//!
//! 3. **Provable I/O** — `IREAD key` / `IWRITE key` access the committed
//!    I/O buffer at addresses 224–239.  `prove_io_execution_stark` produces a
//!    STARK proof over the kernel execution and binds the external input
//!    values via `ProvedIoExecution`, enabling a two-layer audit.

use llm_provable_computer::assembly::parse_program;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::interpreter::NativeInterpreter;
use llm_provable_computer::model::TransformerVm;
use llm_provable_computer::proof::{prove_execution_stark, verify_execution_stark};
use llm_provable_computer::{
    prove_io_execution_stark, verify_io_execution_stark_full, IoBuffer, VanillaStarkProofOptions,
};

fn main() {
    // ── 1. Local variables ────────────────────────────────────────────────────
    //
    // Compute ACC = 3 + 7 using a local variable slot to hold the first operand
    // across two LOADI calls.  This shows that SETLOCAL/GETLOCAL persist
    // independently of the accumulator.

    println!("── Local variables ───────────────────────────────────────────");

    let local_source = r#"
.memory 255
LOADI 3
SETLOCAL 0      ; local[0] = 3
LOADI 7
GETLOCAL 0      ; ACC = local[0] = 3  (ACC was 7, now 3)
ADD 7           ; ACC = 10
HALT
"#;

    let program = parse_program(local_source).expect("parse ok");
    let model = TransformerVm::new(TransformerVmConfig::default(), program).expect("model ok");

    let mut interp = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        64,
    );
    interp.run().expect("run ok");
    let state = interp.state();
    println!("  Final ACC:         {}", state.acc); // 10
    println!("  local[0] (mem[240]): {}", state.memory[240]); // 3

    // ── 2. Structured loop ────────────────────────────────────────────────────
    //
    // Count down from 5 to 0 using BRIF.  The LOOP/ENDLOOP markers are
    // purely stylistic (NOP-equivalent).

    println!("\n── Structured loop ───────────────────────────────────────────");

    let loop_source = r#"
.memory 4
LOADI 5
loop_start: LOOP
  SUB 1
  BRIF loop_start
ENDLOOP
HALT
"#;

    let program = parse_program(loop_source).expect("parse ok");
    let model = TransformerVm::new(TransformerVmConfig::default(), program).expect("model ok");

    let mut interp = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        64,
    );
    interp.run().expect("run ok");
    println!("  Final ACC: {} (counted down from 5)", interp.state().acc); // 0
    println!("  Steps:     {}", interp.step_count());

    // Prove the loop and verify
    let proof = prove_execution_stark(&model, 64).expect("prove ok");
    let valid = verify_execution_stark(&proof).expect("verify ok");
    println!("  Proof valid: {valid}");
    println!("  Claim: final ACC = {}", proof.claim.final_state.acc);

    // ── 3. Provable I/O ───────────────────────────────────────────────────────
    //
    // An external system supplies two values via the I/O buffer.  The program
    // reads them, sums them, writes the result, and halts.  The proof covers
    // the full kernel execution; the committed input_values close the external
    // witness binding.

    println!("\n── Provable I/O ──────────────────────────────────────────────");

    // Read two values, sum, write result:
    let io_source = r#"
.memory 240
IREAD 0         ; ACC = input[0]
STORE 0         ; save to MEM[0]
IREAD 1         ; ACC = input[1]
ADDM 0          ; ACC += MEM[0]  => sum
IWRITE 2        ; output[2] = sum
HALT
"#;

    let program = parse_program(io_source).expect("parse ok");
    let model = TransformerVm::new(TransformerVmConfig::default(), program).expect("model ok");

    // Provide external inputs
    let mut io_input = IoBuffer::new();
    io_input.set(0, 19); // input[0] = 19
    io_input.set(1, 23); // input[1] = 23
                         // expected output[2] = 42

    println!("  Input[0] = {}", io_input.get(0));
    println!("  Input[1] = {}", io_input.get(1));

    // Run interpreter with pre-loaded I/O buffer to check result first
    let mut initial_memory = model.program().initial_memory().to_vec();
    io_input.inject_into_memory(&mut initial_memory);
    let initial_state = llm_provable_computer::state::MachineState::with_memory(initial_memory);
    let mut interp = NativeInterpreter::new_with_state(
        model.program().clone(),
        model.config().attention_mode.clone(),
        initial_state,
        64,
    );
    interp.run().expect("run ok");
    println!("  Output[2] = {} (sum)", interp.state().memory[226]); // 42

    // Prove + verify (small proof options for speed in this example)
    let options = VanillaStarkProofOptions {
        expansion_factor: 4,
        num_colinearity_checks: 2,
        security_level: 2,
    };

    print!("  Generating proof...");
    let proved = prove_io_execution_stark(&model, &io_input, 32, options).expect("prove ok");
    println!(" done");

    let valid = verify_io_execution_stark_full(&proved).expect("verify ok");
    println!("  Proof valid:    {valid}");
    println!("  Output[2]:      {}", proved.output_values[2]);
    println!("  Input committed: {:?}", &proved.input_values[..3]);

    // Show the two-layer audit manually
    let initial_mem = &proved.kernel_proof.claim.initial_state.memory;
    let io_region_matches = proved
        .input_values
        .iter()
        .enumerate()
        .all(|(i, &v)| initial_mem.get(224 + i).copied().unwrap_or(0) == v);
    println!("  I/O region binding verified: {io_region_matches}");
}
