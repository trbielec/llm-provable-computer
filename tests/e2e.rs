//! End-to-end tests: parse → compile → execute → verify for complete programs.
//!
//! Every test runs the full pipeline through both the transformer model and
//! the native interpreter, then compares results.

use llm_provable_computer::{
    compile_cdmn_table_with_metadata, parse_program, verify_model_against_native, Attention2DMode,
    CdmnTable, Collection, CollectionField, Constraint as CdmnConstraint,
    Definition as CdmnDefinition, ExecutionRuntime, MachineState, NativeInterpreter,
    ProgramCompiler, TransformerVmConfig, Variable as CdmnVariable,
};

fn run_program(source: &str, max_steps: usize) -> MachineState {
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model, max_steps);
    let result = runtime.run().expect("run");
    result.final_state
}

fn run_and_verify(source: &str, max_steps: usize, layers: usize) -> MachineState {
    let config = TransformerVmConfig {
        num_layers: layers,
        ..TransformerVmConfig::default()
    };
    let model = ProgramCompiler
        .compile_source(source, config)
        .expect("compile");
    let comparison = verify_model_against_native(model, max_steps).expect("verify");
    comparison.transformer.final_state
}

fn cdmn_variable(name: &str, memory: u8) -> CdmnVariable {
    CdmnVariable {
        name: name.to_string(),
        memory,
        presence_memory: None,
        unknown_memory: None,
    }
}

fn cdmn_field(name: &str, offset: u8) -> CollectionField {
    CollectionField {
        name: name.to_string(),
        offset,
        presence_offset: None,
        unknown_offset: None,
    }
}

fn cdmn_constraint(
    id: &str,
    description: &str,
    logic: &str,
    reason_code: Option<i64>,
    reason_label: Option<&str>,
) -> CdmnConstraint {
    CdmnConstraint {
        id: id.to_string(),
        description: description.to_string(),
        logic: logic.to_string(),
        reason_code,
        reason_label: reason_label.map(str::to_string),
        insufficient_data_reason_code: None,
        insufficient_data_reason_label: None,
    }
}

fn medication_policy_table() -> CdmnTable {
    CdmnTable {
        name: "Medication Safety".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![
            cdmn_variable("max_dose", 0),
            cdmn_variable("required_class", 1),
            cdmn_variable("backup_class", 2),
            cdmn_variable("prohibited_class", 3),
        ],
        definitions: vec![],
        collections: vec![Collection {
            name: "medications".to_string(),
            count_memory: 4,
            base_memory: 5,
            stride: 2,
            max_items: 3,
            fields: vec![cdmn_field("dose", 0), cdmn_field("class_code", 1)],
        }],
        constraints: vec![
            cdmn_constraint(
                "M1",
                "Each medication must respect the max dose",
                "forall medication in medications (medication.dose <= max_dose)",
                Some(101),
                Some("dose_exceeds_max"),
            ),
            cdmn_constraint(
                "M2",
                "At least one required or backup class must be present",
                "exists medication in medications ((medication.class_code == required_class) or (medication.class_code == backup_class))",
                Some(102),
                Some("required_or_backup_class_missing"),
            ),
            cdmn_constraint(
                "M3",
                "No medication may use a prohibited class",
                "forall medication in medications (not (medication.class_code == prohibited_class))",
                Some(103),
                Some("prohibited_class_present"),
            ),
        ],
    }
}

fn lab_policy_table() -> CdmnTable {
    CdmnTable {
        name: "Lab and Vitals Guardrails".to_string(),
        table_type: "cDMN".to_string(),
        variables: vec![
            cdmn_variable("wbc_count", 0),
            cdmn_variable("wbc_min", 1),
            cdmn_variable("wbc_max", 2),
            cdmn_variable("systolic_bp", 3),
            cdmn_variable("diastolic_bp", 4),
            cdmn_variable("pulse_pressure_min", 5),
            cdmn_variable("pulse_pressure_max", 6),
        ],
        definitions: vec![CdmnDefinition {
            name: "pulse_pressure".to_string(),
            expression: "systolic_bp - diastolic_bp".to_string(),
            description: "Derived systolic-diastolic pulse pressure".to_string(),
        }],
        collections: vec![],
        constraints: vec![
            cdmn_constraint(
                "L1",
                "White blood cell count must remain within range",
                "wbc_count between wbc_min and wbc_max",
                Some(301),
                Some("wbc_out_of_range"),
            ),
            cdmn_constraint(
                "L2",
                "Pulse pressure must remain within range",
                "pulse_pressure between pulse_pressure_min and pulse_pressure_max",
                Some(302),
                Some("pulse_pressure_out_of_range"),
            ),
        ],
    }
}

// --- Shipped programs ---

#[test]
fn e2e_addition_program() {
    let source = std::fs::read_to_string("programs/addition.tvm").expect("fixture");
    let state = run_and_verify(&source, 32, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 8);
}

#[test]
fn e2e_counter_program() {
    let source = std::fs::read_to_string("programs/counter.tvm").expect("fixture");
    let state = run_and_verify(&source, 128, 2);
    assert!(state.halted);
    assert_eq!(state.acc, 5);
    assert_eq!(state.memory[0], 5);
}

#[test]
fn e2e_fibonacci_program() {
    let source = std::fs::read_to_string("programs/fibonacci.tvm").expect("fixture");
    let state = run_and_verify(&source, 512, 3);
    assert!(state.halted);
    assert_eq!(state.acc, 21, "Fibonacci(8) = 21");
}

#[test]
fn e2e_factorial_recursive_program() {
    let source = std::fs::read_to_string("programs/factorial_recursive.tvm").expect("fixture");
    let state = run_and_verify(&source, 128, 2);
    assert!(state.halted);
    assert_eq!(state.acc, 120, "5! = 120");
    assert_eq!(
        state.sp, 11,
        "stack pointer restored after recursion unwinds"
    );
}

#[test]
fn e2e_multiply_program() {
    let source = std::fs::read_to_string("programs/multiply.tvm").expect("fixture");
    let state = run_and_verify(&source, 256, 2);
    assert!(state.halted);
    assert_eq!(state.acc, 42, "6 * 7 = 42");
}

#[test]
fn e2e_subroutine_addition() {
    let source = std::fs::read_to_string("programs/subroutine_addition.tvm").expect("fixture");
    let state = run_and_verify(&source, 32, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 42);
    assert_eq!(state.sp, 8);
}

#[test]
fn e2e_stack_roundtrip() {
    let source = std::fs::read_to_string("programs/stack_roundtrip.tvm").expect("fixture");
    let state = run_and_verify(&source, 32, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 42);
    assert_eq!(state.sp, 8);
}

#[test]
fn e2e_memory_roundtrip() {
    let source = std::fs::read_to_string("programs/memory_roundtrip.tvm").expect("fixture");
    let state = run_and_verify(&source, 32, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 42);
}

// --- Soft attention programs ---

#[test]
fn e2e_soft_attention_average_hard() {
    let source = std::fs::read_to_string("programs/soft_attention_memory.tvm").expect("fixture");
    let model = ProgramCompiler
        .compile_source(
            &source,
            TransformerVmConfig {
                attention_mode: Attention2DMode::AverageHard { query: [1.0, 0.0] },
                ..TransformerVmConfig::default()
            },
        )
        .expect("compile");
    let comparison = verify_model_against_native(model, 32).expect("verify");
    assert_eq!(comparison.transformer.final_state.acc, 10);
}

#[test]
fn e2e_soft_attention_softmax() {
    let source = std::fs::read_to_string("programs/soft_attention_memory.tvm").expect("fixture");
    let model = ProgramCompiler
        .compile_source(
            &source,
            TransformerVmConfig {
                attention_mode: Attention2DMode::Softmax { query: [1.0, 0.0] },
                ..TransformerVmConfig::default()
            },
        )
        .expect("compile");
    let comparison = verify_model_against_native(model, 32).expect("verify");
    assert_eq!(comparison.transformer.final_state.acc, 9);
}

#[test]
fn e2e_soft_attention_hard_softmax() {
    let source = std::fs::read_to_string("programs/soft_attention_memory.tvm").expect("fixture");
    let model = ProgramCompiler
        .compile_source(
            &source,
            TransformerVmConfig {
                attention_mode: Attention2DMode::HardSoftmax {
                    temperature: 10.0,
                    query: [1.0, 0.0],
                },
                ..TransformerVmConfig::default()
            },
        )
        .expect("compile");
    let comparison = verify_model_against_native(model, 32).expect("verify");
    assert_eq!(comparison.transformer.final_state.acc, 4);
}

// --- Individual instruction E2E ---

#[test]
fn e2e_each_arithmetic_instruction() {
    let cases = [
        ("LOADI 100\nADD 50\nHALT\n", 150),
        ("LOADI 100\nSUB 30\nHALT\n", 70),
        ("LOADI 6\nMUL 7\nHALT\n", 42),
        ("LOADI -5\nADD 5\nHALT\n", 0),
        ("LOADI 0\nSUB 1\nHALT\n", -1),
    ];

    for (source, expected_acc) in cases {
        let state = run_and_verify(source, 16, 1);
        assert!(state.halted);
        assert_eq!(state.acc, expected_acc, "failed for: {source}");
    }
}

#[test]
fn e2e_overflowing_addition_saturates_and_matches_transformer() {
    let state = run_and_verify(
        "LOADI 4611686018427387904\nADD 4611686018427387904\nHALT\n",
        16,
        1,
    );
    assert_eq!(state.acc, i64::MAX);
    assert!(state.carry_flag);
}

#[test]
fn e2e_each_bitwise_instruction() {
    let cases = [
        ("LOADI 15\nAND 9\nHALT\n", 9),  // 0b1111 & 0b1001 = 0b1001
        ("LOADI 5\nOR 10\nHALT\n", 15),  // 0b0101 | 0b1010 = 0b1111
        ("LOADI 15\nXOR 9\nHALT\n", 6),  // 0b1111 ^ 0b1001 = 0b0110
        ("LOADI 255\nAND 0\nHALT\n", 0), // anything & 0 = 0
        ("LOADI 0\nOR 42\nHALT\n", 42),  // 0 | x = x
        ("LOADI 42\nXOR 42\nHALT\n", 0), // x ^ x = 0
    ];

    for (source, expected_acc) in cases {
        let state = run_and_verify(source, 16, 1);
        assert!(state.halted);
        assert_eq!(state.acc, expected_acc, "failed for: {source}");
    }
}

#[test]
fn e2e_each_memory_instruction() {
    let source = r#"
        .memory 4
        .init 0 100
        LOADI 42
        STORE 1
        LOAD 0
        ADDM 1
        HALT
    "#;
    let state = run_and_verify(source, 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 142, "100 + 42");
    assert_eq!(state.memory[1], 42);
}

#[test]
fn e2e_memory_arithmetic_instructions() {
    let source = r#"
        .memory 4
        .init 0 10
        .init 1 3
        .init 2 5
        LOAD 0
        SUBM 1
        MULM 2
        HALT
    "#;
    let state = run_and_verify(source, 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 35, "(10 - 3) * 5 = 35");
}

#[test]
fn e2e_each_branch_instruction() {
    // JZ takes branch
    let state = run_program("LOADI 0\nJZ 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 0);

    // JZ falls through
    let state = run_program("LOADI 1\nJZ 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 99);

    // JNZ takes branch
    let state = run_program("LOADI 1\nJNZ 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 1);

    // JNZ falls through
    let state = run_program("LOADI 0\nJNZ 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 99);

    // JS takes branch
    let state = run_program("LOADI -1\nJS 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, -1);

    // JS falls through
    let state = run_program("LOADI 1\nJS 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 99);

    // JNS takes branch
    let state = run_program("LOADI 1\nJNS 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 1);

    // JNS falls through
    let state = run_program("LOADI -1\nJNS 3\nLOADI 99\nHALT\n", 16);
    assert_eq!(state.acc, 99);
}

#[test]
fn e2e_quantified_cdmn_medication_policy() {
    let compiled =
        compile_cdmn_table_with_metadata(&medication_policy_table()).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    initial_memory[0] = 500;
    initial_memory[1] = 11;
    initial_memory[2] = 12;
    initial_memory[3] = 99;
    initial_memory[4] = 2;
    initial_memory[5] = 400;
    initial_memory[6] = 7;
    initial_memory[7] = 450;
    initial_memory[8] = 12;

    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 2_000_000);
    let result = runtime.run().expect("run");
    let final_state = result.final_state;
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(final_state.memory[status_idx], 1);
    assert_eq!(final_state.memory[reason_idx], 0);
}

#[test]
fn e2e_quantified_cdmn_reports_reason_code_on_failure() {
    let compiled =
        compile_cdmn_table_with_metadata(&medication_policy_table()).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    initial_memory[0] = 500;
    initial_memory[1] = 11;
    initial_memory[2] = 12;
    initial_memory[3] = 99;
    initial_memory[4] = 2;
    initial_memory[5] = 600;
    initial_memory[6] = 7;
    initial_memory[7] = 450;
    initial_memory[8] = 12;

    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 2_000_000);
    let result = runtime.run().expect("run");
    let final_state = result.final_state;
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(final_state.memory[status_idx], 0);
    assert_eq!(final_state.memory[reason_idx], 101);
    let reason = compiled.reason_for_code(final_state.memory[reason_idx]);
    assert_eq!(
        reason.map(|entry| entry.label.as_str()),
        Some("dose_exceeds_max")
    );
}

#[test]
fn e2e_quantified_cdmn_reports_reason_code_for_prohibited_class() {
    let compiled =
        compile_cdmn_table_with_metadata(&medication_policy_table()).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    initial_memory[0] = 500;
    initial_memory[1] = 11;
    initial_memory[2] = 12;
    initial_memory[3] = 99;
    initial_memory[4] = 2;
    initial_memory[5] = 400;
    initial_memory[6] = 99;
    initial_memory[7] = 450;
    initial_memory[8] = 12;

    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 2_000_000);
    let result = runtime.run().expect("run");
    let final_state = result.final_state;
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert!(final_state.halted);
    assert_eq!(final_state.memory[status_idx], 0);
    assert_eq!(final_state.memory[reason_idx], 103);
    let reason = compiled.reason_for_code(final_state.memory[reason_idx]);
    assert_eq!(
        reason.map(|entry| entry.label.as_str()),
        Some("prohibited_class_present")
    );
}

#[test]
fn e2e_cdmn_lab_policy_supports_ranges_and_derived_expressions() {
    let compiled = compile_cdmn_table_with_metadata(&lab_policy_table()).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    initial_memory[0] = 8_000;
    initial_memory[1] = 4_000;
    initial_memory[2] = 11_000;
    initial_memory[3] = 120;
    initial_memory[4] = 80;
    initial_memory[5] = 30;
    initial_memory[6] = 60;

    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 2_000_000);
    let result = runtime.run().expect("run");
    let final_state = result.final_state;
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(final_state.memory[status_idx], 1);
    assert_eq!(final_state.memory[reason_idx], 0);
}

#[test]
fn e2e_cdmn_lab_policy_reports_range_failure_reason_code() {
    let compiled = compile_cdmn_table_with_metadata(&lab_policy_table()).expect("compile cdmn");
    let program = parse_program(&compiled.assembly).expect("parse");
    let mut initial_memory = vec![0; program.memory_size()];
    initial_memory[0] = 8_000;
    initial_memory[1] = 4_000;
    initial_memory[2] = 11_000;
    initial_memory[3] = 170;
    initial_memory[4] = 90;
    initial_memory[5] = 30;
    initial_memory[6] = 60;

    let program = program
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let mut runtime = NativeInterpreter::new(program, Attention2DMode::default(), 2_000_000);
    let result = runtime.run().expect("run");
    let final_state = result.final_state;
    let status_idx = usize::from(compiled.output.status_register);
    let reason_idx = usize::from(compiled.output.reason_register);
    assert_eq!(final_state.memory[status_idx], 0);
    assert_eq!(final_state.memory[reason_idx], 302);
    let reason = compiled.reason_for_code(final_state.memory[reason_idx]);
    assert_eq!(
        reason.map(|entry| entry.label.as_str()),
        Some("pulse_pressure_out_of_range")
    );
}

#[test]
fn e2e_cmp_sets_flags_correctly() {
    // acc < value: carry=true, zero=false
    let source = "LOADI 5\nCMP 10\nHALT\n";
    let state = run_and_verify(source, 16, 1);
    assert!(state.carry_flag, "5 < 10 should set carry");
    assert!(!state.zero_flag);
    assert_eq!(state.acc, -5, "CMP stores difference");

    // acc == value: carry=false, zero=true
    let source = "LOADI 10\nCMP 10\nHALT\n";
    let state = run_and_verify(source, 16, 1);
    assert!(!state.carry_flag);
    assert!(state.zero_flag);

    // acc > value: carry=false, zero=false
    let source = "LOADI 10\nCMP 5\nHALT\n";
    let state = run_and_verify(source, 16, 1);
    assert!(!state.carry_flag, "10 > 5 should not set carry");
    assert!(!state.zero_flag);
}

#[test]
fn e2e_cmpm_with_memory_operand() {
    let source = r#"
        .memory 2
        .init 0 50
        LOADI 50
        CMPM 0
        HALT
    "#;
    let state = run_and_verify(source, 16, 1);
    assert!(state.zero_flag);
    assert!(!state.carry_flag);
    assert_eq!(state.acc, 0);
}

// --- Edge cases ---

#[test]
fn e2e_single_halt() {
    let state = run_and_verify("HALT\n", 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 0);
    assert_eq!(state.pc, 0);
}

#[test]
fn e2e_nop_then_halt() {
    let state = run_and_verify("NOP\nNOP\nNOP\nHALT\n", 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 0);
    assert_eq!(state.pc, 3);
}

#[test]
fn e2e_crossing_old_i16_max_no_longer_overflows() {
    let state = run_and_verify("LOADI 32767\nADD 1\nHALT\n", 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 32_768);
    assert!(!state.carry_flag);
}

#[test]
fn e2e_crossing_old_i16_min_no_longer_underflows() {
    let state = run_and_verify("LOADI -32768\nSUB 1\nHALT\n", 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, -32_769);
    assert!(!state.carry_flag);
}

#[test]
fn e2e_max_steps_stops_infinite_loop() {
    let source = "loop: JMP loop\n";
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model, 100);
    let result = runtime.run().expect("run");
    assert!(!result.halted);
    assert_eq!(result.steps, 100);
}

// --- Multi-layer tests ---

#[test]
fn e2e_multi_layer_fibonacci() {
    let source = std::fs::read_to_string("programs/fibonacci.tvm").expect("fixture");
    for layers in 1..=4 {
        let state = run_and_verify(&source, 512, layers);
        assert!(state.halted, "layers={layers}");
        assert_eq!(state.acc, 21, "Fibonacci(8) with {layers} layers");
    }
}

#[test]
fn e2e_multi_layer_multiply() {
    let source = std::fs::read_to_string("programs/multiply.tvm").expect("fixture");
    for layers in 1..=3 {
        let state = run_and_verify(&source, 256, layers);
        assert!(state.halted, "layers={layers}");
        assert_eq!(state.acc, 42, "6*7 with {layers} layers");
    }
}

// --- Complex programs ---

#[test]
fn e2e_sum_array() {
    // Sum memory[0..4] = 10 + 20 + 30 + 40 = 100
    let source = r#"
        .memory 8
        .init 0 10
        .init 1 20
        .init 2 30
        .init 3 40
        .init 4 0
        .init 5 4

        LOADI 0
        STORE 4
        LOADI 0
    loop:
        LOAD 4
        SUBM 5
        JZ done
        LOAD 4
        ADDM 4
        STORE 4
        ; This approach won't work for generic array sum due to computed addresses.
        ; Instead, just sum them inline.
        JMP done
    done:
        LOAD 0
        ADDM 1
        ADDM 2
        ADDM 3
        HALT
    "#;
    let state = run_and_verify(source, 64, 2);
    assert!(state.halted);
    assert_eq!(state.acc, 100);
}

#[test]
fn e2e_absolute_value() {
    // Compute abs(-42)
    let source = r#"
        LOADI -42
        CMP 0
        JZ done
        ; acc now holds -42 - 0 = -42
        ; If carry was set (acc < 0 as signed), negate by MUL -1
        MUL -1
    done:
        HALT
    "#;
    let state = run_and_verify(source, 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 42);
}

#[test]
fn e2e_nested_subroutines() {
    let source = r#"
        .memory 8
        LOADI 0
        CALL add_ten
        CALL add_ten
        CALL add_ten
        HALT
    add_ten:
        ADD 10
        RET
    "#;
    let state = run_and_verify(source, 32, 2);
    assert!(state.halted);
    assert_eq!(state.acc, 30);
    assert_eq!(state.sp, 8, "stack pointer restored after all calls");
}

#[test]
fn e2e_bitwise_memory_ops() {
    let source = r#"
        .memory 4
        .init 0 255
        .init 1 15
        .init 2 170
        LOADI 170
        ANDM 1
        HALT
    "#;
    let state = run_and_verify(source, 16, 1);
    assert!(state.halted);
    assert_eq!(state.acc, 10, "0b10101010 & 0b00001111 = 0b00001010");
}

#[test]
fn e2e_xor_swap() {
    // XOR swap: a ^= b; b ^= a; a ^= b
    let source = r#"
        .memory 2
        .init 0 42
        .init 1 99
        LOAD 0
        XORM 1
        STORE 0
        LOAD 1
        XORM 0
        STORE 1
        LOAD 0
        XORM 1
        STORE 0
        HALT
    "#;
    let state = run_and_verify(source, 32, 2);
    assert!(state.halted);
    assert_eq!(state.memory[0], 99, "values should be swapped");
    assert_eq!(state.memory[1], 42, "values should be swapped");
}

// --- Determinism tests ---

#[test]
fn e2e_repeated_execution_is_deterministic() {
    let source = std::fs::read_to_string("programs/fibonacci.tvm").expect("fixture");

    let results: Vec<_> = (0..5)
        .map(|_| {
            let model = ProgramCompiler
                .compile_source(&source, TransformerVmConfig::default())
                .expect("compile");
            let mut runtime = ExecutionRuntime::new(model, 512);
            runtime.run().expect("run")
        })
        .collect();

    for result in &results[1..] {
        assert_eq!(result.final_state, results[0].final_state);
        assert_eq!(result.steps, results[0].steps);
    }
}

// --- Trace completeness ---

#[test]
fn e2e_trace_has_one_entry_per_step() {
    let source = "LOADI 1\nADD 2\nSUB 1\nHALT\n";
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model, 16);
    let result = runtime.run().expect("run");

    assert_eq!(result.steps, 4);
    assert_eq!(runtime.events().len(), 4, "one event per step");
    assert_eq!(runtime.trace().len(), 5, "initial + one per step");

    // First trace entry is initial state
    assert_eq!(runtime.trace()[0].pc, 0);
    assert_eq!(runtime.trace()[0].acc, 0);
}

#[test]
fn e2e_native_trace_has_one_entry_per_step() {
    let source = "LOADI 1\nADD 2\nHALT\n";
    let program = parse_program(source).expect("parse");
    let mut interpreter = NativeInterpreter::new(
        program,
        Attention2DMode::AverageHard { query: [1.0, 0.0] },
        16,
    );
    let result = interpreter.run().expect("run");

    assert_eq!(result.steps, 3);
    assert_eq!(interpreter.events().len(), 3);
    assert_eq!(interpreter.trace().len(), 4);
}

// --- Error propagation ---

#[test]
fn e2e_parse_error_propagates() {
    let result = ProgramCompiler.compile_source("INVALID_OP 42\n", TransformerVmConfig::default());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown instruction"));
}

#[test]
fn e2e_runtime_propagates_pc_out_of_bounds() {
    let source = "JMP 100\n";
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model, 16);
    let err = runtime.run().expect_err("should fail on OOB jump");
    assert!(err.to_string().contains("out of bounds"));
}
