use llm_provable_computer::{
    build_shared_transformer_staged_trace, ExecutionRuntime, ProgramCompiler,
    SharedTransformerTraceRow, SharedTransformerTraceStage, TransformerVmConfig,
};

fn witnessed_execution(
    source: &str,
    max_steps: usize,
) -> llm_provable_computer::SharedTransformerExecutionWitness {
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model, max_steps);
    let (_result, witness) = runtime.run_with_witness().expect("run with witness");
    witness
}

#[test]
fn staged_trace_uses_expected_row_schedule() {
    let witness = witnessed_execution(
        r#"
        .memory 2
        LOADI 7
        HALT
    "#,
        8,
    );

    let ff_chunk_size = 32;
    let trace = build_shared_transformer_staged_trace(&witness, ff_chunk_size).expect("trace");
    let expected_ff_rows = witness.steps[0].ff_hidden.len().div_ceil(ff_chunk_size);
    let expected_rows_per_step = 4 + expected_ff_rows;

    assert_eq!(trace.row_count_for_step(1), expected_rows_per_step);
    assert_eq!(trace.row_count_for_step(2), expected_rows_per_step);

    let first_stages = trace
        .rows
        .iter()
        .take(expected_rows_per_step)
        .map(SharedTransformerTraceRow::stage)
        .collect::<Vec<_>>();
    assert_eq!(first_stages[0], SharedTransformerTraceStage::InputState);
    assert_eq!(first_stages[1], SharedTransformerTraceStage::Attention);
    assert!(first_stages[2..(2 + expected_ff_rows)]
        .iter()
        .all(|stage| *stage == SharedTransformerTraceStage::FeedForwardChunk));
    assert_eq!(
        first_stages[2 + expected_ff_rows],
        SharedTransformerTraceStage::Transition
    );
    assert_eq!(
        first_stages[3 + expected_ff_rows],
        SharedTransformerTraceStage::OutputState
    );
}

#[test]
fn staged_trace_preserves_exact_integer_rows() {
    let witness = witnessed_execution(
        r#"
        .memory 4
        .init 0 41
        LOAD 0
        ADD 1
        STORE 1
        LOAD 1
        HALT
    "#,
        16,
    );

    let trace = build_shared_transformer_staged_trace(&witness, 64).expect("trace");

    let input_row = match &trace.rows[0] {
        SharedTransformerTraceRow::InputState(row) => row,
        other => panic!("expected input-state row, got {other:?}"),
    };
    assert!(input_row
        .input_token
        .iter()
        .all(|value| matches!(value, -1..=1)));

    let attention_row = trace
        .rows
        .iter()
        .find_map(|row| match row {
            SharedTransformerTraceRow::Attention(row) => Some(row),
            _ => None,
        })
        .expect("attention row");
    assert_eq!(attention_row.input_vector[0], 1);
}

#[test]
fn staged_trace_rejects_non_discrete_witness_values() {
    let mut witness = witnessed_execution(
        r#"
        .memory 1
        LOADI 5
        HALT
    "#,
        8,
    );
    witness.steps[0].ff_output[0] += 0.5;

    let err = build_shared_transformer_staged_trace(&witness, 32).unwrap_err();
    assert!(err
        .to_string()
        .contains("shared transformer staged trace requires exact integer values"));
}

#[test]
fn staged_trace_preserves_overflowing_raw_accumulator_output() {
    let witness = witnessed_execution(
        r#"
        .memory 1
        LOADI 4611686018427387904
        ADD 4611686018427387904
        HALT
    "#,
        8,
    );

    let trace = build_shared_transformer_staged_trace(&witness, 64).expect("trace");
    let raw_acc_values = trace
        .rows
        .iter()
        .filter_map(|row| match row {
            SharedTransformerTraceRow::Transition(row) => Some(row.ff_output[1]),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        raw_acc_values
            .iter()
            .copied()
            .any(|raw_acc| raw_acc > i128::from(i64::MAX)),
        "transition trace should retain the true overflowing raw accumulator output"
    );
}
