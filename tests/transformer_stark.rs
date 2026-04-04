use llm_provable_computer::{
    build_shared_transformer_staged_trace, estimate_shared_transformer_forward_pass_stark_cost,
    estimate_shared_transformer_staged_skeleton_stark_cost,
    prove_shared_transformer_forward_pass_stark, prove_shared_transformer_staged_skeleton_stark,
    prove_shared_transformer_staged_skeleton_stark_profiled,
    validate_shared_transformer_staged_trace, verify_shared_transformer_forward_pass_stark,
    verify_shared_transformer_staged_skeleton_stark, Attention2DMode, ExecutionRuntime,
    ProgramCompiler, SharedTransformerTraceRow, TransformerVmConfig, VanillaStarkProofOptions,
};

fn compile_and_trace(
    source: &str,
    max_steps: usize,
    ff_chunk_size: usize,
) -> (
    llm_provable_computer::TransformerVm,
    llm_provable_computer::SharedTransformerStagedTrace,
) {
    let model = ProgramCompiler
        .compile_source(source, TransformerVmConfig::default())
        .expect("compile");
    let mut runtime = ExecutionRuntime::new(model.clone(), max_steps);
    let (_result, witness) = runtime.run_with_witness().expect("run with witness");
    let trace = build_shared_transformer_staged_trace(&witness, ff_chunk_size).expect("trace");
    (model, trace)
}

fn validate_forward_pass_trace_case(source: &str, max_steps: usize, ff_chunk_size: usize) {
    let (model, trace) = compile_and_trace(source, max_steps, ff_chunk_size);
    let report = validate_shared_transformer_staged_trace(&model, &trace).expect("validate");
    assert!(
        report.validated_steps > 0,
        "expected at least one validated step"
    );
}

#[test]
fn staged_trace_validator_accepts_valid_trace() {
    let (model, trace) = compile_and_trace(
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
        32,
    );

    let report = validate_shared_transformer_staged_trace(&model, &trace).expect("validate");
    assert_eq!(report.validated_steps, 5);
    assert_eq!(report.row_count, trace.rows.len());
}

#[test]
fn staged_trace_validator_rejects_tampered_attention_row() {
    let (model, mut trace) = compile_and_trace(
        r#"
        .memory 1
        LOADI 5
        HALT
    "#,
        8,
        16,
    );

    let attention_row = trace
        .rows
        .iter_mut()
        .find_map(|row| match row {
            SharedTransformerTraceRow::Attention(row) => Some(row),
            _ => None,
        })
        .expect("attention row");
    attention_row.input_vector[0] += 1;

    let err = validate_shared_transformer_staged_trace(&model, &trace).unwrap_err();
    assert!(err
        .to_string()
        .contains("attention row does not match shared build_input_vector semantics"));
}

#[test]
fn staged_trace_validator_rejects_tampered_feed_forward_chunk() {
    let (model, mut trace) = compile_and_trace(
        r#"
        .memory 1
        LOADI 5
        HALT
    "#,
        8,
        8,
    );

    let chunk_row = trace
        .rows
        .iter_mut()
        .find_map(|row| match row {
            SharedTransformerTraceRow::FeedForwardChunk(row) => Some(row),
            _ => None,
        })
        .expect("chunk row");
    chunk_row.hidden_values[0] += 1;

    let err = validate_shared_transformer_staged_trace(&model, &trace).unwrap_err();
    assert!(err
        .to_string()
        .contains("feed-forward chunk rows do not match shared FF hidden activations"));
}

#[test]
fn staged_trace_validator_rejects_tampered_output_token() {
    let (model, mut trace) = compile_and_trace(
        r#"
        .memory 1
        LOADI 5
        HALT
    "#,
        8,
        16,
    );

    let output_row = trace
        .rows
        .iter_mut()
        .find_map(|row| match row {
            SharedTransformerTraceRow::OutputState(row) => Some(row),
            _ => None,
        })
        .expect("output row");
    output_row.output_token[0] *= -1;

    let err = validate_shared_transformer_staged_trace(&model, &trace).unwrap_err();
    assert!(err
        .to_string()
        .contains("output token does not match shared encode_state semantics"));
}

#[test]
fn forward_pass_trace_validator_covers_instruction_families() {
    for (source, max_steps) in [
        (
            r#"
            .memory 2
            LOADI 7
            ADD 3
            SUB 2
            MUL 4
            HALT
        "#,
            16usize,
        ),
        (
            r#"
            .memory 3
            .init 0 5
            LOAD 0
            STORE 1
            ADDM 1
            HALT
        "#,
            16usize,
        ),
        (
            r#"
            .memory 1
            LOADI 6
            AND 3
            OR 8
            XOR 2
            HALT
        "#,
            16usize,
        ),
        (
            r#"
            .memory 1
            LOADI -1
            CMP 0
            JS neg
            LOADI 9
        neg:
            HALT
        "#,
            16usize,
        ),
        (
            r#"
            .memory 4
            LOADI 5
            PUSH
            LOADI 2
            POP
            CALL sub
            HALT
        sub:
            ADD 1
            RET
        "#,
            24usize,
        ),
    ] {
        validate_forward_pass_trace_case(source, max_steps, 16);
    }
}

#[test]
#[ignore = "shared transformer proof round-trips remain expensive in the default suite"]
fn staged_skeleton_proof_round_trips_for_average_hard_execution() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 3
            .init 0 11
            .init 1 4
            LOAD 0
            ADD 1
            STORE 2
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let proof =
        prove_shared_transformer_staged_skeleton_stark(&model, 16, 512).expect("prove skeleton");
    assert!(verify_shared_transformer_staged_skeleton_stark(&proof).expect("verify"));

    let mut proof = proof;
    proof.claim.trace_digest = proof.claim.trace_digest.wrapping_add(1);

    assert!(!verify_shared_transformer_staged_skeleton_stark(&proof).expect("verify"));
}

#[test]
fn staged_skeleton_proof_rejects_softmax_attention_mode() {
    let config = TransformerVmConfig {
        attention_mode: Attention2DMode::Softmax { query: [1.0, 0.0] },
        ..TransformerVmConfig::default()
    };
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 5
            HALT
        "#,
            config,
        )
        .expect("compile");

    let err = prove_shared_transformer_staged_skeleton_stark(&model, 8, 512).unwrap_err();
    assert!(err
        .to_string()
        .contains("currently supports only `average-hard` attention"));
}

#[test]
fn forward_pass_proof_api_rejects_softmax_attention_mode() {
    let config = TransformerVmConfig {
        attention_mode: Attention2DMode::Softmax { query: [1.0, 0.0] },
        ..TransformerVmConfig::default()
    };
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 5
            HALT
        "#,
            config,
        )
        .expect("compile");

    let err = prove_shared_transformer_forward_pass_stark(&model, 8, 512).unwrap_err();
    assert!(err
        .to_string()
        .contains("currently supports only `average-hard` attention"));
}

#[test]
#[ignore = "shared transformer cost estimation remains expensive in the default suite"]
fn staged_skeleton_cost_estimate_reports_trace_shape() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 5
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let estimate = estimate_shared_transformer_staged_skeleton_stark_cost(
        &model,
        8,
        8,
        VanillaStarkProofOptions::default(),
    )
    .expect("estimate");

    assert_eq!(estimate.steps, 2);
    assert_eq!(
        estimate.trace_rows,
        estimate.steps * (estimate.ff_chunk_rows * estimate.linear_feature_chunk_rows + 4)
    );
    assert!(estimate.register_count > 0);
    assert!(estimate.transition_constraint_count > 0);
    assert!(estimate.fri_domain_length >= estimate.omicron_domain_length);
}

#[test]
#[ignore = "shared transformer cost estimation remains expensive in the default suite"]
fn forward_pass_cost_estimate_matches_staged_cost_estimate() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 5
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let staged = estimate_shared_transformer_staged_skeleton_stark_cost(
        &model,
        8,
        8,
        VanillaStarkProofOptions::default(),
    )
    .expect("staged estimate");
    let forward = estimate_shared_transformer_forward_pass_stark_cost(
        &model,
        8,
        8,
        VanillaStarkProofOptions::default(),
    )
    .expect("forward estimate");

    assert_eq!(forward.trace_rows, staged.trace_rows);
    assert_eq!(forward.register_count, staged.register_count);
    assert_eq!(
        forward.transition_constraint_count,
        staged.transition_constraint_count
    );
}

#[test]
#[ignore = "shared transformer profiled proof remains expensive in the default suite"]
fn staged_skeleton_profiled_proof_round_trips_for_average_hard_execution() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 3
            .init 0 11
            .init 1 4
            LOAD 0
            ADD 1
            STORE 2
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let report =
        prove_shared_transformer_staged_skeleton_stark_profiled(&model, 16, 512).expect("prove");
    assert!(verify_shared_transformer_staged_skeleton_stark(&report.proof).expect("verify"));
    assert!(report.profile.witness_build_micros > 0);
    assert!(report.profile.stark.total_micros > 0);
    assert!(report.profile.total_micros >= report.profile.stark.total_micros);
}

#[test]
#[ignore = "shared transformer proof round-trips remain expensive in the default suite"]
fn forward_pass_proof_round_trips_for_average_hard_execution() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 3
            .init 0 11
            .init 1 4
            LOAD 0
            ADD 1
            STORE 2
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let proof =
        prove_shared_transformer_forward_pass_stark(&model, 16, 512).expect("prove forward pass");
    assert!(verify_shared_transformer_forward_pass_stark(&proof).expect("verify"));
}

#[test]
#[ignore = "shared transformer overflow proof remains expensive in the default suite"]
fn staged_skeleton_proof_round_trips_overflowing_raw_accumulator() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 4611686018427387904
            ADD 4611686018427387904
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let proof =
        prove_shared_transformer_staged_skeleton_stark(&model, 8, 512).expect("prove skeleton");
    assert!(verify_shared_transformer_staged_skeleton_stark(&proof).expect("verify"));
}

#[test]
#[ignore = "shared transformer compare proof remains expensive in the default suite"]
fn staged_skeleton_proof_round_trips_compare_branch_execution() {
    let model = ProgramCompiler
        .compile_source(
            r#"
            .memory 1
            LOADI 5
            CMP 5
            JZ done
            LOADI 99
        done:
            HALT
        "#,
            TransformerVmConfig::default(),
        )
        .expect("compile");

    let proof =
        prove_shared_transformer_staged_skeleton_stark(&model, 16, 512).expect("prove skeleton");
    assert!(verify_shared_transformer_staged_skeleton_stark(&proof).expect("verify"));
}
