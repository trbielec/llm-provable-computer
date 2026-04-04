use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llm_provable_computer::commit_and_audit::{
    audit_execution, commit_execution, verify_audit, AuditRequest, AuditTarget,
};
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::instruction::Program;
use llm_provable_computer::model::{
    DispatchInfo, SharedTransformerExecutionWitness, SharedTransformerStepWitness, TransformerVm,
};
use llm_provable_computer::state::MachineState;
use std::time::Duration;

fn setup_benchmark_witness(config: &TransformerVmConfig) -> SharedTransformerExecutionWitness {
    let initial_state = MachineState::new(1);
    SharedTransformerExecutionWitness {
        initial_state: initial_state.clone(),
        steps: vec![SharedTransformerStepWitness {
            step_number: 0,
            pre_state: initial_state.clone(),
            input_token: vec![0.0; config.d_model],
            decoded_state: initial_state.clone(),
            dispatch: DispatchInfo {
                instruction: llm_provable_computer::instruction::Instruction::Halt,
                layer_idx: 0,
            },
            operand: 0,
            input_vector: vec![0.0; 9 + 2 * llm_provable_computer::state::WORD_BITS],
            ff_gate: vec![0.0; config.ff_dim],
            ff_value: vec![0.0; config.ff_dim],
            ff_hidden: vec![0.0; config.ff_dim],
            ff_output: vec![0.0; 6],
            transition: llm_provable_computer::model::SharedTransformerTransitionWitness {
                pc: 0,
                acc: 0,
                sp: 0,
                zero_flag: true,
                carry_flag: false,
                halted: true,
                memory_write: None,
            },
            post_write_state: initial_state.clone(),
            output_token: vec![0.0; config.d_model],
            final_state: initial_state.clone(),
        }],
        final_state: initial_state.clone(),
    }
}

fn bench_commit_and_audit(c: &mut Criterion) {
    let config = TransformerVmConfig {
        num_layers: 1,
        ff_dim: 256,
        ..TransformerVmConfig::default()
    };
    let program = Program::new(
        vec![llm_provable_computer::instruction::Instruction::Halt],
        1,
    );
    let vm = TransformerVm::new(config.clone(), program).unwrap();
    let witness = setup_benchmark_witness(&config);

    let mut group = c.benchmark_group("commit_and_audit");

    group.bench_function("commit", |b| {
        b.iter(|| commit_execution(black_box(&vm), black_box(&witness)).unwrap())
    });

    let (receipt, m_trace, w_commitment) = commit_execution(&vm, &witness).unwrap();
    let staged_trace =
        llm_provable_computer::transformer_trace::build_shared_transformer_staged_trace(
            &witness,
            config.ff_dim,
        )
        .unwrap();
    let request = AuditRequest {
        target: AuditTarget::Steps(vec![3]),
        challenge: None,
    };

    group.bench_function("audit_generate", |b| {
        b.iter(|| {
            audit_execution(
                black_box(&vm),
                black_box(&m_trace),
                black_box(&w_commitment),
                black_box(&staged_trace),
                black_box(&request),
            )
            .unwrap()
        })
    });

    let response = audit_execution(&vm, &m_trace, &w_commitment, &staged_trace, &request).unwrap();

    group.bench_function("audit_verify", |b| {
        b.iter(|| verify_audit(black_box(&receipt), black_box(&response), None))
    });

    group.finish();
}

criterion_group! {
    name = commit_audit_benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = bench_commit_and_audit
}

criterion_main!(commit_audit_benches);
