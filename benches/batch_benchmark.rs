use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use llm_provable_computer::batch::BatchRunner;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::instruction::{Instruction, Program};
use llm_provable_computer::model::TransformerVm;
use llm_provable_computer::state::{MachineState, Word};

fn build_states(count: usize, memory_size: usize) -> Vec<MachineState> {
    (0..count)
        .map(|value| {
            let mut state = MachineState::new(memory_size);
            state.acc = value as Word;
            if let Some(slot) = state.memory.get_mut(0) {
                *slot = value as Word;
            }
            state
        })
        .collect()
}

fn bench_batch_runner(c: &mut Criterion) {
    let config = TransformerVmConfig::default();
    let program = Program::new(vec![Instruction::Halt], 1);
    let model = Arc::new(TransformerVm::new(config, program).unwrap());
    let runner = BatchRunner::new(Arc::clone(&model), 32);

    let mut group = c.benchmark_group("batch_runner");

    for &batch_size in &[64usize, 256, 1024, 4096] {
        let states = build_states(batch_size, 4);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &states,
            |b, states| {
                b.iter_batched(
                    || states.clone(),
                    |cloned_states| {
                        black_box(runner.run(cloned_states));
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = batch_benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = bench_batch_runner
}

criterion_main!(batch_benches);
