use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_provable_computer::candidate_eval::SelectionStrategy;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::overlap::{FallbackProgram, OverlapOrchestrator, OverlapStrategy};
use llm_provable_computer::planner_bridge::{PlannerContext, StaticPlanner};
use llm_provable_computer::state::MachineState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_context() -> PlannerContext {
    PlannerContext::new(
        b"bench-prompt".to_vec(),
        "bench-model".to_string(),
        MachineState::default(),
    )
}

fn make_fallback() -> FallbackProgram {
    FallbackProgram::from_source("LOADI 0\nHALT").expect("fallback program must parse")
}

fn k4_planner() -> StaticPlanner {
    StaticPlanner::new(vec![
        ("LOADI 3\nHALT".to_string(), b"a".to_vec()),
        ("LOADI 7\nHALT".to_string(), b"b".to_vec()),
        ("LOADI 12\nHALT".to_string(), b"c".to_vec()),
        ("LOADI 5\nHALT".to_string(), b"d".to_vec()),
    ])
}

fn make_orchestrator(strategy: OverlapStrategy) -> OverlapOrchestrator {
    OverlapOrchestrator::new(
        TransformerVmConfig::default(),
        64,
        SelectionStrategy::HighestAccumulator,
        make_fallback(),
    )
    .with_strategy(strategy)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap");

    // Sequential k=4: measures baseline pipeline latency with no parallelism.
    group.bench_function("sequential_k4", |b| {
        let orchestrator = make_orchestrator(OverlapStrategy::Sequential);
        let planner = k4_planner();
        b.iter(|| {
            let report = orchestrator
                .run(black_box(&planner), black_box(make_context()))
                .expect("sequential benchmark run must succeed");
            black_box(report)
        });
    });

    // Overlapped k=4: measures pipelined latency; savings vs sequential are the key metric.
    group.bench_function("overlapped_k4", |b| {
        let orchestrator = make_orchestrator(OverlapStrategy::Overlapped);
        let planner = k4_planner();
        b.iter(|| {
            let report = orchestrator
                .run(black_box(&planner), black_box(make_context()))
                .expect("overlapped benchmark run must succeed");
            black_box(report)
        });
    });

    group.finish();
}

criterion_group! {
    name = overlap_benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = bench_overlap
}

criterion_main!(overlap_benches);
