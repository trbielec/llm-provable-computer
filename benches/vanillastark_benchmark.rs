use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llm_provable_computer::vanillastark::{FieldElement, MPolynomial, RescuePrime, Stark};
use llm_provable_computer::{
    compile_cdmn_table_to_decision_program, compile_cdmn_table_with_metadata,
    compile_dmn_table_with_metadata, estimate_compiled_cdmn_decision_stark_direct_cost,
    estimate_execution_stark_cost, prove_cdmn_decision_stark_direct, prove_execution_stark,
    verify_cdmn_decision_stark_direct, verify_execution_stark, CdmnDecisionStarkDirectCostEstimate,
    CdmnTable, DmnTable, ProgramCompiler, TransformerVm, TransformerVmConfig,
    VanillaStarkCostEstimate, VanillaStarkProofOptions,
};

/// Shared setup: build the Stark instance, trace, AIR, and boundary once.
struct StarkFixture {
    stark: Stark,
    trace: Vec<Vec<FieldElement>>,
    air: Vec<MPolynomial>,
    boundary: Vec<(usize, usize, FieldElement)>,
}

struct PolicyProofFixture {
    label: &'static str,
    model: TransformerVm,
    max_steps: usize,
    metrics: VanillaStarkCostEstimate,
}

struct DirectPolicyProofFixture {
    label: &'static str,
    table: CdmnTable,
    initial_memory: Vec<i64>,
    metrics: CdmnDecisionStarkDirectCostEstimate,
}

fn setup_fixture() -> StarkFixture {
    let rp = RescuePrime::new();
    let input = FieldElement::sample(b"0xdeadbeef");
    let output = rp.hash(input);

    let expansion_factor = 4;
    let num_colinearity_checks = 2;
    let security_level = 2;
    let num_cycles = rp.n + 1;
    let state_width = rp.m;

    let stark = Stark::new(
        expansion_factor,
        num_colinearity_checks,
        security_level,
        state_width,
        num_cycles,
        2,
    );

    let trace = rp.trace(input);
    let air = rp.transition_constraints(stark.omicron);
    let boundary = rp.boundary_constraints(output);

    StarkFixture {
        stark,
        trace,
        air,
        boundary,
    }
}

fn build_policy_fixture(
    label: &'static str,
    assembly: &str,
    memory_updates: &[(usize, i64)],
    max_steps: usize,
) -> PolicyProofFixture {
    let config = TransformerVmConfig::default();
    let compiled = ProgramCompiler
        .compile_source(assembly, config.clone())
        .expect("compile policy program");
    let mut initial_memory = compiled.program().initial_memory().to_vec();
    for &(address, value) in memory_updates {
        initial_memory[address] = value;
    }
    let program = compiled
        .program()
        .clone()
        .with_initial_memory(initial_memory)
        .expect("initial memory");
    let model = TransformerVm::new(config, program).expect("transformer vm");
    let metrics =
        estimate_execution_stark_cost(&model, max_steps, VanillaStarkProofOptions::default())
            .expect("estimate stark cost");
    PolicyProofFixture {
        label,
        model,
        max_steps,
        metrics,
    }
}

fn policy_benchmark_fixtures() -> Vec<PolicyProofFixture> {
    let dmn_table: DmnTable =
        serde_json::from_str(include_str!("../programs/loan_rules.dmn.json")).expect("dmn json");
    let dmn_compiled = compile_dmn_table_with_metadata(&dmn_table).expect("compile dmn");

    let medication_table: CdmnTable =
        serde_json::from_str(include_str!("../programs/medication_safety.cdmn.json"))
            .expect("medication cdmn json");
    let medication_compiled =
        compile_cdmn_table_with_metadata(&medication_table).expect("compile medication cdmn");

    let lab_table: CdmnTable =
        serde_json::from_str(include_str!("../programs/lab_vitals_guardrail.cdmn.json"))
            .expect("lab cdmn json");
    let lab_compiled = compile_cdmn_table_with_metadata(&lab_table).expect("compile lab cdmn");

    vec![
        build_policy_fixture(
            "dmn_loan_first_hit",
            &dmn_compiled.assembly,
            &[(0, 750), (1, 25_000)],
            256,
        ),
        build_policy_fixture(
            "cdmn_medication_quantified",
            &medication_compiled.assembly,
            &[
                (0, 500),
                (1, 11),
                (2, 12),
                (3, 99),
                (4, 2),
                (5, 400),
                (6, 7),
                (7, 450),
                (8, 12),
            ],
            1_000,
        ),
        build_policy_fixture(
            "cdmn_lab_ranges",
            &lab_compiled.assembly,
            &[
                (0, 8_000),
                (1, 4_000),
                (2, 11_000),
                (3, 120),
                (4, 80),
                (5, 30),
                (6, 60),
            ],
            1_000,
        ),
    ]
}

fn direct_policy_benchmark_fixtures() -> Vec<DirectPolicyProofFixture> {
    let medication_table: CdmnTable =
        serde_json::from_str(include_str!("../programs/medication_safety.cdmn.json"))
            .expect("medication cdmn json");
    let medication_program = compile_cdmn_table_to_decision_program(&medication_table)
        .expect("compile medication decision");
    let medication_metrics = estimate_compiled_cdmn_decision_stark_direct_cost(
        &medication_program,
        VanillaStarkProofOptions::default(),
    )
    .expect("estimate medication direct cost");

    let lab_table: CdmnTable =
        serde_json::from_str(include_str!("../programs/lab_vitals_guardrail.cdmn.json"))
            .expect("lab cdmn json");
    let lab_program =
        compile_cdmn_table_to_decision_program(&lab_table).expect("compile lab decision");
    let lab_metrics = estimate_compiled_cdmn_decision_stark_direct_cost(
        &lab_program,
        VanillaStarkProofOptions::default(),
    )
    .expect("estimate lab direct cost");

    vec![
        DirectPolicyProofFixture {
            label: "direct_cdmn_medication_quantified",
            table: medication_table,
            initial_memory: vec![500, 11, 12, 99, 2, 400, 11, 450, 12, 0, 0],
            metrics: medication_metrics,
        },
        DirectPolicyProofFixture {
            label: "direct_cdmn_lab_ranges",
            table: lab_table,
            initial_memory: vec![8_000, 4_000, 11_000, 120, 80, 30, 60],
            metrics: lab_metrics,
        },
    ]
}

fn bench_rescue_prime_hash(c: &mut Criterion) {
    let rp = RescuePrime::new();
    let input = FieldElement::sample(b"0xdeadbeef");

    c.bench_function("rescue_prime_hash", |b| b.iter(|| rp.hash(input)));
}

fn bench_rescue_prime_trace(c: &mut Criterion) {
    let rp = RescuePrime::new();
    let input = FieldElement::sample(b"0xdeadbeef");

    c.bench_function("rescue_prime_trace", |b| b.iter(|| rp.trace(input)));
}

fn bench_transition_constraints(c: &mut Criterion) {
    let rp = RescuePrime::new();
    let stark = Stark::new(4, 2, 2, rp.m, rp.n + 1, 2);

    c.bench_function("transition_constraints", |b| {
        b.iter(|| rp.transition_constraints(stark.omicron))
    });
}

fn bench_stark_prove(c: &mut Criterion) {
    let fixture = setup_fixture();

    c.bench_function("stark_prove", |b| {
        b.iter(|| {
            fixture
                .stark
                .prove(&fixture.trace, &fixture.air, &fixture.boundary)
        })
    });
}

fn bench_stark_verify(c: &mut Criterion) {
    let fixture = setup_fixture();
    let proof = fixture
        .stark
        .prove(&fixture.trace, &fixture.air, &fixture.boundary);

    c.bench_function("stark_verify", |b| {
        b.iter(|| {
            fixture
                .stark
                .verify(&proof, &fixture.air, &fixture.boundary)
        })
    });
}

fn bench_stark_e2e(c: &mut Criterion) {
    let fixture = setup_fixture();

    c.bench_function("stark_e2e", |b| {
        b.iter(|| {
            let proof = fixture
                .stark
                .prove(&fixture.trace, &fixture.air, &fixture.boundary);
            let verdict = fixture
                .stark
                .verify(&proof, &fixture.air, &fixture.boundary);
            assert!(verdict);
        })
    });
}

fn bench_stark_proof_size(c: &mut Criterion) {
    let fixture = setup_fixture();

    c.bench_function("stark_proof_size", |b| {
        b.iter(|| {
            let proof = fixture
                .stark
                .prove(&fixture.trace, &fixture.air, &fixture.boundary);
            proof.len()
        })
    });
}

fn bench_field_ops(c: &mut Criterion) {
    let a = FieldElement::new(85408008396924667383611388730472331217);
    let b = FieldElement::new(174420698556543096520990950387834928928);

    let mut group = c.benchmark_group("field_ops");
    group.bench_function("add", |bench| bench.iter(|| a + b));
    group.bench_function("mul", |bench| bench.iter(|| a * b));
    group.bench_function("inv", |bench| bench.iter(|| a.inverse()));
    group.bench_function("pow_small", |bench| bench.iter(|| a.pow(3)));
    group.bench_function("pow_large", |bench| bench.iter(|| a.pow(b.value())));
    group.finish();
}

fn bench_polynomial_ops(c: &mut Criterion) {
    use llm_provable_computer::vanillastark::Polynomial;

    let coeffs: Vec<FieldElement> = (0..64).map(|i| FieldElement::new(i + 1)).collect();
    let poly = Polynomial::new(coeffs);
    let point = FieldElement::new(42);

    let mut group = c.benchmark_group("polynomial_ops");
    group.bench_function("evaluate_deg63", |b| b.iter(|| poly.evaluate(point)));

    let domain: Vec<FieldElement> = (0..8).map(|i| FieldElement::new(i + 1)).collect();
    let values: Vec<FieldElement> = (0..8).map(|i| FieldElement::new(i * 7 + 3)).collect();
    group.bench_function("interpolate_8pts", |b| {
        b.iter(|| Polynomial::interpolate_domain(&domain, &values))
    });
    group.finish();
}

fn bench_fri(c: &mut Criterion) {
    use llm_provable_computer::vanillastark::{Fri, Polynomial, ProofStream};

    let degree = 63;
    let expansion_factor = 4;
    let num_colinearity_tests = 17;
    let codeword_length = (degree + 1) * expansion_factor;

    let omega = FieldElement::primitive_nth_root(codeword_length as u128);
    let generator = FieldElement::generator();
    let fri = Fri::new(
        generator,
        omega,
        codeword_length,
        expansion_factor,
        num_colinearity_tests,
    );

    let polynomial = Polynomial::new((0..=degree).map(|i| FieldElement::new(i as u128)).collect());
    let domain: Vec<FieldElement> = (0..codeword_length).map(|i| omega.pow(i as u128)).collect();
    let codeword = polynomial.evaluate_domain(&domain);

    let mut group = c.benchmark_group("fri");

    group.bench_function("prove_deg63", |b| {
        b.iter(|| {
            let mut ps = ProofStream::new();
            fri.prove(&codeword, &mut ps);
        })
    });

    // Pre-generate proof for verify bench
    let mut ps = ProofStream::new();
    fri.prove(&codeword, &mut ps);
    let verify_ps = ps.clone();

    group.bench_function("verify_deg63", |b| {
        b.iter(|| {
            let mut ps = verify_ps.clone();
            ps.read_index = 0;
            let mut points = Vec::new();
            fri.verify(&mut ps, &mut points);
        })
    });

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let rp = RescuePrime::new();
    let input = FieldElement::sample(b"0xdeadbeef");
    let output = rp.hash(input);
    let trace = rp.trace(input);
    let num_cycles = rp.n + 1;
    let state_width = rp.m;

    let mut group = c.benchmark_group("scaling");

    for &checks in &[2, 4] {
        let security_level = 2 * checks;
        let stark = Stark::new(4, checks, security_level, state_width, num_cycles, 2);
        let air = rp.transition_constraints(stark.omicron);
        let boundary = rp.boundary_constraints(output);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("checks={}", checks)),
            &checks,
            |b, _| b.iter(|| stark.prove(&trace, &air, &boundary)),
        );

        let proof = stark.prove(&trace, &air, &boundary);
        group.bench_with_input(
            BenchmarkId::new("verify", format!("checks={}", checks)),
            &checks,
            |b, _| b.iter(|| stark.verify(&proof, &air, &boundary)),
        );
    }

    group.finish();
}

fn bench_policy_proofs(c: &mut Criterion) {
    let fixtures = policy_benchmark_fixtures();
    let mut group = c.benchmark_group("policy_proofs");
    group.sample_size(3);
    group.measurement_time(Duration::from_secs(20));

    for fixture in &fixtures {
        let id = format!(
            "{}_rows{}_cols{}_fri{}",
            fixture.label,
            fixture.metrics.trace_rows,
            fixture.metrics.trace_width,
            fixture.metrics.fri_domain_length
        );
        group.bench_with_input(
            BenchmarkId::new("prove", id.clone()),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    prove_execution_stark(black_box(&fixture.model), fixture.max_steps)
                        .expect("prove")
                })
            },
        );

        let proof = prove_execution_stark(&fixture.model, fixture.max_steps).expect("prove");
        group.bench_with_input(
            BenchmarkId::new("verify", id.clone()),
            fixture,
            |b, _fixture| {
                b.iter(|| {
                    assert!(verify_execution_stark(black_box(&proof)).expect("verify"));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("proof_size", id),
            fixture,
            |b, _fixture| {
                b.iter(|| {
                    let proof = prove_execution_stark(black_box(&fixture.model), fixture.max_steps)
                        .expect("prove");
                    black_box(proof.proof.len())
                })
            },
        );
    }
    group.finish();
}

fn bench_direct_policy_proofs(c: &mut Criterion) {
    let fixtures = direct_policy_benchmark_fixtures();
    let mut group = c.benchmark_group("direct_policy_proofs");
    group.sample_size(3);
    group.measurement_time(Duration::from_secs(20));

    for fixture in &fixtures {
        let id = format!(
            "{}_rows{}_cols{}_fri{}",
            fixture.label,
            fixture.metrics.trace_rows,
            fixture.metrics.register_count,
            fixture.metrics.fri_domain_length
        );
        group.bench_with_input(
            BenchmarkId::new("prove", id.clone()),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    prove_cdmn_decision_stark_direct(
                        black_box(&fixture.table),
                        black_box(&fixture.initial_memory),
                    )
                    .expect("direct prove")
                })
            },
        );

        let proof = prove_cdmn_decision_stark_direct(&fixture.table, &fixture.initial_memory)
            .expect("direct prove");
        group.bench_with_input(
            BenchmarkId::new("verify", id.clone()),
            fixture,
            |b, _fixture| {
                b.iter(|| {
                    assert!(verify_cdmn_decision_stark_direct(black_box(&proof))
                        .expect("direct verify"));
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("proof_size", id), fixture, |b, fixture| {
            b.iter(|| {
                let proof = prove_cdmn_decision_stark_direct(
                    black_box(&fixture.table),
                    black_box(&fixture.initial_memory),
                )
                .expect("direct prove");
                black_box(proof.proof.len())
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = field_benches;
    config = Criterion::default().sample_size(100);
    targets = bench_field_ops
}

criterion_group! {
    name = polynomial_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_polynomial_ops
}

criterion_group! {
    name = hash_benches;
    config = Criterion::default().sample_size(50);
    targets = bench_rescue_prime_hash, bench_rescue_prime_trace, bench_transition_constraints
}

criterion_group! {
    name = fri_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_fri
}

criterion_group! {
    name = stark_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_stark_prove, bench_stark_verify, bench_stark_e2e, bench_stark_proof_size
}

criterion_group! {
    name = scaling_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_scaling
}

criterion_group! {
    name = policy_benches;
    config = Criterion::default().sample_size(3).measurement_time(Duration::from_secs(20));
    targets = bench_policy_proofs, bench_direct_policy_proofs
}

criterion_main!(
    field_benches,
    polynomial_benches,
    hash_benches,
    fri_benches,
    stark_benches,
    scaling_benches,
    policy_benches,
);
