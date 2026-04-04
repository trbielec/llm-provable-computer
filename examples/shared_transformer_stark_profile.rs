use llm_provable_computer::{
    prove_shared_transformer_staged_skeleton_stark_profiled,
    verify_shared_transformer_staged_skeleton_stark, ProgramCompiler, TransformerVmConfig,
};

fn main() {
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
        prove_shared_transformer_staged_skeleton_stark_profiled(&model, 16, 32).expect("prove");
    let verified = verify_shared_transformer_staged_skeleton_stark(&report.proof).expect("verify");

    println!("verified: {verified}");
    println!(
        "shared transformer profile: total={}us witness={}us staged_trace={}us validate={}us skeleton_trace={}us air={}us boundary={}us",
        report.profile.total_micros,
        report.profile.witness_build_micros,
        report.profile.staged_trace_build_micros,
        report.profile.trace_validation_micros,
        report.profile.skeleton_trace_build_micros,
        report.profile.air_build_micros,
        report.profile.boundary_build_micros,
    );
    println!(
        "stark profile: total={}us randomizers={}us interpolate={}us boundary_commit={}us transition={}us randomizer_commit={}us combination={}us fri={}us openings={}us",
        report.profile.stark.total_micros,
        report.profile.stark.randomizer_rows_micros,
        report.profile.stark.trace_interpolation_micros,
        report.profile.stark.boundary_quotient_commit_micros,
        report.profile.stark.transition_quotient_micros,
        report.profile.stark.randomizer_commit_micros,
        report.profile.stark.combination_micros,
        report.profile.stark.fri_prove_micros,
        report.profile.stark.opening_micros,
    );
}
