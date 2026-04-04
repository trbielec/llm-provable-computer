use llm_provable_computer::{
    prove_cdmn_decision_stark_direct_profiled, verify_cdmn_decision_stark_direct, CdmnTable,
};

fn main() {
    let table: CdmnTable =
        serde_json::from_str(include_str!("../programs/lab_vitals_guardrail.cdmn.json"))
            .expect("lab cdmn json");
    let memory = vec![8_000, 4_000, 11_000, 170, 90, 30, 60];

    let report =
        prove_cdmn_decision_stark_direct_profiled(&table, &memory).expect("profiled prove");
    let verified = verify_cdmn_decision_stark_direct(&report.proof).expect("verify");

    println!("verified: {verified}");
    println!(
        "direct profile: total={}us circuit={}us witness={}us trace={}us air={}us boundary={}us",
        report.profile.total_micros,
        report.profile.circuit_build_micros,
        report.profile.witness_build_micros,
        report.profile.trace_materialization_micros,
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
