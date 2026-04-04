use llm_provable_computer::{
    prove_aggregate_verification, prove_execution_stark, verify_aggregate_proof, ProgramCompiler,
    RecursiveAggregateProof, TransformerVmConfig,
};

fn addition_proof() -> llm_provable_computer::VanillaStarkExecutionProof {
    let source = std::fs::read_to_string("programs/addition.tvm").expect("program source");
    let model = ProgramCompiler
        .compile_source(&source, TransformerVmConfig::default())
        .expect("compile");
    prove_execution_stark(&model, 32).expect("prove")
}

#[test]
fn aggregate_proof_round_trips_for_single_inner_proof() {
    let proof = addition_proof();
    let aggregate = prove_aggregate_verification(&[proof]).expect("aggregate");
    assert!(verify_aggregate_proof(&aggregate).expect("verify aggregate"));
}

#[test]
fn aggregate_proof_rejects_tampered_claim_digest() {
    let proof_a = addition_proof();
    let proof_b = addition_proof();
    let mut aggregate: RecursiveAggregateProof =
        prove_aggregate_verification(&[proof_a, proof_b]).expect("aggregate");
    aggregate.inner_proofs[0].claim_digest = aggregate.inner_proofs[0].claim_digest
        + llm_provable_computer::vanillastark::FieldElement::one();
    assert!(!verify_aggregate_proof(&aggregate).expect("verify aggregate"));
}
