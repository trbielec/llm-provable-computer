//! Minimal vanilla STARK prover/verifier used as the Milestone 2 baseline.

pub mod field;
pub mod fri;
pub mod merkle;
pub mod multivariate;
pub mod ntt;
pub mod polynomial;
pub mod proof_stream;
pub mod recursive_stark;
pub mod rescue_merkle;
pub mod rescue_prime;
pub mod rescue_proof_stream;
pub mod stark;
pub mod verifier_air;

pub use field::{FieldElement, P};
pub use fri::Fri;
pub use merkle::Merkle;
pub use multivariate::MPolynomial;
pub use polynomial::Polynomial;
pub use proof_stream::{ProofObject, ProofStream};
pub use recursive_stark::{
    prove_aggregate_verification, prove_recursive_inner, prove_verification,
    verify_aggregate_proof, verify_recursive_inner, RecursiveAggregateProof, RecursiveInnerProof,
};
pub use rescue_merkle::RescueMerkleTree;
pub use rescue_prime::RescuePrime;
pub use rescue_proof_stream::{digest_bytes as rescue_digest_bytes, RescueProofStream};
pub use stark::{Stark, StarkProofProfile};
pub use verifier_air::VerifierAirPublicInput;

pub trait TranscriptHasher {
    type Digest;

    fn digest_bytes(bytes: &[u8]) -> Self::Digest;
}

pub trait MerkleHasher {
    type Digest;

    fn hash_leaf(bytes: &[u8]) -> Self::Digest;
    fn hash_pair(left: &Self::Digest, right: &Self::Digest) -> Self::Digest;
}
