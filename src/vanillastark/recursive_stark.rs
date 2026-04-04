use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::proof::{verify_execution_stark, VanillaStarkExecutionProof};

use super::field::FieldElement;
use super::rescue_proof_stream::digest_bytes;
use super::verifier_air::VerifierAirPublicInput;

const RECURSIVE_PROOF_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecursiveInnerProof {
    pub proof: VanillaStarkExecutionProof,
    pub claim_digest: FieldElement,
    pub transcript_digest: FieldElement,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecursiveAggregateProof {
    pub version: u32,
    pub inner_proofs: Vec<RecursiveInnerProof>,
    pub aggregate_root: FieldElement,
    pub accepted: bool,
}

pub fn prove_recursive_inner(proof: VanillaStarkExecutionProof) -> Result<RecursiveInnerProof> {
    let accepted = verify_execution_stark(&proof)?;
    Ok(RecursiveInnerProof {
        claim_digest: claim_digest(&proof)?,
        transcript_digest: transcript_digest(&proof),
        proof,
        accepted,
    })
}

pub fn verify_recursive_inner(proof: &RecursiveInnerProof) -> Result<bool> {
    let accepted = verify_execution_stark(&proof.proof)?;
    Ok(accepted == proof.accepted
        && claim_digest(&proof.proof)? == proof.claim_digest
        && transcript_digest(&proof.proof) == proof.transcript_digest)
}

pub fn prove_verification(proof: &VanillaStarkExecutionProof) -> Result<RecursiveAggregateProof> {
    prove_aggregate_verification(std::slice::from_ref(proof))
}

pub fn prove_aggregate_verification(
    proofs: &[VanillaStarkExecutionProof],
) -> Result<RecursiveAggregateProof> {
    let mut inner_proofs = Vec::with_capacity(proofs.len());
    for proof in proofs {
        inner_proofs.push(prove_recursive_inner(proof.clone())?);
    }

    let public_input = aggregate_public_input(&inner_proofs);
    Ok(RecursiveAggregateProof {
        version: RECURSIVE_PROOF_VERSION,
        aggregate_root: public_input.aggregate_root(),
        accepted: public_input.accepted,
        inner_proofs,
    })
}

pub fn verify_aggregate_proof(proof: &RecursiveAggregateProof) -> Result<bool> {
    if proof.version != RECURSIVE_PROOF_VERSION {
        return Ok(false);
    }

    let mut accepted = true;
    for inner in &proof.inner_proofs {
        if !verify_recursive_inner(inner)? {
            accepted = false;
            break;
        }
    }

    let public_input = aggregate_public_input(&proof.inner_proofs);
    Ok(accepted
        && public_input.accepted == proof.accepted
        && public_input.aggregate_root() == proof.aggregate_root)
}

fn aggregate_public_input(inner_proofs: &[RecursiveInnerProof]) -> VerifierAirPublicInput {
    VerifierAirPublicInput {
        claim_digests: inner_proofs
            .iter()
            .map(|proof| proof.claim_digest)
            .collect(),
        transcript_digests: inner_proofs
            .iter()
            .map(|proof| proof.transcript_digest)
            .collect(),
        accepted: inner_proofs.iter().all(|proof| proof.accepted),
    }
}

fn claim_digest(proof: &VanillaStarkExecutionProof) -> Result<FieldElement> {
    let bytes = serde_json::to_vec(&proof.claim)
        .map_err(|error| crate::error::VmError::Serialization(error.to_string()))?;
    Ok(digest_bytes(&bytes))
}

fn transcript_digest(proof: &VanillaStarkExecutionProof) -> FieldElement {
    digest_bytes(&proof.proof)
}
