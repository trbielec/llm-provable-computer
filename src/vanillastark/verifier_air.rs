use serde::{Deserialize, Serialize};

use super::field::FieldElement;
use super::rescue_proof_stream::digest_elements;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierAirPublicInput {
    pub claim_digests: Vec<FieldElement>,
    pub transcript_digests: Vec<FieldElement>,
    pub accepted: bool,
}

impl VerifierAirPublicInput {
    pub fn aggregate_root(&self) -> FieldElement {
        let mut elements =
            Vec::with_capacity(self.claim_digests.len() + self.transcript_digests.len() + 1);
        elements.extend(self.claim_digests.iter().copied());
        elements.extend(self.transcript_digests.iter().copied());
        elements.push(if self.accepted {
            FieldElement::one()
        } else {
            FieldElement::zero()
        });
        digest_elements(&elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_root_changes_when_claim_digest_changes() {
        let first = VerifierAirPublicInput {
            claim_digests: vec![FieldElement::new(1)],
            transcript_digests: vec![FieldElement::new(2)],
            accepted: true,
        };
        let second = VerifierAirPublicInput {
            claim_digests: vec![FieldElement::new(3)],
            transcript_digests: vec![FieldElement::new(2)],
            accepted: true,
        };
        assert_ne!(first.aggregate_root(), second.aggregate_root());
    }
}
