//! Typed, auditable record that an upstream probabilistic model produced a proposal.
//!
//! A [`ProposalReceipt`] binds the prompt, model identity, raw output, and sampling
//! parameters together under a single Blake2b-256 receipt hash that can be
//! independently recomputed by any party holding the same inputs.

use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Sampling parameters that were active during model decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeMetadata {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub seed: Option<u64>,
}

/// Optional external attestation (e.g. a signed hash from a trusted endpoint).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttestationField {
    pub issuer: String,
    pub signature_bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Canonical inner struct used only for receipt_hash computation
// ---------------------------------------------------------------------------

/// Subset of [`ProposalReceipt`] fields that are hashed to produce
/// `receipt_hash`.  `attestation` and `receipt_hash` itself are excluded so
/// that the hash is stable regardless of whether an attestation is present.
#[derive(Serialize)]
struct ReceiptHashInput<'a> {
    prompt_hash: &'a [u8; 32],
    model_id: &'a str,
    output_hash: &'a [u8; 32],
    decode_metadata: &'a DecodeMetadata,
}

// ---------------------------------------------------------------------------
// ProposalReceipt
// ---------------------------------------------------------------------------

/// A typed, auditable record that an upstream probabilistic model produced a
/// proposal.
///
/// All hash fields are computed with Blake2b-512 truncated to 32 bytes.
/// `receipt_hash` covers `prompt_hash`, `model_id`, `output_hash`, and
/// `decode_metadata` serialized as canonical JSON via `serde_json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalReceipt {
    /// Blake2b-256 hash of the prompt/context bytes.
    pub prompt_hash: [u8; 32],
    /// Identifier of the model that generated the proposal.
    pub model_id: String,
    /// Blake2b-256 hash of the raw model output bytes.
    pub output_hash: [u8; 32],
    /// Sampling parameters used during decode.
    pub decode_metadata: DecodeMetadata,
    /// Optional external attestation.
    pub attestation: Option<AttestationField>,
    /// Blake2b-256 hash of the canonical serialization of the other fields
    /// (computed on construction, not supplied by the caller).
    pub receipt_hash: [u8; 32],
}

impl ProposalReceipt {
    /// Create a new receipt.  Computes `receipt_hash` automatically.
    pub fn new(
        prompt_bytes: &[u8],
        model_id: impl Into<String>,
        output_bytes: &[u8],
        decode_metadata: DecodeMetadata,
        attestation: Option<AttestationField>,
    ) -> Self {
        let model_id = model_id.into();
        let prompt_hash = blake2b256(prompt_bytes);
        let output_hash = blake2b256(output_bytes);

        let receipt_hash =
            compute_receipt_hash(&prompt_hash, &model_id, &output_hash, &decode_metadata);

        Self {
            prompt_hash,
            model_id,
            output_hash,
            decode_metadata,
            attestation,
            receipt_hash,
        }
    }

    /// Verify that `receipt_hash` matches a recomputation over the other fields.
    pub fn verify_integrity(&self) -> bool {
        let expected = compute_receipt_hash(
            &self.prompt_hash,
            &self.model_id,
            &self.output_hash,
            &self.decode_metadata,
        );
        self.receipt_hash == expected
    }

    /// Return `receipt_hash` as a lowercase hex string.
    pub fn receipt_hash_hex(&self) -> String {
        self.receipt_hash
            .iter()
            .fold(String::with_capacity(64), |mut acc, byte| {
                use std::fmt::Write;
                write!(acc, "{byte:02x}").expect("writing to String is infallible");
                acc
            })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Blake2b-512 hash, truncated to 32 bytes.
fn blake2b256(data: &[u8]) -> [u8; 32] {
    let full = Blake2b512::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&full[..32]);
    out
}

/// Deterministic receipt hash: `blake2b256(serde_json(ReceiptHashInput))`.
///
/// `serde_json::to_vec` is deterministic for structs with named fields when
/// serialized in a fixed field order, which is guaranteed by the
/// `ReceiptHashInput` definition above.
fn compute_receipt_hash(
    prompt_hash: &[u8; 32],
    model_id: &str,
    output_hash: &[u8; 32],
    decode_metadata: &DecodeMetadata,
) -> [u8; 32] {
    let canonical = ReceiptHashInput {
        prompt_hash,
        model_id,
        output_hash,
        decode_metadata,
    };
    let json_bytes =
        serde_json::to_vec(&canonical).expect("ReceiptHashInput is always serializable to JSON");
    blake2b256(&json_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> DecodeMetadata {
        DecodeMetadata {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 256,
            seed: Some(42),
        }
    }

    #[test]
    fn new_produces_valid_receipt() {
        let receipt = ProposalReceipt::new(
            b"hello world prompt",
            "gpt-4o",
            b"model output text",
            sample_metadata(),
            None,
        );
        assert!(receipt.verify_integrity());
    }

    #[test]
    fn receipt_hash_hex_is_64_chars() {
        let receipt = ProposalReceipt::new(
            b"prompt",
            "claude-3-5-sonnet",
            b"output",
            sample_metadata(),
            None,
        );
        assert_eq!(receipt.receipt_hash_hex().len(), 64);
    }

    #[test]
    fn tampered_model_id_fails_integrity() {
        let mut receipt =
            ProposalReceipt::new(b"prompt", "gpt-4o", b"output", sample_metadata(), None);
        receipt.model_id = "evil-model".into();
        assert!(!receipt.verify_integrity());
    }

    #[test]
    fn tampered_prompt_hash_fails_integrity() {
        let mut receipt =
            ProposalReceipt::new(b"prompt", "gpt-4o", b"output", sample_metadata(), None);
        receipt.prompt_hash[0] ^= 0xff;
        assert!(!receipt.verify_integrity());
    }

    #[test]
    fn attestation_does_not_affect_receipt_hash() {
        let without_attestation =
            ProposalReceipt::new(b"prompt", "gpt-4o", b"output", sample_metadata(), None);
        let with_attestation = ProposalReceipt::new(
            b"prompt",
            "gpt-4o",
            b"output",
            sample_metadata(),
            Some(AttestationField {
                issuer: "trusted-endpoint".into(),
                signature_bytes: vec![1, 2, 3],
            }),
        );
        assert_eq!(
            without_attestation.receipt_hash,
            with_attestation.receipt_hash
        );
    }

    #[test]
    fn round_trip_serde() {
        let receipt = ProposalReceipt::new(
            b"some prompt bytes",
            "claude-3-5-sonnet",
            b"some output bytes",
            sample_metadata(),
            Some(AttestationField {
                issuer: "test-issuer".into(),
                signature_bytes: vec![0xde, 0xad, 0xbe, 0xef],
            }),
        );
        let json = serde_json::to_string(&receipt).expect("serialize");
        let restored: ProposalReceipt = serde_json::from_str(&json).expect("deserialize");
        assert!(restored.verify_integrity());
        assert_eq!(receipt.receipt_hash, restored.receipt_hash);
    }
}
