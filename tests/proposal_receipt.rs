use llm_provable_computer::proposal_receipt::{AttestationField, DecodeMetadata, ProposalReceipt};

#[test]
fn receipt_can_be_created_and_serialized() {
    let receipt = ProposalReceipt::new(
        b"what is the capital of France?",
        "gpt-model-v1",
        b"The capital of France is Paris.",
        DecodeMetadata {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            seed: Some(42),
        },
        None,
    );

    let json = serde_json::to_string(&receipt).expect("serialization should succeed");
    let deserialized: ProposalReceipt =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(receipt, deserialized);
}

#[test]
fn receipt_hash_is_stable() {
    let make_receipt = || {
        ProposalReceipt::new(
            b"stable prompt",
            "model-abc",
            b"stable output",
            DecodeMetadata {
                temperature: 1.0,
                top_p: 1.0,
                max_tokens: 128,
                seed: None,
            },
            None,
        )
    };

    let r1 = make_receipt();
    let r2 = make_receipt();

    assert_eq!(r1.receipt_hash, r2.receipt_hash);
}

#[test]
fn receipt_integrity_passes_for_valid_receipt() {
    let receipt = ProposalReceipt::new(
        b"integrity check prompt",
        "model-v2",
        b"integrity check output",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.95,
            max_tokens: 64,
            seed: Some(99),
        },
        None,
    );

    assert!(receipt.verify_integrity());
}

#[test]
fn receipt_integrity_fails_after_tampering_output_hash() {
    let mut receipt = ProposalReceipt::new(
        b"tamper output prompt",
        "model-v2",
        b"original output",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.95,
            max_tokens: 64,
            seed: None,
        },
        None,
    );

    // Flip the first byte of the output hash
    receipt.output_hash[0] ^= 0xFF;

    assert!(!receipt.verify_integrity());
}

#[test]
fn receipt_integrity_fails_after_tampering_prompt_hash() {
    let mut receipt = ProposalReceipt::new(
        b"tamper prompt prompt",
        "model-v2",
        b"some output",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.95,
            max_tokens: 64,
            seed: None,
        },
        None,
    );

    // Flip the first byte of the prompt hash
    receipt.prompt_hash[0] ^= 0xFF;

    assert!(!receipt.verify_integrity());
}

#[test]
fn receipt_integrity_fails_after_tampering_model_id() {
    let mut receipt = ProposalReceipt::new(
        b"tamper model prompt",
        "model-original",
        b"some output",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.95,
            max_tokens: 64,
            seed: None,
        },
        None,
    );

    // Overwrite the model_id after construction
    receipt.model_id = "model-tampered".to_string();

    assert!(!receipt.verify_integrity());
}

#[test]
fn receipt_hash_hex_is_64_chars() {
    let receipt = ProposalReceipt::new(
        b"hex length check",
        "model-hex",
        b"hex output",
        DecodeMetadata {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 16,
            seed: Some(0),
        },
        None,
    );

    let hex = receipt.receipt_hash_hex();
    assert_eq!(hex.len(), 64);
    assert!(hex
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
}

#[test]
fn receipt_with_attestation_round_trips() {
    let attestation = AttestationField {
        issuer: "trusted-issuer.example.com".to_string(),
        signature_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
    };

    let receipt = ProposalReceipt::new(
        b"attested prompt",
        "attested-model-v1",
        b"attested output",
        DecodeMetadata {
            temperature: 0.8,
            top_p: 0.85,
            max_tokens: 512,
            seed: Some(7),
        },
        Some(attestation),
    );

    let json = serde_json::to_string(&receipt).expect("serialization should succeed");
    let deserialized: ProposalReceipt =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(receipt, deserialized);
    assert!(deserialized.attestation.is_some());
}

#[test]
fn receipt_without_attestation_round_trips() {
    let receipt = ProposalReceipt::new(
        b"unatested prompt",
        "model-no-attest",
        b"unattested output",
        DecodeMetadata {
            temperature: 0.3,
            top_p: 0.7,
            max_tokens: 32,
            seed: None,
        },
        None,
    );

    let json = serde_json::to_string(&receipt).expect("serialization should succeed");
    let deserialized: ProposalReceipt =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(receipt, deserialized);
    assert!(deserialized.attestation.is_none());
}

#[test]
fn different_outputs_produce_different_receipt_hashes() {
    let r1 = ProposalReceipt::new(
        b"shared prompt",
        "shared-model",
        b"output version A",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.9,
            max_tokens: 100,
            seed: Some(1),
        },
        None,
    );

    let r2 = ProposalReceipt::new(
        b"shared prompt",
        "shared-model",
        b"output version B",
        DecodeMetadata {
            temperature: 0.5,
            top_p: 0.9,
            max_tokens: 100,
            seed: Some(1),
        },
        None,
    );

    assert_ne!(r1.receipt_hash, r2.receipt_hash);
}
