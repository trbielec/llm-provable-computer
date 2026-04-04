use llm_provable_computer::{
    canonicalize_for_comparison, compare_approximate, quantize_signed,
    ApproximateVerificationConfig,
};

#[test]
fn compare_approximate_handles_zero_and_negative_values() {
    let config = ApproximateVerificationConfig {
        absolute_epsilon: 0.01,
        relative_epsilon: 0.01,
        max_deviating_fraction: 0.25,
    };

    let result = compare_approximate(&[0.0, -10.0, 5.0], &[0.001, -10.005, 5.0], &config);
    assert!(result.passed);
    assert_eq!(result.deviating_values, 0);
}

#[test]
fn canonicalization_rejects_overflow_and_non_finite_values() {
    assert!(quantize_signed(f64::NAN).is_err());
    assert!(quantize_signed(f64::INFINITY).is_err());
    assert!(quantize_signed(1e40).is_err());
}

#[test]
fn canonicalization_preserves_sign_and_quantization_edges() {
    let values = canonicalize_for_comparison(&[-0.000_000_4, 0.000_000_4, -1.25]).unwrap();
    assert_eq!(values[0], 0.0);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], -1.25);
}

#[test]
fn compare_approximate_rejects_many_small_deviations() {
    let config = ApproximateVerificationConfig {
        absolute_epsilon: 0.001,
        relative_epsilon: 0.001,
        max_deviating_fraction: 0.1,
    };

    let expected = vec![1.0; 20];
    let mut actual = expected.clone();
    for value in &mut actual[..4] {
        *value += 0.01;
    }

    let result = compare_approximate(&expected, &actual, &config);
    assert!(!result.passed);
    assert_eq!(result.deviating_values, 4);
}
