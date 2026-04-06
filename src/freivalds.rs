use crate::approximate::{
    compare_approximate, dequantize_signed, ApproximateComparisonResult,
    ApproximateVerificationConfig,
};
use crate::vanillastark::field::{FieldElement, P};
use serde::{Deserialize, Serialize};

/// Convert a signed `i128` value into a `FieldElement`.
///
/// Non-negative values map directly; negative values are represented as `P - |v|`
/// in the prime field.
pub fn field_element_from_signed(v: i128) -> FieldElement {
    if v >= 0 {
        FieldElement::new(v as u128)
    } else {
        FieldElement::new(P - ((-v) as u128))
    }
}

/// Represents a challenge for exact Freivalds checks over the STARK field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreivaldsChallenge {
    pub v: Vec<FieldElement>,
}

impl FreivaldsChallenge {
    pub fn new(size: usize, seed: &[u8]) -> Self {
        let mut v = Vec::with_capacity(size);
        for i in 0..size {
            let mut hash_input = seed.to_vec();
            hash_input.extend_from_slice(&i.to_be_bytes());
            v.push(FieldElement::sample(&hash_input));
        }
        Self { v }
    }
}

/// Approximate Freivalds challenge over the audit signed fixed-point domain.
///
/// Each coefficient is sampled as +/-1 so the projected accumulator stays in the
/// same numeric domain as the auditable payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximateFreivaldsChallenge {
    pub v: Vec<i128>,
}

impl ApproximateFreivaldsChallenge {
    pub fn new(size: usize, seed: &[u8]) -> Self {
        let mut v = Vec::with_capacity(size);
        for i in 0..size {
            let mut hash_input = seed.to_vec();
            hash_input.extend_from_slice(&i.to_be_bytes());
            let sample = FieldElement::sample(&hash_input);
            v.push(if sample.value() & 1 == 0 { 1 } else { -1 });
        }
        Self { v }
    }
}

/// Verification of matrix-vector multiplication y = Wx using exact Freivalds' algorithm.
pub fn verify_matrix_vector_product(
    challenge: &FreivaldsChallenge,
    x: &[FieldElement],
    y: &[FieldElement],
    vt_w: &[FieldElement],
) -> bool {
    if challenge.v.len() != y.len() || vt_w.len() != x.len() {
        return false;
    }

    let mut left_side = FieldElement::zero();
    for (vi, yi) in challenge.v.iter().zip(y.iter()) {
        left_side = left_side + (*vi * *yi);
    }

    let mut right_side = FieldElement::zero();
    for (vwi, xi) in vt_w.iter().zip(x.iter()) {
        right_side = right_side + (*vwi * *xi);
    }

    left_side == right_side
}

pub fn precompute_vt_w(
    challenge: &FreivaldsChallenge,
    weights: &[Vec<FieldElement>],
) -> Vec<FieldElement> {
    let rows = weights.len();
    if rows == 0 {
        return Vec::new();
    }

    let cols = weights[0].len();
    let mut vt_w = vec![FieldElement::zero(); cols];
    for j in 0..cols {
        for (i, weight_row) in weights.iter().enumerate().take(rows) {
            vt_w[j] = vt_w[j] + (challenge.v[i] * weight_row[j]);
        }
    }
    vt_w
}

/// Precompute v^T * W in the fixed-point audit domain.
///
/// The challenge coefficients are +/-1, so the accumulator already stays in the
/// same scale as `weights`.
pub fn precompute_approximate_vt_w(
    challenge: &ApproximateFreivaldsChallenge,
    weights: &[Vec<i128>],
) -> Vec<i128> {
    let rows = weights.len();
    if rows == 0 {
        return Vec::new();
    }

    let cols = weights[0].len();
    let mut vt_w = vec![0i128; cols];
    for j in 0..cols {
        for (i, weight_row) in weights.iter().enumerate().take(rows) {
            vt_w[j] += challenge.v[i] * weight_row[j];
        }
    }
    vt_w
}

/// Approximate projected Freivalds check in the audit fixed-point domain.
///
/// `scale` is the fixed-point scale shared by `x`, `y`, and `weights`. The
/// projected accumulator divides by `scale` once per multiply-accumulate to stay
/// in that same domain. The final scalar equality is checked with mixed absolute
/// and relative tolerances.
pub fn verify_matrix_vector_product_approximate(
    challenge: &ApproximateFreivaldsChallenge,
    x: &[i128],
    y: &[i128],
    vt_w: &[i128],
    scale: i128,
    config: &ApproximateVerificationConfig,
) -> ApproximateComparisonResult {
    if challenge.v.len() != y.len() || vt_w.len() != x.len() || scale == 0 {
        return ApproximateComparisonResult {
            total_values: 1,
            deviating_values: 1,
            max_absolute_error: f64::INFINITY,
            max_relative_error: f64::INFINITY,
            passed: false,
        };
    }

    let mut left_side = 0i128;
    for (vi, yi) in challenge.v.iter().zip(y.iter()) {
        left_side += vi * yi;
    }

    let mut right_side = 0i128;
    for (vwi, xi) in vt_w.iter().zip(x.iter()) {
        right_side += (vwi * xi) / scale;
    }

    compare_approximate(
        &[dequantize_signed(left_side)],
        &[dequantize_signed(right_side)],
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freivalds_success() {
        let size = 4;
        let challenge = FreivaldsChallenge::new(size, b"test_seed");
        let weights = vec![
            vec![FieldElement::new(1), FieldElement::new(2)],
            vec![FieldElement::new(3), FieldElement::new(4)],
            vec![FieldElement::new(5), FieldElement::new(6)],
            vec![FieldElement::new(7), FieldElement::new(8)],
        ];
        let x = vec![FieldElement::new(10), FieldElement::new(20)];
        let y = vec![
            FieldElement::new(50),
            FieldElement::new(110),
            FieldElement::new(170),
            FieldElement::new(230),
        ];

        let vt_w = precompute_vt_w(&challenge, &weights);
        assert!(verify_matrix_vector_product(&challenge, &x, &y, &vt_w));
    }

    #[test]
    fn test_freivalds_failure() {
        let size = 4;
        let challenge = FreivaldsChallenge::new(size, b"test_seed");
        let weights = vec![
            vec![FieldElement::new(1), FieldElement::new(2)],
            vec![FieldElement::new(3), FieldElement::new(4)],
            vec![FieldElement::new(5), FieldElement::new(6)],
            vec![FieldElement::new(7), FieldElement::new(8)],
        ];
        let x = vec![FieldElement::new(10), FieldElement::new(20)];
        let y = vec![
            FieldElement::new(51),
            FieldElement::new(110),
            FieldElement::new(170),
            FieldElement::new(230),
        ];

        let vt_w = precompute_vt_w(&challenge, &weights);
        assert!(!verify_matrix_vector_product(&challenge, &x, &y, &vt_w));
    }

    #[test]
    fn approximate_freivalds_accepts_small_projected_drift() {
        let challenge = ApproximateFreivaldsChallenge::new(2, b"approx-seed");
        let scale = 1_000_000i128;
        let weights = vec![vec![scale, 2 * scale], vec![3 * scale, 4 * scale]];
        let x = vec![10 * scale, 20 * scale];
        let y = vec![50 * scale + 50, 110 * scale - 50];
        let vt_w = precompute_approximate_vt_w(&challenge, &weights);
        let config = ApproximateVerificationConfig {
            absolute_epsilon: 0.001,
            relative_epsilon: 0.000_01,
            max_deviating_fraction: 0.0,
        };

        let result =
            verify_matrix_vector_product_approximate(&challenge, &x, &y, &vt_w, scale, &config);
        assert!(result.passed);
    }
}
