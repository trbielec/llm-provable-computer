use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const AUDIT_NUMERIC_SCALE: i128 = 1_000_000;
pub const AUDIT_NUMERIC_SCALE_F64: f64 = AUDIT_NUMERIC_SCALE as f64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateNumericError {
    message: String,
}

impl ApproximateNumericError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ApproximateNumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ApproximateNumericError {}

pub type ApproximateResult<T> = std::result::Result<T, ApproximateNumericError>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ApproximateVerificationConfig {
    pub absolute_epsilon: f64,
    pub relative_epsilon: f64,
    pub max_deviating_fraction: f64,
}

impl ApproximateVerificationConfig {
    pub fn validate(&self) -> ApproximateResult<()> {
        if !self.absolute_epsilon.is_finite() || self.absolute_epsilon < 0.0 {
            return Err(ApproximateNumericError::new(
                "absolute_epsilon must be finite and non-negative",
            ));
        }
        if !self.relative_epsilon.is_finite() || self.relative_epsilon < 0.0 {
            return Err(ApproximateNumericError::new(
                "relative_epsilon must be finite and non-negative",
            ));
        }
        if !self.max_deviating_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_deviating_fraction)
        {
            return Err(ApproximateNumericError::new(
                "max_deviating_fraction must be finite and in [0, 1]",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApproximateComparisonResult {
    pub total_values: usize,
    pub deviating_values: usize,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub passed: bool,
}

pub fn quantize_signed(value: f64) -> ApproximateResult<i128> {
    if !value.is_finite() {
        return Err(ApproximateNumericError::new(
            "auditable numeric values must be finite",
        ));
    }

    let scaled = value * AUDIT_NUMERIC_SCALE_F64;
    if !scaled.is_finite() {
        return Err(ApproximateNumericError::new(
            "scaled auditable numeric value overflowed f64",
        ));
    }

    if scaled < i128::MIN as f64 || scaled > i128::MAX as f64 {
        return Err(ApproximateNumericError::new(
            "auditable numeric value is out of range for i128 fixed-point encoding",
        ));
    }

    Ok(scaled.round() as i128)
}

pub fn dequantize_signed(value: i128) -> f64 {
    value as f64 / AUDIT_NUMERIC_SCALE_F64
}

pub fn canonicalize_values(values: &[f64]) -> ApproximateResult<Vec<i128>> {
    values.iter().copied().map(quantize_signed).collect()
}

pub fn canonicalize_for_comparison(values: &[f64]) -> ApproximateResult<Vec<f64>> {
    canonicalize_values(values).map(|values| values.into_iter().map(dequantize_signed).collect())
}

pub fn absolute_error(expected: f64, actual: f64) -> f64 {
    (expected - actual).abs()
}

pub fn relative_error(expected: f64, actual: f64) -> f64 {
    let scale = expected.abs().max(actual.abs());
    if scale == 0.0 {
        0.0
    } else {
        absolute_error(expected, actual) / scale
    }
}

pub fn within_tolerance(
    expected: f64,
    actual: f64,
    config: &ApproximateVerificationConfig,
) -> bool {
    let abs_error = absolute_error(expected, actual);
    abs_error <= config.absolute_epsilon
        || relative_error(expected, actual) <= config.relative_epsilon
}

pub fn compare_approximate(
    expected: &[f64],
    actual: &[f64],
    config: &ApproximateVerificationConfig,
) -> ApproximateComparisonResult {
    if expected.len() != actual.len() {
        let total_values = expected.len().max(actual.len());
        return ApproximateComparisonResult {
            total_values,
            deviating_values: total_values,
            max_absolute_error: f64::INFINITY,
            max_relative_error: f64::INFINITY,
            passed: false,
        };
    }

    let total_values = expected.len();
    if total_values == 0 {
        return ApproximateComparisonResult {
            total_values: 0,
            deviating_values: 0,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            passed: true,
        };
    }

    let mut deviating_values = 0usize;
    let mut max_absolute_error = 0.0f64;
    let mut max_relative_error = 0.0f64;
    let mut invalid_value_seen = false;

    for (&expected_value, &actual_value) in expected.iter().zip(actual.iter()) {
        if !expected_value.is_finite() || !actual_value.is_finite() {
            invalid_value_seen = true;
            deviating_values += 1;
            max_absolute_error = f64::INFINITY;
            max_relative_error = f64::INFINITY;
            continue;
        }

        let abs_error = absolute_error(expected_value, actual_value);
        let rel_error = relative_error(expected_value, actual_value);
        max_absolute_error = max_absolute_error.max(abs_error);
        max_relative_error = max_relative_error.max(rel_error);

        if !within_tolerance(expected_value, actual_value, config) {
            deviating_values += 1;
        }
    }

    let deviating_fraction = deviating_values as f64 / total_values as f64;
    ApproximateComparisonResult {
        total_values,
        deviating_values,
        max_absolute_error,
        max_relative_error,
        passed: !invalid_value_seen && deviating_fraction <= config.max_deviating_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_rejects_non_finite_values() {
        assert!(quantize_signed(f64::NAN).is_err());
        assert!(quantize_signed(f64::INFINITY).is_err());
        assert!(quantize_signed(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn quantization_round_trips_signed_values() {
        let value = -12.345678;
        let quantized = quantize_signed(value).expect("quantize");
        let round_trip = dequantize_signed(quantized);
        assert!((round_trip - value).abs() <= 1.0 / AUDIT_NUMERIC_SCALE_F64);
    }

    #[test]
    fn compare_approximate_accepts_small_drift() {
        let config = ApproximateVerificationConfig {
            absolute_epsilon: 0.01,
            relative_epsilon: 0.001,
            max_deviating_fraction: 0.25,
        };

        let result = compare_approximate(&[1.0, 2.0], &[1.0005, 1.9995], &config);
        assert!(result.passed);
        assert_eq!(result.deviating_values, 0);
    }

    #[test]
    fn compare_approximate_rejects_when_too_many_values_deviate() {
        let config = ApproximateVerificationConfig {
            absolute_epsilon: 0.01,
            relative_epsilon: 0.001,
            max_deviating_fraction: 0.2,
        };

        let result = compare_approximate(
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.02, 0.02, 0.0, 0.0, 0.0],
            &config,
        );
        assert!(!result.passed);
        assert_eq!(result.deviating_values, 2);
    }
}
