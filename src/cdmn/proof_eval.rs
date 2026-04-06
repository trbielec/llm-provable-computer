use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::proof_ir::{
    hash_json, CompiledCdmnDecisionProgram, DecisionArithmeticExpr, DecisionAvailability,
    DecisionAvailabilitySource, DecisionBooleanExpr, DecisionValueSource,
};
use crate::error::{Result, VmError};
use crate::state::Word;

const STATUS_AVAILABLE: Word = 0;
const STATUS_MISSING: Word = 1;
const STATUS_UNKNOWN: Word = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvaluatedValue {
    value: Word,
    status: Word,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvaluatedBoolean {
    value: bool,
    status: Word,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundItemContext {
    base_memory: u8,
    stride: u8,
    item_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnDecisionEvaluation {
    pub status: Word,
    pub reason_code: Word,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
}

pub fn evaluate_decision_program(
    program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<CdmnDecisionEvaluation> {
    validate_inputs(program, initial_memory)?;

    let mut definition_values = Vec::with_capacity(program.definitions.len());
    let mut definition_statuses = Vec::with_capacity(program.definitions.len());

    for definition in &program.definitions {
        let evaluated = eval_arithmetic(
            &definition.expression,
            initial_memory,
            &definition_values,
            &definition_statuses,
            &HashMap::new(),
        )?;
        definition_values.push(evaluated.value);
        definition_statuses.push(evaluated.status);
    }

    let mut final_status = 1;
    let mut final_reason_code = 0;

    for constraint in &program.constraints {
        for bound_check in &constraint.bound_checks {
            let count = read_memory(initial_memory, bound_check.count_memory)?;
            if count > i64::from(bound_check.max_items) {
                final_status = 0;
                final_reason_code = constraint.insufficient_data_reason.code;
                return Ok(CdmnDecisionEvaluation {
                    status: final_status,
                    reason_code: final_reason_code,
                    policy_commitment: program.policy_commitment()?,
                    input_commitment: input_commitment(program, initial_memory)?,
                });
            }
        }

        let evaluated = eval_boolean(
            &constraint.expression,
            initial_memory,
            &definition_values,
            &definition_statuses,
            &HashMap::new(),
        )?;
        if evaluated.status != STATUS_AVAILABLE {
            final_status = 0;
            final_reason_code = constraint.insufficient_data_reason.code;
            return Ok(CdmnDecisionEvaluation {
                status: final_status,
                reason_code: final_reason_code,
                policy_commitment: program.policy_commitment()?,
                input_commitment: input_commitment(program, initial_memory)?,
            });
        }
        if !evaluated.value {
            final_status = 0;
            final_reason_code = constraint.reason.code;
            return Ok(CdmnDecisionEvaluation {
                status: final_status,
                reason_code: final_reason_code,
                policy_commitment: program.policy_commitment()?,
                input_commitment: input_commitment(program, initial_memory)?,
            });
        }
    }

    Ok(CdmnDecisionEvaluation {
        status: final_status,
        reason_code: final_reason_code,
        policy_commitment: program.policy_commitment()?,
        input_commitment: input_commitment(program, initial_memory)?,
    })
}

pub fn input_commitment(
    program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<Vec<u8>> {
    let projection = program
        .input_cells
        .iter()
        .map(|&address| Ok((address, read_memory(initial_memory, address)?)))
        .collect::<Result<Vec<_>>>()?;
    hash_json(&projection)
}

pub(crate) fn evaluate_boolean_expression(
    expr: &DecisionBooleanExpr,
    initial_memory: &[Word],
) -> Result<(bool, Word)> {
    let evaluated = eval_boolean(expr, initial_memory, &[], &[], &HashMap::new())?;
    Ok((evaluated.value, evaluated.status))
}

fn validate_inputs(program: &CompiledCdmnDecisionProgram, initial_memory: &[Word]) -> Result<()> {
    if let Some(&highest_address) = program.input_cells.iter().max() {
        if usize::from(highest_address) >= initial_memory.len() {
            return Err(VmError::MemoryOutOfBounds {
                addr: usize::from(highest_address),
                size: initial_memory.len(),
            });
        }
    }
    Ok(())
}

fn eval_boolean(
    expr: &DecisionBooleanExpr,
    initial_memory: &[Word],
    definition_values: &[Word],
    definition_statuses: &[Word],
    bindings: &HashMap<usize, BoundItemContext>,
) -> Result<EvaluatedBoolean> {
    match expr {
        DecisionBooleanExpr::Constant(value) => Ok(EvaluatedBoolean {
            value: *value,
            status: STATUS_AVAILABLE,
        }),
        DecisionBooleanExpr::CountAtLeast {
            count_memory,
            threshold,
            max_items,
        } => {
            let count = read_memory(initial_memory, *count_memory)?;
            if !(0..=i64::from(*max_items)).contains(&count) {
                return Ok(EvaluatedBoolean {
                    value: false,
                    status: STATUS_AVAILABLE,
                });
            }
            Ok(EvaluatedBoolean {
                value: count >= i64::from(*threshold),
                status: STATUS_AVAILABLE,
            })
        }
        DecisionBooleanExpr::Comparison { lhs, op, rhs } => {
            let lhs = eval_arithmetic(
                lhs,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let rhs = eval_arithmetic(
                rhs,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let status = combine_status(lhs.status, rhs.status);
            if status != STATUS_AVAILABLE {
                return Ok(EvaluatedBoolean {
                    value: false,
                    status,
                });
            }
            Ok(EvaluatedBoolean {
                value: match op {
                    crate::cdmn::ComparisonOp::Eq => lhs.value == rhs.value,
                    crate::cdmn::ComparisonOp::Ne => lhs.value != rhs.value,
                    crate::cdmn::ComparisonOp::Lt => lhs.value < rhs.value,
                    crate::cdmn::ComparisonOp::Le => lhs.value <= rhs.value,
                    crate::cdmn::ComparisonOp::Gt => lhs.value > rhs.value,
                    crate::cdmn::ComparisonOp::Ge => lhs.value >= rhs.value,
                },
                status,
            })
        }
        DecisionBooleanExpr::Availability { kind, value } => {
            let value = eval_arithmetic(
                value,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            Ok(EvaluatedBoolean {
                value: match kind {
                    crate::cdmn::AvailabilityPredicateKind::Present => {
                        value.status == STATUS_AVAILABLE
                    }
                    crate::cdmn::AvailabilityPredicateKind::Missing => {
                        value.status == STATUS_MISSING
                    }
                    crate::cdmn::AvailabilityPredicateKind::Unknown => {
                        value.status == STATUS_UNKNOWN
                    }
                },
                status: STATUS_AVAILABLE,
            })
        }
        DecisionBooleanExpr::Between {
            value,
            lower,
            upper,
        } => {
            let value = eval_arithmetic(
                value,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let lower = eval_arithmetic(
                lower,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let upper = eval_arithmetic(
                upper,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let status = combine_status(value.status, combine_status(lower.status, upper.status));
            if status != STATUS_AVAILABLE {
                return Ok(EvaluatedBoolean {
                    value: false,
                    status,
                });
            }
            Ok(EvaluatedBoolean {
                value: value.value >= lower.value && value.value <= upper.value,
                status,
            })
        }
        DecisionBooleanExpr::And(left, right) => {
            let left = eval_boolean(
                left,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            if left.status != STATUS_AVAILABLE || !left.value {
                return Ok(left);
            }
            eval_boolean(
                right,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )
        }
        DecisionBooleanExpr::Or(left, right) => {
            let left = eval_boolean(
                left,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            if left.status != STATUS_AVAILABLE {
                return Ok(left);
            }
            if left.value {
                return Ok(left);
            }
            eval_boolean(
                right,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )
        }
        DecisionBooleanExpr::Not(inner) => {
            let inner = eval_boolean(
                inner,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            if inner.status != STATUS_AVAILABLE {
                return Ok(inner);
            }
            Ok(EvaluatedBoolean {
                value: !inner.value,
                status: STATUS_AVAILABLE,
            })
        }
        DecisionBooleanExpr::Quantified {
            kind,
            collection,
            predicate,
        } => {
            let count = read_memory(initial_memory, collection.count_memory)?;
            let in_domain = (0..=i64::from(collection.max_items)).contains(&count);
            let mut acc = match kind {
                crate::cdmn::QuantifierKind::ForAll => EvaluatedBoolean {
                    value: true,
                    status: STATUS_AVAILABLE,
                },
                crate::cdmn::QuantifierKind::Exists => EvaluatedBoolean {
                    value: false,
                    status: STATUS_AVAILABLE,
                },
            };
            for item_index in 0..usize::from(collection.max_items) {
                let active = in_domain && count >= i64::try_from(item_index + 1).unwrap();
                if !active {
                    continue;
                }
                let mut nested = bindings.clone();
                nested.insert(
                    collection.binding,
                    BoundItemContext {
                        base_memory: collection.base_memory,
                        stride: collection.stride,
                        item_index,
                    },
                );
                let item = eval_boolean(
                    predicate,
                    initial_memory,
                    definition_values,
                    definition_statuses,
                    &nested,
                )?;
                acc = match kind {
                    crate::cdmn::QuantifierKind::ForAll => {
                        if acc.status != STATUS_AVAILABLE || !acc.value {
                            acc
                        } else {
                            item
                        }
                    }
                    crate::cdmn::QuantifierKind::Exists => {
                        if acc.status != STATUS_AVAILABLE || acc.value {
                            acc
                        } else {
                            item
                        }
                    }
                };
            }
            Ok(acc)
        }
    }
}

fn eval_arithmetic(
    expr: &DecisionArithmeticExpr,
    initial_memory: &[Word],
    definition_values: &[Word],
    definition_statuses: &[Word],
    bindings: &HashMap<usize, BoundItemContext>,
) -> Result<EvaluatedValue> {
    match expr {
        DecisionArithmeticExpr::Number(value) => Ok(EvaluatedValue {
            value: *value,
            status: STATUS_AVAILABLE,
        }),
        DecisionArithmeticExpr::Value(value_ref) => {
            let value = match &value_ref.source {
                DecisionValueSource::InputCell(address) => read_memory(initial_memory, *address)?,
                DecisionValueSource::DefinitionValue(index) => {
                    *definition_values.get(*index).ok_or_else(|| {
                        VmError::InvalidConfig(format!("missing definition value at index {index}"))
                    })?
                }
                DecisionValueSource::BoundField { binding, offset } => {
                    let context = bindings.get(binding).ok_or_else(|| {
                        VmError::InvalidConfig(format!(
                            "missing quantified binding context for binding {binding}"
                        ))
                    })?;
                    let address = collection_address(*offset, *context)?;
                    read_memory(initial_memory, address)?
                }
            };
            Ok(EvaluatedValue {
                value,
                status: eval_availability(
                    &value_ref.availability,
                    initial_memory,
                    definition_statuses,
                    bindings,
                )?,
            })
        }
        DecisionArithmeticExpr::Binary { op, left, right } => {
            let left = eval_arithmetic(
                left,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let right = eval_arithmetic(
                right,
                initial_memory,
                definition_values,
                definition_statuses,
                bindings,
            )?;
            let status = combine_status(left.status, right.status);
            if status != STATUS_AVAILABLE {
                return Ok(EvaluatedValue { value: 0, status });
            }
            Ok(EvaluatedValue {
                value: match op {
                    crate::cdmn::ArithmeticOp::Add => {
                        checked_binary_op("addition", left.value, right.value, |a, b| {
                            a.checked_add(b)
                        })?
                    }
                    crate::cdmn::ArithmeticOp::Sub => {
                        checked_binary_op("subtraction", left.value, right.value, |a, b| {
                            a.checked_sub(b)
                        })?
                    }
                    crate::cdmn::ArithmeticOp::Mul => {
                        checked_binary_op("multiplication", left.value, right.value, |a, b| {
                            a.checked_mul(b)
                        })?
                    }
                },
                status,
            })
        }
    }
}

fn eval_availability(
    availability: &DecisionAvailability,
    initial_memory: &[Word],
    definition_statuses: &[Word],
    bindings: &HashMap<usize, BoundItemContext>,
) -> Result<Word> {
    match availability {
        DecisionAvailability::Always => Ok(STATUS_AVAILABLE),
        DecisionAvailability::Flags { presence, unknown } => {
            if let Some(unknown) = unknown {
                if resolve_availability_source(
                    unknown,
                    initial_memory,
                    definition_statuses,
                    bindings,
                )? != 0
                {
                    return Ok(STATUS_UNKNOWN);
                }
            }
            if let Some(presence) = presence {
                if resolve_availability_source(
                    presence,
                    initial_memory,
                    definition_statuses,
                    bindings,
                )? == 0
                {
                    return Ok(STATUS_MISSING);
                }
            }
            Ok(STATUS_AVAILABLE)
        }
        DecisionAvailability::Status(source) => {
            resolve_availability_source(source, initial_memory, definition_statuses, bindings)
        }
    }
}

fn resolve_availability_source(
    source: &DecisionAvailabilitySource,
    initial_memory: &[Word],
    definition_statuses: &[Word],
    bindings: &HashMap<usize, BoundItemContext>,
) -> Result<Word> {
    match source {
        DecisionAvailabilitySource::InputCell(address) => read_memory(initial_memory, *address),
        DecisionAvailabilitySource::DefinitionStatus(index) => {
            definition_statuses.get(*index).copied().ok_or_else(|| {
                VmError::InvalidConfig(format!("missing definition status at index {index}"))
            })
        }
        DecisionAvailabilitySource::BoundField { binding, offset } => {
            let context = bindings.get(binding).ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "missing quantified availability binding context for binding {binding}"
                ))
            })?;
            let address = collection_address(*offset, *context)?;
            read_memory(initial_memory, address)
        }
    }
}

fn collection_address(offset: u8, context: BoundItemContext) -> Result<u8> {
    let address = usize::from(context.base_memory)
        + context.item_index * usize::from(context.stride)
        + usize::from(offset);
    u8::try_from(address).map_err(|_| {
        VmError::InvalidConfig(
            "quantified collection address exceeds the u8 address space".to_string(),
        )
    })
}

fn combine_status(left: Word, right: Word) -> Word {
    left.max(right)
}

fn read_memory(memory: &[Word], address: u8) -> Result<Word> {
    memory
        .get(usize::from(address))
        .copied()
        .ok_or(VmError::MemoryOutOfBounds {
            addr: usize::from(address),
            size: memory.len(),
        })
}

fn checked_binary_op<F>(name: &str, lhs: Word, rhs: Word, op: F) -> Result<Word>
where
    F: FnOnce(Word, Word) -> Option<Word>,
{
    op(lhs, rhs)
        .ok_or_else(|| VmError::InvalidConfig(format!("decision {name} overflowed i64 semantics")))
}
