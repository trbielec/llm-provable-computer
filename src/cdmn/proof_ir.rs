use std::collections::{BTreeSet, HashMap};

use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};

use super::ast::{ArithmeticOp, AvailabilityPredicateKind, ComparisonOp, QuantifierKind};
use super::ir::{
    ArithmeticIr, AvailabilityIr, AvailabilitySource, BindingId, BooleanIr, CompilationPlan,
    FailureReason, ValueRefIr, ValueSource,
};
use super::schema::ReasonMetadata;
use crate::error::{Result, VmError};
use crate::state::Word;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionValueSource {
    InputCell(u8),
    DefinitionValue(usize),
    BoundField { binding: usize, offset: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionAvailabilitySource {
    InputCell(u8),
    DefinitionStatus(usize),
    BoundField { binding: usize, offset: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionAvailability {
    Always,
    Flags {
        presence: Option<DecisionAvailabilitySource>,
        unknown: Option<DecisionAvailabilitySource>,
    },
    Status(DecisionAvailabilitySource),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DecisionValueRef {
    pub source: DecisionValueSource,
    pub availability: DecisionAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionArithmeticExpr {
    Number(Word),
    Value(DecisionValueRef),
    Binary {
        op: ArithmeticOp,
        left: Box<DecisionArithmeticExpr>,
        right: Box<DecisionArithmeticExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionBooleanExpr {
    Constant(bool),
    CountAtLeast {
        count_memory: u8,
        threshold: u8,
        max_items: u8,
    },
    Comparison {
        lhs: DecisionArithmeticExpr,
        op: ComparisonOp,
        rhs: DecisionArithmeticExpr,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: DecisionArithmeticExpr,
    },
    Between {
        value: DecisionArithmeticExpr,
        lower: DecisionArithmeticExpr,
        upper: DecisionArithmeticExpr,
    },
    And(Box<DecisionBooleanExpr>, Box<DecisionBooleanExpr>),
    Or(Box<DecisionBooleanExpr>, Box<DecisionBooleanExpr>),
    Not(Box<DecisionBooleanExpr>),
    Quantified {
        kind: QuantifierKind,
        collection: DecisionQuantifiedCollection,
        predicate: Box<DecisionBooleanExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DecisionQuantifiedCollection {
    pub binding: usize,
    pub name: String,
    pub count_memory: u8,
    pub base_memory: u8,
    pub stride: u8,
    pub max_items: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionDefinition {
    pub name: String,
    pub description: String,
    pub expression: DecisionArithmeticExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionFailureReason {
    pub code: Word,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionCountBoundCheck {
    pub count_memory: u8,
    pub max_items: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionConstraint {
    pub id: String,
    pub description: String,
    pub bound_checks: Vec<DecisionCountBoundCheck>,
    pub expression: DecisionBooleanExpr,
    pub reason: DecisionFailureReason,
    pub insufficient_data_reason: DecisionFailureReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCdmnDecisionProgram {
    pub name: String,
    pub definitions: Vec<DecisionDefinition>,
    pub constraints: Vec<DecisionConstraint>,
    pub input_cells: Vec<u8>,
    pub reasons: Vec<ReasonMetadata>,
}

impl CompiledCdmnDecisionProgram {
    pub fn reason_for_code(&self, code: Word) -> Option<&ReasonMetadata> {
        self.reasons.iter().find(|reason| reason.code == code)
    }

    pub fn policy_commitment(&self) -> Result<Vec<u8>> {
        hash_json(self)
    }
}

pub(crate) fn lower_to_decision_program(
    name: &str,
    plan: &CompilationPlan,
) -> Result<CompiledCdmnDecisionProgram> {
    let mut lowerer = Lowerer::new(plan);
    let definitions = plan
        .definitions
        .iter()
        .map(|definition| {
            Ok(DecisionDefinition {
                name: definition.name.clone(),
                description: definition.description.clone(),
                expression: lowerer.lower_arithmetic(&definition.expression, &HashMap::new())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let constraints = plan
        .constraints
        .iter()
        .map(|constraint| {
            let mut bound_checks = BTreeSet::new();
            let expression = lowerer.lower_boolean(
                &constraint.expression,
                &HashMap::new(),
                &mut bound_checks,
            )?;
            Ok(DecisionConstraint {
                id: constraint.id.clone(),
                description: constraint.description.clone(),
                bound_checks: bound_checks
                    .into_iter()
                    .map(|(count_memory, max_items)| DecisionCountBoundCheck {
                        count_memory,
                        max_items,
                    })
                    .collect(),
                expression,
                reason: lower_reason(&constraint.reason),
                insufficient_data_reason: lower_reason(&constraint.insufficient_data_reason),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let reasons = constraints
        .iter()
        .flat_map(|constraint| {
            [
                ReasonMetadata {
                    code: constraint.reason.code,
                    constraint_id: constraint.id.clone(),
                    label: constraint.reason.label.clone(),
                    description: constraint.reason.description.clone(),
                },
                ReasonMetadata {
                    code: constraint.insufficient_data_reason.code,
                    constraint_id: constraint.id.clone(),
                    label: constraint.insufficient_data_reason.label.clone(),
                    description: constraint.insufficient_data_reason.description.clone(),
                },
            ]
        })
        .collect();

    Ok(CompiledCdmnDecisionProgram {
        name: name.to_string(),
        definitions,
        constraints,
        input_cells: lowerer.input_cells.into_iter().collect(),
        reasons,
    })
}

pub(crate) fn hash_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| VmError::Serialization(format!("json serialization error: {error}")))?;
    let mut hasher = Blake2b512::new();
    hasher.update(encoded);
    Ok(hasher.finalize().to_vec())
}

fn lower_reason(reason: &FailureReason) -> DecisionFailureReason {
    DecisionFailureReason {
        code: reason.code,
        label: reason.label.clone(),
        description: reason.description.clone(),
    }
}

struct Lowerer<'a> {
    definition_values: HashMap<u8, usize>,
    definition_statuses: HashMap<u8, usize>,
    input_cells: BTreeSet<u8>,
    _plan: &'a CompilationPlan,
}

impl<'a> Lowerer<'a> {
    fn new(plan: &'a CompilationPlan) -> Self {
        let definition_values = plan
            .definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| (definition.memory, index))
            .collect();
        let definition_statuses = plan
            .definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| (definition.status_memory, index))
            .collect();
        Self {
            definition_values,
            definition_statuses,
            input_cells: BTreeSet::new(),
            _plan: plan,
        }
    }

    fn lower_arithmetic(
        &mut self,
        expr: &ArithmeticIr,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionArithmeticExpr> {
        match expr {
            ArithmeticIr::Number(value) => Ok(DecisionArithmeticExpr::Number(*value)),
            ArithmeticIr::Value(value_ref) => Ok(DecisionArithmeticExpr::Value(
                self.lower_value_ref(value_ref, bindings)?,
            )),
            ArithmeticIr::Binary { op, left, right } => Ok(DecisionArithmeticExpr::Binary {
                op: *op,
                left: Box::new(self.lower_arithmetic(left, bindings)?),
                right: Box::new(self.lower_arithmetic(right, bindings)?),
            }),
        }
    }

    fn lower_boolean(
        &mut self,
        expr: &BooleanIr,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
        bound_checks: &mut BTreeSet<(u8, u8)>,
    ) -> Result<DecisionBooleanExpr> {
        match expr {
            BooleanIr::Comparison { lhs, op, rhs } => Ok(DecisionBooleanExpr::Comparison {
                lhs: self.lower_arithmetic(lhs, bindings)?,
                op: *op,
                rhs: self.lower_arithmetic(rhs, bindings)?,
            }),
            BooleanIr::Availability { kind, value } => Ok(DecisionBooleanExpr::Availability {
                kind: *kind,
                value: self.lower_arithmetic(value, bindings)?,
            }),
            BooleanIr::Between {
                value,
                lower,
                upper,
            } => Ok(DecisionBooleanExpr::Between {
                value: self.lower_arithmetic(value, bindings)?,
                lower: self.lower_arithmetic(lower, bindings)?,
                upper: self.lower_arithmetic(upper, bindings)?,
            }),
            BooleanIr::And(left, right) => Ok(DecisionBooleanExpr::And(
                Box::new(self.lower_boolean(left, bindings, bound_checks)?),
                Box::new(self.lower_boolean(right, bindings, bound_checks)?),
            )),
            BooleanIr::Or(left, right) => Ok(DecisionBooleanExpr::Or(
                Box::new(self.lower_boolean(left, bindings, bound_checks)?),
                Box::new(self.lower_boolean(right, bindings, bound_checks)?),
            )),
            BooleanIr::Not(inner) => Ok(DecisionBooleanExpr::Not(Box::new(self.lower_boolean(
                inner,
                bindings,
                bound_checks,
            )?))),
            BooleanIr::Quantified {
                kind,
                collection,
                predicate,
            } => {
                bound_checks.insert((collection.count_memory, collection.max_items));
                self.input_cells.insert(collection.count_memory);
                let mut nested = bindings.clone();
                nested.insert(
                    collection.binding,
                    DecisionQuantifiedCollection {
                        binding: collection.binding.0,
                        name: collection.name.clone(),
                        count_memory: collection.count_memory,
                        base_memory: collection.base_memory,
                        stride: collection.stride,
                        max_items: collection.max_items,
                    },
                );
                self.register_collection_inputs(&nested[&collection.binding]);
                Ok(DecisionBooleanExpr::Quantified {
                    kind: *kind,
                    collection: nested[&collection.binding].clone(),
                    predicate: Box::new(self.lower_boolean(predicate, &nested, bound_checks)?),
                })
            }
        }
    }
    fn lower_value_ref(
        &mut self,
        value_ref: &ValueRefIr,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionValueRef> {
        Ok(DecisionValueRef {
            source: self.lower_value_source(value_ref.source, bindings)?,
            availability: self.lower_availability(&value_ref.availability, bindings)?,
        })
    }

    fn lower_value_source(
        &mut self,
        source: ValueSource,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionValueSource> {
        match source {
            ValueSource::Cell(address) => {
                if let Some(&definition_index) = self.definition_values.get(&address) {
                    Ok(DecisionValueSource::DefinitionValue(definition_index))
                } else {
                    self.input_cells.insert(address);
                    Ok(DecisionValueSource::InputCell(address))
                }
            }
            ValueSource::BoundField { binding, offset } => {
                let collection = bindings.get(&binding).ok_or_else(|| {
                    VmError::InvalidConfig("unbound quantified collection slot".to_string())
                })?;
                self.register_collection_field_inputs(collection, offset);
                Ok(DecisionValueSource::BoundField {
                    binding: collection.binding,
                    offset,
                })
            }
        }
    }

    fn lower_availability(
        &mut self,
        availability: &AvailabilityIr,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionAvailability> {
        match availability {
            AvailabilityIr::Flags { presence, unknown } => {
                let presence = presence
                    .map(|source| self.resolve_input_availability_source(source, bindings))
                    .transpose()?;
                let unknown = unknown
                    .map(|source| self.resolve_input_availability_source(source, bindings))
                    .transpose()?;
                if presence.is_none() && unknown.is_none() {
                    Ok(DecisionAvailability::Always)
                } else {
                    Ok(DecisionAvailability::Flags { presence, unknown })
                }
            }
            AvailabilityIr::Status(source) => Ok(DecisionAvailability::Status(
                self.lower_status_source(*source, bindings)?,
            )),
        }
    }

    fn resolve_input_availability_source(
        &mut self,
        source: AvailabilitySource,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionAvailabilitySource> {
        match source {
            AvailabilitySource::Cell(address) => {
                if self.definition_statuses.contains_key(&address) {
                    return Err(VmError::InvalidConfig(
                        "definition status cannot be used as a presence/unknown flag".to_string(),
                    ));
                }
                self.input_cells.insert(address);
                Ok(DecisionAvailabilitySource::InputCell(address))
            }
            AvailabilitySource::BoundField { binding, offset } => {
                let collection = bindings.get(&binding).ok_or_else(|| {
                    VmError::InvalidConfig("unbound quantified collection slot".to_string())
                })?;
                self.register_collection_field_inputs(collection, offset);
                Ok(DecisionAvailabilitySource::BoundField {
                    binding: collection.binding,
                    offset,
                })
            }
        }
    }

    fn lower_status_source(
        &mut self,
        source: AvailabilitySource,
        bindings: &HashMap<BindingId, DecisionQuantifiedCollection>,
    ) -> Result<DecisionAvailabilitySource> {
        match source {
            AvailabilitySource::Cell(address) => {
                if let Some(&definition_index) = self.definition_statuses.get(&address) {
                    Ok(DecisionAvailabilitySource::DefinitionStatus(
                        definition_index,
                    ))
                } else {
                    self.input_cells.insert(address);
                    Ok(DecisionAvailabilitySource::InputCell(address))
                }
            }
            AvailabilitySource::BoundField { binding, offset } => {
                let collection = bindings.get(&binding).ok_or_else(|| {
                    VmError::InvalidConfig("unbound quantified collection slot".to_string())
                })?;
                self.register_collection_field_inputs(collection, offset);
                Ok(DecisionAvailabilitySource::BoundField {
                    binding: collection.binding,
                    offset,
                })
            }
        }
    }

    fn register_collection_inputs(&mut self, collection: &DecisionQuantifiedCollection) {
        self.input_cells.insert(collection.count_memory);
        for item_index in 0..collection.max_items {
            let slot_base = usize::from(collection.base_memory)
                + usize::from(item_index) * usize::from(collection.stride);
            for offset in 0..collection.stride {
                if let Ok(address) = u8::try_from(slot_base + usize::from(offset)) {
                    self.input_cells.insert(address);
                }
            }
        }
    }

    fn register_collection_field_inputs(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        offset: u8,
    ) {
        self.input_cells.insert(collection.count_memory);
        for item_index in 0..collection.max_items {
            let slot_base = usize::from(collection.base_memory)
                + usize::from(item_index) * usize::from(collection.stride);
            if let Ok(address) = u8::try_from(slot_base + usize::from(offset)) {
                self.input_cells.insert(address);
            }
        }
    }
}
