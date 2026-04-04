use std::collections::{HashMap, HashSet};

use super::ast::{ArithmeticExpr, BooleanExpr};
use super::ir::{
    ArithmeticIr, AvailabilityIr, AvailabilitySource, BindingId, BooleanIr, CompilationPlan,
    FailureReason, QuantifiedCollection, ResolvedConstraint, ResolvedDefinition, ValueRefIr,
    ValueSource,
};
use super::parser::{parse_arithmetic_logic_expr, parse_logic_expr};
use super::schema::{CdmnTable, Collection, Constraint, Variable};
use crate::error::{Result, VmError};
use crate::state::Word;

#[derive(Debug, Clone)]
struct CollectionBinding {
    layout: CollectionLayout,
    field_bindings: HashMap<String, FieldBinding>,
}

#[derive(Debug, Clone)]
struct FieldBinding {
    value_offset: u8,
    presence_offset: Option<u8>,
    unknown_offset: Option<u8>,
}

pub(crate) fn build_compilation_plan(table: &CdmnTable) -> Result<CompilationPlan> {
    validate_table_names(table)?;
    let declared_max = max_declared_memory_cell(table)?;
    let mut resolver = Resolver::new(table.collections.clone(), &table.variables);
    let mut next_memory = next_output_register(declared_max)?;
    let mut definitions = Vec::with_capacity(table.definitions.len());

    for definition in &table.definitions {
        let memory = next_memory;
        next_memory = next_output_register(next_memory)?;
        let status_memory = next_memory;
        next_memory = next_output_register(next_memory)?;
        let parsed = parse_arithmetic_logic_expr(&definition.expression)?;
        let expression = resolver.resolve_arithmetic_expr(&parsed, &HashMap::new())?;
        definitions.push(ResolvedDefinition {
            name: definition.name.clone(),
            description: definition.description.clone(),
            expression,
            memory,
            status_memory,
        });
        resolver.insert_register(
            definition.name.clone(),
            ValueRefIr {
                source: ValueSource::Cell(memory),
                availability: AvailabilityIr::Status(AvailabilitySource::Cell(status_memory)),
            },
        )?;
    }

    let scratch_start = next_memory;
    let reason_plans = plan_constraint_reasons(&table.constraints);
    let constraints = table
        .constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| {
            let parsed = parse_logic_expr(&constraint.logic)?;
            let (reason, insufficient_data_reason) = reason_plans[index].clone();
            Ok(ResolvedConstraint {
                id: constraint.id.clone(),
                description: constraint.description.clone(),
                expression: resolver.resolve_boolean_expr(&parsed, &HashMap::new())?,
                fail_label: format!("constraint_fail_{index}"),
                reason,
                insufficient_data_label: format!("constraint_insufficient_{index}"),
                insufficient_data_reason,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CompilationPlan {
        scratch_start,
        definitions,
        constraints,
    })
}

pub(crate) fn next_output_register(register: u8) -> Result<u8> {
    register.checked_add(1).ok_or_else(|| {
        logic_error("not enough address space for cDMN output registers".to_string())
    })
}

fn plan_constraint_reasons(constraints: &[Constraint]) -> Vec<(FailureReason, FailureReason)> {
    let mut used_codes = HashSet::new();
    let mut primary_codes = Vec::with_capacity(constraints.len());

    for (index, constraint) in constraints.iter().enumerate() {
        let code = constraint.reason_code.unwrap_or(index as Word + 1);
        used_codes.insert(code);
        primary_codes.push(code);
    }

    let mut next_auto_code = used_codes
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| {
            let primary_code = primary_codes[index];
            let primary_label = constraint.reason_label.clone().unwrap_or_else(|| {
                derive_reason_label(if constraint.description.is_empty() {
                    &constraint.id
                } else {
                    &constraint.description
                })
            });
            let insufficient_code = constraint.insufficient_data_reason_code.unwrap_or_else(|| {
                while used_codes.contains(&next_auto_code) {
                    next_auto_code = next_auto_code.saturating_add(1);
                }
                let code = next_auto_code;
                used_codes.insert(code);
                next_auto_code = next_auto_code.saturating_add(1);
                code
            });
            let insufficient_label = constraint
                .insufficient_data_reason_label
                .clone()
                .unwrap_or_else(|| format!("{primary_label}_insufficient_data"));
            (
                FailureReason {
                    code: primary_code,
                    label: primary_label,
                    description: constraint.description.clone(),
                },
                FailureReason {
                    code: insufficient_code,
                    label: insufficient_label,
                    description: if constraint.description.is_empty() {
                        "Insufficient data".to_string()
                    } else {
                        format!("Insufficient data for {}", constraint.description)
                    },
                },
            )
        })
        .collect()
}

fn validate_table_names(table: &CdmnTable) -> Result<()> {
    let mut seen = HashSet::new();

    for variable in &table.variables {
        if !seen.insert(variable.name.clone()) {
            return Err(logic_error(format!(
                "duplicate cDMN variable name `{}`",
                variable.name
            )));
        }
    }

    for definition in &table.definitions {
        if !seen.insert(definition.name.clone()) {
            return Err(logic_error(format!(
                "duplicate cDMN definition name `{}`",
                definition.name
            )));
        }
    }

    for collection in &table.collections {
        if !seen.insert(collection.name.clone()) {
            return Err(logic_error(format!(
                "duplicate cDMN collection name `{}`",
                collection.name
            )));
        }
    }

    Ok(())
}

fn max_declared_memory_cell(table: &CdmnTable) -> Result<u8> {
    let mut max_cell = 0usize;

    for variable in &table.variables {
        max_cell = max_cell
            .max(usize::from(variable.memory))
            .max(variable.presence_memory.map(usize::from).unwrap_or(0))
            .max(variable.unknown_memory.map(usize::from).unwrap_or(0));
    }

    for collection in &table.collections {
        max_cell = max_cell.max(usize::from(collection.count_memory));
        if collection.max_items == 0 {
            continue;
        }
        let stride = usize::from(collection.stride);
        let max_items = usize::from(collection.max_items);
        let base = usize::from(collection.base_memory);
        let last_field_offset = collection
            .fields
            .iter()
            .flat_map(|field| {
                [
                    Some(usize::from(field.offset)),
                    field.presence_offset.map(usize::from),
                    field.unknown_offset.map(usize::from),
                ]
            })
            .flatten()
            .max()
            .unwrap_or(0);
        let last_cell = base
            .saturating_add(stride.saturating_mul(max_items.saturating_sub(1)))
            .saturating_add(last_field_offset);
        max_cell = max_cell.max(last_cell);
    }

    u8::try_from(max_cell).map_err(|_| {
        logic_error("declared cDMN memory layout exceeds the u8 address space".to_string())
    })
}

fn derive_reason_label(input: &str) -> String {
    let mut label = String::new();
    let mut last_was_underscore = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            label.push(ch.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !label.is_empty() && !last_was_underscore {
            label.push('_');
            last_was_underscore = true;
        }
    }
    while label.ends_with('_') {
        label.pop();
    }
    if label.is_empty() {
        "constraint_failure".to_string()
    } else {
        label
    }
}

#[derive(Debug, Clone)]
struct CollectionLayout {
    name: String,
    count_memory: u8,
    base_memory: u8,
    stride: u8,
    max_items: u8,
}

impl CollectionLayout {
    fn bind(&self, binding: BindingId) -> QuantifiedCollection {
        QuantifiedCollection {
            binding,
            name: self.name.clone(),
            count_memory: self.count_memory,
            base_memory: self.base_memory,
            stride: self.stride,
            max_items: self.max_items,
        }
    }
}

#[derive(Debug, Clone)]
struct QuantifierBinding {
    id: BindingId,
    field_bindings: HashMap<String, FieldBinding>,
}

struct Resolver {
    registers: HashMap<String, ValueRefIr>,
    collections: HashMap<String, CollectionBinding>,
    next_binding_id: usize,
}

impl Resolver {
    fn new(collections: Vec<Collection>, variables: &[Variable]) -> Self {
        let registers = variables
            .iter()
            .map(|variable| {
                (
                    variable.name.clone(),
                    ValueRefIr {
                        source: ValueSource::Cell(variable.memory),
                        availability: AvailabilityIr::Flags {
                            presence: variable.presence_memory.map(AvailabilitySource::Cell),
                            unknown: variable.unknown_memory.map(AvailabilitySource::Cell),
                        },
                    },
                )
            })
            .collect();
        let collections = collections
            .into_iter()
            .map(|collection| {
                let layout = CollectionLayout {
                    name: collection.name.clone(),
                    count_memory: collection.count_memory,
                    base_memory: collection.base_memory,
                    stride: collection.stride,
                    max_items: collection.max_items,
                };
                let field_bindings = collection
                    .fields
                    .into_iter()
                    .map(|field| {
                        (
                            field.name,
                            FieldBinding {
                                value_offset: field.offset,
                                presence_offset: field.presence_offset,
                                unknown_offset: field.unknown_offset,
                            },
                        )
                    })
                    .collect();
                (
                    collection.name,
                    CollectionBinding {
                        layout,
                        field_bindings,
                    },
                )
            })
            .collect();
        Self {
            registers,
            collections,
            next_binding_id: 0,
        }
    }

    fn insert_register(&mut self, name: String, value: ValueRefIr) -> Result<()> {
        if self.registers.insert(name.clone(), value).is_some() {
            return Err(logic_error(format!("duplicate cDMN binding name `{name}`")));
        }
        Ok(())
    }

    fn resolve_boolean_expr(
        &mut self,
        expr: &BooleanExpr,
        bindings: &HashMap<String, QuantifierBinding>,
    ) -> Result<BooleanIr> {
        match expr {
            BooleanExpr::Comparison { lhs, op, rhs } => Ok(BooleanIr::Comparison {
                lhs: self.resolve_arithmetic_expr(lhs, bindings)?,
                op: *op,
                rhs: self.resolve_arithmetic_expr(rhs, bindings)?,
            }),
            BooleanExpr::Availability { kind, value } => Ok(BooleanIr::Availability {
                kind: *kind,
                value: self.resolve_arithmetic_expr(value, bindings)?,
            }),
            BooleanExpr::Between {
                value,
                lower,
                upper,
            } => Ok(BooleanIr::Between {
                value: self.resolve_arithmetic_expr(value, bindings)?,
                lower: self.resolve_arithmetic_expr(lower, bindings)?,
                upper: self.resolve_arithmetic_expr(upper, bindings)?,
            }),
            BooleanExpr::And(left, right) => Ok(BooleanIr::And(
                Box::new(self.resolve_boolean_expr(left, bindings)?),
                Box::new(self.resolve_boolean_expr(right, bindings)?),
            )),
            BooleanExpr::Or(left, right) => Ok(BooleanIr::Or(
                Box::new(self.resolve_boolean_expr(left, bindings)?),
                Box::new(self.resolve_boolean_expr(right, bindings)?),
            )),
            BooleanExpr::Not(inner) => Ok(BooleanIr::Not(Box::new(
                self.resolve_boolean_expr(inner, bindings)?,
            ))),
            BooleanExpr::Quantified {
                kind,
                binding,
                collection,
                predicate,
            } => {
                let collection_binding = self
                    .collections
                    .get(collection)
                    .cloned()
                    .ok_or_else(|| logic_error(format!("unknown collection `{collection}`")))?;
                let binding_id = BindingId(self.next_binding_id);
                self.next_binding_id += 1;
                let mut nested_bindings = bindings.clone();
                nested_bindings.insert(
                    binding.clone(),
                    QuantifierBinding {
                        id: binding_id,
                        field_bindings: collection_binding.field_bindings.clone(),
                    },
                );
                Ok(BooleanIr::Quantified {
                    kind: *kind,
                    collection: collection_binding.layout.bind(binding_id),
                    predicate: Box::new(self.resolve_boolean_expr(predicate, &nested_bindings)?),
                })
            }
        }
    }

    fn resolve_arithmetic_expr(
        &self,
        expr: &ArithmeticExpr,
        bindings: &HashMap<String, QuantifierBinding>,
    ) -> Result<ArithmeticIr> {
        match expr {
            ArithmeticExpr::Number(value) => Ok(ArithmeticIr::Number(*value)),
            ArithmeticExpr::Variable(name) => self.resolve_name(name, bindings),
            ArithmeticExpr::Binary { op, left, right } => Ok(ArithmeticIr::Binary {
                op: *op,
                left: Box::new(self.resolve_arithmetic_expr(left, bindings)?),
                right: Box::new(self.resolve_arithmetic_expr(right, bindings)?),
            }),
        }
    }

    fn resolve_name(
        &self,
        name: &str,
        bindings: &HashMap<String, QuantifierBinding>,
    ) -> Result<ArithmeticIr> {
        if let Some((binding_name, field_name)) = name.split_once('.') {
            if let Some(binding) = bindings.get(binding_name) {
                if let Some(field) = binding.field_bindings.get(field_name) {
                    return Ok(ArithmeticIr::Value(ValueRefIr {
                        source: ValueSource::BoundField {
                            binding: binding.id,
                            offset: field.value_offset,
                        },
                        availability: AvailabilityIr::Flags {
                            presence: field.presence_offset.map(|offset| {
                                AvailabilitySource::BoundField {
                                    binding: binding.id,
                                    offset,
                                }
                            }),
                            unknown: field.unknown_offset.map(|offset| {
                                AvailabilitySource::BoundField {
                                    binding: binding.id,
                                    offset,
                                }
                            }),
                        },
                    }));
                }
            }
        }

        self.registers
            .get(name)
            .cloned()
            .map(ArithmeticIr::Value)
            .ok_or_else(|| logic_error(format!("unknown variable `{name}`")))
    }
}

fn logic_error(message: String) -> VmError {
    VmError::InvalidConfig(message)
}
