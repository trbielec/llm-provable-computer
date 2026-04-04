use std::collections::HashMap;

use super::ast::{ArithmeticOp, AvailabilityPredicateKind, ComparisonOp, QuantifierKind};
use super::ir::{
    ArithmeticIr, AvailabilityIr, AvailabilitySource, BindingId, BooleanIr, CompilationPlan,
    QuantifiedCollection, ValueRefIr, ValueSource,
};
use super::schema::{CdmnOutputLayout, CompiledCdmnProgram, ReasonMetadata};
use super::semantics::next_output_register;
use crate::error::{Result, VmError};

pub(crate) fn compile_tvm_program(plan: &CompilationPlan) -> Result<CompiledCdmnProgram> {
    let mut compiler = LogicCompiler::new(plan.scratch_start);
    let mut definition_blocks = Vec::new();
    let mut constraint_blocks = Vec::new();
    let mut fail_blocks = Vec::new();
    let mut reasons = Vec::new();

    for definition in &plan.definitions {
        definition_blocks.push(format!(
            "  # Definition {}{}",
            definition.name,
            if definition.description.is_empty() {
                String::new()
            } else {
                format!(": {}", definition.description)
            }
        ));
        compiler.compile_definition(
            &definition.expression,
            definition.memory,
            definition.status_memory,
            &mut definition_blocks,
        )?;
    }

    for constraint in &plan.constraints {
        constraint_blocks.push(format!(
            "  # Constraint {}: {}",
            constraint.id, constraint.description
        ));
        compiler.compile_boolean_expr(
            &constraint.expression,
            &mut constraint_blocks,
            &constraint.fail_label,
            Some(&constraint.insufficient_data_label),
        )?;
        fail_blocks.push((constraint.fail_label.clone(), constraint.reason.code));
        fail_blocks.push((
            constraint.insufficient_data_label.clone(),
            constraint.insufficient_data_reason.code,
        ));
        reasons.push(ReasonMetadata {
            code: constraint.reason.code,
            constraint_id: constraint.id.clone(),
            label: constraint.reason.label.clone(),
            description: constraint.reason.description.clone(),
        });
        reasons.push(ReasonMetadata {
            code: constraint.insufficient_data_reason.code,
            constraint_id: constraint.id.clone(),
            label: constraint.insufficient_data_reason.label.clone(),
            description: constraint.insufficient_data_reason.description.clone(),
        });
    }

    let status_register = compiler.status_register();
    let reason_register = next_output_register(status_register)?;
    let memory_size = usize::from(reason_register) + 1;
    let mut assembly = String::new();
    assembly.push_str("# Auto-compiled TVM Verifier from cDMN Logic\n");
    assembly.push_str(&format!(".memory {memory_size}\n\nverify_start:\n"));
    for line in definition_blocks {
        assembly.push_str(&line);
        assembly.push('\n');
    }
    for line in constraint_blocks {
        assembly.push_str(&line);
        assembly.push('\n');
    }
    assembly.push_str(&format!(
        "\n  LOADI 1\n  STORE {status_register}\n  LOADI 0\n  STORE {reason_register}\n  HALT\n"
    ));
    for (fail_label, reason_code) in fail_blocks {
        assembly.push_str(&format!(
            "\n{fail_label}:\n  LOADI 0\n  STORE {status_register}\n  LOADI {reason_code}\n  STORE {reason_register}\n  HALT\n"
        ));
    }

    Ok(CompiledCdmnProgram {
        assembly,
        output: CdmnOutputLayout {
            status_register,
            reason_register,
        },
        reasons,
    })
}

pub(crate) struct LogicCompiler {
    scratch_start: u8,
    scratch_high_watermark: u8,
    label_counter: usize,
}

impl LogicCompiler {
    pub(crate) fn new(scratch_start: u8) -> Self {
        Self {
            scratch_start,
            scratch_high_watermark: scratch_start,
            label_counter: 0,
        }
    }

    fn status_register(&self) -> u8 {
        self.scratch_high_watermark
    }

    fn compile_definition(
        &mut self,
        expr: &ArithmeticIr,
        target: u8,
        status_target: u8,
        asm: &mut Vec<String>,
    ) -> Result<()> {
        let mut allocator = TempAllocator::new(self.scratch_start);
        let result = self.compile_arithmetic_expr(expr, asm, &mut allocator, &HashMap::new())?;
        emit_load(asm, result.value);
        emit_store(asm, target);
        if let Some(status) = result.status {
            emit_load(asm, status);
        } else {
            asm.push("  LOADI 0".to_string());
        }
        emit_store(asm, status_target);
        self.scratch_high_watermark = self.scratch_high_watermark.max(allocator.high_watermark());
        Ok(())
    }

    pub(crate) fn compile_boolean_expr(
        &mut self,
        expr: &BooleanIr,
        asm: &mut Vec<String>,
        fail_label: &str,
        insufficient_data_label: Option<&str>,
    ) -> Result<()> {
        let mut allocator = TempAllocator::new(self.scratch_start);
        self.compile_boolean_expr_with_allocator(
            expr,
            asm,
            &mut allocator,
            fail_label,
            insufficient_data_label,
            &HashMap::new(),
        )?;
        self.scratch_high_watermark = self.scratch_high_watermark.max(allocator.high_watermark());
        Ok(())
    }

    pub(crate) fn high_watermark(&self) -> u8 {
        self.scratch_high_watermark
    }

    fn compile_boolean_expr_with_allocator(
        &mut self,
        expr: &BooleanIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        fail_label: &str,
        insufficient_data_label: Option<&str>,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        match expr {
            BooleanIr::Comparison { lhs, op, rhs } => self.compile_comparison(
                lhs,
                *op,
                rhs,
                asm,
                allocator,
                fail_label,
                insufficient_data_label,
                bindings,
            ),
            BooleanIr::Availability { kind, value } => {
                self.compile_availability_check(*kind, value, asm, allocator, fail_label, bindings)
            }
            BooleanIr::Between {
                value,
                lower,
                upper,
            } => self.compile_between(
                value,
                lower,
                upper,
                asm,
                allocator,
                fail_label,
                insufficient_data_label,
                bindings,
            ),
            BooleanIr::And(left, right) => {
                self.compile_boolean_expr_with_allocator(
                    left,
                    asm,
                    allocator,
                    fail_label,
                    insufficient_data_label,
                    bindings,
                )?;
                self.compile_boolean_expr_with_allocator(
                    right,
                    asm,
                    allocator,
                    fail_label,
                    insufficient_data_label,
                    bindings,
                )
            }
            BooleanIr::Or(left, right) => {
                let try_right_label = self.next_label("or_rhs");
                let success_label = self.next_label("or_done");
                self.compile_boolean_expr_with_allocator(
                    left,
                    asm,
                    allocator,
                    &try_right_label,
                    insufficient_data_label,
                    bindings,
                )?;
                asm.push(format!("  JMP {success_label}"));
                asm.push(format!("{try_right_label}:"));
                self.compile_boolean_expr_with_allocator(
                    right,
                    asm,
                    allocator,
                    fail_label,
                    insufficient_data_label,
                    bindings,
                )?;
                asm.push(format!("{success_label}:"));
                Ok(())
            }
            BooleanIr::Not(inner) => {
                let negated_pass_label = self.next_label("not_pass");
                self.compile_boolean_expr_with_allocator(
                    inner,
                    asm,
                    allocator,
                    &negated_pass_label,
                    insufficient_data_label,
                    bindings,
                )?;
                asm.push(format!("  JMP {fail_label}"));
                asm.push(format!("{negated_pass_label}:"));
                Ok(())
            }
            BooleanIr::Quantified {
                kind,
                collection,
                predicate,
            } => self.compile_quantified(
                *kind,
                collection,
                predicate,
                asm,
                allocator,
                fail_label,
                insufficient_data_label,
                bindings,
            ),
        }
    }

    fn compile_comparison(
        &mut self,
        lhs: &ArithmeticIr,
        op: ComparisonOp,
        rhs: &ArithmeticIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        fail_label: &str,
        insufficient_data_label: Option<&str>,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        let lhs_temps = self.compile_arithmetic_expr(lhs, asm, allocator, bindings)?;
        let rhs_temps = self.compile_arithmetic_expr(rhs, asm, allocator, bindings)?;
        let insufficient_label = insufficient_data_label.unwrap_or(fail_label);
        emit_jump_if_unavailable(asm, lhs_temps.status, insufficient_label);
        emit_jump_if_unavailable(asm, rhs_temps.status, insufficient_label);

        emit_load(asm, lhs_temps.value);
        emit_subm(asm, rhs_temps.value);
        match op {
            ComparisonOp::Eq => asm.push(format!("  JNZ {fail_label}")),
            ComparisonOp::Ne => asm.push(format!("  JZ {fail_label}")),
            ComparisonOp::Lt => asm.push(format!("  JNS {fail_label}")),
            ComparisonOp::Le => {
                let pass_label = self.next_label("cmp_le_ok");
                asm.push(format!("  JS {pass_label}"));
                asm.push(format!("  JZ {pass_label}"));
                asm.push(format!("  JMP {fail_label}"));
                asm.push(format!("{pass_label}:"));
            }
            ComparisonOp::Gt => {
                asm.push(format!("  JS {fail_label}"));
                asm.push(format!("  JZ {fail_label}"));
            }
            ComparisonOp::Ge => asm.push(format!("  JS {fail_label}")),
        }
        Ok(())
    }

    fn compile_availability_check(
        &mut self,
        kind: AvailabilityPredicateKind,
        value: &ArithmeticIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        fail_label: &str,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        let value_temps = self.compile_arithmetic_expr(value, asm, allocator, bindings)?;
        match value_temps.status {
            None => match kind {
                AvailabilityPredicateKind::Present => {}
                AvailabilityPredicateKind::Missing | AvailabilityPredicateKind::Unknown => {
                    asm.push(format!("  JMP {fail_label}"));
                }
            },
            Some(status) => {
                emit_load(asm, status);
                match kind {
                    AvailabilityPredicateKind::Present => asm.push(format!("  JNZ {fail_label}")),
                    AvailabilityPredicateKind::Missing => {
                        asm.push("  SUB 1".to_string());
                        asm.push(format!("  JNZ {fail_label}"));
                    }
                    AvailabilityPredicateKind::Unknown => {
                        asm.push("  SUB 2".to_string());
                        asm.push(format!("  JNZ {fail_label}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn compile_between(
        &mut self,
        value: &ArithmeticIr,
        lower: &ArithmeticIr,
        upper: &ArithmeticIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        fail_label: &str,
        insufficient_data_label: Option<&str>,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        let value_temps = self.compile_arithmetic_expr(value, asm, allocator, bindings)?;
        let lower_temps = self.compile_arithmetic_expr(lower, asm, allocator, bindings)?;
        let upper_temps = self.compile_arithmetic_expr(upper, asm, allocator, bindings)?;
        let insufficient_label = insufficient_data_label.unwrap_or(fail_label);
        emit_jump_if_unavailable(asm, value_temps.status, insufficient_label);
        emit_jump_if_unavailable(asm, lower_temps.status, insufficient_label);
        emit_jump_if_unavailable(asm, upper_temps.status, insufficient_label);

        emit_load(asm, value_temps.value);
        emit_subm(asm, lower_temps.value);
        asm.push(format!("  JS {fail_label}"));

        emit_load(asm, value_temps.value);
        emit_subm(asm, upper_temps.value);
        let pass_label = self.next_label("between_ok");
        asm.push(format!("  JS {pass_label}"));
        asm.push(format!("  JZ {pass_label}"));
        asm.push(format!("  JMP {fail_label}"));
        asm.push(format!("{pass_label}:"));
        Ok(())
    }

    fn compile_arithmetic_expr(
        &mut self,
        expr: &ArithmeticIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<ExprTemps> {
        match expr {
            ArithmeticIr::Number(value) => {
                let value_temp = allocator.allocate()?;
                asm.push(format!("  LOADI {value}"));
                emit_store(asm, value_temp);
                Ok(ExprTemps {
                    value: value_temp,
                    status: None,
                })
            }
            ArithmeticIr::Value(value_ref) => {
                self.compile_value_ref(value_ref, asm, allocator, bindings)
            }
            ArithmeticIr::Binary { op, left, right } => {
                let left_temps = self.compile_arithmetic_expr(left, asm, allocator, bindings)?;
                let right_temps = self.compile_arithmetic_expr(right, asm, allocator, bindings)?;
                let value_temp = allocator.allocate()?;

                emit_load(asm, left_temps.value);
                match op {
                    ArithmeticOp::Add => emit_addm(asm, right_temps.value),
                    ArithmeticOp::Mul => emit_mulm(asm, right_temps.value),
                    ArithmeticOp::Sub => emit_subm(asm, right_temps.value),
                }
                emit_store(asm, value_temp);
                if left_temps.status.is_none() && right_temps.status.is_none() {
                    return Ok(ExprTemps {
                        value: value_temp,
                        status: None,
                    });
                }

                let status_temp = allocator.allocate()?;
                let unknown_label = self.next_label("arith_unknown");
                let missing_label = self.next_label("arith_missing");
                let done_label = self.next_label("arith_done");

                emit_jump_if_status_equals(asm, left_temps.status, 2, &unknown_label);
                emit_jump_if_status_equals(asm, right_temps.status, 2, &unknown_label);
                emit_jump_if_unavailable(asm, left_temps.status, &missing_label);
                emit_jump_if_unavailable(asm, right_temps.status, &missing_label);
                asm.push("  LOADI 0".to_string());
                emit_store(asm, status_temp);
                asm.push(format!("  JMP {done_label}"));

                asm.push(format!("{unknown_label}:"));
                asm.push("  LOADI 0".to_string());
                emit_store(asm, value_temp);
                asm.push("  LOADI 2".to_string());
                emit_store(asm, status_temp);
                asm.push(format!("  JMP {done_label}"));

                asm.push(format!("{missing_label}:"));
                asm.push("  LOADI 0".to_string());
                emit_store(asm, value_temp);
                asm.push("  LOADI 1".to_string());
                emit_store(asm, status_temp);

                asm.push(format!("{done_label}:"));
                Ok(ExprTemps {
                    value: value_temp,
                    status: Some(status_temp),
                })
            }
        }
    }

    fn compile_value_ref(
        &mut self,
        value_ref: &ValueRefIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<ExprTemps> {
        let value_temp = allocator.allocate()?;
        emit_load(
            asm,
            self.resolve_value_source_address(value_ref.source, bindings)?,
        );
        emit_store(asm, value_temp);
        if matches!(
            value_ref.availability,
            AvailabilityIr::Flags {
                presence: None,
                unknown: None,
            }
        ) {
            return Ok(ExprTemps {
                value: value_temp,
                status: None,
            });
        }
        let status_temp = allocator.allocate()?;
        self.emit_availability_status(&value_ref.availability, status_temp, asm, bindings)?;
        Ok(ExprTemps {
            value: value_temp,
            status: Some(status_temp),
        })
    }

    fn emit_availability_status(
        &mut self,
        availability: &AvailabilityIr,
        status_target: u8,
        asm: &mut Vec<String>,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        match availability {
            AvailabilityIr::Status(source) => {
                emit_load(
                    asm,
                    self.resolve_availability_source_address(*source, bindings)?,
                );
                emit_store(asm, status_target);
            }
            AvailabilityIr::Flags { presence, unknown } => {
                if presence.is_none() && unknown.is_none() {
                    asm.push("  LOADI 0".to_string());
                    emit_store(asm, status_target);
                    return Ok(());
                }
                let missing_check_label = self.next_label("availability_missing_check");
                let done_label = self.next_label("availability_done");

                if let Some(unknown_source) = unknown {
                    emit_load(
                        asm,
                        self.resolve_availability_source_address(*unknown_source, bindings)?,
                    );
                    asm.push(format!("  JZ {missing_check_label}"));
                    asm.push("  LOADI 2".to_string());
                    emit_store(asm, status_target);
                    asm.push(format!("  JMP {done_label}"));
                }

                asm.push(format!("{missing_check_label}:"));
                if let Some(presence_source) = presence {
                    let present_label = self.next_label("availability_present");
                    emit_load(
                        asm,
                        self.resolve_availability_source_address(*presence_source, bindings)?,
                    );
                    asm.push(format!("  JNZ {present_label}"));
                    asm.push("  LOADI 1".to_string());
                    emit_store(asm, status_target);
                    asm.push(format!("  JMP {done_label}"));
                    asm.push(format!("{present_label}:"));
                }
                asm.push("  LOADI 0".to_string());
                emit_store(asm, status_target);
                asm.push(format!("{done_label}:"));
            }
        }
        Ok(())
    }

    fn compile_quantified(
        &mut self,
        kind: QuantifierKind,
        collection: &QuantifiedCollection,
        predicate: &BooleanIr,
        asm: &mut Vec<String>,
        allocator: &mut TempAllocator,
        fail_label: &str,
        insufficient_data_label: Option<&str>,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<()> {
        let count_fail_label = insufficient_data_label.unwrap_or(fail_label);
        let count_ok_label = self.next_label("collection_count_ok");
        emit_load(asm, collection.count_memory);
        asm.push(format!("  SUB {}", i64::from(collection.max_items)));
        asm.push(format!("  JS {count_ok_label}"));
        asm.push(format!("  JZ {count_ok_label}"));
        asm.push(format!("  JMP {count_fail_label}"));
        asm.push(format!("{count_ok_label}:"));

        match kind {
            QuantifierKind::ForAll => {
                for item_index in 0..collection.max_items {
                    let skip_label = self.next_label("forall_skip");
                    emit_load(asm, collection.count_memory);
                    asm.push(format!("  SUB {}", i64::from(item_index) + 1));
                    asm.push(format!("  JS {skip_label}"));
                    let item_bindings =
                        self.bindings_for_collection_item(collection, item_index, bindings);
                    self.compile_boolean_expr_with_allocator(
                        predicate,
                        asm,
                        allocator,
                        fail_label,
                        insufficient_data_label,
                        &item_bindings,
                    )?;
                    asm.push(format!("{skip_label}:"));
                }
            }
            QuantifierKind::Exists => {
                let success_label = self.next_label("exists_success");
                for item_index in 0..collection.max_items {
                    let inactive_label = self.next_label("exists_inactive");
                    let next_slot_label = self.next_label("exists_next");
                    emit_load(asm, collection.count_memory);
                    asm.push(format!("  SUB {}", i64::from(item_index) + 1));
                    asm.push(format!("  JS {inactive_label}"));
                    let item_bindings =
                        self.bindings_for_collection_item(collection, item_index, bindings);
                    self.compile_boolean_expr_with_allocator(
                        predicate,
                        asm,
                        allocator,
                        &next_slot_label,
                        insufficient_data_label,
                        &item_bindings,
                    )?;
                    asm.push(format!("  JMP {success_label}"));
                    asm.push(format!("{next_slot_label}:"));
                    asm.push(format!("{inactive_label}:"));
                }
                asm.push(format!("  JMP {fail_label}"));
                asm.push(format!("{success_label}:"));
            }
        }

        Ok(())
    }

    fn bindings_for_collection_item(
        &self,
        collection: &QuantifiedCollection,
        item_index: u8,
        inherited: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> HashMap<BindingId, BoundCollectionSlot> {
        let mut bindings = inherited.clone();
        let base = usize::from(collection.base_memory)
            + usize::from(item_index) * usize::from(collection.stride);
        bindings.insert(
            collection.binding,
            BoundCollectionSlot {
                collection_name: collection.name.clone(),
                item_base: base,
            },
        );
        bindings
    }

    fn resolve_value_source_address(
        &self,
        source: ValueSource,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<u8> {
        match source {
            ValueSource::Cell(address) => Ok(address),
            ValueSource::BoundField { binding, offset } => {
                self.resolve_bound_address(binding, offset, bindings)
            }
        }
    }

    fn resolve_availability_source_address(
        &self,
        source: AvailabilitySource,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<u8> {
        match source {
            AvailabilitySource::Cell(address) => Ok(address),
            AvailabilitySource::BoundField { binding, offset } => {
                self.resolve_bound_address(binding, offset, bindings)
            }
        }
    }

    fn resolve_bound_address(
        &self,
        binding: BindingId,
        offset: u8,
        bindings: &HashMap<BindingId, BoundCollectionSlot>,
    ) -> Result<u8> {
        bindings
            .get(&binding)
            .ok_or_else(|| logic_error("unbound quantified collection slot".to_string()))
            .and_then(|slot| {
                let address = slot.item_base + usize::from(offset);
                u8::try_from(address).map_err(|_| {
                    logic_error(format!(
                        "collection `{}` item layout exceeds the u8 address space",
                        slot.collection_name
                    ))
                })
            })
    }

    fn next_label(&mut self, prefix: &str) -> String {
        let label = format!("{prefix}_{}", self.label_counter);
        self.label_counter += 1;
        label
    }
}

#[derive(Debug, Clone, Copy)]
struct ExprTemps {
    value: u8,
    status: Option<u8>,
}

#[derive(Debug, Clone)]
struct BoundCollectionSlot {
    collection_name: String,
    item_base: usize,
}

struct TempAllocator {
    next: u8,
    high_watermark: u8,
}

impl TempAllocator {
    fn new(base: u8) -> Self {
        Self {
            next: base,
            high_watermark: base,
        }
    }

    fn allocate(&mut self) -> Result<u8> {
        if self.next == u8::MAX {
            return Err(logic_error(
                "out of scratch memory while compiling cDMN logic".to_string(),
            ));
        }
        let address = self.next;
        self.next = self.next.saturating_add(1);
        self.high_watermark = self.high_watermark.max(self.next);
        Ok(address)
    }

    fn high_watermark(&self) -> u8 {
        self.high_watermark
    }
}

fn emit_jump_if_unavailable(asm: &mut Vec<String>, status_address: Option<u8>, label: &str) {
    if let Some(status_address) = status_address {
        emit_load(asm, status_address);
        asm.push(format!("  JNZ {label}"));
    }
}

fn emit_jump_if_status_equals(
    asm: &mut Vec<String>,
    status_address: Option<u8>,
    code: i64,
    label: &str,
) {
    if let Some(status_address) = status_address {
        emit_load(asm, status_address);
        asm.push(format!("  SUB {code}"));
        asm.push(format!("  JZ {label}"));
    }
}

fn emit_load(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  LOAD {address}"));
}

fn emit_store(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  STORE {address}"));
}

fn emit_addm(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  ADDM {address}"));
}

fn emit_subm(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  SUBM {address}"));
}

fn emit_mulm(asm: &mut Vec<String>, address: u8) {
    asm.push(format!("  MULM {address}"));
}

fn logic_error(message: String) -> VmError {
    VmError::InvalidConfig(message)
}
