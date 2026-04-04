pub mod ast;
pub(crate) mod backend_tvm;
pub(crate) mod ir;
pub(crate) mod parser;
pub(crate) mod proof_eval;
mod proof_ir;
pub(crate) mod schema;
pub(crate) mod semantics;
mod stark_compat;
mod stark_direct;

pub use self::ast::{
    ArithmeticExpr, ArithmeticOp, AvailabilityPredicateKind, BooleanExpr, ComparisonOp,
    QuantifierKind,
};
use self::backend_tvm::compile_tvm_program;
pub use self::parser::parse_logic_expr;
use self::proof_eval::evaluate_decision_program;
use self::proof_ir::lower_to_decision_program;
pub use self::schema::{
    CdmnOutputLayout, CdmnProgramMetadata, CdmnTable, Collection, CollectionField,
    CompiledCdmnProgram, Constraint, Definition, ReasonMetadata, Variable,
};
use self::semantics::build_compilation_plan;
use crate::error::Result;

pub fn compile_cdmn_table(table: &CdmnTable) -> Result<String> {
    Ok(compile_cdmn_table_with_metadata(table)?.assembly)
}

pub fn compile_cdmn_table_with_metadata(table: &CdmnTable) -> Result<CompiledCdmnProgram> {
    let plan = build_compilation_plan(table)?;
    compile_tvm_program(&plan)
}

pub fn compile_cdmn_table_to_decision_program(
    table: &CdmnTable,
) -> Result<CompiledCdmnDecisionProgram> {
    let plan = build_compilation_plan(table)?;
    lower_to_decision_program(&table.name, &plan)
}

pub fn evaluate_cdmn_decision_program(
    program: &CompiledCdmnDecisionProgram,
    initial_memory: &[crate::state::Word],
) -> Result<CdmnDecisionEvaluation> {
    evaluate_decision_program(program, initial_memory)
}

pub fn evaluate_cdmn_table_decision(
    table: &CdmnTable,
    initial_memory: &[crate::state::Word],
) -> Result<CdmnDecisionEvaluation> {
    let program = compile_cdmn_table_to_decision_program(table)?;
    evaluate_decision_program(&program, initial_memory)
}

pub use self::proof_eval::{
    input_commitment as cdmn_decision_input_commitment, CdmnDecisionEvaluation,
};
pub use self::proof_ir::{
    CompiledCdmnDecisionProgram, DecisionArithmeticExpr, DecisionAvailability,
    DecisionAvailabilitySource, DecisionBooleanExpr, DecisionConstraint, DecisionCountBoundCheck,
    DecisionDefinition, DecisionFailureReason, DecisionValueRef, DecisionValueSource,
};
pub use self::stark_compat::{
    prove_cdmn_decision_stark_compat, prove_cdmn_decision_stark_compat_with_options,
    verify_cdmn_decision_stark_compat, CdmnDecisionStarkCompatClaim, CdmnDecisionStarkCompatProof,
};
pub use self::stark_direct::{
    estimate_cdmn_decision_stark_direct_cost, estimate_compiled_cdmn_decision_stark_direct_cost,
    prove_cdmn_decision_stark_direct, prove_cdmn_decision_stark_direct_profiled,
    prove_cdmn_decision_stark_direct_profiled_with_options,
    prove_cdmn_decision_stark_direct_with_options, prove_compiled_cdmn_decision_stark_direct,
    prove_compiled_cdmn_decision_stark_direct_profiled,
    prove_compiled_cdmn_decision_stark_direct_profiled_with_options,
    prove_compiled_cdmn_decision_stark_direct_with_options, verify_cdmn_decision_stark_direct,
    CdmnDecisionStarkDirectClaim, CdmnDecisionStarkDirectCostEstimate,
    CdmnDecisionStarkDirectProof, CdmnDecisionStarkDirectProvingProfile,
    CdmnDecisionStarkDirectProvingReport,
};
