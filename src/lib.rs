pub mod approximate;
pub mod batch;
#[cfg(all(feature = "metal-batch", target_os = "macos"))]
pub mod metal_batch;
#[cfg(all(feature = "metal-batch", target_os = "macos"))]
pub use metal_batch::{MetalBatchRunResult, MetalBatchRunner};
pub mod assembly;
#[cfg(feature = "burn-model")]
pub mod burn_model;
#[cfg(feature = "burn-model")]
pub mod burn_runtime;
pub mod candidate_eval;
pub mod cdmn;
pub mod commit_and_audit;
pub mod compiler;
pub mod config;
pub mod dmn;
mod dmn_backend_tvm;
mod dmn_decision;
mod dmn_direct;
mod dmn_ir;
pub mod effect_root;
pub mod engine;
pub mod error;
pub mod freivalds;
pub mod geometry;
pub mod instruction;
pub mod interpreter;
pub mod memory;
pub mod merkle_trace;
pub mod model;
#[cfg(feature = "onnx-export")]
pub mod onnx_export;
#[cfg(feature = "onnx-export")]
pub mod onnx_runtime;
pub mod planner_bridge;
pub use planner_bridge::{
    AdmissibilityFilter, AdmissibilityResult, Planner, PlannerBridge, PlannerBridgeResult,
    PlannerContext, ProposedCandidate, StaticPlanner,
};
pub mod poet_proof;
pub mod proof;
pub mod proposal_receipt;
pub mod runtime;
pub mod state;
pub mod transformer_stark;
pub mod transformer_trace;
pub mod tui;
pub mod vanillastark;
pub mod verification;

pub use approximate::{
    canonicalize_for_comparison, canonicalize_values, compare_approximate, dequantize_signed,
    quantize_signed, ApproximateComparisonResult, ApproximateNumericError, ApproximateResult,
    ApproximateVerificationConfig, AUDIT_NUMERIC_SCALE, AUDIT_NUMERIC_SCALE_F64,
};
pub use assembly::parse_program;
pub use batch::{BatchRunResult, BatchRunner};
#[cfg(feature = "burn-model")]
pub use burn_model::{
    load_burn_model, load_burn_model_on_device, save_burn_model, BurnTransformerVm,
};
#[cfg(feature = "burn-model")]
pub use burn_runtime::BurnExecutionRuntime;
pub use cdmn::{
    cdmn_decision_input_commitment, compile_cdmn_table, compile_cdmn_table_to_decision_program,
    compile_cdmn_table_with_metadata, estimate_cdmn_decision_stark_direct_cost,
    estimate_compiled_cdmn_decision_stark_direct_cost, evaluate_cdmn_decision_program,
    evaluate_cdmn_table_decision, parse_logic_expr, prove_cdmn_decision_stark_compat,
    prove_cdmn_decision_stark_compat_with_options, prove_cdmn_decision_stark_direct,
    prove_cdmn_decision_stark_direct_profiled,
    prove_cdmn_decision_stark_direct_profiled_with_options,
    prove_cdmn_decision_stark_direct_with_options, prove_compiled_cdmn_decision_stark_direct,
    prove_compiled_cdmn_decision_stark_direct_profiled,
    prove_compiled_cdmn_decision_stark_direct_profiled_with_options,
    prove_compiled_cdmn_decision_stark_direct_with_options, verify_cdmn_decision_stark_compat,
    verify_cdmn_decision_stark_direct, AvailabilityPredicateKind, BooleanExpr,
    CdmnDecisionEvaluation, CdmnDecisionStarkCompatClaim, CdmnDecisionStarkCompatProof,
    CdmnDecisionStarkDirectClaim, CdmnDecisionStarkDirectCostEstimate,
    CdmnDecisionStarkDirectProof, CdmnDecisionStarkDirectProvingProfile,
    CdmnDecisionStarkDirectProvingReport, CdmnOutputLayout, CdmnProgramMetadata, CdmnTable,
    Collection, CollectionField, CompiledCdmnDecisionProgram, CompiledCdmnProgram, Constraint,
    DecisionArithmeticExpr, DecisionAvailability, DecisionAvailabilitySource, DecisionBooleanExpr,
    DecisionConstraint, DecisionCountBoundCheck, DecisionDefinition, DecisionFailureReason,
    DecisionValueRef, DecisionValueSource, Definition, ReasonMetadata, Variable,
};
pub use commit_and_audit::{
    audit_execution, build_auditable_numeric_rows, commit_execution, receipt_initial_state_hash,
    receipt_program_hash, verify_audit, verify_audit_with_config, verify_receipt_bindings,
    ApproximateAuditVerificationReport, AuditRequest, AuditResponse, AuditTarget,
    AuditableNumericTraceRow, ExecutionReceipt, OpenedStep, OpenedWeight, WeightCommitment,
};
pub use compiler::ProgramCompiler;
pub use config::{Attention2DMode, TransformerVmConfig};
pub use dmn::{
    compile_dmn_table, compile_dmn_table_with_metadata, CompiledDmnProgram, DmnAssignment,
    DmnCollectOperator, DmnDefaultRule, DmnHitPolicy, DmnInput, DmnMatchedOutputLayout, DmnOutput,
    DmnOutputLayout, DmnReasonMetadata, DmnRule, DmnTable, DmnTraceLayout, RuleTraceMetadata,
};
pub use dmn_decision::{
    compile_dmn_table_to_decision_program, evaluate_compiled_dmn_decision_program,
    evaluate_dmn_table_decision, CompiledDmnDecisionDefaultRule, CompiledDmnDecisionProgram,
    CompiledDmnDecisionRule, DmnDecisionEvaluation, DmnSystemReasonSet,
};
pub use dmn_direct::{
    estimate_compiled_dmn_decision_stark_direct_cost, estimate_dmn_decision_stark_direct_cost,
    prove_compiled_dmn_decision_stark_direct, prove_compiled_dmn_decision_stark_direct_profiled,
    prove_compiled_dmn_decision_stark_direct_profiled_with_options,
    prove_compiled_dmn_decision_stark_direct_with_options, prove_dmn_decision_stark_direct,
    prove_dmn_decision_stark_direct_profiled,
    prove_dmn_decision_stark_direct_profiled_with_options,
    prove_dmn_decision_stark_direct_with_options, verify_compiled_dmn_decision_stark_direct,
    verify_dmn_decision_stark_direct, CompiledDmnDecisionStarkDirectClaim,
    CompiledDmnDecisionStarkDirectProof, CompiledDmnDecisionStarkDirectProvingReport,
    DmnDecisionStarkDirectBackend, DmnDecisionStarkDirectClaim, DmnDecisionStarkDirectCostEstimate,
    DmnDecisionStarkDirectProof, DmnDecisionStarkDirectProvingProfile,
    DmnDecisionStarkDirectProvingReport,
};
pub use engine::{
    ExecutionEngine, ExecutionResult, ExecutionTraceEntry, VerificationResult, VerifiedEngine,
};
pub use error::{Result, VmError};
pub use geometry::{HullKvCache, Point2D};
pub use instruction::{Instruction, Program};
pub use interpreter::{NativeExecutionResult, NativeInterpreter, NativeTraceEntry};
pub use memory::AddressedMemory;
pub use model::{
    DispatchInfo, SharedTransformerExactDomainViolation, SharedTransformerExecutionWitness,
    SharedTransformerStepWitness, SharedTransformerTransitionWitness, TransformerVm,
};
#[cfg(feature = "onnx-export")]
pub use onnx_export::{
    export_program_onnx, load_onnx_program_metadata, OnnxInputLayoutEntry, OnnxInstructionMetadata,
    OnnxInstructionRead, OnnxProgramMetadata, ONNX_OUTPUT_DIM,
};
#[cfg(feature = "onnx-export")]
pub use onnx_runtime::OnnxExecutionRuntime;
pub use proof::{
    estimate_execution_stark_cost, load_execution_stark_proof, prove_execution_stark,
    prove_execution_stark_with_options, save_execution_stark_proof, verify_execution_stark,
    VanillaStarkCostEstimate, VanillaStarkExecutionClaim, VanillaStarkExecutionProof,
    VanillaStarkProofOptions,
};
pub use runtime::ExecutionRuntime;
pub use state::{decode_state, encode_state, MachineState, Word, MIN_D_MODEL, WORD_BITS};
pub use transformer_stark::{
    estimate_shared_transformer_forward_pass_stark_cost,
    estimate_shared_transformer_staged_skeleton_stark_cost,
    prove_shared_transformer_forward_pass_stark,
    prove_shared_transformer_forward_pass_stark_profiled,
    prove_shared_transformer_forward_pass_stark_profiled_with_options,
    prove_shared_transformer_forward_pass_stark_with_options,
    prove_shared_transformer_staged_skeleton_stark,
    prove_shared_transformer_staged_skeleton_stark_profiled,
    prove_shared_transformer_staged_skeleton_stark_profiled_with_options,
    prove_shared_transformer_staged_skeleton_stark_with_options,
    validate_shared_transformer_staged_trace, verify_shared_transformer_forward_pass_stark,
    verify_shared_transformer_staged_skeleton_stark, SharedTransformerForwardPassClaim,
    SharedTransformerForwardPassCostEstimate, SharedTransformerForwardPassProof,
    SharedTransformerForwardPassProvingProfile, SharedTransformerForwardPassProvingReport,
    SharedTransformerStagedSkeletonClaim, SharedTransformerStagedSkeletonCostEstimate,
    SharedTransformerStagedSkeletonProof, SharedTransformerStagedSkeletonProvingProfile,
    SharedTransformerStagedSkeletonProvingReport, SharedTransformerTraceValidationReport,
};
pub use transformer_trace::{
    build_shared_transformer_staged_trace, SharedTransformerAttentionTraceRow,
    SharedTransformerFeedForwardChunkTraceRow, SharedTransformerInputTraceRow,
    SharedTransformerOutputTraceRow, SharedTransformerStagedTrace, SharedTransformerTraceRow,
    SharedTransformerTraceStage, SharedTransformerTransitionTraceRow,
};
pub use tui::run_execution_tui;
pub use vanillastark::{
    prove_aggregate_verification, prove_recursive_inner, prove_verification,
    verify_aggregate_proof, verify_recursive_inner, RecursiveAggregateProof, RecursiveInnerProof,
    RescueMerkleTree, RescueProofStream, VerifierAirPublicInput,
};
pub use verification::{verify_engines, verify_model_against_native, ExecutionComparison};
