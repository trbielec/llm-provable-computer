use serde::{Deserialize, Serialize};

use super::{
    compile_cdmn_table_to_decision_program, compile_cdmn_table_with_metadata,
    evaluate_cdmn_decision_program, CdmnDecisionEvaluation, CdmnTable, CompiledCdmnDecisionProgram,
};
use crate::assembly::parse_program;
use crate::config::TransformerVmConfig;
use crate::error::{Result, VmError};
use crate::proof::{
    prove_execution_stark, prove_execution_stark_with_options, verify_execution_stark,
    VanillaStarkExecutionProof, VanillaStarkProofOptions,
};
use crate::state::Word;
use crate::ProgramCompiler;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkCompatClaim {
    pub decision_program: CompiledCdmnDecisionProgram,
    pub status: Word,
    pub reason_code: Word,
    pub vm_status_register: u8,
    pub vm_reason_register: u8,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkCompatProof {
    pub claim: CdmnDecisionStarkCompatClaim,
    pub vm_proof: VanillaStarkExecutionProof,
}

pub fn prove_cdmn_decision_stark_compat(
    table: &CdmnTable,
    initial_memory: &[Word],
    max_steps: usize,
) -> Result<CdmnDecisionStarkCompatProof> {
    prove_cdmn_decision_stark_compat_with_options(
        table,
        initial_memory,
        max_steps,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_cdmn_decision_stark_compat_with_options(
    table: &CdmnTable,
    initial_memory: &[Word],
    max_steps: usize,
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkCompatProof> {
    let decision_program = compile_cdmn_table_to_decision_program(table)?;
    let decision_evaluation = evaluate_cdmn_decision_program(&decision_program, initial_memory)?;
    let (vm_proof, status, reason_code, vm_status_register, vm_reason_register) =
        prove_and_extract_vm_outputs(table, initial_memory, max_steps, Some(options))?;
    validate_decision_matches_vm(&decision_evaluation, status, reason_code)?;

    Ok(CdmnDecisionStarkCompatProof {
        claim: CdmnDecisionStarkCompatClaim {
            decision_program,
            status,
            reason_code,
            vm_status_register,
            vm_reason_register,
            policy_commitment: decision_evaluation.policy_commitment,
            input_commitment: decision_evaluation.input_commitment,
        },
        vm_proof,
    })
}

pub fn verify_cdmn_decision_stark_compat(proof: &CdmnDecisionStarkCompatProof) -> Result<bool> {
    if !verify_execution_stark(&proof.vm_proof)? {
        return Ok(false);
    }

    let initial_memory = proof.vm_proof.claim.program.initial_memory().to_vec();
    let evaluation =
        evaluate_cdmn_decision_program(&proof.claim.decision_program, &initial_memory)?;
    let final_state = &proof.vm_proof.claim.final_state;

    if evaluation.status != proof.claim.status
        || evaluation.reason_code != proof.claim.reason_code
        || evaluation.policy_commitment != proof.claim.policy_commitment
        || evaluation.input_commitment != proof.claim.input_commitment
    {
        return Ok(false);
    }

    let vm_status = *final_state
        .memory
        .get(usize::from(proof.claim.vm_status_register))
        .ok_or_else(|| {
            VmError::InvalidConfig(
                "proved cDMN program does not expose a status register".to_string(),
            )
        })?;
    let vm_reason = *final_state
        .memory
        .get(usize::from(proof.claim.vm_reason_register))
        .ok_or_else(|| {
            VmError::InvalidConfig(
                "proved cDMN program does not expose a reason register".to_string(),
            )
        })?;

    Ok(vm_status == proof.claim.status && vm_reason == proof.claim.reason_code)
}

fn prove_and_extract_vm_outputs(
    table: &CdmnTable,
    initial_memory: &[Word],
    max_steps: usize,
    options: Option<VanillaStarkProofOptions>,
) -> Result<(VanillaStarkExecutionProof, Word, Word, u8, u8)> {
    let compiled = compile_cdmn_table_with_metadata(table)?;
    let mut program = parse_program(&compiled.assembly)?;
    let memory_size = program.memory_size();
    if initial_memory.len() > memory_size {
        return Err(VmError::InvalidConfig(format!(
            "initial memory length {} exceeds compiled cDMN memory size {}",
            initial_memory.len(),
            memory_size
        )));
    }

    let mut padded_memory = vec![0; memory_size];
    padded_memory[..initial_memory.len()].copy_from_slice(initial_memory);
    program = program.with_initial_memory(padded_memory)?;

    let model = ProgramCompiler.compile_program(program, TransformerVmConfig::default())?;
    let vm_proof = match options {
        Some(options) => prove_execution_stark_with_options(&model, max_steps, options)?,
        None => prove_execution_stark(&model, max_steps)?,
    };
    let final_state = &vm_proof.claim.final_state;
    let status = *final_state
        .memory
        .get(usize::from(compiled.output.status_register))
        .ok_or_else(|| {
            VmError::InvalidConfig("missing cDMN status register in final state".to_string())
        })?;
    let reason_code = *final_state
        .memory
        .get(usize::from(compiled.output.reason_register))
        .ok_or_else(|| {
            VmError::InvalidConfig("missing cDMN reason register in final state".to_string())
        })?;
    Ok((
        vm_proof,
        status,
        reason_code,
        compiled.output.status_register,
        compiled.output.reason_register,
    ))
}

fn validate_decision_matches_vm(
    evaluation: &CdmnDecisionEvaluation,
    status: Word,
    reason_code: Word,
) -> Result<()> {
    if evaluation.status != status || evaluation.reason_code != reason_code {
        return Err(VmError::ExecutionMismatch {
            step: 0,
            message: format!(
                "decision IR evaluation produced status={} reason={} but proved VM execution produced status={} reason={}",
                evaluation.status, evaluation.reason_code, status, reason_code
            ),
        });
    }
    Ok(())
}
