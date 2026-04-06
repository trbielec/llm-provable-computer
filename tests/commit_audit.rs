use llm_provable_computer::approximate::ApproximateVerificationConfig;
use llm_provable_computer::commit_and_audit::{
    audit_execution, commit_execution, verify_audit, verify_audit_with_config,
    verify_receipt_bindings, AuditRequest, AuditTarget,
};
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::instruction::Program;
use llm_provable_computer::model::{
    DispatchInfo, SharedTransformerExecutionWitness, SharedTransformerStepWitness, TransformerVm,
};
use llm_provable_computer::state::MachineState;

#[test]
fn test_commit_and_audit_lifecycle() {
    let config = TransformerVmConfig {
        num_layers: 1,
        ff_dim: 256,
        ..TransformerVmConfig::default()
    };
    let program = Program::new(
        vec![llm_provable_computer::instruction::Instruction::Halt],
        1,
    );
    let vm = TransformerVm::new(config.clone(), program).unwrap();

    let initial_state = MachineState::new(1);
    let final_state = MachineState::new(1);

    // Create a minimal witness
    let witness = SharedTransformerExecutionWitness {
        initial_state: initial_state.clone(),
        steps: vec![SharedTransformerStepWitness {
            step_number: 0,
            pre_state: initial_state.clone(),
            input_token: vec![0.0; config.d_model],
            decoded_state: initial_state.clone(),
            dispatch: DispatchInfo {
                instruction: llm_provable_computer::instruction::Instruction::Halt,
                layer_idx: 0,
            },
            operand: 0,
            input_vector: vec![0.0; 9 + 2 * llm_provable_computer::state::WORD_BITS],
            ff_gate: vec![0.0; 256],
            ff_value: vec![0.0; 256],
            ff_hidden: vec![0.0; 256],
            ff_output: vec![0.0; 6],
            transition: llm_provable_computer::model::SharedTransformerTransitionWitness {
                pc: 0,
                acc: 0,
                sp: 0,
                zero_flag: true,
                carry_flag: false,
                halted: true,
                memory_write: None,
            },
            post_write_state: initial_state.clone(),
            output_token: vec![0.0; config.d_model],
            final_state: initial_state.clone(),
        }],
        final_state: final_state.clone(),
    };

    // Commit
    let (receipt, m_trace, w_commitment) = commit_execution(&vm, &witness).unwrap();
    assert_eq!(receipt.step_count, 5);
    assert!(verify_receipt_bindings(&receipt, &vm, &initial_state).unwrap());

    // Audit Request (Step 3 - Transition row, which triggers weight opening)
    let request = AuditRequest {
        target: AuditTarget::Steps(vec![3]),
        challenge: None,
        approximate_config: None,
    };

    // Staged trace for auditing
    let staged_trace =
        llm_provable_computer::transformer_trace::build_shared_transformer_staged_trace(
            &witness,
            config.ff_dim,
        )
        .unwrap();

    // Audit Response
    let response = audit_execution(
        &vm,
        &m_trace,
        &w_commitment,
        &witness,
        &staged_trace,
        &request,
    )
    .unwrap();
    assert_eq!(response.opened_steps.len(), 1);
    assert!(response.opened_steps[0].opened_weight.is_some());

    // Verify
    assert!(verify_audit(&receipt, &response, None));

    // Tamper with weights root and check failure
    let mut tampered_receipt = receipt.clone();
    tampered_receipt.weights_root[0] ^= 0xFF;
    assert!(!verify_audit(&tampered_receipt, &response, None));

    let mut tampered_program_receipt = receipt.clone();
    tampered_program_receipt.program_hash[0] ^= 0xFF;
    assert!(!verify_receipt_bindings(&tampered_program_receipt, &vm, &initial_state).unwrap());

    let mut tampered_initial_state_receipt = receipt.clone();
    tampered_initial_state_receipt.initial_state_hash[0] ^= 0xFF;
    assert!(
        !verify_receipt_bindings(&tampered_initial_state_receipt, &vm, &initial_state).unwrap()
    );
}

#[test]
fn approximate_audit_accepts_small_replay_drift() {
    let config = TransformerVmConfig {
        num_layers: 1,
        ff_dim: 256,
        ..TransformerVmConfig::default()
    };
    let program = Program::new(
        vec![llm_provable_computer::instruction::Instruction::Halt],
        1,
    );
    let vm = TransformerVm::new(config.clone(), program).unwrap();

    let initial_state = MachineState::new(1);
    let witness = SharedTransformerExecutionWitness {
        initial_state: initial_state.clone(),
        steps: vec![SharedTransformerStepWitness {
            step_number: 0,
            pre_state: initial_state.clone(),
            input_token: vec![0.0; config.d_model],
            decoded_state: initial_state.clone(),
            dispatch: DispatchInfo {
                instruction: llm_provable_computer::instruction::Instruction::Halt,
                layer_idx: 0,
            },
            operand: 0,
            input_vector: vec![1.0; 9 + 2 * llm_provable_computer::state::WORD_BITS],
            ff_gate: vec![1.0; 256],
            ff_value: vec![2.0; 256],
            ff_hidden: vec![3.0; 256],
            ff_output: vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            transition: llm_provable_computer::model::SharedTransformerTransitionWitness {
                pc: 0,
                acc: 0,
                sp: 0,
                zero_flag: true,
                carry_flag: false,
                halted: true,
                memory_write: None,
            },
            post_write_state: initial_state.clone(),
            output_token: vec![0.0; config.d_model],
            final_state: initial_state.clone(),
        }],
        final_state: initial_state.clone(),
    };

    let (receipt, m_trace, w_commitment) = commit_execution(&vm, &witness).unwrap();
    let staged_trace =
        llm_provable_computer::transformer_trace::build_shared_transformer_staged_trace(
            &witness,
            config.ff_dim,
        )
        .unwrap();
    let request = AuditRequest {
        target: AuditTarget::Steps(vec![1, 2, 3]),
        challenge: None,
        approximate_config: Some(ApproximateVerificationConfig {
            absolute_epsilon: 0.01,
            relative_epsilon: 0.01,
            max_deviating_fraction: 0.34,
        }),
    };

    let response = audit_execution(
        &vm,
        &m_trace,
        &w_commitment,
        &witness,
        &staged_trace,
        &request,
    )
    .unwrap();

    let mut replay = witness.clone();
    replay.steps[0].input_vector[0] += 0.001;
    replay.steps[0].ff_gate[1] += 0.002;
    replay.steps[0].ff_output[3] += 0.001;

    let report = verify_audit_with_config(
        &receipt,
        &response,
        &replay,
        request.approximate_config.as_ref().unwrap(),
        None,
    )
    .unwrap();

    assert!(report.passed);
    assert_eq!(report.approximate_comparisons.len(), 3);
}
