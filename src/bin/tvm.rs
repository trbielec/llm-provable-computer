use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;
#[cfg(feature = "onnx-export")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "burn-model")]
use burn::backend::NdArray;
use clap::{Parser, Subcommand};
#[cfg(any(feature = "burn-model", feature = "onnx-export"))]
use llm_provable_computer::verify_engines;
#[cfg(feature = "onnx-export")]
use llm_provable_computer::{export_program_onnx, OnnxExecutionRuntime};
use llm_provable_computer::{
    load_execution_stark_proof, prove_aggregate_verification, prove_execution_stark,
    run_execution_tui, save_execution_stark_proof, verify_aggregate_proof, verify_execution_stark,
    verify_model_against_native, ApproximateVerificationConfig, Attention2DMode, ExecutionResult,
    ExecutionRuntime, ExecutionTraceEntry, MachineState, NativeInterpreter, ProgramCompiler,
    RecursiveAggregateProof, TransformerVm, TransformerVmConfig, VmError,
};
#[cfg(feature = "burn-model")]
use llm_provable_computer::{BurnExecutionRuntime, BurnTransformerVm};

#[cfg(feature = "burn-model")]
type CliBurnBackend = NdArray<f64>;

#[derive(Debug, Parser)]
#[command(
    name = "tvm",
    about = "Run deterministic llm-provable-computer programs."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a program and print the final machine state.
    Run {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// Maximum number of execution steps before stopping.
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// Emit the full step-by-step execution trace.
        #[arg(long)]
        trace: bool,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Execution backend to use for the run.
        #[arg(
            long,
            default_value = "transformer",
            value_parser = parse_execution_engine
        )]
        engine: CliExecutionEngine,
        /// Verify the transformer runtime against the native interpreter.
        #[arg(long)]
        verify_native: bool,
        /// Verify the transformer and native runtimes against Burn.
        #[arg(long)]
        verify_burn: bool,
        /// Verify the transformer and native runtimes against ONNX.
        #[arg(long)]
        verify_onnx: bool,
        /// Verify all available runtimes in lockstep.
        #[arg(long, conflicts_with_all = ["verify_native", "verify_burn", "verify_onnx"])]
        verify_all: bool,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
    /// Run the interactive terminal viewer for a program.
    Tui {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// Maximum number of execution steps before stopping.
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// UI refresh interval in milliseconds.
        #[arg(long, default_value_t = 60)]
        tick_ms: u64,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
    /// Export the compiled program as per-instruction ONNX graphs.
    ExportOnnx {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// Directory where ONNX models and metadata will be written.
        #[arg(short = 'o', long = "output-dir")]
        output_dir: PathBuf,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
    /// Produce a STARK proof for a supported execution.
    ProveStark {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// File where the serialized proof JSON will be written.
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
        /// Maximum number of execution steps before stopping.
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
    /// Verify a previously generated STARK proof.
    VerifyStark {
        /// Path to the serialized proof JSON file.
        proof: PathBuf,
    },
    /// Aggregate one or more STARK proofs into a Rescue-bound recursive artifact.
    AggregateProofs {
        /// Paths to the serialized STARK proof JSON files.
        proofs: Vec<PathBuf>,
        /// File where the aggregate proof JSON will be written.
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
    },
    /// Verify a previously generated aggregate proof artifact.
    VerifyAggregate {
        /// Path to the serialized aggregate proof JSON file.
        proof: PathBuf,
    },
    /// Produce a compact execution receipt for commit-and-audit.
    Commit {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// File where the serialized receipt JSON will be written.
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
        /// Maximum number of execution steps before stopping.
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
    /// Run one or more programs across N independent initial states in parallel (CPU batch).
    BatchRun {
        /// Paths to the source `.tvm` programs.
        #[arg(required = true)]
        programs: Vec<PathBuf>,
        /// Number of independent runs.
        #[arg(long, default_value_t = 64)]
        count: usize,
        /// Maximum number of execution steps per run.
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// Number of transformer layers.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
        /// Use Metal GPU batch runner (requires metal-batch feature, macOS only).
        #[arg(long)]
        metal: bool,
        /// After the batch run, verify every result against the native interpreter
        /// and report mismatches. Useful for input-sweep correctness testing.
        #[arg(long)]
        verify_native: bool,
    },
    /// Audit a previously generated execution receipt.
    Audit {
        /// Path to the source `.tvm` program.
        program: PathBuf,
        /// Path to the serialized receipt JSON file.
        receipt: PathBuf,
        /// Specific step index to audit.
        #[arg(long)]
        step: Option<usize>,
        /// Absolute epsilon for approximate replay verification.
        #[arg(long)]
        absolute_epsilon: Option<f64>,
        /// Relative epsilon for approximate replay verification.
        #[arg(long)]
        relative_epsilon: Option<f64>,
        /// Maximum allowed fraction of deviating values in approximate replay verification.
        #[arg(long)]
        max_deviating_fraction: Option<f64>,
        /// Emit structured JSON for approximate comparison results.
        #[arg(long)]
        json: bool,
        /// Number of transformer layers to distribute instructions across.
        #[arg(long, default_value_t = 1)]
        layers: usize,
        /// Attention mode to use for memory reads.
        #[arg(
            long,
            default_value = "average-hard",
            value_parser = parse_attention_mode
        )]
        attention_mode: Attention2DMode,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliExecutionEngine {
    Native,
    Transformer,
    Burn,
    Onnx,
}

impl std::fmt::Display for CliExecutionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::Transformer => f.write_str("transformer"),
            Self::Burn => f.write_str("burn"),
            Self::Onnx => f.write_str("onnx"),
        }
    }
}

#[derive(Debug, Clone)]
struct EngineRunOutput {
    result: ExecutionResult,
    trace: Vec<MachineState>,
    events: Vec<ExecutionTraceEntry>,
}

#[derive(Debug, Clone)]
struct RunCommandOptions {
    program: PathBuf,
    max_steps: usize,
    trace: bool,
    layers: usize,
    engine: CliExecutionEngine,
    verify_native: bool,
    verify_burn: bool,
    verify_onnx: bool,
    verify_all: bool,
    attention_mode: Attention2DMode,
}

#[cfg(feature = "onnx-export")]
struct ScopedTempDir {
    path: PathBuf,
}

#[cfg(feature = "onnx-export")]
impl ScopedTempDir {
    fn new(prefix: &str) -> llm_provable_computer::Result<Self> {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| VmError::InvalidConfig(format!("system clock error: {err}")))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "llm-provable-computer-{prefix}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(feature = "onnx-export")]
impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> llm_provable_computer::Result<()> {
    let cli = Cli::parse_from(normalize_args(std::env::args_os()));
    match cli.command {
        Command::Run {
            program,
            max_steps,
            trace,
            layers,
            engine,
            verify_native,
            verify_burn,
            verify_onnx,
            verify_all,
            attention_mode,
        } => run_program_command(RunCommandOptions {
            program,
            max_steps,
            trace,
            layers,
            engine,
            verify_native,
            verify_burn,
            verify_onnx,
            verify_all,
            attention_mode,
        })?,
        Command::Tui {
            program,
            max_steps,
            layers,
            tick_ms,
            attention_mode,
        } => {
            let mut runtime = load_runtime(&program, max_steps, layers, attention_mode)?;
            run_execution_tui(&program, &mut runtime, Duration::from_millis(tick_ms))?;
        }
        Command::ExportOnnx {
            program,
            output_dir,
            layers,
            attention_mode,
        } => export_onnx_command(&program, &output_dir, layers, attention_mode)?,
        Command::ProveStark {
            program,
            output,
            max_steps,
            layers,
            attention_mode,
        } => prove_stark_command(&program, &output, max_steps, layers, attention_mode)?,
        Command::VerifyStark { proof } => verify_stark_command(&proof)?,
        Command::AggregateProofs { proofs, output } => aggregate_proofs_command(&proofs, &output)?,
        Command::VerifyAggregate { proof } => verify_aggregate_command(&proof)?,
        Command::BatchRun {
            programs,
            count,
            max_steps,
            layers,
            attention_mode,
            metal,
            verify_native,
        } => batch_run_command(
            &programs,
            count,
            max_steps,
            layers,
            attention_mode,
            metal,
            verify_native,
        )?,
        Command::Commit {
            program,
            output,
            max_steps,
            layers,
            attention_mode,
        } => commit_command(&program, &output, max_steps, layers, attention_mode)?,
        Command::Audit {
            program,
            receipt,
            step,
            absolute_epsilon,
            relative_epsilon,
            max_deviating_fraction,
            json,
            layers,
            attention_mode,
        } => audit_command(
            &program,
            &receipt,
            step,
            absolute_epsilon,
            relative_epsilon,
            max_deviating_fraction,
            json,
            layers,
            attention_mode,
        )?,
    }

    Ok(())
}

fn run_program_command(options: RunCommandOptions) -> llm_provable_computer::Result<()> {
    let model = compile_model(
        &options.program,
        options.layers,
        options.attention_mode.clone(),
    )?;
    let executed = execute_engine(&model, options.engine, options.max_steps)?;

    print_execution_summary(&options.program, options.engine, &model, &executed.result);

    if options.verify_native {
        let comparison = verify_model_against_native(model.clone(), options.max_steps)?;
        println!("verified_against_native: true");
        println!("verified_steps: {}", comparison.checked_steps);
        println!(
            "native_elapsed_ms: {:.3}",
            comparison.native.elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "native_throughput_steps_per_sec: {:.2}",
            comparison.native.tokens_per_sec
        );
    }

    if options.verify_burn {
        let verification = verify_burn_engines(&model, options.max_steps)?;
        print_verification_summary(
            "verified_against_burn",
            "verified_burn",
            &verification.checked_steps.to_string(),
            &verification.engines,
        );
    }

    if options.verify_onnx {
        let verification = verify_onnx_engines(&model, options.max_steps)?;
        print_verification_summary(
            "verified_against_onnx",
            "verified_onnx",
            &verification.checked_steps.to_string(),
            &verification.engines,
        );
    }

    if options.verify_all {
        let verification = verify_all_engines(&model, options.max_steps)?;
        print_verification_summary(
            "verified_all",
            "verified_all",
            &verification.checked_steps.to_string(),
            &verification.engines,
        );
    }

    if options.trace {
        print_trace(&executed.trace, &executed.events);
    }

    Ok(())
}

fn export_onnx_command(
    program: &Path,
    output_dir: &Path,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<()> {
    let model = compile_model(program, layers, attention_mode)?;
    export_onnx_command_impl(program, output_dir, &model)
}

fn prove_stark_command(
    program: &Path,
    output: &Path,
    max_steps: usize,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<()> {
    let model = compile_model(program, layers, attention_mode)?;
    let proof = prove_execution_stark(&model, max_steps)?;
    save_execution_stark_proof(&proof, output)?;

    println!("program: {}", program.display());
    println!("proof: {}", output.display());
    println!("steps: {}", proof.claim.steps);
    println!("halted: {}", proof.claim.final_state.halted);
    println!("pc: {}", proof.claim.final_state.pc);
    println!("sp: {}", proof.claim.final_state.sp);
    println!("acc: {}", proof.claim.final_state.acc);
    println!("zero_flag: {}", proof.claim.final_state.zero_flag);
    println!("carry_flag: {}", proof.claim.final_state.carry_flag);
    println!("memory: {:?}", proof.claim.final_state.memory);
    println!("attention_mode: {}", proof.claim.attention_mode);
    println!("proof_bytes: {}", proof.proof.len());

    Ok(())
}

fn verify_stark_command(proof_path: &Path) -> llm_provable_computer::Result<()> {
    let proof = load_execution_stark_proof(proof_path)?;
    if !verify_execution_stark(&proof)? {
        return Err(VmError::InvalidConfig(format!(
            "stark proof verification failed for {}",
            proof_path.display()
        )));
    }

    println!("proof: {}", proof_path.display());
    println!("verified_stark: true");
    println!("steps: {}", proof.claim.steps);
    println!("halted: {}", proof.claim.final_state.halted);
    println!("pc: {}", proof.claim.final_state.pc);
    println!("sp: {}", proof.claim.final_state.sp);
    println!("acc: {}", proof.claim.final_state.acc);
    println!("zero_flag: {}", proof.claim.final_state.zero_flag);
    println!("carry_flag: {}", proof.claim.final_state.carry_flag);
    println!("memory: {:?}", proof.claim.final_state.memory);
    println!("attention_mode: {}", proof.claim.attention_mode);
    println!("instructions: {}", proof.claim.program.instructions().len());
    println!("proof_bytes: {}", proof.proof.len());

    Ok(())
}

fn aggregate_proofs_command(
    proof_paths: &[PathBuf],
    output: &Path,
) -> llm_provable_computer::Result<()> {
    if proof_paths.is_empty() {
        return Err(VmError::InvalidConfig(
            "aggregate-proofs requires at least one proof path".to_string(),
        ));
    }

    let mut proofs = Vec::with_capacity(proof_paths.len());
    for path in proof_paths {
        proofs.push(load_execution_stark_proof(path)?);
    }

    let aggregate = prove_aggregate_verification(&proofs)?;
    let bytes = serde_json::to_vec_pretty(&aggregate)
        .map_err(|error| VmError::Serialization(error.to_string()))?;
    fs::write(output, bytes)?;

    println!("aggregate_proof: {}", output.display());
    println!("proof_count: {}", aggregate.inner_proofs.len());
    println!("accepted: {}", aggregate.accepted);
    println!("aggregate_root: {}", aggregate.aggregate_root);
    Ok(())
}

fn verify_aggregate_command(path: &Path) -> llm_provable_computer::Result<()> {
    let bytes = fs::read(path)?;
    let aggregate: RecursiveAggregateProof = serde_json::from_slice(&bytes)
        .map_err(|error| VmError::Serialization(error.to_string()))?;

    if !verify_aggregate_proof(&aggregate)? {
        return Err(VmError::InvalidConfig(format!(
            "aggregate proof verification failed for {}",
            path.display()
        )));
    }

    println!("aggregate_proof: {}", path.display());
    println!("verified_aggregate: true");
    println!("proof_count: {}", aggregate.inner_proofs.len());
    println!("accepted: {}", aggregate.accepted);
    println!("aggregate_root: {}", aggregate.aggregate_root);
    Ok(())
}

fn commit_command(
    program_path: &Path,
    output_path: &Path,
    max_steps: usize,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<()> {
    let model = compile_model(program_path, layers, attention_mode)?;
    let mut runtime = ExecutionRuntime::new(model.clone(), max_steps);
    let (_result, witness) = runtime.run_with_witness()?;

    let (receipt, _m_trace, _w_commitment) =
        llm_provable_computer::commit_execution(&model, &witness)?;
    save_receipt(&receipt, output_path)?;

    println!("program: {}", program_path.display());
    println!("receipt: {}", output_path.display());
    println!("steps: {}", receipt.step_count);
    println!("halted: {}", receipt.final_state.halted);
    println!("pc: {}", receipt.final_state.pc);
    println!("acc: {}", receipt.final_state.acc);
    println!("trace_root: 0x{}", manual_hex_encode(&receipt.trace_root));
    println!(
        "weights_root: 0x{}",
        manual_hex_encode(&receipt.weights_root)
    );

    Ok(())
}

fn manual_hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[allow(clippy::too_many_arguments)]
fn audit_command(
    program_path: &Path,
    receipt_path: &Path,
    step_idx: Option<usize>,
    absolute_epsilon: Option<f64>,
    relative_epsilon: Option<f64>,
    max_deviating_fraction: Option<f64>,
    json: bool,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<()> {
    let receipt_json = fs::read_to_string(receipt_path).map_err(|e| {
        VmError::InvalidConfig(format!(
            "failed to read receipt {}: {e}",
            receipt_path.display()
        ))
    })?;
    let receipt: llm_provable_computer::ExecutionReceipt = serde_json::from_str(&receipt_json)
        .map_err(|e| VmError::InvalidConfig(format!("failed to parse receipt: {e}")))?;

    let model = compile_model(program_path, layers, attention_mode)?;
    let mut runtime = ExecutionRuntime::new(model.clone(), receipt.step_count);
    let (_result, witness) = runtime.run_with_witness()?;
    if !llm_provable_computer::verify_receipt_bindings(&receipt, &model, &witness.initial_state)? {
        return Err(VmError::InvalidConfig(format!(
            "receipt {} does not match the compiled program/config or initial state",
            receipt_path.display()
        )));
    }

    let staged_trace = llm_provable_computer::build_shared_transformer_staged_trace(
        &witness,
        model.config().ff_dim,
    )?;
    let m_trace = llm_provable_computer::merkle_trace::MerkleizedTrace::new(&staged_trace.rows)?;
    let w_commitment = llm_provable_computer::WeightCommitment::new(&model)?;
    let approximate_config =
        build_approximate_config(absolute_epsilon, relative_epsilon, max_deviating_fraction)?;

    let target_step = step_idx.unwrap_or(0);
    let request = llm_provable_computer::AuditRequest {
        target: llm_provable_computer::AuditTarget::Steps(vec![target_step]),
        challenge: None,
        approximate_config,
    };

    let response = llm_provable_computer::audit_execution(
        &model,
        &m_trace,
        &w_commitment,
        &witness,
        &staged_trace,
        &request,
    )?;

    if let Some(config) = request.approximate_config.as_ref() {
        let report = llm_provable_computer::verify_audit_with_config(
            &receipt, &response, &witness, config, None,
        )?;
        if !report.passed {
            return Err(VmError::InvalidConfig(format!(
                "approximate audit verification failed for receipt {}",
                receipt_path.display()
            )));
        }

        if json {
            let payload = if report.approximate_comparisons.len() == 1 {
                serde_json::to_string_pretty(&report.approximate_comparisons[0])
            } else {
                serde_json::to_string_pretty(&report.approximate_comparisons)
            }
            .map_err(|error| VmError::Serialization(error.to_string()))?;
            println!("{payload}");
            return Ok(());
        }

        println!("receipt: {}", receipt_path.display());
        println!("audit_step: {}", target_step);
        println!("verified_audit: true");
        println!("approximate_mode: true");
        for comparison in &report.approximate_comparisons {
            println!("total_values: {}", comparison.total_values);
            println!("deviating_values: {}", comparison.deviating_values);
            println!("max_absolute_error: {}", comparison.max_absolute_error);
            println!("max_relative_error: {}", comparison.max_relative_error);
        }
    } else if !llm_provable_computer::verify_audit(&receipt, &response, None) {
        return Err(VmError::InvalidConfig(format!(
            "audit verification failed for receipt {}",
            receipt_path.display()
        )));
    } else {
        println!("receipt: {}", receipt_path.display());
        println!("audit_step: {}", target_step);
        println!("verified_audit: true");
        println!("row_type: {:?}", response.opened_steps[0].row.stage());
    }

    Ok(())
}

fn build_approximate_config(
    absolute_epsilon: Option<f64>,
    relative_epsilon: Option<f64>,
    max_deviating_fraction: Option<f64>,
) -> llm_provable_computer::Result<Option<ApproximateVerificationConfig>> {
    if absolute_epsilon.is_none() && relative_epsilon.is_none() && max_deviating_fraction.is_none()
    {
        return Ok(None);
    }

    let config = ApproximateVerificationConfig {
        absolute_epsilon: absolute_epsilon.unwrap_or(1e-6),
        relative_epsilon: relative_epsilon.unwrap_or(1e-6),
        max_deviating_fraction: max_deviating_fraction.unwrap_or(0.0),
    };
    config.validate().map_err(|error| {
        VmError::InvalidConfig(format!("invalid approximate audit configuration: {error}"))
    })?;
    Ok(Some(config))
}

fn save_receipt(
    receipt: &llm_provable_computer::ExecutionReceipt,
    path: &Path,
) -> llm_provable_computer::Result<()> {
    let json = serde_json::to_string_pretty(receipt)
        .map_err(|e| VmError::Serialization(format!("failed to serialize receipt: {e}")))?;
    fs::write(path, json).map_err(|e| {
        VmError::InvalidConfig(format!(
            "failed to write receipt to {}: {e}",
            path.display()
        ))
    })?;
    Ok(())
}

fn batch_run_command(
    program_paths: &[PathBuf],
    count: usize,
    max_steps: usize,
    layers: usize,
    attention_mode: Attention2DMode,
    metal: bool,
    verify_native: bool,
) -> llm_provable_computer::Result<()> {
    use llm_provable_computer::batch::{BatchRunResult, BatchRunner};
    use std::sync::Arc;

    let config = TransformerVmConfig {
        num_layers: layers,
        attention_mode,
        ..TransformerVmConfig::default()
    };

    let mut sources = Vec::new();
    for p in program_paths {
        let s = fs::read_to_string(p).map_err(|io_error| {
            VmError::InvalidConfig(format!(
                "failed to read program {}: {io_error}",
                p.display()
            ))
        })?;
        sources.push(s);
    }
    let sources_str: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();

    let (model_inner, entry_points) = ProgramCompiler.compile_sources(&sources_str, config)?;
    let model = Arc::new(model_inner);
    let initial_memory = model.program().initial_memory().to_vec();

    // Generate `count` initial states distributed across the programs.
    let num_progs = program_paths.len();
    let initial_states: Vec<MachineState> = (0..count)
        .map(|i| {
            let prog_idx = i % num_progs;
            let mut s = MachineState::with_memory(initial_memory.clone());
            s.pc = entry_points[prog_idx];
            s.acc = (i / num_progs) as i64;
            s
        })
        .collect();

    if program_paths.len() == 1 {
        println!("program: {}", program_paths[0].display());
    } else {
        println!("programs: {} total", program_paths.len());
        for (i, p) in program_paths.iter().enumerate() {
            println!("  [{i}] {} (entry pc={})", p.display(), entry_points[i]);
        }
    }
    println!("backend: {}", if metal { "metal" } else { "cpu-rayon" });
    println!("batch_size: {count}");
    println!("max_steps: {max_steps}");

    // Collect (run_idx, final_state) pairs for optional native verification.
    let mut batch_finals: Vec<(usize, MachineState)> = Vec::new();

    if metal {
        #[cfg(all(feature = "metal-batch", target_os = "macos"))]
        {
            use llm_provable_computer::metal_batch::MetalBatchRunner;
            let runner = MetalBatchRunner::new(Arc::clone(&model), max_steps)?;
            let t0 = std::time::Instant::now();
            let results = runner.run(initial_states.clone());
            let elapsed = t0.elapsed();
            let halted = results.iter().filter(|r| r.final_state.halted).count();
            let errors = results.iter().filter(|r| r.result.is_err()).count();
            let total_steps: usize = results
                .iter()
                .filter_map(|r| r.result.as_ref().ok())
                .map(|r| r.steps)
                .sum();
            let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
            let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
            println!("halted_runs: {halted}");
            println!("error_runs: {errors}");
            println!("total_steps: {total_steps}");
            println!("elapsed_ms: {elapsed_ms:.1}");
            println!("steps_per_sec: {steps_per_sec:.0}");
            for r in results {
                batch_finals.push((r.run_idx, r.final_state));
            }
        }
        #[cfg(not(all(feature = "metal-batch", target_os = "macos")))]
        {
            return Err(VmError::InvalidConfig(
                "--metal requires the `metal-batch` feature on macOS".to_string(),
            ));
        }
    } else {
        let runner = BatchRunner::new(Arc::clone(&model), max_steps);
        let t0 = std::time::Instant::now();
        let results: Vec<BatchRunResult> = runner.run(initial_states.clone());
        let elapsed = t0.elapsed();
        let halted = results.iter().filter(|r| r.final_state.halted).count();
        let errors = results.iter().filter(|r| r.result.is_err()).count();
        let total_steps: usize = results
            .iter()
            .filter_map(|r| r.result.as_ref().ok())
            .map(|r| r.steps)
            .sum();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
        println!("halted_runs: {halted}");
        println!("error_runs: {errors}");
        println!("total_steps: {total_steps}");
        println!("elapsed_ms: {elapsed_ms:.1}");
        println!("steps_per_sec: {steps_per_sec:.0}");
        for r in results {
            batch_finals.push((r.run_idx, r.final_state));
        }
    }

    // ── Native verification ────────────────────────────────────────────────
    if verify_native && !batch_finals.is_empty() {
        use llm_provable_computer::NativeInterpreter;

        println!("verifying against native interpreter…");
        let mut mismatches = 0usize;
        for (run_idx, batch_state) in &batch_finals {
            let init = &initial_states[*run_idx];
            let mut native = NativeInterpreter::new_with_state(
                model.program().clone(),
                model.config().attention_mode.clone(),
                init.clone(),
                max_steps,
            );
            let native_result = native.run().expect("native run");
            let native_state = &native_result.final_state;
            if batch_state.acc != native_state.acc || batch_state.halted != native_state.halted {
                mismatches += 1;
                if mismatches <= 10 {
                    println!(
                        "  MISMATCH run {run_idx}: batch acc={} halted={} | native acc={} halted={}",
                        batch_state.acc,
                        batch_state.halted,
                        native_state.acc,
                        native_state.halted,
                    );
                }
            }
        }
        if mismatches == 0 {
            println!("native_mismatches: 0  ✓ all {} results match", batch_finals.len());
        } else {
            println!("native_mismatches: {mismatches}");
            return Err(VmError::InvalidConfig(format!(
                "{mismatches} run(s) produced results that diverge from the native interpreter"
            )));
        }
    }

    Ok(())
}

fn compile_model(
    program: &Path,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<TransformerVm> {
    let source = fs::read_to_string(program).map_err(|io_error| {
        VmError::InvalidConfig(format!(
            "failed to read program {}: {io_error}",
            program.display()
        ))
    })?;

    let config = TransformerVmConfig {
        num_layers: layers,
        attention_mode,
        ..TransformerVmConfig::default()
    };
    ProgramCompiler.compile_source(&source, config)
}

fn load_runtime(
    program: &Path,
    max_steps: usize,
    layers: usize,
    attention_mode: Attention2DMode,
) -> llm_provable_computer::Result<ExecutionRuntime> {
    Ok(ExecutionRuntime::new(
        compile_model(program, layers, attention_mode)?,
        max_steps,
    ))
}

fn execute_engine(
    model: &TransformerVm,
    engine: CliExecutionEngine,
    max_steps: usize,
) -> llm_provable_computer::Result<EngineRunOutput> {
    match engine {
        CliExecutionEngine::Transformer => {
            let mut runtime = ExecutionRuntime::new(model.clone(), max_steps);
            let result = runtime.run()?;
            Ok(EngineRunOutput {
                result,
                trace: runtime.trace().to_vec(),
                events: runtime.events().to_vec(),
            })
        }
        CliExecutionEngine::Native => {
            let mut runtime = NativeInterpreter::new(
                model.program().clone(),
                model.config().attention_mode.clone(),
                max_steps,
            );
            let result = runtime.run()?;
            Ok(EngineRunOutput {
                result,
                trace: runtime.trace().to_vec(),
                events: runtime.events().to_vec(),
            })
        }
        CliExecutionEngine::Burn => execute_burn_engine(model, max_steps),
        CliExecutionEngine::Onnx => execute_onnx_engine(model, max_steps),
    }
}

#[cfg(feature = "burn-model")]
fn execute_burn_engine(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<EngineRunOutput> {
    let device = Default::default();
    let burn_model = BurnTransformerVm::<CliBurnBackend>::from_compiled(model, &device)?;
    let mut runtime = BurnExecutionRuntime::new(burn_model, device, max_steps);
    let result = runtime.run()?;
    Ok(EngineRunOutput {
        result,
        trace: runtime.trace().to_vec(),
        events: runtime.events().to_vec(),
    })
}

#[cfg(not(feature = "burn-model"))]
fn execute_burn_engine(
    _model: &TransformerVm,
    _max_steps: usize,
) -> llm_provable_computer::Result<EngineRunOutput> {
    Err(feature_required_error("engine `burn`", &["burn-model"]))
}

#[cfg(feature = "onnx-export")]
fn execute_onnx_engine(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<EngineRunOutput> {
    let export_dir = ScopedTempDir::new("run-onnx")?;
    export_program_onnx(model, export_dir.path())?;
    let mut runtime = OnnxExecutionRuntime::from_export_dir(export_dir.path(), max_steps)?;
    let result = runtime.run()?;
    Ok(EngineRunOutput {
        result,
        trace: runtime.trace().to_vec(),
        events: runtime.events().to_vec(),
    })
}

#[cfg(not(feature = "onnx-export"))]
fn execute_onnx_engine(
    _model: &TransformerVm,
    _max_steps: usize,
) -> llm_provable_computer::Result<EngineRunOutput> {
    Err(feature_required_error("engine `onnx`", &["onnx-export"]))
}

fn verify_burn_engines(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    verify_burn_engines_impl(model, max_steps)
}

#[cfg(feature = "burn-model")]
fn verify_burn_engines_impl(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    let device = Default::default();
    let burn_model = BurnTransformerVm::<CliBurnBackend>::from_compiled(model, &device)?;
    let mut transformer = ExecutionRuntime::new(model.clone(), max_steps);
    let mut native = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        max_steps,
    );
    let mut burn = BurnExecutionRuntime::new(burn_model, device, max_steps);
    verify_engines(&mut [&mut transformer, &mut native, &mut burn])
}

#[cfg(not(feature = "burn-model"))]
fn verify_burn_engines_impl(
    _model: &TransformerVm,
    _max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    Err(feature_required_error("`--verify-burn`", &["burn-model"]))
}

fn verify_onnx_engines(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    verify_onnx_engines_impl(model, max_steps)
}

#[cfg(feature = "onnx-export")]
fn verify_onnx_engines_impl(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    let export_dir = ScopedTempDir::new("verify-onnx")?;
    export_program_onnx(model, export_dir.path())?;
    let mut transformer = ExecutionRuntime::new(model.clone(), max_steps);
    let mut native = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        max_steps,
    );
    let mut onnx = OnnxExecutionRuntime::from_export_dir(export_dir.path(), max_steps)?;
    verify_engines(&mut [&mut transformer, &mut native, &mut onnx])
}

#[cfg(not(feature = "onnx-export"))]
fn verify_onnx_engines_impl(
    _model: &TransformerVm,
    _max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    Err(feature_required_error("`--verify-onnx`", &["onnx-export"]))
}

fn verify_all_engines(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    verify_all_engines_impl(model, max_steps)
}

#[cfg(all(feature = "burn-model", feature = "onnx-export"))]
fn verify_all_engines_impl(
    model: &TransformerVm,
    max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    let device = Default::default();
    let export_dir = ScopedTempDir::new("verify-all")?;
    export_program_onnx(model, export_dir.path())?;
    let burn_model = BurnTransformerVm::<CliBurnBackend>::from_compiled(model, &device)?;

    let mut transformer = ExecutionRuntime::new(model.clone(), max_steps);
    let mut native = NativeInterpreter::new(
        model.program().clone(),
        model.config().attention_mode.clone(),
        max_steps,
    );
    let mut burn = BurnExecutionRuntime::new(burn_model, device, max_steps);
    let mut onnx = OnnxExecutionRuntime::from_export_dir(export_dir.path(), max_steps)?;
    verify_engines(&mut [&mut transformer, &mut native, &mut burn, &mut onnx])
}

#[cfg(not(all(feature = "burn-model", feature = "onnx-export")))]
fn verify_all_engines_impl(
    _model: &TransformerVm,
    _max_steps: usize,
) -> llm_provable_computer::Result<llm_provable_computer::VerificationResult> {
    Err(feature_required_error(
        "`--verify-all`",
        &["burn-model", "onnx-export"],
    ))
}

#[cfg(feature = "onnx-export")]
fn export_onnx_command_impl(
    program: &Path,
    output_dir: &Path,
    model: &TransformerVm,
) -> llm_provable_computer::Result<()> {
    let metadata = export_program_onnx(model, output_dir)?;

    println!("program: {}", program.display());
    println!("output_dir: {}", output_dir.display());
    println!("instructions: {}", metadata.instructions.len());
    println!("layers: {}", metadata.config.num_layers);
    println!("attention_mode: {}", metadata.config.attention_mode);
    println!("onnx_ir_version: {}", metadata.ir_version);
    println!("onnx_opset_version: {}", metadata.opset_version);
    println!("metadata: {}", output_dir.join("metadata.json").display());

    Ok(())
}

#[cfg(not(feature = "onnx-export"))]
fn export_onnx_command_impl(
    _program: &Path,
    _output_dir: &Path,
    _model: &TransformerVm,
) -> llm_provable_computer::Result<()> {
    Err(feature_required_error("`export-onnx`", &["onnx-export"]))
}

fn print_execution_summary(
    program: &Path,
    engine: CliExecutionEngine,
    model: &TransformerVm,
    result: &ExecutionResult,
) {
    println!("program: {}", program.display());
    println!("engine: {engine}");
    println!("steps: {}", result.steps);
    println!("halted: {}", result.halted);
    println!("pc: {}", result.final_state.pc);
    println!("sp: {}", result.final_state.sp);
    println!("acc: {}", result.final_state.acc);
    println!("zero_flag: {}", result.final_state.zero_flag);
    println!("carry_flag: {}", result.final_state.carry_flag);
    println!("memory: {:?}", result.final_state.memory);
    println!("layers: {}", model.config().num_layers);
    println!("attention_mode: {}", model.config().attention_mode);
    println!("elapsed_ms: {:.3}", result.elapsed.as_secs_f64() * 1000.0);
    println!("throughput_steps_per_sec: {:.2}", result.tokens_per_sec);
}

fn print_verification_summary(
    status_key: &str,
    prefix: &str,
    checked_steps: &str,
    engines: &[llm_provable_computer::VerifiedEngine],
) {
    println!("{status_key}: true");
    println!("{prefix}_steps: {checked_steps}");
    println!(
        "{prefix}_engines: {}",
        engines
            .iter()
            .map(|engine| normalize_engine_name(&engine.name))
            .collect::<Vec<_>>()
            .join(",")
    );
}

fn print_trace(trace: &[MachineState], events: &[ExecutionTraceEntry]) {
    if let Some(initial) = trace.first() {
        println!(
            "trace[000] init pc={} sp={} acc={} zero={} carry={} halted={} memory={:?}",
            initial.pc,
            initial.sp,
            initial.acc,
            initial.zero_flag,
            initial.carry_flag,
            initial.halted,
            initial.memory
        );
    }

    for event in events {
        println!(
            "trace[{step:03}] layer={layer} instr=\"{instr}\" pc={pc} sp={sp} acc={acc} zero={zero} carry={carry} halted={halted} memory={memory:?}",
            step = event.step,
            layer = event.layer_idx.unwrap_or(0),
            instr = event.instruction,
            pc = event.state_after.pc,
            sp = event.state_after.sp,
            acc = event.state_after.acc,
            zero = event.state_after.zero_flag,
            carry = event.state_after.carry_flag,
            halted = event.state_after.halted,
            memory = event.state_after.memory
        );
    }
}

fn normalize_args<I>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = OsString>,
{
    let mut args = args.into_iter().collect::<Vec<_>>();
    let should_insert_run = args
        .get(1)
        .and_then(|arg| arg.to_str())
        .map(needs_run_subcommand)
        .unwrap_or(false);

    if should_insert_run {
        args.insert(1, OsString::from("run"));
    }

    args
}

fn needs_run_subcommand(first_arg: &str) -> bool {
    !first_arg.starts_with('-')
        && !matches!(
            first_arg,
            "run"
                | "tui"
                | "export-onnx"
                | "prove-stark"
                | "verify-stark"
                | "aggregate-proofs"
                | "verify-aggregate"
                | "commit"
                | "audit"
                | "batch-run"
                | "help"
        )
}

fn parse_attention_mode(input: &str) -> Result<Attention2DMode, String> {
    Attention2DMode::from_str(input)
}

fn parse_execution_engine(input: &str) -> Result<CliExecutionEngine, String> {
    let normalized = input.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "native" => Ok(CliExecutionEngine::Native),
        "transformer" => Ok(CliExecutionEngine::Transformer),
        "burn" => Ok(CliExecutionEngine::Burn),
        "onnx" | "onnx-tract" | "onnx/tract" => Ok(CliExecutionEngine::Onnx),
        _ => Err(format!(
            "unknown execution engine `{input}`; expected native, transformer, burn, or onnx"
        )),
    }
}

fn normalize_engine_name(name: &str) -> String {
    match name {
        "onnx/tract" => "onnx".to_string(),
        other => other.to_string(),
    }
}

#[cfg(not(all(feature = "burn-model", feature = "onnx-export")))]
fn feature_required_error(subject: &str, features: &[&str]) -> VmError {
    VmError::InvalidConfig(format!(
        "{subject} requires {}",
        required_features_message(features)
    ))
}

#[cfg(not(all(feature = "burn-model", feature = "onnx-export")))]
fn required_features_message(features: &[&str]) -> String {
    if features.len() == 1 {
        format!("the `{}` feature", features[0])
    } else {
        let joined = features
            .iter()
            .map(|feature| format!("`{feature}`"))
            .collect::<Vec<_>>()
            .join(" and ");
        format!("the {joined} features")
    }
}
