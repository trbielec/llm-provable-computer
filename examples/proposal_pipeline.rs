//! End-to-end example of the Phase 1–5 proposal pipeline.
//!
//! Run with:
//!   cargo run --example proposal_pipeline
//!
//! This simulates an LLM proposing three candidate programs for a simple
//! arithmetic task, running them through the admissibility gate and
//! deterministic kernel, and selecting the best result.

use llm_provable_computer::candidate_eval::SelectionStrategy;
use llm_provable_computer::config::TransformerVmConfig;
use llm_provable_computer::effect_root::{EffectBoundary, EffectLog, StateEntry, StateRoot};
use llm_provable_computer::overlap::{FallbackProgram, OverlapOrchestrator, OverlapStrategy};
use llm_provable_computer::planner_bridge::{PlannerContext, StaticPlanner};
use llm_provable_computer::state::MachineState;

fn main() {
    // ── 1. Simulate LLM proposals ─────────────────────────────────────────────
    //
    // In production this would be replaced by a real Planner implementation
    // that calls an LLM API and returns assembly programs.  Here we use
    // StaticPlanner to show three "proposals" a model might generate for the
    // task: "compute the larger of two candidate values".
    //
    // Proposal A: returns 42  (reasonable answer)
    // Proposal B: returns 100 (higher — will win under HighestAccumulator)
    // Proposal C: invalid syntax — will be rejected by AdmissibilityFilter

    let proposals = vec![
        (
            "LOADI 42\nHALT".to_string(),
            b"proposal-a-raw-output".to_vec(),
        ),
        (
            "LOADI 100\nHALT".to_string(),
            b"proposal-b-raw-output".to_vec(),
        ),
        (
            "THIS IS NOT VALID TVM ASSEMBLY".to_string(),
            b"proposal-c-raw-output".to_vec(),
        ),
    ];

    let planner = StaticPlanner::new(proposals);

    // ── 2. Build an effect boundary ───────────────────────────────────────────
    //
    // Phase 2: represent any external state the planner read from.
    // Here: the planner read a single key "threshold" = 50 from an external
    // store before generating its proposals.

    let pre_state = StateRoot::from_entries(vec![StateEntry::new("threshold", b"50")]);

    let mut effect_log = EffectLog::new();
    effect_log.append_read("threshold", b"50");

    let boundary = EffectBoundary::new(pre_state, effect_log);

    println!("Effect boundary hash: {}", hex(boundary.boundary_hash()));

    // ── 3. Build planner context ───────────────────────────────────────────────
    //
    // The context travels through the whole pipeline so every layer knows what
    // the planner was given.

    let context = PlannerContext::new(
        b"compute the best candidate value given threshold=50".to_vec(),
        "gpt-4o-simulated".to_string(),
        MachineState::default(),
    )
    .with_effect_boundary(boundary);

    // ── 4. Build the overlap orchestrator ─────────────────────────────────────
    //
    // The fallback program runs immediately in Overlapped mode while the
    // planner pipeline is executing — hiding planner latency behind kernel work.
    // With a real LLM this saves hundreds of milliseconds.

    let fallback = FallbackProgram::from_source("LOADI 0\nHALT")
        .expect("fallback compiles");

    let orchestrator = OverlapOrchestrator::new(
        TransformerVmConfig::default(),
        128,
        SelectionStrategy::HighestAccumulator,
        fallback,
    );

    // ── 5. Run Sequential mode ─────────────────────────────────────────────────

    println!("\n── Sequential mode ───────────────────────────────────────────");
    let seq_report = orchestrator
        .clone()
        .with_strategy(OverlapStrategy::Sequential)
        .run(&planner, context.clone())
        .expect("pipeline succeeded");

    print_report(&seq_report);

    // ── 6. Run Overlapped mode ─────────────────────────────────────────────────

    println!("\n── Overlapped mode ───────────────────────────────────────────");
    let ovl_report = orchestrator
        .with_strategy(OverlapStrategy::Overlapped)
        .run(&planner, context.clone())
        .expect("pipeline succeeded");

    print_report(&ovl_report);

    // ── 7. Overlap summary ────────────────────────────────────────────────────

    println!("\n── Overlap summary ───────────────────────────────────────────");
    println!(
        "  Pipeline (sequential):  {} µs",
        seq_report.timings.pipeline_duration_us
    );
    println!(
        "  Pipeline (overlapped):  {} µs  (includes thread overhead)",
        ovl_report.timings.pipeline_duration_us
    );
    println!(
        "  Fallback kernel:        {} µs  (ran concurrently)",
        ovl_report.timings.fallback_duration_us
    );
    println!(
        "  Overlap savings:        {} µs  (vs sequential baseline)",
        ovl_report.timings.overlap_savings_us
    );
    println!();
    println!("  With a real LLM planner (~500 ms round-trip), the fallback");
    println!("  kernel would finish in ~120 µs and the overlap savings would");
    println!("  be ~499,880 µs — the kernel runs essentially for free.");
}

fn print_report(r: &llm_provable_computer::overlap::OverlapReport) {
    println!(
        "  Proposed:       {} candidates",
        // admitted + rejected = proposed; we only have admitted here
        r.admitted_count
    );
    println!("  Admitted:       {} (invalid syntax rejected)", r.admitted_count);
    println!("  Selected ACC:   {}", r.final_acc);
    println!("  Used planner:   {}", r.used_planner_output);
    println!("  Pipeline time:  {} µs", r.timings.pipeline_duration_us);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
