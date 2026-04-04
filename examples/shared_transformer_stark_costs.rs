use llm_provable_computer::{
    estimate_shared_transformer_staged_skeleton_stark_cost, ProgramCompiler, TransformerVmConfig,
    VanillaStarkProofOptions,
};

fn main() {
    let options = VanillaStarkProofOptions::default();

    for (label, source, max_steps, ff_chunk_size) in [
        (
            "average_hard_add_store",
            r#"
            .memory 3
            .init 0 11
            .init 1 4
            LOAD 0
            ADD 1
            STORE 2
            HALT
        "#,
            16usize,
            32usize,
        ),
        (
            "compare_branch",
            r#"
            .memory 1
            LOADI 5
            CMP 5
            JZ done
            LOADI 99
        done:
            HALT
        "#,
            16usize,
            32usize,
        ),
    ] {
        let model = ProgramCompiler
            .compile_source(source, TransformerVmConfig::default())
            .expect("compile");
        let estimate = estimate_shared_transformer_staged_skeleton_stark_cost(
            &model,
            max_steps,
            ff_chunk_size,
            options.clone(),
        )
        .expect("estimate shared transformer cost");

        println!(
            "{label}\n  steps={} rows={} regs={} mem={} d_model={} ff_chunk_rows={} linear_feature_chunk_rows={} hidden_chunk_width={}\n  width: memory={} tokens={} input_features={} chunk_values={} output_acc={} linear_acc={} output_slot_acc={} compare={} raw_acc={}\n  constraints={} degree={} randomizers={} omicron={} fri={}",
            estimate.steps,
            estimate.trace_rows,
            estimate.register_count,
            estimate.memory_size,
            estimate.d_model,
            estimate.ff_chunk_rows,
            estimate.linear_feature_chunk_rows,
            estimate.hidden_chunk_width,
            estimate.memory_column_count,
            estimate.token_column_count,
            estimate.input_feature_column_count,
            estimate.chunk_value_column_count,
            estimate.output_accumulator_column_count,
            estimate.linear_accumulator_column_count,
            estimate.output_slot_accumulator_column_count,
            estimate.compare_column_count,
            estimate.raw_acc_column_count,
            estimate.transition_constraint_count,
            estimate.transition_degree_bound,
            estimate.num_randomizers,
            estimate.omicron_domain_length,
            estimate.fri_domain_length,
        );
    }
}
