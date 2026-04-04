use llm_provable_computer::{
    compile_cdmn_table_to_decision_program, estimate_compiled_cdmn_decision_stark_direct_cost,
    estimate_execution_stark_cost, CdmnTable, ProgramCompiler, TransformerVm, TransformerVmConfig,
    VanillaStarkProofOptions,
};

fn main() {
    let options = VanillaStarkProofOptions::default();

    for (label, json, memory_updates, max_steps) in [
        (
            "medication_safety",
            include_str!("../programs/medication_safety.cdmn.json"),
            vec![
                (0usize, 500i64),
                (1, 11),
                (2, 12),
                (3, 99),
                (4, 2),
                (5, 400),
                (6, 11),
                (7, 450),
                (8, 12),
            ],
            1_000usize,
        ),
        (
            "lab_vitals_guardrail",
            include_str!("../programs/lab_vitals_guardrail.cdmn.json"),
            vec![
                (0usize, 8_000i64),
                (1, 4_000),
                (2, 11_000),
                (3, 120),
                (4, 80),
                (5, 30),
                (6, 60),
            ],
            1_000usize,
        ),
    ] {
        let table: CdmnTable = serde_json::from_str(json).expect("fixture");
        let decision_program =
            compile_cdmn_table_to_decision_program(&table).expect("compile decision program");
        let direct =
            estimate_compiled_cdmn_decision_stark_direct_cost(&decision_program, options.clone())
                .expect("direct cost");

        let compiled =
            llm_provable_computer::compile_cdmn_table_with_metadata(&table).expect("compile tvm");
        let config = TransformerVmConfig::default();
        let compiled_program = ProgramCompiler::default()
            .compile_source(&compiled.assembly, config.clone())
            .expect("compile source");
        let mut initial_memory = compiled_program.program().initial_memory().to_vec();
        for (addr, value) in memory_updates {
            initial_memory[addr] = value;
        }
        let program = compiled_program
            .program()
            .clone()
            .with_initial_memory(initial_memory)
            .expect("initial memory");
        let model = TransformerVm::new(config, program).expect("vm");
        let vm =
            estimate_execution_stark_cost(&model, max_steps, options.clone()).expect("vm cost");

        println!(
            "{label}\n  direct: rows={} regs={} constraints={} degree={} fri={} quantified_nodes={} quantified_items={} equality_aux={} ordered_aux={}\n  vm: rows={} regs={} constraints={} degree={} fri={}",
            direct.trace_rows,
            direct.register_count,
            direct.transition_constraint_count,
            direct.transition_degree_bound,
            direct.fri_domain_length,
            direct.quantified_node_count,
            direct.quantified_item_predicate_count,
            direct.equality_aux_count,
            direct.ordered_aux_count,
            vm.trace_rows,
            vm.trace_width,
            vm.transition_constraint_count,
            vm.transition_degree_bound,
            vm.fri_domain_length,
        );
    }
}
