use std::fs;

use llm_provable_computer::{compile_dmn_table_with_metadata, DmnTable};

fn main() {
    let dmn_json = fs::read_to_string("programs/loan_rules.dmn.json").expect("Read failed");
    let table: DmnTable = serde_json::from_str(&dmn_json).expect("Parse failed");

    let compiled = compile_dmn_table_with_metadata(&table).expect("Compile failed");
    fs::write("programs/loan_rules.tvm", &compiled.assembly).expect("Write failed");
    let metadata = serde_json::to_vec_pretty(&compiled).expect("Failed to serialize DMN metadata");
    fs::write("programs/loan_rules.metadata.json", metadata).expect("Write failed");
    println!("Compiled DMN to programs/loan_rules.tvm with metadata sidecar");
}
