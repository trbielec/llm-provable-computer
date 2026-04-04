use std::fs;

use llm_provable_computer::{compile_cdmn_table_with_metadata, CdmnTable};

fn main() {
    let cdmn_json =
        fs::read_to_string("programs/ratio_constraints.cdmn.json").expect("Read failed");
    let table: CdmnTable = serde_json::from_str(&cdmn_json).expect("Parse failed");

    let compiled = compile_cdmn_table_with_metadata(&table).expect("Compile failed");
    fs::write("programs/budget_verifier.tvm", &compiled.assembly).expect("Write failed");
    let metadata =
        serde_json::to_vec_pretty(&compiled.metadata()).expect("Failed to serialize cDMN metadata");
    fs::write("programs/budget_verifier.metadata.json", metadata).expect("Write failed");
    println!("Compiled cDMN logic to programs/budget_verifier.tvm with metadata sidecar");
}
