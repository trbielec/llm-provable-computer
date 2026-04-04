use std::fs;

use llm_provable_computer::{compile_cdmn_table_with_metadata, CdmnTable};

fn curated_examples() -> [&'static str; 9] {
    [
        "programs/budget_constraints.cdmn.json",
        "programs/ratio_constraints.cdmn.json",
        "programs/medication_safety.cdmn.json",
        "programs/lab_vitals_guardrail.cdmn.json",
        "programs/prior_authorization_imaging.cdmn.json",
        "programs/trial_eligibility_screen.cdmn.json",
        "programs/loan_affordability_underwriting.cdmn.json",
        "programs/insurance_claim_line_review.cdmn.json",
        "programs/vendor_due_diligence.cdmn.json",
    ]
}

#[test]
fn curated_cdmn_examples_compile() {
    for path in curated_examples() {
        let source = fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
        let table: CdmnTable =
            serde_json::from_str(&source).unwrap_or_else(|err| panic!("{path}: {err}"));
        let compiled =
            compile_cdmn_table_with_metadata(&table).unwrap_or_else(|err| panic!("{path}: {err}"));
        assert!(
            !compiled.assembly.is_empty(),
            "{path}: expected generated assembly"
        );
        assert!(
            !compiled.reasons.is_empty(),
            "{path}: expected at least one reason entry"
        );
    }
}
