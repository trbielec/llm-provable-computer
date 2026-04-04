# cDMN Example Pack

Curated cDMN examples that compile against the current frontend/backend.

## Healthcare

- [medication_safety.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/medication_safety.cdmn.json)
  Quantifiers over collections, `forall`, `exists`, `or`, `not`, reason labels.
- [lab_vitals_guardrail.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/lab_vitals_guardrail.cdmn.json)
  Derived definitions plus inclusive `between` ranges.
- [prior_authorization_imaging.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/prior_authorization_imaging.cdmn.json)
  Prior auth style gating with `or` logic and an explicit insufficient-data reason.
- [trial_eligibility_screen.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/trial_eligibility_screen.cdmn.json)
  Eligibility windows, scalar thresholds, and quantifiers over diagnoses and therapies.

## Finance

- [budget_constraints.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/budget_constraints.cdmn.json)
  Basic arithmetic comparisons and exact-ratio checks.
- [ratio_constraints.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/ratio_constraints.cdmn.json)
  Cross-multiplied ratio guardrail using `>=`.
- [loan_affordability_underwriting.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/loan_affordability_underwriting.cdmn.json)
  Definitions, threshold checks, and insufficient-data semantics.

## Insurance

- [insurance_claim_line_review.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/insurance_claim_line_review.cdmn.json)
  Collection-wide claim-line checks with required/prohibited modifier logic.

## Compliance / Procurement

- [vendor_due_diligence.cdmn.json](/Users/tombielecki/Documents/Code/llm-provable-computer/programs/vendor_due_diligence.cdmn.json)
  Risk thresholding plus jurisdiction and certification screening.

## Current Capability Surface

These examples only use the semantics that are implemented today:

- scalar arithmetic and comparisons
- `and`, `or`, `not`
- inclusive `between`
- top-level arithmetic `definitions`
- bounded collections
- `forall` and `exists`
- explicit reason codes / labels
- explicit insufficient-data reasons

They intentionally avoid temporal logic, units, null propagation rules, and other parts of the broader cDMN/DMN roadmap that are not implemented yet.
