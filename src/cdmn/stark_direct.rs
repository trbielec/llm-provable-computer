use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::ast::{ArithmeticOp, AvailabilityPredicateKind, ComparisonOp, QuantifierKind};
use super::proof_eval::{evaluate_decision_program, input_commitment};
use super::proof_ir::{
    CompiledCdmnDecisionProgram, DecisionArithmeticExpr, DecisionAvailability,
    DecisionAvailabilitySource, DecisionBooleanExpr, DecisionCountBoundCheck,
    DecisionFailureReason, DecisionQuantifiedCollection, DecisionValueRef, DecisionValueSource,
};
use super::{compile_cdmn_table_to_decision_program, CdmnTable};
use crate::error::{Result, VmError};
use crate::proof::VanillaStarkProofOptions;
use crate::state::{Word, WORD_BITS};
use crate::vanillastark::{FieldElement, MPolynomial, Stark, StarkProofProfile};

const TRACE_ROWS: usize = 2;
const STATUS_AVAILABLE: Word = 0;
const STATUS_MISSING: Word = 1;
const STATUS_UNKNOWN: Word = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkDirectClaim {
    pub decision_program: CompiledCdmnDecisionProgram,
    pub projected_inputs: Vec<(u8, Word)>,
    pub status: Word,
    pub reason_code: Word,
    pub policy_commitment: Vec<u8>,
    pub input_commitment: Vec<u8>,
    pub options: VanillaStarkProofOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkDirectProof {
    pub claim: CdmnDecisionStarkDirectClaim,
    pub proof: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkDirectProvingProfile {
    pub circuit_build_micros: u64,
    pub witness_build_micros: u64,
    pub trace_materialization_micros: u64,
    pub air_build_micros: u64,
    pub boundary_build_micros: u64,
    pub stark: StarkProofProfile,
    pub total_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkDirectProvingReport {
    pub proof: CdmnDecisionStarkDirectProof,
    pub profile: CdmnDecisionStarkDirectProvingProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnDecisionStarkDirectCostEstimate {
    pub trace_rows: usize,
    pub register_count: usize,
    pub projected_input_count: usize,
    pub definition_count: usize,
    pub arithmetic_node_count: usize,
    pub boolean_node_count: usize,
    pub constraint_count: usize,
    pub quantified_node_count: usize,
    pub quantified_item_predicate_count: usize,
    pub quantified_max_items_total: usize,
    pub equality_aux_count: usize,
    pub ordered_aux_count: usize,
    pub transition_constraint_count: usize,
    pub transition_degree_bound: usize,
    pub num_randomizers: usize,
    pub omicron_domain_length: usize,
    pub fri_domain_length: usize,
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub security_level: usize,
}

#[derive(Debug, Clone)]
struct InputRegister {
    address: u8,
    register: usize,
}

#[derive(Debug, Clone, Default)]
struct InputKinds {
    boolean_flags: BTreeSet<u8>,
    status_cells: BTreeSet<u8>,
    collection_counts: BTreeMap<u8, u8>,
}

#[derive(Debug, Clone, Copy)]
struct ConcreteQuantifiedBinding {
    base_memory: u8,
    stride: u8,
    item_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ArithmeticSource {
    Constant(Word),
    Input {
        register: usize,
        availability: DecisionAvailability,
    },
    DefinitionValue(usize),
    Node(usize),
}

#[derive(Debug, Clone)]
struct ArithmeticNode {
    op: ArithmeticOp,
    left: ArithmeticSource,
    right: ArithmeticSource,
    value_register: usize,
    status_register: usize,
    bit_registers: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DefinitionBinding {
    source: ArithmeticSource,
    value_register: usize,
    status_register: usize,
    bit_registers: Vec<usize>,
    materialized: bool,
}

#[derive(Debug, Clone)]
struct EqualityAux {
    delta_register: usize,
    inverse_register: usize,
}

#[derive(Debug, Clone)]
struct OrderedComparisonAux {
    diff_register: usize,
    diff_bits: Vec<usize>,
    inverse_register: usize,
}

#[derive(Debug, Clone)]
enum QuantifiedPredicate {
    Constant(bool),
    Comparison {
        lhs: ArithmeticSource,
        op: ComparisonOp,
        rhs: ArithmeticSource,
        equality_aux: Option<EqualityAux>,
        ordered_aux: Option<OrderedComparisonAux>,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: ArithmeticSource,
    },
    Between {
        value: ArithmeticSource,
        lower: ArithmeticSource,
        upper: ArithmeticSource,
        lower_aux: OrderedComparisonAux,
        upper_aux: OrderedComparisonAux,
    },
    And(Box<QuantifiedPredicate>, Box<QuantifiedPredicate>),
    Or(Box<QuantifiedPredicate>, Box<QuantifiedPredicate>),
    Not(Box<QuantifiedPredicate>),
}

#[derive(Debug, Clone)]
enum BooleanNodeKind {
    Constant(bool),
    CountAtLeast {
        register: usize,
        threshold: u8,
        max_items: u8,
    },
    Comparison {
        lhs: ArithmeticSource,
        op: ComparisonOp,
        rhs: ArithmeticSource,
        equality_aux: Option<EqualityAux>,
        ordered_aux: Option<OrderedComparisonAux>,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: ArithmeticSource,
    },
    Between {
        value: ArithmeticSource,
        lower: ArithmeticSource,
        upper: ArithmeticSource,
        lower_aux: OrderedComparisonAux,
        upper_aux: OrderedComparisonAux,
    },
    And {
        left: usize,
        right: usize,
    },
    Or {
        left: usize,
        right: usize,
    },
    Not {
        inner: usize,
    },
    Quantified {
        kind: QuantifierKind,
        count_register: usize,
        max_items: u8,
        item_predicates: Vec<QuantifiedPredicate>,
    },
}

#[derive(Debug, Clone)]
struct BooleanNode {
    kind: BooleanNodeKind,
    value_register: usize,
    status_register: usize,
}

#[derive(Debug, Clone)]
struct ConstraintStage {
    bound_checks: Vec<DecisionCountBoundCheck>,
    expression: ConstraintExpression,
    reason: DecisionFailureReason,
    insufficient_data_reason: DecisionFailureReason,
}

#[derive(Debug, Clone, Copy)]
enum ConstraintExpression {
    BooleanNode(usize),
    CompactQuantified(usize),
}

#[derive(Debug, Clone)]
struct DirectDecisionCircuit {
    always_available_register: usize,
    input_registers: Vec<InputRegister>,
    input_register_by_address: BTreeMap<u8, usize>,
    input_kinds: InputKinds,
    arithmetic_nodes: Vec<ArithmeticNode>,
    definitions: Vec<DefinitionBinding>,
    boolean_nodes: Vec<BooleanNode>,
    compact_quantified_stages: Vec<CompactQuantifiedStage>,
    stages: Vec<ConstraintStage>,
    final_status_register: usize,
    final_reason_register: usize,
    row_counter_register: Option<usize>,
    trace_rows: usize,
    register_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct EvaluatedArithmetic {
    value: Word,
    status: Word,
}

#[derive(Debug, Clone, Copy)]
struct EvaluatedBoolean {
    value: bool,
    status: Word,
}

#[derive(Debug, Clone)]
struct EqualityWitness {
    delta: i128,
    inverse: FieldElement,
}

#[derive(Debug, Clone)]
struct OrderedWitness {
    diff: Word,
    inverse: FieldElement,
}

#[derive(Debug, Clone)]
struct DirectDecisionWitness {
    projected_inputs: Vec<(u8, Word)>,
    input_values: Vec<Word>,
    arithmetic_nodes: Vec<EvaluatedArithmetic>,
    arithmetic_bits: Vec<Vec<bool>>,
    definitions: Vec<EvaluatedArithmetic>,
    definition_bits: Vec<Vec<bool>>,
    boolean_nodes: Vec<EvaluatedBoolean>,
    compact_quantified_stages: Vec<CompactQuantifiedStageWitness>,
    equality_aux: HashMap<usize, EqualityWitness>,
    ordered_aux: HashMap<usize, OrderedWitness>,
    final_status: Word,
    final_reason: Word,
    trace_rows: usize,
}

#[derive(Debug, Clone)]
struct CompactQuantifiedStage {
    kind: QuantifierKind,
    collection: DecisionQuantifiedCollection,
    predicate: CompactQuantifiedPredicate,
    start_row: usize,
    row_count: usize,
    result_value_register: usize,
    result_status_register: usize,
    accumulator_value_register: usize,
    accumulator_status_register: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CompactAvailabilitySource {
    InputRegister(usize),
    DefinitionStatus(usize),
    ItemRegisters(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CompactAvailability {
    Always,
    Flags {
        presence: Option<CompactAvailabilitySource>,
        unknown: Option<CompactAvailabilitySource>,
    },
    Status(CompactAvailabilitySource),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CompactValueSource {
    InputRegister(usize),
    DefinitionValue(usize),
    ItemRegisters(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CompactValueRef {
    source: CompactValueSource,
    availability: CompactAvailability,
}

#[derive(Debug, Clone)]
enum CompactQuantifiedPredicate {
    Constant(bool),
    Comparison {
        lhs: CompactArithmeticExpr,
        op: ComparisonOp,
        rhs: CompactArithmeticExpr,
        equality_aux: Option<EqualityAux>,
        ordered_aux: Option<OrderedComparisonAux>,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: CompactArithmeticExpr,
    },
    Between {
        value: CompactArithmeticExpr,
        lower: CompactArithmeticExpr,
        upper: CompactArithmeticExpr,
        lower_aux: OrderedComparisonAux,
        upper_aux: OrderedComparisonAux,
    },
    And(
        Box<CompactQuantifiedPredicate>,
        Box<CompactQuantifiedPredicate>,
    ),
    Or(
        Box<CompactQuantifiedPredicate>,
        Box<CompactQuantifiedPredicate>,
    ),
    Not(Box<CompactQuantifiedPredicate>),
}

#[derive(Debug, Clone)]
enum CompactArithmeticExpr {
    Number(Word),
    Value(CompactValueRef),
}

#[derive(Debug, Clone)]
struct CompactQuantifiedStageWitness {
    result: EvaluatedBoolean,
    accumulator_rows: Vec<EvaluatedBoolean>,
    equality_aux_rows: Vec<HashMap<usize, EqualityWitness>>,
    ordered_aux_rows: Vec<HashMap<usize, OrderedWitness>>,
}

pub fn prove_cdmn_decision_stark_direct(
    table: &CdmnTable,
    initial_memory: &[Word],
) -> Result<CdmnDecisionStarkDirectProof> {
    prove_cdmn_decision_stark_direct_with_options(
        table,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_cdmn_decision_stark_direct_profiled(
    table: &CdmnTable,
    initial_memory: &[Word],
) -> Result<CdmnDecisionStarkDirectProvingReport> {
    prove_cdmn_decision_stark_direct_profiled_with_options(
        table,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn estimate_cdmn_decision_stark_direct_cost(
    table: &CdmnTable,
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectCostEstimate> {
    let decision_program = compile_cdmn_table_to_decision_program(table)?;
    estimate_compiled_cdmn_decision_stark_direct_cost(&decision_program, options)
}

pub fn prove_cdmn_decision_stark_direct_with_options(
    table: &CdmnTable,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectProof> {
    let decision_program = compile_cdmn_table_to_decision_program(table)?;
    prove_compiled_cdmn_decision_stark_direct_with_options(
        &decision_program,
        initial_memory,
        options,
    )
}

pub fn prove_cdmn_decision_stark_direct_profiled_with_options(
    table: &CdmnTable,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectProvingReport> {
    let decision_program = compile_cdmn_table_to_decision_program(table)?;
    prove_compiled_cdmn_decision_stark_direct_profiled_with_options(
        &decision_program,
        initial_memory,
        options,
    )
}

pub fn prove_compiled_cdmn_decision_stark_direct(
    decision_program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<CdmnDecisionStarkDirectProof> {
    prove_compiled_cdmn_decision_stark_direct_with_options(
        decision_program,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn prove_compiled_cdmn_decision_stark_direct_profiled(
    decision_program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<CdmnDecisionStarkDirectProvingReport> {
    prove_compiled_cdmn_decision_stark_direct_profiled_with_options(
        decision_program,
        initial_memory,
        VanillaStarkProofOptions::default(),
    )
}

pub fn estimate_compiled_cdmn_decision_stark_direct_cost(
    decision_program: &CompiledCdmnDecisionProgram,
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectCostEstimate> {
    let circuit = DirectDecisionCircuitBuilder::build(decision_program)?;
    let air = DirectDecisionAir::new(circuit.clone());
    let transition_constraints = air.transition_constraints();
    let transition_degree_bound =
        DirectDecisionAir::transition_degree_bound_for(&transition_constraints);
    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        circuit.trace_rows(),
        transition_degree_bound,
    );

    Ok(CdmnDecisionStarkDirectCostEstimate {
        trace_rows: circuit.trace_rows(),
        register_count: air.register_count(),
        projected_input_count: decision_program.input_cells.len(),
        definition_count: circuit.definitions.len(),
        arithmetic_node_count: circuit.arithmetic_nodes.len(),
        boolean_node_count: circuit.boolean_nodes.len(),
        constraint_count: circuit.stages.len(),
        quantified_node_count: circuit.quantified_node_count(),
        quantified_item_predicate_count: circuit.quantified_item_predicate_count(),
        quantified_max_items_total: circuit.quantified_max_items_total(),
        equality_aux_count: circuit.equality_aux_count(),
        ordered_aux_count: circuit.ordered_aux_count(),
        transition_constraint_count: transition_constraints.len(),
        transition_degree_bound,
        num_randomizers: stark.num_randomizers,
        omicron_domain_length: stark.omicron_domain.len(),
        fri_domain_length: stark.fri.domain_length,
        expansion_factor: options.expansion_factor,
        num_colinearity_checks: options.num_colinearity_checks,
        security_level: options.security_level,
    })
}

pub fn prove_compiled_cdmn_decision_stark_direct_with_options(
    decision_program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectProof> {
    Ok(
        prove_compiled_cdmn_decision_stark_direct_profiled_with_options(
            decision_program,
            initial_memory,
            options,
        )?
        .proof,
    )
}

pub fn prove_compiled_cdmn_decision_stark_direct_profiled_with_options(
    decision_program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
    options: VanillaStarkProofOptions,
) -> Result<CdmnDecisionStarkDirectProvingReport> {
    let total_start = Instant::now();
    let phase_start = Instant::now();
    let circuit = DirectDecisionCircuitBuilder::build(decision_program)?;
    let circuit_build_micros = elapsed_micros(phase_start);
    let phase_start = Instant::now();
    let witness = circuit.build_witness(decision_program, initial_memory)?;
    let witness_build_micros = elapsed_micros(phase_start);
    let phase_start = Instant::now();
    let trace = circuit.materialize_trace_rows(&witness);
    let trace_materialization_micros = elapsed_micros(phase_start);
    let phase_start = Instant::now();
    let air = DirectDecisionAir::new(circuit);
    let air_build_micros = elapsed_micros(phase_start);
    let transition_constraints = air.transition_constraints();
    let transition_degree_bound =
        DirectDecisionAir::transition_degree_bound_for(&transition_constraints);
    let claim = CdmnDecisionStarkDirectClaim {
        decision_program: decision_program.clone(),
        projected_inputs: witness.projected_inputs.clone(),
        status: witness.final_status,
        reason_code: witness.final_reason,
        policy_commitment: decision_program.policy_commitment()?,
        input_commitment: input_commitment(decision_program, initial_memory)?,
        options: options.clone(),
    };
    let phase_start = Instant::now();
    let boundary = air.boundary_constraints(&claim)?;
    let boundary_build_micros = elapsed_micros(phase_start);

    let stark = Stark::new(
        options.expansion_factor,
        options.num_colinearity_checks,
        options.security_level,
        air.register_count(),
        witness.trace_rows,
        transition_degree_bound,
    );
    let (proof, stark_profile) =
        stark.prove_with_profile(&trace, &transition_constraints, &boundary);

    Ok(CdmnDecisionStarkDirectProvingReport {
        proof: CdmnDecisionStarkDirectProof { claim, proof },
        profile: CdmnDecisionStarkDirectProvingProfile {
            circuit_build_micros,
            witness_build_micros,
            trace_materialization_micros,
            air_build_micros,
            boundary_build_micros,
            stark: stark_profile,
            total_micros: elapsed_micros(total_start),
        },
    })
}

pub fn verify_cdmn_decision_stark_direct(proof: &CdmnDecisionStarkDirectProof) -> Result<bool> {
    let circuit = DirectDecisionCircuitBuilder::build(&proof.claim.decision_program)?;
    let materialized =
        materialize_projected_inputs(&proof.claim.decision_program, &proof.claim.projected_inputs)?;
    let evaluation = evaluate_decision_program(&proof.claim.decision_program, &materialized)?;
    if evaluation.status != proof.claim.status
        || evaluation.reason_code != proof.claim.reason_code
        || evaluation.policy_commitment != proof.claim.policy_commitment
        || evaluation.input_commitment != proof.claim.input_commitment
    {
        return Ok(false);
    }

    let trace_rows = circuit.trace_rows();
    let air = DirectDecisionAir::new(circuit);
    let transition_constraints = air.transition_constraints();
    let transition_degree_bound =
        DirectDecisionAir::transition_degree_bound_for(&transition_constraints);
    let stark = Stark::new(
        proof.claim.options.expansion_factor,
        proof.claim.options.num_colinearity_checks,
        proof.claim.options.security_level,
        air.register_count(),
        trace_rows,
        transition_degree_bound,
    );
    Ok(stark.verify(
        &proof.proof,
        &transition_constraints,
        &air.boundary_constraints(&proof.claim)?,
    ))
}

struct DirectDecisionCircuitBuilder {
    circuit: DirectDecisionCircuit,
    arithmetic_cache: HashMap<DecisionArithmeticExpr, ArithmeticSource>,
    boolean_cache: HashMap<DecisionBooleanExpr, usize>,
    equality_aux_cache: HashMap<(ArithmeticSource, ArithmeticSource), EqualityAux>,
    ordered_aux_cache: HashMap<(ArithmeticSource, ArithmeticSource), OrderedComparisonAux>,
}

impl DirectDecisionCircuitBuilder {
    fn build(program: &CompiledCdmnDecisionProgram) -> Result<DirectDecisionCircuit> {
        let mut builder = Self {
            circuit: DirectDecisionCircuit {
                always_available_register: 0,
                input_registers: Vec::new(),
                input_register_by_address: BTreeMap::new(),
                input_kinds: InputKinds::default(),
                arithmetic_nodes: Vec::new(),
                definitions: Vec::new(),
                boolean_nodes: Vec::new(),
                compact_quantified_stages: Vec::new(),
                stages: Vec::new(),
                final_status_register: 0,
                final_reason_register: 0,
                row_counter_register: None,
                trace_rows: TRACE_ROWS,
                register_count: 1,
            },
            arithmetic_cache: HashMap::new(),
            boolean_cache: HashMap::new(),
            equality_aux_cache: HashMap::new(),
            ordered_aux_cache: HashMap::new(),
        };

        for &address in &program.input_cells {
            let register = builder.allocate_register();
            builder
                .circuit
                .input_registers
                .push(InputRegister { address, register });
            builder
                .circuit
                .input_register_by_address
                .insert(address, register);
        }

        for definition in &program.definitions {
            let source = builder.lower_arithmetic(&definition.expression)?;
            let (value_register, status_register, bit_registers, materialized) =
                builder.materialize_definition_binding(&source);
            builder.circuit.definitions.push(DefinitionBinding {
                source,
                value_register,
                status_register,
                bit_registers,
                materialized,
            });
        }

        let mut next_compact_row = 0usize;
        for constraint in &program.constraints {
            for bound_check in &constraint.bound_checks {
                builder
                    .circuit
                    .input_kinds
                    .collection_counts
                    .entry(bound_check.count_memory)
                    .and_modify(|current| *current = (*current).max(bound_check.max_items))
                    .or_insert(bound_check.max_items);
            }
            let expression = if let Some(stage_index) = builder
                .try_add_compact_quantified_stage(&constraint.expression, &mut next_compact_row)?
            {
                ConstraintExpression::CompactQuantified(stage_index)
            } else {
                ConstraintExpression::BooleanNode(builder.lower_boolean(&constraint.expression)?)
            };
            builder.circuit.stages.push(ConstraintStage {
                bound_checks: constraint.bound_checks.clone(),
                expression,
                reason: constraint.reason.clone(),
                insufficient_data_reason: constraint.insufficient_data_reason.clone(),
            });
        }

        if !builder.circuit.compact_quantified_stages.is_empty() {
            builder.circuit.row_counter_register = Some(builder.allocate_register());
            builder.circuit.trace_rows = next_compact_row.max(TRACE_ROWS);
        }

        builder.circuit.final_status_register = builder.allocate_register();
        builder.circuit.final_reason_register = builder.allocate_register();

        Ok(builder.circuit)
    }

    fn lower_arithmetic(&mut self, expr: &DecisionArithmeticExpr) -> Result<ArithmeticSource> {
        if let Some(source) = self.arithmetic_cache.get(expr) {
            return Ok(source.clone());
        }

        let source = match expr {
            DecisionArithmeticExpr::Number(value) => ArithmeticSource::Constant(*value),
            DecisionArithmeticExpr::Value(value_ref) => self.lower_value_ref(value_ref)?,
            DecisionArithmeticExpr::Binary { op, left, right } => {
                let left = self.lower_arithmetic(left)?;
                let right = self.lower_arithmetic(right)?;
                let value_register = self.allocate_register();
                let status_register = if self.arithmetic_source_is_always_available(&left)
                    && self.arithmetic_source_is_always_available(&right)
                {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let bit_registers = self.allocate_bits();
                let node_index = self.circuit.arithmetic_nodes.len();
                self.circuit.arithmetic_nodes.push(ArithmeticNode {
                    op: *op,
                    left,
                    right,
                    value_register,
                    status_register,
                    bit_registers,
                });
                ArithmeticSource::Node(node_index)
            }
        };

        self.arithmetic_cache.insert(expr.clone(), source.clone());
        Ok(source)
    }

    fn lower_boolean(&mut self, expr: &DecisionBooleanExpr) -> Result<usize> {
        if let Some(index) = self.boolean_cache.get(expr).copied() {
            return Ok(index);
        }

        if let DecisionBooleanExpr::Quantified {
            kind,
            collection,
            predicate,
        } = expr
        {
            self.circuit
                .input_kinds
                .collection_counts
                .entry(collection.count_memory)
                .and_modify(|current| *current = (*current).max(collection.max_items))
                .or_insert(collection.max_items);
            let count_register = self.input_register(collection.count_memory)?;
            let item_predicates = (0..usize::from(collection.max_items))
                .map(|item_index| {
                    let mut bindings = HashMap::new();
                    bindings.insert(
                        collection.binding,
                        ConcreteQuantifiedBinding {
                            base_memory: collection.base_memory,
                            stride: collection.stride,
                            item_index,
                        },
                    );
                    self.lower_quantified_predicate(predicate, &bindings)
                })
                .collect::<Result<Vec<_>>>()?;
            let value_register = self.allocate_register();
            let status_register = if item_predicates
                .iter()
                .all(|predicate| self.quantified_predicate_is_always_available(predicate))
            {
                self.circuit.always_available_register
            } else {
                self.allocate_register()
            };
            let index = self.circuit.boolean_nodes.len();
            self.circuit.boolean_nodes.push(BooleanNode {
                kind: BooleanNodeKind::Quantified {
                    kind: *kind,
                    count_register,
                    max_items: collection.max_items,
                    item_predicates,
                },
                value_register,
                status_register,
            });
            self.boolean_cache.insert(expr.clone(), index);
            return Ok(index);
        }

        if let Some((kind, count_memory, max_items, predicates)) = extract_quantifier_pattern(expr)
        {
            self.circuit
                .input_kinds
                .collection_counts
                .entry(count_memory)
                .and_modify(|current| *current = (*current).max(max_items))
                .or_insert(max_items);
            let item_predicates = predicates
                .into_iter()
                .map(|predicate| self.lower_quantified_predicate(predicate, &HashMap::new()))
                .collect::<Result<Vec<_>>>()?;
            let value_register = self.allocate_register();
            let status_register = if item_predicates
                .iter()
                .all(|predicate| self.quantified_predicate_is_always_available(predicate))
            {
                self.circuit.always_available_register
            } else {
                self.allocate_register()
            };
            let index = self.circuit.boolean_nodes.len();
            self.circuit.boolean_nodes.push(BooleanNode {
                kind: BooleanNodeKind::Quantified {
                    kind,
                    count_register: self.input_register(count_memory)?,
                    max_items,
                    item_predicates,
                },
                value_register,
                status_register,
            });
            self.boolean_cache.insert(expr.clone(), index);
            return Ok(index);
        }

        let index = match expr {
            DecisionBooleanExpr::Constant(value) => {
                let value_register = self.allocate_register();
                let status_register = self.circuit.always_available_register;
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::Constant(*value),
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::CountAtLeast {
                count_memory,
                threshold,
                max_items,
            } => {
                self.circuit
                    .input_kinds
                    .collection_counts
                    .entry(*count_memory)
                    .and_modify(|current| *current = (*current).max(*max_items))
                    .or_insert(*max_items);
                let value_register = self.allocate_register();
                let status_register = self.circuit.always_available_register;
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::CountAtLeast {
                        register: self.input_register(*count_memory)?,
                        threshold: *threshold,
                        max_items: *max_items,
                    },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::Comparison { lhs, op, rhs } => {
                let lhs = self.lower_arithmetic(lhs)?;
                let rhs = self.lower_arithmetic(rhs)?;
                self.add_comparison_node(lhs, *op, rhs)?
            }
            DecisionBooleanExpr::Availability { kind, value } => {
                let value = self.lower_arithmetic(value)?;
                let value_register = self.allocate_register();
                let status_register = self.circuit.always_available_register;
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::Availability { kind: *kind, value },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::Between { value, lower, upper } => {
                let value = self.lower_arithmetic(value)?;
                let lower = self.lower_arithmetic(lower)?;
                let upper = self.lower_arithmetic(upper)?;
                let value_register = self.allocate_register();
                let status_register = if self.arithmetic_source_is_always_available(&value)
                    && self.arithmetic_source_is_always_available(&lower)
                    && self.arithmetic_source_is_always_available(&upper)
                {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let lower_aux = self.allocate_ordered_aux();
                let upper_aux = self.allocate_ordered_aux();
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::Between {
                        value,
                        lower,
                        upper,
                        lower_aux,
                        upper_aux,
                    },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::And(left, right) => {
                let left = self.lower_boolean(left)?;
                let right = self.lower_boolean(right)?;
                let value_register = self.allocate_register();
                let status_register = if self.boolean_node_is_always_available(left)
                    && self.boolean_node_is_always_available(right)
                {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::And { left, right },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::Or(left, right) => {
                let left = self.lower_boolean(left)?;
                let right = self.lower_boolean(right)?;
                let value_register = self.allocate_register();
                let status_register = if self.boolean_node_is_always_available(left)
                    && self.boolean_node_is_always_available(right)
                {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::Or { left, right },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::Not(inner) => {
                let inner = self.lower_boolean(inner)?;
                let value_register = self.allocate_register();
                let status_register = if self.boolean_node_is_always_available(inner) {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let index = self.circuit.boolean_nodes.len();
                self.circuit.boolean_nodes.push(BooleanNode {
                    kind: BooleanNodeKind::Not { inner },
                    value_register,
                    status_register,
                });
                index
            }
            DecisionBooleanExpr::Quantified { .. } => unreachable!(
                "top-level quantified expressions are handled before the generic boolean lowering match"
            ),
        };

        self.boolean_cache.insert(expr.clone(), index);
        Ok(index)
    }

    fn try_add_compact_quantified_stage(
        &mut self,
        expr: &DecisionBooleanExpr,
        next_row: &mut usize,
    ) -> Result<Option<usize>> {
        let DecisionBooleanExpr::Quantified {
            kind,
            collection,
            predicate,
        } = expr
        else {
            return Ok(None);
        };

        let Some(predicate) = self.lower_compact_quantified_predicate(collection, predicate)?
        else {
            return Ok(None);
        };

        self.circuit
            .input_kinds
            .collection_counts
            .entry(collection.count_memory)
            .and_modify(|current| *current = (*current).max(collection.max_items))
            .or_insert(collection.max_items);
        let result_value_register = self.allocate_register();
        let result_status_register = self.allocate_register();
        let accumulator_value_register = self.allocate_register();
        let accumulator_status_register = self.allocate_register();
        let stage_index = self.circuit.compact_quantified_stages.len();
        self.circuit
            .compact_quantified_stages
            .push(CompactQuantifiedStage {
                kind: *kind,
                collection: collection.clone(),
                predicate,
                start_row: *next_row,
                row_count: usize::from(collection.max_items) + 1,
                result_value_register,
                result_status_register,
                accumulator_value_register,
                accumulator_status_register,
            });
        *next_row += usize::from(collection.max_items) + 1;
        Ok(Some(stage_index))
    }

    fn add_comparison_node(
        &mut self,
        lhs: ArithmeticSource,
        op: ComparisonOp,
        rhs: ArithmeticSource,
    ) -> Result<usize> {
        let value_register = self.allocate_register();
        let status_register = if self.arithmetic_source_is_always_available(&lhs)
            && self.arithmetic_source_is_always_available(&rhs)
        {
            self.circuit.always_available_register
        } else {
            self.allocate_register()
        };
        let (equality_aux, ordered_aux) = match op {
            ComparisonOp::Eq | ComparisonOp::Ne => (
                Some(self.get_or_allocate_equality_aux(lhs.clone(), rhs.clone())),
                None,
            ),
            ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => (
                None,
                Some(self.get_or_allocate_ordered_aux(lhs.clone(), rhs.clone())),
            ),
        };
        let index = self.circuit.boolean_nodes.len();
        self.circuit.boolean_nodes.push(BooleanNode {
            kind: BooleanNodeKind::Comparison {
                lhs,
                op,
                rhs,
                equality_aux,
                ordered_aux,
            },
            value_register,
            status_register,
        });
        Ok(index)
    }

    fn lower_value_ref(&mut self, value_ref: &DecisionValueRef) -> Result<ArithmeticSource> {
        match &value_ref.source {
            DecisionValueSource::InputCell(address) => Ok(ArithmeticSource::Input {
                register: self.input_register(*address)?,
                availability: self.mark_input_availability(&value_ref.availability)?,
            }),
            DecisionValueSource::DefinitionValue(index) => Ok(ArithmeticSource::DefinitionValue(*index)),
            DecisionValueSource::BoundField { .. } => Err(VmError::InvalidConfig(
                "bound fields must be concretized inside quantified lowering before direct-STARK circuit construction".to_string(),
            )),
        }
    }

    fn mark_input_availability(
        &mut self,
        availability: &DecisionAvailability,
    ) -> Result<DecisionAvailability> {
        match availability {
            DecisionAvailability::Always => Ok(DecisionAvailability::Always),
            DecisionAvailability::Flags { presence, unknown } => {
                if let Some(presence) = presence {
                    self.mark_boolean_flag_source(presence)?;
                }
                if let Some(unknown) = unknown {
                    self.mark_boolean_flag_source(unknown)?;
                }
                Ok(availability.clone())
            }
            DecisionAvailability::Status(source) => {
                self.mark_status_source(source)?;
                Ok(availability.clone())
            }
        }
    }

    fn mark_boolean_flag_source(&mut self, source: &DecisionAvailabilitySource) -> Result<()> {
        match source {
            DecisionAvailabilitySource::InputCell(address) => {
                self.circuit.input_kinds.boolean_flags.insert(*address);
                self.ensure_input_exists(*address)
            }
            DecisionAvailabilitySource::DefinitionStatus(_) => Err(VmError::InvalidConfig(
                "definition status cannot be used as a boolean availability flag".to_string(),
            )),
            DecisionAvailabilitySource::BoundField { .. } => Err(VmError::InvalidConfig(
                "bound-field availability flags must be concretized before direct-STARK lowering"
                    .to_string(),
            )),
        }
    }

    fn mark_status_source(&mut self, source: &DecisionAvailabilitySource) -> Result<()> {
        match source {
            DecisionAvailabilitySource::InputCell(address) => {
                self.circuit.input_kinds.status_cells.insert(*address);
                self.ensure_input_exists(*address)
            }
            DecisionAvailabilitySource::DefinitionStatus(_) => Ok(()),
            DecisionAvailabilitySource::BoundField { .. } => Err(VmError::InvalidConfig(
                "bound-field status sources must be concretized before direct-STARK lowering"
                    .to_string(),
            )),
        }
    }

    fn ensure_input_exists(&self, address: u8) -> Result<()> {
        self.circuit
            .input_register_by_address
            .get(&address)
            .copied()
            .ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "decision program references input cell {address} that is not in the input projection"
                ))
            })?;
        Ok(())
    }

    fn input_register(&self, address: u8) -> Result<usize> {
        self.circuit
            .input_register_by_address
            .get(&address)
            .copied()
            .ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "decision program references input cell {address} that is not in the input projection"
                ))
            })
    }

    fn allocate_register(&mut self) -> usize {
        let register = self.circuit.register_count;
        self.circuit.register_count += 1;
        register
    }

    fn allocate_bits(&mut self) -> Vec<usize> {
        (0..WORD_BITS).map(|_| self.allocate_register()).collect()
    }

    fn allocate_equality_aux(&mut self) -> EqualityAux {
        EqualityAux {
            delta_register: self.allocate_register(),
            inverse_register: self.allocate_register(),
        }
    }

    fn allocate_ordered_aux(&mut self) -> OrderedComparisonAux {
        OrderedComparisonAux {
            diff_register: self.allocate_register(),
            diff_bits: self.allocate_bits(),
            inverse_register: self.allocate_register(),
        }
    }

    fn arithmetic_source_is_always_available(&self, source: &ArithmeticSource) -> bool {
        match source {
            ArithmeticSource::Constant(_) => true,
            ArithmeticSource::Input { availability, .. } => {
                matches!(availability, DecisionAvailability::Always)
            }
            ArithmeticSource::DefinitionValue(index) => {
                self.circuit.definitions[*index].status_register
                    == self.circuit.always_available_register
            }
            ArithmeticSource::Node(index) => {
                self.circuit.arithmetic_nodes[*index].status_register
                    == self.circuit.always_available_register
            }
        }
    }

    fn boolean_node_is_always_available(&self, index: usize) -> bool {
        self.circuit.boolean_nodes[index].status_register == self.circuit.always_available_register
    }

    fn lower_quantified_predicate(
        &mut self,
        expr: &DecisionBooleanExpr,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<QuantifiedPredicate> {
        match expr {
            DecisionBooleanExpr::Constant(value) => Ok(QuantifiedPredicate::Constant(*value)),
            DecisionBooleanExpr::Comparison { lhs, op, rhs } => {
                let lhs = self.lower_bound_arithmetic(lhs, bindings)?;
                let rhs = self.lower_bound_arithmetic(rhs, bindings)?;
                let (equality_aux, ordered_aux) = match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        (Some(self.get_or_allocate_equality_aux(lhs.clone(), rhs.clone())), None)
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => (
                        None,
                        Some(self.get_or_allocate_ordered_aux(lhs.clone(), rhs.clone())),
                    ),
                };
                Ok(QuantifiedPredicate::Comparison {
                    lhs,
                    op: *op,
                    rhs,
                    equality_aux,
                    ordered_aux,
                })
            }
            DecisionBooleanExpr::Availability { kind, value } => {
                let value = self.lower_bound_arithmetic(value, bindings)?;
                Ok(QuantifiedPredicate::Availability {
                    kind: *kind,
                    value,
                })
            }
            DecisionBooleanExpr::Between { value, lower, upper } => {
                let value = self.lower_bound_arithmetic(value, bindings)?;
                let lower = self.lower_bound_arithmetic(lower, bindings)?;
                let upper = self.lower_bound_arithmetic(upper, bindings)?;
                Ok(QuantifiedPredicate::Between {
                    value,
                    lower,
                    upper,
                    lower_aux: self.allocate_ordered_aux(),
                    upper_aux: self.allocate_ordered_aux(),
                })
            }
            DecisionBooleanExpr::And(left, right) => Ok(QuantifiedPredicate::And(
                Box::new(self.lower_quantified_predicate(left, bindings)?),
                Box::new(self.lower_quantified_predicate(right, bindings)?),
            )),
            DecisionBooleanExpr::Or(left, right) => Ok(QuantifiedPredicate::Or(
                Box::new(self.lower_quantified_predicate(left, bindings)?),
                Box::new(self.lower_quantified_predicate(right, bindings)?),
            )),
            DecisionBooleanExpr::Not(inner) => Ok(QuantifiedPredicate::Not(Box::new(
                self.lower_quantified_predicate(inner, bindings)?,
            ))),
            DecisionBooleanExpr::CountAtLeast { .. } => {
                Err(VmError::InvalidConfig(
                    "unexpected nested quantifier/count expression in quantified predicate lowering"
                        .to_string(),
                ))
            }
            DecisionBooleanExpr::Quantified { .. } => Err(VmError::UnsupportedProof(
                "nested quantified predicates are not yet supported by the direct cDMN STARK backend"
                    .to_string(),
            )),
        }
    }

    fn lower_compact_quantified_predicate(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        expr: &DecisionBooleanExpr,
    ) -> Result<Option<CompactQuantifiedPredicate>> {
        match expr {
            DecisionBooleanExpr::Constant(value) => {
                Ok(Some(CompactQuantifiedPredicate::Constant(*value)))
            }
            DecisionBooleanExpr::Comparison { lhs, op, rhs } => {
                let Some(lhs) = self.lower_compact_arithmetic_expr(collection, lhs)? else {
                    return Ok(None);
                };
                let Some(rhs) = self.lower_compact_arithmetic_expr(collection, rhs)? else {
                    return Ok(None);
                };
                let (equality_aux, ordered_aux) = match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        (Some(self.allocate_equality_aux()), None)
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => {
                        (None, Some(self.allocate_ordered_aux()))
                    }
                };
                Ok(Some(CompactQuantifiedPredicate::Comparison {
                    lhs,
                    op: *op,
                    rhs,
                    equality_aux,
                    ordered_aux,
                }))
            }
            DecisionBooleanExpr::Availability { kind, value } => {
                let Some(value) = self.lower_compact_arithmetic_expr(collection, value)? else {
                    return Ok(None);
                };
                Ok(Some(CompactQuantifiedPredicate::Availability {
                    kind: *kind,
                    value,
                }))
            }
            DecisionBooleanExpr::Between {
                value,
                lower,
                upper,
            } => {
                let Some(value) = self.lower_compact_arithmetic_expr(collection, value)? else {
                    return Ok(None);
                };
                let Some(lower) = self.lower_compact_arithmetic_expr(collection, lower)? else {
                    return Ok(None);
                };
                let Some(upper) = self.lower_compact_arithmetic_expr(collection, upper)? else {
                    return Ok(None);
                };
                Ok(Some(CompactQuantifiedPredicate::Between {
                    value,
                    lower,
                    upper,
                    lower_aux: self.allocate_ordered_aux(),
                    upper_aux: self.allocate_ordered_aux(),
                }))
            }
            DecisionBooleanExpr::And(left, right) => {
                let Some(left) = self.lower_compact_quantified_predicate(collection, left)? else {
                    return Ok(None);
                };
                let Some(right) = self.lower_compact_quantified_predicate(collection, right)?
                else {
                    return Ok(None);
                };
                Ok(Some(CompactQuantifiedPredicate::And(
                    Box::new(left),
                    Box::new(right),
                )))
            }
            DecisionBooleanExpr::Or(left, right) => {
                let Some(left) = self.lower_compact_quantified_predicate(collection, left)? else {
                    return Ok(None);
                };
                let Some(right) = self.lower_compact_quantified_predicate(collection, right)?
                else {
                    return Ok(None);
                };
                Ok(Some(CompactQuantifiedPredicate::Or(
                    Box::new(left),
                    Box::new(right),
                )))
            }
            DecisionBooleanExpr::Not(inner) => {
                let Some(inner) = self.lower_compact_quantified_predicate(collection, inner)?
                else {
                    return Ok(None);
                };
                Ok(Some(CompactQuantifiedPredicate::Not(Box::new(inner))))
            }
            DecisionBooleanExpr::CountAtLeast { .. } | DecisionBooleanExpr::Quantified { .. } => {
                Ok(None)
            }
        }
    }

    fn lower_compact_arithmetic_expr(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        expr: &DecisionArithmeticExpr,
    ) -> Result<Option<CompactArithmeticExpr>> {
        match expr {
            DecisionArithmeticExpr::Number(value) => {
                Ok(Some(CompactArithmeticExpr::Number(*value)))
            }
            DecisionArithmeticExpr::Value(value_ref) => Ok(Some(CompactArithmeticExpr::Value(
                self.lower_compact_value_ref(collection, value_ref)?,
            ))),
            DecisionArithmeticExpr::Binary { .. } => Ok(None),
        }
    }

    fn lower_compact_value_ref(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        value_ref: &DecisionValueRef,
    ) -> Result<CompactValueRef> {
        Ok(CompactValueRef {
            source: self.lower_compact_value_source(collection, &value_ref.source)?,
            availability: self.lower_compact_availability(collection, &value_ref.availability)?,
        })
    }

    fn lower_compact_value_source(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        source: &DecisionValueSource,
    ) -> Result<CompactValueSource> {
        match source {
            DecisionValueSource::InputCell(address) => Ok(CompactValueSource::InputRegister(
                self.input_register(*address)?,
            )),
            DecisionValueSource::DefinitionValue(index) => {
                Ok(CompactValueSource::DefinitionValue(*index))
            }
            DecisionValueSource::BoundField { binding, offset } => {
                if *binding != collection.binding {
                    return Err(VmError::UnsupportedProof(
                        "compact quantified direct STARK currently supports a single quantified binding per top-level constraint".to_string(),
                    ));
                }
                Ok(CompactValueSource::ItemRegisters(
                    self.bound_registers(collection, *offset)?,
                ))
            }
        }
    }

    fn lower_compact_availability(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        availability: &DecisionAvailability,
    ) -> Result<CompactAvailability> {
        match availability {
            DecisionAvailability::Always => Ok(CompactAvailability::Always),
            DecisionAvailability::Flags { presence, unknown } => Ok(CompactAvailability::Flags {
                presence: presence
                    .as_ref()
                    .map(|source| self.lower_compact_availability_source(collection, source))
                    .transpose()?,
                unknown: unknown
                    .as_ref()
                    .map(|source| self.lower_compact_availability_source(collection, source))
                    .transpose()?,
            }),
            DecisionAvailability::Status(source) => Ok(CompactAvailability::Status(
                self.lower_compact_availability_source(collection, source)?,
            )),
        }
    }

    fn lower_compact_availability_source(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        source: &DecisionAvailabilitySource,
    ) -> Result<CompactAvailabilitySource> {
        match source {
            DecisionAvailabilitySource::InputCell(address) => Ok(
                CompactAvailabilitySource::InputRegister(self.input_register(*address)?),
            ),
            DecisionAvailabilitySource::DefinitionStatus(index) => {
                Ok(CompactAvailabilitySource::DefinitionStatus(*index))
            }
            DecisionAvailabilitySource::BoundField { binding, offset } => {
                if *binding != collection.binding {
                    return Err(VmError::UnsupportedProof(
                        "compact quantified direct STARK currently supports a single quantified binding per top-level constraint".to_string(),
                    ));
                }
                Ok(CompactAvailabilitySource::ItemRegisters(
                    self.bound_registers(collection, *offset)?,
                ))
            }
        }
    }

    fn bound_registers(
        &mut self,
        collection: &DecisionQuantifiedCollection,
        offset: u8,
    ) -> Result<Vec<usize>> {
        (0..usize::from(collection.max_items))
            .map(|item_index| {
                let address = u8::try_from(
                    usize::from(collection.base_memory)
                        + item_index * usize::from(collection.stride)
                        + usize::from(offset),
                )
                .map_err(|_| {
                    VmError::InvalidConfig(
                        "quantified collection address exceeds the u8 address space".to_string(),
                    )
                })?;
                self.input_register(address)
            })
            .collect()
    }

    fn lower_bound_arithmetic(
        &mut self,
        expr: &DecisionArithmeticExpr,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<ArithmeticSource> {
        match expr {
            DecisionArithmeticExpr::Number(value) => Ok(ArithmeticSource::Constant(*value)),
            DecisionArithmeticExpr::Value(value_ref) => {
                self.lower_bound_value_ref(value_ref, bindings)
            }
            DecisionArithmeticExpr::Binary { op, left, right } => {
                let left = self.lower_bound_arithmetic(left, bindings)?;
                let right = self.lower_bound_arithmetic(right, bindings)?;
                let value_register = self.allocate_register();
                let status_register = if self.arithmetic_source_is_always_available(&left)
                    && self.arithmetic_source_is_always_available(&right)
                {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let bit_registers = self.allocate_bits();
                let node_index = self.circuit.arithmetic_nodes.len();
                self.circuit.arithmetic_nodes.push(ArithmeticNode {
                    op: *op,
                    left,
                    right,
                    value_register,
                    status_register,
                    bit_registers,
                });
                Ok(ArithmeticSource::Node(node_index))
            }
        }
    }

    fn lower_bound_value_ref(
        &mut self,
        value_ref: &DecisionValueRef,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<ArithmeticSource> {
        match &value_ref.source {
            DecisionValueSource::InputCell(address) => Ok(ArithmeticSource::Input {
                register: self.input_register(*address)?,
                availability: self.mark_input_availability(&value_ref.availability)?,
            }),
            DecisionValueSource::DefinitionValue(index) => {
                Ok(ArithmeticSource::DefinitionValue(*index))
            }
            DecisionValueSource::BoundField { binding, offset } => {
                let address = self.bound_input_address(*binding, *offset, bindings)?;
                let availability =
                    self.concretize_bound_availability(&value_ref.availability, bindings)?;
                Ok(ArithmeticSource::Input {
                    register: self.input_register(address)?,
                    availability,
                })
            }
        }
    }

    fn concretize_bound_availability(
        &mut self,
        availability: &DecisionAvailability,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<DecisionAvailability> {
        let concretized = match availability {
            DecisionAvailability::Always => DecisionAvailability::Always,
            DecisionAvailability::Flags { presence, unknown } => DecisionAvailability::Flags {
                presence: presence
                    .as_ref()
                    .map(|source| self.concretize_availability_source(source, bindings))
                    .transpose()?,
                unknown: unknown
                    .as_ref()
                    .map(|source| self.concretize_availability_source(source, bindings))
                    .transpose()?,
            },
            DecisionAvailability::Status(source) => {
                DecisionAvailability::Status(self.concretize_availability_source(source, bindings)?)
            }
        };
        self.mark_input_availability(&concretized)
    }

    fn concretize_availability_source(
        &mut self,
        source: &DecisionAvailabilitySource,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<DecisionAvailabilitySource> {
        match source {
            DecisionAvailabilitySource::InputCell(address) => {
                Ok(DecisionAvailabilitySource::InputCell(*address))
            }
            DecisionAvailabilitySource::DefinitionStatus(index) => {
                Ok(DecisionAvailabilitySource::DefinitionStatus(*index))
            }
            DecisionAvailabilitySource::BoundField { binding, offset } => {
                Ok(DecisionAvailabilitySource::InputCell(
                    self.bound_input_address(*binding, *offset, bindings)?,
                ))
            }
        }
    }

    fn bound_input_address(
        &self,
        binding: usize,
        offset: u8,
        bindings: &HashMap<usize, ConcreteQuantifiedBinding>,
    ) -> Result<u8> {
        let context = bindings.get(&binding).ok_or_else(|| {
            VmError::InvalidConfig(format!(
                "missing concrete quantified binding {binding} in direct-STARK lowering"
            ))
        })?;
        let address = usize::from(context.base_memory)
            + context.item_index * usize::from(context.stride)
            + usize::from(offset);
        let address = u8::try_from(address).map_err(|_| {
            VmError::InvalidConfig(
                "quantified collection address exceeds the u8 address space".to_string(),
            )
        })?;
        self.ensure_input_exists(address)?;
        Ok(address)
    }

    fn quantified_predicate_is_always_available(&self, predicate: &QuantifiedPredicate) -> bool {
        match predicate {
            QuantifiedPredicate::Constant(_) => true,
            QuantifiedPredicate::Comparison { lhs, rhs, .. } => {
                self.arithmetic_source_is_always_available(lhs)
                    && self.arithmetic_source_is_always_available(rhs)
            }
            QuantifiedPredicate::Availability { .. } => true,
            QuantifiedPredicate::Between {
                value,
                lower,
                upper,
                ..
            } => {
                self.arithmetic_source_is_always_available(value)
                    && self.arithmetic_source_is_always_available(lower)
                    && self.arithmetic_source_is_always_available(upper)
            }
            QuantifiedPredicate::And(left, right) | QuantifiedPredicate::Or(left, right) => {
                self.quantified_predicate_is_always_available(left)
                    && self.quantified_predicate_is_always_available(right)
            }
            QuantifiedPredicate::Not(inner) => self.quantified_predicate_is_always_available(inner),
        }
    }

    fn materialize_definition_binding(
        &mut self,
        source: &ArithmeticSource,
    ) -> (usize, usize, Vec<usize>, bool) {
        match source {
            ArithmeticSource::Node(index) => {
                let node = &self.circuit.arithmetic_nodes[*index];
                (
                    node.value_register,
                    node.status_register,
                    node.bit_registers.clone(),
                    false,
                )
            }
            ArithmeticSource::DefinitionValue(index) => {
                let definition = &self.circuit.definitions[*index];
                (
                    definition.value_register,
                    definition.status_register,
                    definition.bit_registers.clone(),
                    false,
                )
            }
            ArithmeticSource::Constant(_) | ArithmeticSource::Input { .. } => {
                let value_register = self.allocate_register();
                let status_register = if self.arithmetic_source_is_always_available(source) {
                    self.circuit.always_available_register
                } else {
                    self.allocate_register()
                };
                let bit_registers = self.allocate_bits();
                (value_register, status_register, bit_registers, true)
            }
        }
    }

    fn get_or_allocate_equality_aux(
        &mut self,
        lhs: ArithmeticSource,
        rhs: ArithmeticSource,
    ) -> EqualityAux {
        let key = (lhs, rhs);
        if let Some(aux) = self.equality_aux_cache.get(&key) {
            return aux.clone();
        }
        let aux = self.allocate_equality_aux();
        self.equality_aux_cache.insert(key, aux.clone());
        aux
    }

    fn get_or_allocate_ordered_aux(
        &mut self,
        lhs: ArithmeticSource,
        rhs: ArithmeticSource,
    ) -> OrderedComparisonAux {
        let key = (lhs, rhs);
        if let Some(aux) = self.ordered_aux_cache.get(&key) {
            return aux.clone();
        }
        let aux = self.allocate_ordered_aux();
        self.ordered_aux_cache.insert(key, aux.clone());
        aux
    }
}

impl DirectDecisionCircuit {
    fn trace_rows(&self) -> usize {
        self.trace_rows
    }

    fn compact_mutable_registers(&self) -> HashSet<usize> {
        let mut registers = HashSet::new();
        if let Some(row_counter_register) = self.row_counter_register {
            registers.insert(row_counter_register);
        }
        for stage in &self.compact_quantified_stages {
            registers.insert(stage.accumulator_value_register);
            registers.insert(stage.accumulator_status_register);
            collect_compact_predicate_registers(&stage.predicate, &mut registers);
        }
        registers
    }

    fn quantified_node_count(&self) -> usize {
        self.boolean_nodes
            .iter()
            .filter(|node| matches!(node.kind, BooleanNodeKind::Quantified { .. }))
            .count()
            + self.compact_quantified_stages.len()
    }

    fn quantified_item_predicate_count(&self) -> usize {
        self.boolean_nodes
            .iter()
            .map(|node| match &node.kind {
                BooleanNodeKind::Quantified {
                    item_predicates, ..
                } => item_predicates.len(),
                _ => 0,
            })
            .sum::<usize>()
            + self
                .compact_quantified_stages
                .iter()
                .map(|stage| usize::from(stage.collection.max_items))
                .sum::<usize>()
    }

    fn quantified_max_items_total(&self) -> usize {
        self.boolean_nodes
            .iter()
            .map(|node| match &node.kind {
                BooleanNodeKind::Quantified { max_items, .. } => usize::from(*max_items),
                _ => 0,
            })
            .sum::<usize>()
            + self
                .compact_quantified_stages
                .iter()
                .map(|stage| usize::from(stage.collection.max_items))
                .sum::<usize>()
    }

    fn equality_aux_count(&self) -> usize {
        let mut aux = HashSet::new();
        for node in &self.boolean_nodes {
            match &node.kind {
                BooleanNodeKind::Comparison {
                    equality_aux: Some(aux_regs),
                    ..
                } => {
                    aux.insert(aux_regs.delta_register);
                }
                BooleanNodeKind::Quantified {
                    item_predicates, ..
                } => {
                    for predicate in item_predicates {
                        collect_quantified_equality_aux(predicate, &mut aux);
                    }
                }
                _ => {}
            }
        }
        for stage in &self.compact_quantified_stages {
            collect_compact_equality_aux(&stage.predicate, &mut aux);
        }
        aux.len()
    }

    fn ordered_aux_count(&self) -> usize {
        let mut aux = HashSet::new();
        for node in &self.boolean_nodes {
            match &node.kind {
                BooleanNodeKind::Comparison {
                    ordered_aux: Some(aux_regs),
                    ..
                } => {
                    aux.insert(aux_regs.diff_register);
                }
                BooleanNodeKind::Between {
                    lower_aux,
                    upper_aux,
                    ..
                } => {
                    aux.insert(lower_aux.diff_register);
                    aux.insert(upper_aux.diff_register);
                }
                BooleanNodeKind::Quantified {
                    item_predicates, ..
                } => {
                    for predicate in item_predicates {
                        collect_quantified_ordered_aux(predicate, &mut aux);
                    }
                }
                _ => {}
            }
        }
        for stage in &self.compact_quantified_stages {
            collect_compact_ordered_aux(&stage.predicate, &mut aux);
        }
        aux.len()
    }

    fn build_witness(
        &self,
        decision_program: &CompiledCdmnDecisionProgram,
        initial_memory: &[Word],
    ) -> Result<DirectDecisionWitness> {
        let projected_inputs = project_inputs(decision_program, initial_memory)?;
        let materialized = materialize_projected_inputs(decision_program, &projected_inputs)?;
        let evaluation = evaluate_decision_program(decision_program, &materialized)?;
        let input_lookup: BTreeMap<u8, Word> = projected_inputs.iter().copied().collect();
        let input_values = self
            .input_registers
            .iter()
            .map(|input| {
                input_lookup.get(&input.address).copied().ok_or_else(|| {
                    VmError::InvalidConfig(format!(
                        "missing projected input cell {} in direct cDMN witness",
                        input.address
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        for &address in &self.input_kinds.boolean_flags {
            let value = *input_lookup.get(&address).ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "missing boolean availability flag cell {address} in direct cDMN witness"
                ))
            })?;
            if value != 0 && value != 1 {
                return Err(VmError::UnsupportedProof(format!(
                    "direct cDMN STARK currently requires boolean availability flags; cell {address} had value {value}"
                )));
            }
        }
        for &address in &self.input_kinds.status_cells {
            let value = *input_lookup.get(&address).ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "missing status-coded input cell {address} in direct cDMN witness"
                ))
            })?;
            if !(STATUS_AVAILABLE..=STATUS_UNKNOWN).contains(&value) {
                return Err(VmError::UnsupportedProof(format!(
                    "direct cDMN STARK currently requires status-coded inputs in {{0,1,2}}; cell {address} had value {value}"
                )));
            }
        }

        let mut arithmetic_nodes = Vec::with_capacity(self.arithmetic_nodes.len());
        let mut arithmetic_bits = Vec::with_capacity(self.arithmetic_nodes.len());
        let mut definitions = Vec::with_capacity(self.definitions.len());
        let mut definition_bits = Vec::with_capacity(self.definitions.len());
        let mut boolean_nodes: Vec<EvaluatedBoolean> = Vec::with_capacity(self.boolean_nodes.len());
        let mut compact_quantified_stages =
            Vec::with_capacity(self.compact_quantified_stages.len());
        let mut equality_aux = HashMap::new();
        let mut ordered_aux = HashMap::new();

        for node in &self.arithmetic_nodes {
            let left = self.eval_arithmetic_source(
                &node.left,
                &input_values,
                &definitions,
                &arithmetic_nodes,
            )?;
            let right = self.eval_arithmetic_source(
                &node.right,
                &input_values,
                &definitions,
                &arithmetic_nodes,
            )?;
            let status = combine_status(left.status, right.status);
            let value = if status == STATUS_AVAILABLE {
                checked_binary_op(node.op, left.value, right.value)?
            } else {
                0
            };
            arithmetic_nodes.push(EvaluatedArithmetic { value, status });
            arithmetic_bits.push(signed_bits(value));
        }

        for definition in &self.definitions {
            let evaluated = self.eval_arithmetic_source(
                &definition.source,
                &input_values,
                &definitions,
                &arithmetic_nodes,
            )?;
            definitions.push(evaluated);
            definition_bits.push(signed_bits(evaluated.value));
        }

        for (index, node) in self.boolean_nodes.iter().enumerate() {
            let evaluated = match &node.kind {
                BooleanNodeKind::Constant(value) => EvaluatedBoolean {
                    value: *value,
                    status: STATUS_AVAILABLE,
                },
                BooleanNodeKind::CountAtLeast {
                    register,
                    threshold,
                    max_items,
                } => {
                    let index = self
                        .input_registers
                        .iter()
                        .position(|input| input.register == *register)
                        .ok_or_else(|| {
                            VmError::InvalidConfig(format!(
                                "unknown count-at-least input register {register}"
                            ))
                        })?;
                    let count = input_values[index];
                    EvaluatedBoolean {
                        value: (0..=i64::from(*max_items)).contains(&count)
                            && count >= i64::from(*threshold),
                        status: STATUS_AVAILABLE,
                    }
                }
                BooleanNodeKind::Comparison {
                    lhs,
                    op,
                    rhs,
                    equality_aux: eq_aux,
                    ordered_aux: ord_aux,
                } => {
                    let lhs = self.eval_arithmetic_source(
                        lhs,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    let rhs = self.eval_arithmetic_source(
                        rhs,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    let status = combine_status(lhs.status, rhs.status);
                    if status != STATUS_AVAILABLE {
                        if let Some(eq_aux) = eq_aux {
                            equality_aux.insert(
                                eq_aux.delta_register,
                                EqualityWitness {
                                    delta: 0,
                                    inverse: FieldElement::zero(),
                                },
                            );
                        }
                        if let Some(ord_aux) = ord_aux {
                            ordered_aux.insert(
                                ord_aux.diff_register,
                                OrderedWitness {
                                    diff: 0,
                                    inverse: FieldElement::zero(),
                                },
                            );
                        }
                        EvaluatedBoolean {
                            value: false,
                            status,
                        }
                    } else {
                        match op {
                            ComparisonOp::Eq | ComparisonOp::Ne => {
                                let delta = i128::from(lhs.value) - i128::from(rhs.value);
                                let is_zero = delta == 0;
                                let value = match op {
                                    ComparisonOp::Eq => is_zero,
                                    ComparisonOp::Ne => !is_zero,
                                    _ => unreachable!(),
                                };
                                let aux = eq_aux.as_ref().ok_or_else(|| {
                                    VmError::InvalidConfig(
                                        "comparison node missing equality auxiliary registers"
                                            .to_string(),
                                    )
                                })?;
                                equality_aux.insert(
                                    aux.delta_register,
                                    EqualityWitness {
                                        delta,
                                        inverse: if is_zero {
                                            FieldElement::zero()
                                        } else {
                                            field_from_i128(delta).inverse()
                                        },
                                    },
                                );
                                EvaluatedBoolean { value, status }
                            }
                            ComparisonOp::Lt
                            | ComparisonOp::Le
                            | ComparisonOp::Gt
                            | ComparisonOp::Ge => {
                                let diff = lhs.value.checked_sub(rhs.value).ok_or_else(|| {
                                    VmError::UnsupportedProof(
                                        "direct cDMN STARK currently requires ordered comparisons whose subtraction fits i64"
                                            .to_string(),
                                    )
                                })?;
                                let value = match op {
                                    ComparisonOp::Lt => diff < 0,
                                    ComparisonOp::Le => diff <= 0,
                                    ComparisonOp::Gt => diff > 0,
                                    ComparisonOp::Ge => diff >= 0,
                                    _ => unreachable!(),
                                };
                                let aux = ord_aux.as_ref().ok_or_else(|| {
                                    VmError::InvalidConfig(
                                        "comparison node missing ordered auxiliary registers"
                                            .to_string(),
                                    )
                                })?;
                                ordered_aux.insert(
                                    aux.diff_register,
                                    OrderedWitness {
                                        diff,
                                        inverse: if diff == 0 {
                                            FieldElement::zero()
                                        } else {
                                            field_from_word(diff).inverse()
                                        },
                                    },
                                );
                                EvaluatedBoolean { value, status }
                            }
                        }
                    }
                }
                BooleanNodeKind::Availability { kind, value } => {
                    let value = self.eval_arithmetic_source(
                        value,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    EvaluatedBoolean {
                        value: match kind {
                            AvailabilityPredicateKind::Present => value.status == STATUS_AVAILABLE,
                            AvailabilityPredicateKind::Missing => value.status == STATUS_MISSING,
                            AvailabilityPredicateKind::Unknown => value.status == STATUS_UNKNOWN,
                        },
                        status: STATUS_AVAILABLE,
                    }
                }
                BooleanNodeKind::Between {
                    value,
                    lower,
                    upper,
                    lower_aux,
                    upper_aux,
                } => {
                    let value = self.eval_arithmetic_source(
                        value,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    let lower = self.eval_arithmetic_source(
                        lower,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    let upper = self.eval_arithmetic_source(
                        upper,
                        &input_values,
                        &definitions,
                        &arithmetic_nodes,
                    )?;
                    let status =
                        combine_status(value.status, combine_status(lower.status, upper.status));
                    if status != STATUS_AVAILABLE {
                        ordered_aux.insert(
                            lower_aux.diff_register,
                            OrderedWitness {
                                diff: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                        ordered_aux.insert(
                            upper_aux.diff_register,
                            OrderedWitness {
                                diff: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                        EvaluatedBoolean {
                            value: false,
                            status,
                        }
                    } else {
                        let lower_diff = value.value.checked_sub(lower.value).ok_or_else(|| {
                            VmError::UnsupportedProof(
                                "direct cDMN STARK currently requires `between` lower-bound subtraction to fit i64"
                                    .to_string(),
                            )
                        })?;
                        let upper_diff = value.value.checked_sub(upper.value).ok_or_else(|| {
                            VmError::UnsupportedProof(
                                "direct cDMN STARK currently requires `between` upper-bound subtraction to fit i64"
                                    .to_string(),
                            )
                        })?;
                        ordered_aux.insert(
                            lower_aux.diff_register,
                            OrderedWitness {
                                diff: lower_diff,
                                inverse: if lower_diff == 0 {
                                    FieldElement::zero()
                                } else {
                                    field_from_word(lower_diff).inverse()
                                },
                            },
                        );
                        ordered_aux.insert(
                            upper_aux.diff_register,
                            OrderedWitness {
                                diff: upper_diff,
                                inverse: if upper_diff == 0 {
                                    FieldElement::zero()
                                } else {
                                    field_from_word(upper_diff).inverse()
                                },
                            },
                        );
                        EvaluatedBoolean {
                            value: value.value >= lower.value && value.value <= upper.value,
                            status,
                        }
                    }
                }
                BooleanNodeKind::And { left, right } => {
                    let left = boolean_nodes[*left];
                    let right = boolean_nodes[*right];
                    if left.status != STATUS_AVAILABLE || !left.value {
                        left
                    } else {
                        right
                    }
                }
                BooleanNodeKind::Or { left, right } => {
                    let left = boolean_nodes[*left];
                    let right = boolean_nodes[*right];
                    if left.status != STATUS_AVAILABLE || left.value {
                        left
                    } else {
                        right
                    }
                }
                BooleanNodeKind::Not { inner } => {
                    let inner = boolean_nodes[*inner];
                    if inner.status != STATUS_AVAILABLE {
                        inner
                    } else {
                        EvaluatedBoolean {
                            value: !inner.value,
                            status: STATUS_AVAILABLE,
                        }
                    }
                }
                BooleanNodeKind::Quantified {
                    kind,
                    count_register,
                    max_items,
                    item_predicates,
                } => {
                    let count_index = self
                        .input_registers
                        .iter()
                        .position(|input| input.register == *count_register)
                        .ok_or_else(|| {
                            VmError::InvalidConfig(format!(
                                "unknown quantified count register {count_register}"
                            ))
                        })?;
                    let count = input_values[count_index];
                    let in_domain = (0..=i64::from(*max_items)).contains(&count);
                    let mut acc = match kind {
                        QuantifierKind::ForAll => EvaluatedBoolean {
                            value: true,
                            status: STATUS_AVAILABLE,
                        },
                        QuantifierKind::Exists => EvaluatedBoolean {
                            value: false,
                            status: STATUS_AVAILABLE,
                        },
                    };
                    for (index, predicate) in item_predicates.iter().enumerate() {
                        let active = in_domain && count >= i64::try_from(index + 1).unwrap();
                        if !active {
                            continue;
                        }
                        let item = self.eval_quantified_predicate(
                            predicate,
                            &input_values,
                            &definitions,
                            &arithmetic_nodes,
                            &mut equality_aux,
                            &mut ordered_aux,
                        )?;
                        acc = match kind {
                            QuantifierKind::ForAll => {
                                if acc.status != STATUS_AVAILABLE || !acc.value {
                                    acc
                                } else {
                                    item
                                }
                            }
                            QuantifierKind::Exists => {
                                if acc.status != STATUS_AVAILABLE || acc.value {
                                    acc
                                } else {
                                    item
                                }
                            }
                        };
                    }
                    acc
                }
            };
            let _ = index;
            boolean_nodes.push(evaluated);
        }

        for stage in &self.compact_quantified_stages {
            compact_quantified_stages.push(self.eval_compact_quantified_stage(
                stage,
                &input_values,
                &definitions,
                &arithmetic_nodes,
            )?);
        }

        let mut current_status = 1;
        let mut current_reason = 0;
        for stage in &self.stages {
            if current_status == 0 {
                continue;
            }
            let bound_ok = stage.bound_checks.iter().all(|bound_check| {
                input_lookup[&bound_check.count_memory] >= 0
                    && input_lookup[&bound_check.count_memory] <= i64::from(bound_check.max_items)
            });
            let expr = match stage.expression {
                ConstraintExpression::BooleanNode(index) => boolean_nodes[index],
                ConstraintExpression::CompactQuantified(index) => {
                    compact_quantified_stages[index].result
                }
            };
            if !bound_ok || expr.status != STATUS_AVAILABLE {
                current_status = 0;
                current_reason = stage.insufficient_data_reason.code;
            } else if !expr.value {
                current_status = 0;
                current_reason = stage.reason.code;
            } else {
                current_status = 1;
                current_reason = 0;
            }
        }

        let (final_status, final_reason) = if self.stages.is_empty() {
            (1, 0)
        } else {
            (current_status, current_reason)
        };
        if final_status != evaluation.status || final_reason != evaluation.reason_code {
            return Err(VmError::ExecutionMismatch {
                step: 0,
                message: format!(
                    "direct cDMN circuit witness produced status={} reason={} but evaluator produced status={} reason={}",
                    final_status, final_reason, evaluation.status, evaluation.reason_code
                ),
            });
        }

        Ok(DirectDecisionWitness {
            projected_inputs,
            input_values,
            arithmetic_nodes,
            arithmetic_bits,
            definitions,
            definition_bits,
            boolean_nodes,
            compact_quantified_stages,
            equality_aux,
            ordered_aux,
            final_status,
            final_reason,
            trace_rows: self.trace_rows(),
        })
    }

    fn eval_arithmetic_source(
        &self,
        source: &ArithmeticSource,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
        arithmetic_nodes: &[EvaluatedArithmetic],
    ) -> Result<EvaluatedArithmetic> {
        match source {
            ArithmeticSource::Constant(value) => Ok(EvaluatedArithmetic {
                value: *value,
                status: STATUS_AVAILABLE,
            }),
            ArithmeticSource::Input {
                register,
                availability,
            } => {
                let index = self
                    .input_registers
                    .iter()
                    .position(|input| input.register == *register)
                    .ok_or_else(|| {
                        VmError::InvalidConfig(format!("unknown direct input register {register}"))
                    })?;
                Ok(EvaluatedArithmetic {
                    value: input_values[index],
                    status: self.evaluate_availability(availability, input_values, definitions)?,
                })
            }
            ArithmeticSource::DefinitionValue(index) => {
                definitions.get(*index).copied().ok_or_else(|| {
                    VmError::InvalidConfig(format!(
                        "missing evaluated definition {index} in direct cDMN witness"
                    ))
                })
            }
            ArithmeticSource::Node(index) => {
                arithmetic_nodes.get(*index).copied().ok_or_else(|| {
                    VmError::InvalidConfig(format!(
                        "missing evaluated arithmetic node {index} in direct cDMN witness"
                    ))
                })
            }
        }
    }

    fn eval_quantified_predicate(
        &self,
        predicate: &QuantifiedPredicate,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
        arithmetic_nodes: &[EvaluatedArithmetic],
        equality_aux: &mut HashMap<usize, EqualityWitness>,
        ordered_aux: &mut HashMap<usize, OrderedWitness>,
    ) -> Result<EvaluatedBoolean> {
        match predicate {
            QuantifiedPredicate::Constant(value) => Ok(EvaluatedBoolean {
                value: *value,
                status: STATUS_AVAILABLE,
            }),
            QuantifiedPredicate::Comparison {
                lhs,
                op,
                rhs,
                equality_aux: eq_aux,
                ordered_aux: ord_aux,
            } => {
                let lhs =
                    self.eval_arithmetic_source(lhs, input_values, definitions, arithmetic_nodes)?;
                let rhs =
                    self.eval_arithmetic_source(rhs, input_values, definitions, arithmetic_nodes)?;
                let status = combine_status(lhs.status, rhs.status);
                if status != STATUS_AVAILABLE {
                    if let Some(eq_aux) = eq_aux {
                        equality_aux.insert(
                            eq_aux.delta_register,
                            EqualityWitness {
                                delta: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                    }
                    if let Some(ord_aux) = ord_aux {
                        ordered_aux.insert(
                            ord_aux.diff_register,
                            OrderedWitness {
                                diff: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                    }
                    return Ok(EvaluatedBoolean {
                        value: false,
                        status,
                    });
                }
                match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        let delta = i128::from(lhs.value) - i128::from(rhs.value);
                        let is_zero = delta == 0;
                        let value = match op {
                            ComparisonOp::Eq => is_zero,
                            ComparisonOp::Ne => !is_zero,
                            _ => unreachable!(),
                        };
                        let aux = eq_aux.as_ref().ok_or_else(|| {
                            VmError::InvalidConfig(
                                "quantified comparison missing equality auxiliary registers"
                                    .to_string(),
                            )
                        })?;
                        equality_aux.insert(
                            aux.delta_register,
                            EqualityWitness {
                                delta,
                                inverse: if is_zero {
                                    FieldElement::zero()
                                } else {
                                    field_from_i128(delta).inverse()
                                },
                            },
                        );
                        Ok(EvaluatedBoolean { value, status })
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => {
                        let diff = lhs.value.checked_sub(rhs.value).ok_or_else(|| {
                            VmError::UnsupportedProof(
                                "direct cDMN STARK currently requires ordered comparisons whose subtraction fits i64"
                                    .to_string(),
                            )
                        })?;
                        let value = match op {
                            ComparisonOp::Lt => diff < 0,
                            ComparisonOp::Le => diff <= 0,
                            ComparisonOp::Gt => diff > 0,
                            ComparisonOp::Ge => diff >= 0,
                            _ => unreachable!(),
                        };
                        let aux = ord_aux.as_ref().ok_or_else(|| {
                            VmError::InvalidConfig(
                                "quantified comparison missing ordered auxiliary registers"
                                    .to_string(),
                            )
                        })?;
                        ordered_aux.insert(
                            aux.diff_register,
                            OrderedWitness {
                                diff,
                                inverse: if diff == 0 {
                                    FieldElement::zero()
                                } else {
                                    field_from_word(diff).inverse()
                                },
                            },
                        );
                        Ok(EvaluatedBoolean { value, status })
                    }
                }
            }
            QuantifiedPredicate::Availability { kind, value } => {
                let value = self.eval_arithmetic_source(
                    value,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                Ok(EvaluatedBoolean {
                    value: match kind {
                        AvailabilityPredicateKind::Present => value.status == STATUS_AVAILABLE,
                        AvailabilityPredicateKind::Missing => value.status == STATUS_MISSING,
                        AvailabilityPredicateKind::Unknown => value.status == STATUS_UNKNOWN,
                    },
                    status: STATUS_AVAILABLE,
                })
            }
            QuantifiedPredicate::Between {
                value,
                lower,
                upper,
                lower_aux,
                upper_aux,
            } => {
                let value = self.eval_arithmetic_source(
                    value,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let lower = self.eval_arithmetic_source(
                    lower,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let upper = self.eval_arithmetic_source(
                    upper,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let status =
                    combine_status(value.status, combine_status(lower.status, upper.status));
                if status != STATUS_AVAILABLE {
                    ordered_aux.insert(
                        lower_aux.diff_register,
                        OrderedWitness {
                            diff: 0,
                            inverse: FieldElement::zero(),
                        },
                    );
                    ordered_aux.insert(
                        upper_aux.diff_register,
                        OrderedWitness {
                            diff: 0,
                            inverse: FieldElement::zero(),
                        },
                    );
                    return Ok(EvaluatedBoolean {
                        value: false,
                        status,
                    });
                }
                let lower_diff = value.value.checked_sub(lower.value).ok_or_else(|| {
                    VmError::UnsupportedProof(
                        "direct cDMN STARK currently requires `between` lower-bound subtraction to fit i64"
                            .to_string(),
                    )
                })?;
                let upper_diff = value.value.checked_sub(upper.value).ok_or_else(|| {
                    VmError::UnsupportedProof(
                        "direct cDMN STARK currently requires `between` upper-bound subtraction to fit i64"
                            .to_string(),
                    )
                })?;
                ordered_aux.insert(
                    lower_aux.diff_register,
                    OrderedWitness {
                        diff: lower_diff,
                        inverse: if lower_diff == 0 {
                            FieldElement::zero()
                        } else {
                            field_from_word(lower_diff).inverse()
                        },
                    },
                );
                ordered_aux.insert(
                    upper_aux.diff_register,
                    OrderedWitness {
                        diff: upper_diff,
                        inverse: if upper_diff == 0 {
                            FieldElement::zero()
                        } else {
                            field_from_word(upper_diff).inverse()
                        },
                    },
                );
                Ok(EvaluatedBoolean {
                    value: value.value >= lower.value && value.value <= upper.value,
                    status,
                })
            }
            QuantifiedPredicate::And(left, right) => {
                let left = self.eval_quantified_predicate(
                    left,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if left.status != STATUS_AVAILABLE || !left.value {
                    Ok(left)
                } else {
                    self.eval_quantified_predicate(
                        right,
                        input_values,
                        definitions,
                        arithmetic_nodes,
                        equality_aux,
                        ordered_aux,
                    )
                }
            }
            QuantifiedPredicate::Or(left, right) => {
                let left = self.eval_quantified_predicate(
                    left,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if left.status != STATUS_AVAILABLE || left.value {
                    Ok(left)
                } else {
                    self.eval_quantified_predicate(
                        right,
                        input_values,
                        definitions,
                        arithmetic_nodes,
                        equality_aux,
                        ordered_aux,
                    )
                }
            }
            QuantifiedPredicate::Not(inner) => {
                let inner = self.eval_quantified_predicate(
                    inner,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if inner.status != STATUS_AVAILABLE {
                    Ok(inner)
                } else {
                    Ok(EvaluatedBoolean {
                        value: !inner.value,
                        status: STATUS_AVAILABLE,
                    })
                }
            }
        }
    }

    fn eval_compact_quantified_stage(
        &self,
        stage: &CompactQuantifiedStage,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
        arithmetic_nodes: &[EvaluatedArithmetic],
    ) -> Result<CompactQuantifiedStageWitness> {
        let count_index = self.input_index_for_address(stage.collection.count_memory)?;
        let count = input_values[count_index];
        let in_domain = (0..=i64::from(stage.collection.max_items)).contains(&count);
        let mut accumulator_rows = Vec::with_capacity(stage.row_count);
        let mut equality_aux_rows = Vec::with_capacity(stage.row_count);
        let mut ordered_aux_rows = Vec::with_capacity(stage.row_count);

        let mut acc = match stage.kind {
            QuantifierKind::ForAll => EvaluatedBoolean {
                value: true,
                status: STATUS_AVAILABLE,
            },
            QuantifierKind::Exists => EvaluatedBoolean {
                value: false,
                status: STATUS_AVAILABLE,
            },
        };
        accumulator_rows.push(acc);
        for item_index in 0..usize::from(stage.collection.max_items) {
            let active = in_domain && count >= i64::try_from(item_index + 1).unwrap();
            let mut equality_aux = HashMap::new();
            let mut ordered_aux = HashMap::new();
            let item = self.eval_compact_quantified_predicate(
                &stage.predicate,
                item_index,
                input_values,
                definitions,
                arithmetic_nodes,
                &mut equality_aux,
                &mut ordered_aux,
            )?;
            equality_aux_rows.push(equality_aux);
            ordered_aux_rows.push(ordered_aux);
            if active {
                acc = match stage.kind {
                    QuantifierKind::ForAll => {
                        if acc.status != STATUS_AVAILABLE || !acc.value {
                            acc
                        } else {
                            item
                        }
                    }
                    QuantifierKind::Exists => {
                        if acc.status != STATUS_AVAILABLE || acc.value {
                            acc
                        } else {
                            item
                        }
                    }
                };
            }
            accumulator_rows.push(acc);
        }
        while equality_aux_rows.len() < stage.row_count {
            equality_aux_rows.push(HashMap::new());
            ordered_aux_rows.push(HashMap::new());
        }
        Ok(CompactQuantifiedStageWitness {
            result: acc,
            accumulator_rows,
            equality_aux_rows,
            ordered_aux_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_compact_quantified_predicate(
        &self,
        predicate: &CompactQuantifiedPredicate,
        item_index: usize,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
        arithmetic_nodes: &[EvaluatedArithmetic],
        equality_aux: &mut HashMap<usize, EqualityWitness>,
        ordered_aux: &mut HashMap<usize, OrderedWitness>,
    ) -> Result<EvaluatedBoolean> {
        match predicate {
            CompactQuantifiedPredicate::Constant(value) => Ok(EvaluatedBoolean {
                value: *value,
                status: STATUS_AVAILABLE,
            }),
            CompactQuantifiedPredicate::Comparison {
                lhs,
                op,
                rhs,
                equality_aux: eq_aux,
                ordered_aux: ord_aux,
            } => {
                let lhs = self.eval_compact_arithmetic_expr(
                    lhs,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let rhs = self.eval_compact_arithmetic_expr(
                    rhs,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let status = combine_status(lhs.status, rhs.status);
                if status != STATUS_AVAILABLE {
                    if let Some(eq_aux) = eq_aux {
                        equality_aux.insert(
                            eq_aux.delta_register,
                            EqualityWitness {
                                delta: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                    }
                    if let Some(ord_aux) = ord_aux {
                        ordered_aux.insert(
                            ord_aux.diff_register,
                            OrderedWitness {
                                diff: 0,
                                inverse: FieldElement::zero(),
                            },
                        );
                    }
                    return Ok(EvaluatedBoolean {
                        value: false,
                        status,
                    });
                }
                match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        let delta = i128::from(lhs.value) - i128::from(rhs.value);
                        let is_zero = delta == 0;
                        let value = matches!(op, ComparisonOp::Eq) == is_zero;
                        let aux = eq_aux.as_ref().ok_or_else(|| {
                            VmError::InvalidConfig(
                                "compact quantified comparison missing equality auxiliary registers"
                                    .to_string(),
                            )
                        })?;
                        equality_aux.insert(
                            aux.delta_register,
                            EqualityWitness {
                                delta,
                                inverse: if is_zero {
                                    FieldElement::zero()
                                } else {
                                    field_from_i128(delta).inverse()
                                },
                            },
                        );
                        Ok(EvaluatedBoolean { value, status })
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => {
                        let diff = lhs.value.checked_sub(rhs.value).ok_or_else(|| {
                            VmError::UnsupportedProof(
                                "compact quantified direct STARK currently requires ordered comparisons whose subtraction fits i64".to_string(),
                            )
                        })?;
                        let value = match op {
                            ComparisonOp::Lt => diff < 0,
                            ComparisonOp::Le => diff <= 0,
                            ComparisonOp::Gt => diff > 0,
                            ComparisonOp::Ge => diff >= 0,
                            _ => unreachable!(),
                        };
                        let aux = ord_aux.as_ref().ok_or_else(|| {
                            VmError::InvalidConfig(
                                "compact quantified comparison missing ordered auxiliary registers"
                                    .to_string(),
                            )
                        })?;
                        ordered_aux.insert(
                            aux.diff_register,
                            OrderedWitness {
                                diff,
                                inverse: if diff == 0 {
                                    FieldElement::zero()
                                } else {
                                    field_from_word(diff).inverse()
                                },
                            },
                        );
                        Ok(EvaluatedBoolean { value, status })
                    }
                }
            }
            CompactQuantifiedPredicate::Availability { kind, value } => {
                let value = self.eval_compact_arithmetic_expr(
                    value,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                Ok(EvaluatedBoolean {
                    value: match kind {
                        AvailabilityPredicateKind::Present => value.status == STATUS_AVAILABLE,
                        AvailabilityPredicateKind::Missing => value.status == STATUS_MISSING,
                        AvailabilityPredicateKind::Unknown => value.status == STATUS_UNKNOWN,
                    },
                    status: STATUS_AVAILABLE,
                })
            }
            CompactQuantifiedPredicate::Between {
                value,
                lower,
                upper,
                lower_aux,
                upper_aux,
            } => {
                let value = self.eval_compact_arithmetic_expr(
                    value,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let lower = self.eval_compact_arithmetic_expr(
                    lower,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let upper = self.eval_compact_arithmetic_expr(
                    upper,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                )?;
                let status =
                    combine_status(value.status, combine_status(lower.status, upper.status));
                if status != STATUS_AVAILABLE {
                    ordered_aux.insert(
                        lower_aux.diff_register,
                        OrderedWitness {
                            diff: 0,
                            inverse: FieldElement::zero(),
                        },
                    );
                    ordered_aux.insert(
                        upper_aux.diff_register,
                        OrderedWitness {
                            diff: 0,
                            inverse: FieldElement::zero(),
                        },
                    );
                    return Ok(EvaluatedBoolean {
                        value: false,
                        status,
                    });
                }
                let lower_diff = value.value.checked_sub(lower.value).ok_or_else(|| {
                    VmError::UnsupportedProof(
                        "compact quantified direct STARK currently requires `between` lower-bound subtraction to fit i64".to_string(),
                    )
                })?;
                let upper_diff = value.value.checked_sub(upper.value).ok_or_else(|| {
                    VmError::UnsupportedProof(
                        "compact quantified direct STARK currently requires `between` upper-bound subtraction to fit i64".to_string(),
                    )
                })?;
                ordered_aux.insert(
                    lower_aux.diff_register,
                    OrderedWitness {
                        diff: lower_diff,
                        inverse: if lower_diff == 0 {
                            FieldElement::zero()
                        } else {
                            field_from_word(lower_diff).inverse()
                        },
                    },
                );
                ordered_aux.insert(
                    upper_aux.diff_register,
                    OrderedWitness {
                        diff: upper_diff,
                        inverse: if upper_diff == 0 {
                            FieldElement::zero()
                        } else {
                            field_from_word(upper_diff).inverse()
                        },
                    },
                );
                Ok(EvaluatedBoolean {
                    value: value.value >= lower.value && value.value <= upper.value,
                    status,
                })
            }
            CompactQuantifiedPredicate::And(left, right) => {
                let left = self.eval_compact_quantified_predicate(
                    left,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                let right = self.eval_compact_quantified_predicate(
                    right,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if left.status != STATUS_AVAILABLE || !left.value {
                    Ok(left)
                } else {
                    Ok(right)
                }
            }
            CompactQuantifiedPredicate::Or(left, right) => {
                let left = self.eval_compact_quantified_predicate(
                    left,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                let right = self.eval_compact_quantified_predicate(
                    right,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if left.status != STATUS_AVAILABLE || left.value {
                    Ok(left)
                } else {
                    Ok(right)
                }
            }
            CompactQuantifiedPredicate::Not(inner) => {
                let inner = self.eval_compact_quantified_predicate(
                    inner,
                    item_index,
                    input_values,
                    definitions,
                    arithmetic_nodes,
                    equality_aux,
                    ordered_aux,
                )?;
                if inner.status != STATUS_AVAILABLE {
                    Ok(inner)
                } else {
                    Ok(EvaluatedBoolean {
                        value: !inner.value,
                        status: STATUS_AVAILABLE,
                    })
                }
            }
        }
    }

    fn eval_compact_arithmetic_expr(
        &self,
        expr: &CompactArithmeticExpr,
        item_index: usize,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
        _arithmetic_nodes: &[EvaluatedArithmetic],
    ) -> Result<EvaluatedArithmetic> {
        match expr {
            CompactArithmeticExpr::Number(value) => Ok(EvaluatedArithmetic {
                value: *value,
                status: STATUS_AVAILABLE,
            }),
            CompactArithmeticExpr::Value(value_ref) => {
                let value = match &value_ref.source {
                    CompactValueSource::InputRegister(register) => {
                        input_values[self.input_index_for_register(*register)?]
                    }
                    CompactValueSource::DefinitionValue(index) => definitions
                        .get(*index)
                        .map(|value| value.value)
                        .ok_or_else(|| {
                            VmError::InvalidConfig(format!(
                                "missing definition value {index} in compact quantified witness"
                            ))
                        })?,
                    CompactValueSource::ItemRegisters(registers) => {
                        input_values[self.input_index_for_register(registers[item_index])?]
                    }
                };
                Ok(EvaluatedArithmetic {
                    value,
                    status: self.evaluate_compact_availability(
                        &value_ref.availability,
                        item_index,
                        input_values,
                        definitions,
                    )?,
                })
            }
        }
    }

    fn evaluate_compact_availability(
        &self,
        availability: &CompactAvailability,
        item_index: usize,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
    ) -> Result<Word> {
        match availability {
            CompactAvailability::Always => Ok(STATUS_AVAILABLE),
            CompactAvailability::Flags { presence, unknown } => {
                if let Some(unknown) = unknown {
                    if self.compact_availability_source_value(
                        unknown,
                        item_index,
                        input_values,
                        definitions,
                    )? != 0
                    {
                        return Ok(STATUS_UNKNOWN);
                    }
                }
                if let Some(presence) = presence {
                    if self.compact_availability_source_value(
                        presence,
                        item_index,
                        input_values,
                        definitions,
                    )? == 0
                    {
                        return Ok(STATUS_MISSING);
                    }
                }
                Ok(STATUS_AVAILABLE)
            }
            CompactAvailability::Status(source) => self.compact_availability_source_value(
                source,
                item_index,
                input_values,
                definitions,
            ),
        }
    }

    fn compact_availability_source_value(
        &self,
        source: &CompactAvailabilitySource,
        item_index: usize,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
    ) -> Result<Word> {
        match source {
            CompactAvailabilitySource::InputRegister(register) => {
                Ok(input_values[self.input_index_for_register(*register)?])
            }
            CompactAvailabilitySource::DefinitionStatus(index) => definitions
                .get(*index)
                .map(|value| value.status)
                .ok_or_else(|| {
                    VmError::InvalidConfig(format!(
                        "missing definition status {index} in compact quantified witness"
                    ))
                }),
            CompactAvailabilitySource::ItemRegisters(registers) => {
                Ok(input_values[self.input_index_for_register(registers[item_index])?])
            }
        }
    }

    fn evaluate_availability(
        &self,
        availability: &DecisionAvailability,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
    ) -> Result<Word> {
        match availability {
            DecisionAvailability::Always => Ok(STATUS_AVAILABLE),
            DecisionAvailability::Flags { presence, unknown } => {
                if let Some(unknown) = unknown {
                    if self.availability_source_value(unknown, input_values, definitions)? != 0 {
                        return Ok(STATUS_UNKNOWN);
                    }
                }
                if let Some(presence) = presence {
                    if self.availability_source_value(presence, input_values, definitions)? == 0 {
                        return Ok(STATUS_MISSING);
                    }
                }
                Ok(STATUS_AVAILABLE)
            }
            DecisionAvailability::Status(source) => match source {
                DecisionAvailabilitySource::InputCell(address) => {
                    Ok(input_values[self.input_index_for_address(*address)?])
                }
                DecisionAvailabilitySource::DefinitionStatus(index) => definitions
                    .get(*index)
                    .map(|value| value.status)
                    .ok_or_else(|| {
                        VmError::InvalidConfig(format!(
                            "missing definition status {index} in direct cDMN witness"
                        ))
                    }),
                DecisionAvailabilitySource::BoundField { .. } => Err(VmError::InvalidConfig(
                    "bound-field status sources must be concretized before direct witness evaluation"
                        .to_string(),
                )),
            },
        }
    }

    fn availability_source_value(
        &self,
        source: &DecisionAvailabilitySource,
        input_values: &[Word],
        definitions: &[EvaluatedArithmetic],
    ) -> Result<Word> {
        match source {
            DecisionAvailabilitySource::InputCell(address) => {
                Ok(input_values[self.input_index_for_address(*address)?])
            }
            DecisionAvailabilitySource::DefinitionStatus(index) => definitions
                .get(*index)
                .map(|value| value.status)
                .ok_or_else(|| {
                    VmError::InvalidConfig(format!(
                        "missing definition status {index} in direct cDMN witness"
                    ))
                }),
            DecisionAvailabilitySource::BoundField { .. } => Err(VmError::InvalidConfig(
                "bound-field availability sources must be concretized before direct witness evaluation"
                    .to_string(),
            )),
        }
    }

    fn input_index_for_address(&self, address: u8) -> Result<usize> {
        self.input_registers
            .iter()
            .position(|input| input.address == address)
            .ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "missing direct input cell {address} in cDMN witness"
                ))
            })
    }

    fn input_index_for_register(&self, register: usize) -> Result<usize> {
        self.input_registers
            .iter()
            .position(|input| input.register == register)
            .ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "missing direct input register {register} in cDMN witness"
                ))
            })
    }

    fn materialize_trace_rows(&self, witness: &DirectDecisionWitness) -> Vec<Vec<FieldElement>> {
        let mut base_row = vec![FieldElement::zero(); self.register_count];
        base_row[self.always_available_register] = field_from_word(STATUS_AVAILABLE);

        for (input, value) in self.input_registers.iter().zip(witness.input_values.iter()) {
            base_row[input.register] = field_from_word(*value);
        }

        for ((node, evaluated), bits) in self
            .arithmetic_nodes
            .iter()
            .zip(witness.arithmetic_nodes.iter())
            .zip(witness.arithmetic_bits.iter())
        {
            base_row[node.value_register] = field_from_word(evaluated.value);
            base_row[node.status_register] = field_from_word(evaluated.status);
            for (register, bit) in node.bit_registers.iter().zip(bits.iter()) {
                base_row[*register] = field_from_bool(*bit);
            }
        }

        for ((definition, evaluated), bits) in self
            .definitions
            .iter()
            .zip(witness.definitions.iter())
            .zip(witness.definition_bits.iter())
        {
            base_row[definition.value_register] = field_from_word(evaluated.value);
            base_row[definition.status_register] = field_from_word(evaluated.status);
            for (register, bit) in definition.bit_registers.iter().zip(bits.iter()) {
                base_row[*register] = field_from_bool(*bit);
            }
        }

        for (node, evaluated) in self.boolean_nodes.iter().zip(witness.boolean_nodes.iter()) {
            base_row[node.value_register] = field_from_bool(evaluated.value);
            base_row[node.status_register] = field_from_word(evaluated.status);
            match &node.kind {
                BooleanNodeKind::Comparison {
                    equality_aux,
                    ordered_aux,
                    ..
                } => {
                    if let Some(aux) = equality_aux {
                        if let Some(witness_aux) = witness.equality_aux.get(&aux.delta_register) {
                            base_row[aux.delta_register] = field_from_i128(witness_aux.delta);
                            base_row[aux.inverse_register] = witness_aux.inverse;
                        }
                    }
                    if let Some(aux) = ordered_aux {
                        if let Some(witness_aux) = witness.ordered_aux.get(&aux.diff_register) {
                            base_row[aux.diff_register] = field_from_word(witness_aux.diff);
                            base_row[aux.inverse_register] = witness_aux.inverse;
                            for (register, bit) in aux
                                .diff_bits
                                .iter()
                                .zip(signed_bits(witness_aux.diff).iter())
                            {
                                base_row[*register] = field_from_bool(*bit);
                            }
                        }
                    }
                }
                BooleanNodeKind::Between {
                    lower_aux,
                    upper_aux,
                    ..
                } => {
                    if let Some(witness_aux) = witness.ordered_aux.get(&lower_aux.diff_register) {
                        base_row[lower_aux.diff_register] = field_from_word(witness_aux.diff);
                        base_row[lower_aux.inverse_register] = witness_aux.inverse;
                        for (register, bit) in lower_aux
                            .diff_bits
                            .iter()
                            .zip(signed_bits(witness_aux.diff).iter())
                        {
                            base_row[*register] = field_from_bool(*bit);
                        }
                    }
                    if let Some(witness_aux) = witness.ordered_aux.get(&upper_aux.diff_register) {
                        base_row[upper_aux.diff_register] = field_from_word(witness_aux.diff);
                        base_row[upper_aux.inverse_register] = witness_aux.inverse;
                        for (register, bit) in upper_aux
                            .diff_bits
                            .iter()
                            .zip(signed_bits(witness_aux.diff).iter())
                        {
                            base_row[*register] = field_from_bool(*bit);
                        }
                    }
                }
                BooleanNodeKind::Quantified {
                    item_predicates, ..
                } => {
                    for predicate in item_predicates {
                        self.write_quantified_predicate_aux_rows(predicate, witness, &mut base_row);
                    }
                }
                _ => {}
            }
        }

        for (stage, stage_witness) in self
            .compact_quantified_stages
            .iter()
            .zip(witness.compact_quantified_stages.iter())
        {
            base_row[stage.result_value_register] = field_from_bool(stage_witness.result.value);
            base_row[stage.result_status_register] = field_from_word(stage_witness.result.status);
        }

        base_row[self.final_status_register] = field_from_word(witness.final_status);
        base_row[self.final_reason_register] = field_from_word(witness.final_reason);

        let mut rows = vec![base_row.clone(); witness.trace_rows];
        if let Some(row_counter_register) = self.row_counter_register {
            for (index, row) in rows.iter_mut().enumerate() {
                row[row_counter_register] = field_from_word(index as Word);
            }
        }
        for (stage, stage_witness) in self
            .compact_quantified_stages
            .iter()
            .zip(witness.compact_quantified_stages.iter())
        {
            for local_row in 0..stage.row_count {
                let row_index = stage.start_row + local_row;
                let row = &mut rows[row_index];
                let acc = stage_witness.accumulator_rows[local_row];
                row[stage.accumulator_value_register] = field_from_bool(acc.value);
                row[stage.accumulator_status_register] = field_from_word(acc.status);
                if let Some(equality_aux) = stage_witness.equality_aux_rows.get(local_row) {
                    for (register, witness_aux) in equality_aux {
                        row[*register] = field_from_i128(witness_aux.delta);
                        let aux = find_equality_aux_by_register(&stage.predicate, *register)
                            .expect("compact equality aux");
                        row[aux.inverse_register] = witness_aux.inverse;
                    }
                }
                if let Some(ordered_aux) = stage_witness.ordered_aux_rows.get(local_row) {
                    for (register, witness_aux) in ordered_aux {
                        row[*register] = field_from_word(witness_aux.diff);
                        let aux = find_ordered_aux_by_register(&stage.predicate, *register)
                            .expect("compact ordered aux");
                        row[aux.inverse_register] = witness_aux.inverse;
                        for (bit_register, bit) in aux
                            .diff_bits
                            .iter()
                            .zip(signed_bits(witness_aux.diff).iter())
                        {
                            row[*bit_register] = field_from_bool(*bit);
                        }
                    }
                }
            }
        }
        rows
    }

    fn write_quantified_predicate_aux_rows(
        &self,
        predicate: &QuantifiedPredicate,
        witness: &DirectDecisionWitness,
        row: &mut [FieldElement],
    ) {
        match predicate {
            QuantifiedPredicate::Comparison {
                equality_aux,
                ordered_aux,
                ..
            } => {
                if let Some(aux) = equality_aux {
                    if let Some(witness_aux) = witness.equality_aux.get(&aux.delta_register) {
                        row[aux.delta_register] = field_from_i128(witness_aux.delta);
                        row[aux.inverse_register] = witness_aux.inverse;
                    }
                }
                if let Some(aux) = ordered_aux {
                    if let Some(witness_aux) = witness.ordered_aux.get(&aux.diff_register) {
                        row[aux.diff_register] = field_from_word(witness_aux.diff);
                        row[aux.inverse_register] = witness_aux.inverse;
                        for (register, bit) in aux
                            .diff_bits
                            .iter()
                            .zip(signed_bits(witness_aux.diff).iter())
                        {
                            row[*register] = field_from_bool(*bit);
                        }
                    }
                }
            }
            QuantifiedPredicate::Between {
                lower_aux,
                upper_aux,
                ..
            } => {
                if let Some(witness_aux) = witness.ordered_aux.get(&lower_aux.diff_register) {
                    row[lower_aux.diff_register] = field_from_word(witness_aux.diff);
                    row[lower_aux.inverse_register] = witness_aux.inverse;
                    for (register, bit) in lower_aux
                        .diff_bits
                        .iter()
                        .zip(signed_bits(witness_aux.diff).iter())
                    {
                        row[*register] = field_from_bool(*bit);
                    }
                }
                if let Some(witness_aux) = witness.ordered_aux.get(&upper_aux.diff_register) {
                    row[upper_aux.diff_register] = field_from_word(witness_aux.diff);
                    row[upper_aux.inverse_register] = witness_aux.inverse;
                    for (register, bit) in upper_aux
                        .diff_bits
                        .iter()
                        .zip(signed_bits(witness_aux.diff).iter())
                    {
                        row[*register] = field_from_bool(*bit);
                    }
                }
            }
            QuantifiedPredicate::And(left, right) | QuantifiedPredicate::Or(left, right) => {
                self.write_quantified_predicate_aux_rows(left, witness, row);
                self.write_quantified_predicate_aux_rows(right, witness, row);
            }
            QuantifiedPredicate::Not(inner) => {
                self.write_quantified_predicate_aux_rows(inner, witness, row);
            }
            QuantifiedPredicate::Constant(_) | QuantifiedPredicate::Availability { .. } => {}
        }
    }
}

struct DirectDecisionAir {
    circuit: DirectDecisionCircuit,
}

impl DirectDecisionAir {
    fn new(circuit: DirectDecisionCircuit) -> Self {
        Self { circuit }
    }

    fn register_count(&self) -> usize {
        self.circuit.register_count
    }

    fn transition_degree_bound_for(constraints: &[MPolynomial]) -> usize {
        constraints
            .iter()
            .flat_map(|constraint| constraint.dictionary.keys())
            .map(|powers| powers.iter().sum())
            .max()
            .unwrap_or(1)
            .max(2)
    }

    fn boundary_constraints(
        &self,
        claim: &CdmnDecisionStarkDirectClaim,
    ) -> Result<Vec<(usize, usize, FieldElement)>> {
        let projected: BTreeMap<u8, Word> = claim.projected_inputs.iter().copied().collect();
        let mut boundary = Vec::new();
        for input in &self.circuit.input_registers {
            let value = projected.get(&input.address).copied().ok_or_else(|| {
                VmError::InvalidConfig(format!(
                    "direct cDMN claim is missing projected input cell {}",
                    input.address
                ))
            })?;
            boundary.push((0, input.register, field_from_word(value)));
        }
        boundary.push((
            0,
            self.circuit.final_status_register,
            field_from_word(claim.status),
        ));
        boundary.push((
            0,
            self.circuit.final_reason_register,
            field_from_word(claim.reason_code),
        ));
        if let Some(row_counter_register) = self.circuit.row_counter_register {
            boundary.push((0, row_counter_register, field_from_word(0)));
            boundary.push((
                self.circuit.trace_rows() - 1,
                row_counter_register,
                field_from_word((self.circuit.trace_rows() - 1) as Word),
            ));
        }
        boundary.push((
            self.circuit.trace_rows() - 1,
            self.circuit.final_status_register,
            field_from_word(claim.status),
        ));
        boundary.push((
            self.circuit.trace_rows() - 1,
            self.circuit.final_reason_register,
            field_from_word(claim.reason_code),
        ));
        Ok(boundary)
    }

    fn transition_constraints(&self) -> Vec<MPolynomial> {
        let variables = MPolynomial::variables(1 + 2 * self.register_count());
        let current = variables[1..1 + self.register_count()].to_vec();
        let next = variables[1 + self.register_count()..].to_vec();

        let one = mp_constant(1);
        let mut constraints = Vec::new();
        let mut emitted_equality_aux = HashSet::new();
        let mut emitted_ordered_aux = HashSet::new();
        let mutable_registers = self.circuit.compact_mutable_registers();

        for register in 0..self.register_count() {
            if !mutable_registers.contains(&register) {
                constraints.push(next[register].clone() - current[register].clone());
            }
        }
        constraints.push(current[self.circuit.always_available_register].clone());
        if let Some(row_counter_register) = self.circuit.row_counter_register {
            constraints.push(
                next[row_counter_register].clone()
                    - current[row_counter_register].clone()
                    - one.clone(),
            );
        }

        for &address in &self.circuit.input_kinds.boolean_flags {
            if let Some(register) = self.circuit.input_register_by_address.get(&address) {
                let value = current[*register].clone();
                constraints.push(value.clone() * (value - one.clone()));
            }
        }
        for &address in &self.circuit.input_kinds.status_cells {
            if let Some(register) = self.circuit.input_register_by_address.get(&address) {
                let value = current[*register].clone();
                constraints.push(status_range_constraint(value));
            }
        }

        for node in &self.circuit.arithmetic_nodes {
            let value = current[node.value_register].clone();
            let status = current[node.status_register].clone();
            let left_value = self.arithmetic_source_value(&node.left, &current);
            let left_status = self.arithmetic_source_status(&node.left, &current);
            let right_value = self.arithmetic_source_value(&node.right, &current);
            let right_status = self.arithmetic_source_status(&node.right, &current);
            let expected_status = combine_status_poly(left_status, right_status);
            let available = status_selector(&status, STATUS_AVAILABLE);
            let operation = match node.op {
                ArithmeticOp::Add => left_value + right_value,
                ArithmeticOp::Sub => left_value - right_value,
                ArithmeticOp::Mul => left_value * right_value,
            };
            constraints.push(status.clone() - expected_status);
            constraints.push(value.clone() - available.clone() * operation);
            constraints.push(status_range_constraint(status));
            constraints.extend(range_constraints(value, &node.bit_registers, &current));
        }

        for definition in &self.circuit.definitions {
            let value = current[definition.value_register].clone();
            let status = current[definition.status_register].clone();
            if definition.materialized {
                constraints.push(
                    value.clone() - self.arithmetic_source_value(&definition.source, &current),
                );
                constraints.push(
                    status.clone() - self.arithmetic_source_status(&definition.source, &current),
                );
                constraints.push(status_range_constraint(status));
                constraints.extend(range_constraints(
                    value,
                    &definition.bit_registers,
                    &current,
                ));
            }
        }

        for node in &self.circuit.boolean_nodes {
            let value = current[node.value_register].clone();
            let status = current[node.status_register].clone();
            constraints.push(value.clone() * (value.clone() - one.clone()));
            constraints.push(status_range_constraint(status.clone()));
            match &node.kind {
                BooleanNodeKind::Constant(expected) => {
                    constraints.push(status.clone());
                    constraints.push(value - mp_constant(if *expected { 1 } else { 0 }));
                }
                BooleanNodeKind::CountAtLeast {
                    register,
                    threshold,
                    max_items,
                } => {
                    let count = current[*register].clone();
                    let in_domain = count_in_domain_poly(&count, *max_items);
                    let expected = count_at_least_poly(&count, *threshold, *max_items);
                    constraints.push(status);
                    constraints.push(in_domain * (value - expected));
                }
                BooleanNodeKind::Comparison {
                    lhs,
                    op,
                    rhs,
                    equality_aux,
                    ordered_aux,
                } => {
                    let lhs_value = self.arithmetic_source_value(lhs, &current);
                    let lhs_status = self.arithmetic_source_status(lhs, &current);
                    let rhs_value = self.arithmetic_source_value(rhs, &current);
                    let rhs_status = self.arithmetic_source_status(rhs, &current);
                    let expected_status = combine_status_poly(lhs_status, rhs_status);
                    let available = status_selector(&status, STATUS_AVAILABLE);
                    constraints.push(status.clone() - expected_status);
                    match op {
                        ComparisonOp::Eq | ComparisonOp::Ne => {
                            let aux = equality_aux.as_ref().expect("equality aux");
                            let delta = current[aux.delta_register].clone();
                            let inverse = current[aux.inverse_register].clone();
                            let is_zero = one.clone() - delta.clone() * inverse.clone();
                            if emitted_equality_aux.insert(aux.delta_register) {
                                constraints.push(
                                    delta.clone() - available.clone() * (lhs_value - rhs_value),
                                );
                                constraints.push(delta.clone() * is_zero.clone());
                                constraints.push(
                                    delta.clone() * inverse - (one.clone() - is_zero.clone()),
                                );
                            }
                            let expected = match op {
                                ComparisonOp::Eq => available.clone() * is_zero,
                                ComparisonOp::Ne => available.clone() * (one.clone() - is_zero),
                                _ => unreachable!(),
                            };
                            constraints.push(value - expected);
                        }
                        ComparisonOp::Lt
                        | ComparisonOp::Le
                        | ComparisonOp::Gt
                        | ComparisonOp::Ge => {
                            let aux = ordered_aux.as_ref().expect("ordered aux");
                            let diff = current[aux.diff_register].clone();
                            let inverse = current[aux.inverse_register].clone();
                            let is_zero = one.clone() - diff.clone() * inverse.clone();
                            let sign = current[*aux.diff_bits.last().expect("sign bit")].clone();
                            if emitted_ordered_aux.insert(aux.diff_register) {
                                constraints.push(
                                    diff.clone() - available.clone() * (lhs_value - rhs_value),
                                );
                                constraints.push(diff.clone() * is_zero.clone());
                                constraints
                                    .push(diff.clone() * inverse - (one.clone() - is_zero.clone()));
                                constraints.extend(range_constraints(
                                    diff.clone(),
                                    &aux.diff_bits,
                                    &current,
                                ));
                            }
                            let expected = match op {
                                ComparisonOp::Lt => available.clone() * sign.clone(),
                                ComparisonOp::Le => {
                                    available.clone() * (sign.clone() + is_zero.clone())
                                }
                                ComparisonOp::Gt => {
                                    available.clone()
                                        * (one.clone() - sign.clone() - is_zero.clone())
                                }
                                ComparisonOp::Ge => {
                                    available.clone() * (one.clone() - sign.clone())
                                }
                                _ => unreachable!(),
                            };
                            constraints.push(value - expected);
                        }
                    }
                }
                BooleanNodeKind::Availability {
                    kind,
                    value: source,
                } => {
                    let source_status = self.arithmetic_source_status(source, &current);
                    let expected = match kind {
                        AvailabilityPredicateKind::Present => {
                            status_selector(&source_status, STATUS_AVAILABLE)
                        }
                        AvailabilityPredicateKind::Missing => {
                            status_selector(&source_status, STATUS_MISSING)
                        }
                        AvailabilityPredicateKind::Unknown => {
                            status_selector(&source_status, STATUS_UNKNOWN)
                        }
                    };
                    constraints.push(status);
                    constraints.push(value - expected);
                }
                BooleanNodeKind::Between {
                    value: between_value,
                    lower,
                    upper,
                    lower_aux,
                    upper_aux,
                } => {
                    let between_value_value = self.arithmetic_source_value(between_value, &current);
                    let between_value_status =
                        self.arithmetic_source_status(between_value, &current);
                    let lower_value = self.arithmetic_source_value(lower, &current);
                    let lower_status = self.arithmetic_source_status(lower, &current);
                    let upper_value = self.arithmetic_source_value(upper, &current);
                    let upper_status = self.arithmetic_source_status(upper, &current);
                    let expected_status = combine_status_poly(
                        between_value_status,
                        combine_status_poly(lower_status, upper_status),
                    );
                    let available = status_selector(&status, STATUS_AVAILABLE);
                    constraints.push(status.clone() - expected_status);

                    let lower_diff = current[lower_aux.diff_register].clone();
                    let lower_inverse = current[lower_aux.inverse_register].clone();
                    let lower_is_zero = one.clone() - lower_diff.clone() * lower_inverse.clone();
                    let lower_sign =
                        current[*lower_aux.diff_bits.last().expect("lower sign bit")].clone();
                    if emitted_ordered_aux.insert(lower_aux.diff_register) {
                        constraints.push(
                            lower_diff.clone()
                                - available.clone() * (between_value_value.clone() - lower_value),
                        );
                        constraints.push(lower_diff.clone() * lower_is_zero.clone());
                        constraints.push(
                            lower_diff.clone() * lower_inverse
                                - (one.clone() - lower_is_zero.clone()),
                        );
                        constraints.extend(range_constraints(
                            lower_diff.clone(),
                            &lower_aux.diff_bits,
                            &current,
                        ));
                    }

                    let upper_diff = current[upper_aux.diff_register].clone();
                    let upper_inverse = current[upper_aux.inverse_register].clone();
                    let upper_is_zero = one.clone() - upper_diff.clone() * upper_inverse.clone();
                    let upper_sign =
                        current[*upper_aux.diff_bits.last().expect("upper sign bit")].clone();
                    if emitted_ordered_aux.insert(upper_aux.diff_register) {
                        constraints.push(
                            upper_diff.clone()
                                - available.clone() * (between_value_value - upper_value),
                        );
                        constraints.push(upper_diff.clone() * upper_is_zero.clone());
                        constraints.push(
                            upper_diff.clone() * upper_inverse
                                - (one.clone() - upper_is_zero.clone()),
                        );
                        constraints.extend(range_constraints(
                            upper_diff.clone(),
                            &upper_aux.diff_bits,
                            &current,
                        ));
                    }

                    let ge_lower = one.clone() - lower_sign;
                    let le_upper = upper_sign + upper_is_zero;
                    constraints.push(value - available * ge_lower * le_upper);
                }
                BooleanNodeKind::And { left, right } => {
                    let left_value =
                        current[self.circuit.boolean_nodes[*left].value_register].clone();
                    let left_status =
                        current[self.circuit.boolean_nodes[*left].status_register].clone();
                    let right_value =
                        current[self.circuit.boolean_nodes[*right].value_register].clone();
                    let right_status =
                        current[self.circuit.boolean_nodes[*right].status_register].clone();
                    let left_available = status_selector(&left_status, STATUS_AVAILABLE);
                    let take_left = one.clone() - left_available.clone() * left_value.clone();
                    let take_right = left_available * left_value;
                    constraints.push(
                        status.clone()
                            - (take_left.clone() * left_status + take_right.clone() * right_status),
                    );
                    constraints.push(
                        value
                            - (take_left
                                * current[self.circuit.boolean_nodes[*left].value_register]
                                    .clone()
                                + take_right * right_value),
                    );
                }
                BooleanNodeKind::Or { left, right } => {
                    let left_value =
                        current[self.circuit.boolean_nodes[*left].value_register].clone();
                    let left_status =
                        current[self.circuit.boolean_nodes[*left].status_register].clone();
                    let right_value =
                        current[self.circuit.boolean_nodes[*right].value_register].clone();
                    let right_status =
                        current[self.circuit.boolean_nodes[*right].status_register].clone();
                    let left_available = status_selector(&left_status, STATUS_AVAILABLE);
                    let take_right = left_available.clone() * (one.clone() - left_value.clone());
                    let take_left = one.clone() - take_right.clone();
                    constraints.push(
                        status.clone()
                            - (take_left.clone() * left_status + take_right.clone() * right_status),
                    );
                    constraints.push(
                        value
                            - (take_left
                                * current[self.circuit.boolean_nodes[*left].value_register]
                                    .clone()
                                + take_right * right_value),
                    );
                }
                BooleanNodeKind::Not { inner } => {
                    let inner_value =
                        current[self.circuit.boolean_nodes[*inner].value_register].clone();
                    let inner_status =
                        current[self.circuit.boolean_nodes[*inner].status_register].clone();
                    let inner_available = status_selector(&inner_status, STATUS_AVAILABLE);
                    constraints.push(status - inner_status);
                    constraints.push(
                        value
                            - (inner_available.clone() * (one.clone() - inner_value.clone())
                                + (one.clone() - inner_available) * inner_value),
                    );
                }
                BooleanNodeKind::Quantified {
                    kind,
                    count_register,
                    max_items,
                    item_predicates,
                } => {
                    let count = current[*count_register].clone();
                    let in_domain = count_in_domain_poly(&count, *max_items);
                    let mut acc_status = mp_constant(match kind {
                        QuantifierKind::ForAll => STATUS_AVAILABLE.into(),
                        QuantifierKind::Exists => STATUS_AVAILABLE.into(),
                    });
                    let mut acc_value = mp_constant(match kind {
                        QuantifierKind::ForAll => 1,
                        QuantifierKind::Exists => 0,
                    });
                    for (index, predicate) in item_predicates.iter().enumerate() {
                        let active = in_domain.clone()
                            * count_at_least_poly(
                                &count,
                                u8::try_from(index + 1).expect("quantifier threshold fits in u8"),
                                *max_items,
                            );
                        let (item_status, item_value) = self.quantified_predicate_polys(
                            predicate,
                            &current,
                            &mut constraints,
                            &mut emitted_equality_aux,
                            &mut emitted_ordered_aux,
                        );
                        let term_value = match kind {
                            QuantifierKind::ForAll => {
                                active.clone() * item_value + (one.clone() - active.clone())
                            }
                            QuantifierKind::Exists => active.clone() * item_value,
                        };
                        let term_status = active * item_status;
                        (acc_status, acc_value) = match kind {
                            QuantifierKind::ForAll => {
                                boolean_and_poly(acc_status, acc_value, term_status, term_value)
                            }
                            QuantifierKind::Exists => {
                                boolean_or_poly(acc_status, acc_value, term_status, term_value)
                            }
                        };
                    }
                    constraints.push(status - acc_status);
                    constraints.push(value - acc_value);
                }
            }
        }

        if let Some(row_counter_register) = self.circuit.row_counter_register {
            let counter = current[row_counter_register].clone();
            for stage in &self.circuit.compact_quantified_stages {
                self.compact_stage_constraints(
                    stage,
                    &counter,
                    &current,
                    &next,
                    &mut constraints,
                    &mut emitted_equality_aux,
                    &mut emitted_ordered_aux,
                );
            }
        }

        let mut previous_status = mp_constant(1);
        let mut previous_reason = mp_constant(0);
        for stage in &self.circuit.stages {
            let bound_ok = stage
                .bound_checks
                .iter()
                .fold(one.clone(), |acc, bound_check| {
                    let register =
                        self.circuit.input_register_by_address[&bound_check.count_memory];
                    acc * count_in_domain_poly(&current[register].clone(), bound_check.max_items)
                });
            let (expr_value, expr_status) = match stage.expression {
                ConstraintExpression::BooleanNode(index) => (
                    current[self.circuit.boolean_nodes[index].value_register].clone(),
                    current[self.circuit.boolean_nodes[index].status_register].clone(),
                ),
                ConstraintExpression::CompactQuantified(index) => (
                    current[self.circuit.compact_quantified_stages[index].result_value_register]
                        .clone(),
                    current[self.circuit.compact_quantified_stages[index].result_status_register]
                        .clone(),
                ),
            };
            let expr_available = status_selector(&expr_status, STATUS_AVAILABLE);

            let freeze = one.clone() - previous_status.clone();
            let fail_bound = previous_status.clone() * (one.clone() - bound_ok.clone());
            let fail_insuff =
                previous_status.clone() * bound_ok.clone() * (one.clone() - expr_available.clone());
            let fail_reason = previous_status.clone()
                * bound_ok.clone()
                * expr_available.clone()
                * (one.clone() - expr_value.clone());
            let pass = previous_status.clone() * bound_ok * expr_available * expr_value;

            previous_reason = freeze * previous_reason.clone()
                + (fail_bound + fail_insuff)
                    * mp_constant(i128::from(stage.insufficient_data_reason.code))
                + fail_reason * mp_constant(i128::from(stage.reason.code));
            previous_status = pass;
        }

        constraints.push(current[self.circuit.final_status_register].clone() - previous_status);
        constraints.push(current[self.circuit.final_reason_register].clone() - previous_reason);

        constraints
    }

    fn arithmetic_source_value(
        &self,
        source: &ArithmeticSource,
        current: &[MPolynomial],
    ) -> MPolynomial {
        match source {
            ArithmeticSource::Constant(value) => mp_constant(i128::from(*value)),
            ArithmeticSource::Input { register, .. } => current[*register].clone(),
            ArithmeticSource::DefinitionValue(index) => {
                current[self.circuit.definitions[*index].value_register].clone()
            }
            ArithmeticSource::Node(index) => {
                current[self.circuit.arithmetic_nodes[*index].value_register].clone()
            }
        }
    }

    fn arithmetic_source_status(
        &self,
        source: &ArithmeticSource,
        current: &[MPolynomial],
    ) -> MPolynomial {
        match source {
            ArithmeticSource::Constant(_) => mp_constant(STATUS_AVAILABLE.into()),
            ArithmeticSource::Input {
                register: _,
                availability,
            } => self.availability_status(availability, current),
            ArithmeticSource::DefinitionValue(index) => {
                current[self.circuit.definitions[*index].status_register].clone()
            }
            ArithmeticSource::Node(index) => {
                current[self.circuit.arithmetic_nodes[*index].status_register].clone()
            }
        }
    }

    fn availability_status(
        &self,
        availability: &DecisionAvailability,
        current: &[MPolynomial],
    ) -> MPolynomial {
        let one = mp_constant(1);
        match availability {
            DecisionAvailability::Always => mp_constant(STATUS_AVAILABLE.into()),
            DecisionAvailability::Flags { presence, unknown } => {
                let unknown_expr = unknown
                    .as_ref()
                    .map(|source| self.availability_source_poly(source, current))
                    .unwrap_or_else(MPolynomial::zero);
                let presence_expr = presence
                    .as_ref()
                    .map(|source| self.availability_source_poly(source, current))
                    .unwrap_or_else(|| one.clone());
                unknown_expr.clone() * mp_constant(2)
                    + (one.clone() - unknown_expr) * (one - presence_expr)
            }
            DecisionAvailability::Status(source) => match source {
                DecisionAvailabilitySource::InputCell(address) => {
                    current[self.circuit.input_register_by_address[address]].clone()
                }
                DecisionAvailabilitySource::DefinitionStatus(index) => {
                    current[self.circuit.definitions[*index].status_register].clone()
                }
                DecisionAvailabilitySource::BoundField { .. } => {
                    panic!(
                        "bound-field status sources must be concretized before direct AIR construction"
                    )
                }
            },
        }
    }

    fn availability_source_poly(
        &self,
        source: &DecisionAvailabilitySource,
        current: &[MPolynomial],
    ) -> MPolynomial {
        match source {
            DecisionAvailabilitySource::InputCell(address) => {
                current[self.circuit.input_register_by_address[address]].clone()
            }
            DecisionAvailabilitySource::DefinitionStatus(index) => {
                current[self.circuit.definitions[*index].status_register].clone()
            }
            DecisionAvailabilitySource::BoundField { .. } => {
                panic!(
                    "bound-field availability sources must be concretized before direct AIR construction"
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_stage_constraints(
        &self,
        stage: &CompactQuantifiedStage,
        counter: &MPolynomial,
        current: &[MPolynomial],
        next: &[MPolynomial],
        constraints: &mut Vec<MPolynomial>,
        emitted_equality_aux: &mut HashSet<usize>,
        emitted_ordered_aux: &mut HashSet<usize>,
    ) {
        let one = mp_constant(1);
        let trace_domain: Vec<Word> = (0..self.circuit.trace_rows())
            .map(|row| i64::try_from(row).expect("trace row fits in i64"))
            .collect();
        let row_selectors: Vec<MPolynomial> = (0..stage.row_count)
            .map(|local_row| {
                lagrange_selector(
                    counter,
                    &trace_domain,
                    i64::try_from(stage.start_row + local_row).expect("stage row fits in i64"),
                )
            })
            .collect();
        let item_row_selectors = &row_selectors[..usize::from(stage.collection.max_items)];
        let item_phase_selector = item_row_selectors
            .iter()
            .cloned()
            .fold(MPolynomial::zero(), |acc, selector| acc + selector);
        let stage_selector = row_selectors
            .iter()
            .cloned()
            .fold(MPolynomial::zero(), |acc, selector| acc + selector);
        let count_register = self.circuit.input_register_by_address[&stage.collection.count_memory];
        let count = current[count_register].clone();
        let in_domain = count_in_domain_poly(&count, stage.collection.max_items);
        let initial_value = mp_constant(match stage.kind {
            QuantifierKind::ForAll => 1,
            QuantifierKind::Exists => 0,
        });
        let initial_status = mp_constant(STATUS_AVAILABLE.into());
        let current_acc_value = current[stage.accumulator_value_register].clone();
        let current_acc_status = current[stage.accumulator_status_register].clone();

        constraints.push(
            stage_selector.clone()
                * current_acc_value.clone()
                * (current_acc_value.clone() - one.clone()),
        );
        constraints
            .push(stage_selector.clone() * status_range_constraint(current_acc_status.clone()));
        constraints
            .push(row_selectors[0].clone() * (current_acc_value.clone() - initial_value.clone()));
        constraints
            .push(row_selectors[0].clone() * (current_acc_status.clone() - initial_status.clone()));

        let (item_status, item_value) = self.compact_quantified_predicate_polys(
            &stage.predicate,
            item_row_selectors,
            &item_phase_selector,
            current,
            constraints,
            emitted_equality_aux,
            emitted_ordered_aux,
        );

        for (local_row, row_selector) in row_selectors
            .iter()
            .enumerate()
            .take(usize::from(stage.collection.max_items))
        {
            let selector = row_selector.clone();
            let active = in_domain.clone()
                * count_at_least_poly(
                    &count,
                    u8::try_from(local_row + 1).expect("quantifier threshold fits in u8"),
                    stage.collection.max_items,
                );
            let term_value = match stage.kind {
                QuantifierKind::ForAll => {
                    active.clone() * item_value.clone() + (one.clone() - active.clone())
                }
                QuantifierKind::Exists => active.clone() * item_value.clone(),
            };
            let term_status = active * item_status.clone();
            let (expected_next_status, expected_next_value) = match stage.kind {
                QuantifierKind::ForAll => boolean_and_poly(
                    current_acc_status.clone(),
                    current_acc_value.clone(),
                    term_status,
                    term_value,
                ),
                QuantifierKind::Exists => boolean_or_poly(
                    current_acc_status.clone(),
                    current_acc_value.clone(),
                    term_status,
                    term_value,
                ),
            };
            constraints.push(
                selector.clone()
                    * (next[stage.accumulator_status_register].clone() - expected_next_status),
            );
            constraints.push(
                selector * (next[stage.accumulator_value_register].clone() - expected_next_value),
            );
        }

        if let Some(final_selector) = row_selectors.last() {
            constraints.push(
                final_selector.clone()
                    * (current[stage.result_value_register].clone()
                        - current[stage.accumulator_value_register].clone()),
            );
            constraints.push(
                final_selector.clone()
                    * (current[stage.result_status_register].clone()
                        - current[stage.accumulator_status_register].clone()),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_quantified_predicate_polys(
        &self,
        predicate: &CompactQuantifiedPredicate,
        item_row_selectors: &[MPolynomial],
        row_gate: &MPolynomial,
        current: &[MPolynomial],
        constraints: &mut Vec<MPolynomial>,
        emitted_equality_aux: &mut HashSet<usize>,
        emitted_ordered_aux: &mut HashSet<usize>,
    ) -> (MPolynomial, MPolynomial) {
        let one = mp_constant(1);
        match predicate {
            CompactQuantifiedPredicate::Constant(value) => (
                mp_constant(STATUS_AVAILABLE.into()),
                mp_constant(if *value { 1 } else { 0 }),
            ),
            CompactQuantifiedPredicate::Comparison {
                lhs,
                op,
                rhs,
                equality_aux,
                ordered_aux,
            } => {
                let lhs_value =
                    self.compact_arithmetic_expr_value_poly(lhs, item_row_selectors, current);
                let lhs_status =
                    self.compact_arithmetic_expr_status_poly(lhs, item_row_selectors, current);
                let rhs_value =
                    self.compact_arithmetic_expr_value_poly(rhs, item_row_selectors, current);
                let rhs_status =
                    self.compact_arithmetic_expr_status_poly(rhs, item_row_selectors, current);
                let status = combine_status_poly(lhs_status, rhs_status);
                let available = status_selector(&status, STATUS_AVAILABLE);
                match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        let aux = equality_aux.as_ref().expect("compact equality aux");
                        let delta = current[aux.delta_register].clone();
                        let inverse = current[aux.inverse_register].clone();
                        let is_zero = one.clone() - delta.clone() * inverse.clone();
                        if emitted_equality_aux.insert(aux.delta_register) {
                            constraints.push(
                                row_gate.clone()
                                    * (delta.clone() - available.clone() * (lhs_value - rhs_value)),
                            );
                            constraints.push(row_gate.clone() * delta.clone() * is_zero.clone());
                            constraints.push(
                                row_gate.clone()
                                    * (delta.clone() * inverse - (one.clone() - is_zero.clone())),
                            );
                        }
                        let value = match op {
                            ComparisonOp::Eq => available.clone() * is_zero,
                            ComparisonOp::Ne => available.clone() * (one.clone() - is_zero),
                            _ => unreachable!(),
                        };
                        (status, value)
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => {
                        let aux = ordered_aux.as_ref().expect("compact ordered aux");
                        let diff = current[aux.diff_register].clone();
                        let inverse = current[aux.inverse_register].clone();
                        let is_zero = one.clone() - diff.clone() * inverse.clone();
                        let sign = current[*aux.diff_bits.last().expect("sign bit")].clone();
                        if emitted_ordered_aux.insert(aux.diff_register) {
                            constraints.push(
                                row_gate.clone()
                                    * (diff.clone() - available.clone() * (lhs_value - rhs_value)),
                            );
                            constraints.push(row_gate.clone() * diff.clone() * is_zero.clone());
                            constraints.push(
                                row_gate.clone()
                                    * (diff.clone() * inverse - (one.clone() - is_zero.clone())),
                            );
                            constraints.extend(
                                range_constraints(diff.clone(), &aux.diff_bits, current)
                                    .into_iter()
                                    .map(|constraint| row_gate.clone() * constraint),
                            );
                        }
                        let value = match op {
                            ComparisonOp::Lt => available.clone() * sign.clone(),
                            ComparisonOp::Le => {
                                available.clone() * (sign.clone() + is_zero.clone())
                            }
                            ComparisonOp::Gt => {
                                available.clone() * (one.clone() - sign.clone() - is_zero.clone())
                            }
                            ComparisonOp::Ge => available.clone() * (one.clone() - sign.clone()),
                            _ => unreachable!(),
                        };
                        (status, value)
                    }
                }
            }
            CompactQuantifiedPredicate::Availability { kind, value } => {
                let source_status =
                    self.compact_arithmetic_expr_status_poly(value, item_row_selectors, current);
                let value = match kind {
                    AvailabilityPredicateKind::Present => {
                        status_selector(&source_status, STATUS_AVAILABLE)
                    }
                    AvailabilityPredicateKind::Missing => {
                        status_selector(&source_status, STATUS_MISSING)
                    }
                    AvailabilityPredicateKind::Unknown => {
                        status_selector(&source_status, STATUS_UNKNOWN)
                    }
                };
                (mp_constant(STATUS_AVAILABLE.into()), value)
            }
            CompactQuantifiedPredicate::Between {
                value,
                lower,
                upper,
                lower_aux,
                upper_aux,
            } => {
                let between_value =
                    self.compact_arithmetic_expr_value_poly(value, item_row_selectors, current);
                let between_status =
                    self.compact_arithmetic_expr_status_poly(value, item_row_selectors, current);
                let lower_value =
                    self.compact_arithmetic_expr_value_poly(lower, item_row_selectors, current);
                let lower_status =
                    self.compact_arithmetic_expr_status_poly(lower, item_row_selectors, current);
                let upper_value =
                    self.compact_arithmetic_expr_value_poly(upper, item_row_selectors, current);
                let upper_status =
                    self.compact_arithmetic_expr_status_poly(upper, item_row_selectors, current);
                let status = combine_status_poly(
                    between_status,
                    combine_status_poly(lower_status, upper_status),
                );
                let available = status_selector(&status, STATUS_AVAILABLE);

                let lower_diff = current[lower_aux.diff_register].clone();
                let lower_inverse = current[lower_aux.inverse_register].clone();
                let lower_is_zero = one.clone() - lower_diff.clone() * lower_inverse.clone();
                let lower_sign =
                    current[*lower_aux.diff_bits.last().expect("lower sign bit")].clone();
                if emitted_ordered_aux.insert(lower_aux.diff_register) {
                    constraints.push(
                        row_gate.clone()
                            * (lower_diff.clone()
                                - available.clone() * (between_value.clone() - lower_value)),
                    );
                    constraints.push(row_gate.clone() * lower_diff.clone() * lower_is_zero.clone());
                    constraints.push(
                        row_gate.clone()
                            * (lower_diff.clone() * lower_inverse
                                - (one.clone() - lower_is_zero.clone())),
                    );
                    constraints.extend(
                        range_constraints(lower_diff.clone(), &lower_aux.diff_bits, current)
                            .into_iter()
                            .map(|constraint| row_gate.clone() * constraint),
                    );
                }

                let upper_diff = current[upper_aux.diff_register].clone();
                let upper_inverse = current[upper_aux.inverse_register].clone();
                let upper_is_zero = one.clone() - upper_diff.clone() * upper_inverse.clone();
                let upper_sign =
                    current[*upper_aux.diff_bits.last().expect("upper sign bit")].clone();
                if emitted_ordered_aux.insert(upper_aux.diff_register) {
                    constraints.push(
                        row_gate.clone()
                            * (upper_diff.clone()
                                - available.clone() * (between_value - upper_value)),
                    );
                    constraints.push(row_gate.clone() * upper_diff.clone() * upper_is_zero.clone());
                    constraints.push(
                        row_gate.clone()
                            * (upper_diff.clone() * upper_inverse
                                - (one.clone() - upper_is_zero.clone())),
                    );
                    constraints.extend(
                        range_constraints(upper_diff.clone(), &upper_aux.diff_bits, current)
                            .into_iter()
                            .map(|constraint| row_gate.clone() * constraint),
                    );
                }

                let ge_lower = one.clone() - lower_sign;
                let le_upper = upper_sign + upper_is_zero;
                (status, available * ge_lower * le_upper)
            }
            CompactQuantifiedPredicate::And(left, right) => {
                let (left_status, left_value) = self.compact_quantified_predicate_polys(
                    left,
                    item_row_selectors,
                    row_gate,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let (right_status, right_value) = self.compact_quantified_predicate_polys(
                    right,
                    item_row_selectors,
                    row_gate,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                boolean_and_poly(left_status, left_value, right_status, right_value)
            }
            CompactQuantifiedPredicate::Or(left, right) => {
                let (left_status, left_value) = self.compact_quantified_predicate_polys(
                    left,
                    item_row_selectors,
                    row_gate,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let (right_status, right_value) = self.compact_quantified_predicate_polys(
                    right,
                    item_row_selectors,
                    row_gate,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                boolean_or_poly(left_status, left_value, right_status, right_value)
            }
            CompactQuantifiedPredicate::Not(inner) => {
                let (inner_status, inner_value) = self.compact_quantified_predicate_polys(
                    inner,
                    item_row_selectors,
                    row_gate,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let inner_available = status_selector(&inner_status, STATUS_AVAILABLE);
                (
                    inner_status,
                    inner_available.clone() * (one.clone() - inner_value.clone())
                        + (one - inner_available) * inner_value,
                )
            }
        }
    }

    fn compact_arithmetic_expr_value_poly(
        &self,
        expr: &CompactArithmeticExpr,
        item_row_selectors: &[MPolynomial],
        current: &[MPolynomial],
    ) -> MPolynomial {
        match expr {
            CompactArithmeticExpr::Number(value) => mp_constant(i128::from(*value)),
            CompactArithmeticExpr::Value(value_ref) => {
                self.compact_value_ref_value_poly(value_ref, item_row_selectors, current)
            }
        }
    }

    fn compact_arithmetic_expr_status_poly(
        &self,
        expr: &CompactArithmeticExpr,
        item_row_selectors: &[MPolynomial],
        current: &[MPolynomial],
    ) -> MPolynomial {
        match expr {
            CompactArithmeticExpr::Number(_) => mp_constant(STATUS_AVAILABLE.into()),
            CompactArithmeticExpr::Value(value_ref) => self.compact_availability_status_poly(
                &value_ref.availability,
                item_row_selectors,
                current,
            ),
        }
    }

    fn compact_value_ref_value_poly(
        &self,
        value_ref: &CompactValueRef,
        item_row_selectors: &[MPolynomial],
        current: &[MPolynomial],
    ) -> MPolynomial {
        match &value_ref.source {
            CompactValueSource::InputRegister(register) => current[*register].clone(),
            CompactValueSource::DefinitionValue(index) => {
                current[self.circuit.definitions[*index].value_register].clone()
            }
            CompactValueSource::ItemRegisters(registers) => registers
                .iter()
                .zip(item_row_selectors.iter())
                .fold(MPolynomial::zero(), |acc, (register, selector)| {
                    acc + selector.clone() * current[*register].clone()
                }),
        }
    }

    fn compact_availability_status_poly(
        &self,
        availability: &CompactAvailability,
        item_row_selectors: &[MPolynomial],
        current: &[MPolynomial],
    ) -> MPolynomial {
        let one = mp_constant(1);
        match availability {
            CompactAvailability::Always => mp_constant(STATUS_AVAILABLE.into()),
            CompactAvailability::Flags { presence, unknown } => {
                let unknown_expr = unknown
                    .as_ref()
                    .map(|source| {
                        self.compact_availability_source_poly(source, item_row_selectors, current)
                    })
                    .unwrap_or_else(MPolynomial::zero);
                let presence_expr = presence
                    .as_ref()
                    .map(|source| {
                        self.compact_availability_source_poly(source, item_row_selectors, current)
                    })
                    .unwrap_or_else(|| one.clone());
                unknown_expr.clone() * mp_constant(2)
                    + (one.clone() - unknown_expr) * (one - presence_expr)
            }
            CompactAvailability::Status(source) => {
                self.compact_availability_source_poly(source, item_row_selectors, current)
            }
        }
    }

    fn compact_availability_source_poly(
        &self,
        source: &CompactAvailabilitySource,
        item_row_selectors: &[MPolynomial],
        current: &[MPolynomial],
    ) -> MPolynomial {
        match source {
            CompactAvailabilitySource::InputRegister(register) => current[*register].clone(),
            CompactAvailabilitySource::DefinitionStatus(index) => {
                current[self.circuit.definitions[*index].status_register].clone()
            }
            CompactAvailabilitySource::ItemRegisters(registers) => registers
                .iter()
                .zip(item_row_selectors.iter())
                .fold(MPolynomial::zero(), |acc, (register, selector)| {
                    acc + selector.clone() * current[*register].clone()
                }),
        }
    }

    fn quantified_predicate_polys(
        &self,
        predicate: &QuantifiedPredicate,
        current: &[MPolynomial],
        constraints: &mut Vec<MPolynomial>,
        emitted_equality_aux: &mut HashSet<usize>,
        emitted_ordered_aux: &mut HashSet<usize>,
    ) -> (MPolynomial, MPolynomial) {
        let one = mp_constant(1);
        match predicate {
            QuantifiedPredicate::Constant(value) => (
                mp_constant(STATUS_AVAILABLE.into()),
                mp_constant(if *value { 1 } else { 0 }),
            ),
            QuantifiedPredicate::Comparison {
                lhs,
                op,
                rhs,
                equality_aux,
                ordered_aux,
            } => {
                let lhs_value = self.arithmetic_source_value(lhs, current);
                let lhs_status = self.arithmetic_source_status(lhs, current);
                let rhs_value = self.arithmetic_source_value(rhs, current);
                let rhs_status = self.arithmetic_source_status(rhs, current);
                let status = combine_status_poly(lhs_status, rhs_status);
                let available = status_selector(&status, STATUS_AVAILABLE);
                match op {
                    ComparisonOp::Eq | ComparisonOp::Ne => {
                        let aux = equality_aux.as_ref().expect("equality aux");
                        let delta = current[aux.delta_register].clone();
                        let inverse = current[aux.inverse_register].clone();
                        let is_zero = one.clone() - delta.clone() * inverse.clone();
                        if emitted_equality_aux.insert(aux.delta_register) {
                            constraints
                                .push(delta.clone() - available.clone() * (lhs_value - rhs_value));
                            constraints.push(delta.clone() * is_zero.clone());
                            constraints
                                .push(delta.clone() * inverse - (one.clone() - is_zero.clone()));
                        }
                        let value = match op {
                            ComparisonOp::Eq => available.clone() * is_zero,
                            ComparisonOp::Ne => available.clone() * (one.clone() - is_zero),
                            _ => unreachable!(),
                        };
                        (status, value)
                    }
                    ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge => {
                        let aux = ordered_aux.as_ref().expect("ordered aux");
                        let diff = current[aux.diff_register].clone();
                        let inverse = current[aux.inverse_register].clone();
                        let is_zero = one.clone() - diff.clone() * inverse.clone();
                        let sign = current[*aux.diff_bits.last().expect("sign bit")].clone();
                        if emitted_ordered_aux.insert(aux.diff_register) {
                            constraints
                                .push(diff.clone() - available.clone() * (lhs_value - rhs_value));
                            constraints.push(diff.clone() * is_zero.clone());
                            constraints
                                .push(diff.clone() * inverse - (one.clone() - is_zero.clone()));
                            constraints.extend(range_constraints(
                                diff.clone(),
                                &aux.diff_bits,
                                current,
                            ));
                        }
                        let value = match op {
                            ComparisonOp::Lt => available.clone() * sign.clone(),
                            ComparisonOp::Le => {
                                available.clone() * (sign.clone() + is_zero.clone())
                            }
                            ComparisonOp::Gt => {
                                available.clone() * (one.clone() - sign.clone() - is_zero.clone())
                            }
                            ComparisonOp::Ge => available.clone() * (one.clone() - sign.clone()),
                            _ => unreachable!(),
                        };
                        (status, value)
                    }
                }
            }
            QuantifiedPredicate::Availability { kind, value } => {
                let source_status = self.arithmetic_source_status(value, current);
                let value = match kind {
                    AvailabilityPredicateKind::Present => {
                        status_selector(&source_status, STATUS_AVAILABLE)
                    }
                    AvailabilityPredicateKind::Missing => {
                        status_selector(&source_status, STATUS_MISSING)
                    }
                    AvailabilityPredicateKind::Unknown => {
                        status_selector(&source_status, STATUS_UNKNOWN)
                    }
                };
                (mp_constant(STATUS_AVAILABLE.into()), value)
            }
            QuantifiedPredicate::Between {
                value,
                lower,
                upper,
                lower_aux,
                upper_aux,
            } => {
                let between_value_value = self.arithmetic_source_value(value, current);
                let between_value_status = self.arithmetic_source_status(value, current);
                let lower_value = self.arithmetic_source_value(lower, current);
                let lower_status = self.arithmetic_source_status(lower, current);
                let upper_value = self.arithmetic_source_value(upper, current);
                let upper_status = self.arithmetic_source_status(upper, current);
                let status = combine_status_poly(
                    between_value_status,
                    combine_status_poly(lower_status, upper_status),
                );
                let available = status_selector(&status, STATUS_AVAILABLE);

                let lower_diff = current[lower_aux.diff_register].clone();
                let lower_inverse = current[lower_aux.inverse_register].clone();
                let lower_is_zero = one.clone() - lower_diff.clone() * lower_inverse.clone();
                let lower_sign =
                    current[*lower_aux.diff_bits.last().expect("lower sign bit")].clone();
                if emitted_ordered_aux.insert(lower_aux.diff_register) {
                    constraints.push(
                        lower_diff.clone()
                            - available.clone() * (between_value_value.clone() - lower_value),
                    );
                    constraints.push(lower_diff.clone() * lower_is_zero.clone());
                    constraints.push(
                        lower_diff.clone() * lower_inverse - (one.clone() - lower_is_zero.clone()),
                    );
                    constraints.extend(range_constraints(
                        lower_diff.clone(),
                        &lower_aux.diff_bits,
                        current,
                    ));
                }

                let upper_diff = current[upper_aux.diff_register].clone();
                let upper_inverse = current[upper_aux.inverse_register].clone();
                let upper_is_zero = one.clone() - upper_diff.clone() * upper_inverse.clone();
                let upper_sign =
                    current[*upper_aux.diff_bits.last().expect("upper sign bit")].clone();
                if emitted_ordered_aux.insert(upper_aux.diff_register) {
                    constraints.push(
                        upper_diff.clone()
                            - available.clone() * (between_value_value - upper_value),
                    );
                    constraints.push(upper_diff.clone() * upper_is_zero.clone());
                    constraints.push(
                        upper_diff.clone() * upper_inverse - (one.clone() - upper_is_zero.clone()),
                    );
                    constraints.extend(range_constraints(
                        upper_diff.clone(),
                        &upper_aux.diff_bits,
                        current,
                    ));
                }

                let ge_lower = one.clone() - lower_sign;
                let le_upper = upper_sign + upper_is_zero;
                (status, available * ge_lower * le_upper)
            }
            QuantifiedPredicate::And(left, right) => {
                let (left_status, left_value) = self.quantified_predicate_polys(
                    left,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let (right_status, right_value) = self.quantified_predicate_polys(
                    right,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let left_available = status_selector(&left_status, STATUS_AVAILABLE);
                let take_left = one.clone() - left_available.clone() * left_value.clone();
                let take_right = left_available * left_value.clone();
                (
                    take_left.clone() * left_status + take_right.clone() * right_status,
                    take_left * left_value + take_right * right_value,
                )
            }
            QuantifiedPredicate::Or(left, right) => {
                let (left_status, left_value) = self.quantified_predicate_polys(
                    left,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let (right_status, right_value) = self.quantified_predicate_polys(
                    right,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let left_available = status_selector(&left_status, STATUS_AVAILABLE);
                let take_right = left_available.clone() * (one.clone() - left_value.clone());
                let take_left = one.clone() - take_right.clone();
                (
                    take_left.clone() * left_status + take_right.clone() * right_status,
                    take_left * left_value + take_right * right_value,
                )
            }
            QuantifiedPredicate::Not(inner) => {
                let (inner_status, inner_value) = self.quantified_predicate_polys(
                    inner,
                    current,
                    constraints,
                    emitted_equality_aux,
                    emitted_ordered_aux,
                );
                let inner_available = status_selector(&inner_status, STATUS_AVAILABLE);
                (
                    inner_status,
                    inner_available.clone() * (one.clone() - inner_value.clone())
                        + (one - inner_available) * inner_value,
                )
            }
        }
    }
}

fn extract_quantifier_pattern(
    expr: &DecisionBooleanExpr,
) -> Option<(QuantifierKind, u8, u8, Vec<&DecisionBooleanExpr>)> {
    extract_forall_pattern(expr).or_else(|| extract_exists_pattern(expr))
}

fn extract_forall_pattern(
    expr: &DecisionBooleanExpr,
) -> Option<(QuantifierKind, u8, u8, Vec<&DecisionBooleanExpr>)> {
    let mut predicates = Vec::new();
    let mut thresholds = Vec::new();
    let mut current = expr;
    let mut count_memory = None;
    let mut max_items = None;

    loop {
        match current {
            DecisionBooleanExpr::Constant(true) => {
                let max_items = max_items?;
                let expected = (1..=max_items).collect::<Vec<_>>();
                thresholds.reverse();
                if thresholds != expected {
                    return None;
                }
                predicates.reverse();
                return Some((QuantifierKind::ForAll, count_memory?, max_items, predicates));
            }
            DecisionBooleanExpr::And(left, right) => {
                let (active, predicate) = match right.as_ref() {
                    DecisionBooleanExpr::Or(not_active, predicate) => match not_active.as_ref() {
                        DecisionBooleanExpr::Not(active) => (active.as_ref(), predicate.as_ref()),
                        _ => return None,
                    },
                    _ => return None,
                };
                match active {
                    DecisionBooleanExpr::CountAtLeast {
                        count_memory: active_count,
                        threshold,
                        max_items: active_max,
                    } => {
                        if let Some(current_count) = count_memory {
                            if current_count != *active_count {
                                return None;
                            }
                        } else {
                            count_memory = Some(*active_count);
                        }
                        if let Some(current_max) = max_items {
                            if current_max != *active_max {
                                return None;
                            }
                        } else {
                            max_items = Some(*active_max);
                        }
                        thresholds.push(*threshold);
                        predicates.push(predicate);
                        current = left.as_ref();
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

fn extract_exists_pattern(
    expr: &DecisionBooleanExpr,
) -> Option<(QuantifierKind, u8, u8, Vec<&DecisionBooleanExpr>)> {
    let mut predicates = Vec::new();
    let mut thresholds = Vec::new();
    let mut current = expr;
    let mut count_memory = None;
    let mut max_items = None;

    loop {
        match current {
            DecisionBooleanExpr::Constant(false) => {
                let max_items = max_items?;
                let expected = (1..=max_items).collect::<Vec<_>>();
                thresholds.reverse();
                if thresholds != expected {
                    return None;
                }
                predicates.reverse();
                return Some((QuantifierKind::Exists, count_memory?, max_items, predicates));
            }
            DecisionBooleanExpr::Or(left, right) => {
                let (active, predicate) = match right.as_ref() {
                    DecisionBooleanExpr::And(active, predicate) => {
                        (active.as_ref(), predicate.as_ref())
                    }
                    _ => return None,
                };
                match active {
                    DecisionBooleanExpr::CountAtLeast {
                        count_memory: active_count,
                        threshold,
                        max_items: active_max,
                    } => {
                        if let Some(current_count) = count_memory {
                            if current_count != *active_count {
                                return None;
                            }
                        } else {
                            count_memory = Some(*active_count);
                        }
                        if let Some(current_max) = max_items {
                            if current_max != *active_max {
                                return None;
                            }
                        } else {
                            max_items = Some(*active_max);
                        }
                        thresholds.push(*threshold);
                        predicates.push(predicate);
                        current = left.as_ref();
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

fn combine_status(left: Word, right: Word) -> Word {
    left.max(right)
}

fn checked_binary_op(op: ArithmeticOp, lhs: Word, rhs: Word) -> Result<Word> {
    match op {
        ArithmeticOp::Add => lhs.checked_add(rhs),
        ArithmeticOp::Sub => lhs.checked_sub(rhs),
        ArithmeticOp::Mul => lhs.checked_mul(rhs),
    }
    .ok_or_else(|| {
        VmError::UnsupportedProof(
            "direct cDMN STARK currently requires arithmetic whose intermediate results fit i64"
                .to_string(),
        )
    })
}

fn project_inputs(
    program: &CompiledCdmnDecisionProgram,
    initial_memory: &[Word],
) -> Result<Vec<(u8, Word)>> {
    program
        .input_cells
        .iter()
        .map(|&address| {
            initial_memory
                .get(usize::from(address))
                .copied()
                .map(|value| (address, value))
                .ok_or(VmError::MemoryOutOfBounds {
                    addr: usize::from(address),
                    size: initial_memory.len(),
                })
        })
        .collect()
}

fn materialize_projected_inputs(
    program: &CompiledCdmnDecisionProgram,
    projected_inputs: &[(u8, Word)],
) -> Result<Vec<Word>> {
    let mut memory = vec![
        0;
        program
            .input_cells
            .iter()
            .max()
            .map(|address| usize::from(*address) + 1)
            .unwrap_or(0)
    ];
    let lookup: BTreeMap<u8, Word> = projected_inputs.iter().copied().collect();
    for &address in &program.input_cells {
        memory[usize::from(address)] = *lookup.get(&address).ok_or_else(|| {
            VmError::InvalidConfig(format!(
                "missing projected input cell {address} for direct cDMN verification"
            ))
        })?;
    }
    Ok(memory)
}

fn range_constraints(
    value: MPolynomial,
    bit_registers: &[usize],
    current: &[MPolynomial],
) -> Vec<MPolynomial> {
    let mut constraints = Vec::new();
    let one = mp_constant(1);
    let mut bit_sum = MPolynomial::zero();
    for (index, register) in bit_registers.iter().enumerate().take(WORD_BITS - 1) {
        let bit = current[*register].clone();
        constraints.push(bit.clone() * (bit.clone() - one.clone()));
        bit_sum = bit_sum + bit * mp_constant(1i128 << index);
    }
    let sign_bit = current[*bit_registers.last().expect("sign bit present")].clone();
    constraints.push(sign_bit.clone() * (sign_bit.clone() - one.clone()));
    bit_sum = bit_sum - sign_bit * mp_constant(1i128 << (WORD_BITS - 1));
    constraints.push(value - bit_sum);
    constraints
}

fn status_selector(status: &MPolynomial, target: Word) -> MPolynomial {
    lagrange_selector(
        status,
        &[STATUS_AVAILABLE, STATUS_MISSING, STATUS_UNKNOWN],
        target,
    )
}

fn boolean_and_poly(
    left_status: MPolynomial,
    left_value: MPolynomial,
    right_status: MPolynomial,
    right_value: MPolynomial,
) -> (MPolynomial, MPolynomial) {
    let one = mp_constant(1);
    let left_available = status_selector(&left_status, STATUS_AVAILABLE);
    let take_left = one.clone() - left_available.clone() * left_value.clone();
    let take_right = left_available * left_value.clone();
    (
        take_left.clone() * left_status + take_right.clone() * right_status,
        take_left * left_value + take_right * right_value,
    )
}

fn boolean_or_poly(
    left_status: MPolynomial,
    left_value: MPolynomial,
    right_status: MPolynomial,
    right_value: MPolynomial,
) -> (MPolynomial, MPolynomial) {
    let one = mp_constant(1);
    let left_available = status_selector(&left_status, STATUS_AVAILABLE);
    let take_right = left_available.clone() * (one.clone() - left_value.clone());
    let take_left = one.clone() - take_right.clone();
    (
        take_left.clone() * left_status + take_right.clone() * right_status,
        take_left * left_value + take_right * right_value,
    )
}

fn count_in_domain_poly(count: &MPolynomial, max_items: u8) -> MPolynomial {
    let domain: Vec<Word> = (0..=max_items).map(i64::from).collect();
    domain.iter().fold(MPolynomial::zero(), |acc, value| {
        acc + lagrange_selector(count, &domain, *value)
    })
}

fn count_at_least_poly(count: &MPolynomial, threshold: u8, max_items: u8) -> MPolynomial {
    if threshold > max_items {
        return MPolynomial::zero();
    }
    let domain: Vec<Word> = (0..=max_items).map(i64::from).collect();
    (threshold..=max_items).fold(MPolynomial::zero(), |acc, value| {
        acc + lagrange_selector(count, &domain, i64::from(value))
    })
}

fn combine_status_poly(left: MPolynomial, right: MPolynomial) -> MPolynomial {
    let left0 = status_selector(&left, STATUS_AVAILABLE);
    let left1 = status_selector(&left, STATUS_MISSING);
    let right0 = status_selector(&right, STATUS_AVAILABLE);
    let right1 = status_selector(&right, STATUS_MISSING);
    let status1 = left1.clone() * (right0.clone() + right1.clone()) + left0 * right1;
    let status2 = mp_constant(1)
        - status_selector(&left, STATUS_AVAILABLE) * status_selector(&right, STATUS_AVAILABLE)
        - status1.clone();
    status1 + mp_constant(2) * status2
}

fn status_range_constraint(status: MPolynomial) -> MPolynomial {
    status.clone() * (status.clone() - mp_constant(1)) * (status - mp_constant(2))
}

fn lagrange_selector(variable: &MPolynomial, domain: &[Word], point: Word) -> MPolynomial {
    let numerator = domain
        .iter()
        .filter(|&&other| other != point)
        .fold(mp_constant(1), |acc, &other| {
            acc * (variable.clone() - mp_constant(i128::from(other)))
        });
    let denominator = domain
        .iter()
        .filter(|&&other| other != point)
        .fold(field_from_i128(1), |acc, &other| {
            acc * field_from_i128(i128::from(point - other))
        });
    numerator * MPolynomial::constant(denominator.inverse())
}

fn signed_bits(value: Word) -> Vec<bool> {
    let bits = u64::from_le_bytes(value.to_le_bytes());
    (0..WORD_BITS).map(|bit| ((bits >> bit) & 1) == 1).collect()
}

fn mp_constant(value: i128) -> MPolynomial {
    MPolynomial::constant(field_from_i128(value))
}

fn field_from_bool(value: bool) -> FieldElement {
    field_from_i128(if value { 1 } else { 0 })
}

fn field_from_word(value: Word) -> FieldElement {
    field_from_i128(i128::from(value))
}

fn field_from_i128(value: i128) -> FieldElement {
    if value >= 0 {
        FieldElement::new(value as u128)
    } else {
        -FieldElement::new(value.unsigned_abs())
    }
}

fn elapsed_micros(start: Instant) -> u64 {
    start.elapsed().as_micros().try_into().unwrap_or(u64::MAX)
}

fn collect_quantified_equality_aux(predicate: &QuantifiedPredicate, aux: &mut HashSet<usize>) {
    match predicate {
        QuantifiedPredicate::Comparison {
            equality_aux: Some(aux_regs),
            ..
        } => {
            aux.insert(aux_regs.delta_register);
        }
        QuantifiedPredicate::And(left, right) | QuantifiedPredicate::Or(left, right) => {
            collect_quantified_equality_aux(left, aux);
            collect_quantified_equality_aux(right, aux);
        }
        QuantifiedPredicate::Not(inner) => collect_quantified_equality_aux(inner, aux),
        QuantifiedPredicate::Constant(_)
        | QuantifiedPredicate::Availability { .. }
        | QuantifiedPredicate::Between { .. }
        | QuantifiedPredicate::Comparison {
            equality_aux: None, ..
        } => {}
    }
}

fn collect_quantified_ordered_aux(predicate: &QuantifiedPredicate, aux: &mut HashSet<usize>) {
    match predicate {
        QuantifiedPredicate::Comparison {
            ordered_aux: Some(aux_regs),
            ..
        } => {
            aux.insert(aux_regs.diff_register);
        }
        QuantifiedPredicate::Between {
            lower_aux,
            upper_aux,
            ..
        } => {
            aux.insert(lower_aux.diff_register);
            aux.insert(upper_aux.diff_register);
        }
        QuantifiedPredicate::And(left, right) | QuantifiedPredicate::Or(left, right) => {
            collect_quantified_ordered_aux(left, aux);
            collect_quantified_ordered_aux(right, aux);
        }
        QuantifiedPredicate::Not(inner) => collect_quantified_ordered_aux(inner, aux),
        QuantifiedPredicate::Constant(_)
        | QuantifiedPredicate::Availability { .. }
        | QuantifiedPredicate::Comparison {
            ordered_aux: None, ..
        } => {}
    }
}

fn collect_compact_predicate_registers(
    predicate: &CompactQuantifiedPredicate,
    registers: &mut HashSet<usize>,
) {
    match predicate {
        CompactQuantifiedPredicate::Comparison {
            equality_aux,
            ordered_aux,
            ..
        } => {
            if let Some(aux) = equality_aux {
                registers.insert(aux.delta_register);
                registers.insert(aux.inverse_register);
            }
            if let Some(aux) = ordered_aux {
                registers.insert(aux.diff_register);
                registers.insert(aux.inverse_register);
                registers.extend(aux.diff_bits.iter().copied());
            }
        }
        CompactQuantifiedPredicate::Between {
            lower_aux,
            upper_aux,
            ..
        } => {
            registers.insert(lower_aux.diff_register);
            registers.insert(lower_aux.inverse_register);
            registers.extend(lower_aux.diff_bits.iter().copied());
            registers.insert(upper_aux.diff_register);
            registers.insert(upper_aux.inverse_register);
            registers.extend(upper_aux.diff_bits.iter().copied());
        }
        CompactQuantifiedPredicate::And(left, right)
        | CompactQuantifiedPredicate::Or(left, right) => {
            collect_compact_predicate_registers(left, registers);
            collect_compact_predicate_registers(right, registers);
        }
        CompactQuantifiedPredicate::Not(inner) => {
            collect_compact_predicate_registers(inner, registers);
        }
        CompactQuantifiedPredicate::Constant(_)
        | CompactQuantifiedPredicate::Availability { .. } => {}
    }
}

fn collect_compact_equality_aux(predicate: &CompactQuantifiedPredicate, aux: &mut HashSet<usize>) {
    match predicate {
        CompactQuantifiedPredicate::Comparison {
            equality_aux: Some(aux_regs),
            ..
        } => {
            aux.insert(aux_regs.delta_register);
        }
        CompactQuantifiedPredicate::And(left, right)
        | CompactQuantifiedPredicate::Or(left, right) => {
            collect_compact_equality_aux(left, aux);
            collect_compact_equality_aux(right, aux);
        }
        CompactQuantifiedPredicate::Not(inner) => collect_compact_equality_aux(inner, aux),
        CompactQuantifiedPredicate::Constant(_)
        | CompactQuantifiedPredicate::Availability { .. }
        | CompactQuantifiedPredicate::Between { .. }
        | CompactQuantifiedPredicate::Comparison {
            equality_aux: None, ..
        } => {}
    }
}

fn collect_compact_ordered_aux(predicate: &CompactQuantifiedPredicate, aux: &mut HashSet<usize>) {
    match predicate {
        CompactQuantifiedPredicate::Comparison {
            ordered_aux: Some(aux_regs),
            ..
        } => {
            aux.insert(aux_regs.diff_register);
        }
        CompactQuantifiedPredicate::Between {
            lower_aux,
            upper_aux,
            ..
        } => {
            aux.insert(lower_aux.diff_register);
            aux.insert(upper_aux.diff_register);
        }
        CompactQuantifiedPredicate::And(left, right)
        | CompactQuantifiedPredicate::Or(left, right) => {
            collect_compact_ordered_aux(left, aux);
            collect_compact_ordered_aux(right, aux);
        }
        CompactQuantifiedPredicate::Not(inner) => collect_compact_ordered_aux(inner, aux),
        CompactQuantifiedPredicate::Constant(_)
        | CompactQuantifiedPredicate::Availability { .. }
        | CompactQuantifiedPredicate::Comparison {
            ordered_aux: None, ..
        } => {}
    }
}

fn find_equality_aux_by_register(
    predicate: &CompactQuantifiedPredicate,
    register: usize,
) -> Option<&EqualityAux> {
    match predicate {
        CompactQuantifiedPredicate::Comparison {
            equality_aux: Some(aux),
            ..
        } if aux.delta_register == register => Some(aux),
        CompactQuantifiedPredicate::And(left, right)
        | CompactQuantifiedPredicate::Or(left, right) => {
            find_equality_aux_by_register(left, register)
                .or_else(|| find_equality_aux_by_register(right, register))
        }
        CompactQuantifiedPredicate::Not(inner) => find_equality_aux_by_register(inner, register),
        _ => None,
    }
}

fn find_ordered_aux_by_register(
    predicate: &CompactQuantifiedPredicate,
    register: usize,
) -> Option<&OrderedComparisonAux> {
    match predicate {
        CompactQuantifiedPredicate::Comparison {
            ordered_aux: Some(aux),
            ..
        } if aux.diff_register == register => Some(aux),
        CompactQuantifiedPredicate::Between {
            lower_aux,
            upper_aux,
            ..
        } if lower_aux.diff_register == register => Some(lower_aux),
        CompactQuantifiedPredicate::Between {
            lower_aux: _,
            upper_aux,
            ..
        } if upper_aux.diff_register == register => Some(upper_aux),
        CompactQuantifiedPredicate::And(left, right)
        | CompactQuantifiedPredicate::Or(left, right) => {
            find_ordered_aux_by_register(left, register)
                .or_else(|| find_ordered_aux_by_register(right, register))
        }
        CompactQuantifiedPredicate::Not(inner) => find_ordered_aux_by_register(inner, register),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdmn::{
        compile_cdmn_table_to_decision_program, CdmnTable, Collection, CollectionField, Constraint,
        Variable,
    };

    #[test]
    fn direct_cdmn_proof_round_trips_small_policy() {
        let table = CdmnTable {
            name: "Direct Proof Smoke".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "dose".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "max_dose".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "D1".to_string(),
                description: "Dose must be within limit".to_string(),
                logic: "dose <= max_dose".to_string(),
                reason_code: Some(41),
                reason_label: Some("dose_above_limit".to_string()),
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let proof = prove_cdmn_decision_stark_direct(&table, &[10, 12]).unwrap();
        assert!(verify_cdmn_decision_stark_direct(&proof).unwrap());
        assert_eq!(proof.claim.status, 1);
        assert_eq!(proof.claim.reason_code, 0);

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let expected_policy_commitment = program.policy_commitment().unwrap();
        assert_eq!(proof.claim.policy_commitment, expected_policy_commitment);
        assert_eq!(
            proof.claim.input_commitment,
            crate::cdmn::proof_ir::hash_json(&vec![(0u8, 10i64), (1u8, 12i64)]).unwrap()
        );
    }

    #[test]
    fn direct_cdmn_cost_estimate_reports_circuit_shape() {
        let table = CdmnTable {
            name: "Direct Cost".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "x".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "y".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "x must be <= y".to_string(),
                logic: "x <= y".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let estimate =
            estimate_cdmn_decision_stark_direct_cost(&table, VanillaStarkProofOptions::default())
                .unwrap();
        assert_eq!(estimate.trace_rows, TRACE_ROWS);
        assert_eq!(estimate.projected_input_count, 2);
        assert_eq!(estimate.constraint_count, 1);
        assert_eq!(estimate.quantified_node_count, 0);
        assert_eq!(estimate.quantified_item_predicate_count, 0);
        assert!(estimate.register_count > 0);
        assert!(estimate.transition_constraint_count > 0);
    }

    #[test]
    fn direct_cdmn_cost_estimate_reports_quantified_expansion() {
        let table = CdmnTable {
            name: "Quantified Cost".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![Variable {
                name: "max_dose".to_string(),
                memory: 0,
                presence_memory: None,
                unknown_memory: None,
            }],
            definitions: vec![],
            collections: vec![Collection {
                name: "medications".to_string(),
                count_memory: 1,
                base_memory: 2,
                stride: 1,
                max_items: 2,
                fields: vec![CollectionField {
                    name: "dose".to_string(),
                    offset: 0,
                    presence_offset: None,
                    unknown_offset: None,
                }],
            }],
            constraints: vec![Constraint {
                id: "Q1".to_string(),
                description: "all doses must be within limit".to_string(),
                logic: "forall medication in medications (medication.dose <= max_dose)".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let estimate =
            estimate_cdmn_decision_stark_direct_cost(&table, VanillaStarkProofOptions::default())
                .unwrap();
        assert_eq!(estimate.quantified_node_count, 1);
        assert_eq!(estimate.quantified_item_predicate_count, 2);
        assert_eq!(estimate.quantified_max_items_total, 2);
        assert_eq!(estimate.ordered_aux_count, 1);
        assert_eq!(estimate.equality_aux_count, 0);
    }

    #[test]
    fn direct_builder_reuses_repeated_subexpressions() {
        let table = CdmnTable {
            name: "Repeated Structure".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "x".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "y".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "z".to_string(),
                    memory: 2,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "repeat x + y >= z".to_string(),
                logic: "(x + y >= z) and (x + y >= z)".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        assert_eq!(circuit.arithmetic_nodes.len(), 1);
        assert_eq!(circuit.boolean_nodes.len(), 2);
    }

    #[test]
    fn direct_builder_reuses_ordered_aux_for_distinct_comparisons_on_same_pair() {
        let table = CdmnTable {
            name: "Shared Ordered Aux".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "value".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "lower".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "upper".to_string(),
                    memory: 2,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "distinct lower-bound comparisons".to_string(),
                logic: "(value >= lower) and (value > lower)".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        let comparison_diffs = circuit
            .boolean_nodes
            .iter()
            .filter_map(|node| match &node.kind {
                BooleanNodeKind::Comparison {
                    op: ComparisonOp::Ge | ComparisonOp::Gt,
                    ordered_aux: Some(aux),
                    ..
                } => Some(aux.diff_register),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(comparison_diffs.len(), 2);
        assert_eq!(comparison_diffs[0], comparison_diffs[1]);
    }

    #[test]
    fn direct_builder_keeps_between_aux_distinct_from_standalone_comparison() {
        let table = CdmnTable {
            name: "Distinct Ordered Aux".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "value".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "lower".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "upper".to_string(),
                    memory: 2,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "upper_present".to_string(),
                    memory: 3,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "between and lower-bound comparison".to_string(),
                logic: "(value between lower and upper_present) and (value >= lower)".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        let mut between_lower_diff = None;
        let mut comparison_diff = None;
        for node in &circuit.boolean_nodes {
            match &node.kind {
                BooleanNodeKind::Between { lower_aux, .. } => {
                    between_lower_diff = Some(lower_aux.diff_register);
                }
                BooleanNodeKind::Comparison {
                    op: ComparisonOp::Ge,
                    ordered_aux: Some(aux),
                    ..
                } => {
                    comparison_diff = Some(aux.diff_register);
                }
                _ => {}
            }
        }
        assert_ne!(between_lower_diff, comparison_diff);
    }

    #[test]
    fn direct_builder_reuses_always_available_status_register() {
        let table = CdmnTable {
            name: "Always Available Status".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "x".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "y".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "z".to_string(),
                    memory: 2,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "simple always-available arithmetic".to_string(),
                logic: "not (((x + y) >= z) and ((x + y) between z and z))".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        let always = circuit.always_available_register;

        assert_eq!(circuit.arithmetic_nodes.len(), 1);
        assert_eq!(circuit.arithmetic_nodes[0].status_register, always);

        let comparison_statuses = circuit
            .boolean_nodes
            .iter()
            .filter_map(|node| match node.kind {
                BooleanNodeKind::Comparison { .. } | BooleanNodeKind::Between { .. } => {
                    Some(node.status_register)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!comparison_statuses.is_empty());
        assert!(comparison_statuses.iter().all(|&status| status == always));

        let and_or_not_statuses = circuit
            .boolean_nodes
            .iter()
            .filter_map(|node| match node.kind {
                BooleanNodeKind::And { .. }
                | BooleanNodeKind::Or { .. }
                | BooleanNodeKind::Not { .. } => Some(node.status_register),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!and_or_not_statuses.is_empty());
        assert!(and_or_not_statuses.iter().all(|&status| status == always));
    }

    #[test]
    fn direct_builder_aliases_definition_to_arithmetic_source() {
        let table = CdmnTable {
            name: "Definition Alias".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![
                Variable {
                    name: "x".to_string(),
                    memory: 0,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "y".to_string(),
                    memory: 1,
                    presence_memory: None,
                    unknown_memory: None,
                },
                Variable {
                    name: "z".to_string(),
                    memory: 2,
                    presence_memory: None,
                    unknown_memory: None,
                },
            ],
            collections: vec![],
            definitions: vec![crate::cdmn::Definition {
                name: "sum".to_string(),
                expression: "x + y".to_string(),
                description: "Derived sum".to_string(),
            }],
            constraints: vec![Constraint {
                id: "C1".to_string(),
                description: "sum must exceed z".to_string(),
                logic: "sum >= z".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        assert_eq!(circuit.arithmetic_nodes.len(), 1);
        assert_eq!(circuit.definitions.len(), 1);
        let arithmetic = &circuit.arithmetic_nodes[0];
        let definition = &circuit.definitions[0];
        assert!(!definition.materialized);
        assert_eq!(definition.value_register, arithmetic.value_register);
        assert_eq!(definition.status_register, arithmetic.status_register);
        assert_eq!(definition.bit_registers, arithmetic.bit_registers);
    }

    #[test]
    fn direct_builder_keeps_quantified_item_predicates_inline() {
        let table = CdmnTable {
            name: "Quantified Inline".to_string(),
            table_type: "cDMN".to_string(),
            variables: vec![Variable {
                name: "max_dose".to_string(),
                memory: 0,
                presence_memory: None,
                unknown_memory: None,
            }],
            definitions: vec![],
            collections: vec![Collection {
                name: "medications".to_string(),
                count_memory: 1,
                base_memory: 2,
                stride: 1,
                max_items: 3,
                fields: vec![CollectionField {
                    name: "dose".to_string(),
                    offset: 0,
                    presence_offset: None,
                    unknown_offset: None,
                }],
            }],
            constraints: vec![Constraint {
                id: "Q1".to_string(),
                description: "all doses must be within limit".to_string(),
                logic: "forall medication in medications (medication.dose <= max_dose)".to_string(),
                reason_code: None,
                reason_label: None,
                insufficient_data_reason_code: None,
                insufficient_data_reason_label: None,
            }],
        };

        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        assert!(circuit.boolean_nodes.is_empty());
        assert_eq!(circuit.compact_quantified_stages.len(), 1);
        let stage = &circuit.compact_quantified_stages[0];
        assert_eq!(stage.collection.max_items, 3);
        assert_eq!(stage.row_count, 4);
        match &stage.predicate {
            CompactQuantifiedPredicate::Comparison {
                op: ComparisonOp::Le,
                ordered_aux: Some(_),
                ..
            } => {}
            other => panic!("expected compact quantified comparison, found {other:?}"),
        }
    }

    #[test]
    #[ignore = "deep AIR/trace regression for compact quantified medication policy"]
    fn medication_policy_compact_trace_satisfies_transition_constraints() {
        let source =
            std::fs::read_to_string("programs/medication_safety.cdmn.json").expect("fixture");
        let table: CdmnTable = serde_json::from_str(&source).expect("parse fixture");
        let program = compile_cdmn_table_to_decision_program(&table).unwrap();
        let memory = vec![
            500, // max_dose
            11,  // required_class
            12,  // backup_class
            99,  // prohibited_class
            2,   // medication count
            400, // med0 dose
            11,  // med0 class
            450, // med1 dose
            12,  // med1 class
            0, 0,
        ];
        let circuit = DirectDecisionCircuitBuilder::build(&program).unwrap();
        let witness = circuit.build_witness(&program, &memory).unwrap();
        let trace = circuit.materialize_trace_rows(&witness);
        let air = DirectDecisionAir::new(circuit);
        let constraints = air.transition_constraints();
        for row in 0..trace.len() - 1 {
            let mut point = Vec::with_capacity(1 + 2 * trace[row].len());
            point.push(FieldElement::zero());
            point.extend(trace[row].iter().copied());
            point.extend(trace[row + 1].iter().copied());
            for (constraint_index, constraint) in constraints.iter().enumerate() {
                let value = constraint.evaluate(&point);
                assert!(
                    value.is_zero(),
                    "constraint {constraint_index} failed at row {row}: {value:?}"
                );
            }
        }
    }
}
