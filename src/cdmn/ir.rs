use super::ast::{ArithmeticOp, AvailabilityPredicateKind, ComparisonOp, QuantifierKind};
use crate::state::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BindingId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ValueSource {
    Cell(u8),
    BoundField { binding: BindingId, offset: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AvailabilitySource {
    Cell(u8),
    BoundField { binding: BindingId, offset: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AvailabilityIr {
    Flags {
        presence: Option<AvailabilitySource>,
        unknown: Option<AvailabilitySource>,
    },
    Status(AvailabilitySource),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValueRefIr {
    pub(crate) source: ValueSource,
    pub(crate) availability: AvailabilityIr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArithmeticIr {
    Number(Word),
    Value(ValueRefIr),
    Binary {
        op: ArithmeticOp,
        left: Box<ArithmeticIr>,
        right: Box<ArithmeticIr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QuantifiedCollection {
    pub(crate) binding: BindingId,
    pub(crate) name: String,
    pub(crate) count_memory: u8,
    pub(crate) base_memory: u8,
    pub(crate) stride: u8,
    pub(crate) max_items: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BooleanIr {
    Comparison {
        lhs: ArithmeticIr,
        op: ComparisonOp,
        rhs: ArithmeticIr,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: ArithmeticIr,
    },
    Between {
        value: ArithmeticIr,
        lower: ArithmeticIr,
        upper: ArithmeticIr,
    },
    And(Box<BooleanIr>, Box<BooleanIr>),
    Or(Box<BooleanIr>, Box<BooleanIr>),
    Not(Box<BooleanIr>),
    Quantified {
        kind: QuantifierKind,
        collection: QuantifiedCollection,
        predicate: Box<BooleanIr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) expression: ArithmeticIr,
    pub(crate) memory: u8,
    pub(crate) status_memory: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FailureReason {
    pub(crate) code: Word,
    pub(crate) label: String,
    pub(crate) description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedConstraint {
    pub(crate) id: String,
    pub(crate) description: String,
    pub(crate) expression: BooleanIr,
    pub(crate) fail_label: String,
    pub(crate) reason: FailureReason,
    pub(crate) insufficient_data_label: String,
    pub(crate) insufficient_data_reason: FailureReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompilationPlan {
    pub(crate) scratch_start: u8,
    pub(crate) definitions: Vec<ResolvedDefinition>,
    pub(crate) constraints: Vec<ResolvedConstraint>,
}
