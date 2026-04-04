use serde::{Deserialize, Serialize};

use crate::state::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithmeticExpr {
    Number(Word),
    Variable(String),
    Binary {
        op: ArithmeticOp,
        left: Box<ArithmeticExpr>,
        right: Box<ArithmeticExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantifierKind {
    ForAll,
    Exists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AvailabilityPredicateKind {
    Present,
    Missing,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BooleanExpr {
    Comparison {
        lhs: ArithmeticExpr,
        op: ComparisonOp,
        rhs: ArithmeticExpr,
    },
    Availability {
        kind: AvailabilityPredicateKind,
        value: ArithmeticExpr,
    },
    Between {
        value: ArithmeticExpr,
        lower: ArithmeticExpr,
        upper: ArithmeticExpr,
    },
    And(Box<BooleanExpr>, Box<BooleanExpr>),
    Or(Box<BooleanExpr>, Box<BooleanExpr>),
    Not(Box<BooleanExpr>),
    Quantified {
        kind: QuantifierKind,
        binding: String,
        collection: String,
        predicate: Box<BooleanExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    Identifier(String),
    Number(Word),
    Plus,
    Minus,
    Star,
    LParen,
    RParen,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    ForAll,
    Exists,
    In,
    Between,
}
