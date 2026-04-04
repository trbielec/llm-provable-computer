use llm_provable_computer::{
    cdmn::{ArithmeticExpr, ArithmeticOp, ComparisonOp, QuantifierKind},
    parse_logic_expr, AvailabilityPredicateKind, BooleanExpr,
};

#[test]
fn parses_current_budget_constraint() {
    let expr = parse_logic_expr("marketing + research <= total_budget").unwrap();
    assert_eq!(
        expr,
        BooleanExpr::Comparison {
            lhs: ArithmeticExpr::Binary {
                op: ArithmeticOp::Add,
                left: Box::new(ArithmeticExpr::Variable("marketing".to_string())),
                right: Box::new(ArithmeticExpr::Variable("research".to_string())),
            },
            op: ComparisonOp::Le,
            rhs: ArithmeticExpr::Variable("total_budget".to_string()),
        }
    );
}

#[test]
fn parses_quantified_expression() {
    let expr =
        parse_logic_expr("forall medication in medications (medication.dose <= max_dose)").unwrap();
    assert!(matches!(
        expr,
        BooleanExpr::Quantified {
            kind: QuantifierKind::ForAll,
            ..
        }
    ));
}

#[test]
fn parses_grouped_boolean_expression() {
    let expr = parse_logic_expr(
        "exists medication in medications ((medication.class_code == required_class) or (medication.class_code == backup_class)) and not (max_dose == 0)",
    )
    .unwrap();
    assert!(matches!(
        expr,
        BooleanExpr::And(left, right)
            if matches!(
                left.as_ref(),
                BooleanExpr::Quantified {
                    kind: QuantifierKind::Exists,
                    predicate,
                    ..
                } if matches!(predicate.as_ref(), BooleanExpr::Or(_, _))
            ) && matches!(right.as_ref(), BooleanExpr::Not(_))
    ));
}

#[test]
fn parses_between_expression() {
    let expr = parse_logic_expr("wbc_count between wbc_min and wbc_max").unwrap();
    assert_eq!(
        expr,
        BooleanExpr::Between {
            value: ArithmeticExpr::Variable("wbc_count".to_string()),
            lower: ArithmeticExpr::Variable("wbc_min".to_string()),
            upper: ArithmeticExpr::Variable("wbc_max".to_string()),
        }
    );
}

#[test]
fn parses_availability_predicates() {
    let expr = parse_logic_expr("present(lab_value) and not unknown(panel.result)").unwrap();
    assert!(matches!(
        expr,
        BooleanExpr::And(left, right)
            if matches!(
                left.as_ref(),
                BooleanExpr::Availability {
                    kind: AvailabilityPredicateKind::Present,
                    value: ArithmeticExpr::Variable(name),
                } if name == "lab_value"
            ) && matches!(
                right.as_ref(),
                BooleanExpr::Not(inner)
                    if matches!(
                        inner.as_ref(),
                        BooleanExpr::Availability {
                            kind: AvailabilityPredicateKind::Unknown,
                            value: ArithmeticExpr::Variable(name),
                        } if name == "panel.result"
                    )
            )
    ));
}
