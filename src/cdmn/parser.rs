use super::ast::{
    ArithmeticExpr, ArithmeticOp, AvailabilityPredicateKind, BooleanExpr, ComparisonOp,
    QuantifierKind, Token,
};
use crate::error::{Result, VmError};
use crate::state::Word;

pub fn parse_logic_expr(input: &str) -> Result<BooleanExpr> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_boolean_expr()?;
    parser.ensure_exhausted()?;
    Ok(expr)
}

pub(crate) fn parse_arithmetic_logic_expr(input: &str) -> Result<ArithmeticExpr> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_arithmetic_expr()?;
    parser.ensure_exhausted()?;
    Ok(expr)
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let chars = input.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut idx = 0usize;

    while idx < chars.len() {
        let ch = chars[idx];
        match ch {
            ch if ch.is_ascii_whitespace() => idx += 1,
            '(' => {
                tokens.push(Token::LParen);
                idx += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                idx += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                idx += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                idx += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                idx += 1;
            }
            '=' => {
                if chars.get(idx + 1) == Some(&'=') {
                    tokens.push(Token::EqEq);
                    idx += 2;
                } else {
                    return Err(parser_error(format!(
                        "unexpected `=` at byte offset {idx}; expected `==`"
                    )));
                }
            }
            '!' => {
                if chars.get(idx + 1) == Some(&'=') {
                    tokens.push(Token::NotEq);
                    idx += 2;
                } else {
                    return Err(parser_error(format!(
                        "unexpected `!` at byte offset {idx}; expected `!=`"
                    )));
                }
            }
            '<' => {
                if chars.get(idx + 1) == Some(&'=') {
                    tokens.push(Token::Le);
                    idx += 2;
                } else {
                    tokens.push(Token::Lt);
                    idx += 1;
                }
            }
            '>' => {
                if chars.get(idx + 1) == Some(&'=') {
                    tokens.push(Token::Ge);
                    idx += 2;
                } else {
                    tokens.push(Token::Gt);
                    idx += 1;
                }
            }
            ch if ch.is_ascii_digit() => {
                let start = idx;
                idx += 1;
                while chars.get(idx).is_some_and(char::is_ascii_digit) {
                    idx += 1;
                }
                let number = input[start..idx].parse::<Word>().map_err(|_| {
                    parser_error(format!("invalid integer `{}`", &input[start..idx]))
                })?;
                tokens.push(Token::Number(number));
            }
            ch if is_identifier_start(ch) => {
                let start = idx;
                idx += 1;
                while chars.get(idx).is_some_and(|next| is_identifier_part(*next)) {
                    idx += 1;
                }
                let identifier = &input[start..idx];
                tokens.push(match identifier {
                    "and" => Token::And,
                    "between" => Token::Between,
                    "or" => Token::Or,
                    "not" => Token::Not,
                    "forall" => Token::ForAll,
                    "exists" => Token::Exists,
                    "in" => Token::In,
                    _ => Token::Identifier(identifier.to_string()),
                });
            }
            _ => {
                return Err(parser_error(format!(
                    "unexpected character `{ch}` at byte offset {idx}"
                )));
            }
        }
    }

    Ok(tokens)
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_identifier_part(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.')
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn ensure_exhausted(&self) -> Result<()> {
        if self.pos == self.tokens.len() {
            Ok(())
        } else {
            Err(parser_error(format!(
                "unexpected trailing token {:?}",
                self.tokens[self.pos]
            )))
        }
    }

    fn parse_boolean_expr(&mut self) -> Result<BooleanExpr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<BooleanExpr> {
        let mut expr = self.parse_and()?;
        while self.take_if(|token| matches!(token, Token::Or)).is_some() {
            let rhs = self.parse_and()?;
            expr = BooleanExpr::Or(Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<BooleanExpr> {
        let mut expr = self.parse_unary_boolean()?;
        while self.take_if(|token| matches!(token, Token::And)).is_some() {
            let rhs = self.parse_unary_boolean()?;
            expr = BooleanExpr::And(Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_unary_boolean(&mut self) -> Result<BooleanExpr> {
        if self.take_if(|token| matches!(token, Token::Not)).is_some() {
            return Ok(BooleanExpr::Not(Box::new(self.parse_unary_boolean()?)));
        }
        if matches!(self.peek(), Some(Token::ForAll | Token::Exists)) {
            return self.parse_quantified();
        }
        if self.looks_like_availability_check() {
            return self.parse_availability_check();
        }
        if matches!(self.peek(), Some(Token::LParen)) {
            let checkpoint = self.pos;
            self.bump();
            let grouped = self.parse_boolean_expr().and_then(|expr| {
                self.expect_token(Token::RParen, "closing `)` after boolean expression")?;
                Ok(expr)
            });
            if let Ok(expr) = grouped {
                return Ok(expr);
            }
            self.pos = checkpoint;
        }
        self.parse_comparison()
    }

    fn parse_quantified(&mut self) -> Result<BooleanExpr> {
        let kind = match self.bump() {
            Some(Token::ForAll) => QuantifierKind::ForAll,
            Some(Token::Exists) => QuantifierKind::Exists,
            _ => return Err(parser_error("expected `forall` or `exists`".to_string())),
        };
        let binding = self.expect_identifier("quantifier binding")?;
        self.expect_token(Token::In, "`in` after quantifier binding")?;
        let collection = self.expect_identifier("quantifier collection")?;
        let predicate = if self
            .take_if(|token| matches!(token, Token::LParen))
            .is_some()
        {
            let expr = self.parse_boolean_expr()?;
            self.expect_token(Token::RParen, "closing `)` after quantified predicate")?;
            expr
        } else {
            self.parse_unary_boolean()?
        };
        Ok(BooleanExpr::Quantified {
            kind,
            binding,
            collection,
            predicate: Box::new(predicate),
        })
    }

    fn parse_comparison(&mut self) -> Result<BooleanExpr> {
        let lhs = self.parse_arithmetic_expr()?;
        if self
            .take_if(|token| matches!(token, Token::Between))
            .is_some()
        {
            let lower = self.parse_arithmetic_expr()?;
            self.expect_token(Token::And, "`and` separating range bounds")?;
            let upper = self.parse_arithmetic_expr()?;
            return Ok(BooleanExpr::Between {
                value: lhs,
                lower,
                upper,
            });
        }
        let op = match self.bump() {
            Some(Token::EqEq) => ComparisonOp::Eq,
            Some(Token::NotEq) => ComparisonOp::Ne,
            Some(Token::Lt) => ComparisonOp::Lt,
            Some(Token::Le) => ComparisonOp::Le,
            Some(Token::Gt) => ComparisonOp::Gt,
            Some(Token::Ge) => ComparisonOp::Ge,
            other => {
                return Err(parser_error(format!(
                    "expected comparison operator after arithmetic expression, found {other:?}"
                )));
            }
        };
        let rhs = self.parse_arithmetic_expr()?;
        Ok(BooleanExpr::Comparison { lhs, op, rhs })
    }

    fn parse_availability_check(&mut self) -> Result<BooleanExpr> {
        let kind = match self.bump() {
            Some(Token::Identifier(name)) if name == "present" => {
                AvailabilityPredicateKind::Present
            }
            Some(Token::Identifier(name)) if name == "missing" => {
                AvailabilityPredicateKind::Missing
            }
            Some(Token::Identifier(name)) if name == "unknown" => {
                AvailabilityPredicateKind::Unknown
            }
            other => {
                return Err(parser_error(format!(
                    "expected availability predicate, found {other:?}"
                )));
            }
        };
        self.expect_token(Token::LParen, "opening `(` after availability predicate")?;
        let value = self.parse_arithmetic_expr()?;
        self.expect_token(Token::RParen, "closing `)` after availability predicate")?;
        Ok(BooleanExpr::Availability { kind, value })
    }

    fn parse_arithmetic_expr(&mut self) -> Result<ArithmeticExpr> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<ArithmeticExpr> {
        let mut expr = self.parse_mul()?;
        loop {
            let op = if self.take_if(|token| matches!(token, Token::Plus)).is_some() {
                Some(ArithmeticOp::Add)
            } else if self
                .take_if(|token| matches!(token, Token::Minus))
                .is_some()
            {
                Some(ArithmeticOp::Sub)
            } else {
                None
            };
            let Some(op) = op else {
                break;
            };
            let rhs = self.parse_mul()?;
            expr = ArithmeticExpr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_mul(&mut self) -> Result<ArithmeticExpr> {
        let mut expr = self.parse_unary_arithmetic()?;
        while self.take_if(|token| matches!(token, Token::Star)).is_some() {
            let rhs = self.parse_unary_arithmetic()?;
            expr = ArithmeticExpr::Binary {
                op: ArithmeticOp::Mul,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_unary_arithmetic(&mut self) -> Result<ArithmeticExpr> {
        if self
            .take_if(|token| matches!(token, Token::Minus))
            .is_some()
        {
            let expr = self.parse_unary_arithmetic()?;
            return Ok(match expr {
                ArithmeticExpr::Number(value) => ArithmeticExpr::Number(-value),
                other => ArithmeticExpr::Binary {
                    op: ArithmeticOp::Sub,
                    left: Box::new(ArithmeticExpr::Number(0)),
                    right: Box::new(other),
                },
            });
        }
        self.parse_arithmetic_primary()
    }

    fn parse_arithmetic_primary(&mut self) -> Result<ArithmeticExpr> {
        match self.bump() {
            Some(Token::Number(value)) => Ok(ArithmeticExpr::Number(value)),
            Some(Token::Identifier(name)) => Ok(ArithmeticExpr::Variable(name)),
            Some(Token::LParen) => {
                let expr = self.parse_arithmetic_expr()?;
                self.expect_token(Token::RParen, "closing `)` after arithmetic expression")?;
                Ok(expr)
            }
            other => Err(parser_error(format!(
                "expected number, identifier, or parenthesized expression, found {other:?}"
            ))),
        }
    }

    fn expect_identifier(&mut self, subject: &str) -> Result<String> {
        match self.bump() {
            Some(Token::Identifier(name)) => Ok(name),
            other => Err(parser_error(format!("expected {subject}, found {other:?}"))),
        }
    }

    fn expect_token(&mut self, expected: Token, subject: &str) -> Result<()> {
        match self.bump() {
            Some(token) if token == expected => Ok(()),
            other => Err(parser_error(format!("expected {subject}, found {other:?}"))),
        }
    }

    fn take_if(&mut self, predicate: impl FnOnce(&Token) -> bool) -> Option<Token> {
        if self.peek().is_some_and(predicate) {
            self.bump()
        } else {
            None
        }
    }

    fn looks_like_availability_check(&self) -> bool {
        matches!(
            (self.peek(), self.tokens.get(self.pos + 1)),
            (Some(Token::Identifier(name)), Some(Token::LParen))
                if matches!(name.as_str(), "present" | "missing" | "unknown")
        )
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned()?;
        self.pos += 1;
        Some(token)
    }
}

fn parser_error(message: String) -> VmError {
    VmError::InvalidConfig(message)
}
