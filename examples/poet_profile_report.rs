use llm_provable_computer::instruction::{Instruction, Program};
use llm_provable_computer::poet_proof::inspect_poet_program;

fn main() {
    let program = Program::new(
        vec![
            Instruction::LoadImmediate(1),
            Instruction::JumpIfZero(4),
            Instruction::XorImmediate(1),
            Instruction::Halt,
        ],
        0,
    );

    let analysis = inspect_poet_program(&program);
    println!("poet reduced profile");
    println!("  supported: {}", analysis.is_fully_supported());
    println!("  required_ff_dim: {}", analysis.required_ff_dim);
    println!("  word_bits: {}", analysis.word_bits);
    println!("  token_len: {}", analysis.token_len);

    for instruction in analysis.instructions {
        match instruction.unsupported_reason {
            Some(reason) => println!(
                "  pc {}: {} unsupported ({})",
                instruction.pc, instruction.instruction, reason
            ),
            None => println!(
                "  pc {}: {} supported (ff_dim >= {})",
                instruction.pc,
                instruction.instruction,
                instruction.required_ff_dim.unwrap_or(0)
            ),
        }
    }
}
