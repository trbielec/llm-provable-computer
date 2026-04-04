pub mod config;
pub mod error;
pub mod instruction;
pub mod proof;
pub mod state;
pub mod vanillastark;

pub use config::Attention2DMode;
pub use error::{Result, VmError};
pub use instruction::{Instruction, Program};
pub use proof::{
    deserialize_execution_stark_proof_json, serialize_execution_stark_proof_json,
    verify_execution_stark, VanillaStarkExecutionClaim, VanillaStarkExecutionProof,
    VanillaStarkProofOptions, EXECUTION_PROOF_FORMAT_VERSION,
};
pub use state::{MachineState, Word, WORD_BITS};
