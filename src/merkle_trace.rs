use crate::error::Result;
use crate::transformer_trace::SharedTransformerTraceRow;
use crate::vanillastark::merkle::{Merkle, MerkleTree};
use serde_json;

pub struct MerkleizedTrace {
    pub tree: MerkleTree,
    pub original_len: usize,
}

impl MerkleizedTrace {
    pub fn new(rows: &[SharedTransformerTraceRow]) -> Result<Self> {
        let original_len = rows.len();
        let mut hashed_leaves = Vec::new();
        for row in rows {
            let bytes = serde_json::to_vec(row)
                .map_err(|e| crate::error::VmError::Serialization(e.to_string()))?;
            hashed_leaves.push(hash_bytes(&bytes));
        }

        if hashed_leaves.is_empty() {
            hashed_leaves.push(vec![0u8; 64]);
        }

        // Pad to power of two
        let padded_len = hashed_leaves.len().next_power_of_two();
        while hashed_leaves.len() < padded_len {
            hashed_leaves.push(vec![0u8; 64]);
        }

        Ok(Self {
            tree: MerkleTree::from_hashed_leaves(&hashed_leaves),
            original_len,
        })
    }

    pub fn root(&self) -> Vec<u8> {
        self.tree.root().to_vec()
    }

    pub fn open(&self, index: usize) -> Vec<Vec<u8>> {
        self.tree.open(index)
    }

    pub fn verify(
        root: &[u8],
        index: usize,
        path: &[Vec<u8>],
        row: &SharedTransformerTraceRow,
    ) -> bool {
        let bytes = match serde_json::to_vec(row) {
            Ok(b) => b,
            Err(_) => return false,
        };
        Merkle::verify_(root, index, path, &hash_bytes(&bytes))
    }
}

fn hash_bytes(data: &[u8]) -> Vec<u8> {
    use blake2::{Blake2b512, Digest};
    let mut hasher = Blake2b512::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DispatchInfo;
    use crate::state::MachineState;
    use crate::transformer_trace::SharedTransformerInputTraceRow;

    #[test]
    fn test_merkle_trace_roundtrip() {
        let row = SharedTransformerTraceRow::InputState(SharedTransformerInputTraceRow {
            step_number: 1,
            pre_state: MachineState::new(10),
            input_token: vec![1, 0, -1],
            decoded_state: MachineState::new(10),
            dispatch: DispatchInfo {
                layer_idx: 0,
                instruction: crate::instruction::Instruction::Halt,
            },
        });
        let rows = vec![row.clone()];
        let m_trace = MerkleizedTrace::new(&rows).unwrap();
        let root = m_trace.root();
        let path = m_trace.open(0);

        assert!(MerkleizedTrace::verify(&root, 0, &path, &row));
    }
}
