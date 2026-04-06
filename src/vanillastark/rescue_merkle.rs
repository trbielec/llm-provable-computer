use super::field::FieldElement;
use super::rescue_prime::RescuePrime;
use super::MerkleHasher;

#[derive(Clone, Debug)]
pub struct RescueMerkleTree {
    levels: Vec<Vec<FieldElement>>,
}

pub struct RescueMerkleHasher;

impl MerkleHasher for RescueMerkleHasher {
    type Digest = FieldElement;

    fn hash_leaf(bytes: &[u8]) -> Self::Digest {
        let mut padded = [0u8; 16];
        let len = bytes.len().min(16);
        padded[..len].copy_from_slice(&bytes[..len]);
        FieldElement::new(u128::from_le_bytes(padded))
    }

    fn hash_pair(left: &Self::Digest, right: &Self::Digest) -> Self::Digest {
        RescuePrime::new().hash_pair(*left, *right)
    }
}

impl RescueMerkleTree {
    pub fn from_leaves(leaves: &[FieldElement]) -> Self {
        assert!(
            !leaves.is_empty(),
            "cannot build Rescue Merkle tree from zero leaves"
        );

        let rescue = RescuePrime::new();
        let mut current = leaves.to_vec();
        let target_len = current.len().next_power_of_two();
        while current.len() < target_len {
            current.push(FieldElement::zero());
        }

        let mut levels = vec![current];
        while levels.last().expect("levels").len() > 1 {
            let parents = levels
                .last()
                .expect("current level")
                .chunks_exact(2)
                .map(|pair| rescue.hash_pair(pair[0], pair[1]))
                .collect();
            levels.push(parents);
        }

        Self { levels }
    }

    pub fn root(&self) -> FieldElement {
        self.levels.last().expect("root level")[0]
    }

    pub fn open(&self, index: usize) -> Vec<FieldElement> {
        let mut path = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut current_index = index;
        for level in &self.levels[..self.levels.len() - 1] {
            path.push(level[current_index ^ 1]);
            current_index >>= 1;
        }
        path
    }

    pub fn verify(
        root: FieldElement,
        index: usize,
        path: &[FieldElement],
        leaf: FieldElement,
    ) -> bool {
        let rescue = RescuePrime::new();
        let mut current = leaf;
        let mut current_index = index;

        for sibling in path {
            current = if current_index.is_multiple_of(2) {
                rescue.hash_pair(current, *sibling)
            } else {
                rescue.hash_pair(*sibling, current)
            };
            current_index >>= 1;
        }

        current == root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescue_merkle_round_trip() {
        let leaves = vec![
            FieldElement::new(1),
            FieldElement::new(2),
            FieldElement::new(3),
            FieldElement::new(4),
        ];
        let tree = RescueMerkleTree::from_leaves(&leaves);
        let path = tree.open(2);
        assert!(RescueMerkleTree::verify(tree.root(), 2, &path, leaves[2]));
    }
}
