/// Merkle tree using blake2b.
/// Ported from the stark-anatomy Python reference implementation.
use blake2::{Blake2b512, Digest};

use super::MerkleHasher;

pub struct Merkle;
pub struct Blake2MerkleHasher;

#[derive(Clone, Debug)]
pub struct MerkleTree {
    levels: Vec<Vec<Vec<u8>>>,
}

fn hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Blake2b512::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

impl MerkleHasher for Blake2MerkleHasher {
    type Digest = Vec<u8>;

    fn hash_leaf(bytes: &[u8]) -> Self::Digest {
        hash(bytes)
    }

    fn hash_pair(left: &Self::Digest, right: &Self::Digest) -> Self::Digest {
        let mut combined = left.clone();
        combined.extend_from_slice(right);
        hash(&combined)
    }
}

impl MerkleTree {
    pub fn from_hashed_leaves(leaves: &[Vec<u8>]) -> Self {
        assert!(
            leaves.len().is_power_of_two(),
            "length must be power of two"
        );
        assert!(
            !leaves.is_empty(),
            "cannot build Merkle tree from zero leaves"
        );

        let mut levels = Vec::new();
        levels.push(leaves.to_vec());

        while levels.last().unwrap().len() > 1 {
            let current = levels.last().unwrap();
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks_exact(2) {
                let mut combined = pair[0].clone();
                combined.extend_from_slice(&pair[1]);
                next.push(hash(&combined));
            }
            levels.push(next);
        }

        Self { levels }
    }

    pub fn from_data(data_array: &[Vec<u8>]) -> Self {
        let hashed: Vec<Vec<u8>> = data_array.iter().map(|data| hash(data)).collect();
        Self::from_hashed_leaves(&hashed)
    }

    pub fn root(&self) -> &[u8] {
        self.levels
            .last()
            .expect("Merkle tree must have at least one level")[0]
            .as_slice()
    }

    pub fn open(&self, index: usize) -> Vec<Vec<u8>> {
        let mut path = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut current_index = index;
        assert!(
            current_index < self.levels[0].len(),
            "cannot open invalid index"
        );

        for level in &self.levels[..self.levels.len() - 1] {
            path.push(level[current_index ^ 1].clone());
            current_index >>= 1;
        }
        path
    }

    pub fn open_many(&self, indices: &[usize]) -> Vec<Vec<Vec<u8>>> {
        indices.iter().map(|&index| self.open(index)).collect()
    }
}

impl Merkle {
    /// Commit to pre-hashed leaves (each leaf is already a hash digest).
    pub fn commit_(leafs: &[Vec<u8>]) -> Vec<u8> {
        MerkleTree::from_hashed_leaves(leafs).root().to_vec()
    }

    /// Commit to raw data (hash each element first, then build tree).
    pub fn commit(data_array: &[Vec<u8>]) -> Vec<u8> {
        MerkleTree::from_data(data_array).root().to_vec()
    }

    /// Open a leaf in the pre-hashed tree (return authentication path).
    pub fn open_(index: usize, leafs: &[Vec<u8>]) -> Vec<Vec<u8>> {
        MerkleTree::from_hashed_leaves(leafs).open(index)
    }

    /// Open a data element (hash each element first, then open in tree).
    pub fn open(index: usize, data_array: &[Vec<u8>]) -> Vec<Vec<u8>> {
        MerkleTree::from_data(data_array).open(index)
    }

    /// Verify a pre-hashed leaf against a root and authentication path.
    pub fn verify_(root: &[u8], index: usize, path: &[Vec<u8>], leaf: &[u8]) -> bool {
        if path.is_empty() {
            return root == leaf;
        }

        let mut current_hash = leaf.to_vec();
        let mut current_index = index;

        for sibling in path {
            let mut combined = Vec::with_capacity(current_hash.len() + sibling.len());
            if current_index % 2 == 0 {
                combined.extend_from_slice(&current_hash);
                combined.extend_from_slice(sibling);
            } else {
                combined.extend_from_slice(sibling);
                combined.extend_from_slice(&current_hash);
            }
            current_hash = hash(&combined);
            current_index >>= 1;
        }

        root == current_hash
    }

    /// Verify a data element against a root and authentication path.
    pub fn verify(root: &[u8], index: usize, path: &[Vec<u8>], data_element: &[u8]) -> bool {
        Self::verify_(root, index, path, &hash(data_element))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_commit_open_verify() {
        let n = 16;
        let leafs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 32]).collect();
        let root = Merkle::commit_(&leafs);

        for i in 0..n {
            let path = Merkle::open_(i, &leafs);
            assert!(Merkle::verify_(&root, i, &path, &leafs[i]));
        }
    }

    #[test]
    fn test_merkle_wrong_leaf_fails() {
        let n = 16;
        let leafs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 32]).collect();
        let root = Merkle::commit_(&leafs);

        let path = Merkle::open_(0, &leafs);
        assert!(!Merkle::verify_(&root, 0, &path, &[255u8; 32]));
    }

    #[test]
    fn test_merkle_wrong_index_fails() {
        let n = 16;
        let leafs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 32]).collect();
        let root = Merkle::commit_(&leafs);

        let path = Merkle::open_(0, &leafs);
        assert!(!Merkle::verify_(&root, 1, &path, &leafs[0]));
    }

    #[test]
    fn test_merkle_tree_open_many_matches_single_open() {
        let n = 16;
        let leaves: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 32]).collect();
        let tree = MerkleTree::from_hashed_leaves(&leaves);
        let indices = vec![0, 3, 7, 12];
        let batched = tree.open_many(&indices);

        for (position, path) in indices.iter().zip(batched.iter()) {
            assert_eq!(*path, tree.open(*position));
            assert!(Merkle::verify_(
                tree.root(),
                *position,
                path,
                &leaves[*position]
            ));
        }
    }

    #[test]
    fn test_merkle_tree_from_data_matches_legacy_helpers() {
        let n = 8;
        let data: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 8]).collect();
        let tree = MerkleTree::from_data(&data);
        let root = Merkle::commit(&data);

        assert_eq!(tree.root(), root.as_slice());
        for index in 0..n {
            let path = tree.open(index);
            assert_eq!(path, Merkle::open(index, &data));
            assert!(Merkle::verify(tree.root(), index, &path, &data[index]));
        }
    }
}
