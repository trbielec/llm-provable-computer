use super::field::FieldElement;
use super::rescue_prime::RescuePrime;
use super::TranscriptHasher;

#[derive(Clone, Debug)]
pub struct RescueProofStream {
    objects: Vec<FieldElement>,
    read_index: usize,
}

pub struct RescueTranscriptHasher;

impl Default for RescueProofStream {
    fn default() -> Self {
        Self::new()
    }
}

impl RescueProofStream {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            read_index: 0,
        }
    }

    pub fn push(&mut self, value: FieldElement) {
        self.objects.push(value);
    }

    pub fn pull(&mut self) -> FieldElement {
        assert!(
            self.read_index < self.objects.len(),
            "RescueProofStream is empty"
        );
        let value = self.objects[self.read_index];
        self.read_index += 1;
        value
    }

    pub fn prover_fiat_shamir(&self) -> FieldElement {
        digest_elements(&self.objects)
    }

    pub fn verifier_fiat_shamir(&self) -> FieldElement {
        digest_elements(&self.objects[..self.read_index])
    }
}

pub fn digest_bytes(bytes: &[u8]) -> FieldElement {
    let elements: Vec<FieldElement> = bytes
        .chunks(16)
        .map(|chunk| {
            let mut padded = [0u8; 16];
            padded[..chunk.len()].copy_from_slice(chunk);
            FieldElement::new(u128::from_le_bytes(padded))
        })
        .collect();
    digest_elements(&elements)
}

pub fn digest_elements(elements: &[FieldElement]) -> FieldElement {
    let rescue = RescuePrime::new();
    let mut state = FieldElement::zero();
    for element in elements {
        state = rescue.hash_pair(state, *element);
    }
    state
}

impl TranscriptHasher for RescueTranscriptHasher {
    type Digest = FieldElement;

    fn digest_bytes(bytes: &[u8]) -> Self::Digest {
        digest_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_bytes_changes_when_input_changes() {
        assert_ne!(digest_bytes(b"abc"), digest_bytes(b"abd"));
    }

    #[test]
    fn proof_stream_transcript_depends_on_prefix() {
        let mut stream = RescueProofStream::new();
        stream.push(FieldElement::new(1));
        stream.push(FieldElement::new(2));
        let full = stream.prover_fiat_shamir();
        let _ = stream.pull();
        let prefix = stream.verifier_fiat_shamir();
        assert_ne!(full, prefix);
    }
}
