use serde::{Deserialize, Serialize};

use crate::error::{Result, VmError};

pub type Word = i64;

pub const WORD_BITS: usize = 64;
pub const MIN_D_MODEL: usize = 84; // 8 (pc) + WORD_BITS + 8 (sp) + 3 (flags) + 1 padding

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineState {
    pub pc: u8,
    pub acc: Word,
    pub sp: u8,
    pub zero_flag: bool,
    pub carry_flag: bool,
    pub halted: bool,
    pub memory: Vec<Word>,
}

impl Default for MachineState {
    /// Creates a state with a small default memory (16 words, all zeroed).
    fn default() -> Self {
        Self::new(16)
    }
}

impl MachineState {
    pub fn new(memory_size: usize) -> Self {
        Self {
            pc: 0,
            acc: 0,
            sp: memory_size.min(usize::from(u8::MAX)) as u8,
            zero_flag: true,
            carry_flag: false,
            halted: false,
            memory: vec![0; memory_size],
        }
    }

    pub fn with_memory(memory: Vec<Word>) -> Self {
        let memory_size = memory.len();
        Self {
            memory,
            ..Self::new(memory_size)
        }
    }
}

pub fn encode_state(state: &MachineState, d_model: usize) -> Result<Vec<f32>> {
    if d_model < MIN_D_MODEL {
        return Err(VmError::InvalidConfig(format!(
            "d_model must be at least {MIN_D_MODEL}, got {d_model}"
        )));
    }

    let mut token = vec![0.0f32; d_model];

    for (bit, slot) in token.iter_mut().enumerate().take(8) {
        *slot = encode_bit(((state.pc >> bit) & 1) == 1);
    }

    let acc_bits = state.acc as u64;
    for (bit, slot) in token.iter_mut().enumerate().skip(8).take(WORD_BITS) {
        *slot = encode_bit(((acc_bits >> (bit - 8)) & 1) == 1);
    }

    for (bit, slot) in token.iter_mut().enumerate().skip(8 + WORD_BITS).take(8) {
        *slot = encode_bit(((state.sp >> (bit - (8 + WORD_BITS))) & 1) == 1);
    }

    let flags_start = 8 + WORD_BITS + 8;
    token[flags_start] = encode_bit(state.zero_flag);
    token[flags_start + 1] = encode_bit(state.carry_flag);
    token[flags_start + 2] = encode_bit(state.halted);
    token[flags_start + 3] = 0.0;

    Ok(token)
}

pub fn decode_state(token: &[f32], memory: Vec<Word>) -> Result<MachineState> {
    if token.len() < MIN_D_MODEL {
        return Err(VmError::InvalidConfig(format!(
            "token must have at least {MIN_D_MODEL} dimensions, got {}",
            token.len()
        )));
    }

    let mut pc = 0u8;
    for (bit, value) in token.iter().enumerate().take(8) {
        if decode_bit(*value) {
            pc |= 1 << bit;
        }
    }

    let mut acc_bits = 0u64;
    for (bit, value) in token.iter().enumerate().skip(8).take(WORD_BITS) {
        if decode_bit(*value) {
            acc_bits |= 1 << (bit - 8);
        }
    }

    let mut sp = 0u8;
    for (bit, value) in token.iter().enumerate().skip(8 + WORD_BITS).take(8) {
        if decode_bit(*value) {
            sp |= 1 << (bit - (8 + WORD_BITS));
        }
    }

    let flags_start = 8 + WORD_BITS + 8;

    Ok(MachineState {
        pc,
        acc: acc_bits as Word,
        sp,
        zero_flag: decode_bit(token[flags_start]),
        carry_flag: decode_bit(token[flags_start + 1]),
        halted: decode_bit(token[flags_start + 2]),
        memory,
    })
}

fn encode_bit(value: bool) -> f32 {
    if value {
        1.0
    } else {
        -1.0
    }
}

fn decode_bit(value: f32) -> bool {
    value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_zero_defaults() {
        let state = MachineState::new(8);
        assert_eq!(state.pc, 0);
        assert_eq!(state.acc, 0);
        assert_eq!(state.sp, 8);
        assert!(state.zero_flag);
        assert!(!state.carry_flag);
        assert!(!state.halted);
        assert_eq!(state.memory, vec![0; 8]);
    }

    #[test]
    fn new_state_sp_capped_at_u8_max() {
        let state = MachineState::new(300);
        assert_eq!(state.sp, 255);
    }

    #[test]
    fn with_memory_preserves_values() {
        let state = MachineState::with_memory(vec![1, 2, 3]);
        assert_eq!(state.memory, vec![1, 2, 3]);
        assert_eq!(state.sp, 3);
    }

    #[test]
    fn encode_decode_round_trips_zero_state() {
        let state = MachineState::new(4);
        let token = encode_state(&state, MIN_D_MODEL).unwrap();
        let decoded = decode_state(&token, state.memory.clone()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn encode_decode_round_trips_max_values() {
        let state = MachineState {
            pc: 255,
            acc: Word::MAX,
            sp: 255,
            zero_flag: true,
            carry_flag: true,
            halted: true,
            memory: vec![0; 4],
        };
        let token = encode_state(&state, MIN_D_MODEL).unwrap();
        let decoded = decode_state(&token, state.memory.clone()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn encode_decode_round_trips_negative_acc() {
        let state = MachineState {
            pc: 0,
            acc: -1,
            sp: 4,
            zero_flag: false,
            carry_flag: false,
            halted: false,
            memory: vec![0; 4],
        };
        let token = encode_state(&state, MIN_D_MODEL).unwrap();
        let decoded = decode_state(&token, state.memory.clone()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn encode_decode_round_trips_min_acc() {
        let state = MachineState {
            pc: 42,
            acc: Word::MIN,
            sp: 8,
            zero_flag: false,
            carry_flag: true,
            halted: false,
            memory: vec![0; 8],
        };
        let token = encode_state(&state, MIN_D_MODEL).unwrap();
        let decoded = decode_state(&token, state.memory.clone()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn encode_rejects_d_model_too_small() {
        let state = MachineState::new(4);
        let err = encode_state(&state, MIN_D_MODEL - 1).unwrap_err();
        assert!(err.to_string().contains("d_model must be at least"));
    }

    #[test]
    fn decode_rejects_token_too_short() {
        let token = vec![0.0; MIN_D_MODEL - 1];
        let err = decode_state(&token, vec![0; 4]).unwrap_err();
        assert!(err.to_string().contains("at least"));
    }

    #[test]
    fn encode_accepts_larger_d_model() {
        let state = MachineState::new(4);
        let larger_d_model = MIN_D_MODEL + 32;
        let token = encode_state(&state, larger_d_model).unwrap();
        assert_eq!(token.len(), larger_d_model);
        // Extra dimensions should be zero
        for &val in &token[MIN_D_MODEL..] {
            assert_eq!(val, 0.0);
        }
        let decoded = decode_state(&token, state.memory.clone()).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn encode_bit_maps_correctly() {
        assert_eq!(encode_bit(true), 1.0);
        assert_eq!(encode_bit(false), -1.0);
    }

    #[test]
    fn decode_bit_threshold_at_zero() {
        assert!(decode_bit(0.0));
        assert!(decode_bit(0.5));
        assert!(decode_bit(1.0));
        assert!(!decode_bit(-0.001));
        assert!(!decode_bit(-1.0));
    }
}
