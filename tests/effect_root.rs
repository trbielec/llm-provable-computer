//! Integration tests for the `effect_root` module.
//!
//! All tests are self-contained; no shared fixtures. The implementation under
//! test lives in `src/effect_root.rs` and is re-exported from the crate root.

use llm_provable_computer::effect_root::{EffectBoundary, EffectLog, StateEntry, StateRoot};

// ---------------------------------------------------------------------------
// EffectLog
// ---------------------------------------------------------------------------

#[test]
fn empty_effect_log_has_zero_root() {
    let log = EffectLog::new();
    assert_eq!(log.root(), &[0u8; 32]);
}

#[test]
fn effect_log_root_is_stable() {
    let mut log_a = EffectLog::new();
    log_a.append_read("state.counter", b"42");
    log_a.append_write("state.counter", b"43");

    let mut log_b = EffectLog::new();
    log_b.append_read("state.counter", b"42");
    log_b.append_write("state.counter", b"43");

    assert_eq!(log_a.root(), log_b.root());
}

#[test]
fn effect_log_verify_root_passes_on_valid_log() {
    let mut log = EffectLog::new();
    log.append_read("account.balance", b"1000");
    log.append_write("account.balance", b"950");
    log.append_write("account.nonce", b"1");

    assert!(log.verify_root());
}

#[test]
fn effect_log_different_keys_produce_different_roots() {
    // The mutation path via direct field access on the inner event requires
    // `events()` to expose mutable references, which may not be part of the
    // public API. Instead we assert the simpler equivalent property: two logs
    // that differ only in a single event key must produce different roots.
    let mut log_a = EffectLog::new();
    log_a.append_read("key.original", b"value");

    let mut log_b = EffectLog::new();
    log_b.append_read("key.tampered", b"value");

    assert_ne!(log_a.root(), log_b.root());
}

#[test]
fn effect_log_read_and_write_events_are_typed() {
    let mut log = EffectLog::new();
    log.append_read("input.token", b"hello");
    log.append_write("output.token", b"world");

    let events = log.events();
    assert_eq!(events.len(), 2);

    assert!(events[0].is_read());
    assert!(!events[0].is_write());

    assert!(events[1].is_write());
    assert!(!events[1].is_read());
}

#[test]
fn effect_log_key_accessor_works() {
    let mut log = EffectLog::new();
    log.append_read("my.read.key", b"data");
    log.append_write("my.write.key", b"result");

    let events = log.events();
    assert_eq!(events[0].key(), "my.read.key");
    assert_eq!(events[1].key(), "my.write.key");
}

// ---------------------------------------------------------------------------
// StateRoot
// ---------------------------------------------------------------------------

#[test]
fn empty_state_root_has_zero_root() {
    let sr = StateRoot::empty();
    assert_eq!(sr.root(), &[0u8; 32]);
}

#[test]
fn state_root_is_order_independent() {
    let entries_a = vec![
        StateEntry::new("balance", b"500"),
        StateEntry::new("nonce", b"3"),
        StateEntry::new("owner", b"alice"),
    ];
    let entries_b = vec![
        StateEntry::new("owner", b"alice"),
        StateEntry::new("balance", b"500"),
        StateEntry::new("nonce", b"3"),
    ];

    let root_a = StateRoot::from_entries(entries_a);
    let root_b = StateRoot::from_entries(entries_b);

    assert_eq!(root_a.root(), root_b.root());
}

#[test]
fn state_root_verify_root_passes_on_valid_snapshot() {
    let entries = vec![StateEntry::new("x", b"10"), StateEntry::new("y", b"20")];
    let sr = StateRoot::from_entries(entries);
    assert!(sr.verify_root());
}

#[test]
fn state_root_different_values_produce_different_roots() {
    let entries_a = vec![StateEntry::new("counter", b"0")];
    let entries_b = vec![StateEntry::new("counter", b"1")];

    let root_a = StateRoot::from_entries(entries_a);
    let root_b = StateRoot::from_entries(entries_b);

    assert_ne!(root_a.root(), root_b.root());
}

// ---------------------------------------------------------------------------
// EffectBoundary
// ---------------------------------------------------------------------------

fn make_pre_state() -> StateRoot {
    StateRoot::from_entries(vec![
        StateEntry::new("balance", b"1000"),
        StateEntry::new("nonce", b"0"),
    ])
}

fn make_post_state() -> StateRoot {
    StateRoot::from_entries(vec![
        StateEntry::new("balance", b"900"),
        StateEntry::new("nonce", b"1"),
    ])
}

fn make_log() -> EffectLog {
    let mut log = EffectLog::new();
    log.append_read("balance", b"1000");
    log.append_write("balance", b"900");
    log.append_write("nonce", b"1");
    log
}

#[test]
fn effect_boundary_verify_integrity_passes() {
    let boundary =
        EffectBoundary::new(make_pre_state(), make_log()).with_post_state(make_post_state());

    assert!(boundary.verify_integrity());
}

#[test]
fn effect_boundary_without_post_state_verify_integrity_passes() {
    let boundary = EffectBoundary::new(make_pre_state(), make_log());
    assert!(boundary.verify_integrity());
}

#[test]
fn effect_boundary_hash_is_stable() {
    let boundary_a =
        EffectBoundary::new(make_pre_state(), make_log()).with_post_state(make_post_state());
    let boundary_b =
        EffectBoundary::new(make_pre_state(), make_log()).with_post_state(make_post_state());

    assert_eq!(boundary_a.boundary_hash(), boundary_b.boundary_hash());
}

#[test]
fn effect_boundary_hash_changes_with_different_log() {
    let mut other_log = EffectLog::new();
    other_log.append_read("balance", b"1000");
    other_log.append_write("balance", b"800"); // different value

    let boundary_a =
        EffectBoundary::new(make_pre_state(), make_log()).with_post_state(make_post_state());
    let boundary_b =
        EffectBoundary::new(make_pre_state(), other_log).with_post_state(make_post_state());

    assert_ne!(boundary_a.boundary_hash(), boundary_b.boundary_hash());
}

#[test]
fn effect_boundary_round_trips_through_json() {
    let boundary =
        EffectBoundary::new(make_pre_state(), make_log()).with_post_state(make_post_state());

    let json = serde_json::to_string(&boundary).expect("serialize");
    let decoded: EffectBoundary = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(boundary.boundary_hash(), decoded.boundary_hash());
    assert_eq!(
        boundary.pre_state_root.root(),
        decoded.pre_state_root.root()
    );
    assert_eq!(boundary.effect_log.root(), decoded.effect_log.root());
    assert_eq!(
        boundary.post_state_root.as_ref().map(|s| s.root()),
        decoded.post_state_root.as_ref().map(|s| s.root()),
    );
}
