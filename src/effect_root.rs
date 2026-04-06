use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};

fn hash_bytes_32(data: &[u8]) -> [u8; 32] {
    let full = Blake2b512::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&full[..32]);
    out
}

// ---------------------------------------------------------------------------
// ReadEvent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadEvent {
    pub key: String,
    pub value_hash: [u8; 32],
}

impl ReadEvent {
    pub fn new(key: impl Into<String>, value_bytes: &[u8]) -> Self {
        Self {
            key: key.into(),
            value_hash: hash_bytes_32(value_bytes),
        }
    }
}

// ---------------------------------------------------------------------------
// WriteEvent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteEvent {
    pub key: String,
    pub value_hash: [u8; 32],
}

impl WriteEvent {
    pub fn new(key: impl Into<String>, value_bytes: &[u8]) -> Self {
        Self {
            key: key.into(),
            value_hash: hash_bytes_32(value_bytes),
        }
    }
}

// ---------------------------------------------------------------------------
// EffectEvent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffectEvent {
    Read(ReadEvent),
    Write(WriteEvent),
}

impl EffectEvent {
    pub fn key(&self) -> &str {
        match self {
            EffectEvent::Read(e) => &e.key,
            EffectEvent::Write(e) => &e.key,
        }
    }

    pub fn is_read(&self) -> bool {
        matches!(self, EffectEvent::Read(_))
    }

    pub fn is_write(&self) -> bool {
        matches!(self, EffectEvent::Write(_))
    }
}

// ---------------------------------------------------------------------------
// EffectLog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectLog {
    events: Vec<EffectEvent>,
    root: [u8; 32],
}

impl Default for EffectLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectLog {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            root: [0u8; 32],
        }
    }

    pub fn append_read(&mut self, key: impl Into<String>, value_bytes: &[u8]) {
        self.events
            .push(EffectEvent::Read(ReadEvent::new(key, value_bytes)));
        self.root = compute_event_log_root(&self.events);
    }

    pub fn append_write(&mut self, key: impl Into<String>, value_bytes: &[u8]) {
        self.events
            .push(EffectEvent::Write(WriteEvent::new(key, value_bytes)));
        self.root = compute_event_log_root(&self.events);
    }

    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }

    pub fn events(&self) -> &[EffectEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn verify_root(&self) -> bool {
        let recomputed = compute_event_log_root(&self.events);
        recomputed == self.root
    }
}

fn compute_event_log_root(events: &[EffectEvent]) -> [u8; 32] {
    if events.is_empty() {
        return [0u8; 32];
    }
    let mut hasher = Blake2b512::new();
    for event in events {
        if let Ok(bytes) = serde_json::to_vec(event) {
            let leaf = hash_bytes_32(&bytes);
            hasher.update(leaf);
        }
    }
    let full = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&full[..32]);
    out
}

// ---------------------------------------------------------------------------
// StateEntry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateEntry {
    pub key: String,
    pub value_hash: [u8; 32],
}

impl StateEntry {
    pub fn new(key: impl Into<String>, value_bytes: &[u8]) -> Self {
        Self {
            key: key.into(),
            value_hash: hash_bytes_32(value_bytes),
        }
    }
}

// ---------------------------------------------------------------------------
// StateRoot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateRoot {
    entries: Vec<StateEntry>,
    root: [u8; 32],
}

impl StateRoot {
    /// Build from entries. Sorts by key before computing root.
    pub fn from_entries(mut entries: Vec<StateEntry>) -> Self {
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        let root = compute_state_root(&entries);
        Self { entries, root }
    }

    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            root: [0u8; 32],
        }
    }

    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }

    pub fn entries(&self) -> &[StateEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn verify_root(&self) -> bool {
        let recomputed = compute_state_root(&self.entries);
        recomputed == self.root
    }
}

fn compute_state_root(entries: &[StateEntry]) -> [u8; 32] {
    if entries.is_empty() {
        return [0u8; 32];
    }
    let mut hasher = Blake2b512::new();
    for entry in entries {
        if let Ok(bytes) = serde_json::to_vec(entry) {
            let leaf = hash_bytes_32(&bytes);
            hasher.update(leaf);
        }
    }
    let full = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&full[..32]);
    out
}

// ---------------------------------------------------------------------------
// EffectBoundary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectBoundary {
    pub pre_state_root: StateRoot,
    pub effect_log: EffectLog,
    pub post_state_root: Option<StateRoot>,
    boundary_hash: [u8; 32],
}

impl EffectBoundary {
    pub fn new(pre_state_root: StateRoot, effect_log: EffectLog) -> Self {
        let boundary_hash = compute_boundary_hash(&pre_state_root, &effect_log, None);
        Self {
            pre_state_root,
            effect_log,
            post_state_root: None,
            boundary_hash,
        }
    }

    pub fn with_post_state(mut self, post_state_root: StateRoot) -> Self {
        self.boundary_hash = compute_boundary_hash(
            &self.pre_state_root,
            &self.effect_log,
            Some(&post_state_root),
        );
        self.post_state_root = Some(post_state_root);
        self
    }

    pub fn boundary_hash(&self) -> &[u8; 32] {
        &self.boundary_hash
    }

    /// Checks pre_state_root, effect_log, and post_state_root (if Some) all
    /// pass verify_root; recomputes boundary_hash.
    pub fn verify_integrity(&self) -> bool {
        if !self.pre_state_root.verify_root() {
            return false;
        }
        if !self.effect_log.verify_root() {
            return false;
        }
        if let Some(post) = &self.post_state_root {
            if !post.verify_root() {
                return false;
            }
        }
        let recomputed = compute_boundary_hash(
            &self.pre_state_root,
            &self.effect_log,
            self.post_state_root.as_ref(),
        );
        recomputed == self.boundary_hash
    }
}

fn compute_boundary_hash(pre: &StateRoot, log: &EffectLog, post: Option<&StateRoot>) -> [u8; 32] {
    let post_root = post.map(|r| *r.root()).unwrap_or([0u8; 32]);
    let mut data = [0u8; 96];
    data[..32].copy_from_slice(pre.root());
    data[32..64].copy_from_slice(log.root());
    data[64..].copy_from_slice(&post_root);
    hash_bytes_32(&data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_event_hashes_value() {
        let e = ReadEvent::new("temperature", b"42.5");
        assert_ne!(e.value_hash, [0u8; 32]);
    }

    #[test]
    fn write_event_hashes_value() {
        let e = WriteEvent::new("counter", b"1");
        assert_ne!(e.value_hash, [0u8; 32]);
    }

    #[test]
    fn effect_event_key_and_predicates() {
        let r = EffectEvent::Read(ReadEvent::new("k", b"v"));
        assert_eq!(r.key(), "k");
        assert!(r.is_read());
        assert!(!r.is_write());

        let w = EffectEvent::Write(WriteEvent::new("k2", b"v2"));
        assert_eq!(w.key(), "k2");
        assert!(!w.is_read());
        assert!(w.is_write());
    }

    #[test]
    fn empty_effect_log() {
        let log = EffectLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(*log.root(), [0u8; 32]);
        assert!(log.verify_root());
    }

    #[test]
    fn effect_log_append_and_verify() {
        let mut log = EffectLog::new();
        log.append_read("key1", b"value1");
        log.append_write("key2", b"value2");
        assert_eq!(log.len(), 2);
        assert_ne!(*log.root(), [0u8; 32]);
        assert!(log.verify_root());
    }

    #[test]
    fn state_root_empty() {
        let sr = StateRoot::empty();
        assert!(sr.is_empty());
        assert_eq!(*sr.root(), [0u8; 32]);
        assert!(sr.verify_root());
    }

    #[test]
    fn state_root_from_entries_sorted() {
        let entries = vec![
            StateEntry::new("z_key", b"z_val"),
            StateEntry::new("a_key", b"a_val"),
        ];
        let sr = StateRoot::from_entries(entries);
        assert_eq!(sr.entries()[0].key, "a_key");
        assert_eq!(sr.entries()[1].key, "z_key");
        assert!(sr.verify_root());
    }

    #[test]
    fn effect_boundary_integrity() {
        let pre = StateRoot::empty();
        let mut log = EffectLog::new();
        log.append_read("db.user", b"alice");
        let boundary = EffectBoundary::new(pre, log);
        assert!(boundary.verify_integrity());

        let post = StateRoot::from_entries(vec![StateEntry::new("db.user", b"bob")]);
        let boundary = boundary.with_post_state(post);
        assert!(boundary.verify_integrity());
    }
}
