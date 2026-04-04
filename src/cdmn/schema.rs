use serde::{Deserialize, Serialize};

use crate::state::Word;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub memory: u8,
    #[serde(default)]
    pub presence_memory: Option<u8>,
    #[serde(default)]
    pub unknown_memory: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionField {
    pub name: String,
    pub offset: u8,
    #[serde(default)]
    pub presence_offset: Option<u8>,
    #[serde(default)]
    pub unknown_offset: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub count_memory: u8,
    pub base_memory: u8,
    pub stride: u8,
    pub max_items: u8,
    pub fields: Vec<CollectionField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Definition {
    pub name: String,
    pub expression: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    pub id: String,
    pub description: String,
    pub logic: String,
    #[serde(default)]
    pub reason_code: Option<Word>,
    #[serde(default)]
    pub reason_label: Option<String>,
    #[serde(default)]
    pub insufficient_data_reason_code: Option<Word>,
    #[serde(default)]
    pub insufficient_data_reason_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnOutputLayout {
    pub status_register: u8,
    pub reason_register: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasonMetadata {
    pub code: Word,
    pub constraint_id: String,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnProgramMetadata {
    pub output: CdmnOutputLayout,
    pub reasons: Vec<ReasonMetadata>,
}

impl CdmnProgramMetadata {
    pub fn reason_for_code(&self, code: Word) -> Option<&ReasonMetadata> {
        self.reasons.iter().find(|reason| reason.code == code)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCdmnProgram {
    pub assembly: String,
    pub output: CdmnOutputLayout,
    pub reasons: Vec<ReasonMetadata>,
}

impl CompiledCdmnProgram {
    pub fn reason_for_code(&self, code: Word) -> Option<&ReasonMetadata> {
        self.reasons.iter().find(|reason| reason.code == code)
    }

    pub fn metadata(&self) -> CdmnProgramMetadata {
        CdmnProgramMetadata {
            output: self.output.clone(),
            reasons: self.reasons.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdmnTable {
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub table_type: String,
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub definitions: Vec<Definition>,
    #[serde(default)]
    pub collections: Vec<Collection>,
    pub constraints: Vec<Constraint>,
}
