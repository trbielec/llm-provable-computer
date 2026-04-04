use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn verify_stark_proof(proof_json: &str) -> Result<JsValue, JsValue> {
    let result =
        match tvm_verify_core::deserialize_execution_stark_proof_json(proof_json.as_bytes()) {
            Ok(proof) => match tvm_verify_core::verify_execution_stark(&proof) {
                Ok(accepted) => build_result(
                    accepted,
                    Some(proof.claim.steps),
                    Some(proof.claim.program.instructions().len()),
                    Some(proof.claim.attention_mode.to_string()),
                    None,
                ),
                Err(error) => build_result(
                    false,
                    Some(proof.claim.steps),
                    Some(proof.claim.program.instructions().len()),
                    Some(proof.claim.attention_mode.to_string()),
                    Some(error.to_string()),
                ),
            },
            Err(error) => build_result(false, None, None, None, Some(error.to_string())),
        }?;

    Ok(result.into())
}

fn build_result(
    accepted: bool,
    steps: Option<usize>,
    instruction_count: Option<usize>,
    attention_mode: Option<String>,
    error: Option<String>,
) -> Result<Object, JsValue> {
    let object = Object::new();
    set(&object, "accepted", JsValue::from_bool(accepted))?;
    set_optional_usize(&object, "steps", steps)?;
    set_optional_usize(&object, "instructionCount", instruction_count)?;
    set_optional_string(&object, "attentionMode", attention_mode)?;
    set_optional_string(&object, "error", error)?;
    Ok(object)
}

fn set(object: &Object, key: &str, value: JsValue) -> Result<(), JsValue> {
    Reflect::set(object, &JsValue::from_str(key), &value).map(|_| ())
}

fn set_optional_usize(object: &Object, key: &str, value: Option<usize>) -> Result<(), JsValue> {
    match value {
        Some(value) => set(object, key, JsValue::from_f64(value as f64)),
        None => set(object, key, JsValue::NULL),
    }
}

fn set_optional_string(object: &Object, key: &str, value: Option<String>) -> Result<(), JsValue> {
    match value {
        Some(value) => set(object, key, JsValue::from_str(&value)),
        None => set(object, key, JsValue::NULL),
    }
}
