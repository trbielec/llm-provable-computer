# tvm-verify-wasm

Browser and Node.js bindings for verifying `llm-provable-computer` STARK proofs.

## Prerequisites

```bash
rustup target add wasm32-unknown-unknown
cargo check -p tvm-verify-core --target wasm32-unknown-unknown
cargo build -p tvm-verify-wasm --target wasm32-unknown-unknown
```

## JavaScript usage

```js
import init, { verify_stark_proof } from "@llm-provable-computer/verifier";

await init();
const result = verify_stark_proof(proofJsonString);
console.log(result.accepted, result.error);
```
