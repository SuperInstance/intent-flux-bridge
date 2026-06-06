# intent-flux-bridge

Experiment: maps natural-language GPU intent to Flux bytecode. Tests the pincher→flux-core synergy where 'LLM as compiler' generates portable bytecode from intent.

## Overview

# intent-flux-bridge

Maps natural-language GPU computation intent to Flux bytecode.

## Stats

- **Tests**: 8
- **LOC**: 401
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use intent_flux_bridge::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
