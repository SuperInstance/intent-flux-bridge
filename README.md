# intent-flux-bridge

Experiment: maps natural-language GPU intent to Flux bytecode. Tests the pincher→flux-core integration where 'LLM as compiler' generates portable bytecode from intent.

## Why This Matters

# intent-flux-bridge
Maps natural-language GPU computation intent to Flux bytecode.
Tests the pincher→flux-core integration: the LLM-compiler generates
portable bytecode from "what I want" rather than "how to do it".

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct Intent
pub enum IntentType
pub struct GpuRequirements
pub enum FluxOp
pub struct FluxProgram
pub struct IntentCompiler
pub struct CompilerStats
pub fn new
pub fn compile
pub fn available_patterns
pub fn stats
pub fn compose
```

## Usage

```toml
[dependencies]
intent-flux-bridge = "0.1.0"
```

```rust
use intent_flux_bridge::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/intent-flux-bridge.git
cd intent-flux-bridge
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 402 |
| Public API | 13 items |

## License

Apache-2.0
