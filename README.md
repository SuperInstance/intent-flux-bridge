# intent-flux-bridge

An experimental Rust crate that maps **natural-language GPU computation intent** to **Flux bytecode** — a portable IR for heterogeneous compute. It demonstrates the "LLM as compiler" paradigm: rather than writing CUDA/Metal/SYCL kernels by hand, you describe *what* you want ("sum all elements", "ternary attention Q·K") and the compiler generates Flux instructions.

## Why It Matters

GPU programming today requires:
- **Platform-specific kernels** — CUDA for NVIDIA, Metal for Apple, ROCm for AMD, SYCL for Intel
- **Manual optimization** — shared memory, warp-level primitives, occupancy tuning
- **Steep learning curve** — each platform has its own dialect and toolchain

The intent-flux-bridge explores an alternative: **declarative intent → portable bytecode**. The LLM acts as a front-end compiler, translating natural language into Flux ops (`TADD`, `TMUL`, `SyncThreads`), which then lower to platform-specific code.

Key innovation: **ternary arithmetic** (balanced ternary {-1, 0, +1}) is a first-class concern via `TADD`/`TMUL` ops, enabling efficient multi-agent consensus and neural network inference on ternary weights.

## How It Works

### Intent Classification

Natural-language descriptions are matched to computation patterns:

```
"sum all elements"           → reduce_sum pattern
"absolute value of each"     → abs_map pattern
"ternary vote consensus"     → ternary_vote pattern (uses TADD/TMUL)
"ternary attention Q times K" → ternary_attention pattern
"scale by constant"          → scale pattern
"ternary filter mask"        → ternary_filter pattern
```

Matching is keyword-based in this prototype; production would use LLM-powered semantic matching.

### Flux Bytecode

Each compiled program is a sequence of `FluxOp` instructions:

| Op | Semantics | Cycles |
|----|-----------|--------|
| `MOVI {reg, imm}` | Load immediate to register | 1 |
| `ADD/SUB/MUL {rd, rs1, rs2}` | Binary arithmetic | 1 |
| `TADD {rd, ra, rb}` | Ternary addition (mod 3 arithmetic) | 2 |
| `TMUL {rd, ra, rb}` | Ternary multiplication | 2 |
| `ThreadIdx {dest, dim}` | Get thread index (0=x, 1=y, 2=z) | 1 |
| `BlockDim {dest, dim}` | Get block dimension | 1 |
| `SyncThreads` | Block-level barrier | ~20 |
| `Load/Store` | Global memory access | ~400 (DRAM) |
| `Branch {cond, target}` | Conditional branch | 2 |
| `Halt` | Terminate kernel | 0 |

### Example: Ternary Vote Compilation

Intent: *"perform ternary voting consensus across agents"*

```
ThreadIdx dest=0, dim=0     // thread ID
Load       rd=1, addr=0     // load vote value
TADD       rd=2, ra=1, rb=0 // ternary accumulate
SyncThreads                // barrier
TMUL       rd=3, ra=2, rb=1 // ternary product
SyncThreads
Store      addr=1, rs=3
Halt
```

Estimated cycles: ~36 (9 ops × 4 avg cycles)

### Program Composition

The `compose()` method pipelines two programs: the first program's `Halt` is removed and the second's ops are appended. This enables dataflow composition ("filter then reduce").

### Complexity Analysis

| Operation | Complexity |
|-----------|-----------|
| Intent matching | O(k·d) where k = keywords, d = description length |
| Pattern lookup | O(1) — HashMap by string key |
| Compilation | O(p) where p = pattern size |
| Composition | O(p₁ + p₂) |
| Register counting | O(p) — single pass over ops |

## Quick Start

```toml
[dependencies]
intent-flux-bridge = "0.1.0"
```

```rust
use intent_flux_bridge::{IntentCompiler, Intent, IntentType, GpuRequirements};

fn main() {
    let mut compiler = IntentCompiler::new();
    
    let intent = Intent {
        description: "compute ternary attention Q times K".into(),
        intent_type: IntentType::Attention,
        gpu_requirements: GpuRequirements {
            input_size: 512, output_size: 512, block_dim: 256,
            needs_shared_memory: true, needs_sync: true,
        },
    };
    
    let program = compiler.compile(&intent).unwrap();
    println!("Program: {} ({} ops, {} cycles, ternary={})",
        program.name, program.ops.len(), program.estimated_cycles, program.uses_ternary);
}
```

## API

| Type | Description |
|------|-------------|
| `IntentCompiler` | The intent→bytecode compiler |
| `Intent` | Natural-language computation description |
| `IntentType` | Map / Reduce / MatMul / Scan / Consensus / Attention |
| `GpuRequirements` | Resource specification (sizes, block dim, sync needs) |
| `FluxProgram` | Compiled bytecode with metadata |
| `FluxOp` | Individual instruction (14 opcodes) |
| `CompilerStats` | Compilation telemetry |

## Architecture Notes

This crate is the **compilation bridge (η)**: it translates human intent into executable Flux bytecode (γ, the IR). The ternary ops (`TADD`, `TMUL`) connect to balanced-ternary arithmetic research, where numbers in {-1, 0, +1} enable ultra-efficient neural network inference — ternary weights need only 2 bits, and `TMUL` maps to sign-multiply logic. The γ + η = C model: Flux bytecode definitions (γ) + this compiler (η) = a complete intent-to-execution pipeline.

## References

- Zhu et al., *Trained Ternary Quantization* (2016) — Ternary neural networks
- Knuth, *The Art of Computer Programming, Vol. 2* — Balanced ternary arithmetic (§4.1)
- Lattner & Adve, *LLVM: A Compilation Framework* (2004) — Multi-stage IR design
- [MLIR dialect design](https://mlir.llvm.org/) — Modern IR composition patterns

## License

MIT
