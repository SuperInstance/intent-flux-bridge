//! # intent-flux-bridge
//!
//! Maps natural-language GPU computation intent to Flux bytecode.
//! Tests the pincher→flux-core synergy: the LLM-compiler generates
//! portable bytecode from "what I want" rather than "how to do it".

use std::collections::HashMap;

/// Natural language intent expressing a GPU computation.
#[derive(Debug, Clone)]
pub struct Intent {
    pub description: String,
    pub intent_type: IntentType,
    pub gpu_requirements: GpuRequirements,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentType {
    /// Element-wise transformation (map each value).
    Map,
    /// Reduce across elements (sum, max, vote).
    Reduce,
    /// Matrix multiplication or linear algebra.
    MatMul,
    /// Scan/prefix operation (cumulative sum, sort).
    Scan,
    /// Multi-agent coordination (consensus, voting).
    Consensus,
    /// Attention mechanism.
    Attention,
}

#[derive(Debug, Clone)]
pub struct GpuRequirements {
    pub input_size: usize,
    pub output_size: usize,
    pub block_dim: u32,
    pub needs_shared_memory: bool,
    pub needs_sync: bool,
}

/// Flux bytecode instruction (produced from intent).
#[derive(Debug, Clone, PartialEq)]
pub enum FluxOp {
    MOVI { reg: u8, imm: i16 },
    ADD { rd: u8, rs1: u8, rs2: u8 },
    SUB { rd: u8, rs1: u8, rs2: u8 },
    MUL { rd: u8, rs1: u8, rs2: u8 },
    TADD { rd: u8, ra: u8, rb: u8 },
    TMUL { rd: u8, ra: u8, rb: u8 },
    ThreadIdx { dest: u8, dim: u8 },
    BlockDim { dest: u8, dim: u8 },
    SyncThreads,
    Load { rd: u8, addr: u8 },
    Store { addr: u8, rs: u8 },
    Branch { cond: u8, target: usize },
    Halt,
}

/// A compiled Flux program from intent.
#[derive(Debug, Clone)]
pub struct FluxProgram {
    pub name: String,
    pub original_intent: String,
    pub ops: Vec<FluxOp>,
    pub register_count: usize,
    pub uses_ternary: bool,
    pub estimated_cycles: u64,
}

/// The intent→bytecode compiler.
pub struct IntentCompiler {
    /// Named patterns that map intent keywords to Flux sequences.
    patterns: HashMap<String, Vec<FluxOp>>,
    /// Statistics.
    stats: CompilerStats,
}

#[derive(Debug, Clone, Default)]
pub struct CompilerStats {
    pub intents_compiled: u64,
    pub ops_generated: u64,
    pub ternary_programs: u64,
    pub patterns_matched: u64,
}

impl IntentCompiler {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Pattern: "abs of each element"
        patterns.insert("abs_map".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },
            FluxOp::MOVI { reg: 2, imm: 0 },
            FluxOp::SUB { rd: 3, rs1: 2, rs2: 1 },
            FluxOp::Branch { cond: 1, target: 6 },  // if val < 0
            FluxOp::Store { addr: 1, rs: 3 },
            FluxOp::Branch { cond: 0, target: 8 },
            FluxOp::Store { addr: 1, rs: 1 },
            FluxOp::Halt,
        ]);

        // Pattern: "sum all elements"
        patterns.insert("reduce_sum".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },
            FluxOp::SyncThreads,
            FluxOp::ADD { rd: 2, rs1: 2, rs2: 1 },
            FluxOp::SyncThreads,
            FluxOp::Halt,
        ]);

        // Pattern: "ternary vote" — {-1,0,+1} consensus
        patterns.insert("ternary_vote".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },
            FluxOp::TADD { rd: 2, ra: 1, rb: 0 },  // accumulate ternary
            FluxOp::SyncThreads,
            FluxOp::TMUL { rd: 3, ra: 2, rb: 1 },  // ternary product
            FluxOp::SyncThreads,
            FluxOp::Store { addr: 1, rs: 3 },
            FluxOp::Halt,
        ]);

        // Pattern: "ternary attention" — simplified
        patterns.insert("ternary_attention".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },  // query
            FluxOp::Load { rd: 2, addr: 1 },  // key
            FluxOp::TMUL { rd: 3, ra: 1, rb: 2 },  // Q·K ternary
            FluxOp::TADD { rd: 4, ra: 3, rb: 3 },  // accumulate
            FluxOp::SyncThreads,
            FluxOp::Store { addr: 2, rs: 4 },
            FluxOp::Halt,
        ]);

        // Pattern: "scale by constant"
        patterns.insert("scale".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },
            FluxOp::MOVI { reg: 2, imm: 3 },  // scale factor
            FluxOp::MUL { rd: 3, rs1: 1, rs2: 2 },
            FluxOp::Store { addr: 1, rs: 3 },
            FluxOp::Halt,
        ]);

        // Pattern: "ternary filter" — zero out elements that don't match
        patterns.insert("ternary_filter".into(), vec![
            FluxOp::ThreadIdx { dest: 0, dim: 0 },
            FluxOp::Load { rd: 1, addr: 0 },
            FluxOp::MOVI { reg: 2, imm: 1 },  // threshold
            FluxOp::TMUL { rd: 3, ra: 1, rb: 2 },  // ternary mask
            FluxOp::Store { addr: 1, rs: 3 },
            FluxOp::Halt,
        ]);

        Self { patterns, stats: CompilerStats::default() }
    }

    /// Compile a natural-language intent into Flux bytecode.
    pub fn compile(&mut self, intent: &Intent) -> Result<FluxProgram, CompileError> {
        let pattern_key = self.match_intent(&intent.description, &intent.intent_type)?;

        let ops = self.patterns.get(&pattern_key)
            .ok_or_else(|| CompileError::NoPattern(pattern_key.clone()))?
            .clone();

        let uses_ternary = ops.iter().any(|op|
            matches!(op, FluxOp::TADD { .. } | FluxOp::TMUL { .. })
        );

        let max_reg = ops.iter().filter_map(|op| match op {
            FluxOp::MOVI { reg, .. } => Some(*reg),
            FluxOp::ADD { rd, .. } | FluxOp::SUB { rd, .. } | FluxOp::MUL { rd, .. } => Some(*rd),
            FluxOp::TADD { rd, .. } | FluxOp::TMUL { rd, .. } => Some(*rd),
            FluxOp::ThreadIdx { dest, .. } => Some(*dest),
            FluxOp::Load { rd, .. } => Some(*rd),
            FluxOp::BlockDim { dest, .. } => Some(*dest),
            _ => None,
        }).max().map(|r| r as usize + 1).unwrap_or(1);

        let ops_len = ops.len();

        self.stats.intents_compiled += 1;
        self.stats.ops_generated += ops_len as u64;
        if uses_ternary { self.stats.ternary_programs += 1; }
        self.stats.patterns_matched += 1;

        Ok(FluxProgram {
            name: pattern_key,
            original_intent: intent.description.clone(),
            ops,
            register_count: max_reg,
            uses_ternary,
            estimated_cycles: ops_len as u64 * 4, // ~4 cycles per op
        })
    }

    /// Match intent description to a known pattern.
    fn match_intent(&self, description: &str, intent_type: &IntentType) -> Result<String, CompileError> {
        let desc = description.to_lowercase();

        // Direct keyword matching (in production, this would be LLM-powered)
        let match_result = if desc.contains("ternary") && desc.contains("attention") {
            Some("ternary_attention".into())
        } else if desc.contains("ternary") && (desc.contains("vote") || desc.contains("consensus")) {
            Some("ternary_vote".into())
        } else if desc.contains("ternary") && (desc.contains("filter") || desc.contains("mask")) {
            Some("ternary_filter".into())
        } else if desc.contains("abs") || (desc.contains("absolute") && desc.contains("value")) {
            Some("abs_map".into())
        } else if desc.contains("sum") || desc.contains("reduce") || desc.contains("total") {
            Some("reduce_sum".into())
        } else if desc.contains("scale") || desc.contains("multiply") && desc.contains("constant") {
            Some("scale".into())
        } else {
            // Fallback to intent type
            match intent_type {
                IntentType::Map => Some("abs_map".into()),
                IntentType::Reduce => Some("reduce_sum".into()),
                IntentType::MatMul => Some("scale".into()),
                IntentType::Consensus => Some("ternary_vote".into()),
                IntentType::Attention => Some("ternary_attention".into()),
                IntentType::Scan => Some("reduce_sum".into()),
            }
        };

        match_result.ok_or_else(|| CompileError::UnrecognizedIntent(description.into()))
    }

    /// Get all available patterns.
    pub fn available_patterns(&self) -> Vec<&str> {
        self.patterns.keys().map(|s| s.as_str()).collect()
    }

    /// Get compiler statistics.
    pub fn stats(&self) -> &CompilerStats { &self.stats }

    /// Compose two programs sequentially (pipeline).
    pub fn compose(&self, first: &FluxProgram, second: &FluxProgram) -> FluxProgram {
        let mut ops = first.ops.clone();
        // Remove the first program's HALT
        ops.retain(|op| !matches!(op, FluxOp::Halt));
        ops.extend(second.ops.clone());

        FluxProgram {
            name: format!("{}_then_{}", first.name, second.name),
            original_intent: format!("{} then {}", first.original_intent, second.original_intent),
            ops,
            register_count: first.register_count.max(second.register_count),
            uses_ternary: first.uses_ternary || second.uses_ternary,
            estimated_cycles: first.estimated_cycles + second.estimated_cycles,
        }
    }
}

impl Default for IntentCompiler {
    fn default() -> Self { Self::new() }
}

/// Compile errors.
#[derive(Debug, Clone)]
pub enum CompileError {
    UnrecognizedIntent(String),
    NoPattern(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnrecognizedIntent(desc) => write!(f, "unrecognized intent: {}", desc),
            Self::NoPattern(name) => write!(f, "no pattern: {}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_intent() {
        let mut compiler = IntentCompiler::new();
        let intent = Intent {
            description: "compute the absolute value of each element".into(),
            intent_type: IntentType::Map,
            gpu_requirements: GpuRequirements {
                input_size: 1024, output_size: 1024, block_dim: 256,
                needs_shared_memory: false, needs_sync: false,
            },
        };
        let program = compiler.compile(&intent).unwrap();
        assert_eq!(program.name, "abs_map");
        assert!(!program.ops.is_empty());
        assert!(!program.uses_ternary);
    }

    #[test]
    fn test_ternary_vote_intent() {
        let mut compiler = IntentCompiler::new();
        let intent = Intent {
            description: "perform ternary voting consensus across agents".into(),
            intent_type: IntentType::Consensus,
            gpu_requirements: GpuRequirements {
                input_size: 256, output_size: 1, block_dim: 256,
                needs_shared_memory: true, needs_sync: true,
            },
        };
        let program = compiler.compile(&intent).unwrap();
        assert_eq!(program.name, "ternary_vote");
        assert!(program.uses_ternary);
    }

    #[test]
    fn test_ternary_attention_intent() {
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
        assert_eq!(program.name, "ternary_attention");
        assert!(program.uses_ternary);
    }

    #[test]
    fn test_reduce_sum_intent() {
        let mut compiler = IntentCompiler::new();
        let intent = Intent {
            description: "sum all elements in the array".into(),
            intent_type: IntentType::Reduce,
            gpu_requirements: GpuRequirements {
                input_size: 4096, output_size: 1, block_dim: 256,
                needs_shared_memory: true, needs_sync: true,
            },
        };
        let program = compiler.compile(&intent).unwrap();
        assert_eq!(program.name, "reduce_sum");
    }

    #[test]
    fn test_compose_programs() {
        let mut compiler = IntentCompiler::new();
        let p1 = compiler.compile(&Intent {
            description: "ternary filter the data".into(),
            intent_type: IntentType::Map,
            gpu_requirements: GpuRequirements { input_size: 256, output_size: 256, block_dim: 256, needs_shared_memory: false, needs_sync: false },
        }).unwrap();
        let p2 = compiler.compile(&Intent {
            description: "sum the filtered results".into(),
            intent_type: IntentType::Reduce,
            gpu_requirements: GpuRequirements { input_size: 256, output_size: 1, block_dim: 256, needs_shared_memory: true, needs_sync: true },
        }).unwrap();

        let composed = compiler.compose(&p1, &p2);
        assert!(composed.name.contains("then"));
        assert!(composed.uses_ternary);
        assert!(composed.ops.len() > p1.ops.len());
    }

    #[test]
    fn test_available_patterns() {
        let compiler = IntentCompiler::new();
        let patterns = compiler.available_patterns();
        assert!(patterns.len() >= 5);
        assert!(patterns.contains(&"abs_map"));
        assert!(patterns.contains(&"ternary_vote"));
        assert!(patterns.contains(&"ternary_attention"));
    }

    #[test]
    fn test_stats_tracking() {
        let mut compiler = IntentCompiler::new();
        for _ in 0..5 {
            compiler.compile(&Intent {
                description: "ternary vote on data".into(),
                intent_type: IntentType::Consensus,
                gpu_requirements: GpuRequirements { input_size: 100, output_size: 1, block_dim: 32, needs_shared_memory: false, needs_sync: false },
            }).unwrap();
        }
        assert_eq!(compiler.stats().intents_compiled, 5);
        assert_eq!(compiler.stats().ternary_programs, 5);
    }

    #[test]
    fn test_fallback_by_type() {
        let mut compiler = IntentCompiler::new();
        // Unknown description but known type
        let program = compiler.compile(&Intent {
            description: "do something weird".into(),
            intent_type: IntentType::Consensus,
            gpu_requirements: GpuRequirements { input_size: 100, output_size: 1, block_dim: 32, needs_shared_memory: false, needs_sync: false },
        }).unwrap();
        assert_eq!(program.name, "ternary_vote"); // fallback
    }
}
