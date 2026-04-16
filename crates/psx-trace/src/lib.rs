//! Trace record format for parity testing.
//!
//! Both the emulator (Rust) and the PCSX-Redux oracle (via Lua exports)
//! emit records in this format. The parity runner parses JSONL streams
//! from both sides and compares them record-for-record.
//!
//! **Why JSONL instead of a packed binary format?** Readability during
//! development outweighs the size overhead. Every field can be inspected
//! with `jq`, diffs are line-oriented, and Redux's Lua has a JSON
//! encoder built in. When trace volume becomes prohibitive we can add
//! a compact binary encoding alongside this one — the struct shape
//! stays canonical.

use serde::{Deserialize, Serialize};

/// Format version. Bump whenever [`InstructionRecord`] gains or removes
/// a field. Readers should reject records with an unexpected version.
pub const FORMAT_VERSION: u32 = 1;

/// State snapshot after one instruction has executed.
///
/// Fields marked "future" are omitted for now; JSON's schema flexibility
/// lets us add them later without breaking old readers, so long as they
/// default to a sensible value on the emitter side.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionRecord {
    /// Monotonically increasing cycle count since reset.
    pub tick: u64,
    /// PC at which `instr` was fetched (not the updated PC afterward).
    pub pc: u32,
    /// Raw 32-bit instruction word that was executed.
    pub instr: u32,
    /// All 32 general-purpose registers after execution. `gprs[0]` must
    /// always be zero; emitters must not rely on the bus to maintain it.
    pub gprs: [u32; 32],
}

impl InstructionRecord {
    /// Serialize to a single JSONL line (no trailing newline).
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("InstructionRecord always serializes")
    }

    /// Parse from a single JSONL line.
    pub fn from_json_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_round_trips() {
        let rec = InstructionRecord {
            tick: 42,
            pc: 0xBFC0_0000,
            instr: 0x3C08_0013,
            gprs: [0; 32],
        };
        let line = rec.to_json_line();
        let parsed = InstructionRecord::from_json_line(&line).unwrap();
        assert_eq!(rec, parsed);
    }

    #[test]
    fn gpr0_is_canonical_zero() {
        // Emitters are responsible; we just assert the type preserves it.
        let rec = InstructionRecord {
            tick: 0,
            pc: 0,
            instr: 0,
            gprs: [0; 32],
        };
        assert_eq!(rec.gprs[0], 0);
    }
}
