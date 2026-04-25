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

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// Format version. Bump whenever [`InstructionRecord`] gains or removes
/// a field. Readers should reject records with an unexpected version.
pub const FORMAT_VERSION: u32 = 2;

/// State snapshot after one instruction has executed.
///
/// `cop2_data` and `cop2_ctl` mirror the 32 GTE data + 32 GTE control
/// registers as exposed by `MFC2` / `CFC2`. They are captured every
/// step (not just on COP2 ops) so a divergence in GTE state surfaces
/// at the instruction that produced it, not millions of cycles later
/// when the bad value finally leaks into a GPR. The snapshot uses
/// "software-visible" semantics (e.g. `H` reads back sign-extended,
/// IRGB packs from IR1..3), not raw internal struct fields.
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
    /// All 32 GTE data registers (`MFC2` view) after execution.
    #[serde(default = "zero_regs", deserialize_with = "deserialize_reg_array")]
    pub cop2_data: [u32; 32],
    /// All 32 GTE control registers (`CFC2` view) after execution.
    #[serde(default = "zero_regs", deserialize_with = "deserialize_reg_array")]
    pub cop2_ctl: [u32; 32],
}

/// Default helper for `[u32; 32]` so older JSONL records (pre-v2)
/// deserialize without the COP2 arrays — they show up as all-zero,
/// which the parity comparator can treat as "no COP2 information
/// available for this record". Lets us inspect a stray v1 JSONL line
/// without crashing, even though normal cache invalidation throws v1
/// caches away by file-format version.
fn zero_regs() -> [u32; 32] {
    [0; 32]
}

fn deserialize_reg_array<'de, D>(deserializer: D) -> Result<[u32; 32], D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<serde_json::Value>::deserialize(deserializer)?;
    if values.len() != 32 {
        return Err(D::Error::custom(format!(
            "expected 32 registers, got {}",
            values.len()
        )));
    }
    let mut out = [0u32; 32];
    for (idx, value) in values.into_iter().enumerate() {
        let Some(n) = value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)) else {
            return Err(D::Error::custom(format!(
                "register {idx} is not an integer"
            )));
        };
        out[idx] = if n < 0 {
            (n as i32) as u32
        } else {
            u32::try_from(n)
                .map_err(|_| D::Error::custom(format!("register {idx} out of u32 range: {n}")))?
        };
    }
    Ok(out)
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

    fn sample(tick: u64) -> InstructionRecord {
        let mut cop2_data = [0u32; 32];
        cop2_data[6] = 0x8080_8080; // RGBC
        cop2_data[8] = 0x0000_1234; // IR0
        let mut cop2_ctl = [0u32; 32];
        cop2_ctl[31] = 0x8000_F000; // FLAG
        InstructionRecord {
            tick,
            pc: 0xBFC0_0000,
            instr: 0x3C08_0013,
            gprs: [0; 32],
            cop2_data,
            cop2_ctl,
        }
    }

    #[test]
    fn jsonl_round_trips() {
        let rec = sample(42);
        let line = rec.to_json_line();
        let parsed = InstructionRecord::from_json_line(&line).unwrap();
        assert_eq!(rec, parsed);
    }

    #[test]
    fn v1_record_without_cop2_fields_parses_with_zero_defaults() {
        // Pre-v2 records didn't carry cop2_data/cop2_ctl. The serde
        // defaults must let the new schema parse them so a v1 cache
        // can be inspected after upgrade — even though normal cache
        // invalidation throws v1 caches away by file-format version.
        let v1 = r#"{"tick":1,"pc":0,"instr":0,"gprs":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let parsed = InstructionRecord::from_json_line(v1).unwrap();
        assert_eq!(parsed.cop2_data, [0u32; 32]);
        assert_eq!(parsed.cop2_ctl, [0u32; 32]);
    }

    #[test]
    fn redux_signed_cop2_json_parses_as_twos_complement() {
        let mut line =
            r#"{"tick":1,"pc":0,"instr":0,"gprs":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"cop2_data":["#
                .to_string();
        line.push_str(
            &std::iter::once("-431".to_string())
                .chain(std::iter::repeat("0".to_string()).take(31))
                .collect::<Vec<_>>()
                .join(","),
        );
        line.push_str(
            r#"],"cop2_ctl":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#,
        );

        let parsed = InstructionRecord::from_json_line(&line).unwrap();
        assert_eq!(parsed.cop2_data[0], (-431i32) as u32);
    }

    #[test]
    fn gpr0_is_canonical_zero() {
        // Emitters are responsible; we just assert the type preserves it.
        let rec = InstructionRecord {
            tick: 0,
            pc: 0,
            instr: 0,
            gprs: [0; 32],
            cop2_data: [0; 32],
            cop2_ctl: [0; 32],
        };
        assert_eq!(rec.gprs[0], 0);
    }
}
