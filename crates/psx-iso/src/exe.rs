//! PSX-EXE parsing.
//!
//! PSX-EXE is the raw executable format the BIOS knows how to load.
//! Every commercial disc's main executable is a PSX-EXE; homebrew
//! uses the same format. The file layout:
//!
//! ```text
//!   00h-07h  "PS-X EXE" magic
//!   08h-0Fh  zero-filled
//!   10h      initial PC
//!   14h      initial GP (`$r28`)
//!   18h      load address in RAM (t_addr)
//!   1Ch      payload size (t_size, excluding the 2 KiB header)
//!   20h-27h  data section (address, size) — usually zero-sized
//!   28h-2Fh  BSS section (address, size) — zeroed by the BIOS
//!   30h      initial SP base (`$r29` high bits)
//!   34h      initial SP offset added to the base
//!   38h-4Bh  reserved
//!   4Ch-7FFh ASCII region marker ("Sony Computer Entertainment Inc.")
//!   800h-..  payload bytes; loaded verbatim to `t_addr`
//! ```
//!
//! The header is always exactly 2048 bytes; everything after is copied
//! byte-for-byte into RAM at `t_addr`. Side-loading bypasses the BIOS
//! entirely: set the CPU's PC/GP/SP to the header values after copying
//! the payload and execution starts in the homebrew.

use alloc::vec::Vec;

/// PSX-EXE header size in bytes.
pub const EXE_HEADER_BYTES: usize = 2048;

const MAGIC: &[u8; 8] = b"PS-X EXE";

/// Errors surfaced while parsing a PSX-EXE.
#[derive(Debug, PartialEq, Eq)]
pub enum ExeError {
    /// File shorter than a 2 KiB header.
    TooShort,
    /// Magic string didn't match "PS-X EXE".
    BadMagic,
    /// Header says the payload is N bytes but the file doesn't carry that many.
    TruncatedPayload {
        /// Expected payload size per header.
        expected: usize,
        /// Bytes actually present past the header.
        actual: usize,
    },
}

/// Parsed PSX-EXE. Owns the payload bytes so a caller can drop the
/// original file.
#[derive(Debug)]
pub struct Exe {
    /// Value for `$pc` at entry.
    pub initial_pc: u32,
    /// Value for `$gp` at entry.
    pub initial_gp: u32,
    /// Destination address in RAM where [`Exe::payload`] goes.
    pub load_addr: u32,
    /// `$r29` base. The BIOS initialises the stack pointer as
    /// `initial_sp_base + initial_sp_offset` — we do the same. If both
    /// are zero the caller should leave the CPU's SP alone so the
    /// default reset state survives.
    pub initial_sp_base: u32,
    /// Additive offset on top of [`initial_sp_base`].
    pub initial_sp_offset: u32,
    /// BSS section start address (zero if no BSS).
    pub bss_addr: u32,
    /// BSS section byte count (zero if no BSS).
    pub bss_size: u32,
    /// Code + data payload, verbatim. Length matches the header's
    /// `t_size` field.
    pub payload: Vec<u8>,
}

impl Exe {
    /// Parse a raw PSX-EXE byte stream. The slice must contain the
    /// full header (2 KiB) plus at least `t_size` payload bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, ExeError> {
        if bytes.len() < EXE_HEADER_BYTES {
            return Err(ExeError::TooShort);
        }
        if &bytes[0..8] != MAGIC {
            return Err(ExeError::BadMagic);
        }

        let initial_pc = read_u32_le(&bytes[0x10..]);
        let initial_gp = read_u32_le(&bytes[0x14..]);
        let load_addr = read_u32_le(&bytes[0x18..]);
        let t_size = read_u32_le(&bytes[0x1C..]);
        let bss_addr = read_u32_le(&bytes[0x28..]);
        let bss_size = read_u32_le(&bytes[0x2C..]);
        let initial_sp_base = read_u32_le(&bytes[0x30..]);
        let initial_sp_offset = read_u32_le(&bytes[0x34..]);

        let payload_available = bytes.len() - EXE_HEADER_BYTES;
        if (t_size as usize) > payload_available {
            return Err(ExeError::TruncatedPayload {
                expected: t_size as usize,
                actual: payload_available,
            });
        }

        let payload = bytes[EXE_HEADER_BYTES..EXE_HEADER_BYTES + t_size as usize].to_vec();

        Ok(Self {
            initial_pc,
            initial_gp,
            load_addr,
            initial_sp_base,
            initial_sp_offset,
            bss_addr,
            bss_size,
            payload,
        })
    }

    /// Effective stack pointer — `base + offset`, or `None` when both
    /// fields are zero (caller should keep the CPU's reset SP).
    pub fn initial_sp(&self) -> Option<u32> {
        if self.initial_sp_base == 0 && self.initial_sp_offset == 0 {
            None
        } else {
            Some(self.initial_sp_base.wrapping_add(self.initial_sp_offset))
        }
    }
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_exe(pc: u32, t_addr: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; EXE_HEADER_BYTES];
        v[..8].copy_from_slice(MAGIC);
        v[0x10..0x14].copy_from_slice(&pc.to_le_bytes());
        v[0x18..0x1C].copy_from_slice(&t_addr.to_le_bytes());
        v[0x1C..0x20].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn parses_minimal_exe() {
        let raw = make_exe(0x8001_0000, 0x8001_0000, &[0xDE, 0xAD, 0xBE, 0xEF]);
        let exe = Exe::parse(&raw).unwrap();
        assert_eq!(exe.initial_pc, 0x8001_0000);
        assert_eq!(exe.load_addr, 0x8001_0000);
        assert_eq!(exe.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn rejects_short_file() {
        assert!(matches!(Exe::parse(&[0u8; 10]), Err(ExeError::TooShort)));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = vec![0u8; EXE_HEADER_BYTES];
        raw[..8].copy_from_slice(b"NOT-EXE!");
        assert!(matches!(Exe::parse(&raw), Err(ExeError::BadMagic)));
    }

    #[test]
    fn rejects_truncated_payload() {
        let mut raw = vec![0u8; EXE_HEADER_BYTES];
        raw[..8].copy_from_slice(MAGIC);
        raw[0x1C..0x20].copy_from_slice(&100u32.to_le_bytes());
        assert!(matches!(
            Exe::parse(&raw),
            Err(ExeError::TruncatedPayload {
                expected: 100,
                actual: 0
            })
        ));
    }

    #[test]
    fn initial_sp_returns_none_when_zero() {
        let raw = make_exe(0, 0, &[]);
        let exe = Exe::parse(&raw).unwrap();
        assert_eq!(exe.initial_sp(), None);
    }
}
