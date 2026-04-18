//! BIN/CUE and (eventually) ISO9660 parsing for PS1 disc images.
//!
//! Phase 6d scope: raw-BIN loading. A BIN file is just a sequence of
//! 2352-byte Mode-2 Form-1 sectors, each laid out as:
//!
//! ```text
//!   0..12    sync pattern (0x00, 0xFF × 10, 0x00)
//!  12..16    header: MM SS FF MODE
//!  16..24    sub-header: FILE CHN SUB CI FILE CHN SUB CI
//!  24..2072  user data (2048 bytes)
//! 2072..2076 EDC (optional)
//! 2076..2352 ECC / reserved
//! ```
//!
//! PS1 discs place their data area starting at LBA `0x0000`, which
//! corresponds to MSF `00:02:00` (after the 2-second pre-gap).
//! `Disc::read_sector_raw` here treats LBA 0 as byte offset 0 of the
//! BIN — adequate for hand-assembled homebrew images. CUE-aware
//! handling (pre-gap skipping, multi-track) lands when a real
//! commercial disc boots.
//!
//! ISO9660 filesystem parsing (primary volume descriptor, path
//! tables, directory records) lands in Phase 6e.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

pub mod exe;
pub mod iso9660;
pub use exe::{Exe, ExeError, EXE_HEADER_BYTES};
pub use iso9660::{default_system_cnf, IsoBuilder, IsoFile};

/// One raw CD-ROM sector — always 2352 bytes on a PS1 disc regardless
/// of track mode.
pub const SECTOR_BYTES: usize = 2352;
/// Byte offset of the 2048-byte user-data region within a Mode-2
/// Form-1 sector.
pub const SECTOR_USER_DATA_OFFSET: usize = 24;
/// User-data size per sector.
pub const SECTOR_USER_DATA_BYTES: usize = 2048;

/// A loaded disc image. Holds the raw BIN bytes and answers
/// sector-read requests.
pub struct Disc {
    bytes: Vec<u8>,
}

impl Disc {
    /// Construct a disc from a raw BIN image.
    pub fn from_bin(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Total sector count = floor(size / 2352). Incomplete tail
    /// sectors are ignored.
    pub fn sector_count(&self) -> usize {
        self.bytes.len() / SECTOR_BYTES
    }

    /// Read a raw 2352-byte sector. Returns `None` past end-of-disc.
    pub fn read_sector_raw(&self, lba: u32) -> Option<&[u8]> {
        let start = (lba as usize).checked_mul(SECTOR_BYTES)?;
        let end = start.checked_add(SECTOR_BYTES)?;
        if end > self.bytes.len() {
            return None;
        }
        Some(&self.bytes[start..end])
    }

    /// Read the 2048-byte user-data payload of a sector (mode 2 form 1).
    pub fn read_sector_user(&self, lba: u32) -> Option<&[u8]> {
        let sector = self.read_sector_raw(lba)?;
        Some(&sector[SECTOR_USER_DATA_OFFSET..SECTOR_USER_DATA_OFFSET + SECTOR_USER_DATA_BYTES])
    }
}

/// Convert a BCD byte pair to binary (used for MSF fields in commands).
/// Returns 0xFF if either nibble is out of range.
pub fn bcd_to_bin(bcd: u8) -> u8 {
    let hi = (bcd >> 4) & 0xF;
    let lo = bcd & 0xF;
    if hi > 9 || lo > 9 {
        0xFF
    } else {
        hi * 10 + lo
    }
}

/// Pack a binary 0..=99 value into a BCD byte. Values above 99
/// clamp to 99 — hardware drops the high bits rather than
/// corrupting the low nibble.
pub fn bin_to_bcd(v: u8) -> u8 {
    let v = v.min(99);
    ((v / 10) << 4) | (v % 10)
}

/// Convert an absolute LBA to an MSF triple `(minute, second, frame)`
/// in *binary* form. Caller must `bin_to_bcd` each field before
/// sending over the wire. LBA 0 = MSF 00:02:00 per the 150-frame
/// pre-gap convention.
pub fn lba_to_msf(lba: u32) -> (u8, u8, u8) {
    let abs = lba.saturating_add(150);
    let m = (abs / (60 * 75)) as u8;
    let s = ((abs / 75) % 60) as u8;
    let f = (abs % 75) as u8;
    (m, s, f)
}

/// Convert a 3-byte BCD MSF triple (minute, second, frame) into the
/// absolute LBA the sector lives at.
///
/// PS1 discs start their data at MSF 00:02:00 (after 2-second
/// pre-gap), which we treat as LBA 0 in the BIN layout. So:
///
/// ```text
///     LBA = (minute × 60 + second - 2) × 75 + frame
/// ```
pub fn msf_to_lba(m_bcd: u8, s_bcd: u8, f_bcd: u8) -> u32 {
    let m = bcd_to_bin(m_bcd) as i32;
    let s = bcd_to_bin(s_bcd) as i32;
    let f = bcd_to_bin(f_bcd) as i32;
    let abs = (m * 60 + s) * 75 + f;
    (abs - 150).max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn bcd_decodes_common_values() {
        assert_eq!(bcd_to_bin(0x00), 0);
        assert_eq!(bcd_to_bin(0x42), 42);
        assert_eq!(bcd_to_bin(0x99), 99);
        assert_eq!(bcd_to_bin(0xAB), 0xFF);
    }

    #[test]
    fn msf_to_lba_data_area_starts_at_zero() {
        assert_eq!(msf_to_lba(0x00, 0x02, 0x00), 0);
    }

    #[test]
    fn bin_to_bcd_packs_correctly() {
        assert_eq!(bin_to_bcd(0), 0x00);
        assert_eq!(bin_to_bcd(42), 0x42);
        assert_eq!(bin_to_bcd(99), 0x99);
        assert_eq!(bin_to_bcd(100), 0x99); // clamped
    }

    #[test]
    fn lba_to_msf_round_trips() {
        assert_eq!(lba_to_msf(0), (0, 2, 0));
        assert_eq!(lba_to_msf(75), (0, 3, 0));
        assert_eq!(lba_to_msf(4500), (1, 2, 0));
    }

    #[test]
    fn disc_sector_count_rounds_down() {
        let bytes = vec![0u8; SECTOR_BYTES * 3 + 100];
        let d = Disc::from_bin(bytes);
        assert_eq!(d.sector_count(), 3);
    }

    #[test]
    fn read_sector_returns_slice() {
        let mut bytes = vec![0u8; SECTOR_BYTES];
        bytes[SECTOR_USER_DATA_OFFSET] = 0xAB;
        let d = Disc::from_bin(bytes);
        let s = d.read_sector_raw(0).unwrap();
        assert_eq!(s[SECTOR_USER_DATA_OFFSET], 0xAB);
        let u = d.read_sector_user(0).unwrap();
        assert_eq!(u[0], 0xAB);
    }

    #[test]
    fn read_past_end_returns_none() {
        let d = Disc::from_bin(vec![0u8; SECTOR_BYTES]);
        assert!(d.read_sector_raw(1).is_none());
    }
}
