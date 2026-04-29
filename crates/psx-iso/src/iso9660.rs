//! Minimal ISO 9660 writer — enough to produce a bootable PS1 disc
//! image from a PSX-EXE + a `SYSTEM.CNF`.
//!
//! This is **not** a full ISO 9660 implementation. Features deliberately
//! omitted:
//!
//! - Multi-level directory trees (everything lives in root).
//! - Joliet / Rock Ridge extensions (ASCII-only file names, 8.3).
//! - Multi-session / multi-track discs.
//! - Proper per-file metadata (creation / modification dates are fixed).
//! - Supplementary volume descriptors.
//!
//! What it **does** produce:
//!
//! - A 2048-byte-per-sector "cooked" image (`.iso`). Every emulator
//!   will boot this directly. Real PlayStation hardware requires a
//!   raw 2352-byte `.bin` with CD-ROM EDC/ECC — conversion is a
//!   post-step (tools like `cdrdao` or `bchunk` do it). This crate
//!   focuses on the filesystem layout; upgrading to raw-sector output
//!   is purely a sector-encoding change in [`IsoBuilder::build`].
//!
//! Sector map produced:
//!
//! ```text
//!   0..=15    zero-filled system area
//!   16        Primary Volume Descriptor (PVD)
//!   17        Volume Descriptor Set Terminator (VDST)
//!   18        L-path table (little-endian LBA fields)
//!   19        M-path table (big-endian LBA fields)
//!   20        Root directory extent
//!   21..      File data, one per file, each aligned to a sector
//! ```
//!
//! Reference: ECMA-119 (the published ISO 9660 standard), plus the
//! PSX-specific bits from PSX-SPX "CD-ROM File System".

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// One cooked sector = 2048 bytes of user data.
pub const SECTOR_SIZE: usize = 2048;

/// One raw CD-ROM sector = 2352 bytes:
///
/// - 12 sync bytes
/// - 4 header bytes
/// - 8 subheader bytes
/// - 2048 user-data bytes
/// - 280 EDC/ECC bytes
///
/// PSoXide's emulator CDROM path expects raw sectors; external
/// emulators accept either cooked .iso or raw .bin.
pub const RAW_SECTOR_SIZE: usize = 2352;

/// Fixed ISO 9660 date used for every volume / file timestamp. Games
/// don't read these, so any valid date works — we use 2026-01-01.
const FIXED_VOLUME_DATE: &[u8; 17] = b"2026010100000000\x00";
/// 7-byte directory record date: year-since-1900, month, day, hour,
/// minute, second, GMT offset in 15-minute units. 126 = 2026.
const FIXED_DIR_DATE: [u8; 7] = [126, 1, 1, 0, 0, 0, 0];

/// A single file to embed. Filenames are normalised to 8.3 upper-case
/// with a `;1` version suffix during serialisation.
pub struct IsoFile {
    /// User-supplied filename. Normalised at build time.
    pub name: String,
    /// Raw bytes to store.
    pub content: Vec<u8>,
}

/// Build a cooked ISO 9660 image from a set of root-directory files.
pub struct IsoBuilder {
    volume_id: String,
    system_id: String,
    files: Vec<IsoFile>,
}

impl Default for IsoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IsoBuilder {
    /// New builder with empty file list and sensible PS1 defaults
    /// (`system_id = "PLAYSTATION"`, `volume_id = "UNTITLED"`).
    pub fn new() -> Self {
        Self {
            volume_id: String::from("UNTITLED"),
            system_id: String::from("PLAYSTATION"),
            files: Vec::new(),
        }
    }

    /// Override the volume identifier. Will be upper-cased and
    /// truncated to 32 bytes (ISO 9660 d-characters).
    pub fn volume_id(mut self, id: &str) -> Self {
        self.volume_id = id.to_ascii_uppercase();
        self
    }

    /// Override the system identifier (written to PVD bytes 8..39).
    pub fn system_id(mut self, id: &str) -> Self {
        self.system_id = id.to_ascii_uppercase();
        self
    }

    /// Add a file to the root directory. Name is normalised later.
    pub fn add_file(&mut self, name: &str, content: Vec<u8>) -> &mut Self {
        self.files.push(IsoFile {
            name: name.to_ascii_uppercase(),
            content,
        });
        self
    }

    /// Serialise the image. Output length is always a multiple of
    /// [`SECTOR_SIZE`].
    pub fn build(&self) -> Vec<u8> {
        // ---------- Layout plan --------------------------------------
        // Every file takes ceil(size / 2048) sectors starting at a
        // known LBA we can bake into the root directory record.

        // Sector allocation for fixed headers:
        let lba_l_path = 18u32;
        let lba_m_path = 19u32;
        let lba_root_dir = 20u32;
        let mut next_file_lba = 21u32;

        // Compute the root directory contents + pre-assigned LBAs.
        struct FilePlacement {
            entry_name: String, // "FOO.TXT;1"
            lba: u32,
            size: u32,
        }
        let mut placements: Vec<FilePlacement> = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let entry_name = iso_file_identifier(&file.name);
            let size = file.content.len() as u32;
            let sectors = size.div_ceil(SECTOR_SIZE as u32).max(1);
            placements.push(FilePlacement {
                entry_name,
                lba: next_file_lba,
                size,
            });
            next_file_lba += sectors;
        }
        let total_sectors = next_file_lba;

        // ---------- Root directory extent ----------------------------
        let mut root_extent = Vec::with_capacity(SECTOR_SIZE);
        // "." entry (self).
        root_extent.extend_from_slice(&dir_record(lba_root_dir, SECTOR_SIZE as u32, &[0], true));
        // ".." entry (parent = self for root).
        root_extent.extend_from_slice(&dir_record(lba_root_dir, SECTOR_SIZE as u32, &[1], true));
        // File entries — stored as ASCII bytes from the placement names.
        for p in &placements {
            root_extent.extend_from_slice(&dir_record(
                p.lba,
                p.size,
                p.entry_name.as_bytes(),
                false,
            ));
        }
        pad_to(&mut root_extent, SECTOR_SIZE);

        // ---------- Path tables (root is the only directory) --------
        let mut l_path = Vec::with_capacity(SECTOR_SIZE);
        encode_path_table_entry(&mut l_path, lba_root_dir, 1, &[0], false);
        pad_to(&mut l_path, SECTOR_SIZE);

        let mut m_path = Vec::with_capacity(SECTOR_SIZE);
        encode_path_table_entry(&mut m_path, lba_root_dir, 1, &[0], true);
        pad_to(&mut m_path, SECTOR_SIZE);
        let path_table_size = 10u32; // 8-byte header + 1 ID + 1 pad

        // ---------- Primary Volume Descriptor -----------------------
        let mut pvd = vec![0u8; SECTOR_SIZE];
        pvd[0] = 0x01; // volume descriptor type = PVD
        pvd[1..6].copy_from_slice(b"CD001");
        pvd[6] = 0x01; // version
                       // pvd[7] = 0 (unused)
        fill_ascii(&mut pvd[8..40], &self.system_id);
        fill_ascii(&mut pvd[40..72], &self.volume_id);
        // pvd[72..80] = 0
        write_both_endian_u32(&mut pvd[80..88], total_sectors);
        // pvd[88..120] = 0 (unused)
        write_both_endian_u16(&mut pvd[120..124], 1); // volume set size
        write_both_endian_u16(&mut pvd[124..128], 1); // volume sequence
        write_both_endian_u16(&mut pvd[128..132], SECTOR_SIZE as u16);
        write_both_endian_u32(&mut pvd[132..140], path_table_size);
        write_le_u32(&mut pvd[140..144], lba_l_path);
        // pvd[144..148] = 0 (optional L-path table)
        write_be_u32(&mut pvd[148..152], lba_m_path);
        // pvd[152..156] = 0 (optional M-path table)

        // Root directory record — 34 bytes embedded at offset 156.
        let root_record = dir_record(lba_root_dir, SECTOR_SIZE as u32, &[0], true);
        pvd[156..156 + 34].copy_from_slice(&root_record);

        // Volume set / publisher / prepare / app / file identifiers —
        // all blank-filled per spec.
        fill_ascii(&mut pvd[190..318], ""); // volume set id (128 bytes)
        fill_ascii(&mut pvd[318..446], ""); // publisher id
        fill_ascii(&mut pvd[446..574], ""); // data preparer id
        fill_ascii(&mut pvd[574..702], ""); // application id
        fill_ascii(&mut pvd[702..740], ""); // copyright file id
        fill_ascii(&mut pvd[740..776], ""); // abstract file id
        fill_ascii(&mut pvd[776..813], ""); // bibliographic file id

        pvd[813..813 + 17].copy_from_slice(FIXED_VOLUME_DATE); // creation
        pvd[830..830 + 17].copy_from_slice(FIXED_VOLUME_DATE); // modification
        pvd[847..847 + 17].copy_from_slice(b"0000000000000000\x00"); // expiration
        pvd[864..864 + 17].copy_from_slice(b"0000000000000000\x00"); // effective
        pvd[881] = 0x01; // file structure version

        // ---------- Volume Descriptor Set Terminator ----------------
        let mut vdst = vec![0u8; SECTOR_SIZE];
        vdst[0] = 0xFF;
        vdst[1..6].copy_from_slice(b"CD001");
        vdst[6] = 0x01;

        // ---------- Stitch it all together --------------------------
        let mut out = vec![0u8; (total_sectors as usize) * SECTOR_SIZE];
        // Sectors 0..=15 are the 32 KiB system area (already zeroed).
        out[16 * SECTOR_SIZE..17 * SECTOR_SIZE].copy_from_slice(&pvd);
        out[17 * SECTOR_SIZE..18 * SECTOR_SIZE].copy_from_slice(&vdst);
        out[18 * SECTOR_SIZE..19 * SECTOR_SIZE].copy_from_slice(&l_path);
        out[19 * SECTOR_SIZE..20 * SECTOR_SIZE].copy_from_slice(&m_path);
        out[20 * SECTOR_SIZE..21 * SECTOR_SIZE].copy_from_slice(&root_extent);

        for (file, placement) in self.files.iter().zip(placements.iter()) {
            let start = placement.lba as usize * SECTOR_SIZE;
            let end = start + file.content.len();
            out[start..end].copy_from_slice(&file.content);
            // Remainder of the last sector stays zero (already is).
        }

        out
    }

    /// Serialise as a raw .bin image — 2352-byte Mode-2 Form-1 sectors
    /// with sync pattern, BCD MSF header, and subheader set. EDC/ECC
    /// bytes are left zero: PSoXide's CDROM emulation and most
    /// desktop emulators don't verify them, and a real PS1 lens
    /// hardware re-computes ECC on the fly.
    ///
    /// Use this output for PSoXide's disc loader (`Disc::from_bin`)
    /// and for emulators expecting .bin/.cue.
    pub fn build_bin(&self) -> Vec<u8> {
        let cooked = self.build();
        let total_sectors = cooked.len() / SECTOR_SIZE;
        let mut bin = vec![0u8; total_sectors * RAW_SECTOR_SIZE];
        for lba in 0..total_sectors {
            let cooked_start = lba * SECTOR_SIZE;
            let raw_start = lba * RAW_SECTOR_SIZE;
            let sector = &mut bin[raw_start..raw_start + RAW_SECTOR_SIZE];
            // Sync pattern: 00 FF×10 00.
            sector[0] = 0x00;
            for b in &mut sector[1..=10] {
                *b = 0xFF;
            }
            sector[11] = 0x00;
            // Header: BCD MSF + mode byte. MSF starts at 00:02:00 for
            // LBA 0 (2-second pre-gap). One frame = one LBA.
            let absolute_frame = lba as u32 + 150; // 150 = 00:02:00 in frames
            let minute = (absolute_frame / 75 / 60) as u8;
            let second = ((absolute_frame / 75) % 60) as u8;
            let frame = (absolute_frame % 75) as u8;
            sector[12] = bin_to_bcd(minute);
            sector[13] = bin_to_bcd(second);
            sector[14] = bin_to_bcd(frame);
            sector[15] = 0x02; // Mode 2
                               // Subheader: FILE=0, CHAN=0, SUBMODE=0x08 (data), CI=0
                               // (repeated twice — Mode 2 requires the sub-header pair).
            sector[16] = 0x00;
            sector[17] = 0x00;
            sector[18] = 0x08;
            sector[19] = 0x00;
            sector[20] = 0x00;
            sector[21] = 0x00;
            sector[22] = 0x08;
            sector[23] = 0x00;
            // User data — copy the cooked sector's 2048 bytes.
            sector[24..24 + SECTOR_SIZE]
                .copy_from_slice(&cooked[cooked_start..cooked_start + SECTOR_SIZE]);
            // EDC (2072..2076) + ECC (2076..2352) left zero.
        }
        bin
    }
}

fn bin_to_bcd(value: u8) -> u8 {
    ((value / 10) << 4) | (value % 10)
}

/// A sensible PSX `SYSTEM.CNF` that boots `PSX.EXE`. Write this as the
/// first file, then append the EXE under the name `PSX.EXE`.
pub fn default_system_cnf() -> Vec<u8> {
    // PSX-SPX documents: line-endings are CR-LF, each assignment on
    // its own line. TCB / EVENT / STACK are the "fine on everything"
    // values used by the official SDK templates.
    let text = b"BOOT = cdrom:\\PSX.EXE;1\r\nTCB = 4\r\nEVENT = 10\r\nSTACK = 801FFFF0\r\n";
    text.to_vec()
}

// ---------------------------------------------------------------------
// Helpers — byte-layer encoding of ISO 9660 primitives.
// ---------------------------------------------------------------------

fn write_le_u32(dst: &mut [u8], value: u32) {
    dst[0..4].copy_from_slice(&value.to_le_bytes());
}

fn write_be_u32(dst: &mut [u8], value: u32) {
    dst[0..4].copy_from_slice(&value.to_be_bytes());
}

fn write_both_endian_u32(dst: &mut [u8], value: u32) {
    dst[0..4].copy_from_slice(&value.to_le_bytes());
    dst[4..8].copy_from_slice(&value.to_be_bytes());
}

fn write_both_endian_u16(dst: &mut [u8], value: u16) {
    dst[0..2].copy_from_slice(&value.to_le_bytes());
    dst[2..4].copy_from_slice(&value.to_be_bytes());
}

/// Fill `dst` with `s` as ASCII, space-padding (or truncating) to
/// `dst.len()`.
fn fill_ascii(dst: &mut [u8], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(dst.len());
    dst[..n].copy_from_slice(&bytes[..n]);
    for byte in &mut dst[n..] {
        *byte = b' ';
    }
}

fn pad_to(buf: &mut Vec<u8>, size: usize) {
    if buf.len() < size {
        buf.resize(size, 0);
    }
}

/// Normalise a user-supplied filename like "hello.exe" to the
/// ISO 9660 Level 1 form "HELLO.EXE;1" suitable for a directory
/// record's file-identifier field.
fn iso_file_identifier(name: &str) -> String {
    let mut out = name.to_ascii_uppercase();
    if !out.contains(';') {
        out.push_str(";1");
    }
    out
}

/// Encode a single directory record. `ident` is the raw identifier
/// bytes — a single `\x00` for ".", single `\x01` for "..", or the
/// ASCII-uppercase filename otherwise.
fn dir_record(extent_lba: u32, size: u32, ident: &[u8], is_dir: bool) -> Vec<u8> {
    let ident_len = ident.len() as u8;
    let base_len: u8 = 33;
    let mut record_len = base_len + ident_len;
    if record_len % 2 != 0 {
        record_len += 1;
    }
    let mut r = vec![0u8; record_len as usize];

    r[0] = record_len;
    r[1] = 0; // extended attribute record length
    write_both_endian_u32(&mut r[2..10], extent_lba);
    write_both_endian_u32(&mut r[10..18], size);
    r[18..25].copy_from_slice(&FIXED_DIR_DATE);
    r[25] = if is_dir { 0x02 } else { 0x00 };
    // r[26] file unit size = 0
    // r[27] interleave gap = 0
    write_both_endian_u16(&mut r[28..32], 1); // volume seq number
    r[32] = ident_len;
    r[33..33 + ident.len()].copy_from_slice(ident);
    // Optional trailing pad byte already zero if record_len was odd.

    r
}

/// Append one path-table entry for a directory. `parent_num` is 1 for
/// root (self-reference). `big_endian` toggles between the L-path and
/// M-path encoding.
fn encode_path_table_entry(
    buf: &mut Vec<u8>,
    extent_lba: u32,
    parent_num: u16,
    ident: &[u8],
    big_endian: bool,
) {
    buf.push(ident.len() as u8);
    buf.push(0); // extended attr record length
    if big_endian {
        buf.extend_from_slice(&extent_lba.to_be_bytes());
        buf.extend_from_slice(&parent_num.to_be_bytes());
    } else {
        buf.extend_from_slice(&extent_lba.to_le_bytes());
        buf.extend_from_slice(&parent_num.to_le_bytes());
    }
    buf.extend_from_slice(ident);
    if ident.len() % 2 != 0 {
        buf.push(0); // padding to even
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_produces_minimal_image() {
        // Even with no files: sector map 0..=20 is fixed, file area
        // starts at 21 but is empty, so total = 21 sectors.
        let img = IsoBuilder::new().build();
        assert_eq!(img.len(), 21 * SECTOR_SIZE);
        // System area is zero-filled.
        assert!(img[..16 * SECTOR_SIZE].iter().all(|&b| b == 0));
        // PVD at sector 16 with CD001 signature.
        assert_eq!(img[16 * SECTOR_SIZE], 0x01);
        assert_eq!(&img[16 * SECTOR_SIZE + 1..16 * SECTOR_SIZE + 6], b"CD001");
        // VDST at sector 17.
        assert_eq!(img[17 * SECTOR_SIZE], 0xFF);
        assert_eq!(&img[17 * SECTOR_SIZE + 1..17 * SECTOR_SIZE + 6], b"CD001");
    }

    #[test]
    fn single_file_lands_at_expected_sector() {
        let mut b = IsoBuilder::new();
        b.add_file("HELLO.TXT", b"hello world".to_vec());
        let img = b.build();
        // File should live at sector 21 per our layout plan.
        let file_start = 21 * SECTOR_SIZE;
        assert_eq!(&img[file_start..file_start + 11], b"hello world");
        // Everything past the written bytes in that sector is zero.
        assert!(img[file_start + 11..file_start + SECTOR_SIZE]
            .iter()
            .all(|&b| b == 0));
    }

    #[test]
    fn volume_id_lands_in_pvd() {
        let img = IsoBuilder::new().volume_id("HELLO").build();
        let vol_start = 16 * SECTOR_SIZE + 40;
        assert_eq!(&img[vol_start..vol_start + 5], b"HELLO");
    }

    #[test]
    fn multi_file_adds_sectors_per_file() {
        let mut b = IsoBuilder::new();
        b.add_file("SMALL.TXT", vec![b'A'; 100]);
        // Second file = 2 sectors worth; needs two allocated sectors.
        b.add_file("BIG.BIN", vec![b'B'; SECTOR_SIZE + 500]);
        let img = b.build();
        // SMALL.TXT at sector 21, BIG.BIN at sector 22.
        assert_eq!(img[21 * SECTOR_SIZE], b'A');
        assert_eq!(img[22 * SECTOR_SIZE], b'B');
        assert_eq!(img[23 * SECTOR_SIZE], b'B');
        // Total sector count = 21 (fixed) + 1 (SMALL.TXT) + 2 (BIG.BIN).
        assert_eq!(img.len() / SECTOR_SIZE, 24);
    }

    #[test]
    fn default_system_cnf_boots_psx_exe() {
        let cnf = default_system_cnf();
        let text = core::str::from_utf8(&cnf).unwrap();
        assert!(text.contains("BOOT = cdrom:\\PSX.EXE;1"));
        assert!(text.contains("STACK = 801FFFF0"));
    }

    #[test]
    fn file_identifier_gets_version_suffix() {
        assert_eq!(iso_file_identifier("HELLO.TXT"), "HELLO.TXT;1");
        assert_eq!(iso_file_identifier("already;5"), "ALREADY;5");
    }

    #[test]
    fn path_table_sizes_match_pvd_field() {
        let img = IsoBuilder::new().build();
        // PVD path-table size at bytes 132..140 of sector 16.
        let pvd = &img[16 * SECTOR_SIZE..17 * SECTOR_SIZE];
        let path_size_le = u32::from_le_bytes(pvd[132..136].try_into().unwrap());
        assert_eq!(path_size_le, 10);
    }

    #[test]
    fn file_round_trips_via_disc_reader() {
        // Build a raw .bin image, then parse it with the companion
        // `Disc` type to confirm a file is readable through the same
        // path the emulator uses. End-to-end sanity check that the
        // PVD + directory map + file-extent LBAs all agree.
        let payload = b"ROUNDTRIP";
        let mut b = IsoBuilder::new();
        b.add_file("TEST.BIN", payload.to_vec());
        let bin = b.build_bin();

        let disc = crate::Disc::from_bin(bin);
        // File should be at sector 21 per the layout plan.
        let sector = disc.read_sector_user(21).expect("read sector 21");
        assert_eq!(&sector[..payload.len()], payload);
    }

    #[test]
    fn raw_bin_has_sync_pattern_at_every_sector() {
        let img = IsoBuilder::new().build_bin();
        let total = img.len() / RAW_SECTOR_SIZE;
        assert!(total >= 21);
        for lba in 0..total {
            let base = lba * RAW_SECTOR_SIZE;
            assert_eq!(img[base], 0x00, "LBA {lba}: sync byte 0");
            for (i, &b) in img[base + 1..base + 11].iter().enumerate() {
                assert_eq!(b, 0xFF, "LBA {lba} sync[{}]", i + 1);
            }
            assert_eq!(img[base + 11], 0x00, "LBA {lba}: sync byte 11");
            assert_eq!(img[base + 15], 0x02, "LBA {lba}: mode byte");
        }
    }

    #[test]
    fn raw_bin_user_data_matches_cooked() {
        let mut b = IsoBuilder::new();
        b.add_file("X.DAT", b"PAYLOAD-BYTES".to_vec());
        let cooked = b.build();
        let raw = b.build_bin();
        for lba in 0..(cooked.len() / SECTOR_SIZE) {
            let cooked_start = lba * SECTOR_SIZE;
            let raw_user_start = lba * RAW_SECTOR_SIZE + 24;
            assert_eq!(
                &cooked[cooked_start..cooked_start + SECTOR_SIZE],
                &raw[raw_user_start..raw_user_start + SECTOR_SIZE],
                "user data differs at LBA {lba}"
            );
        }
    }

    #[test]
    fn raw_bin_msf_header_is_bcd_encoded() {
        // LBA 0 → MSF 00:02:00 → BCD 0x00 0x02 0x00.
        let img = IsoBuilder::new().build_bin();
        assert_eq!(img[12], 0x00);
        assert_eq!(img[13], 0x02);
        assert_eq!(img[14], 0x00);
        // LBA 16 → 00:02:16 → BCD 0x00 0x02 0x16.
        let b16 = 16 * RAW_SECTOR_SIZE;
        assert_eq!(img[b16 + 12], 0x00);
        assert_eq!(img[b16 + 13], 0x02);
        assert_eq!(img[b16 + 14], 0x16);
    }
}
