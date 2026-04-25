//! ISO9660 boot-file discovery for PlayStation discs.
//!
//! Retail discs boot through `SYSTEM.CNF`: the BIOS reads the file
//! from the ISO9660 root directory, extracts the `BOOT = cdrom:\...`
//! path, then loads that PSX-EXE. This module implements just that
//! read path so emulator-side fast boot can start game code without
//! depending on the BIOS license-screen handoff.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::{Disc, Exe, ExeError, SECTOR_USER_DATA_BYTES};

const PVD_LBA: u32 = 16;
const ISO_MAGIC: &[u8; 5] = b"CD001";
const ROOT_RECORD_OFFSET: usize = 156;
const SYSTEM_CNF: &str = "SYSTEM.CNF;1";

/// Boot executable found through `SYSTEM.CNF`.
#[derive(Debug)]
pub struct BootExe {
    /// Boot path exactly as parsed and normalized from `SYSTEM.CNF`.
    pub boot_path: String,
    /// Optional stack pointer requested by `SYSTEM.CNF`.
    pub stack_pointer: Option<u32>,
    /// Parsed PSX-EXE referenced by [`BootExe::boot_path`].
    pub exe: Exe,
}

/// Errors surfaced while discovering a disc's `SYSTEM.CNF` boot EXE.
#[derive(Debug, PartialEq, Eq)]
pub enum BootError {
    /// LBA 16 could not be read.
    MissingPrimaryVolumeDescriptor,
    /// LBA 16 was not an ISO9660 Primary Volume Descriptor.
    BadPrimaryVolumeDescriptor,
    /// The embedded root directory record was malformed.
    BadRootDirectoryRecord,
    /// A directory extent could not be read from the disc image.
    DirectoryExtentUnreadable { extent_lba: u32, size: u32 },
    /// A directory entry was malformed or ran past the sector data.
    BadDirectoryRecord,
    /// The requested path was not present in the ISO9660 tree.
    FileNotFound(String),
    /// A path component expected to be a directory was a regular file.
    NotDirectory(String),
    /// `SYSTEM.CNF` did not include a `BOOT = ...` assignment.
    MissingBootPath,
    /// The file named by `BOOT` could not be parsed as a PSX-EXE.
    Exe(ExeError),
}

impl From<ExeError> for BootError {
    fn from(value: ExeError) -> Self {
        Self::Exe(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirEntry {
    extent_lba: u32,
    size: u32,
    flags: u8,
    identifier: String,
}

impl DirEntry {
    fn is_dir(&self) -> bool {
        self.flags & 0x02 != 0
    }
}

/// Locate and parse the PSX-EXE named by `SYSTEM.CNF`.
pub fn load_boot_exe_from_disc(disc: &Disc) -> Result<BootExe, BootError> {
    let root = root_directory(disc)?;
    let system_entry = lookup_path(disc, &root, SYSTEM_CNF)?;
    let system_cnf = read_extent(disc, system_entry.extent_lba, system_entry.size)?;
    let boot_path = parse_system_cnf_boot_path(&system_cnf).ok_or(BootError::MissingBootPath)?;
    let stack_pointer = parse_system_cnf_stack(&system_cnf);
    let exe_entry = lookup_path(disc, &root, &boot_path)?;
    let exe_bytes = read_extent(disc, exe_entry.extent_lba, exe_entry.size)?;
    let exe = Exe::parse(&exe_bytes)?;

    Ok(BootExe {
        boot_path,
        stack_pointer,
        exe,
    })
}

fn root_directory(disc: &Disc) -> Result<DirEntry, BootError> {
    let pvd = disc
        .read_sector_user(PVD_LBA)
        .ok_or(BootError::MissingPrimaryVolumeDescriptor)?;
    if pvd.len() < ROOT_RECORD_OFFSET + 34 || pvd[0] != 1 || &pvd[1..6] != ISO_MAGIC || pvd[6] != 1
    {
        return Err(BootError::BadPrimaryVolumeDescriptor);
    }
    parse_dir_record(&pvd[ROOT_RECORD_OFFSET..]).ok_or(BootError::BadRootDirectoryRecord)
}

fn lookup_path(disc: &Disc, root: &DirEntry, path: &str) -> Result<DirEntry, BootError> {
    let components = normalize_disc_path(path);
    let Some((last, parents)) = components.split_last() else {
        return Err(BootError::FileNotFound(path.to_string()));
    };

    let mut dir = root.clone();
    for component in parents {
        let next = find_child(disc, &dir, component)?
            .ok_or_else(|| BootError::FileNotFound(path.to_string()))?;
        if !next.is_dir() {
            return Err(BootError::NotDirectory(component.clone()));
        }
        dir = next;
    }

    find_child(disc, &dir, last)?.ok_or_else(|| BootError::FileNotFound(path.to_string()))
}

fn find_child(
    disc: &Disc,
    directory: &DirEntry,
    component: &str,
) -> Result<Option<DirEntry>, BootError> {
    let entries = read_directory(disc, directory)?;
    Ok(entries
        .into_iter()
        .find(|entry| identifier_matches(&entry.identifier, component)))
}

fn read_directory(disc: &Disc, directory: &DirEntry) -> Result<Vec<DirEntry>, BootError> {
    if !directory.is_dir() {
        return Err(BootError::NotDirectory(directory.identifier.clone()));
    }
    let bytes = read_extent(disc, directory.extent_lba, directory.size)?;
    let mut entries = Vec::new();
    let mut offset = 0usize;

    while offset < bytes.len() {
        let len = bytes[offset] as usize;
        if len == 0 {
            offset = ((offset / SECTOR_USER_DATA_BYTES) + 1) * SECTOR_USER_DATA_BYTES;
            continue;
        }
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= bytes.len())
            .ok_or(BootError::BadDirectoryRecord)?;
        let entry = parse_dir_record(&bytes[offset..end]).ok_or(BootError::BadDirectoryRecord)?;
        if entry.identifier != "\u{0}" && entry.identifier != "\u{1}" {
            entries.push(entry);
        }
        offset = end;
    }

    Ok(entries)
}

fn parse_dir_record(record: &[u8]) -> Option<DirEntry> {
    let len = *record.first()? as usize;
    if len < 34 || record.len() < len {
        return None;
    }
    let ident_len = *record.get(32)? as usize;
    let ident_start = 33usize;
    let ident_end = ident_start.checked_add(ident_len)?;
    if ident_end > len {
        return None;
    }
    let identifier = String::from_utf8(record[ident_start..ident_end].to_vec()).ok()?;
    Some(DirEntry {
        extent_lba: read_le_u32(record.get(2..6)?),
        size: read_le_u32(record.get(10..14)?),
        flags: *record.get(25)?,
        identifier,
    })
}

fn read_extent(disc: &Disc, extent_lba: u32, size: u32) -> Result<Vec<u8>, BootError> {
    let mut out = Vec::with_capacity(size as usize);
    let sectors = size.div_ceil(SECTOR_USER_DATA_BYTES as u32);
    for sector in 0..sectors {
        let lba = extent_lba.saturating_add(sector);
        let data = disc
            .read_sector_user(lba)
            .ok_or(BootError::DirectoryExtentUnreadable { extent_lba, size })?;
        out.extend_from_slice(data);
    }
    out.truncate(size as usize);
    Ok(out)
}

fn normalize_disc_path(path: &str) -> Vec<String> {
    let mut p = path.trim().trim_matches('\0').trim().to_ascii_uppercase();
    if let Some(rest) = p.strip_prefix("CDROM:") {
        p = rest.to_string();
    } else if let Some(rest) = p.strip_prefix("CDROM0:") {
        p = rest.to_string();
    }
    p = p.trim_start_matches(['\\', '/']).to_string();
    p.split(['\\', '/'])
        .filter(|component| !component.is_empty())
        .map(|component| {
            if component.contains(';') {
                component.to_string()
            } else {
                let mut with_version = component.to_string();
                with_version.push_str(";1");
                with_version
            }
        })
        .collect()
}

fn identifier_matches(identifier: &str, component: &str) -> bool {
    let ident = identifier.to_ascii_uppercase();
    if ident == component {
        return true;
    }
    if let Some((without_version, _)) = ident.split_once(';') {
        without_version == component.trim_end_matches(";1")
    } else {
        ident == component.trim_end_matches(";1")
    }
}

fn parse_system_cnf_boot_path(bytes: &[u8]) -> Option<String> {
    for line in bytes.split(|byte| *byte == b'\n') {
        let line = trim_ascii(line);
        let Some(eq) = line.iter().position(|byte| *byte == b'=') else {
            continue;
        };
        let key = trim_ascii(&line[..eq]);
        if !key.eq_ignore_ascii_case(b"BOOT") {
            continue;
        }
        let value = trim_ascii(&line[eq + 1..]);
        let value = trim_ascii(trim_ascii_byte(value, b'"'));
        let text = core::str::from_utf8(value).ok()?;
        return Some(normalize_disc_path(text).join("\\"));
    }
    None
}

fn parse_system_cnf_stack(bytes: &[u8]) -> Option<u32> {
    for line in bytes.split(|byte| *byte == b'\n') {
        let line = trim_ascii(line);
        let Some(eq) = line.iter().position(|byte| *byte == b'=') else {
            continue;
        };
        let key = trim_ascii(&line[..eq]);
        if !key.eq_ignore_ascii_case(b"STACK") {
            continue;
        }
        let value = trim_ascii(&line[eq + 1..]);
        let text = core::str::from_utf8(value).ok()?.trim();
        return parse_hex_u32(text);
    }
    None
}

fn parse_hex_u32(text: &str) -> Option<u32> {
    let value = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u32::from_str_radix(value, 16).ok()
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && is_ascii_trim_byte(bytes[start]) {
        start += 1;
    }
    while end > start && is_ascii_trim_byte(bytes[end - 1]) {
        end -= 1;
    }
    &bytes[start..end]
}

fn trim_ascii_byte(bytes: &[u8], byte: u8) -> &[u8] {
    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && bytes[start] == byte {
        start += 1;
    }
    while end > start && bytes[end - 1] == byte {
        end -= 1;
    }
    &bytes[start..end]
}

fn is_ascii_trim_byte(byte: u8) -> bool {
    matches!(byte, b'\0' | b'\t' | b'\n' | b'\r' | b' ' | 0x0B | 0x0C)
}

fn read_le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{IsoBuilder, EXE_HEADER_BYTES};

    fn make_exe(pc: u32, load_addr: u32, payload: &[u8]) -> Vec<u8> {
        let mut exe = vec![0u8; EXE_HEADER_BYTES];
        exe[..8].copy_from_slice(b"PS-X EXE");
        exe[0x10..0x14].copy_from_slice(&pc.to_le_bytes());
        exe[0x18..0x1C].copy_from_slice(&load_addr.to_le_bytes());
        exe[0x1C..0x20].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        exe.extend_from_slice(payload);
        exe
    }

    #[test]
    fn finds_root_system_cnf_boot_exe() {
        let exe = make_exe(0x8001_2340, 0x8001_0000, &[1, 2, 3, 4]);
        let mut builder = IsoBuilder::new();
        builder.add_file(
            "SYSTEM.CNF",
            b"BOOT = cdrom:\\PSX.EXE;1\r\nSTACK = 801FFFF0\r\n".to_vec(),
        );
        builder.add_file("PSX.EXE", exe);

        let disc = Disc::from_bin(builder.build_bin());
        let boot = load_boot_exe_from_disc(&disc).unwrap();
        assert_eq!(boot.boot_path, "PSX.EXE;1");
        assert_eq!(boot.stack_pointer, Some(0x801F_FFF0));
        assert_eq!(boot.exe.initial_pc, 0x8001_2340);
        assert_eq!(boot.exe.load_addr, 0x8001_0000);
        assert_eq!(boot.exe.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn reports_missing_boot_assignment() {
        let mut builder = IsoBuilder::new();
        builder.add_file("SYSTEM.CNF", b"TCB = 4\r\n".to_vec());

        let disc = Disc::from_bin(builder.build_bin());
        assert_eq!(
            load_boot_exe_from_disc(&disc).unwrap_err(),
            BootError::MissingBootPath
        );
    }

    #[test]
    fn normalizes_cdrom_paths() {
        assert_eq!(
            normalize_disc_path("cdrom:\\dir\\foo.exe"),
            vec!["DIR;1".to_string(), "FOO.EXE;1".to_string()]
        );
        assert_eq!(
            normalize_disc_path("cdrom0:/SLUS_005.94;1"),
            vec!["SLUS_005.94;1".to_string()]
        );
    }
}
