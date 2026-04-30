//! Scanned game library -- discovery, metadata extraction, and
//! on-disk cache.
//!
//! The scanner walks a configured root directory and classifies
//! each file as either a disc image (`.bin`, `.cue`, `.iso`) or a
//! side-loadable homebrew (`.exe`). For each hit it extracts
//! cheap-to-read metadata so the UI can show a useful label
//! without running the emulator:
//!
//! - **Title** -- from the ISO9660 Primary Volume Descriptor's
//!   *volume identifier* field (for BIN/ISO), or the file stem
//!   (for EXE / anything we can't parse).
//! - **Region** -- inferred from the PSX license-text sector at
//!   LBA 4 (`Licensed by Sony Computer Entertainment America /
//!   Europe / Japan`).
//! - **Stable ID** -- a 16-hex-char FNV-1a-64 fingerprint over the
//!   license text + PVD identifier bytes. Same disc → same ID
//!   regardless of filename, so renaming a BIN doesn't orphan
//!   its savestates.
//!
//! Results are cached in `library.ron` alongside the source file's
//! last-modified time. A subsequent scan skips re-parsing files
//! whose mtime hasn't changed -- fast startup even with a big
//! library.
//!
//! Parsing is best-effort: any file that errors out surfaces as
//! [`LibraryEntry::kind`] == [`GameKind::Unknown`] with a reason
//! recorded in [`LibraryEntry::diagnostic`]. A malformed BIN
//! doesn't derail the whole scan.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::LIBRARY_VERSION;

/// Discovered game category. Drives the UI grid -- disc images,
/// homebrew, unknown/diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameKind {
    /// A full PSX disc image in raw 2352-byte-per-sector format.
    DiscBin,
    /// An ISO9660 image (2048 bytes per sector, no raw subchannel).
    /// We don't *boot* these yet -- the CD controller expects BIN
    /// sector layout -- but they show up in the library so the user
    /// can see them.
    DiscIso,
    /// A `.cue` playlist pointing at one or more BIN files.
    DiscCue,
    /// A PSX-EXE homebrew binary (our SDK's output + many demos).
    Exe,
    /// Didn't match any known format, or parsing failed. The
    /// [`LibraryEntry::diagnostic`] field carries the "why" so
    /// the UI can surface it.
    Unknown,
}

/// Region code inferred from the PSX license-text sector.
/// A real drive refuses non-matching regions; we record the info
/// for the UI but don't *enforce* it (users may swap BIOSes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Region {
    /// NTSC-U (US / Canada).
    NtscU,
    /// PAL (Europe, most of the world).
    Pal,
    /// NTSC-J (Japan + Asia).
    NtscJ,
    /// License text not recognised -- either a pre-release / unlicensed
    /// disc, or our heuristic missed it.
    Unknown,
}

/// A single discovered library entry. Serialised into
/// `library.ron`; small enough to comfortably hold a few thousand
/// entries in RAM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryEntry {
    /// Stable 16-hex-char game ID. Used to name the per-game
    /// directory under `games/<id>/`.
    pub id: String,
    /// Absolute path to the underlying file. Relative paths inside
    /// `library.ron` would be fragile across working directories.
    pub path: PathBuf,
    /// Classification -- what kind of file this is.
    pub kind: GameKind,
    /// Human-readable display name -- prefer the PVD volume
    /// identifier if present, else the file stem.
    pub title: String,
    /// Region code, best-effort.
    pub region: Region,
    /// File size in bytes. Displayed in the UI; also a cheap
    /// sanity-check on corruption.
    pub size: u64,
    /// File mtime (UNIX epoch seconds) captured at scan time.
    /// The next scan skips the parse step if the mtime hasn't
    /// moved -- huge startup win for big libraries.
    pub mtime: u64,
    /// Optional free-text diagnostic -- carries the "why" when
    /// `kind == Unknown`, or any notable warning for other kinds.
    pub diagnostic: Option<String>,
}

/// Full library cache + its version. Top-level of `library.ron`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Library {
    /// Schema version. Bumped when [`LibraryEntry`] changes
    /// incompatibly; older caches are silently discarded and
    /// regenerated on next scan.
    #[serde(default = "default_library_version")]
    pub version: u32,
    /// Discovered entries. Order is file-system walk order -- UI
    /// sorts at display time.
    #[serde(default)]
    pub entries: Vec<LibraryEntry>,
}

fn default_library_version() -> u32 {
    LIBRARY_VERSION
}

/// Errors from library load/save/scan.
#[derive(Debug, Error)]
pub enum LibraryError {
    /// Filesystem error while reading the cache or scanning.
    #[error("library I/O error at {path}: {source}")]
    Io {
        /// The file or directory we were working on.
        path: PathBuf,
        /// The underlying `io::Error`.
        #[source]
        source: io::Error,
    },
    /// Cache file couldn't be parsed. The scanner treats this as
    /// "regenerate from scratch" and logs -- no crash.
    #[error("library parse error at {path}: {source}")]
    Parse {
        /// The file we were parsing.
        path: PathBuf,
        /// RON parser's error.
        #[source]
        source: ron::error::SpannedError,
    },
    /// Serialisation failed.
    #[error("library serialization error: {0}")]
    Serialize(#[from] ron::Error),
}

impl Library {
    /// Load `library.ron` from `path`. Missing file / wrong version
    /// / corrupt file all return an empty [`Library`] rather than
    /// erroring -- the scanner will rebuild it. This keeps a
    /// first-run UX of "just works."
    pub fn load_or_empty(path: &Path) -> Self {
        match Self::load(path) {
            Ok(lib) if lib.version == LIBRARY_VERSION => lib,
            Ok(_) | Err(_) => Self {
                version: LIBRARY_VERSION,
                entries: Vec::new(),
            },
        }
    }

    /// Strict load -- propagates parse errors. Used in tests and by
    /// diagnostics that want to distinguish "missing" from
    /// "corrupt." Most production code wants [`load_or_empty`].
    pub fn load(path: &Path) -> Result<Self, LibraryError> {
        let contents = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    version: LIBRARY_VERSION,
                    entries: Vec::new(),
                });
            }
            Err(source) => {
                return Err(LibraryError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        ron::from_str(&contents).map_err(|source| LibraryError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Write the cache to `path` atomically. Same tmp-and-rename
    /// pattern as [`crate::Settings::save`].
    pub fn save(&self, path: &Path) -> Result<(), LibraryError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| LibraryError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }
        let body = ron::ser::to_string_pretty(
            self,
            ron::ser::PrettyConfig::new()
                .depth_limit(4)
                .indentor("    ".to_string()),
        )?;
        let tmp = path.with_extension("ron.tmp");
        fs::write(&tmp, body).map_err(|source| LibraryError::Io {
            path: tmp.clone(),
            source,
        })?;
        fs::rename(&tmp, path).map_err(|source| LibraryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    /// Walk `root` recursively and update the library in place. For
    /// each file whose extension matches a known format:
    ///
    /// - If the cache already has an entry with the same path AND
    ///   the file's mtime hasn't moved, keep the cached metadata.
    /// - Otherwise, re-parse and either insert or replace.
    ///
    /// Entries whose file no longer exists on disk are pruned.
    /// Returns the number of entries added/updated (for diagnostics).
    pub fn scan(&mut self, root: &Path) -> Result<usize, LibraryError> {
        if !root.is_dir() {
            return Err(LibraryError::Io {
                path: root.to_path_buf(),
                source: io::Error::new(io::ErrorKind::NotFound, "library root is not a directory"),
            });
        }

        // Build a quick path → cache-index lookup so we can reuse
        // unchanged entries without re-parsing.
        let mut by_path: std::collections::HashMap<PathBuf, usize> =
            std::collections::HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            by_path.insert(e.path.clone(), i);
        }

        let mut fresh: Vec<LibraryEntry> = Vec::new();
        let mut changed = 0usize;
        for path in walk(root) {
            let Some(kind) = classify(&path) else {
                continue;
            };
            let meta = match fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let size = meta.len();

            // Reuse cached entry if mtime hasn't moved.
            if let Some(&idx) = by_path.get(&path) {
                let cached = &self.entries[idx];
                if cached.mtime == mtime && cached.size == size {
                    fresh.push(cached.clone());
                    continue;
                }
            }

            let parsed = parse_entry(&path, kind, size, mtime);
            changed += 1;
            fresh.push(parsed);
        }

        self.entries = fresh;
        self.version = LIBRARY_VERSION;
        Ok(changed)
    }

    /// Walk multiple roots and union their scan results into a
    /// single library. Roots that don't exist are silently skipped
    /// (common case: the SDK-examples path only exists on developer
    /// machines with the nightly toolchain; end users won't have
    /// it). Roots that fail with a real error abort the whole
    /// scan and surface the error.
    ///
    /// Internally it does what `scan` does but across every
    /// provided root, using a single `fresh` accumulator so the
    /// final `self.entries` is the union. Useful for the
    /// frontend's "scan retail games + SDK examples together" case.
    pub fn scan_roots(&mut self, roots: &[&Path]) -> Result<usize, LibraryError> {
        // Reuse the existing cache map so mtime-match still short-
        // circuits re-parses across ALL roots.
        let mut by_path: std::collections::HashMap<PathBuf, usize> =
            std::collections::HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            by_path.insert(e.path.clone(), i);
        }

        let mut fresh: Vec<LibraryEntry> = Vec::new();
        let mut changed = 0usize;

        for root in roots {
            if !root.exists() {
                // Missing roots are a normal condition (end-user
                // install without SDK builds). Silently skip.
                continue;
            }
            if !root.is_dir() {
                return Err(LibraryError::Io {
                    path: root.to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "library root exists but is not a directory",
                    ),
                });
            }

            for path in walk(root) {
                let Some(kind) = classify(&path) else {
                    continue;
                };
                let meta = match fs::metadata(&path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let size = meta.len();

                if let Some(&idx) = by_path.get(&path) {
                    let cached = &self.entries[idx];
                    if cached.mtime == mtime && cached.size == size {
                        fresh.push(cached.clone());
                        continue;
                    }
                }

                let parsed = parse_entry(&path, kind, size, mtime);
                changed += 1;
                fresh.push(parsed);
            }
        }

        self.entries = fresh;
        self.version = LIBRARY_VERSION;
        Ok(changed)
    }
}

/// Classify a file by extension only. Returns `None` for files we
/// don't care about (images, archives, etc). Keeps `scan()` cheap --
/// we only hit the disk to parse files we'll actually show.
fn classify(path: &Path) -> Option<GameKind> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "bin" => Some(GameKind::DiscBin),
        "iso" => Some(GameKind::DiscIso),
        "cue" => Some(GameKind::DiscCue),
        "exe" => Some(GameKind::Exe),
        _ => None,
    }
}

/// Directory names the recursive walker skips. Primarily cargo's
/// per-target build-tree siblings -- if the user points a library
/// scanner at a cargo target-dir (the SDK-examples dir is exactly
/// this case), we'd otherwise surface every intermediate
/// `hello_tri-<hash>.exe` living under `deps/` as a separate
/// library entry. Names match cargo's layout; `.fingerprint` +
/// hidden dirs are caught by the leading-dot filter in `walk`.
///
/// Also kept short because a false positive (user has a folder
/// legitimately named `deps/`) is strictly worse than scanning
/// through it -- keep the list tight and explainable.
const SKIP_DIRS: &[&str] = &["deps", "incremental", "build"];

/// Recursive directory walk. Returns a flat list of file paths --
/// a full-blown `WalkDir` dep feels over-engineered for a few
/// dozen lines of plain `read_dir`. Skips directories in
/// [`SKIP_DIRS`] and anything starting with `.`, so cargo's
/// per-target build siblings don't show up as library entries
/// when the scanner is pointed at an SDK-build output tree.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                // Skip cargo's build-artifact siblings (`deps/`,
                // `incremental/`, `build/`) and any hidden dir
                // (`.fingerprint/`, dotfiles). Using `file_name()`
                // rather than full-path matching means a user's
                // real `deps/` folder anywhere in the walk is also
                // skipped -- an acceptable trade for keeping the
                // scanner noise-free when rooted at a cargo
                // target-dir like `build/examples/.../release/`.
                let skip = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with('.') || SKIP_DIRS.contains(&n))
                    .unwrap_or(false);
                if !skip {
                    stack.push(p);
                }
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

/// Parse one file into a `LibraryEntry`. Uses cheap heuristics --
/// read one or two well-known sectors and the file stem. Never
/// loads the whole file (Crash Bandicoot is 600 MiB; scanning
/// wouldn't finish).
fn parse_entry(path: &Path, kind: GameKind, size: u64, mtime: u64) -> LibraryEntry {
    let fallback_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string();

    match kind {
        GameKind::DiscBin => parse_bin(path, size, mtime, &fallback_title),
        GameKind::DiscIso => LibraryEntry {
            id: fingerprint(&[fallback_title.as_bytes()]),
            path: path.to_path_buf(),
            kind,
            title: fallback_title,
            region: Region::Unknown,
            size,
            mtime,
            // ISO parsing shares the PVD path with BIN once we
            // handle 2048-byte sectors -- just not today.
            diagnostic: Some("ISO sector-size parsing not yet implemented".into()),
        },
        GameKind::DiscCue => parse_cue(path, size, mtime, &fallback_title),
        GameKind::Exe => LibraryEntry {
            id: fingerprint(&[fallback_title.as_bytes(), b"exe"]),
            path: path.to_path_buf(),
            kind,
            title: fallback_title,
            region: Region::Unknown,
            size,
            mtime,
            diagnostic: None,
        },
        GameKind::Unknown => LibraryEntry {
            id: fingerprint(&[fallback_title.as_bytes()]),
            path: path.to_path_buf(),
            kind,
            title: fallback_title,
            region: Region::Unknown,
            size,
            mtime,
            diagnostic: Some("unknown file kind".into()),
        },
    }
}

/// Parse a raw 2352-byte-per-sector BIN. Reads:
///
/// - LBA 4 user-data → PSX license text → region
/// - LBA 16 user-data → ISO9660 PVD → volume identifier → title
///
/// Each read is bounded by a sector (2352 B), so the total disk
/// IO for one BIN is ~5 KiB regardless of image size.
fn parse_bin(path: &Path, size: u64, mtime: u64, fallback_title: &str) -> LibraryEntry {
    use psx_iso::{SECTOR_BYTES, SECTOR_USER_DATA_BYTES, SECTOR_USER_DATA_OFFSET};

    // Helper: read the 2048-byte user-data payload of a given LBA
    // directly from the file, without loading it all. `None` on
    // any I/O error or short read.
    let read_user = |lba: u64| -> Option<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = fs::File::open(path).ok()?;
        let byte_offset = lba
            .checked_mul(SECTOR_BYTES as u64)?
            .checked_add(SECTOR_USER_DATA_OFFSET as u64)?;
        f.seek(SeekFrom::Start(byte_offset)).ok()?;
        let mut buf = vec![0u8; SECTOR_USER_DATA_BYTES];
        match f.read_exact(&mut buf) {
            Ok(()) => Some(buf),
            Err(_) => None,
        }
    };

    let license_user = read_user(4);
    let pvd_user = read_user(16);
    let region = license_user
        .as_deref()
        .map(region_from_license_text)
        .unwrap_or(Region::Unknown);
    let title = pvd_user
        .as_deref()
        .and_then(pvd_volume_identifier)
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| fallback_title.to_string());

    // Stable ID: hash the license + PVD system/volume bytes so two
    // BINs of the same disc get the same ID across renames.
    let mut parts: Vec<&[u8]> = Vec::new();
    if let Some(ref bytes) = license_user {
        parts.push(&bytes[..bytes.len().min(256)]);
    }
    if let Some(ref bytes) = pvd_user {
        // Volume identifier region -- stable across burns.
        parts.push(&bytes[40..bytes.len().min(72)]);
    }
    if parts.is_empty() {
        parts.push(fallback_title.as_bytes());
    }
    let id = fingerprint(&parts);

    LibraryEntry {
        id,
        path: path.to_path_buf(),
        kind: GameKind::DiscBin,
        title,
        region,
        size,
        mtime,
        diagnostic: None,
    }
}

fn parse_cue(path: &Path, size: u64, mtime: u64, fallback_title: &str) -> LibraryEntry {
    let Some(bin_path) = primary_bin_from_cue(path) else {
        return LibraryEntry {
            id: fingerprint(&[fallback_title.as_bytes(), b"cue"]),
            path: path.to_path_buf(),
            kind: GameKind::DiscCue,
            title: fallback_title.to_string(),
            region: Region::Unknown,
            size,
            mtime,
            diagnostic: Some("could not resolve a data-track BIN from CUE".into()),
        };
    };

    let mut entry = parse_bin(&bin_path, size, mtime, fallback_title);
    entry.id = fingerprint(&[entry.id.as_bytes(), b"cue"]);
    entry.path = path.to_path_buf();
    entry.kind = GameKind::DiscCue;
    entry.title = fallback_title.to_string();
    entry.size = size;
    entry.mtime = mtime;
    entry.diagnostic = None;
    entry
}

/// Cheap region heuristic -- look for any of the three canonical
/// license strings anywhere in LBA 4's user data. The BIOS checks
/// for these too; if none match, we say `Unknown` rather than
/// guess.
fn region_from_license_text(bytes: &[u8]) -> Region {
    let as_str = String::from_utf8_lossy(bytes);
    if as_str.contains("Sony Computer Entertainment Amer") {
        Region::NtscU
    } else if as_str.contains("Sony Computer Entertainment Euro")
        || as_str.contains("Sony Computer Entertainment Inc. for U.K.")
    {
        Region::Pal
    } else if as_str.contains("Sony Computer Entertainment Inc.") {
        Region::NtscJ
    } else {
        Region::Unknown
    }
}

/// Read the ISO9660 volume identifier out of a Primary Volume
/// Descriptor sector (LBA 16). The spec places it at offset 40 for
/// 32 ASCII bytes, space-padded. We trim trailing whitespace and
/// replace non-printables with `?` so we never return binary
/// garbage as a title.
fn pvd_volume_identifier(user_data: &[u8]) -> Option<String> {
    if user_data.len() < 72 {
        return None;
    }
    // Must be a Primary Volume Descriptor: type=1, magic "CD001",
    // version=1. Otherwise this isn't a valid ISO9660 PVD sector.
    if user_data[0] != 1 || &user_data[1..6] != b"CD001" || user_data[6] != 1 {
        return None;
    }
    let raw = &user_data[40..72];
    let cleaned: String = raw
        .iter()
        .map(|&b| {
            if (0x20..=0x7E).contains(&b) {
                b as char
            } else {
                ' '
            }
        })
        .collect();
    Some(cleaned.trim().to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CueTrackSpec {
    number: u8,
    track_type: psx_iso::TrackType,
    path: PathBuf,
    /// Pregap sectors in disc space before INDEX 01.
    pregap: u32,
    /// Pregap sectors physically present at the start of the track file.
    file_pregap: u32,
}

fn starts_with_keyword(line: &str, keyword: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= keyword.len() && bytes[..keyword.len()].eq_ignore_ascii_case(keyword.as_bytes())
}

fn parse_cue_filename(rest: &str) -> Option<&str> {
    if let Some(rest) = rest.strip_prefix('"') {
        let end = rest.find('"')?;
        Some(&rest[..end])
    } else {
        rest.split_whitespace().next()
    }
}

fn parse_cue_msf(s: &str) -> u32 {
    let mut parts = s.split(':');
    let m = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);
    let s = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);
    let f = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);
    if parts.next().is_some() {
        return 0;
    }
    m * 60 * 75 + s * 75 + f
}

fn detect_track1_embedded_pregap(bytes: &[u8]) -> u32 {
    if bytes.len() < psx_iso::SECTOR_BYTES {
        return 0;
    }
    let sector = &bytes[..psx_iso::SECTOR_BYTES];
    if sector[0] != 0x00 || sector[11] != 0x00 || sector[1..11] != [0xFF; 10] {
        return 0;
    }
    let m = psx_iso::bcd_to_bin(sector[12]);
    let s = psx_iso::bcd_to_bin(sector[13]);
    let f = psx_iso::bcd_to_bin(sector[14]);
    if [m, s, f].contains(&0xFF) {
        return 0;
    }
    let abs_frame = (m as u32) * 60 * 75 + (s as u32) * 75 + (f as u32);
    150u32.saturating_sub(abs_frame)
}

fn parse_cue_tracks(cue_path: &Path) -> Result<Vec<CueTrackSpec>, String> {
    let contents =
        fs::read_to_string(cue_path).map_err(|e| format!("{}: {e}", cue_path.display()))?;
    let dir = cue_path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", cue_path.display()))?;

    let mut tracks = Vec::new();
    let mut current_file: Option<PathBuf> = None;
    let mut current_track_num: Option<u8> = None;
    let mut current_track_type = psx_iso::TrackType::Data;
    let mut cue_pregap = 0u32;

    for line in contents.lines() {
        let trimmed = line.trim();
        if starts_with_keyword(trimmed, "FILE") {
            let rest = trimmed.get(4..).unwrap_or("").trim_start();
            let Some(filename) = parse_cue_filename(rest) else {
                continue;
            };
            current_file = Some(dir.join(filename));
        } else if starts_with_keyword(trimmed, "TRACK") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                current_track_num = parts[1].parse().ok();
                current_track_type = if parts[2].eq_ignore_ascii_case("AUDIO") {
                    psx_iso::TrackType::Audio
                } else {
                    psx_iso::TrackType::Data
                };
                cue_pregap = 0;
            }
        } else if starts_with_keyword(trimmed, "PREGAP") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(msf) = parts.get(1) {
                cue_pregap = parse_cue_msf(msf);
            }
        } else if starts_with_keyword(trimmed, "INDEX 01") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let file_pregap = parts.get(2).map(|msf| parse_cue_msf(msf)).unwrap_or(0);
            let Some(path) = current_file.clone() else {
                continue;
            };
            let Some(number) = current_track_num else {
                continue;
            };
            tracks.push(CueTrackSpec {
                number,
                track_type: current_track_type,
                path,
                pregap: cue_pregap.saturating_add(file_pregap),
                file_pregap,
            });
            cue_pregap = 0;
        }
    }

    if tracks.is_empty() {
        Err(format!(
            "{} contains no INDEX 01 tracks",
            cue_path.display()
        ))
    } else {
        Ok(tracks)
    }
}

/// Load a full multitrack disc model from a CUE sheet. Track timing
/// comes from the CUE; per-track bytes come from the referenced files.
pub fn load_disc_from_cue(cue_path: &Path) -> Result<psx_iso::Disc, String> {
    let specs = parse_cue_tracks(cue_path)?;
    let mut tracks = Vec::with_capacity(specs.len());

    for spec in specs {
        let bytes = fs::read(&spec.path).map_err(|e| format!("{}: {e}", spec.path.display()))?;
        let file_sectors = bytes.len() / psx_iso::SECTOR_BYTES;
        if file_sectors == 0 {
            return Err(format!(
                "{} is too small to contain a raw PS1 sector",
                spec.path.display()
            ));
        }

        let mut file_pregap = spec.file_pregap;
        let mut pregap = spec.pregap;
        if spec.number == 1 && file_pregap == 0 {
            file_pregap = detect_track1_embedded_pregap(&bytes);
            pregap = pregap.max(file_pregap);
        }
        let sector_count = file_sectors.saturating_sub(file_pregap as usize) as u32;
        let start_lba = tracks
            .last()
            .map(|prev: &psx_iso::Track| prev.start_lba + prev.sector_count + pregap)
            .unwrap_or(0);
        tracks.push(psx_iso::Track {
            number: spec.number,
            track_type: spec.track_type,
            start_lba,
            sector_count,
            pregap,
            file_pregap,
            bytes,
        });
    }

    Ok(psx_iso::Disc::from_tracks(tracks))
}

/// Parse a CUE sheet to find the path of its first data track's BIN.
/// Used to collapse CUE + BIN pairs in the UI and to inherit region
/// metadata from the bootable track during library scans.
pub fn primary_bin_from_cue(cue_path: &Path) -> Option<PathBuf> {
    let tracks = parse_cue_tracks(cue_path).ok()?;
    tracks
        .into_iter()
        .find(|track| track.track_type == psx_iso::TrackType::Data)
        .map(|track| track.path)
}

/// FNV-1a-64 over any number of input slices, rendered as a
/// 16-hex-char string. Same algorithm the parity-cache uses -- no
/// adversarial input, just a stable fingerprint.
fn fingerprint(parts: &[&[u8]]) -> String {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    format!("{h:016x}")
}

/// Scratch helper used in tests: build a synthetic 2352-byte
/// sector with the given user-data payload at the standard
/// Mode-2-Form-1 offset. Real BIN parsing uses the same layout.
#[cfg(test)]
fn synth_sector(user_data: &[u8]) -> Vec<u8> {
    use psx_iso::{SECTOR_BYTES, SECTOR_USER_DATA_BYTES, SECTOR_USER_DATA_OFFSET};
    let mut out = vec![0u8; SECTOR_BYTES];
    let n = user_data.len().min(SECTOR_USER_DATA_BYTES);
    out[SECTOR_USER_DATA_OFFSET..SECTOR_USER_DATA_OFFSET + n].copy_from_slice(&user_data[..n]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_test_disc(path: &Path, title: &str, region_msg: &[u8]) {
        let mut bin = vec![0u8; psx_iso::SECTOR_BYTES * 20];

        let mut license = [0u8; psx_iso::SECTOR_USER_DATA_BYTES];
        license[..region_msg.len()].copy_from_slice(region_msg);
        let sec4 = synth_sector(&license);
        let off4 = 4 * psx_iso::SECTOR_BYTES;
        bin[off4..off4 + psx_iso::SECTOR_BYTES].copy_from_slice(&sec4);

        let mut pvd = [0u8; psx_iso::SECTOR_USER_DATA_BYTES];
        pvd[0] = 1;
        pvd[1..6].copy_from_slice(b"CD001");
        pvd[6] = 1;
        pvd[8..19].copy_from_slice(b"PLAYSTATION");
        let title_bytes = title.as_bytes();
        pvd[40..40 + title_bytes.len().min(32)]
            .copy_from_slice(&title_bytes[..title_bytes.len().min(32)]);
        let sec16 = synth_sector(&pvd);
        let off16 = 16 * psx_iso::SECTOR_BYTES;
        bin[off16..off16 + psx_iso::SECTOR_BYTES].copy_from_slice(&sec16);

        std::fs::write(path, &bin).unwrap();
    }

    #[test]
    fn round_trip_empty_library() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("library.ron");
        let lib = Library::default();
        lib.save(&path).unwrap();
        let loaded = Library::load(&path).unwrap();
        assert_eq!(loaded.entries.len(), 0);
    }

    #[test]
    fn missing_file_loads_as_empty() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("no.ron");
        let lib = Library::load_or_empty(&missing);
        assert_eq!(lib.entries.len(), 0);
    }

    #[test]
    fn corrupt_cache_falls_back_to_empty_via_load_or_empty() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad.ron");
        std::fs::write(&path, "{not ron").unwrap();
        let lib = Library::load_or_empty(&path);
        assert_eq!(lib.entries.len(), 0);
    }

    #[test]
    fn classify_matches_known_extensions() {
        assert_eq!(classify(Path::new("g.bin")), Some(GameKind::DiscBin));
        assert_eq!(classify(Path::new("G.BIN")), Some(GameKind::DiscBin));
        assert_eq!(classify(Path::new("g.iso")), Some(GameKind::DiscIso));
        assert_eq!(classify(Path::new("g.cue")), Some(GameKind::DiscCue));
        assert_eq!(classify(Path::new("hello.exe")), Some(GameKind::Exe));
        assert_eq!(classify(Path::new("notes.txt")), None);
        assert_eq!(classify(Path::new("NOEXT")), None);
    }

    #[test]
    fn region_from_license_text_recognises_sce_variants() {
        let us = b"   Licensed  by   Sony Computer Entertainment America ";
        let eu = b"Licensed by Sony Computer Entertainment Europe";
        let jp = b"Licensed by Sony Computer Entertainment Inc.";
        let unknown = b"Some other text";
        assert_eq!(region_from_license_text(us), Region::NtscU);
        assert_eq!(region_from_license_text(eu), Region::Pal);
        assert_eq!(region_from_license_text(jp), Region::NtscJ);
        assert_eq!(region_from_license_text(unknown), Region::Unknown);
    }

    #[test]
    fn pvd_volume_identifier_trims_padding() {
        // Build a synthetic PVD: type=1, "CD001", ver=1, then 32
        // bytes of system identifier followed by the 32-byte volume
        // identifier at offset 40.
        let mut pvd = vec![0u8; 2048];
        pvd[0] = 1;
        pvd[1..6].copy_from_slice(b"CD001");
        pvd[6] = 1;
        pvd[8..19].copy_from_slice(b"PLAYSTATION");
        pvd[40..72].copy_from_slice(b"CRASH_BANDICOOT                 ");
        assert_eq!(
            pvd_volume_identifier(&pvd),
            Some("CRASH_BANDICOOT".to_string())
        );
    }

    #[test]
    fn pvd_volume_identifier_rejects_non_pvd_sector() {
        // Zero sector isn't a PVD -- type byte is 0.
        let zeros = vec![0u8; 2048];
        assert_eq!(pvd_volume_identifier(&zeros), None);
    }

    #[test]
    fn scan_finds_exe_files_and_reuses_cache_on_mtime_match() {
        let tmp = TempDir::new().unwrap();
        // Create two fake EXEs.
        std::fs::write(tmp.path().join("a.exe"), b"fake exe 1").unwrap();
        std::fs::write(tmp.path().join("b.exe"), b"fake exe 2").unwrap();
        let mut lib = Library::default();
        let changed = lib.scan(tmp.path()).unwrap();
        assert_eq!(changed, 2);
        assert_eq!(lib.entries.len(), 2);

        // Second scan -- no files changed, so nothing should be
        // re-parsed.
        let changed2 = lib.scan(tmp.path()).unwrap();
        assert_eq!(changed2, 0);
        assert_eq!(lib.entries.len(), 2);
    }

    #[test]
    fn scan_prunes_entries_whose_files_are_gone() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a.exe");
        let b = tmp.path().join("b.exe");
        std::fs::write(&a, b"1").unwrap();
        std::fs::write(&b, b"2").unwrap();
        let mut lib = Library::default();
        lib.scan(tmp.path()).unwrap();
        assert_eq!(lib.entries.len(), 2);
        std::fs::remove_file(&b).unwrap();
        lib.scan(tmp.path()).unwrap();
        assert_eq!(lib.entries.len(), 1);
        assert_eq!(lib.entries[0].path, a);
    }

    #[test]
    fn scan_skips_cargo_build_artifact_dirs() {
        // Regression: when the library scanner is pointed at an SDK
        // build-output tree (`build/examples/mipsel-sony-psx/release/`),
        // cargo's `deps/` subdirectory contains intermediate
        // `<crate>-<hash>.exe` artifacts. Those used to surface as
        // separate library entries -- the user saw both
        // `hello-tri` and `hello_tri-<hash>` in the Examples column.
        //
        // Fix is in `walk()`: skip `deps/`, `incremental/`, `build/`,
        // and any hidden dir. This test nails down the invariant.
        let tmp = TempDir::new().unwrap();
        // Main release output -- the file the user should see.
        std::fs::write(tmp.path().join("hello-tri.exe"), b"final").unwrap();
        // Cargo's intermediate layout next to it.
        let deps = tmp.path().join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::write(deps.join("hello_tri-0123456789abcdef.exe"), b"dep").unwrap();
        std::fs::write(deps.join("hello_tri-0123456789abcdef.d"), b"").unwrap();
        let incr = tmp.path().join("incremental");
        std::fs::create_dir_all(&incr).unwrap();
        std::fs::write(incr.join("hello_tri-abcd.exe"), b"inc").unwrap();
        let hidden = tmp.path().join(".fingerprint");
        std::fs::create_dir_all(&hidden).unwrap();
        std::fs::write(hidden.join("hello_tri-xxxx.exe"), b"fp").unwrap();
        let builddir = tmp.path().join("build");
        std::fs::create_dir_all(&builddir).unwrap();
        std::fs::write(builddir.join("some-artifact.exe"), b"b").unwrap();

        let mut lib = Library::default();
        lib.scan(tmp.path()).unwrap();
        assert_eq!(
            lib.entries.len(),
            1,
            "only the top-level hello-tri.exe should be surfaced; \
             got entries: {:?}",
            lib.entries.iter().map(|e| &e.path).collect::<Vec<_>>(),
        );
        assert!(lib.entries[0].path.file_name().and_then(|n| n.to_str()) == Some("hello-tri.exe"),);
    }

    #[test]
    fn scan_rejects_non_directory_root() {
        let tmp = TempDir::new().unwrap();
        let not_a_dir = tmp.path().join("file.txt");
        std::fs::write(&not_a_dir, b"").unwrap();
        let mut lib = Library::default();
        assert!(lib.scan(&not_a_dir).is_err());
    }

    #[test]
    fn parse_bin_extracts_region_and_title_from_synthetic_disc() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.bin");
        write_test_disc(
            &path,
            "TEST_TITLE",
            b"    Licensed  by   Sony Computer Entertainment America",
        );
        let e = parse_entry(
            &path,
            GameKind::DiscBin,
            (psx_iso::SECTOR_BYTES * 20) as u64,
            0,
        );
        assert_eq!(e.title, "TEST_TITLE");
        assert_eq!(e.region, Region::NtscU);
        assert_eq!(e.kind, GameKind::DiscBin);
        assert_eq!(e.id.len(), 16);
    }

    #[test]
    fn fingerprint_is_stable() {
        assert_eq!(fingerprint(&[b"hello"]), fingerprint(&[b"hello"]));
        assert_ne!(fingerprint(&[b"hello"]), fingerprint(&[b"world"]));
    }

    #[test]
    fn primary_bin_from_cue_extracts_first_file_line() {
        let tmp = TempDir::new().unwrap();
        let cue_path = tmp.path().join("game.cue");
        // Realistic multi-track PSX CUE -- the data track is track 1.
        std::fs::write(
            &cue_path,
            concat!(
                "FILE \"game (Track 01).bin\" BINARY\n",
                "  TRACK 01 MODE2/2352\n",
                "    INDEX 01 00:00:00\n",
                "FILE \"game (Track 02).bin\" BINARY\n",
                "  TRACK 02 AUDIO\n",
                "    INDEX 00 00:00:00\n",
            ),
        )
        .unwrap();
        let bin = primary_bin_from_cue(&cue_path).unwrap();
        assert_eq!(bin, tmp.path().join("game (Track 01).bin"));
    }

    #[test]
    fn primary_bin_from_cue_handles_lowercase_keyword() {
        let tmp = TempDir::new().unwrap();
        let cue_path = tmp.path().join("g.cue");
        std::fs::write(
            &cue_path,
            "file \"g.bin\" BINARY\n  track 01 mode2/2352\n    index 01 00:00:00\n",
        )
        .unwrap();
        assert_eq!(
            primary_bin_from_cue(&cue_path).unwrap(),
            tmp.path().join("g.bin")
        );
    }

    #[test]
    fn primary_bin_from_cue_returns_none_on_garbage() {
        let tmp = TempDir::new().unwrap();
        let cue_path = tmp.path().join("bad.cue");
        std::fs::write(&cue_path, "REM some comment with no FILE\n").unwrap();
        assert!(primary_bin_from_cue(&cue_path).is_none());
    }

    #[test]
    fn parse_entry_disc_cue_uses_data_track_metadata() {
        let tmp = TempDir::new().unwrap();
        let cue_path = tmp.path().join("Crash Test.cue");
        let track1_path = tmp.path().join("track1.bin");
        write_test_disc(
            &track1_path,
            "CRASH_TEST_DISC",
            b"    Licensed  by   Sony Computer Entertainment Europe",
        );
        std::fs::write(
            &cue_path,
            concat!(
                "FILE \"track1.bin\" BINARY\n",
                "  TRACK 01 MODE2/2352\n",
                "    INDEX 01 00:00:00\n",
            ),
        )
        .unwrap();

        let entry = parse_entry(&cue_path, GameKind::DiscCue, 1234, 5678);
        assert_eq!(entry.kind, GameKind::DiscCue);
        assert_eq!(entry.title, "Crash Test");
        assert_eq!(entry.region, Region::Pal);
        assert_eq!(entry.path, cue_path);
        assert_eq!(entry.size, 1234);
        assert_eq!(entry.mtime, 5678);
        assert_eq!(entry.diagnostic, None);
    }

    #[test]
    fn load_disc_from_cue_positions_later_track_after_pregap() {
        let tmp = TempDir::new().unwrap();
        let cue_path = tmp.path().join("disc.cue");
        let track1_path = tmp.path().join("track1.bin");
        let track2_path = tmp.path().join("track2.bin");
        let mut track1 = vec![0u8; psx_iso::SECTOR_BYTES * 10];
        track1[12] = 0x00;
        track1[13] = 0x02;
        track1[14] = 0x00;
        std::fs::write(&track1_path, track1).unwrap();
        let mut track2 = vec![0u8; psx_iso::SECTOR_BYTES * 4];
        track2[0] = 0xAB;
        std::fs::write(&track2_path, track2).unwrap();
        std::fs::write(
            &cue_path,
            concat!(
                "FILE \"track1.bin\" BINARY\n",
                "  TRACK 01 MODE2/2352\n",
                "    INDEX 01 00:00:00\n",
                "FILE \"track2.bin\" BINARY\n",
                "  TRACK 02 AUDIO\n",
                "    PREGAP 00:00:02\n",
                "    INDEX 01 00:00:00\n",
            ),
        )
        .unwrap();

        let disc = load_disc_from_cue(&cue_path).unwrap();
        let pos = disc.track_position_for_lba(10).unwrap();
        assert_eq!(pos.track_number, 2);
        assert_eq!(pos.index_number, 0);
        assert_eq!(pos.relative_msf, (0, 0, 1));
        assert!(disc.read_sector_raw(10).is_none());
        assert_eq!(disc.read_sector_raw(12).unwrap()[0], 0xAB);
    }

    #[test]
    fn scan_walks_recursively_into_subdirs() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("game.exe"), b"").unwrap();
        let mut lib = Library::default();
        lib.scan(tmp.path()).unwrap();
        assert_eq!(lib.entries.len(), 1);
        assert!(lib.entries[0].path.ends_with("sub/game.exe"));
    }
}
