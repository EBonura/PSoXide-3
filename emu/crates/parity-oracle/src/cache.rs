//! On-disk cache for Redux traces.
//!
//! Running PCSX-Redux through the Lua oracle costs ~25 seconds of
//! wall time per million steps — a 50M parity probe takes half an
//! hour. Our emulator runs the same 50M in a couple of seconds.
//!
//! To keep divergence-hunting fast, we cache the Redux trace on
//! disk after the first run. All subsequent probes load the cache
//! (fraction of a second), re-run *only* our emulator, and compare
//! step-by-step. Redux is only re-invoked when the step count grows
//! or the BIOS image changes.
//!
//! The file format is intentionally simple — raw little-endian
//! `InstructionRecord`s with a header. No compression yet; a 50M
//! step cache is ~7 GiB. When this starts hurting we swap in zstd
//! streaming; the header carries a version so older caches get
//! invalidated automatically.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use std::ffi::OsStr;

use psx_trace::InstructionRecord;

/// Magic bytes at the start of every cache file.
const MAGIC: &[u8; 8] = b"PSXTRACE";
/// Current cache-file format version. Bump whenever the on-disk
/// layout changes in an incompatible way.
const VERSION: u32 = 1;
/// Bytes per record: `tick(8) + pc(4) + instr(4) + gprs(32*4) = 144`.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 * 4;

/// Cache location — environment-overridable, so CI machines can
/// stash caches under a different volume. Defaults to
/// `$CARGO_MANIFEST_DIR/../../../target/parity-cache` which keeps
/// caches with the build artifacts (gitignored).
pub fn default_dir() -> PathBuf {
    if let Ok(over) = std::env::var("PSOXIDE_PARITY_CACHE_DIR") {
        return PathBuf::from(over);
    }
    // Tests run with CWD inside `emu/` — resolve the cache under the
    // workspace's `target` so `cargo clean` wipes it consistently.
    let base = std::env::var("CARGO_TARGET_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    base.join("parity-cache")
}

/// Cache key: (BIOS digest, step count). A fresh BIOS image or a
/// different step count produces a different path. The digest is a
/// cheap xxHash-style fold over the BIOS bytes — collisions aren't
/// a security concern here, only data integrity.
pub fn path_for(dir: &Path, bios_bytes: &[u8], steps: usize) -> PathBuf {
    let hash = fold_hash(bios_bytes);
    dir.join(format!("redux-{hash:016x}-{steps}.bin"))
}

/// Prefix-aware lookup. Searches `dir` for any cache file matching
/// the same BIOS hash with a step count ≥ `min_steps`, and returns
/// the first `min_steps` records from the longest one found. This
/// means a single 50M-step Redux run satisfies 10M, 20M, 30M — any
/// shorter probe loads instantly.
pub fn load_prefix(
    dir: &Path,
    bios_bytes: &[u8],
    min_steps: usize,
) -> Option<Vec<InstructionRecord>> {
    let hash = fold_hash(bios_bytes);
    let prefix = format!("redux-{hash:016x}-");

    // Walk dir looking for matching files. Pick the largest step
    // count ≥ `min_steps` so we don't truncate short.
    let mut best: Option<(usize, PathBuf)> = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if !name.starts_with(&prefix) || !name.ends_with(".bin") {
            continue;
        }
        let step_str = &name[prefix.len()..name.len() - 4]; // strip prefix + ".bin"
        let Ok(steps) = step_str.parse::<usize>() else {
            continue;
        };
        if steps >= min_steps
            && best
                .as_ref()
                .map(|(best_steps, _)| steps > *best_steps)
                .unwrap_or(true)
        {
            best = Some((steps, path));
        }
    }

    let (cached_steps, path) = best?;
    let mut records = load(&path)?;
    if records.len() > min_steps {
        records.truncate(min_steps);
        eprintln!(
            "[parity-cache] prefix hit {} ({} records, truncated from {})",
            path.display(),
            min_steps,
            cached_steps
        );
    } else {
        eprintln!(
            "[parity-cache] prefix hit {} ({} records)",
            path.display(),
            records.len()
        );
    }
    Some(records)
}

/// Load a cached trace if one exists at `path` and passes a basic
/// header validation. Returns `None` on any failure (caller should
/// fall back to invoking Redux). Corrupt/incompatible caches are
/// logged and skipped, never fatal.
pub fn load(path: &Path) -> Option<Vec<InstructionRecord>> {
    let file = File::open(path).ok()?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }

    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf).ok()?;
    if u32::from_le_bytes(u32buf) != VERSION {
        return None;
    }
    r.read_exact(&mut u32buf).ok()?; // reserved

    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u64buf).ok()?;
    let count = u64::from_le_bytes(u64buf) as usize;

    let mut records = Vec::with_capacity(count);
    let mut rec_buf = [0u8; RECORD_BYTES];
    for _ in 0..count {
        r.read_exact(&mut rec_buf).ok()?;
        records.push(decode_record(&rec_buf));
    }
    Some(records)
}

/// Write `records` to `path`, creating parent directories as needed.
/// Atomically renames a temp file on success so a SIGINT mid-write
/// doesn't leave a corrupt cache behind.
pub fn save(path: &Path, records: &[InstructionRecord]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!(
        "tmp.{}",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    {
        let file = File::create(&tmp)?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?; // reserved
        w.write_all(&(records.len() as u64).to_le_bytes())?;
        let mut rec_buf = [0u8; RECORD_BYTES];
        for r in records {
            encode_record(r, &mut rec_buf);
            w.write_all(&rec_buf)?;
        }
        w.flush()?;
    }

    fs::rename(&tmp, path)?;
    Ok(())
}

fn encode_record(r: &InstructionRecord, buf: &mut [u8; RECORD_BYTES]) {
    buf[0..8].copy_from_slice(&r.tick.to_le_bytes());
    buf[8..12].copy_from_slice(&r.pc.to_le_bytes());
    buf[12..16].copy_from_slice(&r.instr.to_le_bytes());
    for (i, g) in r.gprs.iter().enumerate() {
        let off = 16 + i * 4;
        buf[off..off + 4].copy_from_slice(&g.to_le_bytes());
    }
}

fn decode_record(buf: &[u8; RECORD_BYTES]) -> InstructionRecord {
    let tick = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let pc = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let instr = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let mut gprs = [0u32; 32];
    for i in 0..32 {
        let off = 16 + i * 4;
        gprs[i] = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    }
    InstructionRecord {
        tick,
        pc,
        instr,
        gprs,
    }
}

/// Cheap non-cryptographic hash: FNV-1a 64. Good enough to
/// disambiguate BIOS images in file names.
fn fold_hash(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01B3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(tick: u64) -> InstructionRecord {
        let mut gprs = [0u32; 32];
        gprs[3] = tick as u32;
        InstructionRecord {
            tick,
            pc: 0xBFC0_0000 + (tick as u32) * 4,
            instr: 0xDEAD_BEEF,
            gprs,
        }
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.bin");
        let records: Vec<_> = (0..1234).map(sample_record).collect();

        save(&path, &records).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded, records);
    }

    #[test]
    fn load_returns_none_for_missing() {
        assert!(load(Path::new("/nonexistent/path/trace.bin")).is_none());
    }

    #[test]
    fn load_returns_none_for_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        fs::write(&path, b"NOTATRACE").unwrap();
        assert!(load(&path).is_none());
    }

    #[test]
    fn fold_hash_is_stable() {
        assert_eq!(fold_hash(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fold_hash(b"a"), 0xAF63_DC4C_8601_EC8C);
        // Different inputs should hash differently.
        assert_ne!(fold_hash(b"A"), fold_hash(b"B"));
    }

    #[test]
    fn path_for_embeds_hash_and_steps() {
        let p = path_for(Path::new("/tmp/cache"), b"bios-bytes", 100);
        let s = p.to_string_lossy();
        assert!(s.contains("redux-"));
        assert!(s.ends_with("-100.bin"));
    }

    #[test]
    fn load_prefix_truncates_longer_cache() {
        let dir = tempfile::tempdir().unwrap();
        let bios = b"fake-bios";
        // Save a 100-record cache.
        let records: Vec<_> = (0..100).map(sample_record).collect();
        let path = path_for(dir.path(), bios, 100);
        save(&path, &records).unwrap();

        // Request 40: we get the first 40 back.
        let loaded = load_prefix(dir.path(), bios, 40).unwrap();
        assert_eq!(loaded.len(), 40);
        assert_eq!(loaded, records[..40]);
    }

    #[test]
    fn load_prefix_picks_longest_available() {
        let dir = tempfile::tempdir().unwrap();
        let bios = b"fake-bios";
        save(
            &path_for(dir.path(), bios, 10),
            &(0..10).map(sample_record).collect::<Vec<_>>(),
        )
        .unwrap();
        save(
            &path_for(dir.path(), bios, 100),
            &(0..100).map(sample_record).collect::<Vec<_>>(),
        )
        .unwrap();

        // Ask for 50 — should pick the 100-record cache, not the 10.
        let loaded = load_prefix(dir.path(), bios, 50).unwrap();
        assert_eq!(loaded.len(), 50);
    }

    #[test]
    fn load_prefix_returns_none_when_all_shorter() {
        let dir = tempfile::tempdir().unwrap();
        let bios = b"fake-bios";
        save(
            &path_for(dir.path(), bios, 10),
            &(0..10).map(sample_record).collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(load_prefix(dir.path(), bios, 100).is_none());
    }
}
