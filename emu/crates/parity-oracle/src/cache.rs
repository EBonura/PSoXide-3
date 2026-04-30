//! On-disk cache for Redux traces.
//!
//! Running PCSX-Redux through the Lua oracle costs ~25 seconds of
//! wall time per million steps -- a 50M parity probe takes half an
//! hour. Our emulator runs the same 50M in a couple of seconds.
//!
//! To keep divergence-hunting fast, we cache the Redux trace on
//! disk after the first run. All subsequent probes load the cache
//! (fraction of a second), re-run *only* our emulator, and compare
//! step-by-step. Redux is only re-invoked when the step count grows
//! or the BIOS image changes.
//!
//! The file format is intentionally simple -- raw little-endian
//! `InstructionRecord`s with a header. No compression yet; a 50M
//! step cache is ~20 GiB at v2 (was ~7 GiB at v1, before COP2
//! capture). When this starts hurting we swap in zstd streaming;
//! the header carries a version so older caches get invalidated
//! automatically.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use std::ffi::OsStr;

use psx_trace::InstructionRecord;

/// Magic bytes at the start of every cache file.
const MAGIC: &[u8; 8] = b"PSXTRACE";
/// Current cache-file format version. Bump whenever the on-disk
/// layout changes in an incompatible way. v2 added the GTE register
/// snapshot (`cop2_data` + `cop2_ctl`) -- 256 bytes per record.
const VERSION: u32 = 2;
/// Bytes per record:
///   `tick(8) + pc(4) + instr(4) + gprs(32*4) + cop2_data(32*4) + cop2_ctl(32*4) = 400`.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 * 4 + 32 * 4 + 32 * 4;

/// Cache location -- environment-overridable, so CI machines can
/// stash caches under a different volume. Defaults to
/// `$CARGO_MANIFEST_DIR/../../../target/parity-cache` which keeps
/// caches with the build artifacts (gitignored).
pub fn default_dir() -> PathBuf {
    if let Ok(over) = std::env::var("PSOXIDE_PARITY_CACHE_DIR") {
        return PathBuf::from(over);
    }
    // Tests run with CWD inside `emu/` -- resolve the cache under the
    // workspace's `target` so `cargo clean` wipes it consistently.
    let base = std::env::var("CARGO_TARGET_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    base.join("parity-cache")
}

/// Cache key: (BIOS digest, step count). A fresh BIOS image or a
/// different step count produces a different path. The digest is a
/// cheap xxHash-style fold over the BIOS bytes -- collisions aren't
/// a security concern here, only data integrity.
pub fn path_for(dir: &Path, bios_bytes: &[u8], steps: usize) -> PathBuf {
    let hash = fold_hash(bios_bytes);
    dir.join(format!("redux-{hash:016x}-{steps}.bin"))
}

/// Prefix-aware lookup. Searches `dir` for any cache file matching
/// the same BIOS hash with a step count ≥ `min_steps`, and returns
/// the first `min_steps` records from the longest one found. This
/// means a single 50M-step Redux run satisfies 10M, 20M, 30M -- any
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

/// Load the longest cache file on disk matching this BIOS hash,
/// returning all its records (no truncation). Useful when the caller
/// wants "whatever's available" rather than a specific minimum step
/// count -- e.g. the divergence probe, which walks until it runs out
/// of records regardless of how many there are.
pub fn load_longest(dir: &Path, bios_bytes: &[u8]) -> Option<Vec<InstructionRecord>> {
    let hash = fold_hash(bios_bytes);
    let prefix = format!("redux-{hash:016x}-");

    let mut best: Option<(usize, PathBuf)> = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if !name.starts_with(&prefix) || !name.ends_with(".bin") {
            continue;
        }
        let step_str = &name[prefix.len()..name.len() - 4];
        let Ok(steps) = step_str.parse::<usize>() else {
            continue;
        };
        if best
            .as_ref()
            .map(|(best_steps, _)| steps > *best_steps)
            .unwrap_or(true)
        {
            best = Some((steps, path));
        }
    }

    let (_cached_steps, path) = best?;
    let records = load(&path)?;
    eprintln!(
        "[parity-cache] loaded longest {} ({} records)",
        path.display(),
        records.len()
    );
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

/// Streaming writer for traces that wouldn't fit in memory as a single
/// `Vec<InstructionRecord>` (a 100 M-record trace is ~14 GiB). Writes
/// to a temp file and atomically renames on [`finish`]; dropping
/// without `finish` leaves the temp file behind for post-mortem but
/// never produces a corrupt cache at the target path.
///
/// The file header reserves a `count` field that's written at create
/// time using the caller-supplied expected total; pushing fewer records
/// than declared produces a corrupt file (load will read past EOF), so
/// always write exactly `count`.
pub struct StreamingWriter {
    tmp: PathBuf,
    final_path: PathBuf,
    writer: BufWriter<File>,
    rec_buf: [u8; RECORD_BYTES],
    declared: u64,
    written: u64,
}

impl StreamingWriter {
    /// Create a streaming writer. `count` is the exact number of
    /// records that will be pushed before `finish`; it's written into
    /// the header so loaders don't need to stat the file.
    pub fn create(path: &Path, count: u64) -> io::Result<Self> {
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
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // reserved
        writer.write_all(&count.to_le_bytes())?;
        Ok(Self {
            tmp,
            final_path: path.to_path_buf(),
            writer,
            rec_buf: [0u8; RECORD_BYTES],
            declared: count,
            written: 0,
        })
    }

    /// Append one record. Returns an error if the BufWriter fails to
    /// flush its internal buffer to disk (e.g. ENOSPC).
    pub fn push(&mut self, r: &InstructionRecord) -> io::Result<()> {
        encode_record(r, &mut self.rec_buf);
        self.writer.write_all(&self.rec_buf)?;
        self.written += 1;
        Ok(())
    }

    /// Number of records pushed so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Flush, close, and atomically rename the temp file onto the
    /// target path. Fails if `written != declared` so the header stays
    /// honest.
    pub fn finish(mut self) -> io::Result<()> {
        if self.written != self.declared {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "streaming writer finished early: declared {} records, wrote {}",
                    self.declared, self.written
                ),
            ));
        }
        self.writer.flush()?;
        drop(self.writer);
        fs::rename(&self.tmp, &self.final_path)?;
        Ok(())
    }
}

/// Layout (all little-endian):
///   `[0..8) tick | [8..12) pc | [12..16) instr | [16..144) gprs |`
///   `[144..272) cop2_data | [272..400) cop2_ctl`.
const GPRS_OFF: usize = 16;
const COP2_DATA_OFF: usize = GPRS_OFF + 32 * 4;
const COP2_CTL_OFF: usize = COP2_DATA_OFF + 32 * 4;

fn encode_record(r: &InstructionRecord, buf: &mut [u8; RECORD_BYTES]) {
    buf[0..8].copy_from_slice(&r.tick.to_le_bytes());
    buf[8..12].copy_from_slice(&r.pc.to_le_bytes());
    buf[12..16].copy_from_slice(&r.instr.to_le_bytes());
    encode_u32_block(&r.gprs, &mut buf[GPRS_OFF..COP2_DATA_OFF]);
    encode_u32_block(&r.cop2_data, &mut buf[COP2_DATA_OFF..COP2_CTL_OFF]);
    encode_u32_block(&r.cop2_ctl, &mut buf[COP2_CTL_OFF..]);
}

fn decode_record(buf: &[u8; RECORD_BYTES]) -> InstructionRecord {
    let tick = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let pc = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let instr = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let gprs = decode_u32_block(&buf[GPRS_OFF..COP2_DATA_OFF]);
    let cop2_data = decode_u32_block(&buf[COP2_DATA_OFF..COP2_CTL_OFF]);
    let cop2_ctl = decode_u32_block(&buf[COP2_CTL_OFF..]);
    InstructionRecord {
        tick,
        pc,
        instr,
        gprs,
        cop2_data,
        cop2_ctl,
    }
}

fn encode_u32_block(src: &[u32; 32], dst: &mut [u8]) {
    for (i, v) in src.iter().enumerate() {
        let off = i * 4;
        dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn decode_u32_block(src: &[u8]) -> [u32; 32] {
    let mut out = [0u32; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 4;
        *slot = u32::from_le_bytes(src[off..off + 4].try_into().unwrap());
    }
    out
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
        let mut cop2_data = [0u32; 32];
        cop2_data[7] = tick as u32; // OTZ
        cop2_data[24] = (tick as u32).wrapping_mul(7); // MAC0
        let mut cop2_ctl = [0u32; 32];
        cop2_ctl[31] = 0x8000_F000; // FLAG sentinel
        InstructionRecord {
            tick,
            pc: 0xBFC0_0000 + (tick as u32) * 4,
            instr: 0xDEAD_BEEF,
            gprs,
            cop2_data,
            cop2_ctl,
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

        // Ask for 50 -- should pick the 100-record cache, not the 10.
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

    #[test]
    fn streaming_writer_produces_same_bytes_as_save() {
        // The streaming writer must be bit-identical to `save()` so
        // readers can round-trip either. Asserted here by comparing
        // file bytes (magic + version + count + records).
        let dir = tempfile::tempdir().unwrap();
        let records: Vec<_> = (0..500).map(sample_record).collect();

        let save_path = dir.path().join("batched.bin");
        save(&save_path, &records).unwrap();

        let stream_path = dir.path().join("streamed.bin");
        let mut w = StreamingWriter::create(&stream_path, records.len() as u64).unwrap();
        for r in &records {
            w.push(r).unwrap();
        }
        w.finish().unwrap();

        assert_eq!(
            fs::read(&save_path).unwrap(),
            fs::read(&stream_path).unwrap()
        );
        assert_eq!(load(&stream_path).unwrap(), records);
    }

    #[test]
    fn streaming_writer_finish_rejects_short_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.bin");
        let mut w = StreamingWriter::create(&path, 5).unwrap();
        w.push(&sample_record(1)).unwrap();
        let err = w.finish().expect_err("should reject short write");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // Final path must not exist -- only the tmp file.
        assert!(!path.exists());
    }
}
