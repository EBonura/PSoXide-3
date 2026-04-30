//! Read Redux InstructionRecords from the parity cache around a target
//! step so we can see Redux's PC, instr, GPRs, and cycle count side-by-
//! side with our own state.
//!
//! ```bash
//! cargo run -p emulator-core --example cache_inspect --release -- 19472417 8
//! ```
//! Args: TARGET_STEP [WINDOW] -- print [target-WINDOW, target+WINDOW].

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

const MAGIC: &[u8; 8] = b"PSXTRACE";
const HEADER_BYTES: u64 = 8 + 4 + 4 + 8; // magic + version + reserved + count
const RECORD_BYTES_V1: u64 = 8 + 4 + 4 + 32 * 4; // 144
const RECORD_BYTES_V2: u64 = 8 + 4 + 4 + 32 * 4 + 32 * 4 + 32 * 4; // 400

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_472_417);
    let window: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let path = std::env::args()
        .nth(3)
        .map(PathBuf::from)
        .or_else(find_longest_cache)
        .expect("cache path arg or target/parity-cache/redux-*.bin");
    eprintln!("[cache-inspect] {}", path.display());
    let file = File::open(&path).expect("open cache");
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC);
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf).unwrap();
    let version = u32::from_le_bytes(u32buf);
    r.read_exact(&mut u32buf).unwrap(); // reserved
    let record_bytes = match version {
        1 => RECORD_BYTES_V1,
        2 => RECORD_BYTES_V2,
        other => panic!("unsupported cache version {other}"),
    };
    eprintln!("[cache-inspect] version={version} record_bytes={record_bytes}");

    let start = target.saturating_sub(window);
    let end = target + window;

    for idx in start..=end {
        let off = HEADER_BYTES + idx * record_bytes;
        r.seek(SeekFrom::Start(off)).unwrap();
        let mut buf = vec![0u8; record_bytes as usize];
        r.read_exact(&mut buf).unwrap();
        let tick = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let pc = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let instr = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let mut gprs = [0u32; 32];
        for (i, gpr) in gprs.iter_mut().enumerate() {
            let o = 16 + i * 4;
            *gpr = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        }
        let marker = if idx == target { " <==" } else { "" };
        println!(
            "step {idx:>10}  tick={tick:>10}  pc=0x{pc:08x}  instr=0x{instr:08x}  k0=0x{:08x}  k1=0x{:08x}  v0=0x{:08x}{marker}",
            gprs[26], gprs[27], gprs[2]
        );
    }
}

fn find_longest_cache() -> Option<PathBuf> {
    let base = std::env::var("CARGO_TARGET_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let dir = base.join("parity-cache");
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.starts_with("redux-") || !name.ends_with(".bin") {
            continue;
        }
        let Some(step_text) = name
            .strip_suffix(".bin")
            .and_then(|s| s.rsplit_once('-').map(|(_, steps)| steps))
        else {
            continue;
        };
        let Ok(steps) = step_text.parse::<u64>() else {
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
    best.map(|(_, path)| path)
}
