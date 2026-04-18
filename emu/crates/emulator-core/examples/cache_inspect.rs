//! Read Redux InstructionRecords from the parity cache around a target
//! step so we can see Redux's PC, instr, GPRs, and cycle count side-by-
//! side with our own state.
//!
//! ```bash
//! cargo run -p emulator-core --example cache_inspect --release -- 19472417 8
//! ```
//! Args: TARGET_STEP [WINDOW] — print [target-WINDOW, target+WINDOW].

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

const MAGIC: &[u8; 8] = b"PSXTRACE";
const HEADER_BYTES: u64 = 8 + 4 + 4 + 8; // magic + version + reserved + count
const RECORD_BYTES: u64 = 8 + 4 + 4 + 32 * 4; // 144

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_472_417);
    let window: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let path = PathBuf::from(
        "/Users/ebonura/Desktop/repos/PSoXide/emu/crates/emulator-core/target/parity-cache/redux-32b1a0fa4db70c8f-50000000.bin",
    );
    let file = File::open(&path).expect("open cache");
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC);

    let start = target.saturating_sub(window);
    let end = target + window;

    for idx in start..=end {
        let off = HEADER_BYTES + idx * RECORD_BYTES;
        r.seek(SeekFrom::Start(off)).unwrap();
        let mut buf = [0u8; 144];
        r.read_exact(&mut buf).unwrap();
        let tick = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let pc = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let instr = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let mut gprs = [0u32; 32];
        for i in 0..32 {
            let o = 16 + i * 4;
            gprs[i] = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        }
        let marker = if idx == target { " <==" } else { "" };
        println!(
            "step {idx:>10}  tick={tick:>10}  pc=0x{pc:08x}  instr=0x{instr:08x}  k0=0x{:08x}  k1=0x{:08x}  v0=0x{:08x}{marker}",
            gprs[26], gprs[27], gprs[2]
        );
    }
}
