//! Show the GPR delta between two cache records to see exactly what
//! state changed across an unexplained tick gap.
//!
//! ```bash
//! cargo run -p emulator-core --example cache_diff --release -- 19472416 19472418
//! ```

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

const HEADER_BYTES: u64 = 8 + 4 + 4 + 8;
const RECORD_BYTES: u64 = 144;

fn read_record(r: &mut BufReader<File>, idx: u64) -> (u64, u32, u32, [u32; 32]) {
    r.seek(SeekFrom::Start(HEADER_BYTES + idx * RECORD_BYTES)).unwrap();
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
    (tick, pc, instr, gprs)
}

fn main() {
    let a: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).expect("need step A");
    let b: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).expect("need step B");
    let path = PathBuf::from(
        "/Users/ebonura/Desktop/repos/psoxide/emu/crates/emulator-core/target/parity-cache/redux-32b1a0fa4db70c8f-50000000.bin",
    );
    let mut r = BufReader::new(File::open(&path).expect("open cache"));
    let (ta, pca, ia, ga) = read_record(&mut r, a);
    let (tb, pcb, ib, gb) = read_record(&mut r, b);
    println!(
        "step {a} tick={ta} pc=0x{pca:08x} instr=0x{ia:08x}\n\
         step {b} tick={tb} pc=0x{pcb:08x} instr=0x{ib:08x}\n\
         tick delta = {}",
        tb - ta
    );
    println!("GPR diffs:");
    for i in 0..32 {
        if ga[i] != gb[i] {
            println!("  $r{i:<2}: 0x{:08x} -> 0x{:08x}", ga[i], gb[i]);
        }
    }
}
