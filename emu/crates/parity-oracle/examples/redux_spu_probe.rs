//! Snapshot Redux's SPU register state at absolute instruction
//! checkpoints during a disc boot.
//!
//! ```bash
//! cargo run -p parity-oracle --example redux_spu_probe --release -- \
//!     30000000 60000000 100000000
//! ```

use parity_oracle::{OracleConfig, ReduxProcess};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_DISC: &str =
    "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    let mut checkpoints: Vec<u64> = env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    if checkpoints.is_empty() {
        checkpoints = vec![30_000_000, 60_000_000, 100_000_000];
    }
    checkpoints.sort_unstable();

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let bios = env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS));
    let disc = env::var("PSOXIDE_DISC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_DISC));

    let config = OracleConfig::new(bios, lua)
        .expect("Redux binary resolves")
        .with_disc(disc);
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    let mut at = 0u64;
    for cp in checkpoints {
        if cp > at {
            redux
                .send_command(&format!("run {}", cp - at))
                .expect("run cmd");
            let timeout = Duration::from_secs(((cp - at) / 300_000).max(60));
            let _ = redux.wait_for_response(timeout).expect("run reply");
        }
        at = cp;
        dump_snapshot(&mut redux, cp);
    }

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}

fn dump_snapshot(redux: &mut ReduxProcess, step: u64) {
    let spucnt = spu16(redux, 0x1F80_1DAA);
    let main_l = spu16(redux, 0x1F80_1D80);
    let main_r = spu16(redux, 0x1F80_1D82);
    let cd_l = spu16(redux, 0x1F80_1DB0);
    let cd_r = spu16(redux, 0x1F80_1DB2);

    println!("=== redux step {step} ===");
    println!("SPUCNT        = 0x{spucnt:04x}");
    println!("MAIN_VOL_L/R  = 0x{main_l:04x} / 0x{main_r:04x}");
    println!("CD_VOL_L/R    = 0x{cd_l:04x} / 0x{cd_r:04x}");
    println!();
    println!(" v  vol_l  vol_r  pitch  start adsr_l adsr_h  env  repeat");
    for v in 0..24u32 {
        let base = 0x1F80_1C00 + v * 16;
        let vol_l = spu16(redux, base);
        let vol_r = spu16(redux, base + 2);
        let pitch = spu16(redux, base + 4);
        let start = spu16(redux, base + 6);
        let adsr_l = spu16(redux, base + 8);
        let adsr_h = spu16(redux, base + 10);
        let env = spu16(redux, base + 12);
        let repeat = spu16(redux, base + 14);
        if vol_l != 0 || vol_r != 0 || pitch != 0 || env != 0 || start != 0 {
            println!(
                "{v:2}  {vol_l:04x}   {vol_r:04x}   {pitch:04x}   {start:04x}  {adsr_l:04x}   {adsr_h:04x}   {env:04x}  {repeat:04x}"
            );
        }
    }
    println!();
}

fn spu16(redux: &mut ReduxProcess, addr: u32) -> u16 {
    redux
        .send_command(&format!("spu16 0x{addr:08x}"))
        .expect("spu16 cmd");
    let line = redux.wait_for_response(QUERY_TIMEOUT).expect("spu16 reply");
    parse_spu16(&line)
}

fn parse_spu16(line: &str) -> u16 {
    line.strip_prefix("spu16 ")
        .and_then(|s| s.parse::<i64>().ok())
        .map(|v| v as u16)
        .unwrap_or(0)
}
