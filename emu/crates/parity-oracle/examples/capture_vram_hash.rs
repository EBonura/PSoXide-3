//! Launch Redux, step N instructions, and print the resulting VRAM
//! FNV-1a-64 hash. Used to capture milestone goldens that verify our
//! emulator renders the same pixels Redux does at a specific
//! instruction count.
//!
//! ```bash
//! cargo run -p parity-oracle --example capture_vram_hash --release -- 100000000
//! ```
//!
//! Optional `PSOXIDE_DISC=/path/to/game.bin` mounts a disc image —
//! but the current oracle.lua doesn't wire disc-load through Redux
//! yet, so this path works only for no-disc milestones (A, B). The
//! disc-mounting extension lands when Milestone D needs Redux
//! parity and we probe how Redux's CLI accepts a CD image.

use parity_oracle::{OracleConfig, ReduxProcess};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const VRAM_HASH_TIMEOUT: Duration = Duration::from_secs(60);

fn main() {
    let n: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: capture_vram_hash <step_count>");

    let bios_path = env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path, lua).expect("Redux binary resolves");

    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    eprintln!("[capture_vram_hash] running {n} instructions in Redux (silent)...");
    // Timeout scales with step count: assume at least 1M instrs/sec
    // on Redux's native interpreter, plus a generous safety margin.
    let run_timeout = Duration::from_secs((n / 500_000).max(60));
    let tick = redux.run(n, run_timeout).expect("run n succeeded");
    eprintln!("[capture_vram_hash] reached tick={tick}");

    let snap = redux
        .display_hash(VRAM_HASH_TIMEOUT)
        .expect("display_hash succeeded");
    println!("steps={n}");
    println!("redux_tick={tick}");
    println!("redux_width={}", snap.width);
    println!("redux_height={}", snap.height);
    println!("redux_bpp={}", snap.bpp);
    println!("redux_byte_len={}", snap.byte_len);
    println!("redux_display_fnv1a_64=0x{:016x}", snap.hash);

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();
}
