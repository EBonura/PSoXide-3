//! Query the oracle for what `PCSX.GPU.takeScreenShot()` returns.
//! First-run discovery to pick the VRAM dump path.

use parity_oracle::{OracleConfig, ReduxProcess};
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path, lua).expect("Redux binary resolves");

    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(Duration::from_secs(10)).expect("handshake");

    // Step a few million instructions so the GPU has actually
    // drawn something and `takeScreenShot` returns a non-empty
    // snapshot.
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    eprintln!("[screenshot_probe] running {n} steps in Redux...");
    let timeout = Duration::from_secs((n / 500_000).max(30));
    let _ = redux.run(n, timeout).expect("run");

    redux.send_command("screenshot_probe").expect("send");
    let line = redux
        .wait_for_response(Duration::from_secs(10))
        .expect("response");
    println!("{line}");

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}
