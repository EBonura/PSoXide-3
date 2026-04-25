//! Launch Redux and dump every key in PCSX.* and regs.*. Used to
//! discover the VRAM / GPU accessor name in this Redux build so
//! `oracle.lua`'s `vram_hash` command can pick the right one.

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

    redux.send_command("introspect").expect("send");
    // Oracle emits one line per dumped namespace.
    for _ in 0..6 {
        match redux.wait_for_response(Duration::from_secs(3)) {
            Ok(line) => println!("{line}"),
            Err(_) => break, // timeout = no more lines
        }
    }

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}
