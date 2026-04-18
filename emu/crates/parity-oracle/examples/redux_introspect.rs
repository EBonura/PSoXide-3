//! One-shot: boot Redux, ask Lua to dump all `PCSX.*` and `regs.*`
//! keys. Used to discover the right API for reading hardware
//! registers (IRQ controller live state).

use parity_oracle::{OracleConfig, ReduxProcess};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let bios = env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let config = OracleConfig::new(bios, lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(Duration::from_secs(10)).expect("handshake");

    redux.send_command("introspect").expect("cmd");
    for _ in 0..2 {
        let line = redux
            .wait_for_response(Duration::from_secs(5))
            .expect("reply");
        println!("{line}");
    }

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();
}
