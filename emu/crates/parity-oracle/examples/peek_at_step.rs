use parity_oracle::{OracleConfig, ReduxProcess};
use std::path::PathBuf;
use std::time::Duration;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

fn parse_u32(text: &str) -> u32 {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).expect("bad hex addr")
    } else {
        text.parse().expect("bad addr")
    }
}

fn main() {
    let steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: peek_at_step <steps> <addr> [disc]");
    let addr = std::env::args()
        .nth(2)
        .as_deref()
        .map(parse_u32)
        .expect("usage: peek_at_step <steps> <addr> [disc]");
    let disc = std::env::args().nth(3).map(PathBuf::from);

    let bios = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios, lua).expect("Redux binary resolves");
    if let Some(disc) = disc {
        config = config.with_disc(disc);
    }

    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let timeout = Duration::from_secs((steps / 200_000).max(60));
    let tick = redux.run(steps, timeout).expect("run");
    let value = redux.peek32(addr, Duration::from_secs(5)).expect("peek32");
    println!("step={steps} tick={tick} addr=0x{addr:08x} value=0x{value:08x}");

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}
