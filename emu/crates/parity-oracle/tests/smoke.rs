//! Pipe-protocol smoke test.
//!
//! Verifies the full command/response cycle end-to-end:
//!   1. Redux launches and Lua announces `ready`.
//!   2. Harness sends `handshake`, Lua replies `handshake ok`.
//!   3. Harness sends `quit`, Lua replies `bye` and breaks out of its
//!      read loop.
//!   4. Harness SIGKILLs Redux and inspects captured diagnostics.
//!
//! Requires:
//! - A built PCSX-Redux binary at the configured location.
//! - A PS1 BIOS at `PSOXIDE_BIOS` or the default fallback.
//!
//! Marked `#[ignore]`; run via `make oracle-smoke`.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use parity_oracle::{OracleConfig, ReduxProcess};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make oracle-smoke`"]
fn pipe_protocol_handshake_and_quit() {
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path(), lua).expect("Redux binary resolves");

    let mut process = ReduxProcess::launch(&config).expect("Redux launches");

    process.handshake(RESPONSE_TIMEOUT).expect("handshake succeeds");

    process.send_command("quit").expect("send quit");
    let bye = process
        .wait_for_response(RESPONSE_TIMEOUT)
        .expect("bye arrives");
    assert_eq!(bye, "bye");

    let capture = process.terminate();

    eprintln!("---- redux stdout log ----\n{}", capture.stdout);
    eprintln!("---- redux stderr log ----\n{}", capture.stderr);
    eprintln!("---- redux exit code  ----\n{:?}", capture.exit_code);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make oracle-smoke`"]
fn unknown_command_returns_error_response() {
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path(), lua).expect("Redux binary resolves");

    let mut process = ReduxProcess::launch(&config).expect("Redux launches");
    process.handshake(RESPONSE_TIMEOUT).expect("handshake succeeds");

    process.send_command("frobnicate").expect("send");
    let resp = process
        .wait_for_response(RESPONSE_TIMEOUT)
        .expect("error response arrives");
    assert!(
        resp.starts_with("err unknown:"),
        "unexpected response: {resp:?}"
    );

    process.send_command("quit").expect("send quit");
    let _ = process.wait_for_response(RESPONSE_TIMEOUT);
    let _ = process.terminate();
}
