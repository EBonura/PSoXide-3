//! One-shot generator for a long Redux BIOS trace, written to the
//! parity cache in streaming mode so we never hold the full `Vec`
//! in RAM. A 100 M-record trace is ~14 GiB -- this example streams
//! each record from Redux's stdout directly into the cache file.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release -p emulator-core --example generate_trace -- 100000000
//! ```
//!
//! Runtime on Apple Silicon: ~25 s per million steps, i.e. ~40 min for
//! 100 M. Logs progress every million records so you can tell it's
//! alive. Output: `<target>/parity-cache/redux-<biosHash>-<N>.bin`,
//! which all `load_prefix(min_steps <= N)` consumers pick up for
//! free.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use parity_oracle::{cache, OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const STEP_TIMEOUT: Duration = Duration::from_secs(30);

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios_bytes = std::fs::read(&bios_path).expect("BIOS readable");

    let dir = cache::default_dir();
    let target = cache::path_for(&dir, &bios_bytes, n as usize);
    if target.exists() {
        eprintln!(
            "[generate_trace] cache already exists at {} — delete it to regenerate",
            target.display()
        );
        return;
    }

    eprintln!(
        "[generate_trace] launching Redux for {n} steps → {}",
        target.display()
    );
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path, lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    let mut writer = cache::StreamingWriter::create(&target, n).expect("open streaming writer");
    let start = Instant::now();
    redux
        .step_streaming(n, STEP_TIMEOUT, Some(1_000_000), |rec| {
            writer.push(rec).map_err(Into::into)
        })
        .expect("step_streaming");
    let elapsed = start.elapsed();
    eprintln!(
        "[generate_trace] streamed {n} records in {:.1} s ({:.0} steps/s)",
        elapsed.as_secs_f64(),
        n as f64 / elapsed.as_secs_f64()
    );

    writer.finish().expect("finalize cache");
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();

    eprintln!("[generate_trace] wrote {}", target.display());
}
