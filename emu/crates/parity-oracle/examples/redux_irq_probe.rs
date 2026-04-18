//! Boot Redux once, then advance in chunks and query I_STAT/I_MASK,
//! CAUSE, SR at each checkpoint. A single Redux invocation amortises
//! the 10-minute warm-up over many data points.
//!
//! ```bash
//! cargo run -p parity-oracle --example redux_irq_probe --release -- \
//!     500000 1000000 2000000 5000000 10000000 19000000 19258367
//! ```
//! Each argument is an absolute step count to query at; they must be
//! monotonically increasing.

use parity_oracle::{OracleConfig, ReduxProcess};
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const STEP_TIMEOUT: Duration = Duration::from_secs(60 * 30);
const QUERY_TIMEOUT: Duration = Duration::from_secs(10);

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"))
}

fn main() {
    let mut checkpoints: Vec<u32> = env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if checkpoints.is_empty() {
        checkpoints = vec![100_000, 500_000, 2_000_000, 6_000_000, 10_000_000];
    }
    checkpoints.sort();

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path(), lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    let total: u32 = *checkpoints.last().unwrap();
    eprintln!("[probe] {} checkpoints, total {} steps", checkpoints.len(), total);
    let start = Instant::now();

    let mut at: u32 = 0;
    println!("# step          cycles    cause     sr      I_STAT     I_MASK   ack-counts");
    for cp in &checkpoints {
        let delta = cp - at;
        if delta > 0 {
            redux.step(delta, STEP_TIMEOUT).expect("step");
        }
        at = *cp;

        redux.send_command("regs").expect("regs cmd");
        let regs = redux.wait_for_response(QUERY_TIMEOUT).expect("regs reply");
        let cause = field(&regs, "cause=");
        let sr = field(&regs, "sr=");
        let cycles = field(&regs, "cycles=");

        redux.send_command("peek32 0x1F801070").expect("istat cmd");
        let istat = redux.wait_for_response(QUERY_TIMEOUT).expect("istat reply");
        redux.send_command("peek32 0x1F801074").expect("imask cmd");
        let imask = redux.wait_for_response(QUERY_TIMEOUT).expect("imask reply");

        println!(
            "{at:>10}  {cycles:>10}  0x{cause:08x}  0x{sr:08x}   0x{:08x}   0x{:08x}",
            parse_peek(&istat),
            parse_peek(&imask),
        );
        eprintln!("[probe] checkpoint {at} done at {:?}", start.elapsed());
    }

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(QUERY_TIMEOUT);
    let _ = redux.terminate();
}

fn field(line: &str, key: &str) -> u32 {
    let rest = &line[line.find(key).expect("key") + key.len()..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end].parse().expect("parse")
}

fn parse_peek(line: &str) -> u32 {
    line.strip_prefix("peek32 ")
        .and_then(|s| s.parse::<i64>().ok())
        .map(|v| v as u32)
        .unwrap_or(0xDEAD_BEEF)
}
