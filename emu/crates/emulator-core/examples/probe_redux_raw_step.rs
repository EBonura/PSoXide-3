//! Dump Redux's raw instructions inside one folded user-side step.
//!
//! Usage:
//! `cargo run -p emulator-core --release --example probe_redux_raw_step -- <completed_steps> <disc.cue|disc.bin>`

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::Duration;

use parity_oracle::{OracleConfig, ReduxProcess};

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_redux_raw_step <completed_steps> <disc>");
    let disc_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .expect("usage: probe_redux_raw_step <completed_steps> <disc>");

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path, lua)
        .expect("Redux resolves")
        .with_disc(disc_path);
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(Duration::from_secs(30)).expect("handshake");

    eprintln!("[redux] running {start} user-side steps...");
    let timeout = Duration::from_secs((start / 200_000).max(60));
    let tick = redux.run(start, timeout).expect("run");
    eprintln!("[redux] at start tick={tick}; tracing one folded step...");

    let cap = std::env::var("PSOXIDE_REDUX_RAW_CAP")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3000);
    let command = match std::env::var("PSOXIDE_REDUX_STOP_PC") {
        Ok(stop_pc) => format!("manual_trace_until {stop_pc} {cap}"),
        Err(_) => format!("trace_one_step {cap}"),
    };
    redux.send_command(&command).expect("trace command");
    let mut raw_count = 0u64;
    let mut pc_counts: HashMap<u32, u64> = HashMap::new();
    let mut first = Vec::new();
    let mut samples = Vec::new();
    let mut last: VecDeque<String> = VecDeque::new();
    loop {
        let line = redux
            .wait_for_response(Duration::from_secs(30))
            .expect("trace response");
        if line.starts_with("raw ") {
            raw_count += 1;
            if let Some(pc) = parse_raw_pc(&line) {
                *pc_counts.entry(pc).or_insert(0) += 1;
            }
            if first.len() < 12 {
                first.push(line.clone());
            }
            if raw_count % 1000 == 0 {
                samples.push(line.clone());
            }
            last.push_back(line.clone());
            if last.len() > 12 {
                last.pop_front();
            }
        } else {
            println!("{line}");
        }
        if line == "trace_one_step ok"
            || line == "manual_trace_until ok"
            || line.starts_with("err ")
        {
            break;
        }
    }
    println!("raw_count_seen={raw_count} cap={cap}");
    println!("first_raw:");
    for line in &first {
        println!("  {line}");
    }
    println!("sampled_raw:");
    for line in &samples {
        println!("  {line}");
    }
    println!("last_raw:");
    for line in &last {
        println!("  {line}");
    }
    let mut top_pcs = pc_counts.into_iter().collect::<Vec<_>>();
    top_pcs.sort_by_key(|&(pc, count)| (std::cmp::Reverse(count), pc));
    println!("top_pcs:");
    for (pc, count) in top_pcs.into_iter().take(24) {
        println!("  pc=0x{pc:08x} count={count}");
    }
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}

fn parse_raw_pc(line: &str) -> Option<u32> {
    line.split_whitespace()
        .find_map(|part| part.strip_prefix("pc=")?.parse::<u32>().ok())
}
