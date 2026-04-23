//! Capture Redux's and our display-area pixels at step N, compare
//! byte-for-byte, and report exactly what matches and where. Used
//! to establish Redux-anchored display parity for milestone
//! goldens (Sony logo, shell, PlayStation splash, game title) so
//! we know we're rendering the same pixels, not just the same
//! pixels as ourselves.
//!
//! ```bash
//! cargo run -p parity-oracle --example display_parity_at --release -- 100000000
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    let n: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: display_parity_at <step_count> [disc_path]");
    let disc_path = env::args().nth(2);

    let bios_path = env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));

    // --- Capture Redux side ---
    // Redux accepts the disc via `-iso PATH`; OracleConfig threads it
    // through. Passing the same disc to both emulators is what
    // unlocks milestone-D-plus parity (Crash, Tekken) — the old
    // "no-disc-only" restriction is dead.
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path.clone(), lua).expect("Redux binary resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }

    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    eprintln!("[redux] running {n} steps in Redux (silent)...");
    let run_timeout = Duration::from_secs((n / 400_000).max(60));
    let r_tick = redux.run(n, run_timeout).expect("run");
    eprintln!("[redux] reached tick={r_tick}");

    let redux_path = PathBuf::from(format!("/tmp/redux_display_{n}.bin"));
    redux
        .screenshot_save(&redux_path, Duration::from_secs(60))
        .expect("screenshot save");
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    let redux_bytes = std::fs::read(&redux_path).expect("read redux bytes");
    let redux_meta =
        std::fs::read_to_string(format!("{}.txt", redux_path.display())).unwrap_or_default();
    let (r_w, r_h) = parse_wh(&redux_meta);
    eprintln!(
        "[redux] display {r_w}×{r_h}, {} bytes, hash=0x{:016x}",
        redux_bytes.len(),
        fnv1a_64(&redux_bytes)
    );

    // --- Capture ours ---
    let bios = std::fs::read(&bios_path).expect("bios");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(&PathBuf::from(p)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
    let mut cpu = Cpu::new();
    eprintln!("[ours]  running {n} steps in our emulator...");
    for _ in 0..n {
        cpu.step(&mut bus).expect("step");
    }
    let (_ours_hash, ours_w, ours_h, ours_len) = bus.gpu.display_hash();
    // Also dump raw bytes for direct comparison.
    let da = bus.gpu.display_area();
    let mut ours_bytes = Vec::with_capacity(ours_len);
    for dy in 0..ours_h as u16 {
        for dx in 0..ours_w as u16 {
            let pixel = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
            ours_bytes.extend_from_slice(&pixel.to_le_bytes());
        }
    }
    let ours_path = PathBuf::from(format!("/tmp/ours_display_{n}.bin"));
    std::fs::write(&ours_path, &ours_bytes).expect("write ours");
    eprintln!(
        "[ours]  display {ours_w}×{ours_h}, {} bytes, hash=0x{:016x}",
        ours_bytes.len(),
        fnv1a_64(&ours_bytes)
    );

    // --- Compare ---
    println!();
    println!("=== Display parity @ step {n} ===");
    println!("dimensions: redux={r_w}×{r_h}  ours={ours_w}×{ours_h}");
    println!(
        "byte count: redux={}  ours={}",
        redux_bytes.len(),
        ours_bytes.len()
    );
    if r_w == ours_w && r_h == ours_h {
        byte_compare(&redux_bytes, &ours_bytes, r_w);
    } else {
        println!("dimensions differ — comparing overlap only");
        let w = r_w.min(ours_w);
        let h = r_h.min(ours_h);
        let row_bytes = (w * 2) as usize;
        let mut diffs = 0usize;
        let mut first_diff: Option<(u32, u32)> = None;
        for y in 0..h {
            let r_row_start = (y as usize) * (r_w as usize) * 2;
            let o_row_start = (y as usize) * (ours_w as usize) * 2;
            for x in 0..row_bytes {
                let r_off = r_row_start + x;
                let o_off = o_row_start + x;
                if r_off < redux_bytes.len()
                    && o_off < ours_bytes.len()
                    && redux_bytes[r_off] != ours_bytes[o_off]
                {
                    diffs += 1;
                    if first_diff.is_none() {
                        first_diff = Some((y, (x / 2) as u32));
                    }
                }
            }
        }
        let total_bytes = (w * h * 2) as usize;
        println!("overlap: {w}×{h} = {total_bytes} bytes");
        println!(
            "differing bytes: {diffs} / {total_bytes} ({:.2}%)",
            100.0 * diffs as f64 / total_bytes.max(1) as f64
        );
        if let Some((y, x)) = first_diff {
            println!("first diff at (x={x}, y={y})");
        }
    }
    println!();
    println!("Redux raw bytes: {}", redux_path.display());
    println!("Ours raw bytes:  {}", ours_path.display());
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

fn parse_wh(meta: &str) -> (u32, u32) {
    let mut w = 0u32;
    let mut h = 0u32;
    for tok in meta.split_whitespace() {
        if let Some(v) = tok.strip_prefix("w=") {
            w = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("h=") {
            h = v.parse().unwrap_or(0);
        }
    }
    (w, h)
}

fn byte_compare(a: &[u8], b: &[u8], width: u32) {
    let mut diffs = 0usize;
    let mut first_diff: Option<usize> = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            diffs += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }
    let total = a.len().min(b.len());
    println!(
        "differing bytes: {diffs} / {total} ({:.2}%)",
        100.0 * diffs as f64 / total.max(1) as f64
    );
    if let Some(off) = first_diff {
        let px = off / 2;
        let y = px as u32 / width;
        let x = px as u32 % width;
        println!("first diff at byte {off} → pixel (x={x}, y={y})");
    }
}
