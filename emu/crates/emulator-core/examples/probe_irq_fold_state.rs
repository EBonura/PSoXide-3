//! Compare local and Redux state around one folded IRQ user step.
//!
//! Usage:
//! `cargo run -p emulator-core --release --example probe_irq_fold_state -- <completed_steps> <disc.cue|disc.bin>`

#[path = "support/disc.rs"]
mod disc_support;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const PEEK_ADDRS: [u32; 8] = [
    0x1f80_1070, // I_STAT
    0x1f80_1074, // I_MASK
    0x1f80_1040, // SIO data/mode echo
    0x1f80_1044, // SIO stat
    0x1f80_1048, // SIO mode/control
    0x1f80_104c, // SIO misc/baud
    0x1f80_1800, // CDROM index/response
    0x1f80_10f4, // DICR
];

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_irq_fold_state <completed_user_steps> <disc>");
    let disc_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .expect("usage: probe_irq_fold_state <completed_user_steps> <disc>");

    run_local(start, &disc_path);
    run_redux(start, &disc_path);
}

fn run_local(start: u64, disc_path: &PathBuf) {
    eprintln!("[ours] running {start} folded user steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(BIOS).expect("BIOS");
    let disc = disc_support::load_disc_path(disc_path).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    let mut cpu = Cpu::new();

    for _ in 0..start {
        step_folded(&mut cpu, &mut bus);
    }
    eprintln!("[ours] reached start in {:.1}s", t0.elapsed().as_secs_f64());

    print_local_snapshot("before", &cpu, &mut bus);
    let before = bus.cycles();
    let rec = cpu.step(&mut bus).expect("local user step");
    println!(
        "ours step: pc=0x{:08x} instr=0x{:08x} tick={} next_pc=0x{:08x} in_irq={}",
        rec.pc,
        rec.instr,
        rec.tick,
        cpu.pc(),
        cpu.in_irq_handler()
    );
    let mut raw = 0u64;
    while cpu.in_irq_handler() {
        let rec = cpu.step(&mut bus).expect("local isr step");
        raw += 1;
        if raw <= 8 || raw % 500 == 0 {
            println!(
                "ours isr {raw}: pc=0x{:08x} instr=0x{:08x} tick={} next_pc=0x{:08x} istat=0x{:03x} imask=0x{:03x}",
                rec.pc,
                rec.instr,
                rec.tick,
                cpu.pc(),
                bus.irq().stat(),
                bus.irq().mask()
            );
        }
        if raw > 50_000 {
            println!("ours isr stopped after {raw} raw instructions");
            break;
        }
    }
    println!(
        "ours folded delta={} raw_isr={raw}",
        bus.cycles().saturating_sub(before)
    );
    print_local_snapshot("after", &cpu, &mut bus);
}

fn run_redux(start: u64, disc_path: &PathBuf) {
    eprintln!("[redux] launching...");
    let bios = PathBuf::from(BIOS);
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios, lua)
        .expect("Redux resolves")
        .with_disc(disc_path.clone());
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(Duration::from_secs(30)).expect("handshake");

    eprintln!("[redux] running {start} folded user steps...");
    let t0 = Instant::now();
    let timeout = Duration::from_secs((start / 200_000).max(60));
    let tick = redux.run(start, timeout).expect("Redux run");
    eprintln!(
        "[redux] reached start in {:.1}s, tick={tick}",
        t0.elapsed().as_secs_f64()
    );

    print_redux_snapshot("before", &mut redux);
    let trace = redux
        .step(1, Duration::from_secs(30))
        .expect("Redux folded step");
    if let Some(rec) = trace.first() {
        println!(
            "redux step: pc=0x{:08x} instr=0x{:08x} tick={} gpr_v0=0x{:08x} gpr_a0=0x{:08x} gpr_t0=0x{:08x} gpr_ra=0x{:08x}",
            rec.pc,
            rec.instr,
            rec.tick,
            rec.gprs[2],
            rec.gprs[4],
            rec.gprs[8],
            rec.gprs[31]
        );
    }
    print_redux_snapshot("after", &mut redux);

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}

fn step_folded(cpu: &mut Cpu, bus: &mut Bus) {
    let was_in_isr = cpu.in_isr();
    cpu.step(bus).expect("step");
    if !was_in_isr && cpu.in_irq_handler() {
        while cpu.in_irq_handler() {
            cpu.step(bus).expect("isr step");
        }
    }
}

fn print_local_snapshot(label: &str, cpu: &Cpu, bus: &mut Bus) {
    let cop0 = cpu.cop0();
    let sio_stat = bus.sio0().debug_stat();
    let sio_ctrl = bus.sio0().debug_ctrl();
    let sio_pending_irq = bus.sio0().debug_pending_irq();
    let sio_irq_latched = bus.sio0().debug_irq_latched();
    let sio_transfer = bus.sio0().debug_transfer_deadline();
    let sio_ack_start = bus.sio0().debug_ack_deadline();
    let sio_ack_end = bus.sio0().debug_ack_end_deadline();
    println!(
        "ours {label}: pc=0x{:08x} cycles={} sr=0x{:08x} cause=0x{:08x} epc=0x{:08x} istat=0x{:03x} imask=0x{:03x} dicr=0x{:08x} cdflag=0x{:02x} lastcmd=0x{:02x}",
        cpu.pc(),
        bus.cycles(),
        cop0[12],
        cop0[13],
        cop0[14],
        bus.irq().stat(),
        bus.irq().mask(),
        bus.read32(0x1f80_10f4),
        bus.cdrom.irq_flag(),
        bus.cdrom.last_command()
    );
    println!(
        "ours {label} sio: stat=0x{:08x} ctrl=0x{:04x} pending_irq={} latched={} transfer={:?} ack_start={:?} ack_end={:?}",
        sio_stat,
        sio_ctrl,
        sio_pending_irq,
        sio_irq_latched,
        sio_transfer,
        sio_ack_start,
        sio_ack_end
    );
}

fn print_redux_snapshot(label: &str, redux: &mut ReduxProcess) {
    redux.send_command("regs").expect("Redux regs cmd");
    let regs = redux
        .wait_for_response(Duration::from_secs(5))
        .unwrap_or_else(|err| format!("regs timeout: {err}"));
    println!("redux {label}: {regs}");
    for addr in PEEK_ADDRS {
        match redux.peek32(addr, Duration::from_secs(5)) {
            Ok(value) => println!("redux {label} peek32 0x{addr:08x}=0x{value:08x}"),
            Err(err) => println!("redux {label} peek32 0x{addr:08x}=<err {err}>"),
        }
    }
}
