//! Log every CDROM command byte the BIOS issues up to a step count,
//! with step + cycle. Tells us the exact command sequence so we can
//! spot missing commands against Redux.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{fast_boot_disc, fast_boot_disc_with_hle, Bus, Cpu};
use std::path::Path;

fn main() {
    let mut fastboot = false;
    let mut warmup_steps = 0u64;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--fastboot" {
            fastboot = true;
        } else if arg == "--warmup" {
            warmup_steps = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        } else {
            positional.push(arg);
        }
    }
    let target: u64 = positional
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(89_198_894);
    let disc_path = positional.get(1);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    if warmup_steps > 0 {
        for _ in 0..warmup_steps {
            cpu.step(&mut bus).expect("BIOS warmup step");
        }
        println!(
            "warmup={} pc=0x{:08x} cycles={} sr=0x{:08x} istat=0x{:03x} imask=0x{:03x}",
            warmup_steps,
            cpu.pc(),
            bus.cycles(),
            cpu.cop0()[12],
            bus.irq().stat(),
            bus.irq().mask()
        );
    }
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(Path::new(p)).expect("disc readable");
        if fastboot {
            let info = if warmup_steps > 0 {
                fast_boot_disc_with_hle(&mut bus, &mut cpu, &disc, false).expect("fast boot")
            } else {
                fast_boot_disc(&mut bus, &mut cpu, &disc).expect("fast boot")
            };
            println!(
                "fastboot={} entry=0x{:08x} load=0x{:08x} payload={}B",
                info.boot_path, info.initial_pc, info.load_addr, info.payload_len
            );
        }
        bus.cdrom.insert_disc(Some(disc));
        bus.attach_digital_pad_port1();
    }
    bus.cdrom.enable_command_log(4096);
    bus.cdrom.enable_response_log(8192);

    let mut last_log_len = 0usize;
    let mut last_response_log_len = 0usize;

    for step in 1..=target {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }

        let log = bus.cdrom.command_log();
        if log.len() != last_log_len {
            for entry in &log[last_log_len..] {
                println!(
                    "step={step:>10}  cyc={:>12}  cmd_cyc={:>12}  cmd=0x{:02x}  params=[{}]",
                    bus.cycles(),
                    entry.cycle,
                    entry.command,
                    fmt_params(&entry.params[..entry.param_len as usize]),
                );
            }
            last_log_len = log.len();
        }

        let response_log = bus.cdrom.response_log();
        if response_log.len() != last_response_log_len {
            for entry in &response_log[last_response_log_len..] {
                println!(
                    "step={step:>10}  cyc={:>12}  resp_cyc={:>12}  irq={:?}  bytes=[{}]",
                    bus.cycles(),
                    entry.cycle,
                    entry.irq,
                    fmt_params(&entry.bytes[..entry.len as usize]),
                );
            }
            last_response_log_len = response_log.len();
        }
    }
    println!();
    println!(
        "Total: {} commands dispatched by step {target}",
        bus.cdrom.commands_dispatched()
    );
    println!(
        "Final: pc=0x{:08x} cycles={} sr=0x{:08x} istat=0x{:03x} imask=0x{:03x} cd_irq(flag=0x{:02x},mask=0x{:02x}) hle={}",
        cpu.pc(),
        bus.cycles(),
        cpu.cop0()[12],
        bus.irq().stat(),
        bus.irq().mask(),
        bus.cdrom.irq_flag(),
        bus.cdrom.irq_mask_value(),
        summarize_hle(&bus.hle_bios_call_counts()),
    );
}

fn fmt_params(params: &[u8]) -> String {
    params
        .iter()
        .map(|p| format!("{p:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn summarize_hle(counts: &[[u32; 256]; 3]) -> String {
    let labels = ["A", "B", "C"];
    let mut parts = Vec::new();
    for (table, counts) in counts.iter().enumerate() {
        for (func, &count) in counts.iter().enumerate() {
            if count > 0 {
                parts.push(format!("{}({func:02x})={count}", labels[table]));
            }
        }
    }
    if parts.is_empty() {
        "-".to_string()
    } else {
        parts.join(",")
    }
}
