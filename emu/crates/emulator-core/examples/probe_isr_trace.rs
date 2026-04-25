//! Run to the step just before the first cycle-accounting divergence
//! (step 19474544 per earlier probing) and trace every instruction
//! we execute inside the IRQ handler. Compare the path to what the
//! Redux trace says user code does after the RFE — that tells us
//! whether our ISR body has a divergent branch (different registers,
//! different condition), or whether we're simply executing fewer
//! instructions than Redux per ISR pass.

use emulator_core::{Bus, Cpu};
use parity_oracle::cache;
use std::path::PathBuf;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");

    let dir = cache::default_dir();
    let trace = cache::load_prefix(&dir, &bios, 50_000_000).expect("No cached trace long enough");

    let target_step: usize = std::env::var("PSOXIDE_ISR_AT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_474_544);

    eprintln!("Running to step {target_step} and tracing the ISR that Redux folds into that step.");

    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    // Fast-forward to the step BEFORE the divergence point. Walker
    // uses the parity-folded step model so our step counts line up
    // with Redux's trace indices.
    // `target_step` is Redux's 0-based index of the diverging record.
    // We want to end up in the state that matches trace[target_step-1]
    // — i.e., JUST about to execute the instruction Redux is ABOUT to
    // retire as step `target_step`.
    for _i in 0..target_step {
        let was_in_isr = cpu.in_isr();
        let _ = cpu.step_traced(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let _ = cpu.step_traced(&mut bus).expect("step");
            }
        }
    }

    // Validate we're at the expected state.
    let expected = &trace[target_step - 1];
    eprintln!(
        "At step {}: our pc=0x{:08x}, redux pc=0x{:08x}, our tick={}, redux tick={}",
        target_step - 1,
        cpu.pc(),
        expected.pc,
        bus.cycles(),
        expected.tick,
    );

    // Now step through target_step with instruction-level logging so
    // we can see exactly what the ISR body does.
    eprintln!();
    eprintln!("=== Stepping through step {target_step} (should include ISR fold) ===");
    let was_in_isr = cpu.in_isr();
    let pc_before = cpu.pc();
    let cycles_before = bus.cycles();
    eprintln!("Starting: pc=0x{pc_before:08x}  cycles={cycles_before}  in_isr={was_in_isr}",);

    // Dump bus state before stepping — which IRQs are pending?
    let istat = bus.irq().stat();
    let imask = bus.irq().mask();
    let sr = cpu.cop0()[12]; // Status register is COP0 reg 12
    eprintln!(
        "Pre-step: ISTAT=0x{istat:08x}  IMASK=0x{imask:08x}  SR=0x{sr:08x}  (ISTAT&IMASK=0x{:08x})",
        istat & imask,
    );
    eprintln!(
        "           IEc(SR.0)={}  IM[bit10]={}  would_fire={}",
        sr & 1 != 0,
        sr & (1 << 10) != 0,
        (istat & imask) != 0 && sr & 1 != 0 && sr & (1 << 10) != 0,
    );

    // Total IRQ raises per source up to this point — lets us verify
    // whether our VBlank/CDROM/DMA counts match Redux's expected
    // rate.
    let raise_counts = bus.irq().raise_counts();
    let names = [
        "VBlank",
        "Gpu",
        "Cdrom",
        "Dma",
        "Timer0",
        "Timer1",
        "Timer2",
        "Controller",
        "Sio",
        "Spu",
        "Lightpen",
    ];
    eprintln!("  IRQ raise histogram to this point:");
    for (i, &n) in raise_counts.iter().enumerate() {
        if n > 0 {
            eprintln!("    {:>10} = {n}", names[i]);
        }
    }

    // Expected #VBlanks at this cycle. Each VBlank at 564398 cycles.
    let cycles = bus.cycles();
    let expected_vblanks = if cycles >= 521478 {
        1 + (cycles - 521478) / 564398
    } else {
        0
    };
    eprintln!("  cycles={cycles}, expected VBlank count at this cycle: {expected_vblanks}");

    let rec = cpu.step_traced(&mut bus).expect("step");
    eprintln!(
        "  [main] pc=0x{:08x}  tick={}  (delta={})  in_isr={}  in_irq={}",
        rec.pc,
        rec.tick,
        rec.tick - cycles_before,
        cpu.in_isr(),
        cpu.in_irq_handler(),
    );

    if !was_in_isr && cpu.in_irq_handler() {
        let mut isr_step = 0;
        while cpu.in_irq_handler() {
            let prev_cycles = bus.cycles();
            let prev_pc = cpu.pc();
            let r = cpu.step_traced(&mut bus).expect("step");
            isr_step += 1;
            eprintln!(
                "  [isr #{isr_step}] pc=0x{prev_pc:08x} → 0x{:08x}  instr at fetch=0x?  tick={} (+{})  in_irq={}",
                r.pc, r.tick, r.tick - prev_cycles, cpu.in_irq_handler(),
            );
            if isr_step > 300 {
                eprintln!("  ... (truncated after 300 ISR steps)");
                break;
            }
        }
        eprintln!(
            "=== ISR done in {isr_step} instructions, final tick={} ===",
            bus.cycles()
        );
    } else {
        eprintln!("(No ISR was folded in at this step.)");
    }

    eprintln!();
    eprintln!(
        "Redux says step {target_step} should end at tick {} with pc={:08x}.",
        trace[target_step].tick, trace[target_step].pc,
    );
    eprintln!(
        "We end at tick {} with pc=0x{:08x}. Delta tick: {:+}.",
        bus.cycles(),
        cpu.pc(),
        bus.cycles() as i64 - trace[target_step].tick as i64,
    );

    // Probe forward: at what later step does our emulator finally
    // take the ISR that Redux folded into step `target_step`? If
    // `never`, the divergence is permanent — our IRQ source is
    // missing an event Redux raises.
    eprintln!();
    eprintln!("=== Scanning forward to find when our ISR finally fires ===");
    let base_delta = bus.cycles() as i64 - trace[target_step].tick as i64;
    let mut i = target_step;
    let mut fold_found_at: Option<usize> = None;
    while i < trace.len() - 1 && i < target_step + 1000 {
        let was_in_isr = cpu.in_isr();
        // Capture ISTAT/IMASK BEFORE stepping so we can see what
        // subsystem raised the IRQ.
        let pre_istat = bus.irq().stat();
        let pre_imask = bus.irq().mask();
        let _rec = cpu.step_traced(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            // Bit name lookup.
            let names = [
                "VBlank",
                "Gpu",
                "Cdrom",
                "Dma",
                "Timer0",
                "Timer1",
                "Timer2",
                "Controller",
                "Sio",
                "Spu",
                "Lightpen",
            ];
            let pending = pre_istat & pre_imask;
            let mut which = vec![];
            for b in 0..11 {
                if pending & (1 << b) != 0 {
                    which.push(names[b]);
                }
            }
            eprintln!(
                "  Step {i}: IRQ fired. ISTAT=0x{pre_istat:08x} IMASK=0x{pre_imask:08x} source(s)=[{}]",
                which.join(", "),
            );
            let raise_counts = bus.irq().raise_counts();
            for (idx, &n) in raise_counts.iter().enumerate() {
                if n > 0 {
                    eprintln!("    {:>10} raises total: {n}", names[idx]);
                }
            }
            let mut isr_len = 0;
            while cpu.in_irq_handler() {
                let _ = cpu.step_traced(&mut bus).expect("step");
                isr_len += 1;
            }
            eprintln!(
                "  ISR ran {isr_len} instructions. Cycles now {}.",
                bus.cycles(),
            );
            fold_found_at = Some(i);
            break;
        }
        i += 1;
    }
    if let Some(found_at) = fold_found_at {
        eprintln!(
            "Our ISR triggered at step {found_at} ({} steps after target). \
             Delta to Redux changed from {:+} to {:+}.",
            found_at - target_step,
            base_delta,
            bus.cycles() as i64 - trace[found_at].tick as i64,
        );
    } else {
        eprintln!("No ISR fired in our emulator within 1000 steps of target.");
        let final_delta = bus.cycles() as i64 - trace[i].tick as i64;
        eprintln!("Cycle delta after 1000 steps: {:+}", final_delta);
    }
}
