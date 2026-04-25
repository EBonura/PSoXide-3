//! Probe the retail-disc FMV path: CD sector streaming, XA audio,
//! MDEC input/output DMA, macroblock decode, and final GPU uploads.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_fmv_path -- \
//!   300000000 "/path/to/game.cue"
//! PSOXIDE_DISPLAY_DUMP=/tmp/fmv.ppm cargo run --release -p emulator-core --example probe_fmv_path -- 300000000 "/path/to/game.cue"
//! PSOXIDE_VISIBLE_DUMP=/tmp/visible.ppm cargo run --release -p emulator-core --example probe_fmv_path -- 300000000 "/path/to/game.cue"
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use std::path::PathBuf;

use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, Vram,
    DISC_FAST_BOOT_WARMUP_STEPS, VRAM_HEIGHT, VRAM_WIDTH,
};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_DISC: &str = "/Users/ebonura/Downloads/ps1 games/Metal Gear Solid (USA) (Disc 1) (Rev 1)/Metal Gear Solid (USA) (Disc 1) (Rev 1).cue";
const DEFAULT_STEPS: u64 = 300_000_000;
const SAMPLE_EVERY: u64 = 25_000_000;
const SPU_PUMP_CYCLES: u64 = 560_000;
const SPU_FRAME_SAMPLES: usize = 735;

fn main() {
    let mut fastboot = false;
    let mut positional = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--fastboot" {
            fastboot = true;
        } else {
            positional.push(arg);
        }
    }
    let mut args = positional.into_iter();
    let steps = args
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_STEPS);
    let disc_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DISC));
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS));

    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let disc = disc_support::load_disc_path(&disc_path).expect("disc load");

    let mut cpu = Cpu::new();
    let mut bus = Bus::new(bios).expect("bus");
    let dma_log_enabled = std::env::var("PSOXIDE_DMA_LOG").ok().as_deref() == Some("1");
    if dma_log_enabled {
        bus.set_dma_log_enabled(true);
    }
    let mut cycles_at_last_pump = 0u64;
    let mut next_sample = 0u64;
    let mut error = None;
    let trace_display_pixel = std::env::var("PSOXIDE_TRACE_DISPLAY_PIXEL")
        .ok()
        .and_then(|s| parse_xy(&s));
    let trace_vram_pixel = std::env::var("PSOXIDE_TRACE_VRAM_PIXEL")
        .ok()
        .and_then(|s| parse_xy(&s));
    let print_uploads = std::env::var("PSOXIDE_PRINT_UPLOADS").is_ok();
    let force_u8 = std::env::var("PSOXIDE_FORCE_U8")
        .ok()
        .and_then(|s| parse_force_u8(&s));
    if trace_display_pixel.is_some() || trace_vram_pixel.is_some() || print_uploads {
        bus.gpu.enable_pixel_tracer();
    }

    if fastboot {
        warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, DISC_FAST_BOOT_WARMUP_STEPS)
            .expect("BIOS warmup");
        let info = fast_boot_disc_with_hle(&mut bus, &mut cpu, &disc, false).expect("fast boot");
        println!(
            "fastboot={} entry=0x{:08x} load=0x{:08x} payload={}B",
            info.boot_path, info.initial_pc, info.load_addr, info.payload_len
        );
    }
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();

    println!(
        "fmv probe: steps={} mode={} bios={} disc={}",
        steps,
        if fastboot { "fastboot" } else { "bios" },
        bios_path.display(),
        disc_path.display()
    );
    println!(
        "{:>11} {:>12} {:>10} {:>18} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "step",
        "cycles",
        "pc",
        "display_hash",
        "mcmd",
        "mpar",
        "mb",
        "xa_q",
        "cdrdy",
        "fifo_pop",
        "dma0/1",
        "gpu_dma"
    );

    for step in 0..steps {
        if let Some((addr, value)) = force_u8 {
            let _ = bus.write8_safe(addr, value);
        }
        if step >= next_sample {
            print_sample(step, &cpu, &bus);
            next_sample = next_sample.saturating_add(SAMPLE_EVERY);
        }
        if let Err(e) = cpu.step(&mut bus) {
            error = Some(format!("{e:?}"));
            break;
        }
        if bus.cycles().saturating_sub(cycles_at_last_pump) > SPU_PUMP_CYCLES {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(SPU_FRAME_SAMPLES);
            let _ = bus.spu.drain_audio();
        }
    }

    print_sample(steps, &cpu, &bus);
    if let Some(error) = error {
        println!("cpu error: {error}");
    }
    let starts = bus.dma_start_triggers();
    println!("dma_start_triggers: {starts:?}");
    println!(
        "cdrom: irq_counts={:?} pending={} fifo={} audio_queue={} sector_events={} mode=0x{:02x} filter={:?} read_lba={}",
        bus.cdrom.irq_type_counts,
        bus.cdrom.pending_queue_len(),
        bus.cdrom.data_fifo_len(),
        bus.cdrom.cd_audio_queue_len(),
        bus.cdrom.sector_events_scheduled,
        bus.cdrom.debug_mode(),
        bus.cdrom.debug_xa_filter(),
        bus.cdrom.debug_read_lba()
    );
    if let Some((header, subheader)) = bus.cdrom.debug_last_sector() {
        println!("cdrom_last_sector: header={header:02x?} subheader={subheader:02x?}");
    }
    println!(
        "mdec: state={:?} commands={} params={} macroblocks={} queued_rle={} next_rle={:?}",
        bus.mdec.state(),
        bus.mdec.commands_seen(),
        bus.mdec.params_seen(),
        bus.mdec.macroblocks_decoded(),
        bus.mdec.queued_rle_halfwords(),
        bus.mdec.next_rle_halfword()
    );
    println!(
        "mdec_commands: [{}]",
        bus.mdec
            .command_history()
            .iter()
            .map(|v| format!("0x{v:08x}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    print_watch_words(
        &bus,
        &[
            ("re2_vblank_counter", 0x800a_cd00),
            ("re2_wait_base", 0x800a_bbe8),
            ("re2_input_ptr", 0x800a_bbdc),
            ("re2_movie_state", 0x800d_7680),
            ("re2_movie_flags", 0x800d_7684),
            ("re2_movie_ptr0", 0x800d_7670),
            ("re2_movie_ptr1", 0x800d_75d8),
            ("re2_str_ptr", 0x800f_8980),
        ],
    );
    print_pad_trace(&bus);
    let da = bus.gpu.display_area();
    let display_starts = bus
        .gpu
        .display_start_history()
        .map(|(x, y)| format!("({x},{y})"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "display_area: x={} y={} w={} h={} bpp24={} starts=[{}]",
        da.x, da.y, da.width, da.height, da.bpp24, display_starts
    );
    let display_modes = bus
        .gpu
        .display_mode_history()
        .map(|v| format!("0x{v:08x}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("display_modes: [{display_modes}]");
    print_gp1_recent(&bus.gpu, 80);
    println!("gp1_histogram:");
    for (op, count) in bus.gpu.gp1_opcode_histogram().iter().enumerate() {
        if *count > 0 {
            println!("  0x{op:02x}: {count}");
        }
    }
    print_ram_hits(&bus, b"Licensed");
    print_ram_hits(&bus, b"CD001");
    println!("gp0_histogram:");
    for (op, count) in bus.gpu.gp0_opcode_histogram().iter().enumerate() {
        if *count > 0 {
            println!("  0x{op:02x}: {count}");
        }
    }
    if let Ok(limit) = std::env::var("PSOXIDE_PRINT_UPLOADS") {
        let limit = limit.parse::<usize>().unwrap_or(64);
        print_recent_uploads(&bus.gpu, limit);
    }
    if dma_log_enabled {
        let log = bus.drain_dma_log();
        let cdr_log = log
            .iter()
            .enumerate()
            .filter(|(_, (kind, _, _, _))| kind.starts_with("Cdr"))
            .collect::<Vec<_>>();
        println!("dma_log cdr:");
        for (i, (kind, cycle, delta, target)) in cdr_log
            .iter()
            .skip(cdr_log.len().saturating_sub(120))
            .copied()
        {
            println!(
                "  #{:04} {:<8} cycle={} words/cycles={} target={}",
                i + 1,
                kind,
                cycle,
                delta,
                target
            );
        }
        println!("dma_log mdec:");
        for (i, (kind, cycle, delta, target)) in log
            .iter()
            .enumerate()
            .filter(|(_, (kind, _, _, _))| kind.starts_with("Mdec"))
            .take(200)
        {
            println!(
                "  #{:04} {:<8} cycle={} words/cycles={} target={}",
                i + 1,
                kind,
                cycle,
                delta,
                target
            );
        }
    }
    if let Some((x, y)) = trace_display_pixel {
        let da = bus.gpu.display_area();
        trace_pixel_owner(
            &bus.gpu,
            &format!("display ({x},{y})"),
            da.x.wrapping_add(x),
            da.y.wrapping_add(y),
        );
    }
    if let Some((x, y)) = trace_vram_pixel {
        trace_pixel_owner(&bus.gpu, &format!("vram ({x},{y})"), x, y);
    }

    if let Ok(path) = std::env::var("PSOXIDE_DISPLAY_DUMP") {
        dump_vram_ppm(&bus.gpu.vram, &path).expect("display dump");
        println!("display dump: {path}");
    }
    if let Ok(path) = std::env::var("PSOXIDE_VISIBLE_DUMP") {
        dump_visible_ppm(&bus.gpu, &path).expect("visible dump");
        println!("visible dump: {path}");
    }
    if let Ok(path) = std::env::var("PSOXIDE_FORCE24_DUMP") {
        let x = std::env::var("PSOXIDE_FORCE24_X")
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        let y = std::env::var("PSOXIDE_FORCE24_Y")
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        let width = std::env::var("PSOXIDE_FORCE24_WIDTH")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(256);
        let height = std::env::var("PSOXIDE_FORCE24_HEIGHT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or_else(|| bus.gpu.display_area().height);
        dump_force24_ppm(&bus.gpu, x, y, width, height, &path).expect("force24 dump");
        println!(
            "force24 dump: {path} ({width}x{height}) base=({}, {})",
            x.unwrap_or_else(|| bus.gpu.display_area().x),
            y.unwrap_or_else(|| bus.gpu.display_area().y)
        );
    }
}

fn print_pad_trace(bus: &Bus) {
    let cmds = bus
        .port1_pad_recent_commands()
        .into_iter()
        .map(|v| format!("0x{v:02x}"))
        .collect::<Vec<_>>()
        .join(", ");
    let first = bus
        .port1_recent_first_bytes()
        .into_iter()
        .map(|v| format!("0x{v:02x}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("port1_recent_first: [{first}]");
    println!("port1_recent_commands: [{cmds}]");
    println!("port1_recent_polls:");
    for (i, poll) in bus.port1_recent_polls().iter().enumerate() {
        let len = poll.len as usize;
        let tx = poll.tx[..len]
            .iter()
            .map(|v| format!("0x{v:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let rx = poll.rx[..len]
            .iter()
            .map(|v| format!("0x{v:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  #{i}: complete={} len={} tx=[{}] rx=[{}]",
            poll.complete, poll.len, tx, rx
        );
    }
}

fn parse_xy(text: &str) -> Option<(u16, u16)> {
    let (x, y) = text.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

fn parse_force_u8(text: &str) -> Option<(u32, u8)> {
    let (addr, value) = text.split_once(':')?;
    Some((parse_u32(addr)?, parse_u32(value)? as u8))
}

fn parse_u32(text: &str) -> Option<u32> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        text.parse().ok()
    }
}

fn trace_pixel_owner(gpu: &emulator_core::Gpu, label: &str, x: u16, y: u16) {
    let pixel = gpu.vram.get_pixel(x, y);
    println!("trace_pixel: {label} -> vram=({x},{y}) pixel=0x{pixel:04x}");
    match gpu.pixel_owner_at(x, y) {
        Some(entry) => {
            println!(
                "  owner: idx={} op=0x{:02x} words={}",
                entry.index,
                entry.opcode,
                entry.fifo.len()
            );
            for (i, word) in entry.fifo.iter().enumerate() {
                println!("    [{i}] 0x{word:08x}");
            }
        }
        None => println!("  owner: none (likely VRAM upload or untouched before tracer)"),
    }
    if let Some(upload) = find_last_upload_covering(gpu, x, y) {
        println!(
            "  upload_cover: idx={} xy=({}, {}) wh={}x{}",
            upload.0, upload.1, upload.2, upload.3, upload.4
        );
    }
}

fn find_last_upload_covering(
    gpu: &emulator_core::Gpu,
    x: u16,
    y: u16,
) -> Option<(u32, u16, u16, u16, u16)> {
    gpu.cmd_log.iter().rev().find_map(|entry| {
        if entry.opcode != 0xA0 || entry.fifo.len() < 3 {
            return None;
        }
        let xy = entry.fifo[1];
        let wh = entry.fifo[2];
        let ux = (xy & 0x3FF) as u16;
        let uy = ((xy >> 16) & 0x1FF) as u16;
        let raw_w = (wh & 0x3FF) as u16;
        let raw_h = ((wh >> 16) & 0x1FF) as u16;
        let w = if raw_w == 0 { 1024 } else { raw_w };
        let h = if raw_h == 0 { 512 } else { raw_h };
        let in_x = (x.wrapping_sub(ux) as u32) < w as u32;
        let in_y = (y.wrapping_sub(uy) as u32) < h as u32;
        in_x.then_some(())?;
        in_y.then_some((entry.index, ux, uy, w, h))
    })
}

fn print_recent_uploads(gpu: &emulator_core::Gpu, limit: usize) {
    let mut uploads = gpu
        .cmd_log
        .iter()
        .filter(|entry| entry.opcode == 0xA0 && entry.fifo.len() >= 3)
        .collect::<Vec<_>>();
    let start = uploads.len().saturating_sub(limit);
    uploads.drain(..start);
    println!(
        "recent_uploads last={} total_seen={}",
        uploads.len(),
        start + uploads.len()
    );
    for entry in uploads {
        let xy = entry.fifo[1];
        let wh = entry.fifo[2];
        let x = (xy & 0x3FF) as u16;
        let y = ((xy >> 16) & 0x1FF) as u16;
        let raw_w = (wh & 0x3FF) as u16;
        let raw_h = ((wh >> 16) & 0x1FF) as u16;
        let w = if raw_w == 0 { 1024 } else { raw_w };
        let h = if raw_h == 0 { 512 } else { raw_h };
        println!(
            "  idx={} xy=({}, {}) wh={}x{} raw=0x{wh:08x}",
            entry.index, x, y, w, h
        );
    }
}

fn print_gp1_recent(gpu: &emulator_core::Gpu, limit: usize) {
    let history = gpu.gp1_write_history();
    let start = history.len().saturating_sub(limit);
    println!(
        "gp1_recent last={} total_seen={}",
        history.len() - start,
        history.len()
    );
    for (i, &word) in history.iter().enumerate().skip(start) {
        let op = word >> 24;
        let detail = match op {
            0x05 => {
                let x = word & 0x3ff;
                let y = (word >> 10) & 0x1ff;
                format!(" display_start=({x},{y})")
            }
            0x06 => {
                let x1 = word & 0xfff;
                let x2 = (word >> 12) & 0xfff;
                format!(" h_range=({x1},{x2})")
            }
            0x07 => {
                let y1 = word & 0x3ff;
                let y2 = (word >> 10) & 0x3ff;
                format!(" v_range=({y1},{y2})")
            }
            0x08 => {
                let rgb24 = (word >> 4) & 1;
                let interlace = (word >> 5) & 1;
                let hres2 = (word >> 6) & 1;
                format!(
                    " mode_low=0x{:02x} rgb24={} interlace={} hres2={}",
                    word & 0xff,
                    rgb24,
                    interlace,
                    hres2
                )
            }
            _ => String::new(),
        };
        println!("  #{:04} gp1=0x{word:08x} op=0x{op:02x}{detail}", i + 1);
    }
}

fn print_ram_hits(bus: &Bus, needle: &[u8]) {
    let mut hits = Vec::new();
    for addr in 0x8000_0000u32..0x8020_0000u32 {
        let mut ok = true;
        for (i, &b) in needle.iter().enumerate() {
            if bus.try_read8(addr.wrapping_add(i as u32)) != Some(b) {
                ok = false;
                break;
            }
        }
        if ok {
            hits.push(addr);
            if hits.len() == 8 {
                break;
            }
        }
    }
    let label = String::from_utf8_lossy(needle);
    println!(
        "ram hits for {:?}: {}",
        label,
        hits.iter()
            .map(|addr| format!("0x{addr:08x}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for addr in hits.iter().take(2) {
        println!("  dump @ 0x{addr:08x}:");
        for row in 0..4u32 {
            let base = addr.wrapping_add(row * 16);
            let bytes: Vec<u8> = (0..16)
                .map(|i| bus.try_read8(base + i).unwrap_or(0))
                .collect();
            let hex = bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            let ascii = bytes
                .iter()
                .map(|&b| {
                    if (0x20..=0x7e).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect::<String>();
            println!("    0x{base:08x}: {hex}  {ascii}");
        }
    }
}

fn print_watch_words(bus: &Bus, entries: &[(&str, u32)]) {
    println!("watch_words:");
    for &(label, addr) in entries {
        println!("  {label:<18} 0x{addr:08x}=0x{:08x}", peek_u32(bus, addr));
    }
}

fn peek_u32(bus: &Bus, addr: u32) -> u32 {
    let b0 = bus.try_read8(addr).unwrap_or(0);
    let b1 = bus.try_read8(addr.wrapping_add(1)).unwrap_or(0);
    let b2 = bus.try_read8(addr.wrapping_add(2)).unwrap_or(0);
    let b3 = bus.try_read8(addr.wrapping_add(3)).unwrap_or(0);
    u32::from_le_bytes([b0, b1, b2, b3])
}

fn print_sample(step: u64, cpu: &Cpu, bus: &Bus) {
    let (hash, _w, _h, _len) = bus.gpu.display_hash();
    let starts = bus.dma_start_triggers();
    println!(
        "{step:>11} {cycles:>12} 0x{pc:08x} 0x{hash:016x} {mcmd:>7} {mpar:>7} {mb:>7} {xa_q:>7} {cdrdy:>9} {fifo_pop:>9} {dma01:>9} {gpu_dma:>9}",
        cycles = bus.cycles(),
        pc = cpu.pc(),
        mcmd = bus.mdec.commands_seen(),
        mpar = bus.mdec.params_seen(),
        mb = bus.mdec.macroblocks_decoded(),
        xa_q = bus.cdrom.cd_audio_queue_len(),
        cdrdy = bus.cdrom.irq_type_counts[1],
        fifo_pop = bus.cdrom.data_fifo_pops(),
        dma01 = format!("{}/{}", starts[0], starts[1]),
        gpu_dma = starts[2],
    );
}

fn dump_vram_ppm(vram: &Vram, path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6\n{VRAM_WIDTH} {VRAM_HEIGHT}\n255")?;
    let mut rgb = Vec::with_capacity(VRAM_WIDTH * VRAM_HEIGHT * 3);
    for &pix in vram.words() {
        let r5 = (pix & 0x1F) as u8;
        let g5 = ((pix >> 5) & 0x1F) as u8;
        let b5 = ((pix >> 10) & 0x1F) as u8;
        rgb.push((r5 << 3) | (r5 >> 2));
        rgb.push((g5 << 3) | (g5 >> 2));
        rgb.push((b5 << 3) | (b5 >> 2));
    }
    file.write_all(&rgb)
}

fn dump_visible_ppm(gpu: &emulator_core::Gpu, path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let (rgba, width, height) = gpu.display_rgba8();
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6\n{width} {height}\n255")?;
    for px in rgba.chunks_exact(4) {
        file.write_all(&px[..3])?;
    }
    Ok(())
}

fn dump_force24_ppm(
    gpu: &emulator_core::Gpu,
    base_x: Option<u16>,
    base_y: Option<u16>,
    width: u16,
    height: u16,
    path: &str,
) -> std::io::Result<()> {
    use std::io::Write;

    let da = gpu.display_area();
    let x0 = base_x.unwrap_or(da.x);
    let y0 = base_y.unwrap_or(da.y);
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", width, height)?;
    for y in 0..height {
        for x in 0..width {
            let byte_index = x as u32 * 3;
            let r = display_24bpp_byte(gpu, x0, y0.wrapping_add(y), byte_index);
            let g = display_24bpp_byte(gpu, x0, y0.wrapping_add(y), byte_index + 1);
            let b = display_24bpp_byte(gpu, x0, y0.wrapping_add(y), byte_index + 2);
            file.write_all(&[r, g, b])?;
        }
    }
    Ok(())
}

fn display_24bpp_byte(gpu: &emulator_core::Gpu, base_x: u16, y: u16, byte_index: u32) -> u8 {
    let word_x = (byte_index / 2) as u16;
    let byte_off = (byte_index & 1) as usize;
    let word = gpu.vram.get_pixel(base_x.wrapping_add(word_x), y);
    word.to_le_bytes()[byte_off]
}
