#![allow(dead_code)]

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const EXAMPLE_OUT: &str = "build/examples/mipsel-sony-psx/release";
const SPU_PUMP_CYCLES: u64 = 560_000;
const SPU_SAMPLES_PER_PUMP: usize = 735;

pub struct SideLoadedExe {
    pub bus: Bus,
    pub cpu: Cpu,
    cycles_at_last_spu_pump: u64,
}

impl SideLoadedExe {
    pub fn example(name: &str, attach_digital_pad: bool) -> Self {
        let exe_path = repo_root().join(EXAMPLE_OUT).join(format!("{name}.exe"));
        Self::from_exe_path(&exe_path, attach_digital_pad)
    }

    pub fn from_exe_path(exe_path: &Path, attach_digital_pad: bool) -> Self {
        let bios = read_bios();
        let exe_bytes = std::fs::read(exe_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", exe_path.display()));
        let exe = Exe::parse(&exe_bytes).expect("parse PS1 EXE");

        let mut bus = Bus::new(bios).expect("bus");
        bus.load_exe_payload(exe.load_addr, &exe.payload);
        bus.enable_hle_bios();
        if attach_digital_pad {
            bus.attach_digital_pad_port1();
        }

        let mut cpu = Cpu::new();
        cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

        Self {
            bus,
            cpu,
            cycles_at_last_spu_pump: 0,
        }
    }

    pub fn run_until_vblank(&mut self, target: u64) -> bool {
        while self.bus.irq().raise_counts()[0] < target {
            if !step_cpu_and_pump_spu(
                &mut self.cpu,
                &mut self.bus,
                &mut self.cycles_at_last_spu_pump,
            ) {
                return false;
            }
        }
        true
    }

    pub fn dump_display_ppm(&self, stem: &str, frame: u64) -> io::Result<FrameDump> {
        dump_display_ppm(&self.bus, stem, frame)
    }
}

pub fn step_cpu_and_pump_spu(
    cpu: &mut Cpu,
    bus: &mut Bus,
    cycles_at_last_spu_pump: &mut u64,
) -> bool {
    if cpu.step(bus).is_err() {
        return false;
    }
    pump_spu_if_due(bus, cycles_at_last_spu_pump);
    true
}

pub fn pump_spu_if_due(bus: &mut Bus, cycles_at_last_spu_pump: &mut u64) {
    if bus.cycles() - *cycles_at_last_spu_pump > SPU_PUMP_CYCLES {
        *cycles_at_last_spu_pump = bus.cycles();
        bus.run_spu_samples(SPU_SAMPLES_PER_PUMP);
        let _ = bus.spu.drain_audio();
    }
}

pub struct FrameDump {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub display_hash: u64,
    pub display_bytes: usize,
}

impl FrameDump {
    pub fn log(&self, frame: u64) {
        eprintln!(
            "[frame-probe] frame={frame} display={}x{} bytes={} hash=0x{:016x} wrote={}",
            self.width,
            self.height,
            self.display_bytes,
            self.display_hash,
            self.path.display()
        );
    }
}

pub fn dump_display_ppm(bus: &Bus, stem: &str, frame: u64) -> io::Result<FrameDump> {
    let da = bus.gpu.display_area();
    let path = PathBuf::from(format!("/tmp/{stem}-f{frame:03}.ppm"));
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "P6\n{} {}\n255", da.width, da.height)?;

    let mut rgb = Vec::with_capacity((da.width as usize) * (da.height as usize) * 3);
    for dy in 0..da.height {
        for dx in 0..da.width {
            let pix = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
            let r5 = (pix & 0x1F) as u8;
            let g5 = ((pix >> 5) & 0x1F) as u8;
            let b5 = ((pix >> 10) & 0x1F) as u8;
            rgb.push((r5 << 3) | (r5 >> 2));
            rgb.push((g5 << 3) | (g5 >> 2));
            rgb.push((b5 << 3) | (b5 >> 2));
        }
    }
    file.write_all(&rgb)?;

    let (display_hash, width, height, display_bytes) = bus.gpu.display_hash();
    Ok(FrameDump {
        path,
        width,
        height,
        display_hash,
        display_bytes,
    })
}

pub fn display_nonzero_pixels(bus: &Bus) -> usize {
    let da = bus.gpu.display_area();
    let mut nonzero = 0usize;
    for dy in 0..da.height {
        for dx in 0..da.width {
            if bus.gpu.vram.get_pixel(da.x + dx, da.y + dy) != 0 {
                nonzero += 1;
            }
        }
    }
    nonzero
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

pub fn read_bios() -> Vec<u8> {
    std::fs::read(bios_path()).expect("BIOS readable")
}

pub fn bios_path() -> PathBuf {
    std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}
