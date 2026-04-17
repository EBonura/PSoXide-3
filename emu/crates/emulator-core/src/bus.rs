//! System bus: owns physical memory and dispatches loads to regions.
//!
//! Current coverage: RAM, BIOS, scratchpad. Everything else panics on
//! access — deliberately, because we want unmapped reads to be loud
//! until each region's owning module (GPU, SPU, CD-ROM, …) is wired up.

use psx_hw::memory::{self, to_physical};
use thiserror::Error;

use crate::irq::Irq;

/// Physical address of `I_STAT` (interrupt status / ack register).
const IRQ_STAT_ADDR: u32 = 0x1F80_1070;
/// Physical address of `I_MASK` (interrupt enable register).
const IRQ_MASK_ADDR: u32 = 0x1F80_1074;

/// Errors constructing a [`Bus`].
#[derive(Error, Debug)]
pub enum BusError {
    /// BIOS image was not exactly 512 KiB.
    #[error("BIOS image must be exactly {expected} bytes, got {actual}")]
    BiosSize {
        /// Expected size in bytes.
        expected: usize,
        /// Size that was actually provided.
        actual: usize,
    },
}

/// The PS1 system bus.
pub struct Bus {
    ram: Box<[u8; memory::ram::SIZE]>,
    bios: Box<[u8; memory::bios::SIZE]>,
    scratchpad: Box<[u8; memory::scratchpad::SIZE]>,
    /// Write-echoes-on-read buffer for the MMIO window. **Placeholder.**
    /// Individual peripherals with real semantics (IRQ below, later GPU /
    /// SPU / CD-ROM / DMA / timers) intercept their own ranges ahead of
    /// this fallback; the rest of MMIO still round-trips writes to reads.
    io: Box<[u8; memory::io::SIZE]>,
    /// Interrupt controller (`I_STAT` / `I_MASK`). Accessed via the MMIO
    /// dispatch below and queried by the CPU each step to update
    /// `COP0.CAUSE.IP[2]`.
    irq: Irq,
}

impl Bus {
    /// Build a bus with the given BIOS image. RAM and scratchpad are
    /// zero-initialised; hardware leaves them in an undefined state, but
    /// zeroing is deterministic and adequate for a cold-boot harness.
    pub fn new(bios: Vec<u8>) -> Result<Self, BusError> {
        if bios.len() != memory::bios::SIZE {
            return Err(BusError::BiosSize {
                expected: memory::bios::SIZE,
                actual: bios.len(),
            });
        }

        let bios_arr: Box<[u8; memory::bios::SIZE]> = bios
            .into_boxed_slice()
            .try_into()
            .expect("size was just checked");

        Ok(Self {
            ram: zeroed_box(),
            bios: bios_arr,
            scratchpad: zeroed_box(),
            io: zeroed_box(),
            irq: Irq::new(),
        })
    }

    /// Borrow the interrupt controller — caller can `.raise()` sources
    /// or inspect state without going through MMIO.
    pub fn irq_mut(&mut self) -> &mut Irq {
        &mut self.irq
    }

    /// True when some source is both pending in `I_STAT` and enabled
    /// in `I_MASK`. The CPU mirrors this into `COP0.CAUSE.IP[2]`.
    pub fn external_interrupt_pending(&self) -> bool {
        self.irq.pending()
    }

    /// Read a 32-bit little-endian word from a virtual address.
    ///
    /// Panics on any address that does not resolve to a currently-mapped
    /// region. This is intentional — unmapped reads during development
    /// should surface immediately, not return silent zeros.
    pub fn read8(&self, virt: u32) -> u8 {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            return self.ram[(phys as usize) % memory::ram::SIZE];
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            return self.scratchpad[(phys - memory::scratchpad::BASE) as usize];
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return self.bios[(phys - memory::bios::BASE) as usize];
        }
        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            return self.io[(phys - memory::io::BASE) as usize];
        }
        if (memory::expansion2::BASE
            ..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        panic!("bus: unmapped read8 @ virt={virt:#010x} phys={phys:#010x}");
    }

    /// Read a 16-bit little-endian half-word from a virtual address.
    /// Unmapped regions behave identically to [`Bus::read8`] (see the
    /// region-by-region notes there).
    pub fn read16(&self, virt: u32) -> u16 {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            let off = (phys as usize) % memory::ram::SIZE;
            return u16::from_le_bytes([self.ram[off], self.ram[off + 1]]);
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let off = (phys - memory::scratchpad::BASE) as usize;
            return u16::from_le_bytes([self.scratchpad[off], self.scratchpad[off + 1]]);
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            let off = (phys - memory::bios::BASE) as usize;
            return u16::from_le_bytes([self.bios[off], self.bios[off + 1]]);
        }
        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let off = (phys - memory::io::BASE) as usize;
            return u16::from_le_bytes([self.io[off], self.io[off + 1]]);
        }
        if (memory::expansion2::BASE
            ..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        panic!("bus: unmapped read16 @ virt={virt:#010x} phys={phys:#010x}");
    }

    /// Read a 32-bit little-endian word from a virtual address. This is
    /// the instruction-fetch path.
    pub fn read32(&self, virt: u32) -> u32 {
        let phys = to_physical(virt);

        if phys < memory::ram::MIRROR_END {
            let offset = (phys as usize) % memory::ram::SIZE;
            return read_u32_le(&self.ram[offset..]);
        }

        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let offset = (phys - memory::scratchpad::BASE) as usize;
            return read_u32_le(&self.scratchpad[offset..]);
        }

        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::bios::BASE) as usize;
            return read_u32_le(&self.bios[offset..]);
        }

        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF_FFFF;
        }

        if phys == IRQ_STAT_ADDR {
            return self.irq.stat();
        }
        if phys == IRQ_MASK_ADDR {
            return self.irq.mask();
        }

        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::io::BASE) as usize;
            return read_u32_le(&self.io[offset..]);
        }

        panic!("bus: unmapped read32 @ virt={virt:#010x} phys={phys:#010x}");
    }

    /// Write a 32-bit little-endian word to a virtual address.
    ///
    /// - **RAM / scratchpad**: committed to the backing storage.
    /// - **BIOS ROM**: silently dropped (ROM is read-only).
    /// - **Cache-control register** `0xFFFE_0130`: silently dropped
    ///   until we model the I-cache.
    /// - **MMIO window** `0x1F80_1000..0x1F80_2000`: silently dropped
    ///   for now. Individual peripheral stubs will attach as we add
    ///   them; until then, BIOS's memory-controller init writes are
    ///   no-ops for architectural parity.
    pub fn write32(&mut self, virt: u32, value: u32) {
        if virt == memory::cache_control::ADDR {
            return;
        }

        let phys = to_physical(virt);

        if phys == IRQ_STAT_ADDR {
            self.irq.write_stat(value);
            return;
        }
        if phys == IRQ_MASK_ADDR {
            self.irq.write_mask(value);
            return;
        }

        let bytes = value.to_le_bytes();

        if phys < memory::ram::MIRROR_END {
            let offset = (phys as usize) % memory::ram::SIZE;
            self.ram[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let offset = (phys - memory::scratchpad::BASE) as usize;
            self.scratchpad[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::io::BASE) as usize;
            self.io[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }

        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }

        panic!(
            "bus: unmapped write32 @ virt={virt:#010x} phys={phys:#010x} value={value:#010x}"
        );
    }

    /// Write a byte to a virtual address. Unmapped writes in MMIO /
    /// expansion / BIOS ranges are silently dropped (same rationale as
    /// [`Bus::write32`]).
    pub fn write8(&mut self, virt: u32, value: u8) {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            self.ram[(phys as usize) % memory::ram::SIZE] = value;
            return;
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            self.scratchpad[(phys - memory::scratchpad::BASE) as usize] = value;
            return;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            self.io[(phys - memory::io::BASE) as usize] = value;
            return;
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }
        panic!("bus: unmapped write8 @ virt={virt:#010x} phys={phys:#010x} value={value:#04x}");
    }

    /// Write a 16-bit half-word to a virtual address. Same unmapped-region
    /// policy as [`Bus::write32`].
    pub fn write16(&mut self, virt: u32, value: u16) {
        let phys = to_physical(virt);
        let bytes = value.to_le_bytes();
        if phys < memory::ram::MIRROR_END {
            let off = (phys as usize) % memory::ram::SIZE;
            self.ram[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let off = (phys - memory::scratchpad::BASE) as usize;
            self.scratchpad[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let off = (phys - memory::io::BASE) as usize;
            self.io[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }
        panic!("bus: unmapped write16 @ virt={virt:#010x} phys={phys:#010x} value={value:#06x}");
    }
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn zeroed_box<const N: usize>() -> Box<[u8; N]> {
    // Allocates a zero-initialised slice and converts it. The try_into
    // cannot fail because the source slice has exactly N elements.
    vec![0u8; N]
        .into_boxed_slice()
        .try_into()
        .expect("vec length matches const N")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bios() -> Vec<u8> {
        // 512 KiB. First word is 0xDEADBEEF little-endian, then zeros.
        let mut bios = vec![0u8; memory::bios::SIZE];
        bios[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        bios
    }

    #[test]
    fn rejects_wrong_sized_bios() {
        assert!(matches!(
            Bus::new(vec![0u8; 1024]),
            Err(BusError::BiosSize { .. })
        ));
    }

    #[test]
    fn reads_first_bios_word_via_kseg1_reset_vector() {
        let bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(memory::bios::RESET_VECTOR), 0xDEAD_BEEF);
    }

    #[test]
    fn reads_first_bios_word_via_kseg0_and_kuseg() {
        // BIOS physical base mapped into KSEG0 (cached) and KUSEG.
        let bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(0x9FC0_0000), 0xDEAD_BEEF); // KSEG0
        assert_eq!(bus.read32(0x1FC0_0000), 0xDEAD_BEEF); // KUSEG physical alias
    }

    #[test]
    fn ram_starts_zeroed() {
        let bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(0x0000_0000), 0);
        assert_eq!(bus.read32(0x8000_0000), 0); // KSEG0 RAM
    }

    #[test]
    fn ram_mirrors_wrap_within_8mib() {
        // Hardware mirrors the 2 MiB RAM four times up to 0x0080_0000.
        // A write to offset 0 should be visible at +2 MiB, +4 MiB, +6 MiB.
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.ram[0..4].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        assert_eq!(bus.read32(0x0000_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0020_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0040_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0060_0000), 0x1122_3344);
    }
}
