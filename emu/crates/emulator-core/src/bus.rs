//! System bus: owns physical memory and dispatches loads to regions.
//!
//! Current coverage: RAM, BIOS, scratchpad. Everything else panics on
//! access — deliberately, because we want unmapped reads to be loud
//! until each region's owning module (GPU, SPU, CD-ROM, …) is wired up.

use psx_hw::memory::{self, to_physical};
use thiserror::Error;

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
        })
    }

    /// Read a 32-bit little-endian word from a virtual address.
    ///
    /// Panics on any address that does not resolve to a currently-mapped
    /// region. This is intentional — unmapped reads during development
    /// should surface immediately, not return silent zeros.
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
            // MMIO: accepted silently until real peripherals attach.
            return;
        }

        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }

        panic!(
            "bus: unmapped write32 @ virt={virt:#010x} phys={phys:#010x} value={value:#010x}"
        );
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
