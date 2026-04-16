//! PS1 memory map.
//!
//! The MIPS R3000A's 4 GiB virtual address space is divided into four
//! segments. On PS1, three of them mirror the same 512 MiB physical
//! address space; only the cache hint changes.
//!
//! | Segment | Virtual range              | Cached | Physical mapping |
//! |---------|----------------------------|--------|------------------|
//! | KUSEG   | `0x0000_0000..0x8000_0000` | yes    | `v & 0x1FFF_FFFF`|
//! | KSEG0   | `0x8000_0000..0xA000_0000` | yes    | `v & 0x1FFF_FFFF`|
//! | KSEG1   | `0xA000_0000..0xC000_0000` | no     | `v & 0x1FFF_FFFF`|
//! | KSEG2   | `0xC000_0000..0xFFFF_FFFF` | n/a    | cache control only |
//!
//! Reference: nocash PSX-SPX "Memory Map" section.

/// Strip the segment bits from a virtual address to get the physical address.
///
/// Valid for KUSEG, KSEG0, and KSEG1. KSEG2 addresses (≥ `0xC000_0000`)
/// do not participate in normal memory accesses — they are reserved for
/// the cache-control register.
///
/// ```
/// use psx_hw::memory::to_physical;
/// assert_eq!(to_physical(0x0000_1234), 0x0000_1234); // KUSEG
/// assert_eq!(to_physical(0x8000_1234), 0x0000_1234); // KSEG0
/// assert_eq!(to_physical(0xA000_1234), 0x0000_1234); // KSEG1
/// assert_eq!(to_physical(0xBFC0_0000), 0x1FC0_0000); // BIOS via KSEG1
/// ```
#[inline]
pub const fn to_physical(virt: u32) -> u32 {
    virt & 0x1FFF_FFFF
}

/// Main RAM: 2 MiB starting at physical `0x0000_0000`.
pub mod ram {
    /// Physical base address.
    pub const BASE: u32 = 0x0000_0000;
    /// Size in bytes (2 MiB).
    pub const SIZE: usize = 2 * 1024 * 1024;
    /// Hardware-mirrored range: 2 MiB of physical RAM repeats four times,
    /// filling `0x0000_0000..0x0080_0000`.
    pub const MIRROR_END: u32 = 0x0080_0000;
}

/// Expansion Region 1: up to 8 MiB for parallel-port carts.
///
/// Unused by stock hardware; reads return `0xFF`.
pub mod expansion1 {
    /// Physical base address.
    pub const BASE: u32 = 0x1F00_0000;
    /// Maximum size (configurable via `EXP1_DELAY_SIZE` register).
    pub const SIZE: usize = 8 * 1024 * 1024;
}

/// Scratchpad RAM: 1 KiB of fast CPU-local memory.
///
/// Lives inside the D-cache; accessible only through KUSEG or KSEG0, never
/// KSEG1 (uncached access to scratchpad is undefined).
pub mod scratchpad {
    /// Physical base address.
    pub const BASE: u32 = 0x1F80_0000;
    /// Size in bytes (1 KiB).
    pub const SIZE: usize = 1024;
}

/// Hardware I/O registers.
///
/// Every MMIO register lives in this 4 KiB window. Individual register
/// addresses are defined in the module for their owning hardware block
/// (`gpu`, `spu`, `cdrom`, …).
pub mod io {
    /// Physical base address.
    pub const BASE: u32 = 0x1F80_1000;
    /// Size in bytes (4 KiB). Expansion 2 begins immediately after at
    /// `0x1F80_2000`.
    pub const SIZE: usize = 4 * 1024;
}

/// Expansion Region 2: 8 KiB debug / dev-kit port.
pub mod expansion2 {
    /// Physical base address.
    pub const BASE: u32 = 0x1F80_2000;
    /// Size in bytes (8 KiB).
    pub const SIZE: usize = 8 * 1024;
    /// POST status byte: BIOS writes progress codes here during boot.
    pub const POST: u32 = 0x1F80_2041;
}

/// Expansion Region 3: 2 MiB, rarely used.
pub mod expansion3 {
    /// Physical base address.
    pub const BASE: u32 = 0x1FA0_0000;
    /// Size in bytes (2 MiB).
    pub const SIZE: usize = 2 * 1024 * 1024;
}

/// BIOS ROM: 512 KiB.
///
/// Conventionally accessed via KSEG1 at `0xBFC0_0000` so that the initial
/// boot sequence runs uncached (important because the I-cache starts in
/// an undefined state at reset).
pub mod bios {
    /// Physical base address.
    pub const BASE: u32 = 0x1FC0_0000;
    /// Size in bytes (512 KiB).
    pub const SIZE: usize = 512 * 1024;
    /// Canonical reset vector: KSEG1 view of BIOS base.
    pub const RESET_VECTOR: u32 = 0xBFC0_0000;
}

/// Cache control register at `0xFFFE_0130`.
///
/// The only KSEG2 address actually used by PS1 software. Controls the
/// I-cache enable, scratchpad enable, and a handful of debug bits.
pub mod cache_control {
    /// Virtual address. Does not follow the KUSEG/KSEG0/KSEG1 mirror rule.
    pub const ADDR: u32 = 0xFFFE_0130;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bios_reset_vector_resolves_to_physical_bios_base() {
        assert_eq!(to_physical(bios::RESET_VECTOR), bios::BASE);
    }

    #[test]
    fn io_window_fits_inside_scratchpad_to_expansion2_gap() {
        assert!(io::BASE + io::SIZE as u32 <= expansion2::BASE);
        assert!(scratchpad::BASE + scratchpad::SIZE as u32 <= io::BASE);
    }

    #[test]
    fn ram_mirrors_fill_expected_window() {
        assert_eq!(ram::SIZE as u32 * 4, ram::MIRROR_END);
    }
}
