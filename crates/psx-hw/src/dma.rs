//! DMA controller: 7 channels plus a global control/interrupt block.
//!
//! | Channel | Device  | Typical use                         |
//! |---------|---------|-------------------------------------|
//! | 0       | MDEC in | Compressed data → MDEC              |
//! | 1       | MDEC out| MDEC → RAM                          |
//! | 2       | GPU     | Rendering commands, VRAM transfers  |
//! | 3       | CD-ROM  | Sector data → RAM                   |
//! | 4       | SPU     | Voice data → SPU RAM                |
//! | 5       | PIO     | Expansion bus                       |
//! | 6       | OTC     | Ordering-table clear (GPU helper)   |
//!
//! Each channel has three registers at `BASE + ch*0x10 + {0, 4, 8}`:
//! base address, block control, channel control. The global control and
//! interrupt registers live at `DPCR` and `DICR`.
//!
//! To be populated: channel-control bitfield (`CHCR`: direction, step,
//! sync mode, chopping, trigger, enable), `DPCR` priority layout,
//! `DICR` bitfield (channel IRQ enable/flag, master flag).
//!
//! Reference: nocash PSX-SPX "DMA Channels" section.

/// DMA register base.
pub const BASE: u32 = 0x1F80_1080;

/// DMA control register: per-channel priority and enable.
pub const DPCR: u32 = 0x1F80_10F0;

/// DMA interrupt register: per-channel IRQ flags and master enable.
pub const DICR: u32 = 0x1F80_10F4;

/// Number of DMA channels.
pub const CHANNEL_COUNT: usize = 7;

/// Channel assignments, in hardware-defined order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Channel {
    /// Channel 0 -- MDEC input.
    MdecIn = 0,
    /// Channel 1 -- MDEC output.
    MdecOut = 1,
    /// Channel 2 -- GPU (rendering commands and VRAM transfers).
    Gpu = 2,
    /// Channel 3 -- CD-ROM sector data.
    Cdrom = 3,
    /// Channel 4 -- SPU voice data.
    Spu = 4,
    /// Channel 5 -- PIO / expansion.
    Pio = 5,
    /// Channel 6 -- ordering-table clear (OTC).
    Otc = 6,
}
