//! DMA controller MMIO.
//!
//! The PS1 has 7 DMA channels (MDEC-in, MDEC-out, GPU, CD-ROM, SPU,
//! PIO, OTC). Each channel has three registers — `MADR` (memory
//! address), `BCR` (block count), `CHCR` (control) — at fixed
//! 16-byte strides starting at `0x1F80_1080`. Plus global `DPCR`
//! (priority/enable) at `0x1F80_10F0` and `DICR` (IRQ) at `0x1F80_10F4`.

/// Channel index 0..=6 in the order the DMA controller presents them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Channel {
    /// MDEC→RAM (decoded macroblocks in).
    MdecIn = 0,
    /// RAM→MDEC (raw DCT blocks out).
    MdecOut = 1,
    /// RAM↔GPU.
    Gpu = 2,
    /// CD-ROM drive → RAM.
    Cdrom = 3,
    /// SPU ↔ RAM.
    Spu = 4,
    /// PIO (expansion port).
    Pio = 5,
    /// OTC (ordering-table clear).
    Otc = 6,
}

impl Channel {
    /// Base MMIO address of this channel's register block.
    #[inline(always)]
    pub const fn base(self) -> u32 {
        0x1F80_1080 + 0x10 * (self as u32)
    }

    /// Bit position in `DPCR` for this channel's enable flag.
    #[inline(always)]
    pub const fn dpcr_enable_bit(self) -> u32 {
        3 + 4 * (self as u32)
    }
}

/// Global DMA priority / enable register.
pub const DPCR: u32 = 0x1F80_10F0;
/// Global DMA interrupt / completion-flag register.
pub const DICR: u32 = 0x1F80_10F4;

// --- Per-channel offsets ---------------------------------------------------

const MADR_OFF: u32 = 0x0;
const BCR_OFF: u32 = 0x4;
const CHCR_OFF: u32 = 0x8;

// --- CHCR bits used by SDK callers -----------------------------------------

/// CHCR.0 — direction: 0 = device→RAM, 1 = RAM→device.
pub const CHCR_TO_DEVICE: u32 = 1 << 0;
/// CHCR.1 — step: 0 = +4 MADR, 1 = -4 MADR.
pub const CHCR_STEP_BACKWARD: u32 = 1 << 1;
/// CHCR.8 — chopping enable (DMA yields for CPU periodically).
pub const CHCR_CHOPPING_ENABLE: u32 = 1 << 8;

/// CHCR.9..10 — sync mode: manual (0), block (1), linked-list (2).
pub const CHCR_SYNC_MANUAL: u32 = 0 << 9;
/// Block-mode sync: BCR = block-count × block-size (in words).
pub const CHCR_SYNC_BLOCK: u32 = 1 << 9;
/// Linked-list sync: walks a chain of packet headers in RAM.
pub const CHCR_SYNC_LINKED: u32 = 2 << 9;

/// CHCR.24 — start the transfer (busy while set).
pub const CHCR_START: u32 = 1 << 24;
/// CHCR.28 — manual-mode trigger (self-clears when transfer begins).
pub const CHCR_TRIGGER: u32 = 1 << 28;

// --- Register access helpers ----------------------------------------------

/// Write `MADR` (memory address that DMA will source from or drain to).
#[inline(always)]
pub fn set_madr(ch: Channel, addr: u32) {
    unsafe { crate::write32(ch.base() + MADR_OFF, addr) }
}

/// Read `MADR`.
#[inline(always)]
pub fn madr(ch: Channel) -> u32 {
    unsafe { crate::read32(ch.base() + MADR_OFF) }
}

/// Write `BCR` in manual / linked-list mode: just a 16-bit word count.
/// For block-slice mode use [`set_bcr_block`].
#[inline(always)]
pub fn set_bcr_manual(ch: Channel, words: u16) {
    unsafe { crate::write32(ch.base() + BCR_OFF, words as u32) }
}

/// Write `BCR` in block-slice mode:
/// `BS × BA = blocks of block_size words`.
#[inline(always)]
pub fn set_bcr_block(ch: Channel, block_size: u16, block_count: u16) {
    unsafe {
        crate::write32(
            ch.base() + BCR_OFF,
            (block_size as u32) | ((block_count as u32) << 16),
        )
    }
}

/// Write `CHCR` (control). Starts the transfer if `CHCR_START` is set.
#[inline(always)]
pub fn set_chcr(ch: Channel, value: u32) {
    unsafe { crate::write32(ch.base() + CHCR_OFF, value) }
}

/// Read `CHCR`.
#[inline(always)]
pub fn chcr(ch: Channel) -> u32 {
    unsafe { crate::read32(ch.base() + CHCR_OFF) }
}

/// True while the channel is busy with an in-flight transfer.
#[inline(always)]
pub fn is_busy(ch: Channel) -> bool {
    chcr(ch) & CHCR_START != 0
}

/// Enable a channel in `DPCR` without disturbing the others.
pub fn enable_channel(ch: Channel) {
    let dpcr = unsafe { crate::read32(DPCR) };
    unsafe { crate::write32(DPCR, dpcr | (1 << ch.dpcr_enable_bit())) }
}

/// OTC-channel helper: clears a RAM block as a reverse-linked chain
/// the GPU-DMA walker consumes. The last word is the terminator
/// `0x00FF_FFFF`; previous words are "previous address & 0x1FFFFFFF".
///
/// Convenience wrapper: sets up MADR/BCR/CHCR and blocks until done.
pub fn clear_ordering_table(buf: *mut u32, words: u16) {
    set_madr(Channel::Otc, buf as u32);
    set_bcr_manual(Channel::Otc, words);
    // OTC clear: direction backward (step -4), manual sync, trigger bit.
    set_chcr(
        Channel::Otc,
        CHCR_STEP_BACKWARD | CHCR_SYNC_MANUAL | CHCR_START | CHCR_TRIGGER,
    );
    while is_busy(Channel::Otc) {}
}
