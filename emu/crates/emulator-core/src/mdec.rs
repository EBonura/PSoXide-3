//! MDEC — Motion Decoder. Hardware MPEG-ish macroblock decoder.
//!
//! Games use the MDEC to play back pre-compressed FMV (cutscenes,
//! intros, attract-mode loops). The pipeline is:
//!
//! 1. CPU ships a command word (`0x1F80_1820`) saying "decode N
//!    macroblocks, output colour depth = X".
//! 2. CPU ships `N × M` parameter words (Huffman-packed frequency
//!    coefficients for each macroblock, interleaved Y/Cb/Cr).
//! 3. MDEC IDCT's them, converts YUV→RGB, and streams the decoded
//!    pixel words back. Games normally drain via DMA channel 1
//!    (MDEC-out) straight into VRAM, where the GPU displays them.
//!
//! This module is a **defensive stub** right now — it doesn't decode
//! macroblocks, just models the MMIO register shape plausibly enough
//! that games polling MDEC status see "idle, empty, not requesting"
//! instead of the unmapped-fallback 0xFFFF_FFFF. Real IDCT + Huffman
//! + YUV→RGB comes in a dedicated session (it's another
//! 1000-line subsystem when done right).
//!
//! MMIO map:
//!
//! ```text
//!   0x1F80_1820 R/W : command / parameter FIFO
//!   0x1F80_1824 R/W : status read / control write
//! ```
//!
//! Reference: PSX-SPX "Macroblock Decoder (MDEC)".

/// Base address of the MDEC MMIO port.
pub const MDEC_BASE: u32 = 0x1F80_1820;
/// Command / parameter register (writes issue commands or deliver
/// parameter words; reads return queued output pixels).
pub const MDEC_CMD_DATA: u32 = 0x1F80_1820;
/// Status register (read) / control register (write).
pub const MDEC_CTRL_STAT: u32 = 0x1F80_1824;

/// MDEC state. Holds enough of the status register to return
/// plausible values when software polls; real decoding isn't wired
/// yet.
#[derive(Default)]
pub struct Mdec {
    /// Last raw status word we reported. Kept so reset + control
    /// writes round-trip observably even before we do any decoding.
    /// Bit layout per PSX-SPX:
    ///   31   : Data-Out FIFO Empty (1 = empty)
    ///   30   : Data-In FIFO Full (1 = full)
    ///   29   : Command Busy (1 = busy)
    ///   28   : Data-In Request via DMA0
    ///   27   : Data-Out Request via DMA1
    ///   26-25: Output Depth (00=4bpp, 01=8bpp, 10=24bpp, 11=15bpp)
    ///   24   : Output Signed
    ///   23   : Output Bit-15
    ///   18-16: Current block (Y1..Y4, Cr, Cb)
    ///   15-0 : Words remaining in parameter FIFO minus 1
    ///
    /// Reset default (status_idle()) reports "empty out, not full in,
    /// not busy, no requests".
    status: u32,
    /// Enable DMA0 (data-in) — bit 30 of the last control write.
    /// Bookkeeping only; DMA0 isn't wired to deliver words yet.
    dma_in_enabled: bool,
    /// Enable DMA1 (data-out) — bit 29 of the last control write.
    dma_out_enabled: bool,
    /// Diagnostic: raw command words seen since reset. Games might
    /// ship thousands; we just count so probes can tell "MDEC was
    /// spoken to" vs "MDEC was ignored".
    commands_seen: u64,
    /// Diagnostic: raw parameter-data words seen since reset
    /// (everything the CPU wrote to the command port after a
    /// command had been issued). Useful for "how much FMV did the
    /// game ship?" queries without touching the full payload.
    params_seen: u64,
}

impl Mdec {
    /// Fresh MDEC in its post-reset state: data-out FIFO empty, not
    /// busy, no DMA requests, parameter FIFO remaining = 0.
    pub fn new() -> Self {
        Self {
            status: status_idle(),
            ..Self::default()
        }
    }

    /// True when `phys` lies inside the MDEC MMIO block (the two
    /// 32-bit registers at `0x1F80_1820` and `0x1F80_1824`).
    pub fn contains(phys: u32) -> bool {
        (MDEC_BASE..MDEC_BASE + 8).contains(&phys)
    }

    /// Read a 32-bit word from an MDEC register.
    pub fn read32(&self, phys: u32) -> u32 {
        match phys & 0x1F80_1FFF {
            MDEC_CMD_DATA => {
                // Data output read. Real hardware returns the next
                // decoded pixel word; our stub doesn't decode
                // anything so the queue is always empty.
                0
            }
            MDEC_CTRL_STAT => self.status,
            _ => 0,
        }
    }

    /// Write a 32-bit word to an MDEC register. Commands + data
    /// arrive at `0x1820`, control bits at `0x1824`.
    pub fn write32(&mut self, phys: u32, value: u32) {
        match phys & 0x1F80_1FFF {
            MDEC_CMD_DATA => {
                // Commands / parameter data. We just tally them.
                // Top nibble identifies the command when the word
                // is a command header (low 28 bits are parameter
                // count / flags); otherwise it's raw parameter
                // data. For the stub we don't care which it is.
                let is_command = (value >> 29) & 0x7 != 0;
                if is_command {
                    self.commands_seen = self.commands_seen.saturating_add(1);
                } else {
                    self.params_seen = self.params_seen.saturating_add(1);
                }
            }
            MDEC_CTRL_STAT => {
                if value & (1 << 31) != 0 {
                    // Bit 31 write = reset. PSX-SPX: clears
                    // command, FIFOs, and blockiness counters.
                    self.status = status_idle();
                    self.dma_in_enabled = false;
                    self.dma_out_enabled = false;
                    return;
                }
                self.dma_in_enabled = value & (1 << 30) != 0;
                self.dma_out_enabled = value & (1 << 29) != 0;
            }
            _ => {}
        }
    }

    /// Are DMA channel 0 (in) or 1 (out) enabled? Bus uses these
    /// bits when routing DMA transfers — when they're clear, a DMA
    /// trigger on the MDEC channels is a no-op.
    pub fn dma_in_enabled(&self) -> bool {
        self.dma_in_enabled
    }

    /// See [`Mdec::dma_in_enabled`].
    pub fn dma_out_enabled(&self) -> bool {
        self.dma_out_enabled
    }

    /// Diagnostic — total command words the CPU has shipped to
    /// the MDEC since reset.
    pub fn commands_seen(&self) -> u64 {
        self.commands_seen
    }

    /// Diagnostic — total parameter words (non-command writes to
    /// `0x1F80_1820`) seen since reset.
    pub fn params_seen(&self) -> u64 {
        self.params_seen
    }
}

/// Default MDEC status word — bit 31 set (data-out FIFO empty),
/// everything else clear. That's the idle, ready-for-commands state
/// the BIOS observes right after a GP1 reset + MDEC reset.
const fn status_idle() -> u32 {
    1 << 31
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_covers_both_registers() {
        assert!(Mdec::contains(0x1F80_1820));
        assert!(Mdec::contains(0x1F80_1823));
        assert!(Mdec::contains(0x1F80_1824));
        assert!(Mdec::contains(0x1F80_1827));
        assert!(!Mdec::contains(0x1F80_181C));
        assert!(!Mdec::contains(0x1F80_1828));
    }

    #[test]
    fn fresh_status_reports_idle_and_empty() {
        let m = Mdec::new();
        let stat = m.read32(MDEC_CTRL_STAT);
        // Bit 31 (data-out empty) set.
        assert_eq!(stat & (1 << 31), 1 << 31);
        // Busy (bit 29) clear.
        assert_eq!(stat & (1 << 29), 0);
        // No DMA requests (bits 27, 28).
        assert_eq!(stat & (0b11 << 27), 0);
    }

    #[test]
    fn data_port_read_returns_zero_when_queue_empty() {
        let m = Mdec::new();
        assert_eq!(m.read32(MDEC_CMD_DATA), 0);
    }

    #[test]
    fn control_write_reset_bit_clears_state() {
        let mut m = Mdec::new();
        // Enable both DMAs + scramble status.
        m.write32(MDEC_CTRL_STAT, 0x6000_0000);
        assert!(m.dma_in_enabled());
        assert!(m.dma_out_enabled());
        // Reset bit clears them.
        m.write32(MDEC_CTRL_STAT, 0x8000_0000);
        assert!(!m.dma_in_enabled());
        assert!(!m.dma_out_enabled());
        assert_eq!(m.read32(MDEC_CTRL_STAT), status_idle());
    }

    #[test]
    fn control_write_latches_dma_enables() {
        let mut m = Mdec::new();
        m.write32(MDEC_CTRL_STAT, 0x4000_0000); // DMA0 enable only
        assert!(m.dma_in_enabled());
        assert!(!m.dma_out_enabled());
        m.write32(MDEC_CTRL_STAT, 0x2000_0000); // DMA1 enable only
        assert!(!m.dma_in_enabled());
        assert!(m.dma_out_enabled());
    }

    #[test]
    fn data_write_tallies_commands_vs_parameters() {
        let mut m = Mdec::new();
        // Command word — top 3 bits (29..31) non-zero.
        m.write32(MDEC_CMD_DATA, 0x2000_0001);
        // Parameter word — top 3 bits zero.
        m.write32(MDEC_CMD_DATA, 0x0000_FFFF);
        m.write32(MDEC_CMD_DATA, 0x1234_ABCD);
        assert_eq!(m.commands_seen(), 1);
        assert_eq!(m.params_seen(), 2);
    }
}
