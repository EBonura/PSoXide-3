//! MDEC -- Motion Decoder. Hardware MPEG-ish macroblock decoder.
//!
//! Games use the MDEC to play back pre-compressed FMV (cutscenes,
//! intros, attract-mode loops). The pipeline is:
//!
//! 1. CPU uploads quantization tables via DMA0 with command 0x4.
//! 2. CPU issues a decode command (0x3) to `0x1F80_1820` with block count.
//! 3. CPU streams N×M run-length-encoded coefficient halfwords into
//!    the MDEC via DMA0 (CPU→MDEC, channel 0).
//! 4. MDEC dequantizes, IDCT's, and YUV→RGB converts each macroblock
//!    into either 15-bit or 24-bit pixel output.
//! 5. CPU pulls decoded pixels out via DMA1 (MDEC→CPU, channel 1),
//!    which streams 256 pixels (16×16) per macroblock into RAM --
//!    typically destined for VRAM via a follow-up GPU draw.
//!
//! Reference implementations consulted:
//! - PCSX-Redux `src/core/mdec.{h,cc}` (GPL-2+) for the AAN IDCT +
//!   YUV→RGB pipeline + scaling constants.
//! - PSX-SPX "Macroblock Decoder (MDEC)" chapter for register semantics.
//!
//! MMIO map:
//!
//! ```text
//!   0x1F80_1820 R/W : command / parameter FIFO (write: commands + RLE data,
//!                                               read:  decoded pixel words)
//!   0x1F80_1824 R/W : status read / control write
//! ```
//!
//! The CPU never pokes RLE data through the FIFO directly -- it always
//! goes through DMA0 for speed (at typical FMV rates the CPU would spend
//! all its time shipping data otherwise). We expose `dma_write_in` +
//! `dma_read_out` entry points for the bus to call on DMA channel 0 /
//! channel 1 triggers.

// ===============================================================
//  Register addresses + command constants.
// ===============================================================

/// Base address of the MDEC MMIO port.
pub const MDEC_BASE: u32 = 0x1F80_1820;
/// Command / parameter register (writes issue commands or deliver
/// parameter words; reads return queued output pixels).
pub const MDEC_CMD_DATA: u32 = 0x1F80_1820;
/// Status register (read) / control register (write).
pub const MDEC_CTRL_STAT: u32 = 0x1F80_1824;

// Command field in `reg0`:
//   31..28 : command code (3 = decode, 4 = load quantization, 6 = cosine table)
//   27     : (cmd 3 only) output bpp (1 = 15-bit, 0 = 24-bit)
//   25     : (cmd 3 only) set bit 15 on 15-bit output pixels
//   15..0  : parameter count (words) for the command

/// Command 0x3 STP flag -- sets the 15-bit mask bit (bit 15 of each RGB word).
const MDEC0_STP: u32 = 0x0200_0000;
/// Command 0x3 RGB24 flag name follows Redux: set selects 15-bit, clear selects 24-bit.
const MDEC0_RGB24: u32 = 0x0800_0000;

// Status register (`reg1`) bits:
//   31    : Data-Out FIFO Empty
//   30    : Data-In FIFO Full
//   29    : Command Busy (decode in progress)
//   28    : Data-In Request via DMA0
//   27    : Data-Out Request via DMA1
//   26..25: Output Depth (00=4bpp, 01=8bpp, 10=24bpp, 11=15bpp)
//   24    : Output Signed
//   23    : Output Bit-15
//   18..16: Current block (Y1..Y4, Cr, Cb)
//   15..0 : Words remaining in parameter FIFO minus 1

const MDEC1_BUSY: u32 = 0x2000_0000;
#[allow(dead_code)]
const MDEC1_DREQ: u32 = 0x1800_0000;
#[allow(dead_code)]
const MDEC1_FIFO: u32 = 0xC000_0000;
#[allow(dead_code)]
const MDEC1_RGB24: u32 = 0x0200_0000;
#[allow(dead_code)]
const MDEC1_STP: u32 = 0x0080_0000;
const MDEC1_RESET: u32 = 0x8000_0000;
/// Data-Out FIFO Empty bit.
#[allow(dead_code)]
const MDEC1_EMPTY: u32 = 0x8000_0000;
/// Command Busy bit -- set while a decode is in progress.
#[allow(dead_code)]
const MDEC1_COMMAND_BUSY: u32 = 0x2000_0000;
/// Data-Out Request via DMA1.
#[allow(dead_code)]
const MDEC1_DMA_OUT_REQ: u32 = 0x0800_0000;
/// Data-In Request via DMA0.
#[allow(dead_code)]
const MDEC1_DMA_IN_REQ: u32 = 0x1000_0000;
#[allow(dead_code)]
const MDEC1_OUTPUT_DEPTH_MASK: u32 = 0x0600_0000;
#[allow(dead_code)]
const MDEC1_OUTPUT_SIGNED: u32 = 0x0100_0000;
#[allow(dead_code)]
const MDEC1_OUTPUT_BIT15: u32 = 0x0080_0000;

/// End-of-data sentinel in an RLE coefficient stream.
const MDEC_END_OF_DATA: u16 = 0xFE00;

/// Block size constants -- 8×8 DCT blocks, 6 blocks per macroblock
/// (Cb, Cr, Y1, Y2, Y3, Y4).
const DSIZE: usize = 8;
const DSIZE2: usize = DSIZE * DSIZE;
const BLOCKS_PER_MACROBLOCK: usize = 6;

// ===============================================================
//  Scaling constants (AAN IDCT).
// ===============================================================

const AAN_CONST_BITS: i32 = 12;
const AAN_PRESCALE_BITS: i32 = 16;
const AAN_CONST_SIZE: i32 = 24;
const AAN_CONST_SCALE: i32 = AAN_CONST_SIZE - AAN_CONST_BITS;
const AAN_PRESCALE_SIZE: i32 = 20;
const AAN_PRESCALE_SCALE: i32 = AAN_PRESCALE_SIZE - AAN_PRESCALE_BITS;
const AAN_EXTRA: i32 = 12;

/// `SCALER(x, n) = ((x) + ((1 << n) >> 1)) >> n` -- rounded divide by 2^n.
#[inline]
fn scaler(x: i32, n: i32) -> i32 {
    (x + ((1 << n) >> 1)) >> n
}

#[inline]
fn scale(x: i32, n: i32) -> i32 {
    x >> n
}

#[inline]
fn muls(v: i32, c: i32) -> i32 {
    scale(v.wrapping_mul(c), AAN_CONST_BITS)
}

// Pre-scaled IDCT constants.
fn fix_1_082392200() -> i32 {
    scaler(18_159_528, AAN_CONST_SCALE)
}
fn fix_1_414213562() -> i32 {
    scaler(23_726_566, AAN_CONST_SCALE)
}
fn fix_1_847759065() -> i32 {
    scaler(31_000_253, AAN_CONST_SCALE)
}
fn fix_2_613125930() -> i32 {
    scaler(43_840_978, AAN_CONST_SCALE)
}

/// Zig-zag scan order -- maps sequential coefficient index (0..63) to
/// the position in the 8×8 block where it belongs. RLE-encoded
/// coefficients stream in this order; the decoder sprays them out of
/// zigzag into row-major before IDCT.
const ZIG_ZAG_SCAN: [usize; DSIZE2] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// AAN prescaled forward-DCT coefficients. Multiplied into the
/// quantization tables during upload so the IDCT can be table-driven.
const AAN_SCALES: [i32; DSIZE2] = [
    1_048_576, 1_454_417, 1_370_031, 1_232_995, 1_048_576, 823_861, 567_485, 289_301, 1_454_417,
    2_017_334, 1_900_287, 1_710_213, 1_454_417, 1_142_728, 787_125, 401_273, 1_370_031, 1_900_287,
    1_790_031, 1_610_986, 1_370_031, 1_076_426, 741_455, 377_991, 1_232_995, 1_710_213, 1_610_986,
    1_449_849, 1_232_995, 968_758, 667_292, 340_183, 1_048_576, 1_454_417, 1_370_031, 1_232_995,
    1_048_576, 823_861, 567_485, 289_301, 823_861, 1_142_728, 1_076_426, 968_758, 823_861, 647_303,
    445_870, 227_303, 567_485, 787_125, 741_455, 667_292, 567_485, 445_870, 307_121, 156_569,
    289_301, 401_273, 377_991, 340_183, 289_301, 227_303, 156_569, 79_818,
];

/// Bus lines MDEC can report data-in / data-out requests on. Lets the
/// bus decide whether a DMA trigger should fire right now or wait.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MdecState {
    /// Idle -- ready for a new command.
    Idle,
    /// Accepting parameter / RLE words through DMA0.
    AwaitingData,
    /// Decoding -- output pixels are available via DMA1.
    DecodeReady,
}

// ===============================================================
//  MDEC state.
// ===============================================================

/// Full MDEC subsystem state. Owns quantization tables, a decode
/// output buffer, and the register pair visible to software.
pub struct Mdec {
    /// Last write to the command register (`reg0`).
    reg0: u32,
    /// Current status register (`reg1`). PCSX-Redux exposes the latched
    /// value directly rather than synthesizing empty/DREQ/format bits
    /// when software reads the status port.
    reg1: u32,
    /// Luminance quantization table (64 × 16-bit pre-scaled).
    iq_y: [i32; DSIZE2],
    /// Chrominance quantization table.
    iq_uv: [i32; DSIZE2],
    /// Buffered RLE halfwords received via DMA0 since the decode
    /// command was issued. Drained during decode_macroblocks.
    rl_queue: std::collections::VecDeque<u16>,
    /// Decoded pixel output queue -- ready for DMA1 to pull. Format
    /// depends on `MDEC0_RGB24`: 16-bit halfwords for 15-bit output,
    /// packed 8-bit bytes for 24-bit output. We always queue as
    /// halfwords and let the read path reinterpret.
    out_queue: std::collections::VecDeque<u16>,
    /// Current decode command's RLE word count (from the low 16 bits
    /// of reg0). We stop decoding when we've consumed that many
    /// parameter words.
    #[allow(dead_code)]
    expected_param_words: u32,
    /// Diagnostic: raw command words seen since reset. Games might
    /// ship thousands; we just count so probes can tell "MDEC was
    /// spoken to" vs "MDEC was ignored".
    commands_seen: u64,
    /// Diagnostic: raw parameter-data words seen since reset.
    params_seen: u64,
    /// Enable DMA0 (data-in) -- bit 30 of the last control write.
    dma_in_enabled: bool,
    /// Enable DMA1 (data-out) -- bit 29 of the last control write.
    dma_out_enabled: bool,
    /// Diagnostic: total macroblocks decoded since reset.
    macroblocks_decoded: u64,
    /// Recent command-register writes, newest at the end.
    command_history: Vec<u32>,
}

impl Default for Mdec {
    fn default() -> Self {
        Self::new()
    }
}

impl Mdec {
    /// Freshly-reset MDEC in its post-reset state.
    pub fn new() -> Self {
        Self {
            reg0: 0,
            reg1: status_idle(),
            iq_y: [0; DSIZE2],
            iq_uv: [0; DSIZE2],
            rl_queue: std::collections::VecDeque::new(),
            out_queue: std::collections::VecDeque::new(),
            expected_param_words: 0,
            commands_seen: 0,
            params_seen: 0,
            dma_in_enabled: false,
            dma_out_enabled: false,
            macroblocks_decoded: 0,
            command_history: Vec::new(),
        }
    }

    /// True when `phys` lies inside the MDEC MMIO block.
    pub fn contains(phys: u32) -> bool {
        (MDEC_BASE..MDEC_BASE + 8).contains(&phys)
    }

    /// Read a 32-bit word from an MDEC register.
    pub fn read32(&mut self, phys: u32) -> u32 {
        match phys & 0x1F80_1FFF {
            MDEC_CMD_DATA => self.reg0,
            MDEC_CTRL_STAT => self.status_word(),
            _ => 0,
        }
    }

    /// Write a 32-bit word to an MDEC register. Commands + data
    /// arrive at `0x1820`, control bits at `0x1824`.
    pub fn write32(&mut self, phys: u32, value: u32) {
        match phys & 0x1F80_1FFF {
            MDEC_CMD_DATA => self.command_write(value),
            MDEC_CTRL_STAT => self.control_write(value),
            _ => {}
        }
    }

    /// Called by the bus for DMA channel 0 transfers -- CPU→MDEC.
    /// Each word becomes two halfwords in little-endian order.
    pub fn dma_write_in(&mut self, words: &[u32]) {
        self.params_seen = self.params_seen.saturating_add(words.len() as u64);
        self.reg1 |= MDEC1_STP;

        match self.command_code() {
            0x3 => {
                self.reg1 |= MDEC1_BUSY;
                self.rl_queue.clear();
                self.out_queue.clear();
                for &w in words {
                    self.rl_queue.push_back(w as u16);
                    self.rl_queue.push_back((w >> 16) as u16);
                }
                if !self.can_continue_decode() {
                    self.reg1 &= !(MDEC1_BUSY | MDEC1_STP);
                }
            }
            0x4 => {
                self.absorb_quant_upload(words);
            }
            0x6 => {}
            _ => {}
        }
    }

    /// Called by the bus for DMA channel 1 transfers -- MDEC→CPU.
    /// Fills `out` with decoded pixel words.
    pub fn dma_read_out(&mut self, out: &mut [u32]) {
        for slot in out {
            if self.out_queue.len() < 2 {
                self.decode_until_output_words(2);
            }
            *slot = self.pop_output_word();
        }
    }

    /// Called by the bus when DMA channel 1's scheduled completion
    /// fires. Redux keeps MDEC-in DMA busy for decode commands until
    /// the output side has consumed the decoded frame; this returns
    /// `true` exactly when channel 0 may be completed alongside
    /// channel 1.
    pub fn complete_dma_out(&mut self) -> bool {
        if self.command_code() == 0x3
            && self.out_queue.is_empty()
            && (self.rl_queue.is_empty()
                || self
                    .rl_queue
                    .front()
                    .is_some_and(|&word| word == MDEC_END_OF_DATA))
        {
            self.reg1 &= !(MDEC1_BUSY | MDEC1_STP);
            self.rl_queue.clear();
            return true;
        }
        false
    }

    /// Current coarse state -- diagnostic, for UI display / debug.
    pub fn state(&self) -> MdecState {
        if !self.out_queue.is_empty() {
            MdecState::DecodeReady
        } else if self.reg1 & MDEC1_BUSY != 0 {
            MdecState::AwaitingData
        } else {
            MdecState::Idle
        }
    }

    /// Are DMA channel 0 (in) or 1 (out) enabled?
    pub fn dma_in_enabled(&self) -> bool {
        self.dma_in_enabled
    }

    /// See [`Mdec::dma_in_enabled`].
    pub fn dma_out_enabled(&self) -> bool {
        self.dma_out_enabled
    }

    /// Diagnostic -- total command words the CPU has shipped.
    pub fn commands_seen(&self) -> u64 {
        self.commands_seen
    }

    /// Diagnostic -- total parameter words seen.
    pub fn params_seen(&self) -> u64 {
        self.params_seen
    }

    /// Diagnostic -- total macroblocks fully decoded since reset.
    pub fn macroblocks_decoded(&self) -> u64 {
        self.macroblocks_decoded
    }

    /// True when decoded pixel words are ready for DMA channel 1.
    pub fn output_ready(&self) -> bool {
        !self.out_queue.is_empty()
    }

    /// True when channel 1 can make forward progress, either from the
    /// decoded FIFO or by decoding another macroblock on demand.
    pub fn can_dma_out(&self) -> bool {
        self.output_ready() || self.can_continue_decode()
    }

    /// True when a decode DMA0 upload should remain busy until DMA1
    /// drains the corresponding output frame.
    pub fn decode_dma0_waits_for_output(&self) -> bool {
        self.command_code() == 0x3 && self.reg1 & MDEC1_BUSY != 0
    }

    /// Recent raw command-register writes, newest at the end.
    pub fn command_history(&self) -> &[u32] {
        &self.command_history
    }

    /// Diagnostic: queued compressed halfwords still held after the
    /// latest decode attempt.
    pub fn queued_rle_halfwords(&self) -> usize {
        self.rl_queue.len()
    }

    /// Diagnostic: next compressed halfword, if any.
    pub fn next_rle_halfword(&self) -> Option<u16> {
        self.rl_queue.front().copied()
    }

    // ============================================================
    //  Register-level command / control / status.
    // ============================================================

    fn status_word(&self) -> u32 {
        self.reg1
    }

    fn command_write(&mut self, value: u32) {
        self.reg0 = value;
        self.commands_seen = self.commands_seen.saturating_add(1);
        if self.command_history.len() == 64 {
            self.command_history.remove(0);
        }
        self.command_history.push(value);

        // Redux's MDEC write0 only latches reg0; status-format bits
        // are not mirrored here.
    }

    #[allow(dead_code)]
    fn command_write_direct_fifo_legacy(&mut self, value: u32) {
        // Detect whether this word is a new command or parameter data.
        // Top nibble of a real command is always one of the defined
        // codes (3, 4, 6). If we're awaiting parameter data, treat the
        // word as two RLE halfwords instead.
        if self.reg1 & MDEC1_BUSY == 0 {
            // Not decoding -- this might be a fresh command or
            // quantization-table payload depending on the command.
            let cmd = (value >> 28) & 0xF;
            match cmd {
                0x3 => {
                    // Decode macroblocks. Parameter count is in the
                    // low 16 bits (number of parameter *words*).
                    self.reg0 = value;
                    self.reg1 |= MDEC1_BUSY;
                    self.expected_param_words = value & 0xFFFF;
                    self.commands_seen = self.commands_seen.saturating_add(1);
                    // Mirror cmd flags into status reg.
                    if value & MDEC0_STP != 0 {
                        self.reg1 |= MDEC1_STP;
                    } else {
                        self.reg1 &= !MDEC1_STP;
                    }
                    if value & MDEC0_RGB24 != 0 {
                        self.reg1 |= MDEC1_RGB24;
                    } else {
                        self.reg1 &= !MDEC1_RGB24;
                    }
                    self.rl_queue.clear();
                }
                0x4 => {
                    // Quantization table upload -- 128 bytes (64 Y + 64 UV)
                    // streamed in as 32 parameter words (4 bytes per word).
                    self.reg0 = value;
                    self.reg1 |= MDEC1_BUSY;
                    self.expected_param_words = value & 0xFFFF;
                    self.commands_seen = self.commands_seen.saturating_add(1);
                    self.rl_queue.clear();
                }
                0x6 => {
                    // Cosine table upload -- 32 parameter words.
                    // The MDEC doesn't actually use a host-supplied
                    // cosine table; we accept the upload and discard.
                    self.reg0 = value;
                    self.reg1 |= MDEC1_BUSY;
                    self.expected_param_words = value & 0xFFFF;
                    self.commands_seen = self.commands_seen.saturating_add(1);
                    self.rl_queue.clear();
                }
                _ => {
                    // Unknown / no-op command.
                    self.reg0 = value;
                    self.commands_seen = self.commands_seen.saturating_add(1);
                }
            }
            return;
        }

        // Busy -- this word is parameter / RLE data.
        self.params_seen = self.params_seen.saturating_add(1);
        let cmd = self.command_code();
        match cmd {
            0x3 => {
                // RLE coefficient data -- two halfwords per word.
                self.rl_queue.push_back(value as u16);
                self.rl_queue.push_back((value >> 16) as u16);
                // Check if we have enough to decode a macroblock (8 KiB
                // worst case per block; normally a few hundred bytes).
                // We decode eagerly when we see an end-of-data sentinel
                // or when the parameter-count runs out.
                if self.expected_param_words > 0 {
                    self.expected_param_words -= 1;
                    if self.expected_param_words == 0 {
                        self.decode_until_output_words(1);
                    }
                }
            }
            0x4 => {
                // Quantization table upload -- 64 Y bytes then 64 UV bytes.
                // 32 words × 4 bytes = 128 bytes total.
                self.absorb_quant_word(value);
                if self.expected_param_words > 0 {
                    self.expected_param_words -= 1;
                    if self.expected_param_words == 0 {
                        self.reg1 &= !MDEC1_BUSY;
                    }
                }
            }
            0x6 if self.expected_param_words > 0 => {
                // Cosine table -- discard.
                self.expected_param_words -= 1;
                if self.expected_param_words == 0 {
                    self.reg1 &= !MDEC1_BUSY;
                }
            }
            _ => {}
        }
    }

    fn control_write(&mut self, value: u32) {
        if value & MDEC1_RESET != 0 {
            // Reset -- clears state but preserves quantization tables
            // per PSX-SPX (they're written via command 0x4 and need
            // to survive MDEC resets so games don't re-upload).
            self.reg0 = 0;
            self.reg1 = status_idle();
            self.dma_in_enabled = false;
            self.dma_out_enabled = false;
            self.expected_param_words = 0;
            self.rl_queue.clear();
            self.out_queue.clear();
            return;
        }
        self.dma_in_enabled = value & (1 << 30) != 0;
        self.dma_out_enabled = value & (1 << 29) != 0;
    }

    fn pop_output_word(&mut self) -> u32 {
        let lo = self.out_queue.pop_front().unwrap_or(0) as u32;
        let hi = self.out_queue.pop_front().unwrap_or(0) as u32;
        lo | (hi << 16)
    }

    /// Absorb one 32-bit word of quantization-table data. The upload
    /// is 32 words: first 16 words (64 bytes) for iq_y, next 16 for
    /// iq_uv. We decode each byte, multiply by the AAN prescale, and
    /// slot into the iq table at the zigzag position.
    fn absorb_quant_word(&mut self, value: u32) {
        let bytes = value.to_le_bytes();
        // The byte index within the 128-byte quant upload depends on
        // how many params we've seen so far. params_seen is just a
        // counter; we track quant progress separately via
        // `expected_param_words` going from 32 → 0.
        let total_expected = self.reg0 & 0xFFFF;
        let words_delivered = (total_expected - self.expected_param_words) as usize;
        for (byte_index, &b) in bytes.iter().enumerate() {
            let pos = words_delivered * 4 + byte_index;
            if pos < 64 {
                self.iq_y[pos] =
                    (b as i32) * scaler(AAN_SCALES[ZIG_ZAG_SCAN[pos]], AAN_PRESCALE_SCALE);
            } else if pos < 128 {
                let uv_pos = pos - 64;
                self.iq_uv[uv_pos] =
                    (b as i32) * scaler(AAN_SCALES[ZIG_ZAG_SCAN[uv_pos]], AAN_PRESCALE_SCALE);
            }
        }
    }

    fn absorb_quant_upload(&mut self, words: &[u32]) {
        for (word_index, &value) in words.iter().enumerate() {
            let bytes = value.to_le_bytes();
            for (byte_index, &b) in bytes.iter().enumerate() {
                let pos = word_index * 4 + byte_index;
                if pos < 64 {
                    self.iq_y[pos] =
                        (b as i32) * scaler(AAN_SCALES[ZIG_ZAG_SCAN[pos]], AAN_PRESCALE_SCALE);
                } else if pos < 128 {
                    let uv_pos = pos - 64;
                    self.iq_uv[uv_pos] =
                        (b as i32) * scaler(AAN_SCALES[ZIG_ZAG_SCAN[uv_pos]], AAN_PRESCALE_SCALE);
                }
            }
        }
    }

    #[inline]
    fn command_code(&self) -> u32 {
        (self.reg0 >> 28) & 0xF
    }

    /// Decode enough macroblocks to make at least `min_words` 32-bit
    /// words available. Redux decodes from the RLE stream during DMA1
    /// rather than eagerly during DMA0; matching that order prevents a
    /// single frame's end marker from cutting off later DMA1 chunks.
    fn decode_until_output_words(&mut self, min_words: usize) {
        let min_halfwords = min_words.saturating_mul(2);
        while self.out_queue.len() < min_halfwords && self.can_continue_decode() {
            if !self.decode_one_macroblock() {
                break;
            }
        }
    }

    fn can_continue_decode(&self) -> bool {
        self.command_code() == 0x3 && self.reg1 & MDEC1_BUSY != 0 && !self.rl_queue.is_empty()
    }

    /// Decode a single macroblock (6 blocks × 64 coefs → 16×16 pixels).
    /// Returns `true` on success, `false` if we ran out of data.
    fn decode_one_macroblock(&mut self) -> bool {
        let mut blocks = [[0i32; DSIZE2]; BLOCKS_PER_MACROBLOCK];
        for (bi, block) in blocks.iter_mut().enumerate() {
            let iqtab = if bi < 2 { &self.iq_uv } else { &self.iq_y };
            if !decode_block(&mut self.rl_queue, block, iqtab) {
                return false;
            }
        }
        // Convert YUV→RGB and push output. Block order:
        //   0: Cr, 1: Cb, 2..5: Y1..Y4
        self.emit_macroblock_output(&blocks);
        self.macroblocks_decoded = self.macroblocks_decoded.saturating_add(1);
        true
    }

    /// YUV→RGB conversion + output packing. Fills `out_queue` with
    /// either 15-bit RGB halfwords or 24-bit RGB byte triplets (still
    /// queued as halfwords for uniform storage).
    fn emit_macroblock_output(&mut self, blocks: &[[i32; DSIZE2]; BLOCKS_PER_MACROBLOCK]) {
        // Block 0 = Cr, Block 1 = Cb, Blocks 2..5 = Y1..Y4.
        let cr = &blocks[0];
        let cb = &blocks[1];
        let y_blocks: [&[i32; DSIZE2]; 4] = [&blocks[2], &blocks[3], &blocks[4], &blocks[5]];

        let rgb24 = self.reg0 & MDEC0_RGB24 == 0; // 0 = 24-bit, 1 = 15-bit
        let mask_bit_15 = self.reg0 & MDEC0_STP != 0;

        if rgb24 {
            // 24-bit (RGB888) -- 16×16 pixels × 3 bytes = 768 bytes = 384 halfwords.
            let mut image = [0u8; 16 * 16 * 3];
            yuv_to_rgb24(&mut image, cr, cb, y_blocks);
            // Pack bytes into halfwords little-endian: [b0|b1], [b2|b3], ...
            for chunk in image.chunks(2) {
                let lo = chunk[0] as u16;
                let hi = chunk.get(1).copied().unwrap_or(0) as u16;
                self.out_queue.push_back(lo | (hi << 8));
            }
        } else {
            // 15-bit (RGB555) -- 16×16 pixels × 2 bytes = 512 bytes = 256 halfwords.
            let mut image = [0u16; 16 * 16];
            yuv_to_rgb15(&mut image, cr, cb, y_blocks, mask_bit_15);
            for px in image {
                self.out_queue.push_back(px);
            }
        }
    }
}

/// Default MDEC status word. PCSX-Redux initializes and resets MDEC
/// `reg1` to zero; no empty/DREQ bits are synthesized on read.
const fn status_idle() -> u32 {
    0
}

// ===============================================================
//  Block-level decode: RLE → coefficients → IDCT.
// ===============================================================

/// Decode one 8×8 block's worth of RLE coefficients into `block`,
/// applying dequantization via `iqtab` and the AAN IDCT. Returns
/// `false` when we hit the end of data before completing the block.
fn decode_block(
    rl: &mut std::collections::VecDeque<u16>,
    block: &mut [i32; DSIZE2],
    iqtab: &[i32; DSIZE2],
) -> bool {
    // First word: quantization scale (high 6 bits) + DC coefficient (low 10).
    let head = match rl.pop_front() {
        Some(v) => v,
        None => return false,
    };
    let q_scale = rle_run(head);
    block.fill(0);
    block[0] = scaler(iqtab[0] * rle_val(head), AAN_EXTRA - 3);

    let mut k: usize = 0;
    let mut used_col: i32 = 0;
    loop {
        let rl_word = match rl.pop_front() {
            Some(v) => v,
            None => return false,
        };
        if rl_word == MDEC_END_OF_DATA {
            break;
        }
        let run = rle_run(rl_word) as usize;
        k += run + 1;
        if k > 63 {
            // Broken stream -- bail gracefully.
            break;
        }
        let pos = ZIG_ZAG_SCAN[k];
        block[pos] = scaler(rle_val(rl_word) * iqtab[k] * q_scale, AAN_EXTRA);
        // Track used columns to accelerate IDCT.
        if pos > 7 {
            used_col |= 1 << (pos & 7);
        }
    }
    if k == 0 {
        // Only DC coefficient -- fill the block uniformly.
        idct(block, -1);
    } else {
        idct(block, used_col);
    }
    true
}

/// Extract the quantization-scale / run-length field from an RLE word
/// (top 6 bits).
#[inline]
fn rle_run(v: u16) -> i32 {
    (v >> 10) as i32
}

/// Extract the signed 10-bit value field from an RLE word, sign-extended.
#[inline]
fn rle_val(v: u16) -> i32 {
    let bits = 10;
    let shift = 32 - bits;
    ((v as i32) << shift) >> shift
}

/// AAN-optimized 2D IDCT on an 8×8 block. Implements Redux's hybrid
/// row/column traversal: walks columns first (skipping columns with
/// only a DC coefficient when possible via `used_col`), then rows.
#[allow(clippy::too_many_arguments)]
fn idct(block: &mut [i32; DSIZE2], used_col: i32) {
    if used_col == -1 {
        let v = block[0];
        block.fill(v);
        return;
    }

    // Column pass.
    for i in 0..DSIZE {
        if used_col & (1 << i) == 0 {
            // Column either empty or has only DC -- splat DC down.
            if block[i] != 0 {
                fill_col(block, i, block[i]);
            }
            continue;
        }

        let ptr = |r: usize| block[r * DSIZE + i];
        let z10 = ptr(0) + ptr(4);
        let z11 = ptr(0) - ptr(4);
        let z13 = ptr(2) + ptr(6);
        let z12 = muls(ptr(2) - ptr(6), fix_1_414213562()) - z13;

        let tmp0 = z10 + z13;
        let tmp3 = z10 - z13;
        let tmp1 = z11 + z12;
        let tmp2 = z11 - z12;

        let z13 = ptr(3) + ptr(5);
        let z10 = ptr(3) - ptr(5);
        let z11 = ptr(1) + ptr(7);
        let z12 = ptr(1) - ptr(7);

        let tmp7 = z11 + z13;
        let z5 = (z12 - z10) * fix_1_847759065();
        let tmp6 = scale(z10 * fix_2_613125930() + z5, AAN_CONST_BITS) - tmp7;
        let tmp5 = muls(z11 - z13, fix_1_414213562()) - tmp6;
        let tmp4 = scale(z12 * fix_1_082392200() - z5, AAN_CONST_BITS) + tmp5;

        block[i] = tmp0 + tmp7;
        block[7 * DSIZE + i] = tmp0 - tmp7;
        block[DSIZE + i] = tmp1 + tmp6;
        block[6 * DSIZE + i] = tmp1 - tmp6;
        block[2 * DSIZE + i] = tmp2 + tmp5;
        block[5 * DSIZE + i] = tmp2 - tmp5;
        block[4 * DSIZE + i] = tmp3 + tmp4;
        block[3 * DSIZE + i] = tmp3 - tmp4;
    }

    // Row pass.
    if used_col == 1 {
        for i in 0..DSIZE {
            let v = block[DSIZE * i];
            fill_row(block, i, v);
        }
    } else {
        for i in 0..DSIZE {
            let base = i * DSIZE;
            let p = |j: usize| block[base + j];
            let z10 = p(0) + p(4);
            let z11 = p(0) - p(4);
            let z13 = p(2) + p(6);
            let z12 = muls(p(2) - p(6), fix_1_414213562()) - z13;

            let tmp0 = z10 + z13;
            let tmp3 = z10 - z13;
            let tmp1 = z11 + z12;
            let tmp2 = z11 - z12;

            let z13 = p(3) + p(5);
            let z10 = p(3) - p(5);
            let z11 = p(1) + p(7);
            let z12 = p(1) - p(7);

            let tmp7 = z11 + z13;
            let z5 = (z12 - z10) * fix_1_847759065();
            let tmp6 = scale(z10 * fix_2_613125930() + z5, AAN_CONST_BITS) - tmp7;
            let tmp5 = muls(z11 - z13, fix_1_414213562()) - tmp6;
            let tmp4 = scale(z12 * fix_1_082392200() - z5, AAN_CONST_BITS) + tmp5;

            block[base] = tmp0 + tmp7;
            block[base + 7] = tmp0 - tmp7;
            block[base + 1] = tmp1 + tmp6;
            block[base + 6] = tmp1 - tmp6;
            block[base + 2] = tmp2 + tmp5;
            block[base + 5] = tmp2 - tmp5;
            block[base + 4] = tmp3 + tmp4;
            block[base + 3] = tmp3 - tmp4;
        }
    }
}

fn fill_col(block: &mut [i32; DSIZE2], col: usize, v: i32) {
    for r in 0..DSIZE {
        block[r * DSIZE + col] = v;
    }
}

fn fill_row(block: &mut [i32; DSIZE2], row: usize, v: i32) {
    let base = row * DSIZE;
    for j in 0..DSIZE {
        block[base + j] = v;
    }
}

// ===============================================================
//  YUV → RGB conversion.
// ===============================================================

// JPEG-scale YUV→RGB (Y/Cb/Cr[-128..127] → R/G/B[0..255]):
//   R = Y + 1.400*Cr
//   G = Y - 0.343*Cb - 0.711*Cr
//   B = Y + 1.765*Cb

#[inline]
fn mulr(a: i32) -> i32 {
    1434 * a
}
#[inline]
fn mulb(a: i32) -> i32 {
    1807 * a
}
#[inline]
fn mulg2(a: i32, b: i32) -> i32 {
    -351 * a - 728 * b
}
#[inline]
fn muly(a: i32) -> i32 {
    a << 10
}

#[inline]
fn clamp5(c: i32) -> i32 {
    if c < -16 {
        0
    } else if c > 31 - 16 {
        31
    } else {
        c + 16
    }
}

#[inline]
fn clamp8(c: i32) -> i32 {
    if c < -128 {
        0
    } else if c > 255 - 128 {
        255
    } else {
        c + 128
    }
}

#[inline]
fn scale8(c: i32) -> i32 {
    scaler(c, 20)
}
#[inline]
fn scale5(c: i32) -> i32 {
    scaler(c, 23)
}

#[inline]
fn make_rgb15(r: i32, g: i32, b: i32, a: u16) -> u16 {
    (a | ((b as u16) << 10) | ((g as u16) << 5) | (r as u16)).to_le()
}

/// Produce one 16×16 image from one macroblock's 4 Y blocks plus a
/// shared Cr + Cb block, output as 15-bit RGB halfwords.
fn yuv_to_rgb15(
    image: &mut [u16; 16 * 16],
    cr: &[i32; DSIZE2],
    cb: &[i32; DSIZE2],
    y_blocks: [&[i32; DSIZE2]; 4],
    mask_bit_15: bool,
) {
    let mask = if mask_bit_15 { 0x8000 } else { 0 };
    // 16×16 output split into 4 quadrants of 8×8:
    //   top-left = Y1, top-right = Y2, bottom-left = Y3, bottom-right = Y4
    // Cb/Cr are 8×8 for the entire macroblock -- each pixel of Cb/Cr
    // corresponds to a 2×2 block of Y pixels.
    for qy in 0..2 {
        for qx in 0..2 {
            let y = y_blocks[qy * 2 + qx];
            for row in 0..8 {
                for col in 0..8 {
                    let out_row = qy * 8 + row;
                    let out_col = qx * 8 + col;
                    let out_idx = out_row * 16 + out_col;
                    // Cb/Cr are chroma-subsampled: one entry per
                    // 2×2 Y block. The Cb/Cr entry coords are
                    // (out_row/2, out_col/2).
                    let cidx = (out_row / 2) * DSIZE + (out_col / 2);
                    let y_val = muly(y[row * DSIZE + col]);
                    let r_contrib = mulr(cr[cidx]);
                    let g_contrib = mulg2(cb[cidx], cr[cidx]);
                    let b_contrib = mulb(cb[cidx]);
                    let r = clamp5(scale5(y_val + r_contrib));
                    let g = clamp5(scale5(y_val + g_contrib));
                    let b = clamp5(scale5(y_val + b_contrib));
                    image[out_idx] = make_rgb15(r, g, b, mask);
                }
            }
        }
    }
}

/// 24-bit RGB output. Pixels are packed as `[R, G, B]` triplets in
/// row-major order.
fn yuv_to_rgb24(
    image: &mut [u8; 16 * 16 * 3],
    cr: &[i32; DSIZE2],
    cb: &[i32; DSIZE2],
    y_blocks: [&[i32; DSIZE2]; 4],
) {
    for qy in 0..2 {
        for qx in 0..2 {
            let y = y_blocks[qy * 2 + qx];
            for row in 0..8 {
                for col in 0..8 {
                    let out_row = qy * 8 + row;
                    let out_col = qx * 8 + col;
                    let pixel_idx = (out_row * 16 + out_col) * 3;
                    let cidx = (out_row / 2) * DSIZE + (out_col / 2);
                    let y_val = muly(y[row * DSIZE + col]);
                    let r_contrib = mulr(cr[cidx]);
                    let g_contrib = mulg2(cb[cidx], cr[cidx]);
                    let b_contrib = mulb(cb[cidx]);
                    let r = clamp8(scale8(y_val + r_contrib));
                    let g = clamp8(scale8(y_val + g_contrib));
                    let b = clamp8(scale8(y_val + b_contrib));
                    image[pixel_idx] = r as u8;
                    image[pixel_idx + 1] = g as u8;
                    image[pixel_idx + 2] = b as u8;
                }
            }
        }
    }
}

// ===============================================================
//  Tests.
// ===============================================================

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
    fn fresh_status_matches_redux_idle() {
        let mut m = Mdec::new();
        let stat = m.read32(MDEC_CTRL_STAT);
        assert_eq!(stat, 0);
    }

    #[test]
    fn data_port_read_returns_latched_command_register() {
        let mut m = Mdec::new();
        assert_eq!(m.read32(MDEC_CMD_DATA), 0);
        m.write32(MDEC_CMD_DATA, 0x3200_0540);
        assert_eq!(m.read32(MDEC_CMD_DATA), 0x3200_0540);
    }

    #[test]
    fn control_write_reset_bit_clears_state() {
        let mut m = Mdec::new();
        m.write32(MDEC_CTRL_STAT, 0x6000_0000);
        assert!(m.dma_in_enabled());
        assert!(m.dma_out_enabled());
        m.write32(MDEC_CTRL_STAT, 0x8000_0000);
        assert!(!m.dma_in_enabled());
        assert!(!m.dma_out_enabled());
    }

    #[test]
    fn control_write_latches_dma_enables() {
        let mut m = Mdec::new();
        m.write32(MDEC_CTRL_STAT, 0x4000_0000);
        assert!(m.dma_in_enabled());
        assert!(!m.dma_out_enabled());
        m.write32(MDEC_CTRL_STAT, 0x2000_0000);
        assert!(!m.dma_in_enabled());
        assert!(m.dma_out_enabled());
    }

    #[test]
    fn decode_dma_status_matches_redux_latched_bits() {
        let mut m = Mdec::new();
        m.write32(MDEC_CTRL_STAT, 0x6000_0000);
        m.write32(MDEC_CMD_DATA, 0x3200_0006);
        m.dma_write_in(&[0xFE00_0000; 6]);

        let stat = m.read32(MDEC_CTRL_STAT);
        assert_eq!(stat & MDEC1_BUSY, MDEC1_BUSY);
        assert_eq!(stat & MDEC1_STP, MDEC1_STP);
        assert_eq!(stat & MDEC1_DMA_IN_REQ, 0);
        assert_eq!(stat & MDEC1_DMA_OUT_REQ, 0);
        assert_eq!(stat & MDEC1_OUTPUT_DEPTH_MASK, 0);
    }

    #[test]
    fn data_write_tallies_commands_vs_parameters() {
        let mut m = Mdec::new();
        // Quant table upload command -- 32 param words expected.
        m.write32(MDEC_CMD_DATA, 0x4000_0020);
        assert_eq!(m.commands_seen(), 1);
        // DMA0 carries the payload after the command register is latched.
        m.dma_write_in(&[0xDEAD_BEEF]);
        assert_eq!(m.params_seen(), 1);
    }

    #[test]
    fn quant_upload_command_marks_busy_then_clears() {
        let mut m = Mdec::new();
        // Command 4 (quant upload), 32 param words expected.
        m.write32(MDEC_CMD_DATA, 0x4000_0020);
        assert_eq!(m.read32(MDEC_CTRL_STAT) & MDEC1_BUSY, 0);
        // Deliver 32 words through DMA0.
        m.dma_write_in(&[0; 32]);
        assert_eq!(m.read32(MDEC_CTRL_STAT) & MDEC1_BUSY, 0);
    }

    #[test]
    fn decode_command_emits_empty_output_on_sentinel_only_stream() {
        let mut m = Mdec::new();
        // Upload identity quant tables so decode produces well-defined
        // (but all-zero) output.
        m.write32(MDEC_CMD_DATA, 0x4000_0020);
        m.dma_write_in(&[0x01_01_01_01; 32]);
        // Issue decode command -- tiny parameter count.
        m.write32(MDEC_CMD_DATA, 0x3000_0001);
        // Feed a single word holding two END-of-data sentinels.
        let sentinel = MDEC_END_OF_DATA as u32;
        m.dma_write_in(&[sentinel | (sentinel << 16)]);
        // Redux keeps decode DMA0 busy until DMA1 pulls/observes the
        // output side, even for a sentinel-only stream.
        assert_eq!(m.read32(MDEC_CTRL_STAT) & MDEC1_BUSY, MDEC1_BUSY);
        let mut out = [0u32; 1];
        m.dma_read_out(&mut out);
        assert_eq!(out[0], 0);
        assert!(m.complete_dma_out());
        assert_eq!(m.read32(MDEC_CTRL_STAT) & MDEC1_BUSY, 0);
    }

    #[test]
    fn idct_dc_only_fills_block_with_dc() {
        let mut block = [0i32; DSIZE2];
        block[0] = 42;
        idct(&mut block, -1);
        assert!(block.iter().all(|&v| v == 42));
    }

    #[test]
    fn rle_run_and_val_extract_correctly() {
        // Top 6 bits = 0b101010 = 42 → run.
        // Low 10 bits = 0x123 signed = 0x123 (positive, 291).
        let word = 0b1010_1000_0001_0010_u16; // run=42, value=0x12 (hmm)
                                              // Let me pick a cleaner example:
                                              // run = 5 (top 6 bits = 000101), val = 0x1F (low 10 bits = 00_0001_1111 = 31).
        let word2 = (5u16 << 10) | 0x001F;
        assert_eq!(rle_run(word2), 5);
        assert_eq!(rle_val(word2), 31);
        // Negative value: low 10 bits = 0x3FF = -1 signed.
        let word3 = 0x03FF;
        assert_eq!(rle_val(word3), -1);
        // Unused variable so compiler is happy.
        let _ = rle_run(word);
    }

    #[test]
    fn make_rgb15_packs_correctly() {
        // R=0, G=0, B=0, no mask → 0x0000.
        assert_eq!(make_rgb15(0, 0, 0, 0), 0);
        // R=31, G=0, B=0 → 0x001F.
        assert_eq!(make_rgb15(31, 0, 0, 0), 0x001F);
        // R=0, G=31, B=0 → 0x03E0.
        assert_eq!(make_rgb15(0, 31, 0, 0), 0x03E0);
        // R=0, G=0, B=31 → 0x7C00.
        assert_eq!(make_rgb15(0, 0, 31, 0), 0x7C00);
        // Mask bit → 0x8000.
        assert_eq!(make_rgb15(0, 0, 0, 0x8000), 0x8000);
    }

    #[test]
    fn macroblocks_counter_increments_on_successful_decode() {
        let mut m = Mdec::new();
        // Identity quant tables.
        m.write32(MDEC_CMD_DATA, 0x4000_0020);
        m.dma_write_in(&[0x01_01_01_01; 32]);
        // Six DC-only blocks make one black macroblock.
        m.write32(MDEC_CMD_DATA, 0x3000_0006);
        let sentinel = MDEC_END_OF_DATA as u32;
        // Need a DC coefficient + sentinel per block x 6 blocks.
        // With all-zero DC + immediate sentinel per block, we produce
        // a single macroblock of all-black output.
        m.dma_write_in(&[sentinel << 16; 6]);
        let mut out = [0u32; 1];
        m.dma_read_out(&mut out);
        assert_eq!(m.macroblocks_decoded(), 1);
    }

    #[test]
    fn state_reports_transitions() {
        let mut m = Mdec::new();
        assert_eq!(m.state(), MdecState::Idle);
        m.write32(MDEC_CMD_DATA, 0x3000_0006);
        m.dma_write_in(&[0xFE00_0000; 6]);
        assert_eq!(m.state(), MdecState::AwaitingData);
        let mut out = [0u32; 1];
        m.dma_read_out(&mut out);
        assert_eq!(m.state(), MdecState::DecodeReady);
        let mut rest = [0u32; 191];
        m.dma_read_out(&mut rest);
        m.complete_dma_out();
        assert_eq!(m.state(), MdecState::Idle);
    }
}
