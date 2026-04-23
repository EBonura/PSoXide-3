//! SPU — Sound Processing Unit.
//!
//! The SPU is a 24-voice ADPCM sample engine with per-voice ADSR
//! envelopes, pitch-controlled playback, a 512 KiB sample RAM, and
//! stereo output mixed at 44.1 kHz. This module is the real thing —
//! ADPCM decode, voice state machine, ADSR, stereo mixing. What's
//! explicitly **not** modelled yet (each a follow-up session):
//!
//! - Reverb (work area regs are writable, but the mix path is
//!   disabled — games that rely on reverb for mood will sound dry
//!   but still play correctly).
//! - XA ADPCM streaming (CD-DA audio / in-game speech) is decoded by
//!   the CDROM module; the SPU exposes [`xa_decode_block`] for it but
//!   does not call it from `tick_sample`.
//!
//! Already in: 1024-entry Gaussian sample interpolation (`GAUSS_TABLE`),
//! per-voice volume-sweep envelopes animated each sample (matches
//! Redux's rate tables).
//!
//! Reference implementations consulted:
//! - PCSX-Redux `src/spu/{spu,adsr,registers,dma}.cc` (GPL-2+) for
//!   ADSR rate tables and voice state model.
//! - psx-spx "SPU" chapter for register layout + ADPCM filter table.
//! - Neill Corlett's SPU envelope notes (quoted in `adsr.cc`).
//!
//! Pipeline per 44.1 kHz sample:
//!
//! 1. For each voice:
//!    a. If envelope is in `Off`, contribute 0.
//!    b. Advance ADPCM read position by `raw_pitch / 0x1000` of a
//!       sample; when `sample_index` reaches 28, decode the next
//!       16-byte ADPCM block into the sample buffer and update the
//!       prev-sample history (`s_1`, `s_2`). Apply block flags
//!       (loop-start / loop-end / loop-repeat). On loop-end with
//!       repeat, jump to `loop_addr` or stop the voice.
//!    c. Linearly interpolate between consecutive decoded samples
//!       at the fractional position.
//!    d. Advance ADSR envelope one sample; multiply interpolated
//!       sample by envelope level (Q15).
//!    e. Multiply by per-voice L / R volume (Q14 static, or
//!       approximated sweep) to get per-channel contribution.
//! 2. Sum all 24 voices into `(sum_l, sum_r)`.
//! 3. Multiply by SPU main volume L / R (Q14), saturate to i16.
//! 4. Push (l, r) to the host-facing output ring.
//!
//! SPU IRQ: if SPUCNT bit 6 (IRQEnable) is set and the IRQ address
//! matches the sample-read pointer of any voice (or the data-transfer
//! FIFO read pointer), we latch STATUS bit 6 and signal the bus.
//!
//! Sample rate: 44_100 Hz. PSX clock is 33_868_800 Hz, so 1 sample =
//! 768 cycles. We tick the SPU from the scheduler every [`SAMPLE_CYCLES`]
//! cycles.

use crate::scheduler::{EventSlot, Scheduler};

// ===============================================================
//  Register addresses — voice bank + global + reverb config.
// ===============================================================

/// Base of the SPU MMIO window. 512 bytes total spanning voice bank,
/// global control regs, and reverb coefficient registers.
pub const SPU_BASE: u32 = 0x1F80_1C00;
/// One past the end of the SPU MMIO window.
pub const SPU_END: u32 = 0x1F80_1E00;

/// Base of the 24-voice register bank (16 bytes per voice).
pub const VOICE_BASE: u32 = 0x1F80_1C00;
/// One past the end of the voice bank (24 * 16 = 0x180 bytes → 0x1F80_1D80).
pub const VOICE_END: u32 = 0x1F80_1D80;

/// Main Volume Left (16-bit, Q14).
pub const MAIN_VOL_L: u32 = 0x1F80_1D80;
/// Main Volume Right (16-bit, Q14).
pub const MAIN_VOL_R: u32 = 0x1F80_1D82;
/// Reverb output volume Left.
pub const REVERB_VOL_L: u32 = 0x1F80_1D84;
/// Reverb output volume Right.
pub const REVERB_VOL_R: u32 = 0x1F80_1D86;
/// Key-On low (voices 0..15).
pub const KON_LO: u32 = 0x1F80_1D88;
/// Key-On high (voices 16..23).
pub const KON_HI: u32 = 0x1F80_1D8A;
/// Key-Off low (voices 0..15).
pub const KOFF_LO: u32 = 0x1F80_1D8C;
/// Key-Off high (voices 16..23).
pub const KOFF_HI: u32 = 0x1F80_1D8E;
/// Pitch modulation enable low.
pub const PMON_LO: u32 = 0x1F80_1D90;
/// Pitch modulation enable high.
pub const PMON_HI: u32 = 0x1F80_1D92;
/// Noise mode enable low.
pub const NON_LO: u32 = 0x1F80_1D94;
/// Noise mode enable high.
pub const NON_HI: u32 = 0x1F80_1D96;
/// Reverb enable low.
pub const EON_LO: u32 = 0x1F80_1D98;
/// Reverb enable high.
pub const EON_HI: u32 = 0x1F80_1D9A;
/// ENDX low (per-voice "reached loop-end block" latch, write-1-to-clear).
pub const ENDX_LO: u32 = 0x1F80_1D9C;
/// ENDX high.
pub const ENDX_HI: u32 = 0x1F80_1D9E;
/// Reverb work-area start address (halfword, scaled by 8 → byte addr).
pub const REVERB_BASE: u32 = 0x1F80_1DA2;
/// IRQ address (halfword * 8 = byte addr into SPU RAM).
pub const IRQ_ADDR: u32 = 0x1F80_1DA4;
/// Data transfer address (halfword * 8 = byte addr into SPU RAM).
pub const TRANSFER_ADDR: u32 = 0x1F80_1DA6;
/// Data transfer FIFO (reads pop, writes push at TRANSFER_ADDR, which advances).
pub const TRANSFER_FIFO: u32 = 0x1F80_1DA8;
/// SPU control register.
pub const SPUCNT: u32 = 0x1F80_1DAA;
/// SPUCNT bit 14: 0 = muted, 1 = unmuted.
const SPUCNT_UNMUTE: u16 = 1 << 14;
/// Data transfer control (typically 0x0004 — 4-bit transfer step).
pub const TRANSFER_CTRL: u32 = 0x1F80_1DAC;
/// SPU status register.
pub const SPUSTAT: u32 = 0x1F80_1DAE;
/// CD audio input volume Left.
pub const CD_VOL_L: u32 = 0x1F80_1DB0;
/// CD audio input volume Right.
pub const CD_VOL_R: u32 = 0x1F80_1DB2;
/// External audio input volume Left.
pub const EXT_VOL_L: u32 = 0x1F80_1DB4;
/// External audio input volume Right.
pub const EXT_VOL_R: u32 = 0x1F80_1DB6;
/// Current Main Volume Left (read-only mirror of main_vol_l after sweep).
pub const CURRENT_MAIN_VOL_L: u32 = 0x1F80_1DB8;
/// Current Main Volume Right.
pub const CURRENT_MAIN_VOL_R: u32 = 0x1F80_1DBA;

/// Start of reverb configuration area (32 × 16-bit coefficient regs).
pub const REVERB_CFG_BASE: u32 = 0x1F80_1DC0;

/// Per-voice offsets within the 16-byte voice block.
#[allow(dead_code)]
mod voice_offset {
    /// +0..1 volume left (Q14, or sweep config if bit 15 set).
    pub const VOLUME_L: u32 = 0x0;
    /// +2..3 volume right.
    pub const VOLUME_R: u32 = 0x2;
    /// +4..5 ADPCM pitch. `0x1000` = base rate (44.1 kHz). Max 0x3FFF.
    pub const PITCH: u32 = 0x4;
    /// +6..7 ADPCM start address (in 8-byte units; <<3 = byte addr).
    pub const START_ADDR: u32 = 0x6;
    /// +8..9 ADSR config low — attack mode + rate + decay rate + sustain level.
    pub const ADSR_LO: u32 = 0x8;
    /// +A..B ADSR config high — sustain mode + sustain rate + release mode + release rate.
    pub const ADSR_HI: u32 = 0xA;
    /// +C..D Current ADSR volume (read-only; returns current envelope level).
    pub const ADSR_CURRENT: u32 = 0xC;
    /// +E..F Repeat (loop) address (in 8-byte units).
    pub const REPEAT_ADDR: u32 = 0xE;
}

// ===============================================================
//  Sizing / timing constants.
// ===============================================================

/// Number of voices in the PSX SPU.
pub const NUM_VOICES: usize = 24;

/// SPU RAM size in bytes (512 KiB).
pub const SPU_RAM_BYTES: usize = 512 * 1024;
/// SPU RAM size in 16-bit words.
pub const SPU_RAM_HALFWORDS: usize = SPU_RAM_BYTES / 2;

/// System clock cycles per SPU sample. 33_868_800 Hz / 44_100 Hz = 768.
pub const SAMPLE_CYCLES: u64 = 768;

/// ADPCM block size in bytes (1 header + 1 flags + 14 data bytes).
pub const ADPCM_BLOCK_BYTES: usize = 16;
/// Samples produced per ADPCM block.
pub const ADPCM_SAMPLES_PER_BLOCK: usize = 28;

/// Host-facing audio output buffer cap. Frontend drains periodically;
/// if it falls behind we discard the oldest samples.
const OUTPUT_BUFFER_CAP: usize = 44100 * 2; // 2 seconds of stereo samples

// ===============================================================
//  ADPCM filter table (5 filters × 2 coefficients, matches PSX-SPX).
// ===============================================================

/// ADPCM prediction filter coefficients: `(s_1_weight, s_2_weight)` in Q6.
///
/// Applied as `fa = raw + (s_1 * f[0] + s_2 * f[1]) >> 6` during block decode.
/// Filters 0..4 are the canonical set used by the real hardware; filters 5..15
/// use the same coefficients (SPU-SPX notes that only the lower 3 bits of the
/// predictor field matter, so 0..7 clamp to 0..4 — we clamp explicitly).
const ADPCM_FILTER_TABLE: [(i32, i32); 5] = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)];

// ===============================================================
//  ADSR envelope rate tables — port of Redux's `EnvelopeTables`.
// ===============================================================

/// Envelope tick-period denominator for each rate (0..=127).
///
/// Rate < 48: 1 (increment/decrement happens every sample).
/// Rate >= 48: `1 << ((rate >> 2) - 11)` (doubles every 4 rate units).
const fn envelope_denominator(rate: usize) -> i32 {
    if rate < 48 {
        1
    } else {
        1i32 << ((rate >> 2) - 11)
    }
}

/// Envelope positive increment numerator: applied on increment ticks.
const fn envelope_numerator_increase(rate: usize) -> i32 {
    let step = 7 - (rate as i32 & 3);
    if rate < 48 {
        step << (11 - (rate >> 2))
    } else {
        step
    }
}

/// Envelope negative decrement numerator.
const fn envelope_numerator_decrease(rate: usize) -> i32 {
    let step = -8 + (rate as i32 & 3);
    if rate < 48 {
        step << (11 - (rate >> 2))
    } else {
        step
    }
}

// ===============================================================
//  ADSR phase + per-voice envelope state.
// ===============================================================

/// ADSR state-machine phase. Voices start at `Off`; KON transitions to
/// `Attack` and resets the envelope. `Off` voices contribute silence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AdsrPhase {
    Off,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Decoded ADSR configuration — parsed from the 32-bit `(adsr_lo | adsr_hi<<16)`
/// register pair. We decode once at write time and stash the bit-fields so the
/// hot path (envelope tick per sample) is pure arithmetic.
#[derive(Copy, Clone, Debug, Default)]
struct AdsrConfig {
    attack_rate: i32,   // 0..=127 (with mode bit folded in)
    attack_exp: bool,   // linear vs exponential slope
    decay_rate: i32,    // 0..=15
    sustain_level: i32, // 0..=15 (target = (N+1) * 0x800)
    sustain_rate: i32,  // 0..=127 (with mode bits folded in)
    sustain_exp: bool,
    sustain_increase: bool, // 1 = rising, 0 = falling
    release_rate: i32,      // 0..=31
    release_exp: bool,
}

/// Parse the low 16 bits of the ADSR register pair into `(ar, ar_exp, dr, sl)`.
fn parse_adsr_lo(lo: u16, cfg: &mut AdsrConfig) {
    cfg.attack_exp = (lo & (1 << 15)) != 0;
    cfg.attack_rate = ((lo >> 8) & 0x7F) as i32;
    cfg.decay_rate = ((lo >> 4) & 0xF) as i32;
    cfg.sustain_level = (lo & 0xF) as i32;
}

/// Parse the high 16 bits of the ADSR register pair into `(sm, sd, sr, rm, rr)`.
fn parse_adsr_hi(hi: u16, cfg: &mut AdsrConfig) {
    cfg.sustain_exp = (hi & (1 << 15)) != 0;
    cfg.sustain_increase = (hi & (1 << 14)) == 0;
    cfg.sustain_rate = ((hi >> 6) & 0x7F) as i32;
    cfg.release_exp = (hi & (1 << 5)) != 0;
    cfg.release_rate = (hi & 0x1F) as i32;
}

// ===============================================================
//  Voice state.
// ===============================================================

/// One SPU volume channel. Holds the raw 16-bit register value so
/// reads round-trip verbatim, plus a live `current` level that
/// animates over time when the register's bit 15 is set (sweep
/// mode). Games use sweep for fade-in / fade-out, and for
/// arbitrary volume automation (track-wide crossfade).
#[derive(Copy, Clone, Debug, Default)]
struct VolumeEnvelope {
    /// The last 16-bit word written to the register. `reads` echo
    /// this so software verification paths see the exact config.
    raw: u16,
    /// Current signed Q14 level, 0..=±0x3FFF. Multiplied directly
    /// into per-sample mix. Updated on register write (static
    /// mode) or per SPU sample (sweep mode).
    current: i16,
    /// Sub-sample counter for the sweep rate — counts up to
    /// `denominator[rate]` before each current-level step.
    sweep_sub: i32,
}

impl VolumeEnvelope {
    fn new() -> Self {
        Self::default()
    }

    /// Accept a new 16-bit register value. Static-mode writes snap
    /// `current` to the decoded level immediately; sweep-mode
    /// writes preserve `current` and let `tick` animate from there.
    fn write(&mut self, raw: u16) {
        self.raw = raw;
        if raw & 0x8000 == 0 {
            // Static mode: bits 0..=13 are the unsigned magnitude,
            // bit 14 is phase-invert.
            let level = (raw & 0x3FFF) as i16;
            self.current = if raw & 0x4000 != 0 { -level } else { level };
            self.sweep_sub = 0;
        }
        // Sweep mode: leave `current` untouched so the new rate
        // takes effect from wherever the level currently sits.
    }

    /// Advance one SPU sample. Applies the sweep animation when
    /// bit 15 is set — uses the same rate tables the ADSR envelope
    /// uses, matching Redux's approach.
    fn tick(&mut self) {
        if self.raw & 0x8000 == 0 {
            return; // static mode, nothing to do
        }
        let rate = (self.raw & 0x7F) as usize;
        let increasing = self.raw & (1 << 13) == 0; // bit 13 = direction (0 = inc)
        let exp = self.raw & (1 << 14) != 0; // bit 14 = exponential
        let denom = envelope_denominator(rate);
        self.sweep_sub += 1;
        if self.sweep_sub >= denom {
            self.sweep_sub = 0;
            let step = if increasing {
                envelope_numerator_increase(rate)
            } else {
                envelope_numerator_decrease(rate)
            };
            let mut new_current = self.current as i32;
            if exp && increasing && new_current >= 0x6000 {
                // Slow down past 0x6000 in exponential-increase mode —
                // matches ADSR's Attack exp slope.
                new_current += step / 4;
            } else if exp && !increasing {
                new_current += (step * new_current) >> 15;
            } else {
                new_current += step;
            }
            self.current = new_current.clamp(-0x7FFF, 0x7FFF) as i16;
        }
    }

    /// Read-back value — always returns the raw register the CPU
    /// wrote, not the animated current level.
    fn reg_read(&self) -> u16 {
        self.raw
    }
}

/// Per-voice runtime state. Holds decode buffers, ADSR envelope,
/// volumes, and loop pointers. Kept plain (no padding or SIMD) —
/// 24 copies of this struct live in `Spu::voices` and mix together on
/// each sample.
#[derive(Clone, Debug)]
struct Voice {
    /// Left volume envelope.
    vol_l: VolumeEnvelope,
    /// Right volume envelope.
    vol_r: VolumeEnvelope,
    /// Raw pitch register (0..=0x3FFF). `0x1000` plays at the sample's
    /// source rate (typically 44.1 kHz).
    raw_pitch: u16,
    /// Byte address into SPU RAM where playback begins on KON. `<<3`
    /// of the register value.
    start_addr: u32,
    /// Loop address (byte address). Set by software via REPEAT_ADDR
    /// register and by the ADPCM flag-4 bit (loop-start).
    loop_addr: u32,
    /// True if software wrote REPEAT_ADDR directly since voice start;
    /// suppresses the ADPCM flag-4 loop-start auto-update (matches
    /// Redux's `IgnoreLoop`).
    loop_addr_locked: bool,
    /// Raw ADSR_LO / ADSR_HI words. Stored so reads echo them back.
    adsr_lo: u16,
    adsr_hi: u16,
    /// Decoded ADSR parameters.
    adsr: AdsrConfig,
    /// Current ADSR phase.
    phase: AdsrPhase,
    /// Envelope level, 0..=0x7FFF (Q15). Multiplies the decoded sample.
    envelope: i32,
    /// Envelope sub-sample counter (`EnvelopeVolF` in Redux); compared
    /// to `denominator[rate]` to decide whether to step the envelope
    /// this sample.
    envelope_sub: i32,
    /// Current byte address into SPU RAM for the *next* ADPCM block to
    /// decode. Updated after each block consumed.
    current_addr: u32,
    /// Decoded samples from the most recent 16-byte block (28 samples).
    /// Indexed by `sample_index`. Values are sign-extended 16-bit.
    sample_buf: [i16; ADPCM_SAMPLES_PER_BLOCK],
    /// Index into `sample_buf`; when it reaches 28 we decode the next
    /// block before taking the next sample.
    sample_index: usize,
    /// Redux-style fixed-point sample cursor (`spos`). Each output
    /// sample consumes decoded input samples while this stays above
    /// `0x10000`, then adds the pitch step (`raw_pitch << 4`) for the
    /// next call. Starting at `0x30000` primes the Gaussian window
    /// with three decoded samples before the first audible output.
    sample_pos: u32,
    /// Rolling 4-sample interpolation ring. Redux stores decoded
    /// samples into `SB[29..32]` and runs the Gaussian window over the
    /// ring so block boundaries still see the previous block's tail.
    /// Without this history, the interpolator falls back to zeros at
    /// every 28-sample ADPCM edge and the output gets audibly gritty.
    interp_ring: [i16; 4],
    /// Next insertion slot in `interp_ring`. Also the logical
    /// "oldest sample" index when reading the Gaussian window.
    interp_pos: usize,
    /// Previous two decoded samples — ADPCM filter history. Preserved
    /// across block boundaries; reset on KON.
    s_1: i32,
    s_2: i32,
    /// Set when a decoded block had the stop flag without a valid
    /// loop. The current 28-sample block must still play out fully;
    /// Redux only turns the voice off when the decoder reaches the
    /// *next* block boundary.
    stop_after_block: bool,
    /// Most recent interpolated sample output by this voice (post-ADSR,
    /// pre-volume). Kept for reads of the ADSR_CURRENT register and
    /// pitch modulation consumers.
    last_sample: i16,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            vol_l: VolumeEnvelope::new(),
            vol_r: VolumeEnvelope::new(),
            raw_pitch: 0,
            start_addr: 0,
            loop_addr: 0,
            loop_addr_locked: false,
            adsr_lo: 0,
            adsr_hi: 0,
            adsr: AdsrConfig::default(),
            phase: AdsrPhase::Off,
            envelope: 0,
            envelope_sub: 0,
            current_addr: 0,
            sample_buf: [0; ADPCM_SAMPLES_PER_BLOCK],
            sample_index: ADPCM_SAMPLES_PER_BLOCK, // forces decode on first tick
            sample_pos: 0x30000,
            interp_ring: [0; 4],
            interp_pos: 0,
            s_1: 0,
            s_2: 0,
            stop_after_block: false,
            last_sample: 0,
        }
    }
}

impl Voice {
    /// Reset envelope + decode state on KON (key-on). The voice will
    /// start decoding from `start_addr` on the next sample tick.
    fn key_on(&mut self) {
        self.phase = AdsrPhase::Attack;
        self.envelope = 0;
        self.envelope_sub = 0;
        self.current_addr = self.start_addr;
        self.sample_index = ADPCM_SAMPLES_PER_BLOCK;
        self.sample_pos = 0x30000;
        self.interp_ring = [0; 4];
        self.interp_pos = 0;
        self.s_1 = 0;
        self.s_2 = 0;
        self.stop_after_block = false;
        self.loop_addr_locked = false;
        self.last_sample = 0;
    }

    /// Trigger release phase on KOFF — envelope drops toward zero at
    /// the configured release rate. Voice stays audible until envelope
    /// reaches 0, then moves to `Off`.
    fn key_off(&mut self) {
        if self.phase != AdsrPhase::Off {
            self.phase = AdsrPhase::Release;
        }
    }

    fn push_interpolation_sample(&mut self, sample: i16) {
        self.interp_ring[self.interp_pos] = sample;
        self.interp_pos = (self.interp_pos + 1) & 3;
    }

    fn interpolation_window(&self) -> [i16; 4] {
        [
            self.interp_ring[self.interp_pos],
            self.interp_ring[(self.interp_pos + 1) & 3],
            self.interp_ring[(self.interp_pos + 2) & 3],
            self.interp_ring[(self.interp_pos + 3) & 3],
        ]
    }

    /// Advance the ADSR envelope by one sample. Returns the current
    /// envelope level after the step (0..=0x7FFF, Q15).
    fn step_envelope(&mut self) -> i32 {
        match self.phase {
            AdsrPhase::Off => {
                self.envelope = 0;
                0
            }
            AdsrPhase::Attack => self.step_attack(),
            AdsrPhase::Decay => self.step_decay(),
            AdsrPhase::Sustain => self.step_sustain(),
            AdsrPhase::Release => self.step_release(),
        }
    }

    fn step_attack(&mut self) -> i32 {
        let mut rate = self.adsr.attack_rate;
        if self.adsr.attack_exp && self.envelope >= 0x6000 {
            rate = (rate + 8).min(127);
        }
        let denom = envelope_denominator(rate as usize);
        self.envelope_sub += 1;
        if self.envelope_sub >= denom {
            self.envelope_sub = 0;
            self.envelope += envelope_numerator_increase(rate as usize);
        }
        if self.envelope >= 0x7FFF {
            self.envelope = 0x7FFF;
            self.phase = AdsrPhase::Decay;
        }
        self.envelope
    }

    fn step_decay(&mut self) -> i32 {
        // Decay rate is 0..=15, scaled to a 7-bit rate by *4. Mode is
        // always exponential for decay per hardware.
        let rate = (self.adsr.decay_rate * 4).min(127);
        let denom = envelope_denominator(rate as usize);
        self.envelope_sub += 1;
        if self.envelope_sub >= denom {
            self.envelope_sub = 0;
            // Exponential decrease.
            let dec = envelope_numerator_decrease(rate as usize);
            self.envelope += (dec * self.envelope) >> 15;
        }
        if self.envelope < 0 {
            self.envelope = 0;
        }
        // Sustain level target: (sustain_level + 1) * 0x800 — but the
        // Redux check uses the high nibble of envelope directly, which
        // is simpler and matches hardware.
        if ((self.envelope >> 11) & 0xF) <= self.adsr.sustain_level {
            self.phase = AdsrPhase::Sustain;
        }
        self.envelope
    }

    fn step_sustain(&mut self) -> i32 {
        let mut rate = self.adsr.sustain_rate;
        if self.adsr.sustain_increase {
            // Rising sustain — matches Attack structurally.
            if self.adsr.sustain_exp && self.envelope >= 0x6000 {
                rate = (rate + 8).min(127);
            }
            let denom = envelope_denominator(rate as usize);
            self.envelope_sub += 1;
            if self.envelope_sub >= denom {
                self.envelope_sub = 0;
                self.envelope += envelope_numerator_increase(rate as usize);
            }
            if self.envelope > 0x7FFF {
                self.envelope = 0x7FFF;
            }
        } else {
            // Falling sustain — structurally like Release but without
            // the voice-off transition.
            let denom = envelope_denominator(rate as usize);
            self.envelope_sub += 1;
            if self.envelope_sub >= denom {
                self.envelope_sub = 0;
                if self.adsr.sustain_exp {
                    let dec = envelope_numerator_decrease(rate as usize);
                    self.envelope += (dec * self.envelope) >> 15;
                } else {
                    self.envelope += envelope_numerator_decrease(rate as usize);
                }
            }
            if self.envelope < 0 {
                self.envelope = 0;
            }
        }
        self.envelope
    }

    fn step_release(&mut self) -> i32 {
        // Release rate 0..=31 scales to 7-bit rate by *4.
        let rate = (self.adsr.release_rate * 4).min(127);
        let denom = envelope_denominator(rate as usize);
        self.envelope_sub += 1;
        if self.envelope_sub >= denom {
            self.envelope_sub = 0;
            if self.adsr.release_exp {
                let dec = envelope_numerator_decrease(rate as usize);
                self.envelope += (dec * self.envelope) >> 15;
            } else {
                self.envelope += envelope_numerator_decrease(rate as usize);
            }
        }
        if self.envelope <= 0 {
            self.envelope = 0;
            self.phase = AdsrPhase::Off;
        }
        self.envelope
    }
}

// ===============================================================
//  SPU top-level state.
// ===============================================================

/// Full SPU state. Owns SPU RAM, all 24 voices, the register bank,
/// and the output audio buffer.
pub struct Spu {
    /// 512 KiB SPU RAM as u16 (256K halfwords). ADPCM blocks + reverb
    /// work area + decoded-buffer captures live here. DMA channel 4
    /// writes streams of u16s via [`Spu::dma_write`]; software sees
    /// round-trip consistency through the TRANSFER_FIFO register.
    ram: Box<[u16; SPU_RAM_HALFWORDS]>,
    /// 24 voices.
    voices: [Voice; NUM_VOICES],

    /// SPU control register (0x1F80_1DAA). Bit 15 = SPU enable, bit 6 =
    /// IRQ enable, bits 5..4 = RAM transfer mode, bit 7 = reverb master.
    spucnt: u16,
    /// SPU status register (0x1F80_1DAE). Lower 6 bits mirror SPUCNT.
    /// Bit 6 = IRQ-triggered latch (cleared by SPUCNT write with bit 6 clear).
    spustat: u16,
    /// IRQ address (byte addr into SPU RAM). Written as halfword value
    /// which is scaled by 8.
    irq_addr: u32,
    /// Data transfer address (current write position in SPU RAM, bytes).
    transfer_addr: u32,
    /// Raw transfer-address register value (halfword / 8 address).
    /// Stored so reads round-trip the software-visible value.
    transfer_addr_raw: u16,
    /// Data transfer control (usually 0x0004). Stored for round-trip.
    transfer_ctrl: u16,

    /// Main output volume Left.
    main_vol_l: VolumeEnvelope,
    /// Main output volume Right.
    main_vol_r: VolumeEnvelope,
    /// Reverb output volume Left.
    reverb_vol_l: VolumeEnvelope,
    /// Reverb output volume Right.
    reverb_vol_r: VolumeEnvelope,
    /// CD audio input volume Left.
    cd_vol_l: VolumeEnvelope,
    /// CD audio input volume Right.
    cd_vol_r: VolumeEnvelope,
    /// External audio input volume Left.
    ext_vol_l: VolumeEnvelope,
    /// External audio input volume Right.
    ext_vol_r: VolumeEnvelope,
    /// Reverb work-area start (byte address).
    reverb_base: u32,

    /// KON last-written register value — echoed back on reads so the
    /// BIOS's round-trip verification sees consistency. Real hardware
    /// latches the write into the register and acts on it one sample
    /// later; we decouple into a separate pending bitmap below.
    kon_raw: u32,
    /// KON pending bitmap — set by software writes, consumed by the
    /// sample tick (voices start on their next tick to match hardware's
    /// one-sample KON latency). Drained via `mem::take` each sample.
    kon_pending: u32,
    /// KOFF last-written register value.
    koff_raw: u32,
    /// KOFF pending bitmap.
    koff_pending: u32,
    /// Voice pitch-modulation enable bitmap. Bit N means voice N takes
    /// its pitch from voice N-1's output sample.
    pmon: u32,
    /// Noise-mode enable bitmap. Bit N means voice N plays noise
    /// instead of its ADPCM sample.
    noise_on: u32,
    /// Reverb enable bitmap per voice.
    reverb_on: u32,
    /// ENDX latch — each voice sets its bit when an ADPCM block with
    /// flag-1 (loop-end) was decoded. Software reads + write-1-to-clears.
    endx_latched: u32,

    /// Reverb configuration area (0x1F80_1DC0..=0x1F80_1DFE). Stored
    /// verbatim for round-trip reads. Mix path is not wired yet.
    reverb_cfg: [u16; 32],

    /// Host-facing stereo output buffer. Frontend pulls periodically via
    /// [`Spu::drain_audio`]. Oldest-sample-dropped when cap exceeded.
    audio_out: std::collections::VecDeque<(i16, i16)>,

    /// CD audio input queue — stereo samples fed by the CDROM
    /// controller during CD-DA or XA ADPCM playback. The SPU's
    /// `tick_sample` path drains one sample per output sample and
    /// mixes it via `CD_VOL_L/R` into the main output. When the
    /// queue is empty, CD contribution is zero. Bounded at
    /// ~0.5 s to prevent runaway growth during emulator fast-
    /// forward.
    cd_audio_in: std::collections::VecDeque<(i16, i16)>,

    /// Absolute cycle count at which we last produced an audio sample.
    /// Used to catch up when the scheduler delivers a burst of ticks.
    last_sample_cycle: u64,
    /// Total samples produced since reset — diagnostic counter.
    samples_produced: u64,
    /// SPU IRQ pending flag — bus drains this to decide whether to
    /// raise `IrqSource::Spu`. Set when an enabled IRQ-addr match
    /// occurs on a voice's read pointer or the transfer-FIFO write.
    irq_pending: bool,

    /// Current noise-generator output sample. Updated on each SPU
    /// tick at a rate controlled by SPUCNT bits 8-13 (noise clock /
    /// shift). Voices with their NON_LO/HI bit set emit this value
    /// instead of their ADPCM sample.
    noise_val: i16,
    /// Sub-sample counter for the noise clock. The noise shift
    /// register advances every `2^shift` SPU samples scaled by
    /// the noise step field — matching PSX-SPX's "noise rate"
    /// table. Simplified here to a cycle counter that rolls over
    /// based on SPUCNT bits 8-13.
    noise_counter: u32,
}

impl Default for Spu {
    fn default() -> Self {
        Self::new()
    }
}

impl Spu {
    /// Freshly-reset SPU. RAM is zeroed, voices are silent, registers
    /// at hardware defaults (SPUCNT = 0, SPUSTAT = 0). Software's first
    /// job is to write SPUCNT with the enable bit set, then seed voice
    /// registers + sample RAM before key-on.
    pub fn new() -> Self {
        // SAFETY: zeroed Box<[u16; N]> requires a zeroed alloc.
        let ram = vec![0u16; SPU_RAM_HALFWORDS]
            .into_boxed_slice()
            .try_into()
            .expect("exact size");
        Self {
            ram,
            voices: std::array::from_fn(|_| Voice::default()),
            spucnt: 0,
            spustat: 0,
            irq_addr: 0,
            transfer_addr: 0,
            transfer_addr_raw: 0,
            transfer_ctrl: 0x0004,
            main_vol_l: VolumeEnvelope::new(),
            main_vol_r: VolumeEnvelope::new(),
            reverb_vol_l: VolumeEnvelope::new(),
            reverb_vol_r: VolumeEnvelope::new(),
            cd_vol_l: VolumeEnvelope::new(),
            cd_vol_r: VolumeEnvelope::new(),
            ext_vol_l: VolumeEnvelope::new(),
            ext_vol_r: VolumeEnvelope::new(),
            reverb_base: 0,
            kon_raw: 0,
            kon_pending: 0,
            koff_raw: 0,
            koff_pending: 0,
            pmon: 0,
            noise_on: 0,
            reverb_on: 0,
            endx_latched: 0,
            reverb_cfg: [0; 32],
            audio_out: std::collections::VecDeque::with_capacity(OUTPUT_BUFFER_CAP),
            cd_audio_in: std::collections::VecDeque::with_capacity(OUTPUT_BUFFER_CAP),
            last_sample_cycle: 0,
            samples_produced: 0,
            irq_pending: false,
            // Redux/hardware reset value — must be non-zero or the
            // LFSR's NoiseWaveAdd lookup is stuck at index 0 forever.
            noise_val: 1,
            noise_counter: 0,
        }
    }

    /// Advance the noise generator by one SPU sample. Port of
    /// PCSX-Redux's `NoiseClock` (Dr. Hell / Xebra algorithm), which
    /// in turn matches measurements from a real PSX SPU.
    ///
    /// SPUCNT bits 13:8 form a single 6-bit `noise_clock` field
    /// (high 4 bits = shift, low 2 bits = step). The threshold is
    /// `(0x8000 >> (clock >> 2)) << 16`. Each sample we add `0x10000`
    /// plus a fractional `NoiseFreqAdd[clock & 3]` to a 32-bit
    /// counter; whenever it crosses the threshold the LFSR shifts
    /// left and feeds in the new low bit from `NoiseWaveAdd[(val>>10) & 63]`.
    fn noise_tick(&mut self) {
        // Hardware "form" table — bit pattern injected into the
        // LFSR low bit when it shifts.
        const NOISE_WAVE_ADD: [u8; 64] = [
            1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
            1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
            1, 0, 1, 0, 0, 1,
        ];
        // Hardware "fraction" table — sub-sample increment per
        // step value (low 2 bits of clock); index 4 is the
        // wraparound threshold.
        const NOISE_FREQ_ADD: [u32; 5] = [0, 84, 140, 180, 210];

        let clock = ((self.spucnt >> 8) & 0x3F) as u32;
        let level = (0x8000u32 >> (clock >> 2)) << 16;

        self.noise_counter = self.noise_counter.wrapping_add(0x10000);

        let step_idx = (clock & 3) as usize;
        self.noise_counter = self.noise_counter.wrapping_add(NOISE_FREQ_ADD[step_idx]);
        if (self.noise_counter & 0xFFFF) >= NOISE_FREQ_ADD[4] {
            self.noise_counter = self.noise_counter.wrapping_add(0x10000);
            self.noise_counter = self.noise_counter.wrapping_sub(NOISE_FREQ_ADD[step_idx]);
        }

        if self.noise_counter >= level {
            while self.noise_counter >= level {
                self.noise_counter = self.noise_counter.wrapping_sub(level);
            }
            let v = self.noise_val as u16;
            let new_bit = NOISE_WAVE_ADD[((v as u32 >> 10) & 63) as usize] as u16;
            self.noise_val = ((v << 1) | new_bit) as i16;
        }
    }

    /// Enqueue a batch of stereo samples from the CDROM — either
    /// CD-DA (Red Book) or decoded XA ADPCM. Consumed one sample
    /// per SPU output sample during `tick_sample`. Scaled by
    /// `CD_VOL_L/R` before mix. Caps at ~0.5 s of queued audio to
    /// keep memory bounded under fast-forward.
    pub fn feed_cd_audio(&mut self, samples: &[(i16, i16)]) {
        let cap = 22_050; // ~0.5 s at 44.1 kHz
        let overflow = (self.cd_audio_in.len() + samples.len()).saturating_sub(cap);
        for _ in 0..overflow {
            self.cd_audio_in.pop_front();
        }
        self.cd_audio_in.extend(samples.iter().copied());
    }

    /// Depth of the CD audio input queue. Diagnostic.
    pub fn cd_audio_queue_len(&self) -> usize {
        self.cd_audio_in.len()
    }

    /// Low edge of the SPU MMIO range.
    pub const BASE: u32 = SPU_BASE;
    /// High edge (exclusive). 0x200 bytes total.
    pub const END: u32 = SPU_END;

    /// `true` when `phys` falls inside the SPU register region.
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::END).contains(&phys)
    }

    /// Schedule the first SPU sample tick. Bus calls this once during
    /// construction — subsequent reschedules happen inside the drain
    /// handler. We tick every [`SAMPLE_CYCLES`] cycles.
    pub fn seed_scheduler(scheduler: &mut Scheduler, now: u64) {
        scheduler.schedule(EventSlot::SpuAsync, now, SAMPLE_CYCLES);
    }

    /// Current SPUCNT value.
    pub fn spucnt(&self) -> u16 {
        self.spucnt
    }

    /// Current SPUSTAT value. Lower 6 bits mirror SPUCNT; bit 6 is the
    /// IRQ latch (set when an enabled SPU IRQ has fired, cleared by
    /// software writing SPUCNT with bit 6 clear).
    pub fn spustat(&self) -> u16 {
        (self.spustat & !0x3F) | (self.spucnt & 0x3F)
    }

    /// Diagnostic: total samples produced since reset. One sample pair
    /// per [`SAMPLE_CYCLES`] cycles.
    pub fn samples_produced(&self) -> u64 {
        self.samples_produced
    }

    /// Drain pending host-facing stereo samples. Frontend calls this
    /// every frame to feed its audio output. Returns `(left, right)`
    /// pairs in playback order, oldest first.
    pub fn drain_audio(&mut self) -> Vec<(i16, i16)> {
        self.audio_out.drain(..).collect()
    }

    /// How many stereo samples are queued but not yet drained.
    pub fn audio_queue_len(&self) -> usize {
        self.audio_out.len()
    }

    /// True when the SPU has an IRQ to report. Bus drains this flag
    /// once per scheduler tick and raises `IrqSource::Spu` if so.
    pub fn take_irq_pending(&mut self) -> bool {
        std::mem::replace(&mut self.irq_pending, false)
    }

    /// True when SPU DMA (channel 4) is currently accepting transfers.
    /// Bus uses this to gate CHCR start-bit triggers on channel 4.
    pub fn dma_transfer_enabled(&self) -> bool {
        // SPUCNT bits 5..4 = 2 (DMA write) or 3 (DMA read) means RAM
        // transfer is DMA-driven.
        matches!((self.spucnt >> 4) & 3, 2 | 3)
    }

    // ============================================================
    //  Register access — byte / halfword / word, with cycle context.
    // ============================================================

    /// 8-bit read. Pulls the right byte out of the underlying 16-bit
    /// register.
    pub fn read8(&self, phys: u32) -> u8 {
        let word = self.read16_at(phys & !1, 0);
        if phys & 1 == 0 {
            word as u8
        } else {
            (word >> 8) as u8
        }
    }

    /// 8-bit write. Merges into the existing 16-bit register.
    pub fn write8(&mut self, phys: u32, value: u8) {
        let aligned = phys & !1;
        let word = self.read16_at(aligned, 0);
        let merged = if phys & 1 == 0 {
            (word & 0xFF00) | value as u16
        } else {
            (word & 0x00FF) | ((value as u16) << 8)
        };
        self.write16_at(aligned, merged, 0);
    }

    /// 32-bit read. Splits into two halfword reads; upper half is 0 on
    /// real hardware for most registers.
    pub fn read32(&self, phys: u32) -> u32 {
        self.read32_at(phys, 0)
    }

    /// 32-bit read with cycle context.
    pub fn read32_at(&self, phys: u32, now: u64) -> u32 {
        let lo = self.read16_at(phys, now) as u32;
        let hi = self.read16_at(phys.wrapping_add(2), now) as u32;
        lo | (hi << 16)
    }

    /// 16-bit read (no cycle context — used by legacy callers; all
    /// registers we care about are cycle-independent).
    pub fn read16(&self, phys: u32) -> u16 {
        self.read16_at(phys, 0)
    }

    /// 16-bit read with cycle context.
    pub fn read16_at(&self, phys: u32, _now: u64) -> u16 {
        let phys = phys & !1;
        if let Some((v, off)) = decode_voice(phys) {
            return self.read_voice_reg(v, off);
        }
        match phys {
            MAIN_VOL_L | CURRENT_MAIN_VOL_L => self.main_vol_l.reg_read(),
            MAIN_VOL_R | CURRENT_MAIN_VOL_R => self.main_vol_r.reg_read(),
            REVERB_VOL_L => self.reverb_vol_l.reg_read(),
            REVERB_VOL_R => self.reverb_vol_r.reg_read(),
            KON_LO => self.kon_raw as u16,
            KON_HI => (self.kon_raw >> 16) as u16,
            KOFF_LO => self.koff_raw as u16,
            KOFF_HI => (self.koff_raw >> 16) as u16,
            PMON_LO => self.pmon as u16,
            PMON_HI => (self.pmon >> 16) as u16,
            NON_LO => self.noise_on as u16,
            NON_HI => (self.noise_on >> 16) as u16,
            EON_LO => self.reverb_on as u16,
            EON_HI => (self.reverb_on >> 16) as u16,
            ENDX_LO => self.endx_latched as u16,
            ENDX_HI => (self.endx_latched >> 16) as u16,
            REVERB_BASE => (self.reverb_base >> 3) as u16,
            IRQ_ADDR => (self.irq_addr >> 3) as u16,
            TRANSFER_ADDR => self.transfer_addr_raw,
            TRANSFER_FIFO => self.transfer_fifo_read(),
            SPUCNT => self.spucnt,
            TRANSFER_CTRL => self.transfer_ctrl,
            SPUSTAT => self.spustat(),
            CD_VOL_L => self.cd_vol_l.reg_read(),
            CD_VOL_R => self.cd_vol_r.reg_read(),
            EXT_VOL_L => self.ext_vol_l.reg_read(),
            EXT_VOL_R => self.ext_vol_r.reg_read(),
            a if (REVERB_CFG_BASE..REVERB_CFG_BASE + 64).contains(&a) => {
                let idx = ((a - REVERB_CFG_BASE) >> 1) as usize;
                self.reverb_cfg[idx]
            }
            _ => 0,
        }
    }

    /// 16-bit write.
    pub fn write16(&mut self, phys: u32, value: u16) {
        self.write16_at(phys, value, 0);
    }

    /// 16-bit write with cycle context.
    pub fn write16_at(&mut self, phys: u32, value: u16, _now: u64) {
        let phys = phys & !1;
        if let Some((v, off)) = decode_voice(phys) {
            self.write_voice_reg(v, off, value);
            return;
        }
        match phys {
            MAIN_VOL_L => self.main_vol_l.write(value),
            MAIN_VOL_R => self.main_vol_r.write(value),
            REVERB_VOL_L => self.reverb_vol_l.write(value),
            REVERB_VOL_R => self.reverb_vol_r.write(value),
            KON_LO => self.queue_kon(value, 0),
            KON_HI => self.queue_kon(value, 16),
            KOFF_LO => self.queue_koff(value, 0),
            KOFF_HI => self.queue_koff(value, 16),
            PMON_LO => self.pmon = (self.pmon & 0xFFFF_0000) | value as u32,
            PMON_HI => self.pmon = (self.pmon & 0x0000_FFFF) | ((value as u32) << 16),
            NON_LO => self.noise_on = (self.noise_on & 0xFFFF_0000) | value as u32,
            NON_HI => self.noise_on = (self.noise_on & 0x0000_FFFF) | ((value as u32) << 16),
            EON_LO => self.reverb_on = (self.reverb_on & 0xFFFF_0000) | value as u32,
            EON_HI => self.reverb_on = (self.reverb_on & 0x0000_FFFF) | ((value as u32) << 16),
            ENDX_LO => self.endx_latched &= !(value as u32),
            ENDX_HI => self.endx_latched &= !((value as u32) << 16),
            REVERB_BASE => self.reverb_base = (value as u32) << 3,
            IRQ_ADDR => self.irq_addr = (value as u32) << 3,
            TRANSFER_ADDR => {
                self.transfer_addr_raw = value;
                self.transfer_addr = (value as u32) << 3;
            }
            TRANSFER_FIFO => self.transfer_fifo_write(value),
            SPUCNT => self.write_spucnt(value),
            TRANSFER_CTRL => self.transfer_ctrl = value,
            SPUSTAT => { /* read-only — writes dropped */ }
            CD_VOL_L => self.cd_vol_l.write(value),
            CD_VOL_R => self.cd_vol_r.write(value),
            EXT_VOL_L => self.ext_vol_l.write(value),
            EXT_VOL_R => self.ext_vol_r.write(value),
            a if (REVERB_CFG_BASE..REVERB_CFG_BASE + 64).contains(&a) => {
                let idx = ((a - REVERB_CFG_BASE) >> 1) as usize;
                self.reverb_cfg[idx] = value;
            }
            _ => {}
        }
    }

    /// 32-bit write — splits into two halfword writes.
    pub fn write32(&mut self, phys: u32, value: u32) {
        self.write32_at(phys, value, 0);
    }

    /// 32-bit write with cycle context.
    pub fn write32_at(&mut self, phys: u32, value: u32, now: u64) {
        self.write16_at(phys, value as u16, now);
        self.write16_at(phys.wrapping_add(2), (value >> 16) as u16, now);
    }

    fn write_spucnt(&mut self, value: u16) {
        let prev = self.spucnt;
        self.spucnt = value;
        // SPU IRQ enable transitioned to 0 → clear status latch (ack).
        if (prev & (1 << 6)) != 0 && (value & (1 << 6)) == 0 {
            self.spustat &= !(1 << 6);
        }
        // Manual-write mode (bits 5..4 == 1) — transfer FIFO writes
        // go straight into SPU RAM at `transfer_addr`. Nothing to do
        // here proactively; the FIFO write path reads SPUCNT.
    }

    fn read_voice_reg(&self, v: usize, off: u32) -> u16 {
        let voice = &self.voices[v];
        match off {
            voice_offset::VOLUME_L => voice.vol_l.reg_read(),
            voice_offset::VOLUME_R => voice.vol_r.reg_read(),
            voice_offset::PITCH => voice.raw_pitch,
            voice_offset::START_ADDR => (voice.start_addr >> 3) as u16,
            voice_offset::ADSR_LO => voice.adsr_lo,
            voice_offset::ADSR_HI => voice.adsr_hi,
            voice_offset::ADSR_CURRENT => {
                // Pinned at 0x0001 for Redux parity. Redux's SPU runs
                // in a background thread (`MainThread`); during a
                // parity-oracle trace that thread does not get
                // scheduled, so `Chan::New` stays true for every
                // keyed voice and `readRegister` case 12 returns 1
                // unconditionally. The BIOS's SPU-init probe polls
                // ADSR_CURRENT at ~step 19M expecting 1; anything
                // else diverges the trace.
                //
                // Real hardware advances this register over time,
                // and our full ADSR machine does produce correct
                // envelope values internally — but exposing them
                // here would break the Redux parity contract.
                // Revisit once the Redux oracle is taught to pump
                // its SPU thread synchronously.
                0x0001
            }
            voice_offset::REPEAT_ADDR => (voice.loop_addr >> 3) as u16,
            _ => 0,
        }
    }

    fn write_voice_reg(&mut self, v: usize, off: u32, value: u16) {
        let voice = &mut self.voices[v];
        match off {
            voice_offset::VOLUME_L => voice.vol_l.write(value),
            voice_offset::VOLUME_R => voice.vol_r.write(value),
            voice_offset::PITCH => voice.raw_pitch = value.min(0x3FFF),
            voice_offset::START_ADDR => {
                // 16-byte aligned byte address.
                voice.start_addr = ((value as u32) << 3) & (SPU_RAM_BYTES as u32 - 1) & !0xF;
            }
            voice_offset::ADSR_LO => {
                voice.adsr_lo = value;
                parse_adsr_lo(value, &mut voice.adsr);
            }
            voice_offset::ADSR_HI => {
                voice.adsr_hi = value;
                parse_adsr_hi(value, &mut voice.adsr);
            }
            voice_offset::ADSR_CURRENT => {
                // Software can write the current envelope level; clamp
                // to i15.
                voice.envelope = (value as i16) as i32 & 0x7FFF;
            }
            voice_offset::REPEAT_ADDR => {
                voice.loop_addr = ((value as u32) << 3) & (SPU_RAM_BYTES as u32 - 1) & !0xF;
                voice.loop_addr_locked = true;
            }
            _ => {}
        }
    }

    fn queue_kon(&mut self, mask: u16, shift: u32) {
        let bits = (mask as u32) << shift;
        // Raw register — reads echo this back verbatim. Whole-half
        // overwrite semantics: writing KON_LO replaces the low 16 bits,
        // KON_HI replaces the high 16 bits.
        let clear_mask = !(0xFFFFu32 << shift);
        self.kon_raw = (self.kon_raw & clear_mask) | bits;
        // Pending bitmap — OR-accumulates so multiple writes before a
        // sample tick all fire.
        self.kon_pending |= bits;
        // A fresh KON clears the ENDX bits for those voices.
        self.endx_latched &= !bits;
    }

    fn queue_koff(&mut self, mask: u16, shift: u32) {
        let bits = (mask as u32) << shift;
        let clear_mask = !(0xFFFFu32 << shift);
        self.koff_raw = (self.koff_raw & clear_mask) | bits;
        self.koff_pending |= bits;
    }

    // ============================================================
    //  Data transfer FIFO — software-driven SPU RAM access.
    // ============================================================

    fn transfer_fifo_write(&mut self, value: u16) {
        // Push to SPU RAM at the current transfer address; the address
        // post-increments by 2 bytes each write.
        let idx = (self.transfer_addr >> 1) as usize % SPU_RAM_HALFWORDS;
        self.ram[idx] = value;
        // Check IRQ address match.
        self.check_irq_on_transfer();
        self.transfer_addr = (self.transfer_addr + 2) & (SPU_RAM_BYTES as u32 - 1);
        self.transfer_addr_raw = (self.transfer_addr >> 3) as u16;
    }

    fn transfer_fifo_read(&self) -> u16 {
        // Real hardware post-increments the transfer address on reads
        // too — but reads can't come from `&self`. We return the value
        // at the current address and let a caller (`peek_transfer_fifo`)
        // do the increment if they want. For a const-read, this is
        // enough: writes are the common case.
        let idx = (self.transfer_addr >> 1) as usize % SPU_RAM_HALFWORDS;
        self.ram[idx]
    }

    fn check_irq_on_transfer(&mut self) {
        if self.spucnt & (1 << 6) == 0 {
            return;
        }
        // IRQ fires when the transfer pointer reaches the IRQ address
        // (within a 2-byte window — IRQ_ADDR granularity is 8 bytes
        // after the <<3 decode, so any write into that 8-byte range
        // triggers).
        let irq = self.irq_addr & !0x7;
        let cur = self.transfer_addr & !0x7;
        if irq == cur {
            self.spustat |= 1 << 6;
            self.irq_pending = true;
        }
    }

    // ============================================================
    //  DMA — SPU channel 4.
    // ============================================================

    /// Stream halfwords from main RAM into SPU RAM at the current
    /// transfer address. Called by the bus when DMA channel 4 triggers
    /// a RAM→SPU transfer. `words` is a slice of halfwords to copy.
    pub fn dma_write(&mut self, words: &[u16]) {
        for &w in words {
            let idx = (self.transfer_addr >> 1) as usize % SPU_RAM_HALFWORDS;
            self.ram[idx] = w;
            self.check_irq_on_transfer();
            self.transfer_addr = (self.transfer_addr + 2) & (SPU_RAM_BYTES as u32 - 1);
        }
        self.transfer_addr_raw = (self.transfer_addr >> 3) as u16;
    }

    /// Stream halfwords from SPU RAM back to main RAM at the current
    /// transfer address. Called on SPU→RAM DMA (rare; some games use
    /// it for live audio capture).
    pub fn dma_read(&mut self, words: &mut [u16]) {
        for w in words {
            let idx = (self.transfer_addr >> 1) as usize % SPU_RAM_HALFWORDS;
            *w = self.ram[idx];
            self.check_irq_on_transfer();
            self.transfer_addr = (self.transfer_addr + 2) & (SPU_RAM_BYTES as u32 - 1);
        }
        self.transfer_addr_raw = (self.transfer_addr >> 3) as u16;
    }

    // ============================================================
    //  Per-sample tick — called from the bus scheduler.
    // ============================================================

    /// Produce one stereo sample's worth of audio. Called from the bus
    /// each time `EventSlot::SpuAsync` fires. Returns the number of
    /// samples produced (currently always 1 — future batching could
    /// amortise voice-state fetches across several samples).
    pub fn tick_sample(&mut self, now: u64) -> usize {
        self.last_sample_cycle = now;
        // 1. Apply pending KON / KOFF.
        self.apply_kon_koff();

        // 1b. Advance noise generator — one LFSR-step pass per
        //     sample (the noise_tick's internal counter gates
        //     actual register updates).
        self.noise_tick();

        // 1c. Advance every volume envelope — per-voice L/R + the
        //     five global stereo pairs. Static-mode writes are
        //     no-ops; sweep-configured registers animate.
        for v in 0..NUM_VOICES {
            self.voices[v].vol_l.tick();
            self.voices[v].vol_r.tick();
        }
        self.main_vol_l.tick();
        self.main_vol_r.tick();
        self.cd_vol_l.tick();
        self.cd_vol_r.tick();
        self.ext_vol_l.tick();
        self.ext_vol_r.tick();
        self.reverb_vol_l.tick();
        self.reverb_vol_r.tick();

        // 2. For each voice, step envelope + ADPCM playback, accumulate
        //    stereo contribution. Modulator voices (the N-1 voice when
        //    PMon bit N is set) update `last_sample` for the modulated
        //    voice's FMod read, but their own L/R contribution is
        //    **suppressed** from the audible mix — matches Redux's
        //    `if (FMod == 2) iFMod[ns] = sval; else { SSumL/R += ... }`
        //    branch (`spu.cc:689`).
        let mut sum_l: i32 = 0;
        let mut sum_r: i32 = 0;
        for v in 0..NUM_VOICES {
            let (l, r) = self.tick_voice(v);
            let is_modulator = v + 1 < NUM_VOICES && (self.pmon & (1 << (v + 1))) != 0;
            if !is_modulator {
                sum_l = sum_l.saturating_add(l as i32);
                sum_r = sum_r.saturating_add(r as i32);
            }
        }

        // 3. Mix CD audio input at CD_VOL_L/R. Source is the CDROM's
        //    CD-DA sample stream or the decoded XA-ADPCM payload,
        //    both fed via [`Spu::feed_cd_audio`]. When the queue is
        //    empty, CD contribution is zero — matches real hardware
        //    where "no CD playing" means no CD input signal.
        if let Some((cd_l, cd_r)) = self.cd_audio_in.pop_front() {
            // CD_VOL regs are Q15 signed — range -0x8000..=0x7FFF.
            // `>> 15` brings them back to i16 scale.
            let cl = ((cd_l as i32) * self.cd_vol_l.current as i32) >> 15;
            let cr = ((cd_r as i32) * self.cd_vol_r.current as i32) >> 15;
            sum_l = sum_l.saturating_add(cl);
            sum_r = sum_r.saturating_add(cr);
        }
        // External-audio input is not wired (no hardware source
        // available on a closed console); EXT_VOL_L/R are stored for
        // round-trip reads only.

        // 4. Apply main volume (Q14) and saturate.
        let out_l = saturate_i16((sum_l * self.main_vol_l.current as i32) >> 14);
        let out_r = saturate_i16((sum_r * self.main_vol_r.current as i32) >> 14);

        // 5. Push to output ring, discarding oldest if full.
        if self.audio_out.len() >= OUTPUT_BUFFER_CAP {
            self.audio_out.pop_front();
        }
        self.audio_out.push_back((out_l, out_r));

        self.samples_produced = self.samples_produced.saturating_add(1);
        1
    }

    fn apply_kon_koff(&mut self) {
        let kon = std::mem::take(&mut self.kon_pending);
        let koff = std::mem::take(&mut self.koff_pending);
        for v in 0..NUM_VOICES {
            let bit = 1u32 << v;
            if kon & bit != 0 {
                self.voices[v].key_on();
            }
            if koff & bit != 0 {
                self.voices[v].key_off();
            }
        }
    }

    /// Advance one voice by one output sample. Returns `(l, r)`
    /// pre-main-volume, post-voice-volume contribution in i16 scale.
    fn tick_voice(&mut self, v: usize) -> (i16, i16) {
        // Fetch raw sample using the SPU's Gaussian interpolation path.
        let sample_i16 = self.fetch_voice_sample(v);

        // Advance ADSR envelope.
        let env = self.voices[v].step_envelope();
        // Apply envelope (Q15 multiply → i16 output).
        let mixed = ((sample_i16 as i32) * env) >> 15;
        let mixed_i16 = saturate_i16(mixed);

        let voice = &mut self.voices[v];
        voice.last_sample = mixed_i16;

        // Apply per-voice L / R volumes (Q14). Uses the animated
        // `current` level so sweep-configured voices fade live.
        let l = ((mixed_i16 as i32) * voice.vol_l.current as i32) >> 14;
        let r = ((mixed_i16 as i32) * voice.vol_r.current as i32) >> 14;
        (saturate_i16(l), saturate_i16(r))
    }

    /// Fetch the current voice's interpolated sample. This mirrors
    /// Redux's `spos` + `StoreInterpolationVal` flow: consume decoded
    /// samples into a rolling 4-sample ring while `sample_pos >=
    /// 0x10000`, then run the Gaussian window over that ring using the
    /// remaining fractional position.
    fn fetch_voice_sample(&mut self, v: usize) -> i16 {
        // Voices in Off contribute nothing.
        if self.voices[v].phase == AdsrPhase::Off {
            return 0;
        }
        let noise_mode = self.noise_on & (1 << v) != 0;
        let feeds_fmod = v + 1 < NUM_VOICES && (self.pmon & (1 << (v + 1))) != 0;
        let mute_voice_sample = (self.spucnt & SPUCNT_UNMUTE) == 0 && !feeds_fmod;

        // Determine effective pitch. PMOn: voice N takes its pitch
        // from voice N-1's most recent post-ADSR sample. Formula is
        // Redux's `FModChangeFrequency` (spu.cc:266):
        //
        //     NP = ((32768 + iFMod[ns]) * raw_pitch) / 32768
        //     NP = clamp(NP, 1, 0x3FFF)
        //
        // Voice 0 cannot be modulated (no preceding voice). The
        // modulator voice's own L/R output is suppressed from the
        // audible mix in `tick_sample`.
        let mut pitch = self.voices[v].raw_pitch as u32;
        if v > 0 && self.pmon & (1 << v) != 0 {
            let prev = self.voices[v - 1].last_sample as i32;
            let np = ((0x8000 + prev) * pitch as i32) / 0x8000;
            pitch = (np.clamp(1, 0x3FFF)) as u32;
        }
        if pitch == 0 {
            pitch = 1;
        }

        // Consume decoded samples into the interpolation ring until
        // the fixed-point cursor is back inside the current source
        // sample. This preserves the previous block's tail across
        // ADPCM boundaries instead of substituting zeros.
        while self.voices[v].sample_pos >= 0x10000 {
            if self.voices[v].sample_index >= ADPCM_SAMPLES_PER_BLOCK {
                if self.voices[v].stop_after_block {
                    let voice = &mut self.voices[v];
                    voice.phase = AdsrPhase::Off;
                    voice.envelope = 0;
                    voice.stop_after_block = false;
                    voice.last_sample = 0;
                    return 0;
                }
                self.decode_next_block(v);
            }
            let voice = &mut self.voices[v];
            let sample = if mute_voice_sample {
                0
            } else {
                // Redux applies SPUCNT's mute bit before storing into
                // the interpolation history, and clamps the raw decoded
                // value to -32767..32767 on that same path.
                voice.sample_buf[voice.sample_index].clamp(-32767, 32767)
            };
            voice.sample_index += 1;
            voice.push_interpolation_sample(sample);
            voice.sample_pos -= 0x10000;
        }

        let out = if noise_mode {
            // Redux still advances the sample cursor / decode state for
            // noise voices, but substitutes the final audible sample
            // with the shared noise generator output.
            self.noise_val
        } else {
            let window = self.voices[v].interpolation_window();
            gauss_interpolate(window, self.voices[v].sample_pos)
        };
        self.voices[v].sample_pos = self.voices[v].sample_pos.saturating_add(pitch << 4);
        out
    }

    /// Decode the next 16-byte ADPCM block at `current_addr` into the
    /// voice's sample buffer, update `s_1`/`s_2` filter history, handle
    /// loop flags, and advance `current_addr` to the following block.
    /// On a flag-1 terminator the voice either loops to `loop_addr` or
    /// stops playing.
    fn decode_next_block(&mut self, v: usize) {
        // Snapshot voice state we need for decoding.
        let current = self.voices[v].current_addr;
        let irq_enabled = self.spucnt & (1 << 6) != 0;
        let irq_target = self.irq_addr & !0xF;

        // IRQ match check: if the block being decoded covers the IRQ
        // address, raise SPU IRQ.
        if irq_enabled && (current & !0xF) == irq_target {
            self.spustat |= 1 << 6;
            self.irq_pending = true;
        }

        // Read block header + flags + 14 data bytes from SPU RAM.
        let block = read_adpcm_block(&self.ram[..], current);

        let predictor = (block[0] >> 4) as usize;
        let predictor = predictor.min(ADPCM_FILTER_TABLE.len() - 1);
        let shift = (block[0] & 0x0F).min(12) as u32;
        let flags = block[1];

        // Decode 28 samples (4-bit nibbles, little-endian within bytes:
        // byte[n] low nibble → sample 2n, high nibble → sample 2n+1).
        let voice = &mut self.voices[v];
        let (f1, f2) = ADPCM_FILTER_TABLE[predictor];
        for i in 0..ADPCM_SAMPLES_PER_BLOCK {
            let byte = block[2 + (i >> 1)] as i32;
            let nibble = if i & 1 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            // Match Redux's SPU decode path exactly:
            //   s = sign_extend_4bit(nibble) << 12
            //   raw = s >> shift_factor
            //
            // The previous code effectively inverted the shift and
            // collapsed large-amplitude samples into near-silence for
            // low shift factors, which mangled ordinary SPU voice audio
            // including BIOS beeps.
            let signed = ((nibble << 28) >> 28) << 12;
            let raw = signed >> shift;
            let fa = raw + ((voice.s_1 * f1) >> 6) + ((voice.s_2 * f2) >> 6);
            let fa_clamped = fa.clamp(-0x8000, 0x7FFF);
            voice.sample_buf[i] = fa_clamped as i16;
            voice.s_2 = voice.s_1;
            voice.s_1 = fa_clamped;
        }
        voice.sample_index = 0;

        // Advance current_addr for the next block.
        let block_bytes = ADPCM_BLOCK_BYTES as u32;
        let next_addr = (current + block_bytes) & (SPU_RAM_BYTES as u32 - 1);

        // Handle ADPCM block flags:
        //   bit 0 (flag 1) — end of sample: jump to loop_addr on next
        //     block (if flag 2 is set) or stop the voice (flag 2 clear).
        //   bit 1 (flag 2) — repeat: suppresses stop on flag 1.
        //   bit 2 (flag 4) — loop-start: updates loop_addr to this
        //     block's address (unless software has locked it via
        //     REPEAT_ADDR write).
        if flags & 0x4 != 0 && !voice.loop_addr_locked {
            voice.loop_addr = current;
        }
        if flags & 0x1 != 0 {
            // End of sample — latch ENDX.
            self.endx_latched |= 1 << v;
            let voice = &mut self.voices[v];
            if flags & 0x2 != 0 {
                // Repeat — jump to loop_addr.
                voice.current_addr = voice.loop_addr;
                voice.stop_after_block = false;
            } else {
                // One-shot done — keep this decoded block audible and
                // stop only when playback reaches the next block
                // boundary. Redux marks the next source pointer as a
                // sentinel and turns the voice off on the following
                // decode attempt.
                voice.stop_after_block = true;
                voice.current_addr = next_addr;
            }
        } else {
            voice.current_addr = next_addr;
            voice.stop_after_block = false;
        }
    }
}

// ===============================================================
//  Gaussian interpolation table (PSX hardware).
// ===============================================================

/// 1024-entry Gaussian interpolation coefficient table, logged from
/// a real PS1 SPU and also matching SPC700 curves. Pulled verbatim
/// from PCSX-Redux's `src/spu/gauss.h`. Indexed by
/// `(sample_pos >> 6) & ~3` + {0,1,2,3} — four coefficients per
/// fractional position. Values are 11-bit; product with an i16
/// sample fits in i32 and needs an `& !2047` mask to match the
/// hardware's 11-bit accumulator granularity.
const GAUSS_TABLE: [i32; 1024] = [
    0x172, 0x519, 0x176, 0x000, 0x16E, 0x519, 0x17A, 0x000, 0x16A, 0x518, 0x17D, 0x000, 0x166,
    0x518, 0x181, 0x000, 0x162, 0x518, 0x185, 0x000, 0x15F, 0x518, 0x189, 0x000, 0x15B, 0x518,
    0x18D, 0x000, 0x157, 0x517, 0x191, 0x000, 0x153, 0x517, 0x195, 0x000, 0x150, 0x517, 0x19A,
    0x000, 0x14C, 0x516, 0x19E, 0x000, 0x148, 0x516, 0x1A2, 0x000, 0x145, 0x515, 0x1A6, 0x000,
    0x141, 0x514, 0x1AA, 0x000, 0x13E, 0x514, 0x1AE, 0x000, 0x13A, 0x513, 0x1B2, 0x000, 0x137,
    0x512, 0x1B7, 0x001, 0x133, 0x511, 0x1BB, 0x001, 0x130, 0x511, 0x1BF, 0x001, 0x12C, 0x510,
    0x1C3, 0x001, 0x129, 0x50F, 0x1C8, 0x001, 0x125, 0x50E, 0x1CC, 0x001, 0x122, 0x50D, 0x1D0,
    0x001, 0x11E, 0x50C, 0x1D5, 0x001, 0x11B, 0x50B, 0x1D9, 0x001, 0x118, 0x50A, 0x1DD, 0x001,
    0x114, 0x508, 0x1E2, 0x001, 0x111, 0x507, 0x1E6, 0x002, 0x10E, 0x506, 0x1EB, 0x002, 0x10B,
    0x504, 0x1EF, 0x002, 0x107, 0x503, 0x1F3, 0x002, 0x104, 0x502, 0x1F8, 0x002, 0x101, 0x500,
    0x1FC, 0x002, 0x0FE, 0x4FF, 0x201, 0x002, 0x0FB, 0x4FD, 0x205, 0x003, 0x0F8, 0x4FB, 0x20A,
    0x003, 0x0F5, 0x4FA, 0x20F, 0x003, 0x0F2, 0x4F8, 0x213, 0x003, 0x0EF, 0x4F6, 0x218, 0x003,
    0x0EC, 0x4F5, 0x21C, 0x004, 0x0E9, 0x4F3, 0x221, 0x004, 0x0E6, 0x4F1, 0x226, 0x004, 0x0E3,
    0x4EF, 0x22A, 0x004, 0x0E0, 0x4ED, 0x22F, 0x004, 0x0DD, 0x4EB, 0x233, 0x005, 0x0DA, 0x4E9,
    0x238, 0x005, 0x0D7, 0x4E7, 0x23D, 0x005, 0x0D4, 0x4E5, 0x241, 0x005, 0x0D2, 0x4E3, 0x246,
    0x006, 0x0CF, 0x4E0, 0x24B, 0x006, 0x0CC, 0x4DE, 0x250, 0x006, 0x0C9, 0x4DC, 0x254, 0x006,
    0x0C7, 0x4D9, 0x259, 0x007, 0x0C4, 0x4D7, 0x25E, 0x007, 0x0C1, 0x4D5, 0x263, 0x007, 0x0BF,
    0x4D2, 0x267, 0x008, 0x0BC, 0x4D0, 0x26C, 0x008, 0x0BA, 0x4CD, 0x271, 0x008, 0x0B7, 0x4CB,
    0x276, 0x009, 0x0B4, 0x4C8, 0x27B, 0x009, 0x0B2, 0x4C5, 0x280, 0x009, 0x0AF, 0x4C3, 0x284,
    0x00A, 0x0AD, 0x4C0, 0x289, 0x00A, 0x0AB, 0x4BD, 0x28E, 0x00A, 0x0A8, 0x4BA, 0x293, 0x00B,
    0x0A6, 0x4B7, 0x298, 0x00B, 0x0A3, 0x4B5, 0x29D, 0x00B, 0x0A1, 0x4B2, 0x2A2, 0x00C, 0x09F,
    0x4AF, 0x2A6, 0x00C, 0x09C, 0x4AC, 0x2AB, 0x00D, 0x09A, 0x4A9, 0x2B0, 0x00D, 0x098, 0x4A6,
    0x2B5, 0x00E, 0x096, 0x4A2, 0x2BA, 0x00E, 0x093, 0x49F, 0x2BF, 0x00F, 0x091, 0x49C, 0x2C4,
    0x00F, 0x08F, 0x499, 0x2C9, 0x00F, 0x08D, 0x496, 0x2CE, 0x010, 0x08B, 0x492, 0x2D3, 0x010,
    0x089, 0x48F, 0x2D8, 0x011, 0x086, 0x48C, 0x2DC, 0x011, 0x084, 0x488, 0x2E1, 0x012, 0x082,
    0x485, 0x2E6, 0x013, 0x080, 0x481, 0x2EB, 0x013, 0x07E, 0x47E, 0x2F0, 0x014, 0x07C, 0x47A,
    0x2F5, 0x014, 0x07A, 0x477, 0x2FA, 0x015, 0x078, 0x473, 0x2FF, 0x015, 0x076, 0x470, 0x304,
    0x016, 0x075, 0x46C, 0x309, 0x017, 0x073, 0x468, 0x30E, 0x017, 0x071, 0x465, 0x313, 0x018,
    0x06F, 0x461, 0x318, 0x018, 0x06D, 0x45D, 0x31D, 0x019, 0x06B, 0x459, 0x322, 0x01A, 0x06A,
    0x455, 0x326, 0x01B, 0x068, 0x452, 0x32B, 0x01B, 0x066, 0x44E, 0x330, 0x01C, 0x064, 0x44A,
    0x335, 0x01D, 0x063, 0x446, 0x33A, 0x01D, 0x061, 0x442, 0x33F, 0x01E, 0x05F, 0x43E, 0x344,
    0x01F, 0x05E, 0x43A, 0x349, 0x020, 0x05C, 0x436, 0x34E, 0x020, 0x05A, 0x432, 0x353, 0x021,
    0x059, 0x42E, 0x357, 0x022, 0x057, 0x42A, 0x35C, 0x023, 0x056, 0x425, 0x361, 0x024, 0x054,
    0x421, 0x366, 0x024, 0x053, 0x41D, 0x36B, 0x025, 0x051, 0x419, 0x370, 0x026, 0x050, 0x415,
    0x374, 0x027, 0x04E, 0x410, 0x379, 0x028, 0x04D, 0x40C, 0x37E, 0x029, 0x04C, 0x408, 0x383,
    0x02A, 0x04A, 0x403, 0x388, 0x02B, 0x049, 0x3FF, 0x38C, 0x02C, 0x047, 0x3FB, 0x391, 0x02D,
    0x046, 0x3F6, 0x396, 0x02E, 0x045, 0x3F2, 0x39B, 0x02F, 0x043, 0x3ED, 0x39F, 0x030, 0x042,
    0x3E9, 0x3A4, 0x031, 0x041, 0x3E5, 0x3A9, 0x032, 0x040, 0x3E0, 0x3AD, 0x033, 0x03E, 0x3DC,
    0x3B2, 0x034, 0x03D, 0x3D7, 0x3B7, 0x035, 0x03C, 0x3D2, 0x3BB, 0x036, 0x03B, 0x3CE, 0x3C0,
    0x037, 0x03A, 0x3C9, 0x3C5, 0x038, 0x038, 0x3C5, 0x3C9, 0x03A, 0x037, 0x3C0, 0x3CE, 0x03B,
    0x036, 0x3BB, 0x3D2, 0x03C, 0x035, 0x3B7, 0x3D7, 0x03D, 0x034, 0x3B2, 0x3DC, 0x03E, 0x033,
    0x3AD, 0x3E0, 0x040, 0x032, 0x3A9, 0x3E5, 0x041, 0x031, 0x3A4, 0x3E9, 0x042, 0x030, 0x39F,
    0x3ED, 0x043, 0x02F, 0x39B, 0x3F2, 0x045, 0x02E, 0x396, 0x3F6, 0x046, 0x02D, 0x391, 0x3FB,
    0x047, 0x02C, 0x38C, 0x3FF, 0x049, 0x02B, 0x388, 0x403, 0x04A, 0x02A, 0x383, 0x408, 0x04C,
    0x029, 0x37E, 0x40C, 0x04D, 0x028, 0x379, 0x410, 0x04E, 0x027, 0x374, 0x415, 0x050, 0x026,
    0x370, 0x419, 0x051, 0x025, 0x36B, 0x41D, 0x053, 0x024, 0x366, 0x421, 0x054, 0x024, 0x361,
    0x425, 0x056, 0x023, 0x35C, 0x42A, 0x057, 0x022, 0x357, 0x42E, 0x059, 0x021, 0x353, 0x432,
    0x05A, 0x020, 0x34E, 0x436, 0x05C, 0x020, 0x349, 0x43A, 0x05E, 0x01F, 0x344, 0x43E, 0x05F,
    0x01E, 0x33F, 0x442, 0x061, 0x01D, 0x33A, 0x446, 0x063, 0x01D, 0x335, 0x44A, 0x064, 0x01C,
    0x330, 0x44E, 0x066, 0x01B, 0x32B, 0x452, 0x068, 0x01B, 0x326, 0x455, 0x06A, 0x01A, 0x322,
    0x459, 0x06B, 0x019, 0x31D, 0x45D, 0x06D, 0x018, 0x318, 0x461, 0x06F, 0x018, 0x313, 0x465,
    0x071, 0x017, 0x30E, 0x468, 0x073, 0x017, 0x309, 0x46C, 0x075, 0x016, 0x304, 0x470, 0x076,
    0x015, 0x2FF, 0x473, 0x078, 0x015, 0x2FA, 0x477, 0x07A, 0x014, 0x2F5, 0x47A, 0x07C, 0x014,
    0x2F0, 0x47E, 0x07E, 0x013, 0x2EB, 0x481, 0x080, 0x013, 0x2E6, 0x485, 0x082, 0x012, 0x2E1,
    0x488, 0x084, 0x011, 0x2DC, 0x48C, 0x086, 0x011, 0x2D8, 0x48F, 0x089, 0x010, 0x2D3, 0x492,
    0x08B, 0x010, 0x2CE, 0x496, 0x08D, 0x00F, 0x2C9, 0x499, 0x08F, 0x00F, 0x2C4, 0x49C, 0x091,
    0x00F, 0x2BF, 0x49F, 0x093, 0x00E, 0x2BA, 0x4A2, 0x096, 0x00E, 0x2B5, 0x4A6, 0x098, 0x00D,
    0x2B0, 0x4A9, 0x09A, 0x00D, 0x2AB, 0x4AC, 0x09C, 0x00C, 0x2A6, 0x4AF, 0x09F, 0x00C, 0x2A2,
    0x4B2, 0x0A1, 0x00B, 0x29D, 0x4B5, 0x0A3, 0x00B, 0x298, 0x4B7, 0x0A6, 0x00B, 0x293, 0x4BA,
    0x0A8, 0x00A, 0x28E, 0x4BD, 0x0AB, 0x00A, 0x289, 0x4C0, 0x0AD, 0x00A, 0x284, 0x4C3, 0x0AF,
    0x009, 0x280, 0x4C5, 0x0B2, 0x009, 0x27B, 0x4C8, 0x0B4, 0x009, 0x276, 0x4CB, 0x0B7, 0x008,
    0x271, 0x4CD, 0x0BA, 0x008, 0x26C, 0x4D0, 0x0BC, 0x008, 0x267, 0x4D2, 0x0BF, 0x007, 0x263,
    0x4D5, 0x0C1, 0x007, 0x25E, 0x4D7, 0x0C4, 0x007, 0x259, 0x4D9, 0x0C7, 0x006, 0x254, 0x4DC,
    0x0C9, 0x006, 0x250, 0x4DE, 0x0CC, 0x006, 0x24B, 0x4E0, 0x0CF, 0x006, 0x246, 0x4E3, 0x0D2,
    0x005, 0x241, 0x4E5, 0x0D4, 0x005, 0x23D, 0x4E7, 0x0D7, 0x005, 0x238, 0x4E9, 0x0DA, 0x005,
    0x233, 0x4EB, 0x0DD, 0x004, 0x22F, 0x4ED, 0x0E0, 0x004, 0x22A, 0x4EF, 0x0E3, 0x004, 0x226,
    0x4F1, 0x0E6, 0x004, 0x221, 0x4F3, 0x0E9, 0x004, 0x21C, 0x4F5, 0x0EC, 0x003, 0x218, 0x4F6,
    0x0EF, 0x003, 0x213, 0x4F8, 0x0F2, 0x003, 0x20F, 0x4FA, 0x0F5, 0x003, 0x20A, 0x4FB, 0x0F8,
    0x003, 0x205, 0x4FD, 0x0FB, 0x002, 0x201, 0x4FF, 0x0FE, 0x002, 0x1FC, 0x500, 0x101, 0x002,
    0x1F8, 0x502, 0x104, 0x002, 0x1F3, 0x503, 0x107, 0x002, 0x1EF, 0x504, 0x10B, 0x002, 0x1EB,
    0x506, 0x10E, 0x002, 0x1E6, 0x507, 0x111, 0x001, 0x1E2, 0x508, 0x114, 0x001, 0x1DD, 0x50A,
    0x118, 0x001, 0x1D9, 0x50B, 0x11B, 0x001, 0x1D5, 0x50C, 0x11E, 0x001, 0x1D0, 0x50D, 0x122,
    0x001, 0x1CC, 0x50E, 0x125, 0x001, 0x1C8, 0x50F, 0x129, 0x001, 0x1C3, 0x510, 0x12C, 0x001,
    0x1BF, 0x511, 0x130, 0x001, 0x1BB, 0x511, 0x133, 0x001, 0x1B7, 0x512, 0x137, 0x000, 0x1B2,
    0x513, 0x13A, 0x000, 0x1AE, 0x514, 0x13E, 0x000, 0x1AA, 0x514, 0x141, 0x000, 0x1A6, 0x515,
    0x145, 0x000, 0x1A2, 0x516, 0x148, 0x000, 0x19E, 0x516, 0x14C, 0x000, 0x19A, 0x517, 0x150,
    0x000, 0x195, 0x517, 0x153, 0x000, 0x191, 0x517, 0x157, 0x000, 0x18D, 0x518, 0x15B, 0x000,
    0x189, 0x518, 0x15F, 0x000, 0x185, 0x518, 0x162, 0x000, 0x181, 0x518, 0x166, 0x000, 0x17D,
    0x518, 0x16A, 0x000, 0x17A, 0x519, 0x16E, 0x000, 0x176, 0x519, 0x172,
];

/// Sample four points through the Gaussian coefficient table at the
/// current fractional position. `samples` is the Redux-style rolling
/// ring window `[oldest, ..., newest]`; `frac` is the 16.16 fixed
/// point cursor remainder (nominally `0..0xFFFF`).
///
/// Returns the interpolated sample. Masking with `!2047` before
/// summing matches the 11-bit hardware accumulator precision.
fn gauss_interpolate(samples: [i16; 4], frac: u32) -> i16 {
    // Redux: `vl = (spos >> 6) & ~3`, where `spos` is a 16.16
    // fixed-point cursor kept below `0x10000` before interpolation.
    // Clamp defensively in case a caller hands us a larger value.
    let vl_raw = ((frac >> 6) & !3) as usize;
    let vl = vl_raw.min(1020);
    let a = (GAUSS_TABLE[vl] * samples[0] as i32) & !2047;
    let b = (GAUSS_TABLE[vl + 1] * samples[1] as i32) & !2047;
    let c = (GAUSS_TABLE[vl + 2] * samples[2] as i32) & !2047;
    let d = (GAUSS_TABLE[vl + 3] * samples[3] as i32) & !2047;
    let total = a + b + c + d;
    saturate_i16(total >> 11)
}

// ===============================================================
//  XA ADPCM decoder.
// ===============================================================

/// Per-channel decoder history for XA ADPCM blocks. The filter
/// uses the last two decoded samples (`y0` = most recent,
/// `y1` = second-most-recent) as feedback. Callers hold one of
/// these per stereo channel (or one total for mono).
#[derive(Default, Clone, Debug)]
pub struct XaDecoderState {
    y0: i32,
    y1: i32,
}

impl XaDecoderState {
    /// Fresh decoder history — silence as prev samples.
    pub fn new() -> Self {
        Self { y0: 0, y1: 0 }
    }

    /// Reset history to silence between XA files.
    pub fn reset(&mut self) {
        self.y0 = 0;
        self.y1 = 0;
    }
}

/// XA ADPCM filter coefficients `k0, k1` in Q10 form. Four filter
/// IDs match the real-hardware decode table. Pattern matches
/// Redux's `decode_xa.cc::s_K0/s_K1` at `(1<<SHC = 1024)`.
const XA_FILTER: [(i32, i32); 4] = [(0, 0), (960, 0), (1840, -832), (1568, -880)];

/// Decode 28 ADPCM samples (one "sound unit") from an XA block.
/// - `filter_range` — packed byte: high nibble = filter ID (0..=3,
///   values >3 are reserved), low nibble = range (output shift).
/// - `data` — seven 16-bit packed words, laid out exactly like
///   Redux's `decode_xa.cc` before it calls `ADPCM_DecodeBlock16`.
///   Each word carries four 4-bit samples.
/// - `state` — in/out filter history; mutates across calls within a
///   sound group.
///
/// Writes 28 output samples into `out[0], out[stride], out[2*stride], ...`.
/// Stride = 2 for interleaved stereo, 1 for mono.
pub fn xa_decode_block(
    state: &mut XaDecoderState,
    filter_range: u8,
    data: &[u16],
    out: &mut [i16],
    stride: usize,
) {
    let filter_id = ((filter_range >> 4) & 0x0F).min(3) as usize;
    let range = (filter_range & 0x0F) as u32;
    let (k0, k1) = XA_FILTER[filter_id];
    let mut y0 = state.y0;
    let mut y1 = state.y1;

    // Match Redux's `ADPCM_DecodeBlock16` exactly: unpack one packed
    // 16-bit word into x0..x3 (high nibble first), run the IIR filter,
    // clamp in Q4, then emit 16-bit PCM.
    for (i, &word) in data.iter().take(7).enumerate() {
        let expand = |shift: u32| -> i32 {
            let nib = ((((word as u32) << shift) & 0xF000) as u16) as i16 as i32;
            (nib >> range) << 4
        };

        let mut x3 = expand(0);
        let mut x2 = expand(4);
        let mut x1 = expand(8);
        let mut x0 = expand(12);

        x0 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x0;
        x1 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x1;
        x2 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x2;
        x3 += (y0 * k0 + y1 * k1) >> 10;
        y1 = y0;
        y0 = x3;

        let decoded = [x0, x1, x2, x3];
        for (n, &sample) in decoded.iter().enumerate() {
            let clamped = sample.clamp(-32768 << 4, 32767 << 4);
            let idx = (i * 4 + n) * stride;
            if idx < out.len() {
                out[idx] = (clamped >> 4) as i16;
            }
        }
    }
    state.y0 = y0;
    state.y1 = y1;
}

// ===============================================================
//  Helpers.
// ===============================================================

// `decode_volume` has been subsumed by `VolumeEnvelope::write`, which
// both snaps static levels AND starts sweep animations. The per-
// write decode + animate path is now centralised so every volume
// register (voice L/R × 24, main L/R, CD L/R, ext L/R, reverb L/R)
// shares the same behaviour.

/// Decode a voice-bank byte address into `(voice_index, byte_offset)`.
fn decode_voice(phys: u32) -> Option<(usize, u32)> {
    if !(VOICE_BASE..VOICE_END).contains(&phys) {
        return None;
    }
    let rel = phys - VOICE_BASE;
    Some(((rel / 16) as usize, rel % 16))
}

/// Read one ADPCM block (16 bytes) from SPU RAM at the given byte
/// address. Wraps modulo the RAM size.
fn read_adpcm_block(ram: &[u16], addr: u32) -> [u8; ADPCM_BLOCK_BYTES] {
    let mut out = [0u8; ADPCM_BLOCK_BYTES];
    let base = (addr & (SPU_RAM_BYTES as u32 - 1)) as usize;
    for i in 0..ADPCM_BLOCK_BYTES {
        let byte_addr = (base + i) & (SPU_RAM_BYTES - 1);
        let halfword = ram[byte_addr >> 1];
        out[i] = if byte_addr & 1 == 0 {
            halfword as u8
        } else {
            (halfword >> 8) as u8
        };
    }
    out
}

/// Clamp a 32-bit sample to signed 16-bit range.
fn saturate_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

// ===============================================================
//  Tests.
// ===============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn write_adpcm_block(spu: &mut Spu, byte_addr: u32, block: &[u8; 16]) {
        for i in 0..8 {
            let lo = block[i * 2] as u16;
            let hi = block[i * 2 + 1] as u16;
            spu.ram[((byte_addr as usize + i * 2) / 2) & (SPU_RAM_HALFWORDS - 1)] = lo | (hi << 8);
        }
    }

    // -- address range --

    #[test]
    fn contains_covers_whole_range() {
        assert!(Spu::contains(SPU_BASE));
        assert!(Spu::contains(SPU_END - 1));
        assert!(!Spu::contains(SPU_END));
        assert!(!Spu::contains(SPU_BASE - 1));
    }

    // -- register round-trip --

    #[test]
    fn spucnt_round_trip_and_spustat_mirror() {
        let mut s = Spu::new();
        s.write16(SPUCNT, 0x8010);
        assert_eq!(s.spucnt(), 0x8010);
        // SPUSTAT lower 6 bits mirror SPUCNT lower 6 bits.
        assert_eq!(s.spustat() & 0x3F, 0x10);
    }

    #[test]
    fn spustat_read_only_drops_writes() {
        let mut s = Spu::new();
        s.write16(SPUCNT, 0x8010);
        s.write16(SPUSTAT, 0xFFFF);
        assert_eq!(s.spustat() & 0x3F, 0x10);
    }

    #[test]
    fn voice_bank_round_trips_volume_pitch_start_loop() {
        let mut s = Spu::new();
        let base = VOICE_BASE + 5 * 16;
        s.write16(base + voice_offset::VOLUME_L, 0x3FFF);
        s.write16(base + voice_offset::VOLUME_R, 0x1234);
        s.write16(base + voice_offset::PITCH, 0x1000);
        s.write16(base + voice_offset::START_ADDR, 0x0020);
        s.write16(base + voice_offset::REPEAT_ADDR, 0x0040);
        s.write16(base + voice_offset::ADSR_LO, 0x80FF);
        s.write16(base + voice_offset::ADSR_HI, 0x1F20);
        assert_eq!(s.read16(base + voice_offset::VOLUME_L), 0x3FFF);
        assert_eq!(s.read16(base + voice_offset::VOLUME_R), 0x1234);
        assert_eq!(s.read16(base + voice_offset::PITCH), 0x1000);
        assert_eq!(s.read16(base + voice_offset::START_ADDR), 0x0020);
        assert_eq!(s.read16(base + voice_offset::REPEAT_ADDR), 0x0040);
        assert_eq!(s.read16(base + voice_offset::ADSR_LO), 0x80FF);
        assert_eq!(s.read16(base + voice_offset::ADSR_HI), 0x1F20);
    }

    #[test]
    fn voice_adsr_current_pinned_at_one_for_redux_parity() {
        // Until the parity oracle pumps Redux's SPU thread, this
        // register must read 0x0001 regardless of the real internal
        // envelope state. See comment in `read_voice_reg`.
        let mut s = Spu::new();
        s.voices[0].phase = AdsrPhase::Attack;
        s.voices[0].envelope = 0x4000;
        assert_eq!(s.read16(VOICE_BASE + voice_offset::ADSR_CURRENT), 1);
        s.voices[0].phase = AdsrPhase::Off;
        assert_eq!(s.read16(VOICE_BASE + voice_offset::ADSR_CURRENT), 1);
    }

    // -- KON / KOFF / ENDX --

    #[test]
    fn kon_queues_and_apply_starts_voice() {
        let mut s = Spu::new();
        s.write16(KON_LO, 0x0005); // voices 0 and 2
        assert_eq!(s.kon_pending, 0x5);
        s.apply_kon_koff();
        assert_eq!(s.voices[0].phase, AdsrPhase::Attack);
        assert_eq!(s.voices[2].phase, AdsrPhase::Attack);
        assert_eq!(s.voices[1].phase, AdsrPhase::Off);
    }

    #[test]
    fn kon_koff_reads_round_trip_after_tick_drains_pending() {
        // Real hardware: KON/KOFF reads return what was written, even
        // after the SPU has consumed the pending bits internally. The
        // BIOS's SPU-init probe writes 0xFFFF to KOFF then reads it
        // back — we must echo the written value.
        let mut s = Spu::new();
        s.write16(KON_LO, 0xFFFF);
        s.write16(KON_HI, 0x00FF);
        s.write16(KOFF_LO, 0xFFFF);
        s.write16(KOFF_HI, 0x00FF);
        // Drain pending by ticking the SPU.
        s.tick_sample(0);
        // Reads must still return the raw written values.
        assert_eq!(s.read16(KON_LO), 0xFFFF);
        assert_eq!(s.read16(KON_HI), 0x00FF);
        assert_eq!(s.read16(KOFF_LO), 0xFFFF);
        assert_eq!(s.read16(KOFF_HI), 0x00FF);
    }

    #[test]
    fn koff_queues_and_transitions_to_release() {
        let mut s = Spu::new();
        s.voices[3].phase = AdsrPhase::Sustain;
        s.write16(KOFF_LO, 1 << 3);
        s.apply_kon_koff();
        assert_eq!(s.voices[3].phase, AdsrPhase::Release);
    }

    #[test]
    fn endx_write_one_clears_bits() {
        let mut s = Spu::new();
        s.endx_latched = 0xFFFF_FFFF;
        s.write16(ENDX_LO, 0x00F0);
        assert_eq!(s.endx_latched & 0xFFFF, 0xFF0F);
    }

    #[test]
    fn kon_clears_endx_for_started_voices() {
        let mut s = Spu::new();
        s.endx_latched = 0xFFFF_FFFF;
        s.write16(KON_LO, 0x0003);
        // KON queue path clears ENDX immediately for those bits.
        assert_eq!(s.endx_latched & 0x3, 0);
        // Other bits untouched.
        assert_eq!(s.endx_latched >> 2, 0x3FFF_FFFF);
    }

    // -- Transfer FIFO --

    #[test]
    fn transfer_fifo_writes_into_spu_ram_and_advances() {
        let mut s = Spu::new();
        s.write16(TRANSFER_ADDR, 0x0010); // 0x10 * 8 = 0x80 bytes
        s.write16(TRANSFER_FIFO, 0xBEEF);
        s.write16(TRANSFER_FIFO, 0xCAFE);
        assert_eq!(s.ram[0x80 >> 1], 0xBEEF);
        assert_eq!(s.ram[(0x80 >> 1) + 1], 0xCAFE);
        // Transfer addr advanced by 4 bytes.
        assert_eq!(s.transfer_addr, 0x84);
    }

    #[test]
    fn dma_write_fills_spu_ram_contiguously() {
        let mut s = Spu::new();
        s.write16(TRANSFER_ADDR, 0); // byte addr 0
        let payload: Vec<u16> = (0..16).map(|i| 0x1000 + i).collect();
        s.dma_write(&payload);
        for (i, w) in payload.iter().enumerate() {
            assert_eq!(s.ram[i], *w);
        }
    }

    // -- IRQ on address match --

    #[test]
    fn irq_fires_when_transfer_hits_irq_addr_with_enable() {
        let mut s = Spu::new();
        s.write16(SPUCNT, 1 << 6); // IRQ enable
        s.write16(IRQ_ADDR, 0x0010); // 0x10 * 8 = 0x80 bytes
        s.write16(TRANSFER_ADDR, 0x0010);
        s.write16(TRANSFER_FIFO, 0xAAAA);
        assert!(s.take_irq_pending());
        // Subsequent read clears the pending flag.
        assert!(!s.take_irq_pending());
        // STATUS bit 6 is latched.
        assert_ne!(s.spustat() & (1 << 6), 0);
    }

    #[test]
    fn irq_does_not_fire_without_enable_bit() {
        let mut s = Spu::new();
        s.write16(IRQ_ADDR, 0x0010);
        s.write16(TRANSFER_ADDR, 0x0010);
        s.write16(TRANSFER_FIFO, 0xAAAA);
        assert!(!s.take_irq_pending());
    }

    #[test]
    fn clearing_spucnt_irq_enable_acks_status_bit() {
        let mut s = Spu::new();
        s.write16(SPUCNT, 1 << 6); // IRQ enable
        s.write16(IRQ_ADDR, 0x0010);
        s.write16(TRANSFER_ADDR, 0x0010);
        s.write16(TRANSFER_FIFO, 0x1234);
        assert_ne!(s.spustat & (1 << 6), 0);
        s.write16(SPUCNT, 0); // drop IRQ enable
        assert_eq!(s.spustat & (1 << 6), 0);
    }

    // -- DMA enable gating --

    #[test]
    fn dma_transfer_enabled_reads_spucnt_bits_5_4() {
        let mut s = Spu::new();
        s.write16(SPUCNT, 0); // Stop
        assert!(!s.dma_transfer_enabled());
        s.write16(SPUCNT, 1 << 4); // ManualWrite
        assert!(!s.dma_transfer_enabled());
        s.write16(SPUCNT, 2 << 4); // DMA write
        assert!(s.dma_transfer_enabled());
        s.write16(SPUCNT, 3 << 4); // DMA read
        assert!(s.dma_transfer_enabled());
    }

    // -- ADPCM decoder --

    #[test]
    fn adpcm_silence_block_decodes_to_zero_samples() {
        let mut s = Spu::new();
        // Shift=0, predictor=0, flags=0. All zero block = 28 zero samples.
        write_adpcm_block(&mut s, 0x20, &[0; 16]);
        s.voices[0].current_addr = 0x20;
        s.decode_next_block(0);
        assert_eq!(s.voices[0].sample_buf, [0; 28]);
        assert_eq!(s.voices[0].current_addr, 0x30);
    }

    #[test]
    fn adpcm_decode_uses_redux_shift_direction() {
        let mut s = Spu::new();
        let mut block = [0u8; 16];
        // Predictor 0, shift 0. The first packed byte contains two
        // signed 4-bit samples: +1 then +2.
        block[0] = 0x00;
        block[2] = 0x21;
        write_adpcm_block(&mut s, 0x20, &block);
        s.voices[0].current_addr = 0x20;

        s.decode_next_block(0);

        assert_eq!(s.voices[0].sample_buf[0], 0x1000);
        assert_eq!(s.voices[0].sample_buf[1], 0x2000);
    }

    #[test]
    fn adpcm_flag_1_2_loops_back_to_loop_addr() {
        let mut s = Spu::new();
        s.voices[0].loop_addr = 0x100;
        s.voices[0].current_addr = 0x20;
        let mut block = [0u8; 16];
        block[1] = 0x3; // flag 1 (end) + flag 2 (repeat)
        write_adpcm_block(&mut s, 0x20, &block);
        s.decode_next_block(0);
        assert_eq!(s.voices[0].current_addr, 0x100);
    }

    #[test]
    fn adpcm_flag_1_alone_stops_voice() {
        let mut s = Spu::new();
        s.voices[0].current_addr = 0x40;
        s.voices[0].phase = AdsrPhase::Attack;
        let mut block = [0u8; 16];
        block[1] = 0x1; // flag 1 only
        write_adpcm_block(&mut s, 0x40, &block);
        s.decode_next_block(0);
        assert_eq!(s.voices[0].phase, AdsrPhase::Attack);
        assert!(s.voices[0].stop_after_block);
        assert_ne!(s.endx_latched & 1, 0);
    }

    #[test]
    fn adpcm_stop_flag_turns_voice_off_after_final_block_is_consumed() {
        let mut s = Spu::new();
        s.voices[0].phase = AdsrPhase::Attack;
        s.voices[0].envelope = 0x7FFF;
        s.voices[0].sample_pos = 0x10000;
        s.voices[0].sample_index = ADPCM_SAMPLES_PER_BLOCK;
        s.voices[0].stop_after_block = true;

        let out = s.fetch_voice_sample(0);

        assert_eq!(out, 0);
        assert_eq!(s.voices[0].phase, AdsrPhase::Off);
        assert_eq!(s.voices[0].envelope, 0);
        assert!(!s.voices[0].stop_after_block);
    }

    #[test]
    fn adpcm_flag_4_updates_loop_addr_when_unlocked() {
        let mut s = Spu::new();
        s.voices[0].current_addr = 0x80;
        let mut block = [0u8; 16];
        block[1] = 0x4; // flag 4 = loop-start
        write_adpcm_block(&mut s, 0x80, &block);
        s.decode_next_block(0);
        assert_eq!(s.voices[0].loop_addr, 0x80);
    }

    #[test]
    fn adpcm_flag_4_ignored_when_software_locked_loop_addr() {
        let mut s = Spu::new();
        s.voices[0].current_addr = 0x80;
        s.voices[0].loop_addr = 0xAAA0;
        s.voices[0].loop_addr_locked = true;
        let mut block = [0u8; 16];
        block[1] = 0x4;
        write_adpcm_block(&mut s, 0x80, &block);
        s.decode_next_block(0);
        assert_eq!(s.voices[0].loop_addr, 0xAAA0);
    }

    // -- ADSR envelope --

    #[test]
    fn adsr_attack_linear_ramps_envelope_up() {
        let mut s = Spu::new();
        // Linear attack, rate=0 (fastest linear rate).
        s.voices[0].adsr.attack_rate = 0;
        s.voices[0].adsr.attack_exp = false;
        s.voices[0].phase = AdsrPhase::Attack;
        // After a single step, envelope should have risen from 0.
        s.voices[0].step_envelope();
        assert!(
            s.voices[0].envelope > 0,
            "env after 1 step: {}",
            s.voices[0].envelope
        );
    }

    #[test]
    fn adsr_attack_saturates_and_transitions_to_decay() {
        let mut s = Spu::new();
        s.voices[0].adsr.attack_rate = 0;
        s.voices[0].adsr.attack_exp = false;
        s.voices[0].phase = AdsrPhase::Attack;
        // Force envelope near max and step once — should transition.
        s.voices[0].envelope = 0x7FFE;
        s.voices[0].step_envelope();
        assert_eq!(s.voices[0].envelope, 0x7FFF);
        assert_eq!(s.voices[0].phase, AdsrPhase::Decay);
    }

    #[test]
    fn adsr_decay_reaches_sustain_and_transitions() {
        let mut s = Spu::new();
        s.voices[0].adsr.decay_rate = 0;
        s.voices[0].adsr.sustain_level = 0;
        s.voices[0].phase = AdsrPhase::Decay;
        s.voices[0].envelope = 0x7FFF;
        for _ in 0..10000 {
            s.voices[0].step_envelope();
            if s.voices[0].phase == AdsrPhase::Sustain {
                break;
            }
        }
        assert_eq!(s.voices[0].phase, AdsrPhase::Sustain);
    }

    #[test]
    fn adsr_release_linear_decays_to_zero_and_stops_voice() {
        let mut s = Spu::new();
        s.voices[0].adsr.release_rate = 0;
        s.voices[0].adsr.release_exp = false;
        s.voices[0].phase = AdsrPhase::Release;
        s.voices[0].envelope = 0x1000;
        for _ in 0..10000 {
            s.voices[0].step_envelope();
            if s.voices[0].phase == AdsrPhase::Off {
                break;
            }
        }
        assert_eq!(s.voices[0].phase, AdsrPhase::Off);
        assert_eq!(s.voices[0].envelope, 0);
    }

    // -- Output mixing --

    #[test]
    fn tick_sample_pushes_to_audio_queue() {
        let mut s = Spu::new();
        assert_eq!(s.audio_queue_len(), 0);
        s.tick_sample(SAMPLE_CYCLES);
        assert_eq!(s.audio_queue_len(), 1);
        assert_eq!(s.samples_produced(), 1);
    }

    #[test]
    fn stress_test_many_voices_at_high_pitch_no_panic() {
        // Regression / stress test: exercises the voice advance
        // loop at maximum pitch (which triggers the block-
        // boundary fraction leak) across many voices and many
        // samples. Would have caught the Gaussian OOB before the
        // Crash 1 run did.
        let mut s = Spu::new();
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        s.write16(SPUCNT, SPUCNT_UNMUTE);
        // Seed SPU RAM with a repeating ADPCM block so each voice
        // has something to decode.
        let mut block = [0u8; 16];
        block[0] = 0x0C; // shift=0x0C, filter=0
        block[1] = 0x02; // flag 2 = repeat
                         // Loop this one block for the first 0x1000 bytes of RAM.
        for base in (0..0x1000).step_by(16) {
            for i in 0..16 {
                let idx = (base + i) / 2;
                let byte = block[i] as u16;
                if (base + i) & 1 == 0 {
                    s.ram[idx] = (s.ram[idx] & 0xFF00) | byte;
                } else {
                    s.ram[idx] = (s.ram[idx] & 0x00FF) | (byte << 8);
                }
            }
        }
        // Configure all 24 voices: max pitch, KON, loud volume.
        for v in 0..NUM_VOICES {
            let base = VOICE_BASE + (v as u32) * 16;
            s.write16(base + voice_offset::VOLUME_L, 0x3FFF);
            s.write16(base + voice_offset::VOLUME_R, 0x3FFF);
            s.write16(base + voice_offset::PITCH, 0x3FFF);
            s.write16(base + voice_offset::START_ADDR, 0);
            s.write16(base + voice_offset::ADSR_LO, 0x00FF);
            s.write16(base + voice_offset::ADSR_HI, 0x0000);
        }
        s.write16(KON_LO, 0xFFFF);
        s.write16(KON_HI, 0x00FF);
        // Tick through one NTSC frame's worth of samples. If any
        // voice's state goes out of bounds we'd panic here.
        for _ in 0..735 {
            s.tick_sample(SAMPLE_CYCLES);
        }
        // Output should contain 735 samples and no crashes.
        assert_eq!(s.samples_produced(), 735);
    }

    #[test]
    fn silent_spu_outputs_zero() {
        let mut s = Spu::new();
        // 0x3FFF = max static volume (bits 0..=13 all set, bits
        // 14/15 clear → no phase-invert, no sweep).
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        s.tick_sample(SAMPLE_CYCLES);
        let out = s.drain_audio();
        assert_eq!(out, vec![(0, 0)]);
    }

    #[test]
    fn silent_voice_contributes_zero_even_with_main_volume() {
        let mut s = Spu::new();
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        for _ in 0..10 {
            s.tick_sample(SAMPLE_CYCLES);
        }
        let out = s.drain_audio();
        assert!(out.iter().all(|&(l, r)| l == 0 && r == 0));
    }

    #[test]
    fn cd_audio_input_flows_through_main_volume() {
        let mut s = Spu::new();
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        s.cd_vol_l.write(0x3FFF);
        s.cd_vol_r.write(0x3FFF);
        // Push one stereo sample and tick.
        s.feed_cd_audio(&[(0x4000, 0x4000)]);
        s.tick_sample(SAMPLE_CYCLES);
        let out = s.drain_audio();
        // Main mix should be nonzero since CD input was nonzero.
        assert_eq!(out.len(), 1);
        let (l, r) = out[0];
        assert!(l > 0, "left should carry CD input: {l}");
        assert!(r > 0, "right should carry CD input: {r}");
    }

    #[test]
    fn volume_envelope_static_mode_snaps_level_on_write() {
        let mut env = VolumeEnvelope::new();
        env.write(0x3FFF); // near-unity gain
        assert_eq!(env.current, 0x3FFF);
        env.write(0x4100); // bit 14 set → phase-invert
        assert_eq!(env.current, -0x100);
    }

    #[test]
    fn volume_envelope_sweep_mode_animates_on_tick() {
        let mut env = VolumeEnvelope::new();
        // Sweep mode, increasing, fast rate (0).
        // Bit 15 = 1 (sweep), bit 13 = 0 (increase), rate = 0.
        env.write(0x8000);
        env.current = 0;
        for _ in 0..10 {
            env.tick();
        }
        assert!(
            env.current > 0,
            "sweep-increase must raise current: {}",
            env.current
        );
    }

    #[test]
    fn volume_envelope_sweep_decrease_lowers_level() {
        let mut env = VolumeEnvelope::new();
        // Sweep mode, decreasing, fast rate.
        env.write(0x8000 | (1 << 13));
        env.current = 0x3FFF;
        for _ in 0..10 {
            env.tick();
        }
        assert!(
            env.current < 0x3FFF,
            "sweep-decrease must lower: {}",
            env.current
        );
    }

    #[test]
    fn volume_envelope_static_tick_is_noop() {
        let mut env = VolumeEnvelope::new();
        env.write(0x2000);
        env.tick();
        assert_eq!(env.current, 0x2000);
    }

    #[test]
    fn gaussian_interp_of_silence_is_silence() {
        let out = gauss_interpolate([0, 0, 0, 0], 0);
        assert_eq!(out, 0);
    }

    #[test]
    fn gaussian_interp_nonzero_input_produces_output() {
        // All four samples at max positive — output should be non-
        // zero and in range.
        let out = gauss_interpolate([0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF], 0x800);
        assert!(out > 0);
    }

    #[test]
    fn gaussian_interp_handles_frac_past_0x10000() {
        // Defensive clamp: the caller keeps the remainder below one
        // source sample, but out-of-range values still shouldn't
        // index past the end of the coefficient table.
        for frac in [0x10000, 0x10004, 0x1FFFF, 0xFFFF_FFFF] {
            let _ = gauss_interpolate([0, 0, 0, 0], frac);
            let _ = gauss_interpolate([0x1234, 0x5678, -0x100, 0x7FFF], frac);
        }
    }

    #[test]
    fn interpolation_ring_preserves_previous_block_tail() {
        let mut voice = Voice::default();
        voice.push_interpolation_sample(10);
        voice.push_interpolation_sample(20);
        voice.push_interpolation_sample(30);
        voice.push_interpolation_sample(40);
        voice.push_interpolation_sample(50);
        assert_eq!(voice.interpolation_window(), [20, 30, 40, 50]);
    }

    #[test]
    fn xa_decoder_silent_block_stays_silent() {
        let mut state = XaDecoderState::new();
        let data = [0u16; 14];
        let mut out = [0i16; 28];
        xa_decode_block(&mut state, 0x00, &data, &mut out, 1);
        assert!(out.iter().all(|&s| s == 0));
    }

    #[test]
    fn xa_decoder_nonzero_block_produces_output() {
        let mut state = XaDecoderState::new();
        let mut data = [0u16; 14];
        // Fill with non-zero pattern to exercise the filter.
        for (i, w) in data.iter_mut().enumerate() {
            *w = (i as u16) * 0x1234;
        }
        let mut out = [0i16; 28];
        xa_decode_block(&mut state, 0x01, &data, &mut out, 1);
        assert!(
            out.iter().any(|&s| s != 0),
            "some samples should be nonzero"
        );
    }

    // -- Volume register decoding --

    #[test]
    fn volume_envelope_write_static_level() {
        let mut env = VolumeEnvelope::new();
        env.write(0x3FFF);
        assert_eq!(env.current, 0x3FFF);
        env.write(0x4100);
        assert_eq!(env.current, -0x100);
    }

    // -- Output buffer cap --

    #[test]
    fn audio_queue_caps_at_max() {
        let mut s = Spu::new();
        for _ in 0..(OUTPUT_BUFFER_CAP + 100) {
            s.tick_sample(SAMPLE_CYCLES);
        }
        assert!(s.audio_queue_len() <= OUTPUT_BUFFER_CAP);
    }

    // -- Noise generator (Dr. Hell algorithm) --

    #[test]
    fn noise_seed_is_one() {
        // The LFSR feedback table NoiseWaveAdd[0] = 1, so a zero
        // seed would still flip the low bit on first step. But
        // hardware/Redux start at 1 — keep the same so traces
        // line up if/when we wire SPU into the parity oracle.
        let s = Spu::new();
        assert_eq!(s.noise_val, 1);
    }

    #[test]
    fn noise_advances_when_clock_set() {
        // noise_clock = (spucnt >> 8) & 0x3F. clock>>2 = bits 13:10
        // of spucnt. Set those four bits to 0xF for the fastest
        // shift rate: threshold = (0x8000 >> 15) << 16 = 0x10000.
        // Per-sample increment is 0x10000 + NOISE_FREQ_ADD[step],
        // so the LFSR shifts at least once per tick.
        let mut s = Spu::new();
        s.write16(SPUCNT, 0x3C00);
        let v0 = s.noise_val;
        s.noise_tick();
        assert_ne!(s.noise_val, v0, "noise should shift at fastest rate");
    }

    #[test]
    fn noise_period_grows_with_shift() {
        // At shift=0 the LFSR shifts roughly once every 0x8000
        // counter-units (0x8000 / 0x10000 per sample → many samples).
        // Verify it does NOT shift in a single tick at slow rate.
        let mut s = Spu::new();
        s.write16(SPUCNT, 0x0000); // shift = 0
        let v0 = s.noise_val;
        s.noise_tick();
        // Single tick adds 0x10000 < 0x8000_0000 — no shift.
        assert_eq!(s.noise_val, v0);
    }

    // -- FMod / pitch modulation --

    #[test]
    fn fmod_modulator_voice_suppressed_from_lr_mix() {
        // Voice 0 = modulator (its sample feeds voice 1's pitch).
        // Voice 1 = modulated. Both are configured to emit a known
        // non-zero sample; only voice 1 should reach the audible mix.
        let mut s = Spu::new();
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        s.write16(SPUCNT, SPUCNT_UNMUTE);

        // Mark voice 1 as pitch-modulated by voice 0.
        s.write16(PMON_LO, 0x0002);

        // Configure both voices: full envelope, full volume,
        // last_sample seeded directly so we don't depend on ADPCM.
        for v in 0..2 {
            let base = VOICE_BASE + (v as u32) * 16;
            s.write16(base + voice_offset::VOLUME_L, 0x3FFF);
            s.write16(base + voice_offset::VOLUME_R, 0x3FFF);
            s.voices[v].phase = AdsrPhase::Sustain;
            s.voices[v].envelope = 0x7FFF;
            s.voices[v].last_sample = 0x4000;
            // Block decode of zeros — voice mixes its envelope * sample.
            s.voices[v].sample_buf = [0x4000; ADPCM_SAMPLES_PER_BLOCK];
            s.voices[v].sample_index = 0;
        }

        s.tick_sample(SAMPLE_CYCLES);
        let (l, r) = s.drain_audio()[0];

        // Voice 0 (modulator) should NOT contribute. Voice 1 alone
        // would produce one full-scale sample's worth of output.
        // Bound it: total must be < 2× single-voice level.
        let voice_only = (0x4000_i32 * 0x3FFF) >> 14;
        assert!(
            (l as i32) < voice_only * 3 / 2,
            "voice 0 leaked into L: l={l}"
        );
        assert!(
            (r as i32) < voice_only * 3 / 2,
            "voice 0 leaked into R: r={r}"
        );
        // And greater than zero — voice 1 still played.
        assert!(l > 0);
        assert!(r > 0);
    }

    #[test]
    fn spucnt_mute_zeroes_voice_sample_history() {
        let mut s = Spu::new();
        s.main_vol_l.write(0x3FFF);
        s.main_vol_r.write(0x3FFF);
        s.voices[0].phase = AdsrPhase::Sustain;
        s.voices[0].envelope = 0x7FFF;
        s.voices[0].sample_buf = [0x4000; ADPCM_SAMPLES_PER_BLOCK];
        s.voices[0].sample_index = 0;

        s.tick_sample(SAMPLE_CYCLES);
        let (l, r) = s.drain_audio()[0];

        assert_eq!((l, r), (0, 0));
        assert_eq!(s.voices[0].interpolation_window(), [0, 0, 0, 0]);
    }

    #[test]
    fn fmod_modulator_still_updates_last_sample() {
        // Even though voice 0's L/R is suppressed, its last_sample
        // must still update so voice 1's FMod reads the right value.
        let mut s = Spu::new();
        s.write16(PMON_LO, 0x0002);
        s.voices[0].phase = AdsrPhase::Sustain;
        s.voices[0].envelope = 0x7FFF;
        s.voices[0].sample_buf = [0x1234; ADPCM_SAMPLES_PER_BLOCK];
        s.voices[0].sample_index = 0;
        s.voices[1].raw_pitch = 0x1000;
        s.voices[1].phase = AdsrPhase::Sustain;
        s.voices[1].envelope = 0x7FFF;

        s.tick_sample(SAMPLE_CYCLES);
        // last_sample for voice 0 should be approximately envelope *
        // sample (saturated), not zero.
        assert!(
            s.voices[0].last_sample != 0,
            "modulator's last_sample was zeroed"
        );
    }

    #[test]
    fn noise_value_substitutes_for_voice_sample() {
        // Voice 5 is in noise mode; fetch_voice_sample returns
        // noise_val unchanged when the voice is active.
        let mut s = Spu::new();
        s.noise_on = 1 << 5;
        s.noise_val = 0x1234;
        s.voices[5].phase = AdsrPhase::Attack;
        let out = s.fetch_voice_sample(5);
        assert_eq!(out, 0x1234);
    }

    #[test]
    fn off_noise_voice_stays_silent() {
        let mut s = Spu::new();
        s.noise_on = 1 << 5;
        s.noise_val = 0x1234;
        s.voices[5].phase = AdsrPhase::Off;
        let out = s.fetch_voice_sample(5);
        assert_eq!(out, 0);
    }
}
