//! High-level PS1 SPU (Sound Processing Unit) API.
//!
//! Turns SPU-programming-by-magic-numbers into typed primitives:
//!
//! - [`Voice`] — typed voice index in `0..24`.
//! - [`Pitch`] — Q5.12 sample-rate multiplier; `Pitch::UNITY` =
//!   44100 Hz (one sample per SPU tick).
//! - [`Volume`] — per-voice / main-output 15-bit signed linear gain.
//! - [`Adsr`] — typed envelope descriptor; builds the two-word SPU
//!   envelope register pair.
//! - [`SpuAddr`] — 8-byte-aligned SPU RAM pointer (the only shape
//!   voice-start / loop / transfer registers accept).
//!
//! Typical boot sequence:
//!
//! ```ignore
//! spu::init();                              // turn the SPU on, sensible defaults
//! spu::set_main_volume(Volume::MAX, Volume::MAX);
//! spu::upload_adpcm(SpuAddr::new(0x1010), &TONE_SAMPLE);
//! let v = Voice::V0;
//! v.set_volume(Volume::MAX, Volume::MAX);
//! v.set_pitch(Pitch::UNITY);
//! v.set_start_addr(SpuAddr::new(0x1010));
//! v.set_adsr(Adsr::default_tone());
//! Voice::key_on(v.mask());                  // start the tone
//! ```
//!
//! ## What this crate owns vs doesn't
//!
//! - **Owns**: typed register access, key-on/off masks, the ADPCM
//!   byte-stream upload path, ADSR encoding helpers, sample-start
//!   address alignment.
//! - **Doesn't own (yet)**: ADPCM encoder (ship pre-baked samples
//!   instead — see `vendor/tone_*.adpcm`), XA-ADPCM / CD-audio
//!   streaming, reverb preset tables, DMA-based sample upload.
//!   Those land as the ladder pulls them in.
//!
//! ## Q-format conventions
//!
//! - **Pitch**: Q5.12 in a u16. `0x1000` = 1.0 = the sample plays
//!   at its recorded rate (44100 Hz). Halving to `0x0800` drops an
//!   octave; `0x2000` raises one. Pitch field is 14 bits on
//!   hardware; values above `0x3FFF` clamp.
//! - **Volumes**: i16. Sign carries, the magnitude is linear. Max
//!   output is `0x3FFF` (= 1.0); `Volume::MAX` already uses this.
//!   Negative means "phase-inverted," used for stereo tricks we
//!   don't wire up here yet.
//! - **SPU RAM addresses**: stored in registers as `addr / 8`,
//!   [`SpuAddr`] does the divide so the caller passes byte offsets.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use psx_io::spu::{SPU_BASE, SPUCNT, SPUSTAT};

pub mod tones;

// ======================================================================
// MMIO helpers — hand-rolled volatile access at fixed SPU offsets
// ======================================================================

/// Raw MMIO write of a 16-bit SPU register.
///
/// Every SPU register is 16-bit (the controller doesn't care about
/// higher-width accesses — the lower half is what matters). We go
/// through a volatile pointer so the compiler can't reorder writes
/// across register boundaries.
#[inline]
fn write_reg16(addr: u32, value: u16) {
    // SAFETY: SPU registers live in the hardware-MMIO window at
    // 0x1F80_1C00..0x1F80_1FFC. The SDK only exposes functions
    // that compute their `addr` from typed handles whose
    // constructors validate ranges, so no caller can point this at
    // non-SPU memory.
    unsafe {
        core::ptr::write_volatile(addr as *mut u16, value);
    }
}

/// Raw MMIO read of a 16-bit SPU register.
#[inline]
fn read_reg16(addr: u32) -> u16 {
    // SAFETY: same MMIO contract as `write_reg16`.
    unsafe { core::ptr::read_volatile(addr as *const u16) }
}

// Voice register block is 16 bytes per voice starting at 0x1F80_1C00.
const VOICE_STRIDE: u32 = 0x10;
const VOICE_VOL_LEFT: u32 = 0x0;
const VOICE_VOL_RIGHT: u32 = 0x2;
const VOICE_PITCH: u32 = 0x4;
const VOICE_START_ADDR: u32 = 0x6;
const VOICE_ADSR_LO: u32 = 0x8;
const VOICE_ADSR_HI: u32 = 0xA;
// 0xC = current ADSR envelope (read-only); 0xE = repeat addr.

// Global registers (PSX-SPX § "SPU Registers").
const MAIN_VOL_LEFT: u32 = 0x1F80_1D80;
const MAIN_VOL_RIGHT: u32 = 0x1F80_1D82;
const KEY_ON_LO: u32 = 0x1F80_1D88;
const KEY_ON_HI: u32 = 0x1F80_1D8A;
const KEY_OFF_LO: u32 = 0x1F80_1D8C;
const KEY_OFF_HI: u32 = 0x1F80_1D8E;
const PITCH_MOD_LO: u32 = 0x1F80_1D90;
const PITCH_MOD_HI: u32 = 0x1F80_1D92;
const NOISE_LO: u32 = 0x1F80_1D94;
const NOISE_HI: u32 = 0x1F80_1D96;
const REVERB_ENABLE_LO: u32 = 0x1F80_1D98;
const REVERB_ENABLE_HI: u32 = 0x1F80_1D9A;
const TRANSFER_ADDR: u32 = 0x1F80_1DA6;
const TRANSFER_DATA: u32 = 0x1F80_1DA8;
const TRANSFER_CTRL: u32 = 0x1F80_1DAC;

// ======================================================================
// Initialisation
// ======================================================================

/// Reset the SPU to a sane playable state.
///
/// Sets:
/// - SPUCNT: enable, unmute, default reverb off, CD audio off
/// - Main volume: max
/// - Every voice: silenced (volume 0, ADSR release fire, key-off)
/// - Reverb / pitch-mod / noise: all disabled
/// - Transfer mode: 16-bit PIO (the upload path we expose)
///
/// Call once at boot before any voice operations.
pub fn init() {
    // Silence everything immediately — key_off on all 24 voices
    // before we touch any other state, so nothing glitches audibly
    // on cold boot.
    write_reg16(KEY_OFF_LO, 0xFFFF);
    write_reg16(KEY_OFF_HI, 0x00FF);

    // Zero every voice register block.
    for v in 0..24 {
        let base = SPU_BASE + (v as u32) * VOICE_STRIDE;
        write_reg16(base + VOICE_VOL_LEFT, 0);
        write_reg16(base + VOICE_VOL_RIGHT, 0);
        write_reg16(base + VOICE_PITCH, 0);
        write_reg16(base + VOICE_START_ADDR, 0);
        write_reg16(base + VOICE_ADSR_LO, 0);
        write_reg16(base + VOICE_ADSR_HI, 0);
    }

    // Disable reverb, noise, pitch modulation on all voices.
    write_reg16(REVERB_ENABLE_LO, 0);
    write_reg16(REVERB_ENABLE_HI, 0);
    write_reg16(PITCH_MOD_LO, 0);
    write_reg16(PITCH_MOD_HI, 0);
    write_reg16(NOISE_LO, 0);
    write_reg16(NOISE_HI, 0);

    // Main volume to max — per-voice volume still controls the mix.
    write_reg16(MAIN_VOL_LEFT, Volume::MAX.0 as u16);
    write_reg16(MAIN_VOL_RIGHT, Volume::MAX.0 as u16);

    // SPUCNT: bit 15 = SPU enable, bit 14 = mute OFF (i.e. audible),
    // everything else zero. Writing in that order matches PSX-SPX's
    // recommendation — enable-then-unmute avoids a click.
    write_reg16(SPUCNT, 0x8000); // enabled, muted
    wait_spu_status(0x0000); // wait for SPUSTAT to stabilise
    write_reg16(SPUCNT, 0xC000); // enabled + unmuted
    wait_spu_status(0x0000);

    // Transfer mode: "Stop" (bit 0..=2 = 0). Games toggle this to
    // Manual/DMA as needed when uploading.
    write_reg16(TRANSFER_CTRL, 0x0004); // normal mode
}

/// Poll SPUSTAT's "SPU mode" field (low 6 bits) until it matches
/// `want & 0x3F`. Hardware needs ~1 audio frame to reflect SPUCNT
/// changes; without a poll, fast code can miss the state.
fn wait_spu_status(want: u16) {
    let mask = 0x3F;
    while (read_reg16(SPUSTAT) & mask) != (want & mask) {
        core::hint::spin_loop();
    }
}

// ======================================================================
// Typed primitives
// ======================================================================

/// A 16-bit signed SPU volume, `-0x8000..=0x7FFF` on the wire;
/// `-0x4000..=0x3FFF` is the linear-usable range. Positive
/// magnitudes are the normal "louder" direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Volume(pub i16);

impl Volume {
    /// Silence.
    pub const SILENCE: Self = Self(0);
    /// Maximum linear-positive volume (0x3FFF = +1.0).
    pub const MAX: Self = Self(0x3FFF);
    /// Half-scale convenience (0x2000 ≈ 0.5).
    pub const HALF: Self = Self(0x2000);

    /// Build from a normalized float-ish 0.0..=1.0 value without
    /// actually using floats (since we're `no_std`, no FPU). `num`
    /// and `den` are integer — `Volume::linear(3, 4)` is 0.75.
    pub const fn linear(num: u16, den: u16) -> Self {
        let v = ((Self::MAX.0 as u32) * (num as u32)) / (den as u32);
        Self(v as i16)
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self::SILENCE
    }
}

/// Q5.12 pitch / sample-rate multiplier. `0x1000` = play at the
/// sample's recorded rate. Halving = one octave down, doubling =
/// one octave up.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Pitch(u16);

impl Pitch {
    /// `0x1000` — play at the sample's native 44100 Hz rate.
    pub const UNITY: Self = Self(0x1000);
    /// `0x0800` — one octave below recorded rate.
    pub const OCTAVE_DOWN: Self = Self(0x0800);
    /// `0x2000` — one octave above recorded rate.
    pub const OCTAVE_UP: Self = Self(0x2000);
    /// Hardware cap (14-bit field). Higher values clamp.
    pub const MAX: Self = Self(0x3FFF);

    /// Raw constructor. Values above `0x3FFF` clamp on hardware.
    pub const fn raw(v: u16) -> Self {
        Self(v)
    }

    /// Pitch that makes a native-rate sine loop play at `hz_num /
    /// hz_den` where the loop's natural frequency is `base_hz`.
    /// Integer-only so `no_std` users can compute at compile time.
    pub const fn for_frequency(target_hz: u32, base_hz: u32) -> Self {
        let raw = ((0x1000u32) * target_hz) / base_hz;
        if raw > 0x3FFF {
            Self(0x3FFF)
        } else {
            Self(raw as u16)
        }
    }

    /// Underlying 14-bit value.
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

/// An 8-byte-aligned SPU RAM address.
///
/// SPU voice-start / loop / transfer registers all hold `addr / 8`
/// in their 16-bit field. Construction asserts alignment so the
/// division is lossless and caller bugs don't silently round down.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpuAddr(u32);

impl SpuAddr {
    /// Build from a byte offset into SPU RAM. Panics if `addr`
    /// isn't a multiple of 8.
    pub const fn new(addr: u32) -> Self {
        assert!(addr % 8 == 0, "SpuAddr: addr must be multiple of 8");
        assert!(
            addr < 512 * 1024,
            "SpuAddr: past end of 512 KiB SPU RAM",
        );
        Self(addr)
    }

    /// The byte offset this address represents.
    pub const fn byte_offset(self) -> u32 {
        self.0
    }

    /// The 16-bit field value the SPU registers actually want
    /// (`addr / 8`).
    pub const fn reg_field(self) -> u16 {
        (self.0 / 8) as u16
    }
}

/// ADSR envelope descriptor. Builds the pair of 16-bit envelope
/// registers the SPU needs.
///
/// Hardware bit-fields (PSX-SPX § "Voice 0..23 ADSR Register"):
///
/// **ADSR Lower (u16):**
/// ```text
///   bits 0..=3  : sustain level (0..15, sets target after decay)
///   bits 4..=7  : decay shift (0..15)
///   bits 8..=14 : attack shift (0..127)
///   bit  15     : attack mode (0 = linear, 1 = exponential)
/// ```
///
/// **ADSR Upper (u16):**
/// ```text
///   bits 0..=4  : release shift (0..31)
///   bit  5      : release mode (0 = linear, 1 = exponential)
///   bits 6..=12 : sustain shift (0..127)
///   bit  13     : reserved
///   bit  14     : sustain direction (0 = increase, 1 = decrease)
///   bit  15     : sustain mode (0 = linear, 1 = exponential)
/// ```
///
/// You rarely want to set all of these by hand; use
/// [`Adsr::default_tone`] for a generic instrument-like preset or
/// [`Adsr::percussive`] for a snappy one-shot.
#[derive(Copy, Clone, Debug)]
pub struct Adsr {
    /// Pre-packed 16-bit ADSR lower word.
    pub lower: u16,
    /// Pre-packed 16-bit ADSR upper word.
    pub upper: u16,
}

impl Adsr {
    /// Reasonable generic tone envelope — fast attack, short decay
    /// to half-sustain, mid-long release. Audible and "square-ish."
    pub const fn default_tone() -> Self {
        // Attack shift = 0x7F (~0.5s to full), decay shift = 0xA,
        // sustain level = 0xF (max), release shift = 0x10 (~0.1s).
        // Sustain mode = increase-linear at rate 0x7F (hold at
        // target). Values pulled from a known-good PSX homebrew
        // envelope; fine-tune per instrument in your game.
        Self {
            // Linear attack, shift=0x7F, decay shift=0xA, sustain
            // level=0xF → lower word bits.
            lower: (0x7F << 8) | (0xA << 4) | 0xF,
            // Linear release, release shift=0x10 → upper word bits
            // (rest zero = sustain direction increase / mode linear).
            upper: 0x10 | (0x7F << 6),
        }
    }

    /// Very short percussive envelope — for blips, UI clicks, hit
    /// SFX. Snappy attack, almost immediate release.
    pub const fn percussive() -> Self {
        Self {
            // Attack shift = 0x20 (fast), decay shift = 0x8,
            // sustain = 0x0 (silence after decay).
            lower: (0x20 << 8) | (0x8 << 4),
            // Release shift = 0x10, hold on sustain (direction =
            // decrease, mode = exponential).
            upper: 0x10 | (1 << 14) | (1 << 15),
        }
    }

    /// Silent / "no envelope" — voice stays at key-on volume until
    /// key-off. Useful as a placeholder while iterating.
    pub const fn passthrough() -> Self {
        Self { lower: 0, upper: 0 }
    }
}

// ======================================================================
// Voice handle
// ======================================================================

/// A typed voice index in `0..24`. Construct via the `V0`..`V23`
/// constants or [`Voice::new`]. All per-voice operations are
/// methods on this type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Voice(u8);

impl Voice {
    /// Voice index 0.
    pub const V0: Self = Self(0);
    /// Voice index 1.
    pub const V1: Self = Self(1);
    /// Voice index 2.
    pub const V2: Self = Self(2);
    /// Voice index 3.
    pub const V3: Self = Self(3);
    /// Voice index 4.
    pub const V4: Self = Self(4);
    /// Voice index 5.
    pub const V5: Self = Self(5);
    /// Voice index 6.
    pub const V6: Self = Self(6);
    /// Voice index 7.
    pub const V7: Self = Self(7);
    // V8..V23 follow the same pattern; keep them collapsed until a
    // game actually needs 9+ voices in flight. `Voice::new(N)` is
    // always available as the escape hatch.

    /// Build a voice by index. Panics (const-asserts) if `n >= 24`.
    pub const fn new(n: u8) -> Self {
        assert!(n < 24, "Voice: index must be < 24");
        Self(n)
    }

    /// The 0-based voice index this handle points at.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Bitmask this voice occupies in the 24-bit key-on / key-off
    /// register pair. `Voice::V0.mask() == 0b1`.
    pub const fn mask(self) -> u32 {
        1u32 << (self.0 as u32)
    }

    /// Base MMIO address of this voice's 16-byte register block.
    #[inline]
    const fn reg_base(self) -> u32 {
        SPU_BASE + (self.0 as u32) * VOICE_STRIDE
    }

    /// Set per-voice stereo volume. The main-output mix is still
    /// modulated by [`set_main_volume`], but each voice can have
    /// its own level.
    pub fn set_volume(self, left: Volume, right: Volume) {
        write_reg16(self.reg_base() + VOICE_VOL_LEFT, left.0 as u16);
        write_reg16(self.reg_base() + VOICE_VOL_RIGHT, right.0 as u16);
    }

    /// Set the voice's sample-rate pitch (Q5.12). [`Pitch::UNITY`]
    /// = native 44100 Hz.
    pub fn set_pitch(self, pitch: Pitch) {
        write_reg16(self.reg_base() + VOICE_PITCH, pitch.as_u16());
    }

    /// Point the voice at the ADPCM sample starting at `addr`.
    /// Voice will begin playing from this address at the next key-on.
    pub fn set_start_addr(self, addr: SpuAddr) {
        write_reg16(self.reg_base() + VOICE_START_ADDR, addr.reg_field());
    }

    /// Install ADSR envelope parameters on this voice.
    pub fn set_adsr(self, adsr: Adsr) {
        write_reg16(self.reg_base() + VOICE_ADSR_LO, adsr.lower);
        write_reg16(self.reg_base() + VOICE_ADSR_HI, adsr.upper);
    }

    /// Trigger the voices whose bits are set in `mask`. The SPU
    /// begins playing each voice from its configured start address,
    /// applying its ADSR attack phase.
    ///
    /// Example: `Voice::key_on(Voice::V0.mask() | Voice::V3.mask())`.
    pub fn key_on(mask: u32) {
        write_reg16(KEY_ON_LO, mask as u16);
        write_reg16(KEY_ON_HI, (mask >> 16) as u16);
    }

    /// Stop the voices whose bits are set in `mask` — fires the
    /// release phase of the ADSR.
    pub fn key_off(mask: u32) {
        write_reg16(KEY_OFF_LO, mask as u16);
        write_reg16(KEY_OFF_HI, (mask >> 16) as u16);
    }
}

// ======================================================================
// Global controls
// ======================================================================

/// Set the main L/R output volume that every voice mixes through.
pub fn set_main_volume(left: Volume, right: Volume) {
    write_reg16(MAIN_VOL_LEFT, left.0 as u16);
    write_reg16(MAIN_VOL_RIGHT, right.0 as u16);
}

// ======================================================================
// ADPCM upload (PIO path)
// ======================================================================

/// Upload ADPCM sample bytes to SPU RAM via the manual-transfer
/// (PIO) path. Slow — one halfword per write — but simple and
/// doesn't need DMA setup. Games that upload many megabytes of
/// samples at boot use DMA; for per-frame SFX uploads of a few
/// KB, this is fine.
///
/// `bytes` must be a multiple of 2 (one halfword per two bytes).
/// `dest` must be 8-byte-aligned (the SPU voice-start register's
/// resolution).
///
/// Process (PSX-SPX § "SPU Data Transfer"):
/// 1. Write transfer-control = 0 (reset).
/// 2. Write target address register.
/// 3. Write transfer-control = 4 (manual).
/// 4. Push halfword data through 0x1F80_1DA8.
/// 5. Wait for the transfer to drain (SPUSTAT bit 7 = transfer busy).
pub fn upload_adpcm(dest: SpuAddr, bytes: &[u8]) {
    assert!(
        bytes.len() % 2 == 0,
        "upload_adpcm: byte slice must be a multiple of 2",
    );
    // Halt any in-flight transfer.
    write_reg16(TRANSFER_CTRL, 0x0000);
    // Target address.
    write_reg16(TRANSFER_ADDR, dest.reg_field());
    // Select manual write mode.
    write_reg16(TRANSFER_CTRL, 0x0004);

    let mut i = 0;
    while i + 1 < bytes.len() {
        let lo = bytes[i] as u16;
        let hi = bytes[i + 1] as u16;
        write_reg16(TRANSFER_DATA, lo | (hi << 8));
        i += 2;
    }

    // Flush: transfer-control back to "stop" so the SPU latches
    // the writes. A short wait gives hardware time to settle.
    write_reg16(TRANSFER_CTRL, 0x0000);
    for _ in 0..200 {
        core::hint::spin_loop();
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_linear_scales() {
        assert_eq!(Volume::linear(0, 1), Volume::SILENCE);
        assert_eq!(Volume::linear(1, 1), Volume::MAX);
        // 0.5 → 0x1FFF (rounded down from 0x3FFF / 2).
        let half = Volume::linear(1, 2);
        assert_eq!(half.0, 0x1FFF);
    }

    #[test]
    fn pitch_for_frequency() {
        // Play a 440 Hz native sample at 880 Hz → pitch 0x2000.
        let p = Pitch::for_frequency(880, 440);
        assert_eq!(p.as_u16(), 0x2000);
        // Play at native rate.
        let p = Pitch::for_frequency(440, 440);
        assert_eq!(p, Pitch::UNITY);
        // Very high multiplier clamps to MAX.
        let p = Pitch::for_frequency(100_000, 440);
        assert_eq!(p, Pitch::MAX);
    }

    #[test]
    fn spu_addr_reg_field_divides_by_8() {
        assert_eq!(SpuAddr::new(0x1000).reg_field(), 0x0200);
        assert_eq!(SpuAddr::new(0x1008).reg_field(), 0x0201);
    }

    #[test]
    #[should_panic = "multiple of 8"]
    fn spu_addr_rejects_misalignment() {
        let _ = SpuAddr::new(0x1004);
    }

    #[test]
    fn voice_mask_correct() {
        assert_eq!(Voice::V0.mask(), 0x01);
        assert_eq!(Voice::V3.mask(), 0x08);
        assert_eq!(Voice::new(23).mask(), 0x0080_0000);
    }

    #[test]
    #[should_panic = "index must be < 24"]
    fn voice_new_rejects_out_of_range() {
        let _ = Voice::new(24);
    }
}
