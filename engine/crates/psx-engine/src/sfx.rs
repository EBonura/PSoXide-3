//! SFX conveniences on top of `psx-spu`.
//!
//! All three mini-games re-implemented the same two helpers —
//! `configure_sfx_voice` (volume + pitch + address + ADSR) and
//! `play_sfx` (key_on). The logic is identical across games, so it
//! moves into the engine. Games now do
//!
//! ```ignore
//! use psx_engine::sfx;
//!
//! sfx::configure_voice(Voice::V0, addr, Pitch::raw(0x1400));
//! sfx::play(Voice::V0);
//! ```
//!
//! and drop ~10 lines of boilerplate.
//!
//! The engine deliberately doesn't wrap `Voice` / `Pitch` / `SpuAddr`
//! into its own types — those already live in `psx-spu` and carry
//! the hardware meaning precisely. The engine's value-add is only
//! the *combination* of calls, not renaming things.

use psx_spu::{Adsr, Pitch, SpuAddr, Voice, Volume};

/// Configure a voice for SFX playback: half-volume on both stereo
/// channels, the given pitch, the given sample address, and a
/// percussive ADSR (short attack, fast release). The voice stays
/// silent until [`play`] keys it on.
///
/// This is the one-stop voice setup that every mini-game does at
/// boot; the ADSR is the `percussive()` preset because all three
/// games use short attack-decay tone samples rather than held
/// music notes.
pub fn configure_voice(v: Voice, addr: SpuAddr, pitch: Pitch) {
    v.set_volume(Volume::HALF, Volume::HALF);
    v.set_pitch(pitch);
    v.set_start_addr(addr);
    v.set_adsr(Adsr::percussive());
}

/// Fire a pre-configured SFX voice — re-attacks the ADSR envelope
/// so repeated calls replay the sample's attack transient rather
/// than letting the decay tail dominate.
#[inline]
pub fn play(v: Voice) {
    Voice::key_on(v.mask());
}
