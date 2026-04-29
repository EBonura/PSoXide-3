//! `hello-audio` — the audio equivalent of hello-input.
//!
//! Four face buttons trigger four SPU voices, each playing a
//! different built-in [`psx_spu::tones`] waveform at a different
//! pitch so the ear can tell them apart:
//!
//! | Button   | Waveform | Pitch     | Character            |
//! |----------|----------|-----------|----------------------|
//! | CROSS    | SINE     | 0x0C00    | Low, smooth          |
//! | CIRCLE   | SQUARE   | 0x1000    | Mid, buzzy           |
//! | TRIANGLE | TRIANGLE | 0x1400    | High, soft           |
//! | SQUARE   | SAWTOOTH | 0x1800    | Highest, bright      |
//!
//! Visual feedback:
//! - Background flashes the colour of whichever voice just triggered.
//! - Text rows show each button's state (playing / silent) + its
//!   configured pitch and waveform name.
//!
//! Proves the whole pad → SPU → DAC path: the controller IRQ
//! delivers button events, pad state drives key-on / key-off
//! writes to SPU voice registers, and the SPU's internal mixer
//! streams audio out through the bus' sample collector.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{self as gpu, framebuf::FrameBuffer, Resolution, VideoMode};
use psx_pad::{button, poll_port1, ButtonState};
use psx_spu::{self as spu, tones, Adsr, Pitch, SpuAddr, Voice, Volume};
use psx_vram::{Clut, TexDepth, Tpage};

/// Font atlas tpage — past the 320-wide display buffers.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

/// SPU RAM addresses for each tone. Each tone is 16 bytes (one
/// ADPCM block). We park them starting at 0x1010 — the BIOS
/// convention is to leave 0x0000..0x1000 for system use, with the
/// required "zero" block at 0x1000. We skip past that.
const SPU_ADDR_SINE: SpuAddr = SpuAddr::new(0x1010);
const SPU_ADDR_SQUARE: SpuAddr = SpuAddr::new(0x1020);
const SPU_ADDR_TRIANGLE: SpuAddr = SpuAddr::new(0x1030);
const SPU_ADDR_SAWTOOTH: SpuAddr = SpuAddr::new(0x1040);

/// Per-button setup: which voice, which SPU start addr, which
/// pitch, which display label and tint.
struct Channel {
    voice: Voice,
    start: SpuAddr,
    pitch: Pitch,
    label: &'static str,
    tint: (u8, u8, u8),
}

const CROSS_CHAN: Channel = Channel {
    voice: Voice::V0,
    start: SPU_ADDR_SINE,
    pitch: Pitch::raw(0x0C00),
    label: "CROSS    : SINE  lo",
    tint: (80, 160, 220),
};
const CIRCLE_CHAN: Channel = Channel {
    voice: Voice::V1,
    start: SPU_ADDR_SQUARE,
    pitch: Pitch::raw(0x1000),
    label: "CIRCLE   : SQR   mid",
    tint: (220, 80, 80),
};
const TRIANGLE_CHAN: Channel = Channel {
    voice: Voice::V2,
    start: SPU_ADDR_TRIANGLE,
    pitch: Pitch::raw(0x1400),
    label: "TRIANGLE : TRI   hi",
    tint: (80, 220, 80),
};
const SQUARE_CHAN: Channel = Channel {
    voice: Voice::V3,
    start: SPU_ADDR_SAWTOOTH,
    pitch: Pitch::raw(0x1800),
    label: "SQUARE   : SAW   top",
    tint: (220, 80, 220),
};

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(320, 240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // Audio side init: SPU on, unmuted, main volume full.
    spu::init();

    // Upload all four tone samples into SPU RAM.
    spu::upload_adpcm(SPU_ADDR_SINE, tones::SINE);
    spu::upload_adpcm(SPU_ADDR_SQUARE, tones::SQUARE);
    spu::upload_adpcm(SPU_ADDR_TRIANGLE, tones::TRIANGLE);
    spu::upload_adpcm(SPU_ADDR_SAWTOOTH, tones::SAWTOOTH);

    // Configure each voice once: volume, start addr, envelope.
    // Pitch is also set here; it doesn't change while held (per-
    // channel pitch is constant in this demo, though a fancier one
    // could modulate).
    for ch in [&CROSS_CHAN, &CIRCLE_CHAN, &TRIANGLE_CHAN, &SQUARE_CHAN] {
        ch.voice.set_volume(Volume::HALF, Volume::HALF);
        ch.voice.set_pitch(ch.pitch);
        ch.voice.set_start_addr(ch.start);
        ch.voice.set_adsr(Adsr::default_tone());
    }

    let font = FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT);

    // Edge-detect pad state so we only key_on / key_off on transitions
    // — otherwise we'd retrigger the attack phase every frame, which
    // sounds like a constant click.
    let mut prev_pad = ButtonState::NONE;

    loop {
        let pad = poll_port1().buttons;

        // Compute which channels are newly pressed / newly released.
        let mut on_mask: u32 = 0;
        let mut off_mask: u32 = 0;
        for (ch, btn) in [
            (&CROSS_CHAN, button::CROSS),
            (&CIRCLE_CHAN, button::CIRCLE),
            (&TRIANGLE_CHAN, button::TRIANGLE),
            (&SQUARE_CHAN, button::SQUARE),
        ] {
            let now = pad.is_held(btn);
            let was = prev_pad.is_held(btn);
            if now && !was {
                on_mask |= ch.voice.mask();
            } else if !now && was {
                off_mask |= ch.voice.mask();
            }
        }
        if on_mask != 0 {
            Voice::key_on(on_mask);
        }
        if off_mask != 0 {
            Voice::key_off(off_mask);
        }
        prev_pad = pad;

        // Visual feedback — tint the background with the sum of
        // currently-active channel colours for a low-effort "you
        // can see what you hear" cue.
        let (r, g, b) = mix_background(pad);
        fb.clear(r, g, b);

        // Header.
        font.draw_text(4, 4, "hello-audio", (200, 200, 200));
        font.draw_text(4, 14, "Hold face buttons to play tones.", (140, 140, 140));

        // Per-channel row: label + PLAYING / silent indicator.
        let mut y: i16 = 32;
        for (ch, btn) in [
            (&CROSS_CHAN, button::CROSS),
            (&CIRCLE_CHAN, button::CIRCLE),
            (&TRIANGLE_CHAN, button::TRIANGLE),
            (&SQUARE_CHAN, button::SQUARE),
        ] {
            let held = pad.is_held(btn);
            let (line_tint, state_text) = if held {
                (ch.tint, "PLAYING")
            } else {
                ((100, 100, 100), "silent ")
            };
            font.draw_text(4, y, ch.label, line_tint);
            font.draw_text(4 + 8 * 24, y, state_text, line_tint);
            y = y.wrapping_add(12);
        }

        // Footer hint.
        font.draw_text(4, 220, "press any button to hear a tone", (120, 120, 120));

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
    }
}

/// Mix a background tint from the set of currently-held buttons.
/// Additive blend so "all four" is white-ish and "one" is a clear
/// single hue. Keeps output dim so text stays legible on top.
fn mix_background(pad: ButtonState) -> (u8, u8, u8) {
    let mut r: u16 = 8;
    let mut g: u16 = 8;
    let mut b: u16 = 20;
    for (ch, btn) in [
        (&CROSS_CHAN, button::CROSS),
        (&CIRCLE_CHAN, button::CIRCLE),
        (&TRIANGLE_CHAN, button::TRIANGLE),
        (&SQUARE_CHAN, button::SQUARE),
    ] {
        if pad.is_held(btn) {
            r = r.saturating_add((ch.tint.0 >> 2) as u16);
            g = g.saturating_add((ch.tint.1 >> 2) as u16);
            b = b.saturating_add((ch.tint.2 >> 2) as u16);
        }
    }
    (r.min(255) as u8, g.min(255) as u8, b.min(255) as u8)
}
