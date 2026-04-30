//! `showcase-particles` -- fixed-pool particle effects through the
//! engine ordering-table helpers.
//!
//! The SDK's `psx-fx` crate already had a reusable `ParticlePool`,
//! but no standalone visual demo. This example keeps the simulation
//! in `psx-fx` and routes the packets through `OtFrame` +
//! `PrimitiveArena`, so the effect exercises the same render path as
//! the GTE-heavy showcases.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_engine::{button, App, Config, Ctx, OtFrame, PrimitiveArena, Scene};
use psx_font::{fonts::BASIC_8X16, FontAtlas};
use psx_fx::{LcgRng, ParticlePool};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat};
use psx_math::{cos_q12, sin_q12};
use psx_vram::{Clut, TexDepth, Tpage};

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;
const OT_DEPTH: usize = 8;
const BG_SLOT: usize = OT_DEPTH - 1;
const AUTO_SLOT: usize = 4;
const MARKER_SLOT: usize = 1;

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut RECTS: [RectFlat; 192] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 192];
static mut BG_QUAD: QuadGouraud = QuadGouraud {
    tag: 0,
    color0_cmd: 0,
    v0: 0,
    color1: 0,
    v1: 0,
    color2: 0,
    v2: 0,
    color3: 0,
    v3: 0,
};

struct ShowcaseParticles {
    rng: LcgRng,
    particles: ParticlePool<160>,
    emitter: (i16, i16),
    font: Option<FontAtlas>,
}

impl ShowcaseParticles {
    const fn new() -> Self {
        Self {
            rng: LcgRng::new(0xB105_F00D),
            particles: ParticlePool::new(),
            emitter: (SCREEN_W / 2, SCREEN_H / 2),
            font: None,
        }
    }
}

impl Scene for ShowcaseParticles {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.is_held(button::LEFT) {
            self.emitter.0 = self.emitter.0.saturating_sub(2).max(16);
        }
        if ctx.is_held(button::RIGHT) {
            self.emitter.0 = self.emitter.0.saturating_add(2).min(SCREEN_W - 16);
        }
        if ctx.is_held(button::UP) {
            self.emitter.1 = self.emitter.1.saturating_sub(2).max(32);
        }
        if ctx.is_held(button::DOWN) {
            self.emitter.1 = self.emitter.1.saturating_add(2).min(SCREEN_H - 28);
        }

        if ctx.frame % 4 == 0 {
            let angle = ((ctx.frame as u16).wrapping_mul(7)) & 0x0FFF;
            let x = SCREEN_W / 2 + ((sin_q12(angle) * 84) >> 12) as i16;
            let y = 112 + ((cos_q12(angle.wrapping_mul(2)) * 32) >> 12) as i16;
            let color = auto_color(ctx.frame);
            self.particles
                .spawn_burst(&mut self.rng, (x, y), color, 5, 22, 42);
        }

        if ctx.just_pressed(button::CROSS) || ctx.just_pressed(button::CIRCLE) {
            self.particles
                .spawn_burst(&mut self.rng, self.emitter, (255, 210, 96), 36, 58, 60);
        }

        self.particles.update(1);
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut rects = unsafe { PrimitiveArena::new(&mut RECTS) };
        let mut backgrounds = unsafe { PrimitiveArena::new(core::slice::from_mut(&mut BG_QUAD)) };

        let Some(bg) = backgrounds.push(QuadGouraud::new(
            [(0, 0), (SCREEN_W, 0), (0, SCREEN_H), (SCREEN_W, SCREEN_H)],
            [(4, 8, 24), (4, 8, 24), (0, 0, 4), (0, 0, 4)],
        )) else {
            return;
        };
        ot.add_packet(BG_SLOT, bg);

        let _ = render_particles(&self.particles, &mut rects, &mut ot, AUTO_SLOT);
        if let Some(marker) = rects.push(RectFlat::new(
            self.emitter.0 - 3,
            self.emitter.1 - 3,
            6,
            6,
            255,
            240,
            160,
        )) {
            ot.add_packet(MARKER_SLOT, marker);
        }

        ot.submit();

        if let Some(font) = self.font.as_ref() {
            font.draw_text(4, 4, "SHOWCASE-PARTICLES", (220, 230, 250));
            font.draw_text(4, 222, "D-PAD MOVE  X/O BURST", (150, 170, 210));
            font.draw_text(224, 4, "LIVE", (150, 170, 210));
            let live = u16_hex(self.particles.live_count() as u16);
            font.draw_text(264, 4, live.as_str(), (230, 220, 160));
        }
    }
}

fn render_particles<const N: usize, const OT_N: usize>(
    particles: &ParticlePool<N>,
    rects: &mut PrimitiveArena<'_, RectFlat>,
    ot: &mut OtFrame<'_, OT_N>,
    slot: usize,
) -> usize {
    let mut written = 0;
    for p in particles.particles() {
        if !p.alive() {
            continue;
        }

        let denom = p.spawn_ttl.max(1) as u16;
        let scale = p.ttl as u16;
        let r = ((p.r as u16 * scale) / denom) as u8;
        let g = ((p.g as u16 * scale) / denom) as u8;
        let b = ((p.b as u16 * scale) / denom) as u8;
        let size = if (p.ttl as u16) * 2 > denom { 3 } else { 2 };
        let Some(rect) = rects.push(RectFlat::new(p.x, p.y, size, size, r, g, b)) else {
            break;
        };
        ot.add_packet(slot, rect);
        written += 1;
    }
    written
}

fn auto_color(frame: u32) -> (u8, u8, u8) {
    match (frame / 32) & 3 {
        0 => (96, 180, 255),
        1 => (255, 120, 180),
        2 => (120, 255, 160),
        _ => (255, 220, 96),
    }
}

#[no_mangle]
fn main() -> ! {
    let mut scene = ShowcaseParticles::new();
    let config = Config {
        screen_w: SCREEN_W as u16,
        screen_h: SCREEN_H as u16,
        clear_color: (0, 0, 0),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

struct HexU16 {
    bytes: [u8; 6],
}

impl HexU16 {
    fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.bytes) }
    }
}

fn u16_hex(v: u16) -> HexU16 {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = [0u8; 6];
    out[0] = b'0';
    out[1] = b'x';
    out[2] = HEX[((v >> 12) & 0xF) as usize];
    out[3] = HEX[((v >> 8) & 0xF) as usize];
    out[4] = HEX[((v >> 4) & 0xF) as usize];
    out[5] = HEX[(v & 0xF) as usize];
    HexU16 { bytes: out }
}
