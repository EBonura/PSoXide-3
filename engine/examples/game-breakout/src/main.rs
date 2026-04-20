//! `breakout` — mini-game #2, ported to the `psx-engine`
//! Scene/App framework.
//!
//! Same classic brick-buster: paddle at the bottom, 40-brick
//! rainbow wall at the top, ball bouncing between them. Same
//! arcade polish (gradient background, particle bursts, ball
//! trail, screen shake on brick break, paddle flash on hit).
//!
//! The port follows the pong recipe: hand-rolled main loop +
//! `static mut GAME` → [`Scene`] impl on a struct + [`App::run`].
//! `Scene::init` does SPU setup + font upload + `reset_match`;
//! `Scene::update` integrates physics and state transitions;
//! `Scene::render` builds the OT + HUD. The per-game `frame`
//! counter that breakout tracked by hand is gone — the engine's
//! [`Ctx::frame`] replaces it.
//!
//! What stays in `static mut`: the DMA arena (OT + RECTS +
//! BG_QUAD). Those need fixed bus addresses across frames so the
//! DMA walker can pick them up. Everything else — phase, bricks,
//! particles, shake, trail — is a field on the scene, accessed
//! through `&mut self` instead of `unsafe { &mut GAME }`.
//!
//! Originally `sdk/examples/game-breakout/`.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_engine::{App, Config, Ctx, Scene, button, sfx};
use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat};
use psx_spu::{self as spu, Pitch, SpuAddr, Voice, tones};
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Layout
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

const COLS: usize = 8;
const ROWS: usize = 5;
const BRICK_COUNT: usize = COLS * ROWS;
const BRICK_W: u16 = 36;
const BRICK_H: u16 = 12;
const BRICK_GAP: i16 = 4;
const WALL_LEFT: i16 = 2;
const WALL_TOP: i16 = 24;

const PADDLE_W: u16 = 44;
const PADDLE_H: u16 = 6;
const PADDLE_Y: i16 = SCREEN_H - 20;
const PADDLE_SPEED: i16 = 4;

const BALL_SIZE: u16 = 6;
const BALL_SPEED_X_INIT: i16 = 2;
const BALL_SPEED_Y_INIT: i16 = -2;
const BALL_MAX_SPEED: i16 = 4;

const BORDER_W: u16 = 2;
const FLOOR_Y: i16 = SCREEN_H - 2;

const ROW_POINTS: [u16; ROWS] = [50, 40, 30, 20, 10];
const ROW_COLORS: [(u8, u8, u8); ROWS] = [
    (220, 60, 60),
    (220, 140, 60),
    (220, 220, 60),
    (60, 200, 80),
    (80, 140, 240),
];

// ----------------------------------------------------------------------
// VRAM + SPU layout
// ----------------------------------------------------------------------

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

const SPU_WALL: SpuAddr = SpuAddr::new(0x1010);
const SPU_PADDLE: SpuAddr = SpuAddr::new(0x1020);
const SPU_BRICK: SpuAddr = SpuAddr::new(0x1030);
const SPU_LOSE: SpuAddr = SpuAddr::new(0x1040);

const VOICE_WALL: Voice = Voice::V0;
const VOICE_PADDLE: Voice = Voice::V1;
const VOICE_BRICK: Voice = Voice::V2;
const VOICE_LOSE: Voice = Voice::V3;

// ----------------------------------------------------------------------
// Game tunables
// ----------------------------------------------------------------------

const SERVE_AUTO_LAUNCH_FRAMES: u16 = 30;
const MAX_PARTICLES: usize = 32;
const PARTICLES_PER_BURST: usize = 6;
const PARTICLE_TTL: u8 = 30;
const PARTICLE_SPREAD: i16 = 40;
const PARTICLE_GRAVITY: i16 = 2;
const TRAIL_LEN: usize = 4;
const SHAKE_FRAMES_ON_BREAK: u8 = 6;
const PADDLE_FLASH_FRAMES_ON_HIT: u8 = 6;

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq)]
enum Phase {
    Serve,
    Playing,
    Won,
    Lost,
}

struct Breakout {
    phase: Phase,
    paddle_x: i16,
    ball_x: i16,
    ball_y: i16,
    ball_vx: i16,
    ball_vy: i16,
    score: u16,
    lives: u8,
    bricks: [bool; BRICK_COUNT],
    bricks_left: u16,
    serve_wait: u16,
    shake: ShakeState,
    paddle_flash_frames: u8,
    particles: ParticlePool<MAX_PARTICLES>,
    trail_positions: [(i16, i16); TRAIL_LEN],
    trail_head: u8,
    rng: LcgRng,
    font: Option<FontAtlas>,
}

impl Breakout {
    const fn new() -> Self {
        Self {
            phase: Phase::Serve,
            paddle_x: 0,
            ball_x: 0,
            ball_y: 0,
            ball_vx: 0,
            ball_vy: 0,
            score: 0,
            lives: 3,
            bricks: [true; BRICK_COUNT],
            bricks_left: BRICK_COUNT as u16,
            serve_wait: 0,
            shake: ShakeState::new(),
            paddle_flash_frames: 0,
            particles: ParticlePool::new(),
            trail_positions: [(0, 0); TRAIL_LEN],
            trail_head: 0,
            rng: LcgRng::new(0xBEEF_0042),
            font: None,
        }
    }

    fn reset_match(&mut self) {
        self.phase = Phase::Serve;
        self.paddle_x = (SCREEN_W - PADDLE_W as i16) / 2;
        self.score = 0;
        self.lives = 3;
        self.bricks = [true; BRICK_COUNT];
        self.bricks_left = BRICK_COUNT as u16;
        self.serve_wait = 0;
        self.shake = ShakeState::new();
        self.paddle_flash_frames = 0;
        self.particles.clear();
        self.trail_positions = [(0, 0); TRAIL_LEN];
        self.trail_head = 0;
        self.rng = LcgRng::new(0xBEEF_0042);
        self.reset_ball_on_paddle();
    }

    fn reset_ball_on_paddle(&mut self) {
        self.ball_x = self.paddle_x + (PADDLE_W as i16 - BALL_SIZE as i16) / 2;
        self.ball_y = PADDLE_Y - BALL_SIZE as i16;
        self.ball_vx = BALL_SPEED_X_INIT;
        self.ball_vy = BALL_SPEED_Y_INIT;
    }

    fn push_trail(&mut self) {
        self.trail_positions[self.trail_head as usize] = (self.ball_x, self.ball_y);
        self.trail_head = (self.trail_head + 1) % TRAIL_LEN as u8;
    }

    fn paddle_input(&mut self, ctx: &Ctx) {
        if ctx.is_held(button::LEFT) {
            self.paddle_x -= PADDLE_SPEED;
        }
        if ctx.is_held(button::RIGHT) {
            self.paddle_x += PADDLE_SPEED;
        }
        let min = BORDER_W as i16;
        let max = SCREEN_W - BORDER_W as i16 - PADDLE_W as i16;
        if self.paddle_x < min {
            self.paddle_x = min;
        }
        if self.paddle_x > max {
            self.paddle_x = max;
        }
    }

    /// Walk the brick grid, find the first live overlap, reflect on
    /// the axis of smallest penetration, mark the brick dead, play
    /// SFX, trigger particles + shake.
    fn resolve_brick_collision(&mut self) {
        let bx0 = self.ball_x;
        let by0 = self.ball_y;
        let bx1 = bx0 + BALL_SIZE as i16;
        let by1 = by0 + BALL_SIZE as i16;

        let wall_bottom = WALL_TOP + ROWS as i16 * (BRICK_H as i16 + BRICK_GAP) - BRICK_GAP;
        if by1 < WALL_TOP || by0 > wall_bottom {
            return;
        }

        for row in 0..ROWS {
            for col in 0..COLS {
                let idx = row * COLS + col;
                if !self.bricks[idx] {
                    continue;
                }
                let bx = WALL_LEFT + (col as i16) * (BRICK_W as i16 + BRICK_GAP);
                let by = WALL_TOP + (row as i16) * (BRICK_H as i16 + BRICK_GAP);
                let bx_end = bx + BRICK_W as i16;
                let by_end = by + BRICK_H as i16;
                if bx1 <= bx || bx0 >= bx_end || by1 <= by || by0 >= by_end {
                    continue;
                }
                let pen_left = bx1 - bx;
                let pen_right = bx_end - bx0;
                let pen_top = by1 - by;
                let pen_bottom = by_end - by0;
                let pen_x = pen_left.min(pen_right);
                let pen_y = pen_top.min(pen_bottom);
                if pen_x < pen_y {
                    if pen_left < pen_right {
                        self.ball_x = bx - BALL_SIZE as i16;
                        self.ball_vx = -self.ball_vx.abs();
                    } else {
                        self.ball_x = bx_end;
                        self.ball_vx = self.ball_vx.abs();
                    }
                } else if pen_top < pen_bottom {
                    self.ball_y = by - BALL_SIZE as i16;
                    self.ball_vy = -self.ball_vy.abs();
                } else {
                    self.ball_y = by_end;
                    self.ball_vy = self.ball_vy.abs();
                }
                self.bricks[idx] = false;
                self.bricks_left -= 1;
                self.score = self.score.saturating_add(ROW_POINTS[row]);
                let brick_color = ROW_COLORS[row];
                let brick_cx = bx + BRICK_W as i16 / 2;
                let brick_cy = by + BRICK_H as i16 / 2;
                self.particles.spawn_burst(
                    &mut self.rng,
                    (brick_cx, brick_cy),
                    brick_color,
                    PARTICLES_PER_BURST,
                    PARTICLE_SPREAD,
                    PARTICLE_TTL,
                );
                self.shake.trigger(SHAKE_FRAMES_ON_BREAK);
                sfx::play(VOICE_BRICK);
                return;
            }
        }
    }
}

// ----------------------------------------------------------------------
// DMA-facing arenas
// ----------------------------------------------------------------------

static mut OT: OrderingTable<8> = OrderingTable::new();
/// 2 borders + 40 bricks + 32 particles + 4 trail + paddle + ball
/// = 80, padded to 96.
static mut RECTS: [RectFlat; 96] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 96];
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

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Breakout {
    fn init(&mut self, _ctx: &mut Ctx) {
        spu::init();
        spu::upload_adpcm(SPU_WALL, tones::SINE);
        spu::upload_adpcm(SPU_PADDLE, tones::SQUARE);
        spu::upload_adpcm(SPU_BRICK, tones::TRIANGLE);
        spu::upload_adpcm(SPU_LOSE, tones::SAWTOOTH);
        sfx::configure_voice(VOICE_WALL, SPU_WALL, Pitch::raw(0x1200));
        sfx::configure_voice(VOICE_PADDLE, SPU_PADDLE, Pitch::raw(0x0E00));
        sfx::configure_voice(VOICE_BRICK, SPU_BRICK, Pitch::raw(0x1600));
        sfx::configure_voice(VOICE_LOSE, SPU_LOSE, Pitch::raw(0x0600));

        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));

        self.reset_match();
    }

    fn update(&mut self, ctx: &mut Ctx) {
        // Effects tick every frame regardless of phase so
        // transitions stay smooth.
        if self.paddle_flash_frames > 0 {
            self.paddle_flash_frames -= 1;
        }
        self.particles.update(PARTICLE_GRAVITY);
        self.push_trail();

        match self.phase {
            Phase::Won | Phase::Lost => {
                if ctx.is_held(button::START) {
                    self.reset_match();
                }
                return;
            }
            Phase::Serve => {
                self.paddle_input(ctx);
                self.ball_x = self.paddle_x + (PADDLE_W as i16 - BALL_SIZE as i16) / 2;
                self.serve_wait = self.serve_wait.saturating_add(1);
                if ctx.is_held(button::CROSS) || self.serve_wait >= SERVE_AUTO_LAUNCH_FRAMES {
                    self.phase = Phase::Playing;
                    self.serve_wait = 0;
                }
                return;
            }
            Phase::Playing => { /* fall through */ }
        }

        self.paddle_input(ctx);

        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        // Wall collisions (L/R/top; no bottom wall — losing is the bottom).
        if self.ball_x <= BORDER_W as i16 {
            self.ball_x = BORDER_W as i16;
            self.ball_vx = self.ball_vx.abs();
            sfx::play(VOICE_WALL);
        } else if self.ball_x + BALL_SIZE as i16 >= SCREEN_W - BORDER_W as i16 {
            self.ball_x = SCREEN_W - BORDER_W as i16 - BALL_SIZE as i16;
            self.ball_vx = -self.ball_vx.abs();
            sfx::play(VOICE_WALL);
        }
        if self.ball_y <= 2 {
            self.ball_y = 2;
            self.ball_vy = self.ball_vy.abs();
            sfx::play(VOICE_WALL);
        }

        // Paddle collision.
        if self.ball_vy > 0
            && self.ball_y + BALL_SIZE as i16 >= PADDLE_Y
            && self.ball_y <= PADDLE_Y + PADDLE_H as i16
            && self.ball_x + BALL_SIZE as i16 >= self.paddle_x
            && self.ball_x <= self.paddle_x + PADDLE_W as i16
        {
            self.ball_y = PADDLE_Y - BALL_SIZE as i16;
            self.ball_vy = -self.ball_vy.abs();
            let hit_offset = (self.ball_x + BALL_SIZE as i16 / 2) - (self.paddle_x + PADDLE_W as i16 / 2);
            let half = PADDLE_W as i16 / 2;
            self.ball_vx = (hit_offset * BALL_MAX_SPEED / half).clamp(-BALL_MAX_SPEED, BALL_MAX_SPEED);
            if self.ball_vx == 0 {
                self.ball_vx = 1;
            }
            self.paddle_flash_frames = PADDLE_FLASH_FRAMES_ON_HIT;
            sfx::play(VOICE_PADDLE);
        }

        // Ball fell past the paddle?
        if self.ball_y >= FLOOR_Y {
            self.lives = self.lives.saturating_sub(1);
            sfx::play(VOICE_LOSE);
            if self.lives == 0 {
                self.phase = Phase::Lost;
                return;
            }
            self.phase = Phase::Serve;
            self.serve_wait = 0;
            self.reset_ball_on_paddle();
            return;
        }

        self.resolve_brick_collision();

        if self.bricks_left == 0 {
            self.phase = Phase::Won;
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        self.build_frame_ot();
        unsafe { OT.submit() };
        self.draw_hud();
    }
}

impl Breakout {
    fn build_frame_ot(&mut self) {
        let ot = unsafe { &mut OT };
        let rects = unsafe { &mut RECTS };
        let bg = unsafe { &mut BG_QUAD };
        ot.clear();

        let (shake_dx, shake_dy) = self.shake.tick();

        // Slot 7 (back) — gradient background.
        *bg = QuadGouraud::new(
            [
                (0, 0),
                (SCREEN_W, 0),
                (0, SCREEN_H),
                (SCREEN_W, SCREEN_H),
            ],
            [
                (22, 30, 64),
                (22, 30, 64),
                (4, 6, 18),
                (4, 6, 18),
            ],
        );
        ot.add(7, bg, QuadGouraud::WORDS);

        // Slot 6 — side borders.
        rects[0] = RectFlat::new(shake_dx, 0, BORDER_W, SCREEN_H as u16, 140, 140, 180);
        rects[1] = RectFlat::new(
            SCREEN_W - BORDER_W as i16 + shake_dx,
            0,
            BORDER_W,
            SCREEN_H as u16,
            140,
            140,
            180,
        );
        ot.add(6, &mut rects[0], RectFlat::WORDS);
        ot.add(6, &mut rects[1], RectFlat::WORDS);

        // Slot 4 — bricks.
        let mut idx = 2;
        for row in 0..ROWS {
            let (r, gc, b) = ROW_COLORS[row];
            for col in 0..COLS {
                let bidx = row * COLS + col;
                if !self.bricks[bidx] {
                    continue;
                }
                let bx = WALL_LEFT + (col as i16) * (BRICK_W as i16 + BRICK_GAP) + shake_dx;
                let by = WALL_TOP + (row as i16) * (BRICK_H as i16 + BRICK_GAP) + shake_dy;
                rects[idx] = RectFlat::new(bx, by, BRICK_W, BRICK_H, r, gc, b);
                ot.add(4, &mut rects[idx], RectFlat::WORDS);
                idx += 1;
            }
        }

        // Slot 3 — particles. Reserve 6 trailing slots for
        // paddle + ball + trail so particles don't starve them.
        let particle_budget = rects.len().saturating_sub(idx + 6);
        let wrote = self.particles.render_into_ot(
            ot,
            &mut rects[idx..idx + particle_budget],
            3,
            (shake_dx, shake_dy),
        );
        idx += wrote;

        // Slot 2 — ball trail (only during Playing).
        if self.phase == Phase::Playing {
            for i in 0..TRAIL_LEN {
                let slot_idx = (self.trail_head as usize + i) % TRAIL_LEN;
                let (tx, ty) = self.trail_positions[slot_idx];
                if i == TRAIL_LEN - 1 {
                    continue;
                }
                let brightness = (i + 1) as u16 * 50;
                let r = (brightness.min(220)) as u8;
                let gc = (brightness.min(200)) as u8;
                let b = (brightness.min(120)) as u8;
                let size = (3 + i) as u16;
                rects[idx] = RectFlat::new(tx + shake_dx, ty + shake_dy, size, size, r, gc, b);
                ot.add(2, &mut rects[idx], RectFlat::WORDS);
                idx += 1;
            }
        }

        // Slot 1 — paddle (flashes bright after ball hit).
        let (pr, pg, pb) = if self.paddle_flash_frames > 0 {
            (255, 255, 200)
        } else {
            (220, 220, 240)
        };
        rects[idx] = RectFlat::new(
            self.paddle_x + shake_dx,
            PADDLE_Y + shake_dy,
            PADDLE_W,
            PADDLE_H,
            pr,
            pg,
            pb,
        );
        ot.add(1, &mut rects[idx], RectFlat::WORDS);
        idx += 1;

        // Slot 0 (front) — ball.
        rects[idx] = RectFlat::new(
            self.ball_x + shake_dx,
            self.ball_y + shake_dy,
            BALL_SIZE,
            BALL_SIZE,
            255,
            230,
            120,
        );
        ot.add(0, &mut rects[idx], RectFlat::WORDS);
    }

    fn draw_hud(&self) {
        let Some(font) = self.font.as_ref() else { return };
        font.draw_text(4, 4, "SCORE", (180, 180, 220));
        let score = u16_hex(self.score);
        font.draw_text(4 + 8 * 6, 4, score.as_str(), (240, 240, 140));
        font.draw_text(SCREEN_W - 8 * 10, 4, "LIVES", (180, 180, 220));
        let lives = digit_char(self.lives);
        font.draw_text(SCREEN_W - 8 * 2, 4, lives.as_str(), (240, 180, 140));

        match self.phase {
            Phase::Serve => {
                font.draw_text(
                    (SCREEN_W - 8 * 19) / 2,
                    PADDLE_Y - 40,
                    "press X to serve",
                    (200, 200, 200),
                );
            }
            Phase::Won => {
                font.draw_text(
                    (SCREEN_W - 8 * 10) / 2,
                    100,
                    "YOU  WIN!",
                    (255, 220, 120),
                );
                font.draw_text(
                    (SCREEN_W - 8 * 17) / 2,
                    130,
                    "START to restart",
                    (180, 180, 200),
                );
            }
            Phase::Lost => {
                font.draw_text(
                    (SCREEN_W - 8 * 10) / 2,
                    100,
                    "GAME OVER",
                    (255, 120, 120),
                );
                font.draw_text(
                    (SCREEN_W - 8 * 17) / 2,
                    130,
                    "START to restart",
                    (180, 180, 200),
                );
            }
            Phase::Playing => {}
        }
    }
}

// ----------------------------------------------------------------------
// no_std formatting helpers
// ----------------------------------------------------------------------

fn u16_hex(v: u16) -> HexU16 {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = [0u8; 6];
    out[0] = b'0';
    out[1] = b'x';
    out[2] = HEX[((v >> 12) & 0xF) as usize];
    out[3] = HEX[((v >> 8) & 0xF) as usize];
    out[4] = HEX[((v >> 4) & 0xF) as usize];
    out[5] = HEX[(v & 0xF) as usize];
    HexU16(out)
}

struct HexU16([u8; 6]);
impl HexU16 {
    fn as_str(&self) -> &str {
        // SAFETY: all bytes are ASCII hex / '0' / 'x'.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}

fn digit_char(n: u8) -> DigitChar {
    let c = if n <= 9 { b'0' + n } else { b'X' };
    DigitChar([c])
}

struct DigitChar([u8; 1]);
impl DigitChar {
    fn as_str(&self) -> &str {
        // SAFETY: '0'..='9' or 'X'.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let config = Config {
        clear_color: (6, 8, 24),
        ..Config::default()
    };
    let mut game = Breakout::new();
    App::run(config, &mut game);
}
