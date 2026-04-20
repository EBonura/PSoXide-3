//! `invaders` — mini-game #3, ported to the `psx-engine`
//! Scene/App framework.
//!
//! Classic Space Invaders: 5×10 marching alien grid that speeds
//! up as the formation thins; single player bullet + up to 4
//! enemy bombs; wave progression after clearing the screen.
//! Effects polish: gradient background, explosion particle
//! bursts, screen shake on hit, ship flash.
//!
//! Port pattern is the same as pong + breakout: drop the hand-
//! rolled main loop, replace `static mut GAME` with a scene
//! struct, delegate the per-frame cadence to [`App::run`]. The
//! per-game `frame` counter that invaders tracked by hand
//! collapses into [`Ctx::frame`].
//!
//! What stays in `static mut`: the DMA arena (OT + RECTS +
//! BG_QUAD). Fixed bus addresses the walker can follow.
//!
//! Originally `sdk/examples/game-invaders/`.

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

const COLS: usize = 10;
const ROWS: usize = 5;
const ALIEN_COUNT: usize = COLS * ROWS;
const ALIEN_W: u16 = 16;
const ALIEN_H: u16 = 10;
const ALIEN_H_SPACING: i16 = 10;
const ALIEN_V_SPACING: i16 = 6;
const GRID_LEFT: i16 = 20;
const GRID_TOP: i16 = 28;

const SHIP_W: u16 = 24;
const SHIP_H: u16 = 6;
const SHIP_Y: i16 = SCREEN_H - 18;
const SHIP_SPEED: i16 = 3;

const BULLET_W: u16 = 2;
const BULLET_H: u16 = 6;
const PLAYER_BULLET_VY: i16 = -5;
const ENEMY_BOMB_VY: i16 = 2;
const MAX_ENEMY_BOMBS: usize = 4;

const ROW_POINTS: [u16; ROWS] = [30, 20, 20, 10, 10];
const ROW_COLORS: [(u8, u8, u8); ROWS] = [
    (255, 80, 200),
    (180, 80, 240),
    (80, 240, 200),
    (80, 240, 80),
    (240, 240, 80),
];

const INVASION_Y: i16 = SHIP_Y - 4;

const MOVE_INTERVAL_INITIAL: u16 = 36;
const MOVE_INTERVAL_MIN: u16 = 4;
const MARCH_STEP_X: i16 = 4;
const MARCH_STEP_DOWN: i16 = 8;

const START_AUTO_FRAMES: u16 = 30;

// ----------------------------------------------------------------------
// VRAM + SPU layout
// ----------------------------------------------------------------------

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

const SPU_SHOOT: SpuAddr = SpuAddr::new(0x1010);
const SPU_KILL: SpuAddr = SpuAddr::new(0x1020);
const SPU_MARCH: SpuAddr = SpuAddr::new(0x1030);
const SPU_LOSE: SpuAddr = SpuAddr::new(0x1040);

const VOICE_SHOOT: Voice = Voice::V0;
const VOICE_KILL: Voice = Voice::V1;
const VOICE_MARCH: Voice = Voice::V2;
const VOICE_LOSE: Voice = Voice::V3;

// ----------------------------------------------------------------------
// Effects tunables
// ----------------------------------------------------------------------

const MAX_PARTICLES: usize = 48;
const PARTICLES_PER_KILL: usize = 8;
const PARTICLE_TTL: u8 = 26;
const PARTICLE_SPREAD: i16 = 48;
const PARTICLE_GRAVITY: i16 = 1;
const SHAKE_FRAMES_ON_HIT: u8 = 6;
const SHIP_FLASH_FRAMES: u8 = 10;

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

#[derive(Copy, Clone)]
struct Bullet {
    x: i16,
    y: i16,
    alive: bool,
}

impl Bullet {
    const fn dead() -> Self {
        Self {
            x: 0,
            y: 0,
            alive: false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Phase {
    Serve,
    Playing,
    Lost,
}

struct Invaders {
    phase: Phase,
    ship_x: i16,
    aliens: [bool; ALIEN_COUNT],
    aliens_left: u16,
    grid_offset_x: i16,
    grid_offset_y: i16,
    march_direction: i16,
    march_frames_until_step: u16,
    march_tempo: u16,
    wave: u8,
    score: u16,
    lives: u8,
    player_bullet: Bullet,
    enemy_bombs: [Bullet; MAX_ENEMY_BOMBS],
    serve_wait: u16,
    rng: LcgRng,
    shake: ShakeState,
    ship_flash_frames: u8,
    particles: ParticlePool<MAX_PARTICLES>,
    font: Option<FontAtlas>,
}

impl Invaders {
    const fn new() -> Self {
        Self {
            phase: Phase::Serve,
            ship_x: 0,
            aliens: [true; ALIEN_COUNT],
            aliens_left: ALIEN_COUNT as u16,
            grid_offset_x: 0,
            grid_offset_y: 0,
            march_direction: 1,
            march_frames_until_step: 0,
            march_tempo: MOVE_INTERVAL_INITIAL,
            wave: 1,
            score: 0,
            lives: 3,
            player_bullet: Bullet::dead(),
            enemy_bombs: [Bullet::dead(); MAX_ENEMY_BOMBS],
            serve_wait: 0,
            rng: LcgRng::new(0xC0DE_F00D),
            shake: ShakeState::new(),
            ship_flash_frames: 0,
            particles: ParticlePool::new(),
            font: None,
        }
    }

    /// Rebuild the alien wall and reset the ship to centre. If
    /// `full` is set, scores + lives + wave also reset.
    fn reset_wave(&mut self, full: bool) {
        self.phase = Phase::Serve;
        self.ship_x = (SCREEN_W - SHIP_W as i16) / 2;
        self.aliens = [true; ALIEN_COUNT];
        self.aliens_left = ALIEN_COUNT as u16;
        self.grid_offset_x = 0;
        self.grid_offset_y = (self.wave as i16).saturating_sub(1) * MARCH_STEP_DOWN;
        self.march_direction = 1;
        self.march_frames_until_step = MOVE_INTERVAL_INITIAL;
        self.march_tempo = MOVE_INTERVAL_INITIAL
            .saturating_sub((self.wave as u16).saturating_sub(1) * 4);
        if full {
            self.score = 0;
            self.lives = 3;
            self.wave = 1;
        }
        self.player_bullet = Bullet::dead();
        self.enemy_bombs = [Bullet::dead(); MAX_ENEMY_BOMBS];
        self.serve_wait = 0;
        self.shake = ShakeState::new();
        self.ship_flash_frames = 0;
        self.particles.clear();
        self.rng = LcgRng::new(0xC0DE_F00D ^ (self.wave as u32 * 0x9E37_79B1));
    }

    fn alien_bbox(&self, row: usize, col: usize) -> (i16, i16, i16, i16) {
        let x = GRID_LEFT
            + self.grid_offset_x
            + (col as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING);
        let y = GRID_TOP
            + self.grid_offset_y
            + (row as i16) * (ALIEN_H as i16 + ALIEN_V_SPACING);
        (x, y, x + ALIEN_W as i16, y + ALIEN_H as i16)
    }

    fn ship_input(&mut self, ctx: &Ctx) {
        if ctx.is_held(button::LEFT) {
            self.ship_x -= SHIP_SPEED;
        }
        if ctx.is_held(button::RIGHT) {
            self.ship_x += SHIP_SPEED;
        }
        let min = 4;
        let max = SCREEN_W - SHIP_W as i16 - 4;
        if self.ship_x < min {
            self.ship_x = min;
        }
        if self.ship_x > max {
            self.ship_x = max;
        }
    }

    /// CROSS fires a single bullet. Classic Invaders cadence — can't spam.
    fn handle_player_shot(&mut self, ctx: &Ctx) {
        if ctx.is_held(button::CROSS) && !self.player_bullet.alive {
            self.player_bullet = Bullet {
                x: self.ship_x + SHIP_W as i16 / 2 - BULLET_W as i16 / 2,
                y: SHIP_Y - BULLET_H as i16,
                alive: true,
            };
            sfx::play(VOICE_SHOOT);
        }
    }

    fn advance_bullets(&mut self) {
        if self.player_bullet.alive {
            self.player_bullet.y += PLAYER_BULLET_VY;
            if self.player_bullet.y < 0 {
                self.player_bullet.alive = false;
            }
        }
        for bomb in &mut self.enemy_bombs {
            if bomb.alive {
                bomb.y += ENEMY_BOMB_VY;
                if bomb.y > SCREEN_H {
                    bomb.alive = false;
                }
            }
        }
    }

    /// Tick the formation march: timer → step in direction or
    /// step-down + reverse if blocked by a wall.
    fn advance_alien_march(&mut self) {
        if self.march_frames_until_step > 0 {
            self.march_frames_until_step -= 1;
            return;
        }
        let progress = ALIEN_COUNT as u16 - self.aliens_left;
        let tempo = self
            .march_tempo
            .saturating_sub(progress * 2 / 3)
            .max(MOVE_INTERVAL_MIN);
        self.march_frames_until_step = tempo;

        let (mut leftmost, mut rightmost) = (COLS, 0);
        for row in 0..ROWS {
            for col in 0..COLS {
                if self.aliens[row * COLS + col] {
                    if col < leftmost {
                        leftmost = col;
                    }
                    if col > rightmost {
                        rightmost = col;
                    }
                }
            }
        }
        if leftmost == COLS {
            return;
        }

        let leftmost_x = GRID_LEFT
            + self.grid_offset_x
            + (leftmost as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING);
        let rightmost_x = GRID_LEFT
            + self.grid_offset_x
            + (rightmost as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING)
            + ALIEN_W as i16;

        let predicted_left = leftmost_x + self.march_direction * MARCH_STEP_X;
        let predicted_right = rightmost_x + self.march_direction * MARCH_STEP_X;
        if predicted_left < 4 || predicted_right > SCREEN_W - 4 {
            self.grid_offset_y += MARCH_STEP_DOWN;
            self.march_direction = -self.march_direction;
        } else {
            self.grid_offset_x += self.march_direction * MARCH_STEP_X;
        }
        sfx::play(VOICE_MARCH);
    }

    /// Every 40 frames, a random surviving column's lowest alien
    /// drops a bomb (if a free bomb slot exists).
    fn maybe_drop_enemy_bomb(&mut self, frame: u32) {
        if (frame % 40) != 0 {
            return;
        }
        let slot = match self.enemy_bombs.iter().position(|b| !b.alive) {
            Some(s) => s,
            None => return,
        };
        let col = (self.rng.next() as usize) % COLS;
        for row in (0..ROWS).rev() {
            if self.aliens[row * COLS + col] {
                let (ax, ay, _, ay_end) = self.alien_bbox(row, col);
                let _ = ay;
                self.enemy_bombs[slot] = Bullet {
                    x: ax + (ALIEN_W as i16 - BULLET_W as i16) / 2,
                    y: ay_end,
                    alive: true,
                };
                return;
            }
        }
    }

    fn resolve_player_bullet(&mut self) {
        if !self.player_bullet.alive {
            return;
        }
        let bx0 = self.player_bullet.x;
        let by0 = self.player_bullet.y;
        let bx1 = bx0 + BULLET_W as i16;
        let by1 = by0 + BULLET_H as i16;
        for row in 0..ROWS {
            for col in 0..COLS {
                let idx = row * COLS + col;
                if !self.aliens[idx] {
                    continue;
                }
                let (ax0, ay0, ax1, ay1) = self.alien_bbox(row, col);
                if bx1 <= ax0 || bx0 >= ax1 || by1 <= ay0 || by0 >= ay1 {
                    continue;
                }
                self.aliens[idx] = false;
                self.aliens_left -= 1;
                self.score = self.score.saturating_add(ROW_POINTS[row]);
                self.player_bullet.alive = false;
                let cx = (ax0 + ax1) / 2;
                let cy = (ay0 + ay1) / 2;
                self.particles.spawn_burst(
                    &mut self.rng,
                    (cx, cy),
                    ROW_COLORS[row],
                    PARTICLES_PER_KILL,
                    PARTICLE_SPREAD,
                    PARTICLE_TTL,
                );
                self.shake.trigger(SHAKE_FRAMES_ON_HIT);
                sfx::play(VOICE_KILL);
                return;
            }
        }
    }

    fn resolve_enemy_bombs(&mut self) {
        let ship_x0 = self.ship_x;
        let ship_y0 = SHIP_Y;
        let ship_x1 = self.ship_x + SHIP_W as i16;
        let ship_y1 = SHIP_Y + SHIP_H as i16;

        let mut hit_count: u8 = 0;
        for bomb in &mut self.enemy_bombs {
            if !bomb.alive {
                continue;
            }
            let bx0 = bomb.x;
            let by0 = bomb.y;
            let bx1 = bx0 + BULLET_W as i16;
            let by1 = by0 + BULLET_H as i16;
            if bx1 <= ship_x0 || bx0 >= ship_x1 || by1 <= ship_y0 || by0 >= ship_y1 {
                continue;
            }
            bomb.alive = false;
            hit_count = hit_count.saturating_add(1);
        }
        if hit_count == 0 {
            return;
        }

        let cx = (ship_x0 + ship_x1) / 2;
        let cy = (ship_y0 + ship_y1) / 2;
        self.particles.spawn_burst(
            &mut self.rng,
            (cx, cy),
            (255, 180, 80),
            PARTICLES_PER_KILL,
            PARTICLE_SPREAD,
            PARTICLE_TTL,
        );
        self.shake.trigger(SHAKE_FRAMES_ON_HIT * 2);
        self.ship_flash_frames = SHIP_FLASH_FRAMES;
        self.lives = self.lives.saturating_sub(hit_count);
        sfx::play(VOICE_LOSE);
        if self.lives == 0 {
            self.phase = Phase::Lost;
        }
    }

    /// Any alien crossing the ship's row = instant loss.
    fn check_invasion(&mut self) {
        for row in 0..ROWS {
            for col in 0..COLS {
                if !self.aliens[row * COLS + col] {
                    continue;
                }
                let (_, _, _, ay1) = self.alien_bbox(row, col);
                if ay1 >= INVASION_Y {
                    self.phase = Phase::Lost;
                    self.lives = 0;
                    return;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// DMA arenas
// ----------------------------------------------------------------------

static mut OT: OrderingTable<8> = OrderingTable::new();
/// 50 aliens + 48 particles + player bullet + 4 bombs + ship = 104,
/// padded to 128.
static mut RECTS: [RectFlat; 128] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 128];
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

impl Scene for Invaders {
    fn init(&mut self, _ctx: &mut Ctx) {
        spu::init();
        spu::upload_adpcm(SPU_SHOOT, tones::SAWTOOTH);
        spu::upload_adpcm(SPU_KILL, tones::SQUARE);
        spu::upload_adpcm(SPU_MARCH, tones::TRIANGLE);
        spu::upload_adpcm(SPU_LOSE, tones::SINE);
        sfx::configure_voice(VOICE_SHOOT, SPU_SHOOT, Pitch::raw(0x1800));
        sfx::configure_voice(VOICE_KILL, SPU_KILL, Pitch::raw(0x1200));
        sfx::configure_voice(VOICE_MARCH, SPU_MARCH, Pitch::raw(0x0A00));
        sfx::configure_voice(VOICE_LOSE, SPU_LOSE, Pitch::raw(0x0500));

        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));

        self.reset_wave(true);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if self.ship_flash_frames > 0 {
            self.ship_flash_frames -= 1;
        }
        self.particles.update(PARTICLE_GRAVITY);

        match self.phase {
            Phase::Lost => {
                if ctx.is_held(button::START) {
                    self.wave = 1;
                    self.reset_wave(true);
                }
                return;
            }
            Phase::Serve => {
                self.serve_wait = self.serve_wait.saturating_add(1);
                if ctx.is_held(button::CROSS) || self.serve_wait >= START_AUTO_FRAMES {
                    self.phase = Phase::Playing;
                    self.serve_wait = 0;
                }
                self.ship_input(ctx);
                return;
            }
            Phase::Playing => {}
        }

        self.ship_input(ctx);
        self.handle_player_shot(ctx);
        self.advance_bullets();
        self.advance_alien_march();
        self.maybe_drop_enemy_bomb(ctx.frame);
        self.resolve_player_bullet();
        self.resolve_enemy_bombs();
        self.check_invasion();
        if self.aliens_left == 0 {
            self.wave = self.wave.saturating_add(1);
            self.reset_wave(false);
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        self.build_frame_ot(ctx.frame);
        unsafe { OT.submit() };
        self.draw_hud();
    }
}

impl Invaders {
    fn build_frame_ot(&mut self, frame: u32) {
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
            [(20, 10, 50), (20, 10, 50), (2, 2, 10), (2, 2, 10)],
        );
        ot.add(7, bg, QuadGouraud::WORDS);

        // Slot 5 — aliens.
        let mut idx = 0;
        for row in 0..ROWS {
            let (r, gc, b) = ROW_COLORS[row];
            for col in 0..COLS {
                if !self.aliens[row * COLS + col] {
                    continue;
                }
                let (ax, ay, _, _) = self.alien_bbox(row, col);
                rects[idx] = RectFlat::new(
                    ax + shake_dx,
                    ay + shake_dy,
                    ALIEN_W,
                    ALIEN_H,
                    r,
                    gc,
                    b,
                );
                ot.add(5, &mut rects[idx], RectFlat::WORDS);
                idx += 1;
            }
        }

        // Slot 3 — particles. Reserve 6 trailing slots for
        // bullets + ship.
        let particle_budget = rects.len().saturating_sub(idx + 6);
        let wrote = self.particles.render_into_ot(
            ot,
            &mut rects[idx..idx + particle_budget],
            3,
            (shake_dx, shake_dy),
        );
        idx += wrote;

        // Slot 2 — enemy bombs.
        for bomb in &self.enemy_bombs {
            if !bomb.alive {
                continue;
            }
            rects[idx] = RectFlat::new(
                bomb.x + shake_dx,
                bomb.y + shake_dy,
                BULLET_W,
                BULLET_H,
                255,
                160,
                80,
            );
            ot.add(2, &mut rects[idx], RectFlat::WORDS);
            idx += 1;
        }

        // Slot 1 — player bullet.
        if self.player_bullet.alive {
            rects[idx] = RectFlat::new(
                self.player_bullet.x + shake_dx,
                self.player_bullet.y + shake_dy,
                BULLET_W,
                BULLET_H,
                255,
                255,
                180,
            );
            ot.add(1, &mut rects[idx], RectFlat::WORDS);
            idx += 1;
        }

        // Slot 0 (front) — the ship. Flash uses the engine's
        // frame counter for the bit-2 strobe.
        let (sr, sg, sb) = if self.ship_flash_frames > 0 && (frame & 2 != 0) {
            (255, 200, 80)
        } else {
            (120, 220, 255)
        };
        rects[idx] = RectFlat::new(
            self.ship_x + shake_dx,
            SHIP_Y + shake_dy,
            SHIP_W,
            SHIP_H,
            sr,
            sg,
            sb,
        );
        ot.add(0, &mut rects[idx], RectFlat::WORDS);
    }

    fn draw_hud(&self) {
        let Some(font) = self.font.as_ref() else { return };
        font.draw_text(4, 4, "SCORE", (180, 220, 255));
        let score = u16_hex(self.score);
        font.draw_text(4 + 8 * 6, 4, score.as_str(), (240, 240, 140));
        font.draw_text(SCREEN_W / 2 - 8 * 4, 4, "WAVE", (180, 220, 255));
        let wave = digit_char(self.wave.min(9));
        font.draw_text(SCREEN_W / 2 + 8, 4, wave.as_str(), (240, 200, 140));
        font.draw_text(SCREEN_W - 8 * 10, 4, "LIVES", (180, 220, 255));
        let lives = digit_char(self.lives);
        font.draw_text(SCREEN_W - 8 * 2, 4, lives.as_str(), (240, 180, 140));

        match self.phase {
            Phase::Serve => {
                font.draw_text(
                    (SCREEN_W - 8 * 17) / 2,
                    SHIP_Y - 40,
                    "press X to begin",
                    (200, 200, 200),
                );
            }
            Phase::Lost => {
                font.draw_text(
                    (SCREEN_W - 8 * 9) / 2,
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
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut game = Invaders::new();
    App::run(Config::default(), &mut game);
}
