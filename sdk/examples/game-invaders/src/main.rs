//! `invaders` — mini-game #3. Classic Space Invaders, taken up a
//! notch from Pong + Breakout with:
//!
//! - **5×10 alien grid** that marches side-to-side, steps down,
//!   and speeds up as the formation thins.
//! - **Two bullet pools** — one player bullet (single shot held
//!   until it clears the screen or hits) + up to 4 enemy bombs.
//! - **Score by row** — top row rare and worth most (30 pts),
//!   bottom rows common (10 pts each).
//! - **Wave progression** — clearing the screen respawns a fresh
//!   wall that starts one step lower and marches faster.
//!
//! Effects polish (same toolbox as breakout):
//! - Gradient background via `QuadGouraud`.
//! - Explosion particle burst on alien kill.
//! - Screen shake on each hit.
//! - Player ship flashes when struck.
//!
//! State machine:
//! - Serve: fresh wave painted, 30-frame countdown then march begins.
//! - Playing: normal loop.
//! - Lost: aliens reached the bottom OR player ran out of lives.
//!   START to restart.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_pad::{ButtonState, button, poll_port1};
use psx_spu::{self as spu, Adsr, Pitch, SpuAddr, Voice, Volume, tones};
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Layout constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

/// Alien grid.
const COLS: usize = 10;
const ROWS: usize = 5;
const ALIEN_COUNT: usize = COLS * ROWS;
const ALIEN_W: u16 = 16;
const ALIEN_H: u16 = 10;
const ALIEN_H_SPACING: i16 = 10;
const ALIEN_V_SPACING: i16 = 6;
const GRID_LEFT: i16 = 20;
const GRID_TOP: i16 = 28;

/// Player ship.
const SHIP_W: u16 = 24;
const SHIP_H: u16 = 6;
const SHIP_Y: i16 = SCREEN_H - 18;
const SHIP_SPEED: i16 = 3;

/// Bullets.
const BULLET_W: u16 = 2;
const BULLET_H: u16 = 6;
const PLAYER_BULLET_VY: i16 = -5;
const ENEMY_BOMB_VY: i16 = 2;
const MAX_ENEMY_BOMBS: usize = 4;

/// Row points + colours — rarer rows at the top.
const ROW_POINTS: [u16; ROWS] = [30, 20, 20, 10, 10];
const ROW_COLORS: [(u8, u8, u8); ROWS] = [
    (255, 80, 200),   // pink — rare "squid"
    (180, 80, 240),   // purple
    (80, 240, 200),   // cyan
    (80, 240, 80),    // green
    (240, 240, 80),   // yellow
];

/// Row the aliens must never reach or the player loses.
const INVASION_Y: i16 = SHIP_Y - 4;

/// March timing. Higher `MOVE_INTERVAL` = slower march. We scale
/// it down as aliens die — by wave 1 a handful-remaining formation
/// feels frenetic.
const MOVE_INTERVAL_INITIAL: u16 = 36;
const MOVE_INTERVAL_MIN: u16 = 4;
/// Pixels moved per march tick.
const MARCH_STEP_X: i16 = 4;
const MARCH_STEP_DOWN: i16 = 8;

/// Auto-start the match after this many frames if no input.
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
// Effects (particles + shake + ship flash)
// ----------------------------------------------------------------------

const MAX_PARTICLES: usize = 48;
const PARTICLES_PER_KILL: usize = 8;
const PARTICLE_TTL: u8 = 26;
/// Q4.4 velocity spread for explosion scatter.
const PARTICLE_SPREAD: i16 = 48;
/// Q4.4 gravity per frame (1 = 0.0625 px/frame²).
const PARTICLE_GRAVITY: i16 = 1;
const SHAKE_FRAMES_ON_HIT: u8 = 6;
const SHIP_FLASH_FRAMES: u8 = 10;

#[derive(Copy, Clone)]
struct Bullet {
    /// `x`/`y` are the TOP-LEFT of the bullet rect.
    x: i16,
    y: i16,
    /// Active flag — simpler than `Option<Bullet>` in static arrays.
    alive: bool,
}

impl Bullet {
    const fn dead() -> Self {
        Self { x: 0, y: 0, alive: false }
    }
}

// ----------------------------------------------------------------------
// Game state
// ----------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq)]
enum Phase {
    /// Fresh wave is painted, ship is idle, waiting to begin.
    Serve,
    /// Full gameplay.
    Playing,
    /// Aliens reached the ship OR player's lives hit zero.
    Lost,
}

struct Game {
    phase: Phase,
    ship_x: i16,
    /// True = alien alive. Indexed `row * COLS + col`.
    aliens: [bool; ALIEN_COUNT],
    aliens_left: u16,
    /// Anchor-point offset applied to the whole alien grid each
    /// frame; march logic mutates this.
    grid_offset_x: i16,
    grid_offset_y: i16,
    march_direction: i16, // +1 = right, -1 = left
    march_frames_until_step: u16,
    march_tempo: u16, // decreases as aliens die
    wave: u8,
    score: u16,
    lives: u8,
    /// One active player bullet at a time (classic Invaders).
    player_bullet: Bullet,
    /// Up to `MAX_ENEMY_BOMBS` enemy bombs in flight.
    enemy_bombs: [Bullet; MAX_ENEMY_BOMBS],
    /// Frame timer used for auto-start + deterministic RNG step.
    frame: u32,
    /// Counter for Serve phase's auto-start-when-idle.
    serve_wait: u16,
    /// Deterministic LCG for enemy shot picking + particle spread.
    rng: LcgRng,
    /// Shared SDK effects state.
    shake: ShakeState,
    ship_flash_frames: u8,
    particles: ParticlePool<MAX_PARTICLES>,
}

static mut GAME: Game = Game {
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
    frame: 0,
    serve_wait: 0,
    rng: LcgRng::new(0xC0DE_F00D),
    shake: ShakeState::new(),
    ship_flash_frames: 0,
    particles: ParticlePool::new(),
};

static mut OT: OrderingTable<8> = OrderingTable::new();
/// Scene rects: 50 aliens + particles(48) + player bullet + 4 bombs +
/// ship = 104. Round up for headroom.
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
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(SCREEN_W as u16, SCREEN_H as u16);
    gpu::set_draw_area(0, 0, (SCREEN_W - 1) as u16, (SCREEN_H - 1) as u16);
    gpu::set_draw_offset(0, 0);

    spu::init();
    spu::upload_adpcm(SPU_SHOOT, tones::SAWTOOTH);
    spu::upload_adpcm(SPU_KILL, tones::SQUARE);
    spu::upload_adpcm(SPU_MARCH, tones::TRIANGLE);
    spu::upload_adpcm(SPU_LOSE, tones::SINE);
    configure_sfx_voice(VOICE_SHOOT, SPU_SHOOT, Pitch::raw(0x1800));
    configure_sfx_voice(VOICE_KILL, SPU_KILL, Pitch::raw(0x1200));
    configure_sfx_voice(VOICE_MARCH, SPU_MARCH, Pitch::raw(0x0A00));
    configure_sfx_voice(VOICE_LOSE, SPU_LOSE, Pitch::raw(0x0500));

    let font = FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT);

    reset_wave(true);

    loop {
        let pad = poll_port1();

        update_game(pad);

        fb.clear(0, 0, 0);

        build_frame_ot();
        submit_frame_ot();

        draw_hud(&font);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
    }
}

fn configure_sfx_voice(v: Voice, addr: SpuAddr, pitch: Pitch) {
    v.set_volume(Volume::HALF, Volume::HALF);
    v.set_pitch(pitch);
    v.set_start_addr(addr);
    v.set_adsr(Adsr::percussive());
}

fn play_sfx(v: Voice) {
    Voice::key_on(v.mask());
}

// ----------------------------------------------------------------------
// Game logic
// ----------------------------------------------------------------------

/// Rebuild the alien wall and reset the ship to centre. If
/// `full` is set, scores + lives + wave also reset.
fn reset_wave(full: bool) {
    let g = unsafe { &mut GAME };
    g.phase = Phase::Serve;
    g.ship_x = (SCREEN_W - SHIP_W as i16) / 2;
    g.aliens = [true; ALIEN_COUNT];
    g.aliens_left = ALIEN_COUNT as u16;
    g.grid_offset_x = 0;
    // New waves start slightly lower — classic progression.
    g.grid_offset_y = (g.wave as i16).saturating_sub(1) * MARCH_STEP_DOWN;
    g.march_direction = 1;
    g.march_frames_until_step = MOVE_INTERVAL_INITIAL;
    g.march_tempo = MOVE_INTERVAL_INITIAL.saturating_sub((g.wave as u16).saturating_sub(1) * 4);
    if full {
        g.score = 0;
        g.lives = 3;
        g.wave = 1;
    }
    g.player_bullet = Bullet::dead();
    g.enemy_bombs = [Bullet::dead(); MAX_ENEMY_BOMBS];
    g.serve_wait = 0;
    g.shake = ShakeState::new();
    g.ship_flash_frames = 0;
    g.particles.clear();
    // Seed RNG from wave so each wave's enemy-shot pattern differs.
    g.rng = LcgRng::new(0xC0DE_F00D ^ (g.wave as u32 * 0x9E37_79B1));
}

fn alien_bbox(g: &Game, row: usize, col: usize) -> (i16, i16, i16, i16) {
    let x = GRID_LEFT
        + g.grid_offset_x
        + (col as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING);
    let y = GRID_TOP
        + g.grid_offset_y
        + (row as i16) * (ALIEN_H as i16 + ALIEN_V_SPACING);
    (x, y, x + ALIEN_W as i16, y + ALIEN_H as i16)
}

fn update_game(pad: ButtonState) {
    let g = unsafe { &mut GAME };

    g.frame = g.frame.wrapping_add(1);
    // `shake.tick()` is called in the render path — decrementing
    // there keeps the offset read + consumption coupled.
    if g.ship_flash_frames > 0 {
        g.ship_flash_frames -= 1;
    }
    g.particles.update(PARTICLE_GRAVITY);

    match g.phase {
        Phase::Lost => {
            if pad.is_held(button::START) {
                // Full reset.
                g.wave = 1;
                reset_wave(true);
            }
            return;
        }
        Phase::Serve => {
            g.serve_wait = g.serve_wait.saturating_add(1);
            if pad.is_held(button::CROSS) || g.serve_wait >= START_AUTO_FRAMES {
                g.phase = Phase::Playing;
                g.serve_wait = 0;
            }
            ship_input(g, pad);
            return;
        }
        Phase::Playing => {}
    }

    ship_input(g, pad);
    handle_player_shot(g, pad);
    advance_bullets(g);
    advance_alien_march(g);
    maybe_drop_enemy_bomb(g);
    resolve_player_bullet(g);
    resolve_enemy_bombs(g);
    check_invasion(g);
    if g.aliens_left == 0 {
        // Next wave — keep score / lives, raise difficulty.
        g.wave = g.wave.saturating_add(1);
        reset_wave(false);
    }
}

fn ship_input(g: &mut Game, pad: ButtonState) {
    if pad.is_held(button::LEFT) {
        g.ship_x -= SHIP_SPEED;
    }
    if pad.is_held(button::RIGHT) {
        g.ship_x += SHIP_SPEED;
    }
    let min = 4;
    let max = SCREEN_W - SHIP_W as i16 - 4;
    if g.ship_x < min {
        g.ship_x = min;
    }
    if g.ship_x > max {
        g.ship_x = max;
    }
}

/// CROSS fires a single bullet that sits in the air until it
/// exits the top of the screen or hits an alien. Classic Invaders
/// cadence — can't spam.
fn handle_player_shot(g: &mut Game, pad: ButtonState) {
    if pad.is_held(button::CROSS) && !g.player_bullet.alive {
        g.player_bullet = Bullet {
            x: g.ship_x + SHIP_W as i16 / 2 - BULLET_W as i16 / 2,
            y: SHIP_Y - BULLET_H as i16,
            alive: true,
        };
        play_sfx(VOICE_SHOOT);
    }
}

fn advance_bullets(g: &mut Game) {
    if g.player_bullet.alive {
        g.player_bullet.y += PLAYER_BULLET_VY;
        if g.player_bullet.y < 0 {
            g.player_bullet.alive = false;
        }
    }
    for bomb in &mut g.enemy_bombs {
        if bomb.alive {
            bomb.y += ENEMY_BOMB_VY;
            if bomb.y > SCREEN_H {
                bomb.alive = false;
            }
        }
    }
}

/// Tick the alien formation. If it's not time to step, just
/// decrement the frame counter. On step: move in the current
/// direction, or step down + reverse if hitting a wall.
fn advance_alien_march(g: &mut Game) {
    if g.march_frames_until_step > 0 {
        g.march_frames_until_step -= 1;
        return;
    }
    // Tempo scales with surviving aliens — the fewer left, the
    // faster the remaining ones march.
    let progress = ALIEN_COUNT as u16 - g.aliens_left;
    let tempo = g
        .march_tempo
        .saturating_sub(progress * 2 / 3)
        .max(MOVE_INTERVAL_MIN);
    g.march_frames_until_step = tempo;

    // Find min/max column with a live alien — avoids the "dead
    // edge column prevents step-down" bug.
    let (mut leftmost, mut rightmost) = (COLS, 0);
    for row in 0..ROWS {
        for col in 0..COLS {
            if g.aliens[row * COLS + col] {
                if col < leftmost { leftmost = col; }
                if col > rightmost { rightmost = col; }
            }
        }
    }
    if leftmost == COLS {
        // Nothing alive (shouldn't happen — aliens_left would be 0).
        return;
    }

    let leftmost_x = GRID_LEFT
        + g.grid_offset_x
        + (leftmost as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING);
    let rightmost_x = GRID_LEFT
        + g.grid_offset_x
        + (rightmost as i16) * (ALIEN_W as i16 + ALIEN_H_SPACING)
        + ALIEN_W as i16;

    // Predict next position — if it would clip a wall, step down
    // instead + reverse direction.
    let predicted_left = leftmost_x + g.march_direction * MARCH_STEP_X;
    let predicted_right = rightmost_x + g.march_direction * MARCH_STEP_X;
    if predicted_left < 4 || predicted_right > SCREEN_W - 4 {
        g.grid_offset_y += MARCH_STEP_DOWN;
        g.march_direction = -g.march_direction;
    } else {
        g.grid_offset_x += g.march_direction * MARCH_STEP_X;
    }
    play_sfx(VOICE_MARCH);
}

/// Each march-tick, small probability a random column's lowest
/// surviving alien drops a bomb.
fn maybe_drop_enemy_bomb(g: &mut Game) {
    // Fire roughly every N frames scaled by wave.
    if (g.frame % 40) != 0 {
        return;
    }
    // Need a free bomb slot.
    let slot = match g.enemy_bombs.iter().position(|b| !b.alive) {
        Some(s) => s,
        None => return,
    };
    // Pick a random surviving column.
    let col = (g.rng.next() as usize) % COLS;
    // Lowest live alien in that column.
    for row in (0..ROWS).rev() {
        if g.aliens[row * COLS + col] {
            let (ax, ay, ax_end, ay_end) = alien_bbox(g, row, col);
            let _ = ax_end;
            g.enemy_bombs[slot] = Bullet {
                x: ax + (ALIEN_W as i16 - BULLET_W as i16) / 2,
                y: ay_end,
                alive: true,
            };
            let _ = ax;
            let _ = ay;
            return;
        }
    }
}

fn resolve_player_bullet(g: &mut Game) {
    if !g.player_bullet.alive {
        return;
    }
    let bx0 = g.player_bullet.x;
    let by0 = g.player_bullet.y;
    let bx1 = bx0 + BULLET_W as i16;
    let by1 = by0 + BULLET_H as i16;
    for row in 0..ROWS {
        for col in 0..COLS {
            let idx = row * COLS + col;
            if !g.aliens[idx] {
                continue;
            }
            let (ax0, ay0, ax1, ay1) = alien_bbox(g, row, col);
            if bx1 <= ax0 || bx0 >= ax1 || by1 <= ay0 || by0 >= ay1 {
                continue;
            }
            // Hit!
            g.aliens[idx] = false;
            g.aliens_left -= 1;
            g.score = g.score.saturating_add(ROW_POINTS[row]);
            g.player_bullet.alive = false;
            let cx = (ax0 + ax1) / 2;
            let cy = (ay0 + ay1) / 2;
            g.particles.spawn_burst(
                &mut g.rng,
                (cx, cy),
                ROW_COLORS[row],
                PARTICLES_PER_KILL,
                PARTICLE_SPREAD,
                PARTICLE_TTL,
            );
            g.shake.trigger(SHAKE_FRAMES_ON_HIT);
            play_sfx(VOICE_KILL);
            return;
        }
    }
}

fn resolve_enemy_bombs(g: &mut Game) {
    let ship_x0 = g.ship_x;
    let ship_y0 = SHIP_Y;
    let ship_x1 = g.ship_x + SHIP_W as i16;
    let ship_y1 = SHIP_Y + SHIP_H as i16;

    // First pass — mark hits without mutating anything else on
    // `g`. The borrow checker allows `&mut g.enemy_bombs` here
    // because the only field we touch is `enemy_bombs`. We then
    // release that borrow and apply the hit effect in a second
    // pass that freely takes `&mut g`.
    let mut hit_count: u8 = 0;
    for bomb in &mut g.enemy_bombs {
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

    // Second pass — effects.
    let cx = (ship_x0 + ship_x1) / 2;
    let cy = (ship_y0 + ship_y1) / 2;
    g.particles.spawn_burst(
        &mut g.rng,
        (cx, cy),
        (255, 180, 80),
        PARTICLES_PER_KILL,
        PARTICLE_SPREAD,
        PARTICLE_TTL,
    );
    g.shake.trigger(SHAKE_FRAMES_ON_HIT * 2);
    g.ship_flash_frames = SHIP_FLASH_FRAMES;
    g.lives = g.lives.saturating_sub(hit_count);
    play_sfx(VOICE_LOSE);
    if g.lives == 0 {
        g.phase = Phase::Lost;
    }
}

/// Any alien reaching the ship's row = instant loss.
fn check_invasion(g: &mut Game) {
    for row in 0..ROWS {
        for col in 0..COLS {
            if !g.aliens[row * COLS + col] {
                continue;
            }
            let (_, _, _, ay1) = alien_bbox(g, row, col);
            if ay1 >= INVASION_Y {
                g.phase = Phase::Lost;
                g.lives = 0;
                return;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------

fn build_frame_ot() {
    let ot = unsafe { &mut OT };
    let rects = unsafe { &mut RECTS };
    let bg = unsafe { &mut BG_QUAD };
    ot.clear();

    let g = unsafe { &mut GAME };

    // Shake offset (applied to game-field primitives only).
    let (shake_dx, shake_dy) = g.shake.tick();

    // OT convention: slot N-1 = back, slot 0 = front. Gradient
    // goes to slot 7, ship to slot 0.
    //
    // Slot 7 — gradient background. Deep purple-indigo top fading
    // to near-black bottom for that outer-space feel.
    *bg = QuadGouraud::new(
        [
            (0, 0),
            (SCREEN_W, 0),
            (0, SCREEN_H),
            (SCREEN_W, SCREEN_H),
        ],
        [
            (20, 10, 50),   // top — purple-indigo
            (20, 10, 50),
            (2, 2, 10),     // bottom — near-black
            (2, 2, 10),
        ],
    );
    unsafe {
        ot.add(7, bg, QuadGouraud::WORDS);
    }

    // Slot 5 — aliens.
    let mut idx = 0;
    for row in 0..ROWS {
        let (r, gc, b) = ROW_COLORS[row];
        for col in 0..COLS {
            if !g.aliens[row * COLS + col] {
                continue;
            }
            let (ax, ay, _, _) = alien_bbox(g, row, col);
            rects[idx] = RectFlat::new(
                ax + shake_dx,
                ay + shake_dy,
                ALIEN_W,
                ALIEN_H,
                r,
                gc,
                b,
            );
            unsafe {
                ot.add(5, &mut rects[idx], RectFlat::WORDS);
            }
            idx += 1;
        }
    }

    // Slot 3 — particles (front-of-aliens, behind-bullets/ship).
    // Reserve 6 trailing slots for bullets + ship so the write
    // can't starve them.
    let particle_budget = rects.len().saturating_sub(idx + 6);
    let wrote = g
        .particles
        .render_into_ot(ot, &mut rects[idx..idx + particle_budget], 3, (shake_dx, shake_dy));
    idx += wrote;

    // Slot 2 — enemy bombs.
    for bomb in &g.enemy_bombs {
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
        unsafe {
            ot.add(2, &mut rects[idx], RectFlat::WORDS);
        }
        idx += 1;
    }

    // Slot 1 — player bullet + ship.
    if g.player_bullet.alive {
        rects[idx] = RectFlat::new(
            g.player_bullet.x + shake_dx,
            g.player_bullet.y + shake_dy,
            BULLET_W,
            BULLET_H,
            255,
            255,
            180,
        );
        unsafe {
            ot.add(1, &mut rects[idx], RectFlat::WORDS);
        }
        idx += 1;
    }

    // Slot 0 (front) — the ship. Flash briefly on hit.
    let (sr, sg, sb) = if g.ship_flash_frames > 0 && (g.frame & 2 != 0) {
        (255, 200, 80)
    } else {
        (120, 220, 255)
    };
    rects[idx] = RectFlat::new(
        g.ship_x + shake_dx,
        SHIP_Y + shake_dy,
        SHIP_W,
        SHIP_H,
        sr,
        sg,
        sb,
    );
    unsafe {
        ot.add(0, &mut rects[idx], RectFlat::WORDS);
    }
}

fn submit_frame_ot() {
    unsafe {
        OT.submit();
    }
}

fn draw_hud(font: &FontAtlas) {
    let g = unsafe { &GAME };
    font.draw_text(4, 4, "SCORE", (180, 220, 255));
    let score = u16_hex(g.score);
    font.draw_text(4 + 8 * 6, 4, score.as_str(), (240, 240, 140));
    // Centre: WAVE.
    font.draw_text(SCREEN_W / 2 - 8 * 4, 4, "WAVE", (180, 220, 255));
    let wave = digit_char(g.wave.min(9));
    font.draw_text(SCREEN_W / 2 + 8, 4, wave.as_str(), (240, 200, 140));
    // Right: LIVES.
    font.draw_text(SCREEN_W - 8 * 10, 4, "LIVES", (180, 220, 255));
    let lives = digit_char(g.lives);
    font.draw_text(SCREEN_W - 8 * 2, 4, lives.as_str(), (240, 180, 140));

    match g.phase {
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
