//! `breakout` — mini-game #2. Classic brick-buster: paddle at the
//! bottom, 40-brick wall at the top, bounce the ball into the wall.
//!
//! Bigger than Pong and stresses parts of the SDK Pong didn't:
//!
//! - **OT scale**: ~100 primitives per busy frame (40 bricks + 32
//!   particle-pool + 4 ball-trail + paddle + ball + 2 borders +
//!   gradient background). Validates the ordering-table path
//!   handles non-trivial scene counts.
//! - **Axis-resolved collision**: bricks can be hit from any side;
//!   we pick the smallest-penetration axis and reflect that
//!   component only, so the ball visually bounces off the *edge*
//!   that was touched rather than flipping both axes.
//! - **Multiple game states**: serve (ball rides the paddle),
//!   playing, won (cleared the wall), lost (no lives left). START
//!   resets after either end state.
//! - **Row-colored bricks**: five HSV-lookalike rows with
//!   different point values so the scoreboard climbs unevenly,
//!   showing the font's hex-printing helper stays legible over
//!   the game frame.
//! - **Arcade polish effects** — gradient background
//!   (`QuadGouraud`), screen shake on brick break, brick-break
//!   particle burst, ball trail afterimages, paddle flash on hit.
//!   All built from the same `RectFlat` / `QuadGouraud` primitives
//!   already used for the static scene — no new hardware paths,
//!   just clever scheduling in the OT.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC_8X16};
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

/// Brick wall dimensions.
const COLS: usize = 8;
const ROWS: usize = 5;
const BRICK_COUNT: usize = COLS * ROWS;
const BRICK_W: u16 = 36;
const BRICK_H: u16 = 12;
const BRICK_GAP: i16 = 4;
/// Horizontal margin so the grid fits within the play area.
/// `(COLS * (BRICK_W + BRICK_GAP)) - BRICK_GAP = 8*40 - 4 = 316`
/// fits inside 320 with 2px on each side.
const WALL_LEFT: i16 = 2;
const WALL_TOP: i16 = 24;

/// Paddle dimensions + placement.
const PADDLE_W: u16 = 44;
const PADDLE_H: u16 = 6;
const PADDLE_Y: i16 = SCREEN_H - 20;
const PADDLE_SPEED: i16 = 4;

/// Ball.
const BALL_SIZE: u16 = 6;
const BALL_SPEED_X_INIT: i16 = 2;
const BALL_SPEED_Y_INIT: i16 = -2; // up
const BALL_MAX_SPEED: i16 = 4;

/// Play-area borders (left + right vertical strips).
const BORDER_W: u16 = 2;
/// Floor Y — below this the ball is "out" and a life is lost.
const FLOOR_Y: i16 = SCREEN_H - 2;

/// Row point values — top row is rarest / hardest to reach.
const ROW_POINTS: [u16; ROWS] = [50, 40, 30, 20, 10];
/// Row colours, rgb8. Classic rainbow from top.
const ROW_COLORS: [(u8, u8, u8); ROWS] = [
    (220, 60, 60),   // red
    (220, 140, 60),  // orange
    (220, 220, 60),  // yellow
    (60, 200, 80),   // green
    (80, 140, 240),  // blue
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
// Game state
// ----------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq)]
enum Phase {
    /// Ball rides the paddle, waiting for CROSS to serve.
    Serve,
    /// Ball in play.
    Playing,
    /// Wall cleared.
    Won,
    /// No lives left.
    Lost,
}

struct Game {
    phase: Phase,
    paddle_x: i16,
    ball_x: i16,
    ball_y: i16,
    ball_vx: i16,
    ball_vy: i16,
    score: u16,
    lives: u8,
    /// True = brick alive. Indexed by `row * COLS + col`.
    bricks: [bool; BRICK_COUNT],
    /// How many bricks are still alive — cached so the win-check
    /// doesn't walk the array every frame.
    bricks_left: u16,
    /// Frames the current Serve has been waiting. Auto-launch
    /// after [`SERVE_AUTO_LAUNCH_FRAMES`] so the game plays itself
    /// if the pad is idle (good UX + deterministic milestones).
    serve_wait: u16,
    /// Screen-shake countdown. Non-zero = jitter the draw offset.
    shake_frames: u8,
    /// Paddle-flash countdown. Non-zero = tint the paddle brighter.
    paddle_flash_frames: u8,
    /// Particle pool — bursts from breaking bricks. Reused slots.
    particles: [Particle; MAX_PARTICLES],
    /// Ball's last N positions for the trail effect. Ring buffer,
    /// `trail_head` is the index of the oldest entry.
    trail_positions: [(i16, i16); TRAIL_LEN],
    trail_head: u8,
    /// Tiny LCG PRNG state for particle velocity seeding. Deterministic
    /// so the milestone goldens stay stable.
    rng: u32,
    /// Frame counter — drives per-frame effects that want a phase
    /// reference (none currently, but useful for future pulse FX).
    frame: u32,
}

/// Auto-launch the ball when Serve phase has held for this many
/// frames without a CROSS press. ~0.5 s at 60 Hz.
const SERVE_AUTO_LAUNCH_FRAMES: u16 = 30;

/// Maximum concurrent brick-break particles. Each brick hit
/// spawns [`PARTICLES_PER_BURST`] — pool size needs to cover a
/// handful of overlapping bursts.
const MAX_PARTICLES: usize = 32;
/// Particles spawned per brick break.
const PARTICLES_PER_BURST: usize = 6;
/// Particle time-to-live in frames (≈0.5 s at 60 Hz).
const PARTICLE_TTL: u8 = 30;
/// How many previous ball positions the trail samples.
const TRAIL_LEN: usize = 4;
/// Frames the screen stays shaky after a brick break.
const SHAKE_FRAMES_ON_BREAK: u8 = 6;
/// Frames the paddle stays bright after a ball hit.
const PADDLE_FLASH_FRAMES_ON_HIT: u8 = 6;

/// Single particle — a short-lived coloured dot that drifts then fades.
#[derive(Copy, Clone)]
struct Particle {
    x: i16,
    y: i16,
    /// Sub-pixel velocity in Q4.4 — so `vx = 32` is 2 px/frame.
    vx: i16,
    vy: i16,
    /// Packed colour at spawn; renderer fades this by `ttl`.
    r: u8,
    g: u8,
    b: u8,
    /// Frames remaining. 0 = slot empty.
    ttl: u8,
}

impl Particle {
    const fn empty() -> Self {
        Self { x: 0, y: 0, vx: 0, vy: 0, r: 0, g: 0, b: 0, ttl: 0 }
    }
}

static mut GAME: Game = Game {
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
    shake_frames: 0,
    paddle_flash_frames: 0,
    particles: [Particle::empty(); MAX_PARTICLES],
    trail_positions: [(0, 0); TRAIL_LEN],
    trail_head: 0,
    rng: 0xBEEF_0042,
    frame: 0,
};

static mut OT: OrderingTable<8> = OrderingTable::new();
/// Rect slots for the full scene incl. effects:
///   2 borders + 40 bricks + 32 particles + 4 trail + paddle +
///   ball = 80, rounded up to 96 for headroom.
static mut RECTS: [RectFlat; 96] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 96];
/// One gradient-background quad.
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
    spu::upload_adpcm(SPU_WALL, tones::SINE);
    spu::upload_adpcm(SPU_PADDLE, tones::SQUARE);
    spu::upload_adpcm(SPU_BRICK, tones::TRIANGLE);
    spu::upload_adpcm(SPU_LOSE, tones::SAWTOOTH);
    configure_sfx_voice(VOICE_WALL, SPU_WALL, Pitch::raw(0x1200));
    configure_sfx_voice(VOICE_PADDLE, SPU_PADDLE, Pitch::raw(0x0E00));
    configure_sfx_voice(VOICE_BRICK, SPU_BRICK, Pitch::raw(0x1600));
    configure_sfx_voice(VOICE_LOSE, SPU_LOSE, Pitch::raw(0x0600));

    let font = FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT);

    reset_match();

    loop {
        let pad = poll_port1();
        update_game(pad);

        fb.clear(6, 8, 24);

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

fn reset_match() {
    let g = unsafe { &mut GAME };
    g.phase = Phase::Serve;
    g.paddle_x = (SCREEN_W - PADDLE_W as i16) / 2;
    g.score = 0;
    g.lives = 3;
    g.bricks = [true; BRICK_COUNT];
    g.bricks_left = BRICK_COUNT as u16;
    g.serve_wait = 0;
    g.shake_frames = 0;
    g.paddle_flash_frames = 0;
    g.particles = [Particle::empty(); MAX_PARTICLES];
    g.trail_positions = [(0, 0); TRAIL_LEN];
    g.trail_head = 0;
    g.rng = 0xBEEF_0042;
    g.frame = 0;
    reset_ball_on_paddle();
}

/// Park-miller step — 32-bit integer PRNG, deterministic across
/// runs. Not cryptographically interesting; perfect for picking
/// particle velocities.
fn next_rand(g: &mut Game) -> u32 {
    g.rng = g.rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
    g.rng
}

/// Pick a signed random in `-range..=range` using 5 bits of
/// entropy from the LCG — enough granularity for particle spray.
fn rand_sign(g: &mut Game, range: i16) -> i16 {
    let r = next_rand(g);
    let raw = ((r >> 16) & 0x1F) as i16; // 0..=31
    (raw - 16) * range / 16 // scale to -range..=range
}

/// Spawn a burst of particles from a brick centre. Velocities
/// spray outward with small random jitter; colour inherits from
/// the brick that broke.
fn spawn_brick_particles(g: &mut Game, cx: i16, cy: i16, color: (u8, u8, u8)) {
    let mut spawned = 0;
    for slot in 0..MAX_PARTICLES {
        if g.particles[slot].ttl != 0 {
            continue;
        }
        // Spread: outward "plus gravity" feel. Bias upward a bit
        // so the explosion looks satisfying then falls.
        let vx = rand_sign(g, 40);
        let vy = rand_sign(g, 40) - 16; // pulls up
        g.particles[slot] = Particle {
            x: cx,
            y: cy,
            vx,
            vy,
            r: color.0,
            g: color.1,
            b: color.2,
            ttl: PARTICLE_TTL,
        };
        spawned += 1;
        if spawned >= PARTICLES_PER_BURST {
            break;
        }
    }
}

/// Update every live particle: drift + gravity + decrement TTL.
fn update_particles(g: &mut Game) {
    for p in &mut g.particles {
        if p.ttl == 0 {
            continue;
        }
        // Q4.4 velocity → integrate into position at 1/16 px /frame.
        // Tracking a sub-pixel accumulator would be cleaner; for
        // this TTL (30 frames) the drift is fine as integer px.
        p.x = p.x.wrapping_add(p.vx / 16);
        p.y = p.y.wrapping_add(p.vy / 16);
        // Gravity: +2 Q4.4 per frame → +0.125 px/frame²
        p.vy = p.vy.saturating_add(2);
        p.ttl -= 1;
    }
}

/// Record the current ball position into the trail ring buffer
/// (called once per frame).
fn push_trail(g: &mut Game) {
    g.trail_positions[g.trail_head as usize] = (g.ball_x, g.ball_y);
    g.trail_head = (g.trail_head + 1) % TRAIL_LEN as u8;
}

/// Park the ball on top of the paddle for serve.
fn reset_ball_on_paddle() {
    let g = unsafe { &mut GAME };
    g.ball_x = g.paddle_x + (PADDLE_W as i16 - BALL_SIZE as i16) / 2;
    g.ball_y = PADDLE_Y - BALL_SIZE as i16;
    g.ball_vx = BALL_SPEED_X_INIT;
    g.ball_vy = BALL_SPEED_Y_INIT;
}

fn update_game(pad: ButtonState) {
    let g = unsafe { &mut GAME };

    // Effect counters tick every frame so they also decay during
    // Serve / Won / Lost. Keeps transitions smooth.
    g.frame = g.frame.wrapping_add(1);
    if g.shake_frames > 0 {
        g.shake_frames -= 1;
    }
    if g.paddle_flash_frames > 0 {
        g.paddle_flash_frames -= 1;
    }
    update_particles(g);
    push_trail(g);

    match g.phase {
        Phase::Won | Phase::Lost => {
            if pad.is_held(button::START) {
                reset_match();
            }
            return;
        }
        Phase::Serve => {
            // Paddle moves even during serve so the player can aim.
            paddle_input(g, pad);
            // Ball follows paddle.
            g.ball_x = g.paddle_x + (PADDLE_W as i16 - BALL_SIZE as i16) / 2;
            // Manual launch via CROSS, or auto-launch after a
            // brief wait — keeps gameplay responsive to human
            // input while giving the test harness (which can't
            // press buttons) something to measure.
            g.serve_wait = g.serve_wait.saturating_add(1);
            if pad.is_held(button::CROSS) || g.serve_wait >= SERVE_AUTO_LAUNCH_FRAMES {
                g.phase = Phase::Playing;
                g.serve_wait = 0;
            }
            return;
        }
        Phase::Playing => { /* fall through */ }
    }

    paddle_input(g, pad);

    // Integrate ball position.
    g.ball_x += g.ball_vx;
    g.ball_y += g.ball_vy;

    // Wall collisions (left / right / top). Ball doesn't have a
    // bottom wall — falling past the paddle loses a life.
    if g.ball_x <= BORDER_W as i16 {
        g.ball_x = BORDER_W as i16;
        g.ball_vx = g.ball_vx.abs();
        play_sfx(VOICE_WALL);
    } else if g.ball_x + BALL_SIZE as i16 >= SCREEN_W - BORDER_W as i16 {
        g.ball_x = SCREEN_W - BORDER_W as i16 - BALL_SIZE as i16;
        g.ball_vx = -g.ball_vx.abs();
        play_sfx(VOICE_WALL);
    }
    if g.ball_y <= 2 {
        g.ball_y = 2;
        g.ball_vy = g.ball_vy.abs();
        play_sfx(VOICE_WALL);
    }

    // Paddle collision.
    if g.ball_vy > 0
        && g.ball_y + BALL_SIZE as i16 >= PADDLE_Y
        && g.ball_y <= PADDLE_Y + PADDLE_H as i16
        && g.ball_x + BALL_SIZE as i16 >= g.paddle_x
        && g.ball_x <= g.paddle_x + PADDLE_W as i16
    {
        g.ball_y = PADDLE_Y - BALL_SIZE as i16;
        g.ball_vy = -g.ball_vy.abs();
        // Add some horizontal push based on where the ball hit
        // the paddle. The further from centre, the more angle.
        let hit_offset = (g.ball_x + BALL_SIZE as i16 / 2) - (g.paddle_x + PADDLE_W as i16 / 2);
        let half = PADDLE_W as i16 / 2;
        g.ball_vx = (hit_offset * BALL_MAX_SPEED / half).clamp(-BALL_MAX_SPEED, BALL_MAX_SPEED);
        if g.ball_vx == 0 {
            // Guarantee forward motion — a zero vx would stick.
            g.ball_vx = 1;
        }
        g.paddle_flash_frames = PADDLE_FLASH_FRAMES_ON_HIT;
        play_sfx(VOICE_PADDLE);
    }

    // Ball fell past the paddle?
    if g.ball_y >= FLOOR_Y {
        g.lives = g.lives.saturating_sub(1);
        play_sfx(VOICE_LOSE);
        if g.lives == 0 {
            g.phase = Phase::Lost;
            return;
        }
        g.phase = Phase::Serve;
        g.serve_wait = 0;
        reset_ball_on_paddle();
        return;
    }

    // Brick collision — we resolve the first hit found. On the
    // frame after a hit the ball has already been deflected away,
    // so single-hit-per-frame is correct and avoids awkward
    // tunneling edge cases.
    resolve_brick_collision(g);

    if g.bricks_left == 0 {
        g.phase = Phase::Won;
    }
}

fn paddle_input(g: &mut Game, pad: ButtonState) {
    if pad.is_held(button::LEFT) {
        g.paddle_x -= PADDLE_SPEED;
    }
    if pad.is_held(button::RIGHT) {
        g.paddle_x += PADDLE_SPEED;
    }
    let min = BORDER_W as i16;
    let max = SCREEN_W - BORDER_W as i16 - PADDLE_W as i16;
    if g.paddle_x < min {
        g.paddle_x = min;
    }
    if g.paddle_x > max {
        g.paddle_x = max;
    }
}

/// Walk the brick grid, find the first live brick that the ball
/// rect overlaps, reflect the ball's velocity on the axis of
/// smallest penetration, mark the brick dead, play SFX + add
/// score.
fn resolve_brick_collision(g: &mut Game) {
    let bx0 = g.ball_x;
    let by0 = g.ball_y;
    let bx1 = bx0 + BALL_SIZE as i16;
    let by1 = by0 + BALL_SIZE as i16;

    // Grid is axis-aligned — quick cull by Y to skip rows entirely
    // below / above the ball.
    let wall_bottom = WALL_TOP + ROWS as i16 * (BRICK_H as i16 + BRICK_GAP) - BRICK_GAP;
    if by1 < WALL_TOP || by0 > wall_bottom {
        return;
    }

    for row in 0..ROWS {
        for col in 0..COLS {
            let idx = row * COLS + col;
            if !g.bricks[idx] {
                continue;
            }
            let bx = WALL_LEFT + (col as i16) * (BRICK_W as i16 + BRICK_GAP);
            let by = WALL_TOP + (row as i16) * (BRICK_H as i16 + BRICK_GAP);
            let bx_end = bx + BRICK_W as i16;
            let by_end = by + BRICK_H as i16;
            // AABB overlap test.
            if bx1 <= bx || bx0 >= bx_end || by1 <= by || by0 >= by_end {
                continue;
            }
            // Overlap found — compute penetration on each axis and
            // reflect the smaller one. Positive pen_x means the
            // ball is intruding from the left side of the brick.
            let pen_left = bx1 - bx;
            let pen_right = bx_end - bx0;
            let pen_top = by1 - by;
            let pen_bottom = by_end - by0;
            let pen_x = pen_left.min(pen_right);
            let pen_y = pen_top.min(pen_bottom);
            if pen_x < pen_y {
                // Horizontal bounce.
                if pen_left < pen_right {
                    g.ball_x = bx - BALL_SIZE as i16;
                    g.ball_vx = -g.ball_vx.abs();
                } else {
                    g.ball_x = bx_end;
                    g.ball_vx = g.ball_vx.abs();
                }
            } else {
                // Vertical bounce.
                if pen_top < pen_bottom {
                    g.ball_y = by - BALL_SIZE as i16;
                    g.ball_vy = -g.ball_vy.abs();
                } else {
                    g.ball_y = by_end;
                    g.ball_vy = g.ball_vy.abs();
                }
            }
            g.bricks[idx] = false;
            g.bricks_left -= 1;
            g.score = g.score.saturating_add(ROW_POINTS[row]);
            // Effect triggers: particle burst + screen shake.
            let brick_color = ROW_COLORS[row];
            let brick_cx = bx + BRICK_W as i16 / 2;
            let brick_cy = by + BRICK_H as i16 / 2;
            spawn_brick_particles(g, brick_cx, brick_cy, brick_color);
            g.shake_frames = SHAKE_FRAMES_ON_BREAK;
            play_sfx(VOICE_BRICK);
            return;
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

    let g = unsafe { &GAME };

    // Screen shake: deterministic triangle-wave jitter over the
    // last `shake_frames` frames. Keeps the HUD still (it's drawn
    // immediate-mode after the OT submit, outside this shake).
    let (shake_dx, shake_dy) = if g.shake_frames > 0 {
        // Drift sign flips each frame so it reads as a judder.
        let s = g.shake_frames as i16;
        let sign = if s & 1 == 0 { 1 } else { -1 };
        ((s / 2) * sign, ((s + 1) / 2) * -sign)
    } else {
        (0, 0)
    };

    // OT slot convention (see `psx_gpu::ot`): slot N-1 draws
    // FIRST and slot 0 LAST. On a PSX with no Z-buffer, later
    // draws paint over earlier draws — so slot 0 = front, slot 7
    // = back. That's why the gradient goes to slot 7 and the ball
    // goes to slot 0.
    //
    // Slot 7 (back) — gradient background via the new QuadGouraud
    // primitive. Top is a deep indigo, bottom fades to black-navy
    // so the bricks pop off the upper half.
    *bg = QuadGouraud::new(
        [
            (0, 0),
            (SCREEN_W, 0),
            (0, SCREEN_H),
            (SCREEN_W, SCREEN_H),
        ],
        [
            (22, 30, 64),   // top-left   — richer indigo
            (22, 30, 64),
            (4, 6, 18),     // bottom-left — near-black navy
            (4, 6, 18),
        ],
    );
    // SAFETY: `bg` points into BG_QUAD which is a fixed static
    // address; DMA walker sees a stable bus address.
    unsafe {
        ot.add(7, bg, QuadGouraud::WORDS);
    }

    // Slot 6 — side borders just in front of the background.
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
    unsafe {
        ot.add(6, &mut rects[0], RectFlat::WORDS);
        ot.add(6, &mut rects[1], RectFlat::WORDS);
    }

    // Slot 4 — bricks. Shake applied as a vertex offset so only
    // the game field jitters, not the gradient / HUD.
    let mut idx = 2;
    for row in 0..ROWS {
        let (r, gc, b) = ROW_COLORS[row];
        for col in 0..COLS {
            let bidx = row * COLS + col;
            if !g.bricks[bidx] {
                continue;
            }
            let bx = WALL_LEFT + (col as i16) * (BRICK_W as i16 + BRICK_GAP) + shake_dx;
            let by = WALL_TOP + (row as i16) * (BRICK_H as i16 + BRICK_GAP) + shake_dy;
            rects[idx] = RectFlat::new(bx, by, BRICK_W, BRICK_H, r, gc, b);
            unsafe {
                ot.add(4, &mut rects[idx], RectFlat::WORDS);
            }
            idx += 1;
        }
    }

    // Slot 3 — particles, in front of bricks and behind paddle/ball.
    // Size shrinks + colour dims as TTL decays so each burst fades
    // out gracefully.
    for p in &g.particles {
        if p.ttl == 0 {
            continue;
        }
        // Fade: scale colour channels by ttl / PARTICLE_TTL.
        let scale = p.ttl as u16;
        let denom = PARTICLE_TTL as u16;
        let r = ((p.r as u16 * scale) / denom) as u8;
        let gc = ((p.g as u16 * scale) / denom) as u8;
        let b = ((p.b as u16 * scale) / denom) as u8;
        // Size tapers: 3 px at full ttl, 1 px near death.
        let size = if p.ttl > PARTICLE_TTL / 2 { 3 } else { 2 };
        rects[idx] = RectFlat::new(p.x + shake_dx, p.y + shake_dy, size, size, r, gc, b);
        unsafe {
            ot.add(3, &mut rects[idx], RectFlat::WORDS);
        }
        idx += 1;
        if idx >= rects.len() - 6 {
            break; // leave room for paddle + ball + trail
        }
    }

    // Slot 2 — ball trail, behind the paddle + ball.
    // Uses the ring buffer; oldest entry dimmest. Skip drawing
    // during Serve so the trail doesn't drag along the paddle.
    if g.phase == Phase::Playing {
        for i in 0..TRAIL_LEN {
            let slot_idx = (g.trail_head as usize + i) % TRAIL_LEN;
            let (tx, ty) = g.trail_positions[slot_idx];
            // `i=0` is oldest (dimmest), `i=TRAIL_LEN-1` is newest
            // and closest to the actual ball — skip the newest to
            // avoid drawing right on top of the ball.
            if i == TRAIL_LEN - 1 {
                continue;
            }
            // Linear fade by position in history.
            let brightness = (i + 1) as u16 * 50;
            let r = (brightness.min(220)) as u8;
            let gc = (brightness.min(200)) as u8;
            let b = (brightness.min(120)) as u8;
            // Trail rects shrink slightly too.
            let size = 3 + i as u16;
            rects[idx] = RectFlat::new(tx + shake_dx, ty + shake_dy, size as u16, size as u16, r, gc, b);
            unsafe {
                ot.add(2, &mut rects[idx], RectFlat::WORDS);
            }
            idx += 1;
        }
    }

    // Slot 1 — paddle. Flash brighter while `paddle_flash_frames > 0`.
    let (pr, pg, pb) = if g.paddle_flash_frames > 0 {
        (255, 255, 200) // warm white flash
    } else {
        (220, 220, 240)
    };
    rects[idx] = RectFlat::new(
        g.paddle_x + shake_dx,
        PADDLE_Y + shake_dy,
        PADDLE_W,
        PADDLE_H,
        pr,
        pg,
        pb,
    );
    unsafe {
        ot.add(1, &mut rects[idx], RectFlat::WORDS);
    }
    idx += 1;

    // Slot 0 (frontmost) — ball.
    rects[idx] = RectFlat::new(
        g.ball_x + shake_dx,
        g.ball_y + shake_dy,
        BALL_SIZE,
        BALL_SIZE,
        255,
        230,
        120,
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
    // Score + lives at the top.
    font.draw_text(4, 4, "SCORE", (180, 180, 220));
    let score = u16_hex(g.score);
    font.draw_text(4 + 8 * 6, 4, score.as_str(), (240, 240, 140));
    font.draw_text(SCREEN_W - 8 * 10, 4, "LIVES", (180, 180, 220));
    let lives = digit_char(g.lives);
    font.draw_text(SCREEN_W - 8 * 2, 4, lives.as_str(), (240, 180, 140));

    match g.phase {
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

// ----------------------------------------------------------------------
// no_std formatting helpers
// ----------------------------------------------------------------------

/// Render a u16 as `"0xABCD"` into a stack buffer. Used for the
/// score display.
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

/// Render a u8 0..=9 as a single-char string; >9 becomes 'X'.
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
