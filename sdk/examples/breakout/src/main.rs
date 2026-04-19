//! `breakout` — mini-game #2. Classic brick-buster: paddle at the
//! bottom, 40-brick wall at the top, bounce the ball into the wall.
//!
//! Bigger than Pong and stresses parts of the SDK Pong didn't:
//!
//! - **OT scale**: up to 44 primitives per frame (40 bricks + 2
//!   borders + paddle + ball). Validates the ordering-table path
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

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::RectFlat;
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
}

/// Auto-launch the ball when Serve phase has held for this many
/// frames without a CROSS press. ~0.5 s at 60 Hz.
const SERVE_AUTO_LAUNCH_FRAMES: u16 = 30;

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
};

static mut OT: OrderingTable<8> = OrderingTable::new();
/// Enough rect slots for the full scene: 40 bricks + 2 borders +
/// paddle + ball = 44 with a little headroom.
static mut RECTS: [RectFlat; 48] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 48];

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
    reset_ball_on_paddle();
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
    ot.clear();

    let g = unsafe { &GAME };

    // Side borders (slot 0 = back).
    rects[0] = RectFlat::new(0, 0, BORDER_W, SCREEN_H as u16, 140, 140, 180);
    rects[1] = RectFlat::new(
        SCREEN_W - BORDER_W as i16,
        0,
        BORDER_W,
        SCREEN_H as u16,
        140,
        140,
        180,
    );
    unsafe {
        ot.add(0, &mut rects[0], RectFlat::WORDS);
        ot.add(0, &mut rects[1], RectFlat::WORDS);
    }

    // Bricks (slot 3).
    let mut idx = 2;
    for row in 0..ROWS {
        let (r, gc, b) = ROW_COLORS[row];
        for col in 0..COLS {
            let bidx = row * COLS + col;
            if !g.bricks[bidx] {
                continue;
            }
            let bx = WALL_LEFT + (col as i16) * (BRICK_W as i16 + BRICK_GAP);
            let by = WALL_TOP + (row as i16) * (BRICK_H as i16 + BRICK_GAP);
            rects[idx] = RectFlat::new(bx, by, BRICK_W, BRICK_H, r, gc, b);
            unsafe {
                ot.add(3, &mut rects[idx], RectFlat::WORDS);
            }
            idx += 1;
        }
    }

    // Paddle (slot 5).
    rects[idx] = RectFlat::new(g.paddle_x, PADDLE_Y, PADDLE_W, PADDLE_H, 220, 220, 240);
    unsafe {
        ot.add(5, &mut rects[idx], RectFlat::WORDS);
    }
    idx += 1;

    // Ball (slot 7, front).
    rects[idx] = RectFlat::new(g.ball_x, g.ball_y, BALL_SIZE, BALL_SIZE, 255, 230, 120);
    unsafe {
        ot.add(7, &mut rects[idx], RectFlat::WORDS);
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
