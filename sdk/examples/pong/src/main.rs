//! `pong` — the first PSoXide-3 mini-game. Classic two-paddle
//! Pong, composed entirely out of SDK pieces:
//!
//! - **GPU** — `psx_gpu::ot` + `psx_gpu::prim::RectFlat` for ball /
//!   paddles / borders / centre dashes, all depth-sorted through
//!   an OT and submitted via DMA linked-list.
//! - **Pad** — `psx_pad` for player input (port 1 D-pad UP/DOWN
//!   drives the left paddle, START resets a finished game).
//! - **SPU** — `psx_spu` for three SFX: wall bounce, paddle hit,
//!   score. Each uses a different built-in tone waveform so the
//!   player can tell them apart without looking.
//! - **Font** — `psx_font` in the 8×16 BIOS-console face for the
//!   scoreboard + the "P1 WINS" banner.
//! - **FrameBuffer** — `psx_gpu::framebuf::FrameBuffer` for tear-
//!   free presentation, same pattern as the other examples.
//!
//! Gameplay:
//! - First to 7 points wins.
//! - Left paddle: D-pad UP / DOWN.
//! - Right paddle: AI that tracks the ball with a capped move
//!   speed (beatable but not a pushover).
//! - After a score, ball respawns at centre aimed at the scorer.
//! - After a match, press START to reset.

#![no_std]
#![no_main]
// Bare-metal single-threaded PSX homebrew — `static mut` is the
// no_alloc workhorse here, and every access site is serialised
// by `main`'s loop. The Rust-2024 static-mut lints are shouting
// about UB in a multi-threaded context that doesn't apply.
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
// Screen + gameplay constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

/// Height of the top + bottom border strips in pixels.
const BORDER_H: u16 = 3;
/// Y at which the playfield starts (below the border + score row).
const PLAYFIELD_TOP: i16 = 28;
/// Y at which the playfield ends.
const PLAYFIELD_BOT: i16 = SCREEN_H - BORDER_H as i16;

/// Paddle dimensions + offsets from the screen edges.
const PADDLE_W: u16 = 6;
const PADDLE_H: u16 = 36;
const PADDLE_MARGIN: i16 = 12;
/// Paddle vertical speed (pixels / frame).
const PADDLE_SPEED: i16 = 3;
/// AI paddle speed — slightly slower than the player's max so the
/// human can actually beat it. Hysteresis band avoids jitter.
const AI_SPEED: i16 = 2;
const AI_HYSTERESIS: i16 = 3;

/// Ball size + starting speed.
const BALL_SIZE: u16 = 6;
const BALL_START_VX: i16 = 2;
const BALL_START_VY: i16 = 1;
const BALL_MAX_SPEED: i16 = 4;

/// Score needed to win.
const WIN_SCORE: u8 = 7;

// ----------------------------------------------------------------------
// VRAM layout
// ----------------------------------------------------------------------

/// Font atlas tpage — past the 320-wide display buffers.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

// ----------------------------------------------------------------------
// SPU layout — one voice per SFX, tone samples at well-known offsets
// ----------------------------------------------------------------------

const SPU_WALL: SpuAddr = SpuAddr::new(0x1010);
const SPU_PADDLE: SpuAddr = SpuAddr::new(0x1020);
const SPU_SCORE: SpuAddr = SpuAddr::new(0x1030);

const VOICE_WALL: Voice = Voice::V0;
const VOICE_PADDLE: Voice = Voice::V1;
const VOICE_SCORE: Voice = Voice::V2;

// ----------------------------------------------------------------------
// Game state
// ----------------------------------------------------------------------

/// All live state lives in a `static mut` — no_std, no_alloc, and
/// we're single-threaded bare metal so unsynchronized access is
/// fine. Access sites are all inside `main` which serialises them.
#[repr(C)]
struct Game {
    p1_y: i16,
    p2_y: i16,
    p1_score: u8,
    p2_score: u8,
    ball_x: i16,
    ball_y: i16,
    ball_vx: i16,
    ball_vy: i16,
    /// 0 = live, 1 = P1 won, 2 = P2 won.
    winner: u8,
    /// Rotates serve direction after each score.
    serve_dir: i8,
}

static mut GAME: Game = Game {
    p1_y: 0,
    p2_y: 0,
    p1_score: 0,
    p2_score: 0,
    ball_x: 0,
    ball_y: 0,
    ball_vx: 0,
    ball_vy: 0,
    winner: 0,
    serve_dir: 1,
};

/// OT + primitive backing storage. Sized generously so any layout
/// change stays well inside the allocation.
static mut OT: OrderingTable<8> = OrderingTable::new();
/// 2 paddles + ball + 2 border strips + up to 12 centre dashes.
static mut RECTS: [RectFlat; 20] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 20];

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(SCREEN_W as u16, SCREEN_H as u16);
    gpu::set_draw_area(0, 0, (SCREEN_W - 1) as u16, (SCREEN_H - 1) as u16);
    gpu::set_draw_offset(0, 0);

    // Audio boot + SFX sample upload. One short tone per SFX, each
    // using a different waveform so the ear can tell them apart.
    spu::init();
    spu::upload_adpcm(SPU_WALL, tones::SINE);
    spu::upload_adpcm(SPU_PADDLE, tones::SQUARE);
    spu::upload_adpcm(SPU_SCORE, tones::TRIANGLE);
    configure_sfx_voice(VOICE_WALL, SPU_WALL, Pitch::raw(0x1400));
    configure_sfx_voice(VOICE_PADDLE, SPU_PADDLE, Pitch::raw(0x0E00));
    configure_sfx_voice(VOICE_SCORE, SPU_SCORE, Pitch::raw(0x0800));

    // Font atlas for the scoreboard / banner.
    let font = FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT);

    // Initialise game state.
    reset_match();

    loop {
        let pad = poll_port1();

        update_game(pad);

        fb.clear(8, 12, 28);

        build_frame_ot();
        submit_frame_ot();

        draw_scoreboard(&font);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
    }
}

/// Configure a sample-playback voice: half volume on both channels,
/// percussive ADSR (short attack, fast release), fixed pitch.
/// The voice won't make noise until [`Voice::key_on`].
fn configure_sfx_voice(v: Voice, addr: SpuAddr, pitch: Pitch) {
    v.set_volume(Volume::HALF, Volume::HALF);
    v.set_pitch(pitch);
    v.set_start_addr(addr);
    v.set_adsr(Adsr::percussive());
}

/// Fire an SFX: the voice retriggers its ADSR attack, plays a
/// short burst, and fades out. Multiple calls in quick succession
/// just re-attack, so rapid hits still feel snappy.
fn play_sfx(v: Voice) {
    Voice::key_on(v.mask());
}

// ----------------------------------------------------------------------
// Game logic
// ----------------------------------------------------------------------

/// Reset both paddles, scores, and serve. Called on boot + after
/// each match ends and START is pressed.
fn reset_match() {
    // SAFETY: single-threaded access, `main` is the only caller
    // site chain.
    unsafe {
        GAME.p1_y = (SCREEN_H - PADDLE_H as i16) / 2;
        GAME.p2_y = (SCREEN_H - PADDLE_H as i16) / 2;
        GAME.p1_score = 0;
        GAME.p2_score = 0;
        GAME.winner = 0;
        GAME.serve_dir = 1;
        reset_ball();
    }
}

/// Put the ball back at centre aimed at the scorer's side.
/// `serve_dir` flips after each score so both players get the
/// ball equally.
fn reset_ball() {
    // SAFETY: same rationale as `reset_match`.
    unsafe {
        GAME.ball_x = (SCREEN_W - BALL_SIZE as i16) / 2;
        GAME.ball_y = (SCREEN_H - BALL_SIZE as i16) / 2;
        GAME.ball_vx = BALL_START_VX * GAME.serve_dir as i16;
        GAME.ball_vy = BALL_START_VY;
    }
}

/// One simulation tick. Reads pad state, writes game state, fires
/// SFX as appropriate.
fn update_game(pad: ButtonState) {
    // SAFETY: see `reset_match`.
    let g = unsafe { &mut GAME };

    // Post-match input: any START press resets.
    if g.winner != 0 {
        if pad.is_held(button::START) {
            reset_match();
        }
        return;
    }

    // Player paddle follows D-pad.
    if pad.is_held(button::UP) {
        g.p1_y -= PADDLE_SPEED;
    }
    if pad.is_held(button::DOWN) {
        g.p1_y += PADDLE_SPEED;
    }
    clamp_paddle(&mut g.p1_y);

    // AI paddle tracks the ball's Y with a small hysteresis band
    // so it doesn't micro-oscillate when roughly aligned.
    let target = g.ball_y + (BALL_SIZE as i16 / 2);
    let paddle_mid = g.p2_y + (PADDLE_H as i16 / 2);
    if target < paddle_mid - AI_HYSTERESIS {
        g.p2_y -= AI_SPEED;
    } else if target > paddle_mid + AI_HYSTERESIS {
        g.p2_y += AI_SPEED;
    }
    clamp_paddle(&mut g.p2_y);

    // Advance ball.
    g.ball_x += g.ball_vx;
    g.ball_y += g.ball_vy;

    // Wall (top / bottom) bounce.
    if g.ball_y <= PLAYFIELD_TOP {
        g.ball_y = PLAYFIELD_TOP;
        g.ball_vy = g.ball_vy.abs();
        play_sfx(VOICE_WALL);
    } else if g.ball_y + BALL_SIZE as i16 >= PLAYFIELD_BOT {
        g.ball_y = PLAYFIELD_BOT - BALL_SIZE as i16;
        g.ball_vy = -g.ball_vy.abs();
        play_sfx(VOICE_WALL);
    }

    // Paddle collision — left paddle, ball moving left.
    let left_face = PADDLE_MARGIN + PADDLE_W as i16;
    if g.ball_vx < 0
        && g.ball_x <= left_face
        && g.ball_x + BALL_SIZE as i16 >= PADDLE_MARGIN
        && g.ball_y + BALL_SIZE as i16 >= g.p1_y
        && g.ball_y <= g.p1_y + PADDLE_H as i16
    {
        g.ball_x = left_face;
        g.ball_vx = bounce_speed(-g.ball_vx);
        g.ball_vy = spin_from_paddle(g.ball_y, g.p1_y);
        play_sfx(VOICE_PADDLE);
    }

    // Paddle collision — right paddle.
    let right_face = SCREEN_W - PADDLE_MARGIN - PADDLE_W as i16;
    if g.ball_vx > 0
        && g.ball_x + BALL_SIZE as i16 >= right_face
        && g.ball_x <= SCREEN_W - PADDLE_MARGIN
        && g.ball_y + BALL_SIZE as i16 >= g.p2_y
        && g.ball_y <= g.p2_y + PADDLE_H as i16
    {
        g.ball_x = right_face - BALL_SIZE as i16;
        g.ball_vx = -bounce_speed(g.ball_vx);
        g.ball_vy = spin_from_paddle(g.ball_y, g.p2_y);
        play_sfx(VOICE_PADDLE);
    }

    // Score: ball fell off either end.
    if (g.ball_x + BALL_SIZE as i16) < 0 {
        g.p2_score = g.p2_score.saturating_add(1);
        g.serve_dir = -1;
        play_sfx(VOICE_SCORE);
        check_win_and_reset();
    } else if g.ball_x > SCREEN_W {
        g.p1_score = g.p1_score.saturating_add(1);
        g.serve_dir = 1;
        play_sfx(VOICE_SCORE);
        check_win_and_reset();
    }
}

fn check_win_and_reset() {
    // SAFETY: see `reset_match`.
    let g = unsafe { &mut GAME };
    if g.p1_score >= WIN_SCORE {
        g.winner = 1;
    } else if g.p2_score >= WIN_SCORE {
        g.winner = 2;
    }
    reset_ball();
}

/// Clamp a paddle's Y so it stays inside the playfield.
fn clamp_paddle(y: &mut i16) {
    let min = PLAYFIELD_TOP + 2;
    let max = PLAYFIELD_BOT - PADDLE_H as i16 - 2;
    if *y < min {
        *y = min;
    }
    if *y > max {
        *y = max;
    }
}

/// Slightly increase |vx| on each paddle bounce to ramp up rally
/// difficulty, up to [`BALL_MAX_SPEED`]. Sign is caller's
/// responsibility — this just returns the magnitude.
fn bounce_speed(mut speed: i16) -> i16 {
    if speed.abs() < BALL_MAX_SPEED {
        speed += if speed >= 0 { 0 } else { 0 };
        // Magnitude bump: sign-preserving add-1.
        if speed >= 0 {
            speed = (speed + 1).min(BALL_MAX_SPEED);
        } else {
            speed = (speed - 1).max(-BALL_MAX_SPEED);
        }
    }
    speed
}

/// Map the ball's vertical hit position on the paddle to a new vy.
/// Hit near the top third → bounce up; near the bottom → bounce
/// down; middle → no vertical change. Gives the player a way to
/// "aim" deflections.
fn spin_from_paddle(ball_y: i16, paddle_y: i16) -> i16 {
    let relative = (ball_y + BALL_SIZE as i16 / 2) - (paddle_y + PADDLE_H as i16 / 2);
    let h = PADDLE_H as i16 / 2;
    if relative < -h / 3 {
        -2
    } else if relative > h / 3 {
        2
    } else {
        // Keep the ball's existing vertical drift rather than
        // snapping to zero — a grazing hit shouldn't flatten it.
        // SAFETY: single-threaded.
        let vy = unsafe { GAME.ball_vy };
        if vy == 0 { 1 } else { vy.signum() }
    }
}

// ----------------------------------------------------------------------
// Rendering — OT layer
// ----------------------------------------------------------------------

/// Populate [`RECTS`] with the current frame's primitives and drop
/// them into the OT at the right depth slots. Slot 0 = back, 7 = front.
fn build_frame_ot() {
    // SAFETY: single-threaded access; OT + RECTS are the only
    // mutable statics touched.
    let ot = unsafe { &mut OT };
    let rects = unsafe { &mut RECTS };
    ot.clear();

    let g = unsafe { &GAME };

    // Slot 0 (farthest back) — top + bottom border strips.
    rects[0] = RectFlat::new(0, PLAYFIELD_TOP - 1, SCREEN_W as u16, BORDER_H, 140, 140, 180);
    rects[1] = RectFlat::new(
        0,
        PLAYFIELD_BOT,
        SCREEN_W as u16,
        BORDER_H,
        140,
        140,
        180,
    );
    // SAFETY: rects[n] is a valid pointer into the static array;
    // the OT walker follows DMA tags through bus addresses, which
    // stay stable across frames because `static mut` is fixed-
    // address.
    unsafe {
        ot.add(0, &mut rects[0], RectFlat::WORDS);
        ot.add(0, &mut rects[1], RectFlat::WORDS);
    }

    // Slot 2 — centre dashes. 10 short vertical segments of
    // 6×10 with 10-pixel gaps — a classic Pong look.
    let dash_h: u16 = 10;
    let dash_gap: i16 = 10;
    let dash_x: i16 = (SCREEN_W - 4) / 2;
    let mut y = PLAYFIELD_TOP + 6;
    let mut idx = 2;
    while y + dash_h as i16 <= PLAYFIELD_BOT - 4 && idx < 14 {
        rects[idx] = RectFlat::new(dash_x, y, 4, dash_h, 110, 110, 150);
        unsafe {
            ot.add(2, &mut rects[idx], RectFlat::WORDS);
        }
        y += dash_h as i16 + dash_gap;
        idx += 1;
    }

    // Slot 5 — paddles.
    rects[14] = RectFlat::new(
        PADDLE_MARGIN,
        g.p1_y,
        PADDLE_W,
        PADDLE_H,
        240,
        240,
        240,
    );
    rects[15] = RectFlat::new(
        SCREEN_W - PADDLE_MARGIN - PADDLE_W as i16,
        g.p2_y,
        PADDLE_W,
        PADDLE_H,
        240,
        240,
        240,
    );
    unsafe {
        ot.add(5, &mut rects[14], RectFlat::WORDS);
        ot.add(5, &mut rects[15], RectFlat::WORDS);
    }

    // Slot 7 (frontmost) — ball, tinted slightly warm so it reads
    // against the white paddles.
    rects[16] = RectFlat::new(
        g.ball_x,
        g.ball_y,
        BALL_SIZE,
        BALL_SIZE,
        255,
        220,
        120,
    );
    unsafe {
        ot.add(7, &mut rects[16], RectFlat::WORDS);
    }
}

/// Hand the OT to DMA channel 2 and wait for the walker to finish.
fn submit_frame_ot() {
    // SAFETY: `OT::submit` drives the DMA walker; it returns when
    // the chain's terminator tag is consumed.
    unsafe {
        OT.submit();
    }
}

// ----------------------------------------------------------------------
// Rendering — text layer (immediate-mode, on top of OT geometry)
// ----------------------------------------------------------------------

/// Draw scores at the top of the screen + game-over banner when
/// the match has ended.
fn draw_scoreboard(font: &FontAtlas) {
    let g = unsafe { &GAME };
    let p1 = digit_str(g.p1_score);
    let p2 = digit_str(g.p2_score);
    // Scores live at y=6, which is above the top border at y=27.
    font.draw_text(60, 6, p1.as_str(), (220, 220, 240));
    font.draw_text(SCREEN_W - 60 - 8, 6, p2.as_str(), (220, 220, 240));
    // Player labels.
    font.draw_text(24, 6, "P1", (140, 180, 255));
    font.draw_text(SCREEN_W - 24 - 8 * 2, 6, "P2", (255, 180, 140));

    if g.winner != 0 {
        let msg = if g.winner == 1 { "P1 WINS!" } else { "P2 WINS!" };
        // Centre in the playfield. 8 chars × 8 px = 64 px wide.
        font.draw_text(
            (SCREEN_W - 8 * 8) / 2,
            (SCREEN_H - 16) / 2,
            msg,
            (255, 230, 120),
        );
        font.draw_text(
            (SCREEN_W - 17 * 8) / 2,
            (SCREEN_H - 16) / 2 + 24,
            "START to play again",
            (160, 160, 180),
        );
    }
}

/// Render a 0-9 score as a single-digit string in a stack buffer.
/// Scores above 9 are rare in Pong-to-7 but clamp to 'X' for
/// safety.
fn digit_str(n: u8) -> DigitStr {
    let c = if n < 10 {
        b'0' + n
    } else {
        b'X'
    };
    DigitStr([c])
}

struct DigitStr([u8; 1]);
impl DigitStr {
    fn as_str(&self) -> &str {
        // SAFETY: the single byte is always ASCII '0'..='9' or 'X'.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}
