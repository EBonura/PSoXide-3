//! `game-pong` — classic two-paddle Pong, first full-game port to
//! the `psx-engine` framework.
//!
//! Same game it's always been (left paddle = D-pad, right paddle =
//! AI tracking with a hysteresis band, first to 7 wins, START
//! resets after a match ends). The only change is *how it's wired*:
//!
//! - No hand-rolled main loop. [`App::run`] drives the cadence:
//!   poll-pad → update → clear → render → draw-sync → vsync → swap.
//! - [`Scene::update`] and [`Scene::render`] split the per-frame
//!   work, both receive `&mut Ctx`.
//! - Pad edge detection uses [`Ctx::just_pressed`] instead of
//!   tracking previous-frame state by hand.
//! - Game state is a struct field on the scene rather than a
//!   `static mut GAME` global — no more `unsafe { &mut GAME }`
//!   laced through every logic function.
//!
//! What DOESN'T move out of static: the OT + the rect arena. Those
//! are DMA targets; the walker needs fixed bus addresses across
//! frames, so a `.bss`-resident static array is still the right
//! home for them.
//!
//! Original: `sdk/examples/game-pong/` — ~540 lines with its own
//! main, its own pad polling, its own loop structure. Under the
//! engine it's ~480 lines of almost-entirely-game-logic code;
//! roughly 60 lines of boilerplate (setup, loop, unsafe static
//! shuffling) collapse into the engine.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_engine::{App, Config, Ctx, Scene, button};
use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::RectFlat;
use psx_spu::{self as spu, Adsr, Pitch, SpuAddr, Voice, Volume, tones};
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Screen + gameplay constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

const BORDER_H: u16 = 3;
const PLAYFIELD_TOP: i16 = 28;
const PLAYFIELD_BOT: i16 = SCREEN_H - BORDER_H as i16;

const PADDLE_W: u16 = 6;
const PADDLE_H: u16 = 36;
const PADDLE_MARGIN: i16 = 12;
const PADDLE_SPEED: i16 = 3;

const AI_SPEED: i16 = 2;
const AI_HYSTERESIS: i16 = 3;

const BALL_SIZE: u16 = 6;
const BALL_START_VX: i16 = 2;
const BALL_START_VY: i16 = 1;
const BALL_MAX_SPEED: i16 = 4;

const WIN_SCORE: u8 = 7;

// ----------------------------------------------------------------------
// VRAM layout
// ----------------------------------------------------------------------

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
// DMA-facing arenas — stay in .bss so the OT walker sees stable
// addresses frame-to-frame.
// ----------------------------------------------------------------------

static mut OT: OrderingTable<8> = OrderingTable::new();
static mut RECTS: [RectFlat; 20] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 20];

// ----------------------------------------------------------------------
// Game state — now a Scene field rather than a static
// ----------------------------------------------------------------------

struct Pong {
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
    /// Font atlas handle, populated in `init`. `Option` because
    /// `FontAtlas` isn't `Default` and we don't have it at struct
    /// construction time.
    font: Option<FontAtlas>,
}

impl Pong {
    const fn new() -> Self {
        Self {
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
            font: None,
        }
    }

    /// Reset both paddles, scores, and serve. Called on boot + after
    /// each match ends and START is pressed.
    fn reset_match(&mut self) {
        self.p1_y = (SCREEN_H - PADDLE_H as i16) / 2;
        self.p2_y = (SCREEN_H - PADDLE_H as i16) / 2;
        self.p1_score = 0;
        self.p2_score = 0;
        self.winner = 0;
        self.serve_dir = 1;
        self.reset_ball();
    }

    /// Put the ball back at centre aimed at the scorer's side.
    fn reset_ball(&mut self) {
        self.ball_x = (SCREEN_W - BALL_SIZE as i16) / 2;
        self.ball_y = (SCREEN_H - BALL_SIZE as i16) / 2;
        self.ball_vx = BALL_START_VX * self.serve_dir as i16;
        self.ball_vy = BALL_START_VY;
    }

    /// Decide win condition, advance to post-match state or reset
    /// the ball for the next rally.
    fn check_win_and_reset(&mut self) {
        if self.p1_score >= WIN_SCORE {
            self.winner = 1;
        } else if self.p2_score >= WIN_SCORE {
            self.winner = 2;
        }
        self.reset_ball();
    }
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
/// difficulty, up to [`BALL_MAX_SPEED`]. Sign-preserving magnitude
/// bump; caller decides the final sign.
fn bounce_speed(mut speed: i16) -> i16 {
    if speed.abs() < BALL_MAX_SPEED {
        if speed >= 0 {
            speed = (speed + 1).min(BALL_MAX_SPEED);
        } else {
            speed = (speed - 1).max(-BALL_MAX_SPEED);
        }
    }
    speed
}

/// Map ball-vs-paddle hit offset to a new vy so the player can
/// "aim" bounces. Top third → up; bottom third → down; middle →
/// preserve existing drift.
fn spin_from_paddle(ball_y: i16, paddle_y: i16, prev_vy: i16) -> i16 {
    let relative = (ball_y + BALL_SIZE as i16 / 2) - (paddle_y + PADDLE_H as i16 / 2);
    let h = PADDLE_H as i16 / 2;
    if relative < -h / 3 {
        -2
    } else if relative > h / 3 {
        2
    } else if prev_vy == 0 {
        1
    } else {
        prev_vy.signum()
    }
}

// ----------------------------------------------------------------------
// SFX helpers
// ----------------------------------------------------------------------

/// Configure a sample-playback voice: half volume, percussive
/// ADSR, fixed pitch. No sound until `Voice::key_on`.
fn configure_sfx_voice(v: Voice, addr: SpuAddr, pitch: Pitch) {
    v.set_volume(Volume::HALF, Volume::HALF);
    v.set_pitch(pitch);
    v.set_start_addr(addr);
    v.set_adsr(Adsr::percussive());
}

/// Fire an SFX — re-attacks the voice's ADSR.
fn play_sfx(v: Voice) {
    Voice::key_on(v.mask());
}

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Pong {
    fn init(&mut self, _ctx: &mut Ctx) {
        spu::init();
        spu::upload_adpcm(SPU_WALL, tones::SINE);
        spu::upload_adpcm(SPU_PADDLE, tones::SQUARE);
        spu::upload_adpcm(SPU_SCORE, tones::TRIANGLE);
        configure_sfx_voice(VOICE_WALL, SPU_WALL, Pitch::raw(0x1400));
        configure_sfx_voice(VOICE_PADDLE, SPU_PADDLE, Pitch::raw(0x0E00));
        configure_sfx_voice(VOICE_SCORE, SPU_SCORE, Pitch::raw(0x0800));

        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));

        self.reset_match();
    }

    fn update(&mut self, ctx: &mut Ctx) {
        // Post-match: any START press resets.
        if self.winner != 0 {
            if ctx.just_pressed(button::START) {
                self.reset_match();
            }
            return;
        }

        // Player paddle follows D-pad (held, so motion is smooth).
        if ctx.is_held(button::UP) {
            self.p1_y -= PADDLE_SPEED;
        }
        if ctx.is_held(button::DOWN) {
            self.p1_y += PADDLE_SPEED;
        }
        clamp_paddle(&mut self.p1_y);

        // AI paddle tracks ball Y with hysteresis so it doesn't
        // micro-oscillate when aligned.
        let target = self.ball_y + (BALL_SIZE as i16 / 2);
        let paddle_mid = self.p2_y + (PADDLE_H as i16 / 2);
        if target < paddle_mid - AI_HYSTERESIS {
            self.p2_y -= AI_SPEED;
        } else if target > paddle_mid + AI_HYSTERESIS {
            self.p2_y += AI_SPEED;
        }
        clamp_paddle(&mut self.p2_y);

        // Advance ball.
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        // Wall (top / bottom) bounce.
        if self.ball_y <= PLAYFIELD_TOP {
            self.ball_y = PLAYFIELD_TOP;
            self.ball_vy = self.ball_vy.abs();
            play_sfx(VOICE_WALL);
        } else if self.ball_y + BALL_SIZE as i16 >= PLAYFIELD_BOT {
            self.ball_y = PLAYFIELD_BOT - BALL_SIZE as i16;
            self.ball_vy = -self.ball_vy.abs();
            play_sfx(VOICE_WALL);
        }

        // Left paddle bounce (ball moving left).
        let left_face = PADDLE_MARGIN + PADDLE_W as i16;
        if self.ball_vx < 0
            && self.ball_x <= left_face
            && self.ball_x + BALL_SIZE as i16 >= PADDLE_MARGIN
            && self.ball_y + BALL_SIZE as i16 >= self.p1_y
            && self.ball_y <= self.p1_y + PADDLE_H as i16
        {
            self.ball_x = left_face;
            self.ball_vx = bounce_speed(-self.ball_vx);
            self.ball_vy = spin_from_paddle(self.ball_y, self.p1_y, self.ball_vy);
            play_sfx(VOICE_PADDLE);
        }

        // Right paddle bounce.
        let right_face = SCREEN_W - PADDLE_MARGIN - PADDLE_W as i16;
        if self.ball_vx > 0
            && self.ball_x + BALL_SIZE as i16 >= right_face
            && self.ball_x <= SCREEN_W - PADDLE_MARGIN
            && self.ball_y + BALL_SIZE as i16 >= self.p2_y
            && self.ball_y <= self.p2_y + PADDLE_H as i16
        {
            self.ball_x = right_face - BALL_SIZE as i16;
            self.ball_vx = -bounce_speed(self.ball_vx);
            self.ball_vy = spin_from_paddle(self.ball_y, self.p2_y, self.ball_vy);
            play_sfx(VOICE_PADDLE);
        }

        // Score: ball fell off either end.
        if (self.ball_x + BALL_SIZE as i16) < 0 {
            self.p2_score = self.p2_score.saturating_add(1);
            self.serve_dir = -1;
            play_sfx(VOICE_SCORE);
            self.check_win_and_reset();
        } else if self.ball_x > SCREEN_W {
            self.p1_score = self.p1_score.saturating_add(1);
            self.serve_dir = 1;
            play_sfx(VOICE_SCORE);
            self.check_win_and_reset();
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        self.build_frame_ot();
        self.submit_frame_ot();
        self.draw_scoreboard();
    }
}

impl Pong {
    /// Populate RECTS with the current frame's primitives and drop
    /// them into the OT at the right depth slots.
    fn build_frame_ot(&self) {
        let ot = unsafe { &mut OT };
        let rects = unsafe { &mut RECTS };
        ot.clear();

        // Slot 0 (back) — top + bottom border strips.
        rects[0] = RectFlat::new(0, PLAYFIELD_TOP - 1, SCREEN_W as u16, BORDER_H, 140, 140, 180);
        rects[1] = RectFlat::new(0, PLAYFIELD_BOT, SCREEN_W as u16, BORDER_H, 140, 140, 180);
        ot.add(0, &mut rects[0], RectFlat::WORDS);
        ot.add(0, &mut rects[1], RectFlat::WORDS);

        // Slot 2 — centre dashes.
        let dash_h: u16 = 10;
        let dash_gap: i16 = 10;
        let dash_x: i16 = (SCREEN_W - 4) / 2;
        let mut y = PLAYFIELD_TOP + 6;
        let mut idx = 2;
        while y + dash_h as i16 <= PLAYFIELD_BOT - 4 && idx < 14 {
            rects[idx] = RectFlat::new(dash_x, y, 4, dash_h, 110, 110, 150);
            ot.add(2, &mut rects[idx], RectFlat::WORDS);
            y += dash_h as i16 + dash_gap;
            idx += 1;
        }

        // Slot 5 — paddles.
        rects[14] = RectFlat::new(PADDLE_MARGIN, self.p1_y, PADDLE_W, PADDLE_H, 240, 240, 240);
        rects[15] = RectFlat::new(
            SCREEN_W - PADDLE_MARGIN - PADDLE_W as i16,
            self.p2_y,
            PADDLE_W,
            PADDLE_H,
            240,
            240,
            240,
        );
        ot.add(5, &mut rects[14], RectFlat::WORDS);
        ot.add(5, &mut rects[15], RectFlat::WORDS);

        // Slot 7 (front) — ball, warm tint to read against white paddles.
        rects[16] = RectFlat::new(
            self.ball_x,
            self.ball_y,
            BALL_SIZE,
            BALL_SIZE,
            255,
            220,
            120,
        );
        ot.add(7, &mut rects[16], RectFlat::WORDS);
    }

    fn submit_frame_ot(&self) {
        unsafe { OT.submit() };
    }

    /// Scoreboard + game-over banner. Immediate-mode on top of OT.
    fn draw_scoreboard(&self) {
        let Some(font) = self.font.as_ref() else { return };
        let p1 = digit_str(self.p1_score);
        let p2 = digit_str(self.p2_score);
        font.draw_text(60, 6, p1.as_str(), (220, 220, 240));
        font.draw_text(SCREEN_W - 60 - 8, 6, p2.as_str(), (220, 220, 240));
        font.draw_text(24, 6, "P1", (140, 180, 255));
        font.draw_text(SCREEN_W - 24 - 8 * 2, 6, "P2", (255, 180, 140));

        if self.winner != 0 {
            let msg = if self.winner == 1 { "P1 WINS!" } else { "P2 WINS!" };
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
}

/// Render a 0-9 score as a single-digit string in a stack buffer.
fn digit_str(n: u8) -> DigitStr {
    let c = if n < 10 { b'0' + n } else { b'X' };
    DigitStr([c])
}

struct DigitStr([u8; 1]);
impl DigitStr {
    fn as_str(&self) -> &str {
        // SAFETY: byte is always ASCII '0'..='9' or 'X'.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    // Clear colour matches the original's dim navy background.
    let config = Config {
        clear_color: (8, 12, 28),
        ..Config::default()
    };
    let mut game = Pong::new();
    App::run(config, &mut game);
}
