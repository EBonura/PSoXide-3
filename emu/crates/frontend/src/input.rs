//! Host-input router.
//!
//! Owns the single [`Gilrs`] context and tracks every connected
//! gamepad by `GamepadId`. Each frame the shell calls [`poll`]
//! once; the router drains pending events, logs
//! connect / disconnect transitions (so the user can see why a
//! freshly-paired Bluetooth pad isn't recognised), merges every
//! pad's button state into one PSX pad-1 mask, and detects the
//! **Select+Start chord** that opens the Menu overlay.
//!
//! # Multi-pad policy
//!
//! All connected pads OR into port 1. Two users pressing the same
//! button on different pads looks like one user holding it —
//! acceptable for the "anything plugged in just works" era of PS1
//! homebrew. The router's state is keyed by `GamepadId`, so a
//! future settings panel can peel this apart into `port[0]` /
//! `port[1]` without reshaping the pipeline.
//!
//! # Why an explicit tracked-pad set
//!
//! The pre-router code called `gilrs.gamepads().next()` every
//! frame — "whatever comes first in gilrs's internal map". That's
//! fine with one pad plugged in, but a disconnect-then-reconnect
//! can flip iteration order and the game silently starts reading
//! a different device. Worse: the only time events were drained
//! was inside the game-running path, so a BT controller paired
//! while the Menu was open never got registered until the user
//! launched a game.
//!
//! This router:
//! 1. Polls every frame regardless of run state, so hot-plugged
//!    pads show up immediately.
//! 2. Logs every Connected / Disconnected so the user can see
//!    what gilrs actually sees.
//! 3. Keeps its own `HashMap<GamepadId, TrackedPad>` so pad
//!    identity is stable across reorderings and iteration is
//!    deterministic.
//!
//! # Chord semantics
//!
//! `Select + Start` = open / close the Menu. Matches the
//! PS1-era "player holds both shoulders" convention that shipped
//! in everything from game input and launcher navigation.
//!
//! The chord fires on the **rising edge**: both bits must go from
//! "not-both-held last frame" to "both-held this frame". Subsequent
//! frames with the chord still held do nothing — so the menu
//! doesn't flicker open/closed while the user is lazy about
//! releasing.
//!
//! When the chord fires, the individual SELECT + START bits are
//! **stripped** from `pad1_mask` for that frame. Otherwise games
//! with their own "Select + Start = [special action]" handlers
//! would also see the combination and fire the in-game action the
//! moment the menu opens.
//!
//! The DualShock Analog button is routed separately from the
//! standard button mask because PS1 software does not see it as a
//! normal held button. It toggles the controller's Digital/Analog
//! response mode instead.

use emulator_core::button;
use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs};
use std::collections::HashMap;

/// Left-stick deadzone for the D-pad proxy path.
const STICK_DEADZONE: f32 = 0.3;

/// Analog trigger activation threshold.
const TRIGGER_THRESHOLD: f32 = 0.5;

/// The Select+Start combination, precomputed.
const CHORD_MASK: u16 = button::SELECT | button::START;

/// Gamepad hotplug notification surfaced to the frontend toast.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputNotice {
    /// A pad became available to gilrs.
    Connected {
        /// User-facing controller name reported by gilrs.
        name: String,
        /// Mapping source summary, e.g. SDL database vs raw HID.
        mapping: String,
    },
    /// A previously-tracked pad disappeared.
    Disconnected {
        /// User-facing controller name captured before disconnect.
        name: String,
    },
}

impl InputNotice {
    /// Short status-line text for the frontend toast.
    pub fn message(&self) -> String {
        match self {
            Self::Connected { name, mapping } => format!("Gamepad connected: {name} ({mapping})"),
            Self::Disconnected { name } => format!("Gamepad disconnected: {name}"),
        }
    }
}

/// Per-frame summary returned by [`InputRouter::poll`].
///
/// The shell consumes this to drive both the PSX pad input and
/// Menu menu navigation in a single call — separate fields rather
/// than one blob so each consumer's needs are obvious at the call
/// site.
#[derive(Clone, Debug, Default)]
pub struct InputFrame {
    /// Hotplug notices drained this frame.
    pub notices: Vec<InputNotice>,

    /// Merged PSX pad-1 bitmask across every connected pad. The
    /// Select + Start bits are masked out during frames when the
    /// chord fires, so in-game Select+Start handlers don't see a
    /// phantom combo when the user is actually opening the menu.
    pub pad1_mask: u16,

    /// Rising-edge of Select+Start on any pad — goes `true` for
    /// exactly one frame after both become held. The shell wires
    /// this into `MenuInput.toggle_open`, reusing the same path
    /// Escape goes through.
    pub toggle_menu: bool,

    /// Rising-edge of the host pad's Analog/Mode button. The shell
    /// turns this into a DualShock mode toggle rather than a normal
    /// PSX button press.
    pub analog_button: bool,

    /// Menu-navigation edges: `true` the frame a direction just
    /// became held on any pad. Single-press semantics so holding
    /// the D-pad doesn't spam nav events (cf. keyboard's OS-level
    /// repeat, which we wouldn't want to imitate on a stick).
    pub menu_up: bool,
    /// See [`Self::menu_up`].
    pub menu_down: bool,
    /// See [`Self::menu_up`].
    pub menu_left: bool,
    /// See [`Self::menu_up`].
    pub menu_right: bool,
    /// Rising-edge of Cross — the Menu's "enter / confirm".
    pub menu_confirm: bool,
    /// Rising-edge of Circle — the Menu's "back / cancel".
    pub menu_back: bool,

    /// Left stick on the first currently-connected pad, in
    /// −1.0..=1.0 per axis. Zero when no pad is plugged in.
    pub left_stick: (f32, f32),
    /// Right stick on the first currently-connected pad.
    pub right_stick: (f32, f32),
}

/// One tracked pad. Name is captured at connect time so we can
/// still identify the pad in a disconnect log message after gilrs
/// has already dropped the handle.
struct TrackedPad {
    name: String,
}

/// Central input router. One per shell; polled once per frame.
pub struct InputRouter {
    /// `None` when gilrs init failed (headless / locked-down
    /// Linux / missing permissions). Keyboard still works.
    gilrs: Option<Gilrs>,
    /// Every pad gilrs has told us about, still connected.
    /// Updated on `Connected` / `Disconnected` events.
    pads: HashMap<GamepadId, TrackedPad>,
    /// Previous frame's merged mask — used for rising-edge
    /// detection on both the chord and the menu-nav buttons.
    prev_mask: u16,
    /// Previous frame's host Analog/Mode button state.
    prev_analog_button: bool,
    /// Hotplug notices generated during startup before the first
    /// frame could display them.
    pending_notices: Vec<InputNotice>,
}

impl InputRouter {
    /// Initialise gilrs and enumerate any pads already known at
    /// startup. A failed init is non-fatal — subsequent polls
    /// return an empty [`InputFrame`] and the shell happily runs
    /// on keyboard alone.
    pub fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => {
                // Log every pad gilrs already enumerated at this
                // point — the "did it even see my controller?"
                // question users will ask when BT pads go missing.
                for (id, gp) in g.gamepads() {
                    eprintln!(
                        "[input] init: id={id:?} name={:?} connected={} mapping={}",
                        gp.name(),
                        gp.is_connected(),
                        if gp.mapping_source() == gilrs::MappingSource::None {
                            "none (raw HID — face buttons may not map)"
                        } else {
                            "SDL_GameControllerDB"
                        },
                    );
                }
                Some(g)
            }
            Err(e) => {
                eprintln!("[input] gilrs init failed: {e} — keyboard only");
                None
            }
        };

        // Pre-seed the tracked-pad map so `is_connected` is honest
        // from frame zero rather than waiting for the first `poll`.
        let mut pads: HashMap<GamepadId, TrackedPad> = HashMap::new();
        let mut pending_notices = Vec::new();
        if let Some(g) = gilrs.as_ref() {
            for (id, gp) in g.gamepads() {
                let mapping = if gp.mapping_source() == gilrs::MappingSource::None {
                    "raw HID".to_string()
                } else {
                    "SDL mapping".to_string()
                };
                pending_notices.push(InputNotice::Connected {
                    name: gp.name().to_string(),
                    mapping,
                });
                pads.insert(
                    id,
                    TrackedPad {
                        name: gp.name().to_string(),
                    },
                );
            }
        }

        Self {
            gilrs,
            pads,
            prev_mask: 0,
            prev_analog_button: false,
            pending_notices,
        }
    }

    /// Drain pending events, recompute the merged state, and
    /// return the [`InputFrame`] for this tick. Must be called
    /// once per frame — call it even when the game is paused,
    /// otherwise gilrs's event queue never drains and hot-plugged
    /// pads go un-noticed.
    pub fn poll(&mut self) -> InputFrame {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return InputFrame::default();
        };
        let mut notices = std::mem::take(&mut self.pending_notices);

        // 1. Drain events, maintaining our connected-pad set.
        while let Some(Event { id, event, .. }) = gilrs.next_event() {
            match event {
                EventType::Connected => {
                    let gp = gilrs.gamepad(id);
                    let name = gp.name().to_string();
                    let mapping = if gp.mapping_source() == gilrs::MappingSource::None {
                        "none (raw HID — face buttons may not map)"
                    } else {
                        "SDL_GameControllerDB"
                    };
                    eprintln!("[input] connected: id={id:?} name={name:?} mapping={mapping}",);
                    notices.push(InputNotice::Connected {
                        name: name.clone(),
                        mapping: mapping.to_string(),
                    });
                    self.pads.insert(id, TrackedPad { name });
                }
                EventType::Disconnected => {
                    if let Some(p) = self.pads.remove(&id) {
                        eprintln!("[input] disconnected: id={id:?} name={:?}", p.name);
                        notices.push(InputNotice::Disconnected { name: p.name });
                    }
                }
                // Other events (button + axis state) are handled
                // implicitly by re-reading `gp.is_pressed` /
                // `gp.value` below — no per-event bookkeeping.
                _ => {}
            }
        }

        // 2. Sample every connected pad; OR into the merged mask.
        //    Record the first connected pad's sticks for the
        //    caller — analog routing is single-player even when
        //    multiple pads contribute to port 1's digital mask.
        let mut mask: u16 = 0;
        let mut first_sticks: Option<((f32, f32), (f32, f32))> = None;
        let mut analog_down = false;

        // Iterate through our *own* tracked set instead of
        // `gilrs.gamepads()` — deterministic order across frames
        // (HashMap iteration order is randomised per-Map, but
        // stable within one HashMap's lifetime when unchanged).
        for id in self.pads.keys() {
            let gp = gilrs.gamepad(*id);
            if !gp.is_connected() {
                continue;
            }
            mask |= sample_pad(&gp);
            analog_down |= gp.is_pressed(Button::Mode);
            if first_sticks.is_none() {
                first_sticks = Some((
                    (gp.value(Axis::LeftStickX), gp.value(Axis::LeftStickY)),
                    (gp.value(Axis::RightStickX), gp.value(Axis::RightStickY)),
                ));
            }
        }

        // 3. Edge detection. We compute edges *before* stripping
        //    the chord, so `menu_confirm` / `menu_back` pick up on
        //    Cross / Circle normally, but the chord logic sees
        //    the raw combination too.
        let edge = mask & !self.prev_mask;
        let chord_active = (mask & CHORD_MASK) == CHORD_MASK;
        let chord_was_active = (self.prev_mask & CHORD_MASK) == CHORD_MASK;
        let toggle_menu = chord_active && !chord_was_active;
        let analog_button = analog_down && !self.prev_analog_button;

        // 4. Build the frame.
        //    Menu-nav edges are derived from the raw merged mask
        //    *before* we mask out the chord bits — pressing just
        //    Select (without Start) should still behave like the
        //    pad sees it (no menu_* flag fires for Select alone,
        //    since CHORD_MASK bits aren't in any menu_* field).
        let frame = InputFrame {
            notices,
            pad1_mask: if chord_active {
                // Swallow the chord so in-game Select+Start
                // handlers don't see the menu-open combination.
                mask & !CHORD_MASK
            } else {
                mask
            },
            toggle_menu,
            analog_button,
            menu_up: (edge & button::UP) != 0,
            menu_down: (edge & button::DOWN) != 0,
            menu_left: (edge & button::LEFT) != 0,
            menu_right: (edge & button::RIGHT) != 0,
            menu_confirm: (edge & button::CROSS) != 0,
            menu_back: (edge & button::CIRCLE) != 0,
            left_stick: first_sticks.map(|(l, _)| l).unwrap_or((0.0, 0.0)),
            right_stick: first_sticks.map(|(_, r)| r).unwrap_or((0.0, 0.0)),
        };

        // 5. Stash for next frame's edge math. We store the raw
        //    merged mask (chord bits included) so the chord's
        //    rising-edge math works — stashing the chord-masked
        //    value would make the chord re-fire every frame it
        //    stayed held.
        self.prev_mask = mask;
        self.prev_analog_button = analog_down;

        frame
    }

    /// `true` when any pad is currently tracked. Used by the HUD
    /// + startup banner to show a "pad connected" indicator.
    pub fn is_connected(&self) -> bool {
        !self.pads.is_empty()
    }

    /// Comma-separated list of currently-connected pad names.
    /// Empty string when nothing is plugged in. Diagnostic —
    /// surfaced in the startup banner and (eventually) the
    /// settings panel.
    pub fn connected_names(&self) -> String {
        let mut names: Vec<&str> = self.pads.values().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        names.join(", ")
    }
}

/// Sample one pad's PSX button mask. Extracted so the mapping
/// table is in exactly one place — no ambiguity about which
/// flavour of `is_pressed` the router uses vs what a hypothetical
/// "preview current pad" feature might do.
///
/// Xbox layout → DualShock:
///   South (A / bottom) → Cross
///   East  (B / right)  → Circle
///   West  (X / left)   → Square
///   North (Y / top)    → Triangle
fn sample_pad(gp: &gilrs::Gamepad<'_>) -> u16 {
    let mut mask: u16 = 0;

    if gp.is_pressed(Button::South) {
        mask |= button::CROSS;
    }
    if gp.is_pressed(Button::East) {
        mask |= button::CIRCLE;
    }
    if gp.is_pressed(Button::West) {
        mask |= button::SQUARE;
    }
    if gp.is_pressed(Button::North) {
        mask |= button::TRIANGLE;
    }

    // Shoulders + triggers. Analog-trigger fallback covers pads
    // whose drivers expose the triggers as `LeftZ` / `RightZ`
    // axes instead of discrete buttons (macOS + some Xbox BT
    // firmwares behave this way).
    if gp.is_pressed(Button::LeftTrigger) {
        mask |= button::L1;
    }
    if gp.is_pressed(Button::RightTrigger) {
        mask |= button::R1;
    }
    if gp.is_pressed(Button::LeftTrigger2) || gp.value(Axis::LeftZ) > TRIGGER_THRESHOLD {
        mask |= button::L2;
    }
    if gp.is_pressed(Button::RightTrigger2) || gp.value(Axis::RightZ) > TRIGGER_THRESHOLD {
        mask |= button::R2;
    }
    if gp.is_pressed(Button::LeftThumb) {
        mask |= button::L3;
    }
    if gp.is_pressed(Button::RightThumb) {
        mask |= button::R3;
    }

    // D-pad.
    if gp.is_pressed(Button::DPadUp) {
        mask |= button::UP;
    }
    if gp.is_pressed(Button::DPadDown) {
        mask |= button::DOWN;
    }
    if gp.is_pressed(Button::DPadLeft) {
        mask |= button::LEFT;
    }
    if gp.is_pressed(Button::DPadRight) {
        mask |= button::RIGHT;
    }

    // Left stick as D-pad proxy — useful for both games that
    // treat the pad as digital regardless of DualShock capability
    // AND for Menu nav, which needs directional edges.
    let lx = gp.value(Axis::LeftStickX);
    let ly = gp.value(Axis::LeftStickY);
    if ly > STICK_DEADZONE {
        mask |= button::UP;
    }
    if ly < -STICK_DEADZONE {
        mask |= button::DOWN;
    }
    if lx < -STICK_DEADZONE {
        mask |= button::LEFT;
    }
    if lx > STICK_DEADZONE {
        mask |= button::RIGHT;
    }

    // Select / Start.
    if gp.is_pressed(Button::Select) {
        mask |= button::SELECT;
    }
    if gp.is_pressed(Button::Start) {
        mask |= button::START;
    }

    mask
}
