//! Host-gamepad input using `gilrs`.
//!
//! Polls the first connected gamepad each frame and produces a PSX
//! digital-pad bitmask that maps directly onto the emulator-core's
//! `button::` constants (bit set = held). Left stick doubles as a
//! D-pad proxy for games that don't support analog, with a 0.3
//! deadzone. Analog triggers are mapped to L2 / R2 with a 0.5
//! activation threshold.
//!
//! Port of `psoXide/emulator/player/src/gamepad.rs`, adapted to use
//! our active-high pad mask instead of the wire-protocol active-low
//! value. Keeps the same dual-role (XMB navigation + in-game input)
//! but simplified to one public surface — the shell merges this
//! into its keyboard-derived mask via OR each frame.

use emulator_core::button;
use gilrs::{Axis, Button, Gilrs};

/// Left-stick deadzone for the D-pad proxy path.
const STICK_DEADZONE: f32 = 0.3;

/// Threshold beyond which an analog trigger (`LeftZ` / `RightZ`)
/// counts as an L2 / R2 press.
const TRIGGER_THRESHOLD: f32 = 0.5;

/// Host gamepad facade. Owns the gilrs context (which holds OS
/// handles to connected pads) and exposes a single `poll_buttons`
/// entry point returning the current PSX button bitmask.
///
/// A failed gilrs init (headless / missing driver / locked-down
/// Linux sandbox) is non-fatal — `poll_buttons` just returns 0
/// (nothing pressed) so the shell still runs on keyboard input.
pub struct Gamepad {
    gilrs: Option<Gilrs>,
}

impl Gamepad {
    /// Try to initialise gilrs. Returns a `Gamepad` either way —
    /// if init failed, subsequent calls just no-op.
    pub fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("[gamepad] gilrs init failed: {e} — keyboard only");
                None
            }
        };
        Self { gilrs }
    }

    /// Drain pending gilrs events and snapshot the current
    /// button mask for the first connected pad. Returns 0 when
    /// no pad is available. Must be called once per frame.
    ///
    /// The output uses the emulator-core's active-high button
    /// layout (bit N set = button N held), so the shell can OR
    /// it into its keyboard mask with no conversion.
    pub fn poll_buttons(&mut self) -> u16 {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return 0;
        };
        while let Some(_event) = gilrs.next_event() {}

        let Some((_, gp)) = gilrs.gamepads().next() else {
            return 0;
        };

        let mut mask: u16 = 0;

        // Face buttons — Xbox layout mapped to DualShock:
        //   South (A / bottom) → Cross
        //   East  (B / right)  → Circle
        //   West  (X / left)   → Square
        //   North (Y / top)    → Triangle
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

        // Shoulders + triggers.
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

        // Left stick as D-pad proxy — useful for games that treat
        // the pad as digital regardless of DualShock capability.
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

    /// True when any gamepad is currently reachable. Used by the
    /// HUD to show a plugged-in indicator.
    pub fn is_connected(&self) -> bool {
        self.gilrs
            .as_ref()
            .and_then(|g| g.gamepads().next())
            .is_some()
    }

    /// Name of the first connected pad — empty string when none.
    /// Diagnostic; surfaced via the settings panel.
    #[allow(dead_code)]
    pub fn name(&self) -> String {
        self.gilrs
            .as_ref()
            .and_then(|g| g.gamepads().next())
            .map(|(_, gp)| gp.name().to_string())
            .unwrap_or_default()
    }

    /// Snapshot the left-stick position, both axes in -1.0..=1.0.
    /// Used for analog DualShock voices; shell scales to the
    /// PSX byte range (`0x80` centre, 0..=0xFF).
    pub fn left_stick(&mut self) -> (f32, f32) {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return (0.0, 0.0);
        };
        let Some((_, gp)) = gilrs.gamepads().next() else {
            return (0.0, 0.0);
        };
        (gp.value(Axis::LeftStickX), gp.value(Axis::LeftStickY))
    }

    /// Snapshot the right-stick position, both axes in -1.0..=1.0.
    pub fn right_stick(&mut self) -> (f32, f32) {
        let Some(gilrs) = self.gilrs.as_mut() else {
            return (0.0, 0.0);
        };
        let Some((_, gp)) = gilrs.gamepads().next() else {
            return (0.0, 0.0);
        };
        (gp.value(Axis::RightStickX), gp.value(Axis::RightStickY))
    }
}
