//! Embedded editor play-mode state machine.
//!
//! The frontend owns the side effects (cook/build/load/stop). This
//! module owns the phase transitions so play-mode behaviour can be
//! tested without spawning a process or touching emulator state.

use std::process::Child;

use psxed_ui::EditorPlaytestStatus;

/// Pure play-mode phase mirrored to the editor UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddedPlaytestPhase {
    /// No embedded playtest is active.
    #[default]
    Idle,
    /// `make build-editor-playtest` is running in the background.
    Building,
    /// The editor viewport is displaying the live emulator.
    Running {
        /// Whether keyboard/gamepad input is captured by the game.
        input_captured: bool,
    },
    /// Last play attempt failed.
    Failed,
}

impl EmbeddedPlaytestPhase {
    /// Editor-facing status mirror.
    pub const fn editor_status(self) -> EditorPlaytestStatus {
        match self {
            Self::Idle => EditorPlaytestStatus::Idle,
            Self::Building => EditorPlaytestStatus::Building,
            Self::Running { input_captured } => EditorPlaytestStatus::Running { input_captured },
            Self::Failed => EditorPlaytestStatus::Failed,
        }
    }

    /// True when the editor viewport should show the live game.
    pub const fn is_running(self) -> bool {
        matches!(self, Self::Running { .. })
    }

    /// True when editor shortcuts should give way to game input.
    pub const fn input_captured(self) -> bool {
        matches!(
            self,
            Self::Running {
                input_captured: true
            }
        )
    }
}

/// Embedded play-mode state plus the optional build child.
pub struct EmbeddedPlaytest<C = Child> {
    phase: EmbeddedPlaytestPhase,
    build_child: Option<C>,
}

/// Production embedded play state.
pub type EmbeddedPlaytestState = EmbeddedPlaytest<Child>;

impl<C> Default for EmbeddedPlaytest<C> {
    fn default() -> Self {
        Self {
            phase: EmbeddedPlaytestPhase::Idle,
            build_child: None,
        }
    }
}

impl<C> EmbeddedPlaytest<C> {
    /// Current phase.
    #[cfg(test)]
    pub const fn phase(&self) -> EmbeddedPlaytestPhase {
        self.phase
    }

    /// Editor-facing status mirror.
    pub const fn editor_status(&self) -> EditorPlaytestStatus {
        self.phase.editor_status()
    }

    /// True when the editor viewport is currently the live game.
    pub const fn is_running(&self) -> bool {
        self.phase.is_running()
    }

    /// True when keyboard/gamepad input should route to the game.
    pub const fn input_captured(&self) -> bool {
        self.phase.input_captured()
    }

    /// Enter background-build phase and retain the child handle.
    pub fn start_building(&mut self, child: C) {
        self.build_child = Some(child);
        self.phase = EmbeddedPlaytestPhase::Building;
    }

    /// Mutable access to the background child while building.
    pub fn building_child_mut(&mut self) -> Option<&mut C> {
        if self.phase == EmbeddedPlaytestPhase::Building {
            self.build_child.as_mut()
        } else {
            None
        }
    }

    /// Take ownership of the background child, if any.
    pub fn take_build_child(&mut self) -> Option<C> {
        self.build_child.take()
    }

    /// Enter running phase.
    pub fn start_running(&mut self, input_captured: bool) {
        self.build_child = None;
        self.phase = EmbeddedPlaytestPhase::Running { input_captured };
    }

    /// Enter failed phase.
    pub fn fail(&mut self) {
        self.build_child = None;
        self.phase = EmbeddedPlaytestPhase::Failed;
    }

    /// Return to idle.
    pub fn stop(&mut self) {
        self.build_child = None;
        self.phase = EmbeddedPlaytestPhase::Idle;
    }

    /// Capture input for the embedded game.
    pub fn capture_input(&mut self) -> bool {
        if let EmbeddedPlaytestPhase::Running { input_captured } = &mut self.phase {
            *input_captured = true;
            true
        } else {
            false
        }
    }

    /// Release input back to the editor.
    pub fn release_input(&mut self) -> bool {
        if let EmbeddedPlaytestPhase::Running { input_captured } = &mut self.phase {
            *input_captured = false;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_playtest_build_success_reaches_running() {
        let mut state = EmbeddedPlaytest::default();
        state.start_building(7u8);
        assert_eq!(state.phase(), EmbeddedPlaytestPhase::Building);
        assert_eq!(state.building_child_mut().copied(), Some(7));

        state.start_running(true);
        assert_eq!(
            state.phase(),
            EmbeddedPlaytestPhase::Running {
                input_captured: true
            }
        );
        assert!(state.building_child_mut().is_none());
    }

    #[test]
    fn embedded_playtest_build_failure_clears_child() {
        let mut state = EmbeddedPlaytest::default();
        state.start_building(9u8);
        state.fail();
        assert_eq!(state.phase(), EmbeddedPlaytestPhase::Failed);
        assert!(state.take_build_child().is_none());
    }

    #[test]
    fn stop_during_build_returns_child_to_caller() {
        let mut state = EmbeddedPlaytest::default();
        state.start_building(3u8);
        assert_eq!(state.take_build_child(), Some(3));
        state.stop();
        assert_eq!(state.phase(), EmbeddedPlaytestPhase::Idle);
    }

    #[test]
    fn capture_and_release_only_apply_while_running() {
        let mut state = EmbeddedPlaytest::<u8>::default();
        assert!(!state.capture_input());

        state.start_running(false);
        assert!(!state.input_captured());
        assert!(state.capture_input());
        assert!(state.input_captured());
        assert!(state.release_input());
        assert!(!state.input_captured());
    }
}
