//! Reusable movement-space helpers for third-person games.
//!
//! The editor playtest and future examples use [`Angle`] for camera
//! yaw. Camera yaw is the orbit direction from focus to camera.
//! Pushing the stick forward moves in the camera's view direction;
//! pushing right moves toward screen-right.

use crate::{Angle, Q12};

/// One centred analog input axis.
///
/// Values use the conventional PSX/gamepad centred range:
/// negative is left/up depending on the physical axis, positive is
/// right/down, and zero is centred. The type deliberately does not
/// bake in a direction name; call sites decide whether an axis is
/// strafe, forward, yaw, height, and so on.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct InputAxis {
    raw: i16,
}

impl InputAxis {
    /// Centred input.
    pub const ZERO: Self = Self { raw: 0 };

    /// Build from a centred raw axis value.
    pub const fn new(raw: i16) -> Self {
        Self { raw }
    }

    /// Raw centred value.
    pub const fn raw(self) -> i16 {
        self.raw
    }

    /// Same physical axis with inverted sign.
    pub const fn inverted(self) -> Self {
        Self {
            raw: self.raw.wrapping_neg(),
        }
    }

    /// Clamp symmetrically to `[-max, max]`.
    pub fn clamped(self, max: i16) -> Self {
        let max = max.max(0);
        Self {
            raw: self.raw.clamp(-max, max),
        }
    }

    /// Absolute magnitude, saturating `i16::MIN` to `i16::MAX`.
    pub const fn magnitude(self) -> i16 {
        if self.raw == i16::MIN {
            i16::MAX
        } else if self.raw < 0 {
            -self.raw
        } else {
            self.raw
        }
    }

    /// Convert this axis into a signed per-frame step after deadzone.
    pub fn scaled_step(self, profile: InputAxisProfile, max_step: i16) -> i16 {
        let magnitude = self.magnitude().min(profile.axis_max());
        let deadzone = profile.deadzone();
        if magnitude <= deadzone {
            return 0;
        }
        let sign = if self.raw < 0 { -1 } else { 1 };
        let effective = magnitude.saturating_sub(deadzone) as i32;
        let range = (profile.axis_max() - deadzone) as i32;
        (sign * effective * max_step as i32 / range) as i16
    }
}

/// Two centred analog input axes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct InputVector {
    /// Horizontal/local-X axis.
    pub x: InputAxis,
    /// Vertical/local-Y axis.
    pub y: InputAxis,
}

impl InputVector {
    /// Both axes centred.
    pub const ZERO: Self = Self {
        x: InputAxis::ZERO,
        y: InputAxis::ZERO,
    };

    /// Build from typed axes.
    pub const fn new(x: InputAxis, y: InputAxis) -> Self {
        Self { x, y }
    }

    /// Build from raw centred axis values.
    pub const fn from_centered(x: i16, y: i16) -> Self {
        Self::new(InputAxis::new(x), InputAxis::new(y))
    }

    /// Return raw centred axis values.
    pub const fn raw(self) -> (i16, i16) {
        (self.x.raw(), self.y.raw())
    }

    /// Clamp both axes symmetrically to `[-max, max]`.
    pub fn clamped(self, max: i16) -> Self {
        Self::new(self.x.clamped(max), self.y.clamped(max))
    }

    /// Squared 2D magnitude in raw axis units.
    pub fn magnitude_squared(self) -> i32 {
        square_i16(self.x.raw()).saturating_add(square_i16(self.y.raw()))
    }
}

/// Deadzone and maximum magnitude for a centred input axis/vector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InputAxisProfile {
    deadzone: i16,
    axis_max: i16,
}

impl InputAxisProfile {
    /// Build a profile. Negative deadzones are clamped to zero; a
    /// max not larger than the deadzone is raised to `deadzone + 1`
    /// so normalization never divides by zero.
    pub const fn new(deadzone: i16, axis_max: i16) -> Self {
        let deadzone = if deadzone < 0 { 0 } else { deadzone };
        let min_axis_max = if deadzone == i16::MAX {
            i16::MAX
        } else {
            deadzone + 1
        };
        let axis_max = if axis_max < min_axis_max {
            min_axis_max
        } else {
            axis_max
        };
        Self { deadzone, axis_max }
    }

    /// Deadzone threshold.
    pub const fn deadzone(self) -> i16 {
        self.deadzone
    }

    /// Maximum useful centred axis magnitude.
    pub const fn axis_max(self) -> i16 {
        self.axis_max
    }
}

/// World-space movement intent produced from camera-relative input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraRelativeMove {
    /// World-space X intent. [`Q12::ONE`] means full speed toward +X.
    pub x: Q12,
    /// World-space Z intent. [`Q12::ONE`] means full speed toward +Z.
    pub z: Q12,
    /// Signed local forward intent after deadzone. Positive means
    /// the player is pushing forward relative to the camera, negative
    /// means backward, zero means pure strafe or idle.
    pub forward: i8,
}

/// Convert local stick axes into a world-space camera-relative
/// movement vector.
///
/// `strafe` and `forward` are centred stick values, usually in
/// `[-axis_max, axis_max]`. `camera_yaw` is the camera orbit yaw
/// from the focus/player toward the camera, not the view direction.
/// This matches [`ThirdPersonCameraFrame::yaw`](crate::ThirdPersonCameraFrame::yaw).
pub fn camera_relative_move(
    strafe: i16,
    forward: i16,
    camera_yaw: Angle,
    deadzone: i16,
    axis_max: i16,
) -> CameraRelativeMove {
    camera_relative_move_axes(
        InputVector::from_centered(strafe, forward),
        camera_yaw,
        InputAxisProfile::new(deadzone, axis_max),
    )
}

/// Convert typed local stick axes into a world-space camera-relative
/// movement vector.
pub fn camera_relative_move_axes(
    axes: InputVector,
    camera_yaw: Angle,
    profile: InputAxisProfile,
) -> CameraRelativeMove {
    let (strafe, forward) = axes.raw();
    let deadzone = profile.deadzone();
    let axis_max = profile.axis_max();
    let mag = isqrt_i32(square_i16(strafe).saturating_add(square_i16(forward)));
    if mag <= deadzone as i32 {
        return CameraRelativeMove::default();
    }

    let clamped_mag = mag.min(axis_max as i32);
    let scaled_mag = Q12::from_ratio(clamped_mag - deadzone as i32, (axis_max - deadzone) as i32);
    let local_strafe = scaled_mag.mul_ratio(strafe as i32, mag);
    let local_forward = scaled_mag.mul_ratio(forward as i32, mag);

    let forward_yaw = camera_yaw.add(Angle::HALF);
    let right_yaw = forward_yaw.add(Angle::QUARTER);
    let world_x = forward_yaw
        .sin()
        .mul_q12(local_forward)
        .saturating_add(right_yaw.sin().mul_q12(local_strafe));
    let world_z = forward_yaw
        .cos()
        .mul_q12(local_forward)
        .saturating_add(right_yaw.cos().mul_q12(local_strafe));

    CameraRelativeMove {
        x: world_x,
        z: world_z,
        forward: signed_forward_intent(forward, deadzone),
    }
}

/// Compatibility wrapper for callers that still use the old `_q12`
/// function name. The returned scalar fields are typed [`Q12`]
/// values.
pub fn camera_relative_move_q12(
    strafe: i16,
    forward: i16,
    camera_yaw: Angle,
    deadzone: i16,
    axis_max: i16,
) -> CameraRelativeMove {
    camera_relative_move(strafe, forward, camera_yaw, deadzone, axis_max)
}

fn signed_forward_intent(forward: i16, deadzone: i16) -> i8 {
    if forward < -deadzone {
        -1
    } else if forward > deadzone {
        1
    } else {
        0
    }
}

fn square_i16(value: i16) -> i32 {
    let value = value as i32;
    value * value
}

fn isqrt_i32(value: i32) -> i32 {
    if value <= 0 {
        return 0;
    }
    let mut x = value as u32;
    let mut r = 0u32;
    let mut bit = 1u32 << 30;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEADZONE: i16 = 18;
    const AXIS_MAX: i16 = 127;

    #[test]
    fn forward_uses_q12_camera_yaw_without_low_byte_aliasing() {
        let movement = camera_relative_move(0, AXIS_MAX, Angle::QUARTER, DEADZONE, AXIS_MAX);
        assert_eq!(
            movement,
            CameraRelativeMove {
                x: Q12::NEG_ONE,
                z: Q12::ZERO,
                forward: 1,
            }
        );

        let movement = camera_relative_move(0, AXIS_MAX, Angle::THREE_QUARTER, DEADZONE, AXIS_MAX);
        assert_eq!(
            movement,
            CameraRelativeMove {
                x: Q12::ONE,
                z: Q12::ZERO,
                forward: 1,
            }
        );
    }

    #[test]
    fn strafe_right_matches_screen_right() {
        let movement = camera_relative_move(AXIS_MAX, 0, Angle::HALF, DEADZONE, AXIS_MAX);
        assert_eq!(
            movement,
            CameraRelativeMove {
                x: Q12::ONE,
                z: Q12::ZERO,
                forward: 0,
            }
        );

        let movement = camera_relative_move(AXIS_MAX, 0, Angle::ZERO, DEADZONE, AXIS_MAX);
        assert_eq!(
            movement,
            CameraRelativeMove {
                x: Q12::NEG_ONE,
                z: Q12::ZERO,
                forward: 0,
            }
        );
    }

    #[test]
    fn diagonal_input_is_normalized() {
        let movement = camera_relative_move(AXIS_MAX, AXIS_MAX, Angle::HALF, DEADZONE, AXIS_MAX);
        assert!(movement.x.raw() > 2800 && movement.x.raw() < 3000);
        assert!(movement.z.raw() > 2800 && movement.z.raw() < 3000);
        assert_eq!(movement.forward, 1);
    }

    #[test]
    fn deadzone_returns_idle() {
        assert_eq!(
            camera_relative_move(4, -4, Angle::HALF, DEADZONE, AXIS_MAX),
            CameraRelativeMove::default()
        );
    }

    #[test]
    fn input_axis_scaled_step_clamps_to_profile_max() {
        let profile = InputAxisProfile::new(8, 64);
        assert_eq!(InputAxis::new(32767).scaled_step(profile, 12), 12);
        assert_eq!(InputAxis::new(-32768).scaled_step(profile, 12), -12);
    }
}
