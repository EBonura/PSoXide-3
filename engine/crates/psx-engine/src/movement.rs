//! Reusable movement-space helpers for third-person games.
//!
//! The editor playtest and future examples use [`Angle`] for camera
//! yaw. Camera yaw is the orbit direction from focus to camera.
//! Pushing the stick forward moves in the camera's view direction;
//! pushing right moves toward screen-right.

use crate::{Angle, Q12};

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
    let deadzone = deadzone.max(0);
    let axis_max = axis_max.max(deadzone.saturating_add(1));
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
}
