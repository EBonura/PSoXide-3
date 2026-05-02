//! Player movement and camera input helpers.

use psx_engine::{
    camera_relative_move_axes, Angle, InputAxis, InputAxisProfile, InputVector, RoomPoint,
};

use super::*;

pub(crate) fn motor_input(ctx: &Ctx, camera_yaw: Angle) -> CharacterMotorInput {
    let movement = camera_relative_move_axes(local_move_axes(ctx), camera_yaw, move_axis_profile());

    CharacterMotorInput {
        turn: 0,
        walk: movement.forward,
        move_x: movement.x,
        move_z: movement.z,
        sprint: ctx.is_held(RUN_BUTTON),
        evade: false,
    }
}

pub(crate) fn local_move_axes(ctx: &Ctx) -> InputVector {
    let (left_x, left_y) = ctx.pad.sticks.left_centered();
    let left = InputVector::from_centered(left_x, left_y);
    let stick_mag = isqrt_i32(left.magnitude_squared());
    if stick_mag > MOVE_STICK_DEADZONE as i32 {
        return InputVector::new(left.x, left.y.inverted()).clamped(STICK_MAX);
    }

    let mut strafe = 0i16;
    let mut forward = 0i16;
    if ctx.is_held(button::RIGHT) {
        strafe += STICK_MAX;
    }
    if ctx.is_held(button::LEFT) {
        strafe -= STICK_MAX;
    }
    if ctx.is_held(button::UP) {
        forward += STICK_MAX;
    }
    if ctx.is_held(button::DOWN) {
        forward -= STICK_MAX;
    }
    InputVector::from_centered(strafe, forward)
}

pub(crate) fn player_anim_from_motor(anim: CharacterMotorAnim) -> PlayerAnim {
    match anim {
        CharacterMotorAnim::Idle => PlayerAnim::Idle,
        CharacterMotorAnim::Walk => PlayerAnim::Walk,
        CharacterMotorAnim::Run => PlayerAnim::Run,
        CharacterMotorAnim::Roll => PlayerAnim::Run,
        CharacterMotorAnim::Backstep => PlayerAnim::Walk,
    }
}

pub(crate) fn camera_input(ctx: &Ctx) -> ThirdPersonCameraInput {
    let (right_x, _) = ctx.pad.sticks.right_centered();
    ThirdPersonCameraInput {
        yaw_delta_q12: stick_to_yaw_delta(InputAxis::new(right_x.saturating_neg())),
        recenter: ctx.is_held(button::L1),
    }
}

pub(crate) fn stick_to_yaw_delta(axis: InputAxis) -> i16 {
    stick_axis_delta(axis, CAMERA_STICK_YAW_STEP)
}

pub(crate) fn stick_to_radius_delta(axis: InputAxis) -> i32 {
    stick_axis_delta(axis, CAMERA_RADIUS_STEP as i16) as i32
}

pub(crate) fn stick_to_height_delta(axis: InputAxis) -> i32 {
    stick_axis_delta(axis, CAMERA_HEIGHT_STICK_STEP as i16) as i32
}

pub(crate) fn stick_axis_delta(axis: InputAxis, max_step: i16) -> i16 {
    axis.scaled_step(camera_axis_profile(), max_step)
}

fn move_axis_profile() -> InputAxisProfile {
    InputAxisProfile::new(MOVE_STICK_DEADZONE, STICK_MAX)
}

fn camera_axis_profile() -> InputAxisProfile {
    InputAxisProfile::new(CAMERA_STICK_DEADZONE, STICK_MAX)
}

pub(crate) fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub(crate) fn abs_i16(value: i16) -> i16 {
    if value == i16::MIN {
        i16::MAX
    } else if value < 0 {
        -value
    } else {
        value
    }
}

pub(crate) fn distance_xz_sq(a: RoomPoint, b: RoomPoint) -> i32 {
    let dx = a.x.saturating_sub(b.x);
    let dz = a.z.saturating_sub(b.z);
    square_i32_saturating(dx).saturating_add(square_i32_saturating(dz))
}

pub(crate) fn square_i32_saturating(value: i32) -> i32 {
    let abs = abs_i32(value);
    if abs > 46_340 {
        return i32::MAX;
    }
    abs * abs
}

pub(crate) fn isqrt_i32(value: i32) -> i32 {
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

pub(crate) fn abs_i32(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else if value < 0 {
        -value
    } else {
        value
    }
}
