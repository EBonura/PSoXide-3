//! Player movement and camera input helpers.

use super::*;

pub(crate) fn motor_input(ctx: &Ctx, camera_yaw_q12: u16) -> CharacterMotorInput {
    let (strafe, forward) = local_move_axes(ctx);
    let (move_x_q12, move_z_q12, walk) =
        camera_relative_move_q12(strafe, forward, camera_yaw_q12);

    CharacterMotorInput {
        turn: 0,
        walk,
        move_x_q12,
        move_z_q12,
        sprint: ctx.is_held(RUN_BUTTON),
        evade: false,
    }
}

pub(crate) fn local_move_axes(ctx: &Ctx) -> (i16, i16) {
    let (left_x, left_y) = ctx.pad.sticks.left_centered();
    let stick_mag = isqrt(left_x as i64 * left_x as i64 + left_y as i64 * left_y as i64);
    if stick_mag > MOVE_STICK_DEADZONE as i64 {
        return (left_x.clamp(-STICK_MAX, STICK_MAX), (-left_y).clamp(-STICK_MAX, STICK_MAX));
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
    (strafe, forward)
}

pub(crate) fn camera_relative_move_q12(
    strafe: i16,
    forward: i16,
    camera_yaw_q12: u16,
) -> (i16, i16, i8) {
    let mag = isqrt(strafe as i64 * strafe as i64 + forward as i64 * forward as i64);
    if mag <= MOVE_STICK_DEADZONE as i64 {
        return (0, 0, 0);
    }
    let clamped_mag = mag.min(STICK_MAX as i64);
    let scaled_mag_q12 =
        ((clamped_mag - MOVE_STICK_DEADZONE as i64) * 4096)
            / (STICK_MAX - MOVE_STICK_DEADZONE) as i64;
    let local_strafe_q12 = (strafe as i64 * scaled_mag_q12 / mag) as i32;
    let local_forward_q12 = (forward as i64 * scaled_mag_q12 / mag) as i32;

    let forward_yaw = camera_yaw_q12.wrapping_add(HALF_TURN_Q12);
    let right_yaw = forward_yaw.wrapping_sub(1024);
    let world_x = (((sin_1_3_12(forward_yaw) as i32) * local_forward_q12)
        + ((sin_1_3_12(right_yaw) as i32) * local_strafe_q12))
        >> 12;
    let world_z = (((cos_1_3_12(forward_yaw) as i32) * local_forward_q12)
        + ((cos_1_3_12(right_yaw) as i32) * local_strafe_q12))
        >> 12;
    let walk = if forward < -MOVE_STICK_DEADZONE {
        -1
    } else if forward > MOVE_STICK_DEADZONE {
        1
    } else {
        0
    };
    (clamp_i16(world_x), clamp_i16(world_z), walk)
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
        yaw_delta_q12: stick_to_yaw_delta(right_x),
        recenter: ctx.is_held(button::L1),
    }
}

pub(crate) fn stick_to_yaw_delta(axis: i16) -> i16 {
    stick_axis_delta(axis, CAMERA_STICK_YAW_STEP)
}

pub(crate) fn stick_to_radius_delta(axis: i16) -> i32 {
    stick_axis_delta(axis, CAMERA_RADIUS_STEP as i16) as i32
}

pub(crate) fn stick_to_height_delta(axis: i16) -> i32 {
    stick_axis_delta(axis, CAMERA_HEIGHT_STICK_STEP as i16) as i32
}

pub(crate) fn stick_axis_delta(axis: i16, max_step: i16) -> i16 {
    let magnitude = abs_i16(axis);
    if magnitude <= CAMERA_STICK_DEADZONE {
        return 0;
    }
    let sign = if axis < 0 { -1 } else { 1 };
    let effective = magnitude.saturating_sub(CAMERA_STICK_DEADZONE) as i32;
    let range = (STICK_MAX - CAMERA_STICK_DEADZONE) as i32;
    (sign * effective * max_step as i32 / range) as i16
}

pub(crate) fn add_signed_q12(angle: u16, delta: i16) -> u16 {
    ((angle as i32 + delta as i32) & 0x0FFF) as u16
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

pub(crate) fn distance_xz_sq(a: WorldVertex, b: WorldVertex) -> i64 {
    let dx = (a.x - b.x) as i64;
    let dz = (a.z - b.z) as i64;
    dx * dx + dz * dz
}
