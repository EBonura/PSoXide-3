//! Fixed-budget third-person camera controller.
//!
//! The controller is designed for PS1-scale rooms: no heap, no dynamic
//! dispatch, bounded ray work, integer math, and collision probes that
//! read the cooked grid room through [`RoomCollision`]. It supplies the
//! common action-camera pieces a game wants on top of [`WorldCamera`]:
//! manual orbit cooldown, automatic re-alignment, camera lag, lock-on
//! framing, and a spring-arm collision solve that shortens the boom
//! without taking yaw control away from the player.

use crate::{Angle, RoomCollision, RoomPoint, WorldCamera, WorldProjection, Q12};

const RAY_STEPS_MAX: i32 = 8;
const RAY_STEPS_MIN: i32 = 3;

// Mirrors psxed_format::world::direction::* without adding a direct
// psxed-format dependency just for byte constants.
const DIR_NORTH: u8 = 0;
const DIR_EAST: u8 = 1;
const DIR_SOUTH: u8 = 2;
const DIR_WEST: u8 = 3;

/// Tunables for [`ThirdPersonCameraState`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraConfig {
    /// Preferred trailing distance from focus to camera.
    pub distance: i32,
    /// Closest the collision solver may pull the camera.
    pub min_distance: i32,
    /// Furthest distance the camera may ease back out to.
    pub max_distance: i32,
    /// Vertical camera offset above the player origin.
    pub height: i32,
    /// Vertical look-at offset above the player origin.
    pub target_height: i32,
    /// Extra clearance kept between the camera ray and blocking geometry.
    pub collision_margin: i32,
    /// Lowest manual pitch, in signed Q0.12 turn units.
    pub pitch_min_q12: i16,
    /// Highest manual pitch, in signed Q0.12 turn units.
    pub pitch_max_q12: i16,
    /// Frames before auto-alignment resumes after manual camera input.
    pub manual_cooldown_frames: u8,
    /// Maximum auto-align yaw movement per frame.
    pub auto_align_step: Angle,
    /// Position lag strength as a power-of-two divisor.
    pub position_lag_shift: u8,
    /// Focus lag strength as a power-of-two divisor.
    pub focus_lag_shift: u8,
    /// Ease-out strength when collision lets the camera extend again.
    pub distance_lag_shift: u8,
    /// Frames to hold the shortened boom before easing out.
    pub collision_release_delay_frames: u8,
}

impl ThirdPersonCameraConfig {
    /// Build a camera config from the authored Character camera fields.
    pub const fn character(distance: i32, height: i32, target_height: i32) -> Self {
        Self {
            distance,
            min_distance: 384,
            max_distance: distance,
            height,
            target_height,
            collision_margin: 160,
            pitch_min_q12: -192,
            pitch_max_q12: 704,
            manual_cooldown_frames: 42,
            auto_align_step: Angle::from_q12(18),
            position_lag_shift: 2,
            focus_lag_shift: 2,
            distance_lag_shift: 3,
            collision_release_delay_frames: 4,
        }
    }
}

/// Per-frame camera input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ThirdPersonCameraInput {
    /// Signed manual yaw delta in Q0.12 angle units.
    pub yaw_delta_q12: i16,
    /// Signed manual pitch delta in Q0.12 angle units.
    /// Positive raises the camera above the focus point.
    pub pitch_delta_q12: i16,
    /// When true, force the camera to begin easing back behind the player.
    pub recenter: bool,
}

/// Player and optional lock-on target data consumed by the camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraTarget {
    /// Player/root position in room-local world units.
    pub player: RoomPoint,
    /// Player facing yaw.
    pub player_yaw: Angle,
    /// True while the player is intentionally moving.
    pub moving: bool,
    /// Optional lock-on target position in room-local world units.
    pub lock_target: Option<RoomPoint>,
}

/// Camera solve result for the current frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraFrame {
    /// Render camera ready for world/model draw calls.
    pub camera: WorldCamera,
    /// Lagged focus point used by the camera.
    pub focus: RoomPoint,
    /// Camera orbit yaw.
    pub yaw: Angle,
    /// Camera pitch, signed Q0.12 turn units.
    pub pitch_q12: i16,
    /// Current camera distance after collision.
    pub distance: i32,
    /// True when the camera was shortened by collision this frame.
    pub collision_pull_in: bool,
    /// Reserved for older debug overlays; spring-arm collision no
    /// longer steers yaw, so this is currently always false.
    pub collision_rotated: bool,
}

/// Runtime state for the third-person camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraState {
    yaw: Angle,
    pitch_q12: i16,
    distance: i32,
    position: RoomPoint,
    focus: RoomPoint,
    manual_cooldown: u8,
    collision_release_delay: u8,
    initialized: bool,
    last_pull_in: bool,
    last_rotated: bool,
}

impl ThirdPersonCameraState {
    /// Create a camera state with an initial orbit yaw.
    pub const fn new(yaw: Angle) -> Self {
        Self {
            yaw,
            pitch_q12: 0,
            distance: 0,
            position: RoomPoint::ZERO,
            focus: RoomPoint::ZERO,
            manual_cooldown: 0,
            collision_release_delay: 0,
            initialized: false,
            last_pull_in: false,
            last_rotated: false,
        }
    }

    /// Reset the camera immediately behind a player position.
    pub fn snap_to_player(
        &mut self,
        target: ThirdPersonCameraTarget,
        config: ThirdPersonCameraConfig,
    ) {
        let config = normalize_config(config);
        self.yaw = target.player_yaw.add(Angle::HALF);
        self.distance = config
            .distance
            .clamp(config.min_distance, config.max_distance);
        self.pitch_q12 = default_pitch_q12(config);
        self.focus = player_focus(target.player, config.target_height);
        self.position = camera_position(self.focus, self.distance, self.yaw, self.pitch_q12);
        self.manual_cooldown = 0;
        self.collision_release_delay = 0;
        self.initialized = true;
        self.last_pull_in = false;
        self.last_rotated = false;
    }

    /// Advance the controller and build a render camera.
    pub fn update(
        &mut self,
        projection: WorldProjection,
        collision: Option<RoomCollision<'_, '_>>,
        target: ThirdPersonCameraTarget,
        input: ThirdPersonCameraInput,
        config: ThirdPersonCameraConfig,
    ) -> ThirdPersonCameraFrame {
        let config = normalize_config(config);
        if !self.initialized {
            self.snap_to_player(target, config);
        }

        let focus_goal = camera_focus_goal(target, config);

        if input.yaw_delta_q12 != 0 || input.pitch_delta_q12 != 0 {
            self.yaw = self.yaw.add_signed_q12(input.yaw_delta_q12);
            self.pitch_q12 = self
                .pitch_q12
                .saturating_add(input.pitch_delta_q12)
                .clamp(config.pitch_min_q12, config.pitch_max_q12);
            self.manual_cooldown = config.manual_cooldown_frames;
        } else if self.manual_cooldown != 0 {
            self.manual_cooldown -= 1;
        }

        let player_back_yaw = target.player_yaw.add(Angle::HALF);
        let desired_yaw = if let Some(lock) = target.lock_target {
            yaw_to_point(target.player, lock).add(Angle::HALF)
        } else if input.recenter || (target.moving && self.manual_cooldown == 0) {
            player_back_yaw
        } else {
            self.yaw
        };
        self.yaw = self
            .yaw
            .approach_q12(desired_yaw, config.auto_align_step.as_q12());
        if input.recenter {
            self.pitch_q12 = approach_i16(
                self.pitch_q12,
                default_pitch_q12(config),
                config.auto_align_step.as_q12() as i16,
            );
        }

        self.focus = if target.lock_target.is_some() {
            approach_vertex_shift(
                self.focus,
                focus_goal,
                config.focus_lag_shift.saturating_sub(1),
            )
        } else {
            approach_vertex_shift(self.focus, focus_goal, config.focus_lag_shift)
        };

        let collision_solve =
            solve_camera_collision(collision, self.focus, self.yaw, self.pitch_q12, config);

        if collision_solve.distance < self.distance {
            self.distance = collision_solve.distance;
            self.collision_release_delay = config.collision_release_delay_frames;
        } else if self.collision_release_delay != 0 {
            self.collision_release_delay -= 1;
        } else {
            self.distance = approach_i32_shift(
                self.distance,
                collision_solve.distance,
                config.distance_lag_shift,
            );
        }

        let desired_position = camera_position(self.focus, self.distance, self.yaw, self.pitch_q12);
        if collision_solve.pull_in {
            self.position = desired_position;
        } else {
            self.position =
                approach_vertex_shift(self.position, desired_position, config.position_lag_shift);
        }

        self.last_pull_in = collision_solve.pull_in;
        self.last_rotated = false;
        ThirdPersonCameraFrame {
            camera: camera_from_position_focus(projection, self.position, self.focus),
            focus: self.focus,
            yaw: self.yaw,
            pitch_q12: self.pitch_q12,
            distance: self.distance,
            collision_pull_in: self.last_pull_in,
            collision_rotated: self.last_rotated,
        }
    }

    /// Current orbit yaw.
    pub const fn yaw(&self) -> Angle {
        self.yaw
    }

    /// Current orbit pitch in signed Q0.12 units.
    pub const fn pitch_q12(&self) -> i16 {
        self.pitch_q12
    }

    /// Current camera position.
    pub const fn position(&self) -> RoomPoint {
        self.position
    }

    /// Current lagged focus point.
    pub const fn focus(&self) -> RoomPoint {
        self.focus
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CollisionSolve {
    distance: i32,
    pull_in: bool,
}

fn normalize_config(mut config: ThirdPersonCameraConfig) -> ThirdPersonCameraConfig {
    config.min_distance = config.min_distance.max(128);
    config.max_distance = config.max_distance.max(config.min_distance);
    config.distance = config
        .distance
        .clamp(config.min_distance, config.max_distance);
    config.collision_margin = config.collision_margin.max(0);
    if config.pitch_min_q12 > config.pitch_max_q12 {
        let pitch = config.pitch_min_q12;
        config.pitch_min_q12 = config.pitch_max_q12;
        config.pitch_max_q12 = pitch;
    }
    if config.auto_align_step == Angle::ZERO {
        config.auto_align_step = Angle::from_q12(1);
    }
    config.position_lag_shift = config.position_lag_shift.min(6);
    config.focus_lag_shift = config.focus_lag_shift.min(6);
    config.distance_lag_shift = config.distance_lag_shift.min(6);
    config
}

fn player_focus(player: RoomPoint, target_height: i32) -> RoomPoint {
    RoomPoint::new(player.x, player.y.saturating_add(target_height), player.z)
}

fn camera_focus_goal(
    target: ThirdPersonCameraTarget,
    config: ThirdPersonCameraConfig,
) -> RoomPoint {
    let player = player_focus(target.player, config.target_height);
    if let Some(lock) = target.lock_target {
        let lock_focus = player_focus(lock, config.target_height / 2);
        RoomPoint::new(
            midpoint_i32(player.x, lock_focus.x),
            midpoint_i32(player.y, lock_focus.y),
            midpoint_i32(player.z, lock_focus.z),
        )
    } else {
        player
    }
}

fn midpoint_i32(a: i32, b: i32) -> i32 {
    a.saturating_add(b.saturating_sub(a) / 2)
}

fn solve_camera_collision(
    collision: Option<RoomCollision<'_, '_>>,
    focus: RoomPoint,
    yaw: Angle,
    pitch_q12: i16,
    config: ThirdPersonCameraConfig,
) -> CollisionSolve {
    let Some(room) = collision else {
        return CollisionSolve {
            distance: config.distance,
            pull_in: false,
        };
    };

    let desired = camera_position(focus, config.distance, yaw, pitch_q12);
    let clear = probe_clear_distance(room, focus, desired, config.distance, config);
    let distance = clear.clamp(config.min_distance, config.distance);
    CollisionSolve {
        distance,
        pull_in: distance < config.distance,
    }
}

fn probe_clear_distance(
    room: RoomCollision<'_, '_>,
    from: RoomPoint,
    to: RoomPoint,
    max_distance: i32,
    config: ThirdPersonCameraConfig,
) -> i32 {
    let max_distance = max_distance.max(1);
    let sector = room.sector_size().max(1);
    let mut steps = (max_distance / (sector / 4).max(1)).clamp(RAY_STEPS_MIN, RAY_STEPS_MAX);
    if steps <= 0 {
        steps = RAY_STEPS_MIN;
    }

    let mut nearest = max_distance;
    let mut i = 1;
    while i <= steps {
        let sample = lerp_vertex(from, to, i, steps);
        if point_outside_camera_space(room, sample) {
            nearest = ((max_distance * i) / steps).min(nearest);
            break;
        }
        if let Some(hit) = nearest_wall_hit_around(
            room,
            sample,
            from,
            to,
            max_distance,
            config.collision_margin,
        ) {
            nearest = hit.min(nearest);
            break;
        }
        i += 1;
    }

    nearest
        .saturating_sub(config.collision_margin)
        .clamp(config.min_distance, config.distance)
}

fn point_outside_camera_space(room: RoomCollision<'_, '_>, point: RoomPoint) -> bool {
    let s = room.sector_size();
    if s <= 0 || point.x < 0 || point.z < 0 {
        return true;
    }
    let sx = point.x / s;
    let sz = point.z / s;
    if sx < 0 || sz < 0 || sx >= room.width() as i32 || sz >= room.depth() as i32 {
        return true;
    }
    match room.sector(sx as u16, sz as u16) {
        Some(sector) => !sector.has_floor(),
        None => true,
    }
}

fn nearest_wall_hit_around(
    room: RoomCollision<'_, '_>,
    sample: RoomPoint,
    from: RoomPoint,
    to: RoomPoint,
    ray_distance: i32,
    vertical_margin: i32,
) -> Option<i32> {
    let s = room.sector_size();
    if s <= 0 || sample.x < 0 || sample.z < 0 {
        return None;
    }
    let sx = sample.x / s;
    let sz = sample.z / s;
    let mut nearest: Option<i32> = None;
    let mut ox = -1;
    while ox <= 1 {
        let mut oz = -1;
        while oz <= 1 {
            let cx = sx + ox;
            let cz = sz + oz;
            if cx >= 0 && cz >= 0 && cx < room.width() as i32 && cz < room.depth() as i32 {
                if let Some(sector) = room.sector(cx as u16, cz as u16) {
                    let mut i = 0;
                    while i < sector.wall_count() {
                        if let Some(wall) = room.sector_wall(sector, i) {
                            if wall.solid() {
                                if let Some(hit) = segment_wall_hit_distance(
                                    from,
                                    to,
                                    ray_distance,
                                    cx,
                                    cz,
                                    s,
                                    wall.direction(),
                                    wall.heights(),
                                    vertical_margin,
                                ) {
                                    nearest = Some(match nearest {
                                        Some(prev) => prev.min(hit),
                                        None => hit,
                                    });
                                }
                            }
                        }
                        i += 1;
                    }
                }
            }
            oz += 1;
        }
        ox += 1;
    }
    nearest
}

fn segment_wall_hit_distance(
    from: RoomPoint,
    to: RoomPoint,
    ray_distance: i32,
    sx: i32,
    sz: i32,
    sector_size: i32,
    direction: u8,
    heights: [i32; 4],
    vertical_margin: i32,
) -> Option<i32> {
    let x0 = sx.saturating_mul(sector_size);
    let x1 = x0.saturating_add(sector_size);
    let z0 = sz.saturating_mul(sector_size);
    let z1 = z0.saturating_add(sector_size);
    let dx = to.x.saturating_sub(from.x);
    let dz = to.z.saturating_sub(from.z);
    if ray_distance <= 0 {
        return None;
    }
    let t_q12 = match direction {
        DIR_NORTH => intersect_horizontal_q12(from.z, dz, z0),
        DIR_SOUTH => intersect_horizontal_q12(from.z, dz, z1),
        DIR_EAST => intersect_vertical_q12(from.x, dx, x1),
        DIR_WEST => intersect_vertical_q12(from.x, dx, x0),
        _ => None,
    }?;
    if !(0..=Q12::SCALE).contains(&t_q12) {
        return None;
    }
    let t = Q12::from_raw(t_q12);
    let x_at = from.x.saturating_add(t.mul_i32(dx));
    let y_at = from
        .y
        .saturating_add(t.mul_i32(to.y.saturating_sub(from.y)));
    let z_at = from.z.saturating_add(t.mul_i32(dz));
    let wall_axis_q12 = match direction {
        DIR_NORTH | DIR_SOUTH => {
            if x_at < x0 || x_at > x1 {
                return None;
            }
            (x_at.saturating_sub(x0))
                .saturating_mul(Q12::SCALE)
                .checked_div(sector_size.max(1))?
        }
        DIR_EAST | DIR_WEST => {
            if z_at < z0 || z_at > z1 {
                return None;
            }
            (z_at.saturating_sub(z0))
                .saturating_mul(Q12::SCALE)
                .checked_div(sector_size.max(1))?
        }
        _ => return None,
    };
    let axis = Q12::from_raw(wall_axis_q12.clamp(0, Q12::SCALE));
    let (bottom, top) = match direction {
        DIR_NORTH | DIR_EAST => (
            lerp_i32(heights[0], heights[1], axis),
            lerp_i32(heights[3], heights[2], axis),
        ),
        DIR_SOUTH | DIR_WEST => (
            lerp_i32(heights[1], heights[0], axis),
            lerp_i32(heights[2], heights[3], axis),
        ),
        _ => return None,
    };
    let min_y = bottom.min(top).saturating_sub(vertical_margin);
    let max_y = bottom.max(top).saturating_add(vertical_margin);
    if y_at < min_y || y_at > max_y {
        return None;
    }
    Some(t.mul_i32(ray_distance))
}

fn intersect_horizontal_q12(from_z: i32, dz: i32, wall_z: i32) -> Option<i32> {
    if dz == 0 {
        return None;
    }
    wall_z
        .saturating_sub(from_z)
        .saturating_mul(Q12::SCALE)
        .checked_div(dz)
}

fn intersect_vertical_q12(from_x: i32, dx: i32, wall_x: i32) -> Option<i32> {
    if dx == 0 {
        return None;
    }
    wall_x
        .saturating_sub(from_x)
        .saturating_mul(Q12::SCALE)
        .checked_div(dx)
}

fn camera_position(focus: RoomPoint, distance: i32, yaw: Angle, pitch_q12: i16) -> RoomPoint {
    let sin_yaw = yaw.sin();
    let cos_yaw = yaw.cos();
    let pitch = signed_q12_angle(pitch_q12);
    let sin_pitch = pitch.sin();
    let cos_pitch = pitch.cos();
    let horizontal = cos_pitch.mul_i32(distance);
    RoomPoint::new(
        focus.x.saturating_add(sin_yaw.mul_i32(horizontal)),
        focus.y.saturating_add(sin_pitch.mul_i32(distance)),
        focus.z.saturating_add(cos_yaw.mul_i32(horizontal)),
    )
}

fn camera_from_position_focus(
    projection: WorldProjection,
    position: RoomPoint,
    focus: RoomPoint,
) -> WorldCamera {
    let dx = position.x.saturating_sub(focus.x);
    let dz = position.z.saturating_sub(focus.z);
    let radius = isqrt_i32(dx.saturating_mul(dx).saturating_add(dz.saturating_mul(dz))).max(1);
    let target_dy = focus.y.saturating_sub(position.y);
    let pitch_len = isqrt_i32(
        radius
            .saturating_mul(radius)
            .saturating_add(target_dy.saturating_mul(target_dy)),
    )
    .max(1);
    WorldCamera {
        position: position.to_world_vertex(),
        projection,
        sin_yaw: Q12::from_ratio(dx, radius),
        cos_yaw: Q12::from_ratio(dz, radius),
        sin_pitch: Q12::from_ratio(target_dy, pitch_len),
        cos_pitch: Q12::from_ratio(radius, pitch_len),
    }
}

fn default_pitch_q12(config: ThirdPersonCameraConfig) -> i16 {
    pitch_from_vertical_distance(
        config.height.saturating_sub(config.target_height),
        config.distance,
    )
    .clamp(config.pitch_min_q12, config.pitch_max_q12)
}

fn pitch_from_vertical_distance(vertical: i32, horizontal: i32) -> i16 {
    if vertical == 0 {
        return 0;
    }
    let ay = abs_i32(vertical);
    let ax = abs_i32(horizontal).max(1);
    let base = if ay <= ax {
        ay.saturating_mul(512) / ax
    } else {
        1024 - (ax.saturating_mul(512) / ay.max(1))
    }
    .min(1024);
    let signed = if vertical < 0 { -base } else { base };
    signed.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn signed_q12_angle(q12: i16) -> Angle {
    Angle::from_q12(((q12 as i32) & 0x0FFF) as u16)
}

fn lerp_i32(a: i32, b: i32, t: Q12) -> i32 {
    a.saturating_add(t.mul_i32(b.saturating_sub(a)))
}

fn yaw_to_point(from: RoomPoint, to: RoomPoint) -> Angle {
    let dx = to.x.saturating_sub(from.x);
    let dz = to.z.saturating_sub(from.z);
    if dx == 0 && dz == 0 {
        return Angle::ZERO;
    }
    let ax = abs_i32(dx);
    let az = abs_i32(dz);
    let base = if ax <= az {
        ax.saturating_mul(512) / az.max(1)
    } else {
        1024 - (az.saturating_mul(512) / ax.max(1))
    };
    let angle = if dz >= 0 {
        if dx >= 0 {
            base
        } else {
            4096 - base
        }
    } else if dx >= 0 {
        2048 - base
    } else {
        2048 + base
    };
    Angle::from_q12((angle & 0x0FFF) as u16)
}

fn approach_i16(current: i16, target: i16, step: i16) -> i16 {
    let step = step.max(1);
    let delta = target.saturating_sub(current);
    if abs_i16(delta) <= step {
        target
    } else if delta > 0 {
        current.saturating_add(step)
    } else {
        current.saturating_sub(step)
    }
}

fn approach_i32_shift(current: i32, target: i32, shift: u8) -> i32 {
    if current == target {
        return current;
    }
    let shift = shift.min(6);
    let delta = target.saturating_sub(current);
    let step = if shift == 0 { delta } else { delta >> shift };
    if step == 0 {
        current.saturating_add(delta.signum())
    } else {
        current.saturating_add(step)
    }
}

fn approach_vertex_shift(current: RoomPoint, target: RoomPoint, shift: u8) -> RoomPoint {
    RoomPoint::new(
        approach_i32_shift(current.x, target.x, shift),
        approach_i32_shift(current.y, target.y, shift),
        approach_i32_shift(current.z, target.z, shift),
    )
}

fn lerp_vertex(from: RoomPoint, to: RoomPoint, num: i32, den: i32) -> RoomPoint {
    RoomPoint::new(
        from.x + ((to.x - from.x) * num) / den,
        from.y + ((to.y - from.y) * num) / den,
        from.z + ((to.z - from.z) * num) / den,
    )
}

fn isqrt_i32(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut bit = 1 << 30;
    let mut rest = n;
    let mut root = 0;
    while bit > rest {
        bit >>= 2;
    }
    while bit != 0 {
        if rest >= root + bit {
            rest -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn abs_i16(value: i16) -> i16 {
    if value == i16::MIN {
        i16::MAX
    } else if value < 0 {
        -value
    } else {
        value
    }
}

fn abs_i32(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else {
        value.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaw_to_point_matches_cardinal_axes() {
        let origin = RoomPoint::ZERO;
        assert_eq!(yaw_to_point(origin, RoomPoint::new(0, 0, 10)), Angle::ZERO);
        assert_eq!(
            yaw_to_point(origin, RoomPoint::new(10, 0, 0)),
            Angle::QUARTER
        );
        assert_eq!(yaw_to_point(origin, RoomPoint::new(0, 0, -10)), Angle::HALF);
        assert_eq!(
            yaw_to_point(origin, RoomPoint::new(-10, 0, 0)),
            Angle::THREE_QUARTER
        );
    }

    #[test]
    fn approach_angle_takes_shortest_wrapping_path() {
        assert_eq!(
            Angle::from_q12(4090).approach_q12(Angle::from_q12(8), 16),
            Angle::from_q12(8)
        );
        assert_eq!(
            Angle::from_q12(20).approach_q12(Angle::from_q12(4000), 16),
            Angle::from_q12(4)
        );
    }

    #[test]
    fn segment_wall_hit_finds_cardinal_crossing() {
        let from = RoomPoint::new(512, 0, 512);
        let to = RoomPoint::new(1536, 0, 512);
        let heights = [-512, -512, 512, 512];
        assert_eq!(
            segment_wall_hit_distance(from, to, 1024, 0, 0, 1024, DIR_EAST, heights, 0),
            Some(512)
        );
        assert_eq!(
            segment_wall_hit_distance(from, to, 1024, 0, 0, 1024, DIR_NORTH, heights, 0),
            None
        );
    }

    #[test]
    fn segment_wall_hit_ignores_camera_ray_above_wall() {
        let from = RoomPoint::new(512, 900, 512);
        let to = RoomPoint::new(1536, 900, 512);
        let heights = [0, 0, 512, 512];

        assert_eq!(
            segment_wall_hit_distance(from, to, 1024, 0, 0, 1024, DIR_EAST, heights, 0),
            None
        );
    }

    #[test]
    fn manual_input_sets_cooldown_and_prevents_auto_align() {
        let mut camera = ThirdPersonCameraState::new(Angle::HALF);
        let config = ThirdPersonCameraConfig::character(1400, 700, 0);
        let target = ThirdPersonCameraTarget {
            player: RoomPoint::ZERO,
            player_yaw: Angle::ZERO,
            moving: true,
            lock_target: None,
        };
        let frame = camera.update(
            WorldProjection::new(160, 120, 320, 64),
            None,
            target,
            ThirdPersonCameraInput {
                yaw_delta_q12: 128,
                pitch_delta_q12: 0,
                recenter: false,
            },
            config,
        );
        assert_eq!(frame.yaw, Angle::HALF.add_signed_q12(128));
        assert_eq!(frame.pitch_q12, default_pitch_q12(config));
        let frame = camera.update(
            WorldProjection::new(160, 120, 320, 64),
            None,
            target,
            ThirdPersonCameraInput::default(),
            config,
        );
        assert_eq!(frame.yaw, Angle::HALF.add_signed_q12(128));
    }

    #[test]
    fn character_height_offsets_raise_camera_and_focus() {
        let mut camera = ThirdPersonCameraState::new(Angle::HALF);
        let config = ThirdPersonCameraConfig::character(1400, 700, 400);
        let target = ThirdPersonCameraTarget {
            player: RoomPoint::new(128, 32, -64),
            player_yaw: Angle::ZERO,
            moving: false,
            lock_target: None,
        };

        camera.snap_to_player(target, config);

        assert_eq!(camera.focus.y, target.player.y + config.target_height);
        assert!(camera.position.y > camera.focus.y);
        assert_eq!(camera.pitch_q12, default_pitch_q12(config));
    }

    #[test]
    fn manual_pitch_input_clamps_to_config_limits() {
        let mut camera = ThirdPersonCameraState::new(Angle::HALF);
        let mut config = ThirdPersonCameraConfig::character(1400, 700, 0);
        config.pitch_min_q12 = -64;
        config.pitch_max_q12 = 96;
        let target = ThirdPersonCameraTarget {
            player: RoomPoint::ZERO,
            player_yaw: Angle::ZERO,
            moving: false,
            lock_target: None,
        };

        let frame = camera.update(
            WorldProjection::new(160, 120, 320, 64),
            None,
            target,
            ThirdPersonCameraInput {
                yaw_delta_q12: 0,
                pitch_delta_q12: 512,
                recenter: false,
            },
            config,
        );

        assert_eq!(frame.pitch_q12, 96);
    }
}
