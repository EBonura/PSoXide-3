//! Fixed-budget third-person camera controller.
//!
//! The controller is designed for PS1-scale rooms: no heap, no dynamic
//! dispatch, bounded ray work, integer math, and collision probes that
//! read the cooked grid room through [`RoomCollision`]. It supplies the
//! common action-camera pieces a game wants on top of [`WorldCamera`]:
//! manual orbit cooldown, automatic re-alignment, camera lag, lock-on
//! framing, and a small fan of "whisker" probes that tries to rotate
//! around walls before falling back to pulling the camera closer.

use psx_math::{cos_q12, sin_q12};

use crate::{RoomCollision, WorldCamera, WorldProjection, WorldVertex};

const HALF_TURN_Q12: u16 = 2048;
const FULL_TURN_Q12: i32 = 4096;
const WHISKER_COUNT: usize = 3;
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
    /// Angular stride between whisker rays, in 4096-units-per-turn angles.
    pub whisker_stride_q12: u16,
    /// Frames before auto-alignment resumes after manual camera input.
    pub manual_cooldown_frames: u8,
    /// Maximum auto-align yaw movement per frame.
    pub auto_align_step_q12: u16,
    /// Maximum collision-driven yaw movement per frame.
    pub collision_yaw_step_q12: u16,
    /// Position lag strength as a power-of-two divisor.
    pub position_lag_shift: u8,
    /// Focus lag strength as a power-of-two divisor.
    pub focus_lag_shift: u8,
    /// Ease-out strength when collision lets the camera extend again.
    pub distance_lag_shift: u8,
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
            whisker_stride_q12: 112,
            manual_cooldown_frames: 42,
            auto_align_step_q12: 18,
            collision_yaw_step_q12: 42,
            position_lag_shift: 2,
            focus_lag_shift: 2,
            distance_lag_shift: 3,
        }
    }
}

/// Per-frame camera input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ThirdPersonCameraInput {
    /// Signed manual yaw delta in 4096-units-per-turn angle space.
    pub yaw_delta_q12: i16,
    /// When true, force the camera to begin easing back behind the player.
    pub recenter: bool,
}

/// Player and optional lock-on target data consumed by the camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraTarget {
    /// Player/root position in room-local world units.
    pub player: WorldVertex,
    /// Player facing yaw, in 4096-units-per-turn angle space.
    pub player_yaw_q12: u16,
    /// True while the player is intentionally moving.
    pub moving: bool,
    /// Optional lock-on target position in room-local world units.
    pub lock_target: Option<WorldVertex>,
}

/// Camera solve result for the current frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraFrame {
    /// Render camera ready for world/model draw calls.
    pub camera: WorldCamera,
    /// Lagged focus point used by the camera.
    pub focus: WorldVertex,
    /// Camera orbit yaw, in 4096-units-per-turn angle space.
    pub yaw_q12: u16,
    /// Current camera distance after collision.
    pub distance: i32,
    /// True when the camera was shortened by collision this frame.
    pub collision_pull_in: bool,
    /// True when a whisker ray steered the camera around a blocker.
    pub collision_rotated: bool,
}

/// Runtime state for the third-person camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThirdPersonCameraState {
    yaw_q12: u16,
    distance: i32,
    position: WorldVertex,
    focus: WorldVertex,
    manual_cooldown: u8,
    initialized: bool,
    last_pull_in: bool,
    last_rotated: bool,
}

impl ThirdPersonCameraState {
    /// Create a camera state with an initial orbit yaw.
    pub const fn new(yaw_q12: u16) -> Self {
        Self {
            yaw_q12,
            distance: 0,
            position: WorldVertex::ZERO,
            focus: WorldVertex::ZERO,
            manual_cooldown: 0,
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
        self.yaw_q12 = target.player_yaw_q12.wrapping_add(HALF_TURN_Q12);
        self.distance = config
            .distance
            .clamp(config.min_distance, config.max_distance);
        self.focus = player_focus(target.player, config.target_height);
        self.position = camera_position(
            self.focus,
            target.player.y.saturating_add(config.height),
            self.distance,
            self.yaw_q12,
        );
        self.manual_cooldown = 0;
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
        let camera_y = camera_height_goal(focus_goal, target, config);

        if input.yaw_delta_q12 != 0 {
            self.yaw_q12 = add_signed_angle(self.yaw_q12, input.yaw_delta_q12);
            self.manual_cooldown = config.manual_cooldown_frames;
        } else if self.manual_cooldown != 0 {
            self.manual_cooldown -= 1;
        }

        let player_back_yaw = target.player_yaw_q12.wrapping_add(HALF_TURN_Q12);
        let desired_yaw = if let Some(lock) = target.lock_target {
            yaw_to_point_q12(target.player, lock).wrapping_add(HALF_TURN_Q12)
        } else if input.recenter || (target.moving && self.manual_cooldown == 0) {
            player_back_yaw
        } else {
            self.yaw_q12
        };
        self.yaw_q12 = approach_angle(self.yaw_q12, desired_yaw, config.auto_align_step_q12);

        let collision_solve =
            solve_camera_collision(collision, focus_goal, camera_y, self.yaw_q12, config);
        let rotated_to_clear_path = collision_solve.rotated
            && collision_solve
                .distance
                .saturating_add(config.collision_margin)
                >= config.distance;
        self.yaw_q12 = if rotated_to_clear_path {
            collision_solve.yaw_q12
        } else {
            approach_angle(
                self.yaw_q12,
                collision_solve.yaw_q12,
                config.collision_yaw_step_q12,
            )
        };

        if collision_solve.distance < self.distance {
            self.distance = collision_solve.distance;
        } else {
            self.distance = approach_i32_shift(
                self.distance,
                collision_solve.distance,
                config.distance_lag_shift,
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

        let desired_position = camera_position(self.focus, camera_y, self.distance, self.yaw_q12);
        if collision_solve.pull_in || collision_solve.rotated {
            self.position = desired_position;
        } else {
            self.position =
                approach_vertex_shift(self.position, desired_position, config.position_lag_shift);
        }

        self.last_pull_in = collision_solve.pull_in;
        self.last_rotated = collision_solve.rotated;
        ThirdPersonCameraFrame {
            camera: camera_from_position_focus(projection, self.position, self.focus),
            focus: self.focus,
            yaw_q12: self.yaw_q12,
            distance: self.distance,
            collision_pull_in: self.last_pull_in,
            collision_rotated: self.last_rotated,
        }
    }

    /// Current orbit yaw.
    pub const fn yaw_q12(&self) -> u16 {
        self.yaw_q12
    }

    /// Current camera position.
    pub const fn position(&self) -> WorldVertex {
        self.position
    }

    /// Current lagged focus point.
    pub const fn focus(&self) -> WorldVertex {
        self.focus
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CollisionSolve {
    yaw_q12: u16,
    distance: i32,
    pull_in: bool,
    rotated: bool,
}

fn normalize_config(mut config: ThirdPersonCameraConfig) -> ThirdPersonCameraConfig {
    config.min_distance = config.min_distance.max(128);
    config.max_distance = config.max_distance.max(config.min_distance);
    config.distance = config
        .distance
        .clamp(config.min_distance, config.max_distance);
    config.collision_margin = config.collision_margin.max(0);
    config.whisker_stride_q12 = config.whisker_stride_q12.max(1);
    config.auto_align_step_q12 = config.auto_align_step_q12.max(1);
    config.collision_yaw_step_q12 = config.collision_yaw_step_q12.max(1);
    config.position_lag_shift = config.position_lag_shift.min(6);
    config.focus_lag_shift = config.focus_lag_shift.min(6);
    config.distance_lag_shift = config.distance_lag_shift.min(6);
    config
}

fn player_focus(player: WorldVertex, target_height: i32) -> WorldVertex {
    WorldVertex::new(player.x, player.y.saturating_add(target_height), player.z)
}

fn camera_focus_goal(
    target: ThirdPersonCameraTarget,
    config: ThirdPersonCameraConfig,
) -> WorldVertex {
    let player = player_focus(target.player, config.target_height);
    if let Some(lock) = target.lock_target {
        let lock_focus = player_focus(lock, config.target_height / 2);
        WorldVertex::new(
            midpoint_i32(player.x, lock_focus.x),
            midpoint_i32(player.y, lock_focus.y),
            midpoint_i32(player.z, lock_focus.z),
        )
    } else {
        player
    }
}

fn camera_height_goal(
    focus: WorldVertex,
    target: ThirdPersonCameraTarget,
    config: ThirdPersonCameraConfig,
) -> i32 {
    if let Some(lock) = target.lock_target {
        let span = abs_i32(lock.y.saturating_sub(target.player.y)).min(config.height);
        focus
            .y
            .saturating_add(config.height)
            .saturating_add(span / 2)
    } else {
        target.player.y.saturating_add(config.height)
    }
}

fn midpoint_i32(a: i32, b: i32) -> i32 {
    a.saturating_add(b.saturating_sub(a) / 2)
}

fn solve_camera_collision(
    collision: Option<RoomCollision<'_, '_>>,
    focus: WorldVertex,
    camera_y: i32,
    yaw_q12: u16,
    config: ThirdPersonCameraConfig,
) -> CollisionSolve {
    let Some(room) = collision else {
        return CollisionSolve {
            yaw_q12,
            distance: config.distance,
            pull_in: false,
            rotated: false,
        };
    };

    let stride = config.whisker_stride_q12 as i16;
    let whiskers = [0i16, -stride, stride];
    let mut best_yaw = yaw_q12;
    let mut best_distance = 0;
    let mut center_distance = 0;

    let mut i = 0;
    while i < WHISKER_COUNT {
        let candidate_yaw = add_signed_angle(yaw_q12, whiskers[i]);
        let candidate_pos = camera_position(focus, camera_y, config.distance, candidate_yaw);
        let clear = probe_clear_distance(room, focus, candidate_pos, config.distance, config);
        if i == 0 {
            center_distance = clear;
            best_distance = clear;
        } else if clear > best_distance.saturating_add(config.collision_margin / 2) {
            best_distance = clear;
            best_yaw = candidate_yaw;
        }
        i += 1;
    }

    let distance = best_distance.clamp(config.min_distance, config.distance);
    CollisionSolve {
        yaw_q12: best_yaw,
        distance,
        pull_in: distance < config.distance,
        rotated: best_yaw != yaw_q12 && best_distance > center_distance,
    }
}

fn probe_clear_distance(
    room: RoomCollision<'_, '_>,
    from: WorldVertex,
    to: WorldVertex,
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
        if let Some(hit) = nearest_wall_hit_around(room, sample, from, to, max_distance) {
            nearest = hit.min(nearest);
            break;
        }
        i += 1;
    }

    nearest
        .saturating_sub(config.collision_margin)
        .clamp(config.min_distance, config.distance)
}

fn point_outside_camera_space(room: RoomCollision<'_, '_>, point: WorldVertex) -> bool {
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
    sample: WorldVertex,
    from: WorldVertex,
    to: WorldVertex,
    ray_distance: i32,
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
    from: WorldVertex,
    to: WorldVertex,
    ray_distance: i32,
    sx: i32,
    sz: i32,
    sector_size: i32,
    direction: u8,
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
    if !(0..=4096).contains(&t_q12) {
        return None;
    }
    let x_at = from.x.saturating_add((dx.saturating_mul(t_q12)) >> 12);
    let z_at = from.z.saturating_add((dz.saturating_mul(t_q12)) >> 12);
    match direction {
        DIR_NORTH | DIR_SOUTH => {
            if x_at < x0 || x_at > x1 {
                return None;
            }
        }
        DIR_EAST | DIR_WEST => {
            if z_at < z0 || z_at > z1 {
                return None;
            }
        }
        _ => return None,
    }
    Some(ray_distance.saturating_mul(t_q12) >> 12)
}

fn intersect_horizontal_q12(from_z: i32, dz: i32, wall_z: i32) -> Option<i32> {
    if dz == 0 {
        return None;
    }
    wall_z
        .saturating_sub(from_z)
        .saturating_mul(4096)
        .checked_div(dz)
}

fn intersect_vertical_q12(from_x: i32, dx: i32, wall_x: i32) -> Option<i32> {
    if dx == 0 {
        return None;
    }
    wall_x
        .saturating_sub(from_x)
        .saturating_mul(4096)
        .checked_div(dx)
}

fn camera_position(focus: WorldVertex, camera_y: i32, distance: i32, yaw_q12: u16) -> WorldVertex {
    let sin_yaw = sin_q12(yaw_q12);
    let cos_yaw = cos_q12(yaw_q12);
    WorldVertex::new(
        focus
            .x
            .saturating_add((sin_yaw.saturating_mul(distance)) >> 12),
        camera_y,
        focus
            .z
            .saturating_add((cos_yaw.saturating_mul(distance)) >> 12),
    )
}

fn camera_from_position_focus(
    projection: WorldProjection,
    position: WorldVertex,
    focus: WorldVertex,
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
        position,
        projection,
        sin_yaw: q12_ratio_i32(dx, radius),
        cos_yaw: q12_ratio_i32(dz, radius),
        sin_pitch: q12_ratio_i32(target_dy, pitch_len),
        cos_pitch: q12_ratio_i32(radius, pitch_len),
    }
}

fn q12_ratio_i32(numerator: i32, denominator: i32) -> i32 {
    if denominator == 0 {
        return 0;
    }
    numerator.saturating_mul(4096) / denominator
}

fn yaw_to_point_q12(from: WorldVertex, to: WorldVertex) -> u16 {
    let dx = to.x.saturating_sub(from.x);
    let dz = to.z.saturating_sub(from.z);
    if dx == 0 && dz == 0 {
        return 0;
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
            FULL_TURN_Q12 - base
        }
    } else if dx >= 0 {
        2048 - base
    } else {
        2048 + base
    };
    (angle & 0x0FFF) as u16
}

fn add_signed_angle(angle: u16, delta: i16) -> u16 {
    ((angle as i32 + delta as i32) & 0x0FFF) as u16
}

fn approach_angle(current: u16, target: u16, step: u16) -> u16 {
    let delta = shortest_angle_delta(current, target);
    let step = step as i16;
    if abs_i16(delta) <= step {
        target
    } else if delta > 0 {
        add_signed_angle(current, step)
    } else {
        add_signed_angle(current, -step)
    }
}

fn shortest_angle_delta(current: u16, target: u16) -> i16 {
    let mut delta = ((target as i32 - current as i32) & 0x0FFF) as i16;
    if delta > 2048 {
        delta -= 4096;
    }
    delta
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

fn approach_vertex_shift(current: WorldVertex, target: WorldVertex, shift: u8) -> WorldVertex {
    WorldVertex::new(
        approach_i32_shift(current.x, target.x, shift),
        approach_i32_shift(current.y, target.y, shift),
        approach_i32_shift(current.z, target.z, shift),
    )
}

fn lerp_vertex(from: WorldVertex, to: WorldVertex, num: i32, den: i32) -> WorldVertex {
    WorldVertex::new(
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

fn abs_i32(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else {
        value.abs()
    }
}

fn abs_i16(value: i16) -> i16 {
    if value == i16::MIN {
        i16::MAX
    } else {
        value.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaw_to_point_matches_cardinal_axes() {
        let origin = WorldVertex::ZERO;
        assert_eq!(yaw_to_point_q12(origin, WorldVertex::new(0, 0, 10)), 0);
        assert_eq!(yaw_to_point_q12(origin, WorldVertex::new(10, 0, 0)), 1024);
        assert_eq!(yaw_to_point_q12(origin, WorldVertex::new(0, 0, -10)), 2048);
        assert_eq!(yaw_to_point_q12(origin, WorldVertex::new(-10, 0, 0)), 3072);
    }

    #[test]
    fn approach_angle_takes_shortest_wrapping_path() {
        assert_eq!(approach_angle(4090, 8, 16), 8);
        assert_eq!(approach_angle(20, 4000, 16), 4);
    }

    #[test]
    fn segment_wall_hit_finds_cardinal_crossing() {
        let from = WorldVertex::new(512, 0, 512);
        let to = WorldVertex::new(1536, 0, 512);
        assert_eq!(
            segment_wall_hit_distance(from, to, 1024, 0, 0, 1024, DIR_EAST),
            Some(512)
        );
        assert_eq!(
            segment_wall_hit_distance(from, to, 1024, 0, 0, 1024, DIR_NORTH),
            None
        );
    }

    #[test]
    fn manual_input_sets_cooldown_and_prevents_auto_align() {
        let mut camera = ThirdPersonCameraState::new(HALF_TURN_Q12);
        let config = ThirdPersonCameraConfig::character(1400, 700, 0);
        let target = ThirdPersonCameraTarget {
            player: WorldVertex::ZERO,
            player_yaw_q12: 0,
            moving: true,
            lock_target: None,
        };
        let frame = camera.update(
            WorldProjection::new(160, 120, 320, 64),
            None,
            target,
            ThirdPersonCameraInput {
                yaw_delta_q12: 128,
                recenter: false,
            },
            config,
        );
        assert_eq!(frame.yaw_q12, HALF_TURN_Q12 + 128);
        let frame = camera.update(
            WorldProjection::new(160, 120, 320, 64),
            None,
            target,
            ThirdPersonCameraInput::default(),
            config,
        );
        assert_eq!(frame.yaw_q12, HALF_TURN_Q12 + 128);
    }

    #[test]
    fn character_height_offsets_raise_camera_and_focus() {
        let mut camera = ThirdPersonCameraState::new(HALF_TURN_Q12);
        let config = ThirdPersonCameraConfig::character(1400, 700, 400);
        let target = ThirdPersonCameraTarget {
            player: WorldVertex::new(128, 32, -64),
            player_yaw_q12: 0,
            moving: false,
            lock_target: None,
        };

        camera.snap_to_player(target, config);

        assert_eq!(camera.focus.y, target.player.y + config.target_height);
        assert_eq!(camera.position.y, target.player.y + config.height);
    }
}
