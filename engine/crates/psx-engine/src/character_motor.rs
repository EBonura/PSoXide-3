//! Fixed-budget third-person character motor.
//!
//! The motor owns player locomotion state that should be shared by
//! game code and AI controllers: position, yaw, stamina, short evade
//! actions, and the coarse collision commit against cooked room data.
//! Inputs are intent-shaped rather than pad-shaped so callers can feed
//! either player controls or future behaviour-tree output.

use crate::{Angle, RoomCollision, RoomPoint, Q12};

const DEFAULT_STAMINA_MAX_Q12: i32 = 4096;
const MAX_MOTOR_CATCHUP_VBLANKS: u16 = 4;
const SPLIT_NE_SW: u8 = 1;

/// Tunables for [`CharacterMotorState`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CharacterMotorConfig {
    /// Capsule radius in world units. Collision currently samples
    /// the centre plus four cardinal points around this radius.
    pub radius: i32,
    /// Forward/backward walking speed in world units per frame.
    pub walk_speed: i32,
    /// Sprint speed in world units per frame.
    pub run_speed: i32,
    /// Turn speed per frame.
    pub yaw_step: Angle,
    /// Maximum stamina, in Q12-style arbitrary units.
    pub stamina_max_q12: i32,
    /// Minimum stamina required to start sprinting.
    pub sprint_min_q12: i32,
    /// Stamina spent per sprinting frame.
    pub sprint_drain_q12: i32,
    /// Stamina recovered per grounded non-sprint frame.
    pub stamina_recover_q12: i32,
    /// Stamina spent to start a roll.
    pub roll_cost_q12: i32,
    /// Roll travel speed in world units per frame.
    pub roll_speed: i32,
    /// Frames where roll keeps moving.
    pub roll_active_frames: u8,
    /// Recovery frames after roll movement ends.
    pub roll_recovery_frames: u8,
    /// Roll invulnerability frames from action start.
    pub roll_invulnerable_frames: u8,
    /// Stamina spent to start a backstep.
    pub backstep_cost_q12: i32,
    /// Backstep travel speed in world units per frame.
    pub backstep_speed: i32,
    /// Frames where backstep keeps moving.
    pub backstep_active_frames: u8,
    /// Recovery frames after backstep movement ends.
    pub backstep_recovery_frames: u8,
    /// Backstep invulnerability frames from action start.
    pub backstep_invulnerable_frames: u8,
}

impl CharacterMotorConfig {
    /// Build a motor config from authored Character movement fields.
    pub const fn character(radius: i32, walk_speed: i32, run_speed: i32, yaw_step: Angle) -> Self {
        Self {
            radius,
            walk_speed,
            run_speed,
            yaw_step,
            stamina_max_q12: DEFAULT_STAMINA_MAX_Q12,
            sprint_min_q12: 384,
            sprint_drain_q12: 48,
            stamina_recover_q12: 36,
            roll_cost_q12: 768,
            roll_speed: 96,
            roll_active_frames: 14,
            roll_recovery_frames: 12,
            roll_invulnerable_frames: 10,
            backstep_cost_q12: 512,
            backstep_speed: 72,
            backstep_active_frames: 8,
            backstep_recovery_frames: 10,
            backstep_invulnerable_frames: 6,
        }
    }
}

/// Per-frame abstract movement intent.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CharacterMotorInput {
    /// Signed turn intent. Negative turns left, positive turns right.
    pub turn: i8,
    /// Signed forward/back intent. Negative backs up, positive walks forward.
    pub walk: i8,
    /// World-space analog X movement intent. [`Q12::ONE`] is
    /// full-strength movement to +X. When either analog movement
    /// component is non-zero, the motor uses this vector instead of
    /// tank-style `turn` / `walk`.
    pub move_x: Q12,
    /// World-space analog Z movement intent. [`Q12::ONE`] is
    /// full-strength movement to +Z.
    pub move_z: Q12,
    /// True while the actor wants to spend stamina on sprinting.
    pub sprint: bool,
    /// Rising-edge evade request. The motor chooses roll vs backstep
    /// from the current walk intent.
    pub evade: bool,
}

/// Current high-level action.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CharacterMotorAction {
    /// No fixed action is currently playing.
    Idle,
    /// Forward evasive roll.
    Roll,
    /// Backward evasive step.
    Backstep,
}

impl CharacterMotorAction {
    /// `true` when no fixed action is currently playing.
    pub const fn is_idle(self) -> bool {
        matches!(self, Self::Idle)
    }
}

/// Animation intent produced by the motor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CharacterMotorAnim {
    /// Standing still.
    Idle,
    /// Walking or backing up.
    Walk,
    /// Sprinting.
    Run,
    /// Forward evasive roll.
    Roll,
    /// Backward evasive step.
    Backstep,
}

/// Result of one motor update.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CharacterMotorFrame {
    /// Current root position.
    pub position: RoomPoint,
    /// Current facing yaw.
    pub yaw: Angle,
    /// Animation intent for this frame.
    pub anim: CharacterMotorAnim,
    /// Current fixed action, if any.
    pub action: CharacterMotorAction,
    /// True when the root position changed this frame.
    pub moved: bool,
    /// True when requested movement hit coarse room collision.
    pub blocked: bool,
    /// True while a successful sprint is active.
    pub sprinting: bool,
    /// True during action invulnerability frames.
    pub invulnerable: bool,
    /// True during the non-moving tail of a fixed action.
    pub recovery: bool,
    /// Current stamina after this frame.
    pub stamina_q12: i32,
}

/// Runtime character motor state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CharacterMotorState {
    position: RoomPoint,
    yaw: Angle,
    stamina_q12: i32,
    action: CharacterMotorAction,
    action_frame: u8,
    action_yaw: Angle,
    /// Sprint is latched while the button stays held so
    /// `sprint_min_q12` means "minimum to start", not "minimum to
    /// continue".
    sprint_latched: bool,
    /// Prevents held-sprint from pulsing Run/Walk every recovery
    /// frame after stamina reaches zero.
    sprint_exhausted: bool,
}

impl CharacterMotorState {
    /// Create a motor at a root position and yaw.
    pub const fn new(position: RoomPoint, yaw: Angle) -> Self {
        Self {
            position,
            yaw,
            stamina_q12: DEFAULT_STAMINA_MAX_Q12,
            action: CharacterMotorAction::Idle,
            action_frame: 0,
            action_yaw: yaw,
            sprint_latched: false,
            sprint_exhausted: false,
        }
    }

    /// Reset position, yaw, stamina, and any in-progress action.
    pub fn snap_to(&mut self, position: RoomPoint, yaw: Angle) {
        self.position = position;
        self.yaw = yaw;
        self.stamina_q12 = DEFAULT_STAMINA_MAX_Q12;
        self.action = CharacterMotorAction::Idle;
        self.action_frame = 0;
        self.action_yaw = yaw;
        self.sprint_latched = false;
        self.sprint_exhausted = false;
    }

    /// Move the motor to another coordinate space while preserving
    /// yaw, stamina, and any in-progress action. Used by streaming
    /// room transitions where the same physical player position is
    /// re-expressed relative to a newly-current chunk.
    pub fn relocate(&mut self, position: RoomPoint) {
        self.position = position;
    }

    /// Advance the motor by one frame.
    pub fn update(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        input: CharacterMotorInput,
        config: CharacterMotorConfig,
    ) -> CharacterMotorFrame {
        self.update_vblanks(collision, input, config, 1)
    }

    /// Advance the motor by elapsed display ticks.
    ///
    /// Heavy render paths can miss VBlanks. Animation already uses
    /// display time, so the motor catches up with small fixed
    /// substeps instead of scaling one large collision step. The cap
    /// prevents a long pause from spending a whole frame in movement
    /// catch-up.
    pub fn update_vblanks(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        input: CharacterMotorInput,
        config: CharacterMotorConfig,
        delta_vblanks: u16,
    ) -> CharacterMotorFrame {
        let config = normalize_config(config);
        let steps = delta_vblanks.max(1).min(MAX_MOTOR_CATCHUP_VBLANKS);
        let mut final_frame: Option<CharacterMotorFrame> = None;

        for step in 0..steps {
            let mut step_input = input;
            if step > 0 {
                step_input.evade = false;
            }
            let frame = self.update_one_frame(collision, step_input, config);
            final_frame = Some(match final_frame {
                Some(mut aggregate) => {
                    aggregate.position = frame.position;
                    aggregate.yaw = frame.yaw;
                    aggregate.anim = frame.anim;
                    aggregate.action = frame.action;
                    aggregate.moved |= frame.moved;
                    aggregate.blocked |= frame.blocked;
                    aggregate.sprinting = frame.sprinting;
                    aggregate.invulnerable |= frame.invulnerable;
                    aggregate.recovery |= frame.recovery;
                    aggregate.stamina_q12 = frame.stamina_q12;
                    aggregate
                }
                None => frame,
            });
        }

        final_frame.unwrap_or_else(|| {
            self.frame(
                CharacterMotorAnim::Idle,
                self.action,
                false,
                false,
                false,
                false,
                false,
            )
        })
    }

    fn update_one_frame(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        input: CharacterMotorInput,
        config: CharacterMotorConfig,
    ) -> CharacterMotorFrame {
        self.stamina_q12 = self.stamina_q12.clamp(0, config.stamina_max_q12);
        self.snap_floor(collision, config.radius);

        if self.action.is_idle() && input.evade {
            self.try_start_evade(input, config);
        }

        if !self.action.is_idle() {
            return self.update_action(collision, config);
        }

        if let Some((move_x, move_z, move_mag)) = analog_move_vector(input) {
            self.update_sprint_gate(input.sprint, true);
            self.yaw = yaw_from_vector(move_x, move_z);
            let wants_sprint = input.sprint;
            let sprinting = self.can_sprint(wants_sprint, config);
            let base_speed = if sprinting {
                config.run_speed
            } else {
                config.walk_speed
            };
            let speed = move_mag.mul_i32(base_speed);
            let (moved, blocked) =
                self.try_move_vector(collision, move_x, move_z, speed, config.radius);

            if sprinting && moved {
                self.spend_sprint_stamina(config);
            } else {
                self.recover_stamina(config);
            }

            let anim = if !moved && blocked {
                CharacterMotorAnim::Idle
            } else if sprinting {
                CharacterMotorAnim::Run
            } else {
                CharacterMotorAnim::Walk
            };

            return self.frame(
                anim,
                CharacterMotorAction::Idle,
                moved,
                blocked,
                sprinting,
                false,
                false,
            );
        }

        if input.turn > 0 {
            self.yaw = self.yaw.add(config.yaw_step);
        } else if input.turn < 0 {
            self.yaw = self.yaw.sub(config.yaw_step);
        }

        let moving_intent = input.walk != 0;
        self.update_sprint_gate(input.sprint, moving_intent);
        let wants_forward_sprint = input.sprint && input.walk > 0;
        let sprinting = moving_intent && self.can_sprint(wants_forward_sprint, config);
        let speed = if sprinting {
            config.run_speed
        } else {
            config.walk_speed
        };
        let signed_speed = if input.walk < 0 { -speed } else { speed };

        let (moved, blocked) = if moving_intent {
            self.try_move(collision, signed_speed, config.radius)
        } else {
            (false, false)
        };

        if sprinting && moved {
            self.spend_sprint_stamina(config);
        } else {
            self.recover_stamina(config);
        }

        let anim = if !moving_intent || !moved && blocked {
            CharacterMotorAnim::Idle
        } else if sprinting {
            CharacterMotorAnim::Run
        } else {
            CharacterMotorAnim::Walk
        };

        self.frame(
            anim,
            CharacterMotorAction::Idle,
            moved,
            blocked,
            sprinting,
            false,
            false,
        )
    }

    /// Current root position.
    pub const fn position(&self) -> RoomPoint {
        self.position
    }

    /// Current facing yaw.
    pub const fn yaw(&self) -> Angle {
        self.yaw
    }

    /// Current stamina value.
    pub const fn stamina_q12(&self) -> i32 {
        self.stamina_q12
    }

    /// Current fixed action.
    pub const fn action(&self) -> CharacterMotorAction {
        self.action
    }

    fn try_start_evade(&mut self, input: CharacterMotorInput, config: CharacterMotorConfig) {
        let analog = analog_move_vector(input);
        if let Some((move_x, move_z, _)) = analog {
            self.yaw = yaw_from_vector(move_x, move_z);
        }
        let action = if analog.is_none() && input.walk < 0 {
            CharacterMotorAction::Backstep
        } else {
            CharacterMotorAction::Roll
        };
        let cost = match action {
            CharacterMotorAction::Idle => 0,
            CharacterMotorAction::Roll => config.roll_cost_q12,
            CharacterMotorAction::Backstep => config.backstep_cost_q12,
        };
        if self.stamina_q12 < cost {
            return;
        }
        self.stamina_q12 -= cost;
        self.action = action;
        self.action_frame = 0;
        self.action_yaw = self.yaw;
    }

    fn update_action(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        config: CharacterMotorConfig,
    ) -> CharacterMotorFrame {
        let profile = ActionProfile::for_action(self.action, config);
        let frame = self.action_frame;
        let active = frame < profile.active_frames;
        let invulnerable = frame < profile.invulnerable_frames;
        let recovery = frame >= profile.active_frames;

        let (moved, blocked) = if active {
            let signed_speed = profile.speed.saturating_mul(profile.direction as i32);
            self.try_move_at_yaw(collision, self.action_yaw, signed_speed, config.radius)
        } else {
            (false, false)
        };

        self.action_frame = self.action_frame.saturating_add(1);
        let finished = self.action_frame >= profile.total_frames();
        let action = self.action;
        if finished {
            self.action = CharacterMotorAction::Idle;
            self.action_frame = 0;
            self.recover_stamina(config);
        }

        let anim = match action {
            CharacterMotorAction::Idle => CharacterMotorAnim::Idle,
            CharacterMotorAction::Roll => CharacterMotorAnim::Roll,
            CharacterMotorAction::Backstep => CharacterMotorAnim::Backstep,
        };
        self.frame(anim, action, moved, blocked, false, invulnerable, recovery)
    }

    fn try_move(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        signed_speed: i32,
        radius: i32,
    ) -> (bool, bool) {
        self.try_move_at_yaw(collision, self.yaw, signed_speed, radius)
    }

    fn try_move_at_yaw(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        yaw: Angle,
        signed_speed: i32,
        radius: i32,
    ) -> (bool, bool) {
        if signed_speed == 0 {
            return (false, false);
        }
        let sin_yaw = yaw.sin();
        let cos_yaw = yaw.cos();
        let target = RoomPoint::new(
            self.position
                .x
                .saturating_add(sin_yaw.mul_i32(signed_speed)),
            self.position.y,
            self.position
                .z
                .saturating_add(cos_yaw.mul_i32(signed_speed)),
        );

        let Some(room) = collision else {
            self.position = target;
            return (true, false);
        };
        let Some(height) = stand_height(room, target.x, target.z, radius) else {
            return (false, true);
        };
        self.position = target.with_y(height);
        (true, false)
    }

    fn try_move_vector(
        &mut self,
        collision: Option<RoomCollision<'_, '_>>,
        move_x: Q12,
        move_z: Q12,
        speed: i32,
        radius: i32,
    ) -> (bool, bool) {
        if speed == 0 {
            return (false, false);
        }
        let dx = move_x.mul_i32(speed);
        let dz = move_z.mul_i32(speed);
        if dx == 0 && dz == 0 {
            return (false, false);
        }
        let target = RoomPoint::new(
            self.position.x.saturating_add(dx),
            self.position.y,
            self.position.z.saturating_add(dz),
        );

        let Some(room) = collision else {
            self.position = target;
            return (true, false);
        };
        let Some(height) = stand_height(room, target.x, target.z, radius) else {
            return (false, true);
        };
        self.position = target.with_y(height);
        (true, false)
    }

    fn recover_stamina(&mut self, config: CharacterMotorConfig) {
        self.stamina_q12 = self
            .stamina_q12
            .saturating_add(config.stamina_recover_q12)
            .min(config.stamina_max_q12);
    }

    fn update_sprint_gate(&mut self, wants_sprint: bool, moving: bool) {
        if !wants_sprint || !moving {
            self.sprint_latched = false;
            self.sprint_exhausted = false;
        }
    }

    fn can_sprint(&mut self, wants_sprint: bool, config: CharacterMotorConfig) -> bool {
        if !wants_sprint || self.sprint_exhausted || self.stamina_q12 <= 0 {
            self.sprint_latched = false;
            return false;
        }
        if self.sprint_latched || self.stamina_q12 >= config.sprint_min_q12 {
            self.sprint_latched = true;
            true
        } else {
            false
        }
    }

    fn spend_sprint_stamina(&mut self, config: CharacterMotorConfig) {
        self.stamina_q12 = self
            .stamina_q12
            .saturating_sub(config.sprint_drain_q12)
            .max(0);
        if self.stamina_q12 == 0 {
            self.sprint_latched = false;
            self.sprint_exhausted = true;
        }
    }

    fn snap_floor(&mut self, collision: Option<RoomCollision<'_, '_>>, radius: i32) {
        let Some(room) = collision else {
            return;
        };
        if let Some(height) = stand_height(room, self.position.x, self.position.z, radius) {
            self.position.y = height;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn frame(
        &self,
        anim: CharacterMotorAnim,
        action: CharacterMotorAction,
        moved: bool,
        blocked: bool,
        sprinting: bool,
        invulnerable: bool,
        recovery: bool,
    ) -> CharacterMotorFrame {
        CharacterMotorFrame {
            position: self.position,
            yaw: self.yaw,
            anim,
            action,
            moved,
            blocked,
            sprinting,
            invulnerable,
            recovery,
            stamina_q12: self.stamina_q12,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ActionProfile {
    speed: i32,
    direction: i8,
    active_frames: u8,
    recovery_frames: u8,
    invulnerable_frames: u8,
}

impl ActionProfile {
    fn for_action(action: CharacterMotorAction, config: CharacterMotorConfig) -> Self {
        match action {
            CharacterMotorAction::Idle => Self {
                speed: 0,
                direction: 0,
                active_frames: 0,
                recovery_frames: 0,
                invulnerable_frames: 0,
            },
            CharacterMotorAction::Roll => Self {
                speed: config.roll_speed,
                direction: 1,
                active_frames: config.roll_active_frames,
                recovery_frames: config.roll_recovery_frames,
                invulnerable_frames: config.roll_invulnerable_frames,
            },
            CharacterMotorAction::Backstep => Self {
                speed: config.backstep_speed,
                direction: -1,
                active_frames: config.backstep_active_frames,
                recovery_frames: config.backstep_recovery_frames,
                invulnerable_frames: config.backstep_invulnerable_frames,
            },
        }
    }

    fn total_frames(self) -> u8 {
        self.active_frames
            .saturating_add(self.recovery_frames)
            .max(1)
    }
}

fn normalize_config(mut config: CharacterMotorConfig) -> CharacterMotorConfig {
    config.radius = config.radius.max(0);
    config.walk_speed = config.walk_speed.max(0);
    config.run_speed = config.run_speed.max(config.walk_speed);
    if config.yaw_step == Angle::ZERO {
        config.yaw_step = Angle::from_q12(1);
    }
    config.stamina_max_q12 = config.stamina_max_q12.max(1);
    config.sprint_min_q12 = config.sprint_min_q12.clamp(0, config.stamina_max_q12);
    config.sprint_drain_q12 = config.sprint_drain_q12.max(0);
    config.stamina_recover_q12 = config.stamina_recover_q12.max(0);
    config.roll_cost_q12 = config.roll_cost_q12.clamp(0, config.stamina_max_q12);
    config.roll_speed = config.roll_speed.max(0);
    config.roll_active_frames = config.roll_active_frames.max(1);
    config.roll_invulnerable_frames = config.roll_invulnerable_frames.min(
        config
            .roll_active_frames
            .saturating_add(config.roll_recovery_frames),
    );
    config.backstep_cost_q12 = config.backstep_cost_q12.clamp(0, config.stamina_max_q12);
    config.backstep_speed = config.backstep_speed.max(0);
    config.backstep_active_frames = config.backstep_active_frames.max(1);
    config.backstep_invulnerable_frames = config.backstep_invulnerable_frames.min(
        config
            .backstep_active_frames
            .saturating_add(config.backstep_recovery_frames),
    );
    config
}

fn analog_move_vector(input: CharacterMotorInput) -> Option<(Q12, Q12, Q12)> {
    let x = input.move_x.raw();
    let z = input.move_z.raw();
    if x == 0 && z == 0 {
        return None;
    }
    let mag = isqrt_i32(square_i32_saturating(x).saturating_add(square_i32_saturating(z)));
    if mag <= 0 {
        return None;
    }
    if mag <= Q12::SCALE {
        return Some((input.move_x, input.move_z, Q12::from_raw(mag)));
    }
    Some((
        Q12::ONE.mul_ratio(x, mag),
        Q12::ONE.mul_ratio(z, mag),
        Q12::ONE,
    ))
}

fn yaw_from_vector(dx: Q12, dz: Q12) -> Angle {
    let dx = dx.raw();
    let dz = dz.raw();
    if dx == 0 && dz == 0 {
        return Angle::ZERO;
    }
    let ax = abs_i32(dx);
    let az = abs_i32(dz);
    let base = if ax <= az {
        ax * 512 / az.max(1)
    } else {
        1024 - (az * 512 / ax.max(1))
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

fn square_i32_saturating(value: i32) -> i32 {
    let abs = abs_i32(value);
    if abs > 46_340 {
        return i32::MAX;
    }
    abs * abs
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

fn abs_i32(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else if value < 0 {
        -value
    } else {
        value
    }
}

fn stand_height(room: RoomCollision<'_, '_>, x: i32, z: i32, radius: i32) -> Option<i32> {
    let height = floor_height_at(room, x, z)?;
    if radius <= 0 {
        return Some(height);
    }
    let r = radius.max(0);
    floor_height_at(room, x.saturating_sub(r), z)?;
    floor_height_at(room, x.saturating_add(r), z)?;
    floor_height_at(room, x, z.saturating_sub(r))?;
    floor_height_at(room, x, z.saturating_add(r))?;
    Some(height)
}

fn floor_height_at(room: RoomCollision<'_, '_>, x: i32, z: i32) -> Option<i32> {
    let s = room.sector_size();
    if s <= 0 || x < 0 || z < 0 {
        return None;
    }
    let sx = x / s;
    let sz = z / s;
    if sx < 0 || sz < 0 || sx >= room.width() as i32 || sz >= room.depth() as i32 {
        return None;
    }
    let sector = room.sector(sx as u16, sz as u16)?;
    if !sector.has_floor() || !sector.floor_walkable() {
        return None;
    }
    let local_x = (x - sx * s).clamp(0, s);
    let local_z = (z - sz * s).clamp(0, s);
    Some(height_at_local(
        sector.floor_heights(),
        sector.floor_split(),
        local_x,
        local_z,
        s,
    ))
}

fn height_at_local(heights: [i32; 4], split: u8, local_x: i32, local_z: i32, sector: i32) -> i32 {
    let u = local_x.clamp(0, sector);
    let v = local_z.clamp(0, sector);
    let [nw, ne, se, sw] = heights;
    if split == SPLIT_NE_SW {
        if u + v <= sector {
            nw.saturating_add(mul_sector(ne.saturating_sub(nw), u, sector))
                .saturating_add(mul_sector(sw.saturating_sub(nw), v, sector))
        } else {
            sw.saturating_add(mul_sector(se.saturating_sub(sw), u, sector))
                .saturating_add(mul_sector(ne.saturating_sub(se), sector - v, sector))
        }
    } else if v <= u {
        nw.saturating_add(mul_sector(ne.saturating_sub(nw), u - v, sector))
            .saturating_add(mul_sector(se.saturating_sub(nw), v, sector))
    } else {
        nw.saturating_add(mul_sector(se.saturating_sub(sw), u, sector))
            .saturating_add(mul_sector(sw.saturating_sub(nw), v, sector))
    }
}

fn mul_sector(delta: i32, amount: i32, sector: i32) -> i32 {
    if sector <= 0 {
        0
    } else {
        delta.saturating_mul(amount) / sector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CharacterMotorConfig {
        CharacterMotorConfig::character(64, 32, 64, Angle::from_q12(16))
    }

    #[test]
    fn forward_input_moves_along_yaw() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                walk: 1,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.position, RoomPoint::new(0, 0, 32));
        assert_eq!(frame.anim, CharacterMotorAnim::Walk);
        assert!(frame.moved);
    }

    #[test]
    fn turn_input_wraps_yaw() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                turn: -1,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.yaw, Angle::ZERO.add_signed_q12(-16));
    }

    #[test]
    fn analog_vector_moves_without_tank_turning() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                move_x: Q12::ONE,
                move_z: Q12::ZERO,
                walk: 1,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.position, RoomPoint::new(32, 0, 0));
        assert_eq!(frame.yaw, Angle::QUARTER);
        assert_eq!(frame.anim, CharacterMotorAnim::Walk);
        assert!(frame.moved);
    }

    #[test]
    fn analog_vector_scales_speed_by_magnitude() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                move_x: Q12::HALF,
                move_z: Q12::ZERO,
                walk: 1,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.position, RoomPoint::new(8, 0, 0));
        assert_eq!(frame.yaw, Angle::QUARTER);
    }

    #[test]
    fn analog_sprint_reports_run() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                move_x: Q12::ONE,
                move_z: Q12::ZERO,
                walk: 1,
                sprint: true,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.position, RoomPoint::new(64, 0, 0));
        assert_eq!(frame.anim, CharacterMotorAnim::Run);
        assert!(frame.sprinting);
    }

    #[test]
    fn sprint_consumes_stamina_and_reports_run() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                walk: 1,
                sprint: true,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.position, RoomPoint::new(0, 0, 64));
        assert_eq!(frame.anim, CharacterMotorAnim::Run);
        assert!(frame.sprinting);
        assert!(frame.stamina_q12 < DEFAULT_STAMINA_MAX_Q12);
    }

    #[test]
    fn held_sprint_stays_walk_after_exhaustion_until_released() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let mut cfg = config();
        cfg.stamina_max_q12 = 96;
        cfg.sprint_min_q12 = 32;
        cfg.sprint_drain_q12 = 64;
        cfg.stamina_recover_q12 = 16;
        motor.stamina_q12 = cfg.stamina_max_q12;

        let held = CharacterMotorInput {
            walk: 1,
            sprint: true,
            ..CharacterMotorInput::default()
        };

        let first = motor.update(None, held, cfg);
        let second = motor.update(None, held, cfg);
        assert_eq!(first.anim, CharacterMotorAnim::Run);
        assert_eq!(second.anim, CharacterMotorAnim::Run);
        assert_eq!(second.stamina_q12, 0);

        for _ in 0..4 {
            let frame = motor.update(None, held, cfg);
            assert_eq!(frame.anim, CharacterMotorAnim::Walk);
            assert!(!frame.sprinting);
        }

        let released = motor.update(
            None,
            CharacterMotorInput {
                walk: 1,
                sprint: false,
                ..CharacterMotorInput::default()
            },
            cfg,
        );
        assert_eq!(released.anim, CharacterMotorAnim::Walk);

        let restarted = motor.update(None, held, cfg);
        assert_eq!(restarted.anim, CharacterMotorAnim::Run);
        assert!(restarted.sprinting);
    }

    #[test]
    fn vblank_delta_matches_repeated_single_frame_updates() {
        let cfg = config();
        let input = CharacterMotorInput {
            walk: 1,
            sprint: true,
            ..CharacterMotorInput::default()
        };
        let mut stepped = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let mut caught_up = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);

        let _ = stepped.update(None, input, cfg);
        let expected = stepped.update(None, input, cfg);
        let actual = caught_up.update_vblanks(None, input, cfg, 2);

        assert_eq!(actual.position, expected.position);
        assert_eq!(actual.yaw, expected.yaw);
        assert_eq!(actual.anim, expected.anim);
        assert_eq!(actual.stamina_q12, expected.stamina_q12);
        assert_eq!(caught_up.stamina_q12(), stepped.stamina_q12());
    }

    #[test]
    fn vblank_delta_consumes_evade_edge_once() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let mut cfg = config();
        cfg.roll_cost_q12 = 512;
        cfg.roll_speed = 0;
        cfg.roll_active_frames = 1;
        cfg.roll_recovery_frames = 0;
        cfg.roll_invulnerable_frames = 1;
        cfg.stamina_recover_q12 = 0;
        motor.stamina_q12 = 1024;

        let frame = motor.update_vblanks(
            None,
            CharacterMotorInput {
                walk: 1,
                evade: true,
                ..CharacterMotorInput::default()
            },
            cfg,
            2,
        );

        assert_eq!(frame.anim, CharacterMotorAnim::Walk);
        assert_eq!(frame.action, CharacterMotorAction::Idle);
        assert_eq!(frame.stamina_q12, 512);
    }

    #[test]
    fn evade_starts_roll_with_invulnerability() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                walk: 1,
                evade: true,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.action, CharacterMotorAction::Roll);
        assert_eq!(frame.anim, CharacterMotorAnim::Roll);
        assert!(frame.invulnerable);
        assert_eq!(frame.position, RoomPoint::new(0, 0, 96));
    }

    #[test]
    fn backwards_evade_starts_backstep() {
        let mut motor = CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO);
        let frame = motor.update(
            None,
            CharacterMotorInput {
                walk: -1,
                evade: true,
                ..CharacterMotorInput::default()
            },
            config(),
        );
        assert_eq!(frame.action, CharacterMotorAction::Backstep);
        assert_eq!(frame.anim, CharacterMotorAnim::Backstep);
        assert_eq!(frame.position, RoomPoint::new(0, 0, -72));
    }
}
