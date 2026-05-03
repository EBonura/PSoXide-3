//! Lightweight guest-runtime telemetry for PSoXide host tooling.
//!
//! The runtime emits compact stage/counter events through an emulator-observed
//! Expansion 2 port. On non-MIPS host builds these functions compile to no-ops,
//! so editor-side preview code can depend on `psx-engine` without touching host
//! memory.

/// Runtime stage ids. Keep in sync with `emulator_core::telemetry::stage`.
pub mod stage {
    /// Per-frame gameplay/update work.
    pub const UPDATE: u16 = 1;
    /// Framebuffer clear before scene rendering.
    pub const FRAME_CLEAR: u16 = 2;
    /// Whole `Scene::render` call.
    pub const RENDER: u16 = 3;
    /// Present/vblank wait and framebuffer swap.
    pub const PRESENT: u16 = 4;
    /// Editor-playtest camera update.
    pub const CAMERA: u16 = 5;
    /// Grid-room surface rendering.
    pub const ROOM: u16 = 6;
    /// Legacy entity debug marker rendering.
    pub const ENTITY_MARKERS: u16 = 7;
    /// Placed model-instance rendering.
    pub const MODEL_INSTANCES: u16 = 8;
    /// Player model rendering.
    pub const PLAYER: u16 = 9;
    /// Player-attached equipment / weapon rendering and hit-volume evaluation.
    pub const EQUIPMENT: u16 = 12;
    /// Deferred world-command sort and OT insertion.
    pub const WORLD_FLUSH: u16 = 10;
    /// Ordering-table DMA submission.
    pub const OT_SUBMIT: u16 = 11;
}

/// Runtime counter ids. Keep in sync with `emulator_core::telemetry::counter`.
pub mod counter {
    /// Textured primitive packets allocated this frame.
    pub const TRI_PRIMITIVES: u16 = 1;
    /// World render commands queued before flush.
    pub const WORLD_COMMANDS: u16 = 2;
    /// Placed model instances drawn.
    pub const MODEL_INSTANCE_DRAWS: u16 = 3;
    /// Vertices projected for placed model instances.
    pub const MODEL_INSTANCE_PROJECTED_VERTICES: u16 = 4;
    /// Triangles submitted for placed model instances.
    pub const MODEL_INSTANCE_SUBMITTED_TRIS: u16 = 5;
    /// Triangles culled for placed model instances.
    pub const MODEL_INSTANCE_CULLED_TRIS: u16 = 6;
    /// Triangles dropped for placed model instances.
    pub const MODEL_INSTANCE_DROPPED_TRIS: u16 = 7;
    /// Vertices projected for the player model.
    pub const PLAYER_PROJECTED_VERTICES: u16 = 8;
    /// Triangles submitted for the player model.
    pub const PLAYER_SUBMITTED_TRIS: u16 = 9;
    /// Triangles culled for the player model.
    pub const PLAYER_CULLED_TRIS: u16 = 10;
    /// Triangles dropped for the player model.
    pub const PLAYER_DROPPED_TRIS: u16 = 11;
    /// Bitfield of model-render overflow flags observed this frame.
    pub const MODEL_OVERFLOW_FLAGS: u16 = 12;
    /// Non-empty room grid cells considered by the visibility pass.
    pub const ROOM_CELLS_CONSIDERED: u16 = 13;
    /// Room grid cells drawn after visibility culling.
    pub const ROOM_CELLS_DRAWN: u16 = 14;
    /// Room grid cells rejected by the coarse frustum test.
    pub const ROOM_CELLS_CULLED: u16 = 15;
    /// Room floor/ceiling/wall surfaces considered for projection.
    pub const ROOM_SURFACES_CONSIDERED: u16 = 16;
    /// Player-attached equipment visuals drawn.
    pub const EQUIPMENT_DRAWS: u16 = 17;
    /// Active weapon hitboxes this frame.
    pub const EQUIPMENT_ACTIVE_HITBOXES: u16 = 18;
    /// Entity marker hits found by active weapon hitboxes.
    pub const EQUIPMENT_TARGET_HITS: u16 = 19;
    /// Vertices projected for equipment models.
    pub const EQUIPMENT_PROJECTED_VERTICES: u16 = 20;
    /// Triangles submitted for equipment models.
    pub const EQUIPMENT_SUBMITTED_TRIS: u16 = 21;
    /// Triangles culled for equipment models.
    pub const EQUIPMENT_CULLED_TRIS: u16 = 22;
    /// Triangles dropped for equipment models.
    pub const EQUIPMENT_DROPPED_TRIS: u16 = 23;
}

const EVENT_KIND_FRAME_BEGIN: u8 = 1;
const EVENT_KIND_STAGE_BEGIN: u8 = 2;
const EVENT_KIND_STAGE_END: u8 = 3;
const EVENT_KIND_COUNTER: u8 = 4;

#[cfg(target_arch = "mips")]
const EVENT_ADDR: *mut u32 = 0xBF80_2F00 as *mut u32;
#[cfg(target_arch = "mips")]
const VALUE_ADDR: *mut u32 = 0xBF80_2F04 as *mut u32;

/// Mark the start of a guest frame.
#[inline(always)]
pub fn frame_begin(frame: u32) {
    emit_value(frame);
    emit_event(EVENT_KIND_FRAME_BEGIN, 0);
}

/// Mark the start of a named stage.
#[inline(always)]
pub fn stage_begin(stage_id: u16) {
    emit_event(EVENT_KIND_STAGE_BEGIN, stage_id);
}

/// Mark the end of a named stage.
#[inline(always)]
pub fn stage_end(stage_id: u16) {
    emit_event(EVENT_KIND_STAGE_END, stage_id);
}

/// Emit a numeric counter value.
#[inline(always)]
pub fn counter(counter_id: u16, value: u32) {
    emit_value(value);
    emit_event(EVENT_KIND_COUNTER, counter_id);
}

#[cfg(target_arch = "mips")]
#[inline(always)]
fn encode_event(kind: u8, id: u16) -> u32 {
    ((kind as u32) << 24) | id as u32
}

#[cfg(target_arch = "mips")]
#[inline(always)]
fn emit_value(value: u32) {
    unsafe {
        core::ptr::write_volatile(VALUE_ADDR, value);
    }
}

#[cfg(not(target_arch = "mips"))]
#[inline(always)]
fn emit_value(_value: u32) {}

#[cfg(target_arch = "mips")]
#[inline(always)]
fn emit_event(kind: u8, id: u16) {
    unsafe {
        core::ptr::write_volatile(EVENT_ADDR, encode_event(kind, id));
    }
}

#[cfg(not(target_arch = "mips"))]
#[inline(always)]
fn emit_event(_kind: u8, _id: u16) {}
