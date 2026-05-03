//! Guest-runtime telemetry captured out-of-band by the emulator.
//!
//! Instrumented homebrew writes compact event words into a reserved slice of
//! Expansion Region 2. Retail software sees the normal expansion-port
//! behaviour, while PSoXide can timestamp those writes with the emulator's bus
//! cycle counter and surface the result in host-side tooling.

use std::collections::VecDeque;

use psx_hw::memory;

/// Physical base of PSoXide's emulator-only telemetry port.
pub const BASE_PHYS: u32 = memory::expansion2::BASE + 0x0F00;
/// Event command register. A write appends one telemetry event.
pub const EVENT_PHYS: u32 = BASE_PHYS;
/// Event value register. The next command write snapshots this value.
pub const VALUE_PHYS: u32 = BASE_PHYS + 4;

const EVENT_CAP: usize = 8192;
const KIND_SHIFT: u32 = 24;
const KIND_MASK: u32 = 0xFF;
const ID_MASK: u32 = 0xFFFF;

/// Runtime stage id constants shared with `psx-engine::telemetry`.
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

/// Number of stage slots, including index zero for unknown/reserved ids.
pub const STAGE_COUNT: usize = 13;

/// Runtime counter id constants shared with `psx-engine::telemetry`.
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

/// Number of counter slots, including index zero for unknown/reserved ids.
pub const COUNTER_COUNT: usize = 24;

/// Telemetry event kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestTelemetryKind {
    /// A new guest frame began; `value` is the guest frame index.
    FrameBegin,
    /// A named runtime stage began.
    StageBegin,
    /// A named runtime stage ended.
    StageEnd,
    /// A numeric counter was emitted.
    Counter,
    /// Unknown event kind preserved for diagnostics.
    Unknown(u8),
}

impl GuestTelemetryKind {
    fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::FrameBegin,
            2 => Self::StageBegin,
            3 => Self::StageEnd,
            4 => Self::Counter,
            other => Self::Unknown(other),
        }
    }
}

/// One telemetry event timestamped by the emulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestTelemetryEvent {
    /// Bus cycles elapsed when the guest wrote the event command.
    pub cycles: u64,
    /// Event kind.
    pub kind: GuestTelemetryKind,
    /// Stage or counter id, depending on [`kind`](Self::kind).
    pub id: u16,
    /// Latched value from [`VALUE_PHYS`].
    pub value: u32,
}

/// Rolling capture buffer for guest telemetry events.
pub struct GuestTelemetry {
    pending_value: u32,
    events: VecDeque<GuestTelemetryEvent>,
    frames_seen: u64,
}

impl Default for GuestTelemetry {
    fn default() -> Self {
        Self {
            pending_value: 0,
            events: VecDeque::with_capacity(EVENT_CAP),
            frames_seen: 0,
        }
    }
}

impl GuestTelemetry {
    /// Create an empty telemetry buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// True if `phys` lands inside the telemetry port.
    pub const fn contains(phys: u32) -> bool {
        phys == EVENT_PHYS || phys == VALUE_PHYS
    }

    /// Observe a 32-bit write. Returns true if the telemetry port consumed it.
    pub fn observe_write32(&mut self, phys: u32, value: u32, cycles: u64) -> bool {
        match phys {
            VALUE_PHYS => {
                self.pending_value = value;
                true
            }
            EVENT_PHYS => {
                let raw_kind = ((value >> KIND_SHIFT) & KIND_MASK) as u8;
                let id = (value & ID_MASK) as u16;
                self.push(GuestTelemetryEvent {
                    cycles,
                    kind: GuestTelemetryKind::from_raw(raw_kind),
                    id,
                    value: self.pending_value,
                });
                true
            }
            _ => false,
        }
    }

    /// Drain all captured events in chronological order.
    pub fn drain_events(&mut self) -> Vec<GuestTelemetryEvent> {
        self.events.drain(..).collect()
    }

    /// Number of guest frame-begin markers observed since reset.
    pub const fn frames_seen(&self) -> u64 {
        self.frames_seen
    }

    fn push(&mut self, event: GuestTelemetryEvent) {
        if matches!(event.kind, GuestTelemetryKind::FrameBegin) {
            self.frames_seen = self.frames_seen.saturating_add(1);
        }
        if self.events.len() >= EVENT_CAP {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}

/// Aggregated guest telemetry over a span of events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestTelemetrySummary {
    /// Number of guest frame-begin markers observed.
    pub frames: u64,
    /// Total cycles spent in each known stage id.
    pub stage_cycles: [u64; STAGE_COUNT],
    /// Number of completed spans per known stage id.
    pub stage_hits: [u64; STAGE_COUNT],
    /// Summed counter values per known counter id.
    pub counters: [u64; COUNTER_COUNT],
}

impl Default for GuestTelemetrySummary {
    fn default() -> Self {
        Self {
            frames: 0,
            stage_cycles: [0; STAGE_COUNT],
            stage_hits: [0; STAGE_COUNT],
            counters: [0; COUNTER_COUNT],
        }
    }
}

impl GuestTelemetrySummary {
    /// Build a summary from raw telemetry events.
    pub fn from_events(events: &[GuestTelemetryEvent]) -> Self {
        let mut out = Self::default();
        out.add_events(events);
        out
    }

    /// Add raw events to this summary.
    pub fn add_events(&mut self, events: &[GuestTelemetryEvent]) {
        let mut stage_start: [Option<u64>; STAGE_COUNT] = [None; STAGE_COUNT];
        for event in events {
            match event.kind {
                GuestTelemetryKind::FrameBegin => {
                    self.frames = self.frames.saturating_add(1);
                }
                GuestTelemetryKind::StageBegin => {
                    if let Some(slot) = stage_start.get_mut(event.id as usize) {
                        *slot = Some(event.cycles);
                    }
                }
                GuestTelemetryKind::StageEnd => {
                    let Some(slot) = stage_start.get_mut(event.id as usize) else {
                        continue;
                    };
                    let Some(start) = slot.take() else {
                        continue;
                    };
                    let idx = event.id as usize;
                    self.stage_cycles[idx] =
                        self.stage_cycles[idx].saturating_add(event.cycles.saturating_sub(start));
                    self.stage_hits[idx] = self.stage_hits[idx].saturating_add(1);
                }
                GuestTelemetryKind::Counter => {
                    if let Some(counter) = self.counters.get_mut(event.id as usize) {
                        *counter = counter.saturating_add(event.value as u64);
                    }
                }
                GuestTelemetryKind::Unknown(_) => {}
            }
        }
    }

    /// True when at least one event contributed useful data.
    pub fn has_data(&self) -> bool {
        self.frames > 0
            || self.stage_cycles.iter().any(|&cycles| cycles > 0)
            || self.counters.iter().any(|&value| value > 0)
    }
}

/// Human-readable stage name for host tooling.
pub fn stage_name(id: u16) -> &'static str {
    match id {
        stage::UPDATE => "update",
        stage::FRAME_CLEAR => "frame clear",
        stage::RENDER => "render total",
        stage::PRESENT => "present/wait",
        stage::CAMERA => "camera",
        stage::ROOM => "room",
        stage::ENTITY_MARKERS => "entity markers",
        stage::MODEL_INSTANCES => "model instances",
        stage::PLAYER => "player",
        stage::EQUIPMENT => "equipment",
        stage::WORLD_FLUSH => "world flush/sort",
        stage::OT_SUBMIT => "ot submit",
        _ => "unknown",
    }
}

/// Human-readable counter name for host tooling.
pub fn counter_name(id: u16) -> &'static str {
    match id {
        counter::TRI_PRIMITIVES => "tri prims",
        counter::WORLD_COMMANDS => "world commands",
        counter::MODEL_INSTANCE_DRAWS => "model draws",
        counter::MODEL_INSTANCE_PROJECTED_VERTICES => "model verts",
        counter::MODEL_INSTANCE_SUBMITTED_TRIS => "model tris",
        counter::MODEL_INSTANCE_CULLED_TRIS => "model culled",
        counter::MODEL_INSTANCE_DROPPED_TRIS => "model dropped",
        counter::PLAYER_PROJECTED_VERTICES => "player verts",
        counter::PLAYER_SUBMITTED_TRIS => "player tris",
        counter::PLAYER_CULLED_TRIS => "player culled",
        counter::PLAYER_DROPPED_TRIS => "player dropped",
        counter::MODEL_OVERFLOW_FLAGS => "overflow flags",
        counter::ROOM_CELLS_CONSIDERED => "room cells",
        counter::ROOM_CELLS_DRAWN => "room cells drawn",
        counter::ROOM_CELLS_CULLED => "room cells culled",
        counter::ROOM_SURFACES_CONSIDERED => "room surfaces",
        counter::EQUIPMENT_DRAWS => "equipment draws",
        counter::EQUIPMENT_ACTIVE_HITBOXES => "weapon hitboxes",
        counter::EQUIPMENT_TARGET_HITS => "weapon hits",
        counter::EQUIPMENT_PROJECTED_VERTICES => "equipment verts",
        counter::EQUIPMENT_SUBMITTED_TRIS => "equipment tris",
        counter::EQUIPMENT_CULLED_TRIS => "equipment culled",
        counter::EQUIPMENT_DROPPED_TRIS => "equipment dropped",
        _ => "unknown",
    }
}

/// Encode a guest event command word.
pub const fn encode_event(kind: u8, id: u16) -> u32 {
    ((kind as u32) << KIND_SHIFT) | (id as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_port_latches_value_then_event() {
        let mut telemetry = GuestTelemetry::new();
        assert!(telemetry.observe_write32(VALUE_PHYS, 42, 100));
        assert!(telemetry.observe_write32(
            EVENT_PHYS,
            encode_event(4, counter::WORLD_COMMANDS),
            110
        ));

        let events = telemetry.drain_events();
        assert_eq!(telemetry.frames_seen(), 0);
        assert_eq!(
            events,
            [GuestTelemetryEvent {
                cycles: 110,
                kind: GuestTelemetryKind::Counter,
                id: counter::WORLD_COMMANDS,
                value: 42,
            }]
        );
    }

    #[test]
    fn summary_accumulates_stage_spans_and_counters() {
        let events = [
            GuestTelemetryEvent {
                cycles: 10,
                kind: GuestTelemetryKind::FrameBegin,
                id: 0,
                value: 7,
            },
            GuestTelemetryEvent {
                cycles: 20,
                kind: GuestTelemetryKind::StageBegin,
                id: stage::RENDER,
                value: 0,
            },
            GuestTelemetryEvent {
                cycles: 70,
                kind: GuestTelemetryKind::StageEnd,
                id: stage::RENDER,
                value: 0,
            },
            GuestTelemetryEvent {
                cycles: 80,
                kind: GuestTelemetryKind::Counter,
                id: counter::TRI_PRIMITIVES,
                value: 12,
            },
        ];
        let summary = GuestTelemetrySummary::from_events(&events);
        assert_eq!(summary.frames, 1);
        assert_eq!(summary.stage_cycles[stage::RENDER as usize], 50);
        assert_eq!(summary.stage_hits[stage::RENDER as usize], 1);
        assert_eq!(summary.counters[counter::TRI_PRIMITIVES as usize], 12);
    }
}
