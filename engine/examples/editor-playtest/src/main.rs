//! `editor-playtest` — render a level cooked from the editor.
//!
//! Loads `generated/level_manifest.rs` (a Rust source file the
//! editor's playtest compiler produces via
//! [`psxed_project::playtest::write_package`]) containing:
//!
//! * a master [`LevelAssetRecord`] table — every cooked
//!   `.psxw` room blob and `.psxt` texture blob is a record;
//! * per-room [`LevelMaterialRecord`]s mapping each cooked
//!   local material slot to a texture asset id;
//! * per-room [`RoomResidencyRecord`]s declaring required
//!   RAM/VRAM assets;
//! * a [`PlayerSpawnRecord`] and [`EntityRecord`]s.
//!
//! The runtime resolves the active room by walking `ASSETS`,
//! uploads its texture assets through a tiny no-alloc
//! [`ResidencyManager`], builds a `TextureMaterial` table from
//! the room's material slice, and renders. No hardcoded starter
//! textures — the asset table is the source of truth.
//!
//! Controls (free-orbit toggled with SELECT):
//! * D-pad LEFT / RIGHT — yaw camera (orbit) or player (tank).
//! * D-pad UP / DOWN    — zoom (orbit) or walk (tank).

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Texture, World as AssetWorld};
use psx_engine::{
    button, draw_room, App, Config, Ctx, DepthBand, DepthRange, OtFrame, PrimitiveArena,
    RuntimeRoom, Scene, WorldCamera, WorldProjection, WorldRenderPass, WorldSurfaceOptions,
    WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_gte::transform::{cos_1_3_12, sin_1_3_12};
use psx_level::{
    find_asset_of_kind, AssetId, AssetKind, EntityRecord, LevelMaterialRecord, LevelRoomRecord,
    ResidencyManager,
};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

// Placeholder manifests reference unused statics; populated
// manifests reference all of them. Quiet either side here.
#[allow(dead_code, unused_imports)]
mod generated {
    include!("../generated/level_manifest.rs");
}

use generated::{ASSETS, ENTITIES, MATERIALS, PLAYER_SPAWN, ROOMS, ROOM_RESIDENCY};

// VRAM layout. We reserve one tpage at (640, 0) for textured
// materials and stripe textures left-to-right inside it. Each
// material gets one CLUT row above the framebuffer (y >= 480).
// Both ranges are deliberately generous for the per-room cap of
// `MAX_ROOM_MATERIALS`.
const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
/// First CLUT row used by uploaded textures. Row N below this
/// belongs to material N's CLUT.
const CLUT_BASE_Y: u16 = 480;

const SCREEN_CX: i16 = 160;
const SCREEN_CY: i16 = 120;
const FOCAL: i32 = 320;
const NEAR_Z: i32 = 64;
const FAR_Z: i32 = 8192;
const PROJECTION: WorldProjection = WorldProjection::new(SCREEN_CX, SCREEN_CY, FOCAL, NEAR_Z);

const CAMERA_Y_OFFSET: i32 = 1100;
const CAMERA_START_RADIUS: i32 = 2400;
const CAMERA_RADIUS_MIN: i32 = 800;
const CAMERA_RADIUS_MAX: i32 = 5200;
const CAMERA_RADIUS_STEP: i32 = 64;
const CAMERA_START_YAW: u16 = 220;
const CAMERA_YAW_STEP: u16 = 12;

const HALF_TURN_Q12: u16 = 2048;
const FOLLOW_RADIUS: i32 = 1400;
const FOLLOW_HEIGHT: i32 = 700;
const PLAYER_SPEED: i32 = 32;
const PLAYER_YAW_STEP: u16 = 32;

const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);

const MAX_TEXTURED_TRIS: usize = 4096;

/// Cap on the per-room material slot count. Picked to comfortably
/// exceed the cooker's currently-emitted material count without
/// over-reserving VRAM or RAM. If a future room exceeds this,
/// the runtime fails graceful (skips the over-cap material) and
/// the cook report should also flag.
const MAX_ROOM_MATERIALS: usize = 32;

/// Capacity of the residency manager's RAM table.
const MAX_RESIDENT_RAM_ASSETS: usize = 64;
/// Capacity of the residency manager's VRAM table.
const MAX_RESIDENT_VRAM_ASSETS: usize = 32;

/// Marker visualization tuning. Markers are debug stubs — keep
/// them visible at orbit-camera scales without dominating the
/// scene.
const MARKER_HALF: i32 = 96;
const MARKER_LIFT: i32 = MARKER_HALF;
const MARKER_TINT: (u8, u8, u8) = (0xff, 0xa8, 0x40);

const TRI_ZERO: TriTextured = TriTextured::new(
    [(0, 0), (0, 0), (0, 0)],
    [(0, 0), (0, 0), (0, 0)],
    0,
    0,
    (0, 0, 0),
);

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut TEXTURED_TRIS: [TriTextured; MAX_TEXTURED_TRIS] =
    [const { TRI_ZERO }; MAX_TEXTURED_TRIS];
static mut WORLD_COMMANDS: [WorldTriCommand; MAX_TEXTURED_TRIS] =
    [WorldTriCommand::EMPTY; MAX_TEXTURED_TRIS];

/// Residency manager — tracks which AssetIds are RAM/VRAM
/// resident across frames. Static so it survives across the
/// `Scene::init` → `Scene::render` boundary.
static mut RESIDENCY: ResidencyManager<MAX_RESIDENT_RAM_ASSETS, MAX_RESIDENT_VRAM_ASSETS> =
    ResidencyManager::new();

/// Per-asset upload bookkeeping. When a texture asset becomes
/// VRAM-resident we record its CLUT word and tpage half-x stride
/// so the per-frame material build can reconstruct its
/// `TextureMaterial` without re-walking the upload code.
#[derive(Copy, Clone)]
struct VramSlot {
    asset: AssetId,
    clut_word: u16,
    tpage_word: u16,
}

const VRAM_SLOT_EMPTY: Option<VramSlot> = None;
static mut VRAM_SLOTS: [Option<VramSlot>; MAX_RESIDENT_VRAM_ASSETS] =
    [VRAM_SLOT_EMPTY; MAX_RESIDENT_VRAM_ASSETS];
/// Number of VRAM slots used so far (next CLUT row + tpage cursor).
static mut VRAM_SLOT_COUNT: usize = 0;
/// Tpage X cursor (in halfwords). Each uploaded texture advances
/// it by `halfwords_per_row`.
static mut TPAGE_X_CURSOR: u16 = 0;

struct Playtest {
    /// Active room. `None` until `init` runs and only `Some`
    /// when the manifest had at least one room and its bytes
    /// parsed.
    room: Option<RuntimeRoom<'static>>,
    /// Active room's material table, ordered by `local_slot`.
    /// Indexed directly by the slot value the cooked `.psxw`
    /// stores per face.
    materials: [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
    /// `materials[..material_count]` is the in-use slice; rest
    /// is `None`.
    material_count: usize,
    /// Player position in room-local engine units.
    player_x: i32,
    player_y: i32,
    player_z: i32,
    player_yaw: u16,
    /// `true` toggles a free-orbit camera around the spawn for
    /// debug inspection. Default = follow.
    free_orbit: bool,
    orbit_yaw: u16,
    orbit_radius: i32,
    /// Spawn position retained for orbit-mode targeting.
    spawn: WorldVertex,
}

impl Playtest {
    const fn new() -> Self {
        Self {
            room: None,
            materials: [const { None }; MAX_ROOM_MATERIALS],
            material_count: 0,
            player_x: 0,
            player_y: 0,
            player_z: 0,
            player_yaw: 0,
            free_orbit: false,
            orbit_yaw: CAMERA_START_YAW,
            orbit_radius: CAMERA_START_RADIUS,
            spawn: WorldVertex::ZERO,
        }
    }
}

impl Scene for Playtest {
    fn init(&mut self, _ctx: &mut Ctx) {
        // Empty manifest? Boot to a clear-coloured screen.
        let Some(room_record) = ROOMS.first() else {
            return;
        };

        // Walk the residency contract for this room. Required
        // RAM assets are logical-only (every asset is
        // include_bytes!-resident from process start), but we
        // still tick them through the manager so the change-set
        // counts are honest. Required VRAM assets we'll need
        // textures for — actual uploads happen below.
        let residency_record = ROOM_RESIDENCY
            .iter()
            .find(|r| r.room == 0)
            .expect("starter room has a residency record");
        let _ = unsafe { RESIDENCY.ensure_room_resident(residency_record) };

        // Resolve and parse the room's world bytes.
        let world_asset =
            find_asset_of_kind(ASSETS, room_record.world_asset, AssetKind::RoomWorld);
        if let Some(asset) = world_asset {
            if let Ok(world) = AssetWorld::from_bytes(asset.bytes) {
                self.room = Some(RuntimeRoom::from_world(world));
            }
        }

        // Build the material table by walking this room's slice
        // of MATERIALS. For each entry: ensure VRAM-resident
        // (uploading on first sight), then build the
        // TextureMaterial referencing the slot's CLUT/tpage.
        self.material_count = build_room_materials(room_record, &mut self.materials);

        self.player_x = PLAYER_SPAWN.x;
        self.player_y = PLAYER_SPAWN.y;
        self.player_z = PLAYER_SPAWN.z;
        self.player_yaw = PLAYER_SPAWN.yaw as u16;
        self.spawn = WorldVertex::new(PLAYER_SPAWN.x, PLAYER_SPAWN.y, PLAYER_SPAWN.z);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::SELECT) {
            self.free_orbit = !self.free_orbit;
        }
        if self.free_orbit {
            if ctx.is_held(button::RIGHT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_add(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_sub(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::UP) {
                self.orbit_radius =
                    (self.orbit_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
            }
            if ctx.is_held(button::DOWN) {
                self.orbit_radius =
                    (self.orbit_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
            }
        } else {
            if ctx.is_held(button::RIGHT) {
                self.player_yaw = self.player_yaw.wrapping_add(PLAYER_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.player_yaw = self.player_yaw.wrapping_sub(PLAYER_YAW_STEP);
            }
            let sin_y = sin_1_3_12(self.player_yaw) as i32;
            let cos_y = cos_1_3_12(self.player_yaw) as i32;
            if ctx.is_held(button::UP) {
                self.player_x += (sin_y * PLAYER_SPEED) >> 12;
                self.player_z += (cos_y * PLAYER_SPEED) >> 12;
            }
            if ctx.is_held(button::DOWN) {
                self.player_x -= (sin_y * PLAYER_SPEED) >> 12;
                self.player_z -= (cos_y * PLAYER_SPEED) >> 12;
            }
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        let camera = if self.free_orbit {
            WorldCamera::orbit_yaw(
                PROJECTION,
                self.spawn,
                CAMERA_Y_OFFSET,
                self.orbit_radius,
                self.orbit_yaw,
            )
        } else {
            let target = WorldVertex::new(self.player_x, self.player_y, self.player_z);
            WorldCamera::orbit_yaw(
                PROJECTION,
                target,
                self.player_y + FOLLOW_HEIGHT,
                FOLLOW_RADIUS,
                self.player_yaw.wrapping_add(HALF_TURN_Q12),
            )
        };

        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };

        if let Some(room) = self.room {
            // Pack the materials slice down to a contiguous
            // `&[TextureMaterial]` indexed by local_slot. Slots
            // that didn't resolve become a sentinel material —
            // visually obvious without crashing the renderer.
            let mut bound: [TextureMaterial; MAX_ROOM_MATERIALS] =
                [TextureMaterial::opaque(0, TPAGE_WORD, (0x80, 0x80, 0x80)); MAX_ROOM_MATERIALS];
            for i in 0..self.material_count {
                if let Some(m) = self.materials[i] {
                    bound[i] = m;
                }
            }
            let materials = &bound[..self.material_count];
            let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE);
            draw_room(
                room.render(),
                materials,
                &camera,
                options,
                &mut triangles,
                &mut world,
            );
            draw_entity_markers(ENTITIES, materials, &camera, options, &mut triangles, &mut world);
        }

        world.flush();
        ot.submit();
    }
}

/// Walk `room.material_first..material_first + material_count`,
/// resolve each material's texture asset, and build a
/// TextureMaterial in `out` indexed by `local_slot`. Each
/// texture asset is uploaded at most once across the program
/// lifetime — the residency manager + VRAM_SLOTS tracks who's
/// already up.
///
/// Returns the highest `local_slot + 1` so the caller knows the
/// in-use prefix length.
fn build_room_materials(
    room: &LevelRoomRecord,
    out: &mut [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
) -> usize {
    let first = room.material_first as usize;
    let count = room.material_count as usize;
    let slice: &[LevelMaterialRecord] = &MATERIALS[first..first + count];

    let mut max_slot: usize = 0;
    for material in slice {
        let slot = material.local_slot as usize;
        if slot >= MAX_ROOM_MATERIALS {
            // Cooker should already have failed validation;
            // skip rather than crash if it slips through.
            continue;
        }
        let Some(asset) =
            find_asset_of_kind(ASSETS, material.texture_asset, AssetKind::Texture)
        else {
            continue;
        };
        let Some(slot_record) = ensure_texture_uploaded(asset.id, asset.bytes) else {
            continue;
        };
        out[slot] = Some(TextureMaterial::opaque(
            slot_record.clut_word,
            slot_record.tpage_word,
            (
                material.tint_rgb[0],
                material.tint_rgb[1],
                material.tint_rgb[2],
            ),
        ));
        if slot + 1 > max_slot {
            max_slot = slot + 1;
        }
    }
    max_slot
}

/// Upload `asset_bytes` to VRAM if not already resident; return
/// the slot record so the caller can build a TextureMaterial.
/// Returns `None` if the texture parse fails or the VRAM table
/// is full.
fn ensure_texture_uploaded(asset_id: AssetId, asset_bytes: &[u8]) -> Option<VramSlot> {
    // Already uploaded? Look up the slot record.
    if unsafe { RESIDENCY.contains_vram(asset_id) } {
        return unsafe {
            VRAM_SLOTS
                .iter()
                .filter_map(|s| *s)
                .find(|s| s.asset == asset_id)
        };
    }

    let texture = Texture::from_bytes(asset_bytes).ok()?;

    // Capacity check before we touch any VRAM state.
    let count = unsafe { VRAM_SLOT_COUNT };
    if count >= MAX_RESIDENT_VRAM_ASSETS {
        return None;
    }

    // Pick the next CLUT row and tpage stride offset, then
    // upload pixels + CLUT.
    let clut_y = CLUT_BASE_Y - count as u16;
    let tpage_x = SHARED_TPAGE.x() + unsafe { TPAGE_X_CURSOR };

    let pix_rect = VramRect::new(tpage_x, SHARED_TPAGE.y(), texture.halfwords_per_row(), texture.height());
    upload_bytes(pix_rect, texture.pixel_bytes());

    let clut_rect = VramRect::new(0, clut_y, texture.clut_entries(), 1);
    upload_clut(clut_rect, texture.clut_bytes());

    let clut = Clut::new(0, clut_y);
    let slot = VramSlot {
        asset: asset_id,
        clut_word: clut.uv_clut_word(),
        tpage_word: TPAGE_WORD,
    };

    unsafe {
        VRAM_SLOTS[count] = Some(slot);
        VRAM_SLOT_COUNT = count + 1;
        TPAGE_X_CURSOR += texture.halfwords_per_row();
        // Mirror VRAM into the residency tracker. mark_vram_resident
        // returns false if it overflows; we already reserved a
        // slot so this should always succeed.
        let _ = RESIDENCY.mark_vram_resident(asset_id);
    }

    Some(slot)
}

/// Draw one tinted cube per generated entity record. Cubes
/// reuse the room's first material with an override tint so
/// markers stand out from the surrounding geometry without
/// needing a dedicated texture upload.
fn draw_entity_markers(
    entities: &[EntityRecord],
    materials: &[TextureMaterial],
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) {
    if entities.is_empty() || materials.is_empty() {
        return;
    }
    // Reuse the room's first material so we don't need a
    // dedicated marker texture. Tint override picks up the
    // existing CLUT + tpage but recolours.
    let base = materials[0];
    let material = TextureMaterial::opaque(
        material_clut_word(base),
        material_tpage_word(base),
        MARKER_TINT,
    );
    let opts = options.with_material_layer(material);
    const UVS: [(u8, u8); 4] = [(0, 0), (64, 0), (64, 64), (0, 64)];

    for entity in entities {
        let cx = entity.x;
        let cy = entity.y - MARKER_LIFT - MARKER_HALF;
        let cz = entity.z;
        let h = MARKER_HALF;

        let top = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
        ];
        let bottom = [
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        let north = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        let south = [
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
        ];
        let east = [
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
        ];
        let west = [
            WorldVertex::new(cx - h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz + h),
        ];

        for face in [top, bottom, north, south, east, west] {
            if let Some(projected) = camera.project_world_quad(face) {
                let _ = world.submit_textured_quad(triangles, projected, UVS, material, opts);
            }
        }
    }
}

/// Extract a `TextureMaterial`'s CLUT word for re-tinting.
/// `TextureMaterial`'s fields are private, so we reconstruct
/// via a helper that round-trips through the public API.
fn material_clut_word(_m: TextureMaterial) -> u16 {
    // The marker reuses material[0]'s CLUT/tpage. We don't have
    // a public getter, so the cleanest workaround is to use the
    // first uploaded slot's CLUT word from VRAM_SLOTS, which is
    // stable across the program's lifetime. Marker code only
    // runs after `init` has populated at least one slot.
    unsafe {
        VRAM_SLOTS
            .iter()
            .find_map(|s| *s)
            .map(|s| s.clut_word)
            .unwrap_or(0)
    }
}

fn material_tpage_word(_m: TextureMaterial) -> u16 {
    TPAGE_WORD
}

#[no_mangle]
fn main() -> ! {
    let mut scene = Playtest::new();
    let config = Config {
        clear_color: (5, 7, 12),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

/// Stamp the 0x8000 (semi-transparency-disable) bit on every
/// non-zero CLUT entry so opaque textures don't accidentally
/// trigger STP-bit blending.
fn upload_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    if bytes.len() > marked.len() || !bytes.len().is_multiple_of(2) {
        return;
    }

    let mut i = 0;
    while i < bytes.len() {
        let raw = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let stamped = if raw == 0 { 0 } else { raw | 0x8000 };
        let pair = stamped.to_le_bytes();
        marked[i] = pair[0];
        marked[i + 1] = pair[1];
        i += 2;
    }

    upload_bytes(rect, &marked[..bytes.len()]);
}
