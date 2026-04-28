//! Drawing helpers for cooked grid worlds.
//!
//! Walks a [`RoomRender`] and emits its floors / ceilings / walls
//! through [`WorldRenderPass::submit_textured_quad`]. Material slot
//! → runtime material is provided by the caller because the
//! current `.psxw` (VERSION 1) doesn't embed a material table.
//! See `docs/world-format-roadmap.md` for the future compact
//! format that will let this helper resolve materials itself.

use psx_gpu::{material::TextureMaterial, prim::TriTextured};

use crate::{
    render3d::CullMode, PrimitiveArena, RoomRender, WorldCamera, WorldRenderPass,
    WorldSurfaceOptions, WorldVertex,
};

/// Texture-page-relative tile size used by the helper. v1 records
/// don't carry per-face UV info, so floors / ceilings / walls
/// all UV-tile a single 64×64 patch over the quad. Per-face UVs
/// would land alongside the future compact material table — see
/// `docs/world-format-roadmap.md`.
const TILE_UV: u8 = 64;

/// Direction id for the north edge.
///
/// Mirrors `psxed_format::world::direction::NORTH` — kept inline
/// so `psx-engine` doesn't need a direct `psxed-format` dep
/// (it already reaches the format via `psx-asset`, but adding
/// the direct dep just for four byte constants is overkill).
const DIR_NORTH: u8 = 0;
const DIR_EAST: u8 = 1;
const DIR_SOUTH: u8 = 2;
const DIR_WEST: u8 = 3;

const FLOOR_UVS: [(u8, u8); 4] = [(0, 0), (TILE_UV, 0), (TILE_UV, TILE_UV), (0, TILE_UV)];
const WALL_UVS: [(u8, u8); 4] = [(0, TILE_UV), (TILE_UV, TILE_UV), (TILE_UV, 0), (0, 0)];

/// Walk every populated sector of `room`, emitting one textured
/// quad per floor / ceiling face plus one per wall.
///
/// `materials` is indexed by the slot ids returned from
/// [`SectorRender::floor_material`], [`SectorRender::ceiling_material`]
/// and [`WallRender::material`]. A face whose slot points past the
/// table is dropped silently — friendlier than a panic while the
/// author is mid-iteration with partially-assigned materials.
///
/// Cells are corner-rooted at world `(0, 0)`: cell `(sx, sz)`
/// occupies `x ∈ [sx*S, (sx+1)*S]`, `z ∈ [sz*S, (sz+1)*S]`.
/// Position the camera target at the room's centre — typically
/// `(W*S/2, 0, D*S/2)` — so the orbit lands on the geometry.
///
/// `options` carries the depth band + range; the helper flips
/// only `cull_mode` per face kind: floors and ceilings use
/// [`CullMode::None`], walls use [`CullMode::Back`].
///
/// # Quad corner conventions
///
/// All four-corner inputs to [`WorldRenderPass::submit_textured_quad`]
/// are emitted in perimeter order. The renderer splits along the
/// `0`–`2` diagonal (see `TEXTURED_QUAD_TRIANGLES` in `render3d.rs`),
/// so corner positions and UVs must agree on what `0`, `1`, `2`,
/// `3` mean.
///
/// * **Floors / ceilings** — `[NW, NE, SE, SW]`. UVs match in the
///   same order: `[(0,0), (T,0), (T,T), (0,T)]`.
/// * **Walls** — `[bottom-left, bottom-right, top-right, top-left]`,
///   measured from inside the cell looking at the wall. UVs match:
///   `[(0,T), (T,T), (T,0), (0,0)]`.
///
/// [`SectorRender::floor_material`]: crate::SectorRender::floor_material
/// [`SectorRender::ceiling_material`]: crate::SectorRender::ceiling_material
/// [`WallRender::material`]: crate::WallRender::material
pub fn draw_room<const OT: usize>(
    room: RoomRender<'_, '_>,
    materials: &[TextureMaterial],
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let sector_size = room.sector_size();
    for sx in 0..room.width() {
        for sz in 0..room.depth() {
            let Some(sector) = room.sector(sx, sz) else {
                continue;
            };

            if sector.has_floor() {
                if let Some(slot) = sector.floor_material() {
                    if let Some(&material) = materials.get(slot as usize) {
                        emit_floor(
                            sx,
                            sz,
                            sector_size,
                            sector.floor_heights(),
                            material,
                            camera,
                            options,
                            triangles,
                            world,
                        );
                    }
                }
            }

            if sector.has_ceiling() {
                if let Some(slot) = sector.ceiling_material() {
                    if let Some(&material) = materials.get(slot as usize) {
                        emit_ceiling(
                            sx,
                            sz,
                            sector_size,
                            sector.ceiling_heights(),
                            material,
                            camera,
                            options,
                            triangles,
                            world,
                        );
                    }
                }
            }

            let mut i = 0;
            while i < sector.wall_count() {
                if let Some(wall) = room.sector_wall(sector, i) {
                    if let Some(&material) = materials.get(wall.material() as usize) {
                        emit_wall(
                            sx,
                            sz,
                            sector_size,
                            wall.direction(),
                            wall.heights(),
                            material,
                            camera,
                            options,
                            triangles,
                            world,
                        );
                    }
                }
                i += 1;
            }
        }
    }
}

/// Emit one floor quad. Corners feed the renderer in
/// `[NW, NE, SE, SW]` order, which matches the heights array
/// the cooker stores.
fn emit_floor<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    heights: [i32; 4],
    material: TextureMaterial,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let verts = [
        WorldVertex::new(x0, heights[0], z0),
        WorldVertex::new(x1, heights[1], z0),
        WorldVertex::new(x1, heights[2], z1),
        WorldVertex::new(x0, heights[3], z1),
    ];
    submit_quad(camera, options, CullMode::None, material, verts, FLOOR_UVS, triangles, world);
}

/// Emit one ceiling quad. Same `[NW, NE, SE, SW]` height order
/// as the floor — culling is left off so the inward face draws
/// regardless of winding orientation. Once the cooker emits
/// inward-facing winding consistently, this can flip to
/// `CullMode::Back` for free.
fn emit_ceiling<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    heights: [i32; 4],
    material: TextureMaterial,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let verts = [
        WorldVertex::new(x0, heights[0], z0),
        WorldVertex::new(x1, heights[1], z0),
        WorldVertex::new(x1, heights[2], z1),
        WorldVertex::new(x0, heights[3], z1),
    ];
    submit_quad(camera, options, CullMode::None, material, verts, FLOOR_UVS, triangles, world);
}

/// Emit one wall quad. Wall heights `[BL, BR, TR, TL]` map onto
/// the cell's edge endpoints by direction. Diagonal directions
/// (4 / 5) silently drop — the cooker rejects them, so they
/// shouldn't appear, but skipping is cheaper than panicking.
fn emit_wall<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    direction: u8,
    heights: [i32; 4],
    material: TextureMaterial,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let verts = match direction {
        DIR_NORTH => [
            WorldVertex::new(x0, heights[0], z0),
            WorldVertex::new(x1, heights[1], z0),
            WorldVertex::new(x1, heights[2], z0),
            WorldVertex::new(x0, heights[3], z0),
        ],
        DIR_EAST => [
            WorldVertex::new(x1, heights[0], z0),
            WorldVertex::new(x1, heights[1], z1),
            WorldVertex::new(x1, heights[2], z1),
            WorldVertex::new(x1, heights[3], z0),
        ],
        DIR_SOUTH => [
            WorldVertex::new(x1, heights[0], z1),
            WorldVertex::new(x0, heights[1], z1),
            WorldVertex::new(x0, heights[2], z1),
            WorldVertex::new(x1, heights[3], z1),
        ],
        DIR_WEST => [
            WorldVertex::new(x0, heights[0], z1),
            WorldVertex::new(x0, heights[1], z0),
            WorldVertex::new(x0, heights[2], z0),
            WorldVertex::new(x0, heights[3], z1),
        ],
        _ => return,
    };
    submit_quad(camera, options, CullMode::Back, material, verts, WALL_UVS, triangles, world);
}

/// Project + submit one textured quad. Splits the shared
/// boilerplate between floor / ceiling / wall paths.
fn submit_quad<const OT: usize>(
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    cull: CullMode,
    material: TextureMaterial,
    verts: [WorldVertex; 4],
    uvs: [(u8, u8); 4],
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let Some(projected) = camera.project_world_quad(verts) else {
        return;
    };
    let opts = options.with_cull_mode(cull).with_material_layer(material);
    let _ = world.submit_textured_quad(triangles, projected, uvs, material, opts);
}

/// World-space bounds of a sector cell rooted at world `(0, 0)`.
/// Returns `(x0, x1, z0, z1)` so individual quads can pick the
/// corners they need by index.
const fn cell_bounds(sx: u16, sz: u16, sector_size: i32) -> (i32, i32, i32, i32) {
    let x0 = (sx as i32) * sector_size;
    let x1 = ((sx as i32) + 1) * sector_size;
    let z0 = (sz as i32) * sector_size;
    let z1 = ((sz as i32) + 1) * sector_size;
    (x0, x1, z0, z1)
}
