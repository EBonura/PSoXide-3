//! Drawing helpers for cooked grid worlds.
//!
//! Walks a [`RoomRender`] and emits its floors / ceilings / walls
//! through [`WorldRenderPass::submit_textured_quad`]. Material slot
//! → runtime material is provided by the caller because the
//! current `.psxw` (VERSION 2) doesn't embed a material table.
//! See `docs/world-format-roadmap.md` for the future compact
//! format that will let this helper resolve materials itself.

use psx_gpu::{material::TextureMaterial, prim::TriTextured};

use crate::{
    render3d::CullMode, PrimitiveArena, RoomPoint, RoomRender, WorldCamera, WorldRenderPass,
    WorldSurfaceOptions, WorldVertex,
};

/// Which side(s) of a room face should render.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceSidedness {
    /// Authored/front winding only.
    Front,
    /// Opposite winding only.
    Back,
    /// No winding cull.
    Both,
}

/// Runtime material binding for cooked room geometry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldRenderMaterial {
    /// GPU texture/material state.
    pub texture: TextureMaterial,
    /// Face-sidedness policy.
    pub sidedness: SurfaceSidedness,
}

impl WorldRenderMaterial {
    /// Build a front-sided material.
    pub const fn front(texture: TextureMaterial) -> Self {
        Self {
            texture,
            sidedness: SurfaceSidedness::Front,
        }
    }

    /// Build a back-sided material.
    pub const fn back(texture: TextureMaterial) -> Self {
        Self {
            texture,
            sidedness: SurfaceSidedness::Back,
        }
    }

    /// Build a double-sided material.
    pub const fn both(texture: TextureMaterial) -> Self {
        Self {
            texture,
            sidedness: SurfaceSidedness::Both,
        }
    }

    /// Return a copy with the same texture state and sidedness but
    /// a different flat RGB tint.
    pub const fn with_tint(mut self, tint: (u8, u8, u8)) -> Self {
        self.texture = self.texture.with_tint(tint);
        self
    }
}

impl From<TextureMaterial> for WorldRenderMaterial {
    fn from(texture: TextureMaterial) -> Self {
        Self::front(texture)
    }
}

/// Kind of room surface currently being emitted.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorldSurfaceKind {
    /// Sector floor.
    Floor,
    /// Sector ceiling.
    Ceiling,
    /// Sector wall on a runtime cardinal edge.
    Wall {
        /// Runtime wall direction id.
        direction: u8,
    },
}

/// Per-surface data exposed to a room lighting/material pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldSurfaceSample {
    /// Surface kind.
    pub kind: WorldSurfaceKind,
    /// Sector X coordinate.
    pub sx: u16,
    /// Sector Z coordinate.
    pub sz: u16,
    /// Surface centre in the same room-local world coordinates as
    /// the emitted vertices.
    pub center: RoomPoint,
}

/// Coarse grid visibility settings for room rendering.
///
/// This is intentionally cell-based rather than triangle-based: the
/// renderer can reject whole authored sectors before it walks their
/// floor/wall records. `radius_cells` bounds traversal around an
/// anchor such as the player, while the camera test rejects cells that
/// are outside the current view cone.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GridVisibility {
    /// Runtime room-space anchor, usually the player root.
    pub anchor: RoomPoint,
    /// Maximum Chebyshev distance from `anchor` in grid cells.
    pub radius_cells: u16,
    /// Extra projected-pixel margin around the viewport. A non-zero
    /// margin avoids visible popping when a large cell straddles the
    /// frustum edge.
    pub screen_margin: i32,
}

impl GridVisibility {
    /// Build a conservative grid visibility window around an anchor.
    pub const fn around(anchor: RoomPoint, radius_cells: u16) -> Self {
        Self {
            anchor,
            radius_cells,
            screen_margin: 48,
        }
    }

    /// Return a copy with a different projected screen margin.
    pub const fn with_screen_margin(mut self, margin: i32) -> Self {
        self.screen_margin = margin;
        self
    }
}

/// Runtime counters from a grid-visible room draw.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GridVisibilityStats {
    /// Non-empty cells considered inside the traversal radius.
    pub cells_considered: u16,
    /// Cells rejected by the coarse camera-space bounds test.
    pub cells_frustum_culled: u16,
    /// Cells that reached surface emission.
    pub cells_drawn: u16,
    /// Floor/ceiling/wall surfaces handed to the projection path.
    pub surfaces_considered: u16,
}

/// Hook used by [`draw_room_lit`] to vary material tint per room
/// surface.
pub trait WorldSurfaceLighting {
    /// Shade one material for one room surface.
    fn shade(
        &self,
        sample: WorldSurfaceSample,
        material: WorldRenderMaterial,
    ) -> WorldRenderMaterial;
}

/// No-op surface lighting.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NoWorldSurfaceLighting;

impl WorldSurfaceLighting for NoWorldSurfaceLighting {
    fn shade(
        &self,
        _sample: WorldSurfaceSample,
        material: WorldRenderMaterial,
    ) -> WorldRenderMaterial {
        material
    }
}

/// Floor / ceiling split id for the standard NW→SE diagonal --
/// the value the cooker stamps when no rotation has been
/// authored. Mirrors `psxed_format::world::split::NORTH_WEST_SOUTH_EAST`.
/// Used by tests to spell the split id explicitly; runtime
/// emission falls through to this case for any non-`SPLIT_NE_SW`
/// id.
#[cfg(test)]
const SPLIT_NW_SE: u8 = 0;
/// Alternate split id (NE→SW diagonal). Mirrors
/// `psxed_format::world::split::NORTH_EAST_SOUTH_WEST`.
const SPLIT_NE_SW: u8 = 1;

/// Texture-page-relative tile size used by legacy v1 helper tests.
#[cfg(test)]
const TILE_UV: u8 = 64;

/// Direction id for the north edge.
///
/// Mirrors `psxed_format::world::direction::NORTH` -- kept inline
/// so `psx-engine` doesn't need a direct `psxed-format` dep
/// (it already reaches the format via `psx-asset`, but adding
/// the direct dep just for four byte constants is overkill).
const DIR_NORTH: u8 = 0;
const DIR_EAST: u8 = 1;
const DIR_SOUTH: u8 = 2;
const DIR_WEST: u8 = 3;

#[cfg(test)]
const WALL_UVS: [(u8, u8); 4] = [(0, TILE_UV), (TILE_UV, TILE_UV), (TILE_UV, 0), (0, 0)];

/// Walk every populated sector of `room`, emitting one textured
/// quad per floor / ceiling face plus one per wall.
///
/// `materials` is indexed by the slot ids returned from
/// [`SectorRender::floor_material`], [`SectorRender::ceiling_material`]
/// and [`WallRender::material`]. A face whose slot points past the
/// table is dropped silently -- friendlier than a panic while the
/// author is mid-iteration with partially-assigned materials.
///
/// Cells are corner-rooted at world `(0, 0)`: cell `(sx, sz)`
/// occupies `x ∈ [sx*S, (sx+1)*S]`, `z ∈ [sz*S, (sz+1)*S]`.
/// Position the camera target at the room's centre -- typically
/// `(W*S/2, 0, D*S/2)` -- so the orbit lands on the geometry.
///
/// `options` carries the depth band + range. Per-material
/// [`SurfaceSidedness`] selects front-only, back-only, or
/// double-sided emission; front-sided faces use [`CullMode::Back`].
///
/// # Quad corner conventions
///
/// All four-corner inputs to [`WorldRenderPass::submit_textured_quad`]
/// are emitted in perimeter order. The renderer splits along the
/// `0`–`2` diagonal (see `TEXTURED_QUAD_TRIANGLES` in `render3d.rs`),
/// so corner positions and UVs must agree on what `0`, `1`, `2`,
/// `3` mean.
///
/// * **Floors / ceilings** -- records store `[NW, NE, SE, SW]`.
///   Floors keep that top-facing winding; ceilings flip to the
///   inward underside winding. UVs are transformed with the vertices.
/// * **Walls** -- runtime records store `[bottom-left, bottom-right,
///   top-right, top-left]` for an owning cell edge. The renderer flips
///   that to inward-facing winding before culling while keeping the same
///   physical UV mapping.
///
/// [`SectorRender::floor_material`]: crate::SectorRender::floor_material
/// [`SectorRender::ceiling_material`]: crate::SectorRender::ceiling_material
/// [`WallRender::material`]: crate::WallRender::material
pub fn draw_room<const OT: usize>(
    room: RoomRender<'_, '_>,
    materials: &[WorldRenderMaterial],
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    draw_room_lit(
        room,
        materials,
        &NoWorldSurfaceLighting,
        camera,
        options,
        triangles,
        world,
    );
}

/// Draw a room while giving the caller one material-shading hook per
/// emitted floor, ceiling, and wall surface.
#[allow(clippy::too_many_arguments)]
pub fn draw_room_lit<const OT: usize, L: WorldSurfaceLighting>(
    room: RoomRender<'_, '_>,
    materials: &[WorldRenderMaterial],
    lighting: &L,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    for sx in 0..room.width() {
        for sz in 0..room.depth() {
            let Some(sector) = room.sector(sx, sz) else {
                continue;
            };
            let _ = draw_sector_lit(
                room, sx, sz, sector, materials, lighting, camera, options, triangles, world,
            );
        }
    }
}

/// Draw a room through a coarse grid visibility pass.
///
/// Traversal is ring-ordered from farthest to nearest around
/// `visibility.anchor`, which gives bucketed ordering a stable coarse
/// back-to-front submission order before the PS1 ordering table handles
/// per-triangle depth buckets.
#[allow(clippy::too_many_arguments)]
pub fn draw_room_lit_grid_visible<const OT: usize, L: WorldSurfaceLighting>(
    room: RoomRender<'_, '_>,
    materials: &[WorldRenderMaterial],
    lighting: &L,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    visibility: GridVisibility,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) -> GridVisibilityStats {
    let mut stats = GridVisibilityStats::default();
    let width = room.width();
    let depth = room.depth();
    if width == 0 || depth == 0 {
        return stats;
    }

    let sector_size = room.sector_size().max(1);
    let anchor_x = grid_cell_for_world(visibility.anchor.x, sector_size).clamp(0, width as i32 - 1);
    let anchor_z = grid_cell_for_world(visibility.anchor.z, sector_size).clamp(0, depth as i32 - 1);
    let radius = visibility.radius_cells as i32;
    let min_x = (anchor_x - radius).max(0) as u16;
    let max_x = (anchor_x + radius).min(width as i32 - 1) as u16;
    let min_z = (anchor_z - radius).max(0) as u16;
    let max_z = (anchor_z + radius).min(depth as i32 - 1) as u16;

    let max_ring_x = (anchor_x - min_x as i32).max(max_x as i32 - anchor_x);
    let max_ring_z = (anchor_z - min_z as i32).max(max_z as i32 - anchor_z);
    let mut ring = max_ring_x.max(max_ring_z);
    loop {
        let mut sx = min_x;
        while sx <= max_x {
            let mut sz = min_z;
            while sz <= max_z {
                let dx = ((sx as i32) - anchor_x).abs();
                let dz = ((sz as i32) - anchor_z).abs();
                if dx.max(dz) == ring {
                    if let Some(sector) = room.sector(sx, sz) {
                        stats.cells_considered = stats.cells_considered.saturating_add(1);
                        let (min_y, max_y) = sector_y_bounds(room, sector);
                        if !cell_visible_to_camera(
                            camera,
                            options,
                            sx,
                            sz,
                            sector_size,
                            min_y,
                            max_y,
                            visibility.screen_margin,
                        ) {
                            stats.cells_frustum_culled =
                                stats.cells_frustum_culled.saturating_add(1);
                        } else {
                            stats.cells_drawn = stats.cells_drawn.saturating_add(1);
                            stats.surfaces_considered =
                                stats.surfaces_considered.saturating_add(draw_sector_lit(
                                    room, sx, sz, sector, materials, lighting, camera, options,
                                    triangles, world,
                                ));
                        }
                    }
                }
                if sz == max_z {
                    break;
                }
                sz += 1;
            }
            if sx == max_x {
                break;
            }
            sx += 1;
        }
        if ring == 0 {
            break;
        }
        ring -= 1;
    }

    stats
}

#[allow(clippy::too_many_arguments)]
fn draw_sector_lit<const OT: usize, L: WorldSurfaceLighting>(
    room: RoomRender<'_, '_>,
    sx: u16,
    sz: u16,
    sector: crate::SectorRender,
    materials: &[WorldRenderMaterial],
    lighting: &L,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) -> u16 {
    let sector_size = room.sector_size();
    let mut surfaces = 0u16;

    if sector.has_floor() {
        if let Some(slot) = sector.floor_material() {
            if let Some(&base_material) = materials.get(slot as usize) {
                let material = lighting.shade(
                    WorldSurfaceSample {
                        kind: WorldSurfaceKind::Floor,
                        sx,
                        sz,
                        center: horizontal_face_center(sx, sz, sector_size, sector.floor_heights()),
                    },
                    base_material,
                );
                surfaces = surfaces.saturating_add(1);
                emit_floor(
                    sx,
                    sz,
                    sector_size,
                    sector.floor_heights(),
                    sector.floor_split(),
                    sector.floor_uvs(),
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
            if let Some(&base_material) = materials.get(slot as usize) {
                let material = lighting.shade(
                    WorldSurfaceSample {
                        kind: WorldSurfaceKind::Ceiling,
                        sx,
                        sz,
                        center: horizontal_face_center(
                            sx,
                            sz,
                            sector_size,
                            sector.ceiling_heights(),
                        ),
                    },
                    base_material,
                );
                surfaces = surfaces.saturating_add(1);
                emit_ceiling(
                    sx,
                    sz,
                    sector_size,
                    sector.ceiling_heights(),
                    sector.ceiling_split(),
                    sector.ceiling_uvs(),
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
            if let Some(&base_material) = materials.get(wall.material() as usize) {
                let Some(center) =
                    wall_face_center(sx, sz, sector_size, wall.direction(), wall.heights())
                else {
                    i += 1;
                    continue;
                };
                let material = lighting.shade(
                    WorldSurfaceSample {
                        kind: WorldSurfaceKind::Wall {
                            direction: wall.direction(),
                        },
                        sx,
                        sz,
                        center,
                    },
                    base_material,
                );
                surfaces = surfaces.saturating_add(1);
                emit_wall(
                    sx,
                    sz,
                    sector_size,
                    wall.direction(),
                    wall.heights(),
                    wall.uvs(),
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

    surfaces
}

fn grid_cell_for_world(value: i32, sector_size: i32) -> i32 {
    if value >= 0 {
        value / sector_size
    } else {
        (value - sector_size + 1) / sector_size
    }
}

fn sector_y_bounds(room: RoomRender<'_, '_>, sector: crate::SectorRender) -> (i32, i32) {
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut any = false;

    if sector.has_floor() {
        include_heights(&mut min_y, &mut max_y, &mut any, sector.floor_heights());
    }
    if sector.has_ceiling() {
        include_heights(&mut min_y, &mut max_y, &mut any, sector.ceiling_heights());
    }

    let mut i = 0;
    while i < sector.wall_count() {
        if let Some(wall) = room.sector_wall(sector, i) {
            include_heights(&mut min_y, &mut max_y, &mut any, wall.heights());
        }
        i += 1;
    }

    if any {
        (min_y, max_y)
    } else {
        (0, room.sector_size())
    }
}

fn include_heights(min_y: &mut i32, max_y: &mut i32, any: &mut bool, heights: [i32; 4]) {
    let mut i = 0;
    while i < heights.len() {
        *min_y = (*min_y).min(heights[i]);
        *max_y = (*max_y).max(heights[i]);
        *any = true;
        i += 1;
    }
}

fn cell_visible_to_camera(
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    sx: u16,
    sz: u16,
    sector_size: i32,
    min_y: i32,
    max_y: i32,
    screen_margin: i32,
) -> bool {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let center = WorldVertex::new((x0 + x1) / 2, (min_y + max_y) / 2, (z0 + z1) / 2);
    let view = camera.view_vertex(center);
    let half_height = ((max_y - min_y).abs() / 2).max(sector_size / 2);
    let radius = sector_size.saturating_add(half_height);
    let near = camera.projection.near_z.max(1);
    let far = options.depth_range.far().max(near);
    if view.z < near.saturating_sub(radius) || view.z > far.saturating_add(radius) {
        return false;
    }

    let z = view.z.max(near);
    let focal = camera.projection.focal_length.max(1);
    let half_w = (camera.projection.screen_x as i32)
        .saturating_add(screen_margin)
        .max(1);
    let half_h = (camera.projection.screen_y as i32)
        .saturating_add(screen_margin)
        .max(1);
    let projected_x = view.x.abs().saturating_sub(radius).saturating_mul(focal);
    let projected_y = view.y.abs().saturating_sub(radius).saturating_mul(focal);
    projected_x <= half_w.saturating_mul(z) && projected_y <= half_h.saturating_mul(z)
}

/// Emit one floor quad. Cooked corners are `[NW, NE, SE, SW]`,
/// which already faces upward into playable space.
#[allow(clippy::too_many_arguments)]
fn emit_floor<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    heights: [i32; 4],
    split: u8,
    uvs: [(u8, u8); 4],
    material: WorldRenderMaterial,
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
    submit_split_quad(
        camera,
        options,
        CullMode::Back,
        material,
        verts,
        uvs,
        split,
        triangles,
        world,
    );
}

/// Emit one ceiling quad. Cooked corners are `[NW, NE, SE, SW]`;
/// runtime flips them so front-sided ceilings face the room
/// interior/underside.
#[allow(clippy::too_many_arguments)]
fn emit_ceiling<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    heights: [i32; 4],
    split: u8,
    uvs: [(u8, u8); 4],
    material: WorldRenderMaterial,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let verts = reverse_quad_winding([
        WorldVertex::new(x0, heights[0], z0),
        WorldVertex::new(x1, heights[1], z0),
        WorldVertex::new(x1, heights[2], z1),
        WorldVertex::new(x0, heights[3], z1),
    ]);
    submit_split_quad(
        camera,
        options,
        CullMode::Back,
        material,
        verts,
        reverse_quad_winding(uvs),
        split,
        triangles,
        world,
    );
}

/// Emit one wall quad. Wall heights `[BL, BR, TR, TL]` map onto
/// the cell's edge endpoints by direction. Diagonal directions
/// (4 / 5) silently drop -- the cooker rejects them, so they
/// shouldn't appear, but skipping is cheaper than panicking.
#[allow(clippy::too_many_arguments)]
fn emit_wall<const OT: usize>(
    sx: u16,
    sz: u16,
    sector_size: i32,
    direction: u8,
    heights: [i32; 4],
    uvs: [(u8, u8); 4],
    material: WorldRenderMaterial,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let Some(verts) = inward_wall_vertices(sx, sz, sector_size, direction, heights) else {
        return;
    };
    submit_quad(
        camera,
        options,
        CullMode::Back,
        material,
        verts,
        reverse_quad_winding(uvs),
        triangles,
        world,
    );
}

/// Project + submit one textured quad along the standard
/// `submit_textured_quad` 0–2 diagonal. Walls always use this
/// path because their geometry is rectangular and carries no
/// authored split metadata.
#[allow(clippy::too_many_arguments)]
fn submit_quad<const OT: usize>(
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    cull: CullMode,
    material: WorldRenderMaterial,
    verts: [WorldVertex; 4],
    uvs: [(u8, u8); 4],
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    let Some(projected) = camera.project_world_quad(verts) else {
        return;
    };
    submit_sided_projected_quad(world, triangles, projected, uvs, material, options, cull);
}

/// Project + submit a split-aware textured quad. `split == 0`
/// keeps the standard NW→SE diagonal; `split == 1` flips to
/// NE→SW. UVs are kept in the same `[NW, NE, SE, SW]` slot
/// order as the input verts, so the texture orientation
/// doesn't change with the diagonal -- only the triangulation
/// boundary moves.
#[allow(clippy::too_many_arguments)]
fn submit_split_quad<const OT: usize>(
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    cull: CullMode,
    material: WorldRenderMaterial,
    verts: [WorldVertex; 4],
    uvs: [(u8, u8); 4],
    split: u8,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT>,
) {
    if split != SPLIT_NE_SW {
        // Standard split shares the existing helper -- same
        // triangulation `submit_textured_quad` always used.
        submit_quad(
            camera, options, cull, material, verts, uvs, triangles, world,
        );
        return;
    }
    let Some(mut projected) = camera.project_world_quad(verts) else {
        return;
    };
    let mut uvs = uvs;
    if material.sidedness == SurfaceSidedness::Back {
        projected = reverse_quad_winding(projected);
        uvs = reverse_quad_winding(uvs);
    }
    let opts = options
        .with_cull_mode(cull_for_sidedness(material.sidedness, cull))
        .with_material_layer(material.texture);
    let [(a, b, c), (d, e, f)] = SPLIT_NE_SW_TRIANGLES;
    let stats = world.submit_textured_triangle(
        triangles,
        [projected[a], projected[b], projected[c]],
        [uvs[a], uvs[b], uvs[c]],
        material.texture,
        opts,
    );
    if stats.primitive_overflow || stats.command_overflow {
        return;
    }
    let _ = world.submit_textured_triangle(
        triangles,
        [projected[d], projected[e], projected[f]],
        [uvs[d], uvs[e], uvs[f]],
        material.texture,
        opts,
    );
}

fn submit_sided_projected_quad<const OT: usize>(
    world: &mut WorldRenderPass<'_, '_, OT>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    verts: [crate::render3d::ProjectedVertex; 4],
    uvs: [(u8, u8); 4],
    material: WorldRenderMaterial,
    options: WorldSurfaceOptions,
    base_cull: CullMode,
) {
    let (verts, uvs) = match material.sidedness {
        SurfaceSidedness::Back => (reverse_quad_winding(verts), reverse_quad_winding(uvs)),
        SurfaceSidedness::Front | SurfaceSidedness::Both => (verts, uvs),
    };
    let opts = options
        .with_cull_mode(cull_for_sidedness(material.sidedness, base_cull))
        .with_material_layer(material.texture);
    let _ = world.submit_textured_quad(triangles, verts, uvs, material.texture, opts);
}

const fn cull_for_sidedness(sidedness: SurfaceSidedness, base: CullMode) -> CullMode {
    match sidedness {
        SurfaceSidedness::Both => CullMode::None,
        SurfaceSidedness::Front | SurfaceSidedness::Back => base,
    }
}

/// Triangle index pairs used when a sector authors the
/// alternate (NE→SW) diagonal. Corner indexing matches the
/// `[NW=0, NE=1, SE=2, SW=3]` convention every
/// `[i32; 4]` heights array uses.
///
/// Each entry is `(a, b, c)` -- three corner indices producing
/// one of the two emitted triangles. Picking the boundary
/// `1`/`3` (NE/SW) instead of `0`/`2` (NW/SE) splits the quad
/// along the alternate diagonal.
const SPLIT_NE_SW_TRIANGLES: [(usize, usize, usize); 2] = [(0, 1, 3), (1, 2, 3)];

/// Triangle index pairs used by the standard NW→SE diagonal.
/// Public for tests + parity assertions; runtime emission goes
/// through `submit_textured_quad`'s internal indices, which the
/// `split_triangles_match_psx_engine` test pins to this table.
#[cfg(test)]
const SPLIT_NW_SE_TRIANGLES: [(usize, usize, usize); 2] = [(0, 1, 2), (0, 2, 3)];

/// Resolve the per-split triangulation. Default split (0) and
/// every unrecognised id fall back to the NW-SE diagonal so a
/// future split id never silently empties the room.
#[cfg(test)]
const fn split_triangles(split: u8) -> [(usize, usize, usize); 2] {
    if split == SPLIT_NE_SW {
        SPLIT_NE_SW_TRIANGLES
    } else {
        SPLIT_NW_SE_TRIANGLES
    }
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

fn horizontal_face_center(sx: u16, sz: u16, sector_size: i32, heights: [i32; 4]) -> RoomPoint {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let cy = average4_i32(heights[0], heights[1], heights[2], heights[3]);
    RoomPoint::new((x0 + x1) / 2, cy, (z0 + z1) / 2)
}

fn wall_face_center(
    sx: u16,
    sz: u16,
    sector_size: i32,
    direction: u8,
    heights: [i32; 4],
) -> Option<RoomPoint> {
    let verts = inward_wall_vertices(sx, sz, sector_size, direction, heights)?;
    Some(RoomPoint::new(
        average4_i32(verts[0].x, verts[1].x, verts[2].x, verts[3].x),
        average4_i32(verts[0].y, verts[1].y, verts[2].y, verts[3].y),
        average4_i32(verts[0].z, verts[1].z, verts[2].z, verts[3].z),
    ))
}

fn average4_i32(a: i32, b: i32, c: i32, d: i32) -> i32 {
    a.saturating_add(b).saturating_add(c).saturating_add(d) / 4
}

fn inward_wall_vertices(
    sx: u16,
    sz: u16,
    sector_size: i32,
    direction: u8,
    heights: [i32; 4],
) -> Option<[WorldVertex; 4]> {
    let (x0, x1, z0, z1) = cell_bounds(sx, sz, sector_size);
    let bl_br_tr_tl = match direction {
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
        _ => return None,
    };
    Some(reverse_quad_winding(bl_br_tr_tl))
}

#[cfg(test)]
fn inward_wall_uvs() -> [(u8, u8); 4] {
    reverse_quad_winding(WALL_UVS)
}

fn reverse_quad_winding<T: Copy>(corners: [T; 4]) -> [T; 4] {
    [corners[0], corners[3], corners[2], corners[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Angle;
    use crate::{ProjectedVertex, WorldProjection, Q12};

    /// Helper: the two indices both triangles in `[t0, t1]`
    /// share form the diagonal of the split. Returned sorted
    /// so test assertions are stable.
    fn diagonal(triangles: [(usize, usize, usize); 2]) -> [usize; 2] {
        let [t0, t1] = triangles;
        let a = [t0.0, t0.1, t0.2];
        let b = [t1.0, t1.1, t1.2];
        let mut shared = [usize::MAX; 2];
        let mut n = 0;
        for &i in &a {
            if b.contains(&i) && n < 2 {
                shared[n] = i;
                n += 1;
            }
        }
        shared.sort();
        shared
    }

    #[test]
    fn split_zero_uses_nw_se_diagonal() {
        // Standard split -- both triangles meet at corners 0
        // and 2, which is the diagonal `submit_textured_quad`
        // has always used.
        let triangles = split_triangles(SPLIT_NW_SE);
        assert_eq!(triangles[0], (0, 1, 2));
        assert_eq!(triangles[1], (0, 2, 3));
        assert_eq!(diagonal(triangles), [0, 2]);
    }

    #[test]
    fn split_one_uses_ne_sw_diagonal() {
        // Alternate split -- the two triangles share corners
        // 1 (NE) and 3 (SW), which is the perpendicular
        // diagonal. This is the case the prior renderer got
        // wrong: it used the NW→SE diagonal regardless of
        // the cooked / collision split id.
        let triangles = split_triangles(SPLIT_NE_SW);
        assert_eq!(triangles[0], (0, 1, 3));
        assert_eq!(triangles[1], (1, 2, 3));
        assert_eq!(diagonal(triangles), [1, 3]);
    }

    #[test]
    fn unknown_split_id_falls_back_to_nw_se() {
        // Future split-ids (e.g. quad subdivision) shouldn't
        // empty the room -- fall through to the standard
        // diagonal so the user sees something while the
        // schema catches up.
        for unknown in [2u8, 3, 9, 200] {
            assert_eq!(split_triangles(unknown), SPLIT_NW_SE_TRIANGLES);
        }
    }

    #[test]
    fn each_split_covers_every_corner() {
        // Sanity: every triangulation must reference all four
        // corners across its two triangles, otherwise the quad
        // has a hole.
        for split in [SPLIT_NW_SE, SPLIT_NE_SW] {
            let [t0, t1] = split_triangles(split);
            let mut seen = [false; 4];
            for i in [t0.0, t0.1, t0.2, t1.0, t1.1, t1.2] {
                seen[i] = true;
            }
            assert!(seen.iter().all(|&v| v), "split {split} misses a corner");
        }
    }

    #[test]
    fn cardinal_walls_face_their_owning_cell() {
        let projection = WorldProjection::new(160, 120, 200, 16);
        let y = 512;
        let center = WorldVertex::new(512, y, 512);
        let cases = [
            (
                DIR_NORTH,
                WorldCamera::from_basis(
                    projection,
                    center,
                    Q12::ZERO,
                    Q12::ONE,
                    Q12::ZERO,
                    Q12::ONE,
                ),
            ),
            (
                DIR_EAST,
                WorldCamera::from_basis(
                    projection,
                    center,
                    Q12::NEG_ONE,
                    Q12::ZERO,
                    Q12::ZERO,
                    Q12::ONE,
                ),
            ),
            (
                DIR_SOUTH,
                WorldCamera::from_basis(
                    projection,
                    center,
                    Q12::ZERO,
                    Q12::NEG_ONE,
                    Q12::ZERO,
                    Q12::ONE,
                ),
            ),
            (
                DIR_WEST,
                WorldCamera::from_basis(
                    projection,
                    center,
                    Q12::ONE,
                    Q12::ZERO,
                    Q12::ZERO,
                    Q12::ONE,
                ),
            ),
        ];

        for (direction, camera) in cases {
            let verts = inward_wall_vertices(0, 0, 1024, direction, [0, 0, 1024, 1024])
                .expect("cardinal wall");
            let projected = camera
                .project_world_quad(verts)
                .expect("wall projects from owning cell");
            for (a, b, c) in SPLIT_NW_SE_TRIANGLES {
                assert!(
                    projected_triangle_area(projected[a], projected[b], projected[c]) > 0,
                    "direction {direction} should not be culled"
                );
            }
        }
    }

    #[test]
    fn floors_face_playable_interior() {
        let projection = WorldProjection::new(160, 120, 200, 16);
        let camera = WorldCamera::orbit_yaw(
            projection,
            WorldVertex::new(512, 0, 512),
            1100,
            2048,
            Angle::ZERO,
        );
        let verts = [
            WorldVertex::new(0, 0, 0),
            WorldVertex::new(1024, 0, 0),
            WorldVertex::new(1024, 0, 1024),
            WorldVertex::new(0, 0, 1024),
        ];
        let projected = camera
            .project_world_quad(verts)
            .expect("floor projects from playable camera");

        for (a, b, c) in SPLIT_NW_SE_TRIANGLES {
            let area = projected_triangle_area(projected[a], projected[b], projected[c]);
            assert!(
                area > 0,
                "floor should not be culled from above: area={area} projected={projected:?}"
            );
        }
    }

    #[test]
    fn wall_uvs_follow_the_reversed_winding() {
        assert_eq!(
            inward_wall_uvs(),
            [(0, TILE_UV), (0, 0), (TILE_UV, 0), (TILE_UV, TILE_UV)]
        );
    }

    #[test]
    fn horizontal_face_center_uses_cell_midpoint_and_average_height() {
        assert_eq!(
            horizontal_face_center(2, 3, 1024, [0, 512, 1024, 512]),
            RoomPoint::new(2560, 512, 3584)
        );
    }

    #[test]
    fn wall_face_center_uses_emitted_runtime_wall_geometry() {
        assert_eq!(
            wall_face_center(0, 0, 1024, DIR_EAST, [0, 0, 1024, 1024]),
            Some(RoomPoint::new(1024, 512, 512))
        );
        assert_eq!(
            wall_face_center(0, 0, 1024, DIR_NORTH, [0, 0, 1024, 1024]),
            Some(RoomPoint::new(512, 512, 0))
        );
    }

    fn projected_triangle_area(a: ProjectedVertex, b: ProjectedVertex, c: ProjectedVertex) -> i32 {
        let ax = (b.sx as i32) - (a.sx as i32);
        let ay = (b.sy as i32) - (a.sy as i32);
        let bx = (c.sx as i32) - (a.sx as i32);
        let by = (c.sy as i32) - (a.sy as i32);
        ax * by - ay * bx
    }
}
