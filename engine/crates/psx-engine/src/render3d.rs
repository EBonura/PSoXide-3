//! Fixed-budget 3D render passes built on ordering tables.
//!
//! This module owns the part every PS1-scale 3D scene otherwise ends
//! up rewriting: project vertices through the currently loaded GTE
//! transform, cull back-facing faces, build GPU packets into fixed
//! arenas, sort the resulting commands, and finally insert them into
//! the frame ordering table.
//!
//! It deliberately stays renderer-shaped rather than editor-shaped.
//! Games and future editor exports still choose scene layout,
//! materials, animation, and streaming. The engine guarantees that a
//! frame's opaque mesh triangles share one depth policy and one
//! deterministic OT insertion order.

use psx_asset::Mesh;
use psx_gpu::{
    material::TextureMaterial,
    prim::{TriGouraud, TriTextured},
};
use psx_gte::{
    lighting::{project_lit, ProjectedLit},
    math::Vec3I16,
};
use psx_math::{cos_q12, sin_q12};

use crate::render::{DepthBand, DepthRange, DepthSlot, OtFrame, PrimitiveArena};

const PSX_VERTEX_MIN: i16 = -1024;
const PSX_VERTEX_MAX: i16 = 1023;
const PSX_TRI_MAX_DX: i32 = 1023;
const PSX_TRI_MAX_DY: i32 = 511;
const MAX_TEXTURED_HW_SPLIT_DEPTH: u8 = 5;

/// Scalar depth policy used to bucket a triangle into the ordering table.
///
/// The PS1 has no z-buffer, so every triangle eventually becomes one
/// scalar sort key. `Average` mirrors the common GTE `AVSZ3` style.
/// `Nearest` and `Farthest` are useful escape hatches for authored
/// meshes where a long triangle should bias toward one side of the
/// painter's algorithm.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DepthPolicy {
    /// Use the average of the three projected vertex depths.
    Average,
    /// Use the nearest projected vertex depth.
    Nearest,
    /// Use the farthest projected vertex depth.
    Farthest,
}

impl DepthPolicy {
    fn depth_values(self, z0: i32, z1: i32, z2: i32) -> i32 {
        match self {
            DepthPolicy::Average => (z0 + z1 + z2) / 3,
            DepthPolicy::Nearest => z0.min(z1).min(z2),
            DepthPolicy::Farthest => z0.max(z1).max(z2),
        }
    }

    fn depth(self, verts: [ProjectedLit; 3]) -> i32 {
        self.depth_values(verts[0].sz as i32, verts[1].sz as i32, verts[2].sz as i32)
    }
}

/// Triangle culling policy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CullMode {
    /// Submit triangles regardless of screen-space winding.
    None,
    /// Cull clockwise screen-space triangles.
    Back,
}

/// Projected vertex used by CPU-projected world surfaces.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectedVertex {
    /// Screen-space X.
    pub sx: i16,
    /// Screen-space Y.
    pub sy: i16,
    /// Camera-space depth.
    pub sz: i32,
}

impl ProjectedVertex {
    /// Build a projected vertex.
    pub const fn new(sx: i16, sy: i16, sz: i32) -> Self {
        Self { sx, sy, sz }
    }
}

/// Camera-space vertex used by clipped world surfaces.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewVertex {
    /// Camera-space X.
    pub x: i32,
    /// Camera-space Y.
    pub y: i32,
    /// Camera-space depth.
    pub z: i32,
}

impl ViewVertex {
    /// Zero camera-space vertex.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Build a camera-space vertex.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// CPU-side world-space vertex used by [`WorldCamera`].
///
/// This is intentionally raw integer space rather than
/// [`Vec3World`][crate::Vec3World]'s Q19.12 GTE translation space.
/// CPU-projected editor/debug/material surfaces often use compact
/// authored coordinates and a matching focal length; callers choose
/// that scale as long as every vertex, camera position, and
/// [`WorldProjection`] uses the same unit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WorldVertex {
    /// World-space X.
    pub x: i32,
    /// World-space Y.
    pub y: i32,
    /// World-space Z.
    pub z: i32,
}

impl WorldVertex {
    /// Origin.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Build a world-space vertex.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// Textured camera-space vertex used by near-plane clipping.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TexturedViewVertex {
    /// Camera-space position.
    pub position: ViewVertex,
    /// Texture U coordinate.
    pub u: i32,
    /// Texture V coordinate.
    pub v: i32,
}

impl TexturedViewVertex {
    /// Zero textured camera-space vertex.
    pub const ZERO: Self = Self {
        position: ViewVertex::ZERO,
        u: 0,
        v: 0,
    };

    /// Build a textured camera-space vertex.
    pub const fn new(position: ViewVertex, u: i32, v: i32) -> Self {
        Self { position, u, v }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ProjectedTexturedVertex {
    projected: ProjectedVertex,
    u: i32,
    v: i32,
}

impl ProjectedTexturedVertex {
    const fn new(projected: ProjectedVertex, u: i32, v: i32) -> Self {
        Self { projected, u, v }
    }
}

/// Perspective projection settings for CPU-transformed world surfaces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldProjection {
    /// Screen centre X.
    pub screen_x: i16,
    /// Screen centre Y.
    pub screen_y: i16,
    /// Perspective focal length.
    pub focal_length: i32,
    /// Near clipping plane in camera-space depth.
    pub near_z: i32,
}

impl WorldProjection {
    /// Build projection settings for CPU-transformed world surfaces.
    pub const fn new(screen_x: i16, screen_y: i16, focal_length: i32, near_z: i32) -> Self {
        Self {
            screen_x,
            screen_y,
            focal_length,
            near_z,
        }
    }

    /// Project a camera-space vertex into screen space.
    pub fn project_view(self, vertex: ViewVertex) -> Option<ProjectedVertex> {
        if vertex.z <= 0 || vertex.z < self.near_z {
            return None;
        }

        let sx = (self.screen_x as i32) + (vertex.x * self.focal_length) / vertex.z;
        let sy = (self.screen_y as i32) - (vertex.y * self.focal_length) / vertex.z;
        Some(ProjectedVertex::new(clamp_i16(sx), clamp_i16(sy), vertex.z))
    }
}

impl From<ProjectedLit> for ProjectedVertex {
    fn from(value: ProjectedLit) -> Self {
        Self {
            sx: value.sx,
            sy: value.sy,
            sz: value.sz as i32,
        }
    }
}

/// CPU-side perspective camera for authored world surfaces.
///
/// The camera stores a simple orbit-style basis: yaw rotates around
/// the world's Y axis, then pitch tilts the view around the camera's
/// local X axis. It is deliberately small and fixed-point friendly:
/// the trigonometric basis is Q0.12, matching `psx-math`'s sine table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldCamera {
    /// Projection used after world-to-view transformation.
    pub projection: WorldProjection,
    /// Camera position in the same world units as submitted surfaces.
    pub position: WorldVertex,
    /// Sine of yaw, Q0.12.
    pub sin_yaw: i32,
    /// Cosine of yaw, Q0.12.
    pub cos_yaw: i32,
    /// Sine of pitch, Q0.12.
    pub sin_pitch: i32,
    /// Cosine of pitch, Q0.12.
    pub cos_pitch: i32,
}

impl WorldCamera {
    /// Build a camera from an explicit fixed-point basis.
    pub const fn from_basis(
        projection: WorldProjection,
        position: WorldVertex,
        sin_yaw: i32,
        cos_yaw: i32,
        sin_pitch: i32,
        cos_pitch: i32,
    ) -> Self {
        Self {
            projection,
            position,
            sin_yaw,
            cos_yaw,
            sin_pitch,
            cos_pitch,
        }
    }

    /// Build a camera on a horizontal orbit that looks at `target`.
    ///
    /// `yaw_q12` is the argument consumed by [`sin_q12`] /
    /// [`cos_q12`]: 4096 units per full turn. `camera_y` is the
    /// camera's absolute world-space height. Pitch is derived from
    /// `target.y - camera_y`, so dollying the radius keeps the target
    /// centred without per-frame call-site math.
    pub fn orbit_yaw(
        projection: WorldProjection,
        target: WorldVertex,
        camera_y: i32,
        radius: i32,
        yaw_q12: u16,
    ) -> Self {
        let sin_yaw = sin_q12(yaw_q12);
        let cos_yaw = cos_q12(yaw_q12);
        let target_dy = target.y - camera_y;
        let pitch_len =
            isqrt_i32(radius.saturating_mul(radius) + target_dy.saturating_mul(target_dy)).max(1);
        Self {
            projection,
            position: WorldVertex::new(
                target.x + ((sin_yaw * radius) >> 12),
                camera_y,
                target.z + ((cos_yaw * radius) >> 12),
            ),
            sin_yaw,
            cos_yaw,
            sin_pitch: (target_dy * 4096) / pitch_len,
            cos_pitch: (radius * 4096) / pitch_len,
        }
    }

    /// Transform a world-space vertex into camera-space.
    pub fn view_vertex(self, vertex: WorldVertex) -> ViewVertex {
        let dx = vertex.x - self.position.x;
        let dy = vertex.y - self.position.y;
        let dz = vertex.z - self.position.z;

        let x1 = ((dx * self.cos_yaw) - (dz * self.sin_yaw)) >> 12;
        let z1 = ((-dx * self.sin_yaw) - (dz * self.cos_yaw)) >> 12;
        let y2 = ((dy * self.cos_pitch) - (z1 * self.sin_pitch)) >> 12;
        let z2 = ((dy * self.sin_pitch) + (z1 * self.cos_pitch)) >> 12;

        ViewVertex::new(x1, y2, z2)
    }

    /// Transform and project a world-space vertex.
    pub fn project_world(self, vertex: WorldVertex) -> Option<ProjectedVertex> {
        self.projection.project_view(self.view_vertex(vertex))
    }

    /// Transform and project a world-space quad. Returns `None` if
    /// any corner falls behind the projection near plane.
    pub fn project_world_quad(self, verts: [WorldVertex; 4]) -> Option<[ProjectedVertex; 4]> {
        Some([
            self.project_world(verts[0])?,
            self.project_world(verts[1])?,
            self.project_world(verts[2])?,
            self.project_world(verts[3])?,
        ])
    }

    /// Transform a textured world-space vertex to camera-space.
    pub fn textured_view_vertex(self, vertex: WorldVertex, uv: (u8, u8)) -> TexturedViewVertex {
        TexturedViewVertex::new(self.view_vertex(vertex), uv.0 as i32, uv.1 as i32)
    }
}

/// Shared options for projected world surfaces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldSurfaceOptions {
    /// OT slot band reserved for this surface layer.
    pub depth_band: DepthBand,
    /// Camera-space depth range mapped into `depth_band`.
    pub depth_range: DepthRange,
    /// Triangle scalar depth policy.
    pub depth_policy: DepthPolicy,
    /// Signed offset added to scalar depth before slot mapping.
    pub depth_bias: i32,
    /// Triangle culling mode.
    pub cull_mode: CullMode,
}

impl WorldSurfaceOptions {
    /// Build options for a world-geometry depth band and range.
    pub const fn new(depth_band: DepthBand, depth_range: DepthRange) -> Self {
        Self {
            depth_band,
            depth_range,
            depth_policy: DepthPolicy::Average,
            depth_bias: 0,
            cull_mode: CullMode::Back,
        }
    }

    /// Return options with a different scalar depth policy.
    pub const fn with_depth_policy(mut self, depth_policy: DepthPolicy) -> Self {
        self.depth_policy = depth_policy;
        self
    }

    /// Return options with a signed depth bias.
    pub const fn with_depth_bias(mut self, depth_bias: i32) -> Self {
        self.depth_bias = depth_bias;
        self
    }

    /// Return options with a different culling mode.
    pub const fn with_cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }
}

/// Scratch command for a mixed world render pass.
///
/// Commands hold raw packet pointers so one pass can sort and submit
/// several packet kinds. The pointed-to packets must live until
/// [`WorldRenderPass::flush`] has inserted them into the OT.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldTriCommand {
    slot: DepthSlot,
    depth: i32,
    packet_ptr: *mut u32,
    words: u8,
    order: usize,
}

impl WorldTriCommand {
    /// Empty command value for static scratch-buffer initialisation.
    pub const EMPTY: Self = Self {
        slot: DepthSlot::new(0),
        depth: 0,
        packet_ptr: core::ptr::null_mut(),
        words: 0,
        order: 0,
    };
}

/// Per-submit counters and overflow flags for world surfaces.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WorldRenderStats {
    /// Triangles accepted into the pass after culling.
    pub submitted_triangles: u16,
    /// Triangles rejected by back-face culling.
    pub culled_triangles: u16,
    /// Triangles that crossed the near plane and were clipped before
    /// projection.
    pub clipped_triangles: u16,
    /// Oversized projected triangles split to satisfy PS1 hardware
    /// extent limits.
    pub split_triangles: u16,
    /// Triangles dropped before packet emission because they were
    /// fully clipped or could not be made hardware-legal.
    pub dropped_triangles: u16,
    /// True if the primitive packet arena filled up.
    pub primitive_overflow: bool,
    /// True if the command scratch buffer filled up.
    pub command_overflow: bool,
}

/// Mixed world render pass.
///
/// Authoring code can submit surfaces as quads or triangles; the pass
/// stores sorted triangle packets internally so culling, depth bucketing,
/// and same-slot ordering are deterministic across packet kinds.
#[must_use = "call flush() to insert submitted triangles into the ordering table"]
pub struct WorldRenderPass<'a, 'ot, const OT_DEPTH: usize> {
    ot: &'a mut OtFrame<'ot, OT_DEPTH>,
    commands: &'a mut [WorldTriCommand],
    command_len: usize,
    next_order: usize,
}

impl<'a, 'ot, const OT_DEPTH: usize> WorldRenderPass<'a, 'ot, OT_DEPTH> {
    /// Start a world render pass.
    pub fn new(ot: &'a mut OtFrame<'ot, OT_DEPTH>, commands: &'a mut [WorldTriCommand]) -> Self {
        Self {
            ot,
            commands,
            command_len: 0,
            next_order: 0,
        }
    }

    /// Submit a projected textured triangle.
    pub fn submit_textured_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedVertex; 3],
        uvs: [(u8, u8); 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = WorldRenderStats::default();
        if options.cull_mode == CullMode::Back && projected_back_facing(verts) {
            stats.culled_triangles = 1;
            return stats;
        }

        let textured = [
            ProjectedTexturedVertex::new(verts[0], uvs[0].0 as i32, uvs[0].1 as i32),
            ProjectedTexturedVertex::new(verts[1], uvs[1].0 as i32, uvs[1].1 as i32),
            ProjectedTexturedVertex::new(verts[2], uvs[2].0 as i32, uvs[2].1 as i32),
        ];
        merge_world_stats(
            &mut stats,
            self.submit_textured_triangle_split(triangles, textured, material, options, 0),
        );
        stats
    }

    fn submit_textured_triangle_split(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedTexturedVertex; 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
        split_depth: u8,
    ) -> WorldRenderStats {
        let verts = [
            clamp_projected_textured_vertex(verts[0]),
            clamp_projected_textured_vertex(verts[1]),
            clamp_projected_textured_vertex(verts[2]),
        ];

        if projected_textured_exceeds_hw_extent(verts) && split_depth < MAX_TEXTURED_HW_SPLIT_DEPTH
        {
            return self.submit_split_textured_triangle(
                triangles,
                verts,
                material,
                options,
                split_depth,
            );
        }
        if projected_textured_exceeds_hw_extent(verts) {
            return WorldRenderStats {
                dropped_triangles: 1,
                ..WorldRenderStats::default()
            };
        }

        self.submit_textured_triangle_leaf(triangles, verts, material, options)
    }

    fn submit_split_textured_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedTexturedVertex; 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
        split_depth: u8,
    ) -> WorldRenderStats {
        let edge = largest_projected_edge(verts);
        let mut stats = WorldRenderStats {
            split_triangles: 1,
            ..WorldRenderStats::default()
        };

        let (first, second) = match edge {
            0 => {
                let mid = midpoint_projected_textured(verts[0], verts[1]);
                ([verts[0], mid, verts[2]], [mid, verts[1], verts[2]])
            }
            1 => {
                let mid = midpoint_projected_textured(verts[1], verts[2]);
                ([verts[0], verts[1], mid], [verts[0], mid, verts[2]])
            }
            _ => {
                let mid = midpoint_projected_textured(verts[2], verts[0]);
                ([verts[0], verts[1], mid], [mid, verts[1], verts[2]])
            }
        };

        let first_stats = self.submit_textured_triangle_split(
            triangles,
            first,
            material,
            options,
            split_depth + 1,
        );
        merge_world_stats(&mut stats, first_stats);
        if stats.primitive_overflow || stats.command_overflow {
            return stats;
        }

        let second_stats = self.submit_textured_triangle_split(
            triangles,
            second,
            material,
            options,
            split_depth + 1,
        );
        merge_world_stats(&mut stats, second_stats);
        stats
    }

    fn submit_textured_triangle_leaf(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedTexturedVertex; 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = WorldRenderStats::default();
        if self.command_len >= self.commands.len() {
            stats.command_overflow = true;
            return stats;
        }

        let Some(tri) = triangles.push(TriTextured::with_material(
            [
                (verts[0].projected.sx, verts[0].projected.sy),
                (verts[1].projected.sx, verts[1].projected.sy),
                (verts[2].projected.sx, verts[2].projected.sy),
            ],
            [
                (clamp_u8(verts[0].u), clamp_u8(verts[0].v)),
                (clamp_u8(verts[1].u), clamp_u8(verts[1].v)),
                (clamp_u8(verts[2].u), clamp_u8(verts[2].v)),
            ],
            material,
        )) else {
            stats.primitive_overflow = true;
            return stats;
        };

        let depth = options
            .depth_policy
            .depth_values(
                verts[0].projected.sz,
                verts[1].projected.sz,
                verts[2].projected.sz,
            )
            .saturating_add(options.depth_bias);
        self.push_command(
            options
                .depth_band
                .slot::<OT_DEPTH>(options.depth_range, depth),
            depth,
            tri as *mut TriTextured as *mut u32,
            TriTextured::WORDS,
        );
        stats.submitted_triangles = 1;
        stats
    }

    /// Submit a projected textured quad as two independently culled
    /// and sorted textured triangles.
    pub fn submit_textured_quad(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedVertex; 4],
        uvs: [(u8, u8); 4],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = self.submit_textured_triangle(
            triangles,
            [verts[0], verts[1], verts[2]],
            [uvs[0], uvs[1], uvs[2]],
            material,
            options,
        );
        if stats.primitive_overflow || stats.command_overflow {
            return stats;
        }

        let second = self.submit_textured_triangle(
            triangles,
            [verts[2], verts[1], verts[3]],
            [uvs[2], uvs[1], uvs[3]],
            material,
            options,
        );
        stats.submitted_triangles = stats
            .submitted_triangles
            .saturating_add(second.submitted_triangles);
        stats.culled_triangles = stats
            .culled_triangles
            .saturating_add(second.culled_triangles);
        stats.primitive_overflow |= second.primitive_overflow;
        stats.command_overflow |= second.command_overflow;
        stats
    }

    /// Submit a textured triangle in camera space.
    ///
    /// The triangle is clipped against the projection's near plane
    /// before projection. This avoids whole-surface popping when a
    /// floor or wall crosses the camera plane.
    pub fn submit_textured_view_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [TexturedViewVertex; 3],
        projection: WorldProjection,
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut clipped = [TexturedViewVertex::ZERO; 4];
        let count = clip_textured_triangle_to_near(verts, projection.near_z, &mut clipped);
        let mut stats = WorldRenderStats::default();
        if count < 3 {
            stats.dropped_triangles = 1;
            return stats;
        }
        if count != 3 || !verts.iter().all(|v| v.position.z >= projection.near_z) {
            stats.clipped_triangles = 1;
        }

        let first = self.submit_clipped_textured_triangle(
            triangles,
            [clipped[0], clipped[1], clipped[2]],
            projection,
            material,
            options,
        );
        merge_world_stats(&mut stats, first);
        if stats.primitive_overflow || stats.command_overflow || count == 3 {
            return stats;
        }

        let second = self.submit_clipped_textured_triangle(
            triangles,
            [clipped[0], clipped[2], clipped[3]],
            projection,
            material,
            options,
        );
        merge_world_stats(&mut stats, second);
        stats
    }

    /// Submit a textured quad in camera space as two clipped,
    /// independently culled and sorted triangles.
    pub fn submit_textured_view_quad(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [TexturedViewVertex; 4],
        projection: WorldProjection,
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = self.submit_textured_view_triangle(
            triangles,
            [verts[0], verts[1], verts[2]],
            projection,
            material,
            options,
        );
        if stats.primitive_overflow || stats.command_overflow {
            return stats;
        }

        let second = self.submit_textured_view_triangle(
            triangles,
            [verts[2], verts[1], verts[3]],
            projection,
            material,
            options,
        );
        merge_world_stats(&mut stats, second);
        stats
    }

    /// Transform and submit a textured world-space triangle through
    /// `camera`.
    pub fn submit_textured_world_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        camera: WorldCamera,
        verts: [WorldVertex; 3],
        uvs: [(u8, u8); 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        self.submit_textured_view_triangle(
            triangles,
            [
                camera.textured_view_vertex(verts[0], uvs[0]),
                camera.textured_view_vertex(verts[1], uvs[1]),
                camera.textured_view_vertex(verts[2], uvs[2]),
            ],
            camera.projection,
            material,
            options,
        )
    }

    /// Transform and submit a textured world-space quad through
    /// `camera`.
    pub fn submit_textured_world_quad(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        camera: WorldCamera,
        verts: [WorldVertex; 4],
        uvs: [(u8, u8); 4],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        self.submit_textured_view_quad(
            triangles,
            [
                camera.textured_view_vertex(verts[0], uvs[0]),
                camera.textured_view_vertex(verts[1], uvs[1]),
                camera.textured_view_vertex(verts[2], uvs[2]),
                camera.textured_view_vertex(verts[3], uvs[3]),
            ],
            camera.projection,
            material,
            options,
        )
    }

    fn submit_clipped_textured_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [TexturedViewVertex; 3],
        projection: WorldProjection,
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let Some(a) = projection.project_view(verts[0].position) else {
            return WorldRenderStats {
                dropped_triangles: 1,
                ..WorldRenderStats::default()
            };
        };
        let Some(b) = projection.project_view(verts[1].position) else {
            return WorldRenderStats {
                dropped_triangles: 1,
                ..WorldRenderStats::default()
            };
        };
        let Some(c) = projection.project_view(verts[2].position) else {
            return WorldRenderStats {
                dropped_triangles: 1,
                ..WorldRenderStats::default()
            };
        };
        self.submit_textured_triangle(
            triangles,
            [a, b, c],
            [
                (clamp_u8(verts[0].u), clamp_u8(verts[0].v)),
                (clamp_u8(verts[1].u), clamp_u8(verts[1].v)),
                (clamp_u8(verts[2].u), clamp_u8(verts[2].v)),
            ],
            material,
            options,
        )
    }

    /// Submit a Gouraud triangle packet already projected and lit.
    pub fn submit_gouraud_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriGouraud>,
        verts: [ProjectedLit; 3],
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = WorldRenderStats::default();
        let projected = [
            ProjectedVertex::from(verts[0]),
            ProjectedVertex::from(verts[1]),
            ProjectedVertex::from(verts[2]),
        ];
        if options.cull_mode == CullMode::Back && projected_back_facing(projected) {
            stats.culled_triangles = 1;
            return stats;
        }

        if self.command_len >= self.commands.len() {
            stats.command_overflow = true;
            return stats;
        }

        let Some(tri) = triangles.push(TriGouraud::new(
            [
                (verts[0].sx, verts[0].sy),
                (verts[1].sx, verts[1].sy),
                (verts[2].sx, verts[2].sy),
            ],
            [
                (verts[0].r, verts[0].g, verts[0].b),
                (verts[1].r, verts[1].g, verts[1].b),
                (verts[2].r, verts[2].g, verts[2].b),
            ],
        )) else {
            stats.primitive_overflow = true;
            return stats;
        };

        let depth = options
            .depth_policy
            .depth(verts)
            .saturating_add(options.depth_bias);
        self.push_command(
            options
                .depth_band
                .slot::<OT_DEPTH>(options.depth_range, depth),
            depth,
            tri as *mut TriGouraud as *mut u32,
            TriGouraud::WORDS,
        );
        stats.submitted_triangles = 1;
        stats
    }

    fn push_command(&mut self, slot: DepthSlot, depth: i32, packet_ptr: *mut u32, words: u8) {
        self.commands[self.command_len] = WorldTriCommand {
            slot,
            depth,
            packet_ptr,
            words,
            order: self.next_order,
        };
        self.command_len += 1;
        self.next_order = self.next_order.saturating_add(1);
    }

    /// Sort and insert all submitted triangles into the ordering table.
    pub fn flush(self) {
        sort_world_for_ot_insert(&mut self.commands[..self.command_len]);
        let mut i = 0;
        while i < self.command_len {
            let command = self.commands[i];
            if !command.packet_ptr.is_null() {
                // SAFETY: Commands are created only from primitive
                // arenas borrowed by submit methods. Those packets live
                // until after this pass flushes and the frame submits.
                unsafe {
                    self.ot
                        .add_raw_slot(command.slot, command.packet_ptr, command.words)
                };
            }
            i += 1;
        }
    }
}

/// Scratch command for a Gouraud triangle render pass.
///
/// Caller-owned arrays of this type let the engine collect triangles
/// from several meshes before it mutates the ordering table. The
/// fields are private so call sites cannot accidentally depend on the
/// insertion-order details.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GouraudTriCommand {
    slot: DepthSlot,
    depth: i32,
    primitive_index: usize,
}

impl GouraudTriCommand {
    /// Empty command value for static scratch-buffer initialisation.
    pub const EMPTY: Self = Self {
        slot: DepthSlot::new(0),
        depth: 0,
        primitive_index: 0,
    };
}

/// Options for submitting a lit Gouraud mesh.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GouraudMeshOptions {
    /// OT slot band reserved for this mesh layer.
    pub depth_band: DepthBand,
    /// Camera-space depth range mapped into `depth_band`.
    pub depth_range: DepthRange,
    /// Triangle scalar depth policy.
    pub depth_policy: DepthPolicy,
    /// Signed offset added to the scalar depth before slot mapping.
    pub depth_bias: i32,
    /// Pixel offset added to projected vertices before packet build.
    pub screen_offset: (i16, i16),
    /// Normal used when the mesh blob lacks a normal for a vertex.
    pub default_normal: Vec3I16,
    /// Material RGB used when the mesh blob lacks face colours.
    pub default_material: (u8, u8, u8),
    /// Whether clockwise screen-space triangles should be culled.
    pub cull_backfaces: bool,
}

impl GouraudMeshOptions {
    /// Build mesh options for a world-geometry depth band and range.
    pub const fn new(depth_band: DepthBand, depth_range: DepthRange) -> Self {
        Self {
            depth_band,
            depth_range,
            depth_policy: DepthPolicy::Average,
            depth_bias: 0,
            screen_offset: (0, 0),
            default_normal: Vec3I16::ZERO,
            default_material: (128, 128, 128),
            cull_backfaces: true,
        }
    }

    /// Return options with a different scalar depth policy.
    pub const fn with_depth_policy(mut self, depth_policy: DepthPolicy) -> Self {
        self.depth_policy = depth_policy;
        self
    }

    /// Return options with a signed depth bias.
    pub const fn with_depth_bias(mut self, depth_bias: i32) -> Self {
        self.depth_bias = depth_bias;
        self
    }

    /// Return options with a projected-screen-space offset.
    pub const fn with_screen_offset(mut self, screen_offset: (i16, i16)) -> Self {
        self.screen_offset = screen_offset;
        self
    }

    /// Return options with a different fallback normal.
    pub const fn with_default_normal(mut self, default_normal: Vec3I16) -> Self {
        self.default_normal = default_normal;
        self
    }

    /// Return options with a different fallback material colour.
    pub const fn with_default_material(mut self, default_material: (u8, u8, u8)) -> Self {
        self.default_material = default_material;
        self
    }

    /// Return options with back-face culling enabled or disabled.
    pub const fn with_backface_culling(mut self, cull_backfaces: bool) -> Self {
        self.cull_backfaces = cull_backfaces;
        self
    }
}

/// Per-submit counters and overflow flags.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshRenderStats {
    /// Vertices projected into the caller-provided scratch buffer.
    pub projected_vertices: u16,
    /// Triangles accepted into the pass after culling.
    pub submitted_triangles: u16,
    /// Triangles rejected by back-face culling.
    pub culled_triangles: u16,
    /// Triangles skipped because their vertex indices were not projected.
    pub skipped_triangles: u16,
    /// True if the vertex scratch buffer was too small for the mesh.
    pub vertex_overflow: bool,
    /// True if the triangle packet arena filled up.
    pub primitive_overflow: bool,
    /// True if the command scratch buffer filled up.
    pub command_overflow: bool,
}

/// Opaque Gouraud triangle pass.
///
/// Create one pass for a world layer, submit one or more meshes after
/// loading their GTE transforms and lights, then call [`flush`](Self::flush).
/// The pass sorts all submitted triangles together so triangles from
/// different meshes can share the same OT depth policy.
#[must_use = "call flush() to insert submitted triangles into the ordering table"]
pub struct GouraudRenderPass<'a, 'ot, 'arena, const OT_DEPTH: usize> {
    ot: &'a mut OtFrame<'ot, OT_DEPTH>,
    triangles: &'a mut PrimitiveArena<'arena, TriGouraud>,
    commands: &'a mut [GouraudTriCommand],
    command_len: usize,
}

impl<'a, 'ot, 'arena, const OT_DEPTH: usize> GouraudRenderPass<'a, 'ot, 'arena, OT_DEPTH> {
    /// Start an opaque Gouraud triangle pass.
    pub fn new(
        ot: &'a mut OtFrame<'ot, OT_DEPTH>,
        triangles: &'a mut PrimitiveArena<'arena, TriGouraud>,
        commands: &'a mut [GouraudTriCommand],
    ) -> Self {
        Self {
            ot,
            triangles,
            commands,
            command_len: 0,
        }
    }

    /// Submit a lit mesh using the currently loaded GTE transform and light rig.
    ///
    /// The caller must load rotation, translation, projection setup,
    /// and lighting before calling this method. `projected_vertices`
    /// is temporary scratch and may be reused for the next mesh after
    /// the call returns.
    pub fn submit_lit_mesh(
        &mut self,
        mesh: &Mesh<'_>,
        projected_vertices: &mut [ProjectedLit],
        options: GouraudMeshOptions,
    ) -> MeshRenderStats {
        let mut stats = MeshRenderStats::default();
        let mesh_verts = mesh.vert_count() as usize;
        let project_count = mesh_verts.min(projected_vertices.len()).min(256);
        stats.vertex_overflow = mesh_verts > project_count;
        stats.projected_vertices = project_count as u16;

        let mut vi = 0;
        while vi < project_count {
            let vert = vi as u8;
            let p = project_lit(
                mesh.vertex(vert),
                mesh.vertex_normal(vert).unwrap_or(options.default_normal),
                vertex_material(mesh, vert, options.default_material),
            );
            projected_vertices[vi] = ProjectedLit {
                sx: p.sx.saturating_add(options.screen_offset.0),
                sy: p.sy.saturating_add(options.screen_offset.1),
                sz: p.sz,
                r: p.r,
                g: p.g,
                b: p.b,
            };
            vi += 1;
        }

        let face_stats =
            self.submit_projected_mesh(mesh, &projected_vertices[..project_count], options);
        stats.submitted_triangles = face_stats.submitted_triangles;
        stats.culled_triangles = face_stats.culled_triangles;
        stats.skipped_triangles = face_stats.skipped_triangles;
        stats.primitive_overflow = face_stats.primitive_overflow;
        stats.command_overflow = face_stats.command_overflow;
        stats
    }

    /// Submit a mesh whose vertices are already projected and lit.
    ///
    /// This covers CPU-lighting paths: the caller can compute
    /// per-vertex colours however it wants, while the engine still
    /// owns culling, depth policy, command sorting, and OT insertion.
    /// `projected_vertices` must be indexed by the mesh's face
    /// indices.
    pub fn submit_projected_mesh(
        &mut self,
        mesh: &Mesh<'_>,
        projected_vertices: &[ProjectedLit],
        options: GouraudMeshOptions,
    ) -> MeshRenderStats {
        let mut stats = MeshRenderStats {
            projected_vertices: projected_vertices.len().min(u16::MAX as usize) as u16,
            ..MeshRenderStats::default()
        };
        let project_count = projected_vertices.len();

        let mut face_idx = 0;
        while face_idx < mesh.face_count() {
            let (ia, ib, ic) = mesh.face(face_idx);
            if (ia as usize) >= project_count
                || (ib as usize) >= project_count
                || (ic as usize) >= project_count
            {
                stats.skipped_triangles = stats.skipped_triangles.saturating_add(1);
                face_idx += 1;
                continue;
            }

            let verts = [
                projected_vertices[ia as usize],
                projected_vertices[ib as usize],
                projected_vertices[ic as usize],
            ];
            if options.cull_backfaces && back_facing(verts) {
                stats.culled_triangles = stats.culled_triangles.saturating_add(1);
                face_idx += 1;
                continue;
            }

            if self.command_len >= self.commands.len() {
                stats.command_overflow = true;
                break;
            }

            let Some(primitive_index) = self.triangles.push_index(TriGouraud::new(
                [
                    (verts[0].sx, verts[0].sy),
                    (verts[1].sx, verts[1].sy),
                    (verts[2].sx, verts[2].sy),
                ],
                [
                    (verts[0].r, verts[0].g, verts[0].b),
                    (verts[1].r, verts[1].g, verts[1].b),
                    (verts[2].r, verts[2].g, verts[2].b),
                ],
            )) else {
                stats.primitive_overflow = true;
                break;
            };

            let depth = options
                .depth_policy
                .depth(verts)
                .saturating_add(options.depth_bias);
            self.commands[self.command_len] = GouraudTriCommand {
                slot: options
                    .depth_band
                    .slot::<OT_DEPTH>(options.depth_range, depth),
                depth,
                primitive_index,
            };
            self.command_len += 1;
            stats.submitted_triangles = stats.submitted_triangles.saturating_add(1);
            face_idx += 1;
        }

        stats
    }

    /// Sort and insert all submitted triangles into the ordering table.
    pub fn flush(self) {
        sort_for_ot_insert(&mut self.commands[..self.command_len]);
        let mut i = 0;
        while i < self.command_len {
            let command = self.commands[i];
            if let Some(tri) = self.triangles.get_mut(command.primitive_index) {
                self.ot.add_packet_slot(command.slot, tri);
            }
            i += 1;
        }
    }
}

fn vertex_material(mesh: &Mesh<'_>, vert: u8, fallback: (u8, u8, u8)) -> (u8, u8, u8) {
    let mut face_idx = 0;
    while face_idx < mesh.face_count() {
        let (a, b, c) = mesh.face(face_idx);
        if a == vert || b == vert || c == vert {
            return mesh.face_color(face_idx).unwrap_or(fallback);
        }
        face_idx += 1;
    }
    fallback
}

fn back_facing(verts: [ProjectedLit; 3]) -> bool {
    projected_back_facing([
        ProjectedVertex::from(verts[0]),
        ProjectedVertex::from(verts[1]),
        ProjectedVertex::from(verts[2]),
    ])
}

fn projected_back_facing(verts: [ProjectedVertex; 3]) -> bool {
    let ax = (verts[1].sx as i32) - (verts[0].sx as i32);
    let ay = (verts[1].sy as i32) - (verts[0].sy as i32);
    let bx = (verts[2].sx as i32) - (verts[0].sx as i32);
    let by = (verts[2].sy as i32) - (verts[0].sy as i32);
    (ax * by - ay * bx) <= 0
}

fn clamp_projected_textured_vertex(vertex: ProjectedTexturedVertex) -> ProjectedTexturedVertex {
    ProjectedTexturedVertex::new(
        ProjectedVertex::new(
            clamp_i16_range(vertex.projected.sx, PSX_VERTEX_MIN, PSX_VERTEX_MAX),
            clamp_i16_range(vertex.projected.sy, PSX_VERTEX_MIN, PSX_VERTEX_MAX),
            vertex.projected.sz,
        ),
        vertex.u,
        vertex.v,
    )
}

fn projected_textured_exceeds_hw_extent(verts: [ProjectedTexturedVertex; 3]) -> bool {
    projected_edge_exceeds_hw_extent(verts[0], verts[1])
        || projected_edge_exceeds_hw_extent(verts[1], verts[2])
        || projected_edge_exceeds_hw_extent(verts[2], verts[0])
}

fn projected_edge_exceeds_hw_extent(
    a: ProjectedTexturedVertex,
    b: ProjectedTexturedVertex,
) -> bool {
    let dx = ((a.projected.sx as i32) - (b.projected.sx as i32)).abs();
    let dy = ((a.projected.sy as i32) - (b.projected.sy as i32)).abs();
    dx > PSX_TRI_MAX_DX || dy > PSX_TRI_MAX_DY
}

fn largest_projected_edge(verts: [ProjectedTexturedVertex; 3]) -> usize {
    let mut edge = 0;
    let mut score = projected_edge_split_score(verts[0], verts[1]);
    let score_1 = projected_edge_split_score(verts[1], verts[2]);
    if score_1 > score {
        edge = 1;
        score = score_1;
    }
    let score_2 = projected_edge_split_score(verts[2], verts[0]);
    if score_2 > score {
        edge = 2;
    }
    edge
}

fn projected_edge_split_score(a: ProjectedTexturedVertex, b: ProjectedTexturedVertex) -> i32 {
    let dx = ((a.projected.sx as i32) - (b.projected.sx as i32)).abs();
    let dy = ((a.projected.sy as i32) - (b.projected.sy as i32)).abs();
    dx.max(dy.saturating_mul(2))
}

fn midpoint_projected_textured(
    a: ProjectedTexturedVertex,
    b: ProjectedTexturedVertex,
) -> ProjectedTexturedVertex {
    ProjectedTexturedVertex::new(
        ProjectedVertex::new(
            midpoint_i16(a.projected.sx, b.projected.sx),
            midpoint_i16(a.projected.sy, b.projected.sy),
            midpoint_i32(a.projected.sz, b.projected.sz),
        ),
        midpoint_i32(a.u, b.u),
        midpoint_i32(a.v, b.v),
    )
}

fn midpoint_i16(a: i16, b: i16) -> i16 {
    midpoint_i32(a as i32, b as i32) as i16
}

fn midpoint_i32(a: i32, b: i32) -> i32 {
    a + (b - a) / 2
}

fn clip_textured_triangle_to_near(
    verts: [TexturedViewVertex; 3],
    near_z: i32,
    out: &mut [TexturedViewVertex; 4],
) -> usize {
    let mut count = 0;
    let mut prev = verts[2];
    let mut prev_inside = prev.position.z >= near_z;
    let mut i = 0;
    while i < verts.len() {
        let current = verts[i];
        let current_inside = current.position.z >= near_z;
        if current_inside != prev_inside {
            out[count] = intersect_textured_near(prev, current, near_z);
            count += 1;
        }
        if current_inside {
            out[count] = current;
            count += 1;
        }
        prev = current;
        prev_inside = current_inside;
        i += 1;
    }
    count
}

fn intersect_textured_near(
    a: TexturedViewVertex,
    b: TexturedViewVertex,
    near_z: i32,
) -> TexturedViewVertex {
    let dz = b.position.z - a.position.z;
    if dz == 0 {
        return TexturedViewVertex::new(
            ViewVertex::new(a.position.x, a.position.y, near_z),
            a.u,
            a.v,
        );
    }

    let num = near_z - a.position.z;
    TexturedViewVertex::new(
        ViewVertex::new(
            lerp_i32(a.position.x, b.position.x, num, dz),
            lerp_i32(a.position.y, b.position.y, num, dz),
            near_z,
        ),
        lerp_i32(a.u, b.u, num, dz),
        lerp_i32(a.v, b.v, num, dz),
    )
}

fn lerp_i32(a: i32, b: i32, numerator: i32, denominator: i32) -> i32 {
    if denominator == 0 {
        return a;
    }
    a.saturating_add(b.saturating_sub(a).saturating_mul(numerator) / denominator)
}

fn merge_world_stats(stats: &mut WorldRenderStats, next: WorldRenderStats) {
    stats.submitted_triangles = stats
        .submitted_triangles
        .saturating_add(next.submitted_triangles);
    stats.culled_triangles = stats.culled_triangles.saturating_add(next.culled_triangles);
    stats.clipped_triangles = stats
        .clipped_triangles
        .saturating_add(next.clipped_triangles);
    stats.split_triangles = stats.split_triangles.saturating_add(next.split_triangles);
    stats.dropped_triangles = stats
        .dropped_triangles
        .saturating_add(next.dropped_triangles);
    stats.primitive_overflow |= next.primitive_overflow;
    stats.command_overflow |= next.command_overflow;
}

fn clamp_i16(value: i32) -> i16 {
    if value < i16::MIN as i32 {
        i16::MIN
    } else if value > i16::MAX as i32 {
        i16::MAX
    } else {
        value as i16
    }
}

fn clamp_i16_range(value: i16, min: i16, max: i16) -> i16 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn clamp_u8(value: i32) -> u8 {
    if value < 0 {
        0
    } else if value > u8::MAX as i32 {
        u8::MAX
    } else {
        value as u8
    }
}

fn isqrt_i32(value: i32) -> i32 {
    if value <= 0 {
        return 0;
    }

    let mut bit = 1 << 30;
    let mut n = value;
    let mut root = 0;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if n >= root + bit {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn sort_for_ot_insert(commands: &mut [GouraudTriCommand]) {
    let mut i = 1;
    while i < commands.len() {
        let command = commands[i];
        let mut j = i;
        while j > 0 && should_insert_after(commands[j - 1], command) {
            commands[j] = commands[j - 1];
            j -= 1;
        }
        commands[j] = command;
        i += 1;
    }
}

fn should_insert_after(a: GouraudTriCommand, b: GouraudTriCommand) -> bool {
    if a.slot.index() != b.slot.index() {
        return a.slot.index() > b.slot.index();
    }
    if a.depth != b.depth {
        return a.depth > b.depth;
    }
    // OT insertion prepends packets. For exact ties, insert later
    // primitive indices first so the eventual DMA walk preserves the
    // source order for those equal-depth packets.
    a.primitive_index < b.primitive_index
}

fn sort_world_for_ot_insert(commands: &mut [WorldTriCommand]) {
    let mut i = 1;
    while i < commands.len() {
        let command = commands[i];
        let mut j = i;
        while j > 0 && should_insert_world_after(commands[j - 1], command) {
            commands[j] = commands[j - 1];
            j -= 1;
        }
        commands[j] = command;
        i += 1;
    }
}

fn should_insert_world_after(a: WorldTriCommand, b: WorldTriCommand) -> bool {
    if a.slot.index() != b.slot.index() {
        return a.slot.index() > b.slot.index();
    }
    if a.depth != b.depth {
        return a.depth > b.depth;
    }
    a.order < b.order
}

#[cfg(test)]
mod tests {
    use super::*;
    use psx_gpu::ot::OrderingTable;

    const fn command(slot: usize, depth: i32, primitive_index: usize) -> GouraudTriCommand {
        GouraudTriCommand {
            slot: DepthSlot::new(slot),
            depth,
            primitive_index,
        }
    }

    const fn world_command(slot: usize, depth: i32, order: usize) -> WorldTriCommand {
        WorldTriCommand {
            slot: DepthSlot::new(slot),
            depth,
            packet_ptr: core::ptr::null_mut(),
            words: 0,
            order,
        }
    }

    #[test]
    fn depth_policy_picks_expected_scalar() {
        let verts = [
            ProjectedLit {
                sx: 0,
                sy: 0,
                sz: 100,
                r: 0,
                g: 0,
                b: 0,
            },
            ProjectedLit {
                sx: 0,
                sy: 0,
                sz: 400,
                r: 0,
                g: 0,
                b: 0,
            },
            ProjectedLit {
                sx: 0,
                sy: 0,
                sz: 700,
                r: 0,
                g: 0,
                b: 0,
            },
        ];

        assert_eq!(DepthPolicy::Average.depth(verts), 400);
        assert_eq!(DepthPolicy::Nearest.depth(verts), 100);
        assert_eq!(DepthPolicy::Farthest.depth(verts), 700);
    }

    #[test]
    fn commands_sort_in_ot_insertion_order() {
        let mut commands = [
            command(5, 600, 0),
            command(5, 300, 1),
            command(3, 400, 2),
            command(5, 300, 3),
        ];

        sort_for_ot_insert(&mut commands);

        assert_eq!(commands[0], command(3, 400, 2));
        assert_eq!(commands[1], command(5, 300, 3));
        assert_eq!(commands[2], command(5, 300, 1));
        assert_eq!(commands[3], command(5, 600, 0));
    }

    #[test]
    fn projected_backface_culling_uses_screen_winding() {
        let front = [
            ProjectedVertex::new(0, 0, 100),
            ProjectedVertex::new(10, 0, 100),
            ProjectedVertex::new(0, 10, 100),
        ];
        let back = [front[0], front[2], front[1]];

        assert!(!projected_back_facing(front));
        assert!(projected_back_facing(back));
    }

    #[test]
    fn textured_near_clip_keeps_visible_polygon() {
        let input = [
            TexturedViewVertex::new(ViewVertex::new(-20, 0, 20), 0, 0),
            TexturedViewVertex::new(ViewVertex::new(20, 0, 80), 63, 0),
            TexturedViewVertex::new(ViewVertex::new(-20, 40, 80), 0, 63),
        ];
        let mut out = [TexturedViewVertex::ZERO; 4];

        let count = clip_textured_triangle_to_near(input, 40, &mut out);

        assert_eq!(count, 4);
        assert_eq!(out[0].position.z, 40);
        assert_eq!(out[1].position.z, 40);
        assert!(out[..count].iter().all(|v| v.position.z >= 40));
    }

    #[test]
    fn textured_submit_splits_triangles_that_exceed_ps1_extent() {
        const ZERO: TriTextured = TriTextured::new(
            [(0, 0), (0, 0), (0, 0)],
            [(0, 0), (0, 0), (0, 0)],
            0,
            0,
            (0, 0, 0),
        );
        let mut ot_storage = OrderingTable::<8>::new();
        let mut ot = OtFrame::begin(&mut ot_storage);
        let mut triangle_storage = [const { ZERO }; 8];
        let mut triangles = PrimitiveArena::new(&mut triangle_storage);
        let mut commands = [WorldTriCommand::EMPTY; 8];
        let mut pass = WorldRenderPass::new(&mut ot, &mut commands);

        let stats = pass.submit_textured_triangle(
            &mut triangles,
            [
                ProjectedVertex::new(0, 0, 100),
                ProjectedVertex::new(0, 700, 100),
                ProjectedVertex::new(128, 0, 100),
            ],
            [(0, 0), (63, 0), (0, 63)],
            TextureMaterial::opaque(0, 0, (128, 128, 128)),
            WorldSurfaceOptions::new(DepthBand::whole(), DepthRange::new(0, 1000))
                .with_cull_mode(CullMode::None),
        );

        assert!(stats.submitted_triangles > 1);
        assert!(stats.split_triangles > 0);
        assert_eq!(stats.dropped_triangles, 0);
        assert!(!stats.primitive_overflow);
        assert!(!stats.command_overflow);
        pass.flush();
    }

    #[test]
    fn world_camera_orbit_projects_target_to_screen_center() {
        let projection = WorldProjection::new(160, 120, 200, 40);
        let target = WorldVertex::new(0, -90, 0);

        let camera = WorldCamera::orbit_yaw(projection, target, 0, 120, 0);
        let projected = camera.project_world(target).expect("target in front");

        assert_eq!(projected.sx, 160);
        assert_eq!(projected.sy, 120);
        assert!(projected.sz >= projection.near_z);
    }

    #[test]
    fn world_camera_projects_world_quad() {
        let projection = WorldProjection::new(160, 120, 200, 40);
        let camera = WorldCamera::orbit_yaw(projection, WorldVertex::ZERO, 0, 200, 0);

        let projected = camera.project_world_quad([
            WorldVertex::new(-10, 10, 0),
            WorldVertex::new(10, 10, 0),
            WorldVertex::new(-10, -10, 0),
            WorldVertex::new(10, -10, 0),
        ]);

        assert!(projected.is_some());
    }

    #[test]
    fn world_projection_accepts_vertices_on_near_plane() {
        let projection = WorldProjection::new(160, 120, 200, 40);

        let projected = projection.project_view(ViewVertex::new(0, 0, 40));

        assert_eq!(projected, Some(ProjectedVertex::new(160, 120, 40)));
    }

    #[test]
    fn world_commands_sort_in_ot_insertion_order() {
        let mut commands = [
            world_command(5, 600, 0),
            world_command(5, 300, 1),
            world_command(3, 400, 2),
            world_command(5, 300, 3),
        ];

        sort_world_for_ot_insert(&mut commands);

        assert_eq!(commands[0], world_command(3, 400, 2));
        assert_eq!(commands[1], world_command(5, 300, 3));
        assert_eq!(commands[2], world_command(5, 300, 1));
        assert_eq!(commands[3], world_command(5, 600, 0));
    }
}
