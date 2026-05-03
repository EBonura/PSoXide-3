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

use crate::render::{CameraDepth, DepthBand, DepthRange, DepthSlot, OtFrame, PrimitiveArena};
use crate::{Angle, WorldVertex, Q12};
use psx_asset::{Animation, JointPose, Mesh, Model, ModelFaceCorner, ModelVertex};
use psx_gpu::{
    material::TextureMaterial,
    prim::{TriGouraud, TriTextured},
};
use psx_gte::{
    lighting::{project_lit, ProjectedLit},
    math::{Mat3I16, Vec3I16, Vec3I32},
    scene,
};

const PSX_VERTEX_MIN: i16 = -1024;
const PSX_VERTEX_MAX: i16 = 1023;
const PSX_TRI_MAX_DX: i32 = 1023;
const PSX_TRI_MAX_DY: i32 = 511;
const MAX_TEXTURED_HW_SPLIT_DEPTH: u8 = 5;
const WORLD_COMMAND_NONE: u16 = u16::MAX;
const GOURAUD_COMMAND_NONE: u16 = u16::MAX;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum WorldCommandOrdering {
    LinkedSorted,
    DeferredSorted,
    DeferredSlotSorted,
    Bucketed,
}

/// Canonical quad → triangle split.
///
/// Quad corners arrive in perimeter order `[0, 1, 2, 3]`:
///
/// ```text
///   0 ─────── 1
///   │         │
///   3 ─────── 2
/// ```
///
/// Both triangles share the `0`–`2` diagonal so the union covers
/// the whole quad with no overlap. A pre-history version split
/// the second triangle as `(2, 1, 3)`, which uses the OTHER
/// diagonal -- the two halves overlap at the `1`–`2` edge and
/// leave a triangular hole near corner `3`. That manifested as
/// the "black triangular gaps" floor-rendering bug. Centralised
/// here so every quad-submitting path uses the same split.
const TEXTURED_QUAD_TRIANGLES: [[usize; 3]; 2] = [[0, 1, 2], [0, 2, 3]];

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

/// Uniform Q12 scale from dense model-local units to engine world units.
///
/// Imported characters can spend the full signed-16-bit vertex range on
/// their own local detail while rooms keep using stable 1024-unit grid
/// sectors. This helper applies the scale without requiring 64-bit
/// multiplication on the PS1 runtime path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LocalToWorldScale {
    scale: Q12,
}

impl LocalToWorldScale {
    /// Identity scale.
    pub const IDENTITY: Self = Self { scale: Q12::ONE };

    /// Build from a Q12 header value. Zero means unspecified and maps to
    /// identity for compatibility with older cooked blobs.
    pub const fn from_q12(q12: u16) -> Self {
        if q12 == 0 {
            Self::IDENTITY
        } else {
            Self {
                scale: Q12::from_raw(q12 as i32),
            }
        }
    }

    /// Raw Q12 scale value.
    pub const fn q12(self) -> u16 {
        self.scale.raw() as u16
    }

    /// Typed Q12 scale value.
    pub const fn scale(self) -> Q12 {
        self.scale
    }

    /// Apply the scale to one signed coordinate.
    pub fn apply(self, value: i32) -> i32 {
        let whole = value >> 12;
        let frac = value - (whole << 12);
        whole.saturating_mul(self.scale.raw()) + self.scale.mul_i32(frac)
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

/// Projected textured vertex used as scratch by GTE-backed textured model paths.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectedTexturedVertex {
    /// Screen-space position and depth.
    pub projected: ProjectedVertex,
    /// Texture U coordinate.
    pub u: i32,
    /// Texture V coordinate.
    pub v: i32,
}

impl ProjectedTexturedVertex {
    /// Build a projected textured vertex.
    pub const fn new(projected: ProjectedVertex, u: i32, v: i32) -> Self {
        Self { projected, u, v }
    }
}

/// Per-joint world-to-view transform for one render frame.
///
/// `submit_textured_model` fills one entry per skin joint up-front so
/// blend-skin vertices can read both their primary and secondary
/// joint matrices without re-deriving them mid-frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct JointViewTransform {
    /// Combined view × model rotation, Q12.
    pub rotation: Mat3I16,
    /// View-space translation, Q0.
    pub translation: Vec3I32,
}

impl JointViewTransform {
    /// All-zero transform suitable for `static mut` scratch storage.
    pub const ZERO: Self = Self {
        rotation: Mat3I16::ZERO,
        translation: Vec3I32::ZERO,
    };
}

/// Per-joint room/world transform for gameplay attachment points.
///
/// Unlike [`JointViewTransform`], this is not camera-relative. It is
/// the model instance's oriented joint pose in room-local world units,
/// suitable for composing sockets, weapon grips, and hit volumes.
#[derive(Copy, Clone, Debug, Default)]
pub struct JointWorldTransform {
    /// Instance × joint rotation, Q12.
    pub rotation: Mat3I16,
    /// Joint origin in room-local world units.
    pub translation: WorldVertex,
}

impl JointWorldTransform {
    /// All-zero transform suitable for fallbacks/static scratch.
    pub const ZERO: Self = Self {
        rotation: Mat3I16::ZERO,
        translation: WorldVertex::ZERO,
    };
}

/// Perspective projection settings for world-space render passes.
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
    /// Build projection settings for world-space render passes.
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

/// Perspective camera for authored world surfaces and GTE model passes.
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
    /// Sine of yaw.
    pub sin_yaw: Q12,
    /// Cosine of yaw.
    pub cos_yaw: Q12,
    /// Sine of pitch.
    pub sin_pitch: Q12,
    /// Cosine of pitch.
    pub cos_pitch: Q12,
}

impl WorldCamera {
    /// Build a camera from an explicit fixed-point basis.
    pub const fn from_basis(
        projection: WorldProjection,
        position: WorldVertex,
        sin_yaw: Q12,
        cos_yaw: Q12,
        sin_pitch: Q12,
        cos_pitch: Q12,
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
    /// `yaw` is the orbit direction from target to camera. `camera_y`
    /// is the camera's absolute world-space height. Pitch is derived
    /// from `target.y - camera_y`, so dollying the radius keeps the
    /// target centred without per-frame call-site math.
    pub fn orbit_yaw(
        projection: WorldProjection,
        target: WorldVertex,
        camera_y: i32,
        radius: i32,
        yaw: Angle,
    ) -> Self {
        let sin_yaw = yaw.sin();
        let cos_yaw = yaw.cos();
        let target_dy = target.y - camera_y;
        let pitch_len =
            isqrt_i32(radius.saturating_mul(radius) + target_dy.saturating_mul(target_dy)).max(1);
        Self {
            projection,
            position: WorldVertex::new(
                target.x + sin_yaw.mul_i32(radius),
                camera_y,
                target.z + cos_yaw.mul_i32(radius),
            ),
            sin_yaw,
            cos_yaw,
            sin_pitch: Q12::from_ratio(target_dy, pitch_len),
            cos_pitch: Q12::from_ratio(radius, pitch_len),
        }
    }

    /// Build a camera on a full spherical orbit around `target`.
    ///
    /// `yaw` and `pitch` are canonical engine angles. The camera sits
    /// at constant `radius` from `target` and looks at it directly:
    /// positive `pitch` raises the camera above the target so the view
    /// tilts down. Pitch wraps freely, so the orbit can pass through
    /// the poles and view the model upside-down.
    pub fn orbit(
        projection: WorldProjection,
        target: WorldVertex,
        radius: i32,
        yaw: Angle,
        pitch: Angle,
    ) -> Self {
        let sin_yaw = yaw.sin();
        let cos_yaw = yaw.cos();
        let sin_pitch = pitch.sin();
        let cos_pitch = pitch.cos();
        let horiz = cos_pitch.mul_i32(radius);
        Self {
            projection,
            position: WorldVertex::new(
                target.x + sin_yaw.mul_i32(horiz),
                target.y - sin_pitch.mul_i32(radius),
                target.z + cos_yaw.mul_i32(horiz),
            ),
            sin_yaw,
            cos_yaw,
            sin_pitch,
            cos_pitch,
        }
    }

    /// Transform a world-space vertex into camera-space.
    pub fn view_vertex(self, vertex: WorldVertex) -> ViewVertex {
        let dx = vertex.x - self.position.x;
        let dy = vertex.y - self.position.y;
        let dz = vertex.z - self.position.z;

        let sin_yaw = self.sin_yaw.raw();
        let cos_yaw = self.cos_yaw.raw();
        let sin_pitch = self.sin_pitch.raw();
        let cos_pitch = self.cos_pitch.raw();
        let x1 = ((dx * cos_yaw) - (dz * sin_yaw)) >> 12;
        let z1 = ((-dx * sin_yaw) - (dz * cos_yaw)) >> 12;
        let y2 = ((dy * cos_pitch) - (z1 * sin_pitch)) >> 12;
        let z2 = ((dy * sin_pitch) + (z1 * cos_pitch)) >> 12;

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

/// Coarse render layer for world surfaces inside one ordering table.
///
/// PS1 ordering tables are still depth-first. This layer only resolves
/// exact slot/depth ties so translucent packets blend over opaque packets
/// submitted into the same OT cell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorldRenderLayer {
    /// Opaque surfaces that overwrite destination pixels.
    Opaque,
    /// Semi-transparent surfaces that should blend with prior pixels.
    Transparent,
}

impl WorldRenderLayer {
    /// Pick the world render layer implied by a texture material.
    pub const fn for_material(material: TextureMaterial) -> Self {
        if material.is_translucent() {
            Self::Transparent
        } else {
            Self::Opaque
        }
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
    /// Coarse tie-break layer for opaque/translucent surfaces.
    pub render_layer: WorldRenderLayer,
    /// Split oversized projected textured triangles before packet emission.
    ///
    /// Room-scale quads keep this enabled because a floor/wall can span most
    /// of the screen. Compact character meshes can disable it to stay on the
    /// direct packet path and spend their budget on GTE transforms instead of
    /// conservative CPU-side subdivision checks.
    pub split_textured_triangles: bool,
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
            render_layer: WorldRenderLayer::Opaque,
            split_textured_triangles: true,
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

    /// Return options with a different render layer.
    pub const fn with_render_layer(mut self, render_layer: WorldRenderLayer) -> Self {
        self.render_layer = render_layer;
        self
    }

    /// Return options using the render layer implied by `material`.
    pub const fn with_material_layer(mut self, material: TextureMaterial) -> Self {
        self.render_layer = WorldRenderLayer::for_material(material);
        self
    }

    /// Return options with textured triangle splitting enabled/disabled.
    pub const fn with_textured_triangle_splitting(mut self, enabled: bool) -> Self {
        self.split_textured_triangles = enabled;
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
    render_layer: WorldRenderLayer,
    packet_ptr: *mut u32,
    words: u8,
    order: usize,
    next: u16,
}

impl WorldTriCommand {
    /// Empty command value for static scratch-buffer initialisation.
    pub const EMPTY: Self = Self {
        slot: DepthSlot::new(0),
        depth: 0,
        render_layer: WorldRenderLayer::Opaque,
        packet_ptr: core::ptr::null_mut(),
        words: 0,
        order: 0,
        next: WORLD_COMMAND_NONE,
    };

    #[cfg(test)]
    pub(crate) const fn depth_raw(self) -> i32 {
        self.depth
    }
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

/// Per-submit counters and overflow flags for textured model rendering.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TexturedModelRenderStats {
    /// Vertices projected into the caller-provided scratch buffer.
    pub projected_vertices: u16,
    /// Triangles accepted into the pass after culling.
    pub submitted_triangles: u16,
    /// Triangles rejected by back-face culling.
    pub culled_triangles: u16,
    /// Oversized projected triangles split to satisfy PS1 hardware
    /// extent limits.
    pub split_triangles: u16,
    /// Triangles skipped because a face referenced vertices outside
    /// the part's projected vertex range.
    pub skipped_triangles: u16,
    /// Triangles dropped before packet emission because they were
    /// behind the near plane or could not be made hardware-legal.
    pub dropped_triangles: u16,
    /// True if the vertex scratch buffer was too small for any part.
    pub vertex_overflow: bool,
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
    slot_heads: [u16; OT_DEPTH],
    slot_tails: [u16; OT_DEPTH],
    command_len: usize,
    next_order: usize,
    ordering: WorldCommandOrdering,
}

impl<'a, 'ot, const OT_DEPTH: usize> WorldRenderPass<'a, 'ot, OT_DEPTH> {
    /// Start a world render pass.
    pub fn new(ot: &'a mut OtFrame<'ot, OT_DEPTH>, commands: &'a mut [WorldTriCommand]) -> Self {
        Self {
            ot,
            commands,
            slot_heads: [WORLD_COMMAND_NONE; OT_DEPTH],
            slot_tails: [WORLD_COMMAND_NONE; OT_DEPTH],
            command_len: 0,
            next_order: 0,
            ordering: WorldCommandOrdering::LinkedSorted,
        }
    }

    /// Start a world render pass that sorts submitted commands at flush.
    ///
    /// This keeps painter ordering comparable to [`Self::new`] while avoiding
    /// the hot per-triangle linked-list insertion cost. It is the preferred
    /// mode for scenes that submit a few hundred opaque world/model packets.
    pub fn new_deferred_sorted(
        ot: &'a mut OtFrame<'ot, OT_DEPTH>,
        commands: &'a mut [WorldTriCommand],
    ) -> Self {
        Self {
            ot,
            commands,
            slot_heads: [WORLD_COMMAND_NONE; OT_DEPTH],
            slot_tails: [WORLD_COMMAND_NONE; OT_DEPTH],
            command_len: 0,
            next_order: 0,
            ordering: WorldCommandOrdering::DeferredSorted,
        }
    }

    /// Start a world render pass that appends commands into OT buckets and
    /// sorts only within each occupied bucket at flush.
    ///
    /// This preserves the same exact same-slot depth/layer/order semantics as
    /// the global deferred sorter, but avoids comparing triangles that already
    /// landed in different ordering-table slots.
    pub fn new_deferred_slot_sorted(
        ot: &'a mut OtFrame<'ot, OT_DEPTH>,
        commands: &'a mut [WorldTriCommand],
    ) -> Self {
        Self {
            ot,
            commands,
            slot_heads: [WORLD_COMMAND_NONE; OT_DEPTH],
            slot_tails: [WORLD_COMMAND_NONE; OT_DEPTH],
            command_len: 0,
            next_order: 0,
            ordering: WorldCommandOrdering::DeferredSlotSorted,
        }
    }

    /// Start a world render pass that appends commands into coarse OT buckets.
    ///
    /// This is the fastest ordered mode: it preserves submission order within
    /// each depth slot and relies on a sufficiently deep OT for depth
    /// separation. It avoids both per-command insertion sorting and frame-end
    /// global sorting.
    pub fn new_bucketed(
        ot: &'a mut OtFrame<'ot, OT_DEPTH>,
        commands: &'a mut [WorldTriCommand],
    ) -> Self {
        Self {
            ot,
            commands,
            slot_heads: [WORLD_COMMAND_NONE; OT_DEPTH],
            slot_tails: [WORLD_COMMAND_NONE; OT_DEPTH],
            command_len: 0,
            next_order: 0,
            ordering: WorldCommandOrdering::Bucketed,
        }
    }

    /// Number of world/model triangle commands queued in this pass.
    pub const fn command_len(&self) -> usize {
        self.command_len
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

    /// Submit a textured triangle whose vertices are already projected.
    ///
    /// This is the common packet path for pre-projected surfaces and
    /// GTE-projected model/world batches. Projection is intentionally
    /// kept outside so the expensive part can happen once per shared
    /// vertex rather than once per face.
    pub fn submit_projected_textured_triangle(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedTexturedVertex; 3],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let mut stats = WorldRenderStats::default();
        let projected = [verts[0].projected, verts[1].projected, verts[2].projected];
        if options.cull_mode == CullMode::Back && projected_back_facing(projected) {
            stats.culled_triangles = 1;
            return stats;
        }

        merge_world_stats(
            &mut stats,
            self.submit_textured_triangle_split(triangles, verts, material, options, 0),
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
        if !options.split_textured_triangles {
            return self.submit_textured_triangle_leaf(triangles, verts, material, options);
        }

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

        let depth = CameraDepth::new(
            options
                .depth_policy
                .depth_values(
                    verts[0].projected.sz,
                    verts[1].projected.sz,
                    verts[2].projected.sz,
                )
                .saturating_add(options.depth_bias),
        );
        self.push_command(
            options
                .depth_band
                .slot_depth::<OT_DEPTH>(options.depth_range, depth),
            depth.raw(),
            if material.is_translucent() {
                WorldRenderLayer::Transparent
            } else {
                options.render_layer
            },
            tri as *mut TriTextured as *mut u32,
            TriTextured::WORDS,
        );
        stats.submitted_triangles = 1;
        stats
    }

    /// Submit a projected textured quad as two independently culled
    /// and sorted textured triangles.
    ///
    /// Corners arrive in perimeter order `[0, 1, 2, 3]`. Triangles
    /// are split along the `0`–`2` diagonal -- see
    /// [`TEXTURED_QUAD_TRIANGLES`] for why the alternate split is
    /// wrong.
    pub fn submit_textured_quad(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [ProjectedVertex; 4],
        uvs: [(u8, u8); 4],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let [a, b, c] = TEXTURED_QUAD_TRIANGLES[0];
        let mut stats = self.submit_textured_triangle(
            triangles,
            [verts[a], verts[b], verts[c]],
            [uvs[a], uvs[b], uvs[c]],
            material,
            options,
        );
        if stats.primitive_overflow || stats.command_overflow {
            return stats;
        }

        let [a, b, c] = TEXTURED_QUAD_TRIANGLES[1];
        let second = self.submit_textured_triangle(
            triangles,
            [verts[a], verts[b], verts[c]],
            [uvs[a], uvs[b], uvs[c]],
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
    ///
    /// Corners arrive in perimeter order `[0, 1, 2, 3]`. Triangles
    /// share the `0`–`2` diagonal per [`TEXTURED_QUAD_TRIANGLES`].
    pub fn submit_textured_view_quad(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        verts: [TexturedViewVertex; 4],
        projection: WorldProjection,
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> WorldRenderStats {
        let [a, b, c] = TEXTURED_QUAD_TRIANGLES[0];
        let mut stats = self.submit_textured_view_triangle(
            triangles,
            [verts[a], verts[b], verts[c]],
            projection,
            material,
            options,
        );
        if stats.primitive_overflow || stats.command_overflow {
            return stats;
        }

        let [a, b, c] = TEXTURED_QUAD_TRIANGLES[1];
        let second = self.submit_textured_view_triangle(
            triangles,
            [verts[a], verts[b], verts[c]],
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

    /// Submit an animated rigid-skeletal textured model through the GTE.
    ///
    /// The engine loads one GTE transform per model part, projects
    /// each compact skinned vertex once into `projected_vertices`, and
    /// then builds textured triangle packets from face-corner UVs.
    /// Single-bone vertices are batched through `RTPT` three at a
    /// time; blend vertices take the CPU path because they need two
    /// joint transforms before projection.
    /// `frame_q12` is a looping sampled-frame phase with 12
    /// fractional bits, so decimated animation clips can still play
    /// smoothly between stored poses.
    ///
    /// `instance_rotation` is composed *between* the per-joint pose
    /// matrix and the camera view matrix, so callers placing the
    /// model in a world with a non-identity orientation (e.g. a
    /// yawed enemy / NPC) can pass that rotation here. Use
    /// `Mat3I16::IDENTITY` for unrotated instances -- that's the
    /// existing showcase-model behaviour.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_textured_model(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        model: Model<'_>,
        animation: Animation<'_>,
        frame_q12: u32,
        camera: WorldCamera,
        origin: WorldVertex,
        instance_rotation: Mat3I16,
        projected_vertices: &mut [ProjectedVertex],
        joint_view_transforms: &mut [JointViewTransform],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> TexturedModelRenderStats {
        self.submit_textured_model_impl(
            triangles,
            model,
            animation,
            frame_q12,
            camera,
            origin,
            instance_rotation,
            projected_vertices,
            joint_view_transforms,
            material,
            options,
            true,
        )
    }

    /// Submit an animated textured model using each vertex's primary joint only.
    ///
    /// This is the high-throughput NPC/background-character path. It keeps
    /// every vertex on the GTE projection fast path and ignores secondary
    /// blend weights, trading some joint-boundary smoothness for a much lower
    /// CPU cost.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_textured_model_primary_joints(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        model: Model<'_>,
        animation: Animation<'_>,
        frame_q12: u32,
        camera: WorldCamera,
        origin: WorldVertex,
        instance_rotation: Mat3I16,
        projected_vertices: &mut [ProjectedVertex],
        joint_view_transforms: &mut [JointViewTransform],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
    ) -> TexturedModelRenderStats {
        self.submit_textured_model_impl(
            triangles,
            model,
            animation,
            frame_q12,
            camera,
            origin,
            instance_rotation,
            projected_vertices,
            joint_view_transforms,
            material,
            options,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_textured_model_impl(
        &mut self,
        triangles: &mut PrimitiveArena<'_, TriTextured>,
        model: Model<'_>,
        animation: Animation<'_>,
        frame_q12: u32,
        camera: WorldCamera,
        origin: WorldVertex,
        instance_rotation: Mat3I16,
        projected_vertices: &mut [ProjectedVertex],
        joint_view_transforms: &mut [JointViewTransform],
        material: TextureMaterial,
        options: WorldSurfaceOptions,
        blend_vertices: bool,
    ) -> TexturedModelRenderStats {
        let mut stats = TexturedModelRenderStats::default();
        let local_to_world = LocalToWorldScale::from_q12(model.local_to_world_q12());
        load_world_projection_gte(camera.projection);

        let joint_count = (model.joint_count() as usize).min(joint_view_transforms.len());
        for (joint, joint_view_transform) in joint_view_transforms
            .iter_mut()
            .enumerate()
            .take(joint_count)
        {
            *joint_view_transform = match animation.pose_looped_q12(frame_q12, joint as u16) {
                Some(pose) => {
                    let (rotation, translation) = textured_model_part_gte_transform(
                        camera,
                        pose,
                        instance_rotation,
                        local_to_world,
                        origin,
                    );
                    JointViewTransform {
                        rotation,
                        translation,
                    }
                }
                None => JointViewTransform::default(),
            };
        }

        let project_count = (model.vertex_count() as usize)
            .min(projected_vertices.len())
            .min(u16::MAX as usize);
        if project_count < model.vertex_count() as usize {
            stats.vertex_overflow = true;
        }
        stats.projected_vertices = project_count as u16;

        let mut part_index = 0;
        while part_index < model.part_count() {
            let Some(part) = model.part(part_index) else {
                break;
            };
            let primary_joint = part.joint_index() as usize;
            if primary_joint >= joint_count {
                part_index += 1;
                continue;
            }
            let primary = joint_view_transforms[primary_joint];

            scene::load_rotation(&primary.rotation);
            scene::load_translation(primary.translation);

            let mut global_index = part.first_vertex() as usize;
            let part_end = global_index
                .saturating_add(part.vertex_count() as usize)
                .min(project_count);
            while global_index < part_end {
                let Some(vertex) = model.vertex(global_index as u16) else {
                    break;
                };
                if blend_vertices && model_vertex_uses_cpu_blend(vertex, joint_count) {
                    projected_vertices[global_index] = project_textured_model_vertex(
                        vertex,
                        primary,
                        joint_view_transforms,
                        joint_count,
                        camera.projection,
                    );
                    global_index += 1;
                    continue;
                }

                let mut batch = [vertex; 3];
                let mut batch_count = 1usize;
                while batch_count < 3 && global_index + batch_count < part_end {
                    let Some(next) = model.vertex((global_index + batch_count) as u16) else {
                        break;
                    };
                    if blend_vertices && model_vertex_uses_cpu_blend(next, joint_count) {
                        break;
                    }
                    batch[batch_count] = next;
                    batch_count += 1;
                }

                if batch_count == 3 {
                    let projected = scene::project_triangle(
                        batch[0].position,
                        batch[1].position,
                        batch[2].position,
                    );
                    projected_vertices[global_index] = projected_from_gte(projected[0]);
                    projected_vertices[global_index + 1] = projected_from_gte(projected[1]);
                    projected_vertices[global_index + 2] = projected_from_gte(projected[2]);
                } else {
                    let mut batch_index = 0usize;
                    while batch_index < batch_count {
                        projected_vertices[global_index + batch_index] =
                            project_gte_model_vertex(batch[batch_index]);
                        batch_index += 1;
                    }
                }
                global_index += batch_count;
            }

            part_index += 1;
        }

        part_index = 0;
        while part_index < model.part_count() {
            let Some(part) = model.part(part_index) else {
                break;
            };
            let first_face = part.first_face();
            let last_face = first_face.saturating_add(part.face_count());
            let mut face_index = first_face;
            while face_index < last_face {
                let Some(face) = model.face(face_index) else {
                    break;
                };
                let Some(a) =
                    textured_model_corner(projected_vertices, project_count, face.corners[0])
                else {
                    stats.skipped_triangles = stats.skipped_triangles.saturating_add(1);
                    face_index += 1;
                    continue;
                };
                let Some(b) =
                    textured_model_corner(projected_vertices, project_count, face.corners[1])
                else {
                    stats.skipped_triangles = stats.skipped_triangles.saturating_add(1);
                    face_index += 1;
                    continue;
                };
                let Some(c) =
                    textured_model_corner(projected_vertices, project_count, face.corners[2])
                else {
                    stats.skipped_triangles = stats.skipped_triangles.saturating_add(1);
                    face_index += 1;
                    continue;
                };

                if a.projected.sz < camera.projection.near_z
                    || b.projected.sz < camera.projection.near_z
                    || c.projected.sz < camera.projection.near_z
                {
                    stats.dropped_triangles = stats.dropped_triangles.saturating_add(1);
                    face_index += 1;
                    continue;
                }

                let tri_stats = self.submit_projected_textured_triangle(
                    triangles,
                    [a, b, c],
                    material,
                    options,
                );
                merge_textured_model_stats(&mut stats, tri_stats);
                if stats.primitive_overflow || stats.command_overflow {
                    return stats;
                }

                face_index += 1;
            }

            part_index += 1;
        }

        stats
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

        let depth =
            CameraDepth::new(options.depth_policy.depth(verts)).saturating_add(options.depth_bias);
        self.push_command(
            options
                .depth_band
                .slot_depth::<OT_DEPTH>(options.depth_range, depth),
            depth.raw(),
            options.render_layer,
            tri as *mut TriGouraud as *mut u32,
            TriGouraud::WORDS,
        );
        stats.submitted_triangles = 1;
        stats
    }

    fn push_command(
        &mut self,
        slot: DepthSlot,
        depth: i32,
        render_layer: WorldRenderLayer,
        packet_ptr: *mut u32,
        words: u8,
    ) {
        let command_index = self.command_len;
        self.commands[command_index] = WorldTriCommand {
            slot,
            depth,
            render_layer,
            packet_ptr,
            words,
            order: self.next_order,
            next: WORLD_COMMAND_NONE,
        };
        self.command_len += 1;
        self.next_order = self.next_order.saturating_add(1);
        match self.ordering {
            WorldCommandOrdering::LinkedSorted => self.insert_command_in_slot(command_index),
            WorldCommandOrdering::DeferredSorted => {}
            WorldCommandOrdering::DeferredSlotSorted => self.append_command_in_slot(command_index),
            WorldCommandOrdering::Bucketed => self.append_command_in_slot(command_index),
        }
    }

    fn append_command_in_slot(&mut self, command_index: usize) {
        if OT_DEPTH == 0 || command_index >= WORLD_COMMAND_NONE as usize {
            return;
        }

        let slot = self.commands[command_index].slot.index().min(OT_DEPTH - 1);
        let command_link = command_index as u16;
        let tail = self.slot_tails[slot];
        if tail == WORLD_COMMAND_NONE {
            self.slot_heads[slot] = command_link;
        } else {
            self.commands[tail as usize].next = command_link;
        }
        self.slot_tails[slot] = command_link;
    }

    fn insert_command_in_slot(&mut self, command_index: usize) {
        if OT_DEPTH == 0 || command_index >= WORLD_COMMAND_NONE as usize {
            return;
        }

        let slot = self.commands[command_index].slot.index().min(OT_DEPTH - 1);
        let command_link = command_index as u16;
        let head = self.slot_heads[slot];
        if head == WORLD_COMMAND_NONE
            || should_insert_world_before(
                self.commands[command_index],
                self.commands[head as usize],
            )
        {
            self.commands[command_index].next = head;
            self.slot_heads[slot] = command_link;
            return;
        }

        let mut prev = head as usize;
        loop {
            let next = self.commands[prev].next;
            if next == WORLD_COMMAND_NONE
                || should_insert_world_before(
                    self.commands[command_index],
                    self.commands[next as usize],
                )
            {
                self.commands[command_index].next = next;
                self.commands[prev].next = command_link;
                return;
            }
            prev = next as usize;
        }
    }

    fn reverse_bucket_links(&mut self) {
        let mut slot = 0;
        while slot < OT_DEPTH {
            let mut previous = WORLD_COMMAND_NONE;
            let mut current = self.slot_heads[slot];
            self.slot_tails[slot] = current;
            while current != WORLD_COMMAND_NONE {
                let next = self.commands[current as usize].next;
                self.commands[current as usize].next = previous;
                previous = current;
                current = next;
            }
            self.slot_heads[slot] = previous;
            slot += 1;
        }
    }

    fn sort_slot_links(&mut self) {
        let mut slot = 0;
        while slot < OT_DEPTH {
            self.slot_heads[slot] = self.merge_sort_slot_links(self.slot_heads[slot]);
            self.slot_tails[slot] = WORLD_COMMAND_NONE;
            slot += 1;
        }
    }

    fn merge_sort_slot_links(&mut self, head: u16) -> u16 {
        if head == WORLD_COMMAND_NONE {
            return head;
        }
        let next = self.commands[head as usize].next;
        if next == WORLD_COMMAND_NONE {
            return head;
        }

        let mid = self.split_slot_links(head);
        let left = self.merge_sort_slot_links(head);
        let right = self.merge_sort_slot_links(mid);
        self.merge_sorted_slot_links(left, right)
    }

    fn split_slot_links(&mut self, head: u16) -> u16 {
        let mut slow = head;
        let mut fast = self.commands[head as usize].next;
        while fast != WORLD_COMMAND_NONE {
            fast = self.commands[fast as usize].next;
            if fast != WORLD_COMMAND_NONE {
                slow = self.commands[slow as usize].next;
                fast = self.commands[fast as usize].next;
            }
        }

        let mid = self.commands[slow as usize].next;
        self.commands[slow as usize].next = WORLD_COMMAND_NONE;
        mid
    }

    fn merge_sorted_slot_links(&mut self, mut left: u16, mut right: u16) -> u16 {
        let mut head = WORLD_COMMAND_NONE;
        let mut tail = WORLD_COMMAND_NONE;

        while left != WORLD_COMMAND_NONE && right != WORLD_COMMAND_NONE {
            let take_left = !should_insert_world_before(
                self.commands[right as usize],
                self.commands[left as usize],
            );
            let link = if take_left {
                let next = self.commands[left as usize].next;
                let out = left;
                left = next;
                out
            } else {
                let next = self.commands[right as usize].next;
                let out = right;
                right = next;
                out
            };
            self.commands[link as usize].next = WORLD_COMMAND_NONE;
            if head == WORLD_COMMAND_NONE {
                head = link;
            } else {
                self.commands[tail as usize].next = link;
            }
            tail = link;
        }

        let rest = if left != WORLD_COMMAND_NONE {
            left
        } else {
            right
        };
        if head == WORLD_COMMAND_NONE {
            rest
        } else {
            self.commands[tail as usize].next = rest;
            head
        }
    }

    /// Sort and insert all submitted triangles into the ordering table.
    pub fn flush(mut self) {
        if self.ordering == WorldCommandOrdering::DeferredSorted {
            sort_world_for_ot_insert(&mut self.commands[..self.command_len]);
            let mut command_index = 0;
            while command_index < self.command_len {
                let command = self.commands[command_index];
                if !command.packet_ptr.is_null() {
                    // SAFETY: Commands are created only from primitive
                    // arenas borrowed by submit methods. Those packets live
                    // until after this pass flushes and the frame submits.
                    unsafe {
                        self.ot
                            .add_raw_slot(command.slot, command.packet_ptr, command.words)
                    };
                }
                command_index += 1;
            }
            return;
        }

        if self.ordering == WorldCommandOrdering::DeferredSlotSorted {
            self.sort_slot_links();
        } else if self.ordering == WorldCommandOrdering::Bucketed {
            // OrderingTable::insert prepends packets. Bucketed mode appends
            // submissions so reversing each bucket once here makes the final
            // GPU DMA walk preserve same-slot submission order without doing
            // a per-triangle sorted insertion.
            self.reverse_bucket_links();
        }

        let mut slot = 0;
        while slot < OT_DEPTH {
            let mut command_index = self.slot_heads[slot];
            while command_index != WORLD_COMMAND_NONE {
                let command = self.commands[command_index as usize];
                if !command.packet_ptr.is_null() {
                    // SAFETY: Commands are created only from primitive
                    // arenas borrowed by submit methods. Those packets live
                    // until after this pass flushes and the frame submits.
                    unsafe {
                        self.ot
                            .add_raw_slot(command.slot, command.packet_ptr, command.words)
                    };
                }
                command_index = command.next;
            }
            slot += 1;
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
    next: u16,
}

impl GouraudTriCommand {
    /// Empty command value for static scratch-buffer initialisation.
    pub const EMPTY: Self = Self {
        slot: DepthSlot::new(0),
        depth: 0,
        primitive_index: 0,
        next: GOURAUD_COMMAND_NONE,
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
    slot_heads: [u16; OT_DEPTH],
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
            slot_heads: [GOURAUD_COMMAND_NONE; OT_DEPTH],
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
        let project_count = mesh_verts
            .min(projected_vertices.len())
            .min(u16::MAX as usize);
        stats.vertex_overflow = mesh_verts > project_count;
        stats.projected_vertices = project_count as u16;

        let mut vi = 0;
        while vi < project_count {
            let vert = vi as u16;
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

            let depth = CameraDepth::new(options.depth_policy.depth(verts))
                .saturating_add(options.depth_bias);
            let command_index = self.command_len;
            self.commands[command_index] = GouraudTriCommand {
                slot: options
                    .depth_band
                    .slot_depth::<OT_DEPTH>(options.depth_range, depth),
                depth: depth.raw(),
                primitive_index,
                next: GOURAUD_COMMAND_NONE,
            };
            self.command_len += 1;
            self.insert_command_in_slot(command_index);
            stats.submitted_triangles = stats.submitted_triangles.saturating_add(1);
            face_idx += 1;
        }

        stats
    }

    fn insert_command_in_slot(&mut self, command_index: usize) {
        if OT_DEPTH == 0 || command_index >= GOURAUD_COMMAND_NONE as usize {
            return;
        }

        let slot = self.commands[command_index].slot.index().min(OT_DEPTH - 1);
        let command_link = command_index as u16;
        let head = self.slot_heads[slot];
        if head == GOURAUD_COMMAND_NONE
            || should_insert_gouraud_before(
                self.commands[command_index],
                self.commands[head as usize],
            )
        {
            self.commands[command_index].next = head;
            self.slot_heads[slot] = command_link;
            return;
        }

        let mut prev = head as usize;
        loop {
            let next = self.commands[prev].next;
            if next == GOURAUD_COMMAND_NONE
                || should_insert_gouraud_before(
                    self.commands[command_index],
                    self.commands[next as usize],
                )
            {
                self.commands[command_index].next = next;
                self.commands[prev].next = command_link;
                return;
            }
            prev = next as usize;
        }
    }

    /// Sort and insert all submitted triangles into the ordering table.
    pub fn flush(self) {
        let mut slot = 0;
        while slot < OT_DEPTH {
            let mut command_index = self.slot_heads[slot];
            while command_index != GOURAUD_COMMAND_NONE {
                let command = self.commands[command_index as usize];
                if let Some(tri) = self.triangles.get_mut(command.primitive_index) {
                    self.ot.add_packet_slot(command.slot, tri);
                }
                command_index = command.next;
            }
            slot += 1;
        }
    }
}

fn vertex_material(mesh: &Mesh<'_>, vert: u16, fallback: (u8, u8, u8)) -> (u8, u8, u8) {
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

fn textured_model_corner(
    projected_vertices: &[ProjectedVertex],
    project_count: usize,
    corner: ModelFaceCorner,
) -> Option<ProjectedTexturedVertex> {
    if corner.vertex_index as usize >= project_count {
        return None;
    }
    projected_vertices
        .get(corner.vertex_index as usize)
        .copied()
        .map(|projected| {
            ProjectedTexturedVertex::new(projected, corner.uv.0 as i32, corner.uv.1 as i32)
        })
}

fn load_world_projection_gte(projection: WorldProjection) {
    scene::set_screen_offset(
        (projection.screen_x as i32) << 16,
        (projection.screen_y as i32) << 16,
    );
    scene::set_projection_plane(clamp_u16_i32(projection.focal_length));
}

/// Compose the GTE transform for one joint of a placed model
/// instance: `view × instance × pose_model_to_world`. The
/// returned matrix loads into GTE rotation; the returned vector
/// is the camera-space translation (already pre-rotated by the
/// view matrix). Public so the host editor preview can drive
/// the same math without re-implementing it.
pub fn compute_joint_view_transform(
    camera: WorldCamera,
    pose: JointPose,
    instance_rotation: Mat3I16,
    local_to_world: LocalToWorldScale,
    origin: WorldVertex,
) -> (Mat3I16, Vec3I32) {
    textured_model_part_gte_transform(camera, pose, instance_rotation, local_to_world, origin)
}

/// Compose the world-space transform for one animated model joint.
///
/// This shares the same `instance × pose_model_to_world` math used by
/// [`WorldRenderPass::submit_textured_model`], but stops before camera
/// view composition so gameplay systems can attach child objects to
/// animated joints.
pub fn compute_joint_world_transform(
    pose: JointPose,
    instance_rotation: Mat3I16,
    local_to_world: LocalToWorldScale,
    origin: WorldVertex,
) -> JointWorldTransform {
    let model = scaled_pose_matrix(pose, local_to_world);
    let rotation = mat3_mul_q12(&instance_rotation, &model);
    let scaled_pose_translation = Vec3I32::new(
        local_to_world.apply(pose.translation.x),
        local_to_world.apply(pose.translation.y),
        local_to_world.apply(pose.translation.z),
    );
    let rotated_pose_translation =
        rotate_translation_q12(&instance_rotation, scaled_pose_translation);
    JointWorldTransform {
        rotation,
        translation: WorldVertex::new(
            origin.x.saturating_add(rotated_pose_translation.x),
            origin.y.saturating_add(rotated_pose_translation.y),
            origin.z.saturating_add(rotated_pose_translation.z),
        ),
    }
}

/// Project one model vertex using the same GTE/CPU-blend split as
/// [`WorldRenderPass::submit_textured_model`].
///
/// The caller must have already loaded the primary joint transform
/// into the GTE. Vertices without a valid secondary blend joint use
/// the GTE path; blend vertices use the CPU view/projection path so
/// host previews and runtime rendering keep identical deformation.
pub fn project_model_vertex_with_joint_transforms(
    vertex: ModelVertex,
    primary: JointViewTransform,
    joint_view_transforms: &[JointViewTransform],
    joint_count: usize,
    projection: WorldProjection,
) -> ProjectedVertex {
    project_textured_model_vertex(
        vertex,
        primary,
        joint_view_transforms,
        joint_count,
        projection,
    )
}

fn textured_model_part_gte_transform(
    camera: WorldCamera,
    pose: JointPose,
    instance_rotation: Mat3I16,
    local_to_world: LocalToWorldScale,
    origin: WorldVertex,
) -> (Mat3I16, Vec3I32) {
    let view = camera_gte_view_matrix(camera);
    let model = scaled_pose_matrix(pose, local_to_world);
    // Composition order: view × instance × model. `instance` is
    // pre-multiplied through to rotate the joint pose around
    // the model origin in world space; `view` then rotates the
    // already-oriented model into camera space.
    let oriented = mat3_mul_q12(&instance_rotation, &model);
    let rotation = mat3_mul_q12(&view, &oriented);

    // Pose translation needs the same instance rotation before
    // it lands at world space -- otherwise a yawed model's joints
    // would translate along model-local axes rather than world
    // axes.
    let scaled_pose_translation = Vec3I32::new(
        local_to_world.apply(pose.translation.x),
        local_to_world.apply(pose.translation.y),
        local_to_world.apply(pose.translation.z),
    );
    let rotated_pose_translation =
        rotate_translation_q12(&instance_rotation, scaled_pose_translation);

    let world_translation = WorldVertex::new(
        origin.x.saturating_add(rotated_pose_translation.x),
        origin.y.saturating_add(rotated_pose_translation.y),
        origin.z.saturating_add(rotated_pose_translation.z),
    );
    let delta = WorldVertex::new(
        world_translation.x.saturating_sub(camera.position.x),
        world_translation.y.saturating_sub(camera.position.y),
        world_translation.z.saturating_sub(camera.position.z),
    );
    let translation = Vec3I32::new(
        dot_world_q12(view.m[0], delta),
        dot_world_q12(view.m[1], delta),
        dot_world_q12(view.m[2], delta),
    );

    (rotation, translation)
}

/// Apply a Q12 rotation matrix to an i32 translation vector.
/// Runtime pose translations are model-local and bounded by cooked
/// asset scale, so keep this on the PS1's native 32-bit fast path.
fn rotate_translation_q12(rot: &Mat3I16, t: Vec3I32) -> Vec3I32 {
    let row = |r: [i16; 3]| -> i32 {
        let x = (r[0] as i32).saturating_mul(t.x);
        let y = (r[1] as i32).saturating_mul(t.y);
        let z = (r[2] as i32).saturating_mul(t.z);
        x.saturating_add(y).saturating_add(z) >> 12
    };
    Vec3I32::new(row(rot.m[0]), row(rot.m[1]), row(rot.m[2]))
}

fn camera_gte_view_matrix(camera: WorldCamera) -> Mat3I16 {
    let sy_sp = camera.sin_yaw.mul_q12(camera.sin_pitch).raw();
    let cy_sp = camera.cos_yaw.mul_q12(camera.sin_pitch).raw();
    let sy_cp = camera.sin_yaw.mul_q12(camera.cos_pitch).raw();
    let cy_cp = camera.cos_yaw.mul_q12(camera.cos_pitch).raw();

    Mat3I16 {
        m: [
            [
                clamp_i16(camera.cos_yaw.raw()),
                0,
                clamp_i16(-camera.sin_yaw.raw()),
            ],
            [
                clamp_i16(-sy_sp),
                clamp_i16(-camera.cos_pitch.raw()),
                clamp_i16(-cy_sp),
            ],
            [
                clamp_i16(-sy_cp),
                clamp_i16(camera.sin_pitch.raw()),
                clamp_i16(-cy_cp),
            ],
        ],
    }
}

fn scaled_pose_matrix(pose: JointPose, local_to_world: LocalToWorldScale) -> Mat3I16 {
    let scale = local_to_world.scale();
    let mut out = [[0i16; 3]; 3];
    let mut row = 0;
    while row < 3 {
        let mut col = 0;
        while col < 3 {
            out[row][col] = clamp_i16(scale.mul_i32(pose.matrix[col][row] as i32));
            col += 1;
        }
        row += 1;
    }
    Mat3I16 { m: out }
}

fn mat3_mul_q12(a: &Mat3I16, b: &Mat3I16) -> Mat3I16 {
    let mut out = [[0i16; 3]; 3];
    let mut row = 0;
    while row < 3 {
        let mut col = 0;
        while col < 3 {
            let mut sum = 0i32;
            let mut k = 0;
            while k < 3 {
                sum = sum.saturating_add((a.m[row][k] as i32) * (b.m[k][col] as i32));
                k += 1;
            }
            out[row][col] = clamp_i16(sum >> 12);
            col += 1;
        }
        row += 1;
    }
    Mat3I16 { m: out }
}

fn dot_world_q12(row: [i16; 3], v: WorldVertex) -> i32 {
    let x = (row[0] as i32).saturating_mul(v.x);
    let y = (row[1] as i32).saturating_mul(v.y);
    let z = (row[2] as i32).saturating_mul(v.z);
    x.saturating_add(y).saturating_add(z) >> 12
}

/// Software-side equivalent of one GTE RTPS transform stage.
///
/// Used by the blend-skin slow path: a vertex with weight on a second
/// joint cannot stay on the GTE because the rotation/translation
/// registers are loaded for the part's primary joint. We compute its
/// view-space position twice on the CPU, lerp, and project on the CPU
/// using `WorldProjection::project_view`.
#[inline]
fn cpu_view_transform(transform: &JointViewTransform, position: Vec3I16) -> ViewVertex {
    let vx = position.x as i32;
    let vy = position.y as i32;
    let vz = position.z as i32;
    let m = &transform.rotation.m;
    let x = ((m[0][0] as i32) * vx + (m[0][1] as i32) * vy + (m[0][2] as i32) * vz) >> 12;
    let y = ((m[1][0] as i32) * vx + (m[1][1] as i32) * vy + (m[1][2] as i32) * vz) >> 12;
    let z = ((m[2][0] as i32) * vx + (m[2][1] as i32) * vy + (m[2][2] as i32) * vz) >> 12;
    ViewVertex::new(
        x.saturating_add(transform.translation.x),
        y.saturating_add(transform.translation.y),
        z.saturating_add(transform.translation.z),
    )
}

fn project_textured_model_vertex(
    vertex: ModelVertex,
    primary: JointViewTransform,
    joint_view_transforms: &[JointViewTransform],
    joint_count: usize,
    projection: WorldProjection,
) -> ProjectedVertex {
    if model_vertex_uses_cpu_blend(vertex, joint_count) {
        let secondary = joint_view_transforms[vertex.joint1 as usize];
        let view_a = cpu_view_transform(&primary, vertex.position);
        let view_b = cpu_view_transform(&secondary, vertex.position);
        let view_blend = lerp_view_vertex(view_a, view_b, vertex.blend);
        match cpu_project_gte_view(view_blend, projection) {
            Some(proj) => proj,
            None => ProjectedVertex::new(0, 0, projection.near_z - 1),
        }
    } else {
        project_gte_model_vertex(vertex)
    }
}

#[inline]
fn model_vertex_uses_cpu_blend(vertex: ModelVertex, joint_count: usize) -> bool {
    vertex.is_blend() && (vertex.joint1 as usize) < joint_count
}

#[inline]
fn project_gte_model_vertex(vertex: ModelVertex) -> ProjectedVertex {
    projected_from_gte(scene::project_vertex(vertex.position))
}

#[inline]
fn projected_from_gte(projected: scene::Projected) -> ProjectedVertex {
    ProjectedVertex::new(projected.sx, projected.sy, projected.sz as i32)
}

/// CPU projection that matches the GTE RTPS convention used by the
/// rest of the model render path.
///
/// `WorldProjection::project_view` is for *unflipped* camera-space
/// vertices and applies its own `screen_y -= y*H/z` flip. The
/// view-space output of [`cpu_view_transform`] is already pre-flipped
/// in Y by the GTE-style camera matrix in
/// [`camera_gte_view_matrix`], so we project with `screen_y += y*H/z`
/// to avoid the double-flip that put blend verts on the wrong half
/// of the screen.
#[inline]
fn cpu_project_gte_view(view: ViewVertex, projection: WorldProjection) -> Option<ProjectedVertex> {
    if view.z <= 0 || view.z < projection.near_z {
        return None;
    }
    let sx = (projection.screen_x as i32) + (view.x * projection.focal_length) / view.z;
    let sy = (projection.screen_y as i32) + (view.y * projection.focal_length) / view.z;
    Some(ProjectedVertex::new(clamp_i16(sx), clamp_i16(sy), view.z))
}

/// 256-step linear blend between two view-space positions.
///
/// `t` is the cooked blend byte: `0` returns `a` exactly, `255` returns
/// (255 a + 1 b) / 256 -- close enough to `b` for skin-deform purposes
/// and avoids the expensive divide-by-255 a true unit lerp would cost.
#[inline]
fn lerp_view_vertex(a: ViewVertex, b: ViewVertex, t: u8) -> ViewVertex {
    let t = t as i32;
    let inv = 256 - t;
    ViewVertex::new(
        ((a.x.saturating_mul(inv)).saturating_add(b.x.saturating_mul(t))) >> 8,
        ((a.y.saturating_mul(inv)).saturating_add(b.y.saturating_mul(t))) >> 8,
        ((a.z.saturating_mul(inv)).saturating_add(b.z.saturating_mul(t))) >> 8,
    )
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

fn merge_textured_model_stats(stats: &mut TexturedModelRenderStats, next: WorldRenderStats) {
    stats.submitted_triangles = stats
        .submitted_triangles
        .saturating_add(next.submitted_triangles);
    stats.culled_triangles = stats.culled_triangles.saturating_add(next.culled_triangles);
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

fn clamp_u16_i32(value: i32) -> u16 {
    if value < 0 {
        0
    } else if value > u16::MAX as i32 {
        u16::MAX
    } else {
        value as u16
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

#[cfg(test)]
fn sort_for_ot_insert(commands: &mut [GouraudTriCommand]) {
    let mut gap = commands.len() / 2;
    while gap > 0 {
        let mut i = gap;
        while i < commands.len() {
            let command = commands[i];
            let mut j = i;
            while j >= gap && should_insert_after(commands[j - gap], command) {
                commands[j] = commands[j - gap];
                j -= gap;
            }
            commands[j] = command;
            i += 1;
        }
        gap /= 2;
    }
}

#[cfg(test)]
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

fn should_insert_gouraud_before(a: GouraudTriCommand, b: GouraudTriCommand) -> bool {
    if a.depth != b.depth {
        return a.depth < b.depth;
    }
    a.primitive_index > b.primitive_index
}

fn sort_world_for_ot_insert(commands: &mut [WorldTriCommand]) {
    let mut gap = commands.len() / 2;
    while gap > 0 {
        let mut i = gap;
        while i < commands.len() {
            let command = commands[i];
            let mut j = i;
            while j >= gap && should_insert_world_after(commands[j - gap], command) {
                commands[j] = commands[j - gap];
                j -= gap;
            }
            commands[j] = command;
            i += 1;
        }
        gap /= 2;
    }
}

fn should_insert_world_after(a: WorldTriCommand, b: WorldTriCommand) -> bool {
    if a.slot.index() != b.slot.index() {
        return a.slot.index() > b.slot.index();
    }
    if a.depth != b.depth {
        return a.depth > b.depth;
    }
    if a.render_layer != b.render_layer {
        return a.render_layer == WorldRenderLayer::Opaque
            && b.render_layer == WorldRenderLayer::Transparent;
    }
    a.order < b.order
}

fn should_insert_world_before(a: WorldTriCommand, b: WorldTriCommand) -> bool {
    if a.depth != b.depth {
        return a.depth < b.depth;
    }
    if a.render_layer != b.render_layer {
        return a.render_layer == WorldRenderLayer::Transparent
            && b.render_layer == WorldRenderLayer::Opaque;
    }
    a.order > b.order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RoomPoint;
    use psx_gpu::material::BlendMode;
    use psx_gpu::ot::OrderingTable;

    const fn command(slot: usize, depth: i32, primitive_index: usize) -> GouraudTriCommand {
        GouraudTriCommand {
            slot: DepthSlot::new(slot),
            depth,
            primitive_index,
            next: GOURAUD_COMMAND_NONE,
        }
    }

    const fn world_command(slot: usize, depth: i32, order: usize) -> WorldTriCommand {
        world_command_layer(slot, depth, WorldRenderLayer::Opaque, order)
    }

    const fn world_command_layer(
        slot: usize,
        depth: i32,
        render_layer: WorldRenderLayer,
        order: usize,
    ) -> WorldTriCommand {
        WorldTriCommand {
            slot: DepthSlot::new(slot),
            depth,
            render_layer,
            packet_ptr: core::ptr::null_mut(),
            words: 0,
            order,
            next: WORLD_COMMAND_NONE,
        }
    }

    #[test]
    fn bucketed_world_pass_reverses_flush_order_for_ot_prepend() {
        let mut ot_storage = OrderingTable::<8>::new();
        let mut ot = OtFrame::begin(&mut ot_storage);
        let mut commands = [WorldTriCommand::EMPTY; 3];
        let mut pass = WorldRenderPass::new_bucketed(&mut ot, &mut commands);

        pass.push_command(
            DepthSlot::new(4),
            100,
            WorldRenderLayer::Opaque,
            core::ptr::null_mut(),
            0,
        );
        pass.push_command(
            DepthSlot::new(4),
            100,
            WorldRenderLayer::Opaque,
            core::ptr::null_mut(),
            0,
        );
        pass.push_command(
            DepthSlot::new(4),
            100,
            WorldRenderLayer::Opaque,
            core::ptr::null_mut(),
            0,
        );

        assert_eq!(pass.slot_heads[4], 0);
        assert_eq!(pass.commands[0].next, 1);
        assert_eq!(pass.commands[1].next, 2);
        assert_eq!(pass.commands[2].next, WORLD_COMMAND_NONE);

        pass.reverse_bucket_links();

        assert_eq!(pass.slot_heads[4], 2);
        assert_eq!(pass.commands[2].next, 1);
        assert_eq!(pass.commands[1].next, 0);
        assert_eq!(pass.commands[0].next, WORLD_COMMAND_NONE);
    }

    /// The canonical quad split must always share the `0`–`2`
    /// diagonal. The pre-fix bug used `(0,1,2)` + `(2,1,3)`,
    /// which puts the second triangle on the OTHER diagonal and
    /// leaves a triangular hole near corner `3`. This test fails
    /// loudly if anyone reintroduces that pattern.
    #[test]
    fn textured_quad_triangles_share_zero_two_diagonal() {
        assert_eq!(TEXTURED_QUAD_TRIANGLES[0], [0, 1, 2]);
        assert_eq!(TEXTURED_QUAD_TRIANGLES[1], [0, 2, 3]);
        // Both triangles must contain the diagonal endpoints.
        for tri in TEXTURED_QUAD_TRIANGLES {
            assert!(tri.contains(&0), "{tri:?} missing corner 0");
            assert!(tri.contains(&2), "{tri:?} missing corner 2");
        }
        // All four corners must appear at least once across the
        // two triangles -- otherwise some part of the quad is
        // never drawn.
        for corner in 0..4 {
            assert!(
                TEXTURED_QUAD_TRIANGLES
                    .iter()
                    .any(|tri| tri.contains(&corner)),
                "corner {corner} not covered"
            );
        }
    }

    /// For a convex unit square laid out as
    ///
    /// ```text
    ///   (0,0) ─── (1,0)        0 ─ 1
    ///     │         │          │   │
    ///   (0,1) ─── (1,1)        3 ─ 2
    /// ```
    ///
    /// both generated triangles must have the same signed-area
    /// orientation. If they don't, one of them is flipped and a
    /// `CullMode::Back` pass would reject one half -- which is
    /// exactly how the old buggy split looked: half the quad
    /// rendered, half disappeared.
    #[test]
    fn textured_quad_split_produces_consistent_winding() {
        // Screen-space corners as if the renderer just projected
        // a unit-aligned floor quad. Y grows downward in PSX
        // screen space, but the sign of the cross product is
        // what we're checking -- the absolute orientation
        // doesn't matter.
        let v: [(i32, i32); 4] = [(0, 0), (10, 0), (10, 10), (0, 10)];
        let signed_area = |a: (i32, i32), b: (i32, i32), c: (i32, i32)| -> i32 {
            let abx = b.0 - a.0;
            let aby = b.1 - a.1;
            let acx = c.0 - a.0;
            let acy = c.1 - a.1;
            abx * acy - aby * acx
        };
        let [a0, b0, c0] = TEXTURED_QUAD_TRIANGLES[0];
        let [a1, b1, c1] = TEXTURED_QUAD_TRIANGLES[1];
        let area0 = signed_area(v[a0], v[b0], v[c0]);
        let area1 = signed_area(v[a1], v[b1], v[c1]);
        assert!(area0 != 0, "first triangle is degenerate");
        assert!(area1 != 0, "second triangle is degenerate");
        assert_eq!(
            area0.signum(),
            area1.signum(),
            "split halves must wind the same way (got {area0} vs {area1})"
        );
    }

    /// The two halves must tile the quad without overlap. Picking
    /// a point that lies strictly above the `0`–`2` diagonal should
    /// land in exactly one triangle; the same goes for a point
    /// below. Reproduces the old bug: under `(0,1,2)+(2,1,3)`, a
    /// point in the lower-left quadrant (near corner `3`) lies in
    /// neither half.
    #[test]
    fn textured_quad_split_tiles_the_quad_without_holes() {
        let v: [(i32, i32); 4] = [(0, 0), (10, 0), (10, 10), (0, 10)];
        // Inside-triangle test using barycentric sign check.
        let in_triangle = |t: [usize; 3], p: (i32, i32)| -> bool {
            let a = v[t[0]];
            let b = v[t[1]];
            let c = v[t[2]];
            let s1 = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            let s2 = (c.0 - b.0) * (p.1 - b.1) - (c.1 - b.1) * (p.0 - b.0);
            let s3 = (a.0 - c.0) * (p.1 - c.1) - (a.1 - c.1) * (p.0 - c.0);
            (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)
        };
        // Probes carefully chosen so they're strictly *inside* one
        // half or the other, never on the diagonal. The point
        // (2, 7) is the killer probe: under the OLD `(2,1,3)`
        // second triangle it falls into the hole (y > x AND
        // x+y < 10), so the assertion below would have caught
        // the bug at unit-test time.
        for &p in &[(2, 1), (8, 2), (8, 8), (2, 7)] {
            let covered = in_triangle(TEXTURED_QUAD_TRIANGLES[0], p)
                || in_triangle(TEXTURED_QUAD_TRIANGLES[1], p);
            assert!(covered, "point {p:?} fell into the split's hole");
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
    fn local_to_world_scale_applies_q12_without_i64() {
        let half = LocalToWorldScale::from_q12(0x0800);
        assert_eq!(half.apply(8192), 4096);
        assert_eq!(half.apply(-8192), -4096);
        assert_eq!(half.apply(4095), 2047);

        let identity = LocalToWorldScale::from_q12(0);
        assert_eq!(identity.q12(), 0x1000);
        assert_eq!(identity.apply(-12345), -12345);
    }

    #[test]
    fn joint_world_transform_stops_before_camera_view() {
        let pose = JointPose {
            matrix: Mat3I16::IDENTITY.m,
            translation: Vec3I32::new(256, 128, -64),
        };
        let origin = WorldVertex::new(1000, 2000, 3000);
        let joint = compute_joint_world_transform(
            pose,
            Mat3I16::IDENTITY,
            LocalToWorldScale::IDENTITY,
            origin,
        );

        assert_eq!(joint.rotation, Mat3I16::IDENTITY);
        assert_eq!(joint.translation, WorldVertex::new(1256, 2128, 2936));
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

        let camera = WorldCamera::orbit_yaw(projection, target, 0, 120, Angle::ZERO);
        let projected = camera.project_world(target).expect("target in front");

        assert_eq!(projected.sx, 160);
        assert_eq!(projected.sy, 120);
        assert!(projected.sz >= projection.near_z);
    }

    #[test]
    fn world_camera_projects_world_quad() {
        let projection = WorldProjection::new(160, 120, 200, 40);
        let camera = WorldCamera::orbit_yaw(projection, WorldVertex::ZERO, 0, 200, Angle::ZERO);

        let projected = camera.project_world_quad([
            WorldVertex::new(-10, 10, 0),
            WorldVertex::new(10, 10, 0),
            WorldVertex::new(-10, -10, 0),
            WorldVertex::new(10, -10, 0),
        ]);

        assert!(projected.is_some());
    }

    #[test]
    fn textured_model_gte_transform_matches_world_camera_projection() {
        let projection = WorldProjection::new(160, 118, 320, 48);
        let target = WorldVertex::new(0, 512, 0);
        let camera = WorldCamera::orbit_yaw(projection, target, 1120, 2048, Angle::from_q12(220));
        let pose = JointPose {
            matrix: [[0x1000, 0, 0], [0, 0x1000, 0], [0, 0, 0x1000]],
            translation: Vec3I32::new(20, -16, 32),
        };
        let origin = WorldVertex::new(0, 512, 0);
        let local = Vec3I16::new(64, 128, -32);

        let cpu_world = WorldVertex::new(
            origin.x + pose.translation.x + local.x as i32,
            origin.y + pose.translation.y + local.y as i32,
            origin.z + pose.translation.z + local.z as i32,
        );
        let cpu_view = camera.view_vertex(cpu_world);
        let cpu_projected = camera.project_world(cpu_world).expect("in front");

        let (rotation, translation) = textured_model_part_gte_transform(
            camera,
            pose,
            Mat3I16::IDENTITY,
            LocalToWorldScale::IDENTITY,
            origin,
        );
        let gte_x = translation.x + dot_q12_row_i16(rotation.m[0], local);
        let gte_y = translation.y + dot_q12_row_i16(rotation.m[1], local);
        let gte_z = translation.z + dot_q12_row_i16(rotation.m[2], local);

        assert_close_i32(gte_x, cpu_view.x, 4);
        assert_close_i32(gte_y, -cpu_view.y, 4);
        assert_close_i32(gte_z, cpu_view.z, 4);

        let gte_sx = projection.screen_x as i32 + (gte_x * projection.focal_length) / gte_z;
        let gte_sy = projection.screen_y as i32 + (gte_y * projection.focal_length) / gte_z;
        assert_close_i32(gte_sx, cpu_projected.sx as i32, 1);
        assert_close_i32(gte_sy, cpu_projected.sy as i32, 1);
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

    #[test]
    fn world_render_layer_follows_texture_material_transparency() {
        let opaque = TextureMaterial::opaque(0, 0, (128, 128, 128));
        let transparent = TextureMaterial::blended(0, 0, (128, 128, 128), BlendMode::Average);

        assert_eq!(
            WorldRenderLayer::for_material(opaque),
            WorldRenderLayer::Opaque
        );
        assert_eq!(
            WorldRenderLayer::for_material(transparent),
            WorldRenderLayer::Transparent
        );

        assert_eq!(
            WorldSurfaceOptions::new(DepthBand::whole(), DepthRange::new(0, 1000))
                .with_material_layer(transparent)
                .render_layer,
            WorldRenderLayer::Transparent
        );
    }

    #[test]
    fn room_point_converts_to_raw_world_vertex_only_at_boundary() {
        let point = RoomPoint::new(12, -34, 56);
        let vertex = point.to_world_vertex();

        assert_eq!(vertex, WorldVertex::new(12, -34, 56));
        assert_eq!(RoomPoint::from_world_vertex(vertex), point);
        assert_eq!(
            core::mem::size_of::<RoomPoint>(),
            core::mem::size_of::<WorldVertex>()
        );
    }

    #[test]
    fn textured_submit_uses_transparent_layer_for_translucent_material() {
        const ZERO: TriTextured = TriTextured::new(
            [(0, 0), (0, 0), (0, 0)],
            [(0, 0), (0, 0), (0, 0)],
            0,
            0,
            (0, 0, 0),
        );
        let mut ot_storage = OrderingTable::<8>::new();
        let mut ot = OtFrame::begin(&mut ot_storage);
        let mut triangle_storage = [const { ZERO }; 1];
        let mut triangles = PrimitiveArena::new(&mut triangle_storage);
        let mut commands = [WorldTriCommand::EMPTY; 1];
        let material = TextureMaterial::blended(0, 0, (128, 128, 128), BlendMode::Average);

        let stats = {
            let mut pass = WorldRenderPass::new(&mut ot, &mut commands);
            pass.submit_textured_triangle(
                &mut triangles,
                [
                    ProjectedVertex::new(0, 0, 100),
                    ProjectedVertex::new(16, 0, 100),
                    ProjectedVertex::new(0, 16, 100),
                ],
                [(0, 0), (15, 0), (0, 15)],
                material,
                WorldSurfaceOptions::new(DepthBand::whole(), DepthRange::new(0, 1000))
                    .with_cull_mode(CullMode::None),
            )
        };

        assert_eq!(stats.submitted_triangles, 1);
        assert_eq!(commands[0].render_layer, WorldRenderLayer::Transparent);
    }

    #[test]
    fn world_commands_put_transparent_ties_before_opaque_insertions() {
        let mut commands = [
            world_command_layer(5, 300, WorldRenderLayer::Opaque, 0),
            world_command_layer(5, 300, WorldRenderLayer::Transparent, 1),
            world_command_layer(5, 300, WorldRenderLayer::Opaque, 2),
        ];

        sort_world_for_ot_insert(&mut commands);

        assert_eq!(
            commands[0],
            world_command_layer(5, 300, WorldRenderLayer::Transparent, 1)
        );
        assert_eq!(
            commands[1],
            world_command_layer(5, 300, WorldRenderLayer::Opaque, 2)
        );
        assert_eq!(
            commands[2],
            world_command_layer(5, 300, WorldRenderLayer::Opaque, 0)
        );
    }

    fn dot_q12_row_i16(row: [i16; 3], v: Vec3I16) -> i32 {
        ((row[0] as i32) * (v.x as i32)
            + (row[1] as i32) * (v.y as i32)
            + (row[2] as i32) * (v.z as i32))
            >> 12
    }

    fn assert_close_i32(actual: i32, expected: i32, tolerance: i32) {
        let delta = actual.saturating_sub(expected).abs();
        assert!(
            delta <= tolerance,
            "actual {actual}, expected {expected}, delta {delta}, tolerance {tolerance}"
        );
    }
}
