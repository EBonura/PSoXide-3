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
use psx_gpu::prim::TriGouraud;
use psx_gte::{
    lighting::{project_lit, ProjectedLit},
    math::Vec3I16,
};

use crate::render::{DepthBand, DepthRange, DepthSlot, OtFrame, PrimitiveArena};

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
    fn depth(self, verts: [ProjectedLit; 3]) -> i32 {
        let z0 = verts[0].sz as i32;
        let z1 = verts[1].sz as i32;
        let z2 = verts[2].sz as i32;
        match self {
            DepthPolicy::Average => (z0 + z1 + z2) / 3,
            DepthPolicy::Nearest => z0.min(z1).min(z2),
            DepthPolicy::Farthest => z0.max(z1).max(z2),
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
    let ax = (verts[1].sx as i32) - (verts[0].sx as i32);
    let ay = (verts[1].sy as i32) - (verts[0].sy as i32);
    let bx = (verts[2].sx as i32) - (verts[0].sx as i32);
    let by = (verts[2].sy as i32) - (verts[0].sy as i32);
    (ax * by - ay * bx) <= 0
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

#[cfg(test)]
mod tests {
    use super::*;

    const fn command(slot: usize, depth: i32, primitive_index: usize) -> GouraudTriCommand {
        GouraudTriCommand {
            slot: DepthSlot::new(slot),
            depth,
            primitive_index,
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
}
