use egui::{Color32, ColorImage};
use psx_asset::{Animation, Model, ModelVertex};
use psx_engine::{
    compute_joint_view_transform, Angle, JointViewTransform, LocalToWorldScale, Mat3I16,
    ProjectedVertex, ViewVertex, WorldCamera, WorldProjection, WorldVertex,
};
use psx_gte::math::Vec3I16;

pub const PREVIEW_WIDTH: usize = 320;
pub const PREVIEW_HEIGHT: usize = 240;

const PREVIEW_NEAR_Z: i32 = 48;
const PREVIEW_FOCAL_LENGTH: i32 = 320;
const PREVIEW_PLAYBACK_HZ: u16 = 60;

#[derive(Copy, Clone, Debug)]
pub struct ImportPreviewOptions {
    pub world_height: i32,
    pub time_seconds: f64,
    pub yaw_q12: u16,
    pub pitch_q12: u16,
    pub radius: i32,
    pub show_animation_root: bool,
}

pub fn render_import_model_preview_with_options(
    model_bytes: &[u8],
    clip_bytes: &[u8],
    atlas: &ColorImage,
    options: ImportPreviewOptions,
) -> Option<ColorImage> {
    let model = Model::from_bytes(model_bytes).ok()?;
    let animation = Animation::from_bytes(clip_bytes).ok()?;
    if atlas.size[0] == 0 || atlas.size[1] == 0 {
        return None;
    }

    let projection = WorldProjection::new(
        (PREVIEW_WIDTH / 2) as i16,
        (PREVIEW_HEIGHT / 2 + 4) as i16,
        PREVIEW_FOCAL_LENGTH,
        PREVIEW_NEAR_Z,
    );
    let height = options.world_height.max(128);
    let target = WorldVertex::new(0, height / 2, 0);
    let radius = if options.radius > 0 {
        options.radius
    } else {
        height.saturating_mul(3) / 2
    }
    .clamp(640, 8192);
    let camera = WorldCamera::orbit(
        projection,
        target,
        radius,
        Angle::from_q12(options.yaw_q12),
        Angle::from_q12(options.pitch_q12),
    );
    let origin = WorldVertex::new(0, height / 2, 0);
    let frame_q12 = animation.phase_at_tick_q12(
        (options.time_seconds.max(0.0) * PREVIEW_PLAYBACK_HZ as f64) as u32,
        PREVIEW_PLAYBACK_HZ,
    );

    let mut image = ColorImage {
        size: [PREVIEW_WIDTH, PREVIEW_HEIGHT],
        pixels: vec![Color32::from_rgb(8, 10, 14); PREVIEW_WIDTH * PREVIEW_HEIGHT],
    };
    let mut z_buffer = vec![f32::INFINITY; PREVIEW_WIDTH * PREVIEW_HEIGHT];
    let mut joint_transforms =
        vec![JointViewTransform::ZERO; model.joint_count().min(animation.joint_count()) as usize];
    let local_to_world = LocalToWorldScale::from_q12(model.local_to_world_q12());
    for (joint, transform) in joint_transforms.iter_mut().enumerate() {
        let pose = animation.pose_looped_q12(frame_q12, joint as u16)?;
        let (rotation, translation) =
            compute_joint_view_transform(camera, pose, Mat3I16::IDENTITY, local_to_world, origin);
        *transform = JointViewTransform {
            rotation,
            translation,
        };
    }
    let root_projected = joint_transforms.first().and_then(|transform| {
        cpu_project_gte_view(
            ViewVertex::new(
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ),
            projection,
        )
    });

    let mut projected = vec![None; model.vertex_count() as usize];
    for part_index in 0..model.part_count() {
        let Some(part) = model.part(part_index) else {
            continue;
        };
        let primary_joint = part.joint_index() as usize;
        let Some(primary) = joint_transforms.get(primary_joint).copied() else {
            continue;
        };
        let start = part.first_vertex() as usize;
        let end = start
            .saturating_add(part.vertex_count() as usize)
            .min(projected.len());
        for vertex_index in start..end {
            let Some(vertex) = model.vertex(vertex_index as u16) else {
                continue;
            };
            projected[vertex_index] =
                project_import_model_vertex(vertex, primary, &joint_transforms, projection);
        }
    }

    for part_index in 0..model.part_count() {
        let Some(part) = model.part(part_index) else {
            continue;
        };
        let first_face = part.first_face();
        let last_face = first_face.saturating_add(part.face_count());
        for face_index in first_face..last_face {
            let Some(face) = model.face(face_index) else {
                continue;
            };
            let [a, b, c] = face.corners;
            let Some(pa) = projected.get(a.vertex_index as usize).and_then(|v| *v) else {
                continue;
            };
            let Some(pb) = projected.get(b.vertex_index as usize).and_then(|v| *v) else {
                continue;
            };
            let Some(pc) = projected.get(c.vertex_index as usize).and_then(|v| *v) else {
                continue;
            };
            raster_textured_triangle(
                &mut image,
                &mut z_buffer,
                atlas,
                [
                    PreviewVertex::from_projected(pa, a.uv),
                    PreviewVertex::from_projected(pb, b.uv),
                    PreviewVertex::from_projected(pc, c.uv),
                ],
            );
        }
    }

    if options.show_animation_root {
        if let Some(root) = root_projected {
            draw_animation_root_marker(&mut image, root);
        }
    }

    Some(image)
}

#[derive(Copy, Clone)]
struct PreviewVertex {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    v: f32,
}

impl PreviewVertex {
    fn from_projected(projected: ProjectedVertex, uv: (u8, u8)) -> Self {
        Self {
            x: projected.sx as f32,
            y: projected.sy as f32,
            z: projected.sz as f32,
            u: uv.0 as f32,
            v: uv.1 as f32,
        }
    }
}

fn project_import_model_vertex(
    vertex: ModelVertex,
    primary: JointViewTransform,
    joint_transforms: &[JointViewTransform],
    projection: WorldProjection,
) -> Option<ProjectedVertex> {
    let view = if vertex.is_blend() && (vertex.joint1 as usize) < joint_transforms.len() {
        let secondary = joint_transforms[vertex.joint1 as usize];
        lerp_view_vertex(
            cpu_view_transform(&primary, vertex.position),
            cpu_view_transform(&secondary, vertex.position),
            vertex.blend,
        )
    } else {
        cpu_view_transform(&primary, vertex.position)
    };
    cpu_project_gte_view(view, projection)
}

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

fn cpu_project_gte_view(view: ViewVertex, projection: WorldProjection) -> Option<ProjectedVertex> {
    if view.z <= 0 || view.z < projection.near_z {
        return None;
    }
    let sx = (projection.screen_x as i32) + (view.x * projection.focal_length) / view.z;
    let sy = (projection.screen_y as i32) + (view.y * projection.focal_length) / view.z;
    Some(ProjectedVertex::new(clamp_i16(sx), clamp_i16(sy), view.z))
}

fn lerp_view_vertex(a: ViewVertex, b: ViewVertex, t: u8) -> ViewVertex {
    let t = t as i32;
    let inv = 256 - t;
    ViewVertex::new(
        ((a.x.saturating_mul(inv)).saturating_add(b.x.saturating_mul(t))) >> 8,
        ((a.y.saturating_mul(inv)).saturating_add(b.y.saturating_mul(t))) >> 8,
        ((a.z.saturating_mul(inv)).saturating_add(b.z.saturating_mul(t))) >> 8,
    )
}

fn raster_textured_triangle(
    image: &mut ColorImage,
    z_buffer: &mut [f32],
    atlas: &ColorImage,
    tri: [PreviewVertex; 3],
) {
    let area = edge(tri[0], tri[1], tri[2]);
    if area.abs() < f32::EPSILON {
        return;
    }

    let min_x = tri
        .iter()
        .map(|p| p.x.floor() as i32)
        .min()
        .unwrap_or(0)
        .clamp(0, PREVIEW_WIDTH as i32 - 1);
    let max_x = tri
        .iter()
        .map(|p| p.x.ceil() as i32)
        .max()
        .unwrap_or(0)
        .clamp(0, PREVIEW_WIDTH as i32 - 1);
    let min_y = tri
        .iter()
        .map(|p| p.y.floor() as i32)
        .min()
        .unwrap_or(0)
        .clamp(0, PREVIEW_HEIGHT as i32 - 1);
    let max_y = tri
        .iter()
        .map(|p| p.y.ceil() as i32)
        .max()
        .unwrap_or(0)
        .clamp(0, PREVIEW_HEIGHT as i32 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = PreviewVertex {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
                z: 0.0,
                u: 0.0,
                v: 0.0,
            };
            let w0 = edge(tri[1], tri[2], p);
            let w1 = edge(tri[2], tri[0], p);
            let w2 = edge(tri[0], tri[1], p);
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if !inside {
                continue;
            }

            let b0 = w0 / area;
            let b1 = w1 / area;
            let b2 = w2 / area;
            let depth = tri[0].z * b0 + tri[1].z * b1 + tri[2].z * b2;
            let index = y as usize * PREVIEW_WIDTH + x as usize;
            if depth >= z_buffer[index] {
                continue;
            }

            let u = tri[0].u * b0 + tri[1].u * b1 + tri[2].u * b2;
            let v = tri[0].v * b0 + tri[1].v * b1 + tri[2].v * b2;
            image.pixels[index] = sample_atlas(atlas, u, v);
            z_buffer[index] = depth;
        }
    }
}

fn edge(a: PreviewVertex, b: PreviewVertex, c: PreviewVertex) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn sample_atlas(atlas: &ColorImage, u: f32, v: f32) -> Color32 {
    let x = (u.round() as i32).clamp(0, atlas.size[0] as i32 - 1) as usize;
    let y = (v.round() as i32).clamp(0, atlas.size[1] as i32 - 1) as usize;
    atlas.pixels[y * atlas.size[0] + x]
}

fn draw_animation_root_marker(image: &mut ColorImage, projected: ProjectedVertex) {
    let cx = projected.sx as i32;
    let cy = projected.sy as i32;
    for d in -7..=7 {
        put_marker_pixel(image, cx + d, cy, Color32::from_rgb(30, 220, 255));
        put_marker_pixel(image, cx, cy + d, Color32::from_rgb(30, 220, 255));
    }
    for y in -3..=3 {
        for x in -3..=3 {
            if x * x + y * y <= 9 {
                put_marker_pixel(image, cx + x, cy + y, Color32::from_rgb(255, 60, 170));
            }
        }
    }
    for y in -5i32..=5 {
        for x in -5i32..=5 {
            let edge = x.abs().max(y.abs()) == 5;
            if edge {
                put_marker_pixel(image, cx + x, cy + y, Color32::WHITE);
            }
        }
    }
}

fn put_marker_pixel(image: &mut ColorImage, x: i32, y: i32, color: Color32) {
    if x < 0 || y < 0 || x >= PREVIEW_WIDTH as i32 || y >= PREVIEW_HEIGHT as i32 {
        return;
    }
    image.pixels[y as usize * PREVIEW_WIDTH + x as usize] = color;
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use psx_gte::math::Vec3I32;
    use std::path::Path;

    #[test]
    fn invalid_secondary_blend_uses_primary_transform() {
        let primary = JointViewTransform {
            rotation: Mat3I16::IDENTITY,
            translation: Vec3I32::new(0, 0, 100),
        };
        let vertex = ModelVertex {
            position: Vec3I16::new(0, 0, 0),
            joint1: 99,
            blend: 128,
        };
        let projected = project_import_model_vertex(
            vertex,
            primary,
            &[primary],
            WorldProjection::new(160, 120, 320, 1),
        )
        .expect("invalid secondary joint should stay on primary path");

        assert_eq!(projected, ProjectedVertex::new(160, 120, 100));
    }

    #[test]
    fn atlas_sampling_clamps_uvs() {
        let image = ColorImage {
            size: [2, 2],
            pixels: vec![
                Color32::from_rgb(1, 0, 0),
                Color32::from_rgb(2, 0, 0),
                Color32::from_rgb(3, 0, 0),
                Color32::from_rgb(4, 0, 0),
            ],
        };

        assert_eq!(
            sample_atlas(&image, -10.0, -10.0),
            Color32::from_rgb(1, 0, 0)
        );
        assert_eq!(sample_atlas(&image, 99.0, 99.0), Color32::from_rgb(4, 0, 0));
    }

    #[test]
    fn tracked_wraith_model_renders_nonblank_preview() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let model =
            std::fs::read(root.join("assets/models/obsidian_wraith/obsidian_wraith.psxmdl"))
                .expect("tracked model fixture");
        let clip =
            std::fs::read(root.join("assets/models/obsidian_wraith/obsidian_wraith_idle.psxanim"))
                .expect("tracked animation fixture");
        let atlas = ColorImage {
            size: [128, 128],
            pixels: vec![Color32::from_rgb(210, 90, 70); 128 * 128],
        };

        let image = render_import_model_preview_with_options(
            &model,
            &clip,
            &atlas,
            ImportPreviewOptions {
                world_height: 1024,
                time_seconds: 0.0,
                yaw_q12: 340,
                pitch_q12: 350,
                radius: 1536,
                show_animation_root: true,
            },
        )
        .expect("tracked cooked model should render");
        let background = Color32::from_rgb(8, 10, 14);
        let lit_pixels = image
            .pixels
            .iter()
            .filter(|pixel| **pixel != background)
            .count();

        assert!(
            lit_pixels > 32,
            "expected the cooked model preview to draw visible pixels, got {lit_pixels}"
        );
    }
}
