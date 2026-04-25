//! `showcase-textured-sprite` - 3D textured material showcase.
//!
//! A compact 3D material booth.
//!
//! The floor and back wall are cheap flat geometry; only the small
//! upright samples are textured. Transparent samples sit over strong
//! backing colours so the PS1 blend modes are easy to read without
//! turning the demo into a fill-rate test.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_asset::Texture;
use psx_engine::{App, Config, Ctx, Scene};
use psx_gpu::{
    self as gpu,
    material::{BlendMode, TextureMaterial},
};
use psx_math::{cos_q12, sin_q12};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

static BRICK_BLOB: &[u8] = include_bytes!("../../showcase-fog/assets/brick-wall.psxt");
static FLOOR_BLOB: &[u8] = include_bytes!("../../showcase-fog/assets/floor.psxt");

const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const BRICK_CLUT: Clut = Clut::new(0, 480);
const FLOOR_CLUT: Clut = Clut::new(0, 481);

const TEX_W: u16 = 64;
const TEX_H: u16 = 64;
const BRICK_U: u8 = 0;
const FLOOR_U: u8 = 64;

const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
const BRICK_CLUT_WORD: u16 = BRICK_CLUT.uv_clut_word();
const FLOOR_CLUT_WORD: u16 = FLOOR_CLUT.uv_clut_word();
const IDENTITY_TINT: (u8, u8, u8) = (0x80, 0x80, 0x80);

const SCREEN_CX: i32 = 160;
const SCREEN_CY: i32 = 118;
const FOCAL: i32 = 220;
const NEAR_Z: i32 = 48;

const CAMERA_Y: i32 = 170;
const ORBIT_RADIUS: i32 = 540;
const CAMERA_PITCH_Q12: u16 = 4096 - 128;

const FLOOR_X: i32 = 230;
const FLOOR_FRONT_Z: i32 = 145;
const WALL_Z: i32 = -18;
const WALL_TOP: i32 = 130;
const BACKING_SIZE: i32 = 74;
const PANEL_SIZE: i32 = 54;
const PANEL_BOTTOM: i32 = 34;
const BACKING_Z: i32 = -6;
const PANEL_Z: i32 = 0;

#[derive(Copy, Clone)]
struct Vec3 {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Copy, Clone)]
struct Camera {
    x: i32,
    y: i32,
    z: i32,
    sin_yaw: i32,
    cos_yaw: i32,
    sin_pitch: i32,
    cos_pitch: i32,
}

struct Showcase {
    brick_opaque: TextureMaterial,
    glass_average: TextureMaterial,
    glow_add: TextureMaterial,
    shadow_subtract: TextureMaterial,
    highlight_quarter: TextureMaterial,
}

impl Scene for Showcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        upload_sample_textures();
    }

    fn update(&mut self, _ctx: &mut Ctx) {}

    fn render(&mut self, ctx: &mut Ctx) {
        let camera = camera_for(ctx.frame);
        draw_floor(camera);
        draw_panel_backs(camera);
        draw_material_panels(self, camera);
    }
}

#[no_mangle]
fn main() -> ! {
    let mut scene = Showcase::new();
    let config = Config {
        clear_color: (5, 7, 12),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

impl Showcase {
    const fn new() -> Self {
        Self {
            brick_opaque: TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, IDENTITY_TINT),
            glass_average: TextureMaterial::blended(
                FLOOR_CLUT_WORD,
                TPAGE_WORD,
                (0x68, 0x78, 0x98),
                BlendMode::Average,
            ),
            glow_add: TextureMaterial::blended(
                FLOOR_CLUT_WORD,
                TPAGE_WORD,
                (0x50, 0x64, 0x98),
                BlendMode::Add,
            ),
            shadow_subtract: TextureMaterial::blended(
                BRICK_CLUT_WORD,
                TPAGE_WORD,
                (0x78, 0x78, 0x78),
                BlendMode::Subtract,
            ),
            highlight_quarter: TextureMaterial::blended(
                BRICK_CLUT_WORD,
                TPAGE_WORD,
                (0xc0, 0xc0, 0x98),
                BlendMode::AddQuarter,
            ),
        }
    }
}

fn upload_sample_textures() {
    let brick = Texture::from_bytes(BRICK_BLOB).expect("brick-wall.psxt");
    let floor = Texture::from_bytes(FLOOR_BLOB).expect("floor.psxt");

    let brick_pix_rect = VramRect::new(
        SHARED_TPAGE.x(),
        SHARED_TPAGE.y(),
        brick.halfwords_per_row(),
        brick.height(),
    );
    upload_bytes(brick_pix_rect, brick.pixel_bytes());
    let brick_clut_rect = VramRect::new(BRICK_CLUT.x(), BRICK_CLUT.y(), brick.clut_entries(), 1);
    upload_blend_clut(brick_clut_rect, brick.clut_bytes());

    let floor_pix_rect = VramRect::new(
        SHARED_TPAGE.x() + brick.halfwords_per_row(),
        SHARED_TPAGE.y(),
        floor.halfwords_per_row(),
        floor.height(),
    );
    upload_bytes(floor_pix_rect, floor.pixel_bytes());
    let floor_clut_rect = VramRect::new(FLOOR_CLUT.x(), FLOOR_CLUT.y(), floor.clut_entries(), 1);
    upload_blend_clut(floor_clut_rect, floor.clut_bytes());
}

fn upload_blend_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    assert!(bytes.len() <= marked.len());
    assert!(bytes.len() % 2 == 0);

    let mut i = 0;
    while i < bytes.len() {
        let raw = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let marked_color = if raw == 0 { 0 } else { raw | 0x8000 };
        let pair = marked_color.to_le_bytes();
        marked[i] = pair[0];
        marked[i + 1] = pair[1];
        i += 2;
    }

    upload_bytes(rect, &marked[..bytes.len()]);
}

fn camera_for(frame: u32) -> Camera {
    let yaw = 220u16.wrapping_add((frame as u16) / 8);
    let sx = sin_q12(yaw);
    let cz = cos_q12(yaw);
    Camera {
        x: (sx * ORBIT_RADIUS) >> 12,
        y: CAMERA_Y,
        z: (cz * ORBIT_RADIUS) >> 12,
        sin_yaw: sx,
        cos_yaw: cz,
        sin_pitch: sin_q12(CAMERA_PITCH_Q12),
        cos_pitch: cos_q12(CAMERA_PITCH_Q12),
    }
}

fn draw_floor(camera: Camera) {
    draw_world_flat(
        camera,
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        (38, 40, 46),
    );
    draw_world_flat(
        camera,
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X / 2,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        Vec3 {
            x: FLOOR_X / 2,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        (54, 56, 64),
    );
}

fn draw_wall(camera: Camera) {
    draw_world_flat(
        camera,
        Vec3 {
            x: -FLOOR_X,
            y: WALL_TOP,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: WALL_TOP,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: PANEL_BOTTOM - 8,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: PANEL_BOTTOM - 8,
            z: WALL_Z,
        },
        (18, 20, 28),
    );
    draw_world_flat(
        camera,
        Vec3 {
            x: -FLOOR_X,
            y: PANEL_BOTTOM - 8,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: PANEL_BOTTOM - 8,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        (24, 26, 36),
    );
}

fn draw_panel_backs(camera: Camera) {
    draw_wall(camera);
    draw_backing(camera, -128, (248, 248, 232), (20, 24, 36));
    draw_backing(camera, -64, (230, 28, 34), (24, 210, 222));
    draw_backing(camera, 0, (24, 76, 230), (24, 220, 76));
    draw_backing(camera, 64, (248, 220, 40), (220, 28, 214));
    draw_backing(camera, 128, (248, 248, 248), (44, 72, 232));
}

fn draw_backing(camera: Camera, center_x: i32, left: (u8, u8, u8), right: (u8, u8, u8)) {
    let half = BACKING_SIZE / 2;
    let mid = center_x;
    let x0 = center_x - half;
    let x1 = center_x + half;
    let y0 = PANEL_BOTTOM - ((BACKING_SIZE - PANEL_SIZE) / 2);
    let y1 = y0 + BACKING_SIZE;
    draw_world_flat(
        camera,
        Vec3 {
            x: x0,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: mid,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: x0,
            y: y0,
            z: BACKING_Z,
        },
        Vec3 {
            x: mid,
            y: y0,
            z: BACKING_Z,
        },
        left,
    );
    draw_world_flat(
        camera,
        Vec3 {
            x: mid,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: x1,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: mid,
            y: y0,
            z: BACKING_Z,
        },
        Vec3 {
            x: x1,
            y: y0,
            z: BACKING_Z,
        },
        right,
    );
}

fn draw_material_panels(scene: &Showcase, camera: Camera) {
    draw_vertical_square(camera, -128, BRICK_U, scene.brick_opaque);
    draw_vertical_square(camera, -64, FLOOR_U, scene.glass_average);
    draw_vertical_square(camera, 0, FLOOR_U, scene.glow_add);
    draw_vertical_square(camera, 64, BRICK_U, scene.shadow_subtract);
    draw_vertical_square(camera, 128, BRICK_U, scene.highlight_quarter);
}

fn draw_vertical_square(camera: Camera, center_x: i32, base_u: u8, material: TextureMaterial) {
    let half = PANEL_SIZE / 2;
    let x0 = center_x - half;
    let x1 = center_x + half;
    let y0 = PANEL_BOTTOM;
    let y1 = PANEL_BOTTOM + PANEL_SIZE;
    draw_world_textured(
        camera,
        Vec3 {
            x: x0,
            y: y1,
            z: PANEL_Z,
        },
        Vec3 {
            x: x1,
            y: y1,
            z: PANEL_Z,
        },
        Vec3 {
            x: x0,
            y: y0,
            z: PANEL_Z,
        },
        Vec3 {
            x: x1,
            y: y0,
            z: PANEL_Z,
        },
        base_u,
        material,
    );
}

fn draw_world_textured(
    camera: Camera,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    base_u: u8,
    material: TextureMaterial,
) {
    if let Some(verts) = project_quad(camera, [a, b, c, d]) {
        gpu::draw_quad_textured_material(verts, texture_uvs(base_u), material);
    }
}

fn draw_world_flat(camera: Camera, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: (u8, u8, u8)) {
    if let Some(verts) = project_quad(camera, [a, b, c, d]) {
        gpu::draw_quad_flat(verts, color.0, color.1, color.2);
    }
}

fn project_quad(camera: Camera, verts: [Vec3; 4]) -> Option<[(i16, i16); 4]> {
    Some([
        project_vertex(camera, verts[0])?,
        project_vertex(camera, verts[1])?,
        project_vertex(camera, verts[2])?,
        project_vertex(camera, verts[3])?,
    ])
}

fn project_vertex(camera: Camera, v: Vec3) -> Option<(i16, i16)> {
    let dx = v.x - camera.x;
    let dy = v.y - camera.y;
    let dz = v.z - camera.z;

    let x1 = ((dx * camera.cos_yaw) - (dz * camera.sin_yaw)) >> 12;
    let z1 = ((-dx * camera.sin_yaw) - (dz * camera.cos_yaw)) >> 12;
    let y2 = ((dy * camera.cos_pitch) - (z1 * camera.sin_pitch)) >> 12;
    let z2 = ((dy * camera.sin_pitch) + (z1 * camera.cos_pitch)) >> 12;

    if z2 <= NEAR_Z {
        return None;
    }

    let sx = SCREEN_CX + (x1 * FOCAL) / z2;
    let sy = SCREEN_CY - (y2 * FOCAL) / z2;
    Some((sx as i16, sy as i16))
}

fn texture_uvs(base_u: u8) -> [(u8, u8); 4] {
    [
        (base_u, 0),
        (base_u + (TEX_W as u8 - 1), 0),
        (base_u, TEX_H as u8 - 1),
        (base_u + (TEX_W as u8 - 1), TEX_H as u8 - 1),
    ]
}
