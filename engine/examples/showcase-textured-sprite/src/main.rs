//! `showcase-textured-sprite` - 3D textured material showcase.
//!
//! A compact interactive material room.
//!
//! The room is cheap flat geometry with visible neutral walls and a
//! single upright pane in the centre. Controller input swaps the
//! texture sample and blend mode while the HUD names what is currently
//! being shown.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_asset::Texture;
use psx_engine::{button, App, Config, Ctx, Scene};
use psx_font::{fonts::BASIC, FontAtlas};
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
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

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

const FLOOR_X: i32 = 310;
const FLOOR_FRONT_Z: i32 = 245;
const WALL_Z: i32 = -46;
const WALL_TOP: i32 = 154;
const BACKING_SIZE: i32 = 150;
const PANEL_SIZE: i32 = 112;
const PANEL_BOTTOM: i32 = 20;
const BACKING_Z: i32 = -10;
const PANEL_Z: i32 = 0;

const SAMPLE_BRICK: u8 = 0;
const SAMPLE_FLOOR: u8 = 1;
const SAMPLE_COUNT: u8 = 2;
const BLEND_COUNT: u8 = 5;

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
    font: Option<FontAtlas>,
    sample_idx: u8,
    blend_idx: u8,
}

impl Scene for Showcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        upload_sample_textures();
        self.font = Some(FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::RIGHT) || ctx.just_pressed(button::CROSS) {
            self.blend_idx = (self.blend_idx + 1) % BLEND_COUNT;
        }
        if ctx.just_pressed(button::LEFT) || ctx.just_pressed(button::SQUARE) {
            self.blend_idx = (self.blend_idx + BLEND_COUNT - 1) % BLEND_COUNT;
        }
        if ctx.just_pressed(button::UP)
            || ctx.just_pressed(button::DOWN)
            || ctx.just_pressed(button::TRIANGLE)
            || ctx.just_pressed(button::CIRCLE)
        {
            self.sample_idx = (self.sample_idx + 1) % SAMPLE_COUNT;
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        let camera = camera_for(ctx.frame);
        draw_room(camera);
        draw_material_backing(camera);
        draw_material_pane(self, camera);
        if let Some(font) = self.font.as_ref() {
            self.draw_hud(font);
        }
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
            font: None,
            sample_idx: SAMPLE_FLOOR,
            blend_idx: 1,
        }
    }

    fn sample_name(&self) -> &'static str {
        if self.sample_idx == SAMPLE_BRICK {
            "BRICK"
        } else {
            "FLOOR"
        }
    }

    fn blend_name(&self) -> &'static str {
        match self.blend_idx {
            0 => "OPAQUE",
            1 => "AVERAGE",
            2 => "ADDITIVE",
            3 => "SUBTRACT",
            _ => "ADD QUARTER",
        }
    }

    fn material(&self) -> TextureMaterial {
        let clut = if self.sample_idx == SAMPLE_BRICK {
            BRICK_CLUT_WORD
        } else {
            FLOOR_CLUT_WORD
        };
        match self.blend_idx {
            0 => TextureMaterial::opaque(clut, TPAGE_WORD, IDENTITY_TINT),
            1 => TextureMaterial::blended(clut, TPAGE_WORD, (0x88, 0x98, 0xb0), BlendMode::Average),
            2 => TextureMaterial::blended(clut, TPAGE_WORD, (0x58, 0x70, 0xa0), BlendMode::Add),
            3 => {
                TextureMaterial::blended(clut, TPAGE_WORD, (0x78, 0x78, 0x78), BlendMode::Subtract)
            }
            _ => TextureMaterial::blended(
                clut,
                TPAGE_WORD,
                (0xb8, 0xb8, 0x98),
                BlendMode::AddQuarter,
            ),
        }
    }

    fn base_u(&self) -> u8 {
        if self.sample_idx == SAMPLE_BRICK {
            BRICK_U
        } else {
            FLOOR_U
        }
    }

    fn draw_hud(&self, font: &FontAtlas) {
        font.draw_text(8, 8, "MATERIAL VIEWER", (220, 220, 245));
        font.draw_text(8, 24, "TEXTURE", (130, 150, 190));
        font.draw_text(72, 24, self.sample_name(), (235, 235, 210));
        font.draw_text(8, 38, "BLEND", (130, 150, 190));
        font.draw_text(56, 38, self.blend_name(), (235, 235, 210));
        font.draw_text(8, 224, "UP/DN TEXTURE  L/R BLEND", (140, 155, 190));
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

fn draw_room(camera: Camera) {
    draw_wall(camera);
    draw_side_walls(camera);
    draw_floor(camera);
    draw_floor_panels(camera);
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
        (46, 48, 54),
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
        (50, 48, 56),
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
        (62, 58, 64),
    );
}

fn draw_side_walls(camera: Camera) {
    draw_world_flat(
        camera,
        Vec3 {
            x: -FLOOR_X,
            y: WALL_TOP,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: WALL_TOP - 20,
            z: FLOOR_FRONT_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: -FLOOR_X,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        (40, 42, 50),
    );
    draw_world_flat(
        camera,
        Vec3 {
            x: FLOOR_X,
            y: WALL_TOP,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: WALL_TOP - 20,
            z: FLOOR_FRONT_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: WALL_Z,
        },
        Vec3 {
            x: FLOOR_X,
            y: 0,
            z: FLOOR_FRONT_Z,
        },
        (36, 38, 46),
    );
}

fn draw_floor_panels(camera: Camera) {
    draw_floor_panel(camera, -FLOOR_X, -96, WALL_Z, 78, (54, 56, 62));
    draw_floor_panel(camera, -96, 96, WALL_Z, 78, (66, 64, 68));
    draw_floor_panel(camera, 96, FLOOR_X, WALL_Z, 78, (50, 52, 58));
    draw_floor_panel(camera, -FLOOR_X, -96, 78, FLOOR_FRONT_Z, (42, 46, 52));
    draw_floor_panel(camera, -96, 96, 78, FLOOR_FRONT_Z, (58, 58, 64));
    draw_floor_panel(camera, 96, FLOOR_X, 78, FLOOR_FRONT_Z, (44, 48, 54));
}

fn draw_floor_panel(camera: Camera, x0: i32, x1: i32, z0: i32, z1: i32, color: (u8, u8, u8)) {
    draw_world_flat(
        camera,
        Vec3 { x: x0, y: 1, z: z0 },
        Vec3 { x: x1, y: 1, z: z0 },
        Vec3 { x: x0, y: 1, z: z1 },
        Vec3 { x: x1, y: 1, z: z1 },
        color,
    );
}

fn draw_material_backing(camera: Camera) {
    draw_backing(camera);
    draw_backing_crossbar(camera);
}

fn draw_backing(camera: Camera) {
    let half = BACKING_SIZE / 2;
    let mid_x = 0;
    let mid_y = PANEL_BOTTOM + PANEL_SIZE / 2;
    let x0 = -half;
    let x1 = half;
    let y0 = PANEL_BOTTOM - ((BACKING_SIZE - PANEL_SIZE) / 2);
    let y1 = y0 + BACKING_SIZE;
    draw_backing_rect(camera, x0, mid_x, mid_y, y1, (128, 122, 112));
    draw_backing_rect(camera, mid_x, x1, mid_y, y1, (70, 72, 78));
    draw_backing_rect(camera, x0, mid_x, y0, mid_y, (62, 64, 70));
    draw_backing_rect(camera, mid_x, x1, y0, mid_y, (154, 148, 136));
}

fn draw_backing_rect(camera: Camera, x0: i32, x1: i32, y0: i32, y1: i32, color: (u8, u8, u8)) {
    draw_world_flat(
        camera,
        Vec3 {
            x: x0,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: x1,
            y: y1,
            z: BACKING_Z,
        },
        Vec3 {
            x: x0,
            y: y0,
            z: BACKING_Z,
        },
        Vec3 {
            x: x1,
            y: y0,
            z: BACKING_Z,
        },
        color,
    );
}

fn draw_backing_crossbar(camera: Camera) {
    let half = BACKING_SIZE / 2;
    let y0 = PANEL_BOTTOM + PANEL_SIZE / 2 - 4;
    let y1 = y0 + 8;
    draw_world_flat(
        camera,
        Vec3 {
            x: -half,
            y: y1,
            z: BACKING_Z - 1,
        },
        Vec3 {
            x: half,
            y: y1,
            z: BACKING_Z - 1,
        },
        Vec3 {
            x: -half,
            y: y0,
            z: BACKING_Z - 1,
        },
        Vec3 {
            x: half,
            y: y0,
            z: BACKING_Z - 1,
        },
        (176, 170, 158),
    );
}

fn draw_material_pane(scene: &Showcase, camera: Camera) {
    draw_vertical_square(camera, scene.base_u(), scene.material());
}

fn draw_vertical_square(camera: Camera, base_u: u8, material: TextureMaterial) {
    let half = PANEL_SIZE / 2;
    let x0 = -half;
    let x1 = half;
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
