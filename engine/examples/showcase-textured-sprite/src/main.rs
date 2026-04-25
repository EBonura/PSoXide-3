//! `showcase-textured-sprite` - 3D textured material showcase.
//!
//! A compact interactive material room.
//!
//! The room uses the two cooked sample textures directly: dark brick
//! walls and a cobblestone floor. A single upright pane in the centre
//! shows one material at a time against the actual room.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Texture;
use psx_engine::{button, App, Config, Ctx, DepthBand, DepthRange, OtFrame, PrimitiveArena, Scene};
use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{
    material::{BlendMode, TextureMaterial},
    ot::OrderingTable,
    prim::QuadTextured,
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
const CAMERA_START_YAW: u16 = 220;
const CAMERA_ORBIT_PER_FRAME: u16 = 1;

const FLOOR_X: i32 = 310;
const FLOOR_FRONT_Z: i32 = 245;
const SIDE_FRONT_Z: i32 = 150;
const WALL_Z: i32 = -46;
const WALL_TOP: i32 = 166;
const PANEL_SIZE: i32 = 124;
const PANEL_BOTTOM: i32 = 20;
const PANEL_Z: i32 = 0;

const SAMPLE_BRICK: u8 = 0;
const SAMPLE_FLOOR: u8 = 1;
const SAMPLE_COUNT: u8 = 2;
const BLEND_COUNT: u8 = 5;
const FLOOR_X_EDGES: [i32; 6] = [-FLOOR_X, -186, -62, 62, 186, FLOOR_X];
const FLOOR_Z_EDGES: [i32; 5] = [WALL_Z, 28, 100, 172, FLOOR_FRONT_Z];
const WALL_X_EDGES: [i32; 6] = [-FLOOR_X, -186, -62, 62, 186, FLOOR_X];
const WALL_Y_EDGES: [i32; 3] = [0, 83, WALL_TOP];
const SIDE_Z_EDGES: [i32; 4] = [WALL_Z, 18, 84, SIDE_FRONT_Z];
const SIDE_Y_EDGES: [i32; 3] = [0, 74, WALL_TOP - 18];
const OT_DEPTH: usize = 16;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, ORBIT_RADIUS + FLOOR_FRONT_Z);
const MAX_QUADS: usize = 64;

const QUAD_ZERO: QuadTextured = QuadTextured::new(
    [(0, 0), (0, 0), (0, 0), (0, 0)],
    [(0, 0), (0, 0), (0, 0), (0, 0)],
    0,
    0,
    (0, 0, 0),
);

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut QUADS: [QuadTextured; MAX_QUADS] = [const { QUAD_ZERO }; MAX_QUADS];

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

#[derive(Copy, Clone)]
struct ProjectedQuad {
    verts: [(i16, i16); 4],
    depth: i32,
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
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut quads = unsafe { PrimitiveArena::new(&mut QUADS) };
        draw_room(camera, &mut ot, &mut quads);
        draw_material_pane(self, camera, &mut ot, &mut quads);
        ot.submit();
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
        match self.sample_idx {
            SAMPLE_BRICK => "BRICK",
            SAMPLE_FLOOR => "FLOOR",
            _ => "FLOOR",
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
        let clut = match self.sample_idx {
            SAMPLE_BRICK => BRICK_CLUT_WORD,
            SAMPLE_FLOOR => FLOOR_CLUT_WORD,
            _ => FLOOR_CLUT_WORD,
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
        match self.sample_idx {
            SAMPLE_BRICK => BRICK_U,
            SAMPLE_FLOOR => FLOOR_U,
            _ => FLOOR_U,
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
    let yaw = CAMERA_START_YAW.wrapping_add((frame as u16).wrapping_mul(CAMERA_ORBIT_PER_FRAME));
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

fn draw_room(
    camera: Camera,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    draw_floor(camera, ot, quads);
    draw_wall(camera, ot, quads);
    draw_side_walls(camera, ot, quads);
}

fn draw_floor(
    camera: Camera,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    let material = TextureMaterial::opaque(FLOOR_CLUT_WORD, TPAGE_WORD, (0x62, 0x66, 0x6c));
    let mut zi = 0;
    while zi + 1 < FLOOR_Z_EDGES.len() {
        let mut xi = 0;
        while xi + 1 < FLOOR_X_EDGES.len() {
            draw_floor_tile(
                camera,
                FLOOR_X_EDGES[xi],
                FLOOR_X_EDGES[xi + 1],
                FLOOR_Z_EDGES[zi],
                FLOOR_Z_EDGES[zi + 1],
                material,
                ot,
                quads,
            );
            xi += 1;
        }
        zi += 1;
    }
}

fn draw_floor_tile(
    camera: Camera,
    x0: i32,
    x1: i32,
    z0: i32,
    z1: i32,
    material: TextureMaterial,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    draw_world_textured(
        camera,
        Vec3 { x: x0, y: 0, z: z0 },
        Vec3 { x: x1, y: 0, z: z0 },
        Vec3 { x: x0, y: 0, z: z1 },
        Vec3 { x: x1, y: 0, z: z1 },
        FLOOR_U,
        material,
        ot,
        quads,
    );
}

fn draw_wall(
    camera: Camera,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    let material = TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, (0x4c, 0x44, 0x40));
    let mut yi = 0;
    while yi + 1 < WALL_Y_EDGES.len() {
        let mut xi = 0;
        while xi + 1 < WALL_X_EDGES.len() {
            draw_wall_tile(
                camera,
                WALL_X_EDGES[xi],
                WALL_X_EDGES[xi + 1],
                WALL_Y_EDGES[yi],
                WALL_Y_EDGES[yi + 1],
                WALL_Z,
                material,
                ot,
                quads,
            );
            xi += 1;
        }
        yi += 1;
    }
}

fn draw_side_walls(
    camera: Camera,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    let material = TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, (0x28, 0x2c, 0x34));
    if camera.x >= 0 {
        draw_side_wall(camera, -FLOOR_X, material, ot, quads);
        draw_side_wall(camera, FLOOR_X, material, ot, quads);
    } else {
        draw_side_wall(camera, FLOOR_X, material, ot, quads);
        draw_side_wall(camera, -FLOOR_X, material, ot, quads);
    }
}

fn draw_side_wall(
    camera: Camera,
    x: i32,
    material: TextureMaterial,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    let mut yi = 0;
    while yi + 1 < SIDE_Y_EDGES.len() {
        let mut zi = 0;
        while zi + 1 < SIDE_Z_EDGES.len() {
            draw_side_wall_tile(
                camera,
                x,
                SIDE_Z_EDGES[zi],
                SIDE_Z_EDGES[zi + 1],
                SIDE_Y_EDGES[yi],
                SIDE_Y_EDGES[yi + 1],
                material,
                ot,
                quads,
            );
            zi += 1;
        }
        yi += 1;
    }
}

fn draw_wall_tile(
    camera: Camera,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    z: i32,
    material: TextureMaterial,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    draw_world_textured(
        camera,
        Vec3 { x: x0, y: y1, z },
        Vec3 { x: x1, y: y1, z },
        Vec3 { x: x0, y: y0, z },
        Vec3 { x: x1, y: y0, z },
        BRICK_U,
        material,
        ot,
        quads,
    );
}

fn draw_side_wall_tile(
    camera: Camera,
    x: i32,
    z0: i32,
    z1: i32,
    y0: i32,
    y1: i32,
    material: TextureMaterial,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    draw_world_textured(
        camera,
        Vec3 { x, y: y1, z: z0 },
        Vec3 { x, y: y1, z: z1 },
        Vec3 { x, y: y0, z: z0 },
        Vec3 { x, y: y0, z: z1 },
        BRICK_U,
        material,
        ot,
        quads,
    );
}

fn draw_material_pane(
    scene: &Showcase,
    camera: Camera,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    draw_vertical_square(camera, scene.base_u(), scene.material(), ot, quads);
}

fn draw_vertical_square(
    camera: Camera,
    base_u: u8,
    material: TextureMaterial,
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
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
        ot,
        quads,
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
    ot: &mut OtFrame<'_, OT_DEPTH>,
    quads: &mut PrimitiveArena<'_, QuadTextured>,
) {
    if let Some(projected) = project_quad(camera, [a, b, c, d]) {
        let Some(quad) = quads.push(QuadTextured::with_material(
            projected.verts,
            texture_uvs(base_u),
            material,
        )) else {
            return;
        };
        let slot = WORLD_BAND.slot::<OT_DEPTH>(WORLD_DEPTH_RANGE, projected.depth);
        ot.add_packet_slot(slot, quad);
    }
}

fn project_quad(camera: Camera, verts: [Vec3; 4]) -> Option<ProjectedQuad> {
    let a = project_vertex(camera, verts[0])?;
    let b = project_vertex(camera, verts[1])?;
    let c = project_vertex(camera, verts[2])?;
    let d = project_vertex(camera, verts[3])?;
    Some(ProjectedQuad {
        verts: [(a.0, a.1), (b.0, b.1), (c.0, c.1), (d.0, d.1)],
        depth: (a.2 + b.2 + c.2 + d.2) / 4,
    })
}

fn project_vertex(camera: Camera, v: Vec3) -> Option<(i16, i16, i32)> {
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
    Some((sx as i16, sy as i16, z2))
}

fn texture_uvs(base_u: u8) -> [(u8, u8); 4] {
    [
        (base_u, 0),
        (base_u + (TEX_W as u8 - 1), 0),
        (base_u, TEX_H as u8 - 1),
        (base_u + (TEX_W as u8 - 1), TEX_H as u8 - 1),
    ]
}
