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
use psx_engine::{
    button, App, Config, Ctx, CullMode, DepthBand, DepthRange, OtFrame, PrimitiveArena,
    Scene, WorldCamera, WorldProjection, WorldRenderPass, WorldSurfaceOptions, WorldTriCommand,
    WorldVertex,
};
use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{
    material::{BlendMode, TextureMaterial},
    ot::OrderingTable,
    prim::TriTextured,
};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

static BRICK_BLOB: &[u8] = include_bytes!("../../../../assets/textures/brick-wall.psxt");
static FLOOR_BLOB: &[u8] = include_bytes!("../../../../assets/textures/floor.psxt");

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
const FOCAL: i32 = 320;
const NEAR_Z: i32 = 48;
const PROJECTION: WorldProjection =
    WorldProjection::new(SCREEN_CX as i16, SCREEN_CY as i16, FOCAL, NEAR_Z);

const CAMERA_Y: i32 = 170;
const CAMERA_START_RADIUS: i32 = 260;
const CAMERA_RADIUS_MIN: i32 = 150;
const CAMERA_RADIUS_MAX: i32 = 900;
const CAMERA_RADIUS_STEP: i32 = 12;
const CAMERA_START_YAW: u16 = 220;
const CAMERA_YAW_STEP: u16 = 12;

const ROOM_HALF: i32 = 300;
const WALL_TOP: i32 = 176;
const PANEL_SIZE: i32 = 120;
const PANEL_BOTTOM: i32 = 28;
const PANEL_Z: i32 = 0;
const CAMERA_TARGET_X: i32 = 0;
const CAMERA_TARGET_Y: i32 = PANEL_BOTTOM + PANEL_SIZE / 2;
const CAMERA_TARGET_Z: i32 = PANEL_Z;

const SAMPLE_BRICK: u8 = 0;
const SAMPLE_FLOOR: u8 = 1;
const SAMPLE_COUNT: u8 = 2;
const BLEND_COUNT: u8 = 5;
const ROOM_EDGES: [i32; 4] = [-ROOM_HALF, -100, 100, ROOM_HALF];
const WALL_Y_EDGES: [i32; 3] = [0, WALL_TOP / 2, WALL_TOP];
const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, CAMERA_RADIUS_MAX + ROOM_HALF + 80);
const MAX_TEXTURED_TRIS: usize = 192;

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

struct Showcase {
    font: Option<FontAtlas>,
    sample_idx: u8,
    blend_idx: u8,
    camera_yaw: u16,
    camera_radius: i32,
}

impl Scene for Showcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        upload_sample_textures();
        self.font = Some(FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self.camera_yaw.wrapping_add(CAMERA_YAW_STEP);
        }
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self.camera_yaw.wrapping_sub(CAMERA_YAW_STEP);
        }
        if ctx.is_held(button::UP) {
            self.camera_radius = (self.camera_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
        }
        if ctx.is_held(button::DOWN) {
            self.camera_radius = (self.camera_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
        }

        if ctx.just_pressed(button::CROSS) {
            self.blend_idx = (self.blend_idx + 1) % BLEND_COUNT;
        }
        if ctx.just_pressed(button::SQUARE) {
            self.blend_idx = (self.blend_idx + BLEND_COUNT - 1) % BLEND_COUNT;
        }
        if ctx.just_pressed(button::TRIANGLE) || ctx.just_pressed(button::CIRCLE) {
            self.sample_idx = (self.sample_idx + 1) % SAMPLE_COUNT;
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        let camera = camera_for(self.camera_yaw, self.camera_radius);
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };
        draw_room(camera, &mut world, &mut triangles);
        draw_material_pane(self, camera, &mut world, &mut triangles);
        world.flush();
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
            camera_yaw: CAMERA_START_YAW,
            camera_radius: CAMERA_START_RADIUS,
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
        font.draw_text(8, 224, "D-PAD CAMERA  FACE MATERIAL", (140, 155, 190));
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

fn camera_for(yaw: u16, radius: i32) -> WorldCamera {
    WorldCamera::orbit_yaw(
        PROJECTION,
        WorldVertex::new(CAMERA_TARGET_X, CAMERA_TARGET_Y, CAMERA_TARGET_Z),
        CAMERA_Y,
        radius,
        yaw,
    )
}

fn draw_room(
    camera: WorldCamera,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    draw_floor(camera, world, triangles);
    draw_walls(camera, world, triangles);
}

fn draw_floor(
    camera: WorldCamera,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let material = TextureMaterial::opaque(FLOOR_CLUT_WORD, TPAGE_WORD, (0x62, 0x66, 0x6c));
    let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
        .with_cull_mode(CullMode::None)
        .with_material_layer(material);
    let mut zi = 0;
    while zi + 1 < ROOM_EDGES.len() {
        let mut xi = 0;
        while xi + 1 < ROOM_EDGES.len() {
            draw_world_textured(
                camera,
                [
                    WorldVertex {
                        x: ROOM_EDGES[xi],
                        y: 0,
                        z: ROOM_EDGES[zi],
                    },
                    WorldVertex {
                        x: ROOM_EDGES[xi + 1],
                        y: 0,
                        z: ROOM_EDGES[zi],
                    },
                    WorldVertex {
                        x: ROOM_EDGES[xi],
                        y: 0,
                        z: ROOM_EDGES[zi + 1],
                    },
                    WorldVertex {
                        x: ROOM_EDGES[xi + 1],
                        y: 0,
                        z: ROOM_EDGES[zi + 1],
                    },
                ],
                FLOOR_U,
                material,
                options,
                world,
                triangles,
            );
            xi += 1;
        }
        zi += 1;
    }
}

fn draw_walls(
    camera: WorldCamera,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let material = TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, (0x4c, 0x44, 0x40));
    let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
        .with_cull_mode(CullMode::Back)
        .with_material_layer(material);
    draw_z_wall(
        camera, -ROOM_HALF, false, material, options, world, triangles,
    );
    draw_z_wall(camera, ROOM_HALF, true, material, options, world, triangles);
    draw_x_wall(
        camera, -ROOM_HALF, true, material, options, world, triangles,
    );
    draw_x_wall(
        camera, ROOM_HALF, false, material, options, world, triangles,
    );
}

fn draw_z_wall(
    camera: WorldCamera,
    z: i32,
    reverse_x: bool,
    material: TextureMaterial,
    options: WorldSurfaceOptions,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let mut yi = 0;
    while yi + 1 < WALL_Y_EDGES.len() {
        let mut xi = 0;
        while xi + 1 < ROOM_EDGES.len() {
            let x0 = ROOM_EDGES[xi];
            let x1 = ROOM_EDGES[xi + 1];
            let y0 = WALL_Y_EDGES[yi];
            let y1 = WALL_Y_EDGES[yi + 1];
            let verts = if reverse_x {
                [
                    WorldVertex { x: x1, y: y1, z },
                    WorldVertex { x: x0, y: y1, z },
                    WorldVertex { x: x1, y: y0, z },
                    WorldVertex { x: x0, y: y0, z },
                ]
            } else {
                [
                    WorldVertex { x: x0, y: y1, z },
                    WorldVertex { x: x1, y: y1, z },
                    WorldVertex { x: x0, y: y0, z },
                    WorldVertex { x: x1, y: y0, z },
                ]
            };
            draw_wall_textured(camera, verts, BRICK_U, material, options, world, triangles);
            xi += 1;
        }
        yi += 1;
    }
}

fn draw_x_wall(
    camera: WorldCamera,
    x: i32,
    reverse_z: bool,
    material: TextureMaterial,
    options: WorldSurfaceOptions,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let mut yi = 0;
    while yi + 1 < WALL_Y_EDGES.len() {
        let mut zi = 0;
        while zi + 1 < ROOM_EDGES.len() {
            let z0 = ROOM_EDGES[zi];
            let z1 = ROOM_EDGES[zi + 1];
            let y0 = WALL_Y_EDGES[yi];
            let y1 = WALL_Y_EDGES[yi + 1];
            let verts = if reverse_z {
                [
                    WorldVertex { x, y: y1, z: z1 },
                    WorldVertex { x, y: y1, z: z0 },
                    WorldVertex { x, y: y0, z: z1 },
                    WorldVertex { x, y: y0, z: z0 },
                ]
            } else {
                [
                    WorldVertex { x, y: y1, z: z0 },
                    WorldVertex { x, y: y1, z: z1 },
                    WorldVertex { x, y: y0, z: z0 },
                    WorldVertex { x, y: y0, z: z1 },
                ]
            };
            draw_wall_textured(camera, verts, BRICK_U, material, options, world, triangles);
            zi += 1;
        }
        yi += 1;
    }
}

fn draw_material_pane(
    scene: &Showcase,
    camera: WorldCamera,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    draw_vertical_square(camera, scene.base_u(), scene.material(), world, triangles);
}

fn draw_vertical_square(
    camera: WorldCamera,
    base_u: u8,
    material: TextureMaterial,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let half = PANEL_SIZE / 2;
    let x0 = -half;
    let x1 = half;
    let y0 = PANEL_BOTTOM;
    let y1 = PANEL_BOTTOM + PANEL_SIZE;
    let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
        .with_cull_mode(CullMode::None)
        .with_depth_bias(-8)
        .with_material_layer(material);
    draw_world_textured(
        camera,
        [
            WorldVertex {
                x: x0,
                y: y1,
                z: PANEL_Z,
            },
            WorldVertex {
                x: x1,
                y: y1,
                z: PANEL_Z,
            },
            WorldVertex {
                x: x0,
                y: y0,
                z: PANEL_Z,
            },
            WorldVertex {
                x: x1,
                y: y0,
                z: PANEL_Z,
            },
        ],
        base_u,
        material,
        options,
        world,
        triangles,
    );
}

fn draw_world_textured(
    camera: WorldCamera,
    verts: [WorldVertex; 4],
    base_u: u8,
    material: TextureMaterial,
    options: WorldSurfaceOptions,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let uvs = texture_uvs(base_u);
    let _ = world.submit_textured_world_quad(
        triangles,
        camera,
        verts,
        uvs,
        material,
        options,
    );
}

fn draw_wall_textured(
    camera: WorldCamera,
    verts: [WorldVertex; 4],
    base_u: u8,
    material: TextureMaterial,
    options: WorldSurfaceOptions,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    if let Some(projected) = camera.project_world_quad(verts) {
        let _ = world.submit_textured_quad(
            triangles,
            projected,
            texture_uvs(base_u),
            material,
            options,
        );
    }
}

fn texture_uvs(base_u: u8) -> [(u8, u8); 4] {
    [
        (base_u, 0),
        (base_u + (TEX_W as u8 - 1), 0),
        (base_u, TEX_H as u8 - 1),
        (base_u + (TEX_W as u8 - 1), TEX_H as u8 - 1),
    ]
}
