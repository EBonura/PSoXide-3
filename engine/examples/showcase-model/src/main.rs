//! `showcase-model` — first native animated-model demo.
//!
//! This is deliberately tiny: one cooked `.psxmdl`, one sampled
//! `.psxanim`, one 8bpp CLUT texture, and a camera orbiting the model.
//! The point is to validate the new native import path end-to-end
//! before building a richer character viewer/editor workflow on top.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Animation, Model, Texture};
use psx_engine::{
    button, App, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange, OtFrame,
    PrimitiveArena, ProjectedTexturedVertex, ProjectedVertex, Scene, WorldCamera, WorldProjection,
    WorldRenderPass, WorldSurfaceOptions, WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

static MODEL_BLOB: &[u8] =
    include_bytes!("../../../../assets/models/iron_wraith/iron_wraith.psxmdl");
static ANIM_BLOB: &[u8] =
    include_bytes!("../../../../assets/models/iron_wraith/iron_wraith_idle.psxanim");
static TEXTURE_BLOB: &[u8] =
    include_bytes!("../../../../assets/models/iron_wraith/iron_wraith_128x128_8bpp.psxt");

const SCREEN_CX: i32 = 160;
const SCREEN_CY: i32 = 118;
const PROJ_FOCAL: i32 = 320;
const NEAR_Z: i32 = 48;
const PROJECTION: WorldProjection =
    WorldProjection::new(SCREEN_CX as i16, SCREEN_CY as i16, PROJ_FOCAL, NEAR_Z);

const OT_DEPTH: usize = 128;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 2);

const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit8);
const TEX_CLUT: Clut = Clut::new(0, 482);

const MODEL_WORLD_HEIGHT: i32 = 1024;
const MODEL_Y_OFFSET: i32 = MODEL_WORLD_HEIGHT / 2;
const MODEL_TARGET_Y: i32 = MODEL_WORLD_HEIGHT / 2;
const CAMERA_TARGET: WorldVertex = WorldVertex::new(0, MODEL_TARGET_Y, 0);
const CAMERA_Y: i32 = 1120;
const CAMERA_RADIUS_START: i32 = 2048;
const CAMERA_RADIUS_MIN: i32 = 1152;
const CAMERA_RADIUS_MAX: i32 = 4096;
const CAMERA_RADIUS_STEP: i32 = 64;
const CAMERA_AUTO_STEP_PER_VBLANK: u16 = 4;
const CAMERA_PAD_STEP_PER_VBLANK: u16 = 12;
const WORLD_DEPTH_RANGE: DepthRange =
    DepthRange::new(NEAR_Z, CAMERA_RADIUS_MAX + MODEL_WORLD_HEIGHT * 2);

const TRI_CAP: usize = 1536;
const MODEL_VERTEX_CAP: usize = 1024;

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut TEXTURED_TRIS: [TriTextured; TRI_CAP] = [const {
    TriTextured::new(
        [(0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0)],
        0,
        0,
        (0, 0, 0),
    )
}; TRI_CAP];
static mut WORLD_COMMANDS: [WorldTriCommand; TRI_CAP] = [WorldTriCommand::EMPTY; TRI_CAP];
static mut MODEL_VERTICES: [ProjectedTexturedVertex; MODEL_VERTEX_CAP] =
    [ProjectedTexturedVertex::new(ProjectedVertex::new(0, 0, 0), 0, 0); MODEL_VERTEX_CAP];

struct ModelShowcase {
    model: Option<Model<'static>>,
    animation: Option<Animation<'static>>,
    camera_yaw: u16,
    camera_radius: i32,
}

impl Scene for ModelShowcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.model = Some(Model::from_bytes(MODEL_BLOB).expect("iron_wraith.psxmdl"));
        self.animation = Some(Animation::from_bytes(ANIM_BLOB).expect("iron_wraith_idle.psxanim"));
        upload_model_texture();
    }

    fn update(&mut self, ctx: &mut Ctx) {
        let dt = ctx.time.delta_vblanks();
        self.camera_yaw = self
            .camera_yaw
            .wrapping_add(scale_u16(CAMERA_AUTO_STEP_PER_VBLANK, dt));
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self
                .camera_yaw
                .wrapping_sub(scale_u16(CAMERA_PAD_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self
                .camera_yaw
                .wrapping_add(scale_u16(CAMERA_PAD_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::UP) {
            self.camera_radius =
                (self.camera_radius - scale_i32(CAMERA_RADIUS_STEP, dt)).max(CAMERA_RADIUS_MIN);
        }
        if ctx.is_held(button::DOWN) {
            self.camera_radius =
                (self.camera_radius + scale_i32(CAMERA_RADIUS_STEP, dt)).min(CAMERA_RADIUS_MAX);
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        let Some(model) = self.model else {
            return;
        };
        let Some(animation) = self.animation else {
            return;
        };
        let material = TextureMaterial::opaque(
            TEX_CLUT.uv_clut_word(),
            TEX_TPAGE.uv_tpage_word(0),
            (0x80, 0x80, 0x80),
        )
        .with_raw_texture(true);
        let camera = WorldCamera::orbit_yaw(
            PROJECTION,
            CAMERA_TARGET,
            CAMERA_Y,
            self.camera_radius,
            self.camera_yaw,
        );

        let frame_q12 =
            animation.phase_at_tick_q12(ctx.time.elapsed_vblanks(), ctx.time.video_hz());
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };
        draw_animated_model(
            model,
            animation,
            frame_q12,
            camera,
            material,
            &mut world,
            &mut triangles,
        );
        world.flush();
        ot.submit();
    }
}

#[no_mangle]
fn main() -> ! {
    let mut scene = ModelShowcase::new();
    let config = Config {
        clear_color: (5, 6, 10),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

impl ModelShowcase {
    const fn new() -> Self {
        Self {
            model: None,
            animation: None,
            camera_yaw: 0,
            camera_radius: CAMERA_RADIUS_START,
        }
    }
}

fn scale_u16(value: u16, scale: u16) -> u16 {
    value.saturating_mul(scale)
}

fn scale_i32(value: i32, scale: u16) -> i32 {
    value.saturating_mul(scale as i32)
}

fn upload_model_texture() {
    let texture = Texture::from_bytes(TEXTURE_BLOB).expect("iron_wraith texture");
    upload_bytes(
        VramRect::new(
            TEX_TPAGE.x(),
            TEX_TPAGE.y(),
            texture.halfwords_per_row(),
            texture.height(),
        ),
        texture.pixel_bytes(),
    );
    upload_opaque_clut(
        VramRect::new(TEX_CLUT.x(), TEX_CLUT.y(), texture.clut_entries(), 1),
        texture.clut_bytes(),
    );
}

fn upload_opaque_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    assert!(bytes.len() <= marked.len());
    assert!(bytes.len() % 2 == 0);

    let mut i = 0;
    while i < bytes.len() {
        let mut color = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        if color == 0 {
            color = 1;
        }
        marked[i..i + 2].copy_from_slice(&color.to_le_bytes());
        i += 2;
    }
    upload_bytes(rect, &marked[..bytes.len()]);
}

fn draw_animated_model(
    model: Model<'_>,
    animation: Animation<'_>,
    frame_q12: u32,
    camera: WorldCamera,
    material: TextureMaterial,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
        .with_depth_policy(DepthPolicy::Average)
        .with_cull_mode(CullMode::Back)
        .with_material_layer(material);
    let stats = world.submit_textured_model(
        triangles,
        model,
        animation,
        frame_q12,
        camera,
        WorldVertex::new(0, MODEL_Y_OFFSET, 0),
        unsafe { &mut MODEL_VERTICES },
        material,
        options,
    );
    if stats.primitive_overflow || stats.command_overflow || stats.vertex_overflow {
        return;
    }
}
