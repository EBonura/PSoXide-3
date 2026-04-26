//! `showcase-model` — animated-model demo with clip cycling.
//!
//! Loads one cooked `.psxmdl`, one 8bpp CLUT texture, and **all**
//! cooked `.psxanim` clips for the model. The camera orbits the
//! model and Square/Circle steps backward/forward through the
//! clip set. Animation phase is anchored to the engine's PS1-time
//! clock, so each clip restarts cleanly when you switch.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Animation, Model, Texture};
use psx_engine::{
    button, App, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange, JointViewTransform,
    OtFrame, PrimitiveArena, ProjectedTexturedVertex, ProjectedVertex, Scene, WorldCamera,
    WorldProjection, WorldRenderPass, WorldSurfaceOptions, WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

static MODEL_BLOB: &[u8] =
    include_bytes!("../../../../assets/models/obsidian_wraith/obsidian_wraith.psxmdl");
static TEXTURE_BLOB: &[u8] =
    include_bytes!("../../../../assets/models/obsidian_wraith/obsidian_wraith_128x128_8bpp.psxt");

struct ClipEntry {
    label: &'static str,
    blob: &'static [u8],
}

const CLIPS: &[ClipEntry] = &[
    ClipEntry {
        label: "idle",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_idle.psxanim"
        ),
    },
    ClipEntry {
        label: "walking",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_walking.psxanim"
        ),
    },
    ClipEntry {
        label: "running",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_running.psxanim"
        ),
    },
    ClipEntry {
        label: "unsteady_walk",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_unsteady_walk.psxanim"
        ),
    },
    ClipEntry {
        label: "walk_backward",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_walk_backward_inplace.psxanim"
        ),
    },
    ClipEntry {
        label: "double_combo_attack",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_double_combo_attack.psxanim"
        ),
    },
    ClipEntry {
        label: "hit_reaction",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_hit_reaction.psxanim"
        ),
    },
    ClipEntry {
        label: "dead",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_dead.psxanim"
        ),
    },
];

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
const CAMERA_RADIUS_START: i32 = 2048;
const CAMERA_RADIUS_MIN: i32 = 1152;
const CAMERA_RADIUS_MAX: i32 = 4096;
const CAMERA_RADIUS_STEP: i32 = 128;
const CAMERA_YAW_STEP_PER_VBLANK: u16 = 32;
const CAMERA_PITCH_STEP_PER_VBLANK: u16 = 24;
const CAMERA_PITCH_START: u16 = 350;
const WORLD_DEPTH_RANGE: DepthRange =
    DepthRange::new(NEAR_Z, CAMERA_RADIUS_MAX + MODEL_WORLD_HEIGHT * 2);

const TRI_CAP: usize = 1536;
const MODEL_VERTEX_CAP: usize = 1024;
const JOINT_CAP: usize = 32;

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
static mut JOINT_VIEW_TRANSFORMS: [JointViewTransform; JOINT_CAP] =
    [const { JointViewTransform::ZERO }; JOINT_CAP];

struct ModelShowcase {
    model: Option<Model<'static>>,
    clips: [Option<Animation<'static>>; CLIPS.len()],
    current_clip: usize,
    clip_origin_vblanks: u32,
    camera_yaw: u16,
    camera_pitch: u16,
    camera_radius: i32,
}

impl Scene for ModelShowcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.model = Some(Model::from_bytes(MODEL_BLOB).expect("obsidian_wraith.psxmdl"));
        for (i, entry) in CLIPS.iter().enumerate() {
            self.clips[i] = Some(Animation::from_bytes(entry.blob).expect(entry.label));
        }
        upload_model_texture();
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::CIRCLE) || ctx.just_pressed(button::R1) {
            self.advance_clip(ctx, 1);
        }
        if ctx.just_pressed(button::SQUARE) || ctx.just_pressed(button::L1) {
            self.advance_clip(ctx, CLIPS.len() - 1);
        }

        let dt = ctx.time.delta_vblanks();
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self
                .camera_yaw
                .wrapping_sub(scale_u16(CAMERA_YAW_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self
                .camera_yaw
                .wrapping_add(scale_u16(CAMERA_YAW_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::UP) {
            self.camera_pitch = self
                .camera_pitch
                .wrapping_add(scale_u16(CAMERA_PITCH_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::DOWN) {
            self.camera_pitch = self
                .camera_pitch
                .wrapping_sub(scale_u16(CAMERA_PITCH_STEP_PER_VBLANK, dt));
        }
        if ctx.is_held(button::TRIANGLE) {
            self.camera_radius =
                (self.camera_radius - scale_i32(CAMERA_RADIUS_STEP, dt)).max(CAMERA_RADIUS_MIN);
        }
        if ctx.is_held(button::CROSS) {
            self.camera_radius =
                (self.camera_radius + scale_i32(CAMERA_RADIUS_STEP, dt)).min(CAMERA_RADIUS_MAX);
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        let Some(model) = self.model else {
            return;
        };
        let Some(animation) = self.clips[self.current_clip] else {
            return;
        };
        let material = TextureMaterial::opaque(
            TEX_CLUT.uv_clut_word(),
            TEX_TPAGE.uv_tpage_word(0),
            (0x80, 0x80, 0x80),
        )
        .with_raw_texture(true);
        let camera = WorldCamera::orbit(
            PROJECTION,
            CAMERA_TARGET,
            self.camera_radius,
            self.camera_yaw,
            self.camera_pitch,
        );

        let clip_tick = ctx
            .time
            .elapsed_vblanks()
            .wrapping_sub(self.clip_origin_vblanks);
        let frame_q12 = animation.phase_at_tick_q12(clip_tick, ctx.time.video_hz());
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
            clips: [const { None }; CLIPS.len()],
            current_clip: 0,
            clip_origin_vblanks: 0,
            camera_yaw: 0,
            camera_pitch: CAMERA_PITCH_START,
            camera_radius: CAMERA_RADIUS_START,
        }
    }

    fn advance_clip(&mut self, ctx: &Ctx, step: usize) {
        self.current_clip = (self.current_clip + step) % CLIPS.len();
        self.clip_origin_vblanks = ctx.time.elapsed_vblanks();
    }
}

fn scale_u16(value: u16, scale: u16) -> u16 {
    value.saturating_mul(scale)
}

fn scale_i32(value: i32, scale: u16) -> i32 {
    value.saturating_mul(scale as i32)
}

fn upload_model_texture() {
    let texture = Texture::from_bytes(TEXTURE_BLOB).expect("obsidian_wraith texture");
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
        unsafe { &mut JOINT_VIEW_TRANSFORMS },
        material,
        options,
    );
    if stats.primitive_overflow || stats.command_overflow || stats.vertex_overflow {
        return;
    }
}
