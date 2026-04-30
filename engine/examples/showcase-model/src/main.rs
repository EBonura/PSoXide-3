//! `showcase-model` -- animated-model demo with model + clip cycling.
//!
//! Loads two cooked models that share the same Meshy biped rig, plus
//! every cooked `.psxanim` clip for each. The camera spherically
//! orbits the active model; D-pad rotates, Triangle/Cross dolly, and
//! Square/Circle step through the active model's clip set. Select
//! swaps between the two characters -- texture is re-uploaded into
//! the same VRAM slot, animation phase resets so the new clip starts
//! cleanly. An overlay draws the active model name, the current clip
//! name, and a one-line control reminder.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Animation, Model, Texture};
use psx_engine::{
    button, App, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange, JointViewTransform,
    Mat3I16, OtFrame, PrimitiveArena, ProjectedVertex, Scene, WorldCamera, WorldProjection,
    WorldRenderPass, WorldSurfaceOptions, WorldTriCommand, WorldVertex,
};
use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

struct ClipEntry {
    label: &'static str,
    blob: &'static [u8],
}

struct ModelEntry {
    label: &'static str,
    model: &'static [u8],
    texture: &'static [u8],
    clips: &'static [ClipEntry],
}

/// Both models share the Meshy biped rig (24 joints, identical
/// hierarchy and bind pose), so a single cooked clip set animates
/// either character. We cook the eight clips off the obsidian glb
/// because that's the drop with the full move list -- the hooded
/// wretch glb only ships running and walking.
///
/// The source export's animation names are offset from the actual
/// tracks. Labels here describe the observed motion, not the original
/// filename suffix.
const SHARED_CLIPS: &[ClipEntry] = &[
    ClipEntry {
        label: "dead",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_walk_backward_inplace.psxanim"
        ),
    },
    ClipEntry {
        label: "double_combo_attack",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_walking.psxanim"
        ),
    },
    ClipEntry {
        label: "hit_reaction",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_dead.psxanim"
        ),
    },
    ClipEntry {
        label: "idle",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_double_combo_attack.psxanim"
        ),
    },
    ClipEntry {
        label: "running",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_hit_reaction.psxanim"
        ),
    },
    ClipEntry {
        label: "unsteady_walk",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_idle.psxanim"
        ),
    },
    ClipEntry {
        label: "walk_backward_inplace",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_running.psxanim"
        ),
    },
    ClipEntry {
        label: "walking",
        blob: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_unsteady_walk.psxanim"
        ),
    },
];

const MODELS: &[ModelEntry] = &[
    ModelEntry {
        label: "Obsidian Wraith",
        model: include_bytes!("../../../../assets/models/obsidian_wraith/obsidian_wraith.psxmdl"),
        texture: include_bytes!(
            "../../../../assets/models/obsidian_wraith/obsidian_wraith_128x128_8bpp.psxt"
        ),
        clips: SHARED_CLIPS,
    },
    ModelEntry {
        label: "Hooded Wretch",
        model: include_bytes!("../../../../assets/models/hooded_wretch/hooded_wretch.psxmdl"),
        texture: include_bytes!(
            "../../../../assets/models/hooded_wretch/hooded_wretch_128x128_8bpp.psxt"
        ),
        clips: SHARED_CLIPS,
    },
];

const MAX_CLIPS: usize = 8;
const SHOWCASE_START_CLIP: usize = 3;

const SCREEN_CX: i32 = 160;
const SCREEN_CY: i32 = 118;
const PROJ_FOCAL: i32 = 320;
const NEAR_Z: i32 = 48;
const PROJECTION: WorldProjection =
    WorldProjection::new(SCREEN_CX as i16, SCREEN_CY as i16, PROJ_FOCAL, NEAR_Z);

const OT_DEPTH: usize = 128;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 2);

/// Model texture lives in a dedicated 8bpp page well clear of the
/// font atlas at x=320. Re-upload happens on every model swap.
const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit8);
const TEX_CLUT: Clut = Clut::new(0, 482);

/// 4bpp 8x8 BIOS-style font atlas. Sits at x=320 (multiple of 64),
/// clear of both display buffers and the model's 8bpp page at 640.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

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
static mut MODEL_VERTICES: [ProjectedVertex; MODEL_VERTEX_CAP] =
    [ProjectedVertex::new(0, 0, 0); MODEL_VERTEX_CAP];
static mut JOINT_VIEW_TRANSFORMS: [JointViewTransform; JOINT_CAP] =
    [const { JointViewTransform::ZERO }; JOINT_CAP];

struct ModelShowcase {
    current_model: usize,
    current_clip: usize,
    model: Option<Model<'static>>,
    animations: [Option<Animation<'static>>; MAX_CLIPS],
    clip_origin_vblanks: u32,
    camera_yaw: u16,
    camera_pitch: u16,
    camera_radius: i32,
    font: Option<FontAtlas>,
}

impl Scene for ModelShowcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.font = Some(FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT));
        self.activate_model(0, 0);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::SELECT) {
            let next = (self.current_model + 1) % MODELS.len();
            self.activate_model(next, ctx.time.elapsed_vblanks());
        }

        let entry = &MODELS[self.current_model];
        if ctx.just_pressed(button::CIRCLE) || ctx.just_pressed(button::R1) {
            self.advance_clip(ctx, 1);
        }
        if ctx.just_pressed(button::SQUARE) || ctx.just_pressed(button::L1) {
            self.advance_clip(ctx, entry.clips.len().saturating_sub(1));
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
        let entry = &MODELS[self.current_model];
        if let (Some(model), Some(animation)) = (self.model, self.animations[self.current_clip]) {
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

        if let Some(font) = self.font.as_ref() {
            draw_overlay(font, entry, self.current_clip);
        }
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
            current_model: 0,
            current_clip: 0,
            model: None,
            animations: [const { None }; MAX_CLIPS],
            clip_origin_vblanks: 0,
            camera_yaw: 0,
            camera_pitch: CAMERA_PITCH_START,
            camera_radius: CAMERA_RADIUS_START,
            font: None,
        }
    }

    /// Swap to the model at `index`. Re-parses model, re-uploads
    /// texture into the shared VRAM slot, re-parses every clip, and
    /// resets clip phase to `now_vblanks` so the new clip starts at
    /// frame 0.
    fn activate_model(&mut self, index: usize, now_vblanks: u32) {
        self.current_model = index;
        let entry = &MODELS[index];
        self.current_clip = SHOWCASE_START_CLIP.min(entry.clips.len().saturating_sub(1));
        self.clip_origin_vblanks = now_vblanks;
        self.model = Some(Model::from_bytes(entry.model).expect(entry.label));
        upload_model_texture(entry.texture);
        for slot in self.animations.iter_mut() {
            *slot = None;
        }
        for (i, clip) in entry.clips.iter().enumerate() {
            if i >= MAX_CLIPS {
                break;
            }
            self.animations[i] = Some(Animation::from_bytes(clip.blob).expect(clip.label));
        }
    }

    fn advance_clip(&mut self, ctx: &Ctx, step: usize) {
        let count = MODELS[self.current_model].clips.len().max(1);
        self.current_clip = (self.current_clip + step) % count;
        self.clip_origin_vblanks = ctx.time.elapsed_vblanks();
    }
}

fn scale_u16(value: u16, scale: u16) -> u16 {
    value.saturating_mul(scale)
}

fn scale_i32(value: i32, scale: u16) -> i32 {
    value.saturating_mul(scale as i32)
}

fn upload_model_texture(blob: &[u8]) {
    let texture = Texture::from_bytes(blob).expect("model texture");
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
        Mat3I16::IDENTITY,
        unsafe { &mut MODEL_VERTICES },
        unsafe { &mut JOINT_VIEW_TRANSFORMS },
        material,
        options,
    );
    if stats.primitive_overflow || stats.command_overflow || stats.vertex_overflow {
        return;
    }
}

/// One-frame text overlay: model name + clip name at the top, control
/// reminder at the bottom. Drawn after the 3D scene so it sits over
/// the model without participating in the OT.
fn draw_overlay(font: &FontAtlas, entry: &ModelEntry, current_clip: usize) {
    font.draw_text(8, 8, entry.label, (235, 235, 245));

    let clip_label = entry
        .clips
        .get(current_clip)
        .map(|c| c.label)
        .unwrap_or("-");
    font.draw_text(8, 20, clip_label, (160, 220, 160));

    font.draw_text(8, 216, "DPAD orbit  TRI/X zoom", (180, 180, 200));
    font.draw_text(8, 226, "SQ/O clip  SEL model", (180, 180, 200));
}
