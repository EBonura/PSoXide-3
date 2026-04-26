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

use psx_asset::{Animation, JointPose, Model, ModelVertex, Texture};
use psx_engine::{
    button, App, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange, LocalToWorldScale,
    OtFrame, PrimitiveArena, Scene, WorldCamera, WorldProjection, WorldRenderPass,
    WorldSurfaceOptions, WorldTriCommand, WorldVertex,
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
const CAMERA_AUTO_STEP: u16 = 4;
const CAMERA_PAD_STEP: u16 = 12;
const WORLD_DEPTH_RANGE: DepthRange =
    DepthRange::new(NEAR_Z, CAMERA_RADIUS_MAX + MODEL_WORLD_HEIGHT * 2);

const TRI_CAP: usize = 1536;

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
        self.camera_yaw = self.camera_yaw.wrapping_add(CAMERA_AUTO_STEP);
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self.camera_yaw.wrapping_sub(CAMERA_PAD_STEP);
        }
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self.camera_yaw.wrapping_add(CAMERA_PAD_STEP);
        }
        if ctx.is_held(button::UP) {
            self.camera_radius = (self.camera_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
        }
        if ctx.is_held(button::DOWN) {
            self.camera_radius = (self.camera_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
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

        let frame = animation_frame(
            ctx.frame,
            animation.frame_count(),
            animation.sample_rate_hz(),
        );
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };
        draw_animated_model(
            model,
            animation,
            frame,
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

fn animation_frame(frame: u32, frame_count: u16, sample_rate_hz: u16) -> u16 {
    if frame_count == 0 {
        return 0;
    }
    let divisor = (60 / sample_rate_hz.max(1) as u32).max(1);
    ((frame / divisor) % frame_count as u32) as u16
}

fn draw_animated_model(
    model: Model<'_>,
    animation: Animation<'_>,
    frame: u16,
    camera: WorldCamera,
    material: TextureMaterial,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
) {
    let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
        .with_depth_policy(DepthPolicy::Average)
        .with_cull_mode(CullMode::Back)
        .with_material_layer(material);
    let local_to_world = LocalToWorldScale::from_q12(model.local_to_world_q12());

    let mut part_index = 0;
    while part_index < model.part_count() {
        let Some(part) = model.part(part_index) else {
            break;
        };
        let Some(pose) = animation.pose(frame, part.joint_index()) else {
            part_index += 1;
            continue;
        };
        let first_face = part.first_face();
        let last_face = first_face.saturating_add(part.face_count());
        let mut face_index = first_face;
        while face_index < last_face {
            let Some((ia, ib, ic)) = model.face(face_index) else {
                break;
            };
            let Some(a) = model.vertex(ia) else {
                face_index += 1;
                continue;
            };
            let Some(b) = model.vertex(ib) else {
                face_index += 1;
                continue;
            };
            let Some(c) = model.vertex(ic) else {
                face_index += 1;
                continue;
            };

            let stats = world.submit_textured_world_triangle(
                triangles,
                camera,
                [
                    animated_world_vertex(a, pose, local_to_world),
                    animated_world_vertex(b, pose, local_to_world),
                    animated_world_vertex(c, pose, local_to_world),
                ],
                [a.uv, b.uv, c.uv],
                material,
                options,
            );
            if stats.primitive_overflow || stats.command_overflow {
                return;
            }

            face_index += 1;
        }
        part_index += 1;
    }
}

fn animated_world_vertex(
    vertex: ModelVertex,
    pose: JointPose,
    local_to_world: LocalToWorldScale,
) -> WorldVertex {
    let x = vertex.position.x as i32;
    let y = vertex.position.y as i32;
    let z = vertex.position.z as i32;
    let px = transform_q12(
        pose.matrix[0][0],
        pose.matrix[1][0],
        pose.matrix[2][0],
        pose.translation.x,
        x,
        y,
        z,
    );
    let py = transform_q12(
        pose.matrix[0][1],
        pose.matrix[1][1],
        pose.matrix[2][1],
        pose.translation.y,
        x,
        y,
        z,
    );
    let pz = transform_q12(
        pose.matrix[0][2],
        pose.matrix[1][2],
        pose.matrix[2][2],
        pose.translation.z,
        x,
        y,
        z,
    );
    WorldVertex::new(
        local_to_world.apply(px),
        local_to_world.apply(py) + MODEL_Y_OFFSET,
        local_to_world.apply(pz),
    )
}

fn transform_q12(m0: i16, m1: i16, m2: i16, t: i32, x: i32, y: i32, z: i32) -> i32 {
    ((m0 as i32 * x) >> 12) + ((m1 as i32 * y) >> 12) + ((m2 as i32 * z) >> 12) + t
}
