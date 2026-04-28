//! Editor preview textures.
//!
//! Generates a small procedural 4bpp texture per project Material
//! resource and uploads it into the editor `HwRenderer`'s VRAM. A
//! cache keyed on `ResourceId` records the resulting (tpage, CLUT)
//! words so the editor preview can emit `TriTextured` packets that
//! sample the right region.
//!
//! Why procedural: real cooked textures from project PNGs land later
//! once the editor's asset pipeline is hooked up end-to-end. Until
//! then the preview already needs *something* in each material's
//! tpage so the texture-tint render path is exercised — and for the
//! Sims-style build flow, "stone-vs-brick-vs-wood" patterns convey
//! material identity better than flat colours.
//!
//! VRAM layout:
//!
//! ```text
//!   y = 0      ▶ 320×240 frame buffer (sub-rect the editor paints)
//!   y = 0      ▶ tpages 5..15  — 4bpp 64×64 textures, one per material
//!                packed left-to-right starting at x = 320
//!   y = 480    ▶ CLUT row, 16 halfwords per palette
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use psx_asset::Texture;
use psx_gpu_render::{VRAM_HEIGHT, VRAM_WIDTH};
use psxed_project::{MaterialResource, ProjectDocument, Resource, ResourceData, ResourceId};

/// Cached tpage/CLUT for one Material resource.
#[derive(Debug, Clone, Copy)]
pub struct MaterialSlot {
    /// Packed `uv_tpage_word` value the prim format wants in vertex
    /// 1's UV high half.
    pub tpage_word: u16,
    /// Packed `uv_clut_word` value the prim format wants in vertex
    /// 0's UV high half.
    pub clut_word: u16,
    /// Texture dimensions in texels — handy for UV computation.
    pub width: u8,
    pub height: u8,
}

/// Cache row keeping the slot together with the source signature so
/// we know when to invalidate. Re-uploading on every change leaks
/// the previous tpage / CLUT band; that's fine for editor lifetime
/// (we have ~10 tpages × 32 CLUT slots of slack), but we do skip the
/// re-upload entirely when the signature hasn't moved.
#[derive(Debug, Clone)]
struct CacheEntry {
    slot: MaterialSlot,
    /// Path of the `.psxt` resource that produced this slot. Empty
    /// string when the material has no texture or the file couldn't
    /// be read — then the slot holds a procedural fallback pattern.
    signature: String,
}

/// Owns the editor renderer's VRAM mirror plus the per-material
/// texture cache.
///
/// VRAM regions:
///
/// * `y = 0`,  tpages 0..4   — framebuffer (editor paints x=0..320).
/// * `y = 0`,  tpages 5..15  — 4bpp room material textures, 64×64
///   each, packed left-to-right.
/// * `y = 256`, tpage row 1   — 8bpp model atlases, packed
///   left-to-right by halfwords. Disjoint from the room region.
/// * `y = 480`                — 4bpp CLUTs, 16 halfwords each.
/// * `y = 481..`              — 8bpp CLUTs, 256 halfwords each.
///
/// Each Model resource maps to one cache entry keyed by its
/// `ResourceId`; same for Material resources. The two halves of
/// the cache use disjoint VRAM regions so a model atlas upload
/// never overwrites a room material and vice versa.
pub struct EditorTextures {
    vram: Box<[u16]>,
    cache: HashMap<ResourceId, CacheEntry>,
    model_cache: HashMap<ResourceId, ModelAtlasCacheEntry>,
    /// Index into the tpage row, starts at 5 (after the framebuffer
    /// at tpages 0..4) and bumps by 1 per uploaded texture.
    next_tpage: u8,
    /// Halfword X-coord of the next free 4bpp CLUT slot. Each
    /// 4bpp CLUT is 16 halfwords wide.
    next_clut_x: u16,
    /// Halfword X-coord cursor inside the 8bpp model atlas tpage
    /// row at y=256. Each atlas advances this by its halfword
    /// stride.
    next_model_tpage_x: u16,
    /// Y-coord of the next free 8bpp CLUT row. Steps down by 1
    /// per uploaded atlas (256-entry CLUTs are one row each).
    next_model_clut_y: u16,
}

/// Model atlas cache entry. Same shape as `CacheEntry` but
/// keyed by the Model resource id and signed by the atlas path
/// so editing `texture_path` triggers a re-upload on the next
/// `refresh_models` call.
#[derive(Debug, Clone)]
struct ModelAtlasCacheEntry {
    slot: MaterialSlot,
    /// Atlas path that produced this slot; empty when the model
    /// has no atlas (no slot is uploaded in that case — entry
    /// just records the empty signature so we don't re-walk).
    signature: String,
}

impl EditorTextures {
    pub fn new() -> Self {
        Self {
            vram: vec![0u16; (VRAM_WIDTH * VRAM_HEIGHT) as usize].into_boxed_slice(),
            cache: HashMap::new(),
            model_cache: HashMap::new(),
            // tpages 0..4 cover the framebuffer at x=0..256; the
            // editor paints into the 320×240 subrect of x=0..320,
            // so tpage 5 (x=320) is the first one fully clear.
            next_tpage: 5,
            next_clut_x: 0,
            // Model atlases live at the tpage row at y=256.
            // Cursor advances by halfword stride per atlas.
            next_model_tpage_x: 0,
            // 8bpp CLUTs sit just below the 4bpp band.
            next_model_clut_y: 481,
        }
    }

    /// Borrow the VRAM array for `HwRenderer::render_frame`.
    pub fn vram_words(&self) -> &[u16] {
        &self.vram
    }

    /// Look up a material's texture slot, or `None` if it hasn't
    /// been uploaded (the editor preview should fall back to flat
    /// shading in that case).
    pub fn slot(&self, id: ResourceId) -> Option<MaterialSlot> {
        self.cache.get(&id).map(|e| e.slot)
    }

    /// Look up a Model resource's atlas slot, or `None` if no
    /// atlas has been uploaded for that model (no atlas path,
    /// or upload failed).
    pub fn model_atlas_slot(&self, id: ResourceId) -> Option<MaterialSlot> {
        self.model_cache.get(&id).map(|e| e.slot)
    }

    /// Walk every Material resource and ensure its texture is in VRAM.
    ///
    /// Resolution order per material:
    ///
    /// 1. Follow `material.texture` to a `ResourceData::Texture`.
    /// 2. Resolve `psxt_path` (absolute as-is, otherwise against
    ///    `project_root`).
    /// 3. `fs::read` and parse via [`psx_asset::Texture::from_bytes`].
    /// 4. On any failure, fall back to a name-keyed procedural
    ///    pattern (brick / stone / wood / metal / glass / default
    ///    checker) so the preview is never blank.
    ///
    /// Cached signature is `psxt_path` so editing the path
    /// re-uploads on the next refresh; unchanged paths short-circuit.
    pub fn refresh(&mut self, project: &ProjectDocument, project_root: &Path) {
        for resource in &project.resources {
            let ResourceData::Material(material) = &resource.data else {
                continue;
            };
            let signature = texture_path(project, material).unwrap_or_default();
            if self.cache.get(&resource.id).is_some_and(|entry| entry.signature == signature) {
                continue;
            }
            if self.next_tpage > 15 || self.next_clut_x + 16 > VRAM_WIDTH as u16 {
                // Out of slots. Don't insert a stale cache entry.
                continue;
            }
            let slot = self
                .upload_real_psxt(&signature, project_root)
                .unwrap_or_else(|| self.upload_procedural(&resource.name));
            self.cache.insert(
                resource.id,
                CacheEntry {
                    slot,
                    signature,
                },
            );
        }
    }

    /// Read `path` and upload the parsed PSXT into the next free
    /// tpage / CLUT slot. Returns `None` if the path is empty, the
    /// file can't be read, the blob fails to parse, or the depth
    /// is unsupported by the editor preview path (only 4bpp + 8bpp
    /// indexed for now — the runtime supports 15bpp but editor's
    /// procedural fallback covers any holes).
    fn upload_real_psxt(&mut self, path: &str, project_root: &Path) -> Option<MaterialSlot> {
        if path.is_empty() {
            return None;
        }
        let abs = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            project_root.join(path)
        };
        let bytes = std::fs::read(&abs).ok()?;
        let texture = Texture::from_bytes(&bytes).ok()?;
        // PSX UVs are 8-bit so anything >256 wouldn't be addressable
        // from a single primitive anyway; reject taller-than-256
        // textures rather than silently producing wrong UVs.
        let width = u8::try_from(texture.width()).ok()?;
        let height = u8::try_from(texture.height()).ok()?;
        let tpage_index = self.next_tpage as u16;
        let tpage_x = tpage_index * 64;
        let tpage_y = 0;
        let clut_x = self.next_clut_x;
        let clut_y = 480;

        // Pixel halfwords: copy raw little-endian bytes from
        // `pixel_bytes` into VRAM. The runtime path's
        // `psx_vram::upload_bytes` does the same — we mirror it on
        // host because the editor's `HwRenderer` reads VRAM as
        // halfwords through `R16Uint`.
        let halfwords_per_row = texture.halfwords_per_row() as usize;
        let height_px = texture.height() as usize;
        let pixel_bytes = texture.pixel_bytes();
        if pixel_bytes.len() < halfwords_per_row * height_px * 2 {
            return None;
        }
        for row in 0..height_px {
            for hw in 0..halfwords_per_row {
                let off = (row * halfwords_per_row + hw) * 2;
                let word = u16::from_le_bytes([pixel_bytes[off], pixel_bytes[off + 1]]);
                let vram_idx = (tpage_y as usize + row) * VRAM_WIDTH as usize
                    + tpage_x as usize
                    + hw;
                self.vram[vram_idx] = word;
            }
        }

        // CLUT halfwords: 16 entries for 4bpp. The editor allocates
        // one 4bpp-sized CLUT band per material; 8bpp + 15bpp
        // follow-up since they need a wider band. Detect via the
        // declared CLUT entry count rather than the depth enum so
        // the only psx-asset surface this file touches is `Texture`.
        let clut_bytes = texture.clut_bytes();
        if texture.clut_entries() != 16 {
            return None;
        }
        if !clut_bytes.is_empty() {
            for i in 0..16 {
                let off = i * 2;
                if off + 1 >= clut_bytes.len() {
                    break;
                }
                let raw = u16::from_le_bytes([clut_bytes[off], clut_bytes[off + 1]]);
                // STP-bit hack the showcase applies: opaque texels
                // get bit15 set so PSX semi-transparent draws can
                // still discriminate against fully transparent
                // black. Mirroring keeps editor + runtime visuals
                // identical.
                let marked = if raw == 0 { 0 } else { raw | 0x8000 };
                let vram_idx =
                    (clut_y as usize) * VRAM_WIDTH as usize + clut_x as usize + i;
                self.vram[vram_idx] = marked;
            }
        }

        let slot = MaterialSlot {
            tpage_word: pack_tpage_word(tpage_index, 0),
            clut_word: pack_clut_word(clut_x, clut_y),
            width,
            height,
        };
        self.next_tpage += 1;
        self.next_clut_x += 16;
        Some(slot)
    }

    /// Stamp a name-keyed procedural pattern (brick / stone / wood /
    /// metal / glass / default checker) into the next free slot.
    /// Always returns a valid slot — assumes the caller already
    /// confirmed there's room.
    fn upload_procedural(&mut self, material_name: &str) -> MaterialSlot {
        let pattern = pattern_for_name(material_name);
        let tpage_index = self.next_tpage as u16;
        let tpage_x = tpage_index * 64;
        let tpage_y = 0;
        let clut_x = self.next_clut_x;
        let clut_y = 480;
        self.upload_4bpp(tpage_x, tpage_y, &pattern.pixels);
        self.upload_clut(clut_x, clut_y, &pattern.palette);
        let slot = MaterialSlot {
            tpage_word: pack_tpage_word(tpage_index, 0),
            clut_word: pack_clut_word(clut_x, clut_y),
            width: pattern.width,
            height: pattern.height,
        };
        self.next_tpage += 1;
        self.next_clut_x += 16;
        slot
    }

    /// Pack a 4bpp 64×64 pattern into VRAM at `(x, y)` (halfword
    /// coords). The pattern is 64 wide × 64 tall = 16 halfwords ×
    /// 64 rows; each halfword carries four 4bpp pixels (low nibble
    /// = leftmost).
    fn upload_4bpp(&mut self, tpage_x: u16, tpage_y: u16, pixels: &[u8]) {
        let halfwords_per_row = 16usize;
        for row in 0..64usize {
            for hw in 0..halfwords_per_row {
                let p0 = pixels[row * 64 + hw * 4 + 0] & 0x0F;
                let p1 = pixels[row * 64 + hw * 4 + 1] & 0x0F;
                let p2 = pixels[row * 64 + hw * 4 + 2] & 0x0F;
                let p3 = pixels[row * 64 + hw * 4 + 3] & 0x0F;
                let word = (p0 as u16)
                    | ((p1 as u16) << 4)
                    | ((p2 as u16) << 8)
                    | ((p3 as u16) << 12);
                let vram_idx =
                    (tpage_y as usize + row) * VRAM_WIDTH as usize + tpage_x as usize + hw;
                self.vram[vram_idx] = word;
            }
        }
    }

    fn upload_clut(&mut self, clut_x: u16, clut_y: u16, palette: &[u16; 16]) {
        for (i, &entry) in palette.iter().enumerate() {
            let vram_idx = (clut_y as usize) * VRAM_WIDTH as usize + clut_x as usize + i;
            self.vram[vram_idx] = entry;
        }
    }

    /// Walk every Model resource and ensure its atlas (if any)
    /// is uploaded into the dedicated 8bpp model atlas region.
    /// Models without `texture_path` get an empty cache entry so
    /// the walk doesn't repeatedly try to resolve them.
    pub fn refresh_models(&mut self, project: &ProjectDocument, project_root: &Path) {
        for resource in &project.resources {
            let ResourceData::Model(model) = &resource.data else {
                continue;
            };
            let signature = model.texture_path.clone().unwrap_or_default();
            if self
                .model_cache
                .get(&resource.id)
                .is_some_and(|entry| entry.signature == signature)
            {
                continue;
            }
            if signature.is_empty() {
                // No atlas to upload; record an empty signature
                // so subsequent refreshes skip cleanly.
                self.model_cache.remove(&resource.id);
                continue;
            }
            let abs = if Path::new(&signature).is_absolute() {
                PathBuf::from(&signature)
            } else {
                project_root.join(&signature)
            };
            let Some(slot) = self.upload_model_atlas_psxt(&abs) else {
                self.model_cache.remove(&resource.id);
                continue;
            };
            self.model_cache.insert(
                resource.id,
                ModelAtlasCacheEntry { slot, signature },
            );
        }
    }

    /// Read an 8bpp `.psxt` atlas and upload pixels + 256-entry
    /// CLUT into the dedicated model VRAM region. Returns `None`
    /// on missing file, parse failure, unsupported depth (only
    /// 8bpp is allowed), or VRAM exhaustion.
    fn upload_model_atlas_psxt(&mut self, abs: &Path) -> Option<MaterialSlot> {
        let bytes = std::fs::read(abs).ok()?;
        let texture = Texture::from_bytes(&bytes).ok()?;
        if texture.clut_entries() != 256 {
            // Only 8bpp atlases supported in this region — 4bpp
            // model atlases would belong in the room-material
            // path which we leave alone here.
            return None;
        }
        let width = u8::try_from(texture.width()).ok()?;
        let height = u8::try_from(texture.height()).ok()?;

        let halfwords_per_row = texture.halfwords_per_row();
        let height_px = texture.height();
        let pixel_bytes = texture.pixel_bytes();

        // PSX tpage word can only address tpage *bases*: each
        // page is 64 halfwords wide × 256 rows tall, identified
        // by `tpage_index = base_x / 64`. There's no per-atlas
        // base-X offset inside a page, so an atlas placed at a
        // non-64-halfword boundary would sample from the wrong
        // page. Editor-preview contract: each 8bpp model atlas
        // occupies exactly one 64-halfword tpage column. That
        // covers up to a 128-pixel-wide 8bpp atlas (which the
        // current Wraith / Hooded Wretch atlases match).
        const HALFWORDS_PER_TPAGE: u16 = 64;
        if halfwords_per_row > HALFWORDS_PER_TPAGE {
            // Wider than one tpage column — not addressable
            // from a single primitive.
            return None;
        }
        let aligned_tpage_x = align_up_to(self.next_model_tpage_x, HALFWORDS_PER_TPAGE);
        if aligned_tpage_x as u32 + HALFWORDS_PER_TPAGE as u32 > VRAM_WIDTH as u32 {
            return None;
        }
        if self.next_model_clut_y as usize >= VRAM_HEIGHT as usize {
            return None;
        }
        let tpage_x = aligned_tpage_x;
        let tpage_y: u16 = 256;
        let clut_y = self.next_model_clut_y;

        // Pixels.
        if pixel_bytes.len() < (halfwords_per_row as usize) * (height_px as usize) * 2 {
            return None;
        }
        for row in 0..height_px as usize {
            for hw in 0..halfwords_per_row as usize {
                let off = (row * halfwords_per_row as usize + hw) * 2;
                let word = u16::from_le_bytes([pixel_bytes[off], pixel_bytes[off + 1]]);
                let vram_idx = (tpage_y as usize + row) * VRAM_WIDTH as usize
                    + tpage_x as usize
                    + hw;
                self.vram[vram_idx] = word;
            }
        }

        // CLUT: 256 halfwords on a single row. Stamp the STP bit
        // on non-zero entries the same way the room-material
        // path does so opaque atlases never accidentally trigger
        // semi-transparency.
        let clut_bytes = texture.clut_bytes();
        if clut_bytes.len() < 512 {
            return None;
        }
        for i in 0..256 {
            let off = i * 2;
            let raw = u16::from_le_bytes([clut_bytes[off], clut_bytes[off + 1]]);
            let marked = if raw == 0 { 0 } else { raw | 0x8000 };
            let vram_idx = (clut_y as usize) * VRAM_WIDTH as usize + i;
            self.vram[vram_idx] = marked;
        }

        // Tpage word: tpage row at y=256 → tpage_y_block = 1.
        // Tpage X is `tpage_x / 64` (always 0..15 within a row).
        // 8bpp depth bit pattern is `1` (4bpp = 0).
        let tpage_index = tpage_x / HALFWORDS_PER_TPAGE;
        let slot = MaterialSlot {
            tpage_word: pack_8bpp_tpage_word(tpage_index, 1),
            clut_word: pack_clut_word(0, clut_y),
            width,
            height,
        };

        // Advance to the next aligned slot. Each atlas consumes
        // one 64-halfword tpage column regardless of actual
        // halfword stride — even a 32-halfword (64-pixel) atlas
        // burns a whole column to keep the tpage_index math
        // self-contained.
        self.next_model_tpage_x = aligned_tpage_x.saturating_add(HALFWORDS_PER_TPAGE);
        self.next_model_clut_y = self.next_model_clut_y.saturating_add(1);
        Some(slot)
    }
}

/// Resolve a Material's texture link to the underlying `.psxt`
/// path string, or `None` if the link is missing or the linked
/// resource isn't a Texture.
fn texture_path(project: &ProjectDocument, material: &MaterialResource) -> Option<String> {
    let tex_id = material.texture?;
    let resource: &Resource = project.resource(tex_id)?;
    match &resource.data {
        ResourceData::Texture { psxt_path } => Some(psxt_path.clone()),
        _ => None,
    }
}

/// One generated procedural texture: `pixels` are raw 4bpp indices
/// (0..15), one byte per pixel; `palette` is the 16-entry CLUT in
/// PSX BGR555 format.
struct ProceduralTexture {
    pixels: Vec<u8>,
    palette: [u16; 16],
    width: u8,
    height: u8,
}

fn pattern_for_name(name: &str) -> ProceduralTexture {
    let lower = name.to_ascii_lowercase();
    if lower.contains("brick") {
        brick_pattern()
    } else if lower.contains("floor") || lower.contains("stone") {
        stone_pattern()
    } else if lower.contains("glass") {
        glass_pattern()
    } else if lower.contains("wood") {
        wood_pattern()
    } else if lower.contains("metal") {
        metal_pattern()
    } else {
        default_pattern()
    }
}

/// Stamp a 64×64 brick wall: terra-cotta bricks separated by dark
/// mortar, alternating rows offset by half a brick.
fn brick_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x30, 0x18, 0x10)), // 0: mortar (dark)
        psx_555((0xC8, 0x70, 0x40)), // 1: brick base
        psx_555((0xB0, 0x60, 0x38)), // 2: brick darker
        psx_555((0xD8, 0x88, 0x58)), // 3: brick highlight
        psx_555((0x40, 0x20, 0x18)), // 4: deep shadow
        psx_555((0x50, 0x28, 0x18)),
        psx_555((0x60, 0x30, 0x20)),
        psx_555((0x70, 0x40, 0x28)),
        psx_555((0x80, 0x48, 0x30)),
        psx_555((0x90, 0x50, 0x38)),
        psx_555((0xA0, 0x58, 0x38)),
        psx_555((0xC0, 0x68, 0x40)),
        psx_555((0xD0, 0x78, 0x48)),
        psx_555((0xE0, 0x90, 0x60)),
        psx_555((0xF0, 0xA0, 0x70)),
        psx_555((0xFF, 0xB8, 0x88)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64 {
        let brick_row = y / 8;
        let row_offset = if brick_row % 2 == 0 { 0 } else { 8 };
        for x in 0..64 {
            let local_y = y % 8;
            let local_x = (x + row_offset) % 16;
            let nibble = if local_y == 0 || local_y == 7 {
                0 // horizontal mortar
            } else if local_x == 0 || local_x == 15 {
                0 // vertical mortar
            } else if local_y == 1 || local_x == 1 || local_x == 14 {
                2 // brick edge shadow
            } else if local_y == 6 {
                3 // brick top highlight (because wider on top in side-light)
            } else {
                1 // brick base
            };
            pixels[y * 64 + x] = nibble;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

/// 64×64 stone-tile floor: sand-toned squares with slightly darker
/// grout. The grout grid runs every 16 px so a single tile occupies
/// a quarter of the texture.
fn stone_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x60, 0x58, 0x4C)), // 0: grout
        psx_555((0xB6, 0xAC, 0x96)), // 1: stone base
        psx_555((0xA0, 0x96, 0x80)), // 2: stone darker
        psx_555((0xC8, 0xBC, 0xA8)), // 3: stone lighter
        psx_555((0x90, 0x88, 0x70)),
        psx_555((0x98, 0x90, 0x78)),
        psx_555((0xA8, 0x9E, 0x88)),
        psx_555((0xB0, 0xA6, 0x90)),
        psx_555((0xB8, 0xAE, 0x98)),
        psx_555((0xC0, 0xB6, 0xA0)),
        psx_555((0xC8, 0xBE, 0xA8)),
        psx_555((0xD0, 0xC4, 0xB0)),
        psx_555((0xD8, 0xCC, 0xB8)),
        psx_555((0xE0, 0xD4, 0xC0)),
        psx_555((0x88, 0x80, 0x6C)),
        psx_555((0x70, 0x68, 0x58)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64usize {
        for x in 0..64usize {
            let lx = x % 16;
            let ly = y % 16;
            let nibble: u8 = if lx == 0 || ly == 0 {
                0 // grout
            } else if lx == 1 || ly == 1 {
                2 // shadow on the inside of the grout
            } else if lx == 15 || ly == 15 {
                3 // highlight on the opposite edge
            } else {
                // Speckle the interior so it's not flat — pseudo-
                // random nibble derived from coords stays stable.
                let h = ((x as u32).wrapping_mul(73) ^ (y as u32).wrapping_mul(151)) & 0x07;
                if h < 2 {
                    2
                } else if h > 5 {
                    3
                } else {
                    1
                }
            };
            pixels[y * 64 + x] = nibble;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

fn glass_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x10, 0x20, 0x30)),
        psx_555((0x30, 0x60, 0x90)),
        psx_555((0x40, 0x78, 0xA8)),
        psx_555((0x50, 0x88, 0xB8)),
        psx_555((0x60, 0x98, 0xC8)),
        psx_555((0x70, 0xA8, 0xD0)),
        psx_555((0x80, 0xB0, 0xD8)),
        psx_555((0x88, 0xB8, 0xE0)),
        psx_555((0x90, 0xC0, 0xE8)),
        psx_555((0x98, 0xC8, 0xF0)),
        psx_555((0xA0, 0xD0, 0xF8)),
        psx_555((0xB0, 0xD8, 0xFF)),
        psx_555((0xC0, 0xE0, 0xFF)),
        psx_555((0x20, 0x40, 0x68)),
        psx_555((0x28, 0x48, 0x70)),
        psx_555((0x18, 0x30, 0x58)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            let dx = (x as i32) - 32;
            let dy = (y as i32) - 32;
            let r = (dx * dx + dy * dy) as u32;
            // Shade by distance to centre — soft cyan disc-like glow.
            let nibble = (12u32.saturating_sub(r / 96)).min(12) as u8;
            pixels[y * 64 + x] = nibble;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

fn wood_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x30, 0x18, 0x10)),
        psx_555((0x90, 0x60, 0x40)),
        psx_555((0x80, 0x50, 0x30)),
        psx_555((0xA0, 0x70, 0x48)),
        psx_555((0x70, 0x48, 0x28)),
        psx_555((0x60, 0x38, 0x20)),
        psx_555((0x50, 0x30, 0x18)),
        psx_555((0x40, 0x28, 0x18)),
        psx_555((0xB0, 0x80, 0x50)),
        psx_555((0xC0, 0x90, 0x58)),
        psx_555((0xD0, 0xA0, 0x68)),
        psx_555((0xA8, 0x78, 0x48)),
        psx_555((0x98, 0x68, 0x40)),
        psx_555((0x88, 0x58, 0x38)),
        psx_555((0x78, 0x50, 0x30)),
        psx_555((0x68, 0x48, 0x28)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            let plank = y / 16;
            let in_plank_y = y % 16;
            let nibble = if in_plank_y == 0 {
                0 // plank seam
            } else {
                // Grain — pseudo-random nibble per (plank, x); same
                // x in same plank gives same value so vertical lines.
                let seed = (plank as u32).wrapping_mul(31).wrapping_add(x as u32);
                let h = (seed.wrapping_mul(2654435761) >> 28) as u8;
                (h & 0x0F).max(1)
            };
            pixels[y * 64 + x] = nibble;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

fn metal_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x30, 0x30, 0x30)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x90, 0x90, 0x90)),
        psx_555((0xA0, 0xA0, 0xA0)),
        psx_555((0xB0, 0xB0, 0xB0)),
        psx_555((0xC0, 0xC0, 0xC0)),
        psx_555((0xD0, 0xD0, 0xD0)),
        psx_555((0xE0, 0xE0, 0xE0)),
        psx_555((0x70, 0x70, 0x70)),
        psx_555((0x60, 0x60, 0x60)),
        psx_555((0x50, 0x50, 0x50)),
        psx_555((0x40, 0x40, 0x40)),
        psx_555((0xF0, 0xF0, 0xF0)),
        psx_555((0xFF, 0xFF, 0xFF)),
        psx_555((0xC8, 0xC8, 0xC8)),
        psx_555((0xB8, 0xB8, 0xB8)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64usize {
        for x in 0..64usize {
            // Faint horizontal brushed-metal stripes.
            let stripe = (y / 2) & 1;
            let base = if stripe == 0 { 4u32 } else { 5u32 };
            let h = ((x as u32).wrapping_mul(101) ^ (y as u32).wrapping_mul(13)) & 0x03;
            pixels[y * 64 + x] = (base + h) as u8;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

fn default_pattern() -> ProceduralTexture {
    let palette = [
        psx_555((0x40, 0x40, 0x40)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0xA0, 0xA0, 0xA0)),
        psx_555((0xC0, 0xC0, 0xC0)),
        psx_555((0xE0, 0xE0, 0xE0)),
        psx_555((0xFF, 0x80, 0xFF)), // hot-pink for "missing pattern" debug
        psx_555((0xFF, 0xFF, 0x80)),
        psx_555((0x80, 0xFF, 0xFF)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
        psx_555((0x80, 0x80, 0x80)),
    ];
    let mut pixels = vec![0u8; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            let nibble = if (x / 4 + y / 4) & 1 == 0 { 1 } else { 3 };
            pixels[y * 64 + x] = nibble;
        }
    }
    ProceduralTexture {
        pixels,
        palette,
        width: 64,
        height: 64,
    }
}

/// Convert a 24-bit RGB triple to PSX 15bpp BGR555. Bit 15 (the STP
/// flag) stays 0; semitransparency hits later via the polygon
/// translucency bit, not the per-texel STP bit.
fn psx_555(rgb: (u8, u8, u8)) -> u16 {
    let r5 = (rgb.0 >> 3) as u16;
    let g5 = (rgb.1 >> 3) as u16;
    let b5 = (rgb.2 >> 3) as u16;
    (b5 << 10) | (g5 << 5) | r5
}

/// Pack a (tpage_index, tpage_y_block) pair into the GP0
/// uv1-high-half tpage word format. 4bpp depth, blend bits 0,
/// matching `psx_vram::Tpage::uv_tpage_word(0)`.
fn pack_tpage_word(tpage_index: u16, tpage_y_block: u16) -> u16 {
    let depth = 0u16; // 4bpp
    let semi_trans = 0u16; // 0.5*bg + 0.5*fg blend bits
    (tpage_index & 0xF) | (tpage_y_block << 4) | (semi_trans << 5) | (depth << 7)
}

/// Same as `pack_tpage_word` but with the 8bpp depth bit set —
/// used for model atlas slots which always live in the 8bpp
/// model VRAM region.
fn pack_8bpp_tpage_word(tpage_index: u16, tpage_y_block: u16) -> u16 {
    let depth = 1u16; // 8bpp
    let semi_trans = 0u16;
    (tpage_index & 0xF) | (tpage_y_block << 4) | (semi_trans << 5) | (depth << 7)
}

/// Pack a (clut_x_in_halfwords, clut_y) pair into the GP0
/// uv0-high-half CLUT word format.
fn pack_clut_word(clut_x_halfwords: u16, clut_y: u16) -> u16 {
    let cx = (clut_x_halfwords / 16) & 0x3F;
    let cy = clut_y & 0x1FF;
    cx | (cy << 6)
}

/// Round `value` up to the next multiple of `boundary`.
/// `boundary` must be a power of two for the bitmask path; the
/// general formula handles arbitrary moduli but the model
/// atlas allocator only ever passes 64.
fn align_up_to(value: u16, boundary: u16) -> u16 {
    if boundary == 0 {
        return value;
    }
    let rem = value % boundary;
    if rem == 0 {
        value
    } else {
        value.saturating_add(boundary - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::align_up_to;

    #[test]
    fn align_up_to_handles_aligned_and_misaligned_values() {
        assert_eq!(align_up_to(0, 64), 0);
        assert_eq!(align_up_to(1, 64), 64);
        assert_eq!(align_up_to(63, 64), 64);
        assert_eq!(align_up_to(64, 64), 64);
        assert_eq!(align_up_to(65, 64), 128);
        assert_eq!(align_up_to(127, 64), 128);
        assert_eq!(align_up_to(128, 64), 128);
        // Boundary 0 is a no-op (defensive — the allocator
        // only ever passes 64).
        assert_eq!(align_up_to(33, 0), 33);
    }
}
