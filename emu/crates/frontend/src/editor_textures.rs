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

use psx_gpu_render::{VRAM_HEIGHT, VRAM_WIDTH};
use psxed_project::{ProjectDocument, ResourceData, ResourceId};

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

/// Owns the editor renderer's VRAM mirror plus the per-material
/// texture cache.
pub struct EditorTextures {
    vram: Box<[u16]>,
    cache: HashMap<ResourceId, MaterialSlot>,
    /// Index into the tpage row, starts at 5 (after the framebuffer
    /// at tpages 0..4) and bumps by 1 per uploaded texture.
    next_tpage: u8,
    /// Halfword X-coord of the next free CLUT slot. Each 4bpp CLUT
    /// is 16 halfwords wide, so we step by 16.
    next_clut_x: u16,
}

impl EditorTextures {
    pub fn new() -> Self {
        Self {
            vram: vec![0u16; (VRAM_WIDTH * VRAM_HEIGHT) as usize].into_boxed_slice(),
            cache: HashMap::new(),
            // tpages 0..4 cover the framebuffer at x=0..256; the
            // editor paints into the 320×240 subrect of x=0..320,
            // so tpage 5 (x=320) is the first one fully clear.
            next_tpage: 5,
            next_clut_x: 0,
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
        self.cache.get(&id).copied()
    }

    /// Walk every Material resource and ensure it has a procedural
    /// texture in VRAM. Cheap when nothing's changed — the cache
    /// short-circuits per-resource. Call once per frame; anything
    /// not yet seen gets generated and uploaded right then.
    pub fn refresh(&mut self, project: &ProjectDocument) {
        for resource in &project.resources {
            if !matches!(resource.data, ResourceData::Material(_)) {
                continue;
            }
            if self.cache.contains_key(&resource.id) {
                continue;
            }
            // Out of CLUT row, or out of tpages — silently stop.
            // Future work: pack textures inside one tpage so the
            // 11 free tpages aren't a hard ceiling.
            if self.next_tpage > 15 || self.next_clut_x + 16 > VRAM_WIDTH as u16 {
                continue;
            }
            let pattern = pattern_for_name(&resource.name);
            let tpage_x = (self.next_tpage as u16) * 64;
            let tpage_y = 0;
            let clut_x = self.next_clut_x;
            let clut_y = 480;
            self.upload_4bpp(tpage_x, tpage_y, &pattern.pixels);
            self.upload_clut(clut_x, clut_y, &pattern.palette);
            self.cache.insert(
                resource.id,
                MaterialSlot {
                    tpage_word: pack_tpage_word(self.next_tpage as u16, 0),
                    clut_word: pack_clut_word(clut_x, clut_y),
                    width: pattern.width,
                    height: pattern.height,
                },
            );
            self.next_tpage += 1;
            self.next_clut_x += 16;
        }
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

/// Pack a (clut_x_in_halfwords, clut_y) pair into the GP0
/// uv0-high-half CLUT word format.
fn pack_clut_word(clut_x_halfwords: u16, clut_y: u16) -> u16 {
    let cx = (clut_x_halfwords / 16) & 0x3F;
    let cy = clut_y & 0x1FF;
    cx | (cy << 6)
}
