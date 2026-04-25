//! PS1 texture material state.
//!
//! A material here is intentionally small: the GPU still sees plain
//! GP0 packets, but callers can group the pieces that decide how a
//! textured primitive samples and blends: CLUT word, tpage word,
//! tint, raw-texture flag, dither flag, and semi-transparency mode.

use psx_hw::gpu::{gp0, pack_color};
use psx_io::gpu::{wait_cmd_ready, write_gp0};

/// PS1 semi-transparency mode.
///
/// The four non-opaque variants map directly to GP0(E1) / tpage
/// bits 5..6. `Opaque` means the primitive command's semi-transparent
/// bit is kept clear, so the destination pixel is overwritten.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Overwrite destination pixels.
    Opaque,
    /// `(background + foreground) / 2`.
    Average,
    /// `background + foreground`, clamped per channel.
    Add,
    /// `background - foreground`, clamped per channel.
    Subtract,
    /// `background + foreground / 4`, clamped per channel.
    AddQuarter,
}

impl BlendMode {
    /// Decode the two tpage semi-transparency bits.
    ///
    /// Hardware tpage bits do not encode "opaque"; the primitive
    /// opcode decides whether blending is active. This therefore
    /// returns one of the four blending variants.
    pub const fn from_tpage_bits(bits: u8) -> Self {
        match bits & 0x3 {
            0 => Self::Average,
            1 => Self::Add,
            2 => Self::Subtract,
            _ => Self::AddQuarter,
        }
    }

    /// Encode as GP0(E1) / tpage bits 5..6.
    ///
    /// `Opaque` uses the common average-blend encoding because the
    /// primitive semi-transparent bit will be clear.
    pub const fn tpage_bits(self) -> u8 {
        match self {
            Self::Opaque | Self::Average => 0,
            Self::Add => 1,
            Self::Subtract => 2,
            Self::AddQuarter => 3,
        }
    }

    /// True when a primitive using this material should set its
    /// semi-transparent command bit.
    pub const fn is_translucent(self) -> bool {
        !matches!(self, Self::Opaque)
    }
}

/// Textured primitive material.
///
/// `clut_word` is the packed CLUT handle used by indexed textures.
/// `tpage_word` is the packed tpage word normally produced by
/// `Tpage::uv_tpage_word(0)`. The material rewrites the tpage blend
/// bits from [`BlendMode`] so one base tpage can be reused across
/// opaque and translucent variants.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureMaterial {
    clut_word: u16,
    tpage_word: u16,
    tint: (u8, u8, u8),
    blend_mode: BlendMode,
    raw_texture: bool,
    dither: bool,
}

impl TextureMaterial {
    /// Build an opaque, unmodulated texture material.
    pub const fn new(clut_word: u16, tpage_word: u16) -> Self {
        Self::opaque(clut_word, tpage_word, (0x80, 0x80, 0x80))
    }

    /// Build an opaque texture material with a flat tint.
    pub const fn opaque(clut_word: u16, tpage_word: u16, tint: (u8, u8, u8)) -> Self {
        Self {
            clut_word,
            tpage_word,
            tint,
            blend_mode: BlendMode::Opaque,
            raw_texture: false,
            dither: false,
        }
    }

    /// Build a translucent texture material with a flat tint.
    ///
    /// Textured primitives still follow PS1 texel rules: the
    /// primitive command enables transparency, and sampled texels
    /// with bit 15 set actually blend through the selected mode.
    pub const fn blended(
        clut_word: u16,
        tpage_word: u16,
        tint: (u8, u8, u8),
        blend_mode: BlendMode,
    ) -> Self {
        Self {
            clut_word,
            tpage_word,
            tint,
            blend_mode,
            raw_texture: false,
            dither: false,
        }
    }

    /// Return a copy using `blend_mode`.
    pub const fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    /// Return a copy using `tint`.
    pub const fn with_tint(mut self, tint: (u8, u8, u8)) -> Self {
        self.tint = tint;
        self
    }

    /// Return a copy with raw texture modulation enabled or disabled.
    ///
    /// Raw-texture mode bypasses RGB tint modulation. It is useful for
    /// palettes or direct textures that already carry final colors.
    pub const fn with_raw_texture(mut self, raw_texture: bool) -> Self {
        self.raw_texture = raw_texture;
        self
    }

    /// Return a copy with draw-mode dithering enabled or disabled.
    pub const fn with_dither(mut self, dither: bool) -> Self {
        self.dither = dither;
        self
    }

    /// Packed CLUT word for the first textured UV word.
    pub const fn clut_word(self) -> u16 {
        self.clut_word
    }

    /// Packed tpage word for textured polygon UV words.
    ///
    /// The material owns the tpage blend and dither bits; all other
    /// tpage address/depth bits are preserved from the original word.
    pub const fn tpage_word(self) -> u16 {
        let blend_bits = (self.blend_mode.tpage_bits() as u16) << 5;
        let dither_bit = (self.dither as u16) << 9;
        (self.tpage_word & !(0x0060 | 0x0200)) | blend_bits | dither_bit
    }

    /// Flat tint used by non-Gouraud textured primitives.
    pub const fn tint(self) -> (u8, u8, u8) {
        self.tint
    }

    /// Active blend mode.
    pub const fn blend_mode(self) -> BlendMode {
        self.blend_mode
    }

    /// True when the primitive command should set its semi-transparent bit.
    pub const fn is_translucent(self) -> bool {
        self.blend_mode.is_translucent()
    }

    /// True when the primitive command should set its raw-texture bit.
    pub const fn raw_texture(self) -> bool {
        self.raw_texture
    }

    /// True when the material asks GP0(E1) / primitive tpage state for dithering.
    pub const fn dither(self) -> bool {
        self.dither
    }

    /// Textured polygon command bits without the low RGB payload.
    pub const fn textured_polygon_command(self, gouraud: bool, quad: bool) -> u32 {
        gp0::polygon_opcode(gouraud, quad, true, self.is_translucent(), self.raw_texture)
    }

    /// Textured polygon header with the material's flat tint.
    pub const fn flat_textured_polygon_header(self, quad: bool) -> u32 {
        let (r, g, b) = self.tint;
        self.textured_polygon_command(false, quad) | pack_color(r, g, b)
    }

    /// Textured rectangle command with the material's flat tint.
    pub const fn textured_rect_header(self) -> u32 {
        let (r, g, b) = self.tint;
        0x6400_0000
            | ((self.is_translucent() as u32) << 25)
            | ((self.raw_texture as u32) << 24)
            | pack_color(r, g, b)
    }

    /// GP0(E1) draw-mode word implied by this material.
    ///
    /// Textured rectangles read their tpage from this state. Textured
    /// polygons embed the tpage word directly, but applying the same
    /// material state before drawing keeps subsequent sprite draws in
    /// sync.
    pub const fn draw_mode_word(self) -> u32 {
        let tpage = self.tpage_word();
        gp0::draw_mode(
            (tpage & 0x0F) as u32,
            ((tpage >> 4) & 1) as u32,
            self.blend_mode.tpage_bits() as u32,
            ((tpage >> 7) & 0x3) as u32,
            self.dither,
            true,
        )
    }

    /// Apply this material's tpage, blend, depth, and dither state.
    pub fn apply_draw_mode(self) {
        wait_cmd_ready();
        write_gp0(self.draw_mode_word());
    }
}

#[cfg(test)]
mod tests {
    use super::{BlendMode, TextureMaterial};

    #[test]
    fn blend_mode_decodes_tpage_bits() {
        assert_eq!(BlendMode::from_tpage_bits(0), BlendMode::Average);
        assert_eq!(BlendMode::from_tpage_bits(1), BlendMode::Add);
        assert_eq!(BlendMode::from_tpage_bits(2), BlendMode::Subtract);
        assert_eq!(BlendMode::from_tpage_bits(3), BlendMode::AddQuarter);
        assert_eq!(BlendMode::from_tpage_bits(4), BlendMode::Average);
    }

    #[test]
    fn material_encodes_blend_dither_and_command_bits() {
        let material =
            TextureMaterial::blended(0x1234, 0x018F, (0x80, 0x40, 0x20), BlendMode::AddQuarter)
                .with_raw_texture(true)
                .with_dither(true);

        assert_eq!(material.clut_word(), 0x1234);
        assert_eq!((material.tpage_word() >> 5) & 0x3, 3);
        assert_eq!((material.tpage_word() >> 9) & 0x1, 1);
        assert_eq!((material.textured_rect_header() >> 24) & 0xFF, 0x67);
        assert_eq!(
            (material.flat_textured_polygon_header(true) >> 24) & 0xFF,
            0x2F
        );
    }
}
