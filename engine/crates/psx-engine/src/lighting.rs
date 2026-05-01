//! Shared fixed-point point-light helpers.
//!
//! The editor host preview and the PSX runtime both use the same
//! neutral-128 tint convention:
//!
//! ```text
//! light_rgb in 0..=255:
//!   0    black
//!   128  neutral texture tint
//!   255  saturated overbright
//!
//! final_rgb = clamp(base_rgb * light_rgb / 128, 0, 255)
//! ```
//!
//! Keeping the arithmetic here prevents the authoring viewport and
//! embedded play mode from drifting apart.

use crate::Q8;

/// Neutral lighting value: a material renders at its base tint.
pub const LIGHTING_NEUTRAL: u32 = 128;

/// Saturated lighting value before material tint modulation.
pub const LIGHTING_MAX: u32 = 255;

/// 8-bit RGB colour.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rgb8 {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

impl Rgb8 {
    /// Black.
    pub const BLACK: Self = Self::new(0, 0, 0);

    /// Neutral PSX material/light value.
    pub const NEUTRAL: Self = Self::new(128, 128, 128);

    /// White.
    pub const WHITE: Self = Self::new(255, 255, 255);

    /// Build from channels.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Build from `[r, g, b]`.
    pub const fn from_array(channels: [u8; 3]) -> Self {
        Self::new(channels[0], channels[1], channels[2])
    }

    /// Build from `(r, g, b)`.
    pub const fn from_tuple(channels: (u8, u8, u8)) -> Self {
        Self::new(channels.0, channels.1, channels.2)
    }

    /// Return as `[r, g, b]`.
    pub const fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Return as `(r, g, b)`.
    pub const fn to_tuple(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }
}

/// Material tint in the renderer's neutral-128 convention.
///
/// This is stored as RGB8, but the type names the semantics:
/// `(128,128,128)` means "unchanged texture/material", not 50%
/// brightness in a linear colour space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaterialTint {
    rgb: Rgb8,
}

impl MaterialTint {
    /// Neutral tint.
    pub const NEUTRAL: Self = Self { rgb: Rgb8::NEUTRAL };

    /// Build from RGB channels.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self {
            rgb: Rgb8::new(r, g, b),
        }
    }

    /// Build from an [`Rgb8`] value.
    pub const fn from_rgb(rgb: Rgb8) -> Self {
        Self { rgb }
    }

    /// Build from `(r, g, b)`.
    pub const fn from_tuple(channels: (u8, u8, u8)) -> Self {
        Self::from_rgb(Rgb8::from_tuple(channels))
    }

    /// Underlying RGB channels.
    pub const fn rgb(self) -> Rgb8 {
        self.rgb
    }

    /// Return as `(r, g, b)`.
    pub const fn to_tuple(self) -> (u8, u8, u8) {
        self.rgb.to_tuple()
    }
}

impl Default for MaterialTint {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// Lighting accumulator RGB.
///
/// Channels are wider than 8-bit while lights accumulate. Values are
/// clamped to [`LIGHTING_MAX`] before tint modulation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LightingRgb {
    /// Red channel.
    pub r: u32,
    /// Green channel.
    pub g: u32,
    /// Blue channel.
    pub b: u32,
}

impl LightingRgb {
    /// Black/no light.
    pub const BLACK: Self = Self::new(0, 0, 0);

    /// Neutral light.
    pub const NEUTRAL: Self = Self::new(LIGHTING_NEUTRAL, LIGHTING_NEUTRAL, LIGHTING_NEUTRAL);

    /// Build from channels.
    pub const fn new(r: u32, g: u32, b: u32) -> Self {
        Self { r, g, b }
    }

    /// Build from an 8-bit ambient RGB value.
    pub const fn from_rgb8(rgb: Rgb8) -> Self {
        Self::new(rgb.r as u32, rgb.g as u32, rgb.b as u32)
    }

    /// Add channels with saturation.
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self::new(
            self.r.saturating_add(rhs.r),
            self.g.saturating_add(rhs.g),
            self.b.saturating_add(rhs.b),
        )
    }

    /// Clamp all channels to [`LIGHTING_MAX`].
    pub const fn clamped(self) -> Self {
        Self::new(
            if self.r > LIGHTING_MAX {
                LIGHTING_MAX
            } else {
                self.r
            },
            if self.g > LIGHTING_MAX {
                LIGHTING_MAX
            } else {
                self.g
            },
            if self.b > LIGHTING_MAX {
                LIGHTING_MAX
            } else {
                self.b
            },
        )
    }

    /// Return as `(r, g, b)`.
    pub const fn to_tuple(self) -> (u32, u32, u32) {
        (self.r, self.g, self.b)
    }
}

/// One point light in engine units, pre-multiplied for cheap
/// per-surface accumulation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PointLightSample {
    /// Light position in room-local engine units.
    pub position: [i32; 3],
    /// Cutoff radius in engine units.
    pub radius: i32,
    /// `color * intensity.raw()` per channel.
    pub weighted_color: LightingRgb,
}

impl PointLightSample {
    /// Build a sample from typed RGB and Q8.8 intensity.
    pub fn from_rgb_intensity(position: [i32; 3], radius: i32, color: Rgb8, intensity: Q8) -> Self {
        let intensity = intensity.raw();
        Self {
            position,
            radius,
            weighted_color: LightingRgb::new(
                (color.r as u32).saturating_mul(intensity),
                (color.g as u32).saturating_mul(intensity),
                (color.b as u32).saturating_mul(intensity),
            ),
        }
    }

    /// Build a sample from 8-bit RGB and typed Q8.8 intensity.
    pub fn from_color_intensity(
        position: [i32; 3],
        radius: i32,
        color: [u8; 3],
        intensity: Q8,
    ) -> Self {
        Self::from_rgb_intensity(position, radius, Rgb8::from_array(color), intensity)
    }

    /// Build a sample from 8-bit RGB and raw Q8.8 intensity
    /// (`256 == 1.0`).
    pub fn from_color_intensity_q8(
        position: [i32; 3],
        radius: i32,
        color: [u8; 3],
        intensity_q8: u32,
    ) -> Self {
        Self::from_color_intensity(position, radius, color, Q8::from_raw(intensity_q8))
    }
}

/// Accumulate ambient plus all point-light contributions at `point`.
///
/// Contributions use linear distance falloff. The returned channels
/// are already clamped to [`LIGHTING_MAX`] and are ready for
/// [`modulate_tint`].
pub fn accumulate_point_lights<I>(point: [i32; 3], ambient: [u8; 3], lights: I) -> (u32, u32, u32)
where
    I: IntoIterator<Item = PointLightSample>,
{
    accumulate_point_lights_rgb(point, Rgb8::from_array(ambient), lights).to_tuple()
}

/// Typed variant of [`accumulate_point_lights`].
pub fn accumulate_point_lights_rgb<I>(point: [i32; 3], ambient: Rgb8, lights: I) -> LightingRgb
where
    I: IntoIterator<Item = PointLightSample>,
{
    let mut light_rgb = LightingRgb::from_rgb8(ambient);
    for light in lights {
        let Some(weight) = point_light_weight(point, light.position, light.radius) else {
            continue;
        };
        // `weighted_color = color * intensity.raw()`; `weight` is
        // distance falloff. Both are Q8 inputs, so shift by 16. This
        // deliberately stays in 32-bit arithmetic: generated light
        // intensity is u16, and the PS1 MIPS target handles this path
        // much more predictably than widened arithmetic.
        light_rgb = light_rgb.saturating_add(LightingRgb::new(
            light.weighted_color.r.saturating_mul(weight.raw()) >> 16,
            light.weighted_color.g.saturating_mul(weight.raw()) >> 16,
            light.weighted_color.b.saturating_mul(weight.raw()) >> 16,
        ));
    }
    light_rgb.clamped()
}

/// Modulate a base material tint by a clamped light RGB triple.
pub fn modulate_tint(base: (u8, u8, u8), light_rgb: (u32, u32, u32)) -> (u8, u8, u8) {
    modulate_material_tint(
        MaterialTint::from_tuple(base),
        LightingRgb::new(light_rgb.0, light_rgb.1, light_rgb.2),
    )
    .to_tuple()
}

/// Typed variant of [`modulate_tint`].
pub fn modulate_material_tint(base: MaterialTint, light_rgb: LightingRgb) -> MaterialTint {
    let mod_channel = |base: u8, light: u32| -> u8 {
        let blended = (base as u32 * light) / LIGHTING_NEUTRAL;
        blended.min(255) as u8
    };
    MaterialTint::new(
        mod_channel(base.rgb.r, light_rgb.r),
        mod_channel(base.rgb.g, light_rgb.g),
        mod_channel(base.rgb.b, light_rgb.b),
    )
}

/// Accumulate point lights at `point` and modulate `base` by the
/// resulting light RGB.
pub fn shade_tint_with_lights<I>(
    base: (u8, u8, u8),
    point: [i32; 3],
    ambient: [u8; 3],
    lights: I,
) -> (u8, u8, u8)
where
    I: IntoIterator<Item = PointLightSample>,
{
    shade_material_tint_with_lights(
        MaterialTint::from_tuple(base),
        point,
        Rgb8::from_array(ambient),
        lights,
    )
    .to_tuple()
}

/// Typed variant of [`shade_tint_with_lights`].
pub fn shade_material_tint_with_lights<I>(
    base: MaterialTint,
    point: [i32; 3],
    ambient: Rgb8,
    lights: I,
) -> MaterialTint
where
    I: IntoIterator<Item = PointLightSample>,
{
    modulate_material_tint(base, accumulate_point_lights_rgb(point, ambient, lights))
}

fn point_light_weight(point: [i32; 3], light_position: [i32; 3], radius: i32) -> Option<Q8> {
    if radius <= 0 {
        return None;
    }
    let radius = (radius as u32).min(u16::MAX as u32);
    let dx = abs_diff_u32(point[0], light_position[0]);
    let dy = abs_diff_u32(point[1], light_position[1]);
    let dz = abs_diff_u32(point[2], light_position[2]);
    if dx >= radius || dy >= radius || dz >= radius {
        return None;
    }

    let d2 = dx
        .checked_mul(dx)?
        .checked_add(dy.checked_mul(dy)?)?
        .checked_add(dz.checked_mul(dz)?)?;
    let r2 = radius.checked_mul(radius)?;
    if d2 >= r2 {
        return None;
    }

    let d = isqrt_u32(d2);
    Some(Q8::from_ratio(radius - d, radius))
}

fn abs_diff_u32(a: i32, b: i32) -> u32 {
    if a >= b {
        a.saturating_sub(b) as u32
    } else {
        b.saturating_sub(a) as u32
    }
}

fn isqrt_u32(value: u32) -> u32 {
    let mut x = value;
    let mut r = 0u32;
    let mut bit = 1u32 << 30;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_32_is_dim_not_white() {
        let lit = shade_tint_with_lights((128, 128, 128), [0, 0, 0], [32, 32, 32], []);
        assert_eq!(lit, (32, 32, 32));
    }

    #[test]
    fn ambient_128_is_neutral() {
        let lit = shade_tint_with_lights((128, 128, 128), [0, 0, 0], [128, 128, 128], []);
        assert_eq!(lit, (128, 128, 128));
    }

    #[test]
    fn zero_ambient_zero_lights_is_black() {
        let lit = shade_tint_with_lights((255, 255, 255), [0, 0, 0], [0, 0, 0], []);
        assert_eq!(lit, (0, 0, 0));
    }

    #[test]
    fn point_light_inside_radius_brightens() {
        let light =
            PointLightSample::from_color_intensity([0, 0, 0], 100, [255, 255, 255], Q8::ONE);
        let lit = shade_tint_with_lights((128, 128, 128), [0, 0, 0], [32, 32, 32], [light]);
        assert!(lit.0 > 200 && lit.1 > 200 && lit.2 > 200, "got {:?}", lit);
    }

    #[test]
    fn point_light_outside_radius_contributes_zero() {
        let light =
            PointLightSample::from_color_intensity([0, 0, 0], 100, [255, 255, 255], Q8::ONE);
        let lit = shade_tint_with_lights((128, 128, 128), [10_000, 0, 0], [32, 32, 32], [light]);
        let baseline = shade_tint_with_lights((128, 128, 128), [10_000, 0, 0], [32, 32, 32], []);
        assert_eq!(lit, baseline);
    }

    #[test]
    fn two_lights_accumulate_and_clamp() {
        let light =
            PointLightSample::from_color_intensity([0, 0, 0], 100, [255, 255, 255], Q8::ONE);
        let lit =
            shade_tint_with_lights((255, 255, 255), [0, 0, 0], [128, 128, 128], [light, light]);
        assert_eq!(lit, (255, 255, 255));
    }

    #[test]
    fn low_intensity_colored_light_does_not_saturate() {
        let light = PointLightSample::from_color_intensity(
            [4374, 1024, 4165],
            3686,
            [239, 0, 65],
            Q8::from_raw(89),
        );
        let lit =
            shade_tint_with_lights((128, 128, 128), [4374, 1024, 4165], [32, 32, 32], [light]);
        assert_eq!(lit, (115, 32, 54));
    }
}
