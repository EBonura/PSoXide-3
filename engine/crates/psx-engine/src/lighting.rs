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

/// Neutral lighting value: a material renders at its base tint.
pub const LIGHTING_NEUTRAL: u32 = 128;

/// Saturated lighting value before material tint modulation.
pub const LIGHTING_MAX: u32 = 255;

/// One point light in engine units, pre-multiplied for cheap
/// per-surface accumulation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PointLightSample {
    /// Light position in room-local engine units.
    pub position: [i32; 3],
    /// Cutoff radius in engine units.
    pub radius: i32,
    /// `color * intensity_q8` per channel.
    pub weighted_color: [u32; 3],
}

impl PointLightSample {
    /// Build a sample from 8-bit RGB and Q8.8 intensity
    /// (`256 == 1.0`).
    pub const fn from_color_intensity_q8(
        position: [i32; 3],
        radius: i32,
        color: [u8; 3],
        intensity_q8: u32,
    ) -> Self {
        Self {
            position,
            radius,
            weighted_color: [
                color[0] as u32 * intensity_q8,
                color[1] as u32 * intensity_q8,
                color[2] as u32 * intensity_q8,
            ],
        }
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
    let mut light_rgb: [u32; 3] = [ambient[0] as u32, ambient[1] as u32, ambient[2] as u32];
    for light in lights {
        let Some(weight_q8) = point_light_weight_q8(point, light.position, light.radius) else {
            continue;
        };
        // `weighted_color = color * intensity_q8`; `weight_q8` is
        // distance falloff. Both are Q8 inputs, so shift by 16. This
        // deliberately stays in 32-bit arithmetic: generated light
        // intensity is u16, and the PS1 MIPS target handles this path
        // much more predictably than widened arithmetic.
        light_rgb[0] =
            light_rgb[0].saturating_add(light.weighted_color[0].saturating_mul(weight_q8) >> 16);
        light_rgb[1] =
            light_rgb[1].saturating_add(light.weighted_color[1].saturating_mul(weight_q8) >> 16);
        light_rgb[2] =
            light_rgb[2].saturating_add(light.weighted_color[2].saturating_mul(weight_q8) >> 16);
    }
    (
        light_rgb[0].min(LIGHTING_MAX),
        light_rgb[1].min(LIGHTING_MAX),
        light_rgb[2].min(LIGHTING_MAX),
    )
}

/// Modulate a base material tint by a clamped light RGB triple.
pub fn modulate_tint(base: (u8, u8, u8), light_rgb: (u32, u32, u32)) -> (u8, u8, u8) {
    let mod_channel = |base: u8, light: u32| -> u8 {
        let blended = (base as u32 * light) / LIGHTING_NEUTRAL;
        blended.min(255) as u8
    };
    (
        mod_channel(base.0, light_rgb.0),
        mod_channel(base.1, light_rgb.1),
        mod_channel(base.2, light_rgb.2),
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
    modulate_tint(base, accumulate_point_lights(point, ambient, lights))
}

fn point_light_weight_q8(point: [i32; 3], light_position: [i32; 3], radius: i32) -> Option<u32> {
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
    Some(((radius - d) << 8) / radius)
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
        let light = PointLightSample::from_color_intensity_q8([0, 0, 0], 100, [255, 255, 255], 256);
        let lit = shade_tint_with_lights((128, 128, 128), [0, 0, 0], [32, 32, 32], [light]);
        assert!(lit.0 > 200 && lit.1 > 200 && lit.2 > 200, "got {:?}", lit);
    }

    #[test]
    fn point_light_outside_radius_contributes_zero() {
        let light = PointLightSample::from_color_intensity_q8([0, 0, 0], 100, [255, 255, 255], 256);
        let lit = shade_tint_with_lights((128, 128, 128), [10_000, 0, 0], [32, 32, 32], [light]);
        let baseline = shade_tint_with_lights((128, 128, 128), [10_000, 0, 0], [32, 32, 32], []);
        assert_eq!(lit, baseline);
    }

    #[test]
    fn two_lights_accumulate_and_clamp() {
        let light = PointLightSample::from_color_intensity_q8([0, 0, 0], 100, [255, 255, 255], 256);
        let lit =
            shade_tint_with_lights((255, 255, 255), [0, 0, 0], [128, 128, 128], [light, light]);
        assert_eq!(lit, (255, 255, 255));
    }

    #[test]
    fn low_intensity_colored_light_does_not_saturate() {
        let light =
            PointLightSample::from_color_intensity_q8([4374, 1024, 4165], 3686, [239, 0, 65], 89);
        let lit =
            shade_tint_with_lights((128, 128, 128), [4374, 1024, 4165], [32, 32, 32], [light]);
        assert_eq!(lit, (115, 32, 54));
    }
}
