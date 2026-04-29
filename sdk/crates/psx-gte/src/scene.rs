//! Scene / camera / projection helpers.
//!
//! The register macros in [`regs`][crate::regs] are the only thing
//! that actually touches the GTE; everything here is a convenience
//! layer that bundles the ~8 writes a typical 3D frame needs into
//! named functions. All functions are safe — the macros they wrap
//! already contain the `unsafe { asm! }` internally, and there's
//! nothing we can do with a bad matrix value that would be undefined
//! behaviour (worst case: the projected vertex is garbage).
//!
//! Typical frame:
//!
//! ```ignore
//! scene::set_screen_offset(160 << 16, 120 << 16);
//! scene::set_projection_plane(200);
//! let rot = Mat3I16::rotate_y(angle);
//! scene::load_rotation(&rot);
//! scene::load_translation(Vec3I32::new(0, 0, 0x4000));
//! for v in vertices {
//!     let p = scene::project_vertex(v);
//!     draw_point(p.sx, p.sy);
//! }
//! ```

use crate::math::{Mat3I16, Vec3I16, Vec3I32};
use crate::ops;
use crate::regs::pack_xy;
use crate::{cfc2, ctc2, mfc2, mtc2};

/// Result of a single perspective-projected vertex — screen-space
/// (x, y) in pixels plus the MAC3 depth used for ordering-table
/// inserts. `Projected` is `Copy` + trivially packed so the caller
/// can collect per-vertex results into an array and rasterise later.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Projected {
    /// Screen-space X, clamped to GTE's ±0x400 range.
    pub sx: i16,
    /// Screen-space Y.
    pub sy: i16,
    /// Depth post-divide, 0..0xFFFF after saturation.
    pub sz: u16,
}

/// Load the rotation matrix into the GTE's RT control registers (0..=4).
pub fn load_rotation(m: &Mat3I16) {
    ctc2!(0, pack_xy(m.m[0][0], m.m[0][1]));
    ctc2!(1, pack_xy(m.m[0][2], m.m[1][0]));
    ctc2!(2, pack_xy(m.m[1][1], m.m[1][2]));
    ctc2!(3, pack_xy(m.m[2][0], m.m[2][1]));
    ctc2!(4, m.m[2][2] as i32 as u32);
}

/// Load the light-direction matrix (LLM, control 8..=12).
pub fn load_light_matrix(m: &Mat3I16) {
    ctc2!(8, pack_xy(m.m[0][0], m.m[0][1]));
    ctc2!(9, pack_xy(m.m[0][2], m.m[1][0]));
    ctc2!(10, pack_xy(m.m[1][1], m.m[1][2]));
    ctc2!(11, pack_xy(m.m[2][0], m.m[2][1]));
    ctc2!(12, m.m[2][2] as i32 as u32);
}

/// Load the light-colour matrix (LCM, control 16..=20).
pub fn load_light_colour_matrix(m: &Mat3I16) {
    ctc2!(16, pack_xy(m.m[0][0], m.m[0][1]));
    ctc2!(17, pack_xy(m.m[0][2], m.m[1][0]));
    ctc2!(18, pack_xy(m.m[1][1], m.m[1][2]));
    ctc2!(19, pack_xy(m.m[2][0], m.m[2][1]));
    ctc2!(20, m.m[2][2] as i32 as u32);
}

/// Load the translation vector (TR, control 5..=7).
pub fn load_translation(t: Vec3I32) {
    ctc2!(5, t.x as u32);
    ctc2!(6, t.y as u32);
    ctc2!(7, t.z as u32);
}

/// Load the background-colour bias (BK, control 13..=15).
pub fn load_background_colour(c: Vec3I32) {
    ctc2!(13, c.x as u32);
    ctc2!(14, c.y as u32);
    ctc2!(15, c.z as u32);
}

/// Load the far-colour bias (FC, control 21..=23) used by depth-cue
/// interpolation.
pub fn load_far_colour(c: Vec3I32) {
    ctc2!(21, c.x as u32);
    ctc2!(22, c.y as u32);
    ctc2!(23, c.z as u32);
}

/// Set OFX and OFY (control 24, 25) — the screen-space offsets applied
/// post-divide. Values are 15.16 fixed point; `160 << 16` = 160.0 px.
pub fn set_screen_offset(ofx_15_16: i32, ofy_15_16: i32) {
    ctc2!(24, ofx_15_16 as u32);
    ctc2!(25, ofy_15_16 as u32);
}

/// Set the projection-plane distance H (control 26). Larger H = longer
/// focal length = narrower FOV.
pub fn set_projection_plane(h: u16) {
    ctc2!(26, h as i32 as u32);
}

/// Set the depth-cue coefficients DQA / DQB (control 27, 28).
/// Depth-cue outputs IR0 = DQA/H + DQB, scaled to 0..0x1000.
pub fn set_depth_cue(dqa: i16, dqb: i32) {
    ctc2!(27, dqa as i32 as u32);
    ctc2!(28, dqb as u32);
}

/// Set the AVSZ3/AVSZ4 averaging weights (control 29, 30). Typical
/// values: `ZSF3 = 0x555` (= 1/3 in 0.12), `ZSF4 = 0x400` (= 1/4).
pub fn set_avsz_weights(zsf3: i16, zsf4: i16) {
    ctc2!(29, zsf3 as i32 as u32);
    ctc2!(30, zsf4 as i32 as u32);
}

/// Load `v` into the V0 input slot (data registers 0 and 1) and run
/// RTPS to project it. Returns the screen-space pair + depth so the
/// caller can immediately use the result.
///
/// Assumes the rotation matrix, translation, screen offset, and
/// projection plane have already been set.
pub fn project_vertex(v: Vec3I16) -> Projected {
    mtc2!(0, v.xy_packed());
    mtc2!(1, v.z_packed());
    // SAFETY: V0 has just been loaded; RT / TR / H / OFX / OFY are
    // assumed to be set by the caller's scene setup.
    unsafe { ops::rtps() };
    let sxy = mfc2!(14);
    let sz = mfc2!(19) as u16;
    Projected {
        sx: sxy as i16,
        sy: (sxy >> 16) as i16,
        sz,
    }
}

/// Project three vertices as a batch via RTPT — one GTE call, three
/// results out of the SXY FIFO + SZ FIFO. Slightly faster than three
/// successive [`project_vertex`] calls because RTPT shares setup.
///
/// The returned array is `[v0_result, v1_result, v2_result]`.
pub fn project_triangle(v0: Vec3I16, v1: Vec3I16, v2: Vec3I16) -> [Projected; 3] {
    // Load all three vertices first (data regs 0..=5), then fire RTPT.
    mtc2!(0, v0.xy_packed());
    mtc2!(1, v0.z_packed());
    mtc2!(2, v1.xy_packed());
    mtc2!(3, v1.z_packed());
    mtc2!(4, v2.xy_packed());
    mtc2!(5, v2.z_packed());
    // SAFETY: all three vertices are loaded; scene-setup registers
    // are the caller's responsibility.
    unsafe { ops::rtpt() };
    // After RTPT, SXY FIFO holds (v0, v1, v2) in slots 0/1/2, and
    // SZ FIFO holds them in SZ1/SZ2/SZ3.
    let sxy0 = mfc2!(12);
    let sxy1 = mfc2!(13);
    let sxy2 = mfc2!(14);
    let sz1 = mfc2!(17) as u16;
    let sz2 = mfc2!(18) as u16;
    let sz3 = mfc2!(19) as u16;
    [
        Projected {
            sx: sxy0 as i16,
            sy: (sxy0 >> 16) as i16,
            sz: sz1,
        },
        Projected {
            sx: sxy1 as i16,
            sy: (sxy1 >> 16) as i16,
            sz: sz2,
        },
        Projected {
            sx: sxy2 as i16,
            sy: (sxy2 >> 16) as i16,
            sz: sz3,
        },
    ]
}

/// Read the last three projected Z values and compute their average
/// via AVSZ3 (weighted by ZSF3). Returns OTZ — the depth key most
/// renderers use for ordering-table inserts.
pub fn average_z_triangle() -> u16 {
    // SAFETY: no input registers to prepare — AVSZ3 reads SZ1..SZ3
    // which were populated by the most recent RTPT / project_triangle.
    unsafe { ops::avsz3() };
    mfc2!(7) as u16
}

/// Read the GTE FLAG register. Non-zero indicates at least one error
/// bit fired during the last op (overflow, saturation, divide
/// overflow). Useful for debug prints on a frame that looks wrong.
pub fn read_flag() -> u32 {
    cfc2!(31)
}

#[cfg(all(test, not(target_arch = "mips")))]
mod host_smoke {
    //! Smoke tests for the host-side software-GTE shim.
    //!
    //! On hardware these helpers compile to inline COP2 instructions,
    //! so testing them via Rust integration would require running on
    //! a PS1. On host they route through the per-thread Gte from
    //! `psx-gte-core`, which we *can* poke at directly to confirm the
    //! routing produces matching output.
    use super::*;
    use crate::host;

    fn install_identity() {
        load_rotation(&Mat3I16::IDENTITY);
        load_translation(Vec3I32::ZERO);
        set_screen_offset(160 << 16, 120 << 16);
        set_projection_plane(200);
    }

    #[test]
    fn rtps_through_host_shim_projects_an_in_front_vertex() {
        host::reset();
        install_identity();
        // V0 = (0, 0, 1024) — straight ahead, depth 1024. With H=200
        // the GTE divides 200/sz3 (≈0x4000/sz3 internally), giving an
        // X/Y near the screen offset for a vertex at the origin.
        let projected = project_vertex(Vec3I16::new(0, 0, 1024));
        assert_eq!(projected.sx, 160);
        assert_eq!(projected.sy, 120);
        assert!(
            projected.sz > 0,
            "near-plane vertex must yield non-zero depth"
        );
    }

    #[test]
    fn rtpt_through_host_shim_matches_three_separate_rtps_calls() {
        host::reset();
        install_identity();
        let a = Vec3I16::new(-256, 0, 1024);
        let b = Vec3I16::new(256, 0, 1024);
        let c = Vec3I16::new(0, 256, 1024);

        let batch = project_triangle(a, b, c);

        host::reset();
        install_identity();
        let p_a = project_vertex(a);
        let p_b = project_vertex(b);
        let p_c = project_vertex(c);

        assert_eq!(batch[0], p_a);
        assert_eq!(batch[1], p_b);
        assert_eq!(batch[2], p_c);
    }
}
