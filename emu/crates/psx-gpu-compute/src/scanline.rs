//! Host-side port of the CPU rasterizer's `setup_sections` +
//! `next_row` Q16.16 scanline-delta math (`emulator-core/src/gpu.rs`).
//! Used by the compute-shader rasterizer to ship per-row state to the
//! GPU so the shader can do **bit-exact** UV / colour interpolation
//! that matches the CPU rasterizer (and Redux) byte-for-byte.
//!
//! Why this exists
//! --------------
//! The compute shader's pure-barycentric path produces off-by-1 or
//! off-by-2 UVs at some interior pixels because it does a fresh
//! `(w0*u0 + w1*u1 + w2*u2) / area` divide per pixel, while the CPU
//! does cumulative `c_u += dif_u` per row + `c_u += delta_col_u` per
//! pixel. The two are mathematically equivalent affine maps but
//! round differently. To match the CPU exactly, the GPU has to walk
//! the same accumulators with the same rounding.
//!
//! Approach: this module produces a `Vec<RowState>` with each row's
//! `left_x_q16`, `left_u_q16`, `left_v_q16`, `left_r/g/b_q16` plus
//! `right_x_q16` and the per-column deltas (uniform across the
//! primitive). The GPU's per-pixel job then becomes:
//! ```
//! col = px - (left_x_q16 >> 16);
//! u_q16 = left_u_q16 + col * delta_col_u_q16;
//! u = u_q16 >> 16;
//! ```
//! which uses the same arithmetic as the CPU.
//!
//! **DO NOT MODIFY** the math here without keeping
//! `emulator-core::gpu::setup_sections` in sync — every quirk
//! (`shl10_idiv`, the `longest` clamp, sort order, the pop-once-on-
//! degenerate-section dance) was tuned to hit pixel-exact parity
//! with PCSX-Redux and represents non-obvious knowledge.

use bytemuck::{Pod, Zeroable};

/// One vertex as the scanline setup sees it: positions in Q16.16,
/// Y as a plain integer (scanlines step by 1).
#[derive(Copy, Clone, Debug, Default)]
struct SlVertex {
    x: i64,
    y: i32,
    r: i64,
    g: i64,
    b: i64,
    u: i64,
    v: i64,
}

#[derive(Clone, Debug)]
struct SlTriSetup {
    sorted: [SlVertex; 3],
    left_array: [usize; 3],
    right_array: [usize; 3],
    left_section: i32,
    right_section: i32,
    left_x: i64,
    right_x: i64,
    delta_left_x: i64,
    delta_right_x: i64,
    left_section_height: i32,
    right_section_height: i32,
    left_r: i64,
    left_g: i64,
    left_b: i64,
    delta_left_r: i64,
    delta_left_g: i64,
    delta_left_b: i64,
    left_u: i64,
    left_v: i64,
    delta_left_u: i64,
    delta_left_v: i64,
    delta_col_r: i64,
    delta_col_g: i64,
    delta_col_b: i64,
    delta_col_u: i64,
    delta_col_v: i64,
    y_min: i32,
    y_max: i32,
}

/// `(x << 10) / y` with i64 intermediate, matching Redux's
/// `shl10idiv` helper at `soft.h:276`.
fn shl10_idiv(x: i64, y: i64) -> i64 {
    (x << 10) / y
}

impl SlTriSetup {
    fn pop_left_section(&mut self) -> Result<(), ()> {
        self.left_section -= 1;
        if self.left_section <= 0 {
            return Err(());
        }
        self.compute_left_section()
    }

    fn pop_right_section(&mut self) -> Result<(), ()> {
        self.right_section -= 1;
        if self.right_section <= 0 {
            return Err(());
        }
        self.compute_right_section()
    }

    fn compute_left_section(&mut self) -> Result<(), ()> {
        let idx1 = self.left_array[self.left_section as usize];
        let idx2 = self.left_array[(self.left_section - 1) as usize];
        let v1 = self.sorted[idx1];
        let v2 = self.sorted[idx2];
        let height = v2.y - v1.y;
        if height == 0 {
            return Err(());
        }
        let h = height as i64;
        self.delta_left_x = (v2.x - v1.x) / h;
        self.left_x = v1.x;
        self.delta_left_r = (v2.r - v1.r) / h;
        self.delta_left_g = (v2.g - v1.g) / h;
        self.delta_left_b = (v2.b - v1.b) / h;
        self.left_r = v1.r;
        self.left_g = v1.g;
        self.left_b = v1.b;
        self.delta_left_u = (v2.u - v1.u) / h;
        self.delta_left_v = (v2.v - v1.v) / h;
        self.left_u = v1.u;
        self.left_v = v1.v;
        self.left_section_height = height;
        Ok(())
    }

    fn compute_right_section(&mut self) -> Result<(), ()> {
        let idx1 = self.right_array[self.right_section as usize];
        let idx2 = self.right_array[(self.right_section - 1) as usize];
        let v1 = self.sorted[idx1];
        let v2 = self.sorted[idx2];
        let height = v2.y - v1.y;
        if height == 0 {
            return Err(());
        }
        let h = height as i64;
        self.delta_right_x = (v2.x - v1.x) / h;
        self.right_x = v1.x;
        self.right_section_height = height;
        Ok(())
    }

    fn next_row(&mut self) -> Result<(), ()> {
        self.left_section_height -= 1;
        if self.left_section_height <= 0 {
            self.pop_left_section()?;
        } else {
            self.left_x += self.delta_left_x;
            self.left_r += self.delta_left_r;
            self.left_g += self.delta_left_g;
            self.left_b += self.delta_left_b;
            self.left_u += self.delta_left_u;
            self.left_v += self.delta_left_v;
        }
        self.right_section_height -= 1;
        if self.right_section_height <= 0 {
            self.pop_right_section()?;
        } else {
            self.right_x += self.delta_right_x;
        }
        Ok(())
    }
}

fn setup_sections(
    v_x: [i32; 3],
    v_y: [i32; 3],
    v_rgb: [(i32, i32, i32); 3],
    v_uv: [(i32, i32); 3],
) -> Option<SlTriSetup> {
    let mut verts = [SlVertex::default(); 3];
    for i in 0..3 {
        verts[i] = SlVertex {
            x: (v_x[i] as i64) << 16,
            y: v_y[i],
            r: (v_rgb[i].0 as i64) << 16,
            g: (v_rgb[i].1 as i64) << 16,
            b: (v_rgb[i].2 as i64) << 16,
            u: (v_uv[i].0 as i64) << 16,
            v: (v_uv[i].1 as i64) << 16,
        };
    }
    if verts[0].y > verts[1].y {
        verts.swap(0, 1);
    }
    if verts[0].y > verts[2].y {
        verts.swap(0, 2);
    }
    if verts[1].y > verts[2].y {
        verts.swap(1, 2);
    }
    let v1 = &verts[0];
    let v2 = &verts[1];
    let v3 = &verts[2];
    let height = v3.y - v1.y;
    if height == 0 {
        return None;
    }
    let temp = ((v2.y - v1.y) as i64) << 16;
    let temp = temp / (height as i64);
    let longest = temp * ((v3.x - v1.x) >> 16) + (v1.x - v2.x);
    if longest == 0 {
        return None;
    }

    let mut setup = SlTriSetup {
        sorted: [verts[0], verts[1], verts[2]],
        left_array: [0; 3],
        right_array: [0; 3],
        left_section: 0,
        right_section: 0,
        left_x: 0,
        right_x: 0,
        delta_left_x: 0,
        delta_right_x: 0,
        left_section_height: 0,
        right_section_height: 0,
        left_r: 0,
        left_g: 0,
        left_b: 0,
        delta_left_r: 0,
        delta_left_g: 0,
        delta_left_b: 0,
        left_u: 0,
        left_v: 0,
        delta_left_u: 0,
        delta_left_v: 0,
        delta_col_r: 0,
        delta_col_g: 0,
        delta_col_b: 0,
        delta_col_u: 0,
        delta_col_v: 0,
        y_min: v1.y,
        y_max: v3.y - 1,
    };

    if longest < 0 {
        setup.right_array = [2, 1, 0];
        setup.right_section = 2;
        setup.left_array = [2, 0, 0];
        setup.left_section = 1;
        setup.compute_left_section().ok()?;
        if setup.compute_right_section().is_err() {
            setup.right_section -= 1;
            setup.compute_right_section().ok()?;
        }
    } else {
        setup.left_array = [2, 1, 0];
        setup.left_section = 2;
        setup.right_array = [2, 0, 0];
        setup.right_section = 1;
        setup.compute_right_section().ok()?;
        if setup.compute_left_section().is_err() {
            setup.left_section -= 1;
            setup.compute_left_section().ok()?;
        }
    }

    let longest_clamped: i64 = if longest < 0 {
        longest.min(-0x1000)
    } else {
        longest.max(0x1000)
    };
    let compute_col_delta = |a3: i64, a1: i64, a2: i64| -> i64 {
        shl10_idiv(
            (temp * ((a3 - a1) >> 10)) + ((a1 - a2) << 6),
            longest_clamped,
        )
    };
    setup.delta_col_r = compute_col_delta(v3.r, v1.r, v2.r);
    setup.delta_col_g = compute_col_delta(v3.g, v1.g, v2.g);
    setup.delta_col_b = compute_col_delta(v3.b, v1.b, v2.b);
    setup.delta_col_u = compute_col_delta(v3.u, v1.u, v2.u);
    setup.delta_col_v = compute_col_delta(v3.v, v1.v, v2.v);
    Some(setup)
}

// =============================================================
//  GPU-facing structs
// =============================================================

/// Per-row scanline state shipped to the GPU as a storage-buffer
/// array. Each row's left edge has all the accumulated Q16.16 values
/// the per-pixel walk would have computed by the time the CPU
/// rasterizer reaches that row.
///
/// The Q16.16 values are packed as **2 × i32** (high + low). WGSL
/// has no native i64, so the per-pixel arithmetic in the shader does
/// the carry handling explicitly. See the shader for the reconstruction
/// formula.
///
/// Layout pinned at 64 bytes for storage-buffer array stride.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RowState {
    /// `left_x` Q16.16, hi/lo split: `(left_x_q16 >> 32) as i32` and
    /// `left_x_q16 as u32`. The shader reconstructs as
    /// `(i64(hi) << 32) | u64(lo)`.
    pub left_x_hi: i32,
    pub left_x_lo: u32,
    pub right_x_hi: i32,
    pub right_x_lo: u32,
    pub left_u_hi: i32,
    pub left_u_lo: u32,
    pub left_v_hi: i32,
    pub left_v_lo: u32,
    pub left_r_hi: i32,
    pub left_r_lo: u32,
    pub left_g_hi: i32,
    pub left_g_lo: u32,
    pub left_b_hi: i32,
    pub left_b_lo: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Per-primitive constants (uniform across all rows): the per-column
/// deltas and the y_min/y_max scanline range. Same hi/lo i64 split.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ScanlineConsts {
    pub y_min: i32,
    pub y_max: i32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub delta_col_u_hi: i32,
    pub delta_col_u_lo: u32,
    pub delta_col_v_hi: i32,
    pub delta_col_v_lo: u32,
    pub delta_col_r_hi: i32,
    pub delta_col_r_lo: u32,
    pub delta_col_g_hi: i32,
    pub delta_col_g_lo: u32,
    pub delta_col_b_hi: i32,
    pub delta_col_b_lo: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Bundles host-computed per-row state + per-primitive constants.
pub struct ScanlineSetup {
    /// Length matches `consts.y_max - consts.y_min + 1`. Indexed by
    /// `(py - y_min)` from the shader.
    pub rows: Vec<RowState>,
    pub consts: ScanlineConsts,
}

fn split_i64(v: i64) -> (i32, u32) {
    let hi = (v >> 32) as i32;
    let lo = (v as u64 & 0xFFFF_FFFF) as u32;
    (hi, lo)
}

/// Build the per-row state for a triangle. Returns `None` if the
/// triangle degenerates (zero height or zero longest).
pub fn build_setup(
    v: [(i32, i32); 3],
    uv: [(i32, i32); 3],
    rgb: [(i32, i32, i32); 3],
) -> Option<ScanlineSetup> {
    let mut setup = setup_sections([v[0].0, v[1].0, v[2].0], [v[0].1, v[1].1, v[2].1], rgb, uv)?;

    // Walk every scanline, recording the left-edge state.
    let mut rows = Vec::with_capacity((setup.y_max - setup.y_min + 1).max(0) as usize);
    let mut y = setup.y_min;
    while y <= setup.y_max {
        let (lx_hi, lx_lo) = split_i64(setup.left_x);
        let (rx_hi, rx_lo) = split_i64(setup.right_x);
        let (lu_hi, lu_lo) = split_i64(setup.left_u);
        let (lv_hi, lv_lo) = split_i64(setup.left_v);
        let (lr_hi, lr_lo) = split_i64(setup.left_r);
        let (lg_hi, lg_lo) = split_i64(setup.left_g);
        let (lb_hi, lb_lo) = split_i64(setup.left_b);
        rows.push(RowState {
            left_x_hi: lx_hi,
            left_x_lo: lx_lo,
            right_x_hi: rx_hi,
            right_x_lo: rx_lo,
            left_u_hi: lu_hi,
            left_u_lo: lu_lo,
            left_v_hi: lv_hi,
            left_v_lo: lv_lo,
            left_r_hi: lr_hi,
            left_r_lo: lr_lo,
            left_g_hi: lg_hi,
            left_g_lo: lg_lo,
            left_b_hi: lb_hi,
            left_b_lo: lb_lo,
            _pad0: 0,
            _pad1: 0,
        });
        if setup.next_row().is_err() {
            // Walk-out: pad remaining rows with the last computed
            // state. The shader will skip them via the y > y_max
            // check, but having a valid array element keeps indexing
            // safe.
            let last = *rows.last().unwrap();
            while rows.len() < (setup.y_max - setup.y_min + 1) as usize {
                rows.push(last);
            }
            break;
        }
        y += 1;
    }

    let (du_hi, du_lo) = split_i64(setup.delta_col_u);
    let (dv_hi, dv_lo) = split_i64(setup.delta_col_v);
    let (dr_hi, dr_lo) = split_i64(setup.delta_col_r);
    let (dg_hi, dg_lo) = split_i64(setup.delta_col_g);
    let (db_hi, db_lo) = split_i64(setup.delta_col_b);

    Some(ScanlineSetup {
        rows,
        consts: ScanlineConsts {
            y_min: setup.y_min,
            y_max: setup.y_max,
            _pad0: 0,
            _pad1: 0,
            delta_col_u_hi: du_hi,
            delta_col_u_lo: du_lo,
            delta_col_v_hi: dv_hi,
            delta_col_v_lo: dv_lo,
            delta_col_r_hi: dr_hi,
            delta_col_r_lo: dr_lo,
            delta_col_g_hi: dg_hi,
            delta_col_g_lo: dg_lo,
            delta_col_b_hi: db_hi,
            delta_col_b_lo: db_lo,
            _pad2: 0,
            _pad3: 0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_round_trip() {
        for v in [0i64, 1, -1, 0x1234_5678_9ABC, -0x1234_5678_9ABC, i64::MAX, i64::MIN + 1] {
            let (hi, lo) = split_i64(v);
            let reconstructed = ((hi as i64) << 32) | (lo as i64);
            assert_eq!(reconstructed, v, "split_i64({v}) round-trip");
        }
    }

    #[test]
    fn struct_sizes_pinned() {
        assert_eq!(std::mem::size_of::<RowState>(), 64);
        assert_eq!(std::mem::size_of::<ScanlineConsts>(), 64);
    }

    #[test]
    fn build_setup_axis_aligned_right_triangle() {
        // Simple case: (10,10)-(50,10)-(10,50). Top edge at y=10,
        // long edge v0→v2 vertical at x=10 (the "left" side).
        let setup = build_setup(
            [(10, 10), (50, 10), (10, 50)],
            [(0, 0), (32, 0), (0, 32)],
            [(0, 0, 0); 3],
        )
        .expect("setup");
        assert_eq!(setup.consts.y_min, 10);
        assert_eq!(setup.consts.y_max, 49);
        assert_eq!(setup.rows.len(), 40);
        // Row 0 (y=10): left at v0=(10,0), right at v1=(50,0).
        // left_x_q16 = 10 << 16, left_u_q16 = 0.
        let r0 = &setup.rows[0];
        let lx = ((r0.left_x_hi as i64) << 32) | (r0.left_x_lo as i64);
        assert_eq!(lx >> 16, 10);
    }

    #[test]
    fn build_setup_degenerate_triangle_returns_none() {
        // Three collinear points → height = 0 (or longest = 0).
        let zero_h = build_setup(
            [(0, 5), (10, 5), (20, 5)],
            [(0, 0), (0, 0), (0, 0)],
            [(0, 0, 0); 3],
        );
        assert!(zero_h.is_none());
    }
}
