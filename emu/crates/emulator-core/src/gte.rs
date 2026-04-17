//! GTE — Geometry Transformation Engine (COP2).
//!
//! 32 data registers + 32 control registers, all in fixed-point.
//! Vertices and matrix elements are signed 1.3.12 (`i16` × 1/4096);
//! translations and color-bias terms are signed 31.0; intermediate
//! accumulation runs in 64-bit signed with a 44-bit overflow check
//! before truncation to 32-bit `MAC1..3`.
//!
//! Each function call clears `FLAG`, runs its math (which may set the
//! per-result saturation/overflow bits), then folds the error bits
//! into the master `FLAG[31]`. Software polls `FLAG` to detect
//! geometry that would otherwise wrap or clip incorrectly.
//!
//! The division used by `RTPS`/`RTPT` is the documented unsigned
//! Newton-Raphson iteration off a 257-entry seed table — keeping the
//! exact PSX algorithm matters because games consume the resulting
//! `SX2`/`SY2` values directly and any drift moves vertices on screen.
//!
//! Reference: nocash PSX-SPX, section "GTE Coprocessor". Cross-checked
//! against PCSX-Redux's `gte.cc` interpreter.

use core::cmp;

/// One of the 11 documented GTE function opcodes that needs its raw
/// command-word bits decoded for sub-options (`sf`, `lm`, `mx`, `vx`,
/// `cv`). Stored only so call sites read as data, never as parsing.
#[derive(Copy, Clone)]
struct Cmd {
    /// `sf` — fraction shift. `false` = no shift, `true` = shift the
    /// 44-bit MAC result right arithmetically by 12 before truncating.
    sf: bool,
    /// `lm` — IR1..3 saturation lower bound. `false` = -0x8000,
    /// `true` = 0. Used by NCDS / CC / etc. to clamp negative
    /// intermediate components into the unsigned color range.
    lm: bool,
    /// MVMVA matrix selector (bits 18..17). `0` = rotation,
    /// `1` = light, `2` = light-color, `3` = invalid (uses a buggy
    /// hardware path; we emulate it).
    mx: u8,
    /// MVMVA multiplied-vector selector (bits 16..15). `0` = V0,
    /// `1` = V1, `2` = V2, `3` = `[IR1, IR2, IR3]`.
    vx: u8,
    /// MVMVA translation/bias-vector selector (bits 14..13).
    /// `0` = TR, `1` = BK, `2` = FC (buggy), `3` = none.
    cv: u8,
}

impl Cmd {
    fn decode(instr: u32) -> Self {
        Self {
            sf: (instr >> 19) & 1 != 0,
            lm: (instr >> 10) & 1 != 0,
            mx: ((instr >> 17) & 0b11) as u8,
            vx: ((instr >> 15) & 0b11) as u8,
            cv: ((instr >> 13) & 0b11) as u8,
        }
    }

    fn shift(&self) -> u32 {
        if self.sf { 12 } else { 0 }
    }
}

/// FLAG (control register 31) bit positions. Names mirror PSX-SPX.
mod flag {
    pub const IR0_SAT: u32 = 1 << 12;
    pub const SY2_SAT: u32 = 1 << 13;
    pub const SX2_SAT: u32 = 1 << 14;
    pub const MAC0_NEG: u32 = 1 << 15;
    pub const MAC0_POS: u32 = 1 << 16;
    pub const DIV_OVERFLOW: u32 = 1 << 17;
    pub const SZ3_OTZ_SAT: u32 = 1 << 18;
    pub const COLOR_B_SAT: u32 = 1 << 19;
    pub const COLOR_G_SAT: u32 = 1 << 20;
    pub const COLOR_R_SAT: u32 = 1 << 21;
    pub const IR3_SAT: u32 = 1 << 22;
    pub const IR2_SAT: u32 = 1 << 23;
    pub const IR1_SAT: u32 = 1 << 24;
    pub const MAC3_NEG: u32 = 1 << 25;
    pub const MAC2_NEG: u32 = 1 << 26;
    pub const MAC1_NEG: u32 = 1 << 27;
    pub const MAC3_POS: u32 = 1 << 28;
    pub const MAC2_POS: u32 = 1 << 29;
    pub const MAC1_POS: u32 = 1 << 30;
    /// Bits that participate in the FLAG[31] master OR.
    pub const ERROR_MASK: u32 = 0x7F87_E000;
}

/// Full GTE state. Field grouping matches the PSX register map:
/// vectors / FIFOs / accumulators in `data`-prefixed fields, matrices
/// and translations in `ctrl`-prefixed fields. The MFC2/MTC2 paths
/// pack and unpack these into 32-bit views.
pub struct Gte {
    // Data registers ----------------------------------------------------
    /// V0, V1, V2 — input vertex vectors (X, Y, Z), signed 1.3.12.
    v: [[i16; 3]; 3],
    /// RGBC — packed [R, G, B, CODE]. CODE is the texpage / blending
    /// hint; preserved through color-FIFO pushes.
    rgbc: [u8; 4],
    /// OTZ — average Z written by AVSZ3/AVSZ4, unsigned 16-bit.
    otz: u16,
    /// IR0 — scalar accumulator written by INTPL/DPCS/RTPS. Saturates
    /// to 0..0x1000.
    ir0: i16,
    /// IR1, IR2, IR3 — vector accumulators. Saturate to ±0x7FFF
    /// (lm=0) or 0..0x7FFF (lm=1).
    ir: [i16; 3],
    /// SXY FIFO — three slots of (SX, SY) ∈ -1024..1023, plus a
    /// virtual "P" slot at index 3 that aliases SXY2 on read and
    /// rotates the FIFO on write.
    sxy: [[i16; 2]; 3],
    /// SZ FIFO — four slots of unsigned 16-bit Z. New Z values land
    /// at SZ3 and shift the prior contents toward SZ0.
    sz: [u16; 4],
    /// RGB FIFO — three slots of [R, G, B, CODE]. RTPS-family ops
    /// push the latest RGBC through here.
    rgb_fifo: [[u8; 4]; 3],
    /// Reserved word at data-reg 23. Round-trips writes (some games
    /// cache values here knowing the PS1 leaves it untouched).
    res1: u32,
    /// MAC0 — scalar 32-bit accumulator (e.g. perspective math).
    mac0: i32,
    /// MAC1, MAC2, MAC3 — 32-bit truncations of the 44-bit vector
    /// accumulators.
    mac: [i32; 3],
    /// LZCS / LZCR — leading-zero counter input / result. Writing
    /// LZCS recomputes LZCR; reads of LZCR return the cached value.
    lzcs: u32,
    lzcr: u32,

    // Control registers -------------------------------------------------
    /// Rotation matrix RT, signed 1.3.12.
    rotation: [[i16; 3]; 3],
    /// Translation vector TR, signed 31.0.
    translation: [i32; 3],
    /// Light-direction matrix LLM (rows = light directions), signed 1.3.12.
    light: [[i16; 3]; 3],
    /// Background color BK = (RBK, GBK, BBK), signed 19.12. Stored as
    /// 32-bit but only 32 bits round-trip; the documented field width
    /// is "any" — the constraint comes from the math (44-bit MAC).
    bg_color: [i32; 3],
    /// Light-color matrix LCM, signed 1.3.12.
    light_color: [[i16; 3]; 3],
    /// Far color FC = (RFC, GFC, BFC), signed 19.12.
    far_color: [i32; 3],
    /// Screen-offset X, signed 15.16.
    ofx: i32,
    /// Screen-offset Y, signed 15.16.
    ofy: i32,
    /// H — projection plane distance. Unsigned 16-bit; consumed by the
    /// RTPS divisor.
    h: u16,
    /// DQA — depth-cue interpolation coefficient A, signed 7.8.
    dqa: i16,
    /// DQB — depth-cue interpolation bias B, signed 7.24.
    dqb: i32,
    /// ZSF3, ZSF4 — averaging weights for AVSZ3/AVSZ4, signed 0.12.
    zsf3: i16,
    zsf4: i16,
    /// FLAG — error/saturation bits. Bit 31 is the OR of [`flag::ERROR_MASK`].
    flag: u32,
}

impl Gte {
    /// Construct a freshly-reset GTE — all registers cleared. Real
    /// hardware powers on with garbage, but the BIOS zeroes the lot
    /// before first use, so we save a redundant write by starting
    /// clean.
    pub fn new() -> Self {
        Self {
            v: [[0; 3]; 3],
            rgbc: [0; 4],
            otz: 0,
            ir0: 0,
            ir: [0; 3],
            sxy: [[0; 2]; 3],
            sz: [0; 4],
            rgb_fifo: [[0; 4]; 3],
            res1: 0,
            mac0: 0,
            mac: [0; 3],
            lzcs: 0,
            lzcr: 32,
            rotation: [[0; 3]; 3],
            translation: [0; 3],
            light: [[0; 3]; 3],
            bg_color: [0; 3],
            light_color: [[0; 3]; 3],
            far_color: [0; 3],
            ofx: 0,
            ofy: 0,
            h: 0,
            dqa: 0,
            dqb: 0,
            zsf3: 0,
            zsf4: 0,
            flag: 0,
        }
    }

    /// MFC2 view of data register `idx`. Mirrors what the CPU sees on
    /// `mfc2 rt, $rd`.
    pub fn read_data(&self, idx: u8) -> u32 {
        match idx & 31 {
            0 => pack_xy_i16(self.v[0][0], self.v[0][1]),
            1 => sign_extend_16(self.v[0][2]),
            2 => pack_xy_i16(self.v[1][0], self.v[1][1]),
            3 => sign_extend_16(self.v[1][2]),
            4 => pack_xy_i16(self.v[2][0], self.v[2][1]),
            5 => sign_extend_16(self.v[2][2]),
            6 => u32::from_le_bytes(self.rgbc),
            7 => self.otz as u32,
            8 => sign_extend_16(self.ir0),
            9 => sign_extend_16(self.ir[0]),
            10 => sign_extend_16(self.ir[1]),
            11 => sign_extend_16(self.ir[2]),
            12 => pack_xy_i16(self.sxy[0][0], self.sxy[0][1]),
            13 => pack_xy_i16(self.sxy[1][0], self.sxy[1][1]),
            // SXY2 and SXYP both read SXY2 — only their write semantics
            // differ.
            14 | 15 => pack_xy_i16(self.sxy[2][0], self.sxy[2][1]),
            16 => self.sz[0] as u32,
            17 => self.sz[1] as u32,
            18 => self.sz[2] as u32,
            19 => self.sz[3] as u32,
            20 => u32::from_le_bytes(self.rgb_fifo[0]),
            21 => u32::from_le_bytes(self.rgb_fifo[1]),
            22 => u32::from_le_bytes(self.rgb_fifo[2]),
            23 => self.res1,
            24 => self.mac0 as u32,
            25 => self.mac[0] as u32,
            26 => self.mac[1] as u32,
            27 => self.mac[2] as u32,
            // IRGB/ORGB: pack saturated IR1..3 into 5:5:5 (BGR order
            // in the PS1 framebuffer convention).
            28 | 29 => pack_irgb(&self.ir),
            30 => self.lzcs,
            31 => self.lzcr,
            _ => unreachable!(),
        }
    }

    /// MTC2 view of data register `idx`.
    pub fn write_data(&mut self, idx: u8, value: u32) {
        match idx & 31 {
            0 => {
                let (x, y) = unpack_xy_i16(value);
                self.v[0][0] = x;
                self.v[0][1] = y;
            }
            1 => self.v[0][2] = value as i16,
            2 => {
                let (x, y) = unpack_xy_i16(value);
                self.v[1][0] = x;
                self.v[1][1] = y;
            }
            3 => self.v[1][2] = value as i16,
            4 => {
                let (x, y) = unpack_xy_i16(value);
                self.v[2][0] = x;
                self.v[2][1] = y;
            }
            5 => self.v[2][2] = value as i16,
            6 => self.rgbc = value.to_le_bytes(),
            7 => self.otz = value as u16,
            8 => self.ir0 = value as i16,
            9 => self.ir[0] = value as i16,
            10 => self.ir[1] = value as i16,
            11 => self.ir[2] = value as i16,
            12 => {
                let (x, y) = unpack_xy_i16(value);
                self.sxy[0] = [x, y];
            }
            13 => {
                let (x, y) = unpack_xy_i16(value);
                self.sxy[1] = [x, y];
            }
            14 => {
                let (x, y) = unpack_xy_i16(value);
                self.sxy[2] = [x, y];
            }
            // SXYP — push: SXY1 → SXY0, SXY2 → SXY1, value → SXY2.
            15 => {
                let (x, y) = unpack_xy_i16(value);
                self.sxy[0] = self.sxy[1];
                self.sxy[1] = self.sxy[2];
                self.sxy[2] = [x, y];
            }
            16 => self.sz[0] = value as u16,
            17 => self.sz[1] = value as u16,
            18 => self.sz[2] = value as u16,
            19 => self.sz[3] = value as u16,
            20 => self.rgb_fifo[0] = value.to_le_bytes(),
            21 => self.rgb_fifo[1] = value.to_le_bytes(),
            22 => self.rgb_fifo[2] = value.to_le_bytes(),
            23 => self.res1 = value,
            24 => self.mac0 = value as i32,
            25 => self.mac[0] = value as i32,
            26 => self.mac[1] = value as i32,
            27 => self.mac[2] = value as i32,
            // IRGB write — unpack 5:5:5 and replicate into IR1..3 with
            // each component shifted up by 7 (so 5-bit 31 → 0xF80).
            28 => {
                let r = (value & 0x1F) as i16;
                let g = ((value >> 5) & 0x1F) as i16;
                let b = ((value >> 10) & 0x1F) as i16;
                self.ir[0] = r << 7;
                self.ir[1] = g << 7;
                self.ir[2] = b << 7;
            }
            // ORGB is read-only; write is silently dropped.
            29 => {}
            30 => {
                self.lzcs = value;
                self.lzcr = leading_count_signed(value);
            }
            // LZCR is read-only.
            31 => {}
            _ => unreachable!(),
        }
    }

    /// CFC2 view of control register `idx`.
    pub fn read_control(&self, idx: u8) -> u32 {
        match idx & 31 {
            0 => pack_xy_i16(self.rotation[0][0], self.rotation[0][1]),
            1 => pack_xy_i16(self.rotation[0][2], self.rotation[1][0]),
            2 => pack_xy_i16(self.rotation[1][1], self.rotation[1][2]),
            3 => pack_xy_i16(self.rotation[2][0], self.rotation[2][1]),
            // RT33 sits alone in the low halfword, sign-extended.
            4 => sign_extend_16(self.rotation[2][2]),
            5 => self.translation[0] as u32,
            6 => self.translation[1] as u32,
            7 => self.translation[2] as u32,
            8 => pack_xy_i16(self.light[0][0], self.light[0][1]),
            9 => pack_xy_i16(self.light[0][2], self.light[1][0]),
            10 => pack_xy_i16(self.light[1][1], self.light[1][2]),
            11 => pack_xy_i16(self.light[2][0], self.light[2][1]),
            12 => sign_extend_16(self.light[2][2]),
            13 => self.bg_color[0] as u32,
            14 => self.bg_color[1] as u32,
            15 => self.bg_color[2] as u32,
            16 => pack_xy_i16(self.light_color[0][0], self.light_color[0][1]),
            17 => pack_xy_i16(self.light_color[0][2], self.light_color[1][0]),
            18 => pack_xy_i16(self.light_color[1][1], self.light_color[1][2]),
            19 => pack_xy_i16(self.light_color[2][0], self.light_color[2][1]),
            20 => sign_extend_16(self.light_color[2][2]),
            21 => self.far_color[0] as u32,
            22 => self.far_color[1] as u32,
            23 => self.far_color[2] as u32,
            24 => self.ofx as u32,
            25 => self.ofy as u32,
            // H is a hardware quirk: written as unsigned 16-bit but read
            // back **sign-extended** (so writing 0x8000 reads back as
            // 0xFFFF8000). Caught by parity tests against Redux.
            26 => sign_extend_16(self.h as i16),
            27 => sign_extend_16(self.dqa),
            28 => self.dqb as u32,
            29 => sign_extend_16(self.zsf3),
            30 => sign_extend_16(self.zsf4),
            31 => self.flag,
            _ => unreachable!(),
        }
    }

    /// CTC2 view of control register `idx`.
    pub fn write_control(&mut self, idx: u8, value: u32) {
        let (lo, hi) = unpack_xy_i16(value);
        match idx & 31 {
            0 => {
                self.rotation[0][0] = lo;
                self.rotation[0][1] = hi;
            }
            1 => {
                self.rotation[0][2] = lo;
                self.rotation[1][0] = hi;
            }
            2 => {
                self.rotation[1][1] = lo;
                self.rotation[1][2] = hi;
            }
            3 => {
                self.rotation[2][0] = lo;
                self.rotation[2][1] = hi;
            }
            4 => self.rotation[2][2] = value as i16,
            5 => self.translation[0] = value as i32,
            6 => self.translation[1] = value as i32,
            7 => self.translation[2] = value as i32,
            8 => {
                self.light[0][0] = lo;
                self.light[0][1] = hi;
            }
            9 => {
                self.light[0][2] = lo;
                self.light[1][0] = hi;
            }
            10 => {
                self.light[1][1] = lo;
                self.light[1][2] = hi;
            }
            11 => {
                self.light[2][0] = lo;
                self.light[2][1] = hi;
            }
            12 => self.light[2][2] = value as i16,
            13 => self.bg_color[0] = value as i32,
            14 => self.bg_color[1] = value as i32,
            15 => self.bg_color[2] = value as i32,
            16 => {
                self.light_color[0][0] = lo;
                self.light_color[0][1] = hi;
            }
            17 => {
                self.light_color[0][2] = lo;
                self.light_color[1][0] = hi;
            }
            18 => {
                self.light_color[1][1] = lo;
                self.light_color[1][2] = hi;
            }
            19 => {
                self.light_color[2][0] = lo;
                self.light_color[2][1] = hi;
            }
            20 => self.light_color[2][2] = value as i16,
            21 => self.far_color[0] = value as i32,
            22 => self.far_color[1] = value as i32,
            23 => self.far_color[2] = value as i32,
            24 => self.ofx = value as i32,
            25 => self.ofy = value as i32,
            26 => self.h = value as u16,
            27 => self.dqa = value as i16,
            28 => self.dqb = value as i32,
            29 => self.zsf3 = value as i16,
            30 => self.zsf4 = value as i16,
            // FLAG: writes leave the master bit derived from the rest.
            31 => {
                self.flag = value & 0x7FFF_F000;
                self.update_flag_master();
            }
            _ => unreachable!(),
        }
    }

    /// Execute one COP2 function. `instr` is the full 32-bit
    /// instruction word so we can pull `sf`/`lm`/`mx`/`vx`/`cv`.
    /// Unrecognised commands return without updating state — real
    /// hardware decodes them as nops, which matches what we observe
    /// in PCSX-Redux's interpreter.
    pub fn execute(&mut self, instr: u32) {
        let cmd = Cmd::decode(instr);
        let opcode = (instr & 0x3F) as u8;
        self.flag = 0;
        match opcode {
            0x01 => self.op_rtps(cmd, 0, true),
            0x06 => self.op_nclip(),
            0x0C => self.op_op(cmd),
            0x10 => self.op_dpcs(cmd, false),
            0x11 => self.op_intpl(cmd),
            0x12 => self.op_mvmva(cmd),
            0x13 => self.op_ncds(cmd, 0),
            0x14 => self.op_cdp(cmd),
            0x16 => self.op_ncdt(cmd),
            0x1B => self.op_nccs(cmd, 0),
            0x1C => self.op_cc(cmd),
            0x1E => self.op_ncs(cmd, 0),
            0x20 => self.op_nct(cmd),
            0x28 => self.op_sqr(cmd),
            0x29 => self.op_dcpl(cmd),
            0x2A => self.op_dpct(cmd),
            0x2D => self.op_avsz3(),
            0x2E => self.op_avsz4(),
            0x30 => self.op_rtpt(cmd),
            0x3D => self.op_gpf(cmd),
            0x3E => self.op_gpl(cmd),
            0x3F => self.op_ncct(cmd),
            _ => {
                // Real GTE silently ignores undefined commands. We
                // mirror that — the BIOS shouldn't issue any but
                // games occasionally encode trailing instruction words
                // with stray COP2 patterns.
            }
        }
        self.update_flag_master();
    }

    // ------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------

    /// `RTPS` — perspective transformation of `V[idx]`. When `last` is
    /// true, also updates IR0/MAC0 from DQA/DQB (so `RTPT` can call
    /// this for the first two vertices with `last=false`).
    fn op_rtps(&mut self, cmd: Cmd, idx: usize, last: bool) {
        let v = self.v[idx];
        let sf = cmd.shift();

        // [MAC1,MAC2,MAC3] = (TR << 12 + RT * V) >> sf
        let tr = self.translation;
        let rt = self.rotation;
        let mac1 = self.mac_add_row(1, tr[0], &rt[0], &v, sf);
        let mac2 = self.mac_add_row(2, tr[1], &rt[1], &v, sf);
        let mac3 = self.mac_add_row(3, tr[2], &rt[2], &v, sf);

        self.ir[0] = self.saturate_ir(1, mac1, cmd.lm);
        self.ir[1] = self.saturate_ir(2, mac2, cmd.lm);
        // IR3 quirk: the saturation flag is checked against the
        // pre-shift value when sf=0 — same value, but using a
        // different lm choice (always lm=false). The IR3 storage
        // itself uses cmd.lm. Matches Redux.
        let ir3_flag_value = if cmd.sf {
            mac3
        } else {
            (self.mac[2] as i64 >> 12) as i32
        };
        let _ = self.saturate_ir_flag_only(3, ir3_flag_value, false);
        self.ir[2] = self.saturate_value_for_ir(mac3, cmd.lm);

        // Push SZ3 = MAC3 >> ((1-sf)*12), saturated to 0..0xFFFF.
        let sz_value = if cmd.sf {
            self.mac[2]
        } else {
            (self.mac[2] as i64 >> 12) as i32
        };
        self.push_sz(sz_value);

        // Perspective division: divisor = clamp_17((H<<16) / SZ3).
        let divisor = self.unr_divide();

        // SX2 = (divisor * IR1 + OFX) / 0x10000  → clamped to ±0x400.
        let mac0_x = (divisor as i64) * (self.ir[0] as i64) + (self.ofx as i64);
        let mac0_y = (divisor as i64) * (self.ir[1] as i64) + (self.ofy as i64);
        // MAC0 stores the *post*-screen-X computation only briefly;
        // we re-overwrite it with the depth-cue result if last=true.
        let _ = self.check_mac0(mac0_x);
        let sx = self.saturate_screen(mac0_x >> 16, true);
        let _ = self.check_mac0(mac0_y);
        let sy = self.saturate_screen(mac0_y >> 16, false);
        self.push_sxy(sx, sy);

        if last {
            // MAC0 = divisor * DQA + DQB; IR0 = MAC0 / 0x1000 saturated 0..0x1000.
            let mac0 = (divisor as i64) * (self.dqa as i64) + (self.dqb as i64);
            self.mac0 = self.check_mac0(mac0);
            self.ir0 = self.saturate_ir0((mac0 >> 12) as i32);
        } else {
            // For RTPT non-final iterations we still write MAC0 from
            // the screen-Y math so software reading MAC0 between
            // calls sees the post-perspective accumulator. Redux
            // mirrors this.
            self.mac0 = self.check_mac0(mac0_y);
        }
    }

    /// `RTPT` — perspective transform of all three vectors.
    fn op_rtpt(&mut self, cmd: Cmd) {
        self.op_rtps(cmd, 0, false);
        self.op_rtps(cmd, 1, false);
        self.op_rtps(cmd, 2, true);
    }

    /// `NCLIP` — normal clipping. Computes the Z component of the
    /// cross-product `(SXY1 - SXY0) × (SXY2 - SXY0)` to determine
    /// front/back facing.
    ///
    /// `MAC0 = SX0*(SY1-SY2) + SX1*(SY2-SY0) + SX2*(SY0-SY1)`
    fn op_nclip(&mut self) {
        let sx0 = self.sxy[0][0] as i64;
        let sy0 = self.sxy[0][1] as i64;
        let sx1 = self.sxy[1][0] as i64;
        let sy1 = self.sxy[1][1] as i64;
        let sx2 = self.sxy[2][0] as i64;
        let sy2 = self.sxy[2][1] as i64;
        let result = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
        self.mac0 = self.check_mac0(result);
    }

    /// `OP` — outer product of IR vector with the diagonal of the
    /// rotation matrix. Produces a vector cross-product variant used
    /// for normal generation.
    fn op_op(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let d1 = self.rotation[0][0] as i64;
        let d2 = self.rotation[1][1] as i64;
        let d3 = self.rotation[2][2] as i64;
        let ir1 = self.ir[0] as i64;
        let ir2 = self.ir[1] as i64;
        let ir3 = self.ir[2] as i64;
        let mac1 = self.check_mac(1, (ir3 * d2) - (ir2 * d3)) >> sf;
        let mac2 = self.check_mac(2, (ir1 * d3) - (ir3 * d1)) >> sf;
        let mac3 = self.check_mac(3, (ir2 * d1) - (ir1 * d2)) >> sf;
        self.mac[0] = mac1 as i32;
        self.mac[1] = mac2 as i32;
        self.mac[2] = mac3 as i32;
        self.ir[0] = self.saturate_ir(1, self.mac[0], cmd.lm);
        self.ir[1] = self.saturate_ir(2, self.mac[1], cmd.lm);
        self.ir[2] = self.saturate_ir(3, self.mac[2], cmd.lm);
    }

    /// `MVMVA` — multiply matrix by vector and add translation. Both
    /// the matrix, the vector, and the translation are selected by
    /// command bits.
    fn op_mvmva(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let mx = self.select_matrix(cmd.mx);
        let v = self.select_vector(cmd.vx);
        let tr = self.select_translation(cmd.cv);

        // The cv=2 (FC) path is famously buggy: each row's TR term
        // wipes the matrix product before the IR clamp because the
        // hardware drops the matrix-times-V contribution into the
        // wrong adder slot. We emulate by computing the FC + matrix-
        // first-column product, clamping to IR (with lm=false), then
        // doing the rest of the multiply with TR=0.
        if cmd.cv == 2 {
            // Step 1: MAC = (FC << 12) + (MX_col0 * V_x)
            //         IR = saturate(MAC, lm=false)  [flags only]
            // Step 2: MAC = (MX * V), shifted; IR comes from MAC with
            //         the user's lm. Final saturation flags reflect
            //         step 2.
            for i in 0..3 {
                let bias = (tr[i] as i64) << 12;
                let prod = (mx[i][0] as i64) * (v[0] as i64);
                let stage1 = self.check_mac((i + 1) as u8, bias + prod) >> sf;
                let _ = self.saturate_ir_flag_only((i + 1) as u8, stage1 as i32, false);
            }
            for i in 0..3 {
                let prod = (mx[i][0] as i64) * (v[0] as i64)
                    + (mx[i][1] as i64) * (v[1] as i64)
                    + (mx[i][2] as i64) * (v[2] as i64);
                let mac = self.check_mac((i + 1) as u8, prod) >> sf;
                self.mac[i] = mac as i32;
                self.ir[i] = self.saturate_value_for_ir(mac as i32, cmd.lm);
            }
            return;
        }

        for i in 0..3 {
            let bias = (tr[i] as i64) << 12;
            let prod = (mx[i][0] as i64) * (v[0] as i64)
                + (mx[i][1] as i64) * (v[1] as i64)
                + (mx[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
    }

    /// `SQR` — square the IR vector. `MAC = IR * IR`, IR = saturate(MAC).
    fn op_sqr(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        for i in 0..3 {
            let v = self.ir[i] as i64;
            let mac = self.check_mac((i + 1) as u8, v * v) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
    }

    /// `AVSZ3` — average of three Z values in the FIFO. `OTZ = ZSF3 *
    /// (SZ1 + SZ2 + SZ3) >> 12`, saturated to 0..0xFFFF.
    fn op_avsz3(&mut self) {
        let sum = (self.sz[1] as i64) + (self.sz[2] as i64) + (self.sz[3] as i64);
        let mac0 = (self.zsf3 as i64) * sum;
        self.mac0 = self.check_mac0(mac0);
        self.otz = self.saturate_otz(mac0 >> 12);
    }

    /// `AVSZ4` — average of all four Z values. Same shape as AVSZ3.
    fn op_avsz4(&mut self) {
        let sum = (self.sz[0] as i64)
            + (self.sz[1] as i64)
            + (self.sz[2] as i64)
            + (self.sz[3] as i64);
        let mac0 = (self.zsf4 as i64) * sum;
        self.mac0 = self.check_mac0(mac0);
        self.otz = self.saturate_otz(mac0 >> 12);
    }

    /// `DPCS` — depth-cue colour single. Interpolates the current RGBC
    /// toward the far colour using IR0 as the blend factor. `is_dpct`
    /// distinguishes the variant that pushes RGB FIFO state instead of
    /// reading from RGBC.
    fn op_dpcs(&mut self, cmd: Cmd, is_dpct: bool) {
        let (r, g, b) = if is_dpct {
            (
                self.rgb_fifo[0][0],
                self.rgb_fifo[0][1],
                self.rgb_fifo[0][2],
            )
        } else {
            (self.rgbc[0], self.rgbc[1], self.rgbc[2])
        };
        let sf = cmd.shift();
        // [MAC1,2,3] = [R,G,B,...] << 16  +  IR0 * (limE(FC - [R,G,B] << 16))
        // Per PSX-SPX, the difference is computed in 12-bit fractional
        // intermediate via IR clamp (lm=false). We follow that.
        let bases = [r as i64, g as i64, b as i64];
        let fc = self.far_color;
        for i in 0..3 {
            let base = bases[i] << 16;
            let diff = fc[i] as i64 - base;
            let temp = self.check_mac((i + 1) as u8, diff) >> sf;
            let ir = self.saturate_value_for_ir(temp as i32, false);
            let _ = self.saturate_ir_flag_only((i + 1) as u8, temp as i32, false);
            let combined = self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64))
                >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `DPCT` — depth cue triple: DPCS run three times against the
    /// RGB FIFO.
    fn op_dpct(&mut self, cmd: Cmd) {
        for _ in 0..3 {
            self.op_dpcs(cmd, true);
        }
    }

    /// `INTPL` — interpolate IR vector toward FC by IR0.
    fn op_intpl(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let fc = self.far_color;
        let initial_ir = self.ir;
        for i in 0..3 {
            let base = (initial_ir[i] as i64) << 12;
            let diff = (fc[i] as i64) - base;
            let temp = self.check_mac((i + 1) as u8, diff) >> sf;
            let _ = self.saturate_ir_flag_only((i + 1) as u8, temp as i32, false);
            let ir = self.saturate_value_for_ir(temp as i32, false);
            let combined = self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64))
                >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `NCS` — normal colour single. Lights `V[idx]` against the LLM,
    /// then colours via LCM.
    fn op_ncs(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        // [IR1,IR2,IR3] = LLM * V  (with light bg color bias, sf shifted)
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        // [IR1,IR2,IR3] = LCM * IR  +  bg_color << 12
        let n = self.ir;
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light_color[i][0] as i64) * (n[0] as i64)
                + (self.light_color[i][1] as i64) * (n[1] as i64)
                + (self.light_color[i][2] as i64) * (n[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        // [MAC1,2,3] = ([R,G,B] << 4) * IR
        let r = self.rgbc[0] as i64;
        let g = self.rgbc[1] as i64;
        let b = self.rgbc[2] as i64;
        let cs = [r << 4, g << 4, b << 4];
        for i in 0..3 {
            let mac = self.check_mac((i + 1) as u8, cs[i] * (self.ir[i] as i64)) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `NCT` — NCS for V0, V1, V2.
    fn op_nct(&mut self, cmd: Cmd) {
        for i in 0..3 {
            self.op_ncs(cmd, i);
        }
    }

    /// `NCDS` — normal colour depth-cue single. Like NCS but with the
    /// final colour interpolated toward FC by IR0.
    fn op_ncds(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        let n = self.ir;
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light_color[i][0] as i64) * (n[0] as i64)
                + (self.light_color[i][1] as i64) * (n[1] as i64)
                + (self.light_color[i][2] as i64) * (n[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        // Interpolate toward FC.
        let cs = [
            (self.rgbc[0] as i64) << 4,
            (self.rgbc[1] as i64) << 4,
            (self.rgbc[2] as i64) << 4,
        ];
        for i in 0..3 {
            let base = cs[i] * (self.ir[i] as i64);
            let diff = ((self.far_color[i] as i64) << 12) - base;
            let temp = self.check_mac((i + 1) as u8, diff) >> sf;
            let _ = self.saturate_ir_flag_only((i + 1) as u8, temp as i32, false);
            let ir = self.saturate_value_for_ir(temp as i32, false);
            let combined =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64)) >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `NCDT` — NCDS for V0, V1, V2.
    fn op_ncdt(&mut self, cmd: Cmd) {
        for i in 0..3 {
            self.op_ncds(cmd, i);
        }
    }

    /// `NCCS` — normal colour single (no depth cue, but colour multiplied
    /// against the input RGBC).
    fn op_nccs(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        let n = self.ir;
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light_color[i][0] as i64) * (n[0] as i64)
                + (self.light_color[i][1] as i64) * (n[1] as i64)
                + (self.light_color[i][2] as i64) * (n[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        let cs = [
            (self.rgbc[0] as i64) << 4,
            (self.rgbc[1] as i64) << 4,
            (self.rgbc[2] as i64) << 4,
        ];
        for i in 0..3 {
            let mac = self.check_mac((i + 1) as u8, cs[i] * (self.ir[i] as i64)) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    fn op_ncct(&mut self, cmd: Cmd) {
        for i in 0..3 {
            self.op_nccs(cmd, i);
        }
    }

    /// `CC` — colour-colour: blend RGBC against IR using the LCM and
    /// background-colour bias.
    fn op_cc(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let n = self.ir;
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light_color[i][0] as i64) * (n[0] as i64)
                + (self.light_color[i][1] as i64) * (n[1] as i64)
                + (self.light_color[i][2] as i64) * (n[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        let cs = [
            (self.rgbc[0] as i64) << 4,
            (self.rgbc[1] as i64) << 4,
            (self.rgbc[2] as i64) << 4,
        ];
        for i in 0..3 {
            let mac = self.check_mac((i + 1) as u8, cs[i] * (self.ir[i] as i64)) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `CDP` — colour depth-queue: same as CC but with FC interpolation.
    fn op_cdp(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let n = self.ir;
        for i in 0..3 {
            let bias = (self.bg_color[i] as i64) << 12;
            let prod = (self.light_color[i][0] as i64) * (n[0] as i64)
                + (self.light_color[i][1] as i64) * (n[1] as i64)
                + (self.light_color[i][2] as i64) * (n[2] as i64);
            let mac = self.check_mac((i + 1) as u8, bias + prod) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        let cs = [
            (self.rgbc[0] as i64) << 4,
            (self.rgbc[1] as i64) << 4,
            (self.rgbc[2] as i64) << 4,
        ];
        for i in 0..3 {
            let base = cs[i] * (self.ir[i] as i64);
            let diff = ((self.far_color[i] as i64) << 12) - base;
            let temp = self.check_mac((i + 1) as u8, diff) >> sf;
            let _ = self.saturate_ir_flag_only((i + 1) as u8, temp as i32, false);
            let ir = self.saturate_value_for_ir(temp as i32, false);
            let combined =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64)) >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `DCPL` — depth-cue color light. Interpolate `RGBC * IR` toward
    /// FC by IR0, with no LCM stage.
    fn op_dcpl(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        let cs = [
            (self.rgbc[0] as i64) << 4,
            (self.rgbc[1] as i64) << 4,
            (self.rgbc[2] as i64) << 4,
        ];
        let ir = self.ir;
        for i in 0..3 {
            let base = cs[i] * (ir[i] as i64);
            let diff = ((self.far_color[i] as i64) << 12) - base;
            let temp = self.check_mac((i + 1) as u8, diff) >> sf;
            let _ = self.saturate_ir_flag_only((i + 1) as u8, temp as i32, false);
            let irc = self.saturate_value_for_ir(temp as i32, false);
            let combined =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (irc as i64)) >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `GPF` — general-purpose interpolation. `MAC = IR * IR0`, then
    /// IR/RGB push.
    fn op_gpf(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        for i in 0..3 {
            let mac =
                self.check_mac((i + 1) as u8, (self.ir0 as i64) * (self.ir[i] as i64)) >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `GPL` — general-purpose interpolation with base. `MAC = MAC + IR * IR0`.
    fn op_gpl(&mut self, cmd: Cmd) {
        let sf = cmd.shift();
        for i in 0..3 {
            // MAC base is shifted left by sf before addition (so the
            // pre-existing MAC value is treated as a fixed-point with
            // the same scaling as the new product).
            let base = (self.mac[i] as i64) << sf;
            let mac =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (self.ir[i] as i64))
                    >> sf;
            self.mac[i] = mac as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, mac as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Compute one row of `(translation << 12) + matrix * vector`, with
    /// the 44-bit overflow check applied, then arithmetic-shift the
    /// 64-bit result right by `sf` and store the truncated 32-bit form
    /// into `MAC[idx]`.
    fn mac_add_row(&mut self, idx: u8, tr: i32, row: &[i16; 3], v: &[i16; 3], sf: u32) -> i32 {
        let bias = (tr as i64) << 12;
        let prod = (row[0] as i64) * (v[0] as i64)
            + (row[1] as i64) * (v[1] as i64)
            + (row[2] as i64) * (v[2] as i64);
        let checked = self.check_mac(idx, bias + prod);
        let shifted = checked >> sf;
        self.mac[(idx - 1) as usize] = shifted as i32;
        shifted as i32
    }

    /// Apply the 44-bit signed overflow check, returning the value
    /// unchanged. Sets the appropriate `FLAG` bit on overflow.
    fn check_mac(&mut self, idx: u8, value: i64) -> i64 {
        let pos_limit = (1i64 << 43) - 1;
        let neg_limit = -(1i64 << 43);
        if value > pos_limit {
            self.flag |= match idx {
                1 => flag::MAC1_POS,
                2 => flag::MAC2_POS,
                3 => flag::MAC3_POS,
                _ => 0,
            };
        }
        if value < neg_limit {
            self.flag |= match idx {
                1 => flag::MAC1_NEG,
                2 => flag::MAC2_NEG,
                3 => flag::MAC3_NEG,
                _ => 0,
            };
        }
        value
    }

    /// MAC0 32-bit overflow check + truncation.
    fn check_mac0(&mut self, value: i64) -> i32 {
        if value > i32::MAX as i64 {
            self.flag |= flag::MAC0_POS;
        }
        if value < i32::MIN as i64 {
            self.flag |= flag::MAC0_NEG;
        }
        value as i32
    }

    /// Saturate a value to the IR1/IR2/IR3 range and store it. Sets
    /// the corresponding `FLAG` bit on saturation.
    fn saturate_ir(&mut self, idx: u8, value: i32, lm: bool) -> i16 {
        let lo = if lm { 0 } else { -0x8000 };
        let hi = 0x7FFFi32;
        let sat = if value > hi {
            self.set_ir_flag(idx);
            hi
        } else if value < lo {
            self.set_ir_flag(idx);
            lo
        } else {
            value
        };
        sat as i16
    }

    /// Like [`saturate_ir`] but only sets the flag — does not store.
    /// Used when the flag for one IR slot needs to reflect a different
    /// value than the one stored (RTPS IR3 quirk, MVMVA-FC quirk).
    fn saturate_ir_flag_only(&mut self, idx: u8, value: i32, lm: bool) -> i16 {
        let lo = if lm { 0 } else { -0x8000 };
        let hi = 0x7FFFi32;
        if value > hi || value < lo {
            self.set_ir_flag(idx);
        }
        value.clamp(lo, hi) as i16
    }

    /// Saturate-to-IR without touching the FLAG. Used when the flag was
    /// already set by [`saturate_ir_flag_only`].
    fn saturate_value_for_ir(&self, value: i32, lm: bool) -> i16 {
        let lo = if lm { 0 } else { -0x8000 };
        value.clamp(lo, 0x7FFF) as i16
    }

    fn set_ir_flag(&mut self, idx: u8) {
        self.flag |= match idx {
            1 => flag::IR1_SAT,
            2 => flag::IR2_SAT,
            3 => flag::IR3_SAT,
            _ => 0,
        };
    }

    fn saturate_ir0(&mut self, value: i32) -> i16 {
        if value < 0 {
            self.flag |= flag::IR0_SAT;
            0
        } else if value > 0x1000 {
            self.flag |= flag::IR0_SAT;
            0x1000
        } else {
            value as i16
        }
    }

    /// Saturate `value` to -1024..1023 and set the appropriate screen
    /// flag (`is_x` chooses SX2 vs SY2).
    fn saturate_screen(&mut self, value: i64, is_x: bool) -> i16 {
        let lo = -0x400i64;
        let hi = 0x3FFi64;
        let bit = if is_x { flag::SX2_SAT } else { flag::SY2_SAT };
        if value > hi {
            self.flag |= bit;
            hi as i16
        } else if value < lo {
            self.flag |= bit;
            lo as i16
        } else {
            value as i16
        }
    }

    fn saturate_otz(&mut self, value: i64) -> u16 {
        if value < 0 {
            self.flag |= flag::SZ3_OTZ_SAT;
            0
        } else if value > 0xFFFF {
            self.flag |= flag::SZ3_OTZ_SAT;
            0xFFFF
        } else {
            value as u16
        }
    }

    /// Push a Z value onto SZ FIFO with 0..0xFFFF saturation.
    fn push_sz(&mut self, value: i32) {
        let z = if value < 0 {
            self.flag |= flag::SZ3_OTZ_SAT;
            0
        } else if value > 0xFFFF {
            self.flag |= flag::SZ3_OTZ_SAT;
            0xFFFF
        } else {
            value as u16
        };
        self.sz[0] = self.sz[1];
        self.sz[1] = self.sz[2];
        self.sz[2] = self.sz[3];
        self.sz[3] = z;
    }

    /// Push an XY pair onto the SXY FIFO.
    fn push_sxy(&mut self, x: i16, y: i16) {
        self.sxy[0] = self.sxy[1];
        self.sxy[1] = self.sxy[2];
        self.sxy[2] = [x, y];
    }

    /// Pull MAC1/2/3 into a saturated RGB and push onto the colour FIFO.
    /// `code` is taken from the current RGBC.
    fn push_color_from_mac(&mut self, _cmd: Cmd) {
        let r = self.saturate_color((self.mac[0] >> 4) as i32, flag::COLOR_R_SAT);
        let g = self.saturate_color((self.mac[1] >> 4) as i32, flag::COLOR_G_SAT);
        let b = self.saturate_color((self.mac[2] >> 4) as i32, flag::COLOR_B_SAT);
        let code = self.rgbc[3];
        self.rgb_fifo[0] = self.rgb_fifo[1];
        self.rgb_fifo[1] = self.rgb_fifo[2];
        self.rgb_fifo[2] = [r, g, b, code];
    }

    fn saturate_color(&mut self, value: i32, flag_bit: u32) -> u8 {
        if value < 0 {
            self.flag |= flag_bit;
            0
        } else if value > 0xFF {
            self.flag |= flag_bit;
            0xFF
        } else {
            value as u8
        }
    }

    /// Newton-Raphson divide: returns `clamp((H<<16) / SZ3, 0..0x1FFFF)`.
    /// Sets [`flag::DIV_OVERFLOW`] if `SZ3 == 0` or `H >= 2*SZ3`.
    fn unr_divide(&mut self) -> u32 {
        let h = self.h as u32;
        let sz3 = self.sz[3] as u32;
        if h >= sz3 * 2 {
            self.flag |= flag::DIV_OVERFLOW;
            return 0x1FFFF;
        }
        let z = (sz3 as u16).leading_zeros();
        let n = h << z;
        let d = sz3 << z;
        let table_index = ((d.wrapping_sub(0x7FC0)) >> 7) as usize & 0xFF;
        let u = (UNR_TABLE[table_index] as u32) + 0x101;
        let d = (0x2000080u32.wrapping_sub(d.wrapping_mul(u))) >> 8;
        let d = (0x80u32.wrapping_add(d.wrapping_mul(u))) >> 8;
        let result = (((n as u64) * (d as u64)) + 0x8000) >> 16;
        cmp::min(0x1FFFF, result as u32)
    }

    /// Recompute the master `FLAG[31]` bit from the error-mask OR.
    fn update_flag_master(&mut self) {
        if self.flag & flag::ERROR_MASK != 0 {
            self.flag |= 1 << 31;
        }
    }

    fn select_matrix(&self, mx: u8) -> [[i16; 3]; 3] {
        match mx {
            0 => self.rotation,
            1 => self.light,
            2 => self.light_color,
            // mx=3 is the "garbage matrix" — uses RGB-from-RGBC components
            // mixed with rotation diagonal. Implementation matches Redux.
            _ => {
                let r = (self.rgbc[0] as i16) << 4;
                [
                    [-r, r, self.ir[0]],
                    [self.rotation[0][2], self.rotation[0][2], self.rotation[0][2]],
                    [self.rotation[1][1], self.rotation[1][1], self.rotation[1][1]],
                ]
            }
        }
    }

    fn select_vector(&self, vx: u8) -> [i16; 3] {
        match vx {
            0 => self.v[0],
            1 => self.v[1],
            2 => self.v[2],
            _ => self.ir,
        }
    }

    fn select_translation(&self, cv: u8) -> [i32; 3] {
        match cv {
            0 => self.translation,
            1 => self.bg_color,
            2 => self.far_color,
            _ => [0, 0, 0],
        }
    }
}

impl Default for Gte {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Bit-packing helpers
// ---------------------------------------------------------------------

fn pack_xy_i16(x: i16, y: i16) -> u32 {
    ((y as u16 as u32) << 16) | (x as u16 as u32)
}

/// Unpack the low/high halves of a packed XY word into two `i16`s.
fn unpack_xy_i16(value: u32) -> (i16, i16) {
    (value as i16, (value >> 16) as i16)
}

fn sign_extend_16(value: i16) -> u32 {
    value as i32 as u32
}

fn pack_irgb(ir: &[i16; 3]) -> u32 {
    let r = ir[0].clamp(0, 0xF80) >> 7;
    let g = ir[1].clamp(0, 0xF80) >> 7;
    let b = ir[2].clamp(0, 0xF80) >> 7;
    ((b as u32) << 10) | ((g as u32) << 5) | (r as u32)
}

/// LZCR algorithm: count the run of leading bits matching the sign bit.
/// For positive values that's leading zeros; for negative values it's
/// leading ones. The returned count is in 1..=32.
fn leading_count_signed(value: u32) -> u32 {
    let test = if (value as i32) < 0 { !value } else { value };
    if test == 0 { 32 } else { test.leading_zeros() }
}

// ---------------------------------------------------------------------
// UNR division seed table (PSX-SPX section "GTE Division Inaccuracy")
// ---------------------------------------------------------------------

/// 257-entry seed for the Newton-Raphson divider. Indexed by
/// `(d - 0x7FC0) >> 7` after `d` has been left-aligned. The final
/// entry is required so the index can saturate without bounds checks.
#[rustfmt::skip]
static UNR_TABLE: [u8; 257] = [
    0xFF,0xFD,0xFB,0xF9,0xF7,0xF5,0xF3,0xF1, 0xEF,0xEE,0xEC,0xEA,0xE8,0xE6,0xE4,0xE3,
    0xE1,0xDF,0xDD,0xDC,0xDA,0xD8,0xD6,0xD5, 0xD3,0xD1,0xD0,0xCE,0xCD,0xCB,0xC9,0xC8,
    0xC6,0xC5,0xC3,0xC1,0xC0,0xBE,0xBD,0xBB, 0xBA,0xB8,0xB7,0xB5,0xB4,0xB2,0xB1,0xB0,
    0xAE,0xAD,0xAB,0xAA,0xA9,0xA7,0xA6,0xA4, 0xA3,0xA2,0xA0,0x9F,0x9E,0x9C,0x9B,0x9A,
    0x99,0x97,0x96,0x95,0x94,0x92,0x91,0x90, 0x8F,0x8D,0x8C,0x8B,0x8A,0x89,0x87,0x86,
    0x85,0x84,0x83,0x82,0x81,0x7F,0x7E,0x7D, 0x7C,0x7B,0x7A,0x79,0x78,0x77,0x75,0x74,
    0x73,0x72,0x71,0x70,0x6F,0x6E,0x6D,0x6C, 0x6B,0x6A,0x69,0x68,0x67,0x66,0x65,0x64,
    0x63,0x62,0x61,0x60,0x5F,0x5E,0x5D,0x5D, 0x5C,0x5B,0x5A,0x59,0x58,0x57,0x56,0x55,
    0x54,0x53,0x53,0x52,0x51,0x50,0x4F,0x4E, 0x4D,0x4D,0x4C,0x4B,0x4A,0x49,0x48,0x48,
    0x47,0x46,0x45,0x44,0x43,0x43,0x42,0x41, 0x40,0x3F,0x3F,0x3E,0x3D,0x3C,0x3C,0x3B,
    0x3A,0x39,0x39,0x38,0x37,0x36,0x36,0x35, 0x34,0x33,0x33,0x32,0x31,0x31,0x30,0x2F,
    0x2E,0x2E,0x2D,0x2C,0x2C,0x2B,0x2A,0x2A, 0x29,0x28,0x28,0x27,0x26,0x26,0x25,0x24,
    0x24,0x23,0x22,0x22,0x21,0x20,0x20,0x1F, 0x1E,0x1E,0x1D,0x1D,0x1C,0x1B,0x1B,0x1A,
    0x19,0x19,0x18,0x18,0x17,0x16,0x16,0x15, 0x15,0x14,0x14,0x13,0x12,0x12,0x11,0x11,
    0x10,0x0F,0x0F,0x0E,0x0E,0x0D,0x0D,0x0C, 0x0C,0x0B,0x0A,0x0A,0x09,0x09,0x08,0x08,
    0x07,0x07,0x06,0x06,0x05,0x05,0x04,0x04, 0x03,0x03,0x02,0x02,0x01,0x01,0x00,0x00,
    0x00,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_state_is_zero() {
        let g = Gte::new();
        for i in 0..32 {
            // LZCR seeds to 32 — the leading-zero count of the zeroed
            // LZCS input is "all 32 bits are leading zeros".
            let expected = if i == 31 { 32 } else { 0 };
            assert_eq!(g.read_data(i), expected, "data reg {i}");
        }
        for i in 0..31 {
            assert_eq!(g.read_control(i), 0, "ctrl reg {i} should reset to 0");
        }
    }

    #[test]
    fn data_reg_v0_xy_round_trip() {
        let mut g = Gte::new();
        g.write_data(0, 0xDEAD_BEEF);
        // Stored as two i16; reads back the same packed word.
        assert_eq!(g.read_data(0), 0xDEAD_BEEF);
    }

    #[test]
    fn data_reg_vz0_sign_extends_on_read() {
        let mut g = Gte::new();
        g.write_data(1, 0x0000_8000);
        assert_eq!(g.read_data(1), 0xFFFF_8000);
    }

    #[test]
    fn sxyp_write_pushes_fifo() {
        let mut g = Gte::new();
        g.write_data(12, pack_xy_i16(1, 2)); // SXY0
        g.write_data(13, pack_xy_i16(3, 4)); // SXY1
        g.write_data(14, pack_xy_i16(5, 6)); // SXY2
        g.write_data(15, pack_xy_i16(7, 8)); // SXYP — pushes
        assert_eq!(g.read_data(12), pack_xy_i16(3, 4));
        assert_eq!(g.read_data(13), pack_xy_i16(5, 6));
        assert_eq!(g.read_data(14), pack_xy_i16(7, 8));
    }

    #[test]
    fn lzcs_write_recomputes_lzcr() {
        let mut g = Gte::new();
        g.write_data(30, 0x0000_FFFF);
        assert_eq!(g.read_data(31), 16);
        g.write_data(30, 0xFFFF_0000);
        // Negative — count of leading 1-bits.
        assert_eq!(g.read_data(31), 16);
        g.write_data(30, 0);
        assert_eq!(g.read_data(31), 32);
        g.write_data(30, !0);
        assert_eq!(g.read_data(31), 32);
    }

    #[test]
    fn ctrl_h_reads_back_sign_extended() {
        let mut g = Gte::new();
        g.write_control(26, 0x0000_8000);
        assert_eq!(g.read_control(26), 0xFFFF_8000);
    }

    #[test]
    fn flag_reset_at_start_of_each_op() {
        let mut g = Gte::new();
        // Force a stale flag.
        g.write_control(31, flag::MAC1_POS);
        assert_eq!(g.read_control(31) & flag::MAC1_POS, flag::MAC1_POS);
        // NCLIP clears + recomputes; with all zeros it produces 0.
        g.execute(0x06); // NCLIP, sf=0
        assert_eq!(g.read_control(31), 0);
    }

    #[test]
    fn rtps_identity_transform_against_origin() {
        // With RT = identity and TR = 0, RTPS of V0=(0,0,0) should
        // produce IR = 0, MAC = 0, push (0,0,0) onto SZ FIFO. SZ3 = 0
        // → divide overflow flag set, SX2/SY2 saturate.
        let mut g = Gte::new();
        // Identity rotation, scaled by 0x1000 (1.0 in 1.3.12).
        g.write_control(0, pack_xy_i16(0x1000, 0));
        g.write_control(1, pack_xy_i16(0, 0x1000));
        g.write_control(2, pack_xy_i16(0, 0));
        g.write_control(3, pack_xy_i16(0, 0));
        g.write_control(4, 0x1000);
        // V0 = (0, 0, 0)
        g.execute(0x0180_0001); // RTPS sf=1
        // Flag should record divide overflow (sz3=0, h=0 still triggers).
        assert!(g.read_control(31) & flag::DIV_OVERFLOW != 0);
    }

    #[test]
    fn nclip_computes_z_cross_product() {
        // SXY0 = (0,0), SXY1 = (10,0), SXY2 = (0,10):
        // result = 0*(0-10) + 10*(10-0) + 0*(0-0) = 100
        let mut g = Gte::new();
        g.write_data(12, pack_xy_i16(0, 0));
        g.write_data(13, pack_xy_i16(10, 0));
        g.write_data(14, pack_xy_i16(0, 10));
        g.execute(0x06); // NCLIP
        assert_eq!(g.read_data(24) as i32, 100);
    }

    #[test]
    fn avsz3_averages_with_zsf3() {
        let mut g = Gte::new();
        g.write_data(17, 0x0100); // SZ1
        g.write_data(18, 0x0200); // SZ2
        g.write_data(19, 0x0300); // SZ3
        // ZSF3 = 0x555 — close enough to 0x1000/3 to land OTZ near the
        // simple arithmetic mean.
        g.write_control(29, 0x0555);
        g.execute(0x2D); // AVSZ3
        // MAC0 = 0x555 * (0x100 + 0x200 + 0x300) = 0x555 * 0x600 = 0x1FFE00.
        // OTZ = 0x1FFE00 >> 12 = 0x1FF.
        assert_eq!(g.read_data(24) as i32, 0x1FFE00);
        assert_eq!(g.read_data(7), 0x1FF);
    }

    #[test]
    fn sqr_squares_ir() {
        let mut g = Gte::new();
        g.write_data(9, 0x10);
        g.write_data(10, 0x20);
        g.write_data(11, 0x30);
        g.execute(0x28); // SQR sf=0
        assert_eq!(g.read_data(25), 0x100);
        assert_eq!(g.read_data(26), 0x400);
        assert_eq!(g.read_data(27), 0x900);
    }

    #[test]
    fn op_decode_extracts_command_word_fields() {
        // sf=1, lm=1, mx=2, vx=1, cv=0, opcode=0x12 → instr =
        //   (1<<19) | (1<<10) | (2<<17) | (1<<15) | (0<<13) | 0x12
        let instr = (1u32 << 19) | (1 << 10) | (2 << 17) | (1 << 15) | 0x12;
        let cmd = Cmd::decode(instr);
        assert_eq!(cmd.sf, true);
        assert_eq!(cmd.lm, true);
        assert_eq!(cmd.mx, 2);
        assert_eq!(cmd.vx, 1);
        assert_eq!(cmd.cv, 0);
    }

    #[test]
    fn unknown_function_is_silently_ignored() {
        let mut g = Gte::new();
        g.execute(0x00); // not a valid GTE op
        assert_eq!(g.read_control(31), 0);
    }
}
