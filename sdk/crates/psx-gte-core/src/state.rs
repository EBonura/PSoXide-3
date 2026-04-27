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
        if self.sf {
            12
        } else {
            0
        }
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
    /// At reset both are zero — LZCR is *not* eagerly seeded to 32
    /// (the canonical lzcnt of LZCS=0). Real software always writes
    /// LZCS before reading LZCR, but the parity oracle (PCSX-Redux)
    /// snapshots raw register storage which is `memset(0)` on boot,
    /// so we must match that to keep step-0 traces identical.
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
    /// Diagnostic command counter. This deliberately does not drive
    /// timing yet; it lets frontends report GTE pressure separately
    /// from the current CPU/bus cycle model.
    profile_ops: u64,
    /// Sum of documented GTE command latencies for recognised ops.
    /// Real hardware can hide some of this behind independent CPU
    /// work, so consumers should treat it as internal GTE load rather
    /// than extra bus cycles.
    profile_estimated_cycles: u64,
    /// Per-opcode diagnostic counts, indexed by the low 6 command bits.
    profile_opcode_counts: [u64; 64],
}

/// Monotonic diagnostic counters for GTE command pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GteProfileSnapshot {
    /// Recognised GTE function commands executed since reset.
    pub ops: u64,
    /// Sum of estimated internal GTE cycles since reset.
    pub estimated_cycles: u64,
    /// Per-opcode command counts, indexed by the low 6 command bits.
    pub opcode_counts: [u64; 64],
}

/// Documented command latencies from the public PSX GTE tables. The
/// profiler reports these as internal GTE load only; the emulator's
/// bus cycle counter remains the source of truth for elapsed guest time.
fn gte_command_cycles(opcode: u8) -> Option<u8> {
    match opcode {
        0x01 => Some(15), // RTPS
        0x06 => Some(8),  // NCLIP
        0x0C => Some(6),  // OP
        0x10 => Some(8),  // DPCS
        0x11 => Some(8),  // INTPL
        0x12 => Some(8),  // MVMVA
        0x13 => Some(19), // NCDS
        0x14 => Some(13), // CDP
        0x16 => Some(44), // NCDT
        0x1B => Some(17), // NCCS
        0x1C => Some(11), // CC
        0x1E => Some(14), // NCS
        0x20 => Some(30), // NCT
        0x28 => Some(5),  // SQR
        0x29 => Some(8),  // DCPL
        0x2A => Some(17), // DPCT
        0x2D => Some(5),  // AVSZ3
        0x2E => Some(6),  // AVSZ4
        0x30 => Some(23), // RTPT
        0x3D => Some(5),  // GPF
        0x3E => Some(5),  // GPL
        0x3F => Some(39), // NCCT
        _ => None,
    }
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
            lzcr: 0,
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
            profile_ops: 0,
            profile_estimated_cycles: 0,
            profile_opcode_counts: [0; 64],
        }
    }

    /// Current monotonic GTE diagnostic counters.
    pub fn profile_snapshot(&self) -> GteProfileSnapshot {
        GteProfileSnapshot {
            ops: self.profile_ops,
            estimated_cycles: self.profile_estimated_cycles,
            opcode_counts: self.profile_opcode_counts,
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
        if let Some(cycles) = gte_command_cycles(opcode) {
            self.profile_ops = self.profile_ops.saturating_add(1);
            self.profile_estimated_cycles =
                self.profile_estimated_cycles.saturating_add(cycles as u64);
            self.profile_opcode_counts[opcode as usize] =
                self.profile_opcode_counts[opcode as usize].saturating_add(1);
        }
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
        let sum =
            (self.sz[0] as i64) + (self.sz[1] as i64) + (self.sz[2] as i64) + (self.sz[3] as i64);
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
            let combined =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64)) >> sf;
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
            let combined =
                self.check_mac((i + 1) as u8, base + (self.ir0 as i64) * (ir as i64)) >> sf;
            self.mac[i] = combined as i32;
            self.ir[i] = self.saturate_ir((i + 1) as u8, combined as i32, cmd.lm);
        }
        self.push_color_from_mac(cmd);
    }

    /// `NCS` — normal colour single. Lights `V[idx]` against the LLM,
    /// then colours via LCM.
    ///
    /// Stage 1 formula per PSX-SPX: `MAC = (LLM × V) >> (sf×12)` —
    /// *no* BK bias. (The previous implementation added BK in both
    /// stages and overcounted the bias.) Stage 2 is where BK bias
    /// belongs: `MAC = (BK<<12 + LCM × IR) >> (sf×12)`.
    fn op_ncs(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        // [IR1,IR2,IR3] = (LLM * V) >> (sf*12)
        for i in 0..3 {
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, prod) >> sf;
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
    ///
    /// Stage-1 BK bias removed — see [`Gte::op_ncs`] for the rationale.
    fn op_ncds(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        for i in 0..3 {
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, prod) >> sf;
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
    ///
    /// Stage-1 BK bias removed — see [`Gte::op_ncs`] for the rationale.
    fn op_nccs(&mut self, cmd: Cmd, idx: usize) {
        let sf = cmd.shift();
        let v = self.v[idx];
        for i in 0..3 {
            let prod = (self.light[i][0] as i64) * (v[0] as i64)
                + (self.light[i][1] as i64) * (v[1] as i64)
                + (self.light[i][2] as i64) * (v[2] as i64);
            let mac = self.check_mac((i + 1) as u8, prod) >> sf;
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
            let mac = self.check_mac((i + 1) as u8, (self.ir0 as i64) * (self.ir[i] as i64)) >> sf;
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
            let mac = self.check_mac(
                (i + 1) as u8,
                base + (self.ir0 as i64) * (self.ir[i] as i64),
            ) >> sf;
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
        // Table index is `(d - 0x7FC0) >> 7`, range 0..=256 (UNR_TABLE
        // has 257 entries). A `& 0xFF` mask aliases index 256 →
        // index 0, which corrupts the reciprocal whenever the
        // normalised divisor lands in {0xFFFE, 0xFFFF}: table[256]=0x00
        // gives u=0x101 (smallest seed) but table[0]=0xFF gives
        // u=0x200 (largest), producing a divisor ~2000× too small and
        // collapsing projected vertices toward the screen offset.
        // Matches Redux's `gte_divide` and PSX-SPX exactly.
        let table_index = (d.wrapping_sub(0x7FC0) >> 7) as usize;
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
            // mx=3 is the documented "garbage matrix" (PSX-SPX: GTE
            // Opcodes Summary). Row 0 col 2 is **IR0** (data reg 8),
            // not IR1 — catching this was why `mvmva_mx3_garbage_matrix`
            // got added. Rows 1 and 2 replicate RT[0][2] and RT[1][1]
            // across all three columns.
            _ => {
                let r = (self.rgbc[0] as i16) << 4;
                [
                    [-r, r, self.ir0],
                    [
                        self.rotation[0][2],
                        self.rotation[0][2],
                        self.rotation[0][2],
                    ],
                    [
                        self.rotation[1][1],
                        self.rotation[1][1],
                        self.rotation[1][1],
                    ],
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
    if test == 0 {
        32
    } else {
        test.leading_zeros()
    }
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
            // Including LZCR (reg 31): Redux memsets register storage
            // to zero on boot and only updates LZCR when LZCS is
            // written. We mirror that so step-0 parity traces match.
            assert_eq!(g.read_data(i), 0, "data reg {i}");
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

    #[test]
    fn profile_counts_recognised_commands_only() {
        let mut g = Gte::new();

        g.execute(0x01); // RTPS
        g.execute(0x30); // RTPT
        g.execute(0x00); // undefined nop

        let profile = g.profile_snapshot();
        assert_eq!(profile.ops, 2);
        assert_eq!(profile.estimated_cycles, 15 + 23);
        assert_eq!(profile.opcode_counts[0x01], 1);
        assert_eq!(profile.opcode_counts[0x30], 1);
        assert_eq!(profile.opcode_counts[0x00], 0);
    }

    // ------------------------------------------------------------------
    // Broader known-value fixtures — verify arithmetic for the ops games
    // use most, plus the register-view edge cases. Every expected number
    // below comes from hand-computed PSX-SPX math on small inputs; if one
    // of these fires it means a refactor has drifted the implementation.
    // ------------------------------------------------------------------

    /// Install the identity rotation matrix scaled 1.0 in 1.3.12 (i.e.
    /// every diagonal = 0x1000, off-diagonal = 0). Keeps RTPS/MVMVA
    /// fixtures readable: `MAC = V` in the shifted space.
    fn install_identity_rotation(g: &mut Gte) {
        g.write_control(0, pack_xy_i16(0x1000, 0));
        g.write_control(1, pack_xy_i16(0, 0));
        g.write_control(2, pack_xy_i16(0x1000, 0));
        g.write_control(3, pack_xy_i16(0, 0));
        g.write_control(4, 0x1000);
    }

    /// GTE command-word builder: COP2 functions leave the high bits
    /// alone — only sf/lm/mx/vx/cv/opcode matter. Keeps test instructions
    /// readable next to their intent.
    fn cmd_word(sf: bool, lm: bool, mx: u8, vx: u8, cv: u8, opcode: u8) -> u32 {
        ((sf as u32) << 19)
            | ((mx as u32 & 3) << 17)
            | ((vx as u32 & 3) << 15)
            | ((cv as u32 & 3) << 13)
            | ((lm as u32) << 10)
            | (opcode as u32)
    }

    #[test]
    fn rtps_with_h_zero_projects_to_screen_offset() {
        // With H=0 the divisor collapses to 0, so the projected X/Y are
        // just OFX>>16 and OFY>>16 — a crisp fixture for checking every
        // RTPS side-effect (MAC1/2/3, IR1/2/3, SZ push, SXY push, MAC0,
        // IR0) without the Newton-Raphson divider in the way.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(1, 2)); // V0.x, V0.y
        g.write_data(1, 3); // V0.z
        g.write_control(24, 0x0100_0000); // OFX = 256.0 in 15.16
        g.write_control(25, 0x0200_0000); // OFY = 512.0 in 15.16
        g.write_control(26, 0); // H = 0
        g.write_control(27, 100); // DQA
        g.write_control(28, 5000); // DQB
        g.execute(cmd_word(true, false, 0, 0, 0, 0x01));
        // MAC = V (identity × V >> 12 = V for these small inputs)
        assert_eq!(g.read_data(25) as i32, 1, "MAC1");
        assert_eq!(g.read_data(26) as i32, 2, "MAC2");
        assert_eq!(g.read_data(27) as i32, 3, "MAC3");
        assert_eq!(g.read_data(9), 1, "IR1");
        assert_eq!(g.read_data(10), 2, "IR2");
        assert_eq!(g.read_data(11), 3, "IR3");
        // SZ3 pushed from mac[2]=3 (sf=1 path).
        assert_eq!(g.read_data(19), 3, "SZ3");
        // H=0 ⇒ divisor=0 ⇒ SXY2 = (OFX>>16, OFY>>16).
        assert_eq!(g.read_data(14), pack_xy_i16(0x100, 0x200), "SXY2");
        // MAC0 = divisor * DQA + DQB = 0 + 5000.
        assert_eq!(g.read_data(24) as i32, 5000, "MAC0");
        // IR0 = MAC0 >> 12 saturated to 0..0x1000.
        assert_eq!(g.read_data(8), 1, "IR0");
        // H<2*SZ3 so no DIV_OVERFLOW.
        assert_eq!(g.read_control(31) & flag::DIV_OVERFLOW, 0);
    }

    #[test]
    fn rtpt_transforms_all_three_vertices_and_pushes_sz_fifo() {
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(1, 2));
        g.write_data(1, 3); // V0
        g.write_data(2, pack_xy_i16(4, 5));
        g.write_data(3, 6); // V1
        g.write_data(4, pack_xy_i16(7, 8));
        g.write_data(5, 9); // V2
        g.write_control(24, 0);
        g.write_control(25, 0);
        g.write_control(26, 0); // H=0 keeps projection math simple.
        g.execute(cmd_word(true, false, 0, 0, 0, 0x30));
        // Final IR is from the *last* vertex transformed (V2 = (7,8,9)).
        assert_eq!(g.read_data(9), 7);
        assert_eq!(g.read_data(10), 8);
        assert_eq!(g.read_data(11), 9);
        // SZ FIFO after three pushes of 3, 6, 9.
        assert_eq!(g.read_data(17), 3, "SZ1");
        assert_eq!(g.read_data(18), 6, "SZ2");
        assert_eq!(g.read_data(19), 9, "SZ3");
    }

    #[test]
    fn mvmva_rt_v0_tr_computes_rotation_plus_translation() {
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(1, 2));
        g.write_data(1, 3);
        g.write_control(5, 10);
        g.write_control(6, 20);
        g.write_control(7, 30);
        g.execute(cmd_word(true, false, 0, 0, 0, 0x12));
        // MAC_i = ((TR_i<<12) + RT_row_i · V) >> 12 = TR_i + V_i.
        assert_eq!(g.read_data(25) as i32, 11, "MAC1 = TR0 + V0.x");
        assert_eq!(g.read_data(26) as i32, 22, "MAC2 = TR1 + V0.y");
        assert_eq!(g.read_data(27) as i32, 33, "MAC3 = TR2 + V0.z");
        assert_eq!(g.read_data(9), 11);
        assert_eq!(g.read_data(10), 22);
        assert_eq!(g.read_data(11), 33);
    }

    #[test]
    fn mvmva_cv3_drops_translation() {
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(1, 2));
        g.write_data(1, 3);
        g.write_control(5, 100); // TR should be ignored with cv=3.
        g.write_control(6, 200);
        g.write_control(7, 300);
        g.execute(cmd_word(true, false, 0, 0, 3, 0x12));
        assert_eq!(g.read_data(25) as i32, 1, "MAC1 = V.x (no TR)");
        assert_eq!(g.read_data(26) as i32, 2, "MAC2 = V.y");
        assert_eq!(g.read_data(27) as i32, 3, "MAC3 = V.z");
    }

    #[test]
    fn avsz4_averages_using_zsf4() {
        let mut g = Gte::new();
        g.write_data(16, 0x010);
        g.write_data(17, 0x020);
        g.write_data(18, 0x030);
        g.write_data(19, 0x040);
        g.write_control(30, 0x0400); // ZSF4 = 0x400 (1/4 in 0.12)
        g.execute(0x2E);
        // MAC0 = ZSF4 * (SZ0+SZ1+SZ2+SZ3) = 0x400 * 0xA0 = 0x28000.
        assert_eq!(g.read_data(24) as i32, 0x28000, "MAC0");
        assert_eq!(g.read_data(7), 0x28, "OTZ = MAC0>>12");
    }

    #[test]
    fn irgb_write_unpacks_into_ir_vector() {
        // IRGB packed (B<<10) | (G<<5) | R. Each 5-bit component scales
        // to IR_i = component << 7, matching PSX-SPX.
        let mut g = Gte::new();
        g.write_data(28, (0x0A << 10) | (0x10 << 5) | 0x1F);
        assert_eq!(g.read_data(9) as i16, 0x1F << 7);
        assert_eq!(g.read_data(10) as i16, 0x10 << 7);
        assert_eq!(g.read_data(11) as i16, 0x0A << 7);
    }

    #[test]
    fn orgb_packs_ir_into_5_5_5() {
        let mut g = Gte::new();
        g.write_data(9, 0); // R=0
        g.write_data(10, 0xF80); // G=0x1F (max)
        g.write_data(11, 0x780); // B=0x0F
        assert_eq!(g.read_data(29), (0x0F << 10) | (0x1F << 5) | 0);
    }

    #[test]
    fn orgb_write_is_read_only() {
        let mut g = Gte::new();
        g.write_data(9, 0x500);
        g.write_data(29, 0xFFFF); // Writes to ORGB are silently dropped.
        assert_eq!(g.read_data(9) as i16, 0x500);
    }

    #[test]
    fn mac_overflow_sets_positive_flag() {
        // TR<<12 alone sits at 8.795_893e12; adding RT[0][0]·V0.x (another
        // ~1e9 with maxed-out i16s) tips past 2^43-1 = 8.796_093e12.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_control(5, 0x7FFF_FFFF); // TR.x = i32::MAX
        g.write_control(0, pack_xy_i16(0x7FFF, 0)); // RT[0][0] = max i16
        g.write_data(0, pack_xy_i16(0x7FFF, 0)); // V0.x = max i16
        g.execute(cmd_word(false, false, 0, 0, 0, 0x01));
        assert!(g.read_control(31) & flag::MAC1_POS != 0, "MAC1_POS");
        assert!(g.read_control(31) & 0x8000_0000 != 0, "master bit");
    }

    #[test]
    fn gpf_multiplies_ir_by_ir0_and_pushes_color() {
        let mut g = Gte::new();
        g.write_data(8, 0x1000); // IR0 = 1.0
        g.write_data(9, 0x800);
        g.write_data(10, 0x1000); // will saturate color G
        g.write_data(11, 0x200);
        g.execute(cmd_word(true, false, 0, 0, 0, 0x3D));
        assert_eq!(g.read_data(25) as i32, 0x800, "MAC1");
        assert_eq!(g.read_data(26) as i32, 0x1000, "MAC2");
        assert_eq!(g.read_data(27) as i32, 0x200, "MAC3");
        // Newest color: (r,g,b,code) = (0x80, 0xFF, 0x20, 0) packed LE.
        // 0x1000>>4 = 0x100 → clamps to 0xFF, setting COLOR_G_SAT.
        assert_eq!(g.read_data(22), 0x0020_FF80);
        assert!(g.read_control(31) & flag::COLOR_G_SAT != 0);
    }

    #[test]
    fn gpl_accumulates_base_mac_into_product() {
        // GPL reshifts the existing MAC left by sf before adding the new
        // product so the base acts as the same fixed-point scale.
        let mut g = Gte::new();
        g.write_data(25, 0x100);
        g.write_data(26, 0x200);
        g.write_data(27, 0x300);
        g.write_data(8, 0x1000); // IR0
        g.write_data(9, 0x400);
        g.write_data(10, 0x500);
        g.write_data(11, 0x600);
        g.execute(cmd_word(true, false, 0, 0, 0, 0x3E));
        assert_eq!(g.read_data(25) as i32, 0x500, "MAC1 = 0x100 + 0x400");
        assert_eq!(g.read_data(26) as i32, 0x700, "MAC2 = 0x200 + 0x500");
        assert_eq!(g.read_data(27) as i32, 0x900, "MAC3 = 0x300 + 0x600");
    }

    #[test]
    fn flag_master_bit_set_on_any_error() {
        let mut g = Gte::new();
        g.write_control(31, flag::IR1_SAT);
        assert_eq!(g.read_control(31) & flag::IR1_SAT, flag::IR1_SAT);
        assert!(g.read_control(31) & 0x8000_0000 != 0);
    }

    #[test]
    fn flag_master_bit_clear_on_non_error_bits() {
        // MAC0_POS lives in the error mask; a bit outside it (the RTPS
        // divider's SZ3_OTZ_SAT is in-mask, but there are no gaps we can
        // address via write_control since the mask covers bits 12..30
        // except bit 11 which write_control already strips). Instead we
        // verify: explicitly clearing produces no master bit.
        let mut g = Gte::new();
        g.write_control(31, 0);
        assert_eq!(g.read_control(31) & 0x8000_0000, 0);
    }

    #[test]
    fn nclip_collinear_points_produce_zero() {
        // Three points on y=x line ⇒ signed area = 0 regardless of spacing.
        let mut g = Gte::new();
        g.write_data(12, pack_xy_i16(0, 0));
        g.write_data(13, pack_xy_i16(5, 5));
        g.write_data(14, pack_xy_i16(10, 10));
        g.execute(0x06);
        assert_eq!(g.read_data(24) as i32, 0);
    }

    #[test]
    fn op_outer_product_of_ir_with_unit_diagonal() {
        // With diag=(1,1,1) in 1.3.12 the outer product reduces to the
        // standard cross-product of IR with itself-as-axis.
        let mut g = Gte::new();
        g.write_control(0, pack_xy_i16(0x1000, 0));
        g.write_control(2, pack_xy_i16(0x1000, 0));
        g.write_control(4, 0x1000);
        g.write_data(9, 3);
        g.write_data(10, 4);
        g.write_data(11, 5);
        g.execute(cmd_word(true, false, 0, 0, 0, 0x0C));
        // MAC1 = IR3*D2 - IR2*D3 = 5-4 = 1
        // MAC2 = IR1*D3 - IR3*D1 = 3-5 = -2
        // MAC3 = IR2*D1 - IR1*D2 = 4-3 = 1
        assert_eq!(g.read_data(25) as i32, 1, "MAC1");
        assert_eq!(g.read_data(26) as i32, -2, "MAC2");
        assert_eq!(g.read_data(27) as i32, 1, "MAC3");
        assert_eq!(g.read_data(9) as i16, 1);
        assert_eq!(g.read_data(10) as i16, -2);
        assert_eq!(g.read_data(11) as i16, 1);
    }

    #[test]
    fn ir0_saturates_to_upper_bound() {
        // RTPS with H=SZ3 gives divisor exactly 1.0 (0x10000). Then
        // MAC0 = 0x10000 * DQA (for DQB=0) vastly exceeds 0x1000 once
        // shifted by 12 → IR0 clamps, flag asserts.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(0, 0));
        g.write_data(1, 1); // V0.z=1 so SZ3=1 post-RTPS.
        g.write_control(26, 1); // H=1
        g.write_control(27, 0x7FFF); // DQA max
        g.write_control(28, 0); // DQB=0 keeps MAC0 in i32 range.
        g.execute(cmd_word(true, false, 0, 0, 0, 0x01));
        assert_eq!(g.read_data(8) as i16, 0x1000, "IR0 saturated");
        assert!(g.read_control(31) & flag::IR0_SAT != 0);
    }

    #[test]
    fn color_fifo_push_order_across_three_ops() {
        // Three GPFs with distinct IR vectors — after three pushes,
        // slot 0 holds the oldest, slot 2 the newest.
        let mut g = Gte::new();
        g.write_data(8, 0x1000); // IR0 = 1.0
        let instr = cmd_word(true, false, 0, 0, 0, 0x3D);
        g.write_data(9, 0x100);
        g.write_data(10, 0x200);
        g.write_data(11, 0x300);
        g.execute(instr);
        g.write_data(9, 0x400);
        g.write_data(10, 0x500);
        g.write_data(11, 0x600);
        g.execute(instr);
        g.write_data(9, 0x700);
        g.write_data(10, 0x100);
        g.write_data(11, 0x200);
        g.execute(instr);
        // Each push: r=MAC1>>4, g=MAC2>>4, b=MAC3>>4, code=RGBC[3]=0.
        // u32 view packs little-endian: r | (g<<8) | (b<<16) | (code<<24).
        assert_eq!(g.read_data(20), 0x0030_2010, "oldest push");
        assert_eq!(g.read_data(21), 0x0060_5040, "middle push");
        assert_eq!(g.read_data(22), 0x0020_1070, "newest push");
    }

    #[test]
    fn sqr_sf1_scales_product_down_by_4096() {
        let mut g = Gte::new();
        g.write_data(9, 0x1000);
        g.write_data(10, 0x0800);
        g.write_data(11, 0x2000);
        g.execute(cmd_word(true, false, 0, 0, 0, 0x28));
        // MAC_i = (IR_i * IR_i) >> 12
        assert_eq!(g.read_data(25) as i32, 0x1000);
        assert_eq!(g.read_data(26) as i32, 0x400);
        assert_eq!(g.read_data(27) as i32, 0x4000);
    }

    #[test]
    fn rtps_h_equals_2sz3_sets_divide_overflow() {
        // Boundary: H exactly 2*SZ3 triggers DIV_OVERFLOW and forces
        // divisor = 0x1FFFF (clamped max).
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(0, pack_xy_i16(0, 0));
        g.write_data(1, 1); // SZ3 → 1
        g.write_control(26, 2); // H = 2 → H >= 2*SZ3.
        g.execute(cmd_word(true, false, 0, 0, 0, 0x01));
        assert!(g.read_control(31) & flag::DIV_OVERFLOW != 0);
    }

    #[test]
    fn rtps_unr_table_index_256_is_not_aliased() {
        // Regression: when the normalised SZ3 lands in {0xFFFE, 0xFFFF}
        // the UNR-table index is 256 — the upper bound of the
        // 257-entry table. A defensive `& 0xFF` mask on the index
        // aliased it to 0, swapping the smallest reciprocal multiplier
        // (table[256]=0x00 → u=0x101) for the largest
        // (table[0]=0xFF → u=0x200). The resulting divisor was about
        // 2000× too small, which collapses the projected vertex
        // toward (OFX>>16, OFY>>16) — visually "triangles exploding
        // to the screen centre". Pin the boundary so the mask can't
        // sneak back in.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        // V0 = (0x100, 0, 0x7FFF). With identity rotation and sf=1,
        // MAC3 = V.z = 0x7FFF, so SZ3 = 0x7FFF; leading_zeros
        // normalises that to d = 0xFFFE which exercises table[256].
        g.write_data(0, pack_xy_i16(0x100, 0));
        g.write_data(1, 0x7FFF);
        g.write_control(24, 0); // OFX
        g.write_control(25, 0); // OFY
        g.write_control(26, 0x4000); // H
        g.execute(cmd_word(true, false, 0, 0, 0, 0x01));
        assert_eq!(g.read_data(19), 0x7FFF, "SZ3 fixture");
        // Correct divisor is 0x8001 (≈(H<<16)/SZ3=0x8000.4, rounded
        // up by Newton-Raphson). With the buggy mask divisor=0x4 and
        // SX2 collapses to 0. Expected SX2 = (0x8001*0x100)>>16 = 0x80.
        let sxy2 = g.read_data(14);
        let sx2 = (sxy2 & 0xFFFF) as i16;
        assert_eq!(sx2, 0x80, "SX2 should be 0x80, was {sx2:#x}");
    }

    #[test]
    fn unknown_opcode_leaves_all_registers_clear() {
        // Regression: make sure an invalid opcode doesn't accidentally
        // flip flag bits from prior execution of a zeroed sub-path.
        let mut g = Gte::new();
        g.execute(0x3C); // unassigned slot between 0x3B and 0x3D
        for i in 0..31 {
            assert_eq!(g.read_control(i), 0, "ctrl {i}");
        }
        assert_eq!(g.read_control(31), 0, "flag");
    }

    #[test]
    fn mvmva_cv2_fc_bug_flags_stage1_but_stores_stage2() {
        // Documented FC-translation bug: stage 1 computes
        //   (FC<<12) + MX_col0 * V.x   >> sf
        // and saturation-flags the IR from that; stage 2 computes the
        // real (MX · V) >> sf and stores MAC/IR from *that*. So large
        // FC entries set flags even when the final result is small.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_control(21, 0x1000_0000); // FC.r
        g.write_data(0, pack_xy_i16(1, 0));
        g.write_data(1, 0);
        g.execute(cmd_word(true, false, 0, 0, 2, 0x12));
        // Stage 2 math: MAC1 = (identity · V) >> 12 = 1 for V=(1,0,0).
        assert_eq!(g.read_data(25) as i32, 1, "MAC1 from clean stage 2");
        assert_eq!(g.read_data(9), 1, "IR1 from clean stage 2");
        // Stage 1 IR saturation: (FC<<12 + 0x1000*1) >> 12 ≈ 2^28 → IR_SAT.
        assert!(
            g.read_control(31) & flag::IR1_SAT != 0,
            "IR1_SAT from stage 1"
        );
    }

    #[test]
    fn mvmva_mx3_uses_garbage_matrix() {
        // mx=3 builds a junk matrix from RGBC, IR0, RT[0][2] and RT[1][1]
        // per PSX-SPX. Row 0 col 2 is specifically *IR0* (data reg 8),
        // not IR1 — pinning that here so a refactor can't re-introduce
        // the IR1-vs-IR0 confusion.
        let mut g = Gte::new();
        g.write_data(6, 0x0000_0010); // RGBC.r = 0x10
        g.write_data(8, 0x100); // IR0 = 0x100
        g.write_control(1, pack_xy_i16(0x200, 0)); // RT[0][2] = 0x200
        g.write_control(2, pack_xy_i16(0x300, 0)); // RT[1][1] = 0x300
        g.write_data(0, pack_xy_i16(1, 1));
        g.write_data(1, 1); // V0 = (1,1,1)
        g.execute(cmd_word(false, false, 3, 0, 3, 0x12));
        // Garbage matrix:
        //   [[-0x100, 0x100, IR0=0x100], [0x200; 3], [0x300; 3]]
        // With V = (1,1,1), sf=0:
        //   MAC1 = -0x100 + 0x100 + 0x100 = 0x100
        //   MAC2 = 3 * 0x200 = 0x600
        //   MAC3 = 3 * 0x300 = 0x900
        assert_eq!(g.read_data(25) as i32, 0x100, "MAC1");
        assert_eq!(g.read_data(26) as i32, 0x600, "MAC2");
        assert_eq!(g.read_data(27) as i32, 0x900, "MAC3");
    }

    #[test]
    fn mvmva_vx3_uses_ir_as_vector() {
        // vx=3 substitutes [IR1, IR2, IR3] for the multiplied vector —
        // used by the lighting chain when colors feed back into geometry.
        let mut g = Gte::new();
        install_identity_rotation(&mut g);
        g.write_data(9, 5);
        g.write_data(10, 6);
        g.write_data(11, 7);
        // cv=3 (no TR), sf=1
        g.execute(cmd_word(true, false, 0, 3, 3, 0x12));
        assert_eq!(g.read_data(25) as i32, 5);
        assert_eq!(g.read_data(26) as i32, 6);
        assert_eq!(g.read_data(27) as i32, 7);
    }
}
