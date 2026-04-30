//! Minimal self-contained GTE RTPS test. Bypasses the SDK entirely
//! so we can see whether the emulator's RTPS implementation produces
//! sensible screen coordinates for a canonical setup:
//!
//! - Rotation = identity
//! - Translation = (0, 0, 0x1000)   = 1.0 camera-forward
//! - OFX = 160 << 16, OFY = 120 << 16 (centre of a 320×240 screen)
//! - H = 200                          (projection plane)
//! - V0 = (0, 0, 0x1000)              = vertex on the forward axis
//!
//! Expected SXY2 post-RTPS: X = 160, Y = 120 (vertex projects to the
//! principal point -- translation brings it to camera Z = 2.0, the
//! X/Y components are zero so the offset applies cleanly).
//!
//! If the observed SXY2 is (160, 120) the emulator's RTPS is fine
//! and the hello-gte failure is in the SDK wrappers. If it's
//! something else (like 0, 0 or a saturated garbage value), we have
//! an emulator-side concern to surface.

use emulator_core::Gte;

fn pack_i16(lo: i16, hi: i16) -> u32 {
    ((hi as u16 as u32) << 16) | (lo as u16 as u32)
}

fn main() {
    let mut g = Gte::new();

    // === Identity rotation ===
    // PSX-SPX layout: ctc2(0)=(RT00,RT01), (1)=(RT02,RT10),
    // (2)=(RT11,RT12), (3)=(RT20,RT21), (4)=RT22.
    g.write_control(0, pack_i16(0x1000, 0)); // RT00=1.0, RT01=0
    g.write_control(1, pack_i16(0, 0)); // RT02=0, RT10=0
    g.write_control(2, pack_i16(0x1000, 0)); // RT11=1.0, RT12=0
    g.write_control(3, pack_i16(0, 0)); // RT20=0, RT21=0
    g.write_control(4, 0x1000); // RT22=1.0

    // === Translation ===
    g.write_control(5, 0); // TRX
    g.write_control(6, 0); // TRY
    g.write_control(7, 0x1000); // TRZ = 1.0 in 1.3.12

    // === Screen setup ===
    g.write_control(24, (160 << 16) as u32); // OFX
    g.write_control(25, (120 << 16) as u32); // OFY
    g.write_control(26, 200); // H (projection plane)
    g.write_control(27, 0); // DQA
    g.write_control(28, 0); // DQB

    // === Load V0 = (0, 0, 0x1000) ===
    // data reg 0 = (VX0, VY0) packed, data reg 1 = VZ0.
    g.write_data(0, pack_i16(0, 0));
    g.write_data(1, 0x1000);

    eprintln!("=== Inputs ===");
    eprintln!("RT (read back from control 0..=4):");
    eprintln!("  0: 0x{:08x}", g.read_control(0));
    eprintln!("  1: 0x{:08x}", g.read_control(1));
    eprintln!("  2: 0x{:08x}", g.read_control(2));
    eprintln!("  3: 0x{:08x}", g.read_control(3));
    eprintln!("  4: 0x{:08x}", g.read_control(4));
    eprintln!(
        "TR: ({:#x}, {:#x}, {:#x})",
        g.read_control(5),
        g.read_control(6),
        g.read_control(7)
    );
    eprintln!(
        "OFX={:#x} OFY={:#x} H={}",
        g.read_control(24),
        g.read_control(25),
        g.read_control(26)
    );
    eprintln!(
        "V0 (data 0, 1): 0x{:08x} 0x{:08x}",
        g.read_data(0),
        g.read_data(1)
    );

    // === Execute RTPS sf=1 ===
    // Opcode encoding: 0x4A000000 | 1<<19 (sf=1) | 0x01 (RTPS).
    // `execute()` takes the 32-bit opcode (the `.word` that MIPS
    // would emit); let's hand-encode the same thing.
    let opcode = 0x4A08_0001;
    g.execute(opcode);

    // === Read outputs ===
    eprintln!();
    eprintln!("=== Outputs after RTPS ===");
    let sxy0 = g.read_data(12);
    let sxy1 = g.read_data(13);
    let sxy2 = g.read_data(14);
    let sz3 = g.read_data(19);
    let mac1 = g.read_data(25);
    let mac2 = g.read_data(26);
    let mac3 = g.read_data(27);
    let ir1 = g.read_data(9);
    let ir2 = g.read_data(10);
    let ir3 = g.read_data(11);
    let flag = g.read_control(31);
    eprintln!(
        "SXY0: 0x{:08x}  (x={}, y={})",
        sxy0,
        sxy0 as i16,
        (sxy0 >> 16) as i16
    );
    eprintln!(
        "SXY1: 0x{:08x}  (x={}, y={})",
        sxy1,
        sxy1 as i16,
        (sxy1 >> 16) as i16
    );
    eprintln!(
        "SXY2: 0x{:08x}  (x={}, y={})",
        sxy2,
        sxy2 as i16,
        (sxy2 >> 16) as i16
    );
    eprintln!("SZ3:  0x{:08x} ({})", sz3, sz3);
    eprintln!("MAC1: 0x{:08x} ({})", mac1, mac1 as i32);
    eprintln!("MAC2: 0x{:08x} ({})", mac2, mac2 as i32);
    eprintln!("MAC3: 0x{:08x} ({})", mac3, mac3 as i32);
    eprintln!("IR1: 0x{:08x} ({})", ir1, ir1 as i16);
    eprintln!("IR2: 0x{:08x} ({})", ir2, ir2 as i16);
    eprintln!("IR3: 0x{:08x} ({})", ir3, ir3 as i16);
    eprintln!("FLAG: 0x{:08x}", flag);

    let (sx, sy) = (sxy2 as i16, (sxy2 >> 16) as i16);
    let expected = (160_i16, 120_i16);
    eprintln!();
    if (sx, sy) == expected {
        eprintln!(
            "✓ PASS: SXY2 = ({sx}, {sy}) matches expected {:?}",
            expected
        );
    } else {
        eprintln!(
            "✗ FAIL: SXY2 = ({sx}, {sy}) does NOT match expected {:?}. \
             This is an emulator-side concern — the input setup follows PSX-SPX.",
            expected,
        );
    }

    eprintln!();
    eprintln!("=== Second test: vertex off-axis ===");
    // V0 = (0x800, 0x400, 0x1000) -- should project offset from centre.
    // Expected SX = 160 + 200 * 0.5 / 1.0 = 260 (0x800=0.5, z=1.0)
    // Expected SY = 120 + 200 * 0.25 / 1.0 = 170 (0x400=0.25)
    // (Note: TR already applies 1.0 to Z; V0.z=0x1000 pushes SZ3 to 2.0 → halve the offsets:
    //  SX = 160 + 200 * 0.5 / 2.0 = 210, SY = 120 + 200 * 0.25 / 2.0 = 145.)
    g.write_data(0, pack_i16(0x800, 0x400));
    g.write_data(1, 0x1000);
    // Reset flag before second execute so we can tell fresh errors.
    g.write_control(31, 0);
    g.execute(opcode);
    let sxy2b = g.read_data(14);
    eprintln!(
        "SXY2 for V0=(0x800, 0x400, 0x1000): ({}, {})",
        sxy2b as i16,
        (sxy2b >> 16) as i16,
    );
    eprintln!("Expected ≈ (210, 145)");
    eprintln!(
        "MAC1={} MAC2={} SZ3={}",
        g.read_data(25) as i32,
        g.read_data(26) as i32,
        g.read_data(19),
    );
    eprintln!("FLAG after 2nd RTPS: 0x{:08x}", g.read_control(31));
}
