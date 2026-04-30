//! Directly dump what our CDROM path delivers for LBA 16 (PVD),
//! LBA 22 (root dir), LBA 23 (SYSTEM.CNF). Decouples "disc file is
//! correct" from "CDROM ReadN behaviour is correct" -- if the raw
//! bytes here match what's in the BIN file, we know the read path
//! is fine and the bug is in the BIOS-facing delivery (FIFO /
//! DMA3 / SetMode / timing).

use psx_iso::Disc;

fn main() {
    let disc_path = std::env::var("PSOXIDE_DISC").expect("set PSOXIDE_DISC");
    let bytes = std::fs::read(&disc_path).expect("disc readable");
    let disc = Disc::from_bin(bytes);

    for &lba in &[16u32, 22, 23] {
        println!("=== LBA {lba} (user data, 64 bytes) ===");
        match disc.read_sector_user(lba) {
            Some(user) => {
                for chunk in user[..64.min(user.len())].chunks(16) {
                    let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
                    let ascii: String = chunk
                        .iter()
                        .map(|&b| {
                            if (0x20..0x7f).contains(&b) {
                                b as char
                            } else {
                                '.'
                            }
                        })
                        .collect();
                    println!("  {}  {}", hex.join(" "), ascii);
                }
                // Specific fields.
                if lba == 16 && user.len() >= 180 {
                    // Root directory record starts at user[156].
                    println!("  root-dir record byte 0 (length):  0x{:02x}", user[156]);
                    let lba_le = u32::from_le_bytes([user[158], user[159], user[160], user[161]]);
                    let lba_be = u32::from_be_bytes([user[162], user[163], user[164], user[165]]);
                    let len_le = u32::from_le_bytes([user[166], user[167], user[168], user[169]]);
                    println!("  root-dir LBA (LE / BE): {lba_le} / {lba_be}");
                    println!("  root-dir length (LE):   {len_le}");
                }
            }
            None => println!("  (read failed)"),
        }
        println!();
    }
}
