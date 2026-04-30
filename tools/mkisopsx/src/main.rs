//! `mkisopsx` -- wrap a PSX-EXE into a bootable PS1 disc image.
//!
//! Usage:
//!
//! ```text
//!   mkisopsx --exe path/to/hello-tri.exe --out hello.bin [--volume HELLO] [--iso]
//! ```
//!
//! Default output is a raw 2352-byte-per-sector `.bin` image -- the
//! format PSoXide's own CDROM loader expects and that most desktop
//! emulators (PCSX-Redux, Duckstation, Mednafen) also accept. Pass
//! `--iso` to emit a cooked 2048-byte-per-sector `.iso` instead
//! (smaller, accepted by a few tools that dislike raw sectors).
//!
//! Both flavours contain the same ISO 9660 filesystem: `SYSTEM.CNF`
//! (points the BIOS at `PSX.EXE`) and the EXE itself, both in the
//! root directory.
//!
//! The tool is deliberately a tiny CLI -- actual encoding lives in
//! `psx-iso::iso9660` so it's reusable from build scripts, test
//! harnesses, or a future GUI bundler.

use psx_iso::{default_system_cnf, Exe, IsoBuilder};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    exe: PathBuf,
    out: PathBuf,
    volume: String,
    cooked_iso: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut exe: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut volume = String::from("PSOXIDE");
    let mut cooked_iso = false;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--exe" | "-e" => {
                exe = Some(PathBuf::from(
                    it.next().ok_or_else(|| "--exe takes a path".to_string())?,
                ));
            }
            "--out" | "-o" => {
                out = Some(PathBuf::from(
                    it.next().ok_or_else(|| "--out takes a path".to_string())?,
                ));
            }
            "--volume" | "-v" => {
                volume = it
                    .next()
                    .ok_or_else(|| "--volume takes a string".to_string())?;
            }
            "--iso" => {
                cooked_iso = true;
            }
            "--help" | "-h" => {
                return Err(String::from("help"));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let exe = exe.ok_or_else(|| "missing --exe".to_string())?;
    let out = out.ok_or_else(|| "missing --out".to_string())?;
    Ok(Args {
        exe,
        out,
        volume,
        cooked_iso,
    })
}

fn print_usage() {
    eprintln!(
        "usage: mkisopsx --exe <file.exe> --out <file.bin> [--volume <id>] [--iso]\n\
         \n\
         Wraps a PSX-EXE into a bootable PS1 disc image containing\n\
         `SYSTEM.CNF` and `PSX.EXE` in the ISO 9660 root directory.\n\
         \n\
         --exe, -e       Path to the PSX-EXE (e.g. `hello-tri.exe`).\n\
         --out, -o       Destination path for the image.\n\
         --volume, -v    Volume identifier (default: PSOXIDE).\n\
         --iso           Emit a cooked 2048-byte-per-sector .iso\n\
                         instead of the default raw 2352-byte .bin.\n"
    );
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) if msg == "help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("{msg}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let exe_bytes = match fs::read(&args.exe) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {e}", args.exe.display());
            return ExitCode::from(1);
        }
    };

    // Validate the input actually is a PSX-EXE before we bake it into
    // an ISO -- silently packing a corrupt file would produce a disc
    // that just sits at the BIOS screen, which is annoying to debug.
    if let Err(e) = Exe::parse(&exe_bytes) {
        eprintln!("{}: not a valid PSX-EXE ({e:?})", args.exe.display());
        return ExitCode::from(1);
    }

    let mut builder = IsoBuilder::new().volume_id(&args.volume);
    builder.add_file("SYSTEM.CNF", default_system_cnf());
    builder.add_file("PSX.EXE", exe_bytes);

    let (image, sector_size, format_label) = if args.cooked_iso {
        let iso = builder.build();
        (iso, psx_iso::iso9660::SECTOR_SIZE, "cooked .iso")
    } else {
        let bin = builder.build_bin();
        (bin, psx_iso::iso9660::RAW_SECTOR_SIZE, "raw .bin")
    };

    if let Err(e) = fs::write(&args.out, &image) {
        eprintln!("write {}: {e}", args.out.display());
        return ExitCode::from(1);
    }

    println!(
        "wrote {} ({} bytes = {} sectors, {format_label}) from {}",
        args.out.display(),
        image.len(),
        image.len() / sector_size,
        args.exe.display(),
    );
    ExitCode::SUCCESS
}
