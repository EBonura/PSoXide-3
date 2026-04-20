//! `psxed` — PSoXide content-pipeline CLI.
//!
//! Cooks source assets into the compact binary formats the PS1
//! runtime consumes via `include_bytes!`. Invoked by hand, by a
//! `build.rs` hook, or from `make assets`.
//!
//! # Subcommands
//!
//! ## `obj` — Wavefront OBJ → `.psxm`
//!
//! ```bash
//! psxed obj SRC.obj -o DST.psxm [options]
//!
//! Options:
//!   --decimate-grid N   Vertex-cluster to N^3 cells. Omit for no decimation.
//!   --palette NAME      Face-colour palette: warm (default), cool, green.
//!   --no-colors         Skip the face-colour table.
//!   --compute-normals   Add per-vertex normals for lit rendering.
//! ```
//!
//! ## `tex` — PNG/JPG → `.psxt`
//!
//! ```bash
//! psxed tex SRC.{png,jpg,bmp} -o DST.psxt [options]
//!
//! Options:
//!   --size WxH           Target texel dimensions (default 64x64).
//!   --depth 4|8|15       Bits per texel (default 4 = 16-colour CLUT).
//!   --crop X,Y,W,H       Crop window on the source, pre-resize.
//!   --resample nearest|triangle|lanczos3  (default lanczos3)
//! ```
//!
//! ## Future subcommands
//!
//! - `vag`   — WAV → PSX VAG ADPCM audio
//! - `font`  — TTF or bitmap → psx-font atlas
//! - `scene` — edit a .pscene JSON and cook it into runtime format

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let result = match args[1].as_str() {
        "obj" => run_obj(&args[2..]),
        "tex" => run_tex(&args[2..]),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        cmd => Err(format!("unknown subcommand: {cmd}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("psxed: {msg}");
            ExitCode::from(1)
        }
    }
}

const USAGE: &str = "\
psxed — PSoXide content pipeline

USAGE:
    psxed <subcommand> [arguments]

SUBCOMMANDS:
    obj     Convert a Wavefront .obj mesh to .psxm format
    tex     Convert a PNG/JPG/BMP image to .psxt format
    help    Show this message

OBJ SUBCOMMAND:
    psxed obj <input.obj> -o <output.psxm>
                          [--decimate-grid N]
                          [--palette warm|cool|green]
                          [--no-colors]
                          [--compute-normals]

TEX SUBCOMMAND:
    psxed tex <input.png|.jpg|.bmp> -o <output.psxt>
                          [--size WxH]            (default 64x64)
                          [--depth 4|8|15]        (default 4)
                          [--crop X,Y,W,H]        (overrides centre-square)
                          [--no-crop]             (resize-stretch the full source)
                          [--resample nearest|triangle|lanczos3]

    The default crop is centre-square: the largest square that
    fits in the source, positioned at its centre. This avoids
    aspect distortion on arbitrary-aspect photographs. Pass
    --crop X,Y,W,H for manual control, or --no-crop to disable.

EXAMPLES:
    psxed obj vendor/teapot.obj -o build/teapot.psxm --palette cool
    psxed tex ~/Downloads/brick.jpg -o assets/brick.psxt \\
        --size 128x128 --depth 4 --resample lanczos3
";

fn run_obj(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut decimate_grid: Option<u32> = None;
    let mut palette = psxed_obj::Palette::Warm;
    let mut include_face_colors = true;
    let mut include_normals = false;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" | "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    args.get(i).ok_or_else(|| "expected path after -o".to_string())?,
                ));
            }
            "--decimate-grid" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected N after --decimate-grid".to_string())?;
                decimate_grid = Some(
                    val.parse()
                        .map_err(|_| format!("invalid grid value: {val}"))?,
                );
            }
            "--palette" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected palette name".to_string())?;
                palette = match val.as_str() {
                    "warm" => psxed_obj::Palette::Warm,
                    "cool" => psxed_obj::Palette::Cool,
                    "green" => psxed_obj::Palette::Green,
                    other => return Err(format!("unknown palette: {other}")),
                };
            }
            "--no-colors" => {
                include_face_colors = false;
            }
            "--compute-normals" => {
                include_normals = true;
            }
            a if a.starts_with('-') => {
                return Err(format!("unknown flag: {a}\n\n{USAGE}"));
            }
            _ => {
                // First positional = input path.
                if input.is_none() {
                    input = Some(PathBuf::from(a));
                } else {
                    return Err(format!("unexpected positional argument: {a}"));
                }
            }
        }
        i += 1;
    }

    let input = input.ok_or("missing input OBJ path")?;
    let output = output.ok_or("missing -o output path")?;

    let obj_bytes = std::fs::read(&input)
        .map_err(|e| format!("read {}: {e}", input.display()))?;
    let cfg = psxed_obj::Config {
        decimate_grid,
        palette,
        include_face_colors,
        include_normals,
    };
    let psxm =
        psxed_obj::convert(&obj_bytes, &cfg).map_err(|e| format!("convert: {e}"))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&output, &psxm)
        .map_err(|e| format!("write {}: {e}", output.display()))?;

    // One-line status so `make assets` logs stay legible.
    eprintln!(
        "[psxed obj] {} → {} ({} bytes)",
        input.display(),
        output.display(),
        psxm.len()
    );
    Ok(())
}

fn run_tex(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut width: u16 = 64;
    let mut height: u16 = 64;
    let mut depth = psxed_format::texture::Depth::Bit4;
    let mut crop = psxed_tex::CropMode::CentreSquare;
    let mut resampler = psxed_tex::Resampler::Lanczos3;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" | "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    args.get(i).ok_or_else(|| "expected path after -o".to_string())?,
                ));
            }
            "--size" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected WxH after --size".to_string())?;
                let (w, h) = val
                    .split_once('x')
                    .ok_or_else(|| format!("--size expects WxH, got: {val}"))?;
                width = w.parse().map_err(|_| format!("invalid width: {w}"))?;
                height = h.parse().map_err(|_| format!("invalid height: {h}"))?;
            }
            "--depth" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected bit-depth".to_string())?;
                depth = match val.as_str() {
                    "4" => psxed_format::texture::Depth::Bit4,
                    "8" => psxed_format::texture::Depth::Bit8,
                    "15" => psxed_format::texture::Depth::Bit15,
                    other => {
                        return Err(format!("invalid --depth: {other} (expected 4, 8, 15)"));
                    }
                };
            }
            "--crop" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected X,Y,W,H after --crop".to_string())?;
                let parts: Vec<&str> = val.split(',').collect();
                if parts.len() != 4 {
                    return Err(format!("--crop expects X,Y,W,H, got: {val}"));
                }
                let mut nums = [0u32; 4];
                for (j, p) in parts.iter().enumerate() {
                    nums[j] = p
                        .parse()
                        .map_err(|_| format!("invalid crop component: {p}"))?;
                }
                crop = psxed_tex::CropMode::Explicit(psxed_tex::CropRect {
                    x: nums[0],
                    y: nums[1],
                    w: nums[2],
                    h: nums[3],
                });
            }
            "--no-crop" => {
                crop = psxed_tex::CropMode::None;
            }
            "--resample" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected resampler name".to_string())?;
                resampler = match val.as_str() {
                    "nearest" => psxed_tex::Resampler::Nearest,
                    "triangle" => psxed_tex::Resampler::Triangle,
                    "lanczos3" => psxed_tex::Resampler::Lanczos3,
                    other => {
                        return Err(format!(
                            "invalid --resample: {other} (expected nearest|triangle|lanczos3)"
                        ));
                    }
                };
            }
            a if a.starts_with('-') => {
                return Err(format!("unknown flag: {a}\n\n{USAGE}"));
            }
            _ => {
                if input.is_none() {
                    input = Some(PathBuf::from(a));
                } else {
                    return Err(format!("unexpected positional argument: {a}"));
                }
            }
        }
        i += 1;
    }

    let input = input.ok_or("missing input image path")?;
    let output = output.ok_or("missing -o output path")?;

    let src_bytes = std::fs::read(&input)
        .map_err(|e| format!("read {}: {e}", input.display()))?;
    let cfg = psxed_tex::Config {
        width,
        height,
        depth,
        crop,
        resampler,
    };
    let psxt = psxed_tex::convert(&src_bytes, &cfg).map_err(|e| format!("convert: {e}"))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&output, &psxt)
        .map_err(|e| format!("write {}: {e}", output.display()))?;

    eprintln!(
        "[psxed tex] {} → {} ({}×{} {}bpp, {} bytes)",
        input.display(),
        output.display(),
        width,
        height,
        depth as u8,
        psxt.len(),
    );
    Ok(())
}
