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
//! ```
//!
//! ## Future subcommands
//!
//! - `tim`   — PNG/BMP → PSX TIM texture
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
    help    Show this message

OBJ SUBCOMMAND:
    psxed obj <input.obj> -o <output.psxm>
                          [--decimate-grid N]
                          [--palette warm|cool|green]
                          [--no-colors]

EXAMPLE:
    psxed obj vendor/teapot.obj -o build/teapot.psxm --palette cool
";

fn run_obj(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut decimate_grid: Option<u32> = None;
    let mut palette = psxed_obj::Palette::Warm;
    let mut include_face_colors = true;

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
