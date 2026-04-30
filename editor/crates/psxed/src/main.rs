//! `psxed` -- PSoXide content-pipeline CLI.
//!
//! Cooks source assets into the compact binary formats the PS1
//! runtime consumes via `include_bytes!`. Invoked by hand, by a
//! `build.rs` hook, or from `make assets`.
//!
//! # Subcommands
//!
//! ## `obj` -- Wavefront OBJ → `.psxm`
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
//! ## `glb` -- glTF/GLB → `.psxm`
//!
//! ```bash
//! psxed glb SRC.glb -o DST.psxm [options]
//!
//! Options:
//!   --decimate-grid N       Vertex-cluster to N^3 cells. Omit for no decimation.
//!   --palette NAME          Fallback face-colour palette: warm (default), cool, green.
//!   --no-colors             Skip the face-colour table.
//!   --no-normals            Skip computed per-vertex normals.
//!   --no-material-colors    Ignore glTF material base colours and use palette cycling.
//! ```
//!
//! ## `glb-model` -- skinned glTF/GLB → `.psxmdl` + `.psxanim` + `.psxt`
//!
//! ```bash
//! psxed glb-model SRC.glb --out-dir assets/models/name --name name [options]
//!
//! Options:
//!   --texture-size WxH    Target texture dimensions (default 128x128).
//!   --texture-depth 4|8|15  Target texture depth (default 8).
//!   --anim-fps N          Fixed animation sample rate (default 15).
//!   --world-height N      Suggested engine/world height (default 1024).
//! ```
//!
//! ## `tex` -- PNG/JPG → `.psxt`
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
//! - `vag`   -- WAV → PSX VAG ADPCM audio
//! - `font`  -- TTF or bitmap → psx-font atlas
//! - `scene` -- edit a .pscene JSON and cook it into runtime format

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let result = match args[1].as_str() {
        "glb" | "gltf" => run_glb(&args[2..]),
        "glb-model" | "gltf-model" => run_glb_model(&args[2..]),
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
psxed -- PSoXide content pipeline

USAGE:
    psxed <subcommand> [arguments]

SUBCOMMANDS:
    glb     Convert a glTF/.glb scene mesh to .psxm format
    glb-model
            Convert a skinned glTF/.glb model to .psxmdl/.psxanim/.psxt
    obj     Convert a Wavefront .obj mesh to .psxm format
    tex     Convert a PNG/JPG/BMP image to .psxt format
    help    Show this message

OBJ SUBCOMMAND:
    psxed obj <input.obj> -o <output.psxm>
                          [--decimate-grid N]
                          [--palette warm|cool|green]
                          [--no-colors]
                          [--compute-normals]

GLB SUBCOMMAND:
    psxed glb <input.glb|input.gltf> -o <output.psxm>
                          [--decimate-grid N]
                          [--palette warm|cool|green]
                          [--no-colors]
                          [--no-normals]
                          [--no-material-colors]

GLB-MODEL SUBCOMMAND:
    psxed glb-model <input.glb|input.gltf> --out-dir <directory>
                          [--name asset_name]
                          [--texture-size WxH]     (default 128x128)
                          [--texture-depth 4|8|15] (default 8)
                          [--anim-fps N]           (default 15)
                          [--world-height N]       (default 1024)

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
    psxed glb ~/Downloads/model.glb -o assets/model.psxm \\
        --decimate-grid 6
    psxed glb-model ~/Downloads/character.glb --out-dir assets/models/character \\
        --name character --texture-size 128x128 --texture-depth 8 --anim-fps 15 \\
        --world-height 1024
    psxed obj vendor/teapot.obj -o build/teapot.psxm --palette cool
    psxed tex ~/Downloads/brick.jpg -o assets/brick.psxt \\
        --size 128x128 --depth 4 --resample lanczos3
";

fn run_glb(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut decimate_grid: Option<u32> = None;
    let mut palette = psxed_gltf::Palette::Warm;
    let mut include_face_colors = true;
    let mut include_normals = true;
    let mut use_material_colors = true;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" | "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| "expected path after -o".to_string())?,
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
                palette = parse_gltf_palette(val)?;
            }
            "--no-colors" => {
                include_face_colors = false;
            }
            "--compute-normals" => {
                include_normals = true;
            }
            "--no-normals" => {
                include_normals = false;
            }
            "--material-colors" => {
                use_material_colors = true;
            }
            "--no-material-colors" => {
                use_material_colors = false;
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

    let input = input.ok_or("missing input GLB/glTF path")?;
    let output = output.ok_or("missing -o output path")?;
    let cfg = psxed_gltf::Config {
        decimate_grid,
        palette,
        include_face_colors,
        include_normals,
        use_material_colors,
    };
    let psxm = psxed_gltf::convert_path(&input, &cfg).map_err(|e| format!("convert: {e}"))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&output, &psxm).map_err(|e| format!("write {}: {e}", output.display()))?;

    eprintln!(
        "[psxed glb] {} -> {} ({} bytes)",
        input.display(),
        output.display(),
        psxm.len()
    );
    Ok(())
}

fn parse_gltf_palette(name: &str) -> Result<psxed_gltf::Palette, String> {
    match name {
        "warm" => Ok(psxed_gltf::Palette::Warm),
        "cool" => Ok(psxed_gltf::Palette::Cool),
        "green" => Ok(psxed_gltf::Palette::Green),
        other => Err(format!("unknown palette: {other}")),
    }
}

fn parse_size(value: &str, flag: &str) -> Result<(u16, u16), String> {
    let (w, h) = value
        .split_once('x')
        .ok_or_else(|| format!("{flag} expects WxH, got: {value}"))?;
    let width = w.parse().map_err(|_| format!("invalid width: {w}"))?;
    let height = h.parse().map_err(|_| format!("invalid height: {h}"))?;
    Ok((width, height))
}

fn parse_depth(value: &str) -> Result<psxed_format::texture::Depth, String> {
    match value {
        "4" => Ok(psxed_format::texture::Depth::Bit4),
        "8" => Ok(psxed_format::texture::Depth::Bit8),
        "15" => Ok(psxed_format::texture::Depth::Bit15),
        other => Err(format!("invalid bit-depth: {other} (expected 4, 8, 15)")),
    }
}

fn default_asset_name(input: &std::path::Path) -> String {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let mut out = String::with_capacity(stem.len());
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed.to_string()
    }
}

fn run_glb_model(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut texture_width: u16 = 128;
    let mut texture_height: u16 = 128;
    let mut texture_depth = psxed_format::texture::Depth::Bit8;
    let mut animation_fps: u16 = 15;
    let mut world_height: u16 = 1024;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--out-dir" | "--output-dir" => {
                i += 1;
                out_dir =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        "expected directory after --out-dir".to_string()
                    })?));
            }
            "--name" => {
                i += 1;
                name = Some(
                    args.get(i)
                        .ok_or_else(|| "expected asset name after --name".to_string())?
                        .to_string(),
                );
            }
            "--texture-size" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected WxH after --texture-size".to_string())?;
                let (w, h) = parse_size(val, "--texture-size")?;
                texture_width = w;
                texture_height = h;
            }
            "--texture-depth" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected bit-depth after --texture-depth".to_string())?;
                texture_depth = parse_depth(val)?;
            }
            "--anim-fps" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected N after --anim-fps".to_string())?;
                animation_fps = val
                    .parse()
                    .map_err(|_| format!("invalid --anim-fps value: {val}"))?;
            }
            "--world-height" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "expected N after --world-height".to_string())?;
                world_height = val
                    .parse()
                    .map_err(|_| format!("invalid --world-height value: {val}"))?;
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

    let input = input.ok_or("missing input GLB/glTF path")?;
    let out_dir = out_dir.ok_or("missing --out-dir directory")?;
    let name = name.unwrap_or_else(|| default_asset_name(&input));
    let cfg = psxed_gltf::RigidModelConfig {
        texture_width,
        texture_height,
        texture_depth,
        animation_fps,
        world_height,
    };
    let package =
        psxed_gltf::convert_rigid_model_path(&input, &cfg).map_err(|e| format!("convert: {e}"))?;

    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    let model_path = out_dir.join(format!("{name}.psxmdl"));
    std::fs::write(&model_path, &package.model)
        .map_err(|e| format!("write {}: {e}", model_path.display()))?;

    let mut clip_paths: Vec<(std::path::PathBuf, &psxed_gltf::CookedClip)> =
        Vec::with_capacity(package.clips.len());
    for clip in &package.clips {
        let path = out_dir.join(format!("{name}_{}.psxanim", clip.sanitized_name));
        std::fs::write(&path, &clip.bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        clip_paths.push((path, clip));
    }

    let texture_path = if let Some(texture) = &package.texture {
        let path = out_dir.join(format!(
            "{name}_{}x{}_{}bpp.psxt",
            texture_width, texture_height, texture_depth as u8
        ));
        std::fs::write(&path, texture).map_err(|e| format!("write {}: {e}", path.display()))?;
        Some(path)
    } else {
        None
    };

    eprintln!(
        "[psxed glb-model] {} -> {} ({} src verts, {} cooked verts, {} faces, {} parts, {} joints)",
        input.display(),
        model_path.display(),
        package.report.source_vertices,
        package.report.cooked_vertices,
        package.report.faces,
        package.report.parts,
        package.report.joints,
    );
    eprintln!(
        "[psxed glb-model] precision local_height={} local_to_world_q12={} target_world_height={}",
        package.report.local_height, package.report.local_to_world_q12, world_height
    );
    if clip_paths.is_empty() {
        eprintln!("[psxed glb-model] no animations in source");
    } else {
        eprintln!(
            "[psxed glb-model] {} clips @ {}Hz, {} bytes total",
            clip_paths.len(),
            animation_fps,
            package.report.animation_bytes
        );
        for (path, clip) in &clip_paths {
            let label = clip.source_name.as_deref().unwrap_or(&clip.sanitized_name);
            eprintln!(
                "[psxed glb-model]   {} ({} frames, {} bytes) <- {}",
                path.display(),
                clip.frames,
                clip.bytes.len(),
                label
            );
        }
    }
    if let Some(path) = texture_path {
        eprintln!(
            "[psxed glb-model] texture {} ({}x{} {}bpp, {} bytes)",
            path.display(),
            texture_width,
            texture_height,
            texture_depth as u8,
            package.report.texture_bytes
        );
    }
    Ok(())
}

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
                    args.get(i)
                        .ok_or_else(|| "expected path after -o".to_string())?,
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

    let obj_bytes = std::fs::read(&input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let cfg = psxed_obj::Config {
        decimate_grid,
        palette,
        include_face_colors,
        include_normals,
    };
    let psxm = psxed_obj::convert(&obj_bytes, &cfg).map_err(|e| format!("convert: {e}"))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&output, &psxm).map_err(|e| format!("write {}: {e}", output.display()))?;

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
                    args.get(i)
                        .ok_or_else(|| "expected path after -o".to_string())?,
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

    let src_bytes = std::fs::read(&input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let cfg = psxed_tex::Config {
        width,
        height,
        depth,
        crop,
        resampler,
    };
    let psxt = psxed_tex::convert(&src_bytes, &cfg).map_err(|e| format!("convert: {e}"))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&output, &psxt).map_err(|e| format!("write {}: {e}", output.display()))?;

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
