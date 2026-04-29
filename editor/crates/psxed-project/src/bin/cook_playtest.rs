//! CLI: cook the editor's starter project (or a named project
//! file) into the playtest example's `generated/` directory.
//!
//! The editor's Play action calls the same
//! `psxed_project::playtest::cook_to_dir` underneath before the
//! frontend builds and side-loads the runtime. This bin exists so CI
//! scripts and the Makefile can drive the cook without spinning up the
//! full GUI.
//!
//! Usage:
//!   cook-playtest                     — cook the embedded starter
//!   cook-playtest <project.ron>       — cook the named project
//!
//! Exit codes: 0 success, 1 on validation errors, 2 on I/O.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use psxed_project::{
    default_project_dir,
    playtest::{build_package, cook_to_dir, default_generated_dir},
    ProjectDocument,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (project, project_root) = match args.first() {
        None => (ProjectDocument::starter(), default_project_dir()),
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match ProjectDocument::from_ron_str(&text) {
                Ok(p) => {
                    // Texture `psxt_path`s are stored relative to
                    // the project file, so anchor the project root
                    // at its parent directory.
                    let root = Path::new(path)
                        .parent()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("."));
                    (p, root)
                }
                Err(e) => {
                    eprintln!("[cook-playtest] {path}: parse failed: {e}");
                    return ExitCode::from(2);
                }
            },
            Err(e) => {
                eprintln!("[cook-playtest] {path}: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let dir = default_generated_dir();
    match cook_to_dir(&project, &project_root, &dir) {
        Ok(report) => {
            for warn in &report.warnings {
                eprintln!("[cook-playtest] warning: {warn}");
            }
            if !report.is_ok() {
                for err in &report.errors {
                    eprintln!("[cook-playtest] error: {err}");
                }
                return ExitCode::from(1);
            }
            // Re-run build_package to surface package counts in
            // the success line. Cheap (cooks already ran inside
            // cook_to_dir) and gives operators a quick read on
            // what landed in generated/.
            if let (Some(package), _) = build_package(&project, &project_root) {
                // Per-room residency counts: room world is
                // always RAM-required; deduped texture assets
                // (room materials + model atlases) are
                // VRAM-required; model meshes + clips bump RAM.
                let mut total_ram_refs: usize = 0;
                let mut total_vram_refs: usize = 0;
                for (i, r) in package.rooms.iter().enumerate() {
                    let mut ram_seen: Vec<usize> = vec![r.world_asset_index];
                    let mut vram_seen: Vec<usize> = Vec::new();
                    let first = r.material_first as usize;
                    let count = r.material_count as usize;
                    for m in &package.materials[first..first + count] {
                        if !vram_seen.contains(&m.texture_asset_index) {
                            vram_seen.push(m.texture_asset_index);
                        }
                    }
                    let i_u16 = i as u16;
                    let mut seen_models: Vec<u16> = Vec::new();
                    for inst in &package.model_instances {
                        if inst.room != i_u16 || seen_models.contains(&inst.model) {
                            continue;
                        }
                        seen_models.push(inst.model);
                        let model = &package.models[inst.model as usize];
                        if !ram_seen.contains(&model.mesh_asset_index) {
                            ram_seen.push(model.mesh_asset_index);
                        }
                        if let Some(atlas) = model.texture_asset_index {
                            if !vram_seen.contains(&atlas) {
                                vram_seen.push(atlas);
                            }
                        }
                        let cf = model.clip_first as usize;
                        let cc = model.clip_count as usize;
                        for clip in &package.model_clips[cf..cf + cc] {
                            if !ram_seen.contains(&clip.animation_asset_index) {
                                ram_seen.push(clip.animation_asset_index);
                            }
                        }
                    }
                    total_ram_refs += ram_seen.len();
                    total_vram_refs += vram_seen.len();
                }
                println!(
                    "[cook-playtest] Rooms: {}  Assets: {}  Textures: {}  Models: {}  Model instances: {}  Materials: {}  RAM residency refs: {}  VRAM residency refs: {}  Entities: {}",
                    package.rooms.len(),
                    package.assets.len(),
                    package.texture_asset_count(),
                    package.models.len(),
                    package.model_instances.len(),
                    package.materials.len(),
                    total_ram_refs,
                    total_vram_refs,
                    package.entities.len(),
                );
            }
            println!("[cook-playtest] wrote → {}", dir.display());
            println!("[cook-playtest] Build: make build-editor-playtest");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[cook-playtest] write failed: {e}");
            ExitCode::from(2)
        }
    }
}
