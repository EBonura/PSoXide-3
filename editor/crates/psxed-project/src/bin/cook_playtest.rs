//! CLI: cook the editor's starter project (or a named project
//! file) into the playtest example's `generated/` directory.
//!
//! The editor's "Cook & Play" menu action calls the same
//! `psxed_project::playtest::cook_to_dir` underneath. This bin
//! exists so CI scripts and the Makefile can drive the cook
//! without spinning up the full GUI.
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
                let total_required_ram: usize = package.rooms.iter().map(|_| 1).sum();
                let total_required_vram: usize = package
                    .rooms
                    .iter()
                    .map(|r| {
                        let first = r.material_first as usize;
                        let count = r.material_count as usize;
                        let mut seen: Vec<usize> = Vec::with_capacity(count);
                        for m in &package.materials[first..first + count] {
                            if !seen.contains(&m.texture_asset_index) {
                                seen.push(m.texture_asset_index);
                            }
                        }
                        seen.len()
                    })
                    .sum();
                println!(
                    "[cook-playtest] Rooms: {}  Assets: {}  Textures: {}  Materials: {}  RAM residency refs: {}  VRAM residency refs: {}  Entities: {}",
                    package.rooms.len(),
                    package.assets.len(),
                    package.texture_asset_count(),
                    package.materials.len(),
                    total_required_ram,
                    total_required_vram,
                    package.entities.len(),
                );
            }
            println!("[cook-playtest] wrote → {}", dir.display());
            println!("[cook-playtest] Run: make run-editor-playtest");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[cook-playtest] write failed: {e}");
            ExitCode::from(2)
        }
    }
}
