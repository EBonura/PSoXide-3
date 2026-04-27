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

use std::process::ExitCode;

use psxed_project::{
    playtest::{cook_to_dir, default_generated_dir},
    ProjectDocument,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let project = match args.first() {
        None => ProjectDocument::starter(),
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match ProjectDocument::from_ron_str(&text) {
                Ok(p) => p,
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
    match cook_to_dir(&project, &dir) {
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
            println!("[cook-playtest] wrote {} → {}", report.warnings.len(), dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[cook-playtest] write failed: {e}");
            ExitCode::from(2)
        }
    }
}
