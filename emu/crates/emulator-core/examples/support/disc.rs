#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use psx_iso::Disc;

pub fn load_disc_path(path: &Path) -> Result<Disc, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cue"))
    {
        psoxide_settings::library::load_disc_from_cue(path)
    } else {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Disc::from_bin(bytes))
    }
}

pub fn discover_cue_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    visit_for_cues(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn visit_for_cues(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_file() {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("cue"))
        {
            out.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !meta.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
        let entry = entry.map_err(|e| format!("{}: {e}", path.display()))?;
        visit_for_cues(&entry.path(), out)?;
    }
    Ok(())
}
