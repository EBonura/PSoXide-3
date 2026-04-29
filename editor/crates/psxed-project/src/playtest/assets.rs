//! Asset and resource loading helpers for playtest cooking.

use super::*;
use crate::Resource;

pub(super) fn resolve_path(stored: &str, project_root: &Path) -> PathBuf {
    if Path::new(stored).is_absolute() {
        PathBuf::from(stored)
    } else {
        project_root.join(stored)
    }
}

/// Strip a free-form name down to a filesystem-safe stem.
pub(super) fn sanitise_model_dirname(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_was_sep = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed
    }
}

pub(super) fn find_resource(project: &ProjectDocument, id: ResourceId) -> Option<&Resource> {
    project.resources.iter().find(|r| r.id == id)
}

/// Validate a room-material `.psxt` blob is 4bpp (16-entry
/// CLUT). Both the editor preview material upload path and the
/// runtime room material slots assume 4bpp; other depths
/// render with wrong colours.
pub(super) fn expect_room_material_depth(resource: &Resource, bytes: &[u8]) -> Result<(), String> {
    let texture = psx_asset::Texture::from_bytes(bytes)
        .map_err(|e| format!("texture '{}' parse failed: {e:?}", resource.name))?;
    if texture.clut_entries() != 16 {
        return Err(format!(
            "texture '{}' must be 4bpp (16-entry CLUT) for room materials; found {} entries",
            resource.name,
            texture.clut_entries(),
        ));
    }
    Ok(())
}

/// Read the texture's `.psxt` bytes from disk. Resolves
/// `psxt_path` first as-is (absolute paths), then relative to
/// `project_root`. Returns a string error rather than `io::Error`
/// so callers can prepend room/material context.
pub(super) fn load_texture_bytes(
    resource: &Resource,
    project_root: &Path,
) -> Result<Vec<u8>, String> {
    let ResourceData::Texture { psxt_path } = &resource.data else {
        return Err(format!(
            "resource '{}' (#{}) is not a Texture",
            resource.name,
            resource.id.raw(),
        ));
    };
    if psxt_path.is_empty() {
        return Err(format!(
            "texture resource '{}' has empty path",
            resource.name
        ));
    }
    let path = if Path::new(psxt_path).is_absolute() {
        PathBuf::from(psxt_path)
    } else {
        project_root.join(psxt_path)
    };
    std::fs::read(&path).map_err(|e| {
        format!(
            "failed to read texture '{}' at {}: {e}",
            resource.name,
            path.display(),
        )
    })
}
