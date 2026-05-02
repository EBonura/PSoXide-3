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
    if !is_supported_room_material_dimension(texture.width())
        || !is_supported_room_material_dimension(texture.height())
    {
        return Err(format!(
            "texture '{}' must be a power-of-two room material no larger than 64x64 texels and aligned to 8-texel texture-window units; found {}x{}",
            resource.name,
            texture.width(),
            texture.height(),
        ));
    }
    Ok(())
}

fn is_supported_room_material_dimension(size: u16) -> bool {
    size >= 8 && size <= 64 && size.is_power_of_two() && size % 8 == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProjectDocument, ResourceData};

    #[test]
    fn room_material_rejects_non_texture_window_dimensions() {
        let mut bytes =
            std::fs::read(crate::default_project_dir().join("assets/textures/floor.psxt"))
                .expect("starter floor texture exists");
        // AssetHeader is 12 bytes; TextureHeader width/height live at
        // payload offsets 2/4. Mutating only the dimensions is enough
        // to exercise the room-material contract.
        bytes[14..16].copy_from_slice(&48u16.to_le_bytes());
        let mut project = ProjectDocument::new("room-materials");
        let id = project.add_resource(
            "Odd Tile",
            ResourceData::Texture {
                psxt_path: "assets/textures/odd.psxt".to_string(),
            },
        );
        let resource = project.resource(id).expect("resource inserted");

        let error = expect_room_material_depth(&resource, &bytes).expect_err("48-wide rejected");
        assert!(error.contains("power-of-two room material"));
    }
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
