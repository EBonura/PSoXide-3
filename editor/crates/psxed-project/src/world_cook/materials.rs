//! Material resource resolution for cooked world grids.

use super::*;

pub(super) fn material_slot(
    project: &ProjectDocument,
    material: Option<ResourceId>,
    x: u16,
    z: u16,
    face: WorldGridFaceKind,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<u16, WorldGridCookError> {
    let id = material.ok_or(WorldGridCookError::UnassignedMaterial { x, z, face })?;
    if let Some(slot) = material_slots.get(&id).copied() {
        return Ok(slot);
    }

    let resource = project
        .resources
        .iter()
        .find(|resource| resource.id == id)
        .ok_or(WorldGridCookError::MissingMaterial { id })?;
    let ResourceData::Material(material) = &resource.data else {
        return Err(WorldGridCookError::ResourceIsNotMaterial { id });
    };
    if materials.len() >= u16::MAX as usize {
        return Err(WorldGridCookError::TooManyMaterials {
            count: materials.len() + 1,
        });
    }

    let slot = materials.len() as u16;
    materials.push(CookedWorldMaterial::from_resource(slot, id, material));
    material_slots.insert(id, slot);
    Ok(slot)
}
