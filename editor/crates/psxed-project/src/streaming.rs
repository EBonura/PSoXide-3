//! Editor-side planning data for automatic streaming chunks.
//!
//! The authored tree still stores one Room grid today, but budget
//! reporting and future cook work should reason about generated
//! streamable chunks. Keeping that plan in the project crate lets
//! the editor, validation, and cooker converge on the same answer.

use std::collections::HashSet;

use crate::{
    GridDirection, NodeId, NodeKind, ProjectDocument, ResourceData, ResourceId, WorldGrid,
    WorldGridBudget, MAX_ROOM_BYTES, MAX_ROOM_DEPTH, MAX_ROOM_TRIANGLES, MAX_ROOM_WIDTH,
};

/// Tunable limits for generated streaming chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingChunkConfig {
    /// Preferred chunk width in grid sectors before hard caps force
    /// smaller splits.
    pub target_width: u16,
    /// Preferred chunk depth in grid sectors before hard caps force
    /// smaller splits.
    pub target_depth: u16,
    /// Absolute maximum chunk width accepted by the current cooker.
    pub max_width: u16,
    /// Absolute maximum chunk depth accepted by the current cooker.
    pub max_depth: u16,
    /// Absolute maximum triangle estimate accepted by the current cooker.
    pub max_triangles: usize,
    /// Absolute maximum static-lit `.psxw` room asset size accepted
    /// by Embedded Play.
    pub max_bytes: usize,
}

impl Default for StreamingChunkConfig {
    fn default() -> Self {
        Self {
            target_width: 16,
            target_depth: 16,
            max_width: MAX_ROOM_WIDTH,
            max_depth: MAX_ROOM_DEPTH,
            max_triangles: MAX_ROOM_TRIANGLES,
            max_bytes: MAX_ROOM_BYTES,
        }
    }
}

impl StreamingChunkConfig {
    fn normalized(self) -> Self {
        Self {
            target_width: self.target_width.max(1),
            target_depth: self.target_depth.max(1),
            max_width: self.max_width.max(1),
            max_depth: self.max_depth.max(1),
            max_triangles: self.max_triangles.max(1),
            max_bytes: self.max_bytes.max(1),
        }
    }

    fn over_budget(self, budget: &WorldGridBudget) -> bool {
        budget.width > self.max_width
            || budget.depth > self.max_depth
            || budget.triangles > self.max_triangles
            || budget.psxw_static_lit_bytes > self.max_bytes
    }
}

/// One generated streamable chunk inside an authored grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedChunk {
    /// Stable order within the plan.
    pub index: usize,
    /// Top-left chunk origin in room array-sector coordinates.
    pub array_origin: [u16; 2],
    /// Top-left chunk origin in world-cell coordinates.
    pub world_origin: [i32; 2],
    /// Chunk size in sectors.
    pub size: [u16; 2],
    /// Geometry/byte estimate for this chunk only.
    pub budget: WorldGridBudget,
    /// True when the chunk still violates a hard cap after splitting.
    pub over_budget: bool,
}

/// Deterministic chunk plan generated from one authored grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedChunkPlan {
    /// Config used to create the plan.
    pub config: StreamingChunkConfig,
    /// Source grid origin in world-cell coordinates.
    pub source_origin: [i32; 2],
    /// Source grid size in sectors.
    pub source_size: [u16; 2],
    /// Generated chunks in array-order traversal.
    pub chunks: Vec<GeneratedChunk>,
}

impl GeneratedChunkPlan {
    /// Number of generated chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of chunks still over hard limits after subdivision.
    pub fn over_budget_count(&self) -> usize {
        self.chunks.iter().filter(|chunk| chunk.over_budget).count()
    }

    /// Chunk with the largest base geometry `.psxw` byte estimate.
    pub fn largest_psxw_chunk(&self) -> Option<&GeneratedChunk> {
        self.chunks
            .iter()
            .max_by_key(|chunk| chunk.budget.psxw_bytes)
    }

    /// Chunk with the largest Embedded Play room asset estimate.
    pub fn largest_room_asset_chunk(&self) -> Option<&GeneratedChunk> {
        self.chunks
            .iter()
            .max_by_key(|chunk| chunk.budget.psxw_static_lit_bytes)
    }

    /// Chunk with the largest triangle estimate.
    pub fn largest_triangle_chunk(&self) -> Option<&GeneratedChunk> {
        self.chunks
            .iter()
            .max_by_key(|chunk| chunk.budget.triangles)
    }
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: u16,
    z: u16,
    width: u16,
    depth: u16,
}

/// Generate a deterministic streaming subdivision for one grid.
pub fn plan_generated_chunks(grid: &WorldGrid, config: StreamingChunkConfig) -> GeneratedChunkPlan {
    let config = config.normalized();
    let mut rects = Vec::new();
    let footprint = grid.authored_footprint();
    if let Some(footprint) = footprint {
        split_rect(
            grid,
            config,
            Rect {
                x: footprint.x,
                z: footprint.z,
                width: footprint.width,
                depth: footprint.depth,
            },
            &mut rects,
        );
    }

    let chunks = rects
        .into_iter()
        .enumerate()
        .filter_map(|(index, rect)| {
            let budget = grid.budget_for_rect(rect.x, rect.z, rect.width, rect.depth)?;
            Some(GeneratedChunk {
                index,
                array_origin: [rect.x, rect.z],
                world_origin: [
                    grid.origin[0] + rect.x as i32,
                    grid.origin[1] + rect.z as i32,
                ],
                size: [rect.width, rect.depth],
                over_budget: config.over_budget(&budget),
                budget,
            })
        })
        .collect();

    GeneratedChunkPlan {
        config,
        source_origin: footprint
            .map(|footprint| {
                [
                    grid.origin[0] + footprint.x as i32,
                    grid.origin[1] + footprint.z as i32,
                ]
            })
            .unwrap_or(grid.origin),
        source_size: footprint
            .map(|footprint| [footprint.width, footprint.depth])
            .unwrap_or([0, 0]),
        chunks,
    }
}

fn split_rect(grid: &WorldGrid, config: StreamingChunkConfig, rect: Rect, out: &mut Vec<Rect>) {
    let budget = grid
        .budget_for_rect(rect.x, rect.z, rect.width, rect.depth)
        .unwrap_or_default();
    let wants_size_split = rect.width > config.target_width || rect.depth > config.target_depth;
    let wants_budget_split = config.over_budget(&budget);
    let can_split = rect.width > 1 || rect.depth > 1;
    if (wants_size_split || wants_budget_split) && can_split {
        let split_x = split_on_x(rect, config);
        if split_x && rect.width > 1 {
            let first = rect.width / 2;
            let second = rect.width - first;
            split_rect(
                grid,
                config,
                Rect {
                    width: first,
                    ..rect
                },
                out,
            );
            split_rect(
                grid,
                config,
                Rect {
                    x: rect.x + first,
                    width: second,
                    ..rect
                },
                out,
            );
            return;
        }
        if rect.depth > 1 {
            let first = rect.depth / 2;
            let second = rect.depth - first;
            split_rect(
                grid,
                config,
                Rect {
                    depth: first,
                    ..rect
                },
                out,
            );
            split_rect(
                grid,
                config,
                Rect {
                    z: rect.z + first,
                    depth: second,
                    ..rect
                },
                out,
            );
            return;
        }
    }
    out.push(rect);
}

fn split_on_x(rect: Rect, config: StreamingChunkConfig) -> bool {
    if rect.width > config.target_width && rect.depth <= config.target_depth {
        return true;
    }
    if rect.depth > config.target_depth && rect.width <= config.target_width {
        return false;
    }
    if rect.width > config.max_width && rect.depth <= config.max_depth {
        return true;
    }
    if rect.depth > config.max_depth && rect.width <= config.max_width {
        return false;
    }
    rect.width >= rect.depth
}

/// Referenced runtime-facing resources for a scene or room.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SceneResourceUse {
    pub materials: Vec<ResourceId>,
    pub textures: Vec<ResourceId>,
    pub models: Vec<ResourceId>,
    pub meshes: Vec<ResourceId>,
    pub characters: Vec<ResourceId>,
    pub audio: Vec<ResourceId>,
    pub model_instances: usize,
    pub character_controllers: usize,
    pub colliders: usize,
    pub interactables: usize,
    pub lights: usize,
    pub triggers: usize,
    pub portals: usize,
}

/// Collect resources used by the active scene.
pub fn collect_scene_resource_use(project: &ProjectDocument) -> SceneResourceUse {
    collect_resource_use(project, None)
}

/// Collect resources used by one Room and its descendants.
pub fn collect_room_resource_use(project: &ProjectDocument, room_id: NodeId) -> SceneResourceUse {
    collect_resource_use(project, Some(room_id))
}

fn collect_resource_use(
    project: &ProjectDocument,
    room_filter: Option<NodeId>,
) -> SceneResourceUse {
    let scene = project.active_scene();
    let mut use_set = SceneResourceUse::default();
    let mut materials = HashSet::new();
    let mut textures = HashSet::new();
    let mut models = HashSet::new();
    let mut meshes = HashSet::new();
    let mut characters = HashSet::new();
    let mut audio = HashSet::new();

    for node in scene.nodes() {
        if let Some(room_id) = room_filter {
            if !scene.is_descendant_of(node.id, room_id) {
                continue;
            }
        }

        match &node.kind {
            NodeKind::Room { grid } => {
                collect_grid_resources(grid, &mut use_set, &mut materials);
            }
            NodeKind::MeshInstance { mesh, material, .. } => {
                push_material(*material, &mut use_set, &mut materials);
                if let Some(mesh_id) = mesh {
                    use_set.model_instances += 1;
                    match project.resource(*mesh_id).map(|resource| &resource.data) {
                        Some(ResourceData::Model(_)) => {
                            push_unique(*mesh_id, &mut use_set.models, &mut models)
                        }
                        Some(ResourceData::Mesh { .. }) => {
                            push_unique(*mesh_id, &mut use_set.meshes, &mut meshes)
                        }
                        _ => {}
                    }
                }
            }
            NodeKind::ModelRenderer { model, material } => {
                push_material(*material, &mut use_set, &mut materials);
                if let Some(model_id) = model {
                    use_set.model_instances += 1;
                    push_unique(*model_id, &mut use_set.models, &mut models);
                }
            }
            NodeKind::CharacterController { character, .. } => {
                use_set.character_controllers += 1;
                push_character_model(
                    project,
                    *character,
                    &mut use_set,
                    &mut characters,
                    &mut models,
                );
            }
            NodeKind::SpawnPoint { character, .. } => {
                push_character_model(
                    project,
                    *character,
                    &mut use_set,
                    &mut characters,
                    &mut models,
                );
            }
            NodeKind::AudioSource { sound, .. } => {
                if let Some(audio_id) = sound {
                    push_unique(*audio_id, &mut use_set.audio, &mut audio);
                }
            }
            NodeKind::Collider { .. } => use_set.colliders += 1,
            NodeKind::Interactable { .. } => use_set.interactables += 1,
            NodeKind::PointLight { .. } => use_set.lights += 1,
            NodeKind::Trigger { .. } => use_set.triggers += 1,
            NodeKind::Portal { .. } => use_set.portals += 1,
            _ => {}
        }
    }

    for material_id in use_set.materials.clone() {
        let Some(resource) = project.resource(material_id) else {
            continue;
        };
        if let ResourceData::Material(material) = &resource.data {
            if let Some(texture_id) = material.texture {
                push_unique(texture_id, &mut use_set.textures, &mut textures);
            }
        }
    }

    use_set
}

fn collect_grid_resources(
    grid: &WorldGrid,
    use_set: &mut SceneResourceUse,
    materials: &mut HashSet<ResourceId>,
) {
    for sector in grid.sectors.iter().flatten() {
        if let Some(face) = &sector.floor {
            push_material(face.material, use_set, materials);
        }
        if let Some(face) = &sector.ceiling {
            push_material(face.material, use_set, materials);
        }
        for direction in GridDirection::ALL {
            for wall in sector.walls.get(direction) {
                push_material(wall.material, use_set, materials);
            }
        }
    }
}

fn push_material(
    id: Option<ResourceId>,
    use_set: &mut SceneResourceUse,
    materials: &mut HashSet<ResourceId>,
) {
    if let Some(id) = id {
        push_unique(id, &mut use_set.materials, materials);
    }
}

fn push_character_model(
    project: &ProjectDocument,
    character: Option<ResourceId>,
    use_set: &mut SceneResourceUse,
    characters: &mut HashSet<ResourceId>,
    models: &mut HashSet<ResourceId>,
) {
    let Some(character_id) = character else {
        return;
    };
    push_unique(character_id, &mut use_set.characters, characters);
    let Some(resource) = project.resource(character_id) else {
        return;
    };
    let ResourceData::Character(character) = &resource.data else {
        return;
    };
    if let Some(model_id) = character.model {
        push_unique(model_id, &mut use_set.models, models);
    }
}

fn push_unique(id: ResourceId, out: &mut Vec<ResourceId>, seen: &mut HashSet<ResourceId>) {
    if seen.insert(id) {
        out.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CharacterResource, MaterialResource, NodeKind, ResourceData};

    #[test]
    fn small_grid_stays_one_generated_chunk() {
        let grid = WorldGrid::stone_room(4, 4, 1024, None, None);
        let plan = plan_generated_chunks(&grid, StreamingChunkConfig::default());

        assert_eq!(plan.chunk_count(), 1);
        assert_eq!(plan.chunks[0].array_origin, [0, 0]);
        assert_eq!(plan.chunks[0].size, [4, 4]);
        assert_eq!(plan.over_budget_count(), 0);
    }

    #[test]
    fn wide_grid_splits_to_target_sized_chunks() {
        let grid = WorldGrid::stone_room(40, 16, 1024, None, None);
        let plan = plan_generated_chunks(&grid, StreamingChunkConfig::default());

        assert!(plan.chunk_count() > 1);
        assert!(plan
            .chunks
            .iter()
            .all(|chunk| chunk.size[0] <= 16 && chunk.size[1] <= 16));
        assert_eq!(
            plan.chunks
                .iter()
                .map(|chunk| chunk.budget.total_cells)
                .sum::<usize>(),
            40 * 16
        );
    }

    #[test]
    fn generated_chunks_use_authored_footprint() {
        let mut grid = WorldGrid::empty(40, 16, 1024);
        grid.set_floor(10, 4, 0, None);
        grid.set_floor(13, 6, 0, None);
        let plan = plan_generated_chunks(&grid, StreamingChunkConfig::default());

        assert_eq!(plan.chunk_count(), 1);
        assert_eq!(plan.source_origin, [10, 4]);
        assert_eq!(plan.source_size, [4, 3]);
        assert_eq!(plan.chunks[0].array_origin, [10, 4]);
        assert_eq!(plan.chunks[0].world_origin, [10, 4]);
        assert_eq!(plan.chunks[0].size, [4, 3]);
        assert_eq!(plan.chunks[0].budget.total_cells, 12);
    }

    #[test]
    fn budget_for_rect_counts_only_requested_area() {
        let floor = ResourceId(1);
        let mut grid = WorldGrid::empty(4, 4, 1024);
        grid.set_floor(0, 0, 0, Some(floor));
        grid.set_floor(3, 3, 0, Some(floor));

        let left = grid.budget_for_rect(0, 0, 2, 4).unwrap();
        let right = grid.budget_for_rect(2, 0, 2, 4).unwrap();

        assert_eq!(left.total_cells, 8);
        assert_eq!(right.total_cells, 8);
        assert_eq!(left.floors, 1);
        assert_eq!(right.floors, 1);
    }

    #[test]
    fn scene_resource_use_follows_components_and_material_textures() {
        let mut project = ProjectDocument::new("test");
        let texture = project.add_resource(
            "atlas",
            ResourceData::Texture {
                psxt_path: "atlas.psxt".to_string(),
            },
        );
        let material = project.add_resource(
            "mat",
            ResourceData::Material(MaterialResource::opaque(Some(texture))),
        );
        let model = project.add_resource(
            "model",
            ResourceData::Model(crate::ModelResource {
                model_path: "model.psxmdl".to_string(),
                texture_path: None,
                skeleton: None,
                clips: Vec::new(),
                default_clip: None,
                preview_clip: None,
                world_height: 1024,
                scale_q8: [crate::MODEL_SCALE_ONE_Q8; 3],
                attachments: Vec::new(),
            }),
        );
        let character = project.add_resource(
            "character",
            ResourceData::Character(CharacterResource {
                model: Some(model),
                ..CharacterResource::default()
            }),
        );

        let scene = project.active_scene_mut();
        let room = scene.add_node(
            scene.root,
            "Room",
            NodeKind::Room {
                grid: WorldGrid::stone_room(2, 2, 1024, Some(material), None),
            },
        );
        let entity = scene.add_node(room, "Entity", NodeKind::Entity);
        scene.add_node(
            entity,
            "Controller",
            NodeKind::CharacterController {
                character: Some(character),
                player: true,
            },
        );
        scene.add_node(
            entity,
            "Renderer",
            NodeKind::ModelRenderer {
                model: Some(model),
                material: None,
            },
        );

        let use_set = collect_scene_resource_use(&project);

        assert_eq!(use_set.materials, vec![material]);
        assert_eq!(use_set.textures, vec![texture]);
        assert_eq!(use_set.models, vec![model]);
        assert_eq!(use_set.characters, vec![character]);
        assert_eq!(use_set.model_instances, 1);
        assert_eq!(use_set.character_controllers, 1);
    }
}
