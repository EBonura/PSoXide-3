//! Shared editor/cooker resource resolution rules.

use crate::{
    AnimationRole, CharacterResource, ModelResource, ProjectDocument, ResourceData, ResourceId,
};

/// A player spawn's resolved Character resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedSpawnCharacter {
    /// Character resource id.
    pub id: ResourceId,
    /// `true` when the id came from the "only Character in project"
    /// authoring convenience rather than an explicit spawn setting.
    pub auto_picked: bool,
}

/// Why a spawn's Character could not be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnCharacterResolutionError {
    /// Explicit resource id does not exist.
    MissingExplicit(ResourceId),
    /// Explicit resource exists but is not a Character.
    ExplicitNotCharacter(ResourceId),
    /// No explicit assignment and the project defines no Characters.
    NoCharacters,
    /// No explicit assignment and multiple Characters are available.
    AmbiguousCharacters {
        /// Number of Character resources found.
        count: usize,
    },
}

/// Resolve the Character that should drive a player spawn.
///
/// Rule: explicit assignment wins; otherwise exactly one Character in
/// the project is auto-picked. Ambiguity is an error so the cook and
/// preview cannot silently disagree.
pub fn resolve_spawn_character(
    project: &ProjectDocument,
    explicit: Option<ResourceId>,
) -> Result<ResolvedSpawnCharacter, SpawnCharacterResolutionError> {
    if let Some(id) = explicit {
        let Some(resource) = project.resource(id) else {
            return Err(SpawnCharacterResolutionError::MissingExplicit(id));
        };
        return match &resource.data {
            ResourceData::Character(_) => Ok(ResolvedSpawnCharacter {
                id,
                auto_picked: false,
            }),
            _ => Err(SpawnCharacterResolutionError::ExplicitNotCharacter(id)),
        };
    }

    let mut found: Option<ResourceId> = None;
    let mut count = 0usize;
    for resource in &project.resources {
        if matches!(resource.data, ResourceData::Character(_)) {
            found = Some(resource.id);
            count += 1;
        }
    }

    match (count, found) {
        (1, Some(id)) => Ok(ResolvedSpawnCharacter {
            id,
            auto_picked: true,
        }),
        (0, _) => Err(SpawnCharacterResolutionError::NoCharacters),
        (count, _) => Err(SpawnCharacterResolutionError::AmbiguousCharacters { count }),
    }
}

/// Clip shown for a placed Model instance in the editor preview.
pub fn resolve_model_instance_preview_clip(
    model: &ModelResource,
    override_clip: Option<u16>,
) -> Option<u16> {
    override_clip
        .or(model.effective_preview_clip())
        .or_else(|| {
            if model.clips.is_empty() {
                None
            } else {
                Some(0)
            }
        })
}

/// Clip shown for a player spawn's Character in the editor preview.
pub fn resolve_character_idle_preview_clip(
    character: &CharacterResource,
    model: &ModelResource,
) -> Option<u16> {
    character
        .idle_clip
        .or(model.effective_preview_clip())
        .or_else(|| (!model.clips.is_empty()).then_some(0))
}

/// Clip shown for a player spawn's Character in the editor preview
/// when the project has standalone Animation Set resources.
pub fn resolve_character_idle_preview_clip_for_model(
    project: &ProjectDocument,
    character: &CharacterResource,
    model_id: ResourceId,
    model: &ModelResource,
) -> Option<u16> {
    if let Some(animation_id) = character.animation_set.and_then(|set_id| {
        project
            .resource(set_id)
            .and_then(|resource| match &resource.data {
                ResourceData::AnimationSet(set) => set.role_clip(AnimationRole::Idle),
                _ => None,
            })
    }) {
        if let Some(index) = project.resolved_model_animation_index(model_id, animation_id) {
            return Some(index);
        }
    }
    resolve_character_idle_preview_clip(character, model)
        .or_else(|| (!project.resolved_model_animation_clips(model_id).is_empty()).then_some(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CharacterResource;

    #[test]
    fn spawn_character_auto_picks_single_character() {
        let mut project = ProjectDocument::starter();
        project
            .resources
            .retain(|r| matches!(r.data, ResourceData::Character(_)));
        assert_eq!(project.resources.len(), 1);

        let resolved = resolve_spawn_character(&project, None).expect("single character resolves");
        assert!(resolved.auto_picked);
        assert_eq!(resolved.id, project.resources[0].id);
    }

    #[test]
    fn spawn_character_reports_ambiguity() {
        let mut project = ProjectDocument::starter();
        project.add_resource(
            "Second Character",
            ResourceData::Character(CharacterResource::defaults()),
        );

        assert_eq!(
            resolve_spawn_character(&project, None),
            Err(SpawnCharacterResolutionError::AmbiguousCharacters { count: 2 })
        );
    }
}
