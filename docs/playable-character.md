# Playable character

How a `Character` resource flows from authoring → cooked
manifest → editor preview + editor-playtest runtime so a
designer can press **Play** and walk a textured animated
character around an authored room.

## Scope of this pass

- One **Character** resource = backing Model + role-clip mapping
  + capsule + camera defaults.
- One **Player Spawn** references one Character via its
  `character` field.
- Runtime renders that character at the spawn, lets the user
  walk in third-person, switches idle / walk / run animations
  based on input, rejects movement into empty cells, and drives a
  collision-aware third-person camera.

Out of scope (deliberately): enemies, AI, combat, jumping,
animation blending, keyframe editing, runtime streaming, portal
traversal, per-face runtime lighting on characters, model lighting
at runtime.

## Authoring flow

1. **Add a Character resource**: New Resource → Character. The
   inspector exposes a Model picker, four role-clip selectors
   (idle / walk / run / turn), capsule (radius, height),
   controller (walk / run / turn speed), and camera (distance,
   height, target height) sections. Camera height and target height
   are positive upward offsets from the player root.
2. **Pick a Model**: any `ResourceData::Model` resource is
   allowed. The clip pickers populate from the model's clip
   list once a model is bound.
3. **Assign idle + walk clips**: required for the player.
   Optional: run, turn. The "Auto Assign Clips By Name"
   button matches `idle` / `walk` / `run` / `turn` substrings
   case-insensitively against clip names -- works when the
   source bundle uses conventional naming.
4. **Wire the spawn**: select the Player Spawn node; the
   inspector shows a Character dropdown when `player == true`.
   Pick the Character. If exactly one Character resource
   exists project-wide, leaving this unset auto-picks it at
   cook time with a warning.
5. **Play**: the toolbar action validates the project, cooks the
   manifest, builds `editor-playtest`, and side-loads it into the
   editor viewport.

The starter project ships a "Wraith Hero" Character bound to
"Obsidian Wraith" - a fresh clone Plays into a walking player
with no manual setup.

## Resource shape

```rust
pub struct CharacterResource {
    pub model: Option<ResourceId>,

    pub idle_clip: Option<u16>,
    pub walk_clip: Option<u16>,
    pub run_clip: Option<u16>,   // optional
    pub turn_clip: Option<u16>,  // optional

    pub radius: u16,             // engine units
    pub height: u16,             // engine units

    pub walk_speed: i32,         // engine units / 60 Hz frame
    pub run_speed: i32,
    pub turn_speed_degrees_per_second: u16,

    pub camera_distance: i32,
    pub camera_height: i32,
    pub camera_target_height: i32,
}
```

The `model` field can stay `None` while authoring; cook
validation rejects a player whose Character resolves to a
missing model. Capsule + speed + camera fields have sensible
defaults via `CharacterResource::defaults()`.

## Wire format (psx-level)

Cook output emits two static slices the runtime consumes:

```rust
pub static CHARACTERS: &[LevelCharacterRecord] = &[ ... ];
pub static PLAYER_CONTROLLER: Option<PlayerControllerRecord> = Some(...);
```

`LevelCharacterRecord` mirrors `CharacterResource` field-for-
field, with `Option<u16>` clip slots flattened to plain `u16`
where `u16::MAX` (= `CHARACTER_CLIP_NONE`) means "no clip
authored for this role".

`PlayerControllerRecord` carries the resolved `PlayerSpawnRecord`
and a Character index -- one record per cooked package.

Models that drive a Character are deduplicated against placed
`MeshInstance` references: a project where the same Wraith model
is both placed at room centre and assigned to the player only
emits one `LevelModelRecord`.

## Editor preview

The 3D viewport renders the character at the Player Spawn,
animated with its idle clip, using the same model render path
placed model instances follow. If the spawn has no Character
assigned and exactly one Character exists project-wide, the
preview falls back to that Character -- matching the cooker's
auto-pick rule. Spawn selection / drag still operates on the
SpawnPoint node.

## Runtime (editor-playtest)

`engine/examples/editor-playtest/src/main.rs` resolves the
player at startup:

1. Read `PLAYER_CONTROLLER`. `None` → fall back to debug
   camera + invisible player (placeholder manifest path).
2. Read `CHARACTERS[pc.character]`, decode into a
   `RuntimeCharacter` (degrees-per-second turn rate is converted
   to per-frame quanta up front so the per-frame movement code
   is a wrapping add).
3. Initialize player position / yaw from `pc.spawn`.

Per-frame update:

- Left stick, or D-pad as fallback: camera-relative movement.
- No movement: `Idle` animation.
- Normal movement: `Walk` animation at `walk_speed`.
- CIRCLE held while moving: `Run` animation at `run_speed`. If the
  character has no run clip, the walk animation plays at run speed.
- SELECT: toggle a free-orbit debug camera.
- Right stick: manual third-person camera orbit when the pad is in
  analog mode.
- L1: recenter the third-person camera behind the player.
- R3: hard-lock / unlock the most central entity target in range.
- L2 / R2: switch lock-on target left / right.
- Soft-lock: when no hard lock is active, the camera can bias
  toward a central target in range; strong right-stick input
  suppresses it until the player leaves and re-enters range.

Movement is owned by `psx_engine::CharacterMotorState`: game or AI
code feeds abstract intent (`turn`, `walk`, `sprint`, `evade`), and
the motor advances position / yaw / stamina / action state. Movement
is rejected when the destination capsule sample has no walkable
floor -- coarse but enough to keep the player inside the room until
proper capsule sliding lands. The committed Y position follows the
sampled floor height under the player's root.

Animation state for editor-playtest is currently `Idle` / `Walk` / `Run`.
State changes reset the animation phase so transitions don't pop
into the middle of a clip.

Camera: `psx_engine::ThirdPersonCameraState` owns the follow rig.
It starts from the Character camera defaults, placing the focus and
camera at positive upward offsets from the player's floor/root
position, then applies:

- manual input cooldown, so right-stick orbit does not fight
  automatic re-alignment;
- automatic re-alignment behind the player while moving;
- lock-on framing, where the focus is biased between the player and
  the hard- or soft-locked target and vertical pitch is computed
  from that focus;
- position / focus / distance lag for rubber-band motion;
- a fixed fan of whisker probes against `RoomCollision`, first
  trying to rotate around blocking wall geometry, then pulling the
  camera closer when no clear orbit exists.

The implementation is PS1-shaped: no allocation, bounded ray work,
integer math, and direct sampling against the cooked grid room.

## Validation (cook-time)

Hard errors:
- No player spawn or multiple player spawns.
- Player spawn with no Character and either zero or multiple
  Character resources defined.
- Character references missing / non-Character resource.
- Character has no Model assigned.
- Character's model is invalid / has no atlas / has no clips.
- Idle clip unset or out-of-range.
- Walk clip unset or out-of-range.
- Radius / height / walk_speed / turn_speed / camera_distance
  not positive.
- Camera height / target_height negative.

Warnings:
- Run clip missing → falls back to walk at run speed.
- Turn clip missing → currently unused at runtime.
- Auto-picked Character (player spawn had no explicit
  assignment but exactly one Character exists).

## Currently out of scope

- Multiple Character types at once (NPCs, enemies).
- Animation blending or transition crossfades.
- Keyframe editing inside the editor.
- Enemy-specific lock targets. The current vertical slice uses
  cooked `EntityRecord` markers as lock-on targets until enemies
  have their own runtime records.
- Occlusion fades when collision pulls the camera very close.
- Per-face runtime character lighting (room-level Q8 ambient
  applies to room surfaces only).
- Capsule wall sliding and step-height limits.
- Wall-slide collision response.
- Jump / crouch / strafe actions.
- Portal / room-to-room transitions.
- Streaming or async asset baking.

These are natural follow-ups once the base controller settles.

## See also

- `editor/crates/psxed-project/src/lib.rs` -- `CharacterResource`,
  `ResourceData::Character`, `NodeKind::SpawnPoint::character`.
- `editor/crates/psxed-project/src/playtest.rs` -- cook-side
  `PlaytestCharacter` + `PlaytestPlayerController` + validation.
- `engine/crates/psx-level/src/lib.rs` -- wire structs.
- `engine/examples/editor-playtest/src/main.rs` -- runtime
  `RuntimeCharacter`, `PlayerAnim`, `draw_player`,
  `Playtest::can_stand_at`.
- `emu/crates/frontend/src/editor_preview.rs` --
  `walk_player_spawn_preview` for the editor 3D viewport.
- [`docs/editor-model-authoring.md`](editor-model-authoring.md) --
  Model resource shape (Characters layer on top of these).
- [`docs/editor-architecture.md`](editor-architecture.md) --
  scene tree, selection, drag.
