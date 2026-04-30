//! Cooks the editor's starter room into `OUT_DIR/room.psxw` so
//! the example can `include_bytes!` it at compile time.
//!
//! Also pins the slot-ordering invariant the runtime side bakes
//! in -- slot 0 must be the floor material, slot 1 the brick-wall
//! material. If a future cooker reshape flips the order, this
//! build fails loud and the runtime swap is a one-line patch.

use psxed_project::{
    world_cook::{cook_world_grid, encode_world_grid_psxw},
    NodeKind, ProjectDocument, ResourceData,
};

const ROOM_PSXW: &str = "room.psxw";
const FLOOR_PSXT_SUFFIX: &str = "floor.psxt";
const BRICK_PSXT_SUFFIX: &str = "brick-wall.psxt";

fn main() {
    // The starter is baked into psxed-project at compile time
    // (see DEFAULT_PROJECT_RON), so cargo's automatic
    // build-dependency tracking re-runs us when that crate
    // changes. No explicit rerun-if-changed needed beyond Cargo's
    // default.
    let project = ProjectDocument::starter();

    let grid = project
        .active_scene()
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            NodeKind::Room { grid } => Some(grid.clone()),
            _ => None,
        })
        .expect("starter project must contain a Room node");

    let cooked = cook_world_grid(&project, &grid).expect("starter grid cooks cleanly");
    assert_slot_ordering(&project, &cooked);

    let bytes = encode_world_grid_psxw(&project, &grid).expect("starter grid encodes cleanly");

    let out_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_path = std::path::Path::new(&out_dir).join(ROOM_PSXW);
    std::fs::write(&out_path, bytes).expect("write room.psxw to OUT_DIR");
}

/// The runtime-side example expects slot 0 = floor texture,
/// slot 1 = brick-wall texture. The cooker assigns slots in
/// first-use order while iterating sectors `[x * depth + z]`,
/// so the starter (floor first, walls second) yields exactly
/// that order. Pin it.
fn assert_slot_ordering(
    project: &ProjectDocument,
    cooked: &psxed_project::world_cook::CookedWorldGrid,
) {
    assert_eq!(
        cooked.materials.len(),
        2,
        "starter cook must yield exactly 2 material slots, got {}",
        cooked.materials.len()
    );
    expect_slot_texture(project, cooked, 0, FLOOR_PSXT_SUFFIX);
    expect_slot_texture(project, cooked, 1, BRICK_PSXT_SUFFIX);
}

fn expect_slot_texture(
    project: &ProjectDocument,
    cooked: &psxed_project::world_cook::CookedWorldGrid,
    slot: u16,
    suffix: &str,
) {
    let entry = &cooked.materials[slot as usize];
    assert_eq!(entry.slot, slot, "cooked material[{slot}].slot drifted");
    let texture_id = entry
        .texture
        .unwrap_or_else(|| panic!("slot {slot} material has no texture"));
    let texture = project
        .resource(texture_id)
        .unwrap_or_else(|| panic!("slot {slot} texture id missing from resources"));
    let ResourceData::Texture { psxt_path } = &texture.data else {
        panic!("slot {slot} resource is not a Texture");
    };
    assert!(
        psxt_path.ends_with(suffix),
        "slot {slot} expected texture ending with {suffix}, got {psxt_path}"
    );
}
