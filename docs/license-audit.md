# PSoXide License And Provenance Audit

Audit date: 2026-04-29

Scope: repository source, checked-in binary assets, generated artifacts,
third-party references, and public README media.

Publication decision: PSoXide is published as all three:

- PS1 emulator and debugger,
- PS1 SDK/runtime engine,
- editor and game prototype.

That is the right identity for the project, but it means the public
release bar is higher than for a single-purpose emulator or SDK. Code,
assets, docs, canaries, and generated outputs all need a coherent story.

## Current Position

Project Cargo manifests declare:

```text
GPL-2.0-or-later
```

No root `LICENSE-MIT`, `LICENSE-APACHE`, `LICENSE`, or `NOTICE` file is
currently present. That is a release blocker.

## Release Blockers

### 1. Root license files are missing

Add the standard MIT and Apache-2.0 license texts at the repo root, or
change the manifest declarations to the actual intended license before
publishing.

### 2. GPL-derived implementation risk

The codebase contains comments that describe direct ports or close
ports from PCSX-Redux, which is GPL-2.0-or-later. Examples found during
this audit include:

- `emu/crates/emulator-core/src/scheduler.rs`: "port of Redux"
- `emu/crates/emulator-core/src/gpu.rs`: "Direct port of Redux"
- `emu/crates/emulator-core/src/spu.rs`: Redux SPU files and ADSR tables
- `emu/crates/emulator-core/src/dma.rs`: "Direct port of Redux"
- `emu/crates/emulator-core/src/mdec.rs`: Redux AAN IDCT/Huffman path

If those comments accurately describe copied or translated GPL
implementation, the affected code cannot simply be released as
GPL-compatible. Choose one path before public release:

- relicense the affected emulator portions under a GPL-compatible
  license,
- obtain permission for a permissive relicense,
- or rewrite the affected code from public hardware documentation,
  test results, and clean-room notes.

Using PCSX-Redux as an external oracle is fine. Copying implementation
logic into permissively licensed code is the risk.

### 3. Bundled model and texture provenance gaps

The following checked-in cooked assets need source links, author names,
license names, and ideally original source files or regeneration notes:

- `assets/branding/logo-wordmark.svg`
- `assets/branding/logo-icon-player.svg`
- `assets/branding/logo-icon-player.png`
- `assets/models/obsidian_wraith/*`
- `assets/models/hooded_wretch/*`
- `editor/projects/default/assets/models/obsidian_wraith/*`
- `assets/textures/brick-wall.psxt`
- `assets/textures/floor.psxt`
- `editor/projects/default/assets/textures/*.psxt`

The default editor project currently depends on the Obsidian Wraith
model bundle, so this is not optional if the default project ships.
The README also depends on the branding assets, so their ownership
should be explicitly recorded before external publication.

### 4. Demo JPG attribution is incomplete

Tracked source JPGs:

- `sdk/examples/hello-tex/vendor/brick-wall.jpg`
- `sdk/examples/hello-tex/vendor/floor.jpg`
- `engine/examples/showcase-fog/vendor/brick-wall.jpg`
- `engine/examples/showcase-fog/vendor/floor.jpg`

`sdk/examples/hello-tex/vendor/README.md` names Pexels sources, but it
does not include exact URLs, download dates, Pexels license text/link,
or confirmation that the showcase-fog copies share the same provenance.

### 5. Public README media needs a clean capture record

The README uses current editor/play screenshots under `docs/media/`.
Before a public announcement, record:

- exact commit,
- whether the capture includes only project-owned or properly licensed
  assets,
- capture date,
- any shader/filter/settings differences from default.

Do not use commercial game screenshots or BIOS-logo screenshots as
public README media unless their legal status is explicitly accepted.

### 6. Canary screenshots may be redistribution-sensitive

Tracked files under `emu/crates/emulator-core/tests/milestones/` include
BIOS/commercial-game visual output. They are useful tests, but public
redistribution may be questionable. Consider replacing checked-in PNGs
with hashes, synthetic homebrew captures, or a private canary fixture
package.

### 7. Frontend font license files are missing

Tracked frontend fonts:

- `emu/crates/frontend/assets/fonts/VT323-Regular.ttf`
- `emu/crates/frontend/assets/fonts/lucide.ttf`

VT323 embeds SIL Open Font License metadata. Lucide is commonly shipped
under ISC, but this repo should include explicit upstream URLs and
license text for both fonts in a local provenance file.

### 8. SPU tone provenance is missing

Tracked ADPCM tone blobs:

- `sdk/crates/psx-spu/vendor/tone_sine.adpcm`
- `sdk/crates/psx-spu/vendor/tone_square.adpcm`
- `sdk/crates/psx-spu/vendor/tone_triangle.adpcm`
- `sdk/crates/psx-spu/vendor/tone_sawtooth.adpcm`

If generated locally, add the generator command/source. If imported,
add the upstream license.

### 9. Rust dependency license audit has not been run

Cargo manifests identify the local crate license, but this audit did
not resolve every crates.io dependency license across root, `editor`,
`emu`, `engine`, `sdk`, and `tools/*`. Run `cargo deny` or
`cargo about` across every workspace/tool and commit the report before
claiming a clean dependency license story.

## Lower-Risk Or Documented Items

- `sdk/crates/psx-font/vendor/PROVENANCE.md` documents font8x8 and IBM
  VGA font sources as Public Domain.
- `engine/examples/showcase-lights/vendor/cube.obj` is marked
  hand-authored public domain in the file header.
- `engine/examples/showcase-3d/vendor/teapot.obj` identifies itself as
  a simplified Utah Teapot. Add explicit public-domain/attribution
  notes before release.
- `engine/examples/showcase-3d/vendor/suzanne.obj` was generated by
  MeshLab, but the underlying Suzanne mesh provenance/license is not
  written down. Add it before release.
- `engine/examples/editor-playtest/generated/level_manifest.rs` is a
  tracked placeholder with no `include_bytes!`. Cooked generated rooms,
  textures, and models are ignored and regenerated.
- `/build/` is ignored and should stay untracked.

## Generated Artifacts

Expected public-source contract:

- Tracked: `engine/examples/editor-playtest/generated/level_manifest.rs`
  placeholder only.
- Ignored/regenerated: cooked generated `rooms/`, `textures/`, and
  `models/` directories under `engine/examples/editor-playtest/generated/`.
- Ignored/regenerated: `/build/examples/.../*.exe`.
- Tracked cooked demo assets: small `.psxt`, `.psxm`, `.psxmdl`, and
  `.psxanim` blobs that examples or the default project need from a
  clean clone.

The tracked cooked demo assets still need provenance because generated
binary blobs inherit the licensing of their source material.

## Third-Party References

References in docs/comments include:

- nocash PSX-SPX,
- PCSX-Redux,
- DuckStation,
- public PS1 hardware behavior notes.

Reference citations are fine. For public licensing, distinguish clearly
between:

- specifications and observations used to implement behavior,
- external tools used for testing,
- source code translated into this repository.

The third category is the one that changes license obligations.

## Recommended Pre-Publish Tasks

1. Add root license files and a `NOTICE` or `THIRD_PARTY.md`.
2. Decide how to resolve the PCSX-Redux/GPL-derived implementation
   risk.
3. Add `docs/asset-provenance.md` with every bundled asset, source URL,
   author, license, and regeneration command.
4. Add provenance files beside frontend fonts and SPU tone blobs.
5. Replace or private-gate BIOS/commercial-game golden PNGs.
6. Run dependency license tooling and commit the report.
7. Capture fresh README screenshots/video from a clean clone after the
   asset provenance is fixed.
