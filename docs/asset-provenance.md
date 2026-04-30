# Asset and Media Provenance

This document records the source, author, and license for every
tracked binary asset that ships in this repository. PSoXide's *code*
is licensed under GPL-2.0-or-later (see [`LICENSE`](../LICENSE)), but
binary assets each carry their own provenance and license, recorded
here.

Last updated: 2026-04-30.

## Branding

| File | Source | Author | License |
| --- | --- | --- | --- |
| `assets/branding/logo-wordmark.svg` | Original work | Emanuele Bonura | GPL-2.0-or-later |
| `assets/branding/logo-icon-player.svg` | Original work | Emanuele Bonura | GPL-2.0-or-later |
| `assets/branding/logo-icon-player.png` | Original work | Emanuele Bonura | GPL-2.0-or-later |

## 3D Models

The two character models bundled in the repo were generated using
**[Meshy](https://www.meshy.ai/)**, an AI 3D-model generation service.
They are placeholder / development / proof-of-concept assets used by
the engine showcase example and by the default editor project to
exercise the runtime model pipeline. They are not intended as final
shippable game content.

The models were generated under a paid Meshy subscription and are
marked private in the Meshy account. Meshy's current
[pricing](https://www.meshy.ai/pricing) and
[help](https://help.meshy.ai/en/articles/9992022-can-i-sell-the-models-on-other-platforms)
pages describe Pro/Studio/Enterprise generated assets as "private &
customer owned", and state that premium subscribers own the assets they
create with Meshy and may distribute or sell them. Keep the subscription
receipt / invoice and export metadata with the project records so this
remains auditable.

| Directory | Source | License |
| --- | --- | --- |
| `assets/models/obsidian_wraith/*` | Generated via Meshy (AI), paid subscription, private asset | User-owned per Meshy premium plan terms; distributed here under GPL-2.0-or-later |
| `assets/models/hooded_wretch/*` | Generated via Meshy (AI), paid subscription, private asset | User-owned per Meshy premium plan terms; distributed here under GPL-2.0-or-later |
| `editor/projects/default/assets/models/obsidian_wraith/*` | Cooked copy of the above for the default editor project | Same as above |

The two models share the same Meshy biped rig (24 joints), as already
noted in
[`engine/examples/showcase-model/src/main.rs`](../engine/examples/showcase-model/src/main.rs).

Note: Meshy's terms still make the user responsible for ensuring input
materials do not violate third-party rights. These models should remain
acceptable as bundled assets only if their prompts/reference inputs
were rights-clean.

## Textures

Source JPGs are bundled at 512×512 in the example `vendor/`
directories. The `.psxt` files alongside the assets are cooked
(quantised, palettised, VRAM-packed) outputs derived from those JPGs
via the editor's texture cooker (`make assets`).

| File | Source | Author | License |
| --- | --- | --- | --- |
| `sdk/examples/hello-tex/vendor/brick-wall.jpg` | "Brick Wall" via [Pexels](https://www.pexels.com/license/) | Michael Laut | Pexels License |
| `sdk/examples/hello-tex/vendor/floor.jpg` | "Batako Wall Texture Street" via [Pexels](https://www.pexels.com/license/) | (uncredited Pexels contributor) | Pexels License |
| `engine/examples/showcase-fog/vendor/brick-wall.jpg` | Byte-identical copy of `sdk/examples/hello-tex/vendor/brick-wall.jpg` | Michael Laut | Pexels License |
| `engine/examples/showcase-fog/vendor/floor.jpg` | Byte-identical copy of `sdk/examples/hello-tex/vendor/floor.jpg` | (uncredited Pexels contributor) | Pexels License |
| `assets/textures/brick-wall.psxt` | Cooked from `brick-wall.jpg` | (derived) | Pexels License (inherited) |
| `assets/textures/floor.psxt` | Cooked from `floor.jpg` | (derived) | Pexels License (inherited) |
| `editor/projects/default/assets/textures/brick-wall.psxt` | Cooked copy of `brick-wall.jpg` for the default editor project | (derived) | Pexels License (inherited) |
| `editor/projects/default/assets/textures/floor.psxt` | Cooked copy of `floor.jpg` for the default editor project | (derived) | Pexels License (inherited) |
| `assets/models/*/*.psxt` (model textures) | See "3D Models" above | (Meshy) | User-owned per Meshy premium plan terms; distributed here under GPL-2.0-or-later |

The Pexels License (https://www.pexels.com/license/) permits free
commercial and non-commercial use, modification, and redistribution;
attribution is appreciated but not required.

**TODO before public release**: locate and record the exact Pexels
URLs for both source images. The originals were downloaded without
the URLs being captured at the time;
[hello-tex/vendor/README.md](../sdk/examples/hello-tex/vendor/README.md)
records the titles and Pexels as origin but not the URLs.

## Fonts

See also [`emu/crates/frontend/assets/fonts/PROVENANCE.md`](../emu/crates/frontend/assets/fonts/PROVENANCE.md).

| File | Source | Author | License |
| --- | --- | --- | --- |
| `emu/crates/frontend/assets/fonts/VT323-Regular.ttf` | [Google Fonts](https://fonts.google.com/specimen/VT323) | Peter Hull | SIL Open Font License 1.1 |
| `emu/crates/frontend/assets/fonts/lucide.ttf` | [lucide.dev](https://lucide.dev/) | Lucide Contributors | ISC |
| `sdk/crates/psx-font/vendor/font8x8.bin` | font8x8 (PD) | dhepper | Public Domain |
| `sdk/crates/psx-font/vendor/IBM_VGA_8x16.bin` | IBM VGA BIOS font (PD) | IBM, extracted via VileR's font collection | Public Domain |

Font8x8 and IBM VGA fonts are already documented in detail at
[`sdk/crates/psx-font/vendor/PROVENANCE.md`](../sdk/crates/psx-font/vendor/PROVENANCE.md).

## SPU Tone Blobs

| File | Source | License |
| --- | --- | --- |
| `sdk/crates/psx-spu/vendor/tone_sine.adpcm` | Locally generated; generator command lost | GPL-2.0-or-later (treated as project source) |
| `sdk/crates/psx-spu/vendor/tone_square.adpcm` | Locally generated; generator command lost | GPL-2.0-or-later |
| `sdk/crates/psx-spu/vendor/tone_triangle.adpcm` | Locally generated; generator command lost | GPL-2.0-or-later |
| `sdk/crates/psx-spu/vendor/tone_sawtooth.adpcm` | Locally generated; generator command lost | GPL-2.0-or-later |

**Status**: these blobs are non-functional in the current SPU pipeline.
They are kept in the repo as placeholders pending the next SPU work
session, at which point they should be regenerated from a recorded
generator command (and a per-blob SHA-256 captured here). If the SPU
work doesn't pick them up, delete them rather than ship non-working
audio data.

See [`sdk/crates/psx-spu/vendor/PROVENANCE.md`](../sdk/crates/psx-spu/vendor/PROVENANCE.md).

## OBJ Reference Models

These are public-domain or hand-authored reference meshes used by SDK
or engine examples to exercise the rasterizer. Already reasonably
well-documented in their file headers, listed here for completeness.

| File | Source | License |
| --- | --- | --- |
| `engine/examples/showcase-lights/vendor/cube.obj` | Hand-authored | Public Domain (per file header) |
| `engine/examples/showcase-3d/vendor/teapot.obj` | Simplified Utah Teapot | Public Domain (Utah Teapot is public domain by Martin Newell, 1975) |
| `engine/examples/showcase-3d/vendor/suzanne.obj` | Blender's Suzanne mesh, decimated via MeshLab | CC-0 / Public Domain (Blender Foundation) |

## README Media

| File | Author | License | Notes |
| --- | --- | --- | --- |
| `docs/media/editor-3d-view.png` | Emanuele Bonura | GPL-2.0-or-later | Screenshot of the editor 3D viewport. Visible content includes the paid-subscription Meshy-generated character models documented above. |
| `docs/media/embedded-play-mode.png` | Emanuele Bonura | GPL-2.0-or-later | Screenshot of embedded play mode. Same Meshy asset provenance applies. |

## Excluded From Repository

The following categories are deliberately not tracked or distributed:

- PlayStation BIOS images (users supply their own; see [README](../README.md)).
- Commercial game disc images.
- PCSX-Redux binaries or source.
- Direct renders of Sony's BIOS firmware output (logos, splashes, shell)
  — golden-image PNGs were removed from `emu/crates/emulator-core/tests/milestones/`
  on 2026-04-30; tests now rely on hashes only.
- Mednafen / DuckStation source — used only as behavioural references
  (see [`NOTICE.md`](../NOTICE.md)).

## Pre-Publish Action Items

1. Preserve evidence for the Meshy models: paid subscription invoice,
   asset privacy status, export date, and rights-clean prompt/reference
   notes.
2. Capture exact Pexels URLs for `brick-wall.jpg` and `floor.jpg` (or
   replace with newly-sourced textures whose URLs are recorded at
   download time).
3. Decide the SPU tone blobs' fate: regenerate with a recorded
   generator command, or delete.
4. Run `cargo deny` or `cargo about` across every workspace to confirm
   no transitive dependency carries a license incompatible with
   GPL-2.0-or-later, and commit the report.
