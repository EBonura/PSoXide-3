PSoXide
=======

Copyright (C) 2025 Emanuele Bonura.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License (`LICENSE`) for more details.

Third-party references
----------------------

PSoXide is independent software, not affiliated with or endorsed by Sony
Interactive Entertainment Inc., Sony Computer Entertainment, or any of
their subsidiaries. "PlayStation" and related marks are trademarks of
their respective owners and are referenced here solely to describe the
hardware target this software emulates.

The emulator core was developed alongside the following references:

- **PCSX-Redux** (https://github.com/grumpycoders/pcsx-redux),
  GPL-2.0-or-later. Used as a parity oracle and as a reference
  implementation for ADSR envelope rate tables, AAN IDCT + YUV→RGB
  pipeline (MDEC), DMA interrupt-controller register semantics, and
  scanline-delta triangle rasterization. Algorithm shapes and naming
  conventions are kept close to Redux's where doing so simplifies
  side-by-side parity diffing.

- **nocash PSX-SPX** (https://problemkaputt.de/psxspx.htm) -- public
  hardware documentation used for register layout, ADPCM filter
  tables, and observable behaviour.

- **DuckStation** (https://github.com/stenzek/duckstation), CC-BY-NC-ND
  4.0 -- consulted as a behavioural reference. No source code from
  DuckStation is included or derived in this repository; observations
  about its runtime behaviour informed parity testing only.

- **Mednafen-PSX** (https://mednafen.github.io/), GPL-2.0-or-later --
  consulted as a behavioural reference for PSX subsystem semantics.

- **Neill Corlett's SPU envelope notes** -- the canonical write-up of
  PSX SPU envelope decode behaviour, embedded as quoted commentary in
  PCSX-Redux's `src/spu/adsr.cc`. Referenced via that quotation.

PlayStation BIOS images, commercial game disc images, and PCSX-Redux
binaries or source trees are not included in this repository and are
not redistributed by this project. Users must supply their own legally
obtained copies.
