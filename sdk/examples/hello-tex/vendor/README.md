# hello-tex — source texture assets

The cooked `.psxt` blobs in `../assets/` are the repo artifact and
are versioned. The raw source images that produce them are **not
committed** — they're multi-MB photographs and their byte contents
aren't useful to the runtime. `.gitignore` at the repo root keeps
them out.

## Re-cooking

If you want to regenerate the `.psxt` blobs from scratch (different
size, different quantiser, different resample), drop the sources
into this directory with the exact filenames below, then run `make
assets` at the repo root.

| Local filename     | Used as         | Source / attribution                       |
| ------------------ | --------------- | ------------------------------------------ |
| `brick-wall.jpg`   | Wall texture    | "Brick Wall" by Michael Laut (Pexels)      |
| `floor.jpg`        | Floor texture   | "Batako Wall Texture Street" (Pexels)      |

`make assets` checks for each source file before invoking `psxed
tex`; missing sources are skipped silently so fresh clones don't
have to fetch anything. The already-cooked `.psxt` blobs stay
valid.

## Cook settings

The Makefile passes `--size 64x64 --depth 4 --resample lanczos3`
for both textures. 64×64 at 4bpp is the classic PSX wall-tile size
(one `Tpage` fits four of them comfortably). Lanczos3 downscales
cleanly from the multi-megapixel sources.

If you need different dimensions or bit depth, invoke `psxed tex`
directly:

```bash
editor/target/release/psxed tex vendor/brick-wall.jpg \
    -o assets/brick-wall.psxt \
    --size 128x128 --depth 8 --resample lanczos3
```

Remember to rebuild `hello-tex` and refresh the milestone golden if
you change cook parameters (the `.exe` embeds the cooked blob via
`include_bytes!`, so the binary changes whenever the blob changes).
