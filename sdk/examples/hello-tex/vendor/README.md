# hello-tex — source texture assets

Both the **pre-cropped 512×512 source JPGs** and the **cooked
`.psxt` blobs** (in `../assets/`) live in the repo. Keeping both
means:

- `make assets` regenerates the `.psxt` byte-for-byte from the
  committed sources — any fresh clone can rebuild the pipeline
  without downloading extra data.
- The repo stays small: a 512×512 JPG is ~50-150 KB. The
  multi-megapixel photographs these were cropped from (several
  MB each) are *not* committed.

## Re-cropping from a higher-res original

If you want better quality, start from a higher-res source and
bring the pre-cropped version up to 1024×1024 or 2048×2048. On
macOS:

```bash
# Centre-crop to square + resize in one step (sips = built-in).
sips -c 1024 1024 ORIGINAL.jpg --out sdk/examples/hello-tex/vendor/brick-wall.jpg
```

Then `make assets` — the cooker re-quantises from the new source.
Commit both the updated JPG and the regenerated `.psxt`; update
the milestone golden for `hello-tex` to match.

## Sources

| Repo filename     | Used as       | Original / attribution                     |
| ----------------- | ------------- | ------------------------------------------ |
| `brick-wall.jpg`  | Wall texture  | "Brick Wall" by Michael Laut (Pexels)      |
| `floor.jpg`       | Floor texture | "Batako Wall Texture Street" (Pexels)      |

## Cook settings

The Makefile passes `--size 64x64 --depth 4 --resample lanczos3`
for both textures. 64×64 at 4bpp is the classic PSX wall-tile size
(one `Tpage` fits four of them comfortably). Lanczos3 downscales
cleanly from 512×512 and preserves enough detail to read the brick
seams at native 4bpp palette size.

If you need different dimensions or bit depth, invoke `psxed tex`
directly:

```bash
editor/target/release/psxed tex vendor/brick-wall.jpg \
    -o assets/brick-wall.psxt \
    --size 128x128 --depth 8
```

The cooker's default **centre-square-crop** is applied before
resize — so source JPGs don't strictly have to be square. Pass
`--no-crop` to disable (rarely useful; causes aspect distortion
on non-square sources), or `--crop X,Y,W,H` for manual control.
