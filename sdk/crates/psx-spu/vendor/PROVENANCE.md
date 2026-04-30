# SPU Tone Blob Provenance

The four `.adpcm` blobs in this directory are PS1-format ADPCM samples
intended as reference test tones (sine, square, triangle, sawtooth) for
the SPU pipeline.

## Status: non-functional placeholders

The blobs were generated locally during early SPU bring-up. The exact
generator command was not committed and has been lost. The samples do
not currently produce the expected audio when fed through the SPU
voice path; they are kept in the repo as filename placeholders for a
later regeneration pass.

## Action when SPU work resumes

1. Regenerate each tone from a documented generator (Python `numpy` +
   PS1 ADPCM encoder, an online PSX-ADPCM tool, or a script in
   `tools/`).
2. Record the exact generator invocation in this file.
3. Capture each blob's SHA-256 here so future regenerations can be
   verified.
4. If regeneration is not on the near-term roadmap, delete the blobs
   rather than continue to ship non-functional binary data.

## License

Treated as project source under GPL-2.0-or-later (see repo root
[`LICENSE`](../../../../LICENSE)). They contain no third-party data:
synthetic tones generated locally are pure project work.
