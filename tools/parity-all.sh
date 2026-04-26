#!/usr/bin/env bash
# Run the HW-renderer parity harness on every showcase example.
# Per fixture: prints PASS/FAIL with mismatch% and writes the
# cpu/hw/diff PPMs to /tmp/psx-parity/<fixture>/ for inspection.
#
# Usage: tools/parity-all.sh [STEPS] [TOLERANCE]
#   STEPS     CPU instructions to run before snapshotting (default 5_000_000)
#   TOLERANCE per-channel LSB tolerance (default 8)

set -u
STEPS="${1:-5000000}"
TOL="${2:-8}"
BIOS="/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/build/examples/mipsel-sony-psx/release"
OUT_ROOT="/tmp/psx-parity"
PARITY_BIN="$ROOT/emu/target/release/examples/parity"

if [[ ! -x "$PARITY_BIN" ]]; then
    echo "build parity first: (cd emu && cargo build --release -p psx-gpu-render --example parity)" >&2
    exit 2
fi

FIXTURES=(
    hello-tex
    showcase-textured-sprite
    showcase-3d
    showcase-text
    showcase-particles
    showcase-lights
    showcase-fog
)

pass=0
fail=0
missing=0
for f in "${FIXTURES[@]}"; do
    exe="$BIN_DIR/$f.exe"
    if [[ ! -f "$exe" ]]; then
        printf "SKIP   (missing) %s\n" "$f"
        missing=$((missing + 1))
        continue
    fi
    out="$OUT_ROOT/$f"
    mkdir -p "$out"
    if "$PARITY_BIN" "$BIOS" "$exe" "$STEPS" "$out" "$TOL"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
done

echo
echo "summary: $pass pass, $fail fail, $missing missing"
[[ $fail -eq 0 ]]
