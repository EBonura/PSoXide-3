# PSoXide-3 development commands.
#
# Three Cargo workspaces:
#   root - no_std shared crates (psx-hw, psx-iso, psx-trace)
#   emu  - host-side emulator (emulator-core, frontend, parity-oracle)
#   sdk  - MIPS target SDK (psx-io, psx-rt, psx-gpu, psx-pad, psx-sdk)
#
# SDK examples live under sdk/examples/ and are compiled individually
# with cargo build in their own directory so they can use their own
# .cargo/config.toml for the mipsel-sony-psx target.

.PHONY: help check test canaries fmt lint clean fetch-opcode oracle-smoke parity run \
        test-sdk \
        examples hello-tri hello-input hello-ot hello-tex hello-gte hello-audio \
        run-tri run-input run-ot run-tex run-gte run-audio \
        showcase-textured-sprite run-showcase-textured-sprite \
        showcase-text run-showcase-text

help:
	@echo "PSoXide-3 targets:"
	@echo ""
	@echo "  Emulator / host:"
	@echo "    make check        - cargo check on root + emu + sdk workspaces"
	@echo "    make test         - fast unit tests (both workspaces, excludes canaries)"
	@echo "    make canaries     - commercial-game canary tests (Milestones D-K)"
	@echo "    make fmt          - format all code"
	@echo "    make lint         - clippy -D warnings"
	@echo "    make clean        - cargo clean all workspaces"
	@echo "    make run          - launch the desktop frontend (no EXE)"
	@echo "    make parity       - step both emulators and assert bit-identical traces"
	@echo "    make oracle-smoke - smoke: launch headless Redux and verify Lua runs"
	@echo "    make test-sdk     - build every SDK example + run Milestone-C regression suite"
	@echo ""
	@echo "  SDK examples (build mipsel-sony-psx binaries):"
	@echo "    make examples     - build every example"
	@echo "    make hello-tri    - build the direct-GP0 triangle demo"
	@echo "    make hello-input  - build the pad-poll demo"
	@echo "    make hello-ot     - build the DMA linked-list demo"
	@echo "    make hello-tex    - build the textured-sprite demo"
	@echo "    make hello-gte    - build the GTE perspective-transform demo"
	@echo "    make hello-audio  - build the SPU-tone-per-button demo"
	@echo "    make showcase-textured-sprite"
	@echo "                      - build the polished textured-sprite showcase"
	@echo "    make showcase-text"
	@echo "                      - build the text / font capabilities showcase"
	@echo "    make run-tri      - build + side-load hello-tri into the frontend"
	@echo "    make run-input    - build + side-load hello-input into the frontend"
	@echo "    make run-ot       - build + side-load hello-ot into the frontend"
	@echo "    make run-tex      - build + side-load hello-tex into the frontend"
	@echo "    make run-gte      - build + side-load hello-gte into the frontend"
	@echo "    make run-audio    - build + side-load hello-audio into the frontend"
	@echo "    make run-showcase-textured-sprite"
	@echo "                      - build + side-load the textured-sprite showcase"
	@echo "    make run-showcase-text"
	@echo "                      - build + side-load the text capabilities showcase"

run:
	cd emu && cargo run -p frontend --release

check:
	cargo check --workspace --all-features
	cd emu && cargo check --workspace --all-features
	cd sdk && cargo check --workspace --all-features

test:
	cargo test --workspace
	cd emu && cargo test --workspace

canaries:
	cargo test --workspace -- --ignored
	cd emu && cargo test --workspace -- --ignored

fmt:
	cargo fmt --all
	cd emu && cargo fmt --all

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cd emu && cargo clippy --workspace --all-targets --all-features -- -D warnings

clean:
	cargo clean
	cd emu && cargo clean
	cd sdk && cargo clean
	rm -rf build

fetch-opcode:
	@if [ -z "$(BIOS)" ]; then echo "usage: make fetch-opcode BIOS=/path/to/bios.bin"; exit 2; fi
	cd emu && cargo run -p emulator-core --example fetch_first_opcode -- "$(BIOS)"

oracle-smoke:
	cd emu && cargo test -p parity-oracle --test smoke -- --ignored --nocapture

parity:
	cd emu && cargo test -p emulator-core --release --test parity -- --ignored --nocapture

# Milestone-C regression suite — every SDK example side-loaded into
# the emulator, multi-signal state pinned. Depends on `examples` so
# every .exe referenced by the tests exists before we run them; the
# tests themselves skip gracefully when an .exe is missing, but
# gating on `examples` here surfaces build breaks up-front.
test-sdk: examples
	cd emu && cargo test -p emulator-core --release --test sdk_milestones -- --ignored --nocapture

# --- SDK examples ---------------------------------------------------------

EXAMPLE_OUT := build/examples/mipsel-sony-psx/release

hello-tri:
	cd sdk/examples/hello-tri && cargo build --release

hello-input:
	cd sdk/examples/hello-input && cargo build --release

hello-ot:
	cd sdk/examples/hello-ot && cargo build --release

hello-tex:
	cd sdk/examples/hello-tex && cargo build --release

hello-gte:
	cd sdk/examples/hello-gte && cargo build --release

hello-audio:
	cd sdk/examples/hello-audio && cargo build --release

showcase-textured-sprite:
	cd sdk/examples/showcase-textured-sprite && cargo build --release

showcase-text:
	cd sdk/examples/showcase-text && cargo build --release

examples: hello-tri hello-input hello-ot hello-tex hello-gte hello-audio showcase-textured-sprite showcase-text
	@echo ""
	@echo "Built SDK examples:"
	@ls -la $(EXAMPLE_OUT)/*.exe 2>/dev/null || true

# Frontend side-load helpers. PSOXIDE_EXE makes the frontend skip the
# BIOS reset vector and jump straight into the homebrew. HLE BIOS +
# digital pad are auto-enabled for side-loaded EXEs.

run-tri: hello-tri
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-tri.exe cargo run -p frontend --release

run-input: hello-input
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-input.exe cargo run -p frontend --release

run-ot: hello-ot
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-ot.exe cargo run -p frontend --release

run-tex: hello-tex
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-tex.exe cargo run -p frontend --release

run-gte: hello-gte
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-gte.exe cargo run -p frontend --release

run-audio: hello-audio
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-audio.exe cargo run -p frontend --release

run-showcase-textured-sprite: showcase-textured-sprite
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/showcase-textured-sprite.exe cargo run -p frontend --release

run-showcase-text: showcase-text
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/showcase-text.exe cargo run -p frontend --release
