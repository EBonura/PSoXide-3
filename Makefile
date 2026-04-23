# PSoXide development commands.
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
        psxed assets \
        examples hello-tri hello-input hello-ot hello-tex hello-gte hello-audio \
        run-tri run-input run-ot run-tex run-gte run-audio \
        showcase-textured-sprite run-showcase-textured-sprite \
        showcase-text run-showcase-text \
        game-pong run-game-pong \
        game-breakout run-game-breakout \
        game-invaders run-game-invaders \
        showcase-3d run-showcase-3d \
        showcase-lights run-showcase-lights \
        showcase-fog run-showcase-fog \
        hello-engine run-hello-engine

help:
	@echo "PSoXide targets:"
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
	@echo "    make psxed        - build the content-pipeline CLI"
	@echo "    make assets       - cook source assets via psxed"
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
	@echo "    make game-pong     - build the Pong mini-game"
	@echo "    make game-breakout - build the Breakout mini-game"
	@echo "    make game-invaders - build the Space Invaders mini-game"
	@echo "    make showcase-3d    - build the 3D geometry showcase"
	@echo "    make showcase-lights - build the 4-point-light demo"
	@echo "    make showcase-fog   - build the fog / full-GTE-pipeline demo"
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
	@echo "    make run-game-pong     - build + side-load the Pong mini-game"
	@echo "    make run-game-breakout - build + side-load the Breakout mini-game"
	@echo "    make run-game-invaders - build + side-load the Space Invaders mini-game"
	@echo "    make run-showcase-3d - build + side-load the 3D geometry showcase"
	@echo "    make run-showcase-lights - build + side-load the 4-point-light demo"
	@echo "    make run-showcase-fog - build + side-load the fog demo"

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
	cd emu && cargo test -p emulator-core --release --features trace-cop2 --test parity -- --ignored --nocapture

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

# engine/ examples live outside sdk/examples/ — the engine is its
# own domain and its demos exercise the engine framework.
hello-engine:
	cd engine/examples/hello-engine && cargo build --release

hello-tex: assets
	cd sdk/examples/hello-tex && cargo build --release

hello-gte:
	cd sdk/examples/hello-gte && cargo build --release

hello-audio:
	cd sdk/examples/hello-audio && cargo build --release

showcase-textured-sprite:
	cd engine/examples/showcase-textured-sprite && cargo build --release

showcase-text:
	cd engine/examples/showcase-text && cargo build --release

game-pong:
	cd engine/examples/game-pong && cargo build --release

game-breakout:
	cd engine/examples/game-breakout && cargo build --release

game-invaders:
	cd engine/examples/game-invaders && cargo build --release

showcase-3d: assets
	cd engine/examples/showcase-3d && cargo build --release

showcase-lights: assets
	cd engine/examples/showcase-lights && cargo build --release

# showcase-fog uses two cooked textures (brick wall + cobblestone
# floor) on its corridor walls + floor, plus procedural geometry.
showcase-fog: assets
	cd engine/examples/showcase-fog && cargo build --release

# --- Content pipeline (host-side editor tooling) ------------------------

PSXED := editor/target/release/psxed

# Build the content-pipeline CLI. Independent host workspace —
# always builds fast, no MIPS toolchain needed.
psxed:
	cd editor && cargo build --release --bin psxed

# Cook source assets into the binary blobs examples embed via
# include_bytes!. Re-runs whenever an .obj changes. Targets go
# next to the source under `assets/` so a repo clone has the
# runtime input available without having to run the editor.
SHOWCASE_3D := engine/examples/showcase-3d
SHOWCASE_LIGHTS := engine/examples/showcase-lights
SHOWCASE_FOG := engine/examples/showcase-fog
HELLO_TEX := sdk/examples/hello-tex

# Texture sources (.jpg / .png) are gitignored because they're
# multi-MB photographs; the small cooked .psxt blobs in assets/
# are the repo artifact. `make assets` cooks only when the source
# is present, so fresh clones — which lack the sources — finish
# without errors and the committed .psxt blobs remain valid.
define cook_texture
	@if [ -f "$(1)" ]; then \
	    $(PSXED) tex "$(1)" -o "$(2)" --size $(3) --depth $(4) --resample lanczos3 ; \
	else \
	    echo "[psxed tex] skip: source $(1) not present (using committed $(2))" ; \
	fi
endef

assets: psxed
	@mkdir -p $(SHOWCASE_3D)/assets $(SHOWCASE_LIGHTS)/assets $(HELLO_TEX)/assets
	@$(PSXED) obj $(SHOWCASE_3D)/vendor/suzanne.obj \
	    -o $(SHOWCASE_3D)/assets/suzanne.psxm \
	    --palette warm --decimate-grid 6 --compute-normals
	@$(PSXED) obj $(SHOWCASE_3D)/vendor/teapot.obj \
	    -o $(SHOWCASE_3D)/assets/teapot.psxm \
	    --palette cool --compute-normals
	@$(PSXED) obj $(SHOWCASE_LIGHTS)/vendor/cube.obj \
	    -o $(SHOWCASE_LIGHTS)/assets/cube.psxm \
	    --compute-normals --no-colors
	@mkdir -p $(SHOWCASE_FOG)/assets
	$(call cook_texture,$(HELLO_TEX)/vendor/brick-wall.jpg,$(HELLO_TEX)/assets/brick-wall.psxt,64x64,4)
	$(call cook_texture,$(HELLO_TEX)/vendor/floor.jpg,$(HELLO_TEX)/assets/floor.psxt,64x64,4)
	$(call cook_texture,$(SHOWCASE_FOG)/vendor/brick-wall.jpg,$(SHOWCASE_FOG)/assets/brick-wall.psxt,64x64,4)
	$(call cook_texture,$(SHOWCASE_FOG)/vendor/floor.jpg,$(SHOWCASE_FOG)/assets/floor.psxt,64x64,4)

examples: hello-tri hello-input hello-ot hello-tex hello-gte hello-audio showcase-textured-sprite showcase-text game-pong game-breakout game-invaders showcase-3d showcase-lights showcase-fog hello-engine
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

run-game-pong: game-pong
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/game-pong.exe cargo run -p frontend --release

run-game-breakout: game-breakout
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/game-breakout.exe cargo run -p frontend --release

run-game-invaders: game-invaders
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/game-invaders.exe cargo run -p frontend --release

run-showcase-3d: showcase-3d
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/showcase-3d.exe cargo run -p frontend --release

run-showcase-lights: showcase-lights
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/showcase-lights.exe cargo run -p frontend --release

run-showcase-fog: showcase-fog
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/showcase-fog.exe cargo run -p frontend --release

run-hello-engine: hello-engine
	cd emu && PSOXIDE_EXE=$(CURDIR)/$(EXAMPLE_OUT)/hello-engine.exe cargo run -p frontend --release
