# PSoXide development commands.
#
# The repo has two Cargo workspaces: root (shared no_std crates) and emu/
# (host-side emulator). Every target fans out to both.

.PHONY: help check test canaries fmt lint clean fetch-opcode oracle-smoke parity run

help:
	@echo "PSoXide targets:"
	@echo "  make check        - cargo check on root + emu workspaces"
	@echo "  make test         - fast unit tests (both workspaces, excludes canaries)"
	@echo "  make canaries     - run commercial-game canary tests (Milestones D-K)"
	@echo "  make fmt          - format all code in both workspaces"
	@echo "  make lint         - clippy -D warnings on both workspaces"
	@echo "  make clean        - cargo clean in both workspaces"
	@echo "  make run          - launch the desktop frontend"
	@echo "  make fetch-opcode - smoke: print first BIOS opcode (needs BIOS=<path>)"
	@echo "  make oracle-smoke - smoke: launch headless Redux and verify Lua runs"
	@echo "  make parity       - step both emulators and assert bit-identical traces"

run:
	cd emu && cargo run -p frontend --release

check:
	cargo check --workspace --all-features
	cd emu && cargo check --workspace --all-features

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

fetch-opcode:
	@if [ -z "$(BIOS)" ]; then echo "usage: make fetch-opcode BIOS=/path/to/bios.bin"; exit 2; fi
	cd emu && cargo run -p emulator-core --example fetch_first_opcode -- "$(BIOS)"

oracle-smoke:
	cd emu && cargo test -p parity-oracle --test smoke -- --ignored --nocapture

parity:
	cd emu && cargo test -p emulator-core --test parity -- --ignored --nocapture
