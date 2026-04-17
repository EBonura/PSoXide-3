/*
 * PSoXide-3 linker script.
 *
 * Layout a PSX-EXE-shaped binary: a 2 KiB header immediately followed by
 * the program's text + data. The header's magic, entry-point, load
 * address, and payload size are all computed at link time from section
 * symbols, so once this script is adopted, rebuilding a crate yields
 * a bootable binary without any post-processing.
 *
 * KUSEG/KSEG0/KSEG1 map the same physical RAM. We load to KSEG0
 * (cached) at 64 KiB past RAM base to leave the BIOS kernel alone,
 * matching the address the real BIOS's EXE loader uses.
 */

RAM_BASE    = 0x80000000;   /* KSEG0 cached RAM */
RAM_SIZE    = 2M;
BIOS_SIZE   = 64K;          /* First 64 KiB is BIOS / kernel scratch */
HEADER_SIZE = 2K;           /* PSX-EXE header */
LOAD_ADDR   = RAM_BASE + BIOS_SIZE;
STACK_INIT  = RAM_BASE + 0x001FFF00;

MEMORY {
    HEADER    : ORIGIN = LOAD_ADDR - HEADER_SIZE, LENGTH = HEADER_SIZE
    RAM (rwx) : ORIGIN = LOAD_ADDR,               LENGTH = RAM_SIZE - BIOS_SIZE
}

ENTRY(_start)

SECTIONS {
    /* PSX-EXE header, exactly 2048 bytes. The loader (our emulator or
       the real BIOS) copies `.text + .data` from byte offset 0x800 of
       the file to `LOAD_ADDR`, then jumps to `_start`. */
    .psx_exe_header : {
        /* Magic "PS-X EXE" */
        BYTE(0x50); BYTE(0x53); BYTE(0x2D); BYTE(0x58);
        BYTE(0x20); BYTE(0x45); BYTE(0x58); BYTE(0x45);

        /* 8 bytes reserved */
        QUAD(0);

        /* 0x10 initial PC */
        LONG(ABSOLUTE(_start));
        /* 0x14 initial GP (unused by Rust bare-metal) */
        LONG(0);
        /* 0x18 destination (load) address */
        LONG(LOAD_ADDR);
        /* 0x1C payload size (excluding header, excluding BSS) */
        LONG(__bss_start - __text_start);

        /* 0x20 data section start + size (unused when we pack data
           into .text + .data) */
        QUAD(0);
        QUAD(0);

        /* 0x30 SP base */
        LONG(STACK_INIT);
        /* 0x34 SP offset */
        LONG(0);

        /* 0x38..0x4B reserved */
        QUAD(0);
        QUAD(0);
        LONG(0);

        /* Region marker; keep if present, else zero-fill via ALIGN. */
        KEEP(*(.region));

        /* Pad to exactly HEADER_SIZE. */
        . = ALIGN(HEADER_SIZE);
    } > HEADER

    /* Program code. `.text._start` is forced first so the entry
       point lives at `LOAD_ADDR` exactly. */
    .text : {
        __text_start = .;
        KEEP(*(.text._start));
        *(.text .text.*);
        __text_end = .;
    } > RAM

    /* Initialised data + read-only data. PS1 has no MMU so .rodata
       isn't actually read-only at runtime; keep it with .data so
       the linker can pack everything into one contiguous load. */
    .data : {
        __data_start = .;
        *(.data .data.*);
        *(.rodata .rodata.*);
        *(.got);
        . = ALIGN(2048);   /* 2 KiB-align for ISO sector loading later */
        __data_end = .;
    } > RAM

    /* Uninitialised data — zeroed by `_start`, not stored in file. */
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*);
        *(COMMON);
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    /* Heap starts right after BSS and runs up to just below the stack. */
    . = ALIGN(4);
    __heap_start = .;
    __heap_end   = STACK_INIT - 0x100;   /* 256-byte guard before SP */

    /* Discard MIPS ABI glue not needed on bare metal. */
    /DISCARD/ : {
        *(.MIPS.abiflags)
        *(.reginfo)
        *(.eh_frame_hdr)
        *(.eh_frame)
        *(.comment)
        *(.note.*)
    }
}
