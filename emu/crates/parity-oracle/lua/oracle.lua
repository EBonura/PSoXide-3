-- oracle.lua — synchronous command/response protocol.
--
-- Requires a Redux built with these patched-in Lua bindings:
--   PCSX.stepIn()             -- Debug::stepIn
--   PCSX.runExecute()         -- CPU::Execute (synchronous)
--   PCSX.setQuietPauseResume  -- suppress ExecutionFlow::Run/Pause events
--                                during stepping; essential for performance.
--
-- Execution model:
--
--   The `-dofile` entry point installs a single nextTick callback and
--   returns, so Redux's main loop can start. When that callback fires
--   for the first time the main loop is already running and in the
--   paused-update path; we blockingly read commands from stdin.
--
--   Commands reply via the `#PSX3:` sentinel prefix; Redux's own stdout
--   output (boot banner, etc.) goes unprefixed and is routed to the
--   harness's diagnostic log, not its response channel.
--
--   `step N` runs `stepIn` + `runExecute` back-to-back N times, emitting
--   one JSON record per retired instruction. Quiet mode is essential:
--   without it, every stepIn fires ExecutionFlow::Run, which starts the
--   CoreAudio device (~40 ms/call on macOS), and counters::update blocks
--   on `spu->waitForGoal` because audio never advances. Quiet mode
--   short-circuits both.

local ffi = require "ffi"
local bit = require "bit"

ffi.cdef [[
typedef struct {
    psxGPRRegs GPR;
    psxCP0Regs CP0;
    psxCP2Data CP2D;
    psxCP2Ctrl CP2C;
    uint32_t pc;
    uint32_t code;
    uint64_t cycle;
    uint64_t previousCycles;
    uint32_t interrupt;
    uint8_t spuInterrupt;
    uint8_t _spu_pad[3];
    uint64_t intTargets[32];
    uint64_t lowestTarget;
} psxRegistersExt;
]]

local function send(line)
    io.write("#PSX3:" .. line .. "\n")
    io.flush()
end

-- Like `send` but defers the flush. Used inside batched loops.
local function send_nowait(line)
    io.write("#PSX3:" .. line .. "\n")
end

local function get_hw_ptr()
    return PCSX.getScratchPtr()
end

local function get_pad_slot(port)
    local sio0 = PCSX.SIO0
    if not sio0 or not sio0.slots then
        return nil
    end
    local slot = sio0.slots[port]
    if not slot or not slot.pads then
        return nil
    end
    return slot.pads[1]
end

local function apply_pad_mask(port, mask)
    local pad = get_pad_slot(port)
    if not pad then
        return false, "pad slot unavailable"
    end
    for button = 0, 15 do
        if bit.band(mask, bit.lshift(1, button)) ~= 0 then
            pad.setOverride(button)
        else
            pad.clearOverride(button)
        end
    end
    return true
end

local function parse_pad_pulses(spec)
    if spec == nil or spec == "" or spec == "-" then
        return {}
    end
    local pulses = {}
    for entry in string.gmatch(spec, "[^,]+") do
        local mask_s, start_s, frames_s = entry:match("^(%d+)@(%d+)%+(%d+)$")
        if not mask_s then
            return nil, "bad pulse spec"
        end
        local frames = tonumber(frames_s)
        if not frames or frames <= 0 then
            return nil, "bad pulse frames"
        end
        pulses[#pulses + 1] = {
            mask = tonumber(mask_s),
            start_vblank = tonumber(start_s),
            frames = frames,
        }
    end
    return pulses
end

local function effective_pad_mask(base_mask, pulses, current_vblank)
    local mask = base_mask
    for _, pulse in ipairs(pulses) do
        local end_vblank = pulse.start_vblank + pulse.frames
        if current_vblank >= pulse.start_vblank and current_vblank < end_vblank then
            mask = bit.bor(mask, pulse.mask)
        end
    end
    return mask
end

-- Pointers resolved once in `run()` and closed over by helpers. These
-- addresses are stable for the life of the emulator.
local ram_ptr, rom_ptr, regs, regs_ext
local pad_prev_vblank = nil
local pad_vblank_count = 0

-- COP2 (GTE) accessors resolved at startup. `regs.CP2D` and `regs.CP2C`
-- are unions in `psxregisters.h` with a `r[32]` array view; the FFI
-- binding exposes that array directly. If the build doesn't expose
-- them we abort the harness loudly rather than silently emitting zeros
-- that would defeat GTE parity comparison.
local cop2d_ptr, cop2c_ptr

-- Invocation counter for `run` — diagnostic aid: if this ever exceeds 1
-- we know `PCSX.nextTick(run)` is firing more than once and can track
-- down the cause. Sent as a tag on the `ready` message.
local run_invocation = 0

local function read_instruction(pc)
    local phys = bit.band(pc, 0x1FFFFFFF)
    if phys < 0x00800000 then
        local ram_off = phys % 0x00200000
        return tonumber(ffi.cast("uint32_t*", ram_ptr + ram_off)[0])
    end
    if phys >= 0x1FC00000 and phys < 0x1FC80000 then
        return tonumber(ffi.cast("uint32_t*", rom_ptr + (phys - 0x1FC00000))[0])
    end
    return 0
end

-- Format `[v0,v1,...,v31]` from a 32-entry u32 cdata array. Used for
-- GPRs and COP2 control regs where the raw `r[i]` view already
-- matches the software-visible (CFC2) value.
local function format_u32_array(ptr)
    local parts = {}
    for i = 0, 31 do
        parts[i + 1] = tostring(tonumber(ptr[i]))
    end
    return table.concat(parts, ",")
end

-- Sign-extend the low 16 bits of `v` to 32. Used to mirror Redux's
-- `MFC2_internal` canonicalization for the half-word data registers.
local function sext16(v)
    local lo = bit.band(v, 0xFFFF)
    if bit.band(lo, 0x8000) ~= 0 then
        return bit.bor(lo, 0xFFFF0000)
    end
    return lo
end

-- Saturate `v` (treated as int32) to [0, 0x1F]. Used for IRGB packing
-- where each component is `IR / 128` clamped to a 5-bit field.
local function sat5(v)
    if v < 0 then return 0 end
    if v > 0x1F then return 0x1F end
    return v
end

-- Mirror Redux's `MFC2_internal(reg)` semantics WITHOUT mutating
-- Redux's storage. Returns the value software would observe via
-- `mfc2 rt, $rd`. Necessary because between an MTC2 (which stores
-- the raw 32-bit input into `r[reg]`) and the next MFC2 (which
-- canonicalizes), `r[reg]` carries a stale high half for the
-- half-word registers — a non-issue for software but a divergence
-- against any emulator that returns the canonical view directly.
--
-- Cases mirror gte.cc:282-314.
local function cop2_data_view(p, reg)
    local raw = tonumber(p[reg])
    if reg == 1 or reg == 3 or reg == 5
        or reg == 8 or reg == 9 or reg == 10 or reg == 11 then
        -- Sign-extend low halfword: VZ0, VZ1, VZ2, IR0, IR1, IR2, IR3.
        return sext16(raw)
    end
    if reg == 7 or reg == 16 or reg == 17 or reg == 18 or reg == 19 then
        -- Zero-extend low halfword: OTZ, SZ0..SZ3.
        return bit.band(raw, 0xFFFF)
    end
    if reg == 15 then
        -- SXYP mirrors SXY2 on read.
        return tonumber(p[14])
    end
    if reg == 28 or reg == 29 then
        -- IRGB/ORGB: pack `(IRn >> 7)` clamped to 5 bits, low to high.
        local r5 = sat5(bit.arshift(sext16(tonumber(p[9])), 7))
        local g5 = sat5(bit.arshift(sext16(tonumber(p[10])), 7))
        local b5 = sat5(bit.arshift(sext16(tonumber(p[11])), 7))
        return bit.bor(r5, bit.lshift(g5, 5), bit.lshift(b5, 10))
    end
    return raw
end

-- Format the COP2 data-register snapshot as a JSON array, applying
-- the MFC2-canonical view to each register so output matches what
-- the emulator's `read_data(idx)` returns.
local function format_cop2_data(ptr)
    local parts = {}
    for i = 0, 31 do
        parts[i + 1] = tostring(cop2_data_view(ptr, i))
    end
    return table.concat(parts, ",")
end

local function encode_record(tick, pc, instr, gpr_ptr)
    return string.format(
        '{"tick":%d,"pc":%d,"instr":%d,"gprs":[%s],"cop2_data":[%s],"cop2_ctl":[%s]}',
        tick, pc, instr,
        format_u32_array(gpr_ptr),
        format_cop2_data(cop2d_ptr),
        format_u32_array(cop2c_ptr)
    )
end

local function run()
    -- `PCSX.nextTick` is not strictly one-shot in all Redux builds — it
    -- can re-fire on later frames, which would re-enter this function
    -- and corrupt the protocol stream with extra `ready` lines. Guard
    -- against that.
    run_invocation = run_invocation + 1
    if run_invocation > 1 then
        return
    end

    PCSX.pauseEmulator()
    PCSX.setQuietPauseResume(true)
    ram_ptr = PCSX.getMemPtr()
    rom_ptr = PCSX.getRomPtr()
    regs = PCSX.getRegisters()
    regs_ext = ffi.cast("psxRegistersExt*", PCSX.getRegisters())

    -- COP2 register access via the unioned `r[32]` view. Both
    -- `psxCP2Data` and `psxCP2Ctrl` carry the same shape — one or
    -- both being absent means the Redux build's Lua bindings don't
    -- expose the GTE registers, in which case we can't do GTE
    -- parity at all and abort the harness loudly. Silently emitting
    -- zeros would mask every real GTE divergence.
    local ok_d, d_view = pcall(function() return regs.CP2D and regs.CP2D.r end)
    local ok_c, c_view = pcall(function() return regs.CP2C and regs.CP2C.r end)
    if not (ok_d and d_view and ok_c and c_view) then
        send("err startup: COP2 (regs.CP2D.r / regs.CP2C.r) not exposed by this Redux build; rebuild with GTE Lua bindings")
        return
    end
    cop2d_ptr = d_view
    cop2c_ptr = c_view

    send("ready")

    for line in io.lines() do
        local cmd = line:match("^(%S+)")

        if cmd == "handshake" then
            send("handshake ok")

        elseif cmd == "step" then
            local n = tonumber(line:match("^step%s+(%d+)$")) or 1
            for i = 1, n do
                local pc_before = tonumber(regs.pc)
                local instr = read_instruction(pc_before)
                PCSX.stepIn()
                PCSX.runExecute()
                local tick = tonumber(PCSX.getCPUCycles())
                send_nowait(encode_record(tick, pc_before, instr, regs.GPR.r))
                -- Flush periodically so Rust can start consuming while
                -- we keep producing. Without this the kernel pipe
                -- buffer (16 KiB on macOS) fills and Lua blocks on
                -- write until Rust has drained the whole batch.
                if i % 256 == 0 then io.flush() end
            end
            io.flush()

        elseif cmd == "run" then
            -- Like `step N` but WITHOUT emitting per-instruction
            -- records. Used by milestone tests that only want final
            -- state (e.g. a `vram_hash` query after N steps). Avoids
            -- the ~25s-per-million-steps Lua stdout overhead, so a
            -- 600M-step milestone query runs in ~60s instead of
            -- multiple hours.
            local n = tonumber(line:match("^run%s+(%d+)$")) or 1
            for i = 1, n do
                PCSX.stepIn()
                PCSX.runExecute()
            end
            local tick = tonumber(PCSX.getCPUCycles())
            send(string.format("run ok tick=%d", tick))

        elseif cmd == "run_audio_capture" then
            -- `run_audio_capture N CHUNK_STEPS PATH` — run N user-side
            -- steps silently and append mixed audio into PATH as raw
            -- little-endian s16 stereo PCM.
            --
            -- Important: Redux's SPU mixing thread is host-driven and
            -- can run ahead of emulated CPU time when the oracle is in
            -- quiet stepped mode. So we do NOT drain "everything that's
            -- buffered". Instead we drain only the number of frames
            -- implied by the retired CPU cycles (`33868800 / 44100 =
            -- 768 cycles/sample`) and let the queue back-pressure the
            -- producer if it tries to outrun emulated time.
            local n_str, chunk_str, path =
                line:match("^run_audio_capture%s+(%d+)%s+(%d+)%s+(.+)$")
            local n = tonumber(n_str) or 0
            local chunk = tonumber(chunk_str) or 0
            if n == 0 or chunk == 0 or not path then
                send("err run_audio_capture: bad args")
            else
                local ok, result = pcall(function()
                    local batch_frames = 2048
                    local pcm = ffi.new("int16_t[?]", batch_frames * 2)
                    local file = io.open(path, "wb")
                    if not file then
                        error("cannot open " .. path)
                    end

                    -- Drop audio that Redux's host-driven SPU thread
                    -- prebuffered before the capture window. Without
                    -- this, BIOS comparisons measure MiniAudio queue
                    -- latency rather than emulated SPU time.
                    local flushed = 0
                    while flushed < 32768 do
                        local frames = tonumber(PCSX.drainAudioFrames(pcm, batch_frames)) or 0
                        if frames <= 0 then
                            break
                        end
                        flushed = flushed + frames
                    end

                    local total_frames = 0
                    local cycle_remainder = 0
                    local prev_tick = tonumber(PCSX.getCPUCycles()) or 0
                    local remaining = n
                    while remaining > 0 do
                        local take = math.min(remaining, chunk)
                        for i = 1, take do
                            PCSX.stepIn()
                            PCSX.runExecute()
                        end
                        remaining = remaining - take

                        local tick = tonumber(PCSX.getCPUCycles()) or 0
                        local tick_delta = tick - prev_tick
                        prev_tick = tick
                        cycle_remainder = cycle_remainder + tick_delta

                        local wanted = math.floor(cycle_remainder / 768)
                        cycle_remainder = cycle_remainder % 768

                        while wanted > 0 do
                            local request = math.min(batch_frames, wanted)
                            local frames = tonumber(PCSX.drainAudioFrames(pcm, request)) or 0
                            if frames <= 0 then
                                break
                            end
                            file:write(ffi.string(ffi.cast("char*", pcm), frames * 4))
                            total_frames = total_frames + frames
                            wanted = wanted - frames
                        end
                    end

                    file:close()

                    local meta = io.open(path .. ".txt", "w")
                    if meta then
                        meta:write(string.format("rate=44100 channels=2 format=s16le frames=%d\n", total_frames))
                        meta:close()
                    end
                    local tick = tonumber(PCSX.getCPUCycles()) or 0
                    return string.format("tick=%d frames=%d", tick, total_frames)
                end)
                if ok then
                    send("run_audio_capture ok " .. result)
                else
                    send("err run_audio_capture: " .. tostring(result))
                end
            end

        elseif cmd == "log_cdrom_irqs" then
            -- `log_cdrom_irqs N M` — run N user-side steps
            -- silently, emitting one `cdrom_irq tick=... type=...`
            -- line each time bit 2 of I_STAT (CDROM) transitions
            -- from clear to set, up to M entries. Lets a probe
            -- compare every CDROM IRQ's cycle between emulators
            -- and pinpoint the first one that fires at a different
            -- time.
            --
            -- Per-step poll cost is one FFI read of a stable host
            -- address (I_STAT mirror at the hw-regs base); trivial
            -- next to stepIn()+runExecute().
            local n_str, m_str = line:match("^log_cdrom_irqs%s+(%d+)%s+(%d+)$")
            local n = tonumber(n_str) or 0
            local m_max = tonumber(m_str) or 100
            if n == 0 then
                send("err log_cdrom_irqs: bad args")
            else
                local hw_ptr = get_hw_ptr()
                local istat_ptr = ffi.cast("uint32_t*", hw_ptr + 0x1070)
                local cdirq_ptr = ffi.cast("uint8_t*", hw_ptr + 0x1803)
                local prev = bit.band(istat_ptr[0], 0x4)
                local emitted = 0
                for i = 1, n do
                    PCSX.stepIn()
                    PCSX.runExecute()
                    local cur = bit.band(istat_ptr[0], 0x4)
                    if cur ~= 0 and prev == 0 then
                        local tick = tonumber(PCSX.getCPUCycles())
                        local ty = bit.band(cdirq_ptr[0], 0x7)
                        send_nowait(string.format(
                            "cdrom_irq step=%d tick=%d type=%d", i, tick, ty))
                        emitted = emitted + 1
                        if emitted % 32 == 0 then io.flush() end
                        if emitted >= m_max then break end
                    end
                    prev = cur
                end
                io.flush()
                send(string.format(
                    "log_cdrom_irqs ok emitted=%d", emitted))
            end

        elseif cmd == "run_checkpoint" then
            -- `run_checkpoint N M` — run N user-side steps silently,
            -- emitting one `chk step=X tick=Y pc=Z` line every M
            -- steps. This is the coarse variant of `step`: no GPRs,
            -- no per-instruction records, just the cycle counter and
            -- PC at sampled intervals. Used by fast divergence hunts
            -- — a full 100 M-step scan emits 10 K checkpoints (a few
            -- hundred KiB) instead of 14 GiB of trace records, and
            -- finishes in ~30 s instead of ~40 min.
            --
            -- ISR folding falls out naturally from Redux's `stepIn`:
            -- the stepIn breakpoint lives in user code, so IRQ
            -- handlers run-through under `runExecute` and only the
            -- post-RFE user instruction trips the breakpoint. Each
            -- iteration here is one user-side step even if it took
            -- a full VBlank ISR along the way.
            local n_str, m_str = line:match("^run_checkpoint%s+(%d+)%s+(%d+)$")
            local n = tonumber(n_str) or 0
            local m = tonumber(m_str) or 1
            if n == 0 then
                send("err run_checkpoint: bad args")
            else
                local emissions = 0
                for i = 1, n do
                    PCSX.stepIn()
                    PCSX.runExecute()
                    if i % m == 0 then
                        local tick = tonumber(PCSX.getCPUCycles())
                        local pc = tonumber(regs.pc)
                        send_nowait(string.format("chk step=%d tick=%d pc=%d", i, tick, pc))
                        emissions = emissions + 1
                        if emissions % 256 == 0 then io.flush() end
                    end
                end
                io.flush()
                local tick = tonumber(PCSX.getCPUCycles())
                local pc = tonumber(regs.pc)
                send(string.format("run_checkpoint ok step=%d tick=%d pc=%d", n, tick, pc))
            end

        elseif cmd == "run_checkpoint_pad" then
            -- `run_checkpoint_pad N M PORT BASE_MASK PULSES` — same as
            -- `run_checkpoint`, but also applies a VBlank-timed button
            -- schedule to the given controller port. `PULSES` is either
            -- `-` or a comma-separated list of
            -- `<mask>@<start_vblank>+<frames>` entries, using decimal
            -- integers only so the shell and protocol stay boring.
            local n_str, m_str, port_str, base_mask_str, pulse_spec =
                line:match("^run_checkpoint_pad%s+(%d+)%s+(%d+)%s+(%d+)%s+(%d+)%s+(%S+)$")
            local n = tonumber(n_str) or 0
            local m = tonumber(m_str) or 1
            local port = tonumber(port_str) or 0
            local base_mask = tonumber(base_mask_str) or 0
            local pulses, pulse_err = parse_pad_pulses(pulse_spec)
            if n == 0 or port == 0 or not pulses then
                send("err run_checkpoint_pad: " .. tostring(pulse_err or "bad args"))
            else
                local hw_ptr = get_hw_ptr()
                local istat_ptr = ffi.cast("uint32_t*", hw_ptr + 0x1070)
                if pad_prev_vblank == nil then
                    pad_prev_vblank = bit.band(istat_ptr[0], 0x1)
                end
                local current_mask = nil
                local function sync_pad_mask()
                    local next_mask = effective_pad_mask(base_mask, pulses, pad_vblank_count)
                    if current_mask ~= next_mask then
                        local ok, err = apply_pad_mask(port, next_mask)
                        if not ok then
                            return false, err
                        end
                        current_mask = next_mask
                    end
                    return true
                end
                local ok, err = sync_pad_mask()
                if not ok then
                    send("err run_checkpoint_pad: " .. tostring(err))
                else
                    local emissions = 0
                    for i = 1, n do
                        PCSX.stepIn()
                        PCSX.runExecute()
                        local cur_vblank = bit.band(istat_ptr[0], 0x1)
                        if cur_vblank ~= 0 and pad_prev_vblank == 0 then
                            pad_vblank_count = pad_vblank_count + 1
                        end
                        pad_prev_vblank = cur_vblank
                        ok, err = sync_pad_mask()
                        if not ok then
                            send("err run_checkpoint_pad: " .. tostring(err))
                            break
                        end
                        if i % m == 0 then
                            local tick = tonumber(PCSX.getCPUCycles())
                            local pc = tonumber(regs.pc)
                            send_nowait(string.format("chk step=%d tick=%d pc=%d", i, tick, pc))
                            emissions = emissions + 1
                            if emissions % 256 == 0 then io.flush() end
                        end
                    end
                    io.flush()
                    if ok then
                        local tick = tonumber(PCSX.getCPUCycles())
                        local pc = tonumber(regs.pc)
                        send(string.format("run_checkpoint_pad ok step=%d tick=%d pc=%d", n, tick, pc))
                    end
                end
            end

        elseif cmd == "peek32" then
            -- `peek32 ADDR` — return the 32-bit value at the given
            -- physical address (decimal or 0x-prefixed hex). Used to
            -- inspect IO registers (I_STAT/I_MASK/etc) at the current
            -- step from the harness side. Wrapped in pcall because
            -- the MMIO read paths through Redux's API can throw.
            local addr_str = line:match("^peek32%s+(%S+)$")
            local addr = tonumber(addr_str)
            if addr == nil then
                send("err peek32: bad addr")
            else
                local ok, value = pcall(function()
                    local phys = bit.band(addr, 0x1FFFFFFF)
                    if phys < 0x00800000 then
                        local ram_off = phys % 0x00200000
                        return tonumber(ffi.cast("uint32_t*", ram_ptr + ram_off)[0])
                    end
                    if phys >= 0x1FC00000 and phys < 0x1FC80000 then
                        return tonumber(ffi.cast("uint32_t*", rom_ptr + (phys - 0x1FC00000))[0])
                    end
                    -- MMIO read: Redux exposes it via PCSX.getHardwareRegisters()
                    -- which returns the io[] echo buffer (8KB) — same range as
                    -- our `io` slice. The IRQ controller writes its live state
                    -- into io[] on every change, so reading from there is
                    -- equivalent to what software sees.
                    if phys >= 0x1F801000 and phys < 0x1F803000 then
                        local hw = get_hw_ptr()
                        local hw_off = phys - 0x1F800000
                        return tonumber(ffi.cast("uint32_t*", hw + hw_off)[0])
                    end
                    return 0xDEADBEEF
                end)
                if ok then
                    send(string.format("peek32 %d", value or -1))
                else
                    send("err peek32: " .. tostring(value))
                end
            end

        elseif cmd == "regs" then
            -- Return CPU+COP0 snapshot useful for IRQ debugging:
            -- pc, cause, sr, epc, plus the cycle counter. Redux's
            -- regs.CP0 is a union with `r` (32-element array) — see
            -- `psxregisters.h`. Cause is index 13, Status is 12, EPC 14.
            local ok, msg = pcall(function()
                local pc = tonumber(regs.pc)
                local cycles = tonumber(PCSX.getCPUCycles())
                local cp0 = regs.CP0
                local r = cp0 and cp0.r
                local cause = r and r[13]
                local sr = r and r[12]
                local epc = r and r[14]
                local ok_interrupt, interrupt = pcall(function() return regs_ext[0].interrupt end)
                local ok_lowest, lowest = pcall(function() return regs_ext[0].lowestTarget end)
                local queue = "nil"
                if ok_interrupt and interrupt then
                    local mask = tonumber(interrupt) or 0
                    local parts = {}
                    for bit_idx = 0, 14 do
                        if bit.band(mask, bit.lshift(1, bit_idx)) ~= 0 then
                            local ok_target, target = pcall(function()
                                return regs_ext[0].intTargets[bit_idx]
                            end)
                            if ok_target and target then
                                parts[#parts + 1] = string.format("%d@%d", bit_idx, tonumber(target))
                            else
                                parts[#parts + 1] = tostring(bit_idx)
                            end
                        end
                    end
                    if #parts > 0 then
                        queue = table.concat(parts, ",")
                    end
                end
                send(string.format(
                    "regs pc=%d cause=%s sr=%s epc=%s cycles=%d interrupt=%s lowest=%s queue=%s",
                    pc,
                    cause and tostring(tonumber(cause)) or "nil",
                    sr and tostring(tonumber(sr)) or "nil",
                    epc and tostring(tonumber(epc)) or "nil",
                    cycles,
                    (ok_interrupt and interrupt) and tostring(tonumber(interrupt)) or "nil",
                    (ok_lowest and lowest) and tostring(tonumber(lowest)) or "nil",
                    queue
                ))
            end)
            if not ok then
                send("err regs: " .. tostring(msg))
            end

        elseif cmd == "screenshot_probe" then
            -- Probe what `PCSX.GPU.takeScreenShot()` returns in this
            -- Redux build. Reports table fields, bpp value, and
            -- data sub-shape so we can pick the right parser for
            -- `vram_hash`.
            local ok, result = pcall(function()
                if not (PCSX.GPU and PCSX.GPU.takeScreenShot) then
                    error("no takeScreenShot")
                end
                local shot = PCSX.GPU.takeScreenShot()
                local parts = {}
                parts[#parts+1] = "w=" .. tostring(shot.width)
                parts[#parts+1] = "h=" .. tostring(shot.height)
                if type(shot.bpp) == "cdata" then
                    parts[#parts+1] = "bpp_type=" .. tostring(ffi.typeof(shot.bpp))
                    parts[#parts+1] = "bpp=" .. tostring(tonumber(shot.bpp))
                else
                    parts[#parts+1] = "bpp=" .. tostring(shot.bpp)
                end
                if type(shot.data) == "table" then
                    -- Redux Slice pattern: table with _wrapper (cdata) +
                    -- _type (string) + methods. Probe the metatable.
                    local sub_keys = {}
                    for k, v in pairs(shot.data) do
                        sub_keys[#sub_keys+1] = tostring(k) .. ":" .. type(v)
                    end
                    parts[#parts+1] = "data_keys={" .. table.concat(sub_keys, ",") .. "}"
                    if shot.data._type then
                        parts[#parts+1] = "data._type=" .. tostring(shot.data._type)
                    end
                    if shot.data._wrapper then
                        parts[#parts+1] = "data._wrapper=" .. tostring(ffi.typeof(shot.data._wrapper))
                    end
                    local mt = getmetatable(shot.data)
                    if mt then
                        local mt_keys = {}
                        for k, v in pairs(mt) do
                            mt_keys[#mt_keys+1] = tostring(k) .. ":" .. type(v)
                        end
                        parts[#parts+1] = "data_mt={" .. table.concat(mt_keys, ",") .. "}"
                    end
                elseif type(shot.data) == "cdata" then
                    parts[#parts+1] = "data_cdata=" .. tostring(ffi.typeof(shot.data))
                else
                    parts[#parts+1] = "data_type=" .. type(shot.data)
                end
                return table.concat(parts, " ")
            end)
            if ok then
                send("screenshot_probe " .. result)
            else
                send("err screenshot_probe: " .. tostring(result))
            end

        elseif cmd == "screenshot_save" then
            -- `screenshot_save PATH` — writes Redux's current
            -- screenshot as raw little-endian 15bpp pixel bytes to
            -- PATH (plus a sidecar PATH.txt describing dimensions).
            -- Used for direct byte-by-byte parity diffs against our
            -- emulator's display_hash path.
            local path = line:match("^screenshot_save%s+(.+)$")
            if not path then
                send("err screenshot_save: missing path")
            else
                local ok, result = pcall(function()
                    if not (PCSX.GPU and PCSX.GPU.takeScreenShot) then
                        error("no takeScreenShot")
                    end
                    local shot = PCSX.GPU.takeScreenShot()
                    local w = tonumber(shot.width) or 0
                    local h_dim = tonumber(shot.height) or 0
                    local len = #shot.data
                    local bin = io.open(path, "wb")
                    if not bin then error("cannot open " .. path) end
                    -- Slice.__index returns byte values; pack in
                    -- chunks of 4096 to keep the string builder sane.
                    local chunk_size = 4096
                    local buf = {}
                    for i = 0, len - 1 do
                        buf[#buf+1] = string.char(tonumber(shot.data[i]) or 0)
                        if #buf >= chunk_size then
                            bin:write(table.concat(buf))
                            buf = {}
                        end
                    end
                    if #buf > 0 then bin:write(table.concat(buf)) end
                    bin:close()
                    local meta = io.open(path .. ".txt", "w")
                    if meta then
                        meta:write(string.format("w=%d h=%d bpp=%d len=%d\n",
                            w, h_dim, tonumber(shot.bpp) or 0, len))
                        meta:close()
                    end
                    return string.format("w=%d h=%d len=%d", w, h_dim, len)
                end)
                if ok then
                    send("screenshot_save ok " .. result)
                else
                    send("err screenshot_save: " .. tostring(result))
                end
            end

        elseif cmd == "vram_hash" then
            -- FNV-1a-64 over Redux's currently-visible display area
            -- (the output of `PCSX.GPU.takeScreenShot()`). This is
            -- what the BIOS is actually showing on-screen — it
            -- varies between 0×0 (before any display command has
            -- run) and 640×480 (the standard NTSC-interlaced mode).
            -- Hashing only the visible region gives us Redux-
            -- anchored "pixel parity" for milestone tests, which
            -- verifies that we're rendering the same thing Redux
            -- does at a given instruction count — not just that
            -- our emulator is self-consistent run-to-run.
            --
            -- Redux's PCSX Lua API doesn't expose a direct
            -- 1 MiB VRAM pointer in this build; the screenshot
            -- Slice is the closest thing. If / when a VRAM-ptr
            -- accessor shows up we can extend this to full-VRAM
            -- hashing.
            --
            -- The response also carries `w=... h=... bpp=...` so
            -- callers can sanity-check that Redux's display area
            -- matches theirs, since a same-pixel hash on different
            -- dimensions is meaningless.
            local ok, result = pcall(function()
                if not (PCSX.GPU and PCSX.GPU.takeScreenShot) then
                    error("no takeScreenShot")
                end
                local shot = PCSX.GPU.takeScreenShot()
                local w = tonumber(shot.width) or 0
                local h_dim = tonumber(shot.height) or 0
                local bpp = tonumber(shot.bpp) or 0
                local len = #shot.data
                local h = ffi.new("uint64_t", 0xCBF29CE484222325ULL)
                local prime = ffi.new("uint64_t", 0x100000001B3ULL)
                for i = 0, len - 1 do
                    local byte = tonumber(shot.data[i]) or 0
                    h = bit.bxor(h, ffi.new("uint64_t", byte))
                    h = h * prime
                end
                -- Format as two 32-bit halves. `tonumber(u64)` goes
                -- through Lua's `double`, which only has a 53-bit
                -- mantissa — so `%016x` on the result silently
                -- zeroes the bottom ~11 bits of the hash. Splitting
                -- into high and low u32s keeps every bit precise.
                local hi = ffi.cast("uint32_t", bit.rshift(h, 32))
                local lo = ffi.cast("uint32_t", bit.band(h, 0xFFFFFFFFULL))
                return string.format(
                    "%08x%08x w=%d h=%d bpp=%d len=%d",
                    tonumber(hi), tonumber(lo),
                    w, h_dim, bpp, len
                )
            end)
            if ok then
                send("vram_hash " .. result)
            else
                send("err vram_hash: " .. tostring(result))
            end

        elseif cmd == "introspect" then
            -- One-time discovery: dump the namespaces we might pull
            -- peripheral accessors from. Each section is pcall'd so
            -- a single-section error (FFI cdata iteration, missing
            -- namespace) doesn't kill the rest of the introspection.
            local function dump_table(label, t)
                local ok, result = pcall(function()
                    if t == nil then return "<nil>" end
                    local keys = {}
                    for k, v in pairs(t) do
                        keys[#keys+1] = tostring(k) .. ":" .. type(v)
                    end
                    return table.concat(keys, ",")
                end)
                if ok then
                    send(label .. " " .. result)
                else
                    send(label .. " err:" .. tostring(result))
                end
            end
            dump_table("pcsx", PCSX)
            dump_table("gpu", PCSX.GPU)
            dump_table("misc", PCSX.Misc)
            dump_table("sio0", PCSX.SIO0)
            dump_table("consts", PCSX.CONSTS)

        elseif cmd == "quit" then
            send("bye")
            break

        elseif cmd ~= nil then
            send("err unknown: " .. tostring(cmd))
        end
    end
end

PCSX.nextTick(run)
