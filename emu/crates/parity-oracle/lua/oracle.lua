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

local function send(line)
    io.write("#PSX3:" .. line .. "\n")
    io.flush()
end

-- Like `send` but defers the flush. Used inside batched loops.
local function send_nowait(line)
    io.write("#PSX3:" .. line .. "\n")
end

-- Pointers resolved once in `run()` and closed over by helpers. These
-- addresses are stable for the life of the emulator.
local ram_ptr, rom_ptr, regs

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

local function encode_record(tick, pc, instr, gpr_ptr)
    local parts = {}
    for i = 0, 31 do
        parts[i + 1] = tostring(tonumber(gpr_ptr[i]))
    end
    return string.format(
        '{"tick":%d,"pc":%d,"instr":%d,"gprs":[%s]}',
        tick, pc, instr, table.concat(parts, ",")
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
                        local candidates = {
                            "getHardwareRegisters", "getHardwareRegistersPtr",
                            "getHWPtr", "getHardwarePtr", "getHwRegPtr",
                            "getIOPtr", "getIoPtr", "getRegisters",
                        }
                        for _, name in ipairs(candidates) do
                            local fn = PCSX[name]
                            if type(fn) == "function" then
                                local hw = fn()
                                if hw then
                                    local off = phys - 0x1F801000
                                    return tonumber(ffi.cast("uint32_t*", hw + off)[0])
                                end
                            end
                        end
                        -- Last resort: list keys for debugging.
                        local keys = {}
                        for k, _ in pairs(PCSX) do keys[#keys+1] = k end
                        error("no hw-regs accessor; PCSX keys: " .. table.concat(keys, ","))
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
                send(string.format(
                    "regs pc=%d cause=%s sr=%s epc=%s cycles=%d",
                    pc,
                    cause and tostring(tonumber(cause)) or "nil",
                    sr and tostring(tonumber(sr)) or "nil",
                    epc and tostring(tonumber(epc)) or "nil",
                    cycles
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
