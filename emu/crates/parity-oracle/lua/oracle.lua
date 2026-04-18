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

        elseif cmd == "introspect" then
            -- One-time discovery: dump all PCSX.* keys and a sampling
            -- of regs.* keys. Used to find the right Lua API for
            -- reading hardware registers via the IRQ controller.
            local pcsx_keys = {}
            for k, v in pairs(PCSX) do
                pcsx_keys[#pcsx_keys+1] = k .. ":" .. type(v)
            end
            send("pcsx " .. table.concat(pcsx_keys, ","))
            local reg_keys = {}
            for k, v in pairs(regs) do
                reg_keys[#reg_keys+1] = k .. ":" .. type(v)
            end
            send("regs " .. table.concat(reg_keys, ","))

        elseif cmd == "quit" then
            send("bye")
            break

        elseif cmd ~= nil then
            send("err unknown: " .. tostring(cmd))
        end
    end
end

PCSX.nextTick(run)
