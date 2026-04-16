-- oracle.lua — synchronous command/response protocol.
--
-- Requires a Redux built with `stepIn` and `runExecute` Lua bindings
-- (patched into pcsxlua.cc / pcsxffi.lua).
--
-- Execution model:
--
--   The `-dofile` entry point installs a single nextTick callback and
--   returns, so Redux's main loop can start. When that callback fires
--   for the first time the main loop is already running and in the
--   paused-update path; we blockingly read commands from stdin. Each
--   command returns a response via the `#PSX3:` sentinel; Redux's own
--   stdout goes unprefixed and is siphoned off by the harness as
--   diagnostic log.
--
--   `step` runs `stepIn` (priming `m_step = STEP_IN`, resuming the
--   emulator) then calls `runExecute` which pumps the interpreter
--   loop synchronously. The interpreter executes exactly one
--   instruction; `Debug::process` sees the step flag and pauses,
--   so `runExecute` returns after a single retirement. No coroutines
--   or yields needed — everything is stack-linear and debuggable.

local ffi = require "ffi"
local bit = require "bit"

local function send(line)
    io.write("#PSX3:" .. line .. "\n")
    io.flush()
end

local function resolve_read(virt)
    local phys = bit.band(virt, 0x1FFFFFFF)
    if phys < 0x00800000 then
        local ram_off = phys % 0x00200000
        return PCSX.getMemPtr(), ram_off
    end
    if phys >= 0x1FC00000 and phys < 0x1FC80000 then
        return PCSX.getRomPtr(), phys - 0x1FC00000
    end
    return nil, nil
end

local function read_instruction(pc)
    local base, offset = resolve_read(pc)
    if base == nil then return 0 end
    local word_ptr = ffi.cast("uint32_t*", base + offset)
    return tonumber(word_ptr[0])
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

local function step_one()
    local regs = PCSX.getRegisters()
    local pc_before = tonumber(regs.pc)
    local instr = read_instruction(pc_before)

    PCSX.stepIn()
    PCSX.runExecute()

    local regs_after = PCSX.getRegisters()
    local tick = tonumber(PCSX.getCPUCycles())
    return encode_record(tick, pc_before, instr, regs_after.GPR.r)
end

local function run()
    PCSX.pauseEmulator()
    send("ready")

    for line in io.lines() do
        local cmd = line:match("^(%S+)")

        if cmd == "handshake" then
            send("handshake ok")

        elseif cmd == "step" then
            local n = tonumber(line:match("^step%s+(%d+)$")) or 1
            for _ = 1, n do
                send(step_one())
            end

        elseif cmd == "quit" then
            send("bye")
            break

        elseif cmd ~= nil then
            send("err unknown: " .. tostring(cmd))
        end
    end
end

-- Defer command loop until Redux's main loop starts. Blocking on stdin
-- here would prevent the main loop from ever entering, which is what
-- made earlier coroutine designs necessary — this synchronous design
-- works only because Execute() is now callable directly.
PCSX.nextTick(run)
