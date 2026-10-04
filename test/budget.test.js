import { describe, expect, it, vi } from 'vitest'
import {
  Config, RateMeter, apply, compactCostMs, evaluate, routedTarget, selectRange, validateConfig,
} from '../src/index.js'

function fakeCtx(measurement = { totalTokens: 0, nodes: [] }) {
  const listeners = {}
  const commands = []
  const disposers = []
  const ctx = {
    on: (event, cb) => { listeners[event] = cb },
    effect: (setup) => { disposers.push(setup()) },
    logger: { info: vi.fn(), warn: vi.fn(), exporter: vi.fn() },
    tokenMeter: { measure: vi.fn(() => measurement) },
    commands: { register: vi.fn((def) => { commands.push(def) }) },
  }
  return { ctx, listeners, commands, measurement }
}

function fakeAgent({ provider = 'p', id = 'session-abcdef1234', nodes = [1], compaction = { compactRegion: vi.fn(async () => ({ shadowedSeqs: [1], shadowedTokenCount: 5 })) } } = {}) {
  return {
    id,
    session: {
      requestHeader: () => ({ config: { provider, model: 'm' } }),
      surface: { nodes, replaceGeneration: 0 },
    },
    ctx: { get: (key) => (key === 'compaction' ? compaction ?? undefined : undefined) },
    compaction,
  }
}

/** Run the llm/stream listener over `chunks`, advancing fake time by `firstChunkAfterMs` before the first chunk. */
async function stream(listener, options, chunks, firstChunkAfterMs, now) {
  async function* next() {
    for (const [i, chunk] of chunks.entries()) {
      if (i === 0) now.t += firstChunkAfterMs
      yield chunk
    }
  }
  const out = []
  for await (const c of listener(options, next)) out.push(c)
  return out
}

const usage = (inputTokens, cacheReadTokens) => ({ type: 'usage', usage: { inputTokens, outputTokens: 1, cacheReadTokens } })
const finish = { type: 'finish', reason: { kind: 'stop' } }
const setup = (providers, measurement, balanced = () => true) => {
  const f = fakeCtx(measurement)
  const now = { t: 1000 }
  apply(f.ctx, { providers }, { now: () => now.t, balanced })
  return { ...f, now, streamL: f.listeners['llm/stream'], stepL: f.listeners['agent/pre-step'] }
}
const next = vi.fn(async () => ({ kind: 'enter', messages: [] }))
const signal = () => new AbortController().signal

describe('RateMeter', () => {
  it('ignores small samples and averages the rest as tokens per ms', () => {
    const m = new RateMeter(3)
    expect(m.add(500, 1000)).toBe(false)
    expect(m.rate).toBeUndefined()
    expect(m.add(4000, 2000)).toBe(true)
    expect(m.add(2000, 2000)).toBe(true)
    expect(m.rate).toBeCloseTo(6000 / 4000)
    expect(m.count).toBe(2)
  })

  it('drops the oldest sample past the ring size', () => {
    const m = new RateMeter(2)
    m.add(1000, 1000); m.add(2000, 1000); m.add(3000, 1000)
    expect(m.count).toBe(2)
    expect(m.rate).toBeCloseTo(5000 / 2000)
  })
})

describe('evaluate', () => {
  const cfg = { hardCeilingTokens: 100, maxTtftMs: 1000, maxColdPrefillMs: 5000 }
  it('is quiet when nothing trips', () => {
    expect(evaluate(cfg, 100, 1000, 1)).toBeUndefined()
  })
  it('reports the ceiling first', () => {
    expect(evaluate(cfg, 101, 5000, 0.001).check).toBe('ceiling')
  })
  it('reports observed ttft', () => {
    expect(evaluate(cfg, 50, 1001, undefined).check).toBe('observed')
  })
  it('reports predicted cold prefill only with a rate', () => {
    expect(evaluate(cfg, 50, undefined, undefined)).toBeUndefined()
    expect(evaluate(cfg, 50, undefined, 0.001).check).toBe('predicted')
  })
  it('skips unconfigured checks', () => {
    expect(evaluate({ maxTtftMs: 1000 }, 1e9, undefined, 0.0001)).toBeUndefined()
  })
})

describe('compactCostMs', () => {
  it('is (total + retain) / rate, undefined without a rate', () => {
    expect(compactCostMs(100000, 16000, 0.068)).toBeCloseTo(116000 / 0.068)
    expect(compactCostMs(100000, 16000, undefined)).toBeUndefined()
  })
})

describe('routedTarget', () => {
  it('reads the session request header', () => {
    expect(routedTarget({ requestHeader: () => ({ config: { provider: 'b', model: 'm2' } }) })).toEqual({ provider: 'b', model: 'm2' })
    expect(routedTarget({ requestHeader: () => undefined })).toBeUndefined()
    expect(routedTarget({ requestHeader: () => ({ config: { provider: '', model: 'm' } }) })).toBeUndefined()
  })
})

describe('selectRange', () => {
  const nodes = [1, 2, 3, 4, 5, 6].map((seq) => ({ seq, tokens: 10 }))
  const session = { surface: { nodes: [1, 2, 3, 4, 5, 6] } }
  it('keeps the tail and backs up to a balanced boundary', () => {
    const balanced = (s, seq) => seq !== 4
    expect(selectRange(session, { nodes }, 25, balanced)).toEqual({ start: 1, end: 2 })
  })
  it('returns null when the tail keeps everything or the surface is stale', () => {
    expect(selectRange(session, { nodes }, 60, () => true)).toBeNull()
    expect(() => selectRange({ surface: { nodes: [1, 2] } }, { nodes }, 10, () => true)).toThrow(/surface/)
  })
})

describe('Config', () => {
  it('defaults retainTokens and action', () => {
    expect(Config({ providers: { p: { hardCeilingTokens: 5 } } }).providers.p).toEqual({ hardCeilingTokens: 5, retainTokens: 16000, action: 'warn' })
  })
  it('rejects a bad action and a provider with no check', () => {
    expect(() => Config({ providers: { p: { hardCeilingTokens: 5, action: 'x' } } })).toThrow()
    expect(() => validateConfig({ providers: { p: { retainTokens: 1 } } })).toThrow(/no check/)
  })
})

describe('llm/stream measurement', () => {
  it('records rate from ttft and uncached tokens, passes chunks through', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { maxColdPrefillMs: 1 } })
    const chunks = [{ type: 'text-delta', text: 'x' }, usage(5000, 1000), finish]
    const out = await stream(streamL, { provider: 'p', sessionId: 's1' }, chunks, 2000, now)
    expect(out).toEqual(chunks)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 10, nodes: [] })
    await stepL({ agent: fakeAgent({ id: 's1' }), signal: signal() }, next)
    expect(ctx.logger.info.mock.calls[0][0]).toMatch(/^context-budget: p session=s1 predicted: cold prefill/)
  })

  it('ignores unlisted providers and streams without usage', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { maxColdPrefillMs: 1 } })
    await stream(streamL, { provider: 'other' }, [finish], 5000, now)
    await stream(streamL, { provider: 'p' }, [finish], 5000, now)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 1e9, nodes: [] })
    await stepL({ agent: fakeAgent(), signal: signal() }, next)
    expect(ctx.logger.info).not.toHaveBeenCalled()
  })

  it('does not measure compaction summary requests', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { maxTtftMs: 1000, maxColdPrefillMs: 1 } })
    await stream(streamL, { provider: 'p', sessionId: 'session-abcdef1234', purpose: 'compaction' }, [usage(5000, 0), finish], 9000, now)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 1e9, nodes: [] })
    await stepL({ agent: fakeAgent(), signal: signal() }, next)
    expect(ctx.logger.info).not.toHaveBeenCalled()
  })

  it('uses inputTokens as the uncached count without subtracting cache reads', async () => {
    const { streamL, now, ctx, commands } = setup({ p: { maxColdPrefillMs: 1 } })
    await stream(streamL, { provider: 'p', sessionId: 's1' }, [usage(5000, 3000), finish], 2000, now)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 1, nodes: [] })
    expect(commands[0].handler({ agent: fakeAgent(), rawInput: '' }).text).toMatch(/measured rate: 2500 tok\/s/)
  })

  it('forgets an observed ttft once the surface generation changes', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { maxTtftMs: 1000 } })
    const agent = fakeAgent()
    await stepL({ agent, signal: signal() }, next)                       // records generation 0 for this step
    await stream(streamL, { provider: 'p', sessionId: agent.id }, [finish], 5000, now)
    agent.session.surface.replaceGeneration = 1                            // compaction rewrote the surface
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info).not.toHaveBeenCalled()
    await stream(streamL, { provider: 'p', sessionId: agent.id }, [finish], 5000, now)
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info.mock.calls[0][0]).toMatch(/observed: last ttft 5 s/)
  })

  it('records the last ttft per session even when the stream errors later', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { maxTtftMs: 1000 } })
    const agent = fakeAgent({ id: 'session-zz' })
    await stepL({ agent, signal: signal() }, next)
    async function* next2() { now.t += 3000; yield { type: 'text-delta', text: 'x' }; throw new Error('boom') }
    await expect((async () => { for await (const c of streamL({ provider: 'p', sessionId: 'session-zz' }, next2)) void c })()).rejects.toThrow('boom')
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info.mock.calls[0][0]).toMatch(/observed: last ttft 3 s > 1 s/)
  })
})

describe('agent/pre-step decision', () => {
  it('does nothing below the ceiling, warns above it, calls next either way', async () => {
    next.mockClear()
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 100 } })
    const agent = fakeAgent()
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 100, nodes: [] })
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info).not.toHaveBeenCalled()
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 101, nodes: [] })
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info.mock.calls[0][0]).toMatch(/ceiling: 101 tokens > ceiling 100 .*compacting now n\/a/)
    expect(agent.compaction.compactRegion).not.toHaveBeenCalled()
    expect(next).toHaveBeenCalledTimes(2)
  })

  it('compact mode calls compactRegion with the selected range and logs the outcome', async () => {
    const nodes = [1, 2, 3, 4]
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 10, retainTokens: 10, action: 'compact' } })
    const agent = fakeAgent({ nodes })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 40, nodes: nodes.map((seq) => ({ seq, tokens: 10 })) })
    await stepL({ agent, signal: signal() }, next)
    expect(agent.compaction.compactRegion).toHaveBeenCalledWith(1, 3, agent, expect.any(AbortSignal))
    expect(ctx.logger.info.mock.calls.at(-1)[0]).toMatch(/compacted 1 items \(~5 tokens\)/)
  })

  it('compact mode survives a failing compaction and a preset without compaction', async () => {
    next.mockClear()
    const nodes = [1, 2, 3, 4]
    const failing = { compactRegion: vi.fn(async () => { throw new Error('busy') }) }
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 10, retainTokens: 10, action: 'compact' } })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 40, nodes: nodes.map((seq) => ({ seq, tokens: 10 })) })
    await stepL({ agent: fakeAgent({ nodes, compaction: failing }), signal: signal() }, next)
    expect(ctx.logger.warn.mock.calls[0][0]).toMatch(/compaction failed: busy; continuing/)
    await stepL({ agent: fakeAgent({ nodes, compaction: null }), signal: signal() }, next)
    expect(ctx.logger.warn.mock.calls[1][0]).toMatch(/no compaction engine/)
    expect(next).toHaveBeenCalledTimes(2)
  })

  it('reports one check per step, ceiling before observed before predicted', async () => {
    const { streamL, stepL, now, ctx } = setup({ p: { hardCeilingTokens: 10, maxTtftMs: 1, maxColdPrefillMs: 1 } })
    const agent = fakeAgent()
    await stepL({ agent, signal: signal() }, next)
    await stream(streamL, { provider: 'p', sessionId: agent.id }, [usage(5000, 0), finish], 5000, now)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 11, nodes: [] })
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.info).toHaveBeenCalledTimes(1)
    expect(ctx.logger.info.mock.calls[0][0]).toMatch(/ ceiling: /)
  })

  it('compact mode backs the cut up to a balanced boundary', async () => {
    const nodes = [1, 2, 3, 4]
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 10, retainTokens: 10, action: 'compact' } }, undefined, (s, seq) => seq !== 4)
    const agent = fakeAgent({ nodes })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 40, nodes: nodes.map((seq) => ({ seq, tokens: 10 })) })
    await stepL({ agent, signal: signal() }, next)
    expect(agent.compaction.compactRegion).toHaveBeenCalledWith(1, 2, agent, expect.any(AbortSignal))
  })

  it('warns only once per agent about a preset without compaction', async () => {
    const nodes = [1, 2, 3, 4]
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 10, retainTokens: 10, action: 'compact' } })
    const agent = fakeAgent({ nodes, compaction: null })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 40, nodes: nodes.map((seq) => ({ seq, tokens: 10 })) })
    await stepL({ agent, signal: signal() }, next)
    await stepL({ agent, signal: signal() }, next)
    expect(ctx.logger.warn).toHaveBeenCalledTimes(1)
  })

  it('ignores aborted signals, unlisted providers, and sessions without a header', async () => {
    const { stepL, ctx } = setup({ p: { hardCeilingTokens: 1 } })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 100, nodes: [] })
    const aborted = new AbortController(); aborted.abort()
    await stepL({ agent: fakeAgent(), signal: aborted.signal }, next)
    await stepL({ agent: fakeAgent({ provider: 'other' }), signal: signal() }, next)
    const noHeader = fakeAgent(); noHeader.session.requestHeader = () => undefined
    await stepL({ agent: noHeader, signal: signal() }, next)
    expect(ctx.logger.info).not.toHaveBeenCalled()
  })
})

describe('/context-budget command and exporter', () => {
  it('prints measurements with the compact-now estimate', async () => {
    const { streamL, now, ctx, commands } = setup({ p: { hardCeilingTokens: 100, maxColdPrefillMs: 60000, retainTokens: 20 } })
    await stream(streamL, { provider: 'p', sessionId: 'session-abcdef1234' }, [usage(2000, 0), finish], 1000, now)
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 150, nodes: [] })
    const text = commands[0].handler({ agent: fakeAgent(), rawInput: '' }).text
    expect(commands[0].name).toBe('context-budget')
    expect(text).toContain('context: 150 tokens')
    expect(text).toMatch(/ceiling 100.*\[TRIP\]/)
    expect(text).toMatch(/measured rate: 2000 tok\/s \(1 sample\)/)
    expect(text).toMatch(/compact now: ~0.0 min/)
    expect(text).toContain('action: warn')
  })

  it('reports an unconfigured provider and missing samples', () => {
    const { ctx, commands } = setup({ p: { hardCeilingTokens: 100 } })
    ctx.tokenMeter.measure.mockReturnValue({ totalTokens: 5, nodes: [] })
    expect(commands[0].handler({ agent: fakeAgent({ provider: 'other' }), rawInput: '' }).text).toMatch(/not guarded/)
    expect(commands[0].handler({ agent: fakeAgent(), rawInput: '' }).text).toMatch(/no samples yet/)
  })

  it('exporter prints only context-budget lines', () => {
    const { ctx } = setup({ p: { hardCeilingTokens: 1 } })
    const [exporter] = ctx.logger.exporter.mock.calls[0]
    const log = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      exporter.export({ args: ['context-budget: p x'] })
      exporter.export({ args: ['other: y'] })
      expect(log.mock.calls).toEqual([['context-budget: p x']])
    } finally { log.mockRestore() }
  })
})

describe('v0.2: per-sample timestamp and cache state', () => {
  it('RateMeter stores at and warm, and evicts them with the ring', () => {
    const m = new RateMeter(2)
    m.add(1000, 1000, { at: 111, warm: true })
    m.add(2000, 1000, { at: 222, warm: false })
    m.add(3000, 1000, { at: 333 })
    expect(m.samples).toEqual([
      { uncached: 2000, ttftMs: 1000, at: 222, warm: false },
      { uncached: 3000, ttftMs: 1000, at: 333, warm: false },
    ])
  })
  it('a request with cache reads is a warm sample, without is cold', async () => {
    const { streamL, commands, now } = setup({ p: { hardCeilingTokens: 100000 } })
    await stream(streamL, { provider: 'p', sessionId: 'session-abcdef1234' }, [usage(4000, 500), finish], 2000, now)
    now.t += 600000
    await stream(streamL, { provider: 'p', sessionId: 'session-abcdef1234' }, [usage(8000), finish], 3000, now)
    now.t += 120000
    const text = commands[0].handler({ agent: fakeAgent() }).text
    const lines = text.split('\n').filter((l) => l.includes('tok in'))
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('8000 tok in 3 s')
    expect(lines[0]).toContain('cold')
    expect(lines[0]).toContain('2 min ago')
    expect(lines[1]).toContain('4000 tok in 2 s')
    expect(lines[1]).toContain('warm')
    expect(lines[1]).toContain('12 min ago')
  })
  it('no sample lines when there are no samples', () => {
    const { commands } = setup({ p: { hardCeilingTokens: 100000 } })
    const text = commands[0].handler({ agent: fakeAgent() }).text
    expect(text).toContain('no samples yet')
    expect(text).not.toContain('tok in')
  })
})

describe('package', () => {
  it('accepts the dsh versions it was checked against', async () => {
    const { readFile } = await import('node:fs/promises')
    const { default: semver } = await import('semver')
    const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))
    const range = pkg.peerDependencies['@deepseek-ai/dsh-compaction']
    // dsh skips a plugin at startup when its dsh peer range rejects the running version
    for (const version of ['0.1.1-rc.2', '0.2.0-rc.2']) {
      expect(semver.satisfies(version, range, { includePrerelease: true })).toBe(true)
    }
  })
})
