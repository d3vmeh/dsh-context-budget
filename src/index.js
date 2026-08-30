import z from '@deepseek-ai/schemastery'
import { toolPairingBalancedBefore } from '@deepseek-ai/dsh-compaction'

export const name = 'context-guard'
export const inject = ['llm', 'tokenMeter', 'commands']

export const MIN_SAMPLE_TOKENS = 1000
const RING_SIZE = 8

const providerSchema = z.object({
  hardCeilingTokens: z.number().step(1).min(1),
  maxTtftMs: z.number().min(1),
  maxColdPrefillMs: z.number().min(1),
  retainTokens: z.number().step(1).min(0).default(16000),
  action: z.union(['warn', 'compact']).default('warn'),
})

export const Config = z.object({
  providers: z.dict(providerSchema).default({}),
})

export function validateConfig(config) {
  for (const [provider, c] of Object.entries(config.providers ?? {})) {
    if (c.hardCeilingTokens === undefined && c.maxTtftMs === undefined && c.maxColdPrefillMs === undefined) {
      throw new Error(`context-guard: provider "${provider}" sets no check (hardCeilingTokens, maxTtftMs or maxColdPrefillMs)`)
    }
  }
}

/** Rolling prefill-rate samples for one provider: tokens the server had to read per ms of wait. */
export class RateMeter {
  constructor(size = RING_SIZE) {
    this.size = size
    this.samples = []
  }

  add(uncached, ttftMs) {
    if (!(uncached >= MIN_SAMPLE_TOKENS) || !(ttftMs > 0)) return false
    this.samples.push({ uncached, ttftMs })
    if (this.samples.length > this.size) this.samples.shift()
    return true
  }

  get count() {
    return this.samples.length
  }

  get rate() {
    if (this.samples.length === 0) return undefined
    let tokens = 0
    let ms = 0
    for (const s of this.samples) { tokens += s.uncached; ms += s.ttftMs }
    return tokens / ms
  }
}

/** Provider and model of the session's current request header. */
export function routedTarget(session) {
  const config = session.requestHeader()?.config
  if (!config?.provider || !config.model) return undefined
  return { provider: config.provider, model: config.model }
}

/** First check that trips, in order ceiling, observed, predicted; undefined when none. */
export function evaluate(cfg, total, lastTtftMs, rate) {
  if (cfg.hardCeilingTokens !== undefined && total > cfg.hardCeilingTokens) {
    return { check: 'ceiling', detail: `${total} tokens > ceiling ${cfg.hardCeilingTokens}` }
  }
  if (cfg.maxTtftMs !== undefined && lastTtftMs !== undefined && lastTtftMs > cfg.maxTtftMs) {
    return { check: 'observed', detail: `last ttft ${sec(lastTtftMs)} > ${sec(cfg.maxTtftMs)}` }
  }
  if (cfg.maxColdPrefillMs !== undefined && rate !== undefined && total / rate > cfg.maxColdPrefillMs) {
    return { check: 'predicted', detail: `cold prefill ${min(total / rate)} > ${min(cfg.maxColdPrefillMs)}` }
  }
  return undefined
}

/** Rough cost of compacting now: the summary prompt prefills cold, then the retained context prefills cold again. */
export function compactCostMs(total, retainTokens, rate) {
  return rate === undefined ? undefined : (total + retainTokens) / rate
}

/**
 * Same selection compaction-basic makes: keep the most recent `retainTokens`,
 * then back the cut up to a tool-pairing boundary. `balanced` is injectable for tests.
 */
export function selectRange(session, measurement, retainTokens, balanced = toolPairingBalancedBefore) {
  const priced = measurement.nodes
  const surface = session.surface.nodes
  if (priced.length === 0) return null
  if (surface.length !== priced.length || surface.some((seq, i) => seq !== priced[i].seq)) {
    throw new Error('context-guard: token-meter surface does not match the session surface')
  }
  let accumulated = 0
  let keepFrom = priced.length
  for (let i = priced.length - 1; i >= 0; i -= 1) {
    accumulated += priced[i].tokens
    keepFrom = i
    if (accumulated >= retainTokens) break
  }
  while (keepFrom > 0 && !balanced(session, surface[keepFrom])) keepFrom -= 1
  if (keepFrom === 0) return null
  return { start: surface[0], end: surface[keepFrom - 1] }
}

const sec = (ms) => `${(ms / 1000).toFixed(0)} s`
const min = (ms) => `${(ms / 60000).toFixed(1)} min`

const shortId = (id) => String(id).replace(/^session-/, '').slice(0, 8)
const errorMessage = (e) => (e instanceof Error ? e.message : String(e))
const tokPerSec = (rate) => (rate === undefined ? 'n/a' : `${Math.round(rate * 1000)} tok/s`)
const minutes = (ms) => (ms === undefined ? 'n/a' : `~${(ms / 60000).toFixed(1)} min`)

export function apply(ctx, config = {}, internals = {}) {
  validateConfig(config)
  const now = internals.now ?? Date.now
  const balanced = internals.balanced ?? toolPairingBalancedBefore
  const providers = Object.fromEntries(Object.entries(config.providers ?? {}).map(([k, c]) => [k, { retainTokens: 16000, action: 'warn', ...c }]))
  const meters = new Map()
  const lastTtft = new Map()   // session id -> { ttft, gen }: observed TTFT and the surface generation it was measured on
  const stepGen = new Map()    // session id -> surface generation at the start of the current step
  const warnedNoCompaction = new Set()
  const meterFor = (provider) => {
    let m = meters.get(provider)
    if (m === undefined) { m = new RateMeter(); meters.set(provider, m) }
    return m
  }

  // The dsh host keeps plugin logs in memory only; print just this plugin's lines.
  ctx.logger.exporter({
    export: (message) => {
      const line = message.args[0]
      if (typeof line === 'string' && line.startsWith('context-guard:')) console.log(line)
    },
  })

  ctx.on('llm/stream', async function* (options, next) {
    // Compaction summaries are cold, one-off prefills: not representative, so not measured.
    if (providers[options.provider] === undefined || options.purpose === 'compaction') return yield* next()
    const t0 = now()
    let ttft
    let usage
    try {
      for await (const chunk of next()) {
        if (ttft === undefined) ttft = now() - t0
        if (chunk.type === 'usage') usage = chunk.usage
        yield chunk
      }
    } finally {
      try {
        if (ttft !== undefined) {
          const id = options.sessionId === undefined ? undefined : String(options.sessionId)
          if (id !== undefined) lastTtft.set(id, { ttft, gen: stepGen.get(id) })
          // dsh reports inputTokens already net of cache reads.
          if (usage !== undefined) meterFor(options.provider).add(usage.inputTokens, ttft)
        }
      } catch (error) {
        ctx.logger.warn(`context-guard: measurement failed: ${errorMessage(error)}`)
      }
    }
  })

  const generation = (agent) => agent.session.surface?.replaceGeneration

  function observedTtft(agent) {
    const record = lastTtft.get(String(agent.id))
    return record !== undefined && record.gen === generation(agent) ? record.ttft : undefined
  }

  function inspect(agent) {
    const target = routedTarget(agent.session)
    const cfg = target === undefined ? undefined : providers[target.provider]
    if (cfg === undefined) return undefined
    const measurement = ctx.tokenMeter.measure(agent.session)
    const meter = meterFor(target.provider)
    return {
      target, cfg, measurement,
      total: measurement.totalTokens,
      rate: meter.rate,
      samples: meter.count,
      lastTtftMs: observedTtft(agent),
      tag: `${target.provider} session=${shortId(agent.id)}`,
    }
  }

  async function compact(agent, view, signal) {
    const compaction = agent.ctx.get('compaction')
    if (compaction === undefined) {
      if (!warnedNoCompaction.has(agent.id)) {
        warnedNoCompaction.add(agent.id)
        ctx.logger.warn(`context-guard: ${view.tag} no compaction engine in this agent's preset; warning only`)
      }
      return
    }
    try {
      const range = selectRange(agent.session, view.measurement, view.cfg.retainTokens, balanced)
      if (range === null) { ctx.logger.info(`context-guard: ${view.tag} nothing compactable`); return }
      const result = await compaction.compactRegion(range.start, range.end, agent, signal)
      lastTtft.delete(String(agent.id))
      ctx.logger.info(`context-guard: ${view.tag} compacted ${result.shadowedSeqs.length} items (~${result.shadowedTokenCount} tokens)`)
    } catch (error) {
      ctx.logger.warn(`context-guard: ${view.tag} compaction failed: ${errorMessage(error)}; continuing`)
    }
  }

  ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
    try {
      if (!signal.aborted) {
        stepGen.set(String(agent.id), generation(agent))
        const view = inspect(agent)
        const hit = view === undefined ? undefined : evaluate(view.cfg, view.total, view.lastTtftMs, view.rate)
        if (hit !== undefined) {
          const cost = minutes(compactCostMs(view.total, view.cfg.retainTokens, view.rate))
          ctx.logger.info(`context-guard: ${view.tag} ${hit.check}: ${hit.detail} (${view.total} tokens, rate ${tokPerSec(view.rate)}, compacting now ${cost})`)
          if (view.cfg.action === 'compact') await compact(agent, view, signal)
        }
      }
    } catch (error) {
      ctx.logger.warn(`context-guard: ${errorMessage(error)}`)
    }
    return next()
  })

  function report(agent) {
    const view = inspect(agent)
    if (view === undefined) return "context-guard: this session's provider is not guarded (no providers entry)."
    const { cfg, total, rate, samples, lastTtftMs, target } = view
    const trip = (cond) => (cond ? '   [TRIP]' : '')
    const lines = [`context-guard (${target.provider} / ${target.model})`]
    lines.push(`  context: ${total} tokens${cfg.hardCeilingTokens === undefined ? '' : `   ceiling ${cfg.hardCeilingTokens}${trip(total > cfg.hardCeilingTokens)}`}`)
    if (cfg.maxTtftMs !== undefined) lines.push(`  last ttft: ${lastTtftMs === undefined ? 'none yet' : sec(lastTtftMs)}   limit ${sec(cfg.maxTtftMs)}${trip(lastTtftMs !== undefined && lastTtftMs > cfg.maxTtftMs)}`)
    lines.push(`  measured rate: ${rate === undefined ? 'no samples yet' : `${tokPerSec(rate)} (${samples} sample${samples === 1 ? '' : 's'})`}`)
    if (cfg.maxColdPrefillMs !== undefined) lines.push(`  predicted cold: ${rate === undefined ? 'n/a' : minutes(total / rate).replace('~', '')}   limit ${min(cfg.maxColdPrefillMs)}${trip(rate !== undefined && total / rate > cfg.maxColdPrefillMs)}`)
    lines.push(`  compact now: ${minutes(compactCostMs(total, cfg.retainTokens, rate))}   (retain ${cfg.retainTokens})`)
    lines.push(`  action: ${cfg.action}`)
    return lines.join('\n')
  }

  ctx.commands.register({
    name: 'context-guard',
    description: 'Show context guard measurements for this session',
    handler: (invocation) => ({ kind: 'success', text: report(invocation.agent) }),
  })
}
