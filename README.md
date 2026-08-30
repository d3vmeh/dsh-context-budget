# dsh-context-budget

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that keeps a local model's context at a size your GPU handles well. 

dsh compacts a conversation when it reaches a fixed fraction of the model's declared context window. On local hardware the declared window is not very practical. For example, a 27B model that advertises 256K may read prompts at 300 tokens per second when the context is small and at 70 tokens per second past 100K. So a turn could take a few seconds to start early in the conversation, but later in the same session take half an hour. Also, the GPU may hang before the window is full. This plugin measures how expensive the context has become and warns or compacts early before that point.

## What it checks

Before every agent step, for each guarded provider route:

| Check | Setting | Trips when |
|---|---|---|
| Hard ceiling | `hardCeilingTokens` | the context is larger than this many tokens |
| Observed slowness | `maxTtftMs` | the last reply on this context took longer than this before its first token |
| Predicted slowness | `maxColdPrefillMs` | context size divided by the measured prefill rate exceeds this (the wait if the server's prompt cache were lost) |

Any configured check can be tripped. If you leave the check unset, it is skipped. The prefill rate is measured on every model request from the time to the first chunk and the uncached prompt tokens the server reported.

## Install

```
dsh plugin --profile web add dsh-context-budget
```

Then in `~/.dsh/profiles/web/cordis.patch.yml`:

```yaml
- id: context-budget
  config:
    providers:
      llamacpp:
        hardCeilingTokens: 110000    # optional
        maxTtftMs: 180000            # optional, 3 minutes
        maxColdPrefillMs: 600000     # optional, 10 minutes
        retainTokens: 24000          # kept verbatim when compacting (default 16000)
        action: warn                 # warn (default) or compact
```

The provider key is the route name from your dsh settings. Each provider entry needs at least one check. After installing the plugin, restart `dsh web` and open a new session. 

Check the composed config with `dsh --profile web --dump-config`.

## What it does when a check trips

- `warn`: prints one line to the dsh terminal, naming the check and the current values, as well as an estimate of what compacting now would cost:

  ```
  context-budget: llamacpp session=d1e4be82 predicted: cold prefill 37.2 min > 10.0 min (152340 tokens, rate 68 tok/s, compacting now ~41.2 min)
  ```

- `compact`: prints the same line, then asks dsh's compaction engine to summarize everything except the most recent `retainTokens`, cut at a tool-call boundary. The outcome is printed too (`compacted N items (~T tokens)` or `compaction failed: ...; continuing`). The step always continues.

Compaction itself costly on local runs ( summary prompt and the rewritten context both prefill cold) which is what the `compacting now` estimate shows, so a lower ceiling with a larger `retainTokens` usually costs less overall than many small compactions.

## `/context-budget`

Type it in a session to see the current values:

```
context-budget (llamacpp / qwen3.8-long)
  context: 152340 tokens   ceiling 110000   [TRIP]
  last ttft: 41 s   limit 180 s
  measured rate: 68 tok/s (3 samples)
    3 min ago   8400 tok in 37 s   cold
    12 min ago   2100 tok in 8 s   warm
    41 min ago   12800 tok in 55 s   cold
  predicted cold: 37.3 min   limit 10.0 min   [TRIP]
  compact now: ~41.2 min   (retain 24000)
  action: warn
```

Each sample line shows when it was measured, how many uncached tokens the server had to read, the wait for the first token, and whether the server answered partly from its prompt cache (warm) or read everything fresh (cold). Warm samples have flattering timings; the list makes that visible.

## Notes

- Works with any provider dsh streams from since nothing talks to the model server directly. Measurements come from dsh's own usage reports and wall-clock timing.
- The plugin runs beside dsh's `compaction-basic`, which keeps its own threshold. Setting a per-model `thresholdRatio` there gives you a static ceiling with no plugin. The plugin adds the measured checks and the cost estimate.
- Compaction summary requests are not measured, and an observed slow reply is forgotten once the context has been rewritten so one compaction does not trigger another.
- If the session's agent preset has no compaction engine, `compact` behaves as `warn` (logged once per session).
- Log lines are printed only when a check trips; a session that stays under the limits prints nothing.

## License

MIT
