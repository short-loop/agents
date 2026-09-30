# Patch 07 — `llm.ParallelAdapter` (LLM hedging / racing)

| | |
|---|---|
| **Status** | Opt-in (new class; nothing uses it unless the app constructs it) |
| **Origin** | Fork commits `d7081f0` (short-loop/agents#56, "fix: handle parallel tools calls") and `104ed89` (short-loop/agents#58, "llm hedging and observability", "winner id race") |
| **Depends on** | — |
| **Automated tests** | None |

## Why

LLM time-to-first-token has a long tail per provider/region. Racing the same request
against several endpoints (e.g. OpenAI and Azure OpenAI, or two regions) and streaming
whichever answers first cuts p99 reply latency on voice calls.

## Behaviour

- `ParallelAdapter(llm=[ParallelLLMEntry(llm=..., label=...), ...], attempt_timeout=10.0)`
  is an `llm.LLM`. At least two entries are required (ValueError otherwise).
- `ParallelLLMEntry` is a frozen dataclass pairing an LLM with a label. The adapter
  **overwrites each LLM's internal `_label`** with that label, so upstream metrics and
  traces identify the backend by it.
- `model` is "ParallelAdapter"; `provider` is "livekit".
- `chat(...)` returns a `ParallelLLMStream` which, when run:
  1. starts one task per entry calling that LLM's `chat` with the same chat context,
     tools, `parallel_tool_calls`, `tool_choice`, `extra_kwargs`, and connection options
     forced to **no retries** and `timeout = attempt_timeout`;
  2. the first entry to yield **any chunk** wins; all other tasks are cancelled
     immediately (debug log "llm.ParallelAdapter: <label> won the race");
  3. the winner's chunks are forwarded; the stream's `chat_ctx` / `tools` properties
     delegate to the winning stream once known;
  4. failures before a winner exists are logged as warnings; if all entries fail with no
     winner, the stream raises `APIConnectionError("all LLMs failed in parallel (…)")`.
- There is no failover once a winner is chosen: if the winner fails mid-stream, the
  error propagates (wrap with upstream `llm.FallbackAdapter` if needed).
- **Metrics:** the adapter subscribes to each child's `metrics_collected`, re-emits only
  metrics whose `request_id` matches the winning request (the first chunk's `id`), and
  sets `LLMMetrics.parallel_selected = True` on them. The adapter's own stream disables
  the base metrics monitor (so there is no duplicate metric for the adapter itself).
  Tracing span name is `llm_parallel_adapter`.
- `LLMMetrics` gains an optional `parallel_selected` field (None when the metric does
  not come from a ParallelAdapter).
- `aclose()` unsubscribes from the children's metrics events (it does not close the
  child LLMs).

## Implementation walkthrough

- New file `livekit-agents/livekit/agents/llm/parallel_adapter.py`: `ParallelLLMEntry`,
  `ParallelAdapter`, `ParallelLLMStream`. Winner bookkeeping uses a shared
  `_winning_request_ids` set on the adapter, added when the winner's first chunk arrives
  and discarded when the stream finishes; a done-callback on every race task closes the
  internal channel when the winner finishes or all tasks are done.
- `livekit-agents/livekit/agents/llm/__init__.py`: imports and exports `ParallelAdapter`,
  `ParallelLLMEntry`.
- `livekit-agents/livekit/agents/metrics/base.py`: `LLMMetrics.parallel_selected`.

## Re-applying the patch

The new module is fork-only; re-add the two exports and the metrics field. If upstream's
`LLM` / `LLMStream` base classes change signatures, update the adapter to match (see
contracts below).

## Upstream contracts relied upon

- `LLM.__init__()` with no args, `LLM._label` / `LLM.label`, `LLM.on/off/emit` for
  "metrics_collected".
- `LLM.chat(...)` keyword signature: `chat_ctx`, `tools`, `conn_options`,
  `parallel_tool_calls`, `tool_choice`, `extra_kwargs`.
- `LLMStream.__init__(llm, chat_ctx=, tools=, conn_options=)`, `_run`, `_event_ch`,
  `_chat_ctx`, `_tools`, `_metrics_monitor_task(event_aiter)`, class attribute
  `_llm_request_span_name`.
- `ChatChunk.id` equals `LLMMetrics.request_id` of the same request.
- `APIConnectOptions(max_retry=, timeout=)`.

## Conflict guidance

Conflicts only in `llm/__init__.py` (keep both export lists) and `metrics/base.py` (keep
the field). The real risk is **silent breakage** when upstream changes the `LLM.chat`
signature (new keyword args must be forwarded) or `LLMStream` internals — check the
contracts after every sync and run a type check.

## Verification after sync

- Type check passes for `llm/parallel_adapter.py`.
- `LLM.chat` in upstream `llm/llm.py` has no new parameters that the adapter fails to
  accept and forward.
- `from livekit.agents.llm import ParallelAdapter, ParallelLLMEntry` still works.

## Known caveats

- The field docstring says losing streams get `parallel_selected = False`, but losers'
  metrics are filtered out and never emitted; in practice only True or None is observed.
- Racing multiplies provider cost and rate-limit usage by the number of entries (losers
  are cancelled after the first chunk, but prompt tokens are still billed).
- The winner is chosen on the first chunk of any kind (including a role-only chunk).

## Drop criteria

Upstream ships an equivalent hedging adapter with first-chunk racing, cancellation of
losers and winner-only metrics.
