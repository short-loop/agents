# Patch 08 — LiveKit Inference LLM stream tweaks

| | |
|---|---|
| **Status** | Always-on for `inference.LLM`; bracket stripping switchable (`strip_brackets`) and turned off automatically for expressive turns |
| **Origin** | 1.4.6: fork commit `d7081f0` (short-loop/agents#56). 1.8.3: `ef466656d` (P1), `f778357b0` (P2), `376aae4b4` (mypy follow-up) |
| **Depends on** | — |
| **Automated tests** | None dedicated; `tests/test_llm_utils.py` builds `LLMStream` objects without `__init__` and exercises the class default |
| **Code markers** | `fork(patch 08)` |

## Why

1. **`parallel_tool_calls` without tools.** Some providers behind LiveKit Inference (the
   code comment names Azure OpenAI) reject requests that set `parallel_tool_calls` when no
   tools are present. Upstream already strips `tool_choice` in that case but not
   `parallel_tool_calls`.
2. **Bracket artifacts in spoken output.** Models sometimes emit citation markers or
   bracketed annotations ("[1]", "[source]", "[pause]") that the TTS then reads aloud.
   Upstream's expressive mode (1.7+) makes the LLM emit bracketed TTS markup on purpose,
   so the filter must be off for expressive turns (decision D1).

## Behaviour

Both changes are in `livekit-agents/livekit/agents/inference/llm.py`:

1. In `LLMStream._run`, right after upstream's "remove `tool_choice` if no tools" step:
   when there are no tools, `parallel_tool_calls` is also removed from the extra kwargs.
2. In `_parse_choice`, right after upstream strips thinking tokens: if bracket stripping
   is on and the delta's text content contains `[`, **everything from the first `[` to
   the end of that delta is discarded** (kept as in 1.4.6, decision D1).
   - `inference.LLM(strip_brackets=True)` constructor option, stored on `_LLMOptions`,
     changeable with `update_options(strip_brackets=...)`; the stream reads it once at
     creation (`_strip_brackets`, class default `False` for streams built without
     `__init__`).
   - `AgentActivity._pipeline_reply_task_impl` calls `_set_bracket_stripping(self.llm,
     enabled=expressive is None)` right where it resolves the turn's expressive options,
     so the filter is off for expressive turns and back on otherwise. The helper unwraps
     `llm.ParallelAdapter` entries and `llm.FallbackAdapter` instances to reach the
     inference LLMs inside.

## Re-applying the patch

- Wherever upstream builds the request kwargs and removes `tool_choice` for tool-less
  requests, also remove `parallel_tool_calls`.
- Wherever upstream post-processes each streamed text delta (after thinking-token
  stripping), truncate the delta at the first `[` when the option is on; keep the
  expressive-mode gate where the activity resolves expressive options per turn.

## Upstream contracts relied upon

- `LLMStream._extra_kwargs` dict and `_tools` list; `llm_v` (the typed inference LLM) in
  the stream constructor; `_LLMOptions`.
- The per-delta parsing function and its `delta.content` attribute.
- `AgentActivity._resolve_expressive_options()` returning None when expressive is off for
  the turn; `ParallelAdapter._entries`, `FallbackAdapter._llm_instances`.

## Conflict guidance

Small insertions next to upstream code that changes occasionally (provider headers,
thinking tokens, Google thought signatures). Keep upstream's additions and re-insert the
fork blocks at the same relative positions. If upstream reshapes expressive resolution,
move the `_set_bracket_stripping` call with it.

## Verification after sync

- Both blocks still exist in `inference/llm.py`; `strip_brackets` is on `_LLMOptions`, the
  constructor and `update_options`.
- `_set_bracket_stripping` is still called once per pipeline reply.
- `tests/test_llm_utils.py` passes.

## Known caveats (behavioural, keep in mind when debugging)

- The bracket filter is **per streamed delta**, not per sentence: text after a `[` in the
  same delta is lost, the closing `]` and anything after it in *later* deltas are kept
  (so "[source]" split across deltas can leave "source]" behind). Legitimate brackets are
  also stripped. Kept deliberately (D1); a cross-chunk regex was considered and rejected
  for now.
- It applies only to `inference.LLM`, not to plugin LLMs (e.g. `openai.LLM`).
- The gate mutates the shared LLM instance's option per turn; with one LLM per session
  that is fine, a shared LLM across expressive and non-expressive sessions would flap.

## Drop criteria

- Part 1: upstream strips `parallel_tool_calls` for tool-less requests.
- Part 2: product decision — replace with a TTS text transform (upstream supports
  callable `tts_text_transforms`) if a sentence-level filter is preferred.
