# Patch 08 — LiveKit Inference LLM stream tweaks

| | |
|---|---|
| **Status** | Always-on for `inference.LLM` (LiveKit Inference gateway) |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56: "fix: handle parallel tools calls", "add: bracket stripping…") |
| **Depends on** | — |
| **Automated tests** | None |

## Why

1. **`parallel_tool_calls` without tools.** Some providers behind LiveKit Inference (the
   code comment names Azure OpenAI) reject requests that set `parallel_tool_calls` when no
   tools are present. Upstream already strips `tool_choice` in that case but not
   `parallel_tool_calls`.
2. **Bracket artifacts in spoken output.** Models sometimes emit citation markers or
   bracketed annotations ("[1]", "[source]", "[pause]") that the TTS then reads aloud.

## Behaviour

Both changes are in `LLMStream` in `livekit-agents/livekit/agents/inference/llm.py`:

1. In `_run`, right after upstream's "remove `tool_choice` if no tools" step: when there
   are no tools, `parallel_tool_calls` is also removed from the extra kwargs.
2. In `_parse_choice` (the method that builds a `ChatChunk` from a streamed delta),
   right after upstream strips thinking tokens: if the delta's text content contains
   `[`, **everything from the first `[` to the end of that delta is discarded**.

## Re-applying the patch

- Wherever upstream builds the request kwargs and removes `tool_choice` for tool-less
  requests, also remove `parallel_tool_calls`.
- Wherever upstream post-processes each streamed text delta (after thinking-token
  stripping), truncate the delta at the first `[`.

## Upstream contracts relied upon

- `LLMStream._extra_kwargs` dict and `_tools` list in `inference/llm.py`.
- The per-delta parsing function and its `delta.content` attribute.

## Conflict guidance

Small insertions next to upstream code that changes occasionally (provider headers,
thinking tokens, Google thought signatures). Keep upstream's additions and re-insert the
two fork blocks at the same relative positions.

## Verification after sync

- Both blocks still exist in `inference/llm.py`, after `tool_choice` removal and after
  `strip_thinking_tokens` respectively.

## Known caveats (behavioural, keep in mind when debugging)

- The bracket filter is **per streamed delta**, not per sentence: text after a `[` in the
  same delta is lost, the closing `]` and anything after it in *later* deltas are kept
  (so "[source]" split across deltas can leave "source]" behind). Legitimate brackets are
  also stripped.
- It applies only to `inference.LLM`, not to plugin LLMs (e.g. `openai.LLM`).
- The `parallel_tool_calls` check uses "no tools or empty list"; same effect either way.

## Drop criteria

- Part 1: upstream strips `parallel_tool_calls` for tool-less requests.
- Part 2: product decision — replace with a TTS text transform (upstream now supports
  callable `tts_text_transforms`) if a sentence-level filter is preferred.
