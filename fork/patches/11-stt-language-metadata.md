# Patch 11 — STT language metadata (`TimedString.language`, Deepgram `source_languages` + per-word tags)

| | |
|---|---|
| **Status** | Always-on (additive; populated by Deepgram in `language="multi"` mode) |
| **Origin** | 1.4.6: fork commit `e58f51d` (short-loop/agents#60). 1.8.3: part of `d751a2ac1` (P10) |
| **Depends on** | — |
| **Required by** | Patch 13 (multilingual heuristics read per-word language tags) |
| **Automated tests** | `tests/test_multilingual_deepgram_mapping.py` (`test_multi_mode_surfaces_source_languages`, `test_pinned_language_unaffected`, `test_multi_mode_without_languages_key`) |
| **Code markers** | none in code (the `TimedString` docstring says "fork") |

## Why

Deepgram's multilingual mode ("multi", Nova-3) reports all languages detected in a
result and a language per word. Upstream kept only the first language
(`alt["languages"][0]`, with a TODO) and discarded the per-word tags. The multilingual
adapter (patch 13) needs both to score code-switching evidence.

**What changed at 1.8.** Upstream added `SpeechData.source_languages` ("multi-language
detection services: `language` holds the dominant language and `source_languages` all
detected languages sorted by prevalence"), which is exactly the 1.4.6 fork's
`detected_languages`. The fork field was dropped in favour of it (decision D7); the fork
now only populates it in the Deepgram mapper. `TimedString` gained an upstream
`speaker_id` positional parameter, so the fork's `language` comes after it.

## Behaviour

- `types.TimedString` gains a `language` attribute (NotGivenOr string) and a matching
  keyword argument in its constructor (default NOT_GIVEN, **after** `speaker_id`), for
  per-word detected language. Always pass it by keyword.
- Deepgram `live_transcription_to_speech_data`:
  - each word's `TimedString` gets `language` from the word's `language` key if present;
  - when the stream language is "multi" and the alternative has a **non-empty**
    `languages` list, `SpeechData.language` is the first entry (as upstream) and
    `source_languages` is the full list. Upstream's check was "key present"; the fork
    uses "non-empty" so an empty list no longer raises.
- Pinned-language streams are unaffected (fields stay None / NOT_GIVEN).
- `stt/stt.py` is **not** modified any more.

## Implementation walkthrough

- `livekit-agents/livekit/agents/types.py`: attribute annotation, docstring line and
  constructor parameter on `TimedString`, set in `__new__`.
- `livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py`: in
  `live_transcription_to_speech_data`, the word-to-`TimedString` mapping (`language=`)
  and the multi-language block (`sd.source_languages = [...]`).

## Re-applying the patch

Keep the optional attribute last on `TimedString` (keyword-only in practice) and populate
`source_languages` plus per-word `language` in Deepgram's transcript mapping. If other
plugins in the fork start being used as detectors, populate the same fields there.
Deepgram's v2 (`stt_v2.py`, Flux) already populates `source_languages` upstream but has
no per-word tags.

## Upstream contracts relied upon

- `SpeechData.source_languages` (upstream field) and its meaning.
- Deepgram live response shape: `alternatives[].languages` and `words[].language`.
- `TimedString.__new__` positional order (`text, start_time, end_time, confidence,
  start_time_offset, speaker_id`).

## Conflict guidance

- `TimedString` gains parameters upstream from time to time — keep the fork's `language`
  last and always keyword.
- If upstream surfaces per-word languages itself, prefer its attribute and update patch
  13's `_word_language_fraction` (it reads `word.language` via `getattr`).

## Verification after sync

- Run `tests/test_multilingual_deepgram_mapping.py` and the multilingual test files
  (patch 13), which construct `TimedString` with `language=`.

## Drop criteria

Upstream exposes per-word languages on its own field and patch 13 is migrated to it
(`source_languages` is already upstream's).
