# Patch 11 — STT language metadata (`SpeechData.detected_languages`, `TimedString.language`, Deepgram mapping)

| | |
|---|---|
| **Status** | Always-on (additive fields; populated by Deepgram in `language="multi"` mode) |
| **Origin** | Fork commit `e58f51d` (short-loop/agents#60) |
| **Depends on** | — |
| **Required by** | Patch 13 (multilingual heuristics read per-word language tags) |
| **Automated tests** | `tests/test_multilingual_deepgram_mapping.py` (`test_multi_mode_surfaces_detected_languages`, `test_pinned_language_unaffected`, `test_multi_mode_without_languages_key`) |

## Why

Deepgram's multilingual mode ("multi", Nova-3) reports all languages detected in a
result and a language per word. Upstream kept only the first language
(`alt["languages"][0]`, with a TODO) and discarded the per-word tags. The multilingual
adapter (patch 13) needs both to score code-switching evidence.

## Behaviour

- `stt.SpeechData` gains `detected_languages`: a list of `LanguageCode`, most prominent
  first, or None (multilingual models only).
- `types.TimedString` gains a `language` attribute (NotGivenOr string) and a matching
  keyword argument in its constructor (default NOT_GIVEN), for per-word detected
  language.
- Deepgram `live_transcription_to_speech_data`:
  - each word's `TimedString` gets `language` from the word's `language` key if present;
  - when the stream language is "multi" and the alternative has a **non-empty**
    `languages` list, `SpeechData.language` is the first entry (as upstream) and
    `detected_languages` is the full list. Upstream's check was "key present"; the fork
    uses "non-empty" so an empty list no longer raises.
- Pinned-language streams are unaffected (fields stay None / NOT_GIVEN).

## Implementation walkthrough

- `livekit-agents/livekit/agents/stt/stt.py`: new field on the `SpeechData` dataclass
  after `words`, with a docstring.
- `livekit-agents/livekit/agents/types.py`: attribute annotation and constructor
  parameter on `TimedString`, set in `__new__`.
- `livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py`: in
  `live_transcription_to_speech_data`, the word-to-`TimedString` mapping and the
  multi-language block.

## Re-applying the patch

Add the two optional fields at the end of their declarations (keeps positional
compatibility), and populate them in Deepgram's transcript mapping. If other plugins in
the fork start being used as detectors, populate the same fields there.

## Upstream contracts relied upon

- Deepgram live response shape: `alternatives[].languages` and `words[].language`.

## Conflict guidance

- If upstream implements its own multi-language surfacing (their TODO), prefer upstream's
  field names only if patch 13 is updated in the same change; otherwise keep the fork
  fields alongside.
- `SpeechData` / `TimedString` gain fields upstream from time to time — keep all fields;
  keep new fork fields last with defaults.

## Verification after sync

- Run `tests/test_multilingual_deepgram_mapping.py` and the multilingual test files
  (patch 13), which construct `TimedString` with `language=`.

## Drop criteria

Upstream exposes all detected languages and per-word languages on its own fields, and
patch 13 is migrated to them.
