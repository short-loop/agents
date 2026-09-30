# Patch 02 — Endpointing delays for numbers and alphanumeric sequences

| | |
|---|---|
| **Status** | Always-on (no switch) |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56, "chore: migrate sl patches") |
| **Depends on** | Patch 03 (uses its `use_raw_delay` / `delay_reason` structure) |
| **Shares code with** | Patches 03 and 04 (same nested EOU task) |
| **Automated tests** | None |

## Why

Callers frequently dictate phone numbers, account IDs, confirmation codes, postcodes and
spelled-out identifiers ("four five six, seven eight…", "A B 1 2", "alpha bravo 3 4").
People pause between digit groups; the turn detector often judges such fragments as
complete, so stock endpointing commits the turn mid-number and the agent replies to half
an ID. The fork waits longer whenever the transcript *ends* in a number-like or
alphanumeric sequence.

## Behaviour

Evaluated on the accumulated user transcript each time the end-of-utterance task runs,
**before** the turn detector is consulted:

1. **Ends with a number sequence** — the last two or more words (trailing `.,!?`
   stripped) are digits or the English number words zero–nine. The endpointing delay
   becomes `max_endpointing_delay` (reason `ends_with_number`).
2. **Ends with an alphanumeric sequence** — the last four words are each a digit, a
   number word (zero–ten), a single letter, or a NATO phonetic letter (alpha…zulu), and
   at least one of them is numeric. The endpointing delay becomes
   `max_endpointing_delay − 1.0 s` (reason `ends_with_alphanumeric`).
3. Otherwise the normal turn-detector path runs (upstream behaviour, extended by patch 04).

When either pattern matches:

- The turn detector is **skipped** entirely for this evaluation.
- The delay is applied as a **raw** delay (`use_raw_delay`): it is counted from now,
  not reduced by time already elapsed since the user last spoke (see patch 03).
- If an interruption-backoff mode (patch 04, transient or sustained) is active and its
  `backoff_delay` is larger, the mode's delay wins and the reason becomes
  `ends_with_number+<mode>_backoff` / `ends_with_alphanumeric+<mode>_backoff`.

The chosen reason shows up in the `eou sleep` info log (patch 03).

## Implementation walkthrough

All in `livekit-agents/livekit/agents/voice/audio_recognition.py`:

- Module level: word sets `_NUMBER_WORDS` (zero–nine), `_NUMBER_WORDS_EXTENDED` (adds
  "ten"), `_MILITARY_LETTERS` (NATO alphabet; note the spelling "juliett"), and two
  predicates `_ends_with_number_like(transcript)` and `_ends_with_alpha_numeric(transcript)`.
  Both swallow exceptions and log them, returning False.
- Nested coroutine `_bounce_eou_task` inside `AudioRecognition._run_eou_detection`: the
  delay selection is an if / elif chain — number-like, then alphanumeric, then the turn
  detector branch, then the "no turn detector but a mode is active" branch (patch 04).

## Re-applying the patch

Before the turn-detector prediction in the EOU task, test the current accumulated
transcript for the two trailing patterns; when one matches, choose the long delay, skip
the turn detector, apply the delay unreduced, and let an active backoff-mode delay
override it if larger. Keep the reason strings stable — they are used in production log
analysis.

## Upstream contracts relied upon

- `AudioRecognition._audio_transcript` holds the committed-so-far user transcript.
- `AudioRecognition._max_endpointing_delay` is the session's max delay (updated by
  `update_options`).

## Conflict guidance

This lives in the most conflict-prone block of the fork (see FORK.md section 3). When
upstream changes how `endpointing_delay` is chosen (e.g. new dynamic endpointing,
per-language delays, or moving the logic into the turn detector), keep upstream's new
default path as the "else" branch and put the two pattern checks in front of it.

## Verification after sync

- Both predicates still exist and are called at the top of the delay-selection chain.
- Reasons `ends_with_number` and `ends_with_alphanumeric` still appear in the `eou sleep`
  log.
- With default options (`max_endpointing_delay` 3.0 s), a transcript ending in "five six"
  waits ≈3.0 s; one ending in "a b 1 2" waits ≈2.0 s.

## Known caveats

- English number words only; digits work for any language.
- With `max_endpointing_delay` below 1.0 s the alphanumeric delay would be negative; the
  sleep is then skipped unless the 0.5 s floor from patch 03 applies.
- Upper/lower case of single letters is irrelevant (the transcript is lower-cased).

## Drop criteria

Upstream's turn detector (or an upstream endpointing feature) reliably holds the turn for
digit / ID dictation. Validate with production call samples before dropping.
