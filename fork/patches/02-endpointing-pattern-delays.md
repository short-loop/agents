# Patch 02 — Endpointing delays for numbers and alphanumeric sequences

| | |
|---|---|
| **Status** | Always-on; switch `EndpointingOptions.readout_rules` (default True) |
| **Origin** | 1.4.6: fork commit `d7081f0` (short-loop/agents#56). 1.8.3: `dd24f08b2` (P5) and `1ede41bd3` (the `readout_rules` switch) |
| **Depends on** | Patch 03 (uses its `use_raw_delay` / `delay_reason` structure and sleep computation) |
| **Shares code with** | Patches 03 and 04 (same nested EOU task) |
| **Automated tests** | `tests/test_fork_recognition_rules.py` (both predicates); `tests/test_user_turn_exceeded.py` turns the rule off with an autouse fixture because its fixtures are number words |
| **Code markers** | `fork(patch 02)`, `fork(patch 02/03)` |

## Why

Callers frequently dictate phone numbers, account IDs, confirmation codes, postcodes and
spelled-out identifiers ("four five six, seven eight…", "A B 1 2", "alpha bravo 3 4").
People pause between digit groups; the turn detector often judges such fragments as
complete, so stock endpointing commits the turn mid-number and the agent replies to half
an ID. The fork waits longer whenever the transcript *ends* in a number-like or
alphanumeric sequence.

## Behaviour

Evaluated on the accumulated user transcript each time the end-of-utterance task runs,
**before** the turn detector is consulted, when `readout_rules` is on:

1. **Ends with a number sequence** — the last two or more words (trailing `.,!?`
   stripped) are digits or the English number words zero–nine. The endpointing delay
   becomes the endpointing object's `max_delay` (reason `ends_with_number`).
2. **Ends with an alphanumeric sequence** — the last four words are each a digit, a
   number word (zero–ten), a single letter, or a NATO phonetic letter (alpha…zulu), and
   at least one of them is numeric. The delay becomes `max(max_delay − 1.0 s, min_delay)`
   (reason `ends_with_alphanumeric`).
3. Otherwise the normal turn-detector path runs (upstream behaviour), followed by patch
   04's block.

When either pattern matches:

- The turn detector is **skipped** entirely for this evaluation.
- The delay is applied as a **raw** delay (`use_raw_delay`): it is counted from now, not
  reduced by time already elapsed since the user last spoke (see patch 03).
- If an interruption-backoff mode (patch 04, transient or sustained) is active and its
  `backoff_delay` is larger, the mode's delay wins and the reason becomes
  `ends_with_number+<mode>_backoff` / `ends_with_alphanumeric+<mode>_backoff`.

The chosen reason shows up in the `eou sleep` info log (patch 03) and the delay in
upstream's `eou_wait` span attribute (`lk.eou.delay`, set by upstream after the ladder).

`turn_handling={"endpointing": {"readout_rules": False}}` disables both rules.

## Implementation walkthrough

- `livekit-agents/livekit/agents/voice/turn.py`: `EndpointingOptions.readout_rules: bool`,
  default `True` in both `_ENDPOINTING_DEFAULTS` and `_STREAMING_ENDPOINTING_DEFAULTS`.
- `livekit-agents/livekit/agents/voice/endpointing.py`: `BaseEndpointing.readout_rules`
  attribute (default True), populated by `create_endpointing` from the options.
- `livekit-agents/livekit/agents/voice/audio_recognition.py`:
  - module level: `_NUMBER_WORDS` (zero–nine), `_NUMBER_WORDS_EXTENDED` (adds "ten"),
    `_MILITARY_LETTERS` (NATO alphabet; note the spelling "juliett"), predicates
    `_ends_with_number_like(transcript)` and `_ends_with_alpha_numeric(transcript)`. Both
    swallow exceptions and log them, returning False.
  - nested `_bounce_eou_task` inside `AudioRecognition._run_eou_detection`: the
    delay-selection ladder is `if readout_rules and number-like … elif readout_rules and
    alphanumeric … elif turn_detector is not None: <upstream block>`. The `readout_rules`
    local is `self._endpointing.readout_rules is True` (explicit check: tests stub the
    endpointing object with a `MagicMock`).

## Re-applying the patch

Before the turn-detector prediction in the EOU task, test the current accumulated
transcript for the two trailing patterns; when one matches, choose the long delay, skip
the turn detector, apply the delay unreduced, and let an active backoff-mode delay
override it if larger. Keep the reason strings stable — they are used in production log
analysis.

## Upstream contracts relied upon

- `AudioRecognition._audio_transcript` holds the committed-so-far user transcript.
- `AudioRecognition._endpointing` (a `BaseEndpointing`) with `min_delay` / `max_delay`;
  `max_delay` is a fixed ceiling even in dynamic mode.
- `create_endpointing(options)` being the single constructor of that object.

## Conflict guidance

This lives in the most conflict-prone block of the fork (see FORK.md section 3). When
upstream changes how `endpointing_delay` is chosen (per-language delays, moving the logic
into the turn detector, a new streaming-detector branch), keep upstream's new default path
as the last `elif` and put the two pattern checks in front of it.

## Verification after sync

- Both predicates still exist and are called at the top of the delay-selection chain,
  guarded by `readout_rules`.
- Reasons `ends_with_number` and `ends_with_alphanumeric` still appear in the `eou sleep`
  log.
- With default options (`max_delay` 3.0 s), a transcript ending in "five six" waits ≈3.0 s;
  one ending in "a b 1 2" waits ≈2.0 s.
- `tests/test_fork_recognition_rules.py` and `tests/test_user_turn_exceeded.py` pass.

## Known caveats

- English number words only; digits work for any language.
- Upper/lower case of single letters is irrelevant (the transcript is lower-cased).
- Any upstream test whose fixture text is number words needs the rule turned off (the
  `test_user_turn_exceeded` fixture is the template).

## Drop criteria

Upstream's turn detector (or an upstream endpointing feature) reliably holds the turn for
digit / ID dictation. Validate with production call samples before dropping.
