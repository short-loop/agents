# Patch 03 — EOU sleep timing: stale-anchor warning, sleep floor, raw-delay fallback, `eou sleep` log

| | |
|---|---|
| **Status** | Log line and stale-anchor warning always-on. Sleep floor and raw-delay fallback **opt-in** (`EndpointingOptions.sleep_floor`, `stale_anchor_raw_delay`); the stale-anchor **reset** from 1.4.6 is not re-enabled (decision D9) |
| **Origin** | 1.4.6: fork commit `d7081f0` (short-loop/agents#56: "fix last speaking time detection", "fix: floor sleep to 0.5s", "fix: eou sleep log line"). 1.8.3: `dd24f08b2` (P6) and `1ede41bd3` (opt-in keys) |
| **Depends on** | — |
| **Shares code with** | Patches 02 and 04 (same nested EOU task) |
| **Automated tests** | None dedicated; `tests/test_agent_session.py` and `tests/test_eou_wait_span.py` exercise the path with the keys off |
| **Code markers** | `fork(patch 03)`, `fork(patch 03, D9)`, `fork(patch 02/03)` |

## Why

Upstream computes the end-of-turn wait as "endpointing delay minus the time already
elapsed since the user last spoke" (`_last_speaking_time`). In production the 1.4.6 fork
observed two failure modes:

1. **Stale `_last_speaking_time`.** When VAD misses an end-of-speech / inference update
   between two STT finals, `_last_speaking_time` stays pinned to an old moment. The
   subtraction then yields zero or negative sleep and the agent commits the turn — and
   starts talking — instantly, often while the user is still mid-sentence.
2. **Jumping in too fast.** Even without staleness, very small residual sleeps made the
   agent reply with no perceptible pause.

It also needed one greppable log line per EOU decision to tune endpointing from logs.

**What changed at 1.8.** Upstream reworked the speaking-time anchor (STT-provided end
times when there is no VAD anchor, `_vad_speech_started`, the #6557 transcription-delay
fix), which narrows case 1 without proving it gone; the reset is therefore ported as a
**warning only** until 1.8 production logs show it is still needed (D9). The floor and
the raw-delay fallback shift every commit whose final arrives after `min_delay` by up to
the floor value; on by default they broke 25 upstream timing tests and add that latency on
every late transcript, so they are opt-in keys. **The production config must set both
keys to keep the 1.4.6 behaviour.**

## Behaviour

### Stale speaking-time detection (warning only)

- `AudioRecognition` remembers the previous final-transcript time
  (`_second_last_final_transcript_time`, with a class-level `None` default so partially
  constructed instances in upstream tests still work).
- After the anchor update on every STT FINAL and PREFLIGHT transcript,
  `_check_stale_speaking_anchor()` logs **warning "stale last_speaking_time detected"**
  (fields `last_speaking_time`, `second_last_final_transcript_time`, `lag`) when
  `_last_speaking_time` is older than the previous final. Nothing else changes.

### Sleep computation (opt-in)

Given the chosen endpointing delay, after the delay-selection ladder and patch 04's block:

- raw delay (patch 02 patterns) or no `last_speaking_time` → sleep the full delay
  (upstream behaviour);
- `stale_anchor_raw_delay` on **and** the delay has already fully elapsed since
  `last_speaking_time` → sleep the **full** delay again (debug "last_speaking_time appears
  stale, defaulting to raw endpointing delay"). *Upstream would sleep 0 and commit
  immediately.*
- otherwise → upstream's subtraction;
- `sleep_floor` set **and** the endpointing object's `min_delay` ≥ `sleep_floor` **and**
  the computed sleep is below it → raise the sleep to `sleep_floor`.

Both reads are type-checked (`isinstance` / `is True`) because upstream tests stub the
endpointing object with a `MagicMock`.

### `eou sleep` log line

One `logger.info("eou sleep", …)` per EOU evaluation with structured fields: `delay`
(final sleep, rounded), `reason`, `endpointing_delay`, `last_speaking_time`,
`use_raw_delay`, `trigger` (vad / stt / manual), `from_cache` (streaming detector served a
cached prediction), `end_of_turn_probability`, `unlikely_threshold`, `interruption_mode`
(patch 04). Upstream separately logs `eot prediction` (debug) and `user turn committed`
(debug); ours stays the one INFO line.

Reason values produced across patches 02/03/04: `default`, `eou_unlikely`,
`ends_with_number`, `ends_with_alphanumeric`, `ends_with_number+<mode>_backoff`,
`ends_with_alphanumeric+<mode>_backoff`, `<mode>_backoff`, `<mode>_backoff_no_eou`,
`primed_backoff`, where `<mode>` is `transient` or `sustained`. **These strings are used
by production log queries — keep them stable.**

## Implementation walkthrough

- `livekit-agents/livekit/agents/voice/turn.py`: `EndpointingOptions.sleep_floor: float |
  None` and `stale_anchor_raw_delay: bool`; defaults `None` / `False` in both defaults
  dicts.
- `livekit-agents/livekit/agents/voice/endpointing.py`: `BaseEndpointing.sleep_floor`,
  `stale_anchor_raw_delay` attributes, populated by `create_endpointing`.
- `livekit-agents/livekit/agents/voice/audio_recognition.py`:
  - class attribute default and `__init__` field `_second_last_final_transcript_time`;
  - `_process_stt_event` FINAL branch: shift last → second-last before updating, then
    `_check_stale_speaking_anchor()` after the `use_stt_speaking_time` anchor update;
    PREFLIGHT branch: the same call after its anchor update;
  - `_check_stale_speaking_anchor()` next to `get_last_user_language`;
  - nested `_bounce_eou_task`: locals `delay_reason` / `use_raw_delay` (declared before
    the ladder, initialised `from_cache = False` too), `delay_reason = "eou_unlikely"` in
    upstream's unlikely branch, the sleep computation replacing upstream's two-line
    `extra_sleep` formula, and the `eou sleep` log immediately before `delay_completed =
    False`.

## Re-applying the patch

1. Track the previous final-transcript time; warn when `_last_speaking_time` is older
   than it on FINAL and PREFLIGHT transcripts.
2. Replace the sleep formula with: raw → full delay; stale (opt-in) → full delay;
   otherwise upstream's subtraction; then the opt-in floor.
3. Emit the `eou sleep` info log with the fields above.

## Upstream contracts relied upon

- `_last_speaking_time` / `_last_final_transcript_time` semantics and the
  `_process_stt_event` anchor update (`stt_last_speaking_time`, `use_stt_speaking_time`).
- `_bounce_eou_task` receives timestamps captured at scheduling time and defines
  `end_of_turn_probability` / `unlikely_threshold` before the ladder; `trigger` is a
  parameter of `_run_eou_detection`.
- `create_endpointing` / `BaseEndpointing` (`min_delay`).

## Conflict guidance

- If upstream fixes stale anchors at the source, the warning can be dropped; flag for
  review rather than dropping during a sync.
- If upstream changes the sleep formula, keep the fork's two opt-in rules applied on top
  of the new formula.
- Keep the log message text `eou sleep` and its field names.

## Verification after sync

- The warning helper is called after both anchor updates.
- The two keys exist on `EndpointingOptions`, `BaseEndpointing` and `create_endpointing`.
- `eou sleep` still emitted with all fields.
- `tests/test_agent_session.py`, `tests/test_eou_wait_span.py`,
  `tests/test_audio_recognition_turn_detection.py` pass with the keys off.

## Known caveats

- With the keys on, the "delay already elapsed → wait the full delay again" rule adds up
  to one extra endpointing delay of latency in legitimate cases (e.g. a late final after a
  long silence). This is deliberate: a late reply is preferred over talking over the user.
- The 1.4.6 `_ignore_last_speaking_time` flag and the forced anchor reset no longer exist.
- `sleep_floor` is compared against the endpointing object's `min_delay`, which is the
  learned value in dynamic mode.

## Drop criteria

Upstream fixes stale `_last_speaking_time` at the source and production logs show no
instant commits (drops the warning). The floor / fallback and the log line would still
need to be kept or re-evaluated separately (see the dynamic-endpointing comparison in
`fork/MIGRATION-1.8.md` §6).
