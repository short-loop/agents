# Patch 03 — EOU sleep timing: stale VAD detection, 0.5 s floor, `eou sleep` log

| | |
|---|---|
| **Status** | Always-on (no switch) |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56: "fix last speaking time detection", "fix: floor sleep to 0.5s", "fix: eou sleep log line") |
| **Depends on** | — |
| **Shares code with** | Patches 02 and 04 (same nested EOU task) |
| **Automated tests** | None dedicated (exercised indirectly by `tests/test_agent_session.py`) |

## Why

Upstream computes the end-of-turn wait as "endpointing delay minus the time already
elapsed since the user last spoke" (`_last_speaking_time`, set by VAD). In production the
fork observed two failure modes:

1. **Stale `_last_speaking_time`.** When VAD misses an end-of-speech / inference update
   between two STT finals, `_last_speaking_time` stays pinned to an old moment. The
   subtraction then yields zero or negative sleep and the agent commits the turn — and
   starts talking — instantly, often while the user is still mid-sentence.
2. **Jumping in too fast.** Even without staleness, very small residual sleeps made the
   agent reply with no perceptible pause.

It also needed one greppable log line per EOU decision to tune endpointing from logs.

## Behaviour

### Stale speaking-time detection

- `AudioRecognition` now remembers the previous final-transcript time
  (`_second_last_final_transcript_time`) in addition to the last one.
- On every STT FINAL and PREFLIGHT transcript: if `_last_speaking_time` is **older than
  the previous final transcript**, VAD has not updated it since then. The fork resets
  `_last_speaking_time` to now and sets a flag `_ignore_last_speaking_time`.
- The flag is cleared whenever VAD proves it is alive again: START_OF_SPEECH,
  INFERENCE_DONE, END_OF_SPEECH, or when the STT path sets `_last_speaking_time` itself
  (VAD disabled / first transcript).
- The flag value is captured when the EOU task is created (like the other timestamps) and
  passed into it.

### Sleep computation (`compute_sleep`)

Given the chosen endpointing delay:

- flag set → sleep the **full** delay (log "ignore_last_speaking_time set, using raw
  endpointing delay");
- the delay has **already fully elapsed** since `last_speaking_time` → sleep the **full**
  delay again (log "last_speaking_time appears stale, defaulting to raw endpointing
  delay"). *This differs from upstream, which would sleep 0 and commit immediately.*
- otherwise → upstream behaviour: delay minus time elapsed since last speaking;
- no `last_speaking_time` → full delay.

Patterns from patch 02 bypass `compute_sleep` altogether (`use_raw_delay`).

### 0.5 s floor

If the session's `min_endpointing_delay` is at least 0.5 s and the computed sleep is
below 0.5 s, the sleep is raised to 0.5 s. Sessions configured with a smaller minimum
delay are unaffected.

### `eou sleep` log line

One `logger.info("eou sleep", …)` per EOU evaluation with structured fields: `delay`
(final sleep, rounded), `reason` (see list below), `endpointing_delay`,
`last_speaking_time`, `use_raw_delay`, `ignore_last_speaking_time`, `interruption_mode`
(patch 04).

Reason values produced across patches 02/03/04: `default`, `eou_unlikely`,
`ends_with_number`, `ends_with_alphanumeric`, `ends_with_number+<mode>_backoff`,
`ends_with_alphanumeric+<mode>_backoff`, `<mode>_backoff`, `<mode>_backoff_no_eou`,
`primed_backoff`, where `<mode>` is `transient` or `sustained`. **These strings are used
by production log queries — keep them stable.**

### Minor

Upstream's direct assignment `endpointing_delay = max_endpointing_delay` inside the
turn-detector `try` block is replaced by recording `predict_ok` and deciding after the
`try` (restructure needed by patch 04). With interruption backoff disabled the resulting
delay is identical to upstream (`eou_unlikely` → max delay).

## Implementation walkthrough

All in `livekit-agents/livekit/agents/voice/audio_recognition.py`:

- `AudioRecognition.__init__`: new attributes `_second_last_final_transcript_time` and
  `_ignore_last_speaking_time`.
- `_on_stt_event`, FINAL_TRANSCRIPT branch: shift last → second-last final time before
  updating; clear the flag when STT sets `_last_speaking_time`; staleness check after.
- `_on_stt_event`, PREFLIGHT_TRANSCRIPT branch: same flag clear + staleness check (the
  second-last time is only shifted on FINAL).
- `_on_vad_event`: clear the flag on START_OF_SPEECH, INFERENCE_DONE and END_OF_SPEECH.
- `_run_eou_detection` → nested `_bounce_eou_task`: new parameter
  `ignore_last_speaking_time`; local `use_raw_delay`, `delay_reason`; nested helper
  `compute_sleep`; 0.5 s floor; `eou sleep` log. The task creation at the bottom of
  `_run_eou_detection` passes the captured flag (comment changed to "copy the values
  before awaiting").

## Re-applying the patch

1. Track the previous final-transcript time; detect `_last_speaking_time` older than it
   on FINAL and PREFLIGHT transcripts; reset it and raise an "unreliable" flag; clear the
   flag on any VAD event or STT-sourced update.
2. Snapshot the flag with the other timestamps when scheduling the EOU task.
3. In the sleep computation: full delay when the flag is set or the delay has already
   elapsed; otherwise upstream's subtraction.
4. Floor at 0.5 s when `min_endpointing_delay ≥ 0.5`.
5. Emit the `eou sleep` info log with the fields above.

## Upstream contracts relied upon

- `_last_speaking_time` semantics (VAD-driven, updated on INFERENCE_DONE / END_OF_SPEECH
  upstream) and `_last_final_transcript_time`.
- `_bounce_eou_task` receives timestamps captured at scheduling time.

## Conflict guidance

- If upstream starts using STT END_OF_SPEECH timestamps for `_last_speaking_time` (their
  own TODO in the FINAL branch mentions this), the staleness heuristic may become
  unnecessary — flag for review rather than dropping during a sync.
- If upstream changes the sleep formula, keep the fork's two "full delay" rules and the
  floor applied on top of the new formula.
- Keep the log message text `eou sleep` and its field names.

## Verification after sync

- `_ignore_last_speaking_time` is cleared in all three VAD branches and set only by the
  staleness checks.
- The EOU task signature includes the flag, and the scheduling call passes it.
- `eou sleep` log still emitted with all fields.
- `tests/test_agent_session.py` passes (its timing-sensitive tests exercise this path).

## Known caveats

- The "delay already elapsed → wait the full delay again" rule can add up to one extra
  endpointing delay of latency in legitimate cases (e.g. EOU re-evaluated long after the
  user stopped). This is deliberate: a late reply is preferred over talking over the user.

## Drop criteria

Upstream fixes stale `_last_speaking_time` at the source (e.g. STT-provided end-of-speech
timestamps) and production logs show no instant commits. The floor and the log line
would still need to be kept or re-evaluated separately.
