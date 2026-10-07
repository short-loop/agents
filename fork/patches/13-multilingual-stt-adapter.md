# Patch 13 — `stt.MultilingualAdapter` (silent multilingual switching)

| | |
|---|---|
| **Status** | Opt-in (new class; used only when the app wraps its STT with it) |
| **Origin** | 1.4.6: fork commits `e58f51d` (short-loop/agents#60), `79c0347` (short-loop/agents#61), `062d5dc` (short-loop/agents#62). 1.8.3: `d751a2ac1` (P10) with the hook forwarding below |
| **Depends on** | Patch 11 (per-word language tags; `source_languages`), Patch 12 (Deepgram timestamp continuity) |
| **Automated tests** | `tests/test_stt_multilingual.py` (25 async tests), `tests/test_multilingual_heuristics.py` (19 tests), helper `tests/fake_multilingual_stt.py` (`ScriptedSTT` / `ScriptedStream`: streams that emit exactly the events a test injects and record received frames / flushes) |
| **Example** | `examples/voice_agents/multilingual_switching.py` |

## Why

Callers switch language mid-call (e.g. English → Hindi or Spanish). A language-pinned
STT transcribes foreign speech as garbled text in the pinned language; a pure
multilingual model is less accurate than a pinned one for the main language. The adapter
keeps a pinned, accurate **primary** STT and an always-on multilingual **detector**; when
the detector's evidence shows the caller switched language (or the app/LLM asks for a
switch), the primary is moved to the new language **without losing any speech** — the
detector's transcripts cover the transition.

## Public API

All exported from `livekit.agents.stt` (and `livekit.agents.stt.multilingual`):

- `MultilingualAdapter(detector=, initial_language=, primary=None, primary_factory=None, options=LanguageSwitchOptions())`
  — an `stt.STT`. Requires `primary` or `primary_factory`; both primary and detector must
  be streaming-capable (wrap others with `stt.StreamAdapter`); `initial_language` must be
  in the allowlist if one is set; `switch_mode="recreate"` requires a factory. When only
  a factory is given, the adapter builds and owns the initial primary. Capabilities
  mirror the primary's (always streaming; `keyterms` and `chat_context` copied too).
  `model` "MultilingualAdapter", `provider` "livekit".
  - Framework STT hooks are forwarded to **both** children: `_update_session_keyterms`
    (remembered and re-applied to primaries built later by the factory),
    `_push_conversation_item`, `prewarm`.
  - `current_language` property, `options` property.
  - `async switch_language(language, *, reason="manual")` — manual switch (e.g. from an
    LLM function tool). Raises `LanguageNotAllowedError` for non-allowlisted languages and
    `LanguageSwitchFailedError` on rollback. With no live stream, just sets the language
    for the next stream.
  - `recognize()` (non-streaming) delegates to the current primary in the current
    language.
  - Events (emitted on the adapter): `language_switch_started`, `language_switched`,
    `language_switch_suppressed`, `language_switch_evidence`, and re-emitted
    `metrics_collected` from every child STT.
- `MultilingualRecognizeStream` — the stream returned by `stream()`.
- `LanguageSwitchOptions` — see below.
- Errors: `LanguageNotAllowedError` (ValueError; carries `language`, `allowed`),
  `LanguageSwitchFailedError` (RuntimeError; carries `target`, `reason`).
- Event dataclasses (frozen):
  - `LanguageSwitchStartedEvent` (old, new, initiator "heuristic"/"manual", reason) —
    fired **before** the switch so the app can retune TTS voice / prompt for the next
    reply.
  - `LanguageSwitchedEvent` (old, new, initiator, executor "in_place"/"recreate",
    `latency` = wall-clock seconds from first evidence to completion, 0.0 for manual,
    `trigger_transcript`).
  - `LanguageSwitchSuppressedEvent` (target, reason, score). Reasons: `reentry`,
    `allowlist`, `auto_switch_disabled`, `switch_in_progress`, `switch_failed`,
    `no_executor`.
  - `LanguageSwitchEvidenceEvent` (current language, audio time, per-language scores) —
    throttled snapshot.

### `LanguageSwitchOptions` (all speech timers in **audio time**)

| Option | Default | Meaning |
|---|---|---|
| `languages` | None | Allowlist of switch targets (None = any). Evidence for others is tracked and observable but never acted on. |
| `auto_switch` | True | False = telemetry-only (evidence / suppressed events, no automatic switch; manual still works). |
| `switch_mode` | "auto" | "in_place" (primary stream must expose `update_options(language=...)`), "recreate" (needs `primary_factory`), "auto" (in-place if supported, else factory, else none). |
| `switch_threshold` | 2.0 | Evidence needed, in "confident full utterances". |
| `min_detector_confidence` | 0.6 | Below this, a final contributes nothing. |
| `default_confidence` | 0.7 | Used when the detector reports confidence 0.0. |
| `min_transcript_length` | 3 | Transcripts of at most this many characters are ignored. |
| `interim_evidence_weight` | 0.0 | Weight of interims (0 = finals only). |
| `evidence_half_life_s` | 20.0 | Exponential decay of evidence. |
| `script_mismatch_boost` | 1.5 | Multiplier and per-final cap for cross-script finals. |
| `cross_script_length_floor` | 0.5 | Minimum length weight for cross-script finals. |
| `turn_bonus` | 0.5 | Added from the 2nd consecutive final in the same non-current language. |
| `word_length_cap` | 8 | Word count giving full length weight. |
| `reentry_threshold_multiplier` | 2.0 | Bar multiplier for the switched-away language after a heuristic switch, decaying linearly to 1× over `reentry_decay_s`. |
| `manual_reentry_multiplier` | 2.0 | Same after a manual switch. |
| `reentry_decay_s` | 60.0 | Re-entry decay time. |
| `detector_rescue_s` | 2.0 | Rescue buffered detector finals the primary never covered after this long (0 disables). Must be shorter than the session's final-transcript timeout. |
| `rescued_final_boost` | 1.5 | Multiplier and cap when re-scoring a rescued final (1.0 disables). |
| `boundary_silence_s` | 0.8 | Silence after a detector transcript that ends the transition window. |
| `max_detector_owns_s` | 15.0 | Hard cap on the transition window. |
| `switch_timeout_s` | 10.0 | Max time for the executor to produce the new primary. |
| `switch_grace_s` | 2.0 | Wait for a first transcript / failure from the new stream; silence counts as healthy. |
| `evidence_event_interval_s` | 2.0 | Throttle for evidence events. |
| `detector_restart` | True | Restart a failed detector with exponential backoff (1 s → 30 s). |

## How it works

### Topology

`MultilingualRecognizeStream._run` opens two child streams: the **primary** (in the
current language) and the **detector**. An input task fans every pushed audio frame and
flush out to all live children and advances an internal **audio clock**. One pump task
per child pushes its events into a merged channel; a single **gate loop** consumes that
channel and decides what reaches the session. An **owner** flag says which child is the
"transcriber of record": normally `primary`; `detector` during a switch.

Routing rules:

- `RECOGNITION_USAGE` from every child is always forwarded (billing accuracy).
- Primary events are forwarded while the primary owns — except a primary FINAL whose
  `end_time` is already covered by forwarded content (a rescued / replayed detector final
  already delivered that speech).
- Detector events always feed the heuristic engine; they are forwarded only while the
  detector owns (FINAL / PREFLIGHT gated on `end_time` past the flip gate; INTERIM,
  START_OF_SPEECH, END_OF_SPEECH forwarded as-is).
- Events from a shadow (future primary) stream are suppressed until promotion; events
  from retired streams are dropped.

### Heuristic engine

`_HeuristicEngine` in `heuristics.py` is pure and synchronous. Its **module docstring is
the authoritative description of the scoring pipeline** (units, gates, per-final delta,
decay, re-entry bar, rescued-final re-score, a worked example from a real call, and the
trace log format) — read it before touching thresholds. In short: each detector final
passes gates (`untagged`, `current_language` — which also resets streaks, `too_short`,
`low_confidence`), then contributes a delta built from length weight, confidence, word
composition (fraction of words tagged with the candidate language, via patch 11) and a
cross-script boost (dominant Unicode script not belonging to the current language, from
a built-in script/language table), plus a turn bonus for consecutive finals. Scores decay
with the half-life. A switch fires when a candidate's score crosses
`switch_threshold × re-entry multiplier`, subject to allowlist and `auto_switch`
(suppressed events otherwise). After a switch, evidence is cleared and the switched-away
language gets an elevated, linearly decaying bar (never a hard block). A failed switch
also elevates the target's bar to avoid retry storms. Every scored final produces a
trace, logged by the adapter at debug level as
"multilingual adapter: detector heard '…' lang= conf= delta= score= bar= [gate=]".

### Switch flow (`_do_switch`, serialized by a lock)

1. Emit `language_switch_started`.
2. **Flip 1:** owner becomes `detector`; the flip gate is set to the end of the last
   final the session received; empty INTERIM + empty FINAL "clear" events are emitted to
   reset dangling interims downstream; buffered detector finals past the gate are
   **replayed** so the utterance that triggered the switch reaches the session.
3. **Executor:** in-place (`update_options(language=...)` on the primary stream, same
   object) or recreate (factory builds a new STT, adopted and owned by the adapter, and a
   shadow stream already receiving live audio). Bounded by `switch_timeout_s`.
4. **Health:** wait up to `switch_grace_s` for a transcript or failure from the new
   stream; silence counts as healthy.
5. **Transition window:** wait for an utterance boundary — `boundary_silence_s` of
   audio-time silence since the last detector transcript (watchdog polls every 0.1 s), a
   detector END_OF_SPEECH after health, primary end, or `max_detector_owns_s`.
6. **Flip 2:** for recreate, the shadow becomes the primary and the old primary is
   retired (identity registered before the owner flip so there is no await in between);
   owner back to `primary`; clear events; language updated on stream and adapter; engine
   notified; `language_switched` emitted.
7. **Rollback** on any failure: retire the shadow (recreate) or revert
   `update_options` to the old language (in-place), owner back to `primary`, clear
   events, engine notified, `language_switch_suppressed("switch_failed")`, and
   `LanguageSwitchFailedError` raised (logged as warning for heuristic switches).

Heuristic decisions while a switch is running produce `switch_in_progress`; with no
executor available, `no_executor`.

### No-loss / no-duplicate machinery

These rules came from UAT call analysis and each fixes a real incident; **do not
simplify them during a sync**:

- **Dedup watermark** (`_forwarded_final_end_ts`): the furthest `end_time` of any final
  forwarded to the session. Updated by `_note_forwarded_final`, which also prunes
  buffered detector finals now covered, and warns once if a forwarded final regresses
  more than 5 s behind the watermark (indicates a child reset its audio clock).
- **Pending detector finals buffer** (max 16): while the primary owns, detector finals
  not yet covered by the watermark are buffered for replay (flip 1) and rescue.
- **Coverage dedupe (`_dedupe_by_coverage`)** — coverage may drop a buffered detector
  final **only** if it is untagged, in the current language, or not credible (length
  ≤ `min_transcript_length` or confidence below `min_detector_confidence`). A credible
  final in another language is never dropped on coverage alone, because the primary's
  "coverage" of foreign speech is garble. Used at buffer entry, at prune-on-forward and in
  the rescue watchdog. (Incidents: UAT calls 6ab3a2ef and 6ab3b5f6 — "¿Habla español?"
  transcribed by the en primary as "Good afternoon.", which previously deleted the
  Spanish final.)
- **Detector rescue watchdog** (every 0.25 s, only while the primary owns and no switch
  is in flight): a buffered detector final whose `end_time` is at least
  `detector_rescue_s` of audio behind the clock is forwarded if the primary never covered
  it. A final whose `start_time` begins more than 0.25 s before the watermark **and**
  that coverage dedupe allows dropping (same language, untagged or not credible) is
  treated as a duplicate (two engines disagreeing on end timestamps —
  incident "Hi, Joy." committed twice) and dropped. Rescued finals are logged at info
  ("rescued detector final the primary never finalized") and re-scored with
  `rescued_final_boost` — a primary deaf to a whole utterance is itself strong switch
  evidence, and this is what lets a single clear sentence cross an elevated re-entry bar.
- **Flip gate on `end_time`** (not `start_time`) while the detector owns: a final that
  overlaps a committed garbled prefix but extends past it is forwarded — duplicated
  garble is recoverable, lost speech is not.
- **Clock anchoring**: every child is created with `start_time_offset` = adapter offset +
  current audio clock (`_child_anchor`), so late-created children (recreate shadows,
  detector restarts) report on the session clock; setting `start_time_offset` on the
  adapter stream propagates to children with their anchors. Together with patch 12 this
  keeps the watermark valid across switches.

### Failure handling

- Primary failure → raised; the base `RecognizeStream` retry policy decides whether to
  rebuild the whole adapter stream (a retry rebuilds at the adapter's **current**
  language; evidence persists across retries because the engine lives on the stream
  object). Primary ending before input ended → `APIConnectionError`.
- Detector failure or unexpected end → recoverable error emitted, language detection
  paused, restart with backoff if `detector_restart` (backoff resets when the detector
  produces transcripts again).
- The adapter passes connection options with `max_retry=0` to children by default;
  retries are owned by the children / base class.

## Files

New (fork-only) package `livekit-agents/livekit/agents/stt/multilingual/`:

- `__init__.py` — package exports.
- `adapter.py` — `MultilingualAdapter`, `MultilingualRecognizeStream`, internal messages
  (`_ChildEvent`, `_ChildEnded`, `_ChildFailed`, `_SwitchResolved`), `_SwitchContext`,
  tuning constants (buffer size, poll intervals, coverage slack, clock-regression
  threshold, detector restart backoff).
- `config.py` — `LanguageSwitchOptions`, `LanguageNotAllowedError`,
  `LanguageSwitchFailedError`.
- `events.py` — event dataclasses and the `SwitchInitiator` / `SwitchExecutorKind`
  literals.
- `executors.py` — `_supports_language_update` (signature probe for a `language`
  parameter on `update_options`), `_InPlaceExecutor`, `_RecreateExecutor`.
- `heuristics.py` — scoring docs, script tables, `SwitchDecision`, `SwitchSuppressed`,
  `EventTrace`, `_HeuristicEngine`.

Modified upstream file:

- `livekit-agents/livekit/agents/stt/__init__.py` — imports and `__all__` entries for the
  adapter, options, errors and events.

Also: `examples/voice_agents/multilingual_switching.py` (Deepgram nova-3 en primary +
multi detector, allowlist en/hi, `set_language` function tool calling
`switch_language`, event logging), and the tests listed in the header.

## Re-applying the patch

The package is fork-only and merges cleanly; the work after a sync is checking the
upstream contracts below and re-adding the exports in `stt/__init__.py`. Patches 11 and
12 must be re-applied first.

## Upstream contracts relied upon (check after every sync)

- `stt.STT`: constructor with `capabilities=STTCapabilities(...)` (fields `streaming`,
  `interim_results`, `diarization`, `aligned_transcript`, `offline_recognize`,
  `keyterms`, `chat_context`), `recognize` / `_recognize_impl` signatures,
  `stream(language=, conn_options=)`, the hooks `_update_session_keyterms` /
  `_push_conversation_item` / `prewarm`, event emitter (`on`, `off`, `emit`) and the
  `metrics_collected` event.
- `stt.RecognizeStream` internals: `__init__(stt=, conn_options=, sample_rate=)`,
  `_run`, `_input_ch`, `_FlushSentinel`, `_event_ch`, `_start_time_offset` and the
  `start_time_offset` property (overridden), the 1.5+ `start_time` wall-clock anchor,
  `_emit_error(exc, recoverable=)`, `_metrics_monitor_task`, `_conn_options`,
  `push_frame`, `flush`, `end_input`, `aclose`, and the base class retry behaviour on
  `APIError` (including the wall-clock `_start_time_offset` bump in `_main_task`).
- `SpeechEvent` / `SpeechData` fields (`start_time`, `end_time`, `confidence`,
  `language`, `words`, `text`) and `SpeechEventType` members including
  `PREFLIGHT_TRANSCRIPT` and `RECOGNITION_USAGE`.
- `LanguageCode` (`.language` base code, construction from strings).
- `voice/audio_recognition.py` behaviour: its FINAL handler returns early on empty text
  **before** resetting the interim transcript — this is why the clear events include an
  empty INTERIM. If upstream changes that handler, re-check `_emit_clear_events`.
- Deepgram `SpeechStream.update_options(language=...)` performing an in-run reconnect
  (in-place executor) and patch 12's offset compensation.
- `utils.aio.Chan`, `aio.ChanClosed`, `aio.cancel_and_wait`.

## Conflict guidance

- Only `stt/__init__.py` can conflict textually — keep both export lists.
- Silent-breakage risks are the contracts above. Run the multilingual tests after every
  sync; they exercise the real `RecognizeStream` base class through `ScriptedSTT`.
- If upstream adds its own language-switching or multi-connection STT adapter, escalate
  (drop decision).

## Verification after sync

- `tests/test_stt_multilingual.py`, `tests/test_multilingual_heuristics.py`,
  `tests/test_multilingual_deepgram_mapping.py` pass (all `pytest.mark.unit`, so
  `make unit-tests` runs them).
- Type check passes for the `stt/multilingual/` package.
- The example still imports (`stt.MultilingualAdapter`, `stt.LanguageSwitchOptions`,
  event classes, `stt.LanguageNotAllowedError`, `stt.LanguageSwitchFailedError`).

## Known caveats

- Two live STT connections per call (roughly double STT cost).
- Deepgram v2 (Flux, `stt_v2.py`) has no `update_options(language=)`: with it as primary
  only the factory (`recreate`) executor works.
- Irrelevant for realtime-model sessions (no STT in the reply path).
- `_HeuristicEngine.on_primary_event` is a reserved no-op (hook for future
  text-mismatch signals).
- Script table is coarse (Latin-script languages cannot be distinguished by script;
  they rely on tags, composition and turn counts).
- The worked example in the heuristics docstring uses call-specific settings (threshold
  1.0, re-entry 1.5), not the defaults.

## Drop criteria

Upstream (or a provider) offers equivalent silent switching with no-loss transition,
detector rescue and re-entry hysteresis, validated on the fork's UAT scenarios.
