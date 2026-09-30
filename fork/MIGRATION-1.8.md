> **Numbering note.** This document uses the interim `P1`–`P12` patch numbers from the
> migration commits. The fork patch documents under `fork/patches/` use a different
> scheme; the mapping is in `FORK.md` section 1 (P1/P2 → 08, P3 → 07, P4 → 01, P5 → 02,
> P6 → 03, P7/P8 → 06, P9 → 04 and 05, P10 → 11/12/13, P11 → 09, P12 → 10).

# Moving the fork from 1.4.6 to upstream 1.8.x

Historical record of the 1.4.6 → 1.8.3 move. The per-patch documentation now lives in `fork/patches/` (see `FORK.md`); `P1`–`P13` below is the interim numbering explained in the note above. Sections 1–3 are context, section 4 records the
settled decisions, section 5 is the per-patch port plan (how / why / where), section 6 is the one
open question (dynamic endpointing vs. interruption backoff), section 7 the checklist.

## 1. State

| | |
|---|---|
| Fork base | `29b71d45d` (tag `livekit-agents@1.4.6`, 2026-03-16) |
| Target | upstream `main` `15b4bc84c` (2026-09-30) = tag `livekit-agents@1.8.3` (2026-09-26) |
| Upstream commits in between | 1108 |
| `main` sync | **Done 2026-09-30**: `main` pushed directly to upstream `15b4bc84c`, an exact mirror (PR #64 auto-marked merged, sync branch deleted). Future syncs: `git fetch upstream && git push origin upstream/main:main`. |

Growth of the three files that carry most of our hooks (lines, 1.4.6 → 1.8.3):

| File | 1.4.6 | 1.8.3 |
|---|---|---|
| `voice/agent_activity.py` | 2908 | 5129 |
| `voice/agent_session.py` | 1484 | 2407 |
| `voice/audio_recognition.py` | 755 | 2150 |

None of our new files collide with upstream paths.

## 2. Upstream overview, 1.5.0 → 1.8.3

**1.5.0 (turn-handling rewrite).** `AgentSession(turn_handling=TurnHandlingOptions(...))`
consolidates `turn_detection`, `endpointing` (`EndpointingOptions`: mode fixed/dynamic, min_delay,
max_delay, alpha), `interruption` (`InterruptionOptions`: enabled, mode adaptive/vad, min_duration,
min_words, resume_false_interruption, false_interruption_timeout, backchannel_boundary),
`preemptive_generation`, later `user_turn_limit`. Old kwargs are deprecated shims
(`_migrate_turn_handling` in `voice/turn.py`); `Agent(...)` accepts the same dict for overrides.
Adaptive interruption (audio ML model on LiveKit Inference, transcript gate, pause-and-resume of
falsely interrupted playout). Dynamic endpointing (`voice/endpointing.py`: `BaseEndpointing`,
`DynamicEndpointing`, `create_endpointing`); `AudioRecognition` holds `self._endpointing` with
`min_delay`/`max_delay` instead of two floats. Preemptive generation on by default.
`metrics_collected` deprecated for `session_usage_updated` + `ChatMessage.metrics`.

**1.5.x – 1.6.x.** Turn detector v1.0 (streaming `_StreamingTurnDetector` protocol: the EOU
probability may come from a prediction future), `PreemptiveGenerationOptions`, keyterms /
`stt_context_options`, barge-in cooldown for corrections (#5269), realtime fallback adapter,
`TimedString.speaker_id`, async tools + `ctx.with_filler()`, answering-machine detection, user
transcription timeout, `user_turn_limit`, `console`/`dev` deprecated for `lk agent`, Deepgram
reconnect fixes, `_update_last_language` ignores `multi` (#6531).

**1.7.x.** PII redaction (`lk.pii.*` log/trace keys), expressive mode (TTS emotion markup in LLM
output), adaptive-interruption hardening.

**1.8.x.** OTel GenAI semantic conventions (breaking for dashboards), `DuplexModel` (GPT-Live),
`eou_wait`/`agent_turn` spans, event-loop blocking detection, `requires-python >= 3.10`,
`LLMMetrics.cache_creation_tokens`/`reasoning_tokens`, fallback adapters no longer `chat` spans.

## 3. Our deployment context (settled)

1. **Self-hosted**, not on LiveKit Cloud.
2. **No LiveKit Inference** today; may adopt later if there is a clear advantage. Consequences on
   1.8.3, verified in code:
   - Adaptive interruption resolves to **off**: it is disabled by default when not hosted and in
     production mode, and `AdaptiveInterruptionDetector()` raises without credentials (logged, not
     fatal). Sessions run `interruption.mode = "vad"`. The transcript gate, backchannel boundary and
     pause-and-resume machinery are inert.
   - VAD stays local (`inference.VAD(model="silero")` is `livekit-local-inference`).
   - The default `inference.TurnDetector()` is cloud; we keep the local
     `livekit-plugins-turn-detector` (`turn_detector.multilingual.MultilingualModel`), which still
     exposes `unlikely_threshold` / `predict_end_of_turn`, so our threshold composition is unchanged.
3. **No LiveKit adaptive interruption, ever on self-hosted.** We plan our own detector on open-source
   models. The upstream slot is typed to the concrete `inference.AdaptiveInterruptionDetector`;
   `AudioRecognition._interruption_task` drives its stream and consumes `OverlappingSpeechEvent`
   verdicts. A custom detector mirrors that interface and inherits gating, boundary and
   pause-and-resume for free. Tracked as future **P14**; not part of this migration.
4. **Realtime models are likely** (GPT-Live duplex, Gemini Live). With server-side turn detection the
   EOU task, VAD inference hooks and the STT-based endpointing patches (P4 endpoint path, P5, P6)
   are bypassed and STT is transcription-only. What survives: the interruption tracker (realtime
   commits interrupted items too), the playout silence gate (`_realtime_reply_task` waits on the
   same event), `max_volume`. Moot: P1/P2 (inference LLM only), P3, P10. Therefore the
   pipeline-path ports are kept **lean**: no new endpointing heuristics beyond what exists.

## 4. Decisions (settled)

| # | Question | Decision | Rationale |
|---|---|---|---|
| D1 | P2 bracket stripping vs expressive mode | Keep the current truncate-at-`[` behaviour; **gate it off when the session is expressive** | Expressive markup is bracketed; the crude rule is fine for our non-expressive agents |
| D2 | P4 word rules in adaptive mode? | **Apply in both modes** (VAD and adaptive) | Adaptive is off for us anyway; when P14 exists a loud "okay" can still be classed as an interruption |
| D3 | P9 silence gate implementation | **Separate hold event** from the tracker, added to the authorization wait; `_user_silence_event` semantics untouched | Avoids upstream's `_reconcile_playout_pause` pausing/resuming a reply that is merely held |
| D4 | LiveKit adaptive interruption | **Not used** | Cloud-only; own detector planned (P14) |
| D5 | P12 `previous_text` | **Keep the literal** in the plugin | Not worth an option surface |
| D6 | Port strategy | **Re-apply by feature** on `patched-1.8` from tag `livekit-agents@1.8.3`, not `git rebase` | The three hook files grew 2–3×; hunk conflicts everywhere |
| D7 | `SpeechData.detected_languages` | **Rename to upstream `source_languages`** | Same meaning upstream; avoids a parallel field |
| D8 | `interrupt_backoff` deprecation shim, P13 debug log | **Drop** | Fork-only shim; upstream records `interrupt(source=…)` on the span |
| D9 | P6 stale-anchor override | **Port as warning-only first**, re-enable the forced reset only if 1.8 prod logs show the case | Upstream reworked the anchor; case is narrower but not proven gone |
| D10 | P9 vs dynamic endpointing | **Independent switches**; P9 ported as flat per-mode delays | incremental rollout: backoff for all, dynamic endpointing on a few; §6 is a later comparison |

## 5. Per-patch port plan

Legend — **Clean**: hunk applies as-is. **Mechanical**: same idea, new anchor/name.

| # | Patch | Verdict | Where (1.8.3) |
|---|---|---|---|
| P1 | `parallel_tool_calls` pop | Clean | `inference/llm.py` `LLMStream._run`, after the `tool_choice` pop (~L423) |
| P2 | bracket stripping | Mechanical + D1 gate | `inference/llm.py` `_parse_choice` (~L514) |
| P3 | ParallelAdapter | Mechanical + telemetry follow-ups | `llm/parallel_adapter.py` (new), `llm/__init__.py`, `metrics/base.py` |
| P4 | backchannel / commit words | Mechanical, config relocated | `voice/turn.py` `InterruptionOptions`; `voice/audio_recognition.py`; `voice/agent_activity.py` |
| P5 | digit / alphanumeric endpointing | Mechanical | `voice/audio_recognition.py` `_bounce_eou_task` |
| P6 | stale anchor, sleep floor, `eou sleep` log | Mechanical + D9 | `voice/audio_recognition.py` |
| P7 | language length 5→3 | Clean | `voice/audio_recognition.py` constant |
| P8 | `get_last_user_language` | Clean | same three properties |
| P9 | interruption-backoff modes | Hooks mechanical (D3); endpointing parts pending §6 | `voice/interruption_tracker.py` (new), `voice/turn.py`, `voice/agent_session.py`, `voice/agent_activity.py`, `voice/audio_recognition.py` |
| P10 | MultilingualAdapter + Deepgram | Additive + D7 | `stt/multilingual/` (new), `stt/stt.py`, `types.py`, Deepgram `stt.py` |
| P11 | `max_volume` | Mechanical | `voice/room_io/_output.py`, `room_io/types.py`, `room_io/room_io.py` |
| P12 | ElevenLabs `previous_text` | Mechanical (D5) | ElevenLabs `tts.py` request-builder helpers |
| P13 | interrupt debug log | Drop (D8) | — |
| — | `interrupt_backoff` shim | Drop (D8) | — |

### P1 — `parallel_tool_calls` pop
**Why:** Azure OpenAI rejects the flag without tools; upstream still does not handle it.
**Where/how:** `inference/llm.py`, `LLMStream._run`: after `if not self._tools: self._extra_kwargs.pop("tool_choice", None)` add the same pop for `parallel_tool_calls`.

### P2 — bracket stripping (D1)
**Ported.** `inference.LLM(strip_brackets=True)` / `update_options(strip_brackets=)`; `AgentActivity`
turns it off for expressive turns (adapters unwrapped).
**Why:** citation markers must not be spoken. **Where:** `_parse_choice(self, id, choice, thinking_filter)`, right after `strip_thinking_tokens(delta.content, thinking_filter, final=…)`.
**How:** same truncation, wrapped in `if not <expressive>:`. The stream does not know the session; pass the flag down the way `thinking_filter` is passed (an `LLM(strip_brackets=True)` / `LLMStream` ctor flag that the session sets to `False` when `expressive` is on), or have the session skip it via `extra_kwargs`. Pick the ctor flag: it keeps the rule off for any expressive session and leaves a one-line opt-out.

### P3 — ParallelAdapter
**Why:** provider latency hedging. `LLM.chat()` and `LLMStream.__init__` signatures are unchanged; the module drops in.
**How (follow-ups so it behaves like upstream's `FallbackAdapter`):** forward `LLM.prewarm(loop=)` to every entry (the session calls it eagerly); set `_genai_operation_name = None` on `ParallelLLMStream` (#7373) and record the winning entry's provider/model on the span like `fallback_adapter.py` does (#7374); `parallel_selected` still slots into `LLMMetrics` (now `_BaseMetrics`). Add a unit test for the winner-id race (#58 shipped none).

### P4 — backchannel / commit words (D2)
**Ported.** Config is `turn_handling={"interruption": {"backchannel_words": {...}, "commit_words": {...}}}`;
a new `RecognitionHooks.on_commit_word` stores the commit word like a skip_reply turn; commit words
take precedence over the backchannel list on both paths (small deliberate change from 1.4.6).
**Why:** in VAD mode (our mode) a single "okay / mm-hmm" while the agent speaks must neither interrupt nor commit a turn.
**Where/how:**
- Config: add `backchannel_words: set[str] | None` and `commit_words: set[str] | None` keys to `InterruptionOptions` in `voice/turn.py` with `None` defaults in `_INTERRUPTION_DEFAULTS`; `AudioRecognition.__init__` reads them from `session.options.interruption` (it already reads `backchannel_boundary` there). Drop the `AgentSession` kwargs (all siblings are deprecated).
- `is_bot_speaking()` hook: not needed; use `AudioRecognition._agent_speaking` (maintained by `_on_start_of_agent_speech` / `_on_end_of_agent_speech`) in `_run_eou_detection`; in `AgentActivity._interrupt_by_audio_activity` keep `self._session.agent_state == "speaking"`.
- `_interrupt_by_audio_activity`: insert the single-backchannel-word early return before the `interruption_options["min_words"]` check; the transcript attribute is now `self._audio_recognition._current_transcript`.
- Commit-word path: replace `retrieve_chat_ctx().items.append(...)` with the session's `_conversation_item_added(ChatMessage(role="user", …))` so the global context, the `conversation_item_added` event and P9's user-turn counter all see it.
- Apply in both interruption modes (no `mode == "vad"` guard).

### P5 — digit / alphanumeric endpointing
**Ported**, on by default; switchable with `turn_handling={"endpointing": {"readout_rules": False}}`
(the upstream `test_user_turn_exceeded` module turns it off because its fixtures are number words).
**Why:** callers reading numbers pause between groups. **Where:** `_bounce_eou_task`, before `if turn_detector is not None:`.
**How:** helpers unchanged; the ladder sets `endpointing_delay` / `delay_reason` using `self._endpointing.max_delay` (a fixed ceiling even in dynamic mode); also set `ATTR_EOU_DELAY` on `eou_wait_span` after the override so traces match the log.

### P6 — stale anchor, sleep computation, floor, `eou sleep` log (D9)
**Ported** as `_check_stale_speaking_anchor()` (warning only), the INFO line with `trigger`,
`from_cache`, `end_of_turn_probability`, `unlikely_threshold`, and the raw-delay fallback + floor
as **opt-in endpointing keys** (`turn_handling={"endpointing": {"sleep_floor": 0.5,
"stale_anchor_raw_delay": True}}`), default off. Reason: on by default they shift every commit in
which the final arrives after `min_delay` by up to 0.5 s, which broke 25 upstream timing tests
(and adds that latency on every late transcript in prod). **The prod config must set both keys to
keep 1.4.6 behaviour**; this is also the knob to compare against dynamic endpointing in §6.
**Why:** the `INFERENCE_DONE`-missed case produced instant commits; the INFO line feeds the dashboard.
**How:** `_second_last_final_transcript_time` / `_ignore_last_speaking_time` ported with the reset replaced by `logger.warning("stale last_speaking_time detected", …)`; `compute_sleep()` negative-anchor fallback and the 0.5 s floor ported as-is around `extra_sleep = endpointing_delay; if last_speaking_time: …` (unchanged upstream); `logger.info("eou sleep")` kept and extended with upstream's `trigger`, `from_cache`, `end_of_turn_probability`, `unlikely_threshold`.

### P7 — `MIN_LANGUAGE_DETECTION_LENGTH`
Constant still `= 5`, used by `_update_last_language` (which now also ignores `multi`). Set to 3.

### P8 — `get_last_user_language`
No upstream equivalent; `AudioRecognition._last_language` still exists. Same three properties.

### P9 — interruption-backoff modes (D3)
**Ported.** `turn_handling={"interruption_backoff": InterruptionBackoffOptions(...)}` or the
`AgentSession(interruption_backoff=)` alias; the hold is a separate event added to the
authorization waits; the mode logic runs after the endpointing ladder and reads only
`min_delay`/`max_delay` from the active endpointing object, so backoff and dynamic endpointing are
independent switches (rollout: backoff for all customers, dynamic endpointing for a few). §6 stays
open as a later comparison, not a blocker.
**Where the hooks go (all anchors present, same shape):**
- `AgentSession._conversation_item_added` (~L2295): tracker hook, using `message.text_content`.
- Options: `interruption_backoff: InterruptionBackoffOptions | None` as a key of `TurnHandlingOptions`; dataclasses stay in `voice/interruption_tracker.py`; keep `AgentSession(interruption_backoff=)` as a thin alias during transition; drop the `interrupt_backoff` shim.
- Preemptive: `on_preemptive_generation` gates on `preemptive_opts["enabled"]` → add `or tracker.preemptive_disabled()`; `_cancel_preemptive_generation` exists. Preemptive is on by default upstream, so the prod config sets `turn_handling["preemptive_generation"]["enabled"]` explicitly.
- Silence gate (D3): the tracker owns a second `asyncio.Event` ("hold"); `on_end_of_speech` / `on_vad_inference_done` clear/set it from `ev.silence_duration` / `ev.raw_accumulated_silence` against `silence_gate()`; it is appended to `authorization_tasks` in `_tts_task_impl` (~L3153), `_pipeline_reply_task_impl` (~L3642) and `_realtime_reply_task` next to `_user_silence_event.wait()`. `_user_silence_event` is left exactly as upstream, so `_reconcile_playout_pause` is unaffected.
- Hold/latency logs: `_gate_closed_timestamp` / `_log_silence_gate_hold` wrap the same blocks; upstream's `_record_queue_wait(speech_handle)` sits right after, add the hold as a span attribute there too. `_log_reply_latency` moves next to the early-metrics block (`first_tts_gen_data` instead of `tts_gen_data`); upstream also exposes `e2e_latency`, `end_of_turn_delay`, `transcription_delay`, `llm_node_ttft`, `tts_node_ttfb` on `ChatMessage.metrics`, our INFO line stays for grep-ability.
- Counting caveat: with adaptive off (our case) every barge-in that interrupts sets `speech_handle.interrupted`, so counts match 1.4.6 behaviour. If P14 ever resumes false interruptions, counts drop and thresholds need re-tuning.
- `_bounce_eou_task` mode logic: mechanical re-weave (`end_of_turn_probability is None` replaces `predict_ok`; cap is `self._endpointing.max_delay`), **but whether transient/sustained/primed endpointing parts stay as flat delays or fold into dynamic endpointing is open (§6)**. The gate and preemptive control are independent of that outcome.
- Tests: `create_session(actions, speed_factor=, turn_handling=, extra_kwargs=)`, `run_session`, `FakeActions`, `FakeVAD`, `SESSION_TIMEOUT` exist; port the five tests with `turn_handling={"interruption_backoff": …}`; fake endpointing defaults are `min_delay=0.5/speed, max_delay=6.0/speed`.

### P10 — MultilingualAdapter, `SpeechData`, `TimedString`, Deepgram (D7)
**Why:** unchanged; not relevant for a realtime pilot but required for the STT pipeline.
**How:**
- `STTCapabilities` gained `offline_recognize`, `keyterms`, `chat_context`: build the adapter's capabilities by copying the primary's.
- Forward the new STT hooks as `MultiSpeakerAdapter` does: `_update_session_keyterms`, `_push_conversation_item`, `prewarm`, `model`/`provider`, and `RecognizeStream.context` from the owning child.
- `RecognizeStream` now has a `start_time` property and `_main_task` adds the wall-clock gap between runs to `_start_time_offset` on retry; our `start_time_offset` override (adapter ~L366) and child anchoring must still compose; forward `start_time` too.
- `detected_languages` → `source_languages` (D7) in the Deepgram mapper and tests; nothing in `stt/multilingual/` reads it.
- `TimedString`: upstream inserted `speaker_id` as the 5th positional; add `language` after it and pass by keyword.
- Deepgram `stt.py`: `live_transcription_to_speech_data` now has `use_punctuated_word` and `speaker_id`; re-apply per-word `language=`, `alt.get("languages")`, `source_languages=`. `_run` still resets timestamps on the `update_options()` reconnect, so `conn_audio_sent` → `_start_time_offset` is still needed (upstream's wall-clock offset only covers retries; keep our counter in `send_task`).
- Deepgram Flux (`stt_v2.py`) has no `update_options(language=)`; the factory path is required there.
- Example stays under `examples/voice_agents/` (examples are uv workspace members now).

### P11 — `max_volume`
Constructor, `room_io.py` call and the two option dataclasses are the same. `_forward_audio` was rewritten but `frame = self._scale_volume(frame)` goes immediately before `await self._audio_source.capture_frame(frame)` as before.

### P12 — ElevenLabs `previous_text` (D5)
Add the literal to `_build_synthesize_body` (HTTP, ~L1230) and `_build_context_init_packet` (WS, ~L606). Not in `_build_dialogue_context_init_packet` (text-to-dialogue for `eleven_v3*` has no such field).

## 6. Open: dynamic endpointing vs. interruption backoff

Needs more discussion; we may move some backoff features into dynamic endpointing. Facts to reason from:

| | Dynamic endpointing (upstream) | Interruption backoff (P9) |
|---|---|---|
| Signal | pause timing: gaps between utterances inside a turn, and "immediate interruption" (user resumes within `min_delay` of agent speech start) | count of committed interrupted replies, any timing |
| Lever | raises the learned `min_delay` (EMA, bounded `[min_delay, max_delay]`), i.e. the wait for **every** turn incl. EOU-confident ones; `max_delay` never moves | raises only the EOU-unlikely branch (delay + threshold); confident turns stay fast; `primed_max_endpointing` raises the cap |
| Dynamics | continuous, decays when shorter pauses are seen | discrete modes with hysteresis (transient exits after N clean turns; primed/sustained sticky) |
| Extras | none | playout silence gate, preemptive control |
| Self-hosted note | with adaptive off, `on_end_of_speech(interruption=NOT_GIVEN)` treats every overlap as an interruption (fine for learning) | unchanged |

SL-3890 findings it must satisfy: the 0.03–0.1 EOU band (unlikely-ish turns committing at the fast delay) and the 1.5 s cap being shorter than mid-thought pauses. Dynamic endpointing addresses the first only by raising the floor for all turns and cannot address the cap.

Evaluation plan (on 1.8, same dashboard: interrupted-reply rate by EOU band, per-turn commit delta):
1. dynamic endpointing alone; 2. primed mode alone; 3. both. Then decide whether transient/sustained
delays become a multiplier over the learned `min_delay`, or a `BaseEndpointing` decorator owned by
the tracker (the endpointing object already receives start/end of user and agent speech, the same
signals we use; only the EOU-threshold composition would stay in `_bounce_eou_task`).

## 7. Checklist

- [x] docs: `FORK.md` + `fork/patches/` updated for the 1.8 line (from short-loop/agents#65)
- [x] `main` synced to upstream (`15b4bc84c`) — pushed directly, exact mirror
- [x] `patched-1.8` from `livekit-agents@1.8.3` (2026-09-30)
- [x] Additive: P3 (`0e1e1241c`), P10 (`d751a2ac1`)
- [x] Hooks: P1 (`ef466656d`), P2 + D1 gate (`f778357b0`), P7+P8 (`a3e52f52f`), P11 (`01a880b0b`), P12 (`3c69f69e2`)
- [x] Recognition path: P5+P6 with D9 warning-only (`dd24f08b2`), P4 with D2 (`1ade102b0`)
- [x] P9: tracker module, D3 hold event, `turn_handling["interruption_backoff"]` + kwarg alias, flat per-mode delays layered on the active endpointing object (kept independent of dynamic endpointing for incremental rollout)
- [ ] App: python 3.10, `turn_handling=…` incl. `endpointing.sleep_floor=0.5`, `endpointing.stale_anchor_raw_delay=True`, `interruption.backchannel_words/commit_words`; explicit preemptive setting; `lk agent`; `interruption.mode` left to auto (resolves to vad)
- [ ] Observability: OTel attribute renames, `lk.pii.*` keys, `eou sleep` extra fields, dashboard JSON
- [ ] Tag `1.8.3-shortloop.1`, smoke calls (multilingual UAT set, interruption-heavy call)
- [ ] Later: P14 own interruption detector; realtime-model pilot
