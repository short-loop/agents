# FORK.md — short-loop/agents

This repository is a **patched fork** of [livekit/agents](https://github.com/livekit/agents).
It tracks upstream closely and carries a small, well-defined set of behavioural patches
used by ShortLoop's production voice agents.

This file is the entry point for anyone (human or automated agent) who needs to:

- understand what the fork changes and why,
- sync the fork with upstream without losing or breaking a patch,
- decide whether a patch can be dropped because upstream now covers it.

Per-patch documentation lives in [`fork/patches/`](fork/patches/) (layout and authoring
rules in [`fork/README.md`](fork/README.md)). Each patch file is self-contained: purpose,
behaviour, every file and symbol it touches, how to re-apply it if upstream moved the
surrounding code, conflict hot-spots, tests, and drop criteria.
[`fork/MIGRATION-1.8.md`](fork/MIGRATION-1.8.md) is the record of the one-off move from
upstream 1.4.6 to 1.8.3 (context, decisions D1–D10, per-patch port notes, and the open
dynamic-endpointing comparison).

> These documents contain **no code**. They reference files, classes, functions and
> option names, and describe behaviour in natural language. The source of truth for the
> implementation is always the code on the fork working branch.

---

## 1. Fork facts

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/livekit/agents` (default branch `main`) |
| Fork repository | `https://github.com/short-loop/agents` |
| Fork working branch | `patched-1.8` — the 1.8.x line, one fork commit per patch on top of the base below. `patched` is the frozen 1.4.6 line (base `29b71d4`), kept while production releases still ship from it. `main` mirrors upstream `main` exactly. |
| Current upstream base | commit `76de1759b` — "livekit-agents@1.8.5 (#7642)" (2026-10-07), tag `livekit-agents@1.8.5`, merged as `7a04a2297`. Previous base: `1983c39b` (1.8.3). |
| Package version on the base | `livekit-agents` 1.8.5 (plugins at their matching 1.8.5 versions; runtime pin `livekit==1.1.20`); `requires-python >= 3.10` |
| Fork delta vs. base | 35 files, ~5.1k lines added, ~10 lines removed (≈ 60% is tests and the new multilingual STT package) |
| Fork PR numbering | Fork PRs are `short-loop/agents#NN` (two-digit). Upstream PRs are `livekit/agents#NNNN` (four-digit). Commit subjects keep upstream's `(#NNNN)` suffix for upstream commits and `(#NN)` for fork commits. |
| Ticket references | `SL-3890` = interruption-backoff work (patches 04, 05). |
| Code markers | Fork code inside upstream files carries a `fork(patch NN)` comment where `NN` is the patch number below. `grep -rn "fork(patch"` lists every fork touch point. |

### How the fork is structured in git

- Upstream commits are kept verbatim (no rewriting). The last upstream commit in the
  `patched-1.8` history is the base listed above.
- The 1.4.6 patches were **re-applied by feature** onto the 1.8.3 tag (not rebased or
  merged), one commit per patch, because upstream rewrote the turn-handling files between
  1.4.6 and 1.8.0. From 1.8.3 onward, syncs are ordinary merges (section 4).
- Fork commits on `patched-1.8`, oldest first (the commit messages use the interim
  `P1`–`P12` numbering of the migration; the mapping to patch documents is the last column):

| Fork commit | Subject | Patch docs |
|---|---|---|
| `0e1e1241c` | feat(llm): ParallelAdapter for latency hedging (P3) | 07 |
| `d751a2ac1` | feat(stt): MultilingualAdapter silent language switching (P10) | 11, 12, 13 |
| `ef466656d` | fix(inference): drop parallel_tool_calls when no tools are attached (P1) | 08 |
| `f778357b0` | feat(inference): strip bracket artifacts from streamed text, off in expressive mode (P2) | 08 |
| `a3e52f52f` | feat(voice): get_last_user_language accessor and shorter language detection (P7, P8) | 06 |
| `01a880b0b` | feat(room_io): max_volume attenuation on the participant audio output (P11) | 09 |
| `3c69f69e2` | feat(elevenlabs): previous_text prosody steering on synthesize requests (P12) | 10 |
| `dd24f08b2` | feat(voice): digit endpointing rules, sleep floor and eou sleep log (P5, P6) | 02, 03 |
| `1ade102b0` | feat(voice): backchannel and commit words while the agent speaks (P4) | 01 |
| `376aae4b4` | fix(inference): read strip_brackets once per request (P2, mypy) | 08 |
| `1ede41bd3` | feat(voice): make the P5/P6 endpointing behaviours EndpointingOptions keys | 02, 03 |
| `19dabfdac` | feat(voice): interruption-backoff modes (P9, SL-3890) | 04, 05 |
| `7f8e31138` | fix(voice): make the interruption-backoff activity hooks mock-safe (P9) | 04 |
| `c4acb76b5`, `e6342f30b` | chore: number fork markers after fork/patches docs; docs(fork) for the 1.8.3 line | — |
| `64e70183b` | feat(elevenlabs): support eleven_v4 and eleven_v4_turbo — cherry-pick of upstream `7d3a90714`, superseded by the 1.8.5 merge | — |
| `7a04a2297` | Merge tag `livekit-agents@1.8.5` (sync 2026-10-07: no conflicts, upstream touched only telemetry hunks in the hot-spot files) | — |

The quickest way to see the complete fork delta is a diff between the upstream base
commit and the tip of `patched-1.8`. After a sync, the new upstream base commit replaces
`76de1759b` in the table above.

---

## 2. Patch index

Status legend: **Opt-in** = shipped but inactive unless configured. **Always-on** =
changes default behaviour for every user of the fork (a switch may exist to turn it off).
**Dropped** = removed from the code, document kept for history. Every patch that is not
Dropped is active and must be preserved through syncs.

| # | Patch | Area | Status | Upstream-touching files | Doc |
|---|---|---|---|---|---|
| 01 | Backchannel & commit words | voice / turn-taking | Always-on (built-in list); configurable via `InterruptionOptions` | `voice/turn.py`, `voice/audio_recognition.py`, `voice/agent_activity.py` | [01](fork/patches/01-backchannel-and-commit-words.md) |
| 02 | Endpointing delays for numbers / alphanumerics | voice / EOU | Always-on; switch `EndpointingOptions.readout_rules` | `voice/turn.py`, `voice/endpointing.py`, `voice/audio_recognition.py` | [02](fork/patches/02-endpointing-pattern-delays.md) |
| 03 | EOU sleep timing: stale-anchor warning, sleep floor, raw-delay fallback, `eou sleep` log | voice / EOU | Log + warning always-on; floor and fallback **opt-in** (`EndpointingOptions.sleep_floor`, `stale_anchor_raw_delay`) | `voice/turn.py`, `voice/endpointing.py`, `voice/audio_recognition.py` | [03](fork/patches/03-eou-sleep-timing.md) |
| 04 | Interruption-backoff modes (normal / primed / transient / sustained) | voice / turn-taking | Opt-in (`turn_handling["interruption_backoff"]` or `AgentSession(interruption_backoff=)`) | `voice/interruption_tracker.py` (new), `voice/turn.py`, `voice/agent_session.py`, `voice/agent_activity.py`, `voice/audio_recognition.py`, `voice/__init__.py`, `agents/__init__.py` | [04](fork/patches/04-interruption-backoff-modes.md) |
| 05 | Voice observability logs (reply latency, playout hold) | voice / logging | Always-on (logs only) | `voice/agent_activity.py` | [05](fork/patches/05-voice-observability-logs.md) |
| 06 | Last user language accessor + shorter language-detection minimum | voice | Always-on | `voice/audio_recognition.py`, `voice/agent_activity.py`, `voice/agent_session.py` | [06](fork/patches/06-last-user-language.md) |
| 07 | `llm.ParallelAdapter` (LLM hedging / racing) | llm | Opt-in (new class) | `llm/parallel_adapter.py` (new), `llm/__init__.py`, `metrics/base.py` | [07](fork/patches/07-llm-parallel-adapter.md) |
| 08 | LiveKit Inference LLM stream tweaks (drop `parallel_tool_calls` w/o tools, strip `[` artifacts) | inference | Always-on for `inference.LLM`; bracket stripping switchable (`strip_brackets`) and off in expressive mode | `inference/llm.py`, `voice/agent_activity.py` | [08](fork/patches/08-inference-llm-stream-tweaks.md) |
| 09 | Room audio output `max_volume` | voice / room_io | Opt-in (default 1.0 = no-op) | `voice/room_io/_output.py`, `voice/room_io/room_io.py`, `voice/room_io/types.py` | [09](fork/patches/09-room-audio-max-volume.md) |
| 10 | ElevenLabs `previous_text` priming | plugin: elevenlabs | Always-on | `livekit-plugins-elevenlabs/.../tts.py` | [10](fork/patches/10-elevenlabs-previous-text.md) |
| 11 | STT language metadata (`TimedString.language`, Deepgram `source_languages` + per-word tags) | stt / types / deepgram | Always-on (additive) | `types.py`, `livekit-plugins-deepgram/.../stt.py` | [11](fork/patches/11-stt-language-metadata.md) |
| 12 | Deepgram timestamp continuity across in-run reconnects | plugin: deepgram | Always-on | `livekit-plugins-deepgram/.../stt.py` | [12](fork/patches/12-deepgram-reconnect-timestamps.md) |
| 13 | `stt.MultilingualAdapter` (silent language switching) | stt | Opt-in (new class) | `stt/multilingual/*` (new), `stt/__init__.py`, example + tests | [13](fork/patches/13-multilingual-stt-adapter.md) |

All `livekit-agents` paths above are relative to `livekit-agents/livekit/agents/`.
Plugin paths are relative to `livekit-plugins/`.

Dropped during the 1.8 move (see `fork/MIGRATION-1.8.md`, D8): the debug line
"Speech handle interrupted, cancelling tasks" (upstream records the interruption source on
the `agent_turn` span) and the accepted-and-ignored `AgentSession(interrupt_backoff=)` kwarg.

### Patch dependencies

- **13 depends on 11 and 12.** The multilingual heuristics read per-word language tags
  (11) and the adapter's dedup watermark requires continuous Deepgram timestamps across
  `update_options()` reconnects (12).
- **02, 03 and 04 live in one function** — the nested end-of-utterance task inside
  `AudioRecognition._run_eou_detection`. 02 adds the two pattern branches in front of the
  turn-detector branch and sets `delay_reason` / `use_raw_delay`; 04 is a block *after*
  the whole delay-selection ladder; 03 owns the sleep computation and the `eou sleep`
  log below both. Re-apply them together, in that order.
- **01 extends the `RecognitionHooks` protocol** (`on_commit_word`). `AgentActivity` is
  the only implementer. 04 does not extend the protocol: `AudioRecognition` reads the
  session's tracker directly (type-checked, so mocked sessions degrade to upstream).
- **05 depends on 04** (its log lines include `interruption_mode`; the hold log measures
  04's hold event).
- **08 is gated by expressive mode** from `AgentActivity` (the activity turns bracket
  stripping off for expressive turns).
- **06 is used by application code** that picks the TTS voice / prompt language; it has
  no in-repo dependents besides the chain `AgentSession → AgentActivity → AudioRecognition`.

### Configuration surface (what the app sets)

All fork knobs on the session go through upstream's `turn_handling` dict:

```
turn_handling={
    "endpointing": {"sleep_floor": 0.5, "stale_anchor_raw_delay": True,   # patch 03 (opt-in)
                    "readout_rules": True},                               # patch 02 (default)
    "interruption": {"backchannel_words": None, "commit_words": {...}},    # patch 01
    "interruption_backoff": InterruptionBackoffOptions(...),               # patch 04
}
```

`AgentSession(interruption_backoff=...)` remains as an alias for the 04 key. Patch 08's
`inference.LLM(strip_brackets=True)`, patch 09's `RoomOutputOptions.max_volume` and
patch 13's `MultilingualAdapter` are configured on their own objects.

---

## 3. Conflict hot-spot map

Use this to predict which patches an upstream change will collide with. Files are listed
from highest to lowest conflict risk.

| File | Patches | Risk | Notes |
|---|---|---|---|
| `voice/audio_recognition.py` | 01, 02, 03, 04, 06 | **Very high** | Upstream edits `_process_stt_event`, `_on_vad_event`, `_run_eou_detection` frequently (EOU / turn-detection work). The fork adds two pattern branches, a post-ladder backoff block and its own sleep computation inside the nested EOU task, plus the crutch-word block at the top of `_run_eou_detection`. Read patches 02/03/04 together before resolving. |
| `voice/agent_activity.py` | 01, 04, 05, 06, 08 | **High** | Touches `_interrupt_by_audio_activity`, `on_start_of_speech`, `on_end_of_speech`, `on_vad_inference_done`, `on_preemptive_generation`, the four playout authorization blocks (`_tts_task_impl`, `_pipeline_reply_task_impl`, `_realtime_reply_task`, `_realtime_generation_task`), the early-metrics callback, and the expressive-options resolution in `_pipeline_reply_task_impl`. Upstream blocks are kept byte-identical where possible; fork lines carry `fork(patch NN)`. |
| `voice/turn.py` | 01, 02, 03, 04 | Medium | Fork keys on the `EndpointingOptions` / `InterruptionOptions` / `TurnHandlingOptions` TypedDicts and their `_*_DEFAULTS`. Upstream adds keys here regularly — conflicts are mechanical (keep both). |
| `voice/endpointing.py` | 02, 03 | Low–medium | Three attributes on `BaseEndpointing` set by `create_endpointing`. If upstream changes how endpointing objects are built, keep the attributes populated from the options. |
| `voice/agent_session.py` | 04, 06 | Medium | Tracker instance, the `interruption_backoff` kwarg alias and option resolution, and logic in `_conversation_item_added`. |
| `livekit-plugins-deepgram/.../stt.py` | 11, 12 | Medium | `SpeechStream._run` (reconnect loop) and `live_transcription_to_speech_data`. |
| `inference/llm.py` | 08 | Low–medium | `_LLMOptions.strip_brackets`, constructor/`update_options` parameter, and two small insertions in `LLMStream`. |
| `voice/room_io/*` | 09 | Low | Additive parameter plumbing. |
| `livekit-plugins-elevenlabs/.../tts.py` | 10 | Low | One key added to two request-builder helpers. |
| `types.py`, `metrics/base.py` | 11, 07 | Low | Additive attribute / field. |
| `__init__.py` files (`agents`, `voice`, `llm`, `stt`) | 04, 07, 13 | Low | Export lists; keep both sides. |
| New fork-only files (`voice/interruption_tracker.py`, `llm/parallel_adapter.py`, `stt/multilingual/*`, fork tests, example) | 04, 07, 13 | Never conflict | But they **depend on upstream internals** — see each patch's "Upstream contracts" section. A clean merge can still break them. |

---

## 4. Upstream sync playbook

This is the procedure the daily sync workflow should follow. It is written for an
automated agent but works the same for a human.

### 4.1 Preparation

1. Add upstream as a remote (the fork checkout only has `origin`) and fetch upstream
   `main` plus tags.
2. Determine the current base: the most recent upstream commit reachable from
   `patched-1.8` (today `76de1759b`). Record the list of new upstream commits since that
   base.
3. Create a working branch from `patched-1.8` (never sync directly on it).
4. Skim the new upstream commits and flag any that touch files in the hot-spot map
   (section 3). For each flagged commit, open the corresponding patch docs **before**
   merging.

### 4.2 Integrate

- **Preferred: merge** upstream `main` (or the target release tag) into the working
  branch. Merging keeps upstream commit identities intact, which the "Fork facts" table
  relies on.
- Prefer syncing to **upstream release commits** ("livekit-agents@X.Y.Z (#NNNN)") rather
  than arbitrary `main` commits when a release is available; it keeps plugin versions
  coherent. Daily syncs may track `main`, but record the exact base commit.
- A re-apply-by-feature (what the 1.8 move did) is the fallback when upstream rewrites
  the hot-spot files wholesale; it is a human decision, done from the patch documents.

### 4.3 Resolve conflicts

For every conflicted file:

1. Identify the patches involved using the hot-spot map.
2. Read each patch's **"Re-applying the patch"** and **"Conflict guidance"** sections.
3. Keep upstream's new behaviour and re-apply the fork's behaviour on top of it — the
   docs describe the *intent*, so re-implement against the new upstream shape if the
   original hunk no longer fits.
4. If upstream now provides an equivalent feature, check the patch's **"Drop criteria"**.
   Dropping a patch is a product decision: open it as a separate, clearly labelled
   change (or flag it for a human) rather than silently deleting fork behaviour in a
   sync.

Also check for **silent breakage** in non-conflicted fork-only files: every patch lists
the upstream symbols it relies on (private attributes, method signatures, protocol
methods). Search the merged tree for each one.

### 4.4 Verify

Run, from the repository root (see AGENTS.md for tooling):

1. Format check, lint and strict type check (`make check`; the type check is
   `uv run python scripts/check_types.py`).
2. `make unit-tests`, which is `uv run pytest --unit --audio_eot`: tests are selected by
   **category marker**, so every fork test module is included as long as it declares
   `pytestmark = pytest.mark.unit` (collection fails with a hint otherwise). The fork's
   test modules are:
   - `tests/test_interruption_tracker.py` and `tests/test_interruption_backoff.py` (04)
   - `tests/test_fork_recognition_rules.py` (01, 02 helpers)
   - `tests/test_llm_parallel_adapter.py` (07)
   - `tests/test_stt_multilingual.py`, `tests/test_multilingual_heuristics.py`,
     `tests/test_multilingual_deepgram_mapping.py` and helper
     `tests/fake_multilingual_stt.py` (11, 13)
   - `tests/test_user_turn_exceeded.py` carries a one-fixture fork edit (02).

   `tests/test_room.py` needs a local `livekit-server` binary; deselect it when that is
   not installed. `tests/test_loop_monitor.py` has two event-loop timing tests that can
   fail under a full parallel run and pass alone.
3. The per-patch "Verification after sync" checklist for every patch whose files were
   touched by the incoming upstream changes. Several always-on patches (06, 08, 09, 10,
   12) have **no dedicated automated tests**; their checklists describe what to inspect
   manually in the merged code.

### 4.5 Finish

1. Update section 1 of this file: new upstream base commit and version.
2. If a patch's files, symbols or behaviour changed during conflict resolution, update
   that patch's document in the same PR.
3. Open a PR into `patched-1.8` summarising: upstream range merged, conflicted files,
   patches re-applied with changes, patches flagged for possible drop, test results.

### 4.6 Stop and escalate (do not auto-resolve) when

- Upstream reshapes `AudioRecognition`'s EOU flow again (the nested bounce task in
  `_run_eou_detection`, how `endpointing_delay` is chosen, or the `BaseEndpointing`
  object and `create_endpointing`) — patches 02, 03, 04 need a careful re-implementation.
- Upstream changes the `RecognitionHooks` protocol, the speech authorization flow
  (`authorization_tasks` in the reply tasks, `_authorization_allowed`,
  `_user_silence_event`), or `_reconcile_playout_pause` semantics — patches 01, 04, 05.
- Upstream changes `RecognizeStream` internals used by the multilingual adapter
  (`_input_ch`, `_FlushSentinel`, `_event_ch`, `_start_time_offset`, `start_time`,
  `_emit_error`, `_metrics_monitor_task`, the `_main_task` retry offset) or Deepgram's
  `SpeechStream.update_options` / reconnect loop — patches 12, 13.
- Upstream adds an official feature overlapping a fork patch (backchannel vocabulary,
  history-aware interruption backoff, LLM racing, multilingual switching, output
  volume). Flag for a human decision per the patch's drop criteria. Note that upstream's
  **adaptive interruption** and **dynamic endpointing** already exist and were judged
  *not* to replace 01/04 (see `fork/MIGRATION-1.8.md` §3 and §6).
- Any fork test fails after the merge and the fix is not an obvious mechanical update.

---

## 5. Conventions for fork changes

- **Mark fork code.** Fork code inside upstream files carries a `fork(patch NN)`
  comment at the call-site (plus the ticket where one exists, e.g. `fork(patch 04,
  SL-3890)`). Fork-only files need no markers.
- **Prefer additive helpers over edits to upstream blocks.** Patch 05 is the reference
  example: helpers are new methods, and the upstream code gains only one-line calls.
- **Keep new features opt-in** where possible (patches 03's timing changes, 04, 07, 09,
  13 default to upstream behaviour), and put session-level knobs on the `turn_handling`
  TypedDicts rather than new `AgentSession` kwargs (upstream deprecated the latter).
- **Read upstream internals defensively** where tests stub objects: the fork's activity
  and recognition hooks use `getattr` / `isinstance` so a `MagicMock` session or a fake
  activity falls back to upstream behaviour.
- **Every new patch gets a document** in `fork/patches/` using the same section layout,
  and a row in the index (section 2) and hot-spot map (section 3). Test modules need a
  category marker.
- **Commit messages** follow Conventional Commits with an optional `[SL-xxxx]` prefix,
  and explain the production evidence that motivated the change (see the fork commit
  bodies for examples).

---

## 6. Glossary

- **EOU** — end-of-utterance: the decision that the user finished their turn. Driven by
  VAD silence, STT finals, and optionally a turn-detector model that outputs an
  end-of-turn probability (local plugin or a streaming detector whose prediction arrives
  as a future).
- **Endpointing delay** — how long to wait after the user stopped speaking before
  committing the user turn. Upstream 1.5+ keeps it on a `BaseEndpointing` object
  (`min_delay` normally, `max_delay` when the turn detector says the turn is unlikely
  over); `DynamicEndpointing` learns `min_delay` from the caller's pauses.
- **Unlikely threshold** — the turn detector's per-language probability below which the
  turn is considered unfinished.
- **User-silence event** — `AgentActivity._user_silence_event`; agent playout waits on
  it (when interruptions are allowed) so the agent does not start speaking over the
  user. Upstream's pause/resume logic also reads it.
- **Backoff hold event** — `AgentActivity._backoff_hold_event` (patch 04); a second
  event playout waits on, closed until the user has been quiet for the active mode's
  `silence_gate`. Separate from the user-silence event on purpose.
- **Preemptive generation** — upstream feature (on by default since 1.5) that starts
  LLM/TTS generation before the user turn is committed, to cut latency.
- **Adaptive interruption** — upstream's ML overlap classifier (LiveKit Inference only);
  off on self-hosted deployments, where interruption mode resolves to `vad`.
- **Backchannel** — short acknowledgements ("mhm", "okay") that should not interrupt the
  agent.
- **Primary / detector (multilingual)** — the language-pinned STT whose transcripts the
  session sees, and the always-on multilingual STT used only to detect language changes.
- **Audio time** — seconds of audio pushed into a stream, as opposed to wall-clock time.
  Multilingual timers use audio time because input is gapped while the agent speaks.
