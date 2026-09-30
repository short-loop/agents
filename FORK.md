# FORK.md — short-loop/agents

This repository is a **patched fork** of [livekit/agents](https://github.com/livekit/agents).
It tracks upstream closely and carries a small, well-defined set of behavioural patches
used by ShortLoop's production voice agents.

This file is the entry point for anyone (human or automated agent) who needs to:

- understand what the fork changes and why,
- sync the fork with upstream without losing or breaking a patch,
- decide whether a patch can be dropped because upstream now covers it.

Per-patch documentation lives in [`fork/patches/`](fork/patches/) (layout and authoring
rules in [`fork/README.md`](fork/README.md)). Each patch file is
self-contained: purpose, behaviour, every file and symbol it touches, how to re-apply it
if upstream moved the surrounding code, conflict hot-spots, tests, and drop criteria.

> These documents contain **no code**. They reference files, classes, functions and
> option names, and describe behaviour in natural language. The source of truth for the
> implementation is always the code on the `patched` branch.

---

## 1. Fork facts

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/livekit/agents` (default branch `main`) |
| Fork repository | `https://github.com/short-loop/agents` |
| Fork working branch | `patched` (the only long-lived branch; all fork work lands here) |
| Current upstream base | commit `29b71d4` — "livekit-agents 1.4.6 (#5123)" (2026-03-16) |
| Package version on the base | `livekit-agents` 1.4.6 (plugins at their matching 1.4.6-era versions) |
| Fork delta vs. base | 31 files, ~4.9k lines added, ~20 lines removed (≈ 60% is tests and the new multilingual STT package) |
| Fork PR numbering | Fork PRs are `short-loop/agents#NN` (two-digit). Upstream PRs are `livekit/agents#NNNN` (four-digit). Commit subjects keep upstream's `(#NNNN)` suffix for upstream commits and `(#NN)` for fork commits. |
| Ticket references | `SL-3890` = interruption-backoff work (patches 04, 05). |

### How the fork is structured in git

- Upstream commits are kept verbatim (no rewriting). The last upstream commit in the
  `patched` history is the base listed above.
- Fork commits are squash-merged PRs on top of the base. At the time of writing they are,
  oldest first:

| Fork commit | Fork PR | Summary | Patches it contributes to |
|---|---|---|---|
| `d7081f0` | short-loop/agents#56 | "chore: upgrade agent to 1.4.6" — **re-application of all pre-1.4.6 ShortLoop patches** onto the 1.4.6 base (backchannel words, stale speaking-time fix, parallel LLM adapter, bracket stripping, commit words, 0.5 s sleep floor, EOU sleep log, max volume, ElevenLabs previous_text, old `interrupt_backoff`) | 01, 02, 03, 06, 07, 08, 09, 10 (and the now-removed legacy interrupt backoff) |
| `104ed89` | short-loop/agents#58 | LLM hedging fixes + observability for `ParallelAdapter` (winner-id race, metrics filtering) | 07 |
| `d801d7e` | short-loop/agents#59 | [SL-3890] Replace the legacy `interrupt_backoff` with interruption-backoff modes; add silence-gate and reply-latency log lines | 04, 05 |
| `e58f51d` | short-loop/agents#60 | Silent multilingual STT switching (`MultilingualAdapter`), Deepgram language metadata, Deepgram reconnect timestamp continuity | 11, 12, 13 |
| `79c0347` | short-loop/agents#61 | Multilingual: rescue mismatched-language finals despite overlap | 13 |
| `062d5dc` | short-loop/agents#62 | Multilingual: never drop credible foreign detector finals on coverage alone | 13 |
| `d53d905` | short-loop/agents#63 | [SL-3890] `primed_max_endpointing` and `normal_disable_preemptive` | 04 |

Because `d7081f0` is a squash of many older patches, **git history alone does not
separate the patches**. The per-patch documents are the authoritative map of which hunk
belongs to which feature.

The quickest way to see the complete fork delta is a diff between the upstream base
commit and the tip of `patched`. After a sync, the new upstream base commit replaces
`29b71d4` in the table above.

---

## 2. Patch index

Status legend: **Opt-in** = shipped but inactive unless configured. **Always-on** =
changes default behaviour for every user of the fork, with no switch to turn it off.
**Dropped** = removed from the code, document kept for history. Every patch that is not
Dropped is active and must be preserved through syncs.

| # | Patch | Area | Status | Upstream-touching files | Doc |
|---|---|---|---|---|---|
| 01 | Backchannel & commit words | voice / turn-taking | Always-on (defaults) + configurable | `voice/audio_recognition.py`, `voice/agent_activity.py`, `voice/agent_session.py` | [01](fork/patches/01-backchannel-and-commit-words.md) |
| 02 | Endpointing delays for numbers / alphanumerics | voice / EOU | Always-on | `voice/audio_recognition.py` | [02](fork/patches/02-endpointing-pattern-delays.md) |
| 03 | EOU sleep timing: stale VAD detection, 0.5 s floor, `eou sleep` log | voice / EOU | Always-on | `voice/audio_recognition.py` | [03](fork/patches/03-eou-sleep-timing.md) |
| 04 | Interruption-backoff modes (normal / primed / transient / sustained) | voice / turn-taking | Opt-in (`interruption_backoff=`) | `voice/interruption_tracker.py` (new), `voice/agent_session.py`, `voice/agent_activity.py`, `voice/audio_recognition.py`, `voice/__init__.py`, `agents/__init__.py` | [04](fork/patches/04-interruption-backoff-modes.md) |
| 05 | Voice observability logs (reply latency, silence-gate hold, interrupt debug) | voice / logging | Always-on (logs only) | `voice/agent_activity.py` | [05](fork/patches/05-voice-observability-logs.md) |
| 06 | Last user language accessor + shorter language-detection minimum | voice | Always-on | `voice/audio_recognition.py`, `voice/agent_activity.py`, `voice/agent_session.py` | [06](fork/patches/06-last-user-language.md) |
| 07 | `llm.ParallelAdapter` (LLM hedging / racing) | llm | Opt-in (new class) | `llm/parallel_adapter.py` (new), `llm/__init__.py`, `metrics/base.py` | [07](fork/patches/07-llm-parallel-adapter.md) |
| 08 | LiveKit Inference LLM stream tweaks (drop `parallel_tool_calls` w/o tools, strip `[` artifacts) | inference | Always-on | `inference/llm.py` | [08](fork/patches/08-inference-llm-stream-tweaks.md) |
| 09 | Room audio output `max_volume` | voice / room_io | Opt-in (default 1.0 = no-op) | `voice/room_io/_output.py`, `voice/room_io/room_io.py`, `voice/room_io/types.py` | [09](fork/patches/09-room-audio-max-volume.md) |
| 10 | ElevenLabs `previous_text` priming | plugin: elevenlabs | Always-on | `livekit-plugins-elevenlabs/.../tts.py` | [10](fork/patches/10-elevenlabs-previous-text.md) |
| 11 | STT language metadata (`SpeechData.detected_languages`, `TimedString.language`, Deepgram mapping) | stt / types / deepgram | Always-on (additive fields) | `stt/stt.py`, `types.py`, `livekit-plugins-deepgram/.../stt.py` | [11](fork/patches/11-stt-language-metadata.md) |
| 12 | Deepgram timestamp continuity across in-run reconnects | plugin: deepgram | Always-on | `livekit-plugins-deepgram/.../stt.py` | [12](fork/patches/12-deepgram-reconnect-timestamps.md) |
| 13 | `stt.MultilingualAdapter` (silent language switching) | stt | Opt-in (new class) | `stt/multilingual/*` (new), `stt/__init__.py`, example + tests | [13](fork/patches/13-multilingual-stt-adapter.md) |

All `livekit-agents` paths above are relative to `livekit-agents/livekit/agents/`.
Plugin paths are relative to `livekit-plugins/`.

### Patch dependencies

- **13 depends on 11 and 12.** The multilingual heuristics read per-word language tags
  (11) and the adapter's dedup watermark requires continuous Deepgram timestamps across
  `update_options()` reconnects (12).
- **02, 03 and 04 are interleaved in one function** — the nested end-of-utterance task
  inside `AudioRecognition._run_eou_detection`. 03 introduced the `delay_reason` /
  `use_raw_delay` / `compute_sleep` structure; 02 adds the number branches to it; 04
  adds mode-aware thresholds and delays to it. Re-apply them together.
- **01 and 04 both extend the `RecognitionHooks` protocol** (`is_bot_speaking` from 01,
  `interruption_mode` from 04). `AgentActivity` is the only implementer.
- **05 depends on 04** (its log lines include `interruption_mode`).
- **06 is used by application code** that picks the TTS voice / prompt language; it has
  no in-repo dependents besides the chain `AgentSession → AgentActivity → AudioRecognition`.

---

## 3. Conflict hot-spot map

Use this to predict which patches an upstream change will collide with. Files are listed
from highest to lowest conflict risk.

| File | Patches | Risk | Notes |
|---|---|---|---|
| `voice/audio_recognition.py` | 01, 02, 03, 04, 06 | **Very high** | Upstream edits `_on_stt_event`, `_on_vad_event`, `_run_eou_detection` frequently (EOU / turn-detection work). The fork rewrote the delay-selection and sleep computation inside the nested EOU task. Read patches 02/03/04 together before resolving. |
| `voice/agent_activity.py` | 01, 04, 05, 06 | **High** | Touches `_interrupt_by_audio_activity`, `on_end_of_speech`, `on_vad_inference_done`, `on_preemptive_generation`, `_tts_task_impl`, `_pipeline_reply_task_impl`, `interrupt`. The authorization blocks in the two task impls are intentionally left byte-identical to upstream; fork lines around them are marked with a `fork(SL-3890)` comment. |
| `voice/agent_session.py` | 01, 04, 06 | Medium | New constructor kwargs, `AgentSessionOptions` fields, the tracker instance, and logic in `_conversation_item_added`. Upstream adds constructor kwargs regularly — conflicts are usually mechanical (keep both). |
| `livekit-plugins-deepgram/.../stt.py` | 11, 12 | Medium | `SpeechStream._run` (reconnect loop) and `live_transcription_to_speech_data`. |
| `inference/llm.py` | 08 | Low–medium | Two small insertions in `LLMStream`. |
| `voice/room_io/*` | 09 | Low | Additive parameter plumbing. |
| `livekit-plugins-elevenlabs/.../tts.py` | 10 | Low | One key added to two request payloads. |
| `stt/stt.py`, `types.py`, `metrics/base.py` | 11, 07 | Low | Additive dataclass / attribute fields. |
| `__init__.py` files (`agents`, `voice`, `llm`, `stt`) | 04, 07, 13 | Low | Export lists; keep both sides. |
| New fork-only files (`voice/interruption_tracker.py`, `llm/parallel_adapter.py`, `stt/multilingual/*`, fork tests, example) | 04, 07, 13 | Never conflict | But they **depend on upstream internals** — see each patch's "Upstream contracts" section. A clean merge can still break them. |

---

## 4. Upstream sync playbook

This is the procedure the daily sync workflow should follow. It is written for an
automated agent but works the same for a human.

### 4.1 Preparation

1. Add upstream as a remote (the fork checkout only has `origin`) and fetch upstream
   `main` plus tags.
2. Determine the current base: the most recent upstream commit reachable from `patched`
   (today `29b71d4`). Record the list of new upstream commits since that base.
3. Create a working branch from `patched` (never sync directly on `patched`).
4. Skim the new upstream commits and flag any that touch files in the hot-spot map
   (section 3). For each flagged commit, open the corresponding patch docs **before**
   merging.

### 4.2 Integrate

- **Preferred: merge** upstream `main` (or the target release tag) into the working
  branch. The history of `patched` is merge-friendly (fork commits sit on top of the
  base); merging keeps upstream commit identities intact, which the "Fork facts" table
  relies on.
- Rebasing the fork commits onto upstream is acceptable only if the team explicitly
  chooses it; if so, rebase in the order listed in section 1 and resolve per patch.
- Prefer syncing to **upstream release commits** ("livekit-agents X.Y.Z (#NNNN)") rather
  than arbitrary `main` commits when a release is available; it keeps plugin versions
  coherent. Daily syncs may track `main`, but record the exact base commit.

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

1. Format check, lint and strict type check (`make check`).
2. The fork's own test files (all run offline, no cloud credentials), invoked explicitly
   with pytest:
   - `tests/test_interruption_tracker.py`
   - `tests/test_agent_session.py` (includes the fork's interruption-backoff tests)
   - `tests/test_stt_multilingual.py` (uses the fork's `tests/fake_multilingual_stt.py`)
   - `tests/test_multilingual_heuristics.py`
   - `tests/test_multilingual_deepgram_mapping.py`

   **Important:** only `tests/test_agent_session.py` is listed in the `unit-tests` target
   of the root `makefile`, which is what CI (`.github/workflows/tests.yml`) runs. The
   other four fork test files are **not** run by CI today, so the sync workflow must run
   them itself.
3. The general unit-test suite (`make unit-tests` from the repository root).
4. The per-patch "Verification after sync" checklist for every patch whose files were
   touched by the incoming upstream changes. Several always-on patches (01, 02, 03, 06,
   08, 09, 10, 12) have **no dedicated automated tests**; their checklists describe what
   to inspect manually in the merged code.

### 4.5 Finish

1. Update section 1 of this file: new upstream base commit and version.
2. If a patch's files, symbols or behaviour changed during conflict resolution, update
   that patch's document in the same PR.
3. Open a PR into `patched` summarising: upstream range merged, conflicted files, patches
   re-applied with changes, patches flagged for possible drop, test results.

### 4.6 Stop and escalate (do not auto-resolve) when

- Upstream rewrote `AudioRecognition`'s EOU flow (e.g. `_run_eou_detection` or the nested
  bounce task moved, split, or changed how `endpointing_delay` is chosen) — patches 02,
  03, 04 all need a careful re-implementation.
- Upstream changed the `RecognitionHooks` protocol, `AgentActivity`'s user-silence event
  (`_user_silence_event`) semantics, or the speech authorization flow in
  `_tts_task_impl` / `_pipeline_reply_task_impl`.
- Upstream changed `RecognizeStream` internals used by the multilingual adapter
  (`_input_ch`, `_FlushSentinel`, `_event_ch`, `_start_time_offset`, `_emit_error`,
  `_metrics_monitor_task`) or Deepgram's `SpeechStream.update_options` / reconnect loop.
- Upstream added an official feature overlapping a fork patch (backchannel filtering,
  interruption handling / adaptive endpointing, LLM racing, multilingual switching,
  output volume). Flag for a human decision per the patch's drop criteria.
- Any fork test fails after the merge and the fix is not an obvious mechanical update.

---

## 5. Conventions for fork changes

- **Mark fork code.** New fork code inside upstream files should carry a short
  `fork(<ticket>)` comment at the call-site (the SL-3890 work already does this). Older
  patches (01–03, 06–10) pre-date this convention and are not marked — the patch docs are
  the only map for them.
- **Prefer additive helpers over edits to upstream blocks.** Patch 05 is the reference
  example: helpers are new methods, and the upstream code gains only one-line calls.
- **Keep new features opt-in** where possible (patches 04, 07, 09, 13 default to
  upstream behaviour).
- **Every new patch gets a document** in `fork/patches/` using the same section layout,
  and a row in the index (section 2) and hot-spot map (section 3).
- **Commit messages** follow Conventional Commits with an optional `[SL-xxxx]` prefix,
  and explain the production evidence that motivated the change (see the fork commit
  bodies for examples).

---

## 6. Glossary

- **EOU** — end-of-utterance: the decision that the user finished their turn. Driven by
  VAD silence, STT finals, and optionally a turn-detector model that outputs an
  end-of-turn probability.
- **Endpointing delay** — how long to wait after the user stopped speaking before
  committing the user turn. Upstream uses `min_endpointing_delay` normally and
  `max_endpointing_delay` when the turn detector says the turn is unlikely to be over.
- **Unlikely threshold** — the turn detector's per-language probability below which the
  turn is considered unfinished.
- **User-silence event / silence gate** — `AgentActivity._user_silence_event`; agent
  playout waits on it (when interruptions are allowed) so the agent does not start
  speaking over the user. Patch 04 lets a mode require a minimum amount of accumulated
  silence before it opens.
- **Preemptive generation** — upstream feature that starts LLM/TTS generation before the
  user turn is committed, to cut latency.
- **Backchannel** — short acknowledgements ("mhm", "okay") that should not interrupt the
  agent.
- **Primary / detector (multilingual)** — the language-pinned STT whose transcripts the
  session sees, and the always-on multilingual STT used only to detect language changes.
- **Audio time** — seconds of audio pushed into a stream, as opposed to wall-clock time.
  Multilingual timers use audio time because input is gapped while the agent speaks.
