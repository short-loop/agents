# Patch 04 — Interruption-backoff modes (normal / primed / transient / sustained)

| | |
|---|---|
| **Status** | Opt-in: inactive unless `turn_handling={"interruption_backoff": InterruptionBackoffOptions(...)}` (or the alias `AgentSession(interruption_backoff=...)`) is passed |
| **Origin** | 1.4.6: fork commits `d801d7e` (short-loop/agents#59, SL-3890) and `d53d905` (short-loop/agents#63, SL-3890). 1.8.3: `19dabfdac` (P9) and `7f8e31138` (mock-safe hooks) |
| **Depends on** | Patch 03 (EOU task structure: `delay_reason`, `use_raw_delay`) |
| **Shares code with** | Patches 02/03 (EOU task), 05 (logs) |
| **Automated tests** | `tests/test_interruption_tracker.py` (21 tests), `tests/test_interruption_backoff.py` (4 session tests + option-key test) |
| **Code markers** | `fork(patch 04)`, `fork(patch 04, SL-3890)`, `fork(patch 04/05)` |

## Why

Some callers pause mid-thought, speak slowly, or are on noisy lines. With them the agent
repeatedly starts replying and gets interrupted ("collisions"). The legacy fork fix
(`interrupt_backoff`, a flat delay for 3 s after an audio barge-in) was active only in a
hard-coded window, bypassed the EOU model, and undercounted interruptions about 6× because
audio barge-ins were never counted. Production analysis showed it inactive on most
collision-heavy calls.

The replacement tracks interruptions **per session** from committed conversation items
and escalates through modes that make the agent more patient only when the conversation
shows a collision pattern, while keeping latency low for everyone else.

**Relation to upstream's dynamic endpointing (1.5+).** They are independent switches by
design (decision D10): the backoff for all customers, dynamic endpointing piloted on a
few. The backoff reads only `min_delay` / `max_delay` from whatever endpointing object is
active and never touches the endpointing mode. A later comparison is written up in
`fork/MIGRATION-1.8.md` §6.

## Behaviour

### What counts

- **Interruption:** an assistant chat message committed with `interrupted=True` **and**
  non-empty text (a reply that was cut off after actually being heard). Unspoken or empty
  replies do not count.
- **User turn:** a committed user chat message with non-empty text. Mode transitions are
  evaluated only at user-turn commits (except primed entry, which is immediate).
- The tracker lives on `AgentSession`, so it **survives agent handoffs**.
- With upstream's adaptive interruption **off** (our self-hosted case) every barge-in
  that interrupts sets `interrupted`, so counts match 1.4.6. If a future detector
  resumes false interruptions, resumed replies commit with `interrupted=False` and the
  thresholds need re-tuning.

### Modes

| Mode | Entry | Exit | Effect |
|---|---|---|---|
| `normal` | start | — | Upstream behaviour. Optionally disables preemptive generation (`normal_disable_preemptive`). |
| `primed` | first interruption, **only if** `primed_silence_gate` is set; enters immediately (not at the next user turn) so the very next reply is protected | sticky; left only to transient or sustained | Playout hold = `primed_silence_gate`. No endpointing backoff, except: when `primed_max_endpointing` is set, EOU-unlikely turns wait that long instead of the endpointing object's `max_delay` (reason `primed_backoff`). |
| `transient` | ≥ `transient_entry_count` interruptions within the last `transient_entry_window` user turns | `transient_exit_clean_turns` consecutive clean user turns → back to `primed` (if configured) or `normal`; the turn window is cleared on exit | Mode settings `transient` (see below). |
| `sustained` | total interruptions ≥ `sustained_entry_total` (checked first at each user turn, from any mode) | sticky for the session | Mode settings `sustained`. |

Defaults (`InterruptionBackoffOptions`): transient entry 2 in 5 turns, exit after 3 clean
turns, sustained at 4 total, primed disabled, `primed_max_endpointing` None,
`normal_disable_preemptive` False; transient settings = backoff 4.0 s, unlikely threshold
0.3, silence gate 1.0 s, preemptive allowed; sustained settings = backoff 6.0 s,
unlikely threshold 0.5, silence gate 2.0 s, preemptive disabled.

### Per-mode settings (`InterruptionModeSettings`, frozen)

- `backoff_delay` — endpointing delay used when the turn is judged unlikely-finished, or
  when no EOU prediction is available.
- `unlikely_threshold` — minimum EOU probability for a fast commit in this mode; combined
  with the turn detector's own per-language threshold using the maximum of the two.
- `silence_gate` — seconds of accumulated user silence (from VAD) required before agent
  playout may start. Requires VAD.
- `disable_preemptive` — disables preemptive generation while in this mode.

### Endpointing effect (post-ladder block in the EOU task)

Runs after the whole delay-selection ladder (patch 02 patterns → upstream turn-detector
branch), so it sees the final `end_of_turn_probability` / `unlikely_threshold` from either
the classic or the streaming turn detector:

- Raw delay chosen by patch 02: the larger of that delay and the mode's `backoff_delay`
  (reason `…+<mode>_backoff`).
- Transient / sustained with a prediction: effective threshold = max(detector threshold,
  mode threshold). Below it → `backoff_delay` (reason `<mode>_backoff`); above it → the
  fast delay stands. **Confident turns stay fast even in backoff modes.**
- Transient / sustained without a prediction (no turn detector, unsupported language,
  failed or timed-out prediction — `end_of_turn_probability is None`) → flat
  `backoff_delay` (reason `<mode>_backoff_no_eou`).
- Primed with `primed_max_endpointing` and the ladder chose `eou_unlikely` → that value
  (reason `primed_backoff`).

### Playout hold (silence gate)

The 1.4.6 fork bent upstream's `_user_silence_event`; on 1.8 that event also drives
upstream's pause/resume logic (`_reconcile_playout_pause`), so the gate is a **separate
event** (decision D3):

- `AgentActivity._backoff_hold_event` (set = open) and `_backoff_hold_closed_at`.
- `_update_backoff_hold(silence)` closes the hold when a gate is active and the given
  accumulated silence is ≤ gate, opens it otherwise (`None` silence = no VAD evidence →
  open, fails soft). Called from `on_start_of_speech` (silence 0.0), `on_end_of_speech`
  (`ev.silence_duration`, or `None` without a VAD event) and `on_vad_inference_done`
  (`ev.raw_accumulated_silence`; Silero keeps accumulating raw silence across inference
  events after end-of-speech, which is what reopens the hold).
- The hold's `wait()` is appended to `authorization_tasks` next to upstream's
  `_user_silence_event.wait()` in all four reply tasks (`_tts_task_impl`,
  `_pipeline_reply_task_impl`, `_realtime_reply_task`, `_realtime_generation_task`),
  under the same `allow_interruptions` condition.
- Result: a ready reply waits until the user has been quiet long enough; if the user
  resumes during the hold, the reply is interrupted *silently* instead of audibly
  colliding. Upstream's silence event and pause/resume behaviour are unchanged.

### Preemptive generation

- `AgentActivity.on_preemptive_generation` returns early when the tracker says
  preemptive is disabled for the current mode (upstream's `preemptive_opts["enabled"]`
  check gains one `or`).
- `AgentSession._conversation_item_added`: if a user-turn commit changes the mode into
  one that disables preemptive generation, any in-flight preemptive generation is
  cancelled (`AgentActivity._cancel_preemptive_generation`).
- Upstream enables preemptive generation by default since 1.5, so
  `normal_disable_preemptive` matters for every session that adopts the defaults.

### Configuration

`TurnHandlingOptions.interruption_backoff: InterruptionBackoffOptions | None` (session
level; the resolved value is `AgentSessionOptions.interruption_backoff`). The
`AgentSession(interruption_backoff=...)` kwarg is an alias that wins over the key. The
1.4.6 accepted-and-ignored `interrupt_backoff` kwarg no longer exists (D8).

### Logs

- info "interruption mode changed" with `old_mode`, `new_mode`, `reason`
  (`primed_entry`, `transient_entry`, `transient_exit`, `sustained_entry`),
  `total_interruptions`, `window`.
- debug "interruption recorded" with `total_interruptions`, `mode`.
- `interruption_mode` field on the `eou sleep` log (patch 03) and on patch 05's logs.

## Implementation walkthrough

### New file `livekit-agents/livekit/agents/voice/interruption_tracker.py`

- `InterruptionMode` enum (`NORMAL`, `PRIMED`, `TRANSIENT`, `SUSTAINED`; values are the
  lower-case names used in logs).
- `InterruptionModeSettings` and `InterruptionBackoffOptions` frozen dataclasses.
- `InterruptionTracker`: constructed with options or None (disabled → always normal).
  State: mode, total count, pending count since last user turn, and a bounded deque of
  per-user-turn interruption counts. Methods: `enabled`, `mode`, `total_interruptions`,
  `record_interruption()`, `record_user_turn()` (sustained → transient-exit →
  transient-entry), `backoff_params()` (`(unlikely_threshold, backoff_delay)` or None),
  `silence_gate()`, `primed_max_endpointing()`, `preemptive_disabled()`.

### `livekit-agents/livekit/agents/voice/turn.py`

- `TurnHandlingOptions.interruption_backoff` key (imports the options dataclass from the
  tracker module; no cycle, the tracker imports only the logger).

### `livekit-agents/livekit/agents/voice/agent_session.py`

- Imports `InterruptionBackoffOptions`, `InterruptionTracker`.
- `AgentSessionOptions.interruption_backoff` property (reads the `turn_handling` key).
- `AgentSession.__init__`: kwarg alias after `stt_context_options`; resolution next to
  `user_turn_limit`; the key stored in the `TurnHandlingOptions(...)` it builds;
  `self._interruption_tracker` created right after `self._activity`.
- `_conversation_item_added`: tracker feed before upstream's debug log and emit.

### `livekit-agents/livekit/agents/voice/agent_activity.py`

- Imports `InterruptionTracker`; `_backoff_hold_event` / `_backoff_hold_closed_at` next
  to `_user_silence_event`.
- Region `fork(patch 04/05)` just before `retrieve_chat_ctx`: `_backoff_tracker()`
  (returns the session's tracker only if it `isinstance` `InterruptionTracker`, so mocked
  sessions degrade to upstream), `_backoff_mode_name()`, `_update_backoff_hold()`, and
  patch 05's log helpers.
- `on_start_of_speech`, `on_end_of_speech`, `on_vad_inference_done`,
  `on_preemptive_generation`, and the four authorization blocks (hold `wait()` added via
  `getattr` so fakes without the attribute skip it).

### `livekit-agents/livekit/agents/voice/audio_recognition.py`

- Imports `InterruptionMode`, `InterruptionTracker`; post-ladder block in
  `_bounce_eou_task` (tracker via `getattr(self._session, "_interruption_tracker")` +
  `isinstance`); `interruption_mode` in the `eou sleep` log.

### Exports

`InterruptionBackoffOptions`, `InterruptionMode`, `InterruptionModeSettings` from
`livekit.agents.voice` and `livekit.agents`.

## Re-applying the patch

1. Copy the tracker module unchanged (fork-only file).
2. Plumb options: `TurnHandlingOptions` key (+ kwarg alias) → tracker instance on the
   session.
3. Feed the tracker from the single choke point where committed chat messages enter the
   session history (today `_conversation_item_added`).
4. Apply mode thresholds / delays **after** upstream's delay selection, keeping the EOU
   model in the loop (max of thresholds) so confident turns stay fast; treat "no
   prediction" as the flat backoff.
5. Keep the hold as a separate event awaited next to `_user_silence_event`; never bend
   upstream's silence event.
6. Gate preemptive generation, and cancel in-flight preemptive work on mode entry.

## Upstream contracts relied upon

- `ChatMessage.interrupted` and `ChatMessage.text_content` on committed messages.
- `AgentSession._conversation_item_added` being called for every committed user and
  assistant message (including after handoffs).
- The `authorization_tasks` lists in the four reply tasks and
  `speech_handle.wait_if_not_interrupted`.
- `vad.VADEvent.silence_duration` and `raw_accumulated_silence`, and Silero's behaviour of
  continuing to accumulate raw silence after end-of-speech.
- `AgentActivity._cancel_preemptive_generation()`, `preemptive_generation_opts`.
- `_bounce_eou_task` defining `end_of_turn_probability` / `unlikely_threshold` (None when
  no prediction) before the ladder; `self._endpointing.max_delay`.

## Conflict guidance

- Authorization blocks: upstream adds futures to `authorization_tasks` from time to time;
  keep the hold append next to the silence-event append in every task.
- `_conversation_item_added`: keep the tracker update before the emit. If upstream moves
  history insertion elsewhere, move the tracker feed with it.
- If upstream introduces its own history-aware backoff, do not merge the two silently —
  escalate.

## Verification after sync

- `tests/test_interruption_tracker.py` and `tests/test_interruption_backoff.py` pass;
  `tests/test_realtime_reply_chat_ctx.py` and `tests/test_realtime_adaptive_interruption.py`
  (they drive the activity methods on fakes) pass.
- With the option `None` every fork branch is inert (tracker disabled → mode always normal
  → hold always open, no backoff, preemptive untouched).
- The three log messages still exist with their fields.

## Known caveats

- The hold needs VAD; without VAD events it never engages (and opens on `None`).
- `primed_max_endpointing` applies only when the turn detector ran and judged the turn
  unlikely; the no-EOU fallbacks do not use it.
- Agent-level `turn_handling` overrides do not carry `interruption_backoff` (session only).
- The fake VAD in tests emits no `INFERENCE_DONE` after end-of-speech, so session tests
  use `silence_gate=0.0`.

## Drop criteria

Upstream ships history-aware interruption handling usable without LiveKit Inference that
(a) counts interruptions from committed items, (b) holds playout until sufficient user
silence, and (c) lengthens endpointing only for uncertain turns. Requires a product
decision and production comparison (see `fork/MIGRATION-1.8.md` §6).
