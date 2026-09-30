# Patch 04 — Interruption-backoff modes (normal / primed / transient / sustained)

| | |
|---|---|
| **Status** | Opt-in: inactive unless `AgentSession(interruption_backoff=InterruptionBackoffOptions(...))` is passed |
| **Origin** | Fork commits `d801d7e` (short-loop/agents#59, SL-3890) and `d53d905` (short-loop/agents#63, SL-3890). Supersedes the legacy `interrupt_backoff` float from `d7081f0` (short-loop/agents#56). |
| **Depends on** | Patch 03 (EOU task structure) |
| **Shares code with** | Patches 01 (hooks protocol), 02/03 (EOU task), 05 (logs) |
| **Automated tests** | `tests/test_interruption_tracker.py` (21 tests), five tests in `tests/test_agent_session.py` (`test_interruption_backoff_transient_entry`, `test_interruption_backoff_sustained_disables_preemptive`, `test_interruption_backoff_ignores_unspoken_replies`, `test_interruption_backoff_primed_after_single_interruption`, `test_interrupt_backoff_deprecated_kwarg`) |

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

## Behaviour

### What counts

- **Interruption:** an assistant chat message committed with `interrupted=True` **and**
  non-empty text (a reply that was cut off after actually being heard). Unspoken or empty
  replies do not count.
- **User turn:** a committed user chat message with non-empty text. Mode transitions are
  evaluated only at user-turn commits (except primed entry, which is immediate).
- The tracker lives on `AgentSession`, so it **survives agent handoffs**.

### Modes

| Mode | Entry | Exit | Effect |
|---|---|---|---|
| `normal` | start | — | Upstream behaviour. Optionally disables preemptive generation (`normal_disable_preemptive`). |
| `primed` | first interruption, **only if** `primed_silence_gate` is set; enters immediately (not at the next user turn) so the very next reply is protected | sticky; left only to transient or sustained | Playout silence gate = `primed_silence_gate`. No endpointing backoff, except: when `primed_max_endpointing` is set, EOU-unlikely turns wait that long instead of the stock `max_endpointing_delay` (reason `primed_backoff`). |
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

### Endpointing effect (in the EOU task)

- Turn detector available and prediction succeeded: effective threshold = max(detector
  threshold, mode threshold). Below it → `backoff_delay` (reason `<mode>_backoff`); in
  primed with `primed_max_endpointing` → that value (`primed_backoff`); otherwise stock
  max delay (`eou_unlikely`). Above it → the fast default delay. **Confident turns stay
  fast even in backoff modes.**
- Prediction failed, language unsupported by the detector, or no turn detector at all,
  in transient/sustained → flat `backoff_delay` (reason `<mode>_backoff_no_eou`).
- Number / alphanumeric endings (patch 02) take the larger of their delay and the mode's
  backoff.

### Silence-gate effect (playout)

Upstream sets `AgentActivity._user_silence_event` at end-of-speech and clears/sets it on
VAD inference updates; agent playout waits on this event when interruptions are allowed.
When a gate is active (primed / transient / sustained):

- `on_end_of_speech`: the event is set only if the VAD event's `silence_duration` already
  meets the gate (or there is no VAD event).
- `on_vad_inference_done`: instead of upstream's "speaking and short silence" rule, the
  event is cleared while `raw_accumulated_silence` ≤ gate and set once it exceeds it.
  Silero keeps accumulating raw silence across inference events even after end-of-speech,
  which is what eventually opens the gate.
- Result: a ready reply waits until the user has been quiet long enough; if the user
  resumes during the hold, the reply is interrupted *silently* (never heard) instead of
  audibly colliding.

### Preemptive generation

- `AgentActivity.on_preemptive_generation` returns early when the tracker says
  preemptive is disabled for the current mode.
- `AgentSession._conversation_item_added`: if a user-turn commit changes the mode into
  one that disables preemptive generation, any in-flight preemptive generation is
  cancelled (`AgentActivity._cancel_preemptive_generation`).

### Legacy kwarg

`AgentSession(interrupt_backoff=...)` is still accepted (in the deprecated kwargs block)
but ignored; it logs a warning pointing to `interruption_backoff`.

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
- `InterruptionModeSettings` and `InterruptionBackoffOptions` frozen dataclasses (fields
  and defaults above, each with a docstring).
- `InterruptionTracker`: constructed with options or None (disabled → always normal).
  State: mode, total count, pending count since last user turn, and a bounded deque of
  per-user-turn interruption counts (length = max of entry window and exit clean turns).
  Methods: `enabled`, `mode`, `total_interruptions`, `record_interruption()`,
  `record_user_turn()` (flushes pending into the window, evaluates transitions in the
  order sustained → transient-exit → transient-entry, returns the new mode),
  `backoff_params()`, `silence_gate()`, `preemptive_disabled()`.

### `livekit-agents/livekit/agents/voice/agent_session.py`

- Imports `InterruptionBackoffOptions`, `InterruptionTracker`.
- `AgentSessionOptions.interruption_backoff` field.
- `AgentSession.__init__`: kwarg `interruption_backoff` (after `min_interruption_words`),
  docstring entry, deprecated `interrupt_backoff` kwarg with warning, and
  `self._interruption_tracker` created next to the agent/activity attributes.
- `_conversation_item_added`: records interruptions / user turns (only when the tracker
  is enabled and the message has non-empty text) and cancels in-flight preemptive
  generation on mode entry, **before** the `conversation_item_added` event is emitted.

### `livekit-agents/livekit/agents/voice/agent_activity.py`

- Imports `InterruptionMode`.
- New hook method `interruption_mode()` returning the session tracker's mode.
- `on_end_of_speech`, `on_vad_inference_done`: silence-gate logic described above.
- `on_preemptive_generation`: extra early-return condition.

### `livekit-agents/livekit/agents/voice/audio_recognition.py`

- Imports `InterruptionMode`; `RecognitionHooks.interruption_mode()`.
- Nested EOU task: reads the mode and `session.options.interruption_backoff`, derives
  `mode_threshold`, `mode_backoff`, `mode_name`, `primed_max_endpointing`, and applies
  them in every branch of the delay-selection chain (see patches 02/03).

### Exports

`InterruptionBackoffOptions`, `InterruptionMode`, `InterruptionModeSettings` are exported
from `livekit.agents.voice` and from `livekit.agents` (both import list and `__all__`).

## Re-applying the patch

1. Copy the tracker module unchanged (fork-only file).
2. Plumb options: `AgentSession` kwarg → `AgentSessionOptions.interruption_backoff`;
   tracker instance on the session.
3. Feed the tracker from the single choke point where committed chat messages enter the
   session history (today `_conversation_item_added`).
4. Expose the mode to `AudioRecognition` through `RecognitionHooks`.
5. Apply mode thresholds / delays in EOU delay selection, keeping the EOU model in the
   loop (max of thresholds) so confident turns stay fast.
6. Apply the silence gate wherever upstream sets / clears `_user_silence_event`.
7. Gate preemptive generation, and cancel in-flight preemptive work on mode entry.
8. Keep the deprecated `interrupt_backoff` kwarg accepted-and-ignored.

## Upstream contracts relied upon

- `ChatMessage.interrupted` and `ChatMessage.text_content` on assistant messages
  committed after an interruption.
- `AgentSession._conversation_item_added` being called for every committed user and
  assistant message (including after handoffs).
- `AgentActivity._user_silence_event` being the event that gates playout (awaited in
  `_tts_task_impl` / `_pipeline_reply_task_impl` when `allow_interruptions` is true).
- `vad.VADEvent.silence_duration` and `raw_accumulated_silence`, and Silero's behaviour of
  continuing to accumulate raw silence after end-of-speech.
- `AgentActivity._cancel_preemptive_generation()`.
- Turn detector API: `supports_language`, `predict_end_of_turn`, `unlikely_threshold`.

## Conflict guidance

- `on_vad_inference_done`: upstream's condition (speaking and silence ≤ half of
  `min_endpointing_delay` → clear, else set) must remain the `elif` branch taken when no
  gate is active. If upstream reworks it, preserve "gate active → clear until raw silence
  exceeds gate".
- `on_end_of_speech`: upstream unconditionally sets the event; the fork makes it
  conditional only when a gate is active.
- `_conversation_item_added`: keep the tracker update before the emit. If upstream moves
  history insertion elsewhere, move the tracker feed with it.
- If upstream introduces its own adaptive interruption / endpointing (they have been
  iterating on interruption handling), do not merge the two silently — escalate.

## Verification after sync

- Run `tests/test_interruption_tracker.py` and the five interruption-backoff tests in
  `tests/test_agent_session.py`.
- Check that with `interruption_backoff=None` every fork branch is inert (tracker
  disabled → mode always normal → gate None, preemptive not disabled unless options).
- Check the three log messages still exist with their fields.

## Known caveats

- `InterruptionTracker.backoff_params()` is currently unused (the EOU task reads options
  directly).
- The silence gate needs VAD; without VAD events it never engages.
- `primed_max_endpointing` applies only when the turn detector ran and judged the turn
  unlikely; the no-EOU fallbacks do not use it.

## Drop criteria

Upstream ships adaptive, history-aware interruption handling that (a) counts
interruptions from committed items, (b) holds playout until sufficient user silence, and
(c) lengthens endpointing only for uncertain turns. Requires a product decision and
production comparison.
