# Patch 05 — Voice observability logs (reply latency, silence-gate hold, interrupt debug)

| | |
|---|---|
| **Status** | Always-on (logging only, no behaviour change) |
| **Origin** | Fork commit `d801d7e` (short-loop/agents#59, second part: "log silence-gate holds and per-reply e2e latency", SL-3890); the interrupt debug line comes from `d7081f0` (short-loop/agents#56) |
| **Depends on** | Patch 04 (logs include `interruption_mode`) |
| **Automated tests** | None |

## Why

To evaluate the interruption-backoff modes (patch 04) and endpointing changes from
production logs, the team needed (a) how long the silence gate actually held a ready
reply and whether the hold turned a collision into a silent drop, and (b) a per-reply
end-to-end latency line with its breakdown. Upstream computes e2e latency but only
renders it in console mode.

## Behaviour

Three log lines, all in `livekit-agents/livekit/agents/voice/agent_activity.py`:

1. **debug "playout held by silence gate"** — emitted after the playout authorization
   wait, only if the user-silence event was **closed** when the wait began and the speech
   allows interruptions. Fields: `held` (seconds), `interruption_mode`, `speech_id`,
   `interrupted_while_held` (true means the gate converted a would-be collision into a
   silent drop).
2. **info "agent reply latency"** — once per reply, when the first audio frame is played.
   Fields: `e2e_latency` (user stopped speaking → first audio), `end_of_turn_delay`,
   `transcription_delay` (from the user-turn metrics report), `llm_ttft`, `tts_ttfb`,
   `interruption_mode`, `speech_id`. `speech_id` joins it with line 1.
3. **debug "Speech handle interrupted, cancelling tasks"** with `handle_id`, in
   `AgentActivity.interrupt` when the current speech is interrupted.

## Implementation walkthrough

The SL-3890 code is written as **additive helper methods** so that upstream's
authorization blocks stay byte-identical (this minimises merge conflicts). Call-sites are
single lines marked with a `fork(SL-3890)` comment.

- New methods on `AgentActivity`, placed just before `_tts_task_impl`, under a comment
  block explaining the design:
  - `_gate_closed_timestamp(speech_handle)` — returns now if interruptions are allowed
    and `_user_silence_event` is not set, else None.
  - `_log_silence_gate_hold(speech_handle, closed_at)` — logs line 1 when `closed_at` is
    not None.
  - `_log_reply_latency(speech_handle, *, e2e_latency, user_metrics, llm_ttft, tts_ttfb)`
    — logs line 2.
- `_tts_task_impl` and `_pipeline_reply_task_impl`: one line before building the
  `authorization_tasks` list captures `_gate_closed_at`; one line after
  `speech_handle._clear_authorization()` logs the hold.
- `_pipeline_reply_task_impl`: inside the first-frame callback, right after upstream
  computes `early_metrics["e2e_latency"]`, one call to `_log_reply_latency` using
  `llm_gen_data.ttft` and `tts_gen_data.ttfb` (None when there is no TTS).
- `interrupt()`: the debug log after `self._current_speech.interrupt(force=force)`.

## Re-applying the patch

Keep the three helpers as-is. Re-insert the call-sites: gate snapshot immediately before
the playout authorization wait, hold log immediately after the authorization is cleared
(both task impls), and the latency log wherever upstream computes the reply's e2e latency
from the user's `stopped_speaking_at`.

## Upstream contracts relied upon

- `SpeechHandle.allow_interruptions`, `.id`, `.interrupted`, `_wait_for_authorization`,
  `_clear_authorization`.
- The early-metrics computation in `_pipeline_reply_task_impl` (user metrics report with
  `stopped_speaking_at`, `end_of_turn_delay`, `transcription_delay`; `llm_gen_data.ttft`;
  `tts_gen_data.ttfb`).

## Conflict guidance

Conflicts should be limited to the one-line call-sites. If upstream renames or moves the
authorization block or the e2e computation, move the call-sites with it; do not modify
upstream's block itself. If upstream starts logging e2e latency itself, keep the fork's
line anyway until log queries are migrated (they grep for "agent reply latency").

## Verification after sync

- Searching `agent_activity.py` for `fork(SL-3890)` finds seven hits: the five call-sites
  (two gate snapshots, two hold logs, one latency log), the comment block above the
  helpers, and the `_log_reply_latency` docstring.
- The helper methods exist and type-check.

## Known caveats

- The measured `held` includes any concurrent authorization wait; it is only logged when
  the gate was closed at wait start, which is what makes it attributable to user speech.
- Realtime-model replies (`_realtime_reply_task` path) do not get the latency line.

## Drop criteria

Upstream emits equivalent structured per-reply latency and playout-hold telemetry, and
production dashboards have been migrated to it.
