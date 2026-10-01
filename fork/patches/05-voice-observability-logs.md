# Patch 05 — Voice observability logs (reply latency, playout hold)

| | |
|---|---|
| **Status** | Always-on (logging only, no behaviour change) |
| **Origin** | 1.4.6: fork commit `d801d7e` (short-loop/agents#59, second part, SL-3890). 1.8.3: part of `19dabfdac` (P9). The 1.4.6 debug line "Speech handle interrupted, cancelling tasks" was **dropped** in the 1.8 move (D8): upstream now records the interruption source on the `agent_turn` span |
| **Depends on** | Patch 04 (measures its hold event; logs include `interruption_mode`) |
| **Automated tests** | None |
| **Code markers** | `fork(patch 04)` at the call-sites, `fork(patch 04/05)` on the helper region |

## Why

To evaluate the interruption-backoff modes (patch 04) and endpointing changes from
production logs, the team needed (a) how long the playout hold actually held a ready reply
and whether the hold turned a collision into a silent drop, and (b) a per-reply
end-to-end latency line with its breakdown. Upstream 1.8 exposes the same latency values
on `ChatMessage.metrics` and as `lk.e2e_latency` on the `agent_turn` span, but the
production dashboards grep the log line.

## Behaviour

Two log lines, both in `livekit-agents/livekit/agents/voice/agent_activity.py`:

1. **debug "playout held by silence gate"** — emitted after the playout authorization
   wait in the TTS and pipeline reply tasks, only if the hold event was **closed** when
   the wait began and the speech allows interruptions. Fields: `held` (seconds since the
   hold closed), `interruption_mode`, `speech_id`, `interrupted_while_held` (true means
   the hold converted a would-be collision into a silent drop).
2. **info "agent reply latency"** — once per pipeline reply, when the first audio frame is
   played. Fields: `e2e_latency` (user stopped speaking → first audio),
   `end_of_turn_delay`, `transcription_delay` (from the user-turn metrics report),
   `llm_ttft`, `tts_ttfb` (first TTS segment), `interruption_mode`, `speech_id`.
   `speech_id` joins it with line 1.

## Implementation walkthrough

The code is written as **additive helper methods** so that upstream's authorization blocks
stay byte-identical (this minimises merge conflicts). Call-sites are single lines marked
`fork(patch 04)`.

- Helpers in the `fork(patch 04/05)` region before `retrieve_chat_ctx`:
  `_log_backoff_hold(speech_handle, closed_at)` (line 1) and
  `_log_reply_latency(speech_handle, *, e2e_latency, user_metrics, llm_ttft, tts_ttfb)`
  (line 2); both read the mode through `_backoff_mode_name()` (mock-safe).
- `_tts_task_impl` and `_pipeline_reply_task_impl`: one line before building
  `authorization_tasks` snapshots `_hold_closed_at = getattr(self,
  "_backoff_hold_closed_at", None)`; one line after upstream's
  `_record_queue_wait(speech_handle)` logs the hold.
- `_pipeline_reply_task_impl`: inside the first-frame callback, right after upstream
  computes `early_metrics["e2e_latency"]`, one call to `_log_reply_latency` using
  `llm_gen_data.ttft` and `first_tts_gen_data.ttfb` (None when there is no TTS).

## Re-applying the patch

Keep the helpers as-is. Re-insert the call-sites: hold snapshot immediately before the
playout authorization wait, hold log immediately after upstream records the queue wait
(both task impls), and the latency log wherever upstream computes the reply's e2e latency
from the user's `stopped_speaking_at`.

## Upstream contracts relied upon

- `SpeechHandle.allow_interruptions`, `.id`, `.interrupted`, `_wait_for_authorization`,
  `_clear_authorization`; `_record_queue_wait`.
- The early-metrics computation in `_pipeline_reply_task_impl` (user metrics report with
  `stopped_speaking_at`, `end_of_turn_delay`, `transcription_delay`; `llm_gen_data.ttft`;
  `first_tts_gen_data.ttfb`).

## Conflict guidance

Conflicts should be limited to the one-line call-sites. If upstream renames or moves the
authorization block or the e2e computation, move the call-sites with it; do not modify
upstream's block itself. Keep the fork's line even though upstream has equivalent metrics,
until log queries are migrated (they grep for "agent reply latency").

## Verification after sync

- Searching `agent_activity.py` for `fork(patch 04` finds the hold snapshots and hold logs
  in both task impls and the latency call in the pipeline task.
- The helper methods exist and type-check.

## Known caveats

- The measured `held` includes any concurrent authorization wait; it is only logged when
  the hold was closed at wait start, which is what makes it attributable to user speech.
- Realtime-model replies (`_realtime_reply_task`, `_realtime_generation_task`) honour the
  hold but do not get either log line.

## Drop criteria

Production dashboards migrated to upstream's `ChatMessage.metrics` /
`session_usage_updated` / span attributes.
