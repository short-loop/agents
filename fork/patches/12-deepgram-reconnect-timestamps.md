# Patch 12 — Deepgram timestamp continuity across in-run reconnects

| | |
|---|---|
| **Status** | Always-on for the Deepgram STT plugin (bug fix) |
| **Origin** | 1.4.6: fork commit `e58f51d` (short-loop/agents#60). 1.8.3: part of `d751a2ac1` (P10) |
| **Code markers** | comments starting with `fork:` inside `SpeechStream._run` |
| **Depends on** | — |
| **Required by** | Patch 13 (dedup watermark shared between primary and detector) |
| **Automated tests** | None directly for the plugin change; `tests/test_stt_multilingual.py::test_recreate_child_anchored_to_audio_clock` covers the adapter-side counterpart |

## Why

Deepgram word / utterance timestamps are relative to the audio received **on the current
WebSocket connection**. `SpeechStream.update_options()` (e.g. changing language) triggers
an in-run reconnect inside `_run`, after which timestamps silently restart at 0. Any
consumer comparing timestamps across the reconnect breaks. In the multilingual adapter
(patch 13) an in-place language switch caused already-forwarded speech to be replayed,
duplicating the user's sentence in the committed turn.

This is a general upstream bug; it is a good candidate to contribute upstream.

## Behaviour

- `SpeechStream._run` tracks the audio duration sent on the current connection (a local
  counter incremented in the send task for every frame written to the WebSocket).
- When the reconnect event fires and the loop is about to open a new connection, that
  duration is added to the stream's `_start_time_offset` and the counter is reset.
- Result: all `start_time` / `end_time` values emitted by one `SpeechStream` stay on a
  single continuous audio clock across `update_options` reconnects.

## Implementation walkthrough

`livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py`, inside
`SpeechStream._run`:

- local `conn_audio_sent` initialised with an explanatory comment;
- `send_task` declares it `nonlocal` and increments it after each `send_bytes`;
- in the reconnect branch, right after clearing `_reconnect_event`, the offset is
  advanced and the counter reset (comment explains why).

## Re-applying the patch

Wherever the Deepgram stream reconnects mid-run while the stream object lives on, add the
audio duration consumed by the closed connection to `_start_time_offset` before the next
connection starts reporting timestamps.

## Upstream contracts relied upon

- `RecognizeStream._start_time_offset` being added to Deepgram timestamps in
  `live_transcription_to_speech_data`.
- The reconnect loop structure in `_run` (`_reconnect_event`, per-connection tasks).

## Conflict guidance

If upstream refactors the reconnect loop (or fixes the bug differently, e.g. using
server-side offsets), keep exactly one compensation mechanism — double compensation would
shift timestamps forward. Check the multilingual adapter's "forwarded final end_time
regressed" warning in logs after such a change.

## Verification after sync

- The counter is incremented in the send path and applied in the reconnect path.
- Run the multilingual tests (patch 13).

## Known caveats

- Only covers reconnects driven by `_reconnect_event` inside one `_run`. This patch does
  not change what happens on a full `_run` retry by the base `RecognizeStream`, which
  since 1.5 adds the **wall-clock** gap between runs to `_start_time_offset` in
  `_main_task` — a different mechanism for a different event; both stay. Inside
  the multilingual adapter, children created later (shadows, detector restarts, rebuilds
  after a retry) are anchored to the session audio clock by the adapter itself
  (patch 13).

## Drop criteria

Upstream fixes Deepgram timestamp continuity across `update_options` reconnects.
