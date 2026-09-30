# Patch 09 — Room audio output `max_volume`

| | |
|---|---|
| **Status** | Opt-in (default 1.0 = no change) |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56, "chore: migrate sl patches") |
| **Depends on** | — |
| **Automated tests** | None |

## Why

Some TTS voices / telephony paths were too loud for callers. The fork allows attenuating
the agent's published audio at the room output without touching TTS settings.

## Behaviour

- New option `max_volume` (float, 0.0–1.0, default 1.0) on both:
  - `room_io.AudioOutputOptions` (current API, used via `RoomOptions`), and
  - `room_io.RoomOutputOptions` (deprecated upstream API; converted into
    `AudioOutputOptions` by `RoomOptions`' legacy conversion).
- When below 1.0, every frame published by `_ParticipantAudioOutput` is scaled by the
  factor (16-bit PCM, clipped) right before it is captured into the room audio source.
  At 1.0 or above the frame passes through untouched (no copy).

## Implementation walkthrough

- `livekit-agents/livekit/agents/voice/room_io/types.py`: `max_volume` field on
  `AudioOutputOptions` and `RoomOutputOptions`; the legacy conversion
  `RoomOptions._create_from_legacy` passes it through when building
  `AudioOutputOptions` from `RoomOutputOptions`.
- `livekit-agents/livekit/agents/voice/room_io/room_io.py`: `RoomIO` passes
  `output_audio_options.max_volume` when constructing `_ParticipantAudioOutput`.
- `livekit-agents/livekit/agents/voice/room_io/_output.py`: `_ParticipantAudioOutput`
  accepts `max_volume` (keyword, default 1.0), stores it, has a `_scale_volume(frame)`
  helper (imports numpy lazily), and applies it in `_forward_audio` immediately before
  `capture_frame`, after the first-frame / playback-started bookkeeping.

## Re-applying the patch

Plumb one float from the output options to the participant audio output and scale PCM
frames just before they are captured into the `rtc.AudioSource`.

## Upstream contracts relied upon

- `rtc.AudioFrame` with int16 PCM `data`, `sample_rate`, `num_channels`,
  `samples_per_channel`.
- `_ParticipantAudioOutput._forward_audio` being the single place frames reach the audio
  source.

## Conflict guidance

Additive and low-risk. If upstream removes the deprecated `RoomOutputOptions`, drop the
fork field there but keep it on `AudioOutputOptions`. If upstream restructures the
forward loop (e.g. resampling, AEC, buffering), apply scaling as the last step before
`capture_frame`.

## Verification after sync

- `max_volume` still reaches `_ParticipantAudioOutput` through both option paths.
- `_scale_volume` is still called in the forwarding loop.

## Known caveats

- Values above 1.0 are ignored (no amplification). Negative values are not validated.
- Scaling happens after `on_playback_started` and does not affect transcription sync.

## Drop criteria

Upstream exposes an output gain / volume option on room audio output.
