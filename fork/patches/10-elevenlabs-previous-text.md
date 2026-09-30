# Patch 10 — ElevenLabs `previous_text` priming

| | |
|---|---|
| **Status** | Always-on for the ElevenLabs TTS plugin (not configurable) |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56, "chore: migrate sl patches") |
| **Depends on** | — |
| **Automated tests** | None |

## Why

ElevenLabs uses `previous_text` as context that conditions prosody without being spoken.
Priming every request with a fixed soft-spoken narrative prefix makes the voice delivery
calmer and more consistent across short, independent utterances on calls.

## Behaviour

The fixed string **"And she softly spoke : "** is sent as `previous_text` on:

1. the HTTP (non-streaming) synthesis request body in `ChunkedStream._run`, next to
   `text`, `model_id` and `voice_settings`;
2. the WebSocket context-initialisation packet in `_Connection._send_loop` (the packet with
   `"text": " "`, `voice_settings` and `context_id`, sent when a new context starts).

Every ElevenLabs synthesis through this fork therefore carries the prefix, regardless of
voice or language.

## Implementation walkthrough

`livekit-plugins/livekit-plugins-elevenlabs/livekit/plugins/elevenlabs/tts.py`: one
dictionary key added in each of the two payloads.

## Re-applying the patch

Add the same `previous_text` value to every request payload that starts a synthesis
(HTTP body and WebSocket context init). If upstream adds a new synthesis path (e.g. a new
endpoint), add it there too.

## Upstream contracts relied upon

The ElevenLabs API accepting `previous_text` on both endpoints.

## Conflict guidance

If upstream adds its own `previous_text` / `next_text` option to the plugin, **do not end
up sending the key twice or overriding a user-provided value unknowingly**: prefer the
upstream option and set the fork default through it (escalate for a decision).

## Verification after sync

Both payloads still include `previous_text` with the exact string (including the space
before the colon and the trailing space).

## Known caveats

- The prefix is English and gendered ("she"); it is applied even for non-English or
  male voices.
- With multilingual output this may influence pronunciation of the first words.

## Drop criteria

Product decision to stop priming, or migration to an upstream plugin option that sets it.
