# Patch 06 — Last user language accessor and shorter language-detection minimum

| | |
|---|---|
| **Status** | Always-on |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56, "chore: migrate sl patches") |
| **Depends on** | — |
| **Automated tests** | None |

## Why

Application code (ShortLoop agents) needs to know which language the user last spoke in,
to pick the TTS voice / prompt language for the reply. Upstream tracks it privately in
`AudioRecognition._last_language` (used for the turn detector) but exposes no accessor.
Upstream also ignores the STT-reported language for transcripts of 5 characters or
fewer, which dropped the language of short replies ("sí", "haan", "नहीं").

## Behaviour

- New read-only **properties** (despite the `get_` prefix they are properties, not
  methods) named `get_last_user_language`, returning a `LanguageCode` or None:
  - `AgentSession.get_last_user_language` → current activity's value, None when there is
    no activity;
  - `AgentActivity.get_last_user_language` → audio recognition's value, None when audio
    recognition is not running;
  - `AudioRecognition.get_last_user_language` → `_last_language`.
- `MIN_LANGUAGE_DETECTION_LENGTH` lowered from 5 to 3: a transcript longer than 3
  characters (instead of 5) with a reported language updates `_last_language`. This also
  affects the language passed to the turn detector (`supports_language`,
  `unlikely_threshold`) and the `ATTR_EOU_LANGUAGE` trace attribute.

## Implementation walkthrough

- `livekit-agents/livekit/agents/voice/audio_recognition.py`: constant change; property
  after `current_transcript`.
- `livekit-agents/livekit/agents/voice/agent_activity.py`: imports `LanguageCode`;
  property next to `current_speech`.
- `livekit-agents/livekit/agents/voice/agent_session.py`: imports `LanguageCode`;
  property after `tools`.

## Re-applying the patch

Expose upstream's "last detected user language" through the session → activity →
recognition chain as properties with these exact names (application code depends on the
names), and keep the minimum-length constant at 3.

## Upstream contracts relied upon

- `AudioRecognition._last_language`, updated from FINAL and PREFLIGHT transcripts.
- `AgentSession._activity`, `AgentActivity._audio_recognition`.

## Conflict guidance

Additive; conflicts are mechanical. If upstream adds its own public accessor, keep the
fork names as thin aliases rather than breaking application code.

## Verification after sync

- The three properties exist and the constant is 3.
- `_last_language` is still the variable upstream updates on transcripts (if upstream
  renamed it, re-point the property).

## Known caveats

- The `get_` prefix on a property is unusual; do not "fix" it to a method — callers
  access it as an attribute.

## Drop criteria

Upstream exposes the last user language publicly **and** application code has migrated.
The threshold change must be evaluated separately.
