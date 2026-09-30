# Patch 01 — Backchannel & commit words

| | |
|---|---|
| **Status** | Always-on (built-in default word list); word sets configurable |
| **Origin** | Fork commit `d7081f0` (short-loop/agents#56, "add: backchannel words…", "add: … commit persistence", "fix: is_bot_speaking function", "fix: use current_transcript instead") |
| **Depends on** | — |
| **Shares code with** | Patch 04 (`RecognitionHooks` protocol additions) |
| **Automated tests** | None |

## Why

On phone calls, users constantly make short acknowledgement sounds while the agent is
talking ("mhm", "okay", "yeah", "uh"). With stock LiveKit behaviour, once VAD / STT pick
these up they can (a) interrupt the agent's speech and (b) be committed as a user turn,
making the agent stop mid-sentence and respond to "mhm". The fork treats a **single
backchannel word spoken while the agent is speaking** as noise: it neither interrupts the
agent nor produces a user turn.

"Commit words" are the opposite escape hatch: a configurable set of single words that,
when spoken alone while the agent is talking, must be preserved in the conversation
history (added to the agent's chat context) instead of being thrown away, without
triggering a new reply.

## Behaviour

- Applies **only while the agent is speaking** (`AgentSession.agent_state == "speaking"`)
  and **only when the current user transcript is exactly one word**.
- Word matching is case-insensitive and ignores all non-alphanumeric characters (e.g.
  "Mm-hmm." → "mmhmm", "uh-huh" → "uhhuh").
- Default backchannel set (used when the session does not provide one) contains fillers
  and acknowledgements: the empty string, uh, um, ugh, uhh, oof, aye, hi, hello, okay, ok,
  yes, yeah, ya, sure, yep, yup, hm/hmm/hmmm/hmmmm, mm, mhm and its spelling variants,
  uhuh, uhhuh, huh, eh, ah, aah, aaah.
- Two points in the pipeline are affected:
  1. **Interruption:** audio-activity-driven interruption of the agent is suppressed
     when the single word is a backchannel word and not a commit word.
  2. **End-of-turn:** when EOU detection is about to run for a single-word transcript:
     - backchannel word → the accumulated user transcript is cleared and no EOU / user
       turn happens (debug log "backchannel word detected, ignoring");
     - commit word → a user message with the transcript is appended directly to the
       agent's chat context, the transcript is cleared and no reply is generated (debug
       log "commit word detected, adding to context").
- New `AgentSession` constructor kwargs:
  - `backchannel_words` (set of strings, optional) — **replaces** the default set.
  - `commit_words` (set of strings, optional) — empty/None by default (feature off).
  Both are stored on `AgentSessionOptions` as `backchannel_words` / `commit_words`
  (None when not given).

### Side-effect on `min_interruption_words`

Upstream only evaluated the transcript inside `_interrupt_by_audio_activity` when
`min_interruption_words > 0`. The fork evaluates it whenever an STT and audio recognition
exist (needed for the backchannel check) and moves the `min_interruption_words > 0`
condition inside. Net behaviour for `min_interruption_words` is unchanged.

## Implementation walkthrough

### `livekit-agents/livekit/agents/voice/audio_recognition.py`

- Module level: a regex-based word normaliser (`_strip_word`, lower-case, strip all
  non-word characters and underscores), the `DEFAULT_BACKCHANNEL_WORDS` set, and its
  pre-normalised copy `_STRIPPED_BACKCHANNEL_WORDS`. `re` is imported for this.
- `RecognitionHooks` protocol: new method `is_bot_speaking()`.
- `AudioRecognition.__init__`: new optional parameters `backchannel_words` and
  `commit_words`, stored as private attributes.
- New methods `is_backchannel_word(word)` and `is_commit_word(word)`. The backchannel
  check uses the session-provided set if non-empty, else the default set. The commit
  check returns False when no commit words are configured.
- `_run_eou_detection`: right after the "stt enabled but no transcript yet" early return
  and **before** the chat context is copied and the user message is appended, a block
  checks `hooks.is_bot_speaking()` and a single-word `current_transcript`, then applies the
  backchannel / commit behaviour described above and returns early.

### `livekit-agents/livekit/agents/voice/agent_activity.py`

- New method `AgentActivity.is_bot_speaking()` (implements the hook) — true when the
  session's agent state is "speaking".
- `_start_session` (where `AudioRecognition` is constructed): passes
  `backchannel_words` and `commit_words` from the session options.
- `_interrupt_by_audio_activity`: restructured condition (see side-effect above); splits
  `current_transcript` into words once (`split_words(..., split_character=True)`) and
  returns early for a lone backchannel word while the bot is speaking.

### `livekit-agents/livekit/agents/voice/agent_session.py`

- `AgentSessionOptions`: fields `backchannel_words`, `commit_words`.
- `AgentSession.__init__`: kwargs `backchannel_words`, `commit_words` (NotGivenOr),
  stored as None when not given.

## Re-applying the patch

If upstream reshaped this code, re-implement the intent:

1. Keep a normalised backchannel vocabulary and an optional commit vocabulary, both
   configurable on `AgentSession` and plumbed into `AudioRecognition`.
2. Expose "is the agent currently speaking" to `AudioRecognition` via the hooks object.
3. In the path where audio activity would interrupt the agent, bail out for a single
   backchannel word (that is not a commit word) while the agent is speaking. Keep the
   `min_interruption_words` semantics intact.
4. In EOU detection, before building the user message for the turn detector, drop a lone
   backchannel word (clear the pending transcript), or persist a lone commit word to the
   agent chat context without replying.

## Upstream contracts relied upon

- `AgentSession.agent_state` values ("speaking").
- `AudioRecognition.current_transcript` and `_audio_transcript` (the accumulated final
  transcript that EOU commits).
- `RecognitionHooks.retrieve_chat_ctx()` returning the **live** agent chat context
  (`Agent.chat_ctx`), not a copy — the commit-word branch appends to its `items`.
- `split_words(text, split_character=True)` returning `(word, start, end)` tuples.

## Conflict guidance

- `_interrupt_by_audio_activity` is edited upstream fairly often (interruption handling,
  realtime models, adaptive interruption). Keep upstream's new conditions and re-insert
  the single-word backchannel early-return right after the transcript is read.
- In `_run_eou_detection`, the fork block must stay **after** upstream's early returns
  and **before** the user message is added to the copied chat context.
- `AudioRecognition.__init__` signature: upstream adds parameters here; keep the fork's
  two keyword parameters at the end with defaults of None.
- `RecognitionHooks`: if upstream adds methods, keep both. If upstream adds another
  implementer of the protocol, it must implement `is_bot_speaking` (and
  `interruption_mode` from patch 04).

## Verification after sync

- `AgentSession` still accepts `backchannel_words` / `commit_words` and they reach
  `AudioRecognition` (search for the constructor call in `AgentActivity`).
- `is_bot_speaking` exists on `AgentActivity` and in `RecognitionHooks`.
- The early-return blocks exist in both `_interrupt_by_audio_activity` and
  `_run_eou_detection`.
- Type check passes (protocol conformance of `AgentActivity`).

## Known caveats (documented, not bugs to fix during sync)

- A user-provided `backchannel_words` set is **not** normalised; entries must already be
  lower-case with no punctuation. Commit words likewise.
- The default set includes "yes", "hi", "hello", "okay", "sure" — a single "yes" while the
  agent is speaking is intentionally ignored. Configure `commit_words` to preserve
  specific words.
- The commit-word branch mutates the agent chat context directly; the code comment notes
  this may not work with realtime models.
- The EOU check reads `current_transcript` (final + interim) to count words but clears
  only the final transcript buffer.

## Drop criteria

Upstream ships built-in backchannel / filler suppression for both interruption and turn
commit (with configurable vocabulary) that covers the single-word-while-speaking case.
Even then, commit-word persistence would need an equivalent or remain as a smaller patch.
