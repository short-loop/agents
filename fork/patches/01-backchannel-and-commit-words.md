# Patch 01 — Backchannel & commit words

| | |
|---|---|
| **Status** | Always-on (built-in default word list); word sets configurable via `InterruptionOptions` |
| **Origin** | 1.4.6: fork commit `d7081f0` (short-loop/agents#56). 1.8.3: `1ade102b0` "feat(voice): backchannel and commit words while the agent speaks (P4)" |
| **Depends on** | — |
| **Shares code with** | Patches 02/03/04 (`_run_eou_detection`), 05 |
| **Automated tests** | `tests/test_fork_recognition_rules.py` (word normalisation and the default list); the behaviour itself has no session-level test |
| **Code markers** | `fork(patch 01)` |

## Why

On phone calls, users constantly make short acknowledgement sounds while the agent is
talking ("mhm", "okay", "yeah", "uh"). With stock LiveKit behaviour in VAD interruption
mode, once VAD / STT pick these up they can (a) interrupt the agent's speech and (b) be
committed as a user turn, making the agent stop mid-sentence and respond to "mhm". The
fork treats a **single backchannel word spoken while the agent is speaking** as noise: it
neither interrupts the agent nor produces a user turn.

"Commit words" are the opposite escape hatch: a configurable set of single words that,
when spoken alone while the agent is talking, must be preserved in the conversation
history instead of being thrown away, without triggering a new reply.

Upstream's adaptive interruption (1.5+) solves the same problem with an audio model, but
it runs only on LiveKit Inference; self-hosted deployments resolve to `vad` mode, where
this patch is the only guard. It is applied in both modes (decision D2 in
`fork/MIGRATION-1.8.md`): the ML verdict can still say "interruption" for a loud "okay".

## Behaviour

- Applies **only while the agent is speaking** and **only when the current user transcript
  is exactly one word**.
- Word matching is case-insensitive and ignores all non-alphanumeric characters (e.g.
  "Mm-hmm." → "mmhmm", "uh-huh" → "uhhuh"). User-provided sets are normalised the same way
  at session start.
- Default backchannel set (used when the option is `None`) contains fillers and
  acknowledgements: the empty string, uh, um, ugh, uhh, oof, aye, hi, hello, okay, ok, yes,
  yeah, ya, sure, yep, yup, hm/hmm/hmmm/hmmmm, mm, mhm and its spelling variants, uhuh,
  uhhuh, huh, eh, ah, aah, aaah.
- **Commit words take precedence** over the backchannel list on both paths (a word in
  both sets is a commit word).
- Two points in the pipeline are affected:
  1. **Interruption:** `AgentActivity._interrupt_by_audio_activity` returns early when the
     session's agent state is "speaking" and the current transcript is a single
     backchannel word that is not a commit word. Upstream's `min_words` check follows,
     unchanged.
  2. **End-of-turn:** at the top of `AudioRecognition._run_eou_detection` (after
     upstream's "stt enabled but no transcript yet" return, before the chat context is
     copied), while the recognition's `_agent_speaking` flag is set and turn detection is
     not manual, for a single-word `_current_transcript`:
     - commit word → `hooks.on_commit_word(transcript)` stores it as a user turn without a
       reply, the pending transcript is cleared, no EOU runs (debug log "commit word
       detected, adding to context", field `lk.pii.word`);
     - backchannel word → the pending transcript is cleared and no EOU runs (debug log
       "backchannel word detected, ignoring").
- Configuration lives on upstream's interruption options:
  `turn_handling={"interruption": {"backchannel_words": {...} | None,
  "commit_words": {...} | None}}`. `None` (default) keeps the built-in list / disables
  commit words. Agent-level `turn_handling` overrides are not consulted (session only).

## Implementation walkthrough

### `livekit-agents/livekit/agents/voice/turn.py`

- `InterruptionOptions` gains `backchannel_words: set[str] | None` and
  `commit_words: set[str] | None`; `_INTERRUPTION_DEFAULTS` sets both to `None`.

### `livekit-agents/livekit/agents/voice/audio_recognition.py`

- Module level: `_STRIP_PATTERN` / `_strip_word` (lower-case, strip all non-word
  characters and underscores), `DEFAULT_BACKCHANNEL_WORDS`, pre-normalised
  `_STRIPPED_BACKCHANNEL_WORDS`. `re` is imported for this.
- `RecognitionHooks` protocol: new method `on_commit_word(transcript: str)`.
- `AudioRecognition.__init__`: reads both sets from `session.options.interruption`
  (next to upstream's `backchannel_boundary` read) into `_backchannel_words` /
  `_commit_words`, normalised.
- Methods `is_backchannel_word(word)` and `is_commit_word(word)`.
- `_run_eou_detection`: the crutch-word block described above. It uses upstream's
  `_agent_speaking` (maintained by `_on_start_of_agent_speech` /
  `_on_end_of_agent_speech` on every playout start and end, in every interruption mode).

### `livekit-agents/livekit/agents/voice/agent_activity.py`

- `_interrupt_by_audio_activity`: the early-return block inserted **before** upstream's
  `min_words` block, which stays byte-identical (it reads
  `self._audio_recognition._current_transcript`).
- `on_commit_word(transcript)`: builds a user `ChatMessage`, appends it to
  `self._agent._chat_ctx.items` and calls `self._session._conversation_item_added(...)` —
  the same two steps upstream performs for a `skip_reply` user turn, so the session
  history, the `conversation_item_added` event and patch 04's user-turn counter all see it.

## Re-applying the patch

1. Keep a normalised backchannel vocabulary and an optional commit vocabulary as keys of
   `InterruptionOptions`, read by `AudioRecognition` at construction.
2. In the path where audio activity would interrupt the agent, bail out for a single
   backchannel word (that is not a commit word) while the agent is speaking. Keep the
   `min_words` semantics intact.
3. In EOU detection, before building the user message for the turn detector, persist a
   lone commit word through a hook that stores it exactly like a no-reply user turn, or
   drop a lone backchannel word (clear the pending transcript).

## Upstream contracts relied upon

- `AgentSession.agent_state` values ("speaking") and `AudioRecognition._agent_speaking`.
- `AudioRecognition._current_transcript` (final + interim) and `_audio_transcript` (the
  accumulated final transcript that EOU commits).
- `session.options.interruption` being the resolved `InterruptionOptions` dict.
- How upstream stores a `skip_reply` user turn (`_agent._chat_ctx.items.append` +
  `_session._conversation_item_added`), mirrored by `on_commit_word`. `Agent.chat_ctx`
  is a read-only view in 1.8, which is why the hook exists.
- `split_words(text, split_character=True)` returning `(word, start, end)` tuples.

## Conflict guidance

- `_interrupt_by_audio_activity` is edited upstream fairly often (interruption handling,
  realtime models, adaptive interruption). Keep upstream's new conditions and re-insert
  the early-return block right before the `min_words` check.
- In `_run_eou_detection`, the fork block must stay **after** upstream's early returns
  and **before** the chat context copy.
- `RecognitionHooks`: if upstream adds methods, keep both. If upstream adds another
  implementer of the protocol, it must implement `on_commit_word`.
- `InterruptionOptions`: keep the two fork keys and their `None` defaults.

## Verification after sync

- The two keys exist on `InterruptionOptions` and `_INTERRUPTION_DEFAULTS`, and reach
  `AudioRecognition`.
- `on_commit_word` exists on `AgentActivity` and in `RecognitionHooks`.
- The early-return blocks exist in both `_interrupt_by_audio_activity` and
  `_run_eou_detection`.
- `tests/test_fork_recognition_rules.py` passes; type check passes (protocol conformance).

## Known caveats (documented, not bugs to fix during sync)

- The default set includes "yes", "hi", "hello", "okay", "sure" — a single "yes" while the
  agent is speaking is intentionally ignored. Configure `commit_words` to preserve
  specific words.
- The EOU check reads `_current_transcript` (final + interim) to count words but clears
  only the final transcript buffer.
- In VAD mode the interruption check runs when VAD fires, often before the transcript
  exists; a backchannel can still interrupt if STT is slower than VAD (same as 1.4.6).
- Commit words bypass `on_user_turn_completed` and the turn-detector path entirely.

## Drop criteria

Upstream ships built-in backchannel / filler suppression usable **without LiveKit
Inference** for both interruption and turn commit (with configurable vocabulary) that
covers the single-word-while-speaking case. Even then, commit-word persistence would need
an equivalent or remain as a smaller patch. Also see patch 14 (own interruption detector,
planned) in `fork/MIGRATION-1.8.md` §3.
