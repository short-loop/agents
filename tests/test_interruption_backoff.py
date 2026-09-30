"""fork(P9, SL-3890): session-level tests for the interruption-backoff modes."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    AgentStateChangedEvent,
    InterruptionBackoffOptions,
    InterruptionMode,
    InterruptionModeSettings,
)

from .fake_session import FakeActions, create_session, run_session
from .test_agent_session import SESSION_TIMEOUT, MyAgent, check_timestamp

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _backoff_options(speed: float, **overrides) -> InterruptionBackoffOptions:
    # silence_gate=0.0 keeps the playout hold a no-op: FakeVAD emits no INFERENCE_DONE
    # events after END_OF_SPEECH, so a positive gate would never reopen (real VADs emit
    # inference events continuously)
    transient = overrides.pop(
        "transient",
        InterruptionModeSettings(
            backoff_delay=2.0 / speed, unlikely_threshold=0.3, silence_gate=0.0
        ),
    )
    sustained = overrides.pop(
        "sustained",
        InterruptionModeSettings(
            backoff_delay=2.0 / speed,
            unlikely_threshold=0.5,
            silence_gate=0.0,
            disable_preemptive=True,
        ),
    )
    return InterruptionBackoffOptions(transient=transient, sustained=sustained, **overrides)


async def test_interruption_backoff_transient_entry() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s
    actions.add_user_speech(5.0, 6.0, "Stop for a moment.")  # interrupted at 5.5s
    actions.add_llm("Okay, I stopped the first time.")
    actions.add_tts(10.0)
    actions.add_user_speech(9.0, 10.0, "Stop once more.")  # interrupted at 9.5s
    actions.add_llm("Okay, I stopped again.")
    actions.add_tts(2.0)
    # transient mode entered at the 3rd user turn commit (2 interruptions in window);
    # no turn detector in fakes -> flat transient backoff_delay (2.0s) replaces the
    # 0.5s min delay for the next turn
    actions.add_user_speech(14.0, 15.0, "Please continue the story.")
    actions.add_llm("Continuing the story now.")
    actions.add_tts(1.0)

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"interruption_backoff": _backoff_options(speed)},
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)

    # the backoff-delayed commit fires ~2s after the last speech ends; keep the
    # session draining long enough for it to complete
    t_origin = await asyncio.wait_for(
        run_session(session, agent, drain_delay=1.5), timeout=SESSION_TIMEOUT
    )

    tracker = session._interruption_tracker
    assert tracker.total_interruptions == 2
    assert tracker.mode is InterruptionMode.TRANSIENT

    thinking_events = [ev for ev in agent_state_events if ev.new_state == "thinking"]
    assert len(thinking_events) == 4
    # compare each turn's endpointing delta (commit - end of user speech): normal-mode
    # turns use the min delay (0.5s), the transient-mode turn uses backoff_delay (2.0s)
    speech_ends = [2.5, 6.0, 10.0, 15.0]
    deltas = [
        (ev.created_at - t_origin) * speed - end
        for ev, end in zip(thinking_events, speech_ends, strict=True)
    ]
    for normal_delta in deltas[:3]:
        assert normal_delta <= 1.5, f"normal turn delta {normal_delta} unexpectedly large"
        assert deltas[3] >= normal_delta + 0.7, (
            f"transient turn delta {deltas[3]} not >= normal delta {normal_delta} + 0.7"
        )
    assert 1.6 <= deltas[3] <= 3.0, f"transient turn delta {deltas[3]} not near backoff_delay"


async def test_interruption_backoff_sustained_disables_preemptive() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing great, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=0.3)
    actions.add_user_speech(5.0, 6.0, "Wait, stop now.")  # interrupted at 5.5s
    actions.add_llm("Okay, stopping.", ttft=0.1, duration=0.3)
    actions.add_tts(0.5, ttfb=0.3)
    # sustained mode (entry_total=1) entered at the 2nd user turn commit
    actions.add_user_speech(8.5, 9.5, "Tell me more please.", stt_delay=0.2)
    actions.add_llm("Sure, here is more.", ttft=0.1, duration=1.0)
    actions.add_tts(1.0, ttfb=0.3)
    # sustained: commit at 9.5 + 2.0 (backoff) = 11.5; preemptive is disabled, so
    # generation starts at commit: speaking at 11.5 + 1.0 (llm) + 0.3 (ttfb) = 12.8.
    # if preemptive leaked, generation would overlap the backoff wait -> ~11.5

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"preemptive_generation": {"enabled": True}},
        extra_kwargs={
            "interruption_backoff": _backoff_options(speed, sustained_entry_total=1),
        },
    )
    agent = MyAgent()

    agent_state_events: list[AgentStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)

    t_origin = await asyncio.wait_for(
        run_session(session, agent, drain_delay=1.5), timeout=SESSION_TIMEOUT
    )

    tracker = session._interruption_tracker
    assert tracker.mode is InterruptionMode.SUSTAINED
    assert tracker.preemptive_disabled()

    speaking_events = [ev for ev in agent_state_events if ev.new_state == "speaking"]
    check_timestamp(
        speaking_events[-1].created_at - t_origin, 12.8, speed_factor=speed, max_abs_diff=0.3
    )


async def test_interruption_backoff_ignores_unspoken_replies() -> None:
    # an interrupted speech that never produced audio commits no assistant item
    # and must not count as an interruption
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Tell me a story", stt_delay=0.2)
    actions.add_llm("Here is a story for you...", ttft=0.1, duration=0.3)
    actions.add_tts(10.0, ttfb=1.0)  # first audio frame at ~3.8s
    actions.add_user_speech(2.6, 3.2, "about a firefighter.")  # interrupts before speaking
    actions.add_llm("Here is a story about a firefighter...", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.3)

    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={
            # entry_count=1 would enter transient on the first counted interruption
            "interruption_backoff": _backoff_options(speed, transient_entry_count=1),
        },
    )

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    tracker = session._interruption_tracker
    assert tracker.total_interruptions == 0
    assert tracker.mode is InterruptionMode.NORMAL


async def test_interruption_backoff_primed_after_single_interruption() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)
    actions.add_user_speech(5.0, 6.0, "Stop for a moment.")  # single barge-in
    actions.add_llm("Okay, I stopped.")
    actions.add_tts(1.0)

    session = create_session(
        actions,
        speed_factor=speed,
        # gate 0.0 keeps it a no-op for FakeVAD (no INFERENCE_DONE after end of speech)
        extra_kwargs={"interruption_backoff": _backoff_options(speed, primed_silence_gate=0.0)},
    )

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)

    tracker = session._interruption_tracker
    assert tracker.total_interruptions == 1
    # one interruption is below transient entry (2) but primes the gate, sticky
    assert tracker.mode is InterruptionMode.PRIMED
    assert tracker.silence_gate() == 0.0
    assert tracker.backoff_params() is None


def test_turn_handling_key_and_kwarg_alias() -> None:
    from livekit.agents import AgentSession

    opts = _backoff_options(1.0)
    assert AgentSession().options.interruption_backoff is None
    assert (
        AgentSession(turn_handling={"interruption_backoff": opts}).options.interruption_backoff
        is opts
    )
    assert AgentSession(interruption_backoff=opts).options.interruption_backoff is opts
