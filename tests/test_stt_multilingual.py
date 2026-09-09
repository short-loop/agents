from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.language import LanguageCode
from livekit.agents.stt import (
    STT,
    LanguageNotAllowedError,
    LanguageSwitchFailedError,
    LanguageSwitchOptions,
    MultilingualAdapter,
    RecognitionUsage,
    SpeechEvent,
    SpeechEventType,
)

from .fake_multilingual_stt import ScriptedStream, ScriptedSTT, make_frame

FAST_OPTIONS = LanguageSwitchOptions(
    switch_grace_s=0.1,
    boundary_silence_s=0.05,
    max_detector_owns_s=2.0,
    switch_timeout_s=0.5,
    hard_cooldown_s=1.0,
    manual_cooldown_s=1.0,
)


class Harness:
    """Drives a MultilingualAdapter stream: collects output + adapter events, keeps the
    audio clock advancing with a background frame pusher."""

    def __init__(self, adapter: MultilingualAdapter):
        self.adapter = adapter
        self.stream = adapter.stream()
        self.events: list[SpeechEvent] = []
        self.adapter_events: dict[str, list[Any]] = {
            "language_switch_started": [],
            "language_switched": [],
            "language_switch_suppressed": [],
            "language_switch_evidence": [],
        }
        for name in self.adapter_events:
            adapter.on(name, self._recorder(name))  # type: ignore[arg-type]

        self._collect_task = asyncio.create_task(self._collect())
        self._pusher_task: asyncio.Task[None] | None = None

    def _recorder(self, name: str) -> Callable[[Any], None]:
        def _on_event(ev: Any) -> None:
            self.adapter_events[name].append(ev)

        return _on_event

    async def _collect(self) -> None:
        async for ev in self.stream:
            self.events.append(ev)

    def start_pushing_audio(self) -> None:
        async def _push() -> None:
            while True:
                with contextlib.suppress(RuntimeError):
                    self.stream.push_frame(make_frame(duration_ms=10))
                await asyncio.sleep(0.002)

        self._pusher_task = asyncio.create_task(_push())

    def transcripts(self) -> list[tuple[SpeechEventType, str]]:
        return [
            (ev.type, ev.alternatives[0].text)
            for ev in self.events
            if ev.type
            in (
                SpeechEventType.INTERIM_TRANSCRIPT,
                SpeechEventType.PREFLIGHT_TRANSCRIPT,
                SpeechEventType.FINAL_TRANSCRIPT,
            )
            and ev.alternatives
        ]

    def texts(self) -> list[str]:
        return [text for _, text in self.transcripts() if text]

    async def wait_for(self, predicate: Callable[[], bool], timeout: float = 2.0) -> None:
        async def _poll() -> None:
            while not predicate():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_poll(), timeout)

    async def aclose(self) -> None:
        if self._pusher_task is not None:
            self._pusher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pusher_task
        await self.stream.aclose()
        with contextlib.suppress(asyncio.CancelledError):
            await self._collect_task
        await self.adapter.aclose()


def make_adapter(
    *,
    supports_update: bool = True,
    with_factory: bool = False,
    options: LanguageSwitchOptions | None = None,
) -> tuple[MultilingualAdapter, ScriptedSTT, ScriptedSTT, list[ScriptedSTT]]:
    primary = ScriptedSTT(label="primary", supports_update=supports_update)
    detector = ScriptedSTT(label="detector")
    factory_created: list[ScriptedSTT] = []

    def factory(language: LanguageCode) -> STT:
        stt_instance = ScriptedSTT(label=f"factory-{language}")
        factory_created.append(stt_instance)
        return stt_instance

    adapter = MultilingualAdapter(
        primary=primary,
        detector=detector,
        primary_factory=factory if with_factory else None,
        initial_language="en",
        options=options or FAST_OPTIONS,
    )
    return adapter, primary, detector, factory_created


async def test_passthrough_parity() -> None:
    adapter, primary, detector, _ = make_adapter()
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()
    assert isinstance(primary_stream, ScriptedStream)

    harness.stream.push_frame(make_frame())
    await harness.wait_for(lambda: len(primary_stream.received_frames) >= 1)
    await harness.wait_for(lambda: len(detector_stream.received_frames) >= 1)

    primary_stream.send_transcript("hola", event_type=SpeechEventType.INTERIM_TRANSCRIPT)
    primary_stream.send_transcript("hello world", language="en", end_time=1.0)
    detector_stream.send_transcript("should not appear", language="en", end_time=1.0)
    detector_stream.send_event(
        SpeechEvent(
            type=SpeechEventType.RECOGNITION_USAGE,
            recognition_usage=RecognitionUsage(audio_duration=1.0),
        )
    )

    await harness.wait_for(
        lambda: any(ev.type == SpeechEventType.RECOGNITION_USAGE for ev in harness.events)
    )

    assert harness.texts() == ["hola", "hello world"]
    usage_events = [ev for ev in harness.events if ev.type == SpeechEventType.RECOGNITION_USAGE]
    assert len(usage_events) == 1  # detector usage is forwarded

    await harness.aclose()


async def test_detector_death_keeps_primary() -> None:
    adapter, primary, detector, _ = make_adapter()
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()

    errors: list[Any] = []
    adapter.on("error", errors.append)

    detector_stream.fail(APIConnectionError("detector died"))

    # detector restarts with backoff; primary keeps flowing in the meantime
    primary_stream.send_transcript("still alive", language="en")
    await harness.wait_for(lambda: "still alive" in harness.texts())
    await harness.wait_for(lambda: len(errors) >= 1)
    assert errors[0].recoverable

    new_detector_stream = await detector.wait_for_stream(timeout=5.0)
    assert new_detector_stream is not detector_stream

    primary_stream.send_transcript("after restart", language="en")
    await harness.wait_for(lambda: "after restart" in harness.texts())

    await harness.aclose()


async def test_manual_switch_in_place() -> None:
    adapter, primary, detector, _ = make_adapter(supports_update=True)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    await detector.wait_for_stream()
    harness.start_pushing_audio()

    await adapter.switch_language("hi")

    assert primary_stream.update_calls == ["hi"]  # type: ignore[attr-defined]
    assert adapter.current_language == "hi"
    assert len(harness.adapter_events["language_switch_started"]) == 1
    assert len(harness.adapter_events["language_switched"]) == 1

    switched = harness.adapter_events["language_switched"][0]
    assert switched.old_language == "en"
    assert switched.new_language == "hi"
    assert switched.initiator == "manual"
    assert switched.executor == "in_place"

    # started event must precede completion
    started = harness.adapter_events["language_switch_started"][0]
    assert started.new_language == "hi"

    # after flip 2 the primary owns events again
    primary_stream.send_transcript("नमस्ते", language="hi")
    await harness.wait_for(lambda: "नमस्ते" in harness.texts())

    await harness.aclose()


async def test_manual_switch_same_language_noop() -> None:
    adapter, primary, detector, _ = make_adapter()
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    await detector.wait_for_stream()

    await adapter.switch_language("en")

    assert primary_stream.update_calls == []  # type: ignore[attr-defined]
    assert harness.adapter_events["language_switch_started"] == []
    assert harness.adapter_events["language_switched"] == []

    await harness.aclose()


async def test_manual_switch_recreate() -> None:
    adapter, primary, detector, factory_created = make_adapter(
        supports_update=False, with_factory=True
    )
    harness = Harness(adapter)

    old_primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()
    harness.start_pushing_audio()

    switch_task = asyncio.create_task(adapter.switch_language("hi"))

    # the factory creates a new STT pinned to the target language
    await harness.wait_for(lambda: len(factory_created) == 1)
    shadow_stream = await factory_created[0].wait_for_stream()
    assert shadow_stream.language == "hi"

    # the shadow receives live audio while the old primary is still up
    await harness.wait_for(lambda: len(shadow_stream.received_frames) >= 1)

    await asyncio.wait_for(switch_task, 5.0)
    assert adapter.current_language == "hi"
    switched = harness.adapter_events["language_switched"][0]
    assert switched.executor == "recreate"

    # late events from the retired primary stream are dropped
    before = len(harness.texts())
    with contextlib.suppress(Exception):
        old_primary_stream.send_transcript("stale text", language="en")
    shadow_stream.send_transcript("नई भाषा में", language="hi")
    await harness.wait_for(lambda: "नई भाषा में" in harness.texts())
    assert "stale text" not in harness.texts()
    assert len(harness.texts()) == before + 1

    # detector keeps running across the switch
    detector_stream.send_transcript("अभी भी चालू", language="hi", end_time=100.0)
    await asyncio.sleep(0.05)

    await harness.aclose()


async def test_detector_owns_window_forwards_detector() -> None:
    adapter, primary, detector, _ = make_adapter(supports_update=True)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()

    # a committed primary final sets the dedup gate
    primary_stream.send_transcript("committed english", language="en", start_time=0.0, end_time=5.0)
    await harness.wait_for(lambda: "committed english" in harness.texts())

    switch_task = asyncio.create_task(adapter.switch_language("hi"))
    await harness.wait_for(lambda: len(harness.adapter_events["language_switch_started"]) == 1)

    # pre-flip audio: deduped; post-flip audio: forwarded from the detector
    detector_stream.send_transcript("old audio", language="hi", start_time=4.0, end_time=4.9)
    detector_stream.send_transcript("नया वाक्य", language="hi", start_time=5.5, end_time=6.5)
    await harness.wait_for(lambda: "नया वाक्य" in harness.texts())
    assert "old audio" not in harness.texts()

    # detector interims are forwarded during the window
    detector_stream.send_transcript(
        "आधा वाक्य", language="hi", event_type=SpeechEventType.INTERIM_TRANSCRIPT
    )
    await harness.wait_for(lambda: "आधा वाक्य" in harness.texts())

    # END_OF_SPEECH from the detector closes the window (after the health grace period)
    await asyncio.sleep(0.15)
    detector_stream.send_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
    await asyncio.wait_for(switch_task, 5.0)

    # the flip emitted clearing events: an empty interim + empty final
    empty_interims = [
        ev
        for ev in harness.events
        if ev.type == SpeechEventType.INTERIM_TRANSCRIPT
        and ev.alternatives
        and not ev.alternatives[0].text
    ]
    assert len(empty_interims) >= 2  # one per flip

    # after flip 2, detector transcripts are silent again
    before = len(harness.texts())
    detector_stream.send_transcript("मौन", language="hi", start_time=7.0, end_time=8.0)
    primary_stream.send_transcript("primary owns", language="hi", start_time=8.0, end_time=9.0)
    await harness.wait_for(lambda: "primary owns" in harness.texts())
    assert "मौन" not in harness.texts()
    assert len(harness.texts()) == before + 1

    await harness.aclose()


async def test_shadow_failure_rolls_back() -> None:
    adapter, primary, detector, factory_created = make_adapter(
        supports_update=False, with_factory=True
    )
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    await detector.wait_for_stream()

    switch_task = asyncio.create_task(adapter.switch_language("hi"))
    await harness.wait_for(lambda: len(factory_created) == 1)
    shadow_stream = await factory_created[0].wait_for_stream()
    shadow_stream.fail(APIConnectionError("bad language"))

    with pytest.raises(LanguageSwitchFailedError):
        await asyncio.wait_for(switch_task, 5.0)

    assert adapter.current_language == "en"
    suppressed = harness.adapter_events["language_switch_suppressed"]
    assert any(ev.reason == "switch_failed" for ev in suppressed)
    assert harness.adapter_events["language_switched"] == []

    # primary is untouched and still owns the stream
    primary_stream.send_transcript("still english", language="en")
    await harness.wait_for(lambda: "still english" in harness.texts())

    await harness.aclose()


async def test_heuristic_switch_end_to_end() -> None:
    options = LanguageSwitchOptions(
        switch_grace_s=0.1,
        boundary_silence_s=0.05,
        max_detector_owns_s=2.0,
        switch_timeout_s=0.5,
        switch_threshold=1.0,
        min_detector_confidence=0.5,
    )
    adapter, primary, detector, _ = make_adapter(supports_update=True, options=options)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()
    harness.start_pushing_audio()

    # one long, confident, cross-script utterance crosses threshold 1.0 alone:
    # length weight 1.0 (8 words) * confidence 0.9 * script boost 1.5 = 1.35
    detector_stream.send_transcript(
        "नमस्ते आप कैसे हैं मुझे मदद चाहिए धन्यवाद",
        language="hi",
        confidence=0.9,
        start_time=1.0,
        end_time=3.0,
    )

    await harness.wait_for(lambda: len(harness.adapter_events["language_switched"]) == 1, 5.0)

    switched = harness.adapter_events["language_switched"][0]
    assert switched.initiator == "heuristic"
    assert switched.new_language == "hi"
    assert switched.trigger_transcript is not None
    assert primary_stream.update_calls == ["hi"]  # type: ignore[attr-defined]
    assert adapter.current_language == "hi"

    await harness.aclose()


async def test_trigger_utterance_replayed_at_flip() -> None:
    # regression: the pinned primary garbles a foreign utterance into interims and never
    # finalizes it; the detector final that triggered the switch is consumed while the
    # primary still owns the stream. It must be replayed at flip 1 — otherwise the
    # utterance vanishes and the session commits an empty user turn.
    options = LanguageSwitchOptions(
        switch_grace_s=0.1,
        boundary_silence_s=0.05,
        max_detector_owns_s=2.0,
        switch_timeout_s=0.5,
        switch_threshold=1.0,
        min_detector_confidence=0.5,
    )
    adapter, primary, detector, _ = make_adapter(supports_update=True, options=options)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()
    harness.start_pushing_audio()

    # a committed english final sets the forwarded gate at 5.0
    primary_stream.send_transcript("hello there", language="en", start_time=0.0, end_time=5.0)
    detector_stream.send_transcript("hello there", language="en", start_time=0.0, end_time=5.0)
    await harness.wait_for(lambda: "hello there" in harness.texts())

    # the primary only manages a garbled interim for the hindi utterance — no final
    primary_stream.send_transcript(
        "kia upload the data center",
        language="en",
        event_type=SpeechEventType.INTERIM_TRANSCRIPT,
        start_time=6.0,
        end_time=8.0,
    )
    trigger = "क्या आप लोग टोयोटा की गाड़ियां सर्विस करते हैं"
    detector_stream.send_transcript(
        trigger, language="hi", confidence=0.9, start_time=6.0, end_time=8.0
    )

    await harness.wait_for(lambda: len(harness.adapter_events["language_switched"]) == 1, 5.0)

    # the trigger utterance was replayed to the session as a FINAL
    finals = [t for ty, t in harness.transcripts() if ty == SpeechEventType.FINAL_TRANSCRIPT]
    assert trigger in finals
    # ...but the already-committed english final was not duplicated from the detector
    assert harness.texts().count("hello there") == 1

    await harness.aclose()


async def test_heuristic_suppressed_during_switch() -> None:
    adapter, primary, detector, _ = make_adapter(supports_update=True)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    detector_stream = await detector.wait_for_stream()

    switch_task = asyncio.create_task(adapter.switch_language("hi"))
    await harness.wait_for(lambda: len(harness.adapter_events["language_switch_started"]) == 1)

    # strong evidence for a third language mid-switch must not start a nested switch
    for i in range(5):
        detector_stream.send_transcript(
            "hola como estas amigo muy bien gracias por todo",
            language="es",
            confidence=0.95,
            start_time=float(i),
            end_time=float(i) + 0.9,
        )

    await asyncio.sleep(0.15)
    detector_stream.send_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
    await asyncio.wait_for(switch_task, 5.0)

    assert adapter.current_language == "hi"
    assert len(harness.adapter_events["language_switched"]) == 1
    assert primary_stream.update_calls == ["hi"]  # type: ignore[attr-defined]

    await harness.aclose()


async def test_allowlist_manual_raises() -> None:
    options = LanguageSwitchOptions(languages=["en", "hi"])
    adapter, primary, detector, _ = make_adapter(options=options)
    harness = Harness(adapter)

    await primary.wait_for_stream()
    await detector.wait_for_stream()

    with pytest.raises(LanguageNotAllowedError) as exc_info:
        await adapter.switch_language("ta")
    assert exc_info.value.language == "ta"

    await harness.aclose()


async def test_allowlist_rejected_at_init() -> None:
    with pytest.raises(LanguageNotAllowedError):
        MultilingualAdapter(
            primary=ScriptedSTT(),
            detector=ScriptedSTT(),
            initial_language="fr",
            options=LanguageSwitchOptions(languages=["en", "hi"]),
        )


async def test_end_input_drains_cleanly() -> None:
    adapter, primary, detector, _ = make_adapter()
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    await detector.wait_for_stream()

    harness.stream.push_frame(make_frame())
    primary_stream.send_transcript("goodbye", language="en")
    await harness.wait_for(lambda: "goodbye" in harness.texts())

    harness.stream.end_input()
    await asyncio.wait_for(harness._collect_task, 5.0)

    await harness.aclose()


async def test_no_executor_manual_switch_fails() -> None:
    # primary without update_options and no factory: switches must fail cleanly
    adapter, primary, detector, _ = make_adapter(supports_update=False, with_factory=False)
    harness = Harness(adapter)

    primary_stream = await primary.wait_for_stream()
    await detector.wait_for_stream()

    with pytest.raises(LanguageSwitchFailedError):
        await adapter.switch_language("hi")

    assert adapter.current_language == "en"
    primary_stream.send_transcript("still works", language="en")
    await harness.wait_for(lambda: "still works" in harness.texts())

    await harness.aclose()


async def test_switch_language_without_live_stream() -> None:
    adapter, _, _, _ = make_adapter()
    await adapter.switch_language("hi")
    assert adapter.current_language == "hi"
    await adapter.aclose()
