from __future__ import annotations

import asyncio
import contextlib

from livekit import rtc
from livekit.agents import NOT_GIVEN, APIConnectionError, LanguageCode, NotGivenOr, utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.audio import AudioBuffer


class ScriptedStream(RecognizeStream):
    """A stream that stays open and emits exactly the events the test injects."""

    def __init__(self, *, stt: ScriptedSTT, conn_options: APIConnectOptions, language: str | None):
        super().__init__(stt=stt, conn_options=conn_options)
        self.language = language
        self.received_frames: list[rtc.AudioFrame] = []
        self.received_flushes = 0
        self._fail_fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def send_event(self, ev: SpeechEvent) -> None:
        self._event_ch.send_nowait(ev)

    def send_transcript(
        self,
        text: str,
        *,
        language: str = "",
        event_type: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT,
        confidence: float = 1.0,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        self.send_event(
            SpeechEvent(
                type=event_type,
                alternatives=[
                    SpeechData(
                        language=LanguageCode(language),
                        text=text,
                        confidence=confidence,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ],
            )
        )

    def fail(self, exc: Exception) -> None:
        if not self._fail_fut.done():
            self._fail_fut.set_exception(exc)

    async def _run(self) -> None:
        async def _consume_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    self.received_frames.append(data)
                elif isinstance(data, self._FlushSentinel):
                    self.received_flushes += 1

        input_task = asyncio.create_task(_consume_input())
        try:
            done, _ = await asyncio.wait(
                {input_task, self._fail_fut}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._fail_fut in done:
                self._fail_fut.result()  # raises the injected exception
        finally:
            input_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await input_task
            if not self._fail_fut.done():
                self._fail_fut.cancel()


class SwitchableScriptedStream(ScriptedStream):
    """ScriptedStream with a Deepgram-shaped update_options(language=...)."""

    def __init__(self, *, stt: ScriptedSTT, conn_options: APIConnectOptions, language: str | None):
        super().__init__(stt=stt, conn_options=conn_options, language=language)
        self.update_calls: list[str] = []

    def update_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> None:
        if utils.is_given(language):
            self.update_calls.append(str(language))
            self.language = str(language)


class ScriptedSTT(STT):
    def __init__(self, *, label: str = "scripted", supports_update: bool = False) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self._test_label = label
        self._supports_update = supports_update
        self.created_streams: list[ScriptedStream] = []
        self._stream_ch = utils.aio.Chan[ScriptedStream]()

    async def wait_for_stream(self, timeout: float = 2.0) -> ScriptedStream:
        return await asyncio.wait_for(self._stream_ch.recv(), timeout)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise APIConnectionError("batch recognition not scripted")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        lang = str(language) if utils.is_given(language) else None
        stream_cls = SwitchableScriptedStream if self._supports_update else ScriptedStream
        stream = stream_cls(stt=self, conn_options=conn_options, language=lang)
        self.created_streams.append(stream)
        self._stream_ch.send_nowait(stream)
        return stream


def make_frame(duration_ms: int = 10, sample_rate: int = 48000) -> rtc.AudioFrame:
    samples = int(sample_rate * duration_ms / 1000)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )
