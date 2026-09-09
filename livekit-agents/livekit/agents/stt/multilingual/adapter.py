from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from livekit import rtc

from ..._exceptions import APIConnectionError
from ...language import LanguageCode
from ...log import logger
from ...types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ...utils import aio, is_given
from ...utils.audio import AudioBuffer
from ..stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from .config import LanguageNotAllowedError, LanguageSwitchFailedError, LanguageSwitchOptions
from .events import (
    LanguageSwitchedEvent,
    LanguageSwitchEvidenceEvent,
    LanguageSwitchStartedEvent,
    LanguageSwitchSuppressedEvent,
    SwitchExecutorKind,
    SwitchInitiator,
)
from .executors import (
    _InPlaceExecutor,
    _RecreateExecutor,
    _supports_language_update,
    _SwitchExecutor,
)
from .heuristics import SwitchDecision, SwitchSuppressed, _HeuristicEngine

# retries are owned by the child streams (and, when the caller passes retryful
# conn options, by the base RecognizeStream retry loop rebuilding both children)
_ADAPTER_CONN_OPTIONS = APIConnectOptions(max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout)

_DETECTOR_RESTART_INITIAL_BACKOFF = 1.0
_DETECTOR_RESTART_MAX_BACKOFF = 30.0
_BOUNDARY_POLL_INTERVAL = 0.1

_TRANSCRIPT_EVENT_TYPES = (
    SpeechEventType.INTERIM_TRANSCRIPT,
    SpeechEventType.PREFLIGHT_TRANSCRIPT,
    SpeechEventType.FINAL_TRANSCRIPT,
)


@dataclass
class _ChildEvent:
    stream: RecognizeStream
    role: str
    ev: SpeechEvent


@dataclass
class _ChildEnded:
    stream: RecognizeStream
    role: str


@dataclass
class _ChildFailed:
    stream: RecognizeStream
    role: str
    exc: BaseException


@dataclass
class _SwitchResolved:
    pass


_ChildMsg = _ChildEvent | _ChildEnded | _ChildFailed | _SwitchResolved


@dataclass
class _SwitchContext:
    target: LanguageCode
    old_language: LanguageCode
    initiator: SwitchInitiator
    reason: str
    executor_kind: SwitchExecutorKind
    health_event: asyncio.Event
    boundary_event: asyncio.Event
    started_wall_ts: float
    first_evidence_wall_ts: float | None = None
    trigger_transcript: str | None = None
    new_stream: RecognizeStream | None = None
    failed_exc: BaseException | None = None


class MultilingualAdapter(
    STT[
        Literal[
            "language_switch_started",
            "language_switched",
            "language_switch_suppressed",
            "language_switch_evidence",
        ]
    ]
):
    """Dual-connection STT adapter with silent language switching.

    Runs two live STT connections: a language-pinned *primary* (any provider) whose
    events the session sees, and an always-on multilingual *detector* (e.g.
    ``deepgram.STT(model="nova-3", language="multi")``) consumed silently by a heuristic
    engine. When the engine detects that the user switched language — or when
    :meth:`switch_language` is called (e.g. from an LLM function tool) — the primary is
    moved to the new language. During the transition the detector's transcripts are
    forwarded instead of the primary's, so no speech goes untranscribed.

    Example::

        adapter = stt.MultilingualAdapter(
            primary=deepgram.STT(model="nova-3", language="en"),
            detector=deepgram.STT(model="nova-3", language="multi"),
            initial_language="en",
            options=stt.LanguageSwitchOptions(languages=["en", "hi"]),
        )
        session = AgentSession(stt=adapter, ...)
    """

    def __init__(
        self,
        *,
        detector: STT,
        initial_language: LanguageCode | str,
        primary: STT | None = None,
        primary_factory: Callable[[LanguageCode], STT] | None = None,
        options: NotGivenOr[LanguageSwitchOptions] = NOT_GIVEN,
    ) -> None:
        if primary is None and primary_factory is None:
            raise ValueError("either primary or primary_factory must be provided")

        opts = options if is_given(options) else LanguageSwitchOptions()
        language = LanguageCode(initial_language)
        allowed = _normalize_allowlist(opts.languages)
        if allowed is not None and language.language not in allowed:
            raise LanguageNotAllowedError(language, sorted(allowed))

        if opts.switch_mode == "recreate" and primary_factory is None:
            raise ValueError('switch_mode="recreate" requires primary_factory')

        owned_stts: list[STT] = []
        if primary is None:
            assert primary_factory is not None
            primary = primary_factory(language)
            owned_stts.append(primary)

        if not primary.capabilities.streaming or not detector.capabilities.streaming:
            raise ValueError(
                "MultilingualAdapter requires streaming-capable primary and detector STTs; "
                "wrap non-streaming STTs with stt.StreamAdapter first"
            )

        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=primary.capabilities.interim_results,
                diarization=primary.capabilities.diarization,
                aligned_transcript=primary.capabilities.aligned_transcript,
                offline_recognize=primary.capabilities.offline_recognize,
            )
        )

        self._primary = primary
        self._detector = detector
        self._factory = primary_factory
        self._opts = opts
        self._allowed = allowed
        self._current_language = language
        self._owned_stts = owned_stts

        self._streams: weakref.WeakSet[MultilingualRecognizeStream] = weakref.WeakSet()
        self._hooked_stts: list[STT] = []
        self._hook_metrics(primary)
        self._hook_metrics(detector)
        self._recognize_metrics_needed = False  # children already emit their own metrics

    @property
    def model(self) -> str:
        return "MultilingualAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    @property
    def current_language(self) -> LanguageCode:
        return self._current_language

    @property
    def options(self) -> LanguageSwitchOptions:
        return self._opts

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        return await self._primary.recognize(
            buffer,
            language=language if is_given(language) else self._current_language,
            conn_options=conn_options,
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _ADAPTER_CONN_OPTIONS,
    ) -> SpeechEvent:
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _ADAPTER_CONN_OPTIONS,
    ) -> RecognizeStream:
        stream_language = LanguageCode(language) if is_given(language) else self._current_language
        stream = MultilingualRecognizeStream(
            adapter=self, language=stream_language, conn_options=conn_options
        )
        self._streams.add(stream)
        return stream

    async def switch_language(
        self, language: LanguageCode | str, *, reason: str = "manual"
    ) -> None:
        """Manually switch the primary STT language (e.g. from an LLM function tool).

        Goes through the same serialized switch path and transition window as heuristic
        switches, but resets accumulated evidence and applies the (stronger)
        ``manual_cooldown_s`` stickiness afterwards.

        Raises:
            LanguageNotAllowedError: if ``language`` is not in the configured allowlist.
            LanguageSwitchFailedError: if the switch was rolled back.
        """
        target = LanguageCode(language)
        if self._allowed is not None and target.language not in self._allowed:
            raise LanguageNotAllowedError(target, sorted(self._allowed))

        live_streams = [s for s in self._streams if not s._event_ch.closed]
        if not live_streams:
            self._current_language = target
            return

        for stream in live_streams:
            await stream._request_switch(target, initiator="manual", reason=reason)

    async def aclose(self) -> None:
        for stt_instance in self._hooked_stts:
            stt_instance.off("metrics_collected", self._on_metrics_collected)
        self._hooked_stts.clear()

        await asyncio.gather(
            *[stt_instance.aclose() for stt_instance in self._owned_stts],
            return_exceptions=True,
        )
        self._owned_stts.clear()

    def _hook_metrics(self, stt_instance: STT) -> None:
        if stt_instance not in self._hooked_stts:
            stt_instance.on("metrics_collected", self._on_metrics_collected)
            self._hooked_stts.append(stt_instance)

    def _adopt_stt(self, stt_instance: STT) -> None:
        self._owned_stts.append(stt_instance)
        self._hook_metrics(stt_instance)

    def _release_stt(self, stt_instance: STT) -> None:
        if stt_instance in self._owned_stts:
            self._owned_stts.remove(stt_instance)
        if stt_instance in self._hooked_stts:
            stt_instance.off("metrics_collected", self._on_metrics_collected)
            self._hooked_stts.remove(stt_instance)

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)


class MultilingualRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        adapter: MultilingualAdapter,
        language: LanguageCode,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=adapter, conn_options=conn_options, sample_rate=NOT_GIVEN)
        self._adapter = adapter
        self._opts = adapter._opts
        self._language = language

        # evidence persists across base-class retries of _run (same conversation)
        self._engine = _HeuristicEngine(self._opts, language, allowed=adapter._allowed)

        # per-run state, (re)initialized in _run
        self._fanout: list[RecognizeStream] = []
        self._all_children: set[RecognizeStream] = set()
        self._pumps: dict[RecognizeStream, asyncio.Task[None]] = {}
        self._child_stt: dict[RecognizeStream, STT] = {}
        self._merged_ch: aio.Chan[_ChildMsg] = aio.Chan()
        self._primary_stream: RecognizeStream | None = None
        self._detector_stream: RecognizeStream | None = None
        self._executor: _SwitchExecutor | None = None
        self._switch_lock = asyncio.Lock()
        self._active_switch: _SwitchContext | None = None
        self._owner: Literal["primary", "detector"] = "primary"
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._detector_restart_task: asyncio.Task[None] | None = None
        self._detector_backoff = _DETECTOR_RESTART_INITIAL_BACKOFF
        self._audio_clock = 0.0
        self._last_detector_activity_clock = 0.0
        self._last_transcript_end_ts = 0.0
        self._flip_gate_ts = 0.0
        self._input_ended = False
        self._primary_ended = False

    # -- public-ish surface ------------------------------------------------------------

    @property
    def start_time_offset(self) -> float:
        return self._start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._start_time_offset = value
        for child in list(self._fanout):
            child.start_time_offset = value

    async def _request_switch(
        self, target: LanguageCode, *, initiator: SwitchInitiator, reason: str
    ) -> None:
        await self._do_switch(target, initiator=initiator, reason=reason, decision=None)

    # -- core --------------------------------------------------------------------------

    async def _run(self) -> None:
        adapter = self._adapter

        # reset per-run state (the base class may retry _run on APIError)
        self._fanout = []
        self._all_children = set()
        self._pumps = {}
        self._child_stt = {}
        self._merged_ch = aio.Chan[_ChildMsg]()
        self._active_switch = None
        self._owner = "primary"
        self._bg_tasks = set()
        self._detector_restart_task = None
        self._input_ended = False
        self._primary_ended = False

        # a retry must rebuild at the adapter's current language, not the initial one
        self._language = adapter._current_language

        self._primary_stream = self._open_child(
            adapter._primary, role="primary", language=self._language
        )
        self._detector_stream = self._open_child(adapter._detector, role="detector")
        self._executor = self._resolve_executor()

        forward_task = asyncio.create_task(
            self._forward_input_task(), name="MultilingualAdapter.forward_input"
        )

        try:
            await self._gate_loop()
        finally:
            await aio.cancel_and_wait(forward_task)
            if self._detector_restart_task is not None:
                await aio.cancel_and_wait(self._detector_restart_task)
            if self._bg_tasks:
                await aio.cancel_and_wait(*self._bg_tasks)
            if self._pumps:
                await aio.cancel_and_wait(*self._pumps.values())
            for child in self._all_children:
                with contextlib.suppress(Exception):
                    await child.aclose()
            self._merged_ch.close()

    def _open_child(
        self,
        stt_instance: STT,
        *,
        role: str,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> RecognizeStream:
        child = stt_instance.stream(language=language, conn_options=self._conn_options)
        child.start_time_offset = self._start_time_offset
        self._fanout.append(child)
        self._all_children.add(child)
        self._child_stt[child] = stt_instance
        self._pumps[child] = asyncio.create_task(
            self._pump_child(child, role), name=f"MultilingualAdapter.pump_{role}"
        )
        return child

    async def _forward_input_task(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                self._audio_clock += data.duration

            for child in list(self._fanout):
                try:
                    if isinstance(data, rtc.AudioFrame):
                        child.push_frame(data)
                    elif isinstance(data, self._FlushSentinel):
                        child.flush()
                except Exception:
                    # a child may be closed/reconnecting mid-switch; never block the others
                    pass

        self._input_ended = True
        for child in list(self._fanout):
            with contextlib.suppress(RuntimeError):
                child.end_input()

    async def _pump_child(self, child: RecognizeStream, role: str) -> None:
        try:
            async for ev in child:
                self._merged_ch.send_nowait(_ChildEvent(stream=child, role=role, ev=ev))
        except aio.ChanClosed:
            return
        except Exception as exc:
            with contextlib.suppress(aio.ChanClosed):
                self._merged_ch.send_nowait(_ChildFailed(stream=child, role=role, exc=exc))
        else:
            with contextlib.suppress(aio.ChanClosed):
                self._merged_ch.send_nowait(_ChildEnded(stream=child, role=role))

    async def _gate_loop(self) -> None:
        async for msg in self._merged_ch:
            if isinstance(msg, _SwitchResolved):
                if self._primary_ended and self._active_switch is None:
                    return
                continue

            if isinstance(msg, _ChildFailed):
                if self._handle_child_failed(msg):
                    return
                continue

            if isinstance(msg, _ChildEnded):
                if self._handle_child_ended(msg):
                    return
                continue

            self._on_child_event(msg)

    def _handle_child_failed(self, msg: _ChildFailed) -> bool:
        """Returns True when the gate loop should exit (never; primary failure raises)."""
        ctx = self._active_switch

        if msg.stream is self._primary_stream:
            # base RecognizeStream retry policy decides whether the whole node rebuilds
            raise msg.exc

        if msg.stream is self._detector_stream:
            self._handle_detector_down(msg.exc)
            return False

        if ctx is not None and msg.stream is ctx.new_stream:
            ctx.failed_exc = msg.exc
            ctx.health_event.set()
            ctx.boundary_event.set()
            return False

        logger.debug("multilingual adapter: dropping failure from retired stream")
        return False

    def _handle_child_ended(self, msg: _ChildEnded) -> bool:
        """Returns True when the gate loop should exit."""
        ctx = self._active_switch

        if msg.stream is self._primary_stream:
            self._primary_ended = True
            if not self._input_ended:
                raise APIConnectionError("primary STT stream ended unexpectedly")
            if ctx is not None:
                # no more audio is coming: resolve the transition as soon as possible
                ctx.boundary_event.set()
                return False
            return True

        if msg.stream is self._detector_stream:
            if not self._input_ended:
                self._handle_detector_down(None)
            return False

        if ctx is not None and msg.stream is ctx.new_stream:
            ctx.failed_exc = APIConnectionError("new primary stream ended during switch")
            ctx.health_event.set()
            ctx.boundary_event.set()

        return False

    def _on_child_event(self, msg: _ChildEvent) -> None:
        ev = msg.ev
        ctx = self._active_switch

        # usage is real on every connection; billing accuracy beats prettiness
        if ev.type == SpeechEventType.RECOGNITION_USAGE:
            with contextlib.suppress(aio.ChanClosed):
                self._event_ch.send_nowait(ev)
            return

        is_transcript = ev.type in _TRANSCRIPT_EVENT_TYPES
        has_text = bool(ev.alternatives and ev.alternatives[0].text)

        if (
            ctx is not None
            and ctx.new_stream is not None
            and msg.stream is ctx.new_stream
            and is_transcript
            and has_text
        ):
            ctx.health_event.set()

        if msg.stream is self._detector_stream:
            self._on_detector_event(ev, is_transcript=is_transcript, has_text=has_text)
            return

        if msg.stream is self._primary_stream:
            self._engine.on_primary_event(ev)
            if is_transcript and has_text:
                self._note_transcript_ts(ev)
            if self._owner == "primary":
                with contextlib.suppress(aio.ChanClosed):
                    self._event_ch.send_nowait(ev)
            return

        if ctx is not None and msg.stream is ctx.new_stream:
            # shadow stream: future primary, suppressed until promotion
            self._engine.on_primary_event(ev)
            return

        # event from a retired stream: drop

    def _on_detector_event(self, ev: SpeechEvent, *, is_transcript: bool, has_text: bool) -> None:
        adapter = self._adapter
        ctx = self._active_switch

        if is_transcript:
            self._last_detector_activity_clock = self._audio_clock
            self._detector_backoff = _DETECTOR_RESTART_INITIAL_BACKOFF
            if has_text:
                self._note_transcript_ts(ev)

        result = self._engine.on_detector_event(ev)
        self._handle_engine_result(result)

        snapshot = self._engine.evidence_snapshot_if_due()
        if snapshot is not None:
            adapter.emit(
                "language_switch_evidence",
                LanguageSwitchEvidenceEvent(
                    current_language=self._language,
                    audio_time=self._engine.last_audio_ts,
                    scores=snapshot,
                ),
            )

        if self._owner != "detector":
            return

        # DETECTOR_OWNS: the detector is the transcriber of record
        if ev.type in (
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
        ):
            if (
                ev.alternatives
                and ev.alternatives[0].text
                and ev.alternatives[0].start_time >= self._flip_gate_ts
            ):
                with contextlib.suppress(aio.ChanClosed):
                    self._event_ch.send_nowait(ev)
        elif ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
            with contextlib.suppress(aio.ChanClosed):
                self._event_ch.send_nowait(ev)
        elif ev.type == SpeechEventType.START_OF_SPEECH:
            with contextlib.suppress(aio.ChanClosed):
                self._event_ch.send_nowait(ev)
        elif ev.type == SpeechEventType.END_OF_SPEECH:
            with contextlib.suppress(aio.ChanClosed):
                self._event_ch.send_nowait(ev)
            if ctx is not None and ctx.health_event.is_set() and ctx.failed_exc is None:
                ctx.boundary_event.set()

    def _handle_engine_result(self, result: SwitchDecision | SwitchSuppressed | None) -> None:
        adapter = self._adapter
        if result is None:
            return

        if isinstance(result, SwitchSuppressed):
            adapter.emit(
                "language_switch_suppressed",
                LanguageSwitchSuppressedEvent(
                    target_language=result.target, reason=result.reason, score=result.score
                ),
            )
            return

        if self._switch_lock.locked() or self._active_switch is not None:
            adapter.emit(
                "language_switch_suppressed",
                LanguageSwitchSuppressedEvent(
                    target_language=result.target,
                    reason="switch_in_progress",
                    score=result.score,
                ),
            )
            return

        if self._executor is None:
            adapter.emit(
                "language_switch_suppressed",
                LanguageSwitchSuppressedEvent(
                    target_language=result.target, reason="no_executor", score=result.score
                ),
            )
            return

        task = asyncio.create_task(
            self._heuristic_switch_task(result), name="MultilingualAdapter.switch"
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _heuristic_switch_task(self, decision: SwitchDecision) -> None:
        try:
            await self._do_switch(
                decision.target,
                initiator="heuristic",
                reason=decision.reason,
                decision=decision,
            )
        except LanguageSwitchFailedError as e:
            logger.warning(
                "multilingual adapter: heuristic language switch failed",
                extra={"target": str(decision.target), "reason": e.reason},
            )
        except Exception:
            logger.exception("multilingual adapter: unexpected error during language switch")

    async def _do_switch(
        self,
        target: LanguageCode,
        *,
        initiator: SwitchInitiator,
        reason: str,
        decision: SwitchDecision | None,
    ) -> None:
        adapter = self._adapter
        opts = self._opts

        async with self._switch_lock:
            target = LanguageCode(target)
            if target == self._language:
                return

            executor = self._executor
            if executor is None:
                raise LanguageSwitchFailedError(target, "no switch mechanism available")

            old_language = self._language
            ctx = _SwitchContext(
                target=target,
                old_language=old_language,
                initiator=initiator,
                reason=reason,
                executor_kind=executor.kind,
                health_event=asyncio.Event(),
                boundary_event=asyncio.Event(),
                started_wall_ts=time.time(),
                first_evidence_wall_ts=decision.first_evidence_wall_ts if decision else None,
                trigger_transcript=decision.trigger_transcript if decision else None,
            )
            self._active_switch = ctx

            adapter.emit(
                "language_switch_started",
                LanguageSwitchStartedEvent(
                    old_language=old_language,
                    new_language=target,
                    initiator=initiator,
                    reason=reason,
                ),
            )

            # ---- flip 1: the detector becomes the transcriber of record --------------
            self._owner = "detector"
            self._flip_gate_ts = self._last_transcript_end_ts
            self._engine.on_switch_started(target)
            self._emit_clear_events()

            watchdog: asyncio.Task[None] | None = None
            new_stream: RecognizeStream | None = None
            try:
                try:
                    new_stream = await asyncio.wait_for(
                        executor.switch(target), timeout=opts.switch_timeout_s
                    )
                    ctx.new_stream = new_stream

                    # health: first transcript from the new stream, a failure, or a
                    # grace period without failure (the user may simply be silent)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(ctx.health_event.wait(), timeout=opts.switch_grace_s)

                    if ctx.failed_exc is not None:
                        raise ctx.failed_exc

                    # silence within the grace period counts as healthy
                    ctx.health_event.set()
                except Exception as exc:
                    self._rollback_switch(ctx, executor, new_stream, exc)
                    raise LanguageSwitchFailedError(target, str(exc)) from exc

                # ---- transition window: wait for an utterance boundary ---------------
                if self._primary_ended:
                    ctx.boundary_event.set()

                watchdog = asyncio.create_task(
                    self._boundary_watchdog(ctx), name="MultilingualAdapter.boundary"
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        ctx.boundary_event.wait(), timeout=opts.max_detector_owns_s
                    )

                if ctx.failed_exc is not None:
                    self._rollback_switch(ctx, executor, new_stream, ctx.failed_exc)
                    raise LanguageSwitchFailedError(target, str(ctx.failed_exc)) from ctx.failed_exc

                # ---- flip 2: promote --------------------------------------------------
                if executor.kind == "recreate":
                    assert new_stream is not None
                    old_stream = self._primary_stream
                    new_stt = self._child_stt[new_stream]
                    # register the new identity BEFORE flipping ownership so the gate
                    # loop's identity check and the owner flip are atomic (no await
                    # between here and _emit_clear_events)
                    self._primary_stream = new_stream
                    adapter._primary = new_stt
                    if old_stream is not None:
                        self._retire_stream(old_stream)

                self._owner = "primary"
                self._emit_clear_events()
                self._language = target
                adapter._current_language = target
                self._engine.on_switch_completed(target, initiator=initiator)

                latency = 0.0
                if ctx.first_evidence_wall_ts is not None:
                    latency = time.time() - ctx.first_evidence_wall_ts

                adapter.emit(
                    "language_switched",
                    LanguageSwitchedEvent(
                        old_language=old_language,
                        new_language=target,
                        initiator=initiator,
                        executor=executor.kind,
                        latency=latency,
                        trigger_transcript=ctx.trigger_transcript,
                    ),
                )
            finally:
                if watchdog is not None:
                    await aio.cancel_and_wait(watchdog)
                self._active_switch = None
                with contextlib.suppress(aio.ChanClosed):
                    self._merged_ch.send_nowait(_SwitchResolved())

    def _rollback_switch(
        self,
        ctx: _SwitchContext,
        executor: _SwitchExecutor,
        new_stream: RecognizeStream | None,
        exc: BaseException,
    ) -> None:
        adapter = self._adapter
        logger.warning(
            "multilingual adapter: rolling back language switch",
            exc_info=exc,
            extra={"target": str(ctx.target), "initiator": ctx.initiator},
        )

        if executor.kind == "recreate":
            if new_stream is not None:
                self._retire_stream(new_stream)
        elif executor.kind == "in_place" and new_stream is not None:
            # the underlying connection may already be in the target language; revert
            with contextlib.suppress(Exception):
                new_stream.update_options(language=ctx.old_language)  # type: ignore[attr-defined]

        self._owner = "primary"
        self._emit_clear_events()
        self._engine.on_switch_failed(ctx.target)
        adapter.emit(
            "language_switch_suppressed",
            LanguageSwitchSuppressedEvent(
                target_language=ctx.target, reason="switch_failed", score=0.0
            ),
        )

    async def _boundary_watchdog(self, ctx: _SwitchContext) -> None:
        # boundary = enough audio-time silence since the last detector transcript. If the
        # user was already silent at flip 1, this fires almost immediately — there is no
        # in-flight utterance and no seam risk.
        while not ctx.boundary_event.is_set():
            await asyncio.sleep(_BOUNDARY_POLL_INTERVAL)
            silence = self._audio_clock - self._last_detector_activity_clock
            if ctx.health_event.is_set() and silence >= self._opts.boundary_silence_s:
                ctx.boundary_event.set()
                return

    def _open_shadow_stream(self, language: LanguageCode) -> RecognizeStream:
        assert self._adapter._factory is not None
        new_stt = self._adapter._factory(language)
        self._adapter._adopt_stt(new_stt)
        return self._open_child(new_stt, role="shadow", language=language)

    def _retire_stream(self, stream: RecognizeStream) -> None:
        if stream in self._fanout:
            self._fanout.remove(stream)
        pump = self._pumps.pop(stream, None)
        stt_instance = self._child_stt.pop(stream, None)
        self._all_children.discard(stream)

        adapter = self._adapter
        release_stt = (
            stt_instance is not None
            and stt_instance in adapter._owned_stts
            and stt_instance is not adapter._primary
            and stt_instance is not adapter._detector
        )

        async def _close() -> None:
            if pump is not None:
                await aio.cancel_and_wait(pump)
            with contextlib.suppress(Exception):
                await stream.aclose()
            if release_stt and stt_instance is not None:
                adapter._release_stt(stt_instance)
                with contextlib.suppress(Exception):
                    await stt_instance.aclose()

        task = asyncio.create_task(_close(), name="MultilingualAdapter.retire")
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _resolve_executor(self) -> _SwitchExecutor | None:
        opts = self._opts
        adapter = self._adapter
        assert self._primary_stream is not None

        def _get_primary_stream() -> RecognizeStream:
            assert self._primary_stream is not None
            return self._primary_stream

        in_place_capable = _supports_language_update(self._primary_stream)

        if opts.switch_mode == "in_place":
            if not in_place_capable:
                raise ValueError(
                    'switch_mode="in_place" but the primary stream does not expose '
                    "update_options(language=...)"
                )
            return _InPlaceExecutor(_get_primary_stream)

        if opts.switch_mode == "recreate":
            return _RecreateExecutor(self._open_shadow_stream)

        # auto
        if in_place_capable:
            return _InPlaceExecutor(_get_primary_stream)
        if adapter._factory is not None:
            return _RecreateExecutor(self._open_shadow_stream)

        logger.warning(
            "multilingual adapter: no language-switch mechanism available "
            "(primary has no update_options(language=...) and no primary_factory was "
            "provided); switches will be suppressed"
        )
        return None

    def _handle_detector_down(self, exc: BaseException | None) -> None:
        if self._input_ended:
            return

        error = exc if exc is not None else APIConnectionError("detector STT stream ended")
        logger.warning(
            "multilingual adapter: detector stream failed; language detection paused",
            exc_info=exc,
        )
        self._emit_error(
            Exception(error) if not isinstance(error, Exception) else error, recoverable=True
        )

        if not self._opts.detector_restart:
            return
        if self._detector_restart_task is not None and not self._detector_restart_task.done():
            return
        self._detector_restart_task = asyncio.create_task(
            self._restart_detector(), name="MultilingualAdapter.restart_detector"
        )

    async def _restart_detector(self) -> None:
        backoff = self._detector_backoff
        self._detector_backoff = min(backoff * 2, _DETECTOR_RESTART_MAX_BACKOFF)
        await asyncio.sleep(backoff)

        old = self._detector_stream
        if old is not None:
            self._retire_stream(old)

        self._detector_stream = self._open_child(self._adapter._detector, role="detector")
        logger.info("multilingual adapter: detector stream restarted")

    def _note_transcript_ts(self, ev: SpeechEvent) -> None:
        if ev.alternatives:
            self._last_transcript_end_ts = max(
                self._last_transcript_end_ts, ev.alternatives[0].end_time
            )

    def _emit_clear_events(self) -> None:
        # an empty INTERIM is required to clear a dangling interim downstream
        # (audio_recognition's FINAL handler early-returns on empty text before its
        # interim reset); the empty FINAL is kept for other consumers
        with contextlib.suppress(aio.ChanClosed):
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[SpeechData(language=LanguageCode(""), text="")],
                )
            )
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(language=LanguageCode(""), text="")],
                )
            )

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        async for _ in event_aiter:
            pass


def _normalize_allowlist(languages: list[str] | None) -> set[str] | None:
    if languages is None:
        return None
    return {LanguageCode(lang).language for lang in languages}
