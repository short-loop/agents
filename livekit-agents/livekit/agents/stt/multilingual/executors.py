from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from ...language import LanguageCode
from ..stt import RecognizeStream
from .events import SwitchExecutorKind


def _supports_language_update(stream: RecognizeStream) -> bool:
    """True if the stream exposes ``update_options(language=...)`` (Deepgram-shaped)."""
    fn = getattr(stream, "update_options", None)
    if fn is None or not callable(fn):
        return False

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    return "language" in sig.parameters


class _SwitchExecutor(ABC):
    kind: ClassVar[SwitchExecutorKind]

    @abstractmethod
    async def switch(self, language: LanguageCode) -> RecognizeStream:
        """Move the primary to ``language``.

        Returns the stream that will become (or remain) the primary. In-place executors
        return the same stream object; recreate executors return a new shadow stream that
        is already receiving live audio. Waiting for health and promotion are the
        caller's responsibility.
        """


class _InPlaceExecutor(_SwitchExecutor):
    kind = "in_place"

    def __init__(self, get_stream: Callable[[], RecognizeStream]) -> None:
        self._get_stream = get_stream

    async def switch(self, language: LanguageCode) -> RecognizeStream:
        stream = self._get_stream()
        result = stream.update_options(language=language)  # type: ignore[attr-defined]
        if inspect.isawaitable(result):
            await result
        return stream


class _RecreateExecutor(_SwitchExecutor):
    kind = "recreate"

    def __init__(self, open_stream: Callable[[LanguageCode], RecognizeStream]) -> None:
        self._open_stream = open_stream

    async def switch(self, language: LanguageCode) -> RecognizeStream:
        return self._open_stream(language)
