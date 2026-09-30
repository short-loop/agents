from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import ChatContext, ParallelAdapter, ParallelLLMEntry
from livekit.agents.metrics import LLMMetrics

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _llm(content: str, *, ttft: float, duration: float) -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content=content, ttft=ttft, duration=duration)
        ]
    )


async def _collect(adapter: ParallelAdapter) -> str:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    text = ""
    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    return text


def test_requires_two_entries() -> None:
    with pytest.raises(ValueError):
        ParallelAdapter([ParallelLLMEntry(llm=_llm("a", ttft=0.0, duration=0.0), label="a")])


async def test_fastest_entry_wins() -> None:
    fast = _llm("fast reply", ttft=0.02, duration=0.05)
    slow = _llm("slow reply", ttft=0.5, duration=0.6)
    adapter = ParallelAdapter(
        [ParallelLLMEntry(llm=slow, label="slow"), ParallelLLMEntry(llm=fast, label="fast")]
    )
    selected: list[LLMMetrics] = []
    adapter.on("metrics_collected", selected.append)

    assert await _collect(adapter) == "fast reply"
    assert adapter._active_instance is fast
    assert adapter.model == fast.model

    await asyncio.sleep(0.05)  # let the winner's metrics monitor flush
    assert selected and all(m.parallel_selected for m in selected)
    await adapter.aclose()


async def test_all_entries_failing_raises() -> None:
    # no fake response registered for "hello" -> empty stream, not a failure; use a
    # response map keyed on other input so both streams end without chunks
    a = _llm("x", ttft=0.0, duration=0.0)
    b = _llm("y", ttft=0.0, duration=0.0)
    adapter = ParallelAdapter(
        [ParallelLLMEntry(llm=a, label="a"), ParallelLLMEntry(llm=b, label="b")]
    )
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="unknown input")
    from livekit.agents import APIConnectionError

    with pytest.raises(APIConnectionError):
        async with adapter.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass
    await adapter.aclose()
