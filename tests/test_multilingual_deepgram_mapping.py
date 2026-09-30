from __future__ import annotations

import pytest

from livekit.agents.utils import is_given

pytest.importorskip("livekit.plugins.deepgram")

from livekit.plugins.deepgram.stt import live_transcription_to_speech_data  # noqa: E402


def _payload(*, languages: list[str] | None, word_languages: list[str] | None) -> dict:
    words = []
    for i, word in enumerate(("नमस्ते", "hello")):
        entry: dict = {"word": word, "start": float(i), "end": float(i) + 0.5, "speaker": 0}
        if word_languages is not None:
            entry["language"] = word_languages[i]
        words.append(entry)

    alt: dict = {"transcript": "नमस्ते hello", "confidence": 0.92, "words": words}
    if languages is not None:
        alt["languages"] = languages

    return {"channel": {"alternatives": [alt]}}


def test_multi_mode_surfaces_detected_languages() -> None:
    data = _payload(languages=["hi", "en"], word_languages=["hi", "en"])
    speech_data = live_transcription_to_speech_data(
        "multi", data, is_final=True, start_time_offset=0.0
    )

    sd = speech_data[0]
    assert sd.language == "hi"
    assert sd.detected_languages == ["hi", "en"]
    assert sd.words is not None
    assert [w.language for w in sd.words] == ["hi", "en"]


def test_pinned_language_unaffected() -> None:
    data = _payload(languages=None, word_languages=None)
    speech_data = live_transcription_to_speech_data(
        "en-US", data, is_final=True, start_time_offset=0.0
    )

    sd = speech_data[0]
    assert sd.language == "en-US"
    assert sd.detected_languages is None
    assert sd.words is not None
    assert all(not is_given(w.language) for w in sd.words)


def test_multi_mode_without_languages_key() -> None:
    # defensive: multi mode but the payload lacks the languages list
    data = _payload(languages=None, word_languages=None)
    speech_data = live_transcription_to_speech_data(
        "multi", data, is_final=True, start_time_offset=0.0
    )

    sd = speech_data[0]
    assert sd.language == "multi"
    assert sd.detected_languages is None
