"""fork: unit tests for the pure helpers behind patch 01 (crutch words) and patch 02 (digit endpointing)."""

from __future__ import annotations

import pytest

from livekit.agents.voice.audio_recognition import (
    _STRIPPED_BACKCHANNEL_WORDS,
    _ends_with_alpha_numeric,
    _ends_with_number_like,
    _strip_word,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("my number is nine eight seven", True),
        ("call me at 98 76", True),
        ("I was born in 1990", False),  # a single number token
        ("one two three.", True),
        ("hello there", False),
        ("", False),
    ],
)
def test_ends_with_number_like(transcript: str, expected: bool) -> None:
    assert _ends_with_number_like(transcript) is expected


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("the code is a b 4 c", True),
        ("alpha bravo 7 charlie", True),
        ("it is a b c d", False),  # letters only, no number
        ("too short 1 a", False),  # fewer than four words
        ("we will meet on the 5th", False),
    ],
)
def test_ends_with_alpha_numeric(transcript: str, expected: bool) -> None:
    assert _ends_with_alpha_numeric(transcript) is expected


def test_backchannel_words_are_normalized() -> None:
    assert _strip_word(" Okay! ") == "okay"
    assert _strip_word("Mm-hmm") == "mmhmm"
    assert _strip_word("Uh-huh.") in _STRIPPED_BACKCHANNEL_WORDS
    assert _strip_word("Yes,") in _STRIPPED_BACKCHANNEL_WORDS
    assert _strip_word("actually") not in _STRIPPED_BACKCHANNEL_WORDS
