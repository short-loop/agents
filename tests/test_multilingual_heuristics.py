from __future__ import annotations

from livekit.agents.language import LanguageCode
from livekit.agents.stt import LanguageSwitchOptions, SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.stt.multilingual.heuristics import (
    SwitchDecision,
    SwitchSuppressed,
    _HeuristicEngine,
    _is_cross_script,
)
from livekit.agents.types import TimedString


def final(
    text: str,
    *,
    language: str = "hi",
    confidence: float = 0.9,
    end_time: float = 1.0,
    words: list[TimedString] | None = None,
) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                language=LanguageCode(language),
                text=text,
                confidence=confidence,
                start_time=max(0.0, end_time - 1.0),
                end_time=end_time,
                words=words,
            )
        ],
    )


HINDI_LONG = "नमस्ते आप कैसे हैं मुझे मदद चाहिए धन्यवाद"
SPANISH_LONG = "hola como estas amigo muy bien gracias por todo"


def make_engine(**overrides: object) -> _HeuristicEngine:
    opts = LanguageSwitchOptions(**overrides)  # type: ignore[arg-type]
    return _HeuristicEngine(opts, LanguageCode("en"), now_fn=lambda: 1000.0)


def test_cross_script_detection() -> None:
    assert _is_cross_script(LanguageCode("en"), HINDI_LONG)
    assert not _is_cross_script(LanguageCode("en"), SPANISH_LONG)
    assert not _is_cross_script(LanguageCode("hi"), HINDI_LONG)
    assert _is_cross_script(LanguageCode("hi"), "hello world again")
    # unknown current language: no boost possible
    assert not _is_cross_script(LanguageCode("xx"), HINDI_LONG)


def test_threshold_crossing_cross_script() -> None:
    engine = make_engine(switch_threshold=2.0)
    # 8+ words, conf 0.9, cross-script boost 1.5 -> 1.35 per final; second adds turn bonus
    assert engine.on_detector_event(final(HINDI_LONG, end_time=1.0)) is None
    result = engine.on_detector_event(final(HINDI_LONG, end_time=2.0))
    assert isinstance(result, SwitchDecision)
    assert result.target == "hi"
    assert result.trigger_transcript == HINDI_LONG
    assert result.first_evidence_audio_ts == 1.0


def test_same_script_needs_more_evidence() -> None:
    engine = make_engine(switch_threshold=2.0)
    # spanish over english primary: no script boost -> 0.9, then 0.9 + 1.0 turn bonus
    assert engine.on_detector_event(final(SPANISH_LONG, language="es", end_time=1.0)) is None
    result = engine.on_detector_event(final(SPANISH_LONG, language="es", end_time=2.0))
    assert isinstance(result, SwitchDecision)


def test_short_and_low_confidence_gated() -> None:
    engine = make_engine(switch_threshold=0.1)
    assert engine.on_detector_event(final("हाँ", end_time=1.0)) is None  # too short
    assert engine.on_detector_event(final(HINDI_LONG, confidence=0.3, end_time=2.0)) is None
    # zero confidence means "not provided" and falls back to default_confidence
    result = engine.on_detector_event(final(HINDI_LONG, confidence=0.0, end_time=3.0))
    assert isinstance(result, SwitchDecision)


def test_current_language_final_breaks_streak() -> None:
    engine = make_engine(switch_threshold=3.0, turn_bonus=10.0)
    engine.on_detector_event(final(HINDI_LONG, end_time=1.0))
    engine.on_detector_event(final("hello there my friend", language="en", end_time=2.0))
    # streak broken: the next hindi final gets no turn bonus, so no decision
    result = engine.on_detector_event(final(HINDI_LONG, end_time=3.0))
    assert result is None


def test_evidence_decays_over_audio_time() -> None:
    engine = make_engine(switch_threshold=2.0, evidence_half_life_s=10.0, turn_bonus=0.0)
    engine.on_detector_event(final(HINDI_LONG, end_time=1.0))  # score ~1.35
    # 30s of audio later, score decayed by ~8x; a second final must not cross alone
    result = engine.on_detector_event(final(HINDI_LONG, end_time=31.0))
    assert result is None


def test_cooldown_and_reentry() -> None:
    engine = make_engine(
        switch_threshold=1.0,
        hard_cooldown_s=10.0,
        reentry_threshold_multiplier=2.0,
        reentry_decay_s=100.0,
        evidence_half_life_s=10_000.0,
        turn_bonus=0.0,
    )
    decision = engine.on_detector_event(final(HINDI_LONG, end_time=1.0))
    assert isinstance(decision, SwitchDecision)
    engine.on_switch_started(decision.target)
    engine.on_switch_completed(LanguageCode("hi"), initiator="heuristic")

    # switching back to english is blocked during the hard cooldown
    result = engine.on_detector_event(
        final("hello how are you doing today my friend", language="en", end_time=5.0)
    )
    assert isinstance(result, SwitchSuppressed)
    assert result.reason == "cooldown"

    # right after cooldown the re-entry threshold is elevated, decaying back to 1x
    assert engine._reentry_multiplier("en", 11.0) == 2.0
    assert 1.0 < engine._reentry_multiplier("en", 61.0) < 2.0
    assert engine._reentry_multiplier("en", 250.0) == 1.0

    # evidence kept accumulating during cooldown; once past it, the switch fires
    for i in range(4):
        result = engine.on_detector_event(
            final("keep talking in english for a while now ok", language="en", end_time=12.0 + i)
        )
        if isinstance(result, SwitchDecision):
            break
    assert isinstance(result, SwitchDecision)


def test_manual_cooldown_is_stickier() -> None:
    engine = make_engine(
        switch_threshold=1.0,
        hard_cooldown_s=5.0,
        manual_cooldown_s=50.0,
        evidence_half_life_s=10_000.0,
    )
    engine.on_switch_started(LanguageCode("hi"))
    engine.on_switch_completed(LanguageCode("hi"), initiator="manual")

    result = engine.on_detector_event(
        final("hello how are you doing today my friend", language="en", end_time=30.0)
    )
    assert isinstance(result, SwitchSuppressed)
    assert result.reason == "cooldown"


def test_allowlist_suppression() -> None:
    engine = _HeuristicEngine(
        LanguageSwitchOptions(switch_threshold=1.0),
        LanguageCode("en"),
        allowed={"en", "hi"},
        now_fn=lambda: 0.0,
    )
    result = engine.on_detector_event(
        final(SPANISH_LONG, language="es", confidence=0.95, end_time=1.0)
    )
    if result is None:
        result = engine.on_detector_event(
            final(SPANISH_LONG, language="es", confidence=0.95, end_time=2.0)
        )
    assert isinstance(result, SwitchSuppressed)
    assert result.reason == "allowlist"
    # the evidence is still visible in snapshots
    snapshot = engine.evidence_snapshot_if_due()
    assert snapshot is not None and "es" in snapshot


def test_auto_switch_disabled_is_telemetry_only() -> None:
    engine = make_engine(switch_threshold=1.0, auto_switch=False)
    engine.on_detector_event(final(HINDI_LONG, end_time=1.0))
    result = engine.on_detector_event(final(HINDI_LONG, end_time=2.0))
    assert isinstance(result, SwitchSuppressed)
    assert result.reason == "auto_switch_disabled"


def test_word_composition_scaling() -> None:
    engine = make_engine(switch_threshold=100.0, turn_bonus=0.0, script_mismatch_boost=1.0)

    def tagged_words(fraction_hi: float, n: int = 8) -> list[TimedString]:
        n_hi = int(n * fraction_hi)
        return [TimedString("w", language="hi" if i < n_hi else "en") for i in range(n)]

    engine.on_detector_event(
        final(HINDI_LONG, end_time=1.0, confidence=1.0, words=tagged_words(1.0))
    )
    pure_score = engine._evidence["hi"].score

    engine2 = make_engine(switch_threshold=100.0, turn_bonus=0.0, script_mismatch_boost=1.0)
    engine2.on_detector_event(
        final(HINDI_LONG, end_time=1.0, confidence=1.0, words=tagged_words(0.5))
    )
    mixed_score = engine2._evidence["hi"].score

    engine3 = make_engine(switch_threshold=100.0, turn_bonus=0.0, script_mismatch_boost=1.0)
    engine3.on_detector_event(final(HINDI_LONG, end_time=1.0, confidence=1.0, words=None))
    untagged_score = engine3._evidence["hi"].score

    assert pure_score == untagged_score  # no tags -> neutral scale
    assert mixed_score < pure_score  # code-mixed turn contributes less


def test_snapshot_throttling() -> None:
    engine = make_engine(switch_threshold=100.0, evidence_event_interval_s=5.0)
    engine.on_detector_event(final(HINDI_LONG, end_time=1.0))
    assert engine.evidence_snapshot_if_due() is not None
    engine.on_detector_event(final(HINDI_LONG, end_time=2.0))
    assert engine.evidence_snapshot_if_due() is None  # within the throttle window
    engine.on_detector_event(final(HINDI_LONG, end_time=7.0))
    assert engine.evidence_snapshot_if_due() is not None


def test_multi_and_empty_language_ignored() -> None:
    engine = make_engine(switch_threshold=0.1)
    assert engine.on_detector_event(final(HINDI_LONG, language="multi", end_time=1.0)) is None
    assert engine.on_detector_event(final(HINDI_LONG, language="", end_time=2.0)) is None
    assert engine.on_detector_event(final("", language="hi", end_time=3.0)) is None
