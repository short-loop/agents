from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ...language import LanguageCode
from ...types import NotGivenOr
from ...utils import is_given
from ..stt import SpeechEvent, SpeechEventType
from .config import LanguageSwitchOptions

# Unicode script ranges for the cross-script boost (signal 1). Coarse on purpose:
# only letters matter, and only scripts that map unambiguously to language families.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0041, 0x024F, "latin"),
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0E00, 0x0E7F, "thai"),
    (0x1100, 0x11FF, "hangul"),
    (0x3040, 0x30FF, "kana"),
    (0x3400, 0x4DBF, "cjk"),
    (0x4E00, 0x9FFF, "cjk"),
    (0xAC00, 0xD7AF, "hangul"),
]

_LANG_TO_SCRIPTS: dict[str, frozenset[str]] = {
    **dict.fromkeys(
        (
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "pl",
            "tr",
            "vi",
            "id",
            "ms",
            "sv",
            "da",
            "no",
            "fi",
            "cs",
            "ro",
            "hu",
            "ca",
            "sk",
            "hr",
            "tl",
        ),
        frozenset({"latin"}),
    ),
    **dict.fromkeys(("ru", "uk", "bg", "sr", "mk", "be"), frozenset({"cyrillic"})),
    **dict.fromkeys(("hi", "mr", "ne", "sa"), frozenset({"devanagari"})),
    "bn": frozenset({"bengali"}),
    "pa": frozenset({"gurmukhi"}),
    "gu": frozenset({"gujarati"}),
    "ta": frozenset({"tamil"}),
    "te": frozenset({"telugu"}),
    "kn": frozenset({"kannada"}),
    "ml": frozenset({"malayalam"}),
    **dict.fromkeys(("ar", "fa", "ur"), frozenset({"arabic"})),
    "he": frozenset({"hebrew"}),
    "th": frozenset({"thai"}),
    "ko": frozenset({"hangul"}),
    "ja": frozenset({"kana", "cjk"}),
    "zh": frozenset({"cjk"}),
    "el": frozenset({"greek"}),
}


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for lo, hi, script in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] = counts.get(script, 0) + 1
                break

    if not counts:
        return None
    return max(counts, key=lambda s: counts[s])


def _is_cross_script(current: LanguageCode, text: str) -> bool:
    expected = _LANG_TO_SCRIPTS.get(current.language)
    if expected is None:
        return False
    dominant = _dominant_script(text)
    return dominant is not None and dominant not in expected


@dataclass
class SwitchDecision:
    """The engine decided the user switched language; the adapter should act."""

    target: LanguageCode
    score: float
    reason: str
    trigger_transcript: str
    first_evidence_audio_ts: float
    first_evidence_wall_ts: float


@dataclass
class SwitchSuppressed:
    """Evidence crossed the threshold but the switch is blocked."""

    target: LanguageCode
    reason: Literal["cooldown", "allowlist", "auto_switch_disabled"]
    score: float


@dataclass
class _Evidence:
    score: float = 0.0
    last_update_audio_ts: float = 0.0
    first_evidence_audio_ts: float | None = None
    first_evidence_wall_ts: float | None = None
    consecutive_turns: int = 0


class _HeuristicEngine:
    """Pure, synchronous evidence accumulator for language-switch detection.

    All timing is based on event timestamps (audio time), never wall-clock — the audio
    stream is gapped while the agent speaks. ``now_fn`` exists only to stamp the
    wall-clock latency metric and is injectable for tests.
    """

    def __init__(
        self,
        opts: LanguageSwitchOptions,
        initial_language: LanguageCode,
        *,
        allowed: set[str] | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._opts = opts
        self._current = initial_language
        self._allowed = allowed
        self._now = now_fn

        # all dicts keyed by base language (LanguageCode.language)
        self._evidence: dict[str, _Evidence] = {}
        self._target_codes: dict[str, LanguageCode] = {}
        self._cooldown_until: dict[str, float] = {}
        self._last_audio_ts: float = 0.0
        self._last_snapshot_ts: float = float("-inf")
        self._switch_inflight = False

    @property
    def current_language(self) -> LanguageCode:
        return self._current

    @property
    def last_audio_ts(self) -> float:
        return self._last_audio_ts

    def on_detector_event(self, ev: SpeechEvent) -> SwitchDecision | SwitchSuppressed | None:
        opts = self._opts
        if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
            weight = 1.0
            is_final = True
        elif ev.type == SpeechEventType.INTERIM_TRANSCRIPT and opts.interim_evidence_weight > 0:
            weight = opts.interim_evidence_weight
            is_final = False
        else:
            return None

        if not ev.alternatives:
            return None

        sd = ev.alternatives[0]
        text = sd.text.strip()
        if not text:
            return None

        audio_ts = max(sd.end_time, self._last_audio_ts)
        self._last_audio_ts = audio_ts

        base = sd.language.language if sd.language else ""
        if base in ("", "multi"):
            return None

        if base == self._current.language:
            # a turn in the current language breaks every candidate's streak
            for evidence in self._evidence.values():
                evidence.consecutive_turns = 0
            return None

        if len(text) <= opts.min_transcript_length:
            return None

        confidence = sd.confidence if sd.confidence > 0 else opts.default_confidence
        if confidence < opts.min_detector_confidence:
            return None

        evidence = self._evidence.setdefault(base, _Evidence(last_update_audio_ts=audio_ts))
        self._target_codes[base] = sd.language
        self._decay(evidence, audio_ts)

        n_words = len(text.split())
        length_weight = min(n_words, opts.word_length_cap) / opts.word_length_cap

        composition_scale = 1.0
        fraction = self._word_language_fraction(sd.words, base)
        if fraction is not None:
            composition_scale = 0.5 + 0.5 * fraction

        script_mult = opts.script_mismatch_boost if _is_cross_script(self._current, text) else 1.0

        delta = length_weight * confidence * composition_scale * script_mult * weight
        if is_final:
            evidence.consecutive_turns += 1
            if evidence.consecutive_turns >= 2:
                delta += opts.turn_bonus

        evidence.score += delta
        if evidence.first_evidence_audio_ts is None:
            evidence.first_evidence_audio_ts = audio_ts
            evidence.first_evidence_wall_ts = self._now()

        # during the hard cooldown, suppression is reported against the base threshold so
        # the block stays observable; the elevated re-entry threshold applies afterwards
        in_cooldown = audio_ts < self._cooldown_until.get(base, float("-inf"))
        multiplier = 1.0 if in_cooldown else self._reentry_multiplier(base, audio_ts)
        threshold = opts.switch_threshold * multiplier
        if evidence.score < threshold or self._switch_inflight:
            return None

        target = self._target_codes[base]
        if self._allowed is not None and base not in self._allowed:
            return SwitchSuppressed(target=target, reason="allowlist", score=evidence.score)

        if in_cooldown:
            return SwitchSuppressed(target=target, reason="cooldown", score=evidence.score)

        if not opts.auto_switch:
            return SwitchSuppressed(
                target=target, reason="auto_switch_disabled", score=evidence.score
            )

        assert evidence.first_evidence_audio_ts is not None
        assert evidence.first_evidence_wall_ts is not None
        return SwitchDecision(
            target=target,
            score=evidence.score,
            reason="evidence_threshold",
            trigger_transcript=text,
            first_evidence_audio_ts=evidence.first_evidence_audio_ts,
            first_evidence_wall_ts=evidence.first_evidence_wall_ts,
        )

    def on_primary_event(self, ev: SpeechEvent) -> None:
        # reserved for v2 signals (primary confidence collapse, text-content mismatch)
        pass

    def on_switch_started(self, target: LanguageCode) -> None:
        self._switch_inflight = True

    def on_switch_completed(
        self, new_language: LanguageCode, *, initiator: Literal["heuristic", "manual"]
    ) -> None:
        old = self._current
        self._current = new_language
        self._switch_inflight = False
        self._evidence.clear()

        cooldown = (
            self._opts.manual_cooldown_s if initiator == "manual" else self._opts.hard_cooldown_s
        )
        if old.language:
            self._cooldown_until[old.language] = self._last_audio_ts + cooldown
        self._cooldown_until.pop(new_language.language, None)

    def on_switch_failed(self, target: LanguageCode) -> None:
        self._switch_inflight = False
        self._evidence.pop(target.language, None)
        self._cooldown_until[target.language] = self._last_audio_ts + self._opts.hard_cooldown_s

    def evidence_snapshot_if_due(self) -> dict[str, float] | None:
        if not self._evidence:
            return None

        if self._last_audio_ts - self._last_snapshot_ts < self._opts.evidence_event_interval_s:
            return None

        self._last_snapshot_ts = self._last_audio_ts
        snapshot: dict[str, float] = {}
        for base, evidence in self._evidence.items():
            self._decay(evidence, self._last_audio_ts)
            snapshot[base] = round(evidence.score, 4)
        return snapshot

    def _decay(self, evidence: _Evidence, audio_ts: float) -> None:
        elapsed = audio_ts - evidence.last_update_audio_ts
        if elapsed > 0 and self._opts.evidence_half_life_s > 0:
            evidence.score *= 0.5 ** (elapsed / self._opts.evidence_half_life_s)
        evidence.last_update_audio_ts = max(evidence.last_update_audio_ts, audio_ts)

    def _reentry_multiplier(self, base: str, audio_ts: float) -> float:
        cooldown_end = self._cooldown_until.get(base)
        if cooldown_end is None:
            return 1.0

        opts = self._opts
        if audio_ts < cooldown_end:
            return opts.reentry_threshold_multiplier

        elapsed = audio_ts - cooldown_end
        if elapsed >= opts.reentry_decay_s or opts.reentry_decay_s <= 0:
            del self._cooldown_until[base]
            return 1.0

        progress = elapsed / opts.reentry_decay_s
        return opts.reentry_threshold_multiplier + progress * (
            1.0 - opts.reentry_threshold_multiplier
        )

    def _word_language_fraction(self, words: object, base: str) -> float | None:
        """Fraction of language-tagged words in ``base``; None when no tags available."""
        if not isinstance(words, list) or not words:
            return None

        tagged = 0
        matching = 0
        for word in words:
            lang: NotGivenOr[str] = getattr(word, "language", None) or None  # type: ignore[assignment]
            if lang is None or not is_given(lang) or not lang:
                continue
            tagged += 1
            if LanguageCode(lang).language == base:
                matching += 1

        if tagged == 0:
            return None
        return matching / tagged
