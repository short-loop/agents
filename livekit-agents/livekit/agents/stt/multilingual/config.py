from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...language import LanguageCode


@dataclass
class LanguageSwitchOptions:
    """Configuration for :class:`MultilingualAdapter` language switching.

    All time constants that relate to speech are measured in *audio time* (seconds of
    audio pushed to the adapter), not wall-clock time: the input stream is gapped while
    the agent speaks, so wall-clock timers would silently expire during those gaps.
    """

    languages: list[str] | None = None
    """Allowlist of languages the adapter may switch to. ``None`` allows any language.
    Evidence for non-allowlisted languages is still tracked and observable via
    ``language_switch_evidence`` / ``language_switch_suppressed`` events, but never acted on."""

    auto_switch: bool = True
    """When ``False`` the heuristic engine runs in telemetry-only mode: evidence and
    suppressed events are emitted but no automatic switch is performed. Manual switching
    via :meth:`MultilingualAdapter.switch_language` is unaffected."""

    switch_mode: Literal["auto", "in_place", "recreate"] = "auto"
    """How the primary connection is moved to a new language. ``in_place`` requires the
    primary stream to expose ``update_options(language=...)`` (e.g. Deepgram). ``recreate``
    requires ``primary_factory``. ``auto`` probes for in-place support and falls back to
    the factory."""

    switch_threshold: float = 3.0
    """Accumulated evidence score required to trigger a switch."""

    min_detector_confidence: float = 0.6
    """Detector results below this transcription confidence contribute no evidence."""

    default_confidence: float = 0.7
    """Confidence assumed when the detector reports ``0.0`` (i.e. not provided)."""

    min_transcript_length: int = 3
    """Transcripts of at most this many characters are ignored (too ambiguous)."""

    interim_evidence_weight: float = 0.0
    """Weight of interim transcripts as evidence. ``0.0`` (default) means finals only."""

    evidence_half_life_s: float = 20.0
    """Evidence decays exponentially with this half-life (audio time)."""

    script_mismatch_boost: float = 1.5
    """Evidence multiplier when the detector text's Unicode script differs from the
    current language's expected script (cross-script pairs only)."""

    turn_bonus: float = 1.0
    """Extra evidence added for the second and subsequent consecutive turns in the
    same non-primary language."""

    word_length_cap: int = 8
    """Word count at which an utterance contributes full length weight."""

    hard_cooldown_s: float = 10.0
    """After a heuristic switch, no switch back to the previous language for this long."""

    manual_cooldown_s: float = 30.0
    """After a manual switch, no heuristic switch back for this long (manual intent is
    stickier than acoustic evidence)."""

    reentry_threshold_multiplier: float = 2.0
    """After a cooldown expires, the switched-away language needs this multiple of
    ``switch_threshold``, decaying linearly back to 1x over ``reentry_decay_s``."""

    reentry_decay_s: float = 60.0
    """Time for the re-entry threshold multiplier to decay back to 1x (audio time)."""

    boundary_silence_s: float = 0.8
    """During the transition window, this much audio-time silence after a non-empty
    detector final counts as an utterance boundary (transition end)."""

    max_detector_owns_s: float = 15.0
    """Hard cap on the transition window; the flip back to the primary happens after
    this long even if no utterance boundary was observed."""

    switch_timeout_s: float = 10.0
    """Max time for the switch executor to produce the new primary stream before rollback."""

    switch_grace_s: float = 2.0
    """After the executor completes, wait up to this long for a first transcript (or a
    failure) from the new stream. Silence within the grace period counts as healthy —
    the user may simply not be speaking (e.g. a manual switch right after an agent turn)."""

    evidence_event_interval_s: float = 2.0
    """Minimum audio-time interval between ``language_switch_evidence`` events."""

    detector_restart: bool = True
    """Restart the detector stream (with backoff) if it fails mid-session."""


class LanguageNotAllowedError(ValueError):
    """Raised when a requested language is not in the configured allowlist."""

    def __init__(self, language: LanguageCode, allowed: list[str]) -> None:
        self.language = language
        self.allowed = allowed
        super().__init__(f"language {str(language)!r} is not in the allowed languages: {allowed}")


class LanguageSwitchFailedError(RuntimeError):
    """Raised when a language switch could not be completed and was rolled back."""

    def __init__(self, target: LanguageCode, reason: str) -> None:
        self.target = target
        self.reason = reason
        super().__init__(f"failed to switch language to {str(target)!r}: {reason}")
