from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...language import LanguageCode

SwitchInitiator = Literal["heuristic", "manual"]
SwitchExecutorKind = Literal["in_place", "recreate"]


@dataclass(frozen=True)
class LanguageSwitchStartedEvent:
    """Emitted before the switch begins, so the app can retune TTS/prompt before the
    agent's next generation."""

    old_language: LanguageCode
    new_language: LanguageCode
    initiator: SwitchInitiator
    reason: str


@dataclass(frozen=True)
class LanguageSwitchedEvent:
    """Emitted once the new primary connection owns the transcript stream again."""

    old_language: LanguageCode
    new_language: LanguageCode
    initiator: SwitchInitiator
    executor: SwitchExecutorKind
    latency: float
    """Wall-clock seconds from first heuristic evidence to switch completion (0.0 for manual)."""
    trigger_transcript: str | None
    """The detector transcript that triggered the heuristic decision, if any."""


@dataclass(frozen=True)
class LanguageSwitchSuppressedEvent:
    """Emitted when a switch would have fired but was blocked."""

    target_language: LanguageCode
    reason: Literal[
        "cooldown",
        "allowlist",
        "auto_switch_disabled",
        "switch_in_progress",
        "switch_failed",
        "no_executor",
    ]
    score: float


@dataclass(frozen=True)
class LanguageSwitchEvidenceEvent:
    """Throttled snapshot of the heuristic engine's per-language evidence scores."""

    current_language: LanguageCode
    audio_time: float
    scores: dict[str, float] = field(default_factory=dict)
