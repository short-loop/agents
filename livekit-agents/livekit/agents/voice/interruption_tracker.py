from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

from ..log import logger


class InterruptionMode(enum.Enum):
    NORMAL = "normal"
    PRIMED = "primed"
    TRANSIENT = "transient"
    SUSTAINED = "sustained"


@dataclass(frozen=True)
class InterruptionModeSettings:
    backoff_delay: float
    """Endpointing delay (s) applied when the EOU model is below ``unlikely_threshold``
    (or unavailable) while this mode is active."""
    unlikely_threshold: float
    """Minimum end-of-utterance probability required for a fast commit while this mode
    is active. Combined with the turn detector's own threshold via ``max()``."""
    silence_gate: float
    """Accumulated user silence (s) required before agent speech playout may start
    while this mode is active. Requires VAD."""
    disable_preemptive: bool = False
    """Disable preemptive generation while this mode is active."""


@dataclass(frozen=True)
class InterruptionBackoffOptions:
    transient_entry_count: int = 2
    """Enter transient mode when at least this many interruptions occurred within the
    last ``transient_entry_window`` user turns."""
    transient_entry_window: int = 5
    """Number of most recent user turns considered for transient entry."""
    transient_exit_clean_turns: int = 3
    """Exit transient mode once this many consecutive user turns had no interruptions."""
    sustained_entry_total: int = 4
    """Enter sustained mode (sticky for the session) once total interruptions reach this."""
    primed_silence_gate: float | None = None
    """When set, the first recorded interruption puts the session in a sticky "primed"
    state that applies only this playout silence gate (seconds) — no endpointing backoff.
    Covers collisions that occur before/between transient episodes at near-zero latency
    cost. ``None`` disables the primed state."""
    transient: InterruptionModeSettings = field(
        default_factory=lambda: InterruptionModeSettings(
            backoff_delay=4.0, unlikely_threshold=0.3, silence_gate=1.0
        )
    )
    sustained: InterruptionModeSettings = field(
        default_factory=lambda: InterruptionModeSettings(
            backoff_delay=6.0, unlikely_threshold=0.5, silence_gate=2.0, disable_preemptive=True
        )
    )


class InterruptionTracker:
    """Tracks user-perceived interruptions of agent speech and derives the active
    interruption-backoff mode.

    An interruption is recorded when an assistant conversation item is committed with
    ``interrupted=True`` and non-empty spoken text. Mode transitions are evaluated at
    user-turn boundaries (when a user conversation item is committed).
    """

    def __init__(self, opts: InterruptionBackoffOptions | None) -> None:
        self._opts = opts
        self._mode = InterruptionMode.NORMAL
        self._total = 0
        self._pending = 0  # interruptions since the last user-turn commit
        maxlen = 1
        if opts is not None:
            maxlen = max(opts.transient_entry_window, opts.transient_exit_clean_turns)
        self._window: deque[int] = deque(maxlen=maxlen)  # per-user-turn interruption counts

    @property
    def enabled(self) -> bool:
        return self._opts is not None

    @property
    def mode(self) -> InterruptionMode:
        return self._mode

    @property
    def total_interruptions(self) -> int:
        return self._total

    def record_interruption(self) -> None:
        if self._opts is None:
            return

        self._total += 1
        self._pending += 1
        if self._mode is InterruptionMode.NORMAL and self._opts.primed_silence_gate is not None:
            # primed engages immediately (not at the next user-turn boundary) so the
            # playout gate already protects the very next reply
            self._log_transition(InterruptionMode.PRIMED, "primed_entry")
        logger.debug(
            "interruption recorded",
            extra={"total_interruptions": self._total, "mode": self._mode.value},
        )

    def record_user_turn(self) -> InterruptionMode:
        """Flush pending interruptions into the turn window and evaluate mode transitions.

        Returns the (possibly new) active mode.
        """
        if self._opts is None:
            return self._mode

        self._window.append(self._pending)
        self._pending = 0

        if (
            self._mode is not InterruptionMode.SUSTAINED
            and self._total >= self._opts.sustained_entry_total
        ):
            self._log_transition(InterruptionMode.SUSTAINED, "sustained_entry")
        elif self._mode is InterruptionMode.TRANSIENT:
            exit_turns = self._opts.transient_exit_clean_turns
            recent = list(self._window)[-exit_turns:]
            if len(recent) >= exit_turns and sum(recent) == 0:
                # fall back to primed (sticky once an interruption occurred) when
                # configured, otherwise all the way to normal
                if self._opts.primed_silence_gate is not None:
                    self._log_transition(InterruptionMode.PRIMED, "transient_exit")
                else:
                    self._log_transition(InterruptionMode.NORMAL, "transient_exit")
                # clear the window so interruptions older than the clean streak
                # cannot immediately re-trigger transient entry
                self._window.clear()
        elif self._mode in (InterruptionMode.NORMAL, InterruptionMode.PRIMED):
            entry_window = self._opts.transient_entry_window
            recent = list(self._window)[-entry_window:]
            if sum(recent) >= self._opts.transient_entry_count:
                self._log_transition(InterruptionMode.TRANSIENT, "transient_entry")

        return self._mode

    def _log_transition(self, new_mode: InterruptionMode, reason: str) -> None:
        old_mode = self._mode
        self._mode = new_mode
        if new_mode is not old_mode:
            logger.info(
                "interruption mode changed",
                extra={
                    "old_mode": old_mode.value,
                    "new_mode": new_mode.value,
                    "reason": reason,
                    "total_interruptions": self._total,
                    "window": list(self._window),
                },
            )

    def _active_settings(self) -> InterruptionModeSettings | None:
        if self._opts is None:
            return None
        if self._mode is InterruptionMode.TRANSIENT:
            return self._opts.transient
        if self._mode is InterruptionMode.SUSTAINED:
            return self._opts.sustained
        return None

    def backoff_params(self) -> tuple[float, float] | None:
        """Returns ``(unlikely_threshold, backoff_delay)`` for the active mode, or
        ``None`` when disabled or in normal mode."""
        settings = self._active_settings()
        if settings is None:
            return None
        return settings.unlikely_threshold, settings.backoff_delay

    def silence_gate(self) -> float | None:
        """Returns the active mode's playout silence gate in seconds, or ``None`` when
        disabled or in normal mode."""
        if self._opts is not None and self._mode is InterruptionMode.PRIMED:
            return self._opts.primed_silence_gate
        settings = self._active_settings()
        if settings is None:
            return None
        return settings.silence_gate

    def preemptive_disabled(self) -> bool:
        settings = self._active_settings()
        return settings is not None and settings.disable_preemptive
