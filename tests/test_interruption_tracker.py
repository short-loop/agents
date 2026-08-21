from livekit.agents.voice.interruption_tracker import (
    InterruptionBackoffOptions,
    InterruptionMode,
    InterruptionModeSettings,
    InterruptionTracker,
)

OPTS = InterruptionBackoffOptions(
    transient_entry_count=2,
    transient_entry_window=5,
    transient_exit_clean_turns=3,
    sustained_entry_total=4,
    transient=InterruptionModeSettings(backoff_delay=4.0, unlikely_threshold=0.3, silence_gate=1.0),
    sustained=InterruptionModeSettings(
        backoff_delay=6.0, unlikely_threshold=0.5, silence_gate=2.0, disable_preemptive=True
    ),
)


def _turn(tracker: InterruptionTracker, interruptions: int = 0) -> InterruptionMode:
    for _ in range(interruptions):
        tracker.record_interruption()
    return tracker.record_user_turn()


def test_normal_by_default() -> None:
    tracker = InterruptionTracker(OPTS)
    assert tracker.mode is InterruptionMode.NORMAL
    assert tracker.backoff_params() is None
    assert tracker.silence_gate() is None
    assert not tracker.preemptive_disabled()


def test_disabled_never_leaves_normal() -> None:
    tracker = InterruptionTracker(None)
    assert not tracker.enabled
    for _ in range(10):
        tracker.record_interruption()
        assert tracker.record_user_turn() is InterruptionMode.NORMAL
    assert tracker.total_interruptions == 0
    assert tracker.backoff_params() is None
    assert tracker.silence_gate() is None
    assert not tracker.preemptive_disabled()


def test_transient_entry() -> None:
    tracker = InterruptionTracker(OPTS)
    assert _turn(tracker, interruptions=1) is InterruptionMode.NORMAL
    assert _turn(tracker) is InterruptionMode.NORMAL
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT
    assert tracker.backoff_params() == (0.3, 4.0)
    assert tracker.silence_gate() == 1.0
    assert not tracker.preemptive_disabled()


def test_single_interruption_stays_normal() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=1)
    for _ in range(10):
        assert _turn(tracker) is InterruptionMode.NORMAL


def test_transient_window_expiry() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=1)
    for _ in range(5):
        assert _turn(tracker) is InterruptionMode.NORMAL
    # the first interruption slid out of the 5-turn window
    assert _turn(tracker, interruptions=1) is InterruptionMode.NORMAL


def test_transient_exit_hysteresis() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=1)
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT
    # two clean turns are not enough to exit
    assert _turn(tracker) is InterruptionMode.TRANSIENT
    assert _turn(tracker) is InterruptionMode.TRANSIENT
    # third clean turn exits
    assert _turn(tracker) is InterruptionMode.NORMAL
    # window was cleared on exit: the old interruptions must not re-trigger entry
    assert _turn(tracker) is InterruptionMode.NORMAL


def test_transient_reentry_after_exit() -> None:
    # raise the sustained threshold so 4 total interruptions don't escalate
    tracker = InterruptionTracker(InterruptionBackoffOptions(sustained_entry_total=100))
    _turn(tracker, interruptions=1)
    _turn(tracker, interruptions=1)
    for _ in range(3):
        _turn(tracker)
    assert tracker.mode is InterruptionMode.NORMAL
    _turn(tracker, interruptions=1)
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT


def test_interruption_during_transient_resets_exit_streak() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=1)
    _turn(tracker, interruptions=1)
    _turn(tracker)
    _turn(tracker)
    # an interruption on the 3rd turn restarts the clean streak
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT
    _turn(tracker)
    _turn(tracker)
    assert tracker.mode is InterruptionMode.TRANSIENT
    assert _turn(tracker) is InterruptionMode.NORMAL


def test_sustained_entry_and_sticky() -> None:
    tracker = InterruptionTracker(OPTS)
    # spread interruptions so transient exits in between; total still accumulates
    for _ in range(3):
        _turn(tracker, interruptions=1)
        for _ in range(4):
            _turn(tracker)
    assert tracker.mode is InterruptionMode.NORMAL
    assert _turn(tracker, interruptions=1) is InterruptionMode.SUSTAINED
    assert tracker.backoff_params() == (0.5, 6.0)
    assert tracker.silence_gate() == 2.0
    assert tracker.preemptive_disabled()
    # sticky: clean turns never exit sustained
    for _ in range(20):
        assert _turn(tracker) is InterruptionMode.SUSTAINED


def test_sustained_overrides_transient() -> None:
    tracker = InterruptionTracker(OPTS)
    assert _turn(tracker, interruptions=4) is InterruptionMode.SUSTAINED


def test_multiple_interruptions_in_one_turn() -> None:
    tracker = InterruptionTracker(OPTS)
    assert _turn(tracker, interruptions=2) is InterruptionMode.TRANSIENT


def test_transient_escalates_to_sustained() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=1)
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT
    _turn(tracker, interruptions=1)
    assert _turn(tracker, interruptions=1) is InterruptionMode.SUSTAINED


PRIMED_OPTS = InterruptionBackoffOptions(primed_silence_gate=2.0)


def test_primed_disabled_by_default() -> None:
    tracker = InterruptionTracker(OPTS)  # primed_silence_gate=None
    tracker.record_interruption()
    assert tracker.mode is InterruptionMode.NORMAL
    _turn(tracker)
    assert tracker.mode is InterruptionMode.NORMAL
    assert tracker.silence_gate() is None


def test_primed_entry_on_first_interruption() -> None:
    tracker = InterruptionTracker(PRIMED_OPTS)
    tracker.record_interruption()
    # engages immediately, before any user-turn boundary
    assert tracker.mode is InterruptionMode.PRIMED
    assert tracker.silence_gate() == 2.0
    assert tracker.backoff_params() is None
    assert not tracker.preemptive_disabled()


def test_primed_sticky_through_clean_turns() -> None:
    tracker = InterruptionTracker(PRIMED_OPTS)
    tracker.record_interruption()
    for _ in range(20):
        assert _turn(tracker) is InterruptionMode.PRIMED
    assert tracker.silence_gate() == 2.0


def test_primed_to_transient_and_back() -> None:
    tracker = InterruptionTracker(PRIMED_OPTS)
    _turn(tracker, interruptions=1)
    assert tracker.mode is InterruptionMode.PRIMED
    assert _turn(tracker, interruptions=1) is InterruptionMode.TRANSIENT
    assert tracker.backoff_params() is not None
    # transient exits back to primed, not normal
    _turn(tracker)
    _turn(tracker)
    assert _turn(tracker) is InterruptionMode.PRIMED
    assert tracker.silence_gate() == 2.0
    assert tracker.backoff_params() is None


def test_primed_to_sustained() -> None:
    tracker = InterruptionTracker(PRIMED_OPTS)
    assert _turn(tracker, interruptions=4) is InterruptionMode.SUSTAINED
    assert tracker.preemptive_disabled()


def test_total_interruptions_counter() -> None:
    tracker = InterruptionTracker(OPTS)
    _turn(tracker, interruptions=2)
    _turn(tracker)
    _turn(tracker, interruptions=1)
    assert tracker.total_interruptions == 3
