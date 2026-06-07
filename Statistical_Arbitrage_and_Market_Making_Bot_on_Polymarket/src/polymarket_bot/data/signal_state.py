from __future__ import annotations
from dataclasses import dataclass, field
from time import time
from typing import Optional
from signal_engine import MarketSignal, SignalType


@dataclass
class SignalTracker:
    asset_id: str
    current_signal: SignalType = SignalType.NEUTRAL
    previous_signal: SignalType = SignalType.NEUTRAL
    consecutive_count: int = 0
    first_seen_ts: Optional[float] = None
    last_seen_ts: Optional[float] = None
    last_signal: Optional[MarketSignal] = None

    def update(self, signal: MarketSignal) -> None:
        now = time()

        self.previous_signal = self.current_signal

        if signal.signal == self.current_signal:
            self.consecutive_count += 1
        else:
            self.current_signal = signal.signal
            self.consecutive_count = 1
            self.first_seen_ts = now

        self.last_seen_ts = now
        self.last_signal = signal

    @property
    def is_active(self) -> bool:
        return self.current_signal != SignalType.NEUTRAL

    @property
    def age_seconds(self) -> float:
        if self.first_seen_ts is None:
            return 0.0
        return time() - self.first_seen_ts


@dataclass
class SignalState:
    trackers: dict[str, SignalTracker] = field(default_factory=dict)

    def update(self, signal: MarketSignal) -> SignalTracker:
        tracker = self.trackers.get(signal.asset_id)

        if tracker is None:
            tracker = SignalTracker(asset_id=signal.asset_id)
            self.trackers[signal.asset_id] = tracker

        tracker.update(signal)
        return tracker

    def active_trackers(self, min_consecutive_count: int = 2,) -> list[SignalTracker]:
        return [tracker for tracker in self.trackers.values() if tracker.is_active 
                and tracker.consecutive_count >= min_consecutive_count]