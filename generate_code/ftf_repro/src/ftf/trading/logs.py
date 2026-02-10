"""ftf.trading.logs

Structured trading/event logging.

The reproduction plan requires per-day logging (handled by the engine output
DataFrame) *and* a trade-event log capturing state transitions (entries/exits,
stop triggers, derisking).

This module provides a lightweight, deterministic log object that can be
attached to an EngineResult and serialized to JSON.

Design goals:
- minimal dependencies
- stable schema for tests
- append-only semantics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd


EventType = Literal[
    "ENTRY",
    "EXIT_HARD_STOP",
    "EXIT_TRAILING_STOP",
    "EXIT_TIMEOUT",
    "EXIT_DERISK_CLOSE",
    "DERISK_HALF_ON",
    "DERISK_HALF_OFF",
    "FLAT",
]


@dataclass
class TradeEvent:
    """A single discrete trade/state-machine event."""

    date: str
    event: EventType
    price: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingLog:
    """Append-only structured event log.

    Attributes
    ----------
    header:
        Metadata about the run/anchor/frozen parameters.
    events:
        List of TradeEvent dictionaries.
    """

    header: Dict[str, Any] = field(default_factory=dict)
    events: List[TradeEvent] = field(default_factory=list)

    def add(
        self,
        date: pd.Timestamp,
        event: EventType,
        *,
        price: Optional[float] = None,
        **info: Any,
    ) -> None:
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        self.events.append(
            TradeEvent(date=date.strftime("%Y-%m-%d"), event=event, price=price, info=dict(info))
        )

    def extend(self, events: Iterable[TradeEvent]) -> None:
        for e in events:
            if not isinstance(e, TradeEvent):
                raise TypeError("extend expects an iterable of TradeEvent")
            self.events.append(e)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": dict(self.header),
            "events": [asdict(e) for e in self.events],
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert event log to a DataFrame for analysis/reporting."""

        if not self.events:
            return pd.DataFrame(columns=["date", "event", "price"])  # empty
        rows: List[Dict[str, Any]] = []
        for e in self.events:
            row = {"date": e.date, "event": e.event, "price": e.price}
            row.update(e.info or {})
            rows.append(row)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df


__all__ = ["EventType", "TradeEvent", "TradingLog"]
