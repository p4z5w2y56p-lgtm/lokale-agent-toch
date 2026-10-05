"""Append-only JSONL log of everything that happens. The trade diary (and tax record)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def _default(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o):
        return asdict(o)
    return str(o)


class Journal:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data: Any) -> None:
        record = {"event": event, "wall_time": time.time(), **data}
        self.events.append(record)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, default=_default) + "\n")
