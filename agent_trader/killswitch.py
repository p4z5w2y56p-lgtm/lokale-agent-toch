"""Global stop. If it is engaged, nothing trades. Humans own this switch."""
from __future__ import annotations

from pathlib import Path


class KillSwitch:
    def __init__(self, path: str | Path = "KILL") -> None:
        self.path = Path(path)
        self._flag = False

    def engaged(self) -> bool:
        return self._flag or self.path.exists()

    def engage(self, reason: str = "manual") -> None:
        self._flag = True
        self.path.write_text(f"{reason}\n")

    def release(self) -> None:
        self._flag = False
        self.path.unlink(missing_ok=True)
