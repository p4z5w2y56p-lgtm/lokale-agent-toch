"""Global stop. While engaged, nothing can buy; sells that cut risk stay allowed.

Humans own the file switch (`kill` / `unkill`). The safety rules can also `trip` it
in memory for the current run only, so a bad replay never blocks the next one."""
from __future__ import annotations

from pathlib import Path


class KillSwitch:
    def __init__(self, path: str | Path = "KILL") -> None:
        self.path = Path(path)
        self._flag = False
        self.reason: str | None = None

    def engaged(self) -> bool:
        return self._flag or self.path.exists()

    def trip(self, reason: str) -> None:
        """Automatic halt for this process only (no file written)."""
        if not self._flag:
            self._flag, self.reason = True, reason

    def engage(self, reason: str = "manual") -> None:
        self._flag, self.reason = True, reason
        self.path.write_text(f"{reason}\n")

    def release(self) -> None:
        self._flag, self.reason = False, None
        self.path.unlink(missing_ok=True)
