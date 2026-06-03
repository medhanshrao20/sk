"""
Log pipeline failures and allow steps / seasons to continue after errors.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from config import PIPELINE_ERROR_LOG

T = TypeVar("T")

DEFAULT_ERROR_LOG = PIPELINE_ERROR_LOG


class PipelineErrorLogger:
    """Append structured errors to a log file (safe for overnight runs)."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or DEFAULT_ERROR_LOG
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        step: str,
        exc: BaseException,
        *,
        season: str | None = None,
        extra: str | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header = f"[{ts}]"
        if season:
            header += f" season={season}"
        header += f" step={step}"
        if extra:
            header += f" ({extra})"

        block = "\n".join(
            [
                "",
                "=" * 72,
                header,
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
                "=" * 72,
            ]
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(block)

        where = f"{season} / {step}" if season else step
        print(f"[ERROR] {where} — {type(exc).__name__}: {exc}")
        print(f"         Logged to {self.log_path}")

    def log_message(self, message: str, *, season: str | None = None) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        prefix = f"[{ts}]"
        if season:
            prefix += f" season={season}"
        line = f"{prefix} {message}\n"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        print(message)


def run_step(
    logger: PipelineErrorLogger,
    season_name: str,
    step: str,
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    extra: str | None = None,
    **kwargs: Any,
) -> tuple[bool, T | None]:
    """
    Run one pipeline step; on failure log and return (False, default).

    Returns (True, result) on success.
    """
    print(f"\n[{season_name}] >>> {step}")
    try:
        return True, func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — intentional for overnight resilience
        logger.log(step, exc, season=season_name, extra=extra)
        return False, default
