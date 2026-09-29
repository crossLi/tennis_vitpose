from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_RETENTION_DAYS_ENV = "TENNIS_LOG_RETENTION_DAYS"
DEFAULT_RETENTION_DAYS = 7

_CONFIGURED = False
_TASK_LOGGERS: dict[str, logging.Logger] = {}
_LAST_CLEANUP_TS: float = 0.0
_CLEANUP_INTERVAL_SEC: float = 3 * 60 * 60


def retention_days() -> int:
    raw = os.getenv(_LOG_RETENTION_DAYS_ENV)
    if raw is None:
        return DEFAULT_RETENTION_DAYS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS
    return max(1, val)


def _cleanup_expired_logs(force: bool = False) -> None:
    global _LAST_CLEANUP_TS
    now_ts = time.time()
    if not force and now_ts - _LAST_CLEANUP_TS < _CLEANUP_INTERVAL_SEC:
        return
    days = retention_days()
    cutoff_ts = now_ts - days * 86400
    removed = 0
    try:
        for p in LOG_DIR.iterdir():
            if not p.is_file():
                continue
            if not (p.name == "app.log" or p.name.startswith("app.log.") or p.name.startswith("task_") or p.name.startswith("task_")):
                pass
            if not (p.suffix == ".log" or p.name.startswith("app.log.") or p.name.startswith("task_")):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff_ts:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    continue
    except OSError:
        pass
    _LAST_CLEANUP_TS = now_ts
    if removed > 0:
        sys.stderr.write(
            f"[tennis_coach.logging_utils] cleaned {removed} log files older than {days} days\n"
        )
        sys.stderr.flush()


def _build_filter(prefix: str) -> logging.Filter:
    _prefix = prefix

    class PrefixFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            applied = getattr(record, "_task_prefix_applied", None)
            if applied:
                return True
            if not record.msg.startswith(_prefix):
                record.msg = f"{_prefix} {record.msg}"
            record._task_prefix_applied = True  # type: ignore[attr-defined]
            return True

    return PrefixFilter()


def ensure_logging_configured(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    days = retention_days()
    if days != DEFAULT_RETENTION_DAYS:
        sys.stderr.write(f"[tennis_coach.logging_utils] log_retention_days={days} (from env {_LOG_RETENTION_DAYS_ENV})\n")
    else:
        sys.stderr.write(f"[tennis_coach.logging_utils] log_retention_days={days} (default)\n")
    sys.stderr.flush()
    _cleanup_expired_logs(force=True)

    main_log_path = LOG_DIR / "app.log"
    app_logger = logging.getLogger("tennis_coach")
    app_logger.setLevel(numeric_level)
    app_logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "_tennis_coach", False) for h in app_logger.handlers):
        file_handler = RotatingFileHandler(
            main_log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        file_handler._tennis_coach = True  # type: ignore[attr-defined]
        app_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_tennis_coach", False) for h in app_logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        stream_handler._tennis_coach = True  # type: ignore[attr-defined]
        app_logger.addHandler(stream_handler)

    uvicorn_access = logging.getLogger("uvicorn.access")
    try:
        uvicorn_access.setLevel(logging.WARNING)
    except Exception:
        pass
    _CONFIGURED = True


def get_app_logger(name: str = "app") -> logging.Logger:
    ensure_logging_configured()
    _cleanup_expired_logs()
    return logging.getLogger(f"tennis_coach.{name}")


def task_log_prefix(task_id: str, stage: str | None = None) -> str:
    if stage:
        return f"[task_id={task_id} stage={stage}]"
    return f"[task_id={task_id}]"


def get_task_logger(task_id: str, stage: str | None = None) -> logging.Logger:
    ensure_logging_configured()
    _cleanup_expired_logs()
    logger = _TASK_LOGGERS.get(task_id)
    if logger is None:
        logger = logging.getLogger(f"tennis_coach.task.{task_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        task_log_path = LOG_DIR / f"task_{task_id}.log"
        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)
        if not any(
            isinstance(h, RotatingFileHandler) and getattr(h, "_task_file", None) == task_id
            for h in logger.handlers
        ):
            fh = RotatingFileHandler(
                task_log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            fh._task_file = task_id  # type: ignore[attr-defined]
            logger.addHandler(fh)
        logger.addFilter(_build_filter(task_log_prefix(task_id)))
        _TASK_LOGGERS[task_id] = logger
    return logger


def log_with_task(
    logger: logging.Logger,
    task_id: str,
    level: int,
    msg: str,
    *args: Any,
    stage: str | None = None,
    duration_ms: int | float | None = None,
    **extra: Any,
) -> None:
    ensure_logging_configured()
    _cleanup_expired_logs()
    if stage:
        prefix = f"[stage={stage}]"
    else:
        prefix = ""
    parts = [msg]
    if duration_ms is not None:
        if isinstance(duration_ms, float):
            parts.append(f"duration={duration_ms:.1f}ms")
        else:
            parts.append(f"duration={int(duration_ms)}ms")
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    final_msg = " ".join(parts)
    if prefix:
        final_msg = f"{prefix} {final_msg}"
    logger.log(level, final_msg, *args)
