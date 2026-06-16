"""
logger.py
---------
Centralised logging setup for all SignBridge modules.

Features
--------
* Colored console output (INFO=green, WARNING=yellow, ERROR=red, DEBUG=cyan)
* Optional rotating file handler that writes to PROJECT/logs/
* Single call get_logger(__name__) from every module — no config duplication
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI color codes for console output
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


class _ColorFormatter(logging.Formatter):
    """Formatter that prepends ANSI color to the level name."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color  = _COLORS.get(record.levelname, _RESET)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        formatter = logging.Formatter(self.FMT, datefmt=self.DATEFMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    FMT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# ---------------------------------------------------------------------------
# Module-level state — track which loggers are already configured
# ---------------------------------------------------------------------------
_configured_loggers: set[str] = set()
_root_configured = False


def _configure_root(log_level: int, log_dir: Path | None) -> None:
    """Configure the root logger once per process."""
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(_ColorFormatter())
    root.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "signbridge.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(_PlainFormatter())
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("mediapipe", "absl", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _root_configured = True


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_dir: Path | None = None,
) -> logging.Logger:
    """
    Return a named logger.  On first call also configures the root logger.

    Parameters
    ----------
    name : str
        Logger name — use __name__ in every calling module.
    log_level : int
        Logging level (default INFO).  Set to logging.DEBUG for verbose output.
    log_dir : Path | None
        If provided, logs are also written to a rotating file in this directory.
        Pass PROJECT/logs/ from the top-level scripts.

    Returns
    -------
    logging.Logger
    """
    # Try to resolve default log_dir from project structure
    if log_dir is None:
        try:
            from src.utils.config_loader import get_project_root
            log_dir = get_project_root() / "logs"
        except Exception:
            log_dir = None

    _configure_root(log_level, log_dir)
    return logging.getLogger(name)
