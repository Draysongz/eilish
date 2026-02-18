"""Logging configuration for the trading bot."""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "forex_bot",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Configure a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, logs to logs/bot.log)
        level: Logging level (default: INFO)
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatter
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = _ColorFormatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        enable_color=_should_colorize(),
    )
    
    # File handler
    if log_file is None:
        log_file = Path("logs") / "bot.log"
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "forex_bot") -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str) -> None:
    """Emit a visually grouped section header for console logs."""
    line = "═" * max(10, 64 - len(title))
    logger.info("%s %s", title, line)


class _ColorFormatter(logging.Formatter):
    _RESET = "\x1b[0m"
    _LEVEL_COLORS = {
        "DEBUG": "\x1b[38;5;245m",
        "INFO": "\x1b[38;5;250m",
        "WARNING": "\x1b[38;5;214m",
        "ERROR": "\x1b[38;5;196m",
        "CRITICAL": "\x1b[38;5;196m\x1b[1m",
    }
    _KEY_COLOR = "\x1b[38;5;245m"
    _VALUE_COLOR = "\x1b[38;5;221m"
    _SYMBOL_COLOR = "\x1b[38;5;141m"
    _SECTION_COLOR = "\x1b[38;5;178m"
    _LEVEL_ICONS = {
        "DEBUG": "▸",
        "INFO": "●",
        "WARNING": "▲",
        "ERROR": "■",
        "CRITICAL": "✖",
    }

    def __init__(self, *args, enable_color: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enable_color = enable_color

    def format(self, record: logging.LogRecord) -> str:
        icon = self._LEVEL_ICONS.get(record.levelname, "")
        original_msg = record.getMessage()
        if icon:
            record.message = f"{icon} {original_msg}"
        message = super().format(record)
        record.message = original_msg
        if not self._enable_color:
            return message

        parts = message.split(" | ", 2)
        if len(parts) != 3:
            color = self._LEVEL_COLORS.get(record.levelname, "")
            return f"{color}{message}{self._RESET}" if color else message

        timestamp, level, msg = parts
        level_color = self._LEVEL_COLORS.get(record.levelname, "")
        if level_color:
            level = f"{level_color}{level}{self._RESET}"

        msg = self._colorize_message(msg)
        return f"{timestamp} | {level} | {msg}"

    def _colorize_message(self, msg: str) -> str:
        if msg.startswith("BOT ") or msg.startswith("MT5 ") or msg.startswith("AI ") or msg.startswith("TRADING "):
            return f"{self._SECTION_COLOR}{msg}{self._RESET}"

        msg = re.sub(
            r"\[(?P<symbol>[A-Z0-9_]+)\]",
            lambda match: f"{self._SYMBOL_COLOR}[{match.group('symbol')}]{self._RESET}",
            msg,
        )

        def repl(match: re.Match) -> str:
            key = match.group("key")
            val = match.group("val")
            return f"{self._KEY_COLOR}{key}{self._RESET}={self._VALUE_COLOR}{val}{self._RESET}"

        msg = re.sub(r"(?P<key>[a-zA-Z_]+)=(?P<val>[^\s|]+)", repl, msg)
        return msg


def _should_colorize() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()
