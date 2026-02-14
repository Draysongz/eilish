"""Tests for logging functionality."""
import logging
from pathlib import Path
import tempfile

from src.logger import setup_logger, get_logger


def test_setup_logger_creates_file():
    """Test that logger creates log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logger("test_logger", log_file=log_file)
        
        logger.info("Test message")
        
        # Close handlers before cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


def test_setup_logger_console_output(caplog):
    """Test that logger outputs to console."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logger("test_console", log_file=log_file, console=True)
        
        with caplog.at_level(logging.INFO):
            logger.info("Console test")
        
        # Close handlers before cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        assert "Console test" in caplog.text


def test_get_logger_returns_same_instance():
    """Test that get_logger returns the same logger instance."""
    logger1 = get_logger("test_instance")
    logger2 = get_logger("test_instance")
    
    assert logger1 is logger2
