"""
Scholarship Voice Assistant - Logging Module
=============================================
Structured logging with colors for console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorlog for colored output
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

class ScholarshipLogger:
    """Custom logger with colored console output and file logging."""
    
    _instance: Optional['ScholarshipLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger("scholarship_assistant")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if HAS_COLORLOG:
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s",
                datefmt="%H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def add_file_handler(self, log_dir: Path):
        """Add file handler for persistent logging."""
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"scholarship_assistant_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.info(f"📁 Logging to file: {log_file}")
    
    def set_level(self, level: str):
        """Set logging level from string."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level_map.get(level.upper(), logging.INFO))
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    # Convenience methods for voice assistant events
    def user_speech(self, text: str):
        """Log user speech transcription."""
        self.info(f"🎤 USER: {text}")
    
    def assistant_response(self, text: str):
        """Log assistant response."""
        self.info(f"🤖 ASSISTANT: {text}")
    
    def rag_query(self, query: str, num_results: int):
        """Log RAG retrieval."""
        self.debug(f"🔍 RAG Query: '{query}' → {num_results} results")
    
    def api_call(self, service: str, endpoint: str):
        """Log external API call."""
        self.debug(f"🌐 API Call: {service} → {endpoint}")
    
    def latency(self, component: str, ms: float):
        """Log component latency."""
        emoji = "⚡" if ms < 200 else "🐢" if ms > 500 else "⏱️"
        self.debug(f"{emoji} {component} latency: {ms:.0f}ms")
    
    def connection_event(self, event: str, details: str = ""):
        """Log connection events."""
        self.info(f"🔗 {event}: {details}")
    
    def error_with_context(self, component: str, error: Exception, context: str = ""):
        """Log error with component context."""
        self.error(f"❌ [{component}] {type(error).__name__}: {error}")
        if context:
            self.error(f"   Context: {context}")

# Global logger instance
_logger: Optional[ScholarshipLogger] = None

def get_logger() -> ScholarshipLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = ScholarshipLogger()
    return _logger

def setup_logging(level: str = "INFO", log_to_file: bool = False, log_dir: Optional[Path] = None):
    """
    Initialize logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to also log to file
        log_dir: Directory for log files (default: ./logs)
    """
    logger = get_logger()
    logger.set_level(level)
    
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        logger.add_file_handler(log_dir)
    
    logger.info("🚀 Scholarship Voice Assistant - Logging initialized")
    return logger
