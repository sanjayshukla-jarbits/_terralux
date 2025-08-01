"""
Logging Utilities for Orchestrator
==================================

This module provides enhanced logging capabilities for the orchestrator system,
including structured logging, performance monitoring, and context-aware loggers.

Key Features:
- Structured logging with context information
- Performance monitoring and timing
- Memory usage tracking
- Custom log formatters
- Log rotation and management
- Integration with pipeline execution context
"""

import logging
import logging.handlers
import sys
import time
import functools
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import json
import traceback

# Optional imports for monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        # Add context information to the log record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def update_context(self, context: Dict[str, Any]):
        """Update the context information."""
        self.context.update(context)


class JsonFormatter(logging.Formatter):
    """Custom formatter for JSON-structured logs."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context information if available
        for attr in ['pipeline_id', 'step_id', 'execution_id']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color codes
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add colors only if output is a terminal
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            return f"{color}{formatted}{reset}"
        else:
            return formatted


def setup_logger(name: str, 
                log_file: Optional[Union[str, Path]] = None,
                level: Union[int, str] = logging.INFO,
                format_type: str = 'standard',
                max_bytes: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5,
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_type: Format type ('standard', 'detailed', 'json', 'colored')
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger('MyStep', 'logs/mystep.log', level='DEBUG')
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Define formatters
    formatters = {
        'standard': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        'detailed': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        ),
        'json': JsonFormatter(),
        'colored': ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    }
    
    formatter = formatters.get(format_type, formatters['standard'])
    
    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console if available
        if format_type == 'colored' or (format_type == 'standard' and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()):
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = formatter
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging from configuration dictionary.
    
    Args:
        config: Logging configuration dictionary
        
    Example:
        >>> config = {
        ...     "version": 1,
        ...     "formatters": {
        ...         "standard": {
        ...             "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ...         }
        ...     },
        ...     "handlers": {
        ...         "console": {
        ...             "class": "logging.StreamHandler",
        ...             "level": "INFO",
        ...             "formatter": "standard"
        ...         }
        ...     },
        ...     "root": {
        ...         "level": "INFO",
        ...         "handlers": ["console"]
        ...     }
        ... }
        >>> configure_logging(config)
    """
    try:
        import logging.config
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def get_logger_with_context(name: str, context: Dict[str, Any]) -> logging.Logger:
    """
    Get logger with context information.
    
    Args:
        name: Logger name
        context: Context dictionary to include in logs
        
    Returns:
        Logger with context filter applied
        
    Example:
        >>> context = {"pipeline_id": "pipe123", "step_id": "step456"}
        >>> logger = get_logger_with_context("MyStep", context)
        >>> logger.info("Processing data")  # Will include context info
    """
    logger = logging.getLogger(name)
    
    # Add context filter
    context_filter = ContextFilter(context)
    logger.addFilter(context_filter)
    
    return logger


def log_execution_time(func: Optional[Callable] = None, 
                      logger: Optional[logging.Logger] = None,
                      message: Optional[str] = None) -> Union[Callable, float]:
    """
    Decorator or context manager for logging execution time.
    
    Args:
        func: Function to decorate (when used as decorator)
        logger: Logger to use
        message: Custom message for logging
        
    Returns:
        Decorated function or execution time
        
    Example:
        >>> @log_execution_time
        ... def process_data():
        ...     time.sleep(1)
        ...     return "done"
        
        >>> # Or as context manager
        >>> with log_execution_time(logger=my_logger, message="Processing"):
        ...     time.sleep(1)
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or logging.getLogger(f.__module__)
            func_message = message or f"Executing {f.__name__}"
            
            try:
                result = f(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.info(f"{func_message} completed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(f"{func_message} failed after {execution_time:.2f} seconds: {e}")
                raise
        
        return wrapper
    
    if func is None:
        # Used as context manager or with parameters
        return ExecutionTimer(logger, message)
    else:
        # Used as simple decorator
        return decorator(func)


class ExecutionTimer:
    """Context manager for timing execution."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, message: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.message = message or "Operation"
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"{self.message} started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.message} completed in {execution_time:.2f} seconds")
        else:
            self.logger.error(f"{self.message} failed after {execution_time:.2f} seconds")
        
        return False  # Don't suppress exceptions


def log_memory_usage(logger: Optional[logging.Logger] = None, 
                    message: str = "Memory usage") -> None:
    """
    Log current memory usage.
    
    Args:
        logger: Logger to use
        message: Message to include with memory info
        
    Example:
        >>> log_memory_usage(logger, "Before processing")
    """
    if not PSUTIL_AVAILABLE:
        if logger:
            logger.debug("psutil not available for memory monitoring")
        return
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        
        logger.info(f"{message}: Process={memory_mb:.1f}MB, System={system_memory_percent:.1f}% used")
        
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")


def create_log_handler(handler_type: str, 
                      config: Dict[str, Any]) -> logging.Handler:
    """
    Create a log handler based on type and configuration.
    
    Args:
        handler_type: Type of handler ('file', 'console', 'rotating', 'syslog')
        config: Handler configuration
        
    Returns:
        Configured log handler
        
    Example:
        >>> config = {"filename": "app.log", "level": "INFO"}
        >>> handler = create_log_handler("file", config)
    """
    if handler_type == 'console':
        handler = logging.StreamHandler()
    
    elif handler_type == 'file':
        filename = config.get('filename', 'app.log')
        handler = logging.FileHandler(filename)
    
    elif handler_type == 'rotating':
        filename = config.get('filename', 'app.log')
        max_bytes = config.get('max_bytes', 10 * 1024 * 1024)
        backup_count = config.get('backup_count', 5)
        handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=max_bytes, backupCount=backup_count
        )
    
    elif handler_type == 'syslog':
        address = config.get('address', '/dev/log')
        handler = logging.handlers.SysLogHandler(address=address)
    
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")
    
    # Configure handler
    level = config.get('level', 'INFO')
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    format_string = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    return handler


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.metrics[operation] = duration
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.2f} seconds")
        
        del self.start_times[operation]
        return duration
    
    def log_metric(self, name: str, value: Union[int, float], unit: str = "") -> None:
        """Log a custom metric."""
        self.metrics[name] = value
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"Metric '{name}': {value}{unit_str}")
    
    def log_system_resources(self) -> None:
        """Log current system resource usage."""
        if not PSUTIL_AVAILABLE:
            self.logger.debug("psutil not available for system monitoring")
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.log_metric("cpu_usage", cpu_percent, "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.log_metric("memory_usage", memory.percent, "%")
            self.log_metric("memory_available", memory.available / 1024 / 1024, "MB")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.log_metric("disk_usage", disk_percent, "%")
            
        except Exception as e:
            self.logger.warning(f"Failed to log system resources: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            'metrics': self.metrics.copy(),
            'summary_time': datetime.now().isoformat(),
            'active_timers': list(self.start_times.keys())
        }


def setup_pipeline_logging(pipeline_id: str, 
                          log_dir: Union[str, Path] = "logs",
                          level: Union[int, str] = logging.INFO) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for a pipeline execution.
    
    Args:
        pipeline_id: Unique pipeline identifier
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Dictionary of configured loggers
        
    Example:
        >>> loggers = setup_pipeline_logging("pipeline_123", "logs/")
        >>> loggers['main'].info("Pipeline started")
        >>> loggers['performance'].info("Memory usage: 150MB")
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Context for all pipeline loggers
    context = {'pipeline_id': pipeline_id}
    
    # Main pipeline logger
    main_logger = setup_logger(
        f"pipeline.{pipeline_id}",
        log_path / f"{pipeline_id}_main.log",
        level=level,
        format_type='detailed'
    )
    
    # Performance logger
    perf_logger = setup_logger(
        f"pipeline.{pipeline_id}.performance",
        log_path / f"{pipeline_id}_performance.log",
        level=level,
        format_type='json'
    )
    
    # Error logger
    error_logger = setup_logger(
        f"pipeline.{pipeline_id}.errors",
        log_path / f"{pipeline_id}_errors.log",
        level=logging.ERROR,
        format_type='detailed'
    )
    
    # Add context filters
    for logger in [main_logger, perf_logger, error_logger]:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return {
        'main': main_logger,
        'performance': perf_logger,
        'errors': error_logger
    }


# Export main functions
__all__ = [
    'setup_logger', 'configure_logging', 'get_logger_with_context',
    'log_execution_time', 'log_memory_usage', 'create_log_handler',
    'ContextFilter', 'JsonFormatter', 'ColoredFormatter',
    'ExecutionTimer', 'PerformanceLogger', 'setup_pipeline_logging'
]
