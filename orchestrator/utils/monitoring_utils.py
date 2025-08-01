"""
Monitoring Utilities for Orchestrator
=====================================

This module provides performance monitoring and resource tracking utilities
for the orchestrator system.

Key Features:
- Memory usage monitoring
- Disk usage tracking
- CPU utilization monitoring
- Execution time tracking
- System resource reporting
- Performance metrics collection
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Configure logger
logger = logging.getLogger(__name__)

# Optional imports for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: str
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_available_mb: Optional[float] = None
    disk_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None
    process_memory_mb: Optional[float] = None
    process_cpu_percent: Optional[float] = None


def monitor_memory_usage(process_id: Optional[int] = None) -> Dict[str, float]:
    """
    Monitor memory usage for system and optionally specific process.
    
    Args:
        process_id: Process ID to monitor (current process if None)
        
    Returns:
        Dictionary with memory usage information
        
    Example:
        >>> memory_info = monitor_memory_usage()
        >>> print(f"System memory: {memory_info['system_percent']:.1f}%")
        >>> print(f"Process memory: {memory_info['process_mb']:.1f} MB")
    """
    memory_info = {}
    
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available for memory monitoring")
        return memory_info
    
    try:
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info.update({
            'system_total_mb': system_memory.total / 1024 / 1024,
            'system_used_mb': system_memory.used / 1024 / 1024,
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_percent': system_memory.percent
        })
        
        # Process memory
        if process_id:
            process = psutil.Process(process_id)
        else:
            process = psutil.Process()  # Current process
        
        process_memory = process.memory_info()
        memory_info.update({
            'process_rss_mb': process_memory.rss / 1024 / 1024,
            'process_vms_mb': process_memory.vms / 1024 / 1024,
            'process_percent': process.memory_percent()
        })
        
        # Memory-mapped files
        try:
            process_memory_full = process.memory_full_info()
            memory_info['process_uss_mb'] = process_memory_full.uss / 1024 / 1024
            memory_info['process_pss_mb'] = process_memory_full.pss / 1024 / 1024
        except (AttributeError, psutil.AccessDenied):
            # Not available on all platforms
            pass
        
        logger.debug(f"Memory monitoring - System: {memory_info['system_percent']:.1f}%, "
                    f"Process: {memory_info['process_rss_mb']:.1f}MB")
        
    except Exception as e:
        logger.error(f"Failed to monitor memory usage: {e}")
    
    return memory_info


def monitor_disk_usage(path: Union[str, Path] = '/') -> Dict[str, float]:
    """
    Monitor disk usage for specified path.
    
    Args:
        path: Path to check disk usage for
        
    Returns:
        Dictionary with disk usage information
        
    Example:
        >>> disk_info = monitor_disk_usage('/data')
        >>> print(f"Disk usage: {disk_info['percent']:.1f}%")
        >>> print(f"Free space: {disk_info['free_gb']:.1f} GB")
    """
    disk_info = {}
    
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available for disk monitoring")
        return disk_info
    
    try:
        disk_usage = psutil.disk_usage(str(path))
        
        disk_info.update({
            'total_gb': disk_usage.total / 1024 / 1024 / 1024,
            'used_gb': disk_usage.used / 1024 / 1024 / 1024,
            'free_gb': disk_usage.free / 1024 / 1024 / 1024,
            'percent': (disk_usage.used / disk_usage.total) * 100
        })
        
        logger.debug(f"Disk monitoring for {path} - Used: {disk_info['percent']:.1f}%, "
                    f"Free: {disk_info['free_gb']:.1f}GB")
        
    except Exception as e:
        logger.error(f"Failed to monitor disk usage: {e}")
    
    return disk_info


def track_execution_time(func: Optional[Callable] = None,
                        name: Optional[str] = None) -> Union[Callable, 'ExecutionTimeTracker']:
    """
    Track execution time of function or code block.
    
    Args:
        func: Function to track (when used as decorator)
        name: Name for the tracked operation
        
    Returns:
        Decorated function or ExecutionTimeTracker context manager
        
    Example:
        >>> @track_execution_time
        ... def process_data():
        ...     time.sleep(1)
        ...     return "done"
        
        >>> # Or as context manager
        >>> with track_execution_time(name="data_processing") as tracker:
        ...     time.sleep(1)
        ...     print(f"Elapsed: {tracker.elapsed_time}")
    """
    if func is not None:
        # Used as decorator
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_name = name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Execution time for {operation_name}: {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Execution failed for {operation_name} after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    else:
        # Used as context manager
        return ExecutionTimeTracker(name)


class ExecutionTimeTracker:
    """Context manager for tracking execution time."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or "operation"
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Started tracking: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if exc_type is None:
            logger.info(f"Completed {self.name} in {self.elapsed_time:.3f}s")
        else:
            logger.error(f"Failed {self.name} after {self.elapsed_time:.3f}s")
        
        return False  # Don't suppress exceptions


def log_system_resources(logger_instance: Optional[logging.Logger] = None,
                        include_processes: bool = False) -> Dict[str, Any]:
    """
    Log comprehensive system resource information.
    
    Args:
        logger_instance: Logger to use for output
        include_processes: Include top processes information
        
    Returns:
        Dictionary with system resource information
        
    Example:
        >>> resources = log_system_resources(include_processes=True)
        >>> print(f"CPU cores: {resources['cpu_cores']}")
        >>> print(f"Load average: {resources.get('load_average', 'N/A')}")
    """
    if logger_instance is None:
        logger_instance = logger
    
    resources = {
        'timestamp': datetime.now().isoformat(),
        'psutil_available': PSUTIL_AVAILABLE
    }
    
    if not PSUTIL_AVAILABLE:
        logger_instance.warning("psutil not available - limited resource monitoring")
        return resources
    
    try:
        # CPU information
        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
            cpu_info['load_average'] = {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            }
        except (AttributeError, OSError):
            # Not available on Windows
            pass
        
        resources['cpu'] = cpu_info
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'virtual': {
                'total_gb': memory.total / 1024 / 1024 / 1024,
                'available_gb': memory.available / 1024 / 1024 / 1024,
                'used_gb': memory.used / 1024 / 1024 / 1024,
                'percent': memory.percent
            },
            'swap': {
                'total_gb': swap.total / 1024 / 1024 / 1024,
                'used_gb': swap.used / 1024 / 1024 / 1024,
                'percent': swap.percent
            }
        }
        
        resources['memory'] = memory_info
        
        # Disk information
        disk_usage = psutil.disk_usage('/')
        disk_info = {
            'root': {
                'total_gb': disk_usage.total / 1024 / 1024 / 1024,
                'used_gb': disk_usage.used / 1024 / 1024 / 1024,
                'free_gb': disk_usage.free / 1024 / 1024 / 1024,
                'percent': (disk_usage.used / disk_usage.total) * 100
            }
        }
        
        # Add disk I/O stats
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_info['io'] = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                }
        except Exception:
            pass
        
        resources['disk'] = disk_info
        
        # Network information
        try:
            network_io = psutil.net_io_counters()
            if network_io:
                resources['network'] = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
        except Exception:
            pass
        
        # Process information
        if include_processes:
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sort by CPU usage and take top 10
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
                resources['top_processes'] = processes[:10]
                
            except Exception as e:
                logger_instance.warning(f"Failed to get process information: {e}")
        
        # Log summary
        logger_instance.info(f"System Resources - "
                           f"CPU: {cpu_info['cpu_percent']:.1f}%, "
                           f"Memory: {memory_info['virtual']['percent']:.1f}%, "
                           f"Disk: {disk_info['root']['percent']:.1f}%")
        
    except Exception as e:
        logger_instance.error(f"Failed to collect system resources: {e}")
    
    return resources


class PerformanceMonitor:
    """Continuous performance monitoring for long-running operations."""
    
    def __init__(self, name: str, interval: float = 5.0):
        self.name = name
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self.start_time = None
        self.logger = logging.getLogger(f"PerformanceMonitor.{name}")
    
    def start(self) -> None:
        """Start continuous monitoring."""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started performance monitoring for {self.name}")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        if not self.monitoring:
            self.logger.warning("Monitoring not started")
            return {}
        
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=self.interval + 1)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        results = {
            'name': self.name,
            'total_time_seconds': total_time,
            'snapshots_count': len(self.snapshots),
            'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
            'summary': self._calculate_summary()
        }
        
        self.logger.info(f"Stopped performance monitoring for {self.name} "
                        f"({total_time:.1f}s, {len(self.snapshots)} snapshots)")
        
        return results
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # Limit snapshots to prevent memory issues
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-500:]  # Keep last 500
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        timestamp = datetime.now().isoformat()
        
        if not PSUTIL_AVAILABLE:
            return ResourceSnapshot(timestamp=timestamp)
        
        try:
            # System resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process resources
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            return ResourceSnapshot(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / 1024 / 1024 / 1024,
                process_memory_mb=process_memory.rss / 1024 / 1024,
                process_cpu_percent=process_cpu
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to take resource snapshot: {e}")
            return ResourceSnapshot(timestamp=timestamp)
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from snapshots."""
        if not self.snapshots:
            return {}
        
        summary = {}
        
        # Calculate averages, min, max for numeric fields
        numeric_fields = ['cpu_percent', 'memory_percent', 'disk_percent', 
                         'process_memory_mb', 'process_cpu_percent']
        
        for field in numeric_fields:
            values = [getattr(snapshot, field) for snapshot in self.snapshots 
                     if getattr(snapshot, field) is not None]
            
            if values:
                summary[field] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return summary


def create_performance_report(monitoring_results: List[Dict[str, Any]], 
                            output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Create comprehensive performance report from monitoring results.
    
    Args:
        monitoring_results: List of monitoring result dictionaries
        output_path: Path to save report JSON file (optional)
        
    Returns:
        Performance report dictionary
        
    Example:
        >>> results = [monitor1.stop(), monitor2.stop()]
        >>> report = create_performance_report(results, 'performance_report.json')
        >>> print(f"Total operations: {report['summary']['total_operations']}")
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'operations': monitoring_results,
        'summary': {
            'total_operations': len(monitoring_results),
            'total_time_seconds': sum(r.get('total_time_seconds', 0) for r in monitoring_results),
            'total_snapshots': sum(r.get('snapshots_count', 0) for r in monitoring_results)
        }
    }
    
    # Calculate overall statistics
    if monitoring_results:
        all_snapshots = []
        for result in monitoring_results:
            all_snapshots.extend(result.get('snapshots', []))
        
        if all_snapshots:
            # Calculate overall resource usage statistics
            cpu_values = [s.get('cpu_percent') for s in all_snapshots if s.get('cpu_percent') is not None]
            memory_values = [s.get('memory_percent') for s in all_snapshots if s.get('memory_percent') is not None]
            
            if cpu_values:
                report['summary']['cpu_usage'] = {
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'avg': sum(cpu_values) / len(cpu_values)
                }
            
            if memory_values:
                report['summary']['memory_usage'] = {
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'avg': sum(memory_values) / len(memory_values)
                }
    
    # Save report if path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {output_file}")
    
    return report


@contextmanager
def monitor_resources(name: str, interval: float = 5.0):
    """
    Context manager for monitoring resources during operation.
    
    Args:
        name: Name for the monitoring operation
        interval: Monitoring interval in seconds
        
    Yields:
        PerformanceMonitor instance
        
    Example:
        >>> with monitor_resources("data_processing", interval=2.0) as monitor:
        ...     # Do some processing
        ...     time.sleep(10)
        ...     # Monitor automatically stops and returns results
        >>> results = monitor.stop()
    """
    monitor = PerformanceMonitor(name, interval)
    monitor.start()
    
    try:
        yield monitor
    finally:
        results = monitor.stop()
        logger.info(f"Resource monitoring completed for {name}")


# Export main functions
__all__ = [
    'monitor_memory_usage', 'monitor_disk_usage', 'track_execution_time',
    'log_system_resources', 'create_performance_report', 'monitor_resources',
    'ResourceSnapshot', 'ExecutionTimeTracker', 'PerformanceMonitor'
]
