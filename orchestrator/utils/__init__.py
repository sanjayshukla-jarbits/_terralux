"""
Utilities Module for Modular Pipeline Orchestrator
==================================================

This module provides essential utility functions for the orchestrator system,
including file operations, validation, logging, configuration management,
and geospatial utilities.

Key Features:
- File system operations and path management
- Data validation and schema checking
- Enhanced logging with context
- Configuration file handling
- Geospatial data utilities
- Performance monitoring utilities
- Error handling and debugging tools

Quick Start:
-----------
```python
from orchestrator.utils import (
    ensure_directory, validate_file_exists,
    setup_logger, load_config,
    get_raster_info, calculate_bbox_area
)

# File operations
ensure_directory('outputs/results')
if validate_file_exists('data/input.tif'):
    print("File exists and is valid")

# Logging
logger = setup_logger('MyStep', 'logs/mystep.log')
logger.info("Processing started")

# Configuration
config = load_config('config.json')

# Geospatial utilities
info = get_raster_info('data/elevation.tif')
area_km2 = calculate_bbox_area([85.3, 27.6, 85.4, 27.7])
```

Module Organization:
-------------------
- file_utils: File system operations and validation
- validation_utils: Data validation and schema checking
- logging_utils: Enhanced logging capabilities
- config_utils: Configuration file management
- geospatial_utils: Geospatial data operations
- monitoring_utils: Performance and resource monitoring
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0-basic"
__author__ = "Orchestrator Team"

# Import utility modules with error handling
_UTILS_AVAILABLE = {}

try:
    from .file_utils import (
        ensure_directory, validate_file_exists, get_file_size,
        copy_file, move_file, delete_file_safe, create_temp_directory,
        find_files, get_file_hash, compress_directory
    )
    _UTILS_AVAILABLE['file_utils'] = True
    logger.debug("Successfully imported file utilities")
except ImportError as e:
    logger.warning(f"Failed to import file utilities: {e}")
    _UTILS_AVAILABLE['file_utils'] = False

try:
    from .validation_utils import (
        validate_json_schema, validate_bbox, validate_date_format,
        validate_file_format, validate_raster_data, validate_vector_data,
        check_data_integrity, validate_step_config
    )
    _UTILS_AVAILABLE['validation_utils'] = True
    logger.debug("Successfully imported validation utilities")
except ImportError as e:
    logger.warning(f"Failed to import validation utilities: {e}")
    _UTILS_AVAILABLE['validation_utils'] = False

try:
    from .logging_utils import (
        setup_logger, configure_logging, get_logger_with_context,
        log_execution_time, log_memory_usage, create_log_handler
    )
    _UTILS_AVAILABLE['logging_utils'] = True
    logger.debug("Successfully imported logging utilities")
except ImportError as e:
    logger.warning(f"Failed to import logging utilities: {e}")
    _UTILS_AVAILABLE['logging_utils'] = False

try:
    from .config_utils import (
        load_config, save_config, merge_configs, validate_config,
        get_config_value, set_config_value, create_default_config
    )
    _UTILS_AVAILABLE['config_utils'] = True
    logger.debug("Successfully imported config utilities")
except ImportError as e:
    logger.warning(f"Failed to import config utilities: {e}")
    _UTILS_AVAILABLE['config_utils'] = False

try:
    from .geospatial_utils import (
        get_raster_info, calculate_bbox_area, reproject_bbox,
        validate_crs, get_raster_bounds, calculate_pixel_size,
        clip_raster_to_bbox, merge_raster_files
    )
    _UTILS_AVAILABLE['geospatial_utils'] = True
    logger.debug("Successfully imported geospatial utilities")
except ImportError as e:
    logger.warning(f"Failed to import geospatial utilities: {e}")
    _UTILS_AVAILABLE['geospatial_utils'] = False

try:
    from .monitoring_utils import (
        monitor_memory_usage, monitor_disk_usage, track_execution_time,
        log_system_resources, create_performance_report
    )
    _UTILS_AVAILABLE['monitoring_utils'] = True
    logger.debug("Successfully imported monitoring utilities")
except ImportError as e:
    logger.warning(f"Failed to import monitoring utilities: {e}")
    _UTILS_AVAILABLE['monitoring_utils'] = False

# Utility functions
def get_available_utils() -> Dict[str, bool]:
    """Get status of available utility modules."""
    return _UTILS_AVAILABLE.copy()

def print_utils_status():
    """Print status of all utility modules."""
    print(f"\n🔧 Orchestrator Utils Module Status (v{__version__})")
    print("=" * 50)
    
    for module, available in _UTILS_AVAILABLE.items():
        status_icon = "✓" if available else "❌"
        print(f"{status_icon} {module}")
    
    successful = sum(_UTILS_AVAILABLE.values())
    total = len(_UTILS_AVAILABLE)
    print(f"\nStatus: {successful}/{total} utility modules available")
    
    if successful < total:
        print("\n⚠️  Some utilities may have limited functionality")
        print("   Install missing dependencies for full capabilities")

def validate_utils_setup() -> Dict[str, Any]:
    """Validate the utilities setup and return detailed results."""
    validation_results = {
        'overall_status': 'unknown',
        'module_status': _UTILS_AVAILABLE.copy(),
        'missing_modules': [],
        'available_functions': {},
        'issues': []
    }
    
    # Check available functions in each module
    if _UTILS_AVAILABLE['file_utils']:
        validation_results['available_functions']['file_utils'] = [
            'ensure_directory', 'validate_file_exists', 'get_file_size',
            'copy_file', 'move_file', 'delete_file_safe'
        ]
    
    if _UTILS_AVAILABLE['validation_utils']:
        validation_results['available_functions']['validation_utils'] = [
            'validate_json_schema', 'validate_bbox', 'validate_date_format',
            'validate_file_format', 'check_data_integrity'
        ]
    
    if _UTILS_AVAILABLE['logging_utils']:
        validation_results['available_functions']['logging_utils'] = [
            'setup_logger', 'configure_logging', 'get_logger_with_context',
            'log_execution_time', 'log_memory_usage'
        ]
    
    if _UTILS_AVAILABLE['config_utils']:
        validation_results['available_functions']['config_utils'] = [
            'load_config', 'save_config', 'merge_configs', 'validate_config'
        ]
    
    if _UTILS_AVAILABLE['geospatial_utils']:
        validation_results['available_functions']['geospatial_utils'] = [
            'get_raster_info', 'calculate_bbox_area', 'reproject_bbox',
            'validate_crs', 'get_raster_bounds'
        ]
    
    if _UTILS_AVAILABLE['monitoring_utils']:
        validation_results['available_functions']['monitoring_utils'] = [
            'monitor_memory_usage', 'monitor_disk_usage', 'track_execution_time',
            'log_system_resources'
        ]
    
    # Identify missing modules
    for module, available in _UTILS_AVAILABLE.items():
        if not available:
            validation_results['missing_modules'].append(module)
    
    # Determine overall status
    successful_modules = sum(_UTILS_AVAILABLE.values())
    total_modules = len(_UTILS_AVAILABLE)
    
    if successful_modules == 0:
        validation_results['overall_status'] = 'failed'
        validation_results['issues'].append("No utility modules available")
    elif successful_modules == total_modules:
        validation_results['overall_status'] = 'success'
    else:
        validation_results['overall_status'] = 'partial'
        validation_results['issues'].append(f"Only {successful_modules}/{total_modules} modules available")
    
    return validation_results

def get_help() -> str:
    """Get help information for the utils module."""
    return f"""
Orchestrator Utils Module Help (v{__version__})
{'=' * 50}

This module provides essential utilities for the pipeline orchestrator.

Available Modules:
{chr(10).join(f'  • {module} ({("✓" if available else "❌")})' for module, available in _UTILS_AVAILABLE.items())}

Quick Start Examples:

# File Operations
from orchestrator.utils import ensure_directory, validate_file_exists
ensure_directory('outputs/results')
is_valid = validate_file_exists('data/input.tif')

# Logging
from orchestrator.utils import setup_logger
logger = setup_logger('MyStep', 'logs/mystep.log')

# Configuration
from orchestrator.utils import load_config, save_config
config = load_config('config.json')
save_config(config, 'output_config.json')

# Validation
from orchestrator.utils import validate_bbox, validate_date_format
valid_bbox = validate_bbox([85.3, 27.6, 85.4, 27.7])
valid_date = validate_date_format('2023-01-01')

# Geospatial Operations
from orchestrator.utils import get_raster_info, calculate_bbox_area
info = get_raster_info('elevation.tif')
area_km2 = calculate_bbox_area([85.3, 27.6, 85.4, 27.7])

For detailed documentation, see individual utility modules.
"""

# Export public API
__all__ = [
    # File utilities
    'ensure_directory', 'validate_file_exists', 'get_file_size',
    'copy_file', 'move_file', 'delete_file_safe', 'create_temp_directory',
    'find_files', 'get_file_hash', 'compress_directory',
    
    # Validation utilities
    'validate_json_schema', 'validate_bbox', 'validate_date_format',
    'validate_file_format', 'validate_raster_data', 'validate_vector_data',
    'check_data_integrity', 'validate_step_config',
    
    # Logging utilities
    'setup_logger', 'configure_logging', 'get_logger_with_context',
    'log_execution_time', 'log_memory_usage', 'create_log_handler',
    
    # Config utilities
    'load_config', 'save_config', 'merge_configs', 'validate_config',
    'get_config_value', 'set_config_value', 'create_default_config',
    
    # Geospatial utilities
    'get_raster_info', 'calculate_bbox_area', 'reproject_bbox',
    'validate_crs', 'get_raster_bounds', 'calculate_pixel_size',
    'clip_raster_to_bbox', 'merge_raster_files',
    
    # Monitoring utilities
    'monitor_memory_usage', 'monitor_disk_usage', 'track_execution_time',
    'log_system_resources', 'create_performance_report',
    
    # Module functions
    'get_available_utils', 'print_utils_status', 'validate_utils_setup',
    'get_help',
    
    # Module metadata
    '__version__', '__author__'
]

# Initialize module
def _initialize_utils_module():
    """Initialize the utils module."""
    logger.info(f"Initializing orchestrator utils module v{__version__}")
    
    successful_modules = sum(_UTILS_AVAILABLE.values())
    total_modules = len(_UTILS_AVAILABLE)
    
    if successful_modules == total_modules:
        logger.info("✓ All utility modules loaded successfully")
    elif successful_modules > 0:
        logger.info(f"✓ {successful_modules}/{total_modules} utility modules loaded")
        logger.warning("Some utilities may have limited functionality")
    else:
        logger.warning("❌ No utility modules loaded - using minimal functionality")

# Initialize on import
_initialize_utils_module()
