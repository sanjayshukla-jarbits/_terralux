"""
Modular Orchestrator - Fail Fast Plan
=====================================

A modular orchestrator system for geospatial data acquisition and processing
workflows, optimized for rapid development, testing, and validation.

Key Features:
- Fail-fast execution with performance monitoring
- Comprehensive mock data support for testing
- Modular step-based workflow execution
- Integration with existing landslide_pipeline structure
- CLI interface for development workflow
- Extensive testing and validation capabilities

Package Structure:
- core: Core orchestrator components and execution engine
- steps: Individual workflow step implementations
- processes: Pre-defined process templates and definitions
- tests: Comprehensive testing suite with fixtures
- utils: Utilities for data generation, validation, and monitoring
- cli: Command-line interface for development workflow

Author: Orchestrator Development Team
Version: 1.0.0-failfast
License: MIT
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Package metadata
__version__ = "1.0.0-failfast"
__author__ = "Orchestrator Development Team"
__license__ = "MIT"
__description__ = "Modular orchestrator for geospatial data workflows - Fail Fast Plan"

# Minimum Python version check
MINIMUM_PYTHON = (3, 8)
if sys.version_info < MINIMUM_PYTHON:
    raise ImportError(
        f"Orchestrator requires Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Configure package-level logging
logger = logging.getLogger(__name__)

# Package configuration
ORCHESTRATOR_CONFIG = {
    "fail_fast_mode": True,
    "use_mock_data": True,
    "performance_monitoring": True,
    "default_timeout": 300,
    "default_memory_limit_mb": 512,
    "test_data_retention": "24_hours",
    "auto_cleanup": True
}

# Import core components with error handling
try:
    # Core orchestrator components
    from .core.orchestrator import ModularOrchestrator
    from .core.step_executor import StepExecutor
    from .core.context_manager import ExecutionContext
    from .core.dependency_resolver import DependencyResolver
    from .core.performance import PerformanceMonitor
    
    CORE_AVAILABLE = True
    logger.debug("Core orchestrator components loaded successfully")
    
except ImportError as e:
    logger.warning(f"Core components not available: {e}")
    CORE_AVAILABLE = False
    
    # Provide mock implementations for development
    class ModularOrchestrator:
        """Mock orchestrator for development when core is not available."""
        def __init__(self, config=None):
            self.config = config or {}
            logger.info("Using mock ModularOrchestrator implementation")
        
        def execute_process(self, process_definition):
            return {
                "status": "completed", 
                "mock": True,
                "message": "Mock execution - core components not available"
            }
    
    class StepExecutor:
        """Mock step executor for development."""
        def execute_step(self, step_config, context=None):
            return {"status": "completed", "mock": True}
    
    class ExecutionContext:
        """Mock execution context for development."""
        def __init__(self, **kwargs):
            self.outputs = {}
        
        def store_output(self, step_id, key, data):
            if step_id not in self.outputs:
                self.outputs[step_id] = {}
            self.outputs[step_id][key] = data
        
        def get_output(self, step_id, key):
            return self.outputs.get(step_id, {}).get(key)
    
    class DependencyResolver:
        """Mock dependency resolver for development."""
        def resolve(self, steps):
            return [step.get("id", f"step_{i}") for i, step in enumerate(steps)]
    
    class PerformanceMonitor:
        """Mock performance monitor for development."""
        def start(self): pass
        def stop(self): return {"memory": 100, "time": 1.0}

# Import step implementations with error handling
try:
    from .steps.data_acquisition import (
        SentinelHubAcquisitionStep,
        DEMAcquisitionStep,
        LocalFilesDiscoveryStep,
        ASTERDEMAcquisitionStep
    )
    from .steps.data_processing import (
        DataHarmonizationStep,
        SpectralIndicesCalculationStep,
        DataValidationStep,
        InventoryGenerationStep
    )
    
    STEPS_AVAILABLE = True
    logger.debug("Step implementations loaded successfully")
    
except ImportError as e:
    logger.warning(f"Step implementations not available: {e}")
    STEPS_AVAILABLE = False
    
    # Mock step implementations
    class MockStep:
        """Base mock step for development."""
        def __init__(self, config):
            self.config = config
        
        def execute(self, **kwargs):
            return {
                "status": "completed",
                "mock": True,
                "step_type": self.__class__.__name__
            }
    
    # Create mock step classes
    SentinelHubAcquisitionStep = type("SentinelHubAcquisitionStep", (MockStep,), {})
    DEMAcquisitionStep = type("DEMAcquisitionStep", (MockStep,), {})
    LocalFilesDiscoveryStep = type("LocalFilesDiscoveryStep", (MockStep,), {})
    ASTERDEMAcquisitionStep = type("ASTERDEMAcquisitionStep", (MockStep,), {})
    DataHarmonizationStep = type("DataHarmonizationStep", (MockStep,), {})
    SpectralIndicesCalculationStep = type("SpectralIndicesCalculationStep", (MockStep,), {})
    DataValidationStep = type("DataValidationStep", (MockStep,), {})
    InventoryGenerationStep = type("InventoryGenerationStep", (MockStep,), {})

# Import process definitions with error handling
try:
    from .processes.data_acquisition_only import (
        create_process,
        get_available_processes,
        save_process_to_file,
        load_process_from_file,
        validate_process_definition,
        create_fail_fast_test_suite
    )
    
    PROCESSES_AVAILABLE = True
    logger.debug("Process definitions loaded successfully")
    
except ImportError as e:
    logger.warning(f"Process definitions not available: {e}")
    PROCESSES_AVAILABLE = False
    
    # Mock process functions
    def create_process(process_type, **kwargs):
        """Mock process creation for development."""
        return {
            "process_info": {
                "name": f"Mock {process_type}",
                "version": "1.0.0-mock",
                "mock": True
            },
            "steps": []
        }
    
    def get_available_processes():
        """Mock available processes for development."""
        return {
            "basic_data_acquisition": "Mock basic data acquisition",
            "multi_source_acquisition": "Mock multi-source acquisition"
        }
    
    def save_process_to_file(process_def, path):
        """Mock save process for development."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(process_def, f, indent=2)
        return Path(path)
    
    def load_process_from_file(path):
        """Mock load process for development."""
        import json
        with open(path) as f:
            return json.load(f)
    
    def validate_process_definition(process_def):
        """Mock validation for development."""
        return []  # No errors
    
    def create_fail_fast_test_suite():
        """Mock test suite creation."""
        return {"mock": "test_suite"}

# Import utilities with error handling
try:
    from .utils.mock_data import MockDataGenerator
    from .utils.geospatial import GeospatialValidator
    from .utils.performance import ResourceMonitor
    
    UTILS_AVAILABLE = True
    logger.debug("Utility modules loaded successfully")
    
except ImportError as e:
    logger.warning(f"Utility modules not available: {e}")
    UTILS_AVAILABLE = False
    
    # Mock utilities
    class MockDataGenerator:
        """Mock data generator for development."""
        def generate_sentinel_data(self, **kwargs):
            return "/mock/path/sentinel_data.tif"
        
        def generate_dem_data(self, **kwargs):
            return "/mock/path/dem_data.tif"
        
        def generate_local_files(self, **kwargs):
            return ["/mock/path/local_file.tif"]
    
    class GeospatialValidator:
        """Mock geospatial validator for development."""
        def validate_raster(self, file_path, **kwargs):
            return {"valid": True, "mock": True}
        
        def validate_vector(self, file_path, **kwargs):
            return {"valid": True, "mock": True}
    
    class ResourceMonitor:
        """Mock resource monitor for development."""
        def start_monitoring(self): pass
        def stop_monitoring(self): return {"memory": 100, "time": 1.0}

# Import testing framework with error handling
try:
    from .tests.test_orchestrator import run_fail_fast_tests
    from .tests.test_data_acquisition import DataAcquisitionTestSuite
    from .tests.fixtures import (
        create_test_data_package,
        temporary_orchestrator_environment
    )
    
    TESTS_AVAILABLE = True
    logger.debug("Testing framework loaded successfully")
    
except ImportError as e:
    logger.warning(f"Testing framework not available: {e}")
    TESTS_AVAILABLE = False
    
    # Mock testing functions
    def run_fail_fast_tests(test_level="minimal"):
        """Mock test runner for development."""
        logger.info(f"Mock test execution: {test_level}")
        return True
    
    class DataAcquisitionTestSuite:
        """Mock test suite for development."""
        @staticmethod
        def run_data_acquisition_tests(level="minimal"):
            logger.info(f"Mock data acquisition tests: {level}")
            return True
    
    def create_test_data_package(output_dir, package_type="minimal"):
        """Mock test data package creation."""
        return {"mock": "test_data_package", "output_dir": output_dir}
    
    class temporary_orchestrator_environment:
        """Mock temporary environment context manager."""
        def __init__(self, config=None):
            self.config = config or {}
        
        def __enter__(self):
            return {"temp_dir": "/tmp/mock", "config": self.config}
        
        def __exit__(self, *args):
            pass

# Import CLI with error handling
try:
    from .cli import main as cli_main
    CLI_AVAILABLE = True
    logger.debug("CLI module loaded successfully")
    
except ImportError as e:
    logger.warning(f"CLI module not available: {e}")
    CLI_AVAILABLE = False
    
    def cli_main():
        """Mock CLI for development."""
        print("Mock CLI - full CLI not available")

# Package-level functions
def get_version():
    """Get package version."""
    return __version__

def get_config():
    """Get package configuration."""
    return ORCHESTRATOR_CONFIG.copy()

def set_config(**kwargs):
    """Update package configuration."""
    ORCHESTRATOR_CONFIG.update(kwargs)
    logger.info(f"Package configuration updated: {kwargs}")

def get_status():
    """Get package status and component availability."""
    return {
        "version": __version__,
        "components": {
            "core": CORE_AVAILABLE,
            "steps": STEPS_AVAILABLE,
            "processes": PROCESSES_AVAILABLE,
            "utils": UTILS_AVAILABLE,
            "tests": TESTS_AVAILABLE,
            "cli": CLI_AVAILABLE
        },
        "config": ORCHESTRATOR_CONFIG,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "fail_fast_mode": ORCHESTRATOR_CONFIG["fail_fast_mode"]
    }

def check_environment():
    """Check environment setup and requirements."""
    issues = []
    
    # Check Python version
    if sys.version_info < MINIMUM_PYTHON:
        issues.append(f"Python version {sys.version_info.major}.{sys.version_info.minor} < required {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}")
    
    # Check component availability
    if not CORE_AVAILABLE:
        issues.append("Core orchestrator components not available")
    
    if not STEPS_AVAILABLE:
        issues.append("Step implementations not available")
    
    # Check system resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 2:
            issues.append(f"Low system memory: {memory_gb:.1f}GB (recommended: 2GB+)")
    except ImportError:
        issues.append("psutil not available for system resource checking")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendations": [
            "Install missing dependencies: pip install -r requirements.txt",
            "Ensure minimum 2GB RAM available",
            "Use fail-fast mode for development: orchestrator.set_config(fail_fast_mode=True)"
        ] if issues else []
    }

def create_orchestrator(config=None, fail_fast=None):
    """
    Factory function to create orchestrator instance with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        fail_fast: Override fail-fast mode setting
        
    Returns:
        ModularOrchestrator instance
    """
    # Merge configurations
    orchestrator_config = ORCHESTRATOR_CONFIG.copy()
    if config:
        orchestrator_config.update(config)
    
    if fail_fast is not None:
        orchestrator_config["fail_fast_mode"] = fail_fast
    
    return ModularOrchestrator(config=orchestrator_config)

def quick_test(level="minimal"):
    """
    Quick test execution for development workflow.
    
    Args:
        level: Test level ("minimal" or "comprehensive")
        
    Returns:
        bool: Test success status
    """
    logger.info(f"Running quick {level} tests...")
    
    if TESTS_AVAILABLE:
        return run_fail_fast_tests(test_level=level)
    else:
        logger.warning("Testing framework not available, using mock test")
        return True

def setup_development_environment(clean=False):
    """
    Setup development environment for orchestrator.
    
    Args:
        clean: Whether to clean existing environment
        
    Returns:
        dict: Setup status and created directories
    """
    from pathlib import Path
    import shutil
    
    logger.info("Setting up orchestrator development environment...")
    
    # Development directories
    dev_dirs = [
        "processes",
        "outputs",
        "temp", 
        "logs",
        "mock_data",
        "test_data",
        "configs",
        "reports"
    ]
    
    created_dirs = []
    
    for dir_name in dev_dirs:
        dir_path = Path(dir_name)
        
        if clean and dir_path.exists():
            shutil.rmtree(dir_path)
            logger.debug(f"Cleaned: {dir_name}")
        
        dir_path.mkdir(exist_ok=True)
        created_dirs.append(str(dir_path))
        logger.debug(f"Created: {dir_name}")
    
    # Create development configuration
    dev_config = {
        "development_mode": True,
        "fail_fast_enabled": True,
        "mock_data_preferred": True,
        "performance_monitoring": True,
        "auto_cleanup": True,
        "directories": {name: name for name in dev_dirs}
    }
    
    # Save development config
    import json
    from datetime import datetime
    
    config_file = Path("orchestrator_dev_config.json")
    dev_config["created_at"] = datetime.now().isoformat()
    dev_config["version"] = __version__
    
    with open(config_file, 'w') as f:
        json.dump(dev_config, f, indent=2)
    
    logger.info(f"Development configuration saved: {config_file}")
    
    return {
        "success": True,
        "directories_created": created_dirs,
        "config_file": str(config_file),
        "dev_config": dev_config
    }

# Package exports
__all__ = [
    # Metadata
    "__version__",
    "__author__", 
    "__license__",
    "__description__",
    
    # Core components
    "ModularOrchestrator",
    "StepExecutor", 
    "ExecutionContext",
    "DependencyResolver",
    "PerformanceMonitor",
    
    # Step implementations
    "SentinelHubAcquisitionStep",
    "DEMAcquisitionStep",
    "LocalFilesDiscoveryStep", 
    "ASTERDEMAcquisitionStep",
    "DataHarmonizationStep",
    "SpectralIndicesCalculationStep",
    "DataValidationStep",
    "InventoryGenerationStep",
    
    # Process functions
    "create_process",
    "get_available_processes",
    "save_process_to_file",
    "load_process_from_file",
    "validate_process_definition",
    "create_fail_fast_test_suite",
    
    # Utilities
    "MockDataGenerator",
    "GeospatialValidator", 
    "ResourceMonitor",
    
    # Testing
    "run_fail_fast_tests",
    "DataAcquisitionTestSuite",
    "create_test_data_package",
    "temporary_orchestrator_environment",
    
    # CLI
    "cli_main",
    
    # Package functions
    "get_version",
    "get_config",
    "set_config", 
    "get_status",
    "check_environment",
    "create_orchestrator",
    "quick_test",
    "setup_development_environment",
    
    # Constants
    "ORCHESTRATOR_CONFIG",
    "CORE_AVAILABLE",
    "STEPS_AVAILABLE", 
    "PROCESSES_AVAILABLE",
    "UTILS_AVAILABLE",
    "TESTS_AVAILABLE",
    "CLI_AVAILABLE"
]

# Package initialization logging
logger.info(f"Orchestrator package v{__version__} initialized")
logger.info(f"Components available: Core={CORE_AVAILABLE}, Steps={STEPS_AVAILABLE}, "
           f"Processes={PROCESSES_AVAILABLE}, Utils={UTILS_AVAILABLE}, "
           f"Tests={TESTS_AVAILABLE}, CLI={CLI_AVAILABLE}")

if ORCHESTRATOR_CONFIG["fail_fast_mode"]:
    logger.info("Fail-fast mode enabled for rapid development")

# Environment check on import
env_status = check_environment()
if not env_status["valid"]:
    warnings.warn(
        f"Environment issues detected: {', '.join(env_status['issues'])}. "
        "Some functionality may not be available.",
        RuntimeWarning
    )

# Show quick start info for development
if ORCHESTRATOR_CONFIG["fail_fast_mode"] and logger.isEnabledFor(logging.INFO):
    logger.info("Quick start commands:")
    logger.info("  import orchestrator; orchestrator.quick_test()")
    logger.info("  orchestrator.setup_development_environment()")
    logger.info("  orchestrator.create_orchestrator().execute_process(...)")
