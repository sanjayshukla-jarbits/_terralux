"""
Orchestrator Core Module - Fail-Fast Implementation
======================================================

This module provides the core functionality for the modular pipeline orchestrator,
designed for rapid prototyping and fail-fast development approach.

Core Components:
- ModularOrchestrator: Main orchestration engine
- PipelineContext: State and data management
- ProcessLoader: JSON process definition loading
- BaseStep: Abstract base class for pipeline steps
- StepRegistry: Step type registration system

Quick Start:
-----------
```python
from orchestrator.core import ModularOrchestrator, create_context

# Simple usage
orchestrator = ModularOrchestrator()
orchestrator.load_process('process.json', bbox=[85.3, 27.6, 85.4, 27.7])
result = orchestrator.execute()

# With context management
with create_context(pipeline_id='test') as context:
    orchestrator = ModularOrchestrator()
    result = orchestrator.execute_with_context(context)
```
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Configure logging for the core module
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0-dev"
__author__ = "Pipeline Development Team"

# Core imports with error handling for fail-fast approach
try:
    from .orchestrator import ModularOrchestrator
    _ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import ModularOrchestrator: {e}")
    ModularOrchestrator = None
    _ORCHESTRATOR_AVAILABLE = False

try:
    from .context_manager import PipelineContext, create_context
    _CONTEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import PipelineContext: {e}")
    PipelineContext = None
    create_context = None
    _CONTEXT_AVAILABLE = False

try:
    from .process_loader import ProcessLoader, load_process_simple, validate_process_file
    _PROCESS_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import ProcessLoader: {e}")
    ProcessLoader = None
    load_process_simple = None
    validate_process_file = None
    _PROCESS_LOADER_AVAILABLE = False

# Optional enhanced components (may not be available in minimal setup)
try:
    from .dependency_resolver import DependencyResolver
    _DEPENDENCY_RESOLVER_AVAILABLE = True
except ImportError:
    logger.debug("DependencyResolver not available (optional component)")
    DependencyResolver = None
    _DEPENDENCY_RESOLVER_AVAILABLE = False

try:
    from .execution_engine import ExecutionEngine
    _EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    logger.debug("ExecutionEngine not available (optional component)")
    ExecutionEngine = None
    _EXECUTION_ENGINE_AVAILABLE = False

try:
    from .step_factory import StepFactory
    _STEP_FACTORY_AVAILABLE = True
except ImportError:
    logger.debug("StepFactory not available (optional component)")
    StepFactory = None
    _STEP_FACTORY_AVAILABLE = False

# Import base classes from steps module
try:
    from ..steps.base.base_step import BaseStep
    from ..steps.base.step_registry import StepRegistry
    _STEP_BASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import step base classes: {e}")
    BaseStep = None
    StepRegistry = None
    _STEP_BASE_AVAILABLE = False


# Module-level convenience functions
def get_version() -> str:
    """Get the current version of the orchestrator core module."""
    return __version__


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of core components.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    return {
        'orchestrator': _ORCHESTRATOR_AVAILABLE,
        'context_manager': _CONTEXT_AVAILABLE,
        'process_loader': _PROCESS_LOADER_AVAILABLE,
        'dependency_resolver': _DEPENDENCY_RESOLVER_AVAILABLE,
        'execution_engine': _EXECUTION_ENGINE_AVAILABLE,
        'step_factory': _STEP_FACTORY_AVAILABLE,
        'step_base': _STEP_BASE_AVAILABLE
    }


def get_missing_dependencies() -> List[str]:
    """
    Get list of missing core dependencies.
    
    Returns:
        List of missing component names
    """
    deps = check_dependencies()
    return [name for name, available in deps.items() if not available]


def is_core_available() -> bool:
    """
    Check if core functionality is available.
    
    Returns:
        True if essential components are available
    """
    essential_components = [
        _ORCHESTRATOR_AVAILABLE,
        _CONTEXT_AVAILABLE,
        _PROCESS_LOADER_AVAILABLE,
        _STEP_BASE_AVAILABLE
    ]
    return all(essential_components)


def create_minimal_orchestrator(**kwargs) -> Optional['ModularOrchestrator']:
    """
    Create a minimal orchestrator instance for fail-fast testing.
    
    Args:
        **kwargs: Configuration parameters for the orchestrator
        
    Returns:
        ModularOrchestrator instance or None if not available
    """
    if not _ORCHESTRATOR_AVAILABLE:
        logger.error("ModularOrchestrator not available")
        return None
    
    try:
        return ModularOrchestrator(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        return None


def load_and_execute_process(process_path: Union[str, Path, Dict[str, Any]], 
                           template_variables: Optional[Dict[str, Any]] = None,
                           **config) -> Dict[str, Any]:
    """
    Convenience function to load and execute a process in one call.
    
    Args:
        process_path: Path to process JSON file or process dictionary
        template_variables: Variables for template substitution
        **config: Additional configuration parameters
        
    Returns:
        Execution results dictionary
    """
    if not is_core_available():
        missing = get_missing_dependencies()
        return {
            'status': 'failed',
            'error': f'Core components not available: {missing}',
            'missing_dependencies': missing
        }
    
    try:
        # Create orchestrator
        orchestrator = ModularOrchestrator()
        
        # Load process
        if isinstance(process_path, dict):
            orchestrator.load_process(process_path, template_variables)
        else:
            orchestrator.load_process(str(process_path), template_variables)
        
        # Execute with configuration
        return orchestrator.execute_process(config)
        
    except Exception as e:
        logger.error(f"Failed to load and execute process: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': str(e.__traceback__)
        }


def quick_test() -> Dict[str, Any]:
    """
    Run a quick test of core functionality.
    
    Returns:
        Test results dictionary
    """
    test_results = {
        'version': get_version(),
        'dependencies': check_dependencies(),
        'core_available': is_core_available(),
        'tests': {}
    }
    
    # Test 1: Component imports
    test_results['tests']['imports'] = {
        'status': 'success' if is_core_available() else 'failed',
        'details': 'All core components imported successfully' if is_core_available() 
                  else f'Missing components: {get_missing_dependencies()}'
    }
    
    # Test 2: Context creation
    if _CONTEXT_AVAILABLE:
        try:
            with create_context(pipeline_id='test') as context:
                context.set_variable('test_var', 'test_value')
                value = context.get_variable('test_var')
                assert value == 'test_value'
            test_results['tests']['context'] = {
                'status': 'success',
                'details': 'Context creation and variable management working'
            }
        except Exception as e:
            test_results['tests']['context'] = {
                'status': 'failed',
                'details': f'Context test failed: {e}'
            }
    else:
        test_results['tests']['context'] = {
            'status': 'skipped',
            'details': 'Context manager not available'
        }
    
    # Test 3: Process loading
    if _PROCESS_LOADER_AVAILABLE:
        try:
            # Test with minimal process definition
            test_process = {
                "process_info": {"name": "test", "version": "1.0.0"},
                "steps": [{"id": "test_step", "type": "mock"}]
            }
            loader = ProcessLoader()
            loaded = loader.load_process(test_process)
            assert loaded['process_info']['name'] == 'test'
            test_results['tests']['process_loader'] = {
                'status': 'success',
                'details': 'Process loading working correctly'
            }
        except Exception as e:
            test_results['tests']['process_loader'] = {
                'status': 'failed',
                'details': f'Process loader test failed: {e}'
            }
    else:
        test_results['tests']['process_loader'] = {
            'status': 'skipped',
            'details': 'Process loader not available'
        }
    
    # Test 4: Step registry
    if _STEP_BASE_AVAILABLE:
        try:
            # Test step registry
            available_types = StepRegistry.get_registered_types() if hasattr(StepRegistry, 'get_registered_types') else []
            test_results['tests']['step_registry'] = {
                'status': 'success',
                'details': f'Step registry working, {len(available_types)} types registered'
            }
        except Exception as e:
            test_results['tests']['step_registry'] = {
                'status': 'failed',
                'details': f'Step registry test failed: {e}'
            }
    else:
        test_results['tests']['step_registry'] = {
            'status': 'skipped',
            'details': 'Step base classes not available'
        }
    
    # Test 5: End-to-end orchestrator creation
    if is_core_available():
        try:
            orchestrator = create_minimal_orchestrator()
            if orchestrator:
                test_results['tests']['orchestrator'] = {
                    'status': 'success',
                    'details': 'Orchestrator creation successful'
                }
            else:
                test_results['tests']['orchestrator'] = {
                    'status': 'failed',
                    'details': 'Orchestrator creation returned None'
                }
        except Exception as e:
            test_results['tests']['orchestrator'] = {
                'status': 'failed',
                'details': f'Orchestrator creation failed: {e}'
            }
    else:
        test_results['tests']['orchestrator'] = {
            'status': 'skipped',
            'details': 'Core components not available'
        }
    
    # Overall test status
    test_statuses = [test['status'] for test in test_results['tests'].values()]
    failed_tests = sum(1 for status in test_statuses if status == 'failed')
    passed_tests = sum(1 for status in test_statuses if status == 'success')
    
    test_results['summary'] = {
        'total_tests': len(test_statuses),
        'passed': passed_tests,
        'failed': failed_tests,
        'skipped': sum(1 for status in test_statuses if status == 'skipped'),
        'overall_status': 'success' if failed_tests == 0 else 'failed'
    }
    
    return test_results


# Export public API
__all__ = [
    # Core classes (may be None if not available)
    'ModularOrchestrator',
    'PipelineContext', 
    'ProcessLoader',
    'BaseStep',
    'StepRegistry',
    
    # Optional classes
    'DependencyResolver',
    'ExecutionEngine',
    'StepFactory',
    
    # Utility functions
    'create_context',
    'load_process_simple',
    'validate_process_file',
    'create_minimal_orchestrator',
    'load_and_execute_process',
    
    # Module functions
    'get_version',
    'check_dependencies',
    'get_missing_dependencies',
    'is_core_available',
    'quick_test',
    
    # Module metadata
    '__version__',
    '__author__'
]


# Module initialization
def _initialize_module():
    """Initialize the core module and log status."""
    logger.info(f"Initializing orchestrator core module v{__version__}")
    
    # Check dependencies and log status
    deps = check_dependencies()
    available_count = sum(deps.values())
    total_count = len(deps)
    
    if is_core_available():
        logger.info(f"✓ Core functionality available ({available_count}/{total_count} components)")
    else:
        missing = get_missing_dependencies()
        logger.warning(f"⚠ Core functionality incomplete ({available_count}/{total_count} components)")
        logger.warning(f"Missing components: {missing}")
    
    # Register any available step types from imported modules
    if _STEP_BASE_AVAILABLE:
        try:
            # Try to import and register data acquisition steps if available
            from ..steps.data_acquisition import *
            logger.debug("Data acquisition steps imported")
        except ImportError:
            logger.debug("Data acquisition steps not available")
        
        try:
            # Log registered step types
            if hasattr(StepRegistry, 'get_registered_types'):
                registered_types = StepRegistry.get_registered_types()
                if registered_types:
                    logger.info(f"Registered step types: {registered_types}")
                else:
                    logger.info("No step types registered yet")
        except Exception as e:
            logger.debug(f"Could not get registered step types: {e}")


# Initialize module on import
_initialize_module()


# Quick test when run directly
if __name__ == "__main__":
    import json
    
    print("Running orchestrator core module test...")
    results = quick_test()
    
    print(f"\nOrchestrator Core v{results['version']}")
    print(f"Core Available: {results['core_available']}")
    print(f"Tests: {results['summary']['passed']}/{results['summary']['total_tests']} passed")
    
    if results['summary']['failed'] > 0:
        print(f"\nFailed tests:")
        for test_name, test_result in results['tests'].items():
            if test_result['status'] == 'failed':
                print(f"  - {test_name}: {test_result['details']}")
    
    print(f"\nDependency Status:")
    for component, available in results['dependencies'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    
    if not results['core_available']:
        print(f"\n⚠ To enable full functionality, ensure all core components are available.")
        print(f"Missing: {get_missing_dependencies()}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_status'] == 'success' else 1
    exit(exit_code)
