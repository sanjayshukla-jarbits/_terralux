"""
Steps Base Module - Foundation for Pipeline Step Implementation
==============================================================

This module provides the core infrastructure for implementing pipeline steps
in the modular orchestrator system, designed with fail-fast principles.

Core Components:
- BaseStep: Abstract base class for all pipeline steps
- StepRegistry: Global registry for step type management
- StepResult: Standardized result structure
- Utility functions and decorators

Quick Start:
-----------
```python
from orchestrator.steps.base import BaseStep, StepRegistry, register_step

# Method 1: Direct registration
class MyStep(BaseStep):
    def execute(self, context):
        return StepResult(status='success')

StepRegistry.register('my_step', MyStep)

# Method 2: Decorator registration
@register_step('my_step_decorated', category='example')
class MyDecoratedStep(BaseStep):
    def execute(self, context):
        return StepResult(status='success')
```

Development Workflow:
--------------------
1. Create step class inheriting from BaseStep
2. Implement execute() method
3. Register with StepRegistry
4. Test with MockStep if needed
5. Use in JSON process definitions
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union

# Configure logging for the base module
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0-dev"

# Core imports with fail-fast error handling
_BASE_STEP_AVAILABLE = False
_STEP_REGISTRY_AVAILABLE = False
_STEP_RESULT_AVAILABLE = False

try:
    from .base_step import BaseStep, StepResult, MockStep, StepValidationError
    from .base_step import create_mock_step, validate_step_implementation
    _BASE_STEP_AVAILABLE = True
    logger.debug("BaseStep classes imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import BaseStep: {e}")
    BaseStep = None
    StepResult = None
    MockStep = None
    StepValidationError = None
    create_mock_step = None
    validate_step_implementation = None

try:
    from .step_registry import StepRegistry, StepRegistrationError, register_step
    from .step_registry import get_available_step_types, create_step_from_config, is_step_type_available
    _STEP_REGISTRY_AVAILABLE = True
    logger.debug("StepRegistry imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import StepRegistry: {e}")
    StepRegistry = None
    StepRegistrationError = None
    register_step = None
    get_available_step_types = None
    create_step_from_config = None
    is_step_type_available = None

# Check if core functionality is available
_CORE_AVAILABLE = _BASE_STEP_AVAILABLE and _STEP_REGISTRY_AVAILABLE


def is_base_available() -> bool:
    """Check if base step functionality is available."""
    return _BASE_STEP_AVAILABLE


def is_registry_available() -> bool:
    """Check if step registry functionality is available."""
    return _STEP_REGISTRY_AVAILABLE


def is_core_available() -> bool:
    """Check if core base functionality is available."""
    return _CORE_AVAILABLE


def get_component_status() -> Dict[str, bool]:
    """Get status of all base module components."""
    return {
        'base_step': _BASE_STEP_AVAILABLE,
        'step_registry': _STEP_REGISTRY_AVAILABLE,
        'core_available': _CORE_AVAILABLE
    }


def get_missing_components() -> List[str]:
    """Get list of missing components."""
    status = get_component_status()
    return [name for name, available in status.items() if not available and name != 'core_available']


# Convenience functions with fail-fast support
def create_step(step_config: Dict[str, Any]) -> Optional['BaseStep']:
    """
    Create a step instance from configuration with error handling.
    
    Args:
        step_config: Step configuration dictionary
        
    Returns:
        Step instance or None if creation fails
    """
    if not _STEP_REGISTRY_AVAILABLE:
        logger.error("StepRegistry not available for step creation")
        return None
    
    try:
        return StepRegistry.create_step(step_config)
    except Exception as e:
        logger.error(f"Failed to create step: {e}")
        return None


def register_step_safe(step_type: str, 
                      step_class: Type['BaseStep'],
                      **kwargs) -> bool:
    """
    Safely register a step with error handling.
    
    Args:
        step_type: Step type identifier
        step_class: Step class
        **kwargs: Additional registration parameters
        
    Returns:
        True if registration successful, False otherwise
    """
    if not _STEP_REGISTRY_AVAILABLE:
        logger.error("StepRegistry not available for step registration")
        return False
    
    try:
        StepRegistry.register(step_type, step_class, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to register step '{step_type}': {e}")
        return False


def get_step_types() -> List[str]:
    """
    Get list of available step types with error handling.
    
    Returns:
        List of step types or empty list if registry not available
    """
    if not _STEP_REGISTRY_AVAILABLE:
        logger.warning("StepRegistry not available")
        return []
    
    try:
        return StepRegistry.get_registered_types()
    except Exception as e:
        logger.error(f"Failed to get step types: {e}")
        return []


def validate_step_class(step_class: Type['BaseStep']) -> List[str]:
    """
    Validate step class implementation with error handling.
    
    Args:
        step_class: Step class to validate
        
    Returns:
        List of validation errors
    """
    if not _BASE_STEP_AVAILABLE:
        return ["BaseStep not available for validation"]
    
    try:
        return validate_step_implementation(step_class)
    except Exception as e:
        return [f"Validation failed: {e}"]


def create_mock_step_safe(step_id: str, 
                         step_type: str = 'mock',
                         **hyperparameters) -> Optional['MockStep']:
    """
    Create a mock step with error handling.
    
    Args:
        step_id: Step identifier
        step_type: Step type name
        **hyperparameters: Step parameters
        
    Returns:
        MockStep instance or None if creation fails
    """
    if not _BASE_STEP_AVAILABLE:
        logger.error("MockStep not available")
        return None
    
    try:
        return create_mock_step(step_id, step_type, **hyperparameters)
    except Exception as e:
        logger.error(f"Failed to create mock step: {e}")
        return None


# Auto-registration of built-in step types
def _register_builtin_steps():
    """Register built-in step types if available."""
    if not _CORE_AVAILABLE:
        return
    
    try:
        # Register MockStep if available
        if MockStep:
            StepRegistry.register(
                'mock',
                MockStep,
                category='testing',
                aliases=['test', 'mock_step'],
                metadata={
                    'description': 'Mock step for testing and development',
                    'builtin': True
                }
            )
            logger.debug("Registered built-in MockStep")
    
    except Exception as e:
        logger.debug(f"Failed to register built-in steps: {e}")


# Development and testing utilities
def quick_test() -> Dict[str, Any]:
    """
    Run quick test of base module functionality.
    
    Returns:
        Dictionary with test results
    """
    test_results = {
        'component_status': get_component_status(),
        'core_available': is_core_available(),
        'tests': {}
    }
    
    # Test 1: Component imports
    test_results['tests']['imports'] = {
        'status': 'success' if _CORE_AVAILABLE else 'failed',
        'details': f"Core available: {_CORE_AVAILABLE}"
    }
    
    # Test 2: Mock step creation
    if _BASE_STEP_AVAILABLE:
        try:
            mock_step = create_mock_step_safe('test_mock', 'mock_test')
            if mock_step:
                test_results['tests']['mock_creation'] = {
                    'status': 'success',
                    'details': f"Created mock step: {mock_step}"
                }
            else:
                test_results['tests']['mock_creation'] = {
                    'status': 'failed',
                    'details': "Mock step creation returned None"
                }
        except Exception as e:
            test_results['tests']['mock_creation'] = {
                'status': 'failed',
                'details': f"Mock step creation failed: {e}"
            }
    else:
        test_results['tests']['mock_creation'] = {
            'status': 'skipped',
            'details': "BaseStep not available"
        }
    
    # Test 3: Registry operations
    if _STEP_REGISTRY_AVAILABLE:
        try:
            # Test registration
            class TestStep(BaseStep):
                def execute(self, context):
                    return StepResult(status='success') if StepResult else {'status': 'success'}
            
            success = register_step_safe('test_registry_step', TestStep)
            if success:
                # Test retrieval
                types = get_step_types()
                if 'test_registry_step' in types:
                    test_results['tests']['registry'] = {
                        'status': 'success',
                        'details': f"Registry test passed, {len(types)} types available"
                    }
                else:
                    test_results['tests']['registry'] = {
                        'status': 'failed',
                        'details': "Step not found after registration"
                    }
                
                # Cleanup
                try:
                    StepRegistry.unregister('test_registry_step')
                except:
                    pass
            else:
                test_results['tests']['registry'] = {
                    'status': 'failed',
                    'details': "Step registration failed"
                }
                
        except Exception as e:
            test_results['tests']['registry'] = {
                'status': 'failed',
                'details': f"Registry test failed: {e}"
            }
    else:
        test_results['tests']['registry'] = {
            'status': 'skipped',
            'details': "StepRegistry not available"
        }
    
    # Test 4: Step validation
    if _BASE_STEP_AVAILABLE:
        try:
            class ValidStep(BaseStep):
                def execute(self, context):
                    return StepResult(status='success') if StepResult else {'status': 'success'}
            
            errors = validate_step_class(ValidStep)
            if not errors:
                test_results['tests']['validation'] = {
                    'status': 'success',
                    'details': "Step validation passed"
                }
            else:
                test_results['tests']['validation'] = {
                    'status': 'failed',
                    'details': f"Validation errors: {errors}"
                }
        except Exception as e:
            test_results['tests']['validation'] = {
                'status': 'failed',
                'details': f"Validation test failed: {e}"
            }
    else:
        test_results['tests']['validation'] = {
            'status': 'skipped',
            'details': "BaseStep not available"
        }
    
    # Calculate overall status
    test_statuses = [test['status'] for test in test_results['tests'].values()]
    failed_count = sum(1 for status in test_statuses if status == 'failed')
    
    test_results['summary'] = {
        'total_tests': len(test_statuses),
        'passed': sum(1 for status in test_statuses if status == 'success'),
        'failed': failed_count,
        'skipped': sum(1 for status in test_statuses if status == 'skipped'),
        'overall_status': 'success' if failed_count == 0 else 'failed'
    }
    
    return test_results


def print_status():
    """Print comprehensive status of the base module."""
    print(f"\n=== Steps Base Module v{__version__} ===")
    
    status = get_component_status()
    print(f"Core Available: {status['core_available']}")
    
    print("\nComponent Status:")
    for component, available in status.items():
        if component != 'core_available':
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {component}")
    
    if not status['core_available']:
        missing = get_missing_components()
        print(f"\nMissing Components: {missing}")
    
    # Show available step types if registry is available
    if _STEP_REGISTRY_AVAILABLE:
        try:
            types = get_step_types()
            if types:
                print(f"\nRegistered Step Types ({len(types)}):")
                for step_type in sorted(types):
                    print(f"  • {step_type}")
            else:
                print("\nNo step types registered yet")
        except Exception as e:
            print(f"\nCould not get step types: {e}")


def get_help() -> str:
    """Get help text for the base module."""
    return """
Steps Base Module Help
=====================

This module provides the foundation for implementing pipeline steps.

Core Classes:
- BaseStep: Abstract base class for all steps
- StepRegistry: Global registry for step types
- StepResult: Standard result structure
- MockStep: Mock implementation for testing

Key Functions:
- create_step(config): Create step from configuration
- register_step_safe(type, class): Register step type
- get_step_types(): Get available step types
- validate_step_class(class): Validate step implementation

Development Workflow:
1. Import base classes: from orchestrator.steps.base import BaseStep, StepRegistry
2. Create step class inheriting from BaseStep
3. Implement execute() method returning StepResult
4. Register step: StepRegistry.register('step_type', StepClass)
5. Test with create_mock_step() or MockStep
6. Use in JSON process definitions

For detailed documentation, see individual class docstrings.
"""


# Initialize module
def _initialize_module():
    """Initialize the base module."""
    logger.info(f"Initializing steps base module v{__version__}")
    
    # Log component status
    status = get_component_status()
    if status['core_available']:
        logger.info("✓ Core base functionality available")
    else:
        missing = get_missing_components()
        logger.warning(f"⚠ Core functionality incomplete, missing: {missing}")
    
    # Register built-in steps
    if _CORE_AVAILABLE:
        _register_builtin_steps()
        
        # Log registry status
        try:
            type_count = len(get_step_types())
            logger.info(f"Step registry initialized with {type_count} types")
        except:
            logger.debug("Could not get initial step type count")


# Export public API
__all__ = [
    # Core classes (may be None if not available)
    'BaseStep',
    'StepRegistry', 
    'StepResult',
    'MockStep',
    
    # Exception classes
    'StepValidationError',
    'StepRegistrationError',
    
    # Utility functions
    'create_step',
    'register_step_safe',
    'get_step_types',
    'validate_step_class',
    'create_mock_step_safe',
    
    # Registry functions
    'get_available_step_types',
    'create_step_from_config',
    'is_step_type_available',
    'register_step',
    
    # Module functions
    'is_base_available',
    'is_registry_available',
    'is_core_available',
    'get_component_status',
    'get_missing_components',
    'quick_test',
    'print_status',
    'get_help',
    
    # Metadata
    '__version__'
]

# Initialize module on import
_initialize_module()

# Quick test when run directly
if __name__ == "__main__":
    print("Running steps base module test...")
    
    # Print status
    print_status()
    
    # Run comprehensive test
    print("\n" + "="*50)
    results = quick_test()
    
    print(f"\nTest Results:")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Skipped: {results['summary']['skipped']}")
    print(f"Overall: {results['summary']['overall_status'].upper()}")
    
    # Show failed tests
    if results['summary']['failed'] > 0:
        print(f"\nFailed Tests:")
        for test_name, test_result in results['tests'].items():
            if test_result['status'] == 'failed':
                print(f"  ✗ {test_name}: {test_result['details']}")
    
    # Show help
    if not is_core_available():
        print(f"\nFor help getting started:")
        print("python -c \"from orchestrator.steps.base import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_status'] == 'success' else 1
    exit(exit_code)
