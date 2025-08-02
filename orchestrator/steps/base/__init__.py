#!/usr/bin/env python3
"""
Steps Base Module - Foundation for Pipeline Step Implementation
==============================================================

This module provides the core infrastructure for implementing pipeline steps
in the modular orchestrator system, with enhanced registration management.

Core Components:
- BaseStep: Abstract base class for all pipeline steps
- StepRegistry: Global registry for step type management
- Registration utilities and auto-discovery
- Validation and debugging tools

Enhanced Features:
- Automatic step discovery and registration
- Import error handling and fallback mechanisms
- Comprehensive validation and debugging
- Module health monitoring

FIXES APPLIED:
- Enhanced register_step_safe() function with better error handling
- Fixed import error handling for step_registry module
- Added proper fallback mechanisms when components are not available
- Improved error logging and debugging capabilities
"""

import logging
import warnings
import importlib
from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path

# Configure logging for the base module
logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0-fixed"

# Core imports with fail-fast error handling
_BASE_STEP_AVAILABLE = False
_STEP_REGISTRY_AVAILABLE = False

try:
    from .base_step import BaseStep, MockStep, StepValidationError
    from .base_step import create_mock_step, validate_step_implementation
    _BASE_STEP_AVAILABLE = True
    logger.debug("BaseStep classes imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import BaseStep: {e}")
    BaseStep = None
    MockStep = None
    StepValidationError = None
    create_mock_step = None
    validate_step_implementation = None

try:
    from .step_registry import StepRegistry, StepRegistrationError, register_step
    from .step_registry import get_available_step_types, create_step_from_config, is_step_type_available
    # ADDED: Import register_step_safe from step_registry if available
    try:
        from .step_registry import register_step_safe as _registry_register_step_safe
        _REGISTRY_REGISTER_STEP_SAFE_AVAILABLE = True
    except ImportError:
        _registry_register_step_safe = None
        _REGISTRY_REGISTER_STEP_SAFE_AVAILABLE = False
        
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
    _registry_register_step_safe = None
    _REGISTRY_REGISTER_STEP_SAFE_AVAILABLE = False

# Check if core functionality is available
_CORE_AVAILABLE = _BASE_STEP_AVAILABLE and _STEP_REGISTRY_AVAILABLE

# Step registration tracking
_REGISTRATION_ERRORS = {}
_REGISTERED_MODULES = set()
_REGISTRATION_LOG = []


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
        'core_available': _CORE_AVAILABLE,
        'register_step_safe_from_registry': _REGISTRY_REGISTER_STEP_SAFE_AVAILABLE
    }


def get_missing_components() -> List[str]:
    """Get list of missing components."""
    status = get_component_status()
    return [name for name, available in status.items() if not available and name != 'core_available']


# FIXED: Enhanced registration function with improved error handling
def register_step_safe(step_type: str, 
                      step_class: Type['BaseStep'],
                      category: Optional[str] = None,
                      aliases: Optional[List[str]] = None,
                      **kwargs) -> bool:
    """
    FIXED: Safely register a step with comprehensive error handling and logging.
    
    This function now properly handles the category parameter and provides
    better error reporting and fallback mechanisms.
    
    Args:
        step_type: Step type identifier
        step_class: Step class that inherits from BaseStep
        category: Optional category for organization (FIXED: Added this parameter)
        aliases: Optional list of aliases
        **kwargs: Additional registration parameters
        
    Returns:
        True if registration successful, False otherwise
    """
    if not _STEP_REGISTRY_AVAILABLE:
        error_msg = "StepRegistry not available for step registration"
        logger.error(error_msg)
        _REGISTRATION_ERRORS[step_type] = error_msg
        return False
    
    try:
        # FIXED: Use the register_step_safe from step_registry if available
        if _REGISTRY_REGISTER_STEP_SAFE_AVAILABLE and _registry_register_step_safe:
            logger.debug(f"Using step_registry.register_step_safe for {step_type}")
            success = _registry_register_step_safe(
                step_type=step_type,
                step_class=step_class,
                category=category,
                aliases=aliases,
                **kwargs
            )
            if success:
                # Log successful registration
                log_entry = {
                    'step_type': step_type,
                    'class_name': step_class.__name__,
                    'module': step_class.__module__,
                    'status': 'success',
                    'method': 'registry_register_step_safe'
                }
                _REGISTRATION_LOG.append(log_entry)
                logger.info(f"✓ Successfully registered step: {step_type} -> {step_class.__name__}")
            return success
        
        # FALLBACK: Use StepRegistry.register directly
        logger.debug(f"Using StepRegistry.register directly for {step_type}")
        
        # Validate step class before registration
        if BaseStep and not issubclass(step_class, BaseStep):
            error_msg = f"Step class {step_class.__name__} must inherit from BaseStep"
            logger.error(error_msg)
            _REGISTRATION_ERRORS[step_type] = error_msg
            return False
            
        # FIXED: Register with StepRegistry using corrected signature
        StepRegistry.register(
            step_type=step_type,
            step_class=step_class,
            category=category,  # FIXED: Now properly passed
            aliases=aliases,    # FIXED: Now properly passed
            **kwargs
        )
        
        # Log successful registration
        log_entry = {
            'step_type': step_type,
            'class_name': step_class.__name__,
            'module': step_class.__module__,
            'status': 'success',
            'method': 'registry_register_direct'
        }
        _REGISTRATION_LOG.append(log_entry)
        
        logger.info(f"✓ Successfully registered step: {step_type} -> {step_class.__name__}")
        return True
        
    except Exception as e:
        error_msg = f"Failed to register step '{step_type}': {e}"
        logger.error(error_msg)
        _REGISTRATION_ERRORS[step_type] = error_msg
        
        # Log failed registration
        log_entry = {
            'step_type': step_type,
            'class_name': step_class.__name__ if step_class else 'Unknown',
            'module': step_class.__module__ if step_class else 'Unknown',
            'status': 'failed',
            'error': str(e),
            'method': 'error'
        }
        _REGISTRATION_LOG.append(log_entry)
        
        return False


def auto_register_step_modules() -> Dict[str, Any]:
    """
    Automatically discover and register step modules from standard locations.
    
    Returns:
        Registration summary with success/failure counts
    """
    if not _CORE_AVAILABLE:
        logger.warning("Core functionality not available for auto-registration")
        return {'success': 0, 'failed': 0, 'errors': ['Core functionality not available']}
    
    # Standard step module locations
    step_module_paths = [
        'orchestrator.steps.data_acquisition',
        'orchestrator.steps.preprocessing', 
        'orchestrator.steps.feature_extraction',
        'orchestrator.steps.segmentation',
        'orchestrator.steps.modeling',
        'orchestrator.steps.prediction',
        'orchestrator.steps.visualization'
    ]
    
    success_count = 0
    failed_count = 0
    registration_errors = []
    
    for module_path in step_module_paths:
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            _REGISTERED_MODULES.add(module_path)
            
            # Module imports trigger step registrations through __init__.py
            logger.debug(f"✓ Successfully imported step module: {module_path}")
            success_count += 1
            
        except ImportError as e:
            logger.debug(f"Step module not available: {module_path} ({e})")
            failed_count += 1
            registration_errors.append(f"{module_path}: {e}")
        except Exception as e:
            logger.warning(f"Error importing step module {module_path}: {e}")
            failed_count += 1
            registration_errors.append(f"{module_path}: {e}")
    
    # Log summary
    if success_count > 0:
        logger.info(f"✓ Auto-registered {success_count} step modules")
    if failed_count > 0:
        logger.debug(f"⚠ {failed_count} step modules not available")
    
    return {
        'success': success_count,
        'failed': failed_count,
        'errors': registration_errors,
        'registered_modules': list(_REGISTERED_MODULES)
    }


def validate_all_registrations() -> Dict[str, Any]:
    """
    Validate all current step registrations.
    
    Returns:
        Validation summary with details of any issues
    """
    if not _STEP_REGISTRY_AVAILABLE:
        return {
            'valid': False,
            'error': 'StepRegistry not available',
            'details': {}
        }
    
    try:
        registered_types = StepRegistry.get_registered_types()
        validation_results = {}
        
        for step_type in registered_types:
            try:
                step_class = StepRegistry.get_step_class(step_type)
                
                # Basic validation
                validation_errors = []
                if not hasattr(step_class, 'execute'):
                    validation_errors.append("Missing execute method")
                
                if BaseStep and not issubclass(step_class, BaseStep):
                    validation_errors.append("Does not inherit from BaseStep")
                
                validation_results[step_type] = {
                    'valid': len(validation_errors) == 0,
                    'class_name': step_class.__name__,
                    'module': step_class.__module__,
                    'errors': validation_errors
                }
                
            except Exception as e:
                validation_results[step_type] = {
                    'valid': False,
                    'error': str(e)
                }
        
        # Overall summary
        valid_count = sum(1 for r in validation_results.values() if r.get('valid', False))
        total_count = len(validation_results)
        
        return {
            'valid': valid_count == total_count,
            'summary': {
                'total': total_count,
                'valid': valid_count,
                'invalid': total_count - valid_count
            },
            'details': validation_results
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Validation failed: {e}",
            'details': {}
        }


def get_registration_status() -> Dict[str, Any]:
    """Get comprehensive registration status and debugging information.""" 
    status = {
        'core_available': _CORE_AVAILABLE,
        'component_status': get_component_status(),
        'registered_modules': list(_REGISTERED_MODULES),
        'registration_log': _REGISTRATION_LOG.copy(),
        'registration_errors': _REGISTRATION_ERRORS.copy()
    }
    
    if _STEP_REGISTRY_AVAILABLE:
        try:
            status['registered_steps'] = StepRegistry.get_registered_types()
            status['step_count'] = len(status['registered_steps'])
            status['registry_stats'] = StepRegistry.get_registry_stats()
        except Exception as e:
            status['registry_error'] = str(e)
    
    return status


def print_registration_status():
    """Print detailed registration status for debugging."""
    status = get_registration_status()
    
    print("=" * 60)
    print("STEP REGISTRATION STATUS")
    print("=" * 60)
    
    # Core status
    print(f"Core Available: {status['core_available']}")
    print(f"BaseStep Available: {status['component_status']['base_step']}")
    print(f"Registry Available: {status['component_status']['step_registry']}")
    print(f"Registry register_step_safe: {status['component_status']['register_step_safe_from_registry']}")
    
    # Registered modules
    print(f"\nRegistered Modules ({len(status['registered_modules'])}):")
    for module in status['registered_modules']:
        print(f"  ✓ {module}")
    
    # Registered steps
    if 'registered_steps' in status:
        print(f"\nRegistered Steps ({status['step_count']}):")
        for step_type in status['registered_steps']:
            print(f"  ✓ {step_type}")
    
    # Registration errors
    if status['registration_errors']:
        print(f"\nRegistration Errors ({len(status['registration_errors'])}):")
        for step_type, error in status['registration_errors'].items():
            print(f"  ✗ {step_type}: {error}")
    
    # Recent registrations
    if status['registration_log']:
        print(f"\nRecent Registrations ({len(status['registration_log'])}):")
        for entry in status['registration_log'][-10:]:  # Show last 10
            status_icon = "✓" if entry['status'] == 'success' else "✗"
            method = entry.get('method', 'unknown')
            print(f"  {status_icon} {entry['step_type']} -> {entry['class_name']} [{method}]")
    
    print("=" * 60)


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
        # Extract step_id if present, otherwise use step type
        step_id = step_config.get('id', step_config.get('type', 'unnamed_step'))
        return StepRegistry.create_step(step_id, step_config)
    except Exception as e:
        logger.error(f"Failed to create step: {e}")
        return None


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
            success = register_step_safe(
                step_type='mock',
                step_class=MockStep,
                category='testing',
                aliases=['test', 'mock_step'],
                description='Mock step for testing and development',
                builtin=True
            )
            if success:
                logger.debug("✓ Registered built-in MockStep")
    
    except Exception as e:
        logger.debug(f"Failed to register built-in steps: {e}")


# Development and testing utilities
def quick_test() -> Dict[str, Any]:
    """
    Run quick test of base module functionality.
    
    Returns:
        Test results summary
    """
    test_results = {
        'tests': {},
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'overall_status': 'unknown'
        }
    }
    
    # Test 1: Component availability
    test_results['tests']['component_availability'] = {
        'status': 'success' if _CORE_AVAILABLE else 'failed',
        'details': get_component_status()
    }
    test_results['summary']['total_tests'] += 1
    if _CORE_AVAILABLE:
        test_results['summary']['passed'] += 1
    else:
        test_results['summary']['failed'] += 1
    
    # Test 2: Auto-registration
    if _CORE_AVAILABLE:
        try:
            reg_results = auto_register_step_modules()
            test_results['tests']['auto_registration'] = {
                'status': 'success' if reg_results['success'] > 0 else 'partial',
                'details': reg_results
            }
            test_results['summary']['total_tests'] += 1
            if reg_results['success'] > 0:
                test_results['summary']['passed'] += 1
            else:
                test_results['summary']['failed'] += 1
        except Exception as e:
            test_results['tests']['auto_registration'] = {
                'status': 'failed',
                'details': f"Auto-registration failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['auto_registration'] = {
            'status': 'skipped',
            'details': 'Core not available'
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['skipped'] += 1
    
    # Test 3: Step registry functionality with register_step_safe
    if _STEP_REGISTRY_AVAILABLE:
        try:
            # Test basic registry operations
            initial_count = len(get_step_types())
            
            # Test mock step registration with register_step_safe
            if MockStep:
                success = register_step_safe(
                    step_type='test_registry_step',
                    step_class=MockStep,
                    category='testing',
                    aliases=['test_step_alias']
                )
                if success:
                    # Test retrieval
                    types = get_step_types()
                    if 'test_registry_step' in types and len(types) > initial_count:
                        test_results['tests']['registry_operations'] = {
                            'status': 'success',
                            'details': f"Registry test passed, {len(types)} types available"
                        }
                    else:
                        test_results['tests']['registry_operations'] = {
                            'status': 'failed',
                            'details': "Step not found after registration"
                        }
                    
                    # Cleanup
                    try:
                        StepRegistry.unregister('test_registry_step')
                    except:
                        pass
                else:
                    test_results['tests']['registry_operations'] = {
                        'status': 'failed',
                        'details': "Step registration failed"
                    }
            else:
                test_results['tests']['registry_operations'] = {
                    'status': 'skipped',
                    'details': "MockStep not available"
                }
                
            test_results['summary']['total_tests'] += 1
            if test_results['tests']['registry_operations']['status'] == 'success':
                test_results['summary']['passed'] += 1
            elif test_results['tests']['registry_operations']['status'] == 'failed':
                test_results['summary']['failed'] += 1
            else:
                test_results['summary']['skipped'] += 1
                
        except Exception as e:
            test_results['tests']['registry_operations'] = {
                'status': 'failed',
                'details': f"Registry test failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['registry_operations'] = {
            'status': 'skipped',
            'details': "StepRegistry not available"
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['skipped'] += 1
    
    # Test 4: Validation functionality
    if _BASE_STEP_AVAILABLE:
        try:
            class ValidStep(BaseStep):
                def execute(self, context):
                    return {'status': 'success', 'outputs': {}, 'metadata': {}}
            
            errors = validate_step_class(ValidStep)
            test_results['tests']['validation'] = {
                'status': 'success' if len(errors) == 0 else 'failed',
                'details': f"Validation errors: {errors}" if errors else "Validation passed"
            }
            
            test_results['summary']['total_tests'] += 1
            if len(errors) == 0:
                test_results['summary']['passed'] += 1
            else:
                test_results['summary']['failed'] += 1
                
        except Exception as e:
            test_results['tests']['validation'] = {
                'status': 'failed',
                'details': f"Validation test failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['validation'] = {
            'status': 'skipped',
            'details': "BaseStep not available"
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['skipped'] += 1
    
    # Determine overall status
    if test_results['summary']['failed'] == 0:
        if test_results['summary']['passed'] > 0:
            test_results['summary']['overall_status'] = 'success'
        else:
            test_results['summary']['overall_status'] = 'skipped'
    else:
        test_results['summary']['overall_status'] = 'failed'
    
    return test_results


def print_status():
    """Print current module status."""
    print("Steps Base Module Status")
    print("=" * 30)
    print(f"Version: {__version__}")
    print(f"Core Available: {_CORE_AVAILABLE}")
    print(f"BaseStep Available: {_BASE_STEP_AVAILABLE}")
    print(f"Registry Available: {_STEP_REGISTRY_AVAILABLE}")
    print(f"Registry register_step_safe: {_REGISTRY_REGISTER_STEP_SAFE_AVAILABLE}")
    
    if _STEP_REGISTRY_AVAILABLE:
        try:
            step_count = len(get_step_types())
            print(f"Registered Steps: {step_count}")
        except:
            print("Registered Steps: Unknown")
    
    missing = get_missing_components()
    if missing:
        print(f"Missing Components: {missing}")
    else:
        print("All Components Available: ✓")


def get_help() -> str:
    """Get help information for the base module."""
    return """
Steps Base Module Help
=====================

Quick Start:
1. Import base classes: from orchestrator.steps.base import BaseStep, StepRegistry
2. Create step class inheriting from BaseStep
3. Implement execute() method returning Dict with 'status', 'outputs', 'metadata'
4. Register step: register_step_safe('step_type', StepClass, category='your_category')
5. Test with create_mock_step() or MockStep
6. Use in JSON process definitions

Key Fixes Applied:
- Fixed register_step_safe() to properly handle category parameter
- Enhanced error handling and fallback mechanisms
- Improved import handling for step_registry components
- Better debugging and status reporting

Auto-Registration:
- Call auto_register_step_modules() to discover and register all step modules
- Check registration status with print_registration_status()
- Validate registrations with validate_all_registrations()

Debugging:
- Use print_status() to check module health
- Use quick_test() for comprehensive testing
- Use get_registration_status() for detailed debugging

For detailed documentation, see individual class docstrings.
"""


# Initialize module
def _initialize_module():
    """Initialize the base module with enhanced registration."""
    logger.info(f"Initializing steps base module v{__version__}")
    
    # Log component status
    status = get_component_status()
    if status['core_available']:
        logger.info("✓ Core base functionality available")
        
        # Register built-in steps
        _register_builtin_steps()
        
        # Auto-register step modules
        reg_results = auto_register_step_modules()
        if reg_results['success'] > 0:
            logger.info(f"✓ Auto-registered {reg_results['success']} step modules")
        
        # Log final registry status
        try:
            type_count = len(get_step_types())
            logger.info(f"Step registry initialized with {type_count} types")
        except:
            logger.debug("Could not get initial step type count")
            
    else:
        missing = get_missing_components()
        logger.warning(f"⚠ Core functionality incomplete, missing: {missing}")


# Export public API
__all__ = [
    # Core classes (may be None if not available)
    'BaseStep',
    'StepRegistry', 
    'MockStep',
    
    # Exception classes
    'StepValidationError',
    'StepRegistrationError',
    
    # Enhanced registration functions
    'register_step_safe',
    'auto_register_step_modules',
    'validate_all_registrations',
    'get_registration_status',
    'print_registration_status',
    
    # Utility functions
    'create_step',
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
    print("Running enhanced steps base module test...")
    
    # Print status
    print_status()
    
    # Print registration status
    print("\n")
    print_registration_status()
    
    # Run comprehensive test
    print("\n" + "="*60)
    print("COMPREHENSIVE FUNCTIONALITY TEST")
    print("="*60)
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
    
    # Validation test
    print(f"\n" + "="*60)
    print("REGISTRATION VALIDATION")
    print("="*60)
    validation = validate_all_registrations()
    if validation['valid']:
        print("✓ All registrations are valid")
    else:
        print(f"✗ Registration issues found: {validation.get('error', 'See details')}")
        if 'summary' in validation:
            print(f"Valid: {validation['summary']['valid']}/{validation['summary']['total']}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_status'] == 'success' else 1
    exit(exit_code)
