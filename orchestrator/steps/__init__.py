"""
Orchestrator Steps Module - Complete Step Library
================================================

This module provides the complete library of pipeline steps for the modular
orchestrator system, with enhanced registration management and validation.

Step Categories:
- base: Core infrastructure (BaseStep, StepRegistry)
- data_acquisition: Data acquisition from various sources
- preprocessing: Data preprocessing and cleaning
- feature_extraction: Feature calculation and extraction
- segmentation: Image/data segmentation
- modeling: Machine learning and statistical modeling
- prediction: Prediction and mapping
- visualization: Visualization and reporting

Enhanced Features:
- Automatic step discovery and registration
- Comprehensive validation and health monitoring
- Import error handling with graceful degradation
- Module dependency tracking
- Registration status reporting
"""

import logging
import warnings
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Module version
__version__ = "2.0.0"
__author__ = "TerraLux Development Team"

# Step registration tracking
_MODULE_REGISTRY = {}
_REGISTRATION_STATUS = {}
_IMPORT_ERRORS = {}
_STEP_COUNTS = {}

# Core base module (always try to import first)
try:
    from .base import *
    from .base import StepRegistry, BaseStep, register_step_safe
    from .base import auto_register_step_modules, get_registration_status
    _MODULE_REGISTRY['base'] = 'success'
    logger.debug("✓ Base module imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import base module: {e}")
    _MODULE_REGISTRY['base'] = 'failed'
    _IMPORT_ERRORS['base'] = str(e)
    # Create dummy objects to prevent further errors
    StepRegistry = None
    BaseStep = None
    register_step_safe = None


def _safe_import_step_module(module_name: str, module_path: str) -> bool:
    """
    Safely import a step module with error handling.
    
    Args:
        module_name: Human-readable module name
        module_path: Import path for the module
        
    Returns:
        True if import successful, False otherwise
    """
    try:
        module = __import__(module_path, fromlist=[''])
        _MODULE_REGISTRY[module_name] = 'success'
        
        # Count steps if StepRegistry is available
        if StepRegistry:
            try:
                current_steps = len(StepRegistry.get_registered_types())
                _STEP_COUNTS[module_name] = current_steps
            except:
                pass
        
        logger.debug(f"✓ Successfully imported {module_name} module")
        return True
        
    except ImportError as e:
        _MODULE_REGISTRY[module_name] = 'failed'
        _IMPORT_ERRORS[module_name] = str(e)
        logger.debug(f"✗ {module_name} module not available: {e}")
        return False
    except Exception as e:
        _MODULE_REGISTRY[module_name] = 'error'
        _IMPORT_ERRORS[module_name] = str(e)
        logger.warning(f"✗ Error importing {module_name} module: {e}")
        return False


def _register_step_modules():
    """Register all available step modules."""
    logger.info("Discovering and registering step modules...")
    
    # Step module definitions: (name, import_path, description)
    step_modules = [
        ('data_acquisition', 'orchestrator.steps.data_acquisition', 'Data acquisition from various sources'),
        ('preprocessing', 'orchestrator.steps.preprocessing', 'Data preprocessing and cleaning'),
        ('feature_extraction', 'orchestrator.steps.feature_extraction', 'Feature calculation and extraction'),
        ('segmentation', 'orchestrator.steps.segmentation', 'Image and data segmentation'),
        ('modeling', 'orchestrator.steps.modeling', 'Machine learning and statistical modeling'),
        ('prediction', 'orchestrator.steps.prediction', 'Prediction and mapping'),
        ('visualization', 'orchestrator.steps.visualization', 'Visualization and reporting')
    ]
    
    successful_imports = 0
    initial_step_count = 0
    
    # Get initial step count
    if StepRegistry:
        try:
            initial_step_count = len(StepRegistry.get_registered_types())
        except:
            pass
    
    # Import each module
    for module_name, module_path, description in step_modules:
        logger.debug(f"Importing {module_name}: {description}")
        if _safe_import_step_module(module_name, module_path):
            successful_imports += 1
    
    # Calculate final step counts
    final_step_count = 0
    if StepRegistry:
        try:
            final_step_count = len(StepRegistry.get_registered_types())
            new_steps = final_step_count - initial_step_count
            logger.info(f"✓ Registered {new_steps} new steps from {successful_imports} modules")
        except:
            logger.debug("Could not calculate step count changes")
    
    # Log summary
    total_modules = len(step_modules)
    if successful_imports == total_modules:
        logger.info(f"✓ All {total_modules} step modules imported successfully")
    elif successful_imports > 0:
        logger.info(f"✓ {successful_imports}/{total_modules} step modules imported successfully")
        failed_modules = [name for name, status in _MODULE_REGISTRY.items() 
                         if status in ['failed', 'error'] and name != 'base']
        if failed_modules:
            logger.debug(f"Missing modules: {failed_modules}")
    else:
        logger.warning("✗ No step modules could be imported")
    
    return {
        'successful': successful_imports,
        'total': total_modules,
        'initial_steps': initial_step_count,
        'final_steps': final_step_count
    }


def get_module_status() -> Dict[str, Any]:
    """Get comprehensive module status information."""
    status = {
        'module_registry': _MODULE_REGISTRY.copy(),
        'import_errors': _IMPORT_ERRORS.copy(),
        'step_counts': _STEP_COUNTS.copy(),
        'version': __version__
    }
    
    # Add step registry information if available
    if StepRegistry:
        try:
            status['total_registered_steps'] = len(StepRegistry.get_registered_types())
            status['registered_step_types'] = StepRegistry.get_registered_types()
            status['registry_stats'] = StepRegistry.get_registry_stats()
        except Exception as e:
            status['registry_error'] = str(e)
    else:
        status['registry_available'] = False
    
    # Calculate summary statistics
    successful_modules = len([s for s in _MODULE_REGISTRY.values() if s == 'success'])
    total_modules = len(_MODULE_REGISTRY)
    
    status['summary'] = {
        'successful_modules': successful_modules,
        'total_modules': total_modules,
        'success_rate': successful_modules / total_modules if total_modules > 0 else 0,
        'core_available': _MODULE_REGISTRY.get('base') == 'success'
    }
    
    return status


def get_available_step_types() -> List[str]:
    """Get list of all available step types."""
    if not StepRegistry:
        logger.warning("StepRegistry not available")
        return []
    
    try:
        return StepRegistry.get_registered_types()
    except Exception as e:
        logger.error(f"Failed to get step types: {e}")
        return []


def get_step_categories() -> Dict[str, List[str]]:
    """Get step types organized by category."""
    if not StepRegistry:
        return {}
    
    try:
        return StepRegistry.get_categories()
    except Exception as e:
        logger.error(f"Failed to get step categories: {e}")
        return {}


def validate_step_registrations() -> Dict[str, Any]:
    """Validate all step registrations."""
    if not StepRegistry:
        return {
            'valid': False,
            'error': 'StepRegistry not available'
        }
    
    try:
        # Use base module's validation if available
        from .base import validate_all_registrations
        return validate_all_registrations()
    except ImportError:
        # Fallback validation
        try:
            step_types = StepRegistry.get_registered_types()
            valid_count = 0
            total_count = len(step_types)
            validation_errors = []
            
            for step_type in step_types:
                try:
                    step_class = StepRegistry.get_step_class(step_type)
                    if hasattr(step_class, 'execute'):
                        valid_count += 1
                    else:
                        validation_errors.append(f"{step_type}: Missing execute method")
                except Exception as e:
                    validation_errors.append(f"{step_type}: {e}")
            
            return {
                'valid': len(validation_errors) == 0,
                'summary': {
                    'total': total_count,
                    'valid': valid_count,
                    'invalid': total_count - valid_count
                },
                'errors': validation_errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation failed: {e}"
            }


def print_module_status():
    """Print comprehensive module status."""
    status = get_module_status()
    
    print("=" * 70)
    print("ORCHESTRATOR STEPS MODULE STATUS")
    print("=" * 70)
    
    # Basic info
    print(f"Version: {status['version']}")
    print(f"Core Available: {status['summary']['core_available']}")
    print(f"Modules: {status['summary']['successful_modules']}/{status['summary']['total_modules']} successful")
    print(f"Success Rate: {status['summary']['success_rate']:.1%}")
    
    # Module details
    print(f"\nModule Status:")
    for module_name, module_status in status['module_registry'].items():
        icon = "✓" if module_status == 'success' else "✗"
        print(f"  {icon} {module_name}: {module_status}")
        if module_status != 'success' and module_name in status['import_errors']:
            print(f"    Error: {status['import_errors'][module_name]}")
    
    # Step registry info
    if 'total_registered_steps' in status:
        print(f"\nRegistered Steps: {status['total_registered_steps']}")
        
        # Show step categories if available
        categories = get_step_categories()
        if categories:
            print(f"Step Categories:")
            for category, steps in categories.items():
                print(f"  • {category}: {len(steps)} steps")
    else:
        print(f"\nStep Registry: Not available")
    
    # Validation
    validation = validate_step_registrations()
    if validation['valid']:
        print(f"\nValidation: ✓ All registrations valid")
    else:
        print(f"\nValidation: ✗ Issues found")
        if 'summary' in validation:
            print(f"  Valid: {validation['summary']['valid']}/{validation['summary']['total']}")
    
    print("=" * 70)


def run_comprehensive_test() -> Dict[str, Any]:
    """Run comprehensive test of the steps module."""
    test_results = {
        'tests': {},
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
    }
    
    # Test 1: Module imports
    test_results['tests']['module_imports'] = {
        'status': 'success' if _MODULE_REGISTRY.get('base') == 'success' else 'failed',
        'details': get_module_status()
    }
    test_results['summary']['total_tests'] += 1
    if _MODULE_REGISTRY.get('base') == 'success':
        test_results['summary']['passed'] += 1
    else:
        test_results['summary']['failed'] += 1
    
    # Test 2: Step registration
    if StepRegistry:
        try:
            step_types = get_available_step_types()
            test_results['tests']['step_registration'] = {
                'status': 'success' if len(step_types) > 0 else 'failed',
                'details': f"Found {len(step_types)} registered step types"
            }
            test_results['summary']['total_tests'] += 1
            if len(step_types) > 0:
                test_results['summary']['passed'] += 1
            else:
                test_results['summary']['failed'] += 1
        except Exception as e:
            test_results['tests']['step_registration'] = {
                'status': 'failed',
                'details': f"Step registration test failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['step_registration'] = {
            'status': 'skipped',
            'details': 'StepRegistry not available'
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['skipped'] += 1
    
    # Test 3: Validation
    try:
        validation_results = validate_step_registrations()
        test_results['tests']['validation'] = {
            'status': 'success' if validation_results['valid'] else 'failed',
            'details': validation_results
        }
        test_results['summary']['total_tests'] += 1
        if validation_results['valid']:
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
    
    # Test 4: Core functionality
    if StepRegistry and BaseStep:
        try:
            # Test step creation
            from .base import create_mock_step_safe
            mock_step = create_mock_step_safe('test_step', 'mock')
            
            test_results['tests']['core_functionality'] = {
                'status': 'success' if mock_step is not None else 'failed',
                'details': 'Core functionality test completed'
            }
            test_results['summary']['total_tests'] += 1
            if mock_step is not None:
                test_results['summary']['passed'] += 1
            else:
                test_results['summary']['failed'] += 1
        except Exception as e:
            test_results['tests']['core_functionality'] = {
                'status': 'failed',
                'details': f"Core functionality test failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['core_functionality'] = {
            'status': 'skipped',
            'details': 'Core components not available'
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


def get_help() -> str:
    """Get help information for the steps module."""
    return """
Orchestrator Steps Module Help
=============================

This module provides the complete library of pipeline steps for the modular
orchestrator system.

Quick Start:
1. Check module status: print_module_status()
2. List available steps: get_available_step_types()
3. Validate registrations: validate_step_registrations()
4. Run comprehensive test: run_comprehensive_test()

Step Categories:
- data_acquisition: Satellite data, DEM, local files
- preprocessing: Atmospheric correction, geometric correction, cloud masking
- feature_extraction: Spectral indices, topographic features, texture analysis
- segmentation: SLIC, watershed, region growing
- modeling: Random forest, CNN, clustering
- prediction: Risk mapping, mineral prospectivity, uncertainty analysis
- visualization: Maps, plots, reports

Creating Custom Steps:
1. Inherit from BaseStep
2. Implement execute(self, context) method
3. Register with StepRegistry.register() or register_step_safe()
4. Use in JSON process definitions

For detailed examples and documentation, see individual step modules.
"""


# Initialize the module
def _initialize_steps_module():
    """Initialize the steps module with comprehensive registration."""
    logger.info(f"Initializing orchestrator steps module v{__version__}")
    
    # Check if base module is available
    if _MODULE_REGISTRY.get('base') != 'success':
        logger.error("Base module not available - step registration will be limited")
        return
    
    # Register all step modules
    registration_results = _register_step_modules()
    
    # Log final status
    if registration_results['successful'] > 0:
        logger.info(f"✓ Steps module initialized successfully")
        logger.info(f"  - {registration_results['successful']}/{registration_results['total']} modules loaded")
        logger.info(f"  - {registration_results['final_steps']} total registered steps")
    else:
        logger.warning("⚠ Steps module initialized with limited functionality")
    
    # Issue warnings if critical modules are missing
    critical_modules = ['data_acquisition', 'preprocessing', 'feature_extraction']
    missing_critical = [m for m in critical_modules if _MODULE_REGISTRY.get(m) != 'success']
    
    if missing_critical:
        warnings.warn(
            f"Critical step modules not available: {missing_critical}. "
            f"Pipeline functionality may be limited. "
            f"Install missing dependencies or check module implementations.",
            RuntimeWarning
        )


# Export public API
__all__ = [
    # Core functionality (from base module)
    'BaseStep',
    'StepRegistry',
    'register_step_safe',
    
    # Module management
    'get_module_status',
    'get_available_step_types', 
    'get_step_categories',
    'validate_step_registrations',
    'print_module_status',
    'run_comprehensive_test',
    'get_help',
    
    # Version info
    '__version__',
    '__author__'
]

# Initialize module on import
_initialize_steps_module()

# Quick test and status when run directly
if __name__ == "__main__":
    print("Running Orchestrator Steps Module Test...")
    print("=" * 50)
    
    # Print module status
    print_module_status()
    
    # Run comprehensive test
    print("\n" + "=" * 50)
    print("COMPREHENSIVE FUNCTIONALITY TEST")
    print("=" * 50)
    
    test_results = run_comprehensive_test()
    
    print(f"\nTest Results:")
    print(f"Total Tests: {test_results['summary']['total_tests']}")
    print(f"Passed: {test_results['summary']['passed']}")
    print(f"Failed: {test_results['summary']['failed']}")
    print(f"Skipped: {test_results['summary']['skipped']}")
    print(f"Overall Status: {test_results['summary']['overall_status'].upper()}")
    
    # Show test details
    print(f"\nTest Details:")
    for test_name, test_result in test_results['tests'].items():
        status_icon = {"success": "✓", "failed": "✗", "skipped": "⊝"}[test_result['status']]
        print(f"  {status_icon} {test_name}: {test_result['status']}")
    
    # Show failed tests with details
    failed_tests = [name for name, result in test_results['tests'].items() 
                   if result['status'] == 'failed']
    if failed_tests:
        print(f"\nFailed Test Details:")
        for test_name in failed_tests:
            details = test_results['tests'][test_name]['details']
            print(f"  ✗ {test_name}: {details}")
    
    # Show help if there are issues
    if test_results['summary']['failed'] > 0:
        print(f"\nFor help resolving issues:")
        print("python -c \"from orchestrator.steps import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if test_results['summary']['overall_status'] == 'success' else 1
    exit(exit_code)
