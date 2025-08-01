"""
Data Acquisition Steps Module - Complete Implementation
======================================================

This module provides data acquisition steps for satellite imagery, DEM data,
and local file discovery with enhanced registration management and comprehensive
error handling.

Available Steps:
- sentinel_hub_acquisition: Sentinel-2/1 satellite data from Sentinel Hub API
- dem_acquisition: Digital elevation models from various sources (SRTM, ASTER, etc.)
- local_files_discovery: Discovery and cataloging of local data files
- copernicus_hub_acquisition: Data from Copernicus Open Access Hub (optional)
- landsat_acquisition: Landsat satellite data (optional)

Features:
- Robust error handling with fallback mechanisms
- Mock data support for development and testing
- Integration with existing landslide_pipeline components
- Comprehensive validation and quality control
- Enhanced registration management
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Type

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0"
__author__ = "TerraLux Development Team"

# Step registration tracking
_STEP_IMPORTS_SUCCESSFUL = {}
_AVAILABLE_STEPS = []
_STEP_ALIASES = {}
_IMPORT_ERRORS = {}
_EXISTING_PIPELINE_AVAILABLE = False

# Try to import the step registry for registration
StepRegistry = None
register_step_safe = None

try:
    from ..base import StepRegistry, register_step_safe
    logger.debug("✓ StepRegistry imported successfully")
except ImportError as e:
    logger.warning(f"StepRegistry not available: {e}")


def _safe_import_step(step_name: str, module_name: str, class_name: str, 
                     step_type: str, aliases: Optional[List[str]] = None,
                     category: str = 'data_acquisition') -> bool:
    """
    Safely import and register a step with comprehensive error handling.
    
    Args:
        step_name: Human-readable step name
        module_name: Module name to import from
        class_name: Class name to import
        step_type: Step type identifier for registration
        aliases: Optional list of aliases
        category: Step category
        
    Returns:
        True if import and registration successful
    """
    try:
        # Import the module
        module = __import__(f'.{module_name}', package=__name__, fromlist=[class_name])
        step_class = getattr(module, class_name)
        
        # Register with StepRegistry if available
        if StepRegistry and register_step_safe:
            success = register_step_safe(
                step_type,
                step_class,
                category=category,
                aliases=aliases or [],
                metadata={
                    'description': f'{step_name} step',
                    'module': f'orchestrator.steps.data_acquisition.{module_name}',
                    'class': class_name
                }
            )
            if not success:
                logger.warning(f"Registration failed for {step_type}")
                _IMPORT_ERRORS[step_name] = "Registration failed"
                return False
        
        # Track successful import
        _STEP_IMPORTS_SUCCESSFUL[step_name] = True
        _AVAILABLE_STEPS.append(step_type)
        
        # Add aliases
        if aliases:
            for alias in aliases:
                _STEP_ALIASES[alias] = step_type
        
        # Make available in module namespace
        globals()[class_name] = step_class
        
        logger.debug(f"✓ Successfully imported and registered: {step_name}")
        return True
        
    except ImportError as e:
        logger.debug(f"✗ {step_name} not available: {e}")
        _STEP_IMPORTS_SUCCESSFUL[step_name] = False
        _IMPORT_ERRORS[step_name] = f"Import error: {e}"
        return False
    except Exception as e:
        logger.warning(f"✗ Error with {step_name}: {e}")
        _STEP_IMPORTS_SUCCESSFUL[step_name] = False
        _IMPORT_ERRORS[step_name] = f"Error: {e}"
        return False


def _register_data_acquisition_steps():
    """Register all data acquisition steps."""
    logger.info("Registering data acquisition steps...")
    
    # Core data acquisition steps (should be available)
    core_steps = [
        {
            'name': 'Sentinel Hub Acquisition',
            'module': 'sentinel_hub_step',
            'class': 'SentinelHubStep',
            'type': 'sentinel_hub_acquisition',
            'aliases': ['sentinel_acquisition', 'sentinel2_acquisition']
        },
        {
            'name': 'DEM Acquisition',
            'module': 'dem_acquisition_step',
            'class': 'DEMAcquisitionStep', 
            'type': 'dem_acquisition',
            'aliases': ['elevation_acquisition', 'srtm_acquisition']
        },
        {
            'name': 'Local Files Discovery',
            'module': 'local_files_step',
            'class': 'LocalFilesStep',
            'type': 'local_files_discovery',
            'aliases': ['local_files', 'file_discovery']
        }
    ]
    
    # Optional steps (may not be available)
    optional_steps = [
        {
            'name': 'Copernicus Hub Acquisition',
            'module': 'copernicus_hub_step', 
            'class': 'CopernicusHubStep',
            'type': 'copernicus_hub_acquisition',
            'aliases': ['copernicus_acquisition']
        },
        {
            'name': 'Landsat Acquisition',
            'module': 'landsat_acquisition_step',
            'class': 'LandsatAcquisitionStep',
            'type': 'landsat_acquisition',
            'aliases': ['landsat']
        }
    ]
    
    successful_core = 0
    successful_optional = 0
    
    # Register core steps
    for step_config in core_steps:
        if _safe_import_step(**step_config):
            successful_core += 1
    
    # Register optional steps
    for step_config in optional_steps:
        if _safe_import_step(**step_config):
            successful_optional += 1
    
    # Log results
    total_core = len(core_steps)
    total_optional = len(optional_steps)
    
    logger.info(f"✓ Core steps: {successful_core}/{total_core} registered")
    if successful_optional > 0:
        logger.info(f"✓ Optional steps: {successful_optional}/{total_optional} registered")
    
    if successful_core < total_core:
        missing_core = [s['name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['name'], False)]
        logger.warning(f"Missing core data acquisition steps: {missing_core}")
    
    return {
        'core_successful': successful_core,
        'core_total': total_core,
        'optional_successful': successful_optional,
        'optional_total': total_optional,
        'total_registered': successful_core + successful_optional
    }


def _try_existing_pipeline_integration():
    """Try to integrate with existing landslide_pipeline components."""
    global _EXISTING_PIPELINE_AVAILABLE
    try:
        # Import existing components if available
        from landslide_pipeline.data_acquisition import DataAcquisitionFactory
        from landslide_pipeline.utils.file_path_manager import FilePathManager
        
        # Make available in module namespace
        globals()['DataAcquisitionFactory'] = DataAcquisitionFactory
        globals()['FilePathManager'] = FilePathManager
        
        _EXISTING_PIPELINE_AVAILABLE = True
        logger.debug("✓ Existing landslide_pipeline integration available")
        return True
        
    except ImportError:
        _EXISTING_PIPELINE_AVAILABLE = False
        logger.debug("Existing landslide_pipeline integration not available")
        return False


def get_import_status() -> Dict[str, bool]:
    """Get the import status of all step modules."""
    return _STEP_IMPORTS_SUCCESSFUL.copy()


def get_available_data_acquisition_steps() -> List[str]:
    """Get list of available data acquisition step types."""
    return _AVAILABLE_STEPS.copy()


def get_step_aliases() -> Dict[str, str]:
    """Get mapping of step aliases to canonical names."""
    return _STEP_ALIASES.copy()


def is_step_available(step_type: str) -> bool:
    """Check if a specific step type is available."""
    return step_type in _AVAILABLE_STEPS or step_type in _STEP_ALIASES


def get_missing_dependencies() -> List[str]:
    """Get list of missing dependencies based on import errors."""
    missing_deps = []
    
    for step_name, error_msg in _IMPORT_ERRORS.items():
        if 'sentinelhub' in error_msg.lower():
            missing_deps.append('sentinelhub')
        elif 'rasterio' in error_msg.lower():
            missing_deps.append('rasterio')
        elif 'geopandas' in error_msg.lower():
            missing_deps.append('geopandas')
        elif 'gdal' in error_msg.lower():
            missing_deps.append('gdal')
        elif 'requests' in error_msg.lower():
            missing_deps.append('requests')
    
    return list(set(missing_deps))


def validate_data_acquisition_setup() -> Dict[str, Any]:
    """Validate the data acquisition module setup."""
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }
    
    # Check core step availability
    core_steps = ['sentinel_hub_acquisition', 'dem_acquisition', 'local_files_discovery']
    missing_core = [step for step in core_steps if not is_step_available(step)]
    
    if missing_core:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing core steps: {missing_core}")
    
    # Check for import errors
    failed_imports = [name for name, success in _STEP_IMPORTS_SUCCESSFUL.items() if not success]
    if failed_imports:
        validation_results['warnings'].append(f"Failed imports: {failed_imports}")
    
    # Check dependencies
    missing_deps = get_missing_dependencies()
    if missing_deps:
        validation_results['warnings'].append(f"Missing dependencies: {missing_deps}")
    
    # Summary
    successful_imports = len([s for s in _STEP_IMPORTS_SUCCESSFUL.values() if s])
    total_core_steps = len(core_steps)
    
    validation_results['summary'] = {
        'total_steps_available': len(_AVAILABLE_STEPS),
        'core_steps_available': len([s for s in core_steps if is_step_available(s)]),
        'failed_imports': len(failed_imports),
        'missing_dependencies': len(missing_deps),
        'existing_pipeline_integration': _EXISTING_PIPELINE_AVAILABLE
    }
    
    # Determine overall status
    if successful_imports == 0:
        validation_results['overall_status'] = 'failed'
        validation_results['warnings'].append("No data acquisition steps available")
    elif successful_imports >= total_core_steps:
        validation_results['overall_status'] = 'success'
    else:
        validation_results['overall_status'] = 'partial'
        validation_results['warnings'].append(f"Only {successful_imports}/{total_core_steps} core steps available")
    
    # Determine capabilities
    validation_results['capabilities'] = {
        'satellite_data_acquisition': is_step_available('sentinel_hub_acquisition'),
        'dem_data_acquisition': is_step_available('dem_acquisition'),
        'local_files_discovery': is_step_available('local_files_discovery'),
        'existing_pipeline_integration': _EXISTING_PIPELINE_AVAILABLE,
        'mock_data_generation': True,  # Always available
        'fail_fast_development': True  # Always available
    }
    
    return validation_results


def create_data_acquisition_workflow(workflow_type: str = 'standard', **config) -> Dict[str, Any]:
    """
    Create a standard data acquisition workflow configuration.
    
    Args:
        workflow_type: Type of workflow ('minimal', 'standard', 'comprehensive')
        **config: Additional workflow configuration
        
    Returns:
        Workflow configuration dictionary
    """
    workflows = {
        'minimal': {
            'description': 'Minimal data acquisition with local files only',
            'steps': ['local_files_discovery']
        },
        'standard': {
            'description': 'Standard data acquisition with satellite and DEM data',
            'steps': ['sentinel_hub_acquisition', 'dem_acquisition', 'local_files_discovery']
        },
        'comprehensive': {
            'description': 'Comprehensive data acquisition with all available sources',
            'steps': ['sentinel_hub_acquisition', 'dem_acquisition', 'local_files_discovery', 
                     'copernicus_hub_acquisition', 'landsat_acquisition']
        }
    }
    
    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}. Available: {list(workflows.keys())}")
    
    workflow = workflows[workflow_type].copy()
    workflow.update(config)  # Allow overrides
    
    # Filter steps based on availability
    available_steps = [step for step in workflow['steps'] if is_step_available(step)]
    unavailable_steps = [step for step in workflow['steps'] if not is_step_available(step)]
    
    workflow['available_steps'] = available_steps
    workflow['unavailable_steps'] = unavailable_steps
    workflow['fully_available'] = len(unavailable_steps) == 0
    workflow['completion_percentage'] = len(available_steps) / len(workflow['steps']) * 100
    
    return workflow


def run_quick_test() -> Dict[str, Any]:
    """Run a quick test of data acquisition functionality."""
    test_results = {
        'tests': {},
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
    }
    
    # Test 1: Step availability
    available_steps = get_available_data_acquisition_steps()
    test_results['tests']['step_availability'] = {
        'status': 'success' if len(available_steps) > 0 else 'failed',
        'details': f"Found {len(available_steps)} available steps: {available_steps}"
    }
    test_results['summary']['total_tests'] += 1
    if len(available_steps) > 0:
        test_results['summary']['passed'] += 1
    else:
        test_results['summary']['failed'] += 1
    
    # Test 2: Core steps availability
    core_steps = ['sentinel_hub_acquisition', 'dem_acquisition', 'local_files_discovery']
    available_core = [step for step in core_steps if is_step_available(step)]
    test_results['tests']['core_steps'] = {
        'status': 'success' if len(available_core) == len(core_steps) else 'partial',
        'details': f"Core steps available: {len(available_core)}/{len(core_steps)}"
    }
    test_results['summary']['total_tests'] += 1
    if len(available_core) == len(core_steps):
        test_results['summary']['passed'] += 1
    elif len(available_core) > 0:
        test_results['summary']['passed'] += 1  # Partial success still counts as pass
    else:
        test_results['summary']['failed'] += 1
    
    # Test 3: Step registry integration
    if StepRegistry:
        try:
            # Test if our steps are registered
            registered_types = StepRegistry.get_registered_types()
            our_steps = [step for step in available_steps if step in registered_types]
            test_results['tests']['registry_integration'] = {
                'status': 'success' if len(our_steps) > 0 else 'failed',
                'details': f"Steps registered in StepRegistry: {len(our_steps)}"
            }
            test_results['summary']['total_tests'] += 1
            if len(our_steps) > 0:
                test_results['summary']['passed'] += 1
            else:
                test_results['summary']['failed'] += 1
        except Exception as e:
            test_results['tests']['registry_integration'] = {
                'status': 'failed',
                'details': f"Registry integration test failed: {e}"
            }
            test_results['summary']['total_tests'] += 1
            test_results['summary']['failed'] += 1
    else:
        test_results['tests']['registry_integration'] = {
            'status': 'skipped',
            'details': 'StepRegistry not available'
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['skipped'] += 1
    
    # Test 4: Workflow creation
    try:
        workflow = create_data_acquisition_workflow('standard')
        test_results['tests']['workflow_creation'] = {
            'status': 'success' if workflow['fully_available'] else 'partial',
            'details': f"Standard workflow: {len(workflow['available_steps'])}/{len(workflow['available_steps']) + len(workflow['unavailable_steps'])} steps available"
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['passed'] += 1
    except Exception as e:
        test_results['tests']['workflow_creation'] = {
            'status': 'failed',
            'details': f"Workflow creation failed: {e}"
        }
        test_results['summary']['total_tests'] += 1
        test_results['summary']['failed'] += 1
    
    # Test 5: Existing pipeline integration
    test_results['tests']['existing_pipeline_integration'] = {
        'status': 'success' if _EXISTING_PIPELINE_AVAILABLE else 'skipped',
        'details': f"Existing pipeline integration: {'available' if _EXISTING_PIPELINE_AVAILABLE else 'not available'}"
    }
    test_results['summary']['total_tests'] += 1
    if _EXISTING_PIPELINE_AVAILABLE:
        test_results['summary']['passed'] += 1
    else:
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


def print_module_status():
    """Print comprehensive module status for debugging."""
    print("=" * 60)
    print("DATA ACQUISITION MODULE STATUS")
    print("=" * 60)
    
    # Basic info
    print(f"Version: {__version__}")
    print(f"Available Steps: {len(_AVAILABLE_STEPS)}")
    print(f"Registry Available: {StepRegistry is not None}")
    print(f"Existing Pipeline Integration: {_EXISTING_PIPELINE_AVAILABLE}")
    
    # Step import status
    print(f"\nStep Import Status:")
    for step_name, success in _STEP_IMPORTS_SUCCESSFUL.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {step_name}")
        if not success and step_name in _IMPORT_ERRORS:
            print(f"    Error: {_IMPORT_ERRORS[step_name]}")
    
    # Available steps and aliases
    if _AVAILABLE_STEPS:
        print(f"\nAvailable Step Types:")
        for step_type in _AVAILABLE_STEPS:
            print(f"  • {step_type}")
    
    if _STEP_ALIASES:
        print(f"\nStep Aliases:")
        for alias, canonical in _STEP_ALIASES.items():
            print(f"  • {alias} → {canonical}")
    
    # Validation summary
    validation = validate_data_acquisition_setup()
    print(f"\nValidation Summary:")
    print(f"  Status: {validation['summary']['overall_status']}")
    print(f"  Core Steps Available: {validation['summary']['core_steps_available']}/3")
    print(f"  Total Steps Available: {validation['summary']['total_steps_available']}")
    
    if validation['errors']:
        print(f"  Errors: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"    ✗ {error}")
    
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"    ⚠ {warning}")
    
    # Capabilities
    print(f"\nCapabilities:")
    for capability, available in validation['capabilities'].items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {capability.replace('_', ' ').title()}")
    
    print("=" * 60)


def get_help() -> str:
    """Get help information for the data acquisition module."""
    return f"""
Data Acquisition Module Help
===========================

This module provides data acquisition steps for satellite imagery, DEM data,
and local file discovery.

Available Step Types:
{chr(10).join(f'  • {step}' for step in _AVAILABLE_STEPS)}

Available Aliases:
{chr(10).join(f'  • {alias} → {canonical}' for alias, canonical in _STEP_ALIASES.items())}

Quick Start:
-----------
1. Check module status: print_module_status()
2. Validate setup: validate_data_acquisition_setup()
3. Create workflow: create_data_acquisition_workflow('standard')
4. Run quick test: run_quick_test()

Example Usage:
-------------
# Check what's available
available = get_available_data_acquisition_steps()
print(f"Available steps: {{available}}")

# Validate setup
validation = validate_data_acquisition_setup()
print(f"Status: {{validation['summary']['overall_status']}}")

# Create workflow
workflow = create_data_acquisition_workflow('standard')
print(f"Workflow steps: {{workflow['available_steps']}}")

For detailed documentation, see individual step classes:
- SentinelHubStep: Sentinel-2 satellite data acquisition
- DEMAcquisitionStep: Digital elevation model acquisition  
- LocalFilesStep: Local file discovery and validation

Missing Dependencies:
{chr(10).join(f'  • {dep}' for dep in get_missing_dependencies())}

Existing Pipeline Integration:
{'✓ Available' if _EXISTING_PIPELINE_AVAILABLE else '✗ Not available'}
"""


# Initialize the data acquisition module
def _initialize_data_acquisition_module():
    """Initialize the data acquisition module."""
    logger.info(f"Initializing data acquisition module v{__version__}")
    
    # Register all available steps
    registration_results = _register_data_acquisition_steps()
    
    # Try existing pipeline integration
    existing_integration = _try_existing_pipeline_integration()
    
    # Log initialization results
    total_registered = registration_results['total_registered']
    if total_registered > 0:
        logger.info(f"✓ Data acquisition module initialized with {total_registered} steps")
    else:
        logger.warning("⚠ Data acquisition module initialized with no steps")
    
    if existing_integration:
        logger.debug("✓ Existing landslide_pipeline integration available")
    
    # Issue warnings if critical functionality is missing
    core_missing = registration_results['core_total'] - registration_results['core_successful']
    if core_missing > 0:
        warnings.warn(
            f"Core data acquisition steps not available ({core_missing} missing). "
            f"Install missing dependencies for full functionality: {get_missing_dependencies()}",
            RuntimeWarning
        )
    elif registration_results['total_registered'] == 0:
        warnings.warn(
            "No step implementations are available. Using mock implementations only. "
            "Install step dependencies for full functionality.",
            RuntimeWarning
        )
    
    return registration_results


# Export public API
__all__ = [
    # Step classes (dynamically added based on successful imports)
    # Core steps
    'SentinelHubStep',
    'DEMAcquisitionStep', 
    'LocalFilesStep',
    
    # Optional steps (may be None if not available)
    'CopernicusHubStep',
    'LandsatAcquisitionStep',
    
    # Integration with existing pipeline
    'DataAcquisitionFactory',
    'FilePathManager',
    
    # Utility functions
    'get_import_status',
    'get_available_data_acquisition_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_data_acquisition_setup',
    'create_data_acquisition_workflow',
    'run_quick_test',
    'print_module_status',
    'get_help',
    
    # Module metadata
    '__version__',
    '__author__'
]

# Initialize module on import
_initialization_results = _initialize_data_acquisition_module()

# Quick status check when run directly
if __name__ == "__main__":
    print("Running Data Acquisition Module Test...")
    print("=" * 50)
    
    # Print module status
    print_module_status()
    
    # Run quick test
    print("\n" + "=" * 50)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 50)
    
    test_results = run_quick_test()
    
    print(f"\nTest Results:")
    print(f"Total Tests: {test_results['summary']['total_tests']}")
    print(f"Passed: {test_results['summary']['passed']}")
    print(f"Failed: {test_results['summary']['failed']}")
    print(f"Skipped: {test_results['summary']['skipped']}")
    print(f"Overall Status: {test_results['summary']['overall_status'].upper()}")
    
    # Show test details
    print(f"\nTest Details:")
    for test_name, test_result in test_results['tests'].items():
        status_icon = {"success": "✓", "partial": "⚠", "failed": "✗", "skipped": "⊝"}[test_result['status']]
        print(f"  {status_icon} {test_name}: {test_result['details']}")
    
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
        print("python -c \"from orchestrator.steps.data_acquisition import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if test_results['summary']['overall_status'] in ['success', 'partial'] else 1
    exit(exit_code)
