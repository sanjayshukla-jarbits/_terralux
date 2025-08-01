"""
Data Acquisition Steps Module - CORRECTED VERSION
================================================

This module provides data acquisition steps for satellite imagery, DEM data,
and local file discovery with FIXED import handling.

Available Steps:
- sentinel_hub_acquisition: Sentinel-2/1 satellite data from Sentinel Hub API
- dem_acquisition: Digital elevation models from various sources (SRTM, ASTER, etc.)
- local_files_discovery: Discovery and cataloging of local data files
- copernicus_hub_acquisition: Data from Copernicus Open Access Hub (optional)
- landsat_acquisition: Landsat satellite data (optional)

FIXES APPLIED:
- Fixed __import__ calls to remove invalid 'package' parameter
- Added proper error handling for missing modules
- Enhanced registration management
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Type

# Configure logging
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "2.0.0-fixed"
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

# Add proper error handling for imports
try:
    from ..base import StepRegistry, register_step_safe, BaseStep
    logger.debug("✓ StepRegistry imported successfully")
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"StepRegistry not available: {e}")
    StepRegistry = None
    register_step_safe = None
    BaseStep = None
    REGISTRY_AVAILABLE = False

def _safe_import_step(step_name: str, module_name: str, class_name: str, 
                     step_type: str, aliases: Optional[List[str]] = None,
                     category: str = 'data_acquisition',
                     name: Optional[str] = None) -> bool:
    """
    FIXED: Safely import and register a step with corrected import syntax.
    
    Args:
        step_name: Human-readable step name
        module_name: Module name to import from
        class_name: Class name to import
        step_type: Step type identifier for registration
        aliases: Optional list of aliases
        category: Step category
        name: Optional name parameter (added for compatibility)
        
    Returns:
        True if import and registration successful
    """
    try:
        # FIXED: Correct import without invalid 'package' parameter
        # OLD (BROKEN): module = __import__(f'.{module_name}', package=__name__, fromlist=[class_name])
        # NEW (FIXED): Use proper absolute import path
        full_module_path = f'orchestrator.steps.{category}.{module_name}'
        module = __import__(full_module_path, fromlist=[class_name])
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
                    'module': full_module_path,
                    'class': class_name,
                    'name': name or step_name
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
    """Register all data acquisition steps with FIXED imports."""
    logger.info("Registering data acquisition steps...")
    
    # Core data acquisition steps
    core_steps = [
        {
            'step_name': 'Sentinel Hub Acquisition',
            'module_name': 'sentinel_hub_step',
            'class_name': 'SentinelHubAcquisitionStep',
            'step_type': 'sentinel_hub_acquisition',
            'aliases': ['sentinel2_acquisition', 'sentinel_data', 'satellite_data'],
            'name': 'sentinel_hub_acquisition'
        },
        {
            'step_name': 'DEM Acquisition',
            'module_name': 'dem_acquisition_step',
            'class_name': 'DEMAcquisitionStep',
            'step_type': 'dem_acquisition',
            'aliases': ['elevation_data', 'srtm_acquisition', 'dem_loading'],
            'name': 'dem_acquisition'
        },
        {
            'step_name': 'Local Files Discovery',
            'module_name': 'local_files_step',
            'class_name': 'LocalFilesDiscoveryStep',
            'step_type': 'local_files_discovery',
            'aliases': ['local_data', 'file_discovery'],
            'name': 'local_files_discovery'
        }
    ]
    
    # Optional data acquisition steps
    optional_steps = [
        {
            'step_name': 'Copernicus Hub Acquisition',
            'module_name': 'copernicus_hub_step',
            'class_name': 'CopernicusHubAcquisitionStep',
            'step_type': 'copernicus_hub_acquisition',
            'aliases': ['copernicus_data', 'scihub_data'],
            'name': 'copernicus_hub_acquisition'
        },
        {
            'step_name': 'Landsat Acquisition',
            'module_name': 'landsat_acquisition_step',
            'class_name': 'LandsatAcquisitionStep',
            'step_type': 'landsat_acquisition',
            'aliases': ['landsat_data', 'usgs_data'],
            'name': 'landsat_acquisition'
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
    
    logger.info(f"✓ Core data acquisition steps: {successful_core}/{total_core} registered")
    if successful_optional > 0:
        logger.info(f"✓ Optional steps: {successful_optional}/{total_optional} registered")
    
    if successful_core < total_core:
        missing_core = [s['step_name'] for s in core_steps 
                       if not _STEP_IMPORTS_SUCCESSFUL.get(s['step_name'], False)]
        logger.warning(f"Missing core data acquisition steps: {missing_core}")
    
    return {
        'core_successful': successful_core,
        'core_total': total_core,
        'optional_successful': successful_optional,
        'optional_total': total_optional,
        'total_registered': successful_core + successful_optional
    }


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
    for step_name, error in _IMPORT_ERRORS.items():
        if "No module named" in error:
            # Extract module name from error
            module_name = error.split("No module named ")[1].strip("'\"")
            if module_name not in missing_deps:
                missing_deps.append(module_name)
    return missing_deps


def validate_data_acquisition_setup() -> Dict[str, Any]:
    """Validate the data acquisition setup and return status."""
    return {
        'steps_available': len(_AVAILABLE_STEPS),
        'core_steps_missing': 3 - len([s for s in _AVAILABLE_STEPS if s in ['sentinel_hub_acquisition', 'dem_acquisition', 'local_files_discovery']]),
        'import_errors': len(_IMPORT_ERRORS),
        'registry_available': REGISTRY_AVAILABLE,
        'missing_dependencies': get_missing_dependencies()
    }


def print_module_status():
    """Print comprehensive module status."""
    print(f"Data Acquisition Module Status (v{__version__})")
    print("=" * 50)
    print(f"Available Steps: {len(_AVAILABLE_STEPS)}")
    print(f"Import Errors: {len(_IMPORT_ERRORS)}")
    print(f"Registry Available: {REGISTRY_AVAILABLE}")
    
    if _AVAILABLE_STEPS:
        print(f"Steps: {', '.join(_AVAILABLE_STEPS)}")
    
    if _IMPORT_ERRORS:
        print("Import Issues:")
        for step, error in _IMPORT_ERRORS.items():
            print(f"  - {step}: {error[:100]}...")


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
3. List available steps: get_available_data_acquisition_steps()

Missing Dependencies:
{chr(10).join(f'  • {dep}' for dep in get_missing_dependencies())}
"""


# Initialize the data acquisition module
def _initialize_data_acquisition_module():
    """Initialize the data acquisition module."""
    logger.info(f"Initializing data acquisition module v{__version__}")
    
    # Register all available steps
    registration_results = _register_data_acquisition_steps()
    
    # Log initialization results
    total_registered = registration_results['total_registered']
    if total_registered > 0:
        logger.info(f"✓ Data acquisition module initialized with {total_registered} steps")
    else:
        logger.warning("⚠ Data acquisition module initialized with no steps")
    
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
    'SentinelHubAcquisitionStep',
    'DEMAcquisitionStep', 
    'LocalFilesDiscoveryStep',
    'CopernicusHubAcquisitionStep',
    'LandsatAcquisitionStep',
    
    # Utility functions
    'get_import_status',
    'get_available_data_acquisition_steps',
    'get_step_aliases',
    'is_step_available',
    'get_missing_dependencies',
    'validate_data_acquisition_setup',
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
    
    print(f"\nInitialization Results:")
    print(f"Core Steps: {_initialization_results['core_successful']}/{_initialization_results['core_total']}")
    print(f"Optional Steps: {_initialization_results['optional_successful']}/{_initialization_results['optional_total']}")
    print(f"Total Registered: {_initialization_results['total_registered']}")
    
    # Show help if there are issues
    if _initialization_results['total_registered'] == 0:
        print(f"\nFor help resolving issues:")
        print("python -c \"from orchestrator.steps.data_acquisition import get_help; print(get_help())\"")
    
    # Exit with appropriate code
    exit_code = 0 if _initialization_results['total_registered'] > 0 else 1
    exit(exit_code)
